#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSFacePackages.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include <algorithm>
#include <optional>
#include <limits>
#include <vector>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    RHSFaceStageShape RHSFacePackages::StageShape(const RHSKernelControls &controls)
    {
        RHSFaceStageShape result;
        result.components = controls.components;
        if (controls.mechanical)
            return result;
        const bool reconstruct = !controls.firstOrder && (!controls.direct || controls.linear);
        result.reconstruction = reconstruct ? controls.components * (controls.direct ? controls.dimension : controls.modes) : 0;
        result.viscous = controls.primitiveGradient ? 5 * controls.dimension : controls.viscous && controls.distinctViscous ? result.reconstruction
                                                                                                                            : 0;
        result.width = controls.components + result.reconstruction + result.viscous + 4;
        return result;
    }
    size_t RHSFacePackages::LaneScratchBytes(const RHSKernelControls &controls)
    {
        const auto flux = controls.components * sizeof(real) * (controls.produceLow ? 2 : 1);
        return flux + (controls.mechanical ? sizeof(NumericalStatus) : 2 * sizeof(int) + 3 * sizeof(index));
    }
    struct RHSFacePackages::Data
    {
        std::optional<R::Array<index>::ReadLease> selection;
        R::Array<RHSFacePackage> packages;
        R::Array<RHSFaceSegment> segments;
        std::optional<R::Array<RHSFacePackage>::ReadLease> packageRead;
        std::optional<R::Array<RHSFaceSegment>::ReadLease> segmentRead;
        RHSFacePackageView view;
    };
    RHSFacePackages RHSFacePackages::Prepare(const R::Array<index> &selection, SpanView<const index> counts,
                                             const RHSKernelControls &controls, R::ExecutionContext context, R::RowSelectionShape shape)
    {
        controls.Validate();
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && selection.GetDevice() == context.device &&
                                  controls.mapping == RHSFaceMapping::CTASegments,
                              "RHS segments require placed CUDA selection and CTA-segment mapping");
        auto data = std::make_shared<Data>();
        auto &view = data->view;
        view.selection.shape = shape;
        if (shape.form == R::RowSelectionForm::IndexList)
        {
            data->selection.emplace(selection.Read(context));
            view.selection.indices = data->selection->View<R::FixedRows<1>>();
        }
        view.mapping = controls.mapping;
        view.blockSize = controls.blockSize;
        view.produceLow = controls.produceLow;
        view.doubleBuffer = controls.doubleBuffer;
        view.shape = StageShape(controls);
        // Per lane: high/(optional low) flux, error code/detail/row/point,
        // and integer fallback count. No point state leaves CTA memory.
        const size_t scratch = size_t(controls.blockSize) * LaneScratchBytes(controls);
        const size_t stagePerFace = size_t(2 * view.shape.width) * sizeof(real) * (controls.doubleBuffer ? 2 : 1);
        constexpr size_t budget = 48 * 1024;
        DNDS_check_throw_info(scratch < budget && stagePerFace <= budget - scratch,
                              "RHS segment shape/block size exceeds portable 48 KiB shared-memory budget");
        const int capacity = stagePerFace ? int((budget - scratch) / stagePerFace) : controls.blockSize;
        std::optional<R::Array<index>::ReadLease> hostRead;
        R::RowSelection hostRows{shape};
        if (shape.form == R::RowSelectionForm::IndexList)
        {
            hostRead.emplace(selection.To(R::Device::Host(), context).Read());
            hostRows.indices = hostRead->View<R::FixedRows<1>>();
        }
        std::vector<RHSFacePackage> packages;
        std::vector<RHSFaceSegment> segments;
        RHSFacePackage current;
        int cursor = 0;
        const auto finish = [&]
        {
            if (!current.faces)
                return;
            view.maximumFaces = std::max(view.maximumFaces, current.faces);
            view.warpContainedPackages += current.warpContained;
            packages.push_back(current);
            current = {index(segments.size()), 0, 0};
            cursor = 0;
        };
        hostRows.Visit([&](auto rowAt)
                       {
        for (index iRow = 0; iRow < hostRows.RowCount(); ++iRow)
        {
            const index iAux = rowAt(iRow);
            DNDS_check_throw_info(iAux >= 0 && iAux < counts.size() && counts[iAux] > 0,
                                  "RHS package has an unavailable or empty face rule");
            const index points = counts[iAux];
            const int lanes = int(std::min<index>(points, controls.blockSize));
            int start = cursor;
            if (current.faces == capacity || start + lanes > controls.blockSize)
            {
                finish();
                start = 0;
            }
            DNDS_check_throw_info(points / lanes + (points % lanes != 0) <= std::numeric_limits<int>::max(),
                                  "RHS face rule requires too many bounded point tiles");
            segments.push_back({iAux, points, start, lanes});
            ++current.faces;
            current.tiles = std::max(current.tiles, int(points / lanes + (points % lanes != 0)));
            current.maximumLanes = std::max(current.maximumLanes, lanes);
            current.warpContained &= start / 32 == (start + lanes - 1) / 32;
            cursor = start + lanes;
        } });
        finish();
        view.sharedBytes = scratch + size_t(view.maximumFaces) * stagePerFace;
        R::Array<RHSFacePackage> hostPackages(R::RowLayout::Dense(packages.size(), 1));
        R::Array<RHSFaceSegment> hostSegments(R::RowLayout::Dense(segments.size(), 1));
        {
            auto p = hostPackages.Write();
            auto s = hostSegments.Write();
            for (index i = 0; i < index(packages.size()); ++i)
                p.View()(i, 0) = packages[i];
            for (index i = 0; i < index(segments.size()); ++i)
                s.View()(i, 0) = segments[i];
        }
        data->packages = hostPackages.To(context.device, context);
        data->segments = hostSegments.To(context.device, context);
        data->packages.SetName("Cauchy/RHS/facePackages", DNDS_ALLOCATION_SITE());
        data->segments.SetName("Cauchy/RHS/faceSegments", DNDS_ALLOCATION_SITE());
        data->packageRead.emplace(data->packages.Read(context));
        data->segmentRead.emplace(data->segments.Read(context));
        view.packages = data->packageRead->View<R::FixedRows<1>>();
        view.segments = data->segmentRead->View<R::FixedRows<1>>();
        view.ValidateDevice(context.device);
        return RHSFacePackages(std::move(data));
    }
    RHSFacePackageView RHSFacePackages::View() const { return _data->view; }
}
