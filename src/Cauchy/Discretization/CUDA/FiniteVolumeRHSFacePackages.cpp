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
        const bool reconstruct = !controls.firstOrder && (!controls.direct || controls.linear);
        result.reconstruction = reconstruct ? 5 * (controls.direct ? controls.dimension : controls.modes) : 0;
        result.viscous = controls.primitiveGradient ? 5 * controls.dimension : controls.viscous && controls.distinctViscous ? result.reconstruction
                                                                                                                            : 0;
        result.width = 5 + result.reconstruction + result.viscous + 4;
        return result;
    }
    struct RHSFacePackages::Data
    {
        R::Array<index>::ReadLease selection;
        R::Array<RHSFacePackage> packages;
        R::Array<RHSFaceSegment> segments;
        std::optional<R::Array<RHSFacePackage>::ReadLease> packageRead;
        std::optional<R::Array<RHSFaceSegment>::ReadLease> segmentRead;
        RHSFacePackageView view;
        Data(const R::Array<index> &selected, R::ExecutionContext context)
            : selection(selected.Read(context)) {}
    };
    RHSFacePackages RHSFacePackages::Prepare(const R::Array<index> &selection, SpanView<const index> counts,
                                             const RHSKernelControls &controls, R::ExecutionContext context)
    {
        controls.Validate();
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && selection.GetDevice() == context.device &&
                                  (controls.mapping == RHSFaceMapping::WarpSegments || controls.mapping == RHSFaceMapping::CTASegments),
                              "RHS segments require placed CUDA selection and an explicit segmented mapping");
        auto data = std::make_shared<Data>(selection, context);
        auto &view = data->view;
        view.selection = data->selection.View<R::FixedRows<1>>();
        view.mapping = controls.mapping;
        view.blockSize = controls.blockSize;
        view.produceLow = controls.produceLow;
        view.shape = StageShape(controls);
        // Per lane: high/(optional low) flux, error code/detail/row/point,
        // and integer fallback count. No point state leaves CTA memory.
        const size_t scratch = size_t(controls.blockSize) * (5 * sizeof(real) * (controls.produceLow ? 2 : 1) + 2 * sizeof(int) + 3 * sizeof(index));
        const size_t stagePerFace = size_t(2 * view.shape.width) * sizeof(real);
        constexpr size_t budget = 48 * 1024;
        DNDS_check_throw_info(scratch < budget && stagePerFace <= budget - scratch,
                              "RHS segment shape/block size exceeds portable 48 KiB shared-memory budget");
        const int capacity = int((budget - scratch) / stagePerFace);
        const auto host = selection.To(R::Device::Host(), context);
        const auto selected = host.Read();
        const auto ids = selected.View<R::FixedRows<1>>();
        std::vector<RHSFacePackage> packages;
        std::vector<RHSFaceSegment> segments;
        RHSFacePackage current;
        int cursor = 0;
        const auto finish = [&]
        {
            if (!current.faces)
                return;
            view.maximumFaces = std::max(view.maximumFaces, current.faces);
            packages.push_back(current);
            current = {index(segments.size()), 0, 0};
            cursor = 0;
        };
        for (index iRow = 0; iRow < ids.RowCount(); ++iRow)
        {
            const index iAux = ids(iRow, 0);
            DNDS_check_throw_info(iAux >= 0 && iAux < counts.size() && counts[iAux] > 0,
                                  "RHS package has an unavailable or empty face rule");
            const index points = counts[iAux];
            const int lanes = int(std::min<index>(points, view.mapping == RHSFaceMapping::WarpSegments ? 32 : controls.blockSize));
            int start = cursor;
            if (view.mapping == RHSFaceMapping::WarpSegments && start % 32 + lanes > 32)
                start = (start / 32 + 1) * 32;
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
            cursor = start + lanes;
        }
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
