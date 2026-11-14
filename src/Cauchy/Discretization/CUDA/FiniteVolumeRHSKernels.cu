#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Launch.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Inputs.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Staging.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    using GeometryView = R::ArrayView<const RHSGeometry, R::AccessBoundsChecks, R::FixedRows<1>>;
    using ParametersView = R::ArrayView<const RHSKernelParameters, R::AccessBoundsChecks, R::FixedRows<1>>;
    namespace
    {
        template <int Components>
        __global__ void Cells(GeometryView published, RHSInput faces, RHSInput lowFaces,
                              R::ArrayView<const NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>> faceStatus,
                              RHSOutput output, RHSOutput lowOutput, RHSStatus status, RHSIndices selected, bool produceLow)
        {
            const auto &geometry = published(0, 0);
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                std::array<real, Components> sum{}, lowSum{};
                NumericalStatus cellStatus;
                for (index entry = geometry.offsets(iCell, 0); entry < geometry.offsets(iCell + 1, 0);)
                {
                    const auto &sample = geometry.samples(entry, 0);
                    const index iFace = sample.auxiliaryFace;
                    const auto failure = faceStatus(iFace, 0);
                    if (!failure)
                    {
                        cellStatus = failure.At(iCell, failure.point);
                        break;
                    }
                    const auto &anchor = geometry.samples(geometry.faceEntries(iFace, 0), 0);
                    const bool opposite = sample.side != anchor.side;
                    std::array<real, Components> value{}, low{};
                    for (int j = 0; j < Components; ++j)
                    {
                        value[j] = (opposite ? -1 : 1) * faces(iFace, j);
                        if (produceLow)
                            low[j] = (opposite ? -1 : 1) * lowFaces(iFace, j);
                    }
                    if (opposite && sample.changesFrame)
                    {
                        const auto old = value, oldLow = low;
                        for (int d = 0; d < 3; ++d)
                        {
                            value[1 + d] = 0;
                            if (produceLow)
                                low[1 + d] = 0;
                            for (int k = 0; k < 3; ++k)
                            {
                                value[1 + d] += sample.rotation[d + 3 * k] * old[1 + k];
                                if (produceLow)
                                    low[1 + d] += sample.rotation[d + 3 * k] * oldLow[1 + k];
                            }
                        }
                    }
                    for (int j = 0; j < Components; ++j)
                    {
                        sum[j] += value[j];
                        if (produceLow)
                            lowSum[j] += low[j];
                    }
                    entry += geometry.faceOffsets(sample.face + 1, 0) - geometry.faceOffsets(sample.face, 0);
                }
                status(iCell, 0) = cellStatus;
                if (cellStatus)
                    for (int j = 0; j < Components; ++j)
                    {
                        output(iCell, j) = sum[j];
                        if (produceLow)
                            lowOutput(iCell, j) = lowSum[j];
                    }
            }
        }
        void ValidateField(const RHSField &field, index owned, index ghost, index width, bool requireGhost)
        {
            const auto supported = [](auto view)
            { return view.kind == R::RowLayoutKind::Table || (view.kind == R::RowLayoutKind::TiledColMajorTable && view.stride == 32); };
            DNDS_check_throw_info(field.owned.RowCount() == owned && field.owned.Width() == width && supported(field.owned),
                                  "CUDA RHS requires correctly shaped AoS or tiled32 input");
            if (requireGhost || field.ghost.RowCount())
                DNDS_check_throw_info(field.ghost.RowCount() == ghost && field.ghost.Width() == width && supported(field.ghost),
                                      "CUDA RHS ghost field is incomplete or has another layout");
        }
    }
    void EvaluateFluxFaces(R::ExecutionContext context, const RHSGeometryPublication &geometry, ParametersView parameters,
                           RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices faces, bool requireGhost)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && geometry.Device() == context.device, "CUDA RHS geometry/stream mismatch");
        controls.Validate();
        const auto &g = geometry.HostDescriptor();
        g.ValidateDevice(context.device);
        R::ValidateViewDevices(context.device, parameters, fields, output.high, output.status, output.fallbackCounts, faces);
        if (controls.produceLow)
            output.low.ValidateDevice(context.device);
        if (output.wallForce.RowCount())
            output.wallForce.ValidateDevice(context.device);
        const index owned = g.offsets.RowCount() - 1, ghost = g.cells.RowCount() - owned, nFaces = g.faceEntries.RowCount();
        DNDS_check_throw_info(parameters.RowCount() == 1 && output.high.RowCount() == nFaces && output.high.Width() == controls.components &&
                                  output.high.kind == R::RowLayoutKind::Table && output.status.RowCount() == nFaces && output.fallbackCounts.RowCount() == nFaces,
                              "CUDA RHS face output/profile mismatch");
        ValidateField(fields.mean, owned, ghost, controls.components, requireGhost);
        const bool reconstruct = !controls.firstOrder && (!controls.direct || controls.linear);
        if (reconstruct)
        {
            ValidateField(fields.reconstruction, owned, ghost, controls.components * (controls.direct ? controls.dimension : controls.modes), requireGhost);
            if (!controls.direct)
                DNDS_check_throw_info(g.hasBasis && g.basis.polynomial.size == controls.modes + 1 &&
                                          g.basis.availableDerivatives >= (controls.viscous && !controls.primitiveGradient ? controls.dimension + 1 : 1),
                                      "CUDA RHS basis publication is incomplete");
        }
        if (controls.primitiveGradient)
            ValidateField(fields.viscousData, owned, ghost, 5 * controls.dimension, requireGhost);
        else if (reconstruct && controls.viscous && controls.distinctViscous)
            ValidateField(fields.viscousData, owned, ghost, fields.reconstruction.owned.Width(), requireGhost);
        if (fields.hasPreparedMean)
            DNDS_check_throw_info(fields.preparedMean.RowCount() == g.cells.RowCount() && fields.preparedMean.Width() >= 2,
                                  "CUDA RHS prepared means are incomplete");
        if (controls.useStencil && controls.fluxPolicy.transverseSignal)
        {
            DNDS_check_throw_info(fields.hasWave, "CUDA RHS wave stencil was not prepared");
            ValidateField(fields.waveJump, owned, ghost, 1, requireGhost);
        }
        if (controls.useStencil && controls.fluxPolicy.pressureStencil)
        {
            DNDS_check_throw_info(fields.hasPressure, "CUDA RHS pressure stencil was not prepared");
            ValidateField(fields.pressureMinimum, owned, ghost, 1, requireGhost);
        }
        if (fields.hasIncoming)
            DNDS_check_throw_info(fields.incoming.Width() == controls.components, "CUDA RHS interface width differs");
        if (controls.components > 5)
            DNDS_check_throw_info(g.hasWall && g.wallMean.RowCount() == g.cells.RowCount() && g.wallFaces.RowCount() + 1 == g.faceOffsets.RowCount(),
                                  "CUDA RANS RHS requires placed wall samples");
        if (controls.produceLow)
            DNDS_check_throw_info(output.low.RowCount() == nFaces && output.low.Width() == controls.components && output.low.kind == R::RowLayoutKind::Table,
                                  "CUDA RHS low face output differs");
        R::CheckKernelArgumentBudget<GeometryView, ParametersView, RHSKernelControls, RHSFields, RHSFaceOutputs, RHSIndices>();
        if (!faces.RowCount())
            return;
        if (controls.doubleBuffer)
        {
            Detail::LaunchBufferedFluxFaces(context, geometry, parameters, controls, fields, output, faces);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "buffered CUDA RHS face launch failed");
            return;
        }
        using P = DNDS::Math::ComputePrecision;
        if (controls.polynomialPrecision != P::FP64 || controls.fluxPrecision != P::FP64 || controls.dissipationPrecision != P::FP64)
        {
            Detail::LaunchComputeFluxFaces(context, geometry, parameters, controls, fields, output, faces);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA RHS compute face launch failed");
            return;
        }
        switch (controls.components)
        {
        case 5:
        {
            using Table = R::TableRows;
            using Tiled = R::TiledColMajorTable<32>;
            const bool tiledMean = fields.mean.owned.kind == R::RowLayoutKind::TiledColMajorTable;
            const bool tiledRec = fields.reconstruction.owned.RowCount() ? fields.reconstruction.owned.kind == R::RowLayoutKind::TiledColMajorTable : fields.viscousData.owned.kind == R::RowLayoutKind::TiledColMajorTable;
            if (tiledMean && tiledRec)
                Detail::LaunchFluxFaces<5>(context, geometry, parameters, controls, Detail::ProjectRHSFields<Tiled, Tiled>(fields), output, faces);
            else if (tiledMean)
                Detail::LaunchFluxFaces<5>(context, geometry, parameters, controls, Detail::ProjectRHSFields<Tiled, Table>(fields), output, faces);
            else if (tiledRec)
                Detail::LaunchFluxFaces<5>(context, geometry, parameters, controls, Detail::ProjectRHSFields<Table, Tiled>(fields), output, faces);
            else
                Detail::LaunchFluxFaces<5>(context, geometry, parameters, controls, Detail::ProjectRHSFields<Table, Table>(fields), output, faces);
            break;
        }
        case 6:
            Detail::LaunchFluxFaces<6>(context, geometry, parameters, controls, fields, output, faces);
            break;
        case 7:
            Detail::LaunchFluxFaces<7>(context, geometry, parameters, controls, fields, output, faces);
            break;
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA RHS face launch failed");
    }
    void GatherFluxCells(R::ExecutionContext context, const RHSGeometryPublication &geometry, RHSInput highFaces, RHSInput lowFaces,
                         R::ArrayView<const NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>> faceStatus,
                         RHSOutput high, RHSOutput low, RHSStatus status, RHSIndices cells, bool produceLow)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && geometry.Device() == context.device, "CUDA RHS gather geometry/stream mismatch");
        const auto &g = geometry.HostDescriptor();
        g.ValidateDevice(context.device);
        R::ValidateViewDevices(context.device, highFaces, faceStatus, high, status, cells);
        if (produceLow)
            R::ValidateViewDevices(context.device, lowFaces, low);
        DNDS_check_throw_info(high.Width() >= 5 && high.Width() <= 7 && high.RowCount() + 1 == g.offsets.RowCount() &&
                                  highFaces.Width() == high.Width() && highFaces.RowCount() == g.faceEntries.RowCount() &&
                                  faceStatus.RowCount() == highFaces.RowCount() && status.RowCount() == high.RowCount() &&
                                  high.kind == R::RowLayoutKind::Table && highFaces.kind == R::RowLayoutKind::Table,
                              "CUDA RHS gather shape/layout mismatch");
        if (produceLow)
            DNDS_check_throw_info(high.Width() == 5 && low.Width() == 5 && lowFaces.Width() == 5 && low.RowCount() == high.RowCount() &&
                                      lowFaces.RowCount() == highFaces.RowCount() && low.kind == R::RowLayoutKind::Table && lowFaces.kind == R::RowLayoutKind::Table,
                                  "CUDA RHS low gather shape differs");
        R::CheckKernelArgumentBudget<GeometryView, RHSInput, RHSInput, decltype(faceStatus), RHSOutput, RHSOutput, RHSStatus, RHSIndices, bool>();
        if (!cells.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>((cells.RowCount() + 127) / 128, 1024));
        const auto stream = static_cast<cudaStream_t>(context.stream);
        switch (high.Width())
        {
        case 5:
            Cells<5><<<blocks, 128, 0, stream>>>(geometry.View(), highFaces, lowFaces, faceStatus, high, low, status, cells, produceLow);
            break;
        case 6:
            Cells<6><<<blocks, 128, 0, stream>>>(geometry.View(), highFaces, lowFaces, faceStatus, high, low, status, cells, produceLow);
            break;
        case 7:
            Cells<7><<<blocks, 128, 0, stream>>>(geometry.View(), highFaces, lowFaces, faceStatus, high, low, status, cells, produceLow);
            break;
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA RHS gather launch failed");
    }
}
