#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Shapes.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Segments.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "FiniteVolumeRHSKernels_Instantiations.hpp"
#include "FiniteVolumeACMFluxKernels_Instantiations.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeACMFluxKernels.hpp"
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
                              RHSOutput output, RHSOutput lowOutput, RHSStatus status, RHSIndices selected, bool produceLow, int momentumFirst)
        {
            const auto &geometry = published(0, 0);
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                std::array<real, Components> sum{}, lowSum{};
                NumericalStatus cellStatus;
                const auto cellFaces = geometry.stencil.topology.CellFaces(iCell);
                for (rowsize slot = 0; slot < cellFaces.size(); ++slot)
                {
                    const auto sample = geometry.stencil.GradientSample(iCell, slot);
                    const index iFace = sample.auxiliaryFace;
                    const auto failure = faceStatus(iFace, 0);
                    if (!failure)
                    {
                        cellStatus = failure.At(iCell, failure.point);
                        break;
                    }
                    const bool opposite = sample.side != 0;
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
                            value[momentumFirst + d] = 0;
                            if (produceLow)
                                low[momentumFirst + d] = 0;
                            for (int k = 0; k < 3; ++k)
                            {
                                value[momentumFirst + d] += sample.rotation[d + 3 * k] * old[momentumFirst + k];
                                if (produceLow)
                                    low[momentumFirst + d] += sample.rotation[d + 3 * k] * oldLow[momentumFirst + k];
                            }
                        }
                    }
                    for (int j = 0; j < Components; ++j)
                    {
                        sum[j] += value[j];
                        if (produceLow)
                            lowSum[j] += low[j];
                    }
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
            { return !view.offsets && !view.sizes && (!view.RowCount() || view.data) &&
                     ((view.kind == R::RowLayoutKind::Table && view.stride == view.Width()) ||
                      (view.kind == R::RowLayoutKind::TiledColMajorTable && view.stride == 32)); };
            DNDS_check_throw_info(field.owned.RowCount() == owned && field.owned.Width() == width && supported(field.owned),
                                  "CUDA RHS requires correctly shaped compact AoS or tiled32 input; padded field rows are unsupported");
            if (requireGhost || field.ghost.RowCount())
                DNDS_check_throw_info(field.ghost.RowCount() == ghost && field.ghost.Width() == width && supported(field.ghost),
                                      "CUDA RHS ghost field is incomplete or has another layout");
        }
    }
    void EvaluateFluxFaces(R::ExecutionContext context, const RHSGeometryPublication &geometry, ParametersView parameters,
                           RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices faces, bool requireGhost,
                           RHSFacePackageView packages, RHSWallOutput wallShear)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && geometry.Device() == context.device, "CUDA RHS geometry/stream mismatch");
        controls.Validate();
        const auto &g = geometry.HostDescriptor();
        g.ValidateDevice(context.device);
        if (!controls.mechanical)
            parameters.ValidateDevice(context.device);
        R::ValidateViewDevices(context.device, fields, output.high, output.status, output.fallbackCounts, faces);
        if (controls.produceLow)
            output.low.ValidateDevice(context.device);
        if (wallShear.RowCount())
        {
            wallShear.ValidateDevice(context.device);
            DNDS_check_throw_info(controls.mechanical && controls.viscous && wallShear.RowCount() == output.wallForce.RowCount(),
                                  "CUDA wall shear requires a boundary-sized viscous ACM output");
        }
        if (output.wallForce.RowCount())
            output.wallForce.ValidateDevice(context.device);
        const index owned = g.stencil.jacobianOffsets.RowCount() - 1, ghost = g.cells.RowCount() - owned, nFaces = g.stencil.stencilFaces.RowCount();
        DNDS_check_throw_info((controls.mechanical ? g.acm.RowCount() == 1 : parameters.RowCount() == 1 && !g.acm.RowCount()) && output.high.RowCount() == nFaces && output.high.Width() == controls.components &&
                                  output.high.kind == R::RowLayoutKind::Table && output.high.stride == controls.components && output.status.RowCount() == nFaces && output.fallbackCounts.RowCount() == nFaces,
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
            DNDS_check_throw_info(fields.incoming.Width() == controls.components && fields.incoming.kind == R::RowLayoutKind::Table &&
                                      fields.incoming.stride == fields.incoming.Width(),
                                  "CUDA RHS interface values require compact AoS rows");
        if (controls.components > (controls.mechanical ? 4 : 5))
            DNDS_check_throw_info(g.hasWall && g.wallMean.RowCount() == g.cells.RowCount() && g.wallFaces.RowCount() + 1 == g.faceOffsets.RowCount(),
                                  "CUDA RANS RHS requires placed wall samples");
        if (controls.produceLow)
            DNDS_check_throw_info(output.low.RowCount() == nFaces && output.low.Width() == controls.components && output.low.kind == R::RowLayoutKind::Table && output.low.stride == controls.components,
                                  "CUDA RHS low face output differs");
        R::CheckKernelArgumentBudget<GeometryView, ParametersView, RHSKernelControls, RHSFields, RHSFaceOutputs, RHSIndices>();
        if (!faces.RowCount())
            return;
        if (controls.mechanical)
        {
            const auto blocks = controls.mapping == RHSFaceMapping::CTASegments
                                    ? Detail::ValidateSegmentedFluxFaces(context, geometry, controls, faces, packages)
                                    : 0;
            const Detail::RHSFaceLaunch launch{context, geometry, parameters, controls, fields, output, faces, packages, blocks};
            if (!controls.direct && !controls.firstOrder && Detail::TryLaunchFP64ACMRHS(launch, wallShear))
            {
                DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA ACM static polynomial launch failed");
                return;
            }
            switch (controls.components)
            {
            case 4:
                Detail::LaunchACMFaces<4>(launch, wallShear);
                break;
            case 5:
                Detail::LaunchACMFaces<5>(launch, wallShear);
                break;
            case 6:
                Detail::LaunchACMFaces<6>(launch, wallShear);
                break;
            }
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA ACM face launch failed");
            return;
        }
        const unsigned segmentBlocks = controls.mapping == RHSFaceMapping::CTASegments
                                           ? Detail::ValidateSegmentedFluxFaces(context, geometry, controls, faces, packages)
                                           : 0;
        using P = DNDS::Math::ComputePrecision;
        if (controls.polynomialPrecision != P::FP64 || controls.fluxPrecision != P::FP64 || controls.dissipationPrecision != P::FP64)
        {
            Detail::LaunchComputeFluxFaces(context, geometry, parameters, controls, fields, output, faces);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA RHS compute face launch failed");
            return;
        }
        const Detail::RHSFaceLaunch launch{context, geometry, parameters, controls, fields, output, faces, packages, segmentBlocks};
        Detail::LaunchFP64RHS(launch);
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
        DNDS_check_throw_info(high.Width() >= (g.acm.RowCount() ? 4 : 5) && high.Width() <= (g.acm.RowCount() ? 6 : 7) && high.RowCount() + 1 == g.stencil.jacobianOffsets.RowCount() &&
                                  highFaces.Width() == high.Width() && highFaces.RowCount() == g.stencil.stencilFaces.RowCount() &&
                                  faceStatus.RowCount() == highFaces.RowCount() && status.RowCount() == high.RowCount() &&
                                  high.kind == R::RowLayoutKind::Table && highFaces.kind == R::RowLayoutKind::Table,
                              "CUDA RHS gather shape/layout mismatch");
        if (produceLow)
            DNDS_check_throw_info(high.Width() == 5 && low.Width() == 5 && lowFaces.Width() == 5 && low.RowCount() == high.RowCount() &&
                                      lowFaces.RowCount() == highFaces.RowCount() && low.kind == R::RowLayoutKind::Table && lowFaces.kind == R::RowLayoutKind::Table,
                                  "CUDA RHS low gather shape differs");
        R::CheckKernelArgumentBudget<GeometryView, RHSInput, RHSInput, decltype(faceStatus), RHSOutput, RHSOutput, RHSStatus, RHSIndices, bool, int>();
        if (!cells.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>((cells.RowCount() + 127) / 128, 1024));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        const int momentumFirst = g.acm.RowCount() ? 0 : 1;
        switch (high.Width())
        {
        case 4:
            Cells<4><<<blocks, 128, 0, stream>>>(geometry.View(), highFaces, lowFaces, faceStatus, high, low, status, cells, produceLow, momentumFirst);
            break;
        case 5:
            Cells<5><<<blocks, 128, 0, stream>>>(geometry.View(), highFaces, lowFaces, faceStatus, high, low, status, cells, produceLow, momentumFirst);
            break;
        case 6:
            Cells<6><<<blocks, 128, 0, stream>>>(geometry.View(), highFaces, lowFaces, faceStatus, high, low, status, cells, produceLow, momentumFirst);
            break;
        case 7:
            Cells<7><<<blocks, 128, 0, stream>>>(geometry.View(), highFaces, lowFaces, faceStatus, high, low, status, cells, produceLow, momentumFirst);
            break;
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA RHS gather launch failed");
    }
}
