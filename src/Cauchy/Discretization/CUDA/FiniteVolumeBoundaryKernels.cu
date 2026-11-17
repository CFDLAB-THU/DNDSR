#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    void BoundaryKernelParameters::Validate(int components) const
    {
        gas.Validate();
        using Kind = Physics::RANS::ModelKind;
        DNDS_check_throw_info((components == 5 && rans == Kind::None) || (components == 6 && rans == Kind::SA) ||
                                  (components == 7 && (rans == Kind::KOmegaWilcox || rans == Kind::KOmegaSST)),
                              "unsupported CUDA boundary equation profile");
        DNDS_check_throw_info(thermal.components == components && thermal.flowFirst == 0 && thermal.independent == 0 &&
                                  thermal.speciesEnergies[0].empty() && thermal.speciesEnergies[1].empty(),
                              "CUDA boundary parameters require a perfect-gas thermal policy");
    }
    namespace
    {
        void Validate(R::ExecutionContext context, const BoundaryTraceRequest &request, bool primitive)
        {
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA boundary requires a CUDA stream");
            const bool acm = request.acm.RowCount() > 0;
            if (acm)
                R::ValidateViewDevices(context.device, request.acm, request.acmBC);
            else
                R::ValidateViewDevices(context.device, request.parameters, request.conditions);
            R::ValidateViewDevices(context.device, request.stencil, request.faces, request.means, request.output, request.status);
            DNDS_check_throw_info(std::isfinite(request.time) && (acm ? !primitive && request.components >= 4 && request.components <= 6 && request.acm.RowCount() == 1 : request.components >= 5 && request.components <= 7 && request.parameters.RowCount() == 1) &&
                                      request.means.Width() == request.components && request.output.Width() == (primitive ? 5 : request.components) &&
                                      request.status.RowCount() == request.output.RowCount() && request.means.kind == R::RowLayoutKind::Table &&
                                      request.output.kind == R::RowLayoutKind::Table,
                                  "CUDA boundary array/profile mismatch");
            DNDS_check_throw_info(request.stencil.RowCount() == 1 &&
                                      (!request.centers || request.faces.RowCount() == request.output.RowCount()),
                                  "CUDA boundary compact face/output rows differ");
            if (request.polynomial && !primitive)
            {
                R::ValidateViewDevices(context.device, request.basis, request.coefficients);
                DNDS_check_throw_info(request.modes > 0 && request.modes <= 19 && request.basis.polynomial.size == request.modes + 1 &&
                                          request.basis.availableDerivatives > 0 && request.coefficients.RowCount() == request.means.RowCount() &&
                                          request.coefficients.Width() == request.components * request.modes && request.coefficients.kind == R::RowLayoutKind::Table,
                                      "CUDA boundary polynomial shape mismatch");
                if (request.centers)
                    DNDS_check_throw_info(request.basis.cells.RowCount() >= request.means.RowCount() &&
                                              request.basis.moments.RowCount() >= request.means.RowCount() &&
                                              request.basis.moments.Width() == request.modes + 1,
                                          "center polynomial boundary evaluation requires resident point geometry and moments");
            }
            if (request.components == (acm ? 6 : 7) && !primitive)
            {
                request.wallMean.ValidateDevice(context.device);
                // Mesh wall metrics cover owned+ghost cells; this boundary
                // pass consumes only owned means. Retain the complete metric
                // publication instead of manufacturing an owned-only copy.
                DNDS_check_throw_info(request.wallMean.RowCount() >= request.means.RowCount(), "CUDA boundary wall means are missing");
            }
            if (request.hasIncoming)
            {
                request.incoming.ValidateDevice(context.device);
                DNDS_check_throw_info(request.incoming.Width() == request.components, "CUDA interface boundary width differs");
            }
        }
        template <int Components, bool Mechanical = false>
        __global__ void TraceRows(BoundaryTraceRequest request)
        {
            const auto &parameters = [&]() -> const auto &
            { if constexpr (Mechanical) return request.acm(0,0); else return request.parameters(0,0); }();
            const auto &stencil = request.stencil(0, 0);
            for (index iFace = index(blockIdx.x) * blockDim.x + threadIdx.x; iFace < request.faces.RowCount(); iFace += index(gridDim.x) * blockDim.x)
            {
                const auto range = request.faces(iFace, 0);
                const index count = request.centers ? 1 : range.quadratureCount;
                for (index iLocal = 0; iLocal < count; ++iLocal)
                {
                    const index iPoint = request.centers ? iFace : range.quadratureBegin + iLocal;
                    const auto sample = BoundaryTracePoint(stencil, range, iFace, iLocal, request.centers);
                    const auto &condition = [&]() -> const auto &
                    { if constexpr (Mechanical) return request.acmBC(sample.condition,0); else return request.conditions(sample.condition,0); }();
                    auto &status = request.status(iPoint, 0);
                    status = {};
                    if (condition.kind == BoundaryKind::Interface)
                    {
                        if (!request.hasIncoming || sample.interfaceRow < 0 || sample.interfaceRow >= request.incoming.RowCount())
                        {
                            status = NumericalStatus{NumericalError::InvalidInput, 2}.At(sample.cell, iPoint);
                            continue;
                        }
                        for (int j = 0; j < Components; ++j)
                            request.output(iPoint, j) = request.incoming(sample.interfaceRow, j);
                        continue;
                    }
                    std::array<real, Components> mean{}, point{};
                    for (int j = 0; j < Components; ++j)
                        point[j] = mean[j] = request.means(sample.cell, j);
                    if (request.polynomial)
                    {
                        real basis[19];
                        const bool cached = !request.centers && request.basis.cachedDerivatives > 0;
                        const auto pointContext = cached ? CFV::Math::TaylorPointEvaluation{} : request.basis.PreparePoint(sample.cell, sample.location);
                        for (int mode = 0; mode < request.modes; ++mode)
                            basis[mode] = request.centers
                                              ? request.basis.EvaluateAtPoint(sample.cell, pointContext, 0, mode)
                                              : request.basis.Evaluate(sample.cell, sample.point, sample.side, pointContext, 0, mode);
                        for (int j = 0; j < Components; ++j)
                            for (int mode = 0; mode < request.modes; ++mode)
                                point[j] += request.coefficients(sample.cell, j * request.modes + mode) * basis[mode];
                    }
                    if constexpr (Mechanical)
                    {
                        if constexpr (Components > 4)
                            if (request.repairRANS)
                                RANSReconstructionView{parameters.rans.kind, Components - 4}.RepairPoint(
                                    SpanView<const real>{mean.data() + 4, Components - 4}, SpanView<real>{point.data() + 4, Components - 4});
                        real omega = 0;
                        if constexpr (Components == 6)
                            if (condition.kind == BoundaryKind::NoSlipWall)
                            {
                                const real distance = request.wallMean(sample.cell, 0);
                                if (!(distance > 0 && std::isfinite(distance)))
                                {
                                    status = NumericalStatus{NumericalError::InvalidState}.At(sample.cell, iPoint);
                                    continue;
                                }
                                omega = parameters.flow.kinematicViscosity * Physics::RANS::kWallOmegaCoeff / sqr(distance);
                            }
                        std::array<real, Components> exterior{};
                        if (!ACM::Exterior(parameters.flow, condition.kind, point, condition.acmState, condition.velocity, sample.normal, exterior, Components, omega))
                            status = NumericalStatus{NumericalError::UnsupportedRequest}.At(sample.cell, iPoint);
                        else
                            for (int j = 0; j < Components; ++j)
                                request.output(iPoint, j) = request.exterior ? exterior[j] : .5 * (point[j] + exterior[j]);
                    }
                    else
                    {
                        // PP POLICY: repair this temporary unlimited VR boundary input,
                        // never its coefficients. The independent flux beta follows.
                        if (request.polynomial && (parameters.allowPointFallback || request.forcePointFallback) && !parameters.thermal.ClippedReferenceAdmissible(point))
                            point = mean;
                        if constexpr (Components > 5)
                            if (request.repairRANS)
                                RANSReconstructionView{parameters.rans, Components - 5}.RepairPoint(
                                    SpanView<const real>{mean.data() + 5, Components - 5}, SpanView<real>{point.data() + 5, Components - 5});
                        const real wallDistance = Components == 7 ? request.wallMean(sample.cell, 0) : 0;
                        status = TryPerfectGasBoundaryState<Components>(condition, parameters.gas, parameters.rans, point, mean, wallDistance,
                                                                        sample.normal, sample.location, request.time, request.frame, request.exterior);
                        if (!status && request.polynomial && (parameters.allowPointFallback || request.forcePointFallback) && Physics::AllowsMeanFallback(status))
                        {
                            point = mean;
                            status = TryPerfectGasBoundaryState<Components>(condition, parameters.gas, parameters.rans, point, mean, wallDistance,
                                                                            sample.normal, sample.location, request.time, request.frame, request.exterior);
                        }
                        if (!status)
                        {
                            status = status.At(sample.cell, iPoint);
                            continue;
                        }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                        if constexpr (Components == 6)
                            if (condition.kind == BoundaryKind::NoSlipWall)
                                point[5] = 0;
#endif
                        for (int j = 0; j < Components; ++j)
                            request.output(iPoint, j) = point[j];
                    }
                }
            }
        }
        __global__ void PrimitiveRows(BoundaryTraceRequest request)
        {
            const auto &parameters = request.parameters(0, 0);
            for (index iPoint = index(blockIdx.x) * blockDim.x + threadIdx.x; iPoint < request.output.RowCount(); iPoint += index(gridDim.x) * blockDim.x)
            {
                const auto sample = BoundaryTracePoint(request.stencil(0, 0), request.faces(iPoint, 0), iPoint, 0, true);
                const auto &condition = request.conditions(sample.condition, 0);
                auto &status = request.status(iPoint, 0);
                status = {};
                FlowState inside{}, state{};
                for (int j = 0; j < 5; ++j)
                    inside[j] = request.means(sample.cell, j);
                if (condition.kind == BoundaryKind::Interface)
                {
                    if (!request.hasIncoming || sample.interfaceRow < 0 || sample.interfaceRow >= request.incoming.RowCount())
                    {
                        status = NumericalStatus{NumericalError::InvalidInput, 2}.At(sample.cell, iPoint);
                        continue;
                    }
                    for (int j = 0; j < 5; ++j)
                        state[j] = request.incoming(sample.interfaceRow, j);
                }
                else
                {
                    const auto checked = TryPerfectGasPoint(parameters.gas, inside, false);
                    if (!checked)
                    {
                        status = checked.status.At(sample.cell, iPoint);
                        continue;
                    }
                    const auto exterior = TryPerfectGasBoundaryValue(condition, parameters.gas, inside, sample.normal, sample.location,
                                                                     request.time, request.frame, false);
                    if (!exterior)
                    {
                        status = exterior.status.At(sample.cell, iPoint);
                        continue;
                    }
                    for (int j = 0; j < 5; ++j)
                        state[j] = exterior.value[j];
                }
                const auto properties = TryPerfectGasPoint(parameters.gas, state, false);
                if (!properties)
                {
                    status = properties.status.At(sample.cell, iPoint);
                    continue;
                }
                request.output(iPoint, 0) = state[0];
                for (int d = 0; d < 3; ++d)
                    request.output(iPoint, 1 + d) = state[1 + d] / state[0];
                request.output(iPoint, 4) = properties.value.thermo.temperature;
            }
        }
        __global__ void CopyRows(BoundaryInput source, BoundaryOutput output, BoundaryRows rows)
        {
            for (index iPoint = index(blockIdx.x) * blockDim.x + threadIdx.x; iPoint < output.RowCount(); iPoint += index(gridDim.x) * blockDim.x)
                for (index j = 0; j < output.Width(); ++j)
                    output(iPoint, j) = source(rows(iPoint, 0), j);
        }
    }
    void FillBoundaryTraces(R::ExecutionContext context, const BoundaryTraceRequest &request)
    {
        Validate(context, request, false);
        R::CheckKernelArgumentBudget<BoundaryTraceRequest>();
        if (!request.output.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>((request.faces.RowCount() + 127) / 128, 1024));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (request.acm.RowCount())
        {
            if (request.components == 4)
                TraceRows<4, true><<<blocks, 128, 0, stream>>>(request);
            else if (request.components == 5)
                TraceRows<5, true><<<blocks, 128, 0, stream>>>(request);
            else
                TraceRows<6, true><<<blocks, 128, 0, stream>>>(request);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA ACM boundary launch failed");
            return;
        }
        switch (request.components)
        {
        case 5:
            TraceRows<5><<<blocks, 128, 0, stream>>>(request);
            break;
        case 6:
            TraceRows<6><<<blocks, 128, 0, stream>>>(request);
            break;
        case 7:
            TraceRows<7><<<blocks, 128, 0, stream>>>(request);
            break;
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA boundary trace launch failed");
    }
    void FillPrimitiveBoundary(R::ExecutionContext context, const BoundaryTraceRequest &request)
    {
        Validate(context, request, true);
        R::CheckKernelArgumentBudget<BoundaryTraceRequest>();
        if (!request.output.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>((request.faces.RowCount() + 127) / 128, 1024));
        PrimitiveRows<<<blocks, 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(request);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA primitive boundary launch failed");
    }
    void CopyBoundaryRows(R::ExecutionContext context, BoundaryInput source, BoundaryOutput output, BoundaryRows rows)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA boundary copy needs a CUDA stream");
        R::ValidateViewDevices(context.device, source, output, rows);
        R::CheckKernelArgumentBudget<BoundaryInput, BoundaryOutput, BoundaryRows>();
        DNDS_check_throw_info(source.Width() == output.Width() && rows.RowCount() == output.RowCount(), "boundary copy shapes differ");
        if (!output.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>((output.RowCount() + 127) / 128, 1024));
        CopyRows<<<blocks, 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(source, output, rows);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA boundary copy launch failed");
    }
}
