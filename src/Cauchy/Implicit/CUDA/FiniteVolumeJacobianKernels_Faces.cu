#include "Cauchy/Implicit/CUDA/FiniteVolumeJacobianKernels.hpp"
#include "Cauchy/Implicit/CUDA/FrozenJacobianKernels_Point.hpp"
#include "Cauchy/Boundary/BoundaryJacobian.hpp"
#include "Cauchy/Physics/RiemannPreparation.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "Cauchy/Riemann/RiemannFluxMath.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <int N>
        __device__ NumericalStatus FaceRow(const JacobianFaceInput &q, index iCell,
                                           LinearWriteRows diagonal, bool fullBlock, LinearWriteRows magnitudes)
        {
            const auto &p = q.parameters;
            std::array<real, N * N> own{};
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                    own[i + N * j] = fullBlock ? diagonal(iCell, i + N * j) : (i == j ? diagonal(iCell, i) : 0);
            std::array<real, N> mean{};
            for (int j = 0; j < N; ++j)
                mean[j] = q.means(iCell, j);
            const FlowState left{mean[0], mean[1], mean[2], mean[3], mean[4]};
            for (index entry = q.offsets(iCell, 0); entry < q.offsets(iCell + 1, 0); ++entry)
            {
                const auto &sample = q.samples(entry, 0);
                const real weight = q.faces(sample.face, 6), convection = q.radius(sample.auxiliaryFace, 0), diffusion = q.radius(sample.auxiliaryFace, 1);
                if (!(std::isfinite(weight) && weight > 0 && std::isfinite(convection) && convection >= 0 && std::isfinite(diffusion) && diffusion >= 0))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell, entry};
                const real factor = .5 * p.scale * weight, lambda = convection + diffusion;
                const bool boundary = sample.other == UnInitIndex;
                if (!p.roe)
                    for (int i = 0; i < N; ++i)
                        own[i + N * i] += factor * lambda;
                if (boundary && !p.roe && !q.physicalBoundary)
                    continue; // Euler scalar approximation does not evaluate BCs.
                if (!boundary && !p.roe && sample.other != iCell)
                    continue; // Off-diagonal physical work belongs to its own leaf.

                std::array<real, N> right{}, pressureDerivative{};
                real pressure = 0;
                BoundaryKind kind = BoundaryKind::Interior;
                Vector3 wallVelocity{};
                if (boundary)
                {
                    const auto &condition = q.conditions(sample.condition, 0);
                    kind = ResolveBoundaryKind(condition.kind, sample.ownPoint, p.frame);
                    right = mean;
                    if (kind == BoundaryKind::Interface)
                    {
                        if (sample.boundaryPoint < 0 || sample.boundaryPoint >= q.incoming.RowCount())
                            return NumericalStatus{NumericalError::InvalidInput, 2, iCell, entry};
                        for (int j = 0; j < N; ++j)
                            right[j] = q.incoming(sample.boundaryPoint, j);
                    }
                    else
                    {
                        const auto checked = TryPerfectGasBoundaryState<N>(condition, p.gas, q.rans, right, mean,
                                                                           q.wallMean.RowCount() ? q.wallMean(iCell, 0) : 0,
                                                                           sample.normal, sample.ownPoint, q.time, p.frame, true);
                        if (!checked)
                            return checked.At(iCell, entry);
                    }
                    const FlowState flow{right[0], right[1], right[2], right[3], right[4]};
                    const auto point = TryPerfectGasPoint(p.gas, flow, false);
                    if (!point)
                        return point.status.At(iCell, entry);
                    pressure = point.value.thermo.pressure;
                    const auto derivative = p.gas.PressureDerivative(flow);
                    for (int j = 0; j < 5; ++j)
                        pressureDerivative[j] = derivative[j];
                    if (kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall)
                        wallVelocity = PrepareWallVelocity(condition, sample.ownPoint, p.frame);
                }
                else
                {
                    for (int j = 0; j < N; ++j)
                    {
                        right[j] = q.means(sample.other, j);
                        pressureDerivative[j] = q.pressureDerivative(sample.other, j);
                    }
                    pressure = q.pressure(sample.other, 0);
                }
                const JacobianPointRecord point{sample.other, sample.auxiliaryFace, entry, sample.normal, sample.ownPoint,
                                                sample.rotation, weight, !boundary && sample.changesFrame};
                Vector3 signals{};
                if constexpr (N == 5)
                    if (p.roe)
                    {
                        const auto flow = Detail::RotateJacobianFlow({right[0], right[1], right[2], right[3], right[4]}, point);
                        const auto checked = TryPerfectGasCharacteristics(p.gas, left, flow, sample.normal);
                        if (!checked)
                            return checked.status.At(iCell, entry);
                        if (checked.value.componentwiseFallback)
                            return NumericalStatus{NumericalError::InvalidState, 3, iCell, entry};
                        const auto grid = p.frame.GridVelocity(sample.ownPoint);
                        const real leftPressure = q.pressure(iCell, 0);
                        const Riemann::AcousticState leftData{{left, leftPressure}, std::sqrt(p.gas.gamma * leftPressure / left[0])};
                        const Riemann::AcousticState rightData{{flow, pressure}, std::sqrt(p.gas.gamma * pressure / flow[0])};
                        const auto model = PreparePerfectGasRiemann(p.gas, q.solver, leftData, rightData, leftData, rightData, sample.normal, grid);
                        auto settings = q.fluxSettings;
                        if (q.waveJump.RowCount())
                        {
                            real jump = q.waveJump(iCell, 0);
                            if (!boundary)
                                jump = std::max(jump, q.waveJump(sample.other, 0));
                            settings.dLambda = std::max(settings.dLambda, jump);
                        }
                        if (q.pressureMinimum.RowCount())
                        {
                            real minimum = q.pressureMinimum(iCell, 0);
                            if (!boundary)
                                minimum = std::min(minimum, q.pressureMinimum(sample.other, 0));
                            settings.pressureStencilMin = std::min(settings.pressureStencilMin, minimum);
                        }
                        const auto waves = Riemann::Math::Evaluate(q.solver, model.Input(leftData, rightData, sample.normal, grid), settings);
                        signals = {waves.contact, waves.acousticPlus, waves.acousticMinus};
                        for (int j = 0; j < 3; ++j)
                        {
                            if (!(std::isfinite(signals[j]) && signals[j] >= 0))
                                return NumericalStatus{NumericalError::InvalidState, 4, iCell, entry};
                            magnitudes(entry, j) = signals[j];
                        }
                        const auto &basis = checked.value;
                        for (int j = 0; j < 5; ++j)
                            for (int i = 0; i < 5; ++i)
                            {
                                real value = 0;
                                for (int k = 0; k < 5; ++k)
                                    value += (basis.inverse[i + 5 * k] * signals[k < 3 ? 0 : k - 2]) * basis.forward[k + 5 * j];
                                if (i == j)
                                    value += diffusion;
                                own[i + N * j] += factor * value;
                            }
                    }
                if (!boundary && sample.other != iCell)
                    continue;
                // Apply the complete exterior derivative first, then project
                // the completed diagonal. This preserves cross-component wall
                // reflection even when only some equation columns are active.
                const auto wall = ApproximateFlowExteriorJacobian(kind, sample.normal, wallVelocity);
                for (int j = 0; j < N; ++j)
                {
                    std::array<real, N> direction{}, contribution{};
                    if (!boundary)
                        direction[j] = 1;
                    else
                    {
                        if (j < 5)
                            for (int i = 0; i < 5; ++i)
                                direction[i] = wall[i + 5 * j];
                        if (j >= 5 && (kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall))
                            direction[j] = kind == BoundaryKind::NoSlipWall ? -1 : 1;
                    }
                    const auto checked = Detail::AddFrozenJacobianPoint<N>(p, iCell, point, left, right, pressureDerivative,
                                                                           pressure, convection, diffusion, signals, direction, contribution, false);
                    if (!checked)
                        return checked;
                    for (int i = 0; i < N; ++i)
                        own[i + N * j] += contribution[i];
                }
            }
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                {
                    if (!p.active[i] || !p.active[j])
                        own[i + N * j] = real(i == j);
                    if (!std::isfinite(own[i + N * j]))
                        return NumericalStatus{NumericalError::InvalidState, 5, iCell};
                }
            for (int j = 0; j < (fullBlock ? N * N : N); ++j)
                diagonal(iCell, j) = fullBlock ? own[j] : own[j + N * j];
            return {};
        }
        template <int N>
        __global__ void Faces(JacobianFaceInput input, LinearIndexRows selected, LinearWriteRows diagonal,
                              bool fullBlock, LinearWriteRows magnitudes, JacobianStatus status)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                if (status(iCell, 0))
                    status(iCell, 0) = FaceRow<N>(input, iCell, diagonal, fullBlock, magnitudes);
            }
        }
        template <class View>
        void Table(View view, index rows, index width)
        {
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && view.RowCount() == rows && view.Width() == width,
                                  "CUDA Jacobian face table shape differs");
        }
    }
    void AddJacobianFaces(R::ExecutionContext context, const JacobianFaceInput &input,
                          LinearIndexRows selected, LinearWriteRows diagonal, bool fullBlock,
                          LinearWriteRows magnitudes, JacobianStatus status)
    {
        const auto &p = input.parameters;
        const index rows = diagonal.RowCount(), all = input.means.RowCount();
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && p.components >= 5 && p.components <= 7 &&
                                  (!p.roe || (p.components == 5 && fullBlock)) && std::isfinite(p.scale) && p.scale >= 0 && std::isfinite(input.time) &&
                                  (p.kineticComponent == -1 || (p.components == 7 && p.kineticComponent == 5)),
                              "unsupported CUDA Jacobian face request");
        p.gas.Validate();
        using Kind = Physics::RANS::ModelKind;
        DNDS_check_throw_info((p.components == 5 && input.rans == Kind::None) ||
                                  (p.components == 6 && input.rans == Kind::SA) ||
                                  (p.components == 7 && (input.rans == Kind::KOmegaWilcox || input.rans == Kind::KOmegaSST)),
                              "CUDA Jacobian boundary closure does not match its component profile");
        R::ValidateViewDevices(context.device, input.samples, input.offsets, input.faces, input.conditions,
                               input.means, input.pressure, input.pressureDerivative, input.radius, selected, diagonal, status);
        DNDS_check_throw_info(all >= rows && input.offsets.RowCount() == rows + 1 && status.RowCount() == rows,
                              "CUDA Jacobian face row spaces differ");
        DNDS_check_throw_info(input.means.RowCount() == all && input.means.Width() == p.components &&
                                  (input.means.kind == R::RowLayoutKind::Table ||
                                   (input.means.kind == R::RowLayoutKind::TiledColMajorTable && input.means.stride == 32)),
                              "Jacobian means layout or shape differs");
        Table(input.pressure, all, 1);
        Table(input.pressureDerivative, all, p.components);
        Table(input.radius, input.radius.RowCount(), 2);
        Table(diagonal, rows, fullBlock ? p.components * p.components : p.components);
        if (input.incoming.RowCount())
        {
            input.incoming.ValidateDevice(context.device);
            Table(input.incoming, input.incoming.RowCount(), p.components);
        }
        if (input.wallMean.RowCount())
        {
            input.wallMean.ValidateDevice(context.device);
            DNDS_check_throw_info(input.wallMean.RowCount() >= rows, "CUDA Jacobian wall mean extent differs");
        }
        if (p.roe)
        {
            magnitudes.ValidateDevice(context.device);
            Table(magnitudes, input.samples.RowCount(), 3);
            for (const auto field : {input.waveJump, input.pressureMinimum})
                if (field.RowCount())
                {
                    field.ValidateDevice(context.device);
                    Table(field, all, 1);
                }
        }
        for (const auto *field : {input.means.data, input.pressure.data, input.pressureDerivative.data, input.radius.data, input.incoming.data, input.waveJump.data, input.pressureMinimum.data})
            DNDS_check_throw_info(!field || (field != diagonal.data && (!p.roe || field != magnitudes.data)),
                                  "CUDA Jacobian face output aliases input");
        DNDS_check_throw_info(!diagonal.data || (!p.roe || diagonal.data != magnitudes.data), "CUDA Jacobian face outputs alias");
        R::CheckKernelArgumentBudget<JacobianFaceInput, LinearIndexRows, LinearWriteRows, bool, LinearWriteRows, JacobianStatus>();
        if (!selected.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>(65535, (selected.RowCount() + 127) / 128));
        const auto stream = static_cast<cudaStream_t>(context.stream);
        if (p.components == 5)
            Faces<5><<<blocks, 128, 0, stream>>>(input, selected, diagonal, fullBlock, magnitudes, status);
        else if (p.components == 6)
            Faces<6><<<blocks, 128, 0, stream>>>(input, selected, diagonal, fullBlock, magnitudes, status);
        else
            Faces<7><<<blocks, 128, 0, stream>>>(input, selected, diagonal, fullBlock, magnitudes, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Jacobian face launch failed");
    }
}
