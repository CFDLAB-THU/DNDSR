#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "Cauchy/Discretization/CUDA/PositivityReduction.cuh"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        using BaseInput = R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<2>>;
        using BaseOutput = R::ArrayView<real, R::AccessBoundsChecks, R::FixedRows<2>>;
        using ScalarOutput = R::ArrayView<real, R::AccessBoundsChecks, R::FixedRows<1>>;
        using StatusInput = R::ArrayView<const NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>>;
        DNDS_DEVICE_CALLABLE FlowState Load(LVRInput input, index row)
        {
            FlowState value;
            for (int j = 0; j < 5; ++j)
                value[j] = input(row, j);
            return value;
        }
        __global__ void Target(LVRStencil descriptor, LVRInput means, LVROutput coefficients, int dimension,
                               ThermalBounds bounds, int power, real cutoff, LVRStatus status)
        {
            const auto &stencil = descriptor(0, 0);
            const int modes = int(coefficients.Width() / 5);
            NumericalStatus result;
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const auto cell = LimitLVRTargetCell(stencil, means, coefficients, iCell, dimension, modes, bounds, power, cutoff);
                if (result && !cell)
                    result = cell;
            }
            // Producer epilogue: only compact block errors leave the PP
            // kernel. The shared helper handles tails and deterministic errors.
            PositivityReductionDetail::PublishStatusSummary<128>(result, status);
        }
        __global__ void Scalars(LVRInput means, LVROutput scalars, PerfectGas gas, LVRStatus status)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const auto state = Load(means, iCell);
                const auto point = TryPerfectGasPoint(gas, state, false);
                status(iCell, 0) = point ? NumericalStatus{} : point.status.At(iCell);
                if (point)
                {
                    scalars(iCell, 0) = point.value.thermo.pressure;
                    scalars(iCell, 1) = state[0];
                }
            }
        }
        __global__ void Sensors(LVRStencil descriptor, LVRMeans means, LVRMeans target, int dimension,
                                PerfectGas gas, ThermalBounds bounds, CFV::LimitedVRKernelParameters settings,
                                LVROutput output, LVRStatus status)
        {
            const auto &stencil = descriptor(0, 0);
            const int modes = int(target.owned.Width() / 5);
            for (index iAux = index(blockIdx.x) * blockDim.x + threadIdx.x; iAux < output.RowCount(); iAux += index(gridDim.x) * blockDim.x)
            {
                const index iFace = stencil.stencilFaces(iAux, 0);
                const auto parents = stencil.topology.FaceCells(iFace);
                if (parents[1] == UnInitIndex)
                {
                    for (int j = 0; j < 3; ++j)
                        output(iAux, j) = 0;
                    status(iAux, 0) = {};
                    continue;
                }
                const auto center = stencil.metric.GetFaceCent(iFace);
                const auto left = EvaluateLVRLinearTarget(stencil, means, target, parents[0], center, dimension, modes);
                auto right = EvaluateLVRLinearTarget(stencil, means, target, parents[1],
                                                     stencil.metric.periodic.FacePoint(iFace, 1, center), dimension, modes);
                if (stencil.metric.periodic.Rotates(iFace))
                {
                    const auto rotation = stencil.metric.periodic.OtherRotation(iFace, 0);
                    const auto old = right;
                    for (int d = 0; d < 3; ++d)
                    {
                        right[1 + d] = 0;
                        for (int k = 0; k < 3; ++k)
                            right[1 + d] += rotation(d, k) * old[1 + k];
                    }
                }
                if (!(ThermallyAdmissible(left, bounds) && ThermallyAdmissible(right, bounds)))
                {
                    status(iAux, 0) = NumericalStatus{NumericalError::InvalidState, 1}.At(parents[0], iFace);
                    continue;
                }
                const auto l = gas.Primitive(left), r = gas.Primitive(right);
                const auto normal = stencil.metric.GetFaceMeanNorm(iFace);
                real unLeft = 0, unRight = 0;
                for (int d = 0; d < 3; ++d)
                {
                    unLeft += l.velocity[d] * normal[d];
                    unRight += r.velocity[d] * normal[d];
                }
                const auto sensor = CFV::Math::EvaluateLimitedVRFaceSensor(l.density, r.density, l.pressure, r.pressure,
                                                                           unLeft, unRight, l.soundSpeed, r.soundSpeed, settings);
                const auto support = CFV::Math::EvaluateLimitedVRFaceSensor(means(parents[0], 0), means(parents[1], 0),
                                                                            1, 1, 0, 0, 1, 1, settings)
                                         .contact;
                output(iAux, 0) = sensor.shock;
                output(iAux, 1) = sensor.contact;
                output(iAux, 2) = support;
                status(iAux, 0) = {};
            }
        }
        __global__ void Gather(LVRStencil descriptor, LVRInput sensors, StatusInput faceStatus,
                               LVROutput branches, BaseOutput base, LVRStatus status)
        {
            const auto &stencil = descriptor(0, 0);
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < branches.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                std::array<real, 3> values{};
                NumericalStatus result;
                for (const index iFace : stencil.topology.CellFaces(iCell))
                {
                    const index iAux = stencil.routing(iFace, 0).auxiliaryFace;
                    if (!faceStatus(iAux, 0))
                    {
                        result = faceStatus(iAux, 0).At(iCell, iFace);
                        break;
                    }
                    for (int j = 0; j < 3; ++j)
                        values[j] = std::max(values[j], sensors(iAux, j));
                }
                status(iCell, 0) = result;
                if (result)
                {
                    for (int j = 0; j < 3; ++j)
                        branches(iCell, j) = values[j];
                    for (int j = 0; j < 2; ++j)
                        base(iCell, j) = values[j];
                }
            }
        }
        __global__ void Merge(BaseInput base, LVROutput seeds, int column)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < seeds.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                seeds(iCell, column) = std::max(seeds(iCell, column), base(iCell, column));
        }
        __global__ void Publish(LVRInput branches, BaseOutput history, ScalarOutput penalty)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < branches.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                for (int j = 0; j < 2; ++j)
                    history(iCell, j) = branches(iCell, j);
                penalty(iCell, 0) = std::max(branches(iCell, 0), branches(iCell, 1));
            }
        }
        unsigned Blocks(index count) { return unsigned(std::min<index>((count + 127) / 128, 1024)); }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA physical LVR launch failed"); }
        template <class... Views>
        void CheckDevice(R::ExecutionContext context, const Views &...views)
        {
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA LVR requires a CUDA context");
            R::ValidateViewDevices(context.device, views...);
        }
        template <class View>
        void Table(View view, index rows, index width)
        {
            DNDS_check_throw_info(view.RowCount() == rows && view.Width() == width && view.kind == R::RowLayoutKind::Table,
                                  "CUDA physical LVR table shape differs");
        }
    }
    void LimitLVRTarget(R::ExecutionContext c, LVRStencil stencil, LVRInput means, LVROutput coefficients,
                        int dimension, ThermalBounds bounds, int power, real cutoff, LVRStatus status)
    {
        CheckDevice(c, stencil, means, coefficients, status);
        Table(means, means.RowCount(), 5);
        Table(coefficients, means.RowCount(), coefficients.Width());
        DNDS_check_throw_info(stencil.RowCount() == 1 && (dimension == 2 || dimension == 3) &&
                                  coefficients.Width() % 5 == 0 && coefficients.Width() >= 5 * dimension &&
                                  status.RowCount() == Blocks(means.RowCount()) && power >= 0 && std::isfinite(cutoff) && cutoff >= 0 && cutoff <= 1,
                              "invalid CUDA LVR target controls/shape");
        R::CheckKernelArgumentBudget<LVRStencil, LVRInput, LVROutput, int, ThermalBounds, int, real, LVRStatus>();
        if (!means.RowCount())
            return;
        Target<<<Blocks(means.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(stencil, means, coefficients, dimension, bounds, power, cutoff, status);
        CheckLaunch();
    }
    void FillLVRScalars(R::ExecutionContext c, LVRInput means, LVROutput scalars, PerfectGas gas, LVRStatus status)
    {
        CheckDevice(c, means, scalars, status);
        gas.Validate();
        Table(means, means.RowCount(), 5);
        Table(scalars, means.RowCount(), 2);
        DNDS_check_throw_info(status.RowCount() == means.RowCount(), "invalid CUDA LVR scalar status shape");
        R::CheckKernelArgumentBudget<LVRInput, LVROutput, PerfectGas, LVRStatus>();
        if (!means.RowCount())
            return;
        Scalars<<<Blocks(means.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(means, scalars, gas, status);
        CheckLaunch();
    }
    void BuildLVRFaceSensors(R::ExecutionContext c, LVRStencil stencil, LVRMeans means, LVRMeans target, int dimension,
                             PerfectGas gas, ThermalBounds bounds, CFV::LimitedVRKernelParameters settings,
                             LVROutput sensors, LVRStatus status)
    {
        CheckDevice(c, stencil, means, target, sensors, status);
        gas.Validate();
        Table(means.owned, means.owned.RowCount(), 5);
        Table(target.owned, means.owned.RowCount(), target.owned.Width());
        if (means.ghost.RowCount())
        {
            Table(means.ghost, means.ghost.RowCount(), 5);
            Table(target.ghost, means.ghost.RowCount(), target.owned.Width());
        }
        Table(sensors, sensors.RowCount(), 3);
        DNDS_check_throw_info(stencil.RowCount() == 1 && (dimension == 2 || dimension == 3) &&
                                  target.owned.Width() % 5 == 0 && target.owned.Width() >= 5 * dimension &&
                                  status.RowCount() == sensors.RowCount(),
                              "invalid CUDA LVR face status/target shape");
        R::CheckKernelArgumentBudget<LVRStencil, LVRMeans, LVRMeans, int, PerfectGas, ThermalBounds, CFV::LimitedVRKernelParameters, LVROutput, LVRStatus>();
        if (!sensors.RowCount())
            return;
        Sensors<<<Blocks(sensors.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(stencil, means, target, dimension, gas, bounds, settings, sensors, status);
        CheckLaunch();
    }
    void GatherLVRBranches(R::ExecutionContext c, LVRStencil stencil, LVRInput sensors, StatusInput faceStatus,
                           LVROutput branches, BaseOutput base, LVRStatus status)
    {
        CheckDevice(c, stencil, sensors, faceStatus, branches, base, status);
        Table(sensors, sensors.RowCount(), 3);
        Table(branches, branches.RowCount(), 3);
        DNDS_check_throw_info(stencil.RowCount() == 1 && base.RowCount() == branches.RowCount() && status.RowCount() == branches.RowCount() &&
                                  faceStatus.RowCount() == sensors.RowCount(),
                              "invalid CUDA LVR gather shape");
        R::CheckKernelArgumentBudget<LVRStencil, LVRInput, StatusInput, LVROutput, BaseOutput, LVRStatus>();
        if (!branches.RowCount())
            return;
        Gather<<<Blocks(branches.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(stencil, sensors, faceStatus, branches, base, status);
        CheckLaunch();
    }
    void MergeLVRBase(R::ExecutionContext c, BaseInput base, LVROutput seeds, int column)
    {
        CheckDevice(c, base, seeds);
        Table(seeds, base.RowCount(), 3);
        DNDS_check_throw_info(column == 0 || column == 1, "invalid LVR base column");
        R::CheckKernelArgumentBudget<BaseInput, LVROutput, int>();
        if (!seeds.RowCount())
            return;
        Merge<<<Blocks(seeds.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(base, seeds, column);
        CheckLaunch();
    }
    void PublishLVRPenalty(R::ExecutionContext c, LVRInput branches, BaseOutput history, ScalarOutput penalty)
    {
        CheckDevice(c, branches, history, penalty);
        Table(branches, history.RowCount(), 3);
        DNDS_check_throw_info(penalty.RowCount() == branches.RowCount(), "invalid LVR penalty shape");
        R::CheckKernelArgumentBudget<LVRInput, BaseOutput, ScalarOutput>();
        if (!branches.RowCount())
            return;
        Publish<<<Blocks(branches.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(branches, history, penalty);
        CheckLaunch();
    }
}
