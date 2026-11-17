#include "Cauchy/Discretization/CUDA/ModalPositivityKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "Cauchy/Discretization/CUDA/PositivityReduction.cuh"
#include <cuda_runtime.h>
#include <exception>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <bool Block>
        __device__ void GroupSync()
        {
            if constexpr (Block)
                __syncthreads();
            else
                __syncwarp();
        }
        /// One point stays in the sampling lane's registers. The output-row
        /// ordinal is deliberately ignored: this sink contains only that point.
        struct RegisterPoint
        {
            real values[7]{};
            __device__ real &operator()(index, index j) { return values[j]; }
            __device__ real operator[](index j) const { return values[j]; }
        };
        template <bool Block, int Threads>
        __device__ bool GroupAll(bool value, int *warpResults)
        {
            const bool warp = __all_sync(0xffffffffu, value);
            if constexpr (!Block)
                return warp;
            else
            {
                if (threadIdx.x % 32 == 0)
                    warpResults[threadIdx.x / 32] = warp;
                __syncthreads();
                bool result = true;
                for (int iWarp = 0; iWarp < Threads / 32; ++iWarp)
                    result = result && bool(warpResults[iWarp]);
                // Every reader must finish before another group reduction
                // reuses these shared slots. No FP summation/atomics occur.
                __syncthreads();
                return result;
            }
        }
        __device__ bool RANSPointUnchanged(const ModalPositivityPolicy &policy, const RegisterPoint &point)
        {
#ifndef USE_NS_SA_NUT_REDUCED_ORDER
            if (policy.rans.kind == Physics::RANS::ModelKind::SA)
                return true;
#endif
            for (int j = 0; j < policy.rans.components; ++j)
                if (point[policy.ransFirst + j] < 0)
                    return false;
            return true;
        }
        // A trivial shared slot: no default constructor is run in device shared memory.
        struct DecisionSlot
        {
            NumericalError code;
            int detail, degree;
            index point;
            real beta;
            bool changed;
            __device__ ModalPositivityCellResult Load() const
            {
                return {{code, detail, -1, point}, degree, beta, changed};
            }
            __device__ void Store(ModalPositivityCellResult value)
            {
                code = value.status.code;
                detail = value.status.detail;
                point = value.status.point;
                degree = value.degree;
                beta = value.beta;
                changed = value.changed;
            }
        };
        template <bool Block, int Threads>
        // Advertise the actual block size so the 512-thread variants are not
        // compiled with a register footprint that makes them unlaunchable.
        __global__ __launch_bounds__(Threads) void LimitCells(ModalPositivityRequest request, ModalPositivitySummary partials)
        {
            using namespace ModalPositivityDetail;
            extern __shared__ real scratch[];
            __shared__ DecisionSlot decisions[16];
            __shared__ int done[16];
            __shared__ int warpResults[16];
            const int group = Block ? 0 : int(threadIdx.x) / 32;
            const int lane = Block ? int(threadIdx.x) : int(threadIdx.x) % 32;
            const int width = Block ? int(blockDim.x) : 32;
            const int groups = Block ? 1 : int(blockDim.x) / 32;
            const auto &p = request.policy;
            const index coefficientSize = index(p.components) * p.modes;
            const index pointSize = request.geometry.maximumCellPoints * p.components;
            const index groupSize = p.components + coefficientSize + pointSize +
                                    (p.meanBase ? 0 : coefficientSize + pointSize);
            real *start = scratch + group * groupSize;
            const SpanView<real> mean{start, p.components};
            const SpanView<real> coefficients{start + p.components, coefficientSize};
            const SpanView<real> reference{start + p.components + coefficientSize, p.meanBase ? 0 : coefficientSize};
            const PointRows higher{{start + p.components + coefficientSize + reference.size(), pointSize}, p.components};
            const PointRows lower{{higher.values.data() + pointSize, p.meanBase ? 0 : pointSize}, p.components};
            NumericalStatus failure;
            index limited = 0;
            real minimum = 1;
            for (index iCell = index(blockIdx.x) * groups + group; iCell < request.means.RowCount();
                 iCell += index(gridDim.x) * groups)
            {
                for (index j = lane; j < p.components; j += width)
                    mean[j] = request.means(iCell, j);
                for (index k = lane; k < coefficientSize; k += width)
                    coefficients[k] = request.refresh ? request.initial(iCell, k) : request.coefficients(iCell, k);
                GroupSync<Block>();
                // PP POLICY: test the actual output storage, including refresh.
                // This exclusive operation may stage an as-yet-unaccepted value;
                // it is never published on failure. Reading the stored candidate
                // here lets an unchanged polynomial finish after one sample pass.
                if (request.refresh)
                {
                    for (index k = lane; k < coefficientSize; k += width)
                        request.coefficients(iCell, k) = coefficients[k];
                    GroupSync<Block>();
                    for (index k = lane; k < coefficientSize; k += width)
                        coefficients[k] = request.coefficients(iCell, k);
                    GroupSync<Block>();
                }
                const index points = request.geometry.PointNumber(iCell);
                bool unchanged = true;
                for (index q = lane; q < points; q += width)
                {
                    RegisterPoint point;
                    SamplePoint(request.geometry, iCell, q, mean, coefficients, p.modes, point);
                    // Check before spilling/reducing: finite and thermal checks
                    // include all supplied components; RANS retains its separate
                    // policy. An unacceptable point uses the unchanged slow policy.
                    const bool good = p.bounds.RawThermallyAdmissible(point) &&
                                      p.bounds.ClippedReferenceAdmissible(point) && RANSPointUnchanged(p, point);
                    unchanged = unchanged && good;
                }
                unchanged = GroupAll<Block, Threads>(unchanged, warpResults);
                if (lane == 0)
                {
                    bool meanGood = p.bounds.Admissible(mean);
#ifndef USE_NS_SA_NUT_REDUCED_ORDER
                    if (p.rans.kind != Physics::RANS::ModelKind::SA)
#endif
                        for (int j = 0; j < p.rans.components; ++j)
                            meanGood = meanGood && mean[p.ransFirst + j] >= 0;
                    done[group] = unchanged && meanGood;
                    if (done[group])
                        decisions[group].Store({{}, p.maxOrder, 1, false});
                }
                GroupSync<Block>();
                if (!done[group])
                {
                    // Rare limiting/error path: materialize only this cell's
                    // scratch and preserve base selection, beta and first-error
                    // diagnostics exactly. No global sampled-state field exists.
                    for (index q = lane; q < points; q += width)
                        SamplePoint(request.geometry, iCell, q, mean, coefficients, p.modes, higher);
                    GroupSync<Block>();
                    if (lane == 0)
                    {
                        const Sampler sample{request.geometry, iCell, points, mean, p.modes};
                        const auto result = SelectAndCompress(p, mean, coefficients, reference, higher, lower, points, sample);
                        decisions[group].Store(result);
                        done[group] = !result.status || (!request.refresh && !result.changed);
                    }
                }
                GroupSync<Block>();
                // PP POLICY: recovery is local to this cell/ownership group.
                // Every trial stores, orders and resamples actual coefficients.
                // No global sampled-state field or host-observed retry is used.
                for (int trial = 0; trial <= 1000 && !done[group]; ++trial)
                {
                    for (index k = lane; k < coefficientSize; k += width)
                        request.coefficients(iCell, k) = coefficients[k];
                    GroupSync<Block>();
                    for (index k = lane; k < coefficientSize; k += width)
                        coefficients[k] = request.coefficients(iCell, k);
                    GroupSync<Block>();
                    bool storedGood = true;
                    for (index q = lane; q < points; q += width)
                    {
                        RegisterPoint point;
                        SamplePoint(request.geometry, iCell, q, mean, coefficients, p.modes, point);
                        const bool good = p.bounds.ClippedReferenceAdmissible(point);
                        storedGood = storedGood && good;
                    }
                    storedGood = GroupAll<Block, Threads>(storedGood, warpResults);
                    if (lane == 0)
                    {
                        if (storedGood)
                            done[group] = 1;
                        else if (trial == 1000)
                        {
                            auto result = decisions[group].Load();
                            result.status = NumericalStatus{NumericalError::InvalidState, 7};
                            decisions[group].Store(result);
                            done[group] = 1;
                        }
                        else
                        {
                            auto result = decisions[group].Load();
                            result.status = Recover(p, result, coefficients, reference);
                            decisions[group].Store(result);
                            done[group] = !result.status;
                        }
                    }
                    GroupSync<Block>();
                }
                if (lane == 0)
                {
                    const auto result = decisions[group].Load();
                    request.degree(iCell, 0) = result.degree;
                    request.beta(iCell, 0) = result.beta;
                    if (!result.status)
                        failure = PositivityReductionDetail::FirstFailure{}(failure, result.status.At(iCell, result.status.point));
                    else if (result.beta < 1)
                    {
                        ++limited;
                        minimum = std::min(minimum, result.beta);
                    }
                }
                GroupSync<Block>();
            }
            PositivityReductionDetail::PublishPositivitySummary<Threads>({failure, limited, minimum}, partials);
        }
        __global__ void Finalize(ModalPositivitySummary partials, ModalPositivitySummary output)
        {
            NumericalStatus status;
            index count = 0;
            real minimum = 1;
            for (index iRow = threadIdx.x; iRow < partials.status.RowCount(); iRow += blockDim.x)
            {
                status = PositivityReductionDetail::FirstFailure{}(status, partials.status(iRow, 0));
                count += partials.count(iRow, 0);
                minimum = std::min(minimum, partials.minimum(iRow, 0));
            }
            PositivityReductionDetail::PublishPositivitySummary<128>({status, count, minimum}, output);
        }
        void CheckSummary(ModalPositivitySummary summary, R::Device device, index rows)
        {
            summary.ValidateDevice(device);
            DNDS_check_throw_info(summary.status.RowCount() == rows && summary.count.RowCount() == rows && summary.minimum.RowCount() == rows,
                                  "modal PP summary shape differs from launch");
        }
        size_t SharedBytes(const ModalPositivityPolicy &p, index points, ModalPositivityLaunch launch)
        {
            const index groups = launch.mapping == ModalPositivityMapping::BlockPerCell ? 1 : launch.threads / 32;
            DNDS_check_throw_info(points >= 0 && p.components > 0 && points <= INT_MAX / (groups * p.components * (p.meanBase ? 1 : 2) * index(sizeof(real))),
                                  "modal PP shared-memory extent overflows native launch sizing");
            return size_t(groups) * size_t(p.components + index(p.components) * p.modes + points * p.components + (p.meanBase ? 0 : index(p.components) * p.modes + points * p.components)) * sizeof(real);
        }
        int Selection(ModalPositivityLaunch launch)
        {
            return (launch.mapping == ModalPositivityMapping::BlockPerCell ? 1024 : 0) + launch.threads;
        }
        template <bool Block, int Threads>
        void Prepare(R::Device device, size_t bytes)
        {
            cudaDeviceProp properties{};
            DNDS_check_throw_info(cudaGetDeviceProperties(&properties, device.ordinal) == cudaSuccess,
                                  "cannot query modal PP shared-memory capability");
            cudaFuncAttributes attributes{};
            DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, LimitCells<Block, Threads>) == cudaSuccess &&
                                      bytes + attributes.sharedSizeBytes <= size_t(properties.sharedMemPerBlockOptin),
                                  "modal PP candidate exceeds the device shared-memory limit");
            if (bytes + attributes.sharedSizeBytes > size_t(properties.sharedMemPerBlock))
                DNDS_check_throw_info(cudaFuncSetAttribute(LimitCells<Block, Threads>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           int(bytes)) == cudaSuccess,
                                      "cannot enable modal PP candidate dynamic shared memory");
        }
        template <bool Block, int Threads>
        void Launch(R::ExecutionContext context, const ModalPositivityRequest &request,
                    ModalPositivitySummary partials, ModalPositivityLaunch launch)
        {
            LimitCells<Block, Threads><<<unsigned(partials.status.RowCount()), unsigned(Threads), launch.sharedBytes,
                                         static_cast<cudaStream_t>(context.stream)>>>(request, partials);
        }
    }
    index ModalPositivitySummaryRows(index cells, ModalPositivityLaunch launch)
    {
        DNDS_check_throw_info(cells >= 0 && (launch.threads == 128 || launch.threads == 256 || launch.threads == 512) &&
                                  launch.blocks >= 0 && (launch.mapping == ModalPositivityMapping::WarpPerCell || launch.mapping == ModalPositivityMapping::BlockPerCell),
                              "unsupported modal PP mapping/block/grid selection");
        const index groups = launch.mapping == ModalPositivityMapping::WarpPerCell ? launch.threads / 32 : 1;
        return std::min((cells + groups - 1) / groups, launch.blocks ? launch.blocks : index(1024));
    }
    void PrepareModalPositivityLaunch(R::Device device, const ModalPositivityPolicy &policy,
                                      index maximumCellPoints, ModalPositivityLaunch &launch)
    {
        DNDS_check_throw_info(device.space == R::MemorySpace::CUDA && maximumCellPoints >= 0 &&
                                  policy.components >= 5 && policy.components <= 7 && policy.modes > 0 && policy.modes <= 19,
                              "invalid modal PP launch preparation profile");
        (void)ModalPositivitySummaryRows(0, launch);
        // Capability/opt-in attributes are native-device scoped. Setup may be
        // placing a plan for another device while the controller retains its
        // previous default; restore that default on both success and unwind.
        int previous = -1;
        DNDS_check_throw_info(cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device.ordinal) == cudaSuccess,
                              "cannot select modal PP preparation device");
        struct RestoreDevice
        {
            int previous;
            ~RestoreDevice()
            {
                if (cudaSetDevice(previous) != cudaSuccess)
                    std::terminate();
            }
        } restore{previous};
        const size_t bytes = SharedBytes(policy, maximumCellPoints, launch);
        if (launch.mapping == ModalPositivityMapping::BlockPerCell)
        {
            if (launch.threads == 128)
                Prepare<true, 128>(device, bytes);
            else if (launch.threads == 256)
                Prepare<true, 256>(device, bytes);
            else
                Prepare<true, 512>(device, bytes);
        }
        else
        {
            if (launch.threads == 128)
                Prepare<false, 128>(device, bytes);
            else if (launch.threads == 256)
                Prepare<false, 256>(device, bytes);
            else
                Prepare<false, 512>(device, bytes);
        }
        launch.sharedBytes = bytes;
        launch.preparedDevice = device.ordinal;
        launch.preparedSelection = Selection(launch);
    }
    void LimitPositiveModalCoefficients(R::ExecutionContext context, const ModalPositivityRequest &request,
                                        ModalPositivitySummary partials, ModalPositivityLaunch launch)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "modal PP requires a CUDA context");
        R::ValidateViewDevices(context.device, request.geometry, request.means, request.coefficients, request.beta, request.degree);
        const auto &p = request.policy;
        DNDS_check_throw_info(p.components >= 5 && p.components <= 7 && p.bounds.components == p.components &&
                                  p.bounds.flowFirst == 0 && p.bounds.independent == 0 && p.bounds.speciesEnergies[0].empty() &&
                                  p.bounds.speciesEnergies[1].empty() && p.modes > 0 && p.modes <= 19 &&
                                  p.maxOrder > 0 && p.maxOrder <= 3 && p.rans.components + 5 == p.components &&
                                  p.betaPower >= 0 && p.betaCutoff >= 0 && p.betaCutoff <= 1 &&
                                  (p.clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase ||
                                   p.clippedPolicy == ReconstructionClippedReferencePolicy::Backtracking) &&
                                  request.geometry.basis.polynomial.size == p.modes + 1 && p.modeEnd[p.maxOrder] == p.modes &&
                                  request.means.kind == R::RowLayoutKind::Table && request.coefficients.kind == R::RowLayoutKind::Table &&
                                  request.geometry.stencil.RowCount() == 1 &&
                                  request.means.Width() == p.components && request.coefficients.Width() == p.components * p.modes &&
                                  request.coefficients.RowCount() == request.means.RowCount() &&
                                  request.beta.RowCount() == request.means.RowCount() && request.degree.RowCount() == request.means.RowCount() &&
                                  request.geometry.volumeOffsets.RowCount() == request.means.RowCount() + 1 && request.geometry.maximumCellPoints >= 0,
                              "modal PP perfect-gas shape/physics mismatch");
        if (request.refresh)
        {
            request.initial.ValidateDevice(context.device);
            DNDS_check_throw_info(request.initial.RowCount() == request.coefficients.RowCount() &&
                                      request.initial.Width() == request.coefficients.Width(),
                                  "modal PP refreshed coefficient shape differs");
        }
        const index rows = ModalPositivitySummaryRows(request.means.RowCount(), launch);
        CheckSummary(partials, context.device, rows);
        if (launch.preparedDevice != context.device.ordinal || launch.preparedSelection != Selection(launch) ||
            launch.sharedBytes != SharedBytes(p, request.geometry.maximumCellPoints, launch))
            PrepareModalPositivityLaunch(context.device, p, request.geometry.maximumCellPoints, launch);
        R::CheckKernelArgumentBudget<ModalPositivityRequest, ModalPositivitySummary>();
        if (!rows)
            return;
        if (launch.mapping == ModalPositivityMapping::BlockPerCell)
        {
            if (launch.threads == 128)
                Launch<true, 128>(context, request, partials, launch);
            else if (launch.threads == 256)
                Launch<true, 256>(context, request, partials, launch);
            else
                Launch<true, 512>(context, request, partials, launch);
        }
        else
        {
            if (launch.threads == 128)
                Launch<false, 128>(context, request, partials, launch);
            else if (launch.threads == 256)
                Launch<false, 256>(context, request, partials, launch);
            else
                Launch<false, 512>(context, request, partials, launch);
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "modal PP producer launch failed");
    }
    void FinalizeModalPositivity(R::ExecutionContext context, ModalPositivitySummary partials, ModalPositivitySummary result)
    {
        CheckSummary(partials, context.device, partials.status.RowCount());
        CheckSummary(result, context.device, 1);
        Finalize<<<1, 128, 0, static_cast<cudaStream_t>(context.stream)>>>(partials, result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "modal PP summary finalization failed");
    }
}
