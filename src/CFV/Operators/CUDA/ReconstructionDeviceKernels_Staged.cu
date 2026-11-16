#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include "DNDS/Algebra/CUDA/FP64Matrix.cuh"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    namespace
    {
        // This projection omits limiter/basis buffers from the launch arguments.
        // Matrix tables are checked once by the wrapper; state views retain
        // their own strides, including supported SoA/AoSoA DOF views.
        template <class T, class Layout>
        using Table = R::ArrayView<const T, R::AccessBoundsChecks, Layout>;
        template <class Layout, class T>
        Table<T, Layout> Project(R::ArrayView<const T> view)
        {
            return {view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
        }
        struct Sweep
        {
            Table<index, R::FixedRows<1>> offsets, neighbors, faces, boundaryOffsets;
            Table<uint8_t, R::FixedRows<1>> rotationActive, faceSide;
            Table<real, R::TableRows> matrices, vectors, injection, inverse;
            Table<real, R::FixedRows<9>> rotation;
            real relax;
            bool rotated, triangle, faceCoupling;
            explicit Sweep(const ReconstructionDeviceView &p)
                : offsets(Project<R::FixedRows<1>>(p.offsets)), neighbors(Project<R::FixedRows<1>>(p.neighbors)),
                  faces(Project<R::FixedRows<1>>(p.faces)), boundaryOffsets(Project<R::FixedRows<1>>(p.boundaryOffsets)),
                  rotationActive(Project<R::FixedRows<1>>(p.rotationActive)), faceSide(Project<R::FixedRows<1>>(p.faceSide)),
                  matrices(Project<R::TableRows>(p.matrixAInvB)), vectors(Project<R::TableRows>(p.vectorAInvB)),
                  injection(Project<R::TableRows>(p.boundaryInjection)), inverse(Project<R::TableRows>(p.localInverse)),
                  rotation(Project<R::FixedRows<9>>(p.rotation)), relax(p.relax), rotated(p.rotated),
                  triangle(p.inverseTriangle), faceCoupling(p.faceCoupling) {}
        };
        template <bool Warp>
        __device__ void Join()
        {
            if constexpr (Warp)
                __syncwarp();
            else
                __syncthreads();
        }
        template <bool Async>
        __device__ void CopyWord(real *destination, const real *source)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (Async)
            {
                const unsigned address = unsigned(__cvta_generic_to_shared(destination));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(address), "l"(source) : "memory");
            }
            else
#endif
                *destination = *source;
        }
        template <bool Async>
        __device__ void CommitCopies()
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (Async)
                asm volatile("cp.async.commit_group;" ::: "memory");
#endif
        }
        template <bool Warp, bool Async>
        __device__ void WaitCopies()
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (Async)
                asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
            Join<Warp>();
        }
        template <int N, bool Warp, bool Async>
        __device__ void StageEntry(Sweep plan, index entry, real *slot,
                                   R::PairReadView<real> mean, R::PairReadView<real> input,
                                   int components, int rank, int teamSize)
        {
            const index other = plan.neighbors(entry, 0);
            if (other == UnInitIndex)
                return; // Boundary values are consumed through the injection map.
            const index matrixRow = plan.faceCoupling ? plan.faces(entry, 0) : entry;
            const auto matrix = plan.matrices[matrixRow];
            for (int i = rank; i < N * N; i += teamSize)
                CopyWord<Async>(slot + i, &matrix[i]);
            const auto values = input[other];
            for (int i = rank; i < N * components; i += teamSize)
                CopyWord<Async>(slot + N * N + i, &values[i]);
        }
        template <int N>
        __device__ real Coefficient(Sweep plan, DevicePeriodicBlocks vectors, index entry,
                                    const real *coefficients, int component, int mode)
        {
            const auto first = plan.rotated && plan.rotationActive(entry, 0) ? vectors.First(component) : -1;
            if (first < 0)
                return coefficients[component * N + mode];
            real value = 0;
            for (int d = 0; d < 3; ++d)
                value += plan.rotation(entry, d * 3 + component - first) * coefficients[(first + d) * N + mode];
            return value;
        }
        template <int N>
        __device__ real OtherTerms(Sweep plan, DevicePeriodicBlocks vectors, index iCell,
                                   int component, int mode, R::PairReadView<real> mean,
                                   R::ArrayView<const real> boundary)
        {
            real value = 0;
            const auto first = plan.rotated ? vectors.First(component) : -1;
            for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
            {
                const auto other = plan.neighbors(entry, 0);
                if (other != UnInitIndex)
                {
                    real average = mean(other, component);
                    if (first >= 0 && plan.rotationActive(entry, 0))
                    {
                        average = 0;
                        for (int d = 0; d < 3; ++d)
                            average += plan.rotation(entry, d * 3 + component - first) * mean(other, first + d);
                    }
                    value += plan.vectors(entry, mode) * (average - mean(iCell, component));
                }
                else
                    for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                        value += plan.injection(q, mode) * (boundary(q, component) - mean(iCell, component));
            }
            return value;
        }
        /** @brief A team owns one cell and all output modes/variables.
         * @details Both pipelines have exactly two slots. Neighbor buffering
         * prefetches the next incidence; cell-row buffering prefetches all
         * incidences of the next grid-stride cell. Every staged matrix is used
         * by the complete modal batch before reuse. No block-wide barriers
         * occur in the warp-per-cell specialization, including tail warps.
         */
        template <int N, bool Warp, bool CellRow, bool Async, bool Tensor>
        __global__ void StagedJacobi(Sweep plan, bool raw, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                     R::PairReadView<real> mean, R::PairReadView<real> input,
                                     R::ArrayView<const real> boundary, R::ArrayView<real> output,
                                     R::ArrayView<const int> status, int capacity, int teamWords)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if (status(0, 0))
                return;
            extern __shared__ __align__(32) real shared[];
            const int components = int(output.width / N);
            const int entryWords = N * N + N * components;
            const int team = Warp ? threadIdx.x / 32 : 0;
            const int rank = Warp ? threadIdx.x % 32 : threadIdx.x;
            const int teamSize = Warp ? 32 : blockDim.x;
            const int teams = Warp ? blockDim.x / 32 : 1;
            real *slots = shared + team * teamWords;
            real *right = slots + 2 * capacity * entryWords;
            real *inverse = slots + ((2 * capacity * entryWords + N * components + 3) / 4) * 4;
            auto &tile = *reinterpret_cast<Algebra::CUDA::FP64Tile8 *>(inverse + (raw ? ((N * N + 3) / 4) * 4 : 0));
            const index stride = index(gridDim.x) * teams;
            const index firstWork = index(blockIdx.x) * teams + team;
            auto load = [&](index work, int slot, index entry)
            {
                if constexpr (CellRow)
                {
                    const auto cell = selection.Cell(work);
                    const auto begin = plan.offsets(cell, 0), end = plan.offsets(cell + 1, 0);
                    for (index j = begin; j < end; ++j)
                        StageEntry<N, Warp, Async>(plan, j, slots + (slot * capacity + j - begin) * entryWords, mean, input, components, rank, teamSize);
                }
                else
                    StageEntry<N, Warp, Async>(plan, entry, slots + slot * entryWords, mean, input, components, rank, teamSize);
                CommitCopies<Async>();
            };
            if constexpr (CellRow)
                if (firstWork < selection.count)
                    load(firstWork, 0, 0);
            int cellSlot = 0;
            for (index work = firstWork; work < selection.count; work += stride, cellSlot = 1 - cellSlot)
            {
                const auto iCell = selection.Cell(work);
                const index begin = plan.offsets(iCell, 0), end = plan.offsets(iCell + 1, 0);
                real accumulated[5]{};
                Algebra::CUDA::FP64Accumulator8 products[(N + 7) / 8];
                if constexpr (CellRow)
                {
                    WaitCopies<Warp, Async>();
                    if (work + stride < selection.count)
                        load(work + stride, 1 - cellSlot, 0);
                }
                else if (begin < end)
                    load(work, 0, begin);
                for (index entry = begin; entry < end; ++entry)
                {
                    const int slot = CellRow ? cellSlot : int((entry - begin) % 2);
                    if constexpr (!CellRow)
                    {
                        WaitCopies<Warp, Async>();
                        if (entry + 1 < end)
                            load(work, 1 - slot, entry + 1);
                    }
                    const real *matrix = slots + (slot * capacity + (CellRow ? entry - begin : 0)) * entryWords;
                    const real *coefficients = matrix + N * N;
                    const auto other = plan.neighbors(entry, 0);
                    const bool rotated = plan.rotated && plan.rotationActive(entry, 0);
                    const bool transpose = plan.faceCoupling && plan.faceSide(entry, 0);
                    if (other != UnInitIndex && (!raw || other != iCell || rotated))
                    {
                        if constexpr (Tensor)
                        {
                            // All output-mode fragments remain live across the
                            // neighbor loop. A matrix is never globally restaged
                            // for the next 8-mode fragment.
                            for (int modeTile = 0; modeTile < (N + 7) / 8; ++modeTile)
                                for (int k = 0; k < N; k += 4)
                                {
                                    const int mode = modeTile * 8 + rank / 4, ka = k + rank % 4;
                                    tile.a[rank] = mode < N && ka < N ? matrix[transpose ? mode * N + ka : ka * N + mode] : 0;
                                    const int component = rank % 8, kb = k + rank / 8;
                                    tile.b[rank] = component < components && kb < N ? Coefficient<N>(plan, vectors, entry, coefficients, component, kb) : 0;
                                    __syncwarp();
                                    products[modeTile].AddProduct(tile);
                                    __syncwarp();
                                }
                        }
                        else
#    pragma unroll
                            for (int k = 0; k < 5; ++k)
                            {
                                const int i = rank + k * teamSize;
                                if (i < N * components)
                                {
                                    real product = 0;
                                    for (int j = 0; j < N; ++j)
                                        product += matrix[transpose ? (i % N) * N + j : j * N + i % N] * Coefficient<N>(plan, vectors, entry, coefficients, i / N, j);
                                    accumulated[k] += product;
                                }
                            }
                    }
                    Join<Warp>(); // The slot can be reused after every consumer.
                }
                if constexpr (Tensor)
                {
                    for (int m = 0; m < (N + 7) / 8; ++m)
                    {
                        products[m].Store(tile);
                        __syncwarp();
                        for (int i = rank; i < 64; i += 32)
                            if (m * 8 + i / 8 < N && i % 8 < components)
                                right[(i % 8) * N + m * 8 + i / 8] = tile.c[i];
                        __syncwarp();
                    }
                }
                else
                    for (int k = 0; k < 5; ++k)
                        if (rank + k * teamSize < N * components)
                            right[rank + k * teamSize] = accumulated[k];
                Join<Warp>();
                for (int i = rank; i < N * components; i += teamSize)
                    right[i] += OtherTerms<N>(plan, vectors, iCell, i / N, i % N, mean, boundary);
                if (raw)
                {
                    const auto row = plan.inverse[iCell];
                    for (int i = rank; i < N * N; i += teamSize)
                        inverse[i] = row[plan.triangle ? Math::SymmetricTriangleEntry(i % N, i / N) : i];
                }
                Join<Warp>();
                if constexpr (Tensor)
                {
                    if (raw)
                    {
                        for (int m = 0; m < (N + 7) / 8; ++m)
                        {
                            Algebra::CUDA::FP64Accumulator8 update;
                            for (int k = 0; k < N; k += 4)
                            {
                                const int mode = m * 8 + rank / 4, ka = k + rank % 4;
                                tile.a[rank] = mode < N && ka < N ? inverse[ka * N + mode] : 0;
                                const int c = rank % 8, kb = k + rank / 8;
                                tile.b[rank] = c < components && kb < N ? right[c * N + kb] : 0;
                                __syncwarp();
                                update.AddProduct(tile);
                                __syncwarp();
                            }
                            update.Store(tile);
                            __syncwarp();
                            for (int i = rank; i < 64; i += 32)
                            {
                                const int mode = m * 8 + i / 8, c = i % 8;
                                if (mode < N && c < components)
                                    output(iCell, c * N + mode) = (1 - plan.relax) * input(iCell, c * N + mode) + plan.relax * tile.c[i];
                            }
                            __syncwarp();
                        }
                    }
                }
                if (!Tensor || !raw)
                    for (int i = rank; i < N * components; i += teamSize)
                    {
                        real value = right[i];
                        if (raw)
                        {
                            value = 0;
                            for (int j = 0; j < N; ++j)
                                value += inverse[j * N + i % N] * right[(i / N) * N + j];
                        }
                        output(iCell, i) = (1 - plan.relax) * input(iCell, i) + plan.relax * value;
                    }
                Join<Warp>();
            }
            // A final prefetched row never exists beyond selection.count.
#else
            asm("trap;");
#endif
        }
        template <int N, bool Warp, bool CellRow, bool Async, bool Tensor>
        ReconstructionSweepResources Launch(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                            DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                            R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                            const ReconstructionSweepLaunch &options, bool execute)
        {
            const int teams = Warp ? options.threads / 32 : 1;
            const index capacity = CellRow ? std::max<index>(1, plan.maxRowEntries) : 1;
            const index components = output.width / N;
            const index words = 2 * capacity * (N * N + N * components) + N * components;
            const index inverseEnd = (words + 3) / 4 * 4 + (plan.raw ? (N * N + 3) / 4 * 4 : 0);
            const index teamWords = (inverseEnd + (Tensor ? sizeof(Algebra::CUDA::FP64Tile8) / sizeof(real) : 0) + 3) / 4 * 4;
            const index bytes = teamWords * teams * sizeof(real);
            static thread_local int cachedOrdinal = -1;
            static thread_local cudaDeviceProp device{};
            if (cachedOrdinal != context.device.ordinal)
            {
                DNDS_check_throw_info(cudaGetDeviceProperties(&device, context.device.ordinal) == cudaSuccess,
                                      "CUDA VR device properties unavailable");
                DNDS_check_throw_info(cudaFuncSetAttribute(StagedJacobi<N, Warp, CellRow, Async, Tensor>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           int(device.sharedMemPerBlockOptin)) == cudaSuccess,
                                      "CUDA VR shared-memory opt-in failed");
                cachedOrdinal = context.device.ordinal;
            }
            DNDS_check_throw_info(device.major >= 8 && bytes <= device.sharedMemPerBlockOptin && bytes <= INT_MAX,
                                  "CUDA VR staging exceeds device shared memory or requires sm80+; choose fewer warp-cells or neighbor buffering");
            const index needed = (selection.count + teams - 1) / teams;
            const unsigned grid = unsigned(options.gridBlocks ? std::min<index>(needed, options.gridBlocks) : std::min<index>(needed, 65535));
            ReconstructionSweepResources resource{bytes, 0, 0, int(grid), options.threads};
            if (!execute)
            {
                cudaFuncAttributes attributes{};
                DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, StagedJacobi<N, Warp, CellRow, Async, Tensor>) == cudaSuccess &&
                                          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resource.activeBlocksPerSM,
                                                                                        StagedJacobi<N, Warp, CellRow, Async, Tensor>, options.threads, size_t(bytes)) == cudaSuccess,
                                      "CUDA VR kernel resource query failed");
                resource.registersPerThread = attributes.numRegs;
                return resource;
            }
            StagedJacobi<N, Warp, CellRow, Async, Tensor><<<grid, options.threads, size_t(bytes), static_cast<cudaStream_t>(context.stream)>>>(
                Sweep(plan), plan.raw, selection, vectors, mean, input, boundary, output, status, int(capacity), int(teamWords));
            return resource;
        }
    }
    static ReconstructionSweepResources DispatchStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                                             DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                                             R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                                             const ReconstructionSweepLaunch &options, bool tensor, bool execute)
    {
        options.Validate();
        // Native maps are contiguous tables, including padded row pitches.
        // Project their static access pattern once, rather than carrying CSR/
        // SoA/tile branches through every inner product. Solution fields stay
        // layout-generic; only their frozen row spans are staged.
        const auto table = [](auto view, index width)
        { return view.kind == R::RowLayoutKind::Table && view.width == width && view.stride >= width && !view.offsets && !view.sizes && (!view.rows || view.data); };
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && !plan.quantized &&
                                  plan.n > 0 && plan.n <= 19 && output.width % plan.n == 0 &&
                                  selection.count >= 0 && selection.count <= plan.owned &&
                                  (!selection.indirect || selection.rows.rows >= selection.count) &&
                                  plan.maxRowEntries >= 0 && plan.maxRowEntries <= INT_MAX / (19 * 26) &&
                                  output.width / plan.n >= 1 && output.width / plan.n <= 7 &&
                                  table(plan.matrixAInvB, plan.n * plan.n) && table(plan.vectorAInvB, plan.n) &&
                                  plan.matrixAInvB.width == plan.n * plan.n &&
                                  plan.vectorAInvB.width == plan.n && plan.boundaryInjection.width == plan.n &&
                                  table(plan.offsets, 1) && table(plan.neighbors, 1) && table(plan.boundaryOffsets, 1) &&
                                  table(plan.boundaryInjection, plan.n) &&
                                  (!plan.raw || table(plan.localInverse, plan.localInverse.width)) &&
                                  (!plan.faceCoupling || (table(plan.faces, 1) && table(plan.faceSide, 1))) &&
                                  (!plan.rotated || (table(plan.rotation, 9) && table(plan.rotationActive, 1))) &&
                                  plan.offsets.rows == plan.owned + 1 && plan.neighbors.width == 1 &&
                                  plan.boundaryOffsets.rows == plan.neighbors.rows + 1 &&
                                  plan.vectorAInvB.rows == plan.neighbors.rows &&
                                  table(status, 1) && status.rows == 1 &&
                                  (!plan.faceCoupling || (plan.faces.rows == plan.neighbors.rows && plan.faceSide.rows == plan.neighbors.rows)) &&
                                  (!plan.rotated || (plan.rotation.rows == plan.neighbors.rows && plan.rotationActive.rows == plan.neighbors.rows)) &&
                                  (!plan.raw || (plan.localInverse.rows == plan.owned &&
                                                 plan.localInverse.width == (plan.inverseTriangle ? plan.n * (plan.n + 1) / 2 : plan.n * plan.n))) &&
                                  input.owned.width == output.width && mean.owned.width == output.width / plan.n &&
                                  (!input.ghost.rows || input.ghost.width == output.width) &&
                                  (!mean.ghost.rows || mean.ghost.width == mean.owned.width) &&
                                  (!boundary.rows || boundary.width == mean.owned.width) &&
                                  mean.owned.rows == plan.owned && input.owned.rows == plan.owned && output.rows == plan.owned &&
                                  (!output.rows || (output.data != input.owned.data && output.data != input.ghost.data)) &&
                                  (!tensor || (options.mapping == "warp" && FP64TensorCoreAvailable(context.device))) &&
                                  (options.buffering != "cellRow" || plan.maxRowEntries > 0 || !selection.count),
                              "stage-once CUDA VR requires FP64 P1..P3, <=7 components, a prepared row bound and a supported mapping");
        R::ValidateViewDevices(context.device, plan, selection, vectors, mean, input, boundary, output, status);
        R::CheckKernelArgumentBudget<Sweep, bool, decltype(selection), decltype(vectors), decltype(mean), decltype(input),
                                     decltype(boundary), decltype(output), decltype(status), int, int>();
        if (!selection.count)
            return {};
        ReconstructionSweepResources resource;
        const auto modes = [&](auto n)
        {
            constexpr int N = decltype(n)::value;
            const auto mapping = [&](auto warp)
            {
                constexpr bool W = decltype(warp)::value;
                const auto buffer = [&](auto cellRow)
                {
                    constexpr bool C = decltype(cellRow)::value;
                    const auto async = [&](auto asynchronous)
                    {
                        constexpr bool A = decltype(asynchronous)::value;
                        if constexpr (W)
                            if (tensor)
                            {
                                resource = Launch<N, W, C, A, true>(context, plan, selection, vectors, mean, input, boundary, output, status, options, execute);
                                return;
                            }
                        resource = Launch<N, W, C, A, false>(context, plan, selection, vectors, mean, input, boundary, output, status, options, execute);
                    };
                    if (options.asynchronous)
                        async(std::true_type{});
                    else
                        async(std::false_type{});
                };
                if (options.buffering == "cellRow")
                    buffer(std::true_type{});
                else
                    buffer(std::false_type{});
            };
            if (options.mapping == "warp")
                mapping(std::true_type{});
            else
                mapping(std::false_type{});
        };
        switch (plan.n)
        {
#define DNDS_STAGED_CASE(N)                      \
    case N:                                      \
        modes(std::integral_constant<int, N>{}); \
        break
            DNDS_STAGED_CASE(2);
            DNDS_STAGED_CASE(3);
            DNDS_STAGED_CASE(5);
            DNDS_STAGED_CASE(9);
            DNDS_STAGED_CASE(19);
#undef DNDS_STAGED_CASE
        default:
            DNDS_check_throw_info(false, "unsupported staged CUDA VR mode count");
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "stage-once CUDA VR launch failed");
        return resource;
    }
    void LaunchStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                            DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                            R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                            const ReconstructionSweepLaunch &options, bool tensor)
    {
        (void)DispatchStagedJacobi(context, plan, selection, vectors, mean, input, boundary, output, status, options, tensor, true);
    }
    ReconstructionSweepResources InspectStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                                     DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                                     R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                                     const ReconstructionSweepLaunch &options, bool tensor)
    {
        return DispatchStagedJacobi(context, plan, selection, vectors, mean, input, boundary, output, status, options, tensor, false);
    }
}
