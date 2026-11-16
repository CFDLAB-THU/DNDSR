#pragma once

#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include "DNDS/Algebra/CUDA/FP64Matrix.cuh"
#include "DNDS/Math/ComputeScalar.hpp"
#include <mma.h>
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    namespace Staged
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
        template <class Storage>
        struct Sweep
        {
            Table<index, R::FixedRows<1>> offsets, neighbors, faces, boundaryOffsets;
            Table<uint8_t, R::FixedRows<1>> rotationActive, faceSide;
            Table<Storage, R::TableRows> matrices, vectors, injection;
            Table<real, R::TableRows> inverse;
            Table<real, R::FixedRows<9>> rotation;
            real relax;
            bool rotated, triangle, faceCoupling;
            explicit Sweep(const ReconstructionDeviceView &p)
                : offsets(Project<R::FixedRows<1>>(p.offsets)), neighbors(Project<R::FixedRows<1>>(p.neighbors)),
                  faces(Project<R::FixedRows<1>>(p.faces)), boundaryOffsets(Project<R::FixedRows<1>>(p.boundaryOffsets)),
                  rotationActive(Project<R::FixedRows<1>>(p.rotationActive)), faceSide(Project<R::FixedRows<1>>(p.faceSide)),
                  inverse(Project<R::TableRows>(p.localInverse)),
                  rotation(Project<R::FixedRows<9>>(p.rotation)), relax(p.relax), rotated(p.rotated),
                  triangle(p.inverseTriangle), faceCoupling(p.faceCoupling)
            {
                if constexpr (std::is_same_v<Storage, float>)
                    matrices = Project<R::TableRows>(p.matrixAInvB32), vectors = Project<R::TableRows>(p.vectorAInvB32), injection = Project<R::TableRows>(p.boundaryInjection32);
                else
                    matrices = Project<R::TableRows>(p.matrixAInvB), vectors = Project<R::TableRows>(p.vectorAInvB), injection = Project<R::TableRows>(p.boundaryInjection);
            }
        };
        template <bool Warp>
        __device__ void Join()
        {
            if constexpr (Warp)
                __syncwarp();
            else
                __syncthreads();
        }
        template <class T>
        __device__ void CopyWord(real *destination, const T *source)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            {
                const unsigned address = unsigned(__cvta_generic_to_shared(destination));
                if constexpr (sizeof(T) == 8)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(address), "l"(source) : "memory");
                else
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(address), "l"(source) : "memory");
            }
#else
            *reinterpret_cast<T *>(destination) = *source;
#endif
        }
        __device__ inline void CommitCopies()
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            asm volatile("cp.async.commit_group;" ::: "memory");
#endif
        }
        template <bool Warp>
        __device__ void WaitCopies()
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
            Join<Warp>();
        }
        template <int N, class Storage>
        __device__ void StageEntry(Sweep<Storage> plan, index entry, real *slot,
                                   R::PairReadView<real> input,
                                   int components, int rank, int teamSize)
        {
            const index other = plan.neighbors(entry, 0);
            if (other == UnInitIndex)
                return; // Boundary values are consumed through the injection map.
            const index matrixRow = plan.faceCoupling ? plan.faces(entry, 0) : entry;
            const auto matrix = plan.matrices[matrixRow];
            for (int i = rank; i < N * N; i += teamSize)
                CopyWord(slot + i, &matrix[i]);
            const auto values = input[other];
            for (int i = rank; i < N * components; i += teamSize)
                CopyWord(slot + N * N + i, &values[i]);
        }
        template <class Scalar>
        struct alignas(8) Word
        {
            Scalar value;
        };
        template <int N, class Scalar, class Storage>
        __device__ Scalar Coefficient(Sweep<Storage> plan, DevicePeriodicBlocks vectors, index entry,
                                      const Word<Scalar> *coefficients, int component, int mode)
        {
            const auto first = plan.rotated && plan.rotationActive(entry, 0) ? vectors.First(component) : -1;
            if (first < 0)
                return coefficients[component * N + mode].value;
            Scalar value(0);
            for (int d = 0; d < 3; ++d)
                value += Scalar(plan.rotation(entry, d * 3 + component - first)) * coefficients[(first + d) * N + mode].value;
            return value;
        }
        template <int N, class Scalar, class Storage>
        __device__ Scalar OtherTerms(Sweep<Storage> plan, DevicePeriodicBlocks vectors, index iCell,
                                     int component, int mode, R::PairReadView<real> mean,
                                     R::ArrayView<const real> boundary)
        {
            Scalar value(0);
            const auto first = plan.rotated ? vectors.First(component) : -1;
            for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
            {
                const auto other = plan.neighbors(entry, 0);
                if (other != UnInitIndex)
                {
                    Scalar average(mean(other, component));
                    if (first >= 0 && plan.rotationActive(entry, 0))
                    {
                        average = Scalar(0);
                        for (int d = 0; d < 3; ++d)
                            average += Scalar(plan.rotation(entry, d * 3 + component - first)) * Scalar(mean(other, first + d));
                    }
                    value += Scalar(plan.vectors(entry, mode)) * (average - Scalar(mean(iCell, component)));
                }
                else
                    for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                        value += Scalar(plan.injection(q, mode)) * (Scalar(boundary(q, component)) - Scalar(mean(iCell, component)));
            }
            return value;
        }
        /** @brief Convert a completed raw staging slot exactly once.
         * @details cp.async transports FP64 bytes without numerical arithmetic.
         * fp32mp2 conversion preserves both limbs; every later numeric operand
         * is Scalar. The shared slots have identical size/alignment, and no
         * consumer accesses their old FP64 representation after this barrier.
         */
        template <int N, class Storage, class Scalar, bool Warp>
        __device__ void ConvertSlot(real *slot, int words, int rank, int teamSize)
        {
            static_assert(sizeof(Word<Scalar>) == sizeof(real) && std::is_trivially_copyable_v<Scalar>);
            if constexpr (std::is_same_v<Storage, real> && std::is_same_v<Scalar, real>)
                return;
            auto *converted = reinterpret_cast<Word<Scalar> *>(slot);
            for (int i = rank; i < words; i += teamSize)
            {
                if (i < N * N)
                    converted[i].value = Scalar(*reinterpret_cast<const Storage *>(slot + i));
                else
                    converted[i].value = Scalar(slot[i]);
            }
            Join<Warp>();
        }
        struct alignas(32) TF32Tile
        {
            float a[128], b[128], c[256];
        };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        class TF32Accumulator
        {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> _value;

        public:
            __device__ TF32Accumulator() { nvcuda::wmma::fill_fragment(_value, 0.f); }
            __device__ void AddProduct(const TF32Tile &tile)
            {
                namespace w = nvcuda::wmma;
                w::fragment<w::matrix_a, 16, 16, 8, w::precision::tf32, w::row_major> a;
                w::fragment<w::matrix_b, 16, 16, 8, w::precision::tf32, w::row_major> b;
                w::load_matrix_sync(a, tile.a, 8);
                w::load_matrix_sync(b, tile.b, 16);
                w::mma_sync(_value, a, b, _value);
            }
            __device__ void Store(TF32Tile &tile) const
            {
                nvcuda::wmma::store_matrix_sync(tile.c, _value, 16, nvcuda::wmma::mem_row_major);
            }
        };
#endif
        /** @brief Stage one complete neighbor, consuming every output mode.
         * @details SIMT owns one cell/block; MMA owns one cell/warp. Both use
         * two neighbor slots. No full-row or repeated-output-tile staging.
         * Scalar selects ALL numerical arithmetic, including periodic/BC terms,
         * raw inverse application and relaxation. FP64 is only storage/I/O in
         * the float-float specialization. Distinct outputs remain mandatory.
         */
        template <int N, int Components, class Storage, class Scalar, bool Tensor, bool TF32>
        __global__ void StagedJacobi(Sweep<Storage> plan, bool raw, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                     R::PairReadView<real> mean, R::PairReadView<real> input,
                                     R::ArrayView<const real> boundary, R::ArrayView<real> output,
                                     R::ArrayView<const int> status, int teamWords)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if (status(0, 0))
                return;
            extern __shared__ __align__(32) real shared[];
            const int components = Components ? Components : int(output.width / N);
            const int entryWords = N * N + N * components;
            const int team = Tensor ? threadIdx.x / 32 : 0;
            const int rank = Tensor ? threadIdx.x % 32 : threadIdx.x;
            const int teamSize = Tensor ? 32 : blockDim.x;
            const int teams = Tensor ? blockDim.x / 32 : 1;
            real *slots = shared + team * teamWords;
            auto *right = reinterpret_cast<Scalar *>(slots + 2 * entryWords);
            auto *inverse = reinterpret_cast<Scalar *>(slots + ((2 * entryWords + N * components + 3) / 4) * 4);
            using Tile = std::conditional_t<TF32, TF32Tile, Algebra::CUDA::FP64Tile8>;
            auto &tile = *reinterpret_cast<Tile *>(inverse + (raw ? ((N * N + 3) / 4) * 4 : 0));
            constexpr int M = TF32 ? 16 : 8, K = TF32 ? 8 : 4, C = TF32 ? 16 : 8;
            const index stride = index(gridDim.x) * teams;
            const Scalar relax(plan.relax), keep = Scalar(1) - relax;
            for (index work = index(blockIdx.x) * teams + team; work < selection.count; work += stride)
            {
                const auto iCell = selection.Cell(work);
                const index begin = plan.offsets(iCell, 0), end = plan.offsets(iCell + 1, 0);
                Scalar accumulated[(N * (Components ? Components : 7) + 31) / 32]{};
                using Accumulator = std::conditional_t<TF32, TF32Accumulator, Algebra::CUDA::FP64Accumulator8>;
                Accumulator products[(N + M - 1) / M];
                if constexpr (!Components)
                {
                    for (int i = rank; i < N * components; i += teamSize)
                        right[i] = Scalar(0);
                    Join<Tensor>();
                }
                if (begin < end)
                {
                    StageEntry<N>(plan, begin, slots, input, components, rank, teamSize);
                    CommitCopies();
                }
                for (index entry = begin; entry < end; ++entry)
                {
                    const int slot = int((entry - begin) % 2);
                    WaitCopies<Tensor>();
                    if (entry + 1 < end)
                    {
                        StageEntry<N>(plan, entry + 1, slots + (1 - slot) * entryWords, input, components, rank, teamSize);
                        CommitCopies();
                    }
                    const auto other = plan.neighbors(entry, 0);
                    const bool rotated = plan.rotated && plan.rotationActive(entry, 0);
                    const bool transpose = plan.faceCoupling && plan.faceSide(entry, 0);
                    if (other != UnInitIndex && (!raw || other != iCell || rotated))
                    {
                        ConvertSlot<N, Storage, Scalar, Tensor>(slots + slot * entryWords, entryWords, rank, teamSize);
                        const auto *matrix = reinterpret_cast<const Word<Scalar> *>(slots + slot * entryWords);
                        const auto *coefficients = matrix + N * N;
                        if constexpr (Tensor)
                        {
                            for (int cStart = 0; cStart < components; cStart += C)
                            {
                                // Static 5/6/7 keep fragments across neighbors;
                                // dynamic shapes accumulate each component tile
                                // in shared memory, reusing the staged matrix.
                                Accumulator partial[(N + M - 1) / M];
                                for (int modeTile = 0; modeTile < (N + M - 1) / M; ++modeTile)
                                    for (int k = 0; k < N; k += K)
                                    {
                                        for (int i = rank; i < M * K; i += 32)
                                        {
                                            const int mode = modeTile * M + i / K, ka = k + i % K;
                                            Scalar value = mode < N && ka < N ? matrix[transpose ? mode * N + ka : ka * N + mode].value : Scalar(0);
                                            if constexpr (TF32)
                                                value = nvcuda::wmma::__float_to_tf32(value);
                                            tile.a[i] = value;
                                        }
                                        for (int i = rank; i < K * C; i += 32)
                                        {
                                            const int component = cStart + i % C, kb = k + i / C;
                                            Scalar value = component < components && kb < N ? Coefficient<N, Scalar>(plan, vectors, entry, coefficients, component, kb) : Scalar(0);
                                            if constexpr (TF32)
                                                value = nvcuda::wmma::__float_to_tf32(value);
                                            tile.b[i] = value;
                                        }
                                        __syncwarp();
                                        if constexpr (Components)
                                            products[modeTile].AddProduct(tile);
                                        else
                                            partial[modeTile].AddProduct(tile);
                                        __syncwarp();
                                    }
                                if constexpr (!Components)
                                    for (int m = 0; m < (N + M - 1) / M; ++m)
                                    {
                                        partial[m].Store(tile);
                                        __syncwarp();
                                        for (int i = rank; i < M * C; i += 32)
                                            if (m * M + i / C < N && cStart + i % C < components)
                                                right[(cStart + i % C) * N + m * M + i / C] += tile.c[i];
                                        __syncwarp();
                                    }
                            }
                        }
                        else
                        {
                            for (int i = rank; i < N * components; i += teamSize)
                            {
                                Scalar product(0);
                                for (int j = 0; j < N; ++j)
                                    product += matrix[transpose ? (i % N) * N + j : j * N + i % N].value * Coefficient<N, Scalar>(plan, vectors, entry, coefficients, i / N, j);
                                if constexpr (Components)
                                    accumulated[i / teamSize] += product;
                                else
                                    right[i] += product;
                            }
                        }
                    }
                    Join<Tensor>();
                }
                if constexpr (Tensor && Components)
                {
                    for (int m = 0; m < (N + M - 1) / M; ++m)
                    {
                        products[m].Store(tile);
                        __syncwarp();
                        for (int i = rank; i < M * C; i += 32)
                            if (m * M + i / C < N && i % C < components)
                                right[(i % C) * N + m * M + i / C] = tile.c[i];
                        __syncwarp();
                    }
                }
                else if constexpr (!Tensor && Components)
                    for (int i = rank; i < N * components; i += teamSize)
                        right[i] = accumulated[i / teamSize];
                Join<Tensor>();
                for (int i = rank; i < N * components; i += teamSize)
                    right[i] += OtherTerms<N, Scalar>(plan, vectors, iCell, i / N, i % N, mean, boundary);
                if (raw)
                {
                    const auto row = plan.inverse[iCell];
                    for (int i = rank; i < N * N; i += teamSize)
                        inverse[i] = Scalar(row[plan.triangle ? Math::SymmetricTriangleEntry(i % N, i / N) : i]);
                }
                Join<Tensor>();
                if constexpr (Tensor && Components && !TF32)
                {
                    if (raw)
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
                                    output(iCell, c * N + mode) = keep * input(iCell, c * N + mode) + relax * tile.c[i];
                            }
                            __syncwarp();
                        }
                }
                if (!Tensor || !raw || !Components || TF32)
                    for (int i = rank; i < N * components; i += teamSize)
                    {
                        Scalar value = right[i];
                        if (raw)
                        {
                            value = Scalar(0);
                            for (int j = 0; j < N; ++j)
                                value += inverse[j * N + i % N] * right[(i / N) * N + j];
                        }
                        output(iCell, i) = DNDS::Math::ScalarCast<real>(keep * Scalar(input(iCell, i)) + relax * value);
                    }
                Join<Tensor>();
            }
#else
            asm("trap;");
#endif
        }
        template <int N, int Components, class Storage, class Scalar, bool Tensor, bool TF32>
        ReconstructionSweepResources Launch(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                            DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                            R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                            const ReconstructionSweepLaunch &options, bool execute)
        {
            const int threads = options.threads ? options.threads : Tensor && output.width / N <= 7 ? 128
                                                                                                    : 32;
            const int teams = Tensor ? threads / 32 : 1;
            const index components = output.width / N;
            const index words = 2 * (N * N + N * components) + N * components;
            const index inverseEnd = (words + 3) / 4 * 4 + (plan.raw ? (N * N + 3) / 4 * 4 : 0);
            const index teamWords = (inverseEnd + (Tensor ? (TF32 ? sizeof(TF32Tile) : sizeof(Algebra::CUDA::FP64Tile8)) / sizeof(real) : 0) + 3) / 4 * 4;
            const index bytes = teamWords * teams * sizeof(real);
            static thread_local int cachedOrdinal = -1;
            static thread_local cudaDeviceProp device{};
            if (cachedOrdinal != context.device.ordinal)
            {
                DNDS_check_throw_info(cudaGetDeviceProperties(&device, context.device.ordinal) == cudaSuccess,
                                      "CUDA VR device properties unavailable");
                DNDS_check_throw_info(cudaFuncSetAttribute(StagedJacobi<N, Components, Storage, Scalar, Tensor, TF32>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           int(device.sharedMemPerBlockOptin)) == cudaSuccess,
                                      "CUDA VR shared-memory opt-in failed");
                cachedOrdinal = context.device.ordinal;
            }
            DNDS_check_throw_info(device.major >= 8 && bytes <= device.sharedMemPerBlockOptin && bytes <= INT_MAX,
                                  "CUDA VR staging exceeds device shared memory or requires sm80+");
            const index needed = (selection.count + teams - 1) / teams;
            const unsigned grid = unsigned(options.gridBlocks ? std::min<index>(needed, options.gridBlocks) : std::min<index>(needed, 65535));
            ReconstructionSweepResources resource{bytes, 0, 0, int(grid), threads};
            if (!execute)
            {
                cudaFuncAttributes attributes{};
                DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, StagedJacobi<N, Components, Storage, Scalar, Tensor, TF32>) == cudaSuccess &&
                                          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resource.activeBlocksPerSM,
                                                                                        StagedJacobi<N, Components, Storage, Scalar, Tensor, TF32>, threads, size_t(bytes)) == cudaSuccess,
                                      "CUDA VR kernel resource query failed");
                resource.registersPerThread = attributes.numRegs;
                return resource;
            }
            StagedJacobi<N, Components, Storage, Scalar, Tensor, TF32><<<grid, threads, size_t(bytes), static_cast<cudaStream_t>(context.stream)>>>(
                Sweep<Storage>(plan), plan.raw, selection, vectors, mean, input, boundary, output, status, int(teamWords));
            return resource;
        }
    }
    struct StagedSweepArguments
    {
        R::ExecutionContext context;
        ReconstructionDeviceView plan;
        DeviceWorkSelection selection;
        DevicePeriodicBlocks vectors;
        R::PairReadView<real> mean, input;
        R::ArrayView<const real> boundary;
        R::ArrayView<real> output;
        R::ArrayView<const int> status;
        ReconstructionSweepLaunch options;
        bool execute;
    };
    namespace Staged
    {
        template <class Storage, class Scalar, bool Tensor, bool TF32 = false>
        ReconstructionSweepResources RunShapes(const StagedSweepArguments &a)
        {
            ReconstructionSweepResources result;
            const auto modes = [&](auto n)
            {
                const auto components = [&](auto c)
                {
                    result = Launch<decltype(n)::value, decltype(c)::value, Storage, Scalar, Tensor, TF32>(
                        a.context, a.plan, a.selection, a.vectors, a.mean, a.input, a.boundary, a.output, a.status, a.options, a.execute);
                };
                switch (a.mean.owned.width)
                {
                case 5:
                    components(std::integral_constant<int, 5>{});
                    break;
                case 6:
                    components(std::integral_constant<int, 6>{});
                    break;
                case 7:
                    components(std::integral_constant<int, 7>{});
                    break;
                default:
                    components(std::integral_constant<int, 0>{});
                    break;
                }
            };
            switch (a.plan.n)
            {
            case 2:
                modes(std::integral_constant<int, 2>{});
                break;
            case 3:
                modes(std::integral_constant<int, 3>{});
                break;
            case 5:
                modes(std::integral_constant<int, 5>{});
                break;
            case 9:
                modes(std::integral_constant<int, 9>{});
                break;
            case 19:
                modes(std::integral_constant<int, 19>{});
                break;
            default:
                DNDS_check_throw_info(false, "unsupported staged CUDA VR mode count");
            }
            return result;
        }
    }
    ReconstructionSweepResources RunStagedFP64(const StagedSweepArguments &, bool tensor);
    ReconstructionSweepResources RunStagedFloatFloat(const StagedSweepArguments &);
    ReconstructionSweepResources RunStagedFP32(const StagedSweepArguments &, bool tensor);
}
