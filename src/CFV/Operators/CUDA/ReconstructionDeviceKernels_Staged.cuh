#pragma once

#include "DNDS/Runtime/Execution/CUDAStream.hpp"

#include "CFV/Detail/ReconstructionSweepArguments.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include "DNDS/Algebra/CUDA/FP64Matrix.cuh"
#include "DNDS/Math/ComputeScalar.hpp"
#include "DNDS/Runtime/Execution/KernelViews.hpp"
#include <mma.h>
#include <cuda_runtime.h>
#include <cstring>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    namespace Staged
    {
        // This projection omits limiter/basis buffers from the launch arguments.
        // Matrix/field tables are checked once by the wrapper. Static logical
        // field widths use compact AoS pitches; matrix padding stays independent.
        template <class T, class Layout>
        using Table = R::ArrayView<const T, R::AccessBoundsChecks, Layout>;
        template <class Layout, class T>
        Table<T, Layout> Project(R::ArrayView<const T> view)
        {
            return R::ProjectKernelView<Layout>(view);
        }
        template <int N, class Storage, bool Triangle>
        struct Sweep
        {
            using MatrixRows = R::FixedRows<N * N>;
            using InverseRows = R::FixedRows<Triangle ? N *(N + 1) / 2 : N * N>;
            Table<index, R::FixedRows<1>> offsets, neighbors, faces, boundaryOffsets;
            Table<uint8_t, R::FixedRows<1>> rotationActive, faceSide;
            Table<Storage, MatrixRows> matrices;
            Table<Storage, R::FixedRows<N>> vectors, injection;
            Table<real, InverseRows> inverse;
            Table<float, MatrixRows> packedMatrices;
            Table<float, InverseRows> packedInverse;
            Table<real, R::FixedRows<9>> rotation;
            real relax;
            bool rotated, faceCoupling;
            bool usePackedB, usePackedInverse;
            explicit Sweep(const ReconstructionDeviceView &p)
                : offsets(Project<R::FixedRows<1>>(p.offsets)), neighbors(Project<R::FixedRows<1>>(p.neighbors)),
                  faces(Project<R::FixedRows<1>>(p.faces)), boundaryOffsets(Project<R::FixedRows<1>>(p.boundaryOffsets)),
                  rotationActive(Project<R::FixedRows<1>>(p.rotationActive)), faceSide(Project<R::FixedRows<1>>(p.faceSide)),
                  inverse(Project<InverseRows>(p.localInverse)),
                  packedMatrices(Project<MatrixRows>(p.rawMatrixB32)),
                  packedInverse(Project<InverseRows>(p.localInverse32)),
                  rotation(Project<R::FixedRows<9>>(p.rotation)), relax(p.relax), rotated(p.rotated),
                  faceCoupling(p.faceCoupling), usePackedB(p.packedRawB), usePackedInverse(p.packedInverse)
            {
                if constexpr (std::is_same_v<Storage, float>)
                    matrices = Project<MatrixRows>(p.matrixAInvB32), vectors = Project<R::FixedRows<N>>(p.vectorAInvB32), injection = Project<R::FixedRows<N>>(p.boundaryInjection32);
                else
                    matrices = Project<MatrixRows>(p.matrixAInvB), vectors = Project<R::FixedRows<N>>(p.vectorAInvB), injection = Project<R::FixedRows<N>>(p.boundaryInjection);
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
        template <bool Pipeline, class T>
        __device__ void CopyWord(real *destination, const T *source)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (Pipeline)
            {
                const unsigned address = unsigned(__cvta_generic_to_shared(destination));
                if constexpr (sizeof(T) == 8)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(address), "l"(source) : "memory");
                else
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(address), "l"(source) : "memory");
            }
            else
#endif
                *reinterpret_cast<T *>(destination) = *source;
        }
        template <bool Pipeline>
        __device__ inline void CommitCopies()
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (Pipeline)
                asm volatile("cp.async.commit_group;" ::: "memory");
#endif
        }
        template <bool Warp, bool Pipeline>
        __device__ void WaitCopies()
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (Pipeline)
                asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
            Join<Warp>();
        }
        template <int N, bool Pipeline, class Row>
        __device__ void StageMatrix(real *slot, Row matrix, int rank, int teamSize)
        {
            for (int i = rank; i < N * N; i += teamSize)
                CopyWord<Pipeline>(slot + i, &matrix[i]);
        }
        template <int N, bool Pipeline, bool Raw, class Storage, bool Triangle, class Input>
        __device__ void StageEntry(Sweep<N, Storage, Triangle> plan, index entry, real *slot,
                                   Input input,
                                   int components, int rank, int teamSize)
        {
            const index other = plan.neighbors(entry, 0);
            if (other == UnInitIndex)
                return; // Boundary values are consumed through the injection map.
            const index matrixRow = plan.faceCoupling ? plan.faces(entry, 0) : entry;
            if constexpr (Raw)
            {
                if (plan.usePackedB)
                    StageMatrix<N, Pipeline>(slot, plan.packedMatrices[matrixRow], rank, teamSize);
                else
                    StageMatrix<N, Pipeline>(slot, plan.matrices[matrixRow], rank, teamSize);
            }
            else
                StageMatrix<N, Pipeline>(slot, plan.matrices[matrixRow], rank, teamSize);
            const auto values = input[other];
            for (int i = rank; i < N * components; i += teamSize)
                CopyWord<Pipeline>(slot + N * N + i, &values[i]);
        }
        template <class Scalar>
        struct alignas(8) Word
        {
            Scalar value;
        };
        template <int N, class Scalar, class Storage, bool Triangle>
        __device__ Scalar Coefficient(Sweep<N, Storage, Triangle> plan, DevicePeriodicBlocks vectors, index entry,
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
        template <int N, class Scalar, class Storage, bool Triangle, class Mean, class Boundary>
        __device__ Scalar OtherTerms(Sweep<N, Storage, Triangle> plan, DevicePeriodicBlocks vectors, index iCell,
                                     int component, int mode, Mean mean, Boundary boundary)
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
        template <int N, bool Warp>
        __device__ void PromoteRawB(real *slot, int rank, int teamSize)
        {
            // The float is in the first four bytes of its FP64 staging slot.
            for (int i = rank; i < N * N; i += teamSize)
            {
                float stored;
                memcpy(&stored, slot + i, sizeof(stored));
                slot[i] = real(stored);
            }
            Join<Warp>();
        }
        template <int N, bool Triangle, class Scalar, class Row>
        __device__ void StageInverse(Scalar *inverse, Row row, int rank, int teamSize)
        {
            for (int i = rank; i < N * N; i += teamSize)
                inverse[i] = Scalar(row[Triangle ? Math::SymmetricTriangleEntry(i % N, i / N) : i]);
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
        template <int Width>
        using FieldPolicy = std::conditional_t<Width == 0, R::TableRows, R::FixedRows<Width, Width>>;
        template <int Components, int N>
        struct Fields
        {
            R::PairReadView<real, R::AccessBoundsChecks, FieldPolicy<Components>> mean;
            R::PairReadView<real, R::AccessBoundsChecks, FieldPolicy<Components * N>> input;
            Table<real, FieldPolicy<Components>> boundary;
            R::ArrayView<real, R::AccessBoundsChecks, FieldPolicy<Components * N>> output;
            Table<real, FieldPolicy<Components * N>> target;
            Table<real, R::FixedRows<1>> alpha;
        };
        template <class Scalar, class Input, class Target>
        __device__ Scalar Updated(Input input, Target target, index iCell, int i,
                                  Scalar keep, Scalar relax, Scalar penalty, Scalar scale, Scalar increment)
        {
            Scalar value = keep * Scalar(input(iCell, i));
            if (penalty > Scalar(0))
                value += relax * penalty * Scalar(target(iCell, i));
            return value + scale * increment;
        }
        /** @brief Stage one complete neighbor, consuming every output mode.
         * @details SIMT owns one cell/block; MMA owns one cell/warp. Both use
         * two neighbor slots. No full-row or repeated-output-tile staging.
         * Scalar selects ALL numerical arithmetic, including periodic/BC terms,
         * raw inverse application and relaxation. FP64 is only storage/I/O in
         * the float-float specialization. Distinct outputs remain mandatory.
         */
        template <int N, int Components, bool Raw, bool Triangle, bool LVR, class Storage, class Scalar,
                  bool Tensor, bool TF32, bool Pipeline, class Product = Scalar>
        __global__ void StagedJacobi(Sweep<N, Storage, Triangle> plan, Runtime::RowSelection selection, DevicePeriodicBlocks vectors,
                                     Fields<Components, N> fields, R::ArrayView<const int> status, int teamWords)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            static_assert(!Raw || (std::is_same_v<Storage, real> && std::is_same_v<Scalar, real> &&
                                   std::is_same_v<Product, real> && !Tensor));
            if (status(0, 0))
                return;
            extern __shared__ __align__(32) real shared[];
            const auto mean = fields.mean;
            const auto input = fields.input;
            const auto boundary = fields.boundary;
            const auto output = fields.output;
            constexpr bool wideMMA = Tensor && TF32 && std::is_same_v<Scalar, real>;
            const int components = Components ? Components : int(output.width / N);
            const int entryWords = N * N + N * components;
            const int team = Tensor ? threadIdx.x / 32 : 0;
            const int rank = Tensor ? threadIdx.x % 32 : threadIdx.x;
            const int teamSize = Tensor ? 32 : blockDim.x;
            const int teams = Tensor ? blockDim.x / 32 : 1;
            real *slots = shared + team * teamWords;
            auto *right = reinterpret_cast<Scalar *>(slots + (Pipeline ? 2 : 1) * entryWords);
            auto *inverse = reinterpret_cast<Scalar *>(slots + (((Pipeline ? 2 : 1) * entryWords + N * components + 3) / 4) * 4);
            using Tile = std::conditional_t<TF32, TF32Tile, Algebra::CUDA::FP64Tile8>;
            auto &tile = *reinterpret_cast<Tile *>(inverse + (Raw ? ((N * N + 3) / 4) * 4 : 0));
            constexpr int M = TF32 ? 16 : 8, K = TF32 ? 8 : 4, C = TF32 ? 16 : 8;
            const index stride = index(gridDim.x) * teams;
            const Scalar relax(plan.relax), keep = Scalar(1) - relax;
            for (index work = index(blockIdx.x) * teams + team; work < selection.RowCount(); work += stride)
            {
                const auto iCell = selection.RowAt(work);
                const Scalar penalty = LVR ? Scalar(fields.alpha(iCell, 0)) : Scalar(0);
                const Scalar scale = relax * (Scalar(1) - penalty);
                if (scale == Scalar(0))
                {
                    for (int i = rank; i < N * components; i += teamSize)
                        output(iCell, i) = DNDS::Math::ScalarCast<real>(Updated(input, fields.target, iCell, i, keep, relax, penalty, scale, Scalar(0)));
                    Join<Tensor>();
                    continue;
                }
                const index begin = plan.offsets(iCell, 0), end = plan.offsets(iCell + 1, 0);
                Scalar accumulated[Components ? (N * Components + 31) / 32 : 1]{};
                using Accumulator = std::conditional_t<TF32, TF32Accumulator, Algebra::CUDA::FP64Accumulator8>;
                Accumulator products[(N + M - 1) / M];
                if constexpr (!Components || wideMMA)
                {
                    for (int i = rank; i < N * components; i += teamSize)
                        right[i] = Scalar(0);
                    Join<Tensor>();
                }
                if constexpr (Pipeline)
                    if (begin < end)
                    {
                        StageEntry<N, Pipeline, Raw>(plan, begin, slots, input, components, rank, teamSize);
                        CommitCopies<Pipeline>();
                    }
                for (index entry = begin; entry < end; ++entry)
                {
                    const int slot = Pipeline ? int((entry - begin) % 2) : 0;
                    if constexpr (!Pipeline)
                        StageEntry<N, Pipeline, Raw>(plan, entry, slots, input, components, rank, teamSize);
                    WaitCopies<Tensor, Pipeline>();
                    if constexpr (Pipeline)
                        if (entry + 1 < end)
                        {
                            StageEntry<N, Pipeline, Raw>(plan, entry + 1, slots + (1 - slot) * entryWords, input, components, rank, teamSize);
                            CommitCopies<Pipeline>();
                        }
                    const auto other = plan.neighbors(entry, 0);
                    const bool rotated = plan.rotated && plan.rotationActive(entry, 0);
                    const bool transpose = plan.faceCoupling && plan.faceSide(entry, 0);
                    if (other != UnInitIndex && (!Raw || other != iCell || rotated))
                    {
                        auto *entrySlot = slots + slot * entryWords;
                        if constexpr (Raw)
                        {
                            if (plan.usePackedB)
                                PromoteRawB<N, Tensor>(entrySlot, rank, teamSize);
                        }
                        else
                            ConvertSlot<N, Storage, Scalar, Tensor>(entrySlot, entryWords, rank, teamSize);
                        const auto *matrix = reinterpret_cast<const Word<Scalar> *>(entrySlot);
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
                                        if constexpr (Components && !wideMMA)
                                            products[modeTile].AddProduct(tile);
                                        else
                                            partial[modeTile].AddProduct(tile);
                                        __syncwarp();
                                    }
                                if constexpr (!Components || wideMMA)
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
                                Product product(0);
                                for (int j = 0; j < N; ++j)
                                    product += Product(matrix[transpose ? (i % N) * N + j : j * N + i % N].value) * Product(Coefficient<N, Scalar>(plan, vectors, entry, coefficients, i / N, j));
                                if constexpr (Components)
                                    accumulated[i / teamSize] += Scalar(product);
                                else
                                    right[i] += Scalar(product);
                            }
                        }
                    }
                    Join<Tensor>();
                }
                if constexpr (Tensor && Components && !wideMMA)
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
                if constexpr (Raw)
                {
                    if (plan.usePackedInverse)
                        StageInverse<N, Triangle>(inverse, plan.packedInverse[iCell], rank, teamSize);
                    else
                        StageInverse<N, Triangle>(inverse, plan.inverse[iCell], rank, teamSize);
                }
                Join<Tensor>();
                for (int i = rank; i < N * components; i += teamSize)
                {
                    Scalar value = right[i];
                    if constexpr (Raw)
                    {
                        value = Scalar(0);
                        for (int j = 0; j < N; ++j)
                            value += inverse[j * N + i % N] * right[(i / N) * N + j];
                    }
                    output(iCell, i) = DNDS::Math::ScalarCast<real>(Updated(input, fields.target, iCell, i, keep, relax, penalty, scale, value));
                }
                Join<Tensor>();
            }
#else
            asm("trap;");
#endif
        }
        template <int N, int Components, bool Raw, bool Triangle, bool LVR, class Storage, class Scalar,
                  bool Tensor, bool TF32, bool Pipeline, class Product = Scalar>
        ReconstructionSweepResources Launch(R::ExecutionContext context, ReconstructionDeviceView plan, Runtime::RowSelection selection,
                                            DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                            R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                            const ReconstructionSweepLaunch &options, bool execute,
                                            R::ArrayView<const real> target = {}, R::ArrayView<const real> alpha = {})
        {
            const int threads = options.threads ? options.threads : Pipeline ? (Tensor ? 128 : 32)
                                                                             : (Tensor ? 128 : 96);
            R::CheckKernelArgumentBudget<Sweep<N, Storage, Triangle>, Runtime::RowSelection, DevicePeriodicBlocks,
                                         Fields<Components, N>, R::ArrayView<const int>, int>();
            const int teams = Tensor ? threads / 32 : 1;
            const index components = output.width / N;
            const index words = (Pipeline ? 2 : 1) * (N * N + N * components) + N * components;
            const index inverseEnd = (words + 3) / 4 * 4 + (Raw ? (N * N + 3) / 4 * 4 : 0);
            const index teamWords = (inverseEnd + (Tensor ? (TF32 ? sizeof(TF32Tile) : sizeof(Algebra::CUDA::FP64Tile8)) / sizeof(real) : 0) + 3) / 4 * 4;
            const index bytes = teamWords * teams * sizeof(real);
            static thread_local int cachedOrdinal = -1;
            static thread_local cudaDeviceProp device{};
            if (cachedOrdinal != context.device.ordinal)
            {
                DNDS_check_throw_info(cudaGetDeviceProperties(&device, context.device.ordinal) == cudaSuccess,
                                      "CUDA VR device properties unavailable");
                DNDS_check_throw_info(cudaFuncSetAttribute(StagedJacobi<N, Components, Raw, Triangle, LVR, Storage, Scalar, Tensor, TF32, Pipeline, Product>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           int(device.sharedMemPerBlockOptin)) == cudaSuccess,
                                      "CUDA VR shared-memory opt-in failed");
                cachedOrdinal = context.device.ordinal;
            }
            DNDS_check_throw_info(device.major >= 8 && bytes <= device.sharedMemPerBlockOptin && bytes <= INT_MAX,
                                  "CUDA VR staging exceeds device shared memory or requires sm80+");
            const index needed = (selection.RowCount() + teams - 1) / teams;
            const unsigned grid = unsigned(options.gridBlocks ? std::min<index>(needed, options.gridBlocks) : std::min<index>(needed, 65535));
            ReconstructionSweepResources resource{bytes, 0, 0, int(grid), threads};
            if (!execute)
            {
                cudaFuncAttributes attributes{};
                DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, StagedJacobi<N, Components, Raw, Triangle, LVR, Storage, Scalar, Tensor, TF32, Pipeline, Product>) == cudaSuccess &&
                                          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resource.activeBlocksPerSM,
                                                                                        StagedJacobi<N, Components, Raw, Triangle, LVR, Storage, Scalar, Tensor, TF32, Pipeline, Product>, threads, size_t(bytes)) == cudaSuccess,
                                      "CUDA VR kernel resource query failed");
                resource.registersPerThread = attributes.numRegs;
                return resource;
            }
            StagedJacobi<N, Components, Raw, Triangle, LVR, Storage, Scalar, Tensor, TF32, Pipeline, Product><<<grid, threads, size_t(bytes), ::DNDS::Runtime::CUDAStream(context)>>>(
                Sweep<N, Storage, Triangle>(plan), selection, vectors,
                Fields<Components, N>{R::ProjectKernelPair<FieldPolicy<Components>>(mean),
                                      R::ProjectKernelPair<FieldPolicy<Components * N>>(input),
                                      R::ProjectKernelView<FieldPolicy<Components>>(boundary),
                                      R::ProjectKernelView<FieldPolicy<Components * N>>(output),
                                      R::ProjectKernelView<FieldPolicy<Components * N>>(target),
                                      R::ProjectKernelView<R::FixedRows<1>>(alpha)},
                status, int(teamWords));
            return resource;
        }
    }
    namespace Staged
    {
        template <int N, int Components>
        ReconstructionSweepResources ProductionShape(const StagedSweepArguments &a)
        {
            const auto launch = [&](auto raw, auto triangle, auto limited)
            {
                return Launch<N, Components, decltype(raw)::value, decltype(triangle)::value,
                              decltype(limited)::value, real, real, false, false, true>(
                    a.context, a.plan, a.selection, a.vectors, a.mean, a.input, a.boundary,
                    a.output, a.status, a.options, a.execute, a.target, a.alpha);
            };
            const auto limited = [&](auto raw, auto triangle)
            {
                return a.limited ? launch(raw, triangle, std::true_type{}) : launch(raw, triangle, std::false_type{});
            };
            if (!a.plan.raw)
                return limited(std::false_type{}, std::false_type{});
            return a.plan.inverseTriangle ? limited(std::true_type{}, std::true_type{}) : limited(std::true_type{}, std::false_type{});
        }
        template <int N>
        ReconstructionSweepResources ProductionModes(const StagedSweepArguments &a)
        {
            switch (a.mean.owned.Width())
            {
            case 4:
                return ProductionShape<N, 4>(a);
            case 5:
                return ProductionShape<N, 5>(a);
            case 6:
                return ProductionShape<N, 6>(a);
            case 7:
                return ProductionShape<N, 7>(a);
            default:
                return ProductionShape<N, 0>(a);
            }
        }
        template <class Storage, class Scalar, bool Tensor, bool TF32, bool Pipeline, class Product = Scalar>
        ReconstructionSweepResources Comparison(const StagedSweepArguments &a)
        {
            DNDS_check_throw_info(a.plan.n == 19 && a.mean.owned.Width() == 5 && !a.plan.raw && !a.limited,
                                  "this comparison VR selector has only the C=5/M=19 precomputed non-LVR fixture");
            return Launch<19, 5, false, false, false, Storage, Scalar, Tensor, TF32, Pipeline, Product>(
                a.context, a.plan, a.selection, a.vectors, a.mean, a.input, a.boundary,
                a.output, a.status, a.options, a.execute);
        }
    }
    ReconstructionSweepResources RunStagedFP64(const StagedSweepArguments &, bool tensor);
    ReconstructionSweepResources RunMatrixShared(const StagedSweepArguments &);
    ReconstructionSweepResources RunQuantizedComparison(const StagedSweepArguments &, int policy);
    ReconstructionSweepResources RunStagedFloatFloat(const StagedSweepArguments &);
    ReconstructionSweepResources RunStagedFP32(const StagedSweepArguments &, bool tensor);
}
