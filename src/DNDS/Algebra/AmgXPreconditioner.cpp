#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include <cmath>
#include <chrono>
#include <limits>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <vector>
#ifdef DNDS_USE_AMGX
#    include DNDS_AMGX_C_HEADER
#    include <cuda_runtime_api.h>
#    include <cublas_v2.h>
#endif

namespace DNDS::Algebra
{
#ifdef DNDS_USE_AMGX
    int ComputeAmgXBlockInverse(int rows, int width, const int *diag, const double *values, double *inverse);
#endif
    namespace
    {
        template <class T>
        void CheckScalarTable(Runtime::ArrayView<T> view, Runtime::Device device)
        {
            view.ValidateDevice(device);
            DNDS_check_throw_info(view.kind == Runtime::RowLayoutKind::Table && !view.offsets && !view.sizes &&
                                      view.width == 1 && view.stride == 1 && (!view.rows || view.data),
                                  "AmgX requires compact scalar tables");
        }
#ifdef DNDS_USE_AMGX
        // The provider has process-global initialization and configuration state.
        // Serialize adapter calls, not numerical work in unrelated DNDS streams.
        struct ProviderResources;
        struct ProviderRuntime
        {
            std::mutex mutex;
            std::weak_ptr<ProviderResources> resources;
        };
        ProviderRuntime &Provider()
        {
            static ProviderRuntime provider;
            return provider;
        }
        void Check(AMGX_RC code)
        {
            if (code == AMGX_RC_OK)
                return;
            char message[1024]{};
            AMGX_get_error_string(code, message, sizeof(message));
            DNDS_check_throw_info(false, std::string("AmgX: ") + message);
        }
        void CheckCUDA(cudaError_t code)
        {
            DNDS_check_throw_info(code == cudaSuccess, std::string("AmgX CUDA boundary: ") + cudaGetErrorString(code));
        }
        // First-apply diagnostic only. Reduce on the device so a large case
        // copies just one selected scalar per vector to the controller.
        real MaximumMagnitude(const real *data, int count, cublasHandle_t handle)
        {
            int oneBased = 0;
            DNDS_check_throw_info(cublasIdamax(handle, count, data, 1, &oneBased) == CUBLAS_STATUS_SUCCESS &&
                                      oneBased >= 1 && oneBased <= count,
                                  "AmgX diagnostic maximum failed");
            real value;
            CheckCUDA(cudaMemcpy(&value, data + oneBased - 1, sizeof(value), cudaMemcpyDeviceToHost));
            return std::abs(value);
        }
        void ReadAllocatorMemory(AmgXMemorySnapshot &output)
        {
            AMGX_allocator_memory_info info{};
            Check(AMGX_get_allocator_memory(&info));
            output.allocationTracking = info.tracking_enabled;
            output.cudaPoolAvailable = info.cuda_pool_available;
            output.requestedLive = info.requested_live_bytes;
            output.requestedPeak = info.requested_peak_bytes;
            output.liveAllocations = info.live_allocations;
            output.devicePoolUsed = info.device_pool_used_bytes;
            output.devicePoolReserved = info.device_pool_reserved_bytes;
            output.pinnedPoolUsed = info.pinned_pool_used_bytes;
            output.pinnedPoolReserved = info.pinned_pool_reserved_bytes;
            output.cudaPoolUsed = info.cuda_pool_used_bytes;
            output.cudaPoolReserved = info.cuda_pool_reserved_bytes;
            output.cudaPoolUsedPeak = info.cuda_pool_used_peak_bytes;
            output.cudaPoolReservedPeak = info.cuda_pool_reserved_peak_bytes;
        }
        struct DeviceScope
        {
            int previous = -1;
            explicit DeviceScope(int ordinal)
            {
                CheckCUDA(cudaGetDevice(&previous));
                CheckCUDA(cudaSetDevice(ordinal));
            }
            ~DeviceScope()
            {
                if (cudaSetDevice(previous) != cudaSuccess)
                    std::abort();
            }
        };
        std::string Configuration(const AmgXSettings &s)
        {
            // NOSOLVER uses fixed coarsest_sweeps of the same Jacobi smoother.
            // monitor_residual=0 prevents tolerance-dependent nonlinear action.
            // A string-valued smoother selects AmgX's default scope, not its
            // enclosing AMG scope. Put relaxation in the nested smoother:
            // otherwise the requested factor is ignored (upstream default .9).
            std::ostringstream text;
            text << R"({"config_version":2,"determinism_flag":1,"block_format":")"
                 << (s.blockFormat == AmgXBlockFormat::ColumnMajor ? "COL_MAJOR" : "ROW_MAJOR")
                 << R"(","solver":{"solver":"AMG",)"
                 << (s.measureFirstCorrection ? R"("norm":"LMAX","use_scalar_norm":1,)" : "")
                 << (s.geometricAggregation ? R"("weight_formula":2,"structure_reuse_levels":-1,)" : "")
                 << R"("algorithm":"AGGREGATION","selector":"SIZE_)" << s.aggregateSize << R"(","cycle":"V",)"
                 << R"("smoother":{"solver":"BLOCK_JACOBI","monitor_residual":0,"print_solve_stats":0,"relaxation_factor":)"
                 << s.relaxation << R"(},"coarse_solver":"NOSOLVER","monitor_residual":0,)"
                 << R"("print_solve_stats":0,"print_grid_stats":0,"max_iters":)" << s.cycles
                 << R"(,"presweeps":)" << s.preSweeps << R"(,"postsweeps":)" << s.postSweeps
                 << R"(,"coarsest_sweeps":)" << s.coarseSweeps << R"(,"max_levels":)" << s.maxLevels
                 << R"(,"min_coarse_rows":)" << s.minCoarseRows << "}}";
            return text.str();
        }
        /** AmgX Resources owns process-global pools and CUDA library handles.
         * Creating/destroying an independent resource for each matrix can
         * invalidate other live matrices, including during solver replacement
         * and across pMG levels. Keep one shared resource for this rank/device;
         * individual preconditioners still own their configurations and solvers.
         * All ownership transitions happen under Provider().mutex. In particular,
         * the last release occurs after its matrix/vector/solver destruction.
         */
        struct ProviderResources
        {
            Runtime::Device device;
            bool initialized = false;
            AMGX_config_handle config = nullptr;
            AMGX_resources_handle handle = nullptr;
            ~ProviderResources()
            {
                // Called only under the provider lock, on its selected device.
                // No exception recovery or MPI is permitted during retirement.
                auto release = [](AMGX_RC rc)
                { if (rc != AMGX_RC_OK) std::abort(); };
                if (handle)
                    release(AMGX_resources_destroy(handle));
                if (config)
                    release(AMGX_config_destroy(config));
                if (initialized)
                    release(AMGX_finalize());
            }
        };
        std::shared_ptr<ProviderResources> AcquireResources(Runtime::Device device)
        {
            // The caller holds the provider lock and has selected this device.
            if (auto existing = Provider().resources.lock())
            {
                DNDS_check_throw_info(existing->device == device,
                                      "AmgX supports one active device per process; use one MPI rank per GPU");
                return existing;
            }
            auto resources = std::make_shared<ProviderResources>();
            resources->device = device;
            Check(AMGX_initialize());
            resources->initialized = true;
            // Resource parameters are independent of per-matrix cycle settings.
            // Retain this config for the full resource lifetime, even when the
            // preconditioner which first acquired it is destroyed earlier.
            Check(AMGX_config_create(&resources->config, Configuration(AmgXSettings{}).c_str()));
            Check(AMGX_resources_create(&resources->handle, resources->config, nullptr, 1, &device.ordinal));
            Provider().resources = resources;
            return resources;
        }
#endif
    }

    struct AmgXPreconditioner::State
    {
        Runtime::Device device;
        Runtime::Array<int> offsets, columns;
        std::optional<Runtime::Array<int>::ReadLease> offsetRead, columnRead;
        bool separateDiagonal = false, geometricAggregation = false, measureFirstCorrection = false;
        std::optional<Runtime::Array<real>> geometricWeights;
        uint64_t weightEpoch = 0;
        index rows = 0, entries = 0;
        int blockSize = 1;
        bool pattern = false, uploaded = false, ready = false;
        std::optional<Runtime::Array<real>> nativeValues;
        uint64_t nativeEpoch = 0;
        AmgXCopyStatistics copies;
#ifdef DNDS_USE_AMGX
        std::shared_ptr<ProviderResources> provider;
        AMGX_config_handle config = nullptr;
        AMGX_matrix_handle matrix = nullptr;
        AMGX_vector_handle rhs = nullptr, solution = nullptr;
        AMGX_solver_handle solver = nullptr;
        // Native values may already be attached. Reattaching them would reset
        // provider structure and copy indices on every nonlinear iteration.
        void SetupOrRefresh(Runtime::ExecutionContext context)
        {
            const auto start = std::chrono::steady_clock::now();
            const bool refresh = geometricAggregation && uploaded;
            if (refresh)
            {
                Check(AMGX_solver_resetup(solver, matrix));
                ++copies.numericRefreshes;
            }
            else
            {
                if (geometricAggregation)
                {
                    DNDS_check_throw_info(geometricWeights.has_value(), "geometric AmgX needs weights before matrix setup");
                    auto read = geometricWeights->Read(context);
                    DNDS_check_throw_info(geometricWeights->MutationEpoch() == weightEpoch, "AmgX geometric weights changed before setup");
                    Check(AMGX_matrix_attach_edge_weights(matrix, int(entries), read.View().data));
                    CheckCUDA(cudaDeviceSynchronize());
                }
                Check(AMGX_solver_setup(solver, matrix));
            }
            CheckCUDA(cudaDeviceSynchronize());
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (refresh)
                copies.numericRefreshSeconds += seconds;
            else
                copies.hierarchySetupSeconds += seconds;
            ++copies.setups;
        }
        ~State()
        {
            if (!provider)
                return;
            std::lock_guard<std::mutex> lock(Provider().mutex);
            // Destruction is rank-local and cannot throw or start MPI cleanup.
            int previous = -1;
            if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device.ordinal) != cudaSuccess)
                std::abort();
            auto release = [](AMGX_RC rc)
            { if (rc != AMGX_RC_OK) std::abort(); };
            if (solver)
                release(AMGX_solver_destroy(solver));
            if (solution)
                release(AMGX_vector_destroy(solution));
            if (rhs)
                release(AMGX_vector_destroy(rhs));
            if (matrix)
                release(AMGX_matrix_destroy(matrix));
            if (config)
                release(AMGX_config_destroy(config));
            provider.reset();
            if (cudaSetDevice(previous) != cudaSuccess)
                std::abort();
        }
#endif
    };

    bool AmgXPreconditioner::Available() noexcept
    {
#ifdef DNDS_USE_AMGX
        return true;
#else
        return false;
#endif
    }
    AmgXPreconditioner::AmgXPreconditioner(Runtime::Device device, AmgXSettings settings)
        : _state(std::make_unique<State>())
    {
        DNDS_check_throw_info(Available(), "AmgX requested but DNDS_USE_AMGX is disabled");
        DNDS_check_throw_info(device.IsValid() && device.space == Runtime::MemorySpace::CUDA, "AmgX needs one CUDA device");
        DNDS_check_throw_info(settings.aggregateSize == 2 || settings.aggregateSize == 4 || settings.aggregateSize == 8, "AmgX aggregate size must be 2, 4 or 8");
        DNDS_check_throw_info(settings.cycles > 0 && settings.preSweeps >= 0 && settings.postSweeps >= 0 &&
                                  settings.coarseSweeps > 0 && settings.maxLevels > 0 && settings.minCoarseRows > 0 &&
                                  std::isfinite(settings.relaxation) && settings.relaxation > 0 && settings.relaxation <= 1,
                              "invalid fixed-cycle AmgX settings");
        DNDS_check_throw_info(settings.blockFormat == AmgXBlockFormat::RowMajor,
                              "the selected AmgX BLOCK_JACOBI hierarchy requires native row-major blocks; transposition is not implemented");
        _state->device = device;
        _state->geometricAggregation = settings.geometricAggregation;
        _state->measureFirstCorrection = settings.measureFirstCorrection;
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(device.ordinal);
        _state->provider = AcquireResources(device);
        Check(AMGX_config_create(&_state->config, Configuration(settings).c_str()));
        if (settings.robustACMInverse)
        {
            static const AMGX_block_inverse inverse{ComputeAmgXBlockInverse};
            Check(AMGX_config_set_block_inverse(_state->config, &inverse));
        }
        Check(AMGX_matrix_create(&_state->matrix, _state->provider->handle, AMGX_mode_dDDI));
        Check(AMGX_vector_create(&_state->rhs, _state->provider->handle, AMGX_mode_dDDI));
        Check(AMGX_vector_create(&_state->solution, _state->provider->handle, AMGX_mode_dDDI));
        Check(AMGX_solver_create(&_state->solver, _state->provider->handle, AMGX_mode_dDDI, _state->config));
#endif
    }
    AmgXPreconditioner::~AmgXPreconditioner() = default;
    Runtime::Device AmgXPreconditioner::GetDevice() const { return _state->device; }
    index AmgXPreconditioner::RowNumber() const { return _state->rows * _state->blockSize; }

    void AmgXPreconditioner::SetPattern(const Runtime::Array<int> &offsetArray, const Runtime::Array<int> &columnArray,
                                        int blockSize, bool separateDiagonal)
    {
        DNDS_check_throw_info(!_state->nativeValues && !(_state->geometricAggregation && _state->uploaded),
                              "AmgX native attachment freezes the pattern; create another provider for a new pattern");
        DNDS_check_throw_info((blockSize >= 1 && blockSize <= 10) || blockSize == 19, "unsupported AmgX native block width");
        auto offsetRead = offsetArray.Read(), columnRead = columnArray.Read();
        const auto offsets = offsetRead.View(), columns = columnRead.View();
        CheckScalarTable(offsets, Runtime::Device::Host());
        CheckScalarTable(columns, Runtime::Device::Host());
        DNDS_check_throw_info(offsets.rows >= 1 && offsets.rows - 1 <= std::numeric_limits<int>::max() / blockSize &&
                                  columns.rows <= std::numeric_limits<int>::max(),
                              "AmgX block CSR exceeds int32 range");
        const index rows = offsets.rows - 1;
        // The selected provider's Vector resize still multiplies native block
        // counts in int32. Validate numeric extent as well as CSR indices before
        // entering upload/attachment; wide M=19 blocks reach this limit first.
        const index numericBlocks = columns.rows + (separateDiagonal ? rows : 1);
        DNDS_check_throw_info(numericBlocks <= std::numeric_limits<int>::max() / (blockSize * blockSize),
                              "AmgX native numeric extent exceeds provider int32 arithmetic");
        DNDS_check_throw_info(offsets(0, 0) == 0 && offsets(rows, 0) == columns.rows, "invalid AmgX CSR endpoints");
        std::vector<int> row;
        for (index iRow = 0; iRow < rows; ++iRow)
        {
            const int begin = offsets(iRow, 0), end = offsets(iRow + 1, 0);
            DNDS_check_throw_info(begin >= 0 && begin <= end && end <= columns.rows, "invalid AmgX CSR row");
            row.clear();
            bool diagonal = false;
            for (index j = begin; j < end; ++j)
            {
                const int column = columns(j, 0);
                DNDS_check_throw_info(column >= 0 && column < rows, "AmgX requires rank-local native columns; off-rank filtering is unsupported");
                row.push_back(column);
                diagonal |= column == iRow;
            }
            // This local validation scratch never changes the caller's CSR
            // order or numerical block ordinal. AmgX's public upload accepts
            // unique unsorted CSR, including its separate-DIAG representation.
            std::sort(row.begin(), row.end());
            DNDS_check_throw_info(std::adjacent_find(row.begin(), row.end()) == row.end(), "AmgX requires unique native columns");
            DNDS_check_throw_info(separateDiagonal ? !diagonal : diagonal,
                                  "AmgX diagonal presence differs from its declared native representation");
        }
        _state->geometricWeights.reset();
        _state->offsetRead.reset();
        _state->columnRead.reset();
        _state->offsets = offsetArray;
        _state->columns = columnArray;
        _state->offsetRead.emplace(std::move(offsetRead));
        _state->columnRead.emplace(std::move(columnRead));
        _state->rows = rows;
        _state->entries = columns.rows;
        _state->blockSize = blockSize;
        _state->separateDiagonal = separateDiagonal;
        _state->pattern = true;
        _state->uploaded = _state->ready = false;
    }

    void AmgXPreconditioner::SetGeometricWeights(const Runtime::Array<real> &weights)
    {
        DNDS_check_throw_info(_state->geometricAggregation && _state->pattern && !_state->uploaded,
                              "supply geometric weights after setting the pattern and before setup");
        const auto layout = weights.Layout();
        DNDS_check_throw_info(weights.GetDevice() == GetDevice() && layout.Kind() == Runtime::RowLayoutKind::Table &&
                                  layout.Width() == 1 && layout.Stride() == 1 && layout.RowCount() == _state->entries,
                              "AmgX geometry requires one device scalar per CSR slot");
        _state->geometricWeights = weights;
        _state->weightEpoch = weights.MutationEpoch();
    }

    void AmgXPreconditioner::UpdateValuesBorrowed(Runtime::ExecutionContext context, Runtime::ArrayView<const real> values,
                                                  std::optional<Runtime::ArrayView<const real>> diagonal)
    {
        DNDS_check_throw_info(context.device == GetDevice() && _state->pattern, "AmgX update needs its device and pattern");
        DNDS_check_throw_info(!_state->nativeValues, "AmgX native attachment cannot be updated through public upload");
        CheckScalarTable(values, GetDevice());
        DNDS_check_throw_info(values.rows == _state->entries * _state->blockSize * _state->blockSize, "AmgX value count mismatch");
        DNDS_check_throw_info(bool(diagonal) == _state->separateDiagonal, "AmgX separate diagonal is missing or unexpected");
        if (diagonal)
        {
            CheckScalarTable(*diagonal, GetDevice());
            DNDS_check_throw_info(diagonal->rows == _state->rows * _state->blockSize * _state->blockSize,
                                  "AmgX separate diagonal value count mismatch");
        }
        _state->ready = false;
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(context.device.ordinal);
        CheckCUDA(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)));
        if (_state->rows)
        {
            if (_state->uploaded)
                Check(AMGX_matrix_replace_coefficients(_state->matrix, int(_state->rows), int(_state->entries), values.data, diagonal ? diagonal->data : nullptr));
            else
            {
                Check(AMGX_matrix_upload_all(_state->matrix, int(_state->rows), int(_state->entries), _state->blockSize, _state->blockSize,
                                             _state->offsetRead->View().data, _state->columnRead->View().data, values.data, diagonal ? diagonal->data : nullptr));
                Check(AMGX_vector_bind(_state->rhs, _state->matrix));
                Check(AMGX_vector_bind(_state->solution, _state->matrix));
            }
            if (!_state->uploaded)
                _state->copies.patternHostToDevice += uint64_t(sizeof(int)) * (_state->rows + 1 + _state->entries);
            _state->copies.matrixDeviceToDevice += uint64_t(sizeof(real)) * (_state->entries + (_state->separateDiagonal ? _state->rows : 0)) * _state->blockSize * _state->blockSize;
            _state->SetupOrRefresh(context);
        }
        _state->uploaded = _state->ready = true;
#endif
    }
    void AmgXPreconditioner::AttachValues(Runtime::ExecutionContext context, const Runtime::Array<real> &values)
    {
        DNDS_check_throw_info(context.device == GetDevice() && _state->pattern && (!_state->uploaded || _state->nativeValues),
                              "AmgX attachment needs its device, pattern and an unused/native instance");
        const auto layout = values.Layout();
        const index blocks = _state->entries + (_state->separateDiagonal ? _state->rows : 1);
        const index elements = blocks * _state->blockSize * _state->blockSize;
        DNDS_check_throw_info(values.GetDevice() == GetDevice() && layout.Kind() == Runtime::RowLayoutKind::Table &&
                                  layout.Stride() == layout.Width() && layout.RowCount() * layout.Width() == elements,
                              "AmgX attachment requires one compatible compact native values allocation");
        DNDS_check_throw_info(!_state->geometricAggregation || !_state->nativeValues || (values.Identity() == _state->nativeValues->Identity() && values.Storage().SameRange(_state->nativeValues->Storage())),
                              "geometric AmgX refresh requires the originally attached coefficient range");
        _state->ready = false;
        auto read = values.Read(context);
        // Retain before entering the provider so failed setup cannot leave a
        // dangling external numeric view during provider retirement.
        _state->nativeValues = values;
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(context.device.ordinal);
        CheckCUDA(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)));
        if (_state->rows)
        {
            // Matrix owns only a non-owning numeric view; this retained Array
            // owns the actual allocation until after provider destruction.
            if (!_state->geometricAggregation || !_state->uploaded)
            {
                Check(AMGX_matrix_attach_device_values(_state->matrix, int(_state->rows), int(_state->entries), _state->blockSize,
                                                       _state->offsetRead->View().data, _state->columnRead->View().data,
                                                       const_cast<real *>(read.View().data), size_t(elements), _state->separateDiagonal));
                _state->copies.patternHostToDevice += uint64_t(sizeof(int)) * (_state->rows + 1 + _state->entries);
            }
            if (!_state->uploaded)
            {
                Check(AMGX_vector_bind(_state->rhs, _state->matrix));
                Check(AMGX_vector_bind(_state->solution, _state->matrix));
            }
            _state->SetupOrRefresh(context);
        }
#endif
        _state->nativeEpoch = values.MutationEpoch();
        _state->uploaded = _state->ready = true;
    }
    AmgXMemorySnapshot AmgXPreconditioner::MemorySnapshot() const
    {
        AmgXMemorySnapshot output;
        output.nativeStorage = bool(_state->nativeValues);
        output.copies = _state->copies;
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(GetDevice().ordinal);
        CheckCUDA(cudaDeviceSynchronize());
        if (_state->ready && _state->rows)
        {
            int count = 0;
            Check(AMGX_solver_get_hierarchy_memory(_state->solver, 0, nullptr, &count));
            std::vector<AMGX_level_memory_info> levels(size_t(count), AMGX_level_memory_info{});
            Check(AMGX_solver_get_hierarchy_memory(_state->solver, count, levels.data(), &count));
            for (const auto &level : levels)
                output.levels.push_back({level.level, level.block_rows, level.block_nonzeros, level.block_dim,
                                         level.numeric_owned_bytes, level.numeric_borrowed_bytes, level.pattern_bytes,
                                         level.inverse_bytes, level.transfer_bytes, level.workspace_bytes});
        }
        ReadAllocatorMemory(output);
#endif
        return output;
    }
    AmgXMemorySnapshot AmgXPreconditioner::AllocatorMemorySnapshot(Runtime::Device device)
    {
        DNDS_check_throw_info(device.IsValid() && device.space == Runtime::MemorySpace::CUDA && Available(),
                              "AmgX allocator diagnostics require the enabled CUDA provider");
        AmgXMemorySnapshot output;
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(device.ordinal);
        CheckCUDA(cudaDeviceSynchronize());
        ReadAllocatorMemory(output);
#endif
        return output;
    }
    void AmgXPreconditioner::ApplyBorrowed(Runtime::ExecutionContext context, Runtime::ArrayView<const real> residual,
                                           Runtime::ArrayView<real> correction)
    {
        // TODO(AMGX_STREAM_ORDERED, deferred): replace this completed provider
        // boundary with a preallocated caller-stream V-cycle, not a background
        // wrapper around the same device-wide joins. See docs/guides/amgx.md.
        DNDS_check_throw_info(context.device == GetDevice() && _state->ready, "AmgX apply needs its device and completed setup");
        CheckScalarTable(residual, GetDevice());
        CheckScalarTable(correction, GetDevice());
        DNDS_check_throw_info(residual.rows == RowNumber() && correction.rows == RowNumber() &&
                                  (!_state->rows || residual.data != correction.data),
                              "AmgX vector shape or alias mismatch");
        std::optional<Runtime::Array<real>::ReadLease> nativeRead;
        if (_state->nativeValues)
        {
            nativeRead.emplace(_state->nativeValues->Read(context));
            DNDS_check_throw_info(_state->nativeEpoch == _state->nativeValues->MutationEpoch(),
                                  "native AmgX values changed without hierarchy refresh");
            const auto begin = reinterpret_cast<uintptr_t>(nativeRead->View().data);
            const auto bytes = _state->nativeValues->Storage().Bytes();
            const auto overlaps = [begin, bytes](auto view)
            {
                const auto address = reinterpret_cast<uintptr_t>(view.data);
                const auto extent = size_t(view.rows) * sizeof(real);
                return extent && bytes && (address < begin ? begin - address < extent : address - begin < bytes);
            };
            DNDS_check_throw_info(!overlaps(residual) && !overlaps(correction), "AmgX vectors alias attached matrix values");
        }
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(context.device.ordinal);
        CheckCUDA(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)));
        if (!_state->rows)
            return;
        Check(AMGX_vector_upload(_state->rhs, int(_state->rows), _state->blockSize, residual.data));
        Check(AMGX_vector_set_zero(_state->solution, int(_state->rows), _state->blockSize));
        Check(AMGX_solver_solve_with_0_initial_guess(_state->solver, _state->rhs, _state->solution));
        AMGX_SOLVE_STATUS status;
        Check(AMGX_solver_get_status(_state->solver, &status));
        DNDS_check_throw_info(status == AMGX_SOLVE_SUCCESS || status == AMGX_SOLVE_NOT_CONVERGED,
                              "AmgX fixed-cycle preconditioner failed or diverged");
        Check(AMGX_vector_download(_state->solution, correction.data));
        if (_state->measureFirstCorrection && !_state->copies.firstCorrection)
        {
            CheckCUDA(cudaDeviceSynchronize());
            AmgXCopyStatistics::FirstCorrection first{};
            Check(AMGX_solver_calculate_residual_norm(_state->solver, _state->matrix, _state->rhs, _state->solution, &first.defectMax));
            cublasHandle_t handle;
            DNDS_check_throw_info(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "AmgX diagnostic BLAS handle creation failed");
            const auto release = [](cublasContext *h)
            { if (cublasDestroy(h) != CUBLAS_STATUS_SUCCESS) std::abort(); };
            const std::unique_ptr<cublasContext, decltype(release)> owner(handle, release);
            first.rhsMax = MaximumMagnitude(residual.data, int(RowNumber()), handle);
            first.correctionMax = MaximumMagnitude(correction.data, int(RowNumber()), handle);
            _state->copies.firstCorrection = first;
        }
        ++_state->copies.applications;
        _state->copies.rhsDeviceToDevice += uint64_t(sizeof(real)) * RowNumber();
        _state->copies.resultDeviceToDevice += uint64_t(sizeof(real)) * RowNumber();
        CheckCUDA(cudaDeviceSynchronize());
#endif
    }
}
