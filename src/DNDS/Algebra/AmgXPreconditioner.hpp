#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace DNDS::Algebra
{
    enum class AmgXBlockFormat
    {
        RowMajor,
        ColumnMajor
    };
    /// Fixed linear cycle parameters. No residual-based inner stopping or inner
    /// Krylov method: ordinary left GMRES requires a fixed linear action.
    struct AmgXSettings
    {
        int cycles = 1;
        int preSweeps = 2, postSweeps = 2, coarseSweeps = 8;
        int maxLevels = 20, minCoarseRows = 16;
        /// Applied explicitly in the Jacobi smoother scope, including coarsest sweeps.
        real relaxation = 0.75;
        AmgXBlockFormat blockFormat = AmgXBlockFormat::RowMajor;
        int aggregateSize = 2;
        /// Freeze aggregation from supplied face conductances and refresh only
        /// numeric coarse operators and smoother data on coefficient changes.
        bool geometricAggregation = false;
        /// Measure the true residual and correction on the first fixed-cycle application.
        bool measureFirstCorrection = false;
    };

    /// Provider-owned capacities exclude borrowed finest values and allocator reserve.
    struct AmgXLevelMemory
    {
        int level = 0, blockRows = 0, blockNonzeros = 0, blockSize = 0;
        uint64_t numericOwned = 0, numericBorrowed = 0, pattern = 0;
        uint64_t inverse = 0, transfer = 0, workspace = 0;
    };
    struct AmgXCopyStatistics
    {
        uint64_t patternHostToDevice = 0, matrixDeviceToDevice = 0;
        uint64_t rhsDeviceToDevice = 0, resultDeviceToDevice = 0;
        uint64_t setups = 0, applications = 0, numericRefreshes = 0;
        double hierarchySetupSeconds = 0, numericRefreshSeconds = 0;
        struct FirstCorrection
        {
            real rhsMax, correctionMax, defectMax;
        };
        std::optional<FirstCorrection> firstCorrection;
    };
    /** @brief Explicit completed provider diagnostic snapshot.
     * @details requestedLive/Peak are AmgX allocator requests, enabled by
     * AMGX_TRACK_ALLOCATIONS=1 before the first provider resource. They are not
     * physical completion peaks of asynchronous frees. CUDA-pool figures cover
     * this process/device's default CUDA pool, potentially shared with DNDS;
     * do not add them to level capacities or label all reserve as AmgX-owned.
     */
    struct AmgXMemorySnapshot
    {
        std::vector<AmgXLevelMemory> levels;
        AmgXCopyStatistics copies;
        bool allocationTracking = false, nativeStorage = false, cudaPoolAvailable = false;
        uint64_t requestedLive = 0, requestedPeak = 0, liveAllocations = 0;
        uint64_t devicePoolUsed = 0, devicePoolReserved = 0;
        uint64_t pinnedPoolUsed = 0, pinnedPoolReserved = 0;
        uint64_t cudaPoolUsed = 0, cudaPoolReserved = 0, cudaPoolUsedPeak = 0, cudaPoolReservedPeak = 0;
    };

    /** Single-device FP64 AMG for one rank's owned-owned principal matrix.
     * DNDS owns the global operator and every MPI operation. This adapter never
     * receives a communicator. CSR includes all entries of coupled blocks;
     * callers must not discard off-rank contributions from their global operator.
     *
     * Host wrappers below borrow views for the duration of the call. Callers hold
     * leases/queued operand reservations and this object through completion.
     * AmgX has no application-stream argument: calls wait for the supplied CUDA
     * stream, run the provider, then synchronize its device before returning.
     * This is an explicit host boundary, not a captureable native-stream kernel.
     * The scheduler must drain pending host-progressed communication before this
     * boundary; a device-wide provider join must not wait on MPI progress that
     * only the currently blocked controller could perform.
     * Provider failures throw here; production collective solver catches abort.
     * A pattern/value change is forbidden during a GMRES solve by the caller.
     * Concurrent preconditioner owners share one process-wide provider resource
     * on one device, preserving AmgX pools across replacement and pMG levels.
     * Matrix descriptors, coarse values, vectors and configurations stay private;
     * explicit native attachment shares only the managed finest numeric values. Simultaneous
     * different-device owners in one process are rejected; use one rank per GPU.
     * @todo DEFERRED stream-ordered apply/refresh: add a restricted preallocated
     * caller-stream V-cycle to the provider fork; remove its default-stream
     * kernels, lazy scratch, synchronous vector copies and cudaFreeWait joins.
     * Native numeric attachment alone does not provide this capability. Flow
     * also needs queued coarse-value/inverse refresh when its Jacobian changes.
     * Keep hierarchy rebuilds as explicit setup boundaries until implemented.
     * See docs/guides/amgx.md; do not silently move these joins into a worker
     * and label that path stream-ordered or captureable.
     */
    class AmgXPreconditioner
    {
        struct State;
        std::unique_ptr<State> _state;

    public:
        static bool Available() noexcept;
        explicit AmgXPreconditioner(Runtime::Device device, AmgXSettings settings = {});
        ~AmgXPreconditioner();
        AmgXPreconditioner(const AmgXPreconditioner &) = delete;
        AmgXPreconditioner &operator=(const AmgXPreconditioner &) = delete;

        /// Share the caller's native compact int32 CSR allocations, retaining
        /// read leases until replacement/destruction. No conversion, sorting or
        /// index copy occurs. Columns must be unique and local. A separate
        /// diagonal permits the CSR to contain off-diagonal entries only.
        /// Zero-row ranks use offsets={0}, columns={} and perform an empty apply.
        /// Native widths 1/2/3/4/5/6/7/8/9/19 are supported without padding or
        /// scalarization; width 10 remains available for the provider regression
        /// reference. This block catalog does not expand Cauchy's physics scope.
        /// After native attachment the pattern is frozen; construct another
        /// provider for a replacement pattern. This is checked before mutation.
        void SetPattern(const Runtime::Array<int> &offsets,
                        const Runtime::Array<int> &columns, int blockSize = 1, bool separateDiagonal = false);
        /// Supply one immutable CUDA FP64 conductance per CSR slot, after SetPattern
        /// and before the first setup. The fork consumes a copy at first setup.
        /// A new geometry, numbering or connectivity requires a new provider.
        void SetGeometricWeights(const Runtime::Array<real> &weights);
        /// Device values ordered by CSR entry, then configured block order,
        /// stored as a compact scalar table. Upload/rebuild is outside
        /// the Krylov solve. Repeated calls reuse the matrix's published pattern.
        void UpdateValuesBorrowed(Runtime::ExecutionContext context,
                                  Runtime::ArrayView<const real> values,
                                  std::optional<Runtime::ArrayView<const real>> diagonal = {});
        /** @brief Deep integration: attach one managed native numeric allocation.
         * @details Compact row-major device storage contains E off-diagonal
         * blocks followed by N separate diagonal blocks, or E inline-diagonal
         * blocks followed by one zero sentinel block. No numeric copy occurs.
         * The retained Array pins the allocation; read leases are held during
         * setup/apply only. The caller may mutate between solves and must call
         * AttachValues again before applying. Stale values are rejected by epoch.
         * This synchronous provider route still has explicit stream/device joins.
         * Upload and attachment modes cannot be mixed on one provider instance.
         */
        void AttachValues(Runtime::ExecutionContext context, const Runtime::Array<real> &values);
        /// Query capacities, logical allocator peaks and exact boundary copy counts.
        /// Explicitly synchronizes the provider device; diagnostic use only.
        AmgXMemorySnapshot MemorySnapshot() const;
        /// Diagnostic allocator/pool state, also valid after the last provider retires.
        static AmgXMemorySnapshot AllocatorMemorySnapshot(Runtime::Device device);
        /// Apply from zero with fixed cycles. Input/output are compact scalar
        /// tables (N x 1), never aliased. Completion is host-visible on return.
        void ApplyBorrowed(Runtime::ExecutionContext context,
                           Runtime::ArrayView<const real> residual,
                           Runtime::ArrayView<real> correction);
        Runtime::Device GetDevice() const;
        /// Number of scalar unknowns, including components of native blocks.
        index RowNumber() const;
    };
}
