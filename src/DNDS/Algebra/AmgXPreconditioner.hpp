#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include <memory>
#include <optional>

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
     * Matrices, vectors and solver configurations remain private. Simultaneous
     * different-device owners in one process are rejected; use one rank per GPU.
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
        /// blockSize=1 represents a genuinely scalar system; 5/6/7 use native
        /// blocks in settings.blockFormat. The pinned provider patch enables widths 6/7
        /// without padding or scalarizing coupled equations.
        void SetPattern(const Runtime::Array<int> &offsets,
                        const Runtime::Array<int> &columns, int blockSize = 1, bool separateDiagonal = false);
        /// Device values ordered by CSR entry, then configured block order,
        /// stored as a compact scalar table. Upload/rebuild is outside
        /// the Krylov solve. Repeated calls reuse the matrix's published pattern.
        void UpdateValuesBorrowed(Runtime::ExecutionContext context,
                                  Runtime::ArrayView<const real> values,
                                  std::optional<Runtime::ArrayView<const real>> diagonal = {});
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
