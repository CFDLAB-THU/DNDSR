#pragma once

#include "DNDS/Core/RowRanges.hpp"

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Core/Span.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::CFV::CPU
{
    namespace R = Runtime;
    using ReconstructionReadRows = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
    using ReconstructionWriteRows = R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>;
    using ReconstructionTable = R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>;
    using ReconstructionIndexRows = R::ArrayView<const index, R::AccessBoundsChecks, R::FixedRows<1>>;
    using ReconstructionField = R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows>;

    /// Borrowed work metadata, retained by the enclosing operator. An all-cell
    /// selection visits physical row order; split SOR uses core-first logical
    /// order inside each mesh subpartition, with frozen cross-partition reads.
    struct ReconstructionWorkSelection
    {
        SpanView<const index> rows, coreFirstPosition, vectorStarts;
        SpanView<const WorkerRowRanges> workerRanges;
        index owned = 0, coreCount = 0, workers = 1;
        bool all = true, core = false, coreFirst = false;
    };
    struct ReconstructionSweepView
    {
        ReconstructionIndexRows offsets, neighbors, boundaryOffsets;
        ReconstructionReadRows matrices, vectors, injection, inverse;
        ReconstructionIndexRows faces;
        R::ArrayView<const uint8_t, R::AccessBoundsChecks, R::FixedRows<1>> faceSides;
        R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<9>> rotations;
        R::ArrayView<const uint8_t, R::AccessBoundsChecks, R::FixedRows<1>> rotationActive;
        ReconstructionField means;
        ReconstructionReadRows current, ghost, target, penalty;
        ReconstructionTable boundary;
        ReconstructionWriteRows output;
        index modes = 0, components = 0;
        real relaxation = 1;
        bool sor = false, rotated = false, limited = false;
        bool raw = false, inverseTriangle = false, faceCoupling = false;
    };
    /// Jacobi or ordered block SOR into a distinct output. All array access and
    /// lifetime decisions have already been resolved by the operator. Returns
    /// invalid-penalty status before writing any row. No leases, MPI or
    /// exception-based numerical recovery; dense adapters validate once in
    /// the prologue before entering the numerical loops.
    NumericalStatus ReconstructionSweep(const ReconstructionSweepView &, const ReconstructionWorkSelection &);

    struct ReconstructionLimiterView
    {
        ReconstructionIndexRows offsets, neighbors, faces;
        ReconstructionTable secondary, forward, inverse;
        R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<9>> rotations;
        R::ArrayView<const uint8_t, R::AccessBoundsChecks, R::FixedRows<1>> rotationActive;
        ReconstructionField coefficients;
        ReconstructionReadRows indicator;
        ReconstructionWriteRows output;
        R::ArrayView<real, R::AccessBoundsChecks, R::TableRows> scratch;
        index modes = 0, components = 0, begin = 0, orderModes = 0;
        index characteristicFirst = 0, characteristicCount = 0;
        real threshold = 0, centralWeight = 1;
        bool rotated = false, hasIndicator = false;
    };
    /// One WBAP order, preserving other modes and scalar components outside
    /// the optional characteristic range. Six scratch matrices per worker are
    /// supplied explicitly. Invalid indicators are rejected before output.
    NumericalStatus ReconstructionLimiterWBAP(const ReconstructionLimiterView &, const ReconstructionWorkSelection &);

    /// Row-wise copy preserving arbitrary uniform physical field layouts.
    void CommitReconstruction(ReconstructionReadRows source, ReconstructionWriteRows destination);

    /// Frozen linear-reconstruction rows. Operators own all leases and the
    /// two dense scratch rows per logical worker. Kernels neither create
    /// buffers nor enter MPI; Eigen may use its internal product workspace.
    /// Matrix rows store column-major blocks as in setup.
    struct ReconstructionLinearView
    {
        ReconstructionIndexRows offsets, neighbors, boundaryOffsets;
        ReconstructionTable matrices, diagonal, vectors, injection, boundary;
        R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<9>> rotations;
        R::ArrayView<const uint8_t, R::AccessBoundsChecks, R::FixedRows<1>> rotationActive;
        ReconstructionField input;
        ReconstructionWriteRows output;
        R::ArrayView<real, R::AccessBoundsChecks, R::TableRows> scratch;
        SpanView<const index> vectorStarts;
        index modes = 0, components = 0;
        bool rotated = false;
    };
    /// Independent destination rows; input and output must be disjoint.
    void ApplyReconstructionLinearOperator(const ReconstructionLinearView &);
    /// Assemble the fixed mean/BC term of the reconstruction equations.
    void BuildReconstructionRightHandSide(const ReconstructionLinearView &);
    /// Apply each cell's diagonal inverse to its independent input row.
    void ApplyReconstructionLocalInverse(ReconstructionTable inverse, ReconstructionReadRows input,
                                         ReconstructionWriteRows output, index modes);
    /// Replace a computed matrix action by rhs-action on independent rows.
    void FormReconstructionResidual(ReconstructionReadRows rhs, ReconstructionWriteRows output);
}
