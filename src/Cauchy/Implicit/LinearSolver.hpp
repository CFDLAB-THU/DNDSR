#pragma once

#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "Solver/FieldGMRES.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy
{
    /// Resolved numerical choices, independent of frontend configuration layout.
    /// sgs, jacobi, ilu, amg and amgx are stationary iterations; only gmres constructs
    /// Krylov history and uses the preconditioner selection. ILU/AmgX use frozen
    /// rank-local factors/cycles and x_next=x+P(b-A*x), with a normal halo before
    /// evaluating the full distributed A*x. AmgX has no hidden Krylov driver.
    /// Norms use the integrated RHS, without volume normalization. The default
    /// metric is Euclidean; Solve may supply explicit component scales.
    struct LinearSolveSettings
    {
        std::string method = "sgs", preconditioner = "sgs";
        /// residual: tolerance checks/reporting. fixedCount: stationary methods execute
        /// exactly maxIterations, omit diagnostic residuals and check only the
        /// returned update's finiteness. Tolerances are ignored in fixedCount.
        std::string stopping = "residual";
        index maxIterations = 10;
        /// Host stopping observations, not mathematical product dependencies.
        /// Always observe the final stationary iterate. GMRES also observes at
        /// restart/breakdown; Arnoldi dot products remain mandatory each step.
        index residualCheckInterval = 1;
        int subspace = 20, sgsSweeps = 1, iluFill = 0;
        /// One or two local Jacobi sweeps per exchange. Two freezes remote
        /// input values between sweeps; it is not two synchronized steps.
        /// For GMRES/Jacobi this is the fixed local preconditioner sweep count
        /// from zero; no preconditioner halo is needed in that mode.
        int jacobiLocalSweeps = 1;
        int amgxAggregateSize = 2;
        bool robustACMInverse = false;
        /// Optional block-inverse residual corrections: 0 preserves the
        /// original policy, 1..3 preserves the SVD cutoff/retained subspace.
        /// Applies to SGS/Jacobi, not ILU or provider-owned AMG inverses.
        int diagonalInverseRefinement = 0;
        /// Automatic CUDA block inverse: guardedLU on A100, batchedSVD
        /// elsewhere. Either explicit provider overrides this setup choice.
        /// CPU continues to use its canonical Eigen guarded LU/SVD policy.
        std::string cudaDiagonalInverse = "auto";
        /// Assembled CUDA Jacobi mapping. componentGroup shares one cell among
        /// eight lanes, preserving component-dot and sparse-neighbor order.
        std::string cudaJacobiMapping = "thread";
        /// Fixed zero-started ILU defect corrections per GMRES application.
        /// More than one uses the full operator/halo between local solves.
        /// The count is frozen for Solve; stationary ILU uses maxIterations.
        int iluPreconditionerCorrections = 1;
        bool iluCoreFirst = true, exchangeBetweenSweeps = false;
        real relativeTolerance = .1, absoluteTolerance = 1e-12;
        std::string iluOrdering = "natural";
    };
    struct LinearExecution
    {
        struct Streams
        {
            Runtime::ExecutionStream compute, communication;
        };
        bool overlap = true;
        /// Checked mode coordinates local kernel failures before more MPI.
        /// False is caller-ordered eager mode: callers own distributed failure
        /// handling and identical per-call controls/order. Setup agreement stays
        /// enabled; repeated solve-entry checks follow this switch.
        /// A nonfinite fixed-count result aborts the communicator in unchecked
        /// mode; checked mode returns a globally agreed NonFinite report.
        bool collectiveChecks = true;
        /// Production callers bind their domain's execution lanes. Empty uses
        /// an owned inline CPU runtime for a standalone solve, with exactly the
        /// same operator schedule. Streams are weak handles; their context must
        /// outlive Solve. No solver-specific worker pool is created when bound.
        std::optional<Streams> streams;
    };
    /// Domain-local shared CPU/CUDA linear orchestration. Owns reusable native vectors and
    /// persistent bindings; numerical matrices and kernels remain independent.
    /// All domain ranks construct and call Solve together with agreed controls.
    /// The caller serializes solver/operand access and releases debug views
    /// before reuse. No physical state, time history or PP policy lives here.
    class LinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        using Field = Algebra::DofField<real>;
        using HaloBinder = std::function<Runtime::TransferBinding(Field &)>;
        LinearSolver(std::shared_ptr<LinearProblemContext> problem, const Field &prototype, const HaloBinder &bindHalo,
                     LinearSolveSettings settings = {}, LinearExecution execution = {});
        LinearSolver(const LinearSolver &) = delete;
        LinearSolver &operator=(const LinearSolver &) = delete;
        LinearSolver(LinearSolver &&) noexcept = default;
        LinearSolver &operator=(LinearSolver &&) noexcept = default;
        /// Frozen matrix for this solve. Stationary methods default to zero initial
        /// iterate; a warm guess is explicit. Every GMRES preconditioner call
        /// resets its iterate to zero and uses a fixed number of SGS sweeps or
        /// the same local ILU factors. Standard GMRES must not vary that policy.
        /// Optional positive logical component scales define a frozen inner
        /// product sum(x_j/s_j*y_j/s_j). Empty keeps the Euclidean metric.
        /// Caller supplies agreed scales across ranks. Their width and
        /// positivity are checked once per solve;
        /// no physics or mean-state reduction occurs inside the linear solver.
        Linear::LinearIterationReport Solve(const Field &rhs,
                                            OptionalRef<const Field> initialGuess = {},
                                            const std::vector<real> &componentScales = {});
        /// Sharing result handle; storage survives session destruction but may be
        /// overwritten by the next Solve. Copy values explicitly for history. Always inspect its
        /// report; finite fixed-count or iteration-limited results are usable
        /// without being certified converged. Unevaluated residual norms are NaN.
        Field Solution() const;
    };

    /// The same domain-local solve policy on a complete logical equation
    /// state. Flow/RANS/species may share an array or occupy separate fields in
    /// any supported table layout. Dense blocks couple every logical component;
    /// no implicit device fallback occurs. CUDA currently requires one combined
    /// AoS flow5/SA/2EQ field and Jacobi, AmgX defect correction or GMRES with Jacobi/AmgX. AmgX consumes
    /// the primary native int32, row-major assembled blocks directly (one rank
    /// and compact AoS matrix/vector storage for now), while
    /// DNDS retains the full distributed operator and halo schedule.
    ///
    /// All domain ranks construct/call together. The prototype's physical
    /// grouping and schema must agree across ranks before bindHalo is invoked.
    /// The binder prepares each physical field in that order using the domain's
    /// shared row mapping; it must coordinate its own distributed failures.
    /// Callers serialize access, retain MPI owners on the controller, and release
    /// views before Solve. Results and fixed-preconditioner rules match
    /// LinearSolver. Numerical scopes and transfer requests remain separate.
    class DofLinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        using HaloBinder = LinearSolver::HaloBinder;
        DofLinearSolver(std::shared_ptr<LinearProblemContext> problem, const EquationDofs &prototype,
                        const HaloBinder &bindHalo, LinearSolveSettings settings = {}, LinearExecution execution = {});
        DofLinearSolver(const DofLinearSolver &) = delete;
        DofLinearSolver &operator=(const DofLinearSolver &) = delete;
        DofLinearSolver(DofLinearSolver &&) noexcept = default;
        DofLinearSolver &operator=(DofLinearSolver &&) noexcept = default;
        /// Same optional frozen component-scale metric as LinearSolver.
        Linear::LinearIterationReport Solve(const EquationDofs &rhs,
                                            OptionalRef<const EquationDofs> initialGuess = {},
                                            const std::vector<real> &componentScales = {});
        EquationDofs Solution() const;
    };
}
