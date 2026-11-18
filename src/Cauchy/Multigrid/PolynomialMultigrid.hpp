#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"

namespace DNDS::Cauchy
{
    class NonlinearSolveSetup;
    class DomainExecutionContext;
    /// Same-cell FAS hierarchy: high-order VR -> direct-gradient O2 -> first
    /// order convection. All levels retain viscous gradients/physical BCs.
    /// Normalized state, residual and correction maps are identity on means.
    /// Different meshes/coarsening belong to a different transfer adapter.
    /// Logical equation states support full or selected-component principal-
    /// system cycles through the same hierarchy and FAS recursion.
    /// Unselected means remain frozen; all finest residuals stay observable.
    /// Fine state/workspace remain bound. The hierarchy retains a copy of the
    /// finest solver's shared implementation; its original facade may move or
    /// expire. Each cycle reserves that session explicitly; original solver
    /// calls reject while the hierarchy uses it. The hierarchy is move-only.
    /// Its solvers lock the weak runtime at operation entry. Caller owns physical
    /// trial rollback and still coordinates rank schedules.
    class PolynomialMultigrid
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        PolynomialMultigrid(const PolynomialMultigrid &) = delete;
        PolynomialMultigrid &operator=(const PolynomialMultigrid &) = delete;
        PolynomialMultigrid(PolynomialMultigrid &&) noexcept = default;
        PolynomialMultigrid &operator=(PolynomialMultigrid &&) noexcept = default;
        /// Zero-based coarse-level controls, also usable to reproduce a level
        /// operator independently. Level 0 uses direct-gradient O2 convection;
        /// subsequent levels use mean convection with the same gradients for
        /// diffusion and sources. No mesh/state ownership is changed.
        static DomainSettings CoarseSettings(const DomainSettings &fine, index level);
        /// Preserve each physical allocation's layout and every block's
        /// component map on all levels; no packing. Coarse operators retain
        /// the same immutable equation and worker pool.
        PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, EquationDofs &state,
                            NonlinearSolver &fineSolver, Runtime::EagerEngine &engine, Runtime::ExecutionStream stream,
                            const DomainSettings &settings, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt);
        /// Orchestration constructor retaining the domain execution owner.
        PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine,
                            EquationDofs &state, NonlinearSolver &fineSolver,
                            DomainExecutionContext context,
                            const DomainSettings &settings);
        /// Accept only the original finest uncompressed equation, never the
        /// coarse FAS forcing. Caps return unconverged. Coarse levels have no
        /// mutable VR warm guesses; they are reset by restriction every cycle.
        NonlinearIterationReport Solve(const EquationDofs &source, real mass, real rhsScale, real time, int stage = 0,
                                       NonlinearSolveControl control = {});
        /// Prepared finest nonlinear session for simulation-level domain
        /// coordination. Coarse cycles remain bounded corrections inside each
        /// named update and never own the outer iteration cursor.
        NonlinearSession PrepareSolve(
            const NonlinearSolveSetup &setup, const EquationDofs &source,
            real mass, real rhsScale, real time, int stage = 0,
            NonlinearSolveControl control = {});
        /// One finest update, with bounded CorrectOnce smoothing at each level.
        /// No convergence loop/reference reset; pseudoCFL comes from the caller.
        /// Other-stage inputs in source must remain frozen throughout the cycle.
        /// Optional rate is the caller's already evaluated finest dU/dt, as in
        /// CorrectOnce. Caller guarantees current owned means/time/stage and
        /// reconstruction provenance. It is consumed only by the first finest
        /// smoother; further corrections evaluate their updated states normally.
        NonlinearIterationReport ApplyCycle(const EquationDofs &source, real mass, real rhsScale, real time,
                                            int stage = 0, real pseudoCFL = 0,
                                            OptionalRef<const EquationDofs> rate = {}, OptionalRef<const Runtime::Array<real>> derivative = {});
        /// Frozen stage trace shared with coarse levels after checking their
        /// canonical boundary quadrature order. Local alias binding only; no MPI.
        void SetBoundaryInputs(const BoundaryInputs &input);
        /// Retain an ordered internal packet-binding operation for a prepared
        /// domain solve. Invoked only before the next finest evaluation, while
        /// no coarse cycle executes. This changes packet handles, not payloads;
        /// ordinary external callers use the guarded SetBoundaryInputs method.
        std::function<void(const BoundaryInputs &)> PrepareBoundaryInputBinding() const;
        /// Reserved DoF payload bytes and peak simultaneous borrowed slots.
        /// Excludes persistent state/source/reference and reconstruction storage.
        std::pair<size_t, index> ScratchPoolStatistics() const;
    };
}
