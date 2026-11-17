#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"

namespace DNDS::Cauchy
{
    struct NonlinearSolveGroup;
    /// Method-declared contract checked before an update policy is used. A
    /// policy cannot silently change the number/order of coupled unknowns or
    /// observe a residual at another point in the nonlinear iteration.
    struct UpdatePolicyRequirements
    {
        int coupledStages = 1;
        NonlinearDetail::ResidualObservation observation =
            NonlinearDetail::ResidualObservation::BeforeUpdate;
        bool methodStoppingResidual = false;
    };

    struct UpdatePolicyDescriptor
    {
        std::string name;
        int coupledStages = 1;
        NonlinearDetail::ResidualObservation observation =
            NonlinearDetail::ResidualObservation::BeforeUpdate;
        bool methodStoppingResidual = false;
        int schemaVersion = 1;

        void Validate(const UpdatePolicyRequirements &requirements) const;
        Json Metadata() const;
    };

    /// Solver/operator callbacks for one nonlinear solve. Arithmetic, leases
    /// and MPI remain owned by these adapters; the controller supplies the
    /// immutable settings/schema/rank binding separately.
    struct NonlinearProblem
    {
        NonlinearSolveControl control;
        NonlinearIterationReport initial;
        real reconstructionTolerance = 0;
        std::function<void()> evaluate;
        std::function<void(std::vector<real> &)> measure;
        std::function<std::pair<int, real>()> reconstruction;
        /// Perform exactly one named coupled-stage correction and its
        /// method-owned completion. The policy determines invocation order.
        /// If applyCorrection is false, retain the mean but still complete
        /// method-required rate refreshes. A locally converged/capped domain
        /// must participate in later interface rounds until global termination.
        std::function<void(int, NonlinearIterationReport &, bool applyCorrection)> advanceStage;
        /// Optional opaque owner for solver/session exclusion tickets and
        /// method scratch borrowed by callbacks above.
        std::shared_ptr<void> lifetime;
        /// Optional independent constraint/tolerance ratio, updated by measure.
        /// A ratio greater than one keeps a coupled nonlinear solve active.
        std::function<real()> constraintTargetRatio;
        void Validate(const UpdatePolicyDescriptor &descriptor) const;
    };

    class UpdatePolicy
    {
    public:
        virtual ~UpdatePolicy() = default;
        virtual const UpdatePolicyDescriptor &Descriptor() const = 0;
        /**
         * @brief Execute this policy's complete coordinated nonlinear solve.
         * @details Each concrete policy owns its iteration loop and the order
         * of interface refreshes and domain sweeps. The group supplies shared
         * state/publication helpers, never another loop. One local domain is
         * the same protocol with interface callbacks absent. DITR performs all
         * midpoint updates before refreshing and updating any endpoint.
         */
        virtual void Solve(NonlinearSolveGroup &group) const = 0;
    };

    /// Process-local setup registry. Registration is serialized and intended
    /// for startup/tests, never from a numerical loop. Resolution validates the
    /// method contract before returning a prepared stateless policy.
    class UpdatePolicyRegistry
    {
    public:
        using Factory = std::function<std::unique_ptr<UpdatePolicy>()>;
        static void Register(UpdatePolicyDescriptor descriptor, Factory factory);
        static std::unique_ptr<UpdatePolicy> Resolve(
            const std::string &name,
            const UpdatePolicyRequirements &requirements);
        static std::vector<std::string> Names();
    };
}
