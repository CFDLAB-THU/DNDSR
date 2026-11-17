#pragma once

#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "DNDS/Runtime/Array/ArrayView.hpp"

#include <vector>

namespace DNDS::Cauchy
{
    class PointEvaluator;

    /// The mathematical state requested from a prepared physical boundary.
    /// Interface sources ignore this distinction: their sampled receiver-frame
    /// row is already the prescribed point state for every downstream path.
    enum class BoundaryPointStateKind
    {
        FaceTrace,
        ExteriorState
    };

    using InterfaceBoundaryRows =
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>;

    /// Prepared CPU equation-boundary producer. It owns an immutable equation
    /// descriptor and resolved logical offsets; each call receives only point
    /// scratch, geometry and an already-bound interface view. No worker is
    /// acquired here: the caller constructs one PointEvaluator per
    /// concurrent batch and retains it through completion.
    ///
    /// Physical sources evaluate EOS/RANS laws without mutating a cell mean.
    /// Interface sources copy one complete canonical logical row and perform no
    /// EOS, reflection, averaging or physical-BC dispatch. Device backends need
    /// an explicitly resolved producer; this CPU operator is not a fallback.
    class BoundaryPointOperator
    {
        EquationSystem _equation;
        index _components = 0, _flowFirst = -1, _speciesFirst = -1,
              _ransFirst = -1;
        int _independentSpecies = 0;

    public:
        explicit BoundaryPointOperator(EquationSystem equation);

        bool IsACM() const { return _equation.IsACM(); }
        index ComponentNumber() const { return _components; }
        index FlowFirst() const { return _flowFirst; }
        index SpeciesFirst() const { return _speciesFirst; }
        index RANSFirst() const { return _ransFirst; }

        /// Synchronous point kernel. `mean` is required only by a two-equation
        /// no-slip RANS wall. The interface row view is required exactly for an
        /// Interface condition and is otherwise ignored. Point storage is
        /// operator-local scratch and may be mutated; no Array ownership escapes.
        /// Nonthrowing kernel entry. Outputs are invalid on failure.
        NumericalStatus TryEvaluate(
            std::vector<real> &point, const BoundaryCondition &condition,
            index interfaceRow, const Vector3 &normal, const Vector3 &location,
            PointEvaluator &evaluator, const RigidFrame &frame,
            real time, BoundaryPointStateKind stateKind,
            OptionalRef<const std::vector<real>> mean = {},
            real meanWallDistance = 0,
            InterfaceBoundaryRows interfaceValues = {}) const noexcept;
        void Evaluate(
            std::vector<real> &point, const BoundaryCondition &condition,
            index interfaceRow, const Vector3 &normal, const Vector3 &location,
            PointEvaluator &evaluator, const RigidFrame &frame,
            real time, BoundaryPointStateKind stateKind,
            OptionalRef<const std::vector<real>> mean = {},
            real meanWallDistance = 0,
            InterfaceBoundaryRows interfaceValues = {}) const;

        /// Euler-compatible approximate exterior derivative in canonical
        /// logical equation order. Physical/interface targets are frozen.
        /// Wall flow uses the homogeneous reflection; transported scalars are
        /// copied, except no-slip RANS exterior values use `2*target-interior`
        /// and therefore contribute -I. Isothermal/property derivatives remain
        /// outside this approximate LU-SGS model.
        /// Nonthrowing kernel entry. Outputs are invalid on failure.
        NumericalStatus TryApproximateExteriorJacobian(
            Eigen::Ref<Eigen::MatrixXd> output,
            const BoundaryCondition &condition,
            const std::vector<real> &inside, const Vector3 &normal,
            const Vector3 &location, const RigidFrame &frame, real time) const noexcept;
        void ApproximateExteriorJacobian(
            Eigen::Ref<Eigen::MatrixXd> output,
            const BoundaryCondition &condition,
            const std::vector<real> &inside, const Vector3 &normal,
            const Vector3 &location, const RigidFrame &frame, real time) const;
    };
}
