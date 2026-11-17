#pragma once
#include "DNDS/Core/NumericalStatus.hpp"

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Boundary/BoundaryKind.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "Physics/Chemistry/ChemicalSource.hpp"
#include <set>

namespace DNDS::Cauchy
{
    class EquationSystem;
    class PointEvaluator;
    struct PerfectGasBoundaryParameters;
    /// Resolved numerical BC, independent of mesh storage and time scheduling.
    /// Construction validates names; a kernel receives plain parameters.
    struct BoundaryCondition
    {
        BoundaryKind kind = BoundaryKind::Interior;
        FlowState farfield{};
        std::array<real, 6> acmState{};
        Vector3 velocity{};
        real temperature = 1;
        bool isothermal = false;
        bool wallMovesWithMesh = true;
        FlowProfileSettings profile;
        FlowState prescribedPrimitive{};
        /// Generic metadata resolves primitive inputs without inventing a
        /// perfect-gas conservative state. Historical PG methods reject it.
        bool hasPerfectGasState = true;
        /// Resolved Ns-1 composition for prescribed states. Empty for a
        /// species-free profile; never resolve names inside numerical loops.
        std::vector<real> prescribedFractions;
        std::array<real, 2> prescribedRANS{};
        /// Total inlet uses [p0,T0,unit direction] in ground axes and primitive
        /// RANS values; density is computed at the current boundary point.
        std::array<real, 5> totalCondition{};
        std::array<real, 2> totalRANSPrimitive{};
        /// Setup-only projection for a perfect-gas CPU/CUDA kernel registry.
        /// Rejects mixture metadata and unresolved analytic profiles; the result
        /// owns only plain values and needs no host condition lifetime.
        PerfectGasBoundaryParameters PerfectGasKernelParameters() const;
        /// Validate the equation-state producer at operator setup. The legacy
        /// perfect-gas methods retain their existing larger profile set.
        void ValidateEquationProducer() const;
        /// Resolve spatially composite boundaries without copying descriptor
        /// storage. DMR prescribes the analytic state at world x < 1/6 and
        /// applies slip reflection elsewhere, including the junction itself.
        BoundaryKind KindAt(const Vector3 &point,
                            OptionalRef<const RigidFrame> frame = {}) const noexcept;
        /// EOS-aware face/exterior state. rhoY is preallocated Ns-1 point
        /// scratch and is replaced only when the BC changes composition or
        /// density. No exchange, Array access, worker acquisition or allocation.
        /// Prescribed velocities use ground axes; results use mesh axes.
        /// Numerical point entry; failure returns a status, never an exception.
        NumericalResult<FlowState> TryFaceValue(PointEvaluator &evaluator, const FlowState &inside,
                                                Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                                const Vector3 &point = {}, real time = 0,
                                                OptionalRef<const RigidFrame> frame = {}) const noexcept;
        FlowState FaceValue(PointEvaluator &evaluator, const FlowState &inside,
                            Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                            const Vector3 &point = {}, real time = 0,
                            OptionalRef<const RigidFrame> frame = {}) const;
        /// Numerical point entry; failure returns a status, never an exception.
        NumericalResult<FlowState> TryExterior(PointEvaluator &evaluator, const FlowState &inside,
                                               Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                               const Vector3 &point = {}, real time = 0,
                                               OptionalRef<const RigidFrame> frame = {}) const noexcept;
        FlowState Exterior(PointEvaluator &evaluator, const FlowState &inside,
                           Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                           const Vector3 &point = {}, real time = 0,
                           OptionalRef<const RigidFrame> frame = {}) const;
        /// Resolved absolute wall velocity in the mesh-bound Cartesian frame.
        /// Shared kinematics for equation-specific adiabatic wall producers.
        Vector3 WallVelocityAt(const Vector3 &point,
                               OptionalRef<const RigidFrame> frame = {}) const;
        /// Perfect-gas numerical entries return failures explicitly. Checked
        /// convenience wrappers below are for setup, diagnostics and bindings.
        NumericalResult<FlowState> TryFaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal,
                                                const Vector3 &point = {}, real time = 0,
                                                OptionalRef<const RigidFrame> frame = {}) const noexcept;
        NumericalResult<FlowState> TryExterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal,
                                               const Vector3 &point = {}, real time = 0,
                                               OptionalRef<const RigidFrame> frame = {}) const noexcept;
        /// Physical wall trace for diagnostics/prescribed wall data. Wall normal
        /// is outward from the interior parent. Extrapolation leaves U unchanged.
        FlowState FaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal,
                            const Vector3 &point = {}, real time = 0,
                            OptionalRef<const RigidFrame> frame = {}) const;
        /// Euler-compatible Riemann/VR exterior state. Isothermal no-slip
        /// reflects velocity and sets rho=p/(R*T_wall), preserving composition.
        /// Direct conservative gradients instead consume (inside+exterior)/2.
        /// Wall velocity is prescribed; this initial
        /// rigid-frame implementation does not imply deforming ALE support.
        FlowState Exterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal,
                           const Vector3 &point = {}, real time = 0,
                           OptionalRef<const RigidFrame> frame = {}) const;
        /// Column-major derivative of Exterior with respect to its conservative
        /// interior state, with branch, normal, prescribed data and time frozen.
        /// The derivative is one-sided at characteristic/DMR branch switches;
        /// callers linearize a selected physical state, not the switch itself.
        /// This model-owned BC derivative is separate from Riemann dissipation.
        std::array<real, 25> ExteriorJacobian(const PerfectGas &gas, const FlowState &inside,
                                              const Vector3 &normal, const Vector3 &point = {},
                                              real time = 0,
                                              OptionalRef<const RigidFrame> frame = {}) const;
        /// Euler's approximate linMode=1 protocol: homogeneous wall reflection
        /// before thermal/scalar correction; zero for non-wall boundaries.
        /// Distinct from the exact physical derivative above. No branch
        /// derivative or wall-temperature derivative enters LU-SGS/Roe blocks.
        std::array<real, 25> ApproximateExteriorJacobian(const PerfectGas &gas, const FlowState &inside,
                                                         const Vector3 &normal, const Vector3 &point = {},
                                                         real time = 0,
                                                         OptionalRef<const RigidFrame> frame = {}) const;
        /// Equation-generic form of the same approximate protocol. The
        /// homogeneous wall reflection is independent of EOS; prescribed/open
        /// and interface states are frozen. Thermal/species/RANS corrections
        /// are composed by BoundaryPointOperator rather than hidden here.
        std::array<real, 25> ApproximateExteriorJacobian(const FlowState &inside, const Vector3 &normal,
                                                         const Vector3 &point = {}, real time = 0,
                                                         OptionalRef<const RigidFrame> frame = {}) const;
    };
    /// Immutable zone-to-condition registry. Constructors resolve configuration
    /// and equation metadata once; point operators borrow individual conditions
    /// without retaining a chemistry worker, mesh or mutable field.
    class BoundarySystem
    {
        std::map<int, BoundaryCondition> _conditions;
        BoundarySystem(const std::map<std::string, BoundarySettings> &settings,
                       OptionalRef<const PerfectGas> gas);

    public:
        /// Resolve the legacy perfect-gas boundary profile set.
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas &gas);
        /// Resolve equation-specific composition once; retains no worker.
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const EquationSystem &equation);
        /// Resolve metadata only; equation-specific operators prepare values.
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings);
        /// Return the resolved condition for one mesh boundary-zone identifier.
        const BoundaryCondition &At(int zone) const;
        /// Require every locally/globally used zone to have a resolved condition.
        void ValidateUsed(const std::set<int> &zones) const;
        /// Check distinct resolved producers, independent of mesh quadrature
        /// count. This is also valid for standalone auxiliary preparation.
        void ValidateEquationProducers(const std::vector<int> &usedZones) const
        {
            for (const int zone : usedZones)
                At(zone).ValidateEquationProducer();
        }
    };
}
