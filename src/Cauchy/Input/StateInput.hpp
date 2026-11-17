#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"

#include <memory>

namespace DNDS::Cauchy
{
    /// Unit-system metadata only. Explicit referenceScales requires rho0/U0/
    /// L0/T0; legacy mixture fields remain a compatibility input. Conflicting
    /// nondefault old/new scale specifications fail before backend creation.
    Physics::ReferenceScales ResolveReferenceScales(const EquationSettings &settings);

    /// Serial setup context. Owns one EOS worker across all spatial points;
    /// each concurrent initialization worker needs its own converter. No
    /// hidden worker acquisition, collectives or global expression state.
    class PhysicalStateConverter
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        PhysicalStateConverter(const EquationSystem &equation, const Physics::ReferenceScales &scales,
                               Physics::Chemistry::ChemicalWorkerPool::Lease worker = {});
        ~PhysicalStateConverter();
        PhysicalStateConverter(const PhysicalStateConverter &) = delete;
        PhysicalStateConverter &operator=(const PhysicalStateConverter &) = delete;
        /// Complete schema-order state, in code units with total energy.
        std::vector<real> Conservative(Physics::StateInput input);
        /// Boundary-only totalPT[_phy] vector. Converts p0/T0/scalar units,
        /// normalizes the prescribed direction, and repairs mass fractions.
        /// It does not invent a static density or evaluate an inlet speed.
        std::vector<real> TotalCondition(Physics::StateInput input);
        /// Seed an expression in any Euler input representation. Conservative
        /// species stay raw; primitive scalars divide by the actual density,
        /// while thermodynamic queries use repaired composition, as in Euler.
        Physics::StateInput Represent(const std::vector<real> &conservative, Physics::StateInput::Form form, bool physical);
    };

    struct ResolvedPhysicalState
    {
        /// Equation schema order: flow5, optional RANS, independent species.
        std::vector<real> conservative;
        FlowState primitive; // rho,u,v,w,p in code units
        real temperature;
        std::map<std::string, real> massFractions, ransConservative;
    };
    /// Controller/setup operation; no MPI or hidden worker acquisition. Owns
    /// the result and consumes one explicit chemistry lease for mixtures.
    /// Primitive species follow Euler's repair; conservative values retain
    /// their raw species, while thermodynamic queries use repaired composition.
    /// Missing density is never replaced by a unit/reference-scale value.
    ResolvedPhysicalState ResolvePhysicalState(const Json &input, const EquationSystem &equation,
                                               const Physics::ReferenceScales &scales,
                                               Physics::Chemistry::ChemicalWorkerPool::Lease worker = {});

    /// Prepared values are separate from mutable input configuration. Treat
    /// this result as one snapshot; rebuild it after changing input settings.
    struct ResolvedPhysicalInputs
    {
        DomainSettings settings;
        std::vector<real> initialConservative, referenceConservative;
    };
    /// Resolve all three config sites through the same conversion. Returned
    /// settings contain code-unit adapter inputs. Original tagged input remains
    /// in the caller's config/provenance; this function does not mutate it.
    /// No collectives: Domain encloses it in its collective setup error guard.
    ResolvedPhysicalInputs ResolvePhysicalStateInputs(const DomainSettings &settings, const EquationSystem &equation,
                                                      const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers);
}
