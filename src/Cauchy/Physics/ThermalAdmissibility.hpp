#pragma once

#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "Physics/Chemistry/SpeciesAdmissibility.hpp"
#include "DNDS/Runtime/Storage/Metadata.hpp"

namespace DNDS::Cauchy
{
    /// Response to a failed clipped-composition reconstruction-energy check.
    /// The predicate is shared; this enum changes only whether the complete
    /// flow/species polynomial drops to its base or searches an intermediate beta.
    enum class ReconstructionClippedReferencePolicy
    {
        FallbackToBase,
        Backtracking
    };

    namespace ThermalAdmissibilityDetail
    {
        template <class Values>
        struct PointRow
        {
            const Values &values;
            index point;
            DNDS_DEVICE_CALLABLE real operator[](index j) const { return values(point, j); }
        };
        template <class Base, class Delta>
        struct SegmentRow
        {
            const Base &base;
            const Delta &delta;
            real fraction;
            DNDS_DEVICE_CALLABLE real operator[](index j) const { return base[j] + fraction * delta[j]; }
        };
        template <class Base, class Higher>
        struct DifferenceRow
        {
            const Base &base;
            const Higher &higher;
            real scale;
            DNDS_DEVICE_CALLABLE real operator[](index j) const { return scale * (higher[j] - base[j]); }
        };
        template <class Row>
        struct OffsetRow
        {
            const Row &values;
            index first;
            DNDS_DEVICE_CALLABLE real operator[](index j) const
            {
                return values[first + j];
            }
        };
    }

    /// Frozen numerical view. Energy references are prepared e_k(T) values,
    /// not EOS callbacks. Caller retains the policy owner and declared input
    /// scopes. Rows supply operator[] and may span several physical arrays.
    /// This view contains no ownership, Array, worker, MPI or allocation.
    struct ThermalView
    {
        real densityFloor;
        std::array<real, 2> floors, densityCoefficients;
        std::array<SpanView<const real>, 2> speciesEnergies{};
        index components, flowFirst, speciesFirst, independent;

        /// Flow and transported species share reconstruction compression.
        /// RANS reconstruction uses separate scalar rules. This is not the
        /// final increment policy: Euler scales the whole increment there.
        /// Indices are logical equation components, independent of storage layout.
        DNDS_DEVICE_CALLABLE bool CompressesReconstructionComponent(index j) const
        {
            return (j >= flowFirst && j < flowFirst + 5) ||
                   (j >= speciesFirst && j < speciesFirst + independent);
        }

        template <class Row>
        DNDS_DEVICE_CALLABLE FlowState Flow(const Row &values) const
        {
            FlowState result;
            for (int j = 0; j < 5; ++j)
                result[j] = values[flowFirst + j];
            return result;
        }
        /// Raw reference, affine in transported conservative values. It drives
        /// reconstruction's analytic beta and final-mean line-search proofs.
        /// Reconstruction retains these raw rhoY values for species advection
        /// and diffusion, while final point-property acceptance checks only a
        /// temporary clipped copy. The raw reconstruction bound is a limiter
        /// construction, not an additional EOS/T acceptance invariant.
        template <class Row>
        DNDS_DEVICE_CALLABLE real Reference(const Row &values, int constraint) const
        {
            const real rho = values[flowFirst];
            const auto energy = speciesEnergies[constraint];
            if (energy.empty())
                return rho * densityCoefficients[constraint];
            real value = rho * energy[independent];
            for (index k = 0; k < independent; ++k)
                value += values[speciesFirst + k] * (energy[k] - energy[independent]);
            return value;
        }
        /// Property-forecast reference. This mirrors Euler's tolerant temporary
        /// composition (clip negative fractions, restore the dependent species,
        /// then normalize) without changing stored rhoY. It is deliberately
        /// separate from Reference(): the map is not affine and therefore must
        /// never be used by the exact stored-state FixFraction line search.
        template <class Row>
        DNDS_DEVICE_CALLABLE real ClippedReference(const Row &values, int constraint) const
        {
            const real rho = values[flowFirst];
            const auto energy = speciesEnergies[constraint];
            if (energy.empty())
                return rho * densityCoefficients[constraint];
            real sum = 0;
            for (index k = 0; k < independent; ++k)
                sum += std::max(real(0), values[speciesFirst + k] / rho);
            const real dependent = std::max(real(0), 1 - sum), total = sum + dependent;
            real value = 0;
            for (index k = 0; k < independent; ++k)
                value += std::max(real(0), values[speciesFirst + k] / rho) / total * energy[k];
            value += dependent / total * energy[independent];
            return rho * value;
        }
        template <class Row>
        DNDS_DEVICE_CALLABLE real Sensible(const Row &values, int constraint) const
        {
            return PositivityDetail::InternalEnergy(Flow(values)) - Reference(values, constraint);
        }
        template <class Row>
        DNDS_DEVICE_CALLABLE real ClippedSensible(const Row &values, int constraint) const
        {
            return PositivityDetail::InternalEnergy(Flow(values)) - ClippedReference(values, constraint);
        }
        /// Raw affine density/energy predicate. It deliberately ignores the
        /// independent/dependent species simplex and performs no clipping.
        template <class Row>
        DNDS_DEVICE_CALLABLE bool RawThermallyAdmissible(const Row &values) const
        {
            for (index j = 0; j < components; ++j)
                if (!std::isfinite(values[j]))
                    return false;
            if (!(values[flowFirst] >= densityFloor))
                return false;
            for (int c = 0; c < 2; ++c)
                if (!(Sensible(values, c) >= floors[c]))
                    return false;
            return true;
        }
        /// Temporary physical-point predicate. Species are clipped/normalized
        /// only inside ClippedReference(); the supplied row is never mutated.
        /// Passing this bound makes the corresponding EOS temperature floor
        /// representable after point preparation, without imposing a simplex
        /// constraint on reconstruction coefficients.
        template <class Row>
        DNDS_DEVICE_CALLABLE bool ClippedReferenceAdmissible(const Row &values) const
        {
            for (index j = 0; j < components; ++j)
                if (!std::isfinite(values[j]))
                    return false;
            if (!(values[flowFirst] >= densityFloor))
                return false;
            for (int c = 0; c < 2; ++c)
                if (!(ClippedSensible(values, c) >= floors[c]))
                    return false;
            return true;
        }
        /// Accepted mean-state predicate. Unlike reconstruction, a stored mean
        /// must contain the simplex-repaired rhoY that is actually supplied to
        /// EOS/T; that clipped composition defines the physical invariant. Once
        /// the simplex holds, Reference() and ClippedReference() are identical.
        /// The raw check retained here is therefore a matching arithmetic guard,
        /// not permission to accept a second raw-species representation.
        template <class Row>
        DNDS_DEVICE_CALLABLE bool Admissible(const Row &values, bool requireSpecies = true) const
        {
            if (!RawThermallyAdmissible(values) || !ClippedReferenceAdmissible(values))
                return false;
            if (requireSpecies)
            {
                real sum = 0;
                for (index k = 0; k < independent; ++k)
                {
                    if (values[speciesFirst + k] < 0)
                        return false;
                    sum += values[speciesFirst + k];
                }
                if (sum > values[flowFirst])
                    return false;
            }
            return true;
        }
        /// PP POLICY -- whole-vector clipped-reference forecast.
        /// Euler density relaxation and scheme-one energy compression.
        /// Returns alpha for U+alpha*dU: density first, then both relaxed
        /// energy bounds, followed by alpha^(compression*power)*(1-1e-5)
        /// when limited. The caller scales the complete increment/solve RHS,
        /// or uses alpha to restrict its scalar dt/dTau. RANS positivity is
        /// not part of the factor calculation. Species affect the thermal
        /// references and must move with flow along this conservative line.
        /// The old state must be admissible; a nonfinite result must fail.
        /// Scalar repair and final combined-state validation remain required.
        template <class Row, class Increment>
        DNDS_DEVICE_CALLABLE real IncrementFraction(const Row &state, const Increment &increment, real relax, int compression, int power) const
        {
            using namespace ThermalAdmissibilityDetail;
            const real rho = state[flowFirst], deltaRho = increment[flowFirst];
            const real relaxedDensity = densityFloor * relax + rho * (1 - relax);
            const real density = deltaRho < 0 ? std::min(real(1), (rho - relaxedDensity) / (-deltaRho - smallReal * deltaRho)) : 1;
            const SegmentRow<Row, Increment> next{state, increment, density};
            auto delta = Flow(increment);
            for (auto &value : delta)
                value *= density;
            real energy = 1;
            for (int c = 0; c < 2; ++c)
            {
                const real oldEnergy = ClippedSensible(state, c), newEnergy = ClippedSensible(next, c);
                if (!std::isfinite(oldEnergy) || !std::isfinite(newEnergy))
                    return std::numeric_limits<real>::quiet_NaN();
                const real floor = newEnergy < oldEnergy ? floors[c] + (oldEnergy - floors[c]) * (1 - relax) : floors[c];
                if (newEnergy < floor)
                {
                    // Euler's forecast limiter evaluates the same tolerant
                    // property composition as its EOS path. This is a guarded
                    // predictor, not the exact stored-state line search below.
                    const real value = InternalEnergyCompression(Flow(state), delta, floor,
                                                                 ClippedReference(state, c), ClippedReference(next, c));
                    if (!std::isfinite(value))
                        return value;
                    energy = std::min(energy, value);
                }
            }
            const real fraction = density * energy;
            return fraction < 1 ? std::pow(fraction, compression * power) * (1 - 1e-5) : 1;
        }
        /// PP POLICY -- exact stored-mean repair.
        /// Euler final density/energy increment repair (.99 safety, .9 pull-up).
        /// Returns the whole-increment multiplier, or NaN if pull-up fails.
        /// Callers must reject a nonfinite result, scale all components, then
        /// perform the separate RANS repair and validate the combined state.
        /// The supplied increment must already point to the exact simplex-repaired
        /// species target. Those clipped rhoY values are the actual final
        /// stored/exchanged mean and therefore define the EOS/T invariant. Because
        /// both endpoints are in the complete simplex, the raw affine reference
        /// equals that clipped composition everywhere on their connecting line;
        /// the cheaper raw-affine search is only a proof of the clipped-state
        /// physical invariant, never permission to retain a raw out-of-simplex
        /// mean. No projection follows this search because the candidate was
        /// already repaired before it. This is distinct from
        /// reconstruction, where raw polynomial rhoY is retained and clipping is
        /// only a temporary point-property view. See CauchyPositivity.md.
        template <class Row, class Increment>
        DNDS_DEVICE_CALLABLE real FixFraction(const Row &state, const Increment &increment) const
        {
            using namespace ThermalAdmissibilityDetail;
            real fraction = 1;
            if (state[flowFirst] + increment[flowFirst] <= densityFloor)
                fraction = (densityFloor - state[flowFirst]) / (increment[flowFirst] - verySmallReal);
            for (int c = 0; c < 2; ++c)
            {
                SegmentRow<Row, Increment> next{state, increment, fraction};
                if (!std::isfinite(Sensible(next, c)))
                    return std::numeric_limits<real>::quiet_NaN();
                if (Sensible(next, c) <= floors[c])
                {
                    auto delta = Flow(increment);
                    for (auto &value : delta)
                        value *= fraction;
                    const real alpha = InternalEnergyCompression(Flow(state), delta, floors[c], Reference(state, c), Reference(next, c));
                    if (!std::isfinite(alpha))
                        return alpha;
                    fraction *= alpha * .99;
                    for (int iteration = 0; iteration < 1000; ++iteration)
                    {
                        next.fraction = fraction;
                        if (Sensible(next, c) >= floors[c])
                            break;
                        fraction *= .9;
                    }
                    next.fraction = fraction;
                    if (!(Sensible(next, c) >= floors[c]))
                        return std::numeric_limits<real>::quiet_NaN();
                }
            }
            const SegmentRow<Row, Increment> final{state, increment, fraction};
            for (int c = 0; c < 2; ++c)
                if (!(Sensible(final, c) >= floors[c]))
                    return std::numeric_limits<real>::quiet_NaN();
            return fraction;
        }
        /// PP POLICY -- reconstruction analytic beta and physical endpoint.
        /// Euler flow/species quadrature beta. Apply only to components selected
        /// by CompressesReconstructionComponent(), relative to the supplied
        /// admissible base. This differs from whole-vector increment repair.
        /// densityLimit is an optional SA reduced-order ratio in [0,1], merged
        /// into the density stage before the energy test and beta power.
        /// Raw affine references determine the concave analytic ratio. The
        /// clipped-reference predicate is the final physical endpoint because
        /// it matches point preparation before the EOS/T solver; it never
        /// mutates the polynomial or imposes a species-simplex beta.
        ///
        /// TODO(PP-RECONSTRUCTION-CLIPPED-RHOY): evaluate, with matched
        /// Euler/Cauchy oracles, whether the analytic thermal beta itself should
        /// use the clipped/normalized reference. That map is heuristic and
        /// non-convex; do not change it without retaining endpoint backtracking.
        template <class Row, class BaseValues, class HigherValues>
        DNDS_DEVICE_CALLABLE real ReconstructionFraction(const Row &mean, const BaseValues &baseValues, const HigherValues &higherValues,
                                                         index begin, index end, int power, real cutoff, real densityLimit = 1,
                                                         ReconstructionClippedReferencePolicy clippedPolicy = ReconstructionClippedReferencePolicy::FallbackToBase) const
        {
            using namespace ThermalAdmissibilityDetail;
            bool rawGood = true, clippedGood = true;
            for (index q = begin; q < end; ++q)
            {
                const PointRow<HigherValues> point{higherValues, q};
                rawGood = rawGood && RawThermallyAdmissible(point);
                clippedGood = clippedGood && ClippedReferenceAdmissible(point);
            }
            if (rawGood && clippedGood && densityLimit == 1)
                return 1;
            real density = densityLimit;
            for (index q = begin; q < end; ++q)
            {
                const real base = baseValues(q, flowFirst), delta = higherValues(q, flowFirst) - base;
                if (delta < 0)
                    density = std::min(density, (base - densityFloor) / (-delta + verySmallReal));
            }
            real energy = 1;
            for (int c = 0; c < 2; ++c)
            {
                if (Sensible(mean, c) <= 2 * floors[c])
                {
                    energy = 0;
                    continue;
                }
                for (index q = begin; q < end; ++q)
                {
                    const PointRow<BaseValues> base{baseValues, q};
                    const PointRow<HigherValues> high{higherValues, q};
                    const DifferenceRow<decltype(base), decltype(high)> delta{base, high, density};
                    const SegmentRow<decltype(base), decltype(delta)> next{base, delta, 1};
                    if (!std::isfinite(Sensible(next, c)))
                        return std::numeric_limits<real>::quiet_NaN();
                    if (Sensible(next, c) < 2 * floors[c])
                    {
                        const real value = InternalEnergyCompression(Flow(base), Flow(delta), floors[c], Reference(base, c), Reference(next, c));
                        if (!std::isfinite(value))
                            return value;
                        energy = std::min(energy, value);
                    }
                }
            }
            const real fraction = density * energy;
            const real compressed = fraction < 1 ? fraction * std::pow(fraction, power) * (1 - 1e-5) : 1;
            real beta = compressed < cutoff ? 0 : compressed;
            const auto clippedAt = [&](real candidate)
            {
                bool good = true;
                for (index q = begin; q < end; ++q)
                {
                    const PointRow<BaseValues> base{baseValues, q};
                    const PointRow<HigherValues> high{higherValues, q};
                    const DifferenceRow<decltype(base), decltype(high)> delta{base, high, candidate};
                    const SegmentRow<decltype(base), decltype(delta)> point{base, delta, 1};
                    good = good && ClippedReferenceAdmissible(point);
                }
                return good;
            };
            if (!clippedAt(0))
                return std::numeric_limits<real>::quiet_NaN();
            if (clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase)
                return clippedAt(beta) ? beta : 0;
            for (int iteration = 0; iteration < 1000 && !clippedAt(beta); ++iteration)
            {
                beta *= .9;
                if (beta < cutoff)
                    return 0;
            }
            return clippedAt(beta) ? beta : 0;
        }
    };

    /// CPU preparation owner for two affine bounds: e(Tmin,Y) and the model's
    /// reference-energy margin. For a perfect gas these reduce to the existing
    /// temperature and pressure energy constraints. Mixtures use the preserved
    /// backend reference temperature and an explicit sensible-energy margin.
    /// Immutable coefficient metadata uses shared backend allocations. View()
    /// is a host projection; retain this policy for the full operation. A future
    /// device operator must explicitly publish device coefficient descriptors.
    class ThermalPolicy
    {
        std::array<Runtime::MetadataBuffer<real>, 2> _energies;
        ThermalView _view{};

    public:
        ThermalPolicy(const EquationSystem &equation, real densityFloor, real temperatureFloor,
                      real sensibleEnergyFloor, real pressureFloor, Physics::Chemistry::ChemicalWorkerPool::Lease worker = {});
        ThermalView View() const
        {
            auto result = _view;
            for (int c = 0; c < 2; ++c)
                result.speciesEnergies[c] = _energies[c].HostView();
            return result;
        }
    };

    /// PP POLICY -- accepted mean-state boundary.
    /// Frozen complete mean-state admissibility view. Thermal flow/species
    /// constraints and model-specific RANS scalar constraints are resolved once
    /// by AdmissibilityPolicy; row checks perform no EOS inversion,
    /// allocation, worker acquisition, MPI, or physical-layout dispatch.
    struct AdmissibilityView
    {
        ThermalView thermal;
        RANSStateView rans;
        index ransFirst = -1;

        template <class Row>
        DNDS_DEVICE_CALLABLE bool Admissible(const Row &values,
                                             bool requireSpecies = true) const
        {
            if (thermal.flowFirst < 0)
            {
                for (index j = 0; j < thermal.components; ++j)
                    if (!std::isfinite(values[j]))
                        return false;
            }
            else if (!thermal.Admissible(values, requireSpecies))
                return false;
            if (!rans.components)
                return true;
            return rans.Admissible(
                ThermalAdmissibilityDetail::OffsetRow<Row>{values, ransFirst});
        }
    };

    /// CPU owner for the complete equation mean-state admissibility contract.
    /// Mixture energy coefficients are prepared once with the caller's worker;
    /// the resulting View remains valid until this owner is moved, mutated, or
    /// destroyed. RANS repair remains a separate explicit operation, but its
    /// acceptance predicate is part of the same frozen view.
    class AdmissibilityPolicy
    {
        ThermalPolicy _thermal;
        EquationPropertyKey _property;
        RANSStateView _rans;
        index _ransFirst = -1;

    public:
        AdmissibilityPolicy(
            const EquationSystem &equation, real densityFloor,
            real temperatureFloor, real sensibleEnergyFloor,
            real pressureFloor, RANSStateView rans = {},
            Physics::Chemistry::ChemicalWorkerPool::Lease worker = {});

        AdmissibilityView View() const
        {
            return {_thermal.View(), _rans, _ransFirst};
        }
        const EquationPropertyKey &PropertyKey() const { return _property; }
        const RANSStateView &RANS() const { return _rans; }
    };
}
