#include "Cauchy/Input/StateInput.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/RANSState.hpp"

namespace DNDS::Cauchy
{
    struct PhysicalStateConverter::Data
    {
        EquationSystem equation;
        Physics::ReferenceScales scales;
        PointEvaluator evaluator;
        std::vector<real> scalarScales;
        int rans, species;

        Data(const EquationSystem &equationIn, const Physics::ReferenceScales &scalesIn,
             Physics::Chemistry::ChemicalWorkerPool::Lease worker)
            : equation(equationIn), scales(scalesIn),
              evaluator(equation.MakePointEvaluator(std::move(worker), {})),
              rans(equation.HasRANS() ? equation.RANS().ComponentNumber() : 0),
              species(equation.HasMixture() ? equation.Mixture().SpeciesNumber() : 0)
        {
        }
    };

    Physics::ReferenceScales ResolveReferenceScales(const EquationSettings &settings)
    {
        const auto &old = settings.mixture;
        Physics::ReferenceScales result{old.rho0, old.U0, old.L0, old.T0};
        if (!settings.referenceScales.is_null())
        {
            const auto &input = settings.referenceScales;
            DNDS_check_throw_info(input.is_object() && input.size() == 4 && input.contains("rho0") && input.contains("U0") && input.contains("L0") && input.contains("T0"),
                                  "equation.referenceScales requires exactly rho0,U0,L0,T0");
            result = {input.at("rho0").get<real>(), input.at("U0").get<real>(), input.at("L0").get<real>(), input.at("T0").get<real>()};
            DNDS_check_throw_info((old.rho0 == 1 && old.U0 == 1 && old.L0 == 1 && old.T0 == 1) ||
                                      (old.rho0 == result.rho0 && old.U0 == result.U0 && old.L0 == result.L0 && old.T0 == result.T0),
                                  "conflicting equation.referenceScales and mixture scale fields");
        }
        result.Validate("equation.referenceScales");
        return result;
    }
    PhysicalStateConverter::PhysicalStateConverter(const EquationSystem &equation, const Physics::ReferenceScales &scales,
                                                   Physics::Chemistry::ChemicalWorkerPool::Lease worker)
        : _data(std::make_unique<Data>(equation, scales, std::move(worker)))
    {
        scales.Validate();
        _data->scalarScales.assign(_data->rans + (_data->species ? _data->species - 1 : 0), 1);
        if (_data->rans == 2)
        {
            _data->scalarScales[0] = scales.SpecificEnergy();
            _data->scalarScales[1] = (equation.RANS().Kind() == Physics::RANS::ModelKind::RealizableKE ? scales.SpecificEnergy() : 1) / scales.Time();
        }
        if (_data->species)
        {
            const auto &material = equation.Mixture().Scales();
            DNDS_check_throw_info(scales.rho0 == material.rho0 && scales.U0 == material.U0 && scales.L0 == material.L0 && scales.T0 == material.T0,
                                  "state input scales differ from equation material");
        }
    }
    PhysicalStateConverter::~PhysicalStateConverter() = default;
    std::vector<real> PhysicalStateConverter::TotalCondition(Physics::StateInput parsed)
    {
        const auto &scales = _data->scales;
        const auto &scalarScales = _data->scalarScales;
        const int rans = _data->rans, species = _data->species;
        DNDS_check_throw_info(parsed.form == Physics::StateInput::Form::TotalPT,
                              "totalConditionInlet requires totalPT or totalPT_phy");
        parsed.ToCodeUnits(scales, scalarScales);
        auto &v = parsed.values;
        DNDS_check_throw_info(std::all_of(v.begin(), v.end(), [](real value)
                                          { return std::isfinite(value); }),
                              "scaled total inlet state is not finite");
        const real norm = std::hypot(v[2], v[3], v[4]);
        DNDS_check_throw_info(v[0] > 0 && v[1] > 0 && std::isfinite(norm) && norm > 1e-14,
                              "total inlet requires positive total p/T and a finite nonzero direction");
        for (int d = 2; d < 5; ++d)
            v[d] /= norm;
        for (int j = 0; j < rans; ++j)
            DNDS_check_throw_info(v[5 + j] >= 0, "total inlet requires nonnegative primitive RANS values");
        if (species)
        {
            std::vector<real> fractions(species);
            Physics::Chemistry::RepairMassFractions(1, {v.data() + 5 + rans, species - 1},
                                                    {fractions.data(), species});
            for (int k = 0; k < species - 1; ++k)
                v[5 + rans + k] = fractions[k];
        }
        return std::move(v);
    }
    std::vector<real> PhysicalStateConverter::Conservative(Physics::StateInput parsed)
    {
        using Form = Physics::StateInput::Form;
        const auto &equation = _data->equation;
        const auto &scales = _data->scales;
        const auto &scalarScales = _data->scalarScales;
        auto &evaluator = _data->evaluator;
        DNDS_check_throw_info(parsed.form != Form::TotalPT, "totalPT is only valid for a totalConditionInlet boundary");
        const int nr = _data->rans, ns = _data->species, ni = ns ? ns - 1 : 0;
        parsed.ToCodeUnits(scales, scalarScales);
        auto &u = parsed.values;
        DNDS_check_throw_info(std::all_of(u.begin(), u.end(), [](real v)
                                          { return std::isfinite(v); }),
                              "physical state input is not finite");
        const bool conservative = parsed.form == Form::Conservative || parsed.form == Form::Sensible;
        DNDS_check_throw_info(u[0] > 0, "physical state requires positive rho (or T for primTP)");
        std::vector<real> fractions(ns), rhoY(ni);
        if (ns)
        {
            // Here 1 denotes mass-fraction normalization, not a missing density.
            Physics::Chemistry::RepairMassFractions(conservative ? u[0] : 1,
                                                    {u.data() + 5 + nr, ni}, {fractions.data(), ns});
        }
        if (!conservative)
        {
            real gasConstant = equation.HasMixture() ? 0 : equation.Gas().gasConstant;
            if (ns)
                for (int k = 0; k < ns; ++k)
                    gasConstant += fractions[k] * equation.Mixture().SpeciesGasConstants()[k];
            DNDS_check_throw_info(u[4] > 0, "primitive input requires positive pressure or temperature");
            if (parsed.form == Form::TP)
                u[0] = u[4] / (gasConstant * u[0]);
            for (int k = 0; k < ni; ++k)
                u[5 + nr + k] = u[0] * fractions[k];
        }
        for (int k = 0; k < ni; ++k)
            rhoY[k] = u[5 + nr + k];
        FlowState flow;
        std::copy_n(u.begin(), 5, flow.begin());
        if (!conservative)
        {
            flow = evaluator.FromPrimitive(flow, {rhoY.data(), ni}, parsed.form == Form::RhoT ? u[4] : 0);
            for (int j = 0; j < nr; ++j)
                u[5 + j] *= flow[0];
        }
        else if (parsed.form == Form::Sensible && ns)
        {
            // Euler's I/O reference energy uses repaired mass fractions; the
            // transported conservative species themselves are not rewritten.
            for (int k = 0; k < ns; ++k)
                flow[4] += flow[0] * fractions[k] * equation.Mixture().BaseSpecificInternalEnergies()[k];
        }
        std::copy(flow.begin(), flow.end(), u.begin());
        const auto properties = evaluator.Evaluate(flow, {rhoY.data(), ni});
        DNDS_check_throw_info(std::isfinite(properties.thermo.temperature) && properties.thermo.temperature > 0 &&
                                  std::isfinite(properties.thermo.pressure) && properties.thermo.pressure > 0 &&
                                  std::all_of(u.begin(), u.end(), [](real v)
                                              { return std::isfinite(v); }),
                              "resolved physical state is not finite and thermodynamically admissible");
        return std::move(u);
    }
    Physics::StateInput PhysicalStateConverter::Represent(const std::vector<real> &conservative, Physics::StateInput::Form form, bool physical)
    {
        using Form = Physics::StateInput::Form;
        const auto &equation = _data->equation;
        const auto &scales = _data->scales;
        const auto &scalarScales = _data->scalarScales;
        auto &evaluator = _data->evaluator;
        const int rans = _data->rans, species = _data->species;
        DNDS_check_throw_info(form != Form::TotalPT, "totalPT is a boundary input, not a conservative-state representation");
        DNDS_check_throw_info(conservative.size() == 5 + scalarScales.size() && conservative[0] > 0 &&
                                  std::all_of(conservative.begin(), conservative.end(), [](real v)
                                              { return std::isfinite(v); }),
                              "state representation requires a finite complete conservative row with positive density");
        Physics::StateInput result{form, false, conservative};
        const int ni = species ? species - 1 : 0;
        FlowState flow;
        std::copy_n(conservative.begin(), 5, flow.begin());
        const auto point = evaluator.Evaluate(flow, {conservative.data() + 5 + rans, ni});
        auto &values = result.values;
        if (form == Form::Sensible && species)
            for (int k = 0; k < species; ++k)
                values[4] -= flow[0] * point.fractions[k] * equation.Mixture().BaseSpecificInternalEnergies()[k];
        if (form != Form::Conservative && form != Form::Sensible)
        {
            for (index j = 1; j < index(values.size()); ++j)
                values[j] /= flow[0];
            values[4] = form == Form::RhoT ? point.thermo.temperature : point.thermo.pressure;
            if (form == Form::TP)
                values[0] = point.thermo.temperature;
        }
        if (physical)
            result.ToPhysicalUnits(scales, scalarScales);
        return result;
    }
    ResolvedPhysicalState ResolvePhysicalState(const Json &input, const EquationSystem &equation,
                                               const Physics::ReferenceScales &scales, Physics::Chemistry::ChemicalWorkerPool::Lease worker)
    {
        PhysicalStateConverter converter(equation, scales, std::move(worker));
        auto u = converter.Conservative(Physics::StateInput::Parse(input));
        const auto primitive = converter.Represent(u, Physics::StateInput::Form::RhoP, false);
        const auto temperature = converter.Represent(u, Physics::StateInput::Form::RhoT, false).values[4];
        FlowState flow;
        std::copy_n(primitive.values.begin(), 5, flow.begin());
        ResolvedPhysicalState result{u, flow, temperature, {}, {}};
        const int nr = equation.HasRANS() ? equation.RANS().ComponentNumber() : 0;
        const int ns = equation.HasMixture() ? equation.Mixture().SpeciesNumber() : 0;
        if (ns)
        {
            std::vector<real> fractions(ns);
            Physics::Chemistry::RepairMassFractions(u[0], {u.data() + 5 + nr, ns - 1}, {fractions.data(), ns});
            for (int k = 0; k < ns - 1; ++k)
                result.massFractions[equation.Mixture().SpeciesNames()[k]] = fractions[k];
        }
        for (int j = 0; j < nr; ++j)
            result.ransConservative[equation.RANS().ComponentNames()[j]] = u[5 + j];
        ResolveRANSValues(equation, result.ransConservative);
        return result;
    }
    ResolvedPhysicalInputs ResolvePhysicalStateInputs(const DomainSettings &settings, const EquationSystem &equation,
                                                      const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers)
    {
        ResolvedPhysicalInputs prepared{settings, {}, {}};
        auto &result = prepared.settings;
        const auto scales = ResolveReferenceScales(settings.equation);
        const auto resolve = [&](const Json &value)
        {
            return ResolvePhysicalState(value, equation, scales, workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
        };
        if (!settings.initialState.is_null())
        {
            DNDS_check_throw_info(settings.initialPrimitive == DomainSettings{}.initialPrimitive && settings.initialTemperature == 0 && settings.initialMassFractions.empty() && settings.initialRANS.empty(),
                                  "initialState conflicts with legacy initial fields");
            DNDS_check_throw_info(settings.initialProfile.kind == "constant", "tagged initialState currently requires a constant profile");
            auto state = resolve(settings.initialState);
            prepared.initialConservative = std::move(state.conservative);
            result.initialPrimitive = state.primitive;
            result.initialMassFractions = std::move(state.massFractions);
            result.initialRANS = std::move(state.ransConservative);
            result.initialState = nullptr;
        }
        if (!settings.referenceState.is_null())
        {
            DNDS_check_throw_info(settings.numerics.ransState.referenceConservative.empty(), "referenceState conflicts with partial referenceConservative");
            auto state = resolve(settings.referenceState);
            prepared.referenceConservative = state.conservative;
            if (equation.HasRANS())
            {
                result.numerics.ransState.referenceConservative = std::move(state.ransConservative);
                result.numerics.ransState.referenceConservative["rho"] = state.conservative[0];
            }
            result.referenceState = nullptr;
        }
        for (auto &[zone, boundary] : result.boundaries)
            if (!boundary.state.is_null())
            {
                DNDS_check_throw_info(boundary.kind == "farfield" || boundary.kind == "analytic" || boundary.kind == "totalConditionInlet", "tagged boundary state requires a prescribed boundary");
                DNDS_check_throw_info(boundary.primitive == BoundarySettings{}.primitive && boundary.massFractions.empty() && boundary.ransConservative.empty(),
                                      "boundary state conflicts with legacy primitive/scalar fields");
                if (boundary.kind == "totalConditionInlet")
                {
                    PhysicalStateConverter converter(equation, scales, workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
                    boundary.resolvedTotalState = converter.TotalCondition(Physics::StateInput::Parse(boundary.state));
                    DNDS_check_throw_info(boundary.resolvedTotalState[1] >= settings.numerics.temperatureFloor,
                                          "total inlet temperature lies below the configured temperature floor");
                    // Retain the normalized descriptor in prepared settings:
                    // workspace/cache identities serialize those settings.
                    boundary.state = {{"type", "totalPT"}, {"state", boundary.resolvedTotalState}};
                    continue;
                }
                auto state = resolve(boundary.state);
                DNDS_check_throw_info(state.temperature >= settings.numerics.temperatureFloor,
                                      "prescribed boundary state lies below configured temperature floor");
                boundary.primitive = state.primitive;
                boundary.massFractions = std::move(state.massFractions);
                boundary.ransConservative = std::move(state.ransConservative);
                boundary.state = nullptr;
            }
        return prepared;
    }
}
