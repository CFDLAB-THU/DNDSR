#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/PerfectGasCharacteristics.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Physics/RANS/Model.hpp"

namespace DNDS::Cauchy
{
    EquationSystem::EquationSystem(PerfectGas gas, bool viscous) : _gas(gas), _viscous(viscous)
    {
        _gas.Validate();
        _schema = {"compressible", "sensible-total-energy", "conservative-volume", {{"flow", {{"rho", "mass/volume", "mass"}, {"rhou", "mass/(area*time)", "momentum"}, {"rhov", "mass/(area*time)", "momentum"}, {"rhow", "mass/(area*time)", "momentum"}, {"rhoE", "energy/volume", "total-energy"}}, {1}}}};
    }
    EquationSystem::EquationSystem(ACM::Parameters parameters, bool viscous) : _acm(parameters), _viscous(viscous)
    {
        _schema = {"incompressibleACM", "none", "constant-density-velocity-pressure", {{"flow", {{"u", "length/time", "velocity"}, {"v", "length/time", "velocity"}, {"w", "length/time", "velocity"}, {"p", "pressure", "continuity"}}, {0}}}};
    }
    EquationSystem EquationSystem::Resolve(const EquationSettings &settings)
    {
        if (settings.system == "incompressibleACM")
        {
            const bool viscous = settings.transport == "Newtonian";
            DNDS_check_throw_info(settings.eos == "none" && settings.source == "none" &&
                                      (viscous || settings.transport == "inviscid"),
                                  "constant-density ACM requires eos=none and inviscid or Newtonian transport");
            DNDS_check_throw_info(std::isfinite(settings.acmBeta2) && settings.acmBeta2 > 0 && std::isfinite(settings.acmAlpha),
                                  "invalid artificial-compressibility parameters");
            DNDS_check_throw_info(std::isfinite(settings.viscosity) && settings.viscosity >= 0 &&
                                      (viscous || settings.viscosity == 0) &&
                                      (settings.viscosityModel == "constant" || settings.viscosityModel == "constantKinematic"),
                                  "invalid incompressible viscosity");
            ACM::Parameters acm{};
            acm.beta2 = settings.acmBeta2;
            DNDS_check_throw_info(std::isfinite(settings.acmEntropyFixRatio) && settings.acmEntropyFixRatio >= 0, "invalid ACM Roe entropy-fix ratio");
            acm.entropyFix = settings.acmEntropyFixRatio;
            acm.alpha = settings.acmAlpha;
            acm.kinematicViscosity = settings.viscosity;
            return EquationSystem(acm, viscous).WithRANS(settings.rans);
        }
        DNDS_check_throw_info(settings.system == "compressible" && settings.eos == "perfectGas" && settings.source == "none" &&
                                  (settings.transport == "inviscid" || settings.transport == "NewtonianFourier"),
                              "unavailable Cauchy equation profile");
        const bool viscous = settings.transport == "NewtonianFourier";
        DNDS_check_throw_info(viscous || (settings.viscosity == 0 && settings.viscosityModel == "constant"), "unused inviscid viscosity");
        using Model = Physics::Transport::ViscosityModel;
        const auto model = settings.viscosityModel == "constant" ? Model::Constant : settings.viscosityModel == "sutherland" ? Model::Sutherland
                                                                                                                             : Model::ConstantKinematic;
        DNDS_check_throw_info(settings.viscosityModel == "constant" || settings.viscosityModel == "sutherland" || settings.viscosityModel == "constantKinematic", "unknown perfect-gas viscosity law");
        DNDS_check_throw_info(model == Model::Sutherland || (settings.viscosityReferenceTemperature == 1 && settings.sutherlandConstant == 0), "unused Sutherland parameters");
        return EquationSystem({settings.gamma, settings.gasConstant, settings.viscosity, settings.prandtl, model,
                               settings.viscosityReferenceTemperature, settings.sutherlandConstant},
                              viscous)
            .WithRANS(settings.rans);
    }
    EquationSystem EquationSystem::FromIdealGasMixture(Physics::Thermodynamics::IdealGasMixture material, bool viscous)
    {
        EquationSystem equation(PerfectGas{}, viscous);
        equation._mixture = std::move(material);
        equation._schema.energyConvention = "thermochemical-total-energy";
        const auto &species = equation.Mixture().SpeciesNames();
        if (species.size() > 1)
        {
            EquationBlockSchema block{"species", {}, {}};
            for (index k = 0; k < index(species.size()) - 1; ++k)
                block.components.push_back({"rhoY:" + species[k], "mass/volume", "species"});
            equation._schema.blocks.push_back(std::move(block));
        }
        return equation;
    }
    const Physics::Thermodynamics::IdealGasMixture &EquationSystem::Mixture() const
    {
        DNDS_check_throw_info(_mixture, "equation has no ideal-mixture material");
        return *_mixture;
    }
    EquationSystem EquationSystem::WithRANS(const Physics::RANS::ModelSettings &settings) const
    {
        DNDS_check_throw_info(!HasRANS(), "RANS is already attached to this equation");
        DNDS_check_throw_info(!IsACM() || !settings.useQCR, "ACM supports the linear eddy-viscosity stress closure only");
        auto result = *this;
        auto model = std::make_shared<const Physics::RANS::Model>(settings, Physics::RANS::SSTProduction::StrainSquared);
        if (!model->ComponentNumber())
            return result;
        DNDS_check_throw_info(_viscous, "RANS requires molecular viscous transport");
        DNDS_check_throw_info(_mixture || (_acm ? _acm->kinematicViscosity : _gas.viscosity) > 0,
                              "RANS requires positive molecular viscosity");
        EquationBlockSchema block{"rans", {}, {}};
        const auto names = model->ComponentNames();
        if (model->ComponentNumber() == 1)
            block.components.push_back({names[0], "mass/volume", "scaled-turbulent-viscosity"});
        else
        {
            block.components.push_back({names[0], "energy/volume", "turbulent-kinetic-energy"});
            const bool epsilon = model->Kind() == Physics::RANS::ModelKind::RealizableKE;
            block.components.push_back({names[1], epsilon ? "energy/(volume*time)" : "mass/(volume*time)",
                                        epsilon ? "turbulent-dissipation" : "specific-turbulent-dissipation"});
        }
        result._schema.blocks.insert(result._schema.blocks.begin() + 1, std::move(block));
        result._rans = std::move(model);
        return result;
    }
    const Physics::RANS::Model &EquationSystem::RANS() const
    {
        DNDS_check_throw_info(_rans, "equation has no RANS closure");
        return *_rans;
    }
    std::vector<real> EquationSystem::IndependentMassFractions(const std::map<std::string, real> &values) const
    {
        DNDS_check_throw_info(_mixture || values.empty(), "perfect gas has no species composition");
        if (!_mixture)
            return {};
        const auto &names = _mixture->SpeciesNames();
        std::vector<real> fractions(names.size(), 0);
        real sum = 0;
        for (const auto &[name, value] : values)
        {
            const auto found = std::find(names.begin(), names.end(), name);
            DNDS_check_throw_info(found != names.end() && std::isfinite(value) && value >= 0,
                                  "unknown species or invalid mass fraction: " + name);
            fractions[found - names.begin()] = value;
            sum += value;
        }
        DNDS_check_throw_info(sum <= 1 + 1e-14 && (!values.count(names.back()) || std::abs(sum - 1) <= 1e-14),
                              "mass fractions do not leave a valid dependent species remainder");
        // Permit only roundoff at a closed composition; never renormalize a
        // genuinely overfilled or incomplete explicit composition silently.
        if (sum > 1)
            for (auto &fraction : fractions)
                fraction /= sum;
        fractions.pop_back();
        return fractions;
    }
    EquationPropertyKey EquationSystem::PropertyKey() const
    {
        if (_acm)
            return {{}, {-1., _acm->alpha, _acm->beta2, _acm->entropyFix, _acm->kinematicViscosity, _viscous ? 1. : 0.}};
        if (_mixture)
        {
            const auto &scales = _mixture->Scales();
            return {_mixture->ModelIdentity(), {scales.rho0, scales.U0, scales.L0, scales.T0, _viscous ? 1. : 0.}};
        }
        return {{}, {_gas.gamma, _gas.gasConstant, _gas.viscosity, _gas.prandtl, _viscous ? 1. : 0., real(_gas.viscosityModel), _gas.viscosityReferenceTemperature, _gas.sutherlandConstant}};
    }
    EquationKernelProfile EquationSystem::ResolveKernelProfile(Runtime::Device device) const
    {
        DNDS_check_throw_info(device == Runtime::Device::Host(), "Cauchy equation kernels are unavailable on the requested device; no implicit host fallback");
        if (_acm)
            return EquationKernelProfile::ConstantDensityACMCPU;
        return _mixture ? EquationKernelProfile::CompressibleIdealMixtureCPU : EquationKernelProfile::CompressiblePerfectGasCPU;
    }
    FlowCharacteristics EquationSystem::PrepareCharacteristics(const FlowState &left, const FlowState &right, const Vector3 &faceNormal) const
    {
        Gas(); // This signature has no species inputs; never reinterpret a mixture as perfect gas.
        const auto result = TryPerfectGasCharacteristics(_gas, left, right, faceNormal);
        DNDS_check_throw_info(result, "invalid mean-state characteristic input");
        return result.value;
    }
    FlowCharacteristics EquationSystem::PrepareCharacteristics(const FlowState &left, const FlowState &right,
                                                               const CharacteristicThermodynamics &leftThermo,
                                                               const CharacteristicThermodynamics &rightThermo,
                                                               const Vector3 &normal) const
    {
        const auto result = TryCachedMeanCharacteristics(left, right, leftThermo, rightThermo, normal);
        DNDS_check_throw_info(result, "invalid cached characteristic thermodynamics or mean state");
        return result.value;
    }
    PreparedRiemannData EquationSystem::PrepareRiemann(const Riemann::RiemannFlux &flux,
                                                       const Riemann::AcousticState &left, const Riemann::AcousticState &right,
                                                       const FlowState &leftMean, const FlowState &rightMean,
                                                       const Vector3 &normal, const Vector3 &gridVelocity) const
    {
        return PrepareRiemann(flux, left, right, PrepareFluxState(leftMean), PrepareFluxState(rightMean), normal, gridVelocity);
    }
    PreparedRiemannData EquationSystem::PrepareRiemann(const Riemann::RiemannFlux &flux,
                                                       const Riemann::AcousticState &left, const Riemann::AcousticState &right,
                                                       const Riemann::AcousticState &leftMeanData, const Riemann::AcousticState &rightMeanData,
                                                       const Vector3 &normal, const Vector3 &gridVelocity) const
    {
        if (!flux.RequiredInputs().meanStates)
        {
            PreparedRiemannData data{};
            data.required = flux.RequiredInputs();
            return data;
        }
        // Mixture Rusanov requires no Roe/shock/AUSM producer; all other
        // profiles keep the existing explicit perfect-gas model guard.
        return PreparePerfectGasRiemann(Gas(), flux.Kind(), left, right,
                                        leftMeanData, rightMeanData, normal, gridVelocity);
    }
}
