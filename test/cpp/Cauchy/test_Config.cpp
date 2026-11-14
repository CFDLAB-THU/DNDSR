#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"

namespace C = DNDS::Cauchy;
TEST_CASE("Diagonal inverse refinement is opt-in and rejects unrelated inverse providers")
{
    C::DomainSettings settings;
    CHECK(settings.implicit.linear.diagonalInverseRefinement == 0);
    settings.implicit.diagonal = "block";
    settings.implicit.linear.method = "jacobi";
    settings.implicit.linear.diagonalInverseRefinement = 2;
    const auto restored = C::Json(settings).get<C::DomainSettings>();
    CHECK(restored.implicit.linear.diagonalInverseRefinement == 2);
    CHECK_NOTHROW(settings.implicit.Validate());
    settings.implicit.linear.method = "ilu";
    CHECK_THROWS_WITH(settings.implicit.Validate(), doctest::Contains("block-inverse refinement"));
}
TEST_CASE("Reconstruction kernel selection is explicit and retains the FP64 default")
{
    using Kernel = DNDS::CFV::ReconstructionDeviceKernel;
    C::ExecutionSettings execution;
    CHECK(execution.ReconstructionKernel() == Kernel::FP64);
    for (const auto &[name, kernel] : std::vector<std::pair<std::string, Kernel>>{
             {"fp64Tensor", Kernel::FP64TensorCore}, {"fp64TensorShared", Kernel::FP64TensorCoreShared}, {"fp64TensorAsync", Kernel::FP64TensorCoreAsync}, {"fp32", Kernel::FP32}, {"fp32Accum", Kernel::FP32FullAccumulation}, {"fp32FloatFloatAccum", Kernel::FP32FloatFloatAccumulation}, {"tf32", Kernel::TF32TensorCore}})
    {
        execution.reconstructionKernel = name;
        CHECK(execution.ReconstructionKernel() == kernel);
        const C::ExecutionSettings restored = C::Json(execution).get<C::ExecutionSettings>();
        CHECK(restored.ReconstructionKernel() == kernel);
    }
    C::DomainConfig domain;
    domain.settings.execution.reconstructionKernel = "fp32";
    CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("requires CUDA"));
    domain.settings.execution.reconstructionKernel = "automatic-low-precision";
    CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("unknown reconstruction kernel"));
}
TEST_CASE("CUDA capability resolution requires a complete combined perfect gas profile")
{
    C::DomainConfig domain;
    domain.placement.name = "device-fluid";
    auto &settings = domain.settings;
    settings.execution.device = "cuda";
    settings.execution.layout = "aos";
    settings.execution.fieldGrouping = "coupled";
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.faceFallback = false;
    settings.numerics.riemann.useStencil = false;
    settings.implicit.linear.method = "jacobi";
    settings.implicit.multigrid.levels = 2;
#ifdef DNDS_USE_CUDA
    for (const char *model : {"none", "SA", "KOmegaWilcox", "KOmegaSST"})
    {
        settings.equation.rans.model = model;
        settings.equation.transport = "NewtonianFourier";
        CHECK_NOTHROW(C::ResolvedExecutionPlan::Resolve(domain));
    }
    settings.execution.reconstructionKernel = "fp32";
    CHECK_NOTHROW(C::ResolvedExecutionPlan::Resolve(domain));
    settings.numerics.limiter = "LVR";
    CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("LVR currently requires"));
    settings.numerics.limiter = "none";
    settings.execution.reconstructionKernel = "fp64";
    settings.implicit.linear.method = "amgx";
    settings.implicit.linear.stopping = "fixedCount";
    CHECK_NOTHROW(settings.implicit.Validate());
    // Provider compile definitions are private to DNDS; query the linked
    // capability rather than assuming this test TU inherits that macro.
    if (DNDS::Algebra::AmgXPreconditioner::Available())
        CHECK_NOTHROW(C::ResolvedExecutionPlan::Resolve(domain));
    else
        CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("AmgX requires"));
    settings.execution.device = "cpu";
    CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("AmgX requires assembled CUDA"));
    settings.execution.device = "cuda";
    settings.implicit.linear.method = "jacobi";
    settings.implicit.linear.stopping = "residual";
    settings.implicit.multigrid.coarseLevelSettings["1"].solver = {{"linear", {{"method", "ilu"}}}};
    CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("pMG level 1 CUDA linear solve"));
    settings.implicit.multigrid.coarseLevelSettings.clear();
    settings.execution.fieldGrouping = "equationBlocks";
    CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("combined AoS"));
#else
    CHECK_THROWS_WITH(C::ResolvedExecutionPlan::Resolve(domain), doctest::Contains("CUDA-enabled build"));
#endif
}
TEST_CASE("CPU workers require an MPI capability that permits multithreaded execution")
{
    C::ExecutionSettings execution;
    CHECK_NOTHROW(execution.ValidateMPIThreadSupport(MPI_THREAD_SINGLE));
    execution.hostThreads = 2;
    CHECK_THROWS_WITH(execution.ValidateMPIThreadSupport(MPI_THREAD_SINGLE),
                      doctest::Contains("at least MPI_THREAD_FUNNELED"));
    CHECK_NOTHROW(execution.ValidateMPIThreadSupport(MPI_THREAD_FUNNELED));
    CHECK_NOTHROW(execution.ValidateMPIThreadSupport(MPI_THREAD_MULTIPLE));
}

TEST_CASE("Cauchy exposes one canonical layered input schema")
{
    const auto schema = C::SimulationConfig::InputSchema();
    CHECK(schema["$schema"] == "http://json-schema.org/draft-07/schema#");
    CHECK(schema["additionalProperties"] == false);
    for (const auto *section : {"schemaVersion", "profiles", "defaults", "domains", "time", "output", "interfaces"})
        CHECK(schema["properties"].contains(section));
    const auto &domain = schema["properties"]["domains"]["items"];
    CHECK(domain["required"] == C::Json::array({"name"}));
    CHECK(domain["additionalProperties"] == false);
    CHECK(domain["properties"]["overrides"]["additionalProperties"] == false);

    C::Json input{{"profiles", {{"baseline", C::Json(C::DomainSettings{})}}},
                  {"defaults", C::Json(C::DomainSettings{})},
                  {"domains", C::Json::array({{{"name", "fluid"}, {"profiles", {"baseline"}}, {"overrides", C::Json::object()}}})},
                  {"time", C::Json(C::TimeSettings{})},
                  {"output", C::Json(C::OutputSettings{})},
                  {"interfaces", C::Json::array()}};
    CHECK_NOTHROW(C::SimulationConfig::Resolve(input, 1));
    input["unknownRoot"] = true;
    CHECK_THROWS_WITH(C::SimulationConfig::Resolve(input, 1),
                      doctest::Contains("/unknownRoot"));
}

TEST_CASE("MPI fail-fast launch validation is an explicit execution choice")
{
    CHECK(C::ExecutionSettings{}.transferPreflight == "collective");
    CHECK(C::ExecutionSettings{}.layout == "aos");
    CHECK(C::ExecutionSettings{}.fieldGrouping == "coupled");
    CHECK(C::ImplicitSettings{}.sourceJacobianMode == "center");
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"reconstruction", "variational"}}}, {"implicit", {{"sourceJacobianMode", "rhsQuadrature"}}}}}}, 1));
    const auto config = C::SimulationConfig::Resolve({{"defaults", {{"execution", {{"transferPreflight", "abortOnFailure"}}}}}}, 2);
    CHECK(config.Domains()[0].settings.execution.transferPreflight == "abortOnFailure");
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"execution", {{"transferPreflight", "unchecked"}}}}}}, 2));
}
TEST_CASE("DITR inherits common controls and only supplied compatibility fields override them")
{
    C::Json input{{"time", {{"method", "ditr"}}},
                  {"defaults", {{"implicit", {{"iterations", 17}, {"minIterations", 3}, {"relativeTolerance", 2e-5}, {"absoluteTolerance", 7e-9}}}}}};
    const auto inherited = C::SimulationConfig::Resolve(input, 1);
    const auto common = inherited.Domains()[0].settings.implicit;
    auto controls = inherited.Time().ditr.ResolveControls(common);
    CHECK(controls.iterations == 17);
    CHECK(controls.minIterations == 3);
    CHECK(controls.relativeTolerance == 2e-5);
    CHECK(controls.absoluteTolerance == 7e-9);
    CHECK(inherited.Root()["time"]["ditr"]["iterations"].is_null());
    input["time"]["ditr"] = {{"iterations", 9}, {"minIterations", 0}, {"relativeTolerance", 0}, {"absoluteTolerance", nullptr}};
    const auto explicitValues = C::SimulationConfig::Resolve(input, 1);
    controls = explicitValues.Time().ditr.ResolveControls(common);
    CHECK(controls.iterations == 9);
    CHECK(controls.minIterations == 0);
    CHECK(controls.relativeTolerance == 0);
    CHECK(controls.absoluteTolerance == 7e-9);
    const auto roundtrip = C::SimulationConfig::Resolve({{"defaults", C::Json(explicitValues.Domains()[0].settings)}, {"time", C::Json(explicitValues.Time())}}, 1);
    CHECK(roundtrip.Time().ditr.iterations == 9);
    CHECK_FALSE(roundtrip.Time().ditr.absoluteTolerance.has_value());
    input["time"]["ditr"] = {{"iterations", 2}};
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 1)); // Inherited minimum is three.
    input["time"]["ditr"] = {{"iterations", 2.5}};
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 1)); // No narrowing to an integer override.
}
TEST_CASE("Chemical source splitting is explicit and keeps Euler reactor defaults")
{
    CHECK(C::TimeSettings{}.sourceSplitting == "coupled");
    CHECK(C::TimeSettings{}.reactor.rtol == 1e-10);
    CHECK(C::TimeSettings{}.reactor.atol == 1e-18);
#ifdef DNDS_USE_CANTERA
    C::Json input{{"defaults", {{"equation", {{"eos", "idealGasMixture"}, {"source", "chemical"}, {"mixture", {{"mechanismFile", "h2o2.yaml"}}}}}, {"numerics", {{"flux", "rusanov"}, {"faceFallback", false}}}}},
                  {"time", {{"method", "esdirk2"}, {"sourceSplitting", "strang"}, {"reactor", {{"maxOrder", 2}}}}}};
    const auto config = C::SimulationConfig::Resolve(input, 1);
    CHECK(config.Time().sourceSplitting == "strang");
    CHECK(config.Time().reactor.maxOrder == 2);
    input["time"]["steady"] = true;
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 1));
    input["time"]["steady"] = false;
    input["time"]["reactor"]["maxSteps"] = 0;
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 1));
#endif
}
TEST_CASE("Configured RANS exposes Euler DES and QCR controls without a unit-length substitution")
{
    C::Json input{{"defaults", {{"equation", {{"transport", "NewtonianFourier"}, {"viscosity", .001}, {"rans", {{"model", "SA"}, {"desScale", .01}, {"desMode", 3}, {"useQCR", true}}}}}, {"numerics", {{"flux", "rusanov"}, {"faceFallback", false}, {"viscousGradient", "polynomial"}}}}}};
    const auto config = C::SimulationConfig::Resolve(input, 1);
    const auto &rans = config.Domains()[0].settings.equation.rans;
    CHECK(rans.desScale == .01);
    CHECK(rans.desMode == 3);
    CHECK(rans.useQCR);
    input["defaults"]["equation"]["rans"]["model"] = "none";
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 1));
}

TEST_CASE("Halo freshness audits are opt-in and the collective check switch overrides them")
{
    const auto defaults = C::SimulationConfig::Resolve(C::Json::object(), 1);
    CHECK_FALSE(defaults.Domains()[0].settings.execution.haloFreshnessChecks);
    CHECK_FALSE(defaults.Domains()[0].settings.CheckHaloFreshness());
    C::Json input{{"defaults", {{"execution", {{"haloFreshnessChecks", true}}}}}};
    CHECK(C::SimulationConfig::Resolve(input, 1).Domains()[0].settings.CheckHaloFreshness());
    input["defaults"]["implicit"]["collectiveChecks"] = false;
    const auto disabled = C::SimulationConfig::Resolve(input, 1);
    CHECK(disabled.Domains()[0].settings.execution.haloFreshnessChecks);
    CHECK_FALSE(disabled.Domains()[0].settings.CheckHaloFreshness());
}
TEST_CASE("Interface capability checks admit coordinated DITR and honor the steady BDF1 override")
{
    C::Json input{{"domains", C::Json::array({{{"name", "a"}}, {{"name", "b"}}})},
                  {"defaults", {{"boundaries", {{"101", {{"kind", "interface"}}}}}}},
                  {"interfaces", C::Json::array({{{"donor", "a"}, {"receiver", "b"}, {"donorZone", 101}, {"receiverZone", 101}},
                                                 {{"donor", "b"}, {"receiver", "a"}, {"donorZone", 101}, {"receiverZone", 101}}})},
                  {"time", {{"method", "ditr"}}}};
    CHECK(C::SimulationConfig::Resolve(input, 1).Time().method == "ditr");
    input["time"]["steady"] = true;
    CHECK(C::SimulationConfig::Resolve(input, 1).Time().method == "bdf1");
}
TEST_CASE("Polynomial viscous reconstruction is explicit and retains the previous primitive default")
{
    const auto defaults = C::SimulationConfig::Resolve(C::Json::object(), 2);
    CHECK(defaults.Domains()[0].settings.numerics.viscousGradient == "primitive");
    const auto polynomial = C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"viscousGradient", "polynomial"}, {"viscousLimited", false}, {"noGRPOnWall", true}}}}}}, 2);
    CHECK(polynomial.Domains()[0].settings.numerics.viscousGradient == "polynomial");
    CHECK_FALSE(polynomial.Domains()[0].settings.numerics.viscousLimited);
    CHECK(polynomial.Domains()[0].settings.numerics.noGRPOnWall);
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"viscousGradient", "missing"}}}}}}, 2));
}
TEST_CASE("Physical timestep PP has explicit validated controls and is disabled by the steady limit")
{
    CHECK_FALSE(C::SimulationConfig::Resolve(C::Json::object(), 2).Time().positivity.enabled);
    C::Json input{{"time", {{"positivity", {{"enabled", true}, {"relax", .7}, {"scale", 1.5}, {"increaseLimit", 1.25}}}}}};
    const auto config = C::SimulationConfig::Resolve(input, 2);
    CHECK(config.Time().positivity.enabled);
    CHECK(config.Time().positivity.relax == .7);
    CHECK(config.Time().positivity.scale == 1.5);
    CHECK(config.Time().positivity.increaseLimit == 1.25);
    input["time"]["steady"] = true;
    const auto steady = C::SimulationConfig::Resolve(input, 2);
    CHECK_FALSE(steady.Time().positivity.enabled);
    CHECK(steady.Provenance().at("/time/positivity/enabled") == "BDF1 steady limit");
    for (const auto *name : {"relax", "scale", "increaseLimit"})
    {
        auto bad = input;
        bad["time"]["positivity"][name] = 0;
        CHECK_THROWS(C::SimulationConfig::Resolve(bad, 2));
    }
    input["time"]["positivity"]["relax"] = 1.1;
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 2));
}
TEST_CASE("Strict implicit VR acceptance needs an explicit positive reconstruction tolerance")
{
    C::Json input{{"defaults", {{"implicit", {{"requireConvergedReconstruction", true}}}, {"numerics", {{"reconstruction", "variational"}}}}}};
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 1));
    input["defaults"]["numerics"]["reconstructionTolerance"] = 1e-10;
    CHECK(C::SimulationConfig::Resolve(input, 1).Domains()[0].settings.implicit.requireConvergedReconstruction);
}
TEST_CASE("Steady config resolves Euler's BDF1 infinite timestep override with provenance")
{
    const auto config = C::SimulationConfig::Resolve({{"time", {{"steady", true}, {"method", "esdirk4"}, {"steps", 123}, {"finalTime", 10}, {"cfl", 1}, {"maxRetries", 3}}}}, 2);
    CHECK(config.Time().steady);
    CHECK(config.Time().method == "bdf1");
    CHECK(config.Time().steps == 1);
    CHECK(config.Time().tEnd == 0);
    CHECK(config.Time().finalTime == -1);
    CHECK(config.Time().cfl == 0);
    CHECK(config.Time().maxRetries == 0);
    CHECK(config.Root()["time"]["method"] == "bdf1");
    CHECK(config.Provenance().at("/time/method") == "BDF1 steady limit");
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"consoleInterval", -1}}}}}}, 2));
}
TEST_CASE("Deprecated time and output aliases are exclusive with their canonical controls")
{
    const auto oldTime = C::SimulationConfig::Resolve({{"time", {{"finalTime", 2.5}}}}, 1);
    CHECK(oldTime.Time().tEnd == 2.5);
    CHECK(oldTime.Time().finalTime == -1);
    const auto oldOutput = C::SimulationConfig::Resolve({{"output", {{"interval", 7}}}}, 1);
    CHECK(oldOutput.Output().checkpointInterval == 7);
    CHECK(oldOutput.Output().interval == -1);

    CHECK_THROWS_WITH(C::SimulationConfig::Resolve({{"time", {{"tEnd", 2.5}, {"finalTime", 2.5}}}}, 1),
                      doctest::Contains("only time.tEnd"));
    CHECK_THROWS_WITH(C::SimulationConfig::Resolve({{"output", {{"checkpointInterval", 7}, {"interval", 7}}}}, 1),
                      doctest::Contains("only output.checkpointInterval"));
    CHECK_THROWS_WITH(C::SimulationConfig::Resolve({{"time", {{"finalTime", 2.5}}}}, 1,
                                                   {{"/time/tEnd", 2.5}}),
                      doctest::Contains("only time.tEnd"));
    CHECK_THROWS_WITH(C::SimulationConfig::Resolve({{"output", {{"interval", 7}}}}, 1,
                                                   {{"/output/checkpointInterval", 7}}),
                      doctest::Contains("only output.checkpointInterval"));
    CHECK_THROWS_WITH(C::SimulationConfig::Resolve({{"output", {{"implicit", {{"checkpointInterval", 2}}}}}}, 1),
                      doctest::Contains("internal-iteration"));
    const C::Json multiDomain{{"domains", C::Json::array({{{"name", "a"}}, {{"name", "b"}}})},
                              {"time", {{"method", "bdf1"}}},
                              {"output", {{"implicit", {{"checkpointInterval", 2}}}}}};
    CHECK_THROWS_WITH(C::SimulationConfig::Resolve(multiDomain, 2),
                      doctest::Contains("exact internal checkpoints currently require one domain"));
}
TEST_CASE("Reconstruction PP controls are explicit and reject unsupported compatibility combinations")
{
    C::NumericsSettings settings;
    CHECK(settings.positivityReconstruction == "Euler");
    CHECK(settings.positivityClippedReferencePolicy == "fallbackToBase");
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivity = true;
    settings.positivityCompressToMean = false;
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivityReconstruction = "Segment";
    CHECK_THROWS(settings.ValidatePositivity());
    settings.positivityCompressToMean = true;
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivityBetaPower = -1;
    CHECK_THROWS(settings.ValidatePositivity());
    settings.positivityBetaPower = 2;
    CHECK_THROWS(settings.ValidatePositivity());
    settings.positivityReconstruction = "Euler";
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivityReconstruction = "unknown";
    CHECK_THROWS(settings.ValidatePositivity());
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"positivityReconstruction", "unknown"}}}}}}, 2));
    settings.positivityReconstruction = "Euler";
    settings.positivityClippedReferencePolicy = "backtracking";
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivityClippedReferencePolicy = "unknown";
    CHECK_THROWS(settings.ValidatePositivity());
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"positivityClippedReferencePolicy", "unknown"}}}}}}, 2));
}
TEST_CASE("Cauchy implicit controls resolve through the root and reject contradictory paths")
{
    const auto defaults = C::SimulationConfig::Resolve(C::Json::object(), 2);
    CHECK(defaults.Domains()[0].settings.implicit.residualComponents == std::vector<std::string>{"rho"});
    CHECK(defaults.Domains()[0].settings.implicit.linear.stopping == "residual");
    const auto fixed = C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"linear", {{"method", "ilu"}, {"stopping", "fixedCount"}}}}}}}}, 2);
    CHECK(fixed.Domains()[0].settings.implicit.linear.stopping == "fixedCount");
    CHECK(fixed.Root()["domains"][0]["settings"]["implicit"]["linear"]["stopping"] == "fixedCount");
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"linear", {{"method", "gmres"}, {"stopping", "fixedCount"}}}}}}}}, 2));
    const auto named = C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"residualComponents", {"rho", "flow.rhou"}}}}}}}, 2);
    CHECK(named.Domains()[0].settings.implicit.residualComponents == std::vector<std::string>{"rho", "flow.rhou"});
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"residualComponents", {0}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"residualComponents", {"rho", "rho"}}}}}}}, 2));
    for (const std::string method : {"ssprk3", "bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"})
    {
        const auto config = C::SimulationConfig::Resolve({{"time", {{"method", method}}}, {"defaults", {{"implicit", {{"linear", {{"preconditioner", "ilu"}}}}}}}}, 2);
        CHECK(config.Time().method == method);
        CHECK(config.Domains()[0].settings.implicit.linear.preconditioner == "ilu");
        CHECK(config.Domains()[0].settings.implicit.sgs.exchangeBetweenSweeps == false);
    }
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "unavailable"}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "ditr"}, {"ditr", {{"fraction", .9}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "ditr"}, {"ditr", {{"iterations", 0}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "ditr"}, {"ditr", {{"absoluteTolerance", -1}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"roeDissipation", true}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"iterations", 0}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"jacobianAction", "unknown"}}}}}}, 2));
    for (const std::string name : {"directEvaluated", "semiMatrixFree"})
        CHECK(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"jacobianAction", name}}}}}}, 2)
                  .Domains()[0]
                  .settings.implicit.jacobianAction == name);
    CHECK(C::ImplicitSettings{}.jacobianAction == "assembled");
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"linear", {{"preconditioner", "ilu"}}}, {"sgs", {{"exchangeBetweenSweeps", true}}}}}}}}, 2));
    CHECK(C::ImplicitILUSettings{}.ordering == "natural");
    CHECK(C::ImplicitILUSettings{}.preconditionerCorrections == 1);
    CHECK(C::ImplicitLinearSettings{}.componentScaling == "none");
    const auto metric = C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"linear", {{"componentScaling", "meanState"}}}, {"ilu", {{"preconditionerCorrections", 2}}}}}}}}, 2);
    CHECK(metric.Domains().front().settings.implicit.linear.componentScaling == "meanState");
    CHECK(metric.Domains().front().settings.implicit.ilu.preconditionerCorrections == 2);
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"linear", {{"componentScaling", "unknown"}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"ilu", {{"ordering", "unknown"}}}}}}}}, 2));
    CHECK(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"ilu", {{"ordering", "rcm"}}}}}}}}, 2).Domains().front().settings.implicit.ilu.ordering == "rcm");
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"positivity", {{"rhsFPPRelax", 0}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"positivity", {{"rhsFPPReduction", "unknown"}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"multigrid", {{"levels", 3}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"multigrid", {{"levels", 1}}}}}}}}, 2));
    CHECK_NOTHROW(C::SimulationConfig::Resolve({{"time", {{"method", "bdf1"}}}, {"defaults", {{"numerics", {{"reconstruction", "variational"}, {"cfv", {{"maxOrder", 3}}}}}, {"implicit", {{"multigrid", {{"levels", 2}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"reconstruction", "variational"}, {"cfv", {{"maxOrder", 3}}}}}, {"implicit", {{"multigrid", {{"levels", 2}}}}}}}}, 2));
    const auto sor = C::SimulationConfig::Resolve({{"defaults", {{"execution", {{"overlap", false}}}, {"numerics", {{"cfv", {{"SORInstead", true}, {"coreFirst", false}}}}}}}}, 2);
    CHECK(sor.Domains()[0].settings.numerics.cfv.coreFirst);
    CHECK(sor.Root()["domains"][0]["settings"]["numerics"]["cfv"]["coreFirst"].get<bool>());
    CHECK(sor.Provenance().at("/domains/0/settings/numerics/cfv/coreFirst") == "Cauchy SOR logical ordering");
}
TEST_CASE("Cauchy partial analytic boundary profile receives nested schema defaults")
{
    const auto config = C::SimulationConfig::Resolve({{"defaults", {{"boundaries", {{"7", {{"kind", "analytic"}, {"profile", {{"kind", "doubleMachReflection"}}}}}}}}}}, 1);
    const auto &profile = config.Domains()[0].settings.boundaries.at("7").profile;
    CHECK(profile.kind == "doubleMachReflection");
    CHECK(profile.strength == 5);
    CHECK(profile.period == 10);
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"mesh", {{"boundaryIds", {{"FAR", 1}}}}}}}}, 1));
    CHECK(C::SimulationConfig::Resolve(C::Json::object(), 1).Output().hdfDeflateLevel == 0);
    CHECK(C::SimulationConfig::Resolve(C::Json::object(), 1).Output().historyPrecision == 10);
    CHECK(C::SimulationConfig::Resolve({{"output", {{"historyPrecision", 8}}}}, 1).Output().historyPrecision == 8);
    CHECK_THROWS(C::SimulationConfig::Resolve({{"output", {{"historyPrecision", 0}}}}, 1));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"output", {{"historyPrecision", 18}}}}, 1));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"output", {{"hdfDeflateLevel", 10}}}}, 1));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"output", {{"hdfDeflateLevel", -1}}}}, 1));
}
TEST_CASE("Cauchy root resolves profiles overrides CLI and condition defaults with provenance")
{
    const auto input = C::Json::parse(R"json({
        "profiles": {"gas": {"equation": {"gamma": 1.6}, "execution": {"layout": "soa"}}},
        "defaults": {"equation": {"gamma": 1.5}},
        "domains": [
            {"name": "left", "ranks": [2, 0], "profiles": ["gas"],
             "overrides": {"equation": {"gamma": 1.45}, "boundaries": {"7": {"kind": "farfield"}},
                 "numerics": {"cfv": {"boundaryZoneWeights": [[7, [1, 0.5, 0, 0]]]}}}},
            {"name": "right", "ranks": [1, 3]}
        ]
    })json");
    const auto resolved = C::SimulationConfig::Resolve(input, 4, {{"/domains/0/settings/equation/gamma", 1.4}});
    REQUIRE(resolved.Domains().size() == 2);
    CHECK(resolved.Domains()[0].settings.equation.gamma == 1.4);
    CHECK(resolved.Domains()[1].settings.equation.gamma == 1.5);
    CHECK(resolved.Domains()[0].settings.execution.layout == "soa");
    CHECK(resolved.Domains()[1].settings.execution.layout == "aos");
    CHECK(resolved.Domains()[0].settings.boundaries.at("7").primitive[4] == 1);
    CHECK(resolved.Domains()[0].settings.numerics.cfv.boundaryZoneWeights.at(7)[1] == 0.5);
    CHECK(resolved.RankGroups()[0].ranks == std::vector<int>{2, 0});
    CHECK(resolved.Provenance().at("/domains/0/settings/equation/gamma") == "CLI /domains/0/settings/equation/gamma");
    CHECK(resolved.Provenance().at("/domains/0/settings/execution/layout") == "profile gas");
    CHECK(resolved.Provenance().at("/domains/1/settings/equation/gamma") == "solver defaults");
    CHECK(resolved.Provenance().at("/domains/0/settings/boundaries/7/primitive/4") == "schema default");
    auto changed = input;
    changed["defaults"]["equation"]["gamma"] = 1.8;
    CHECK(resolved.Domains()[1].settings.equation.gamma == 1.5);
}
TEST_CASE("Cauchy rejects unknown nested keys unused contradictory settings and unsupported capabilities")
{
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"equation", {{"gama", 1.4}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"cfv", {{"functional", {{"badWeight", 1}}}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"boundaries", {{"7", {{"badBC", 1}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"boundaries", {{"07", C::Json::object()}}}}}}, 2));
    CHECK_NOTHROW(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"riemann", {{"fixScale", 0}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"flux", "Roe_M3"}, {"riemann", {{"fixScale", 0}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"equation", {{"viscosity", 0.1}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"dt", -1}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"steps", 2.5}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"domains", C::Json::array({{{"name", "bad"}, {"ranks", {0, 0}}}})}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve(C::Json::object(), 2, {{"/domains/0/settings/numerics/mispelled", 1}}));
    const C::Json unsupported{{"defaults", {{"execution", {{"device", "cuda"}}}, {"equation", {{"eos", "realGas"}}}}}};
    try
    {
        C::SimulationConfig::Resolve(unsupported, 2);
        FAIL("missing capabilities were accepted");
    }
    catch (const std::exception &error)
    {
        CHECK(std::string(error.what()).find("backend cuda") != std::string::npos);
        CHECK(std::string(error.what()).find("EOS realGas") != std::string::npos);
    }
}
