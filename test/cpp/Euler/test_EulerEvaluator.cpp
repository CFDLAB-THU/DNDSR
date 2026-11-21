/**
 * @file test_EulerEvaluator.cpp
 * @brief Integration test for the EulerEvaluator pipeline: fdt -> frhs -> fsolve.
 *
 * Exercises the full evaluator pipeline on three configurations:
 *   1. Isentropic Vortex (NS, 2D periodic, inviscid)
 *   2. NACA0012 (NS_SA, 2D external, viscous + SA turbulence)
 *   3. 3D Box (NS_3D, 3D periodic, inviscid)
 *
 * For each case the test:
 *   - Reads the JSON config and mesh
 *   - Initializes the evaluator and DOF arrays
 *   - Runs one Newton-like iteration: EvaluateDt -> EvaluateRHS -> LUSGSMatrixInit
 *     -> LUSGSForward -> LUSGSBackward
 *   - Measures the L1 RHS norm (golden regression value)
 *   - Repeats from identical solution/halo data using native mesh authority,
 *     rebuilding FV/wall caches after corrupting detached facade metadata
 *   - Compares every owned dt, RHS and increment component between the runs
 *
 * This is an MPI test registered at np=1, 2, 4.
 * Golden value sentinel: 1e300 means not yet acquired.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Euler/EulerSolver.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "EulerP/EulerP_Evaluator_impl_common.hxx"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <map>
#include <limits>
#include <hdf5.h>
#include <unistd.h>

TEST_CASE("Existing Euler configurations retain full mesh audits when new keys are absent")
{
    using Configuration = DNDS::Euler::EulerSolver<DNDS::Euler::NS>::Configuration;
    Configuration original(DNDS::Euler::getNVars(DNDS::Euler::NS));
    nlohmann::ordered_json input;
    original.ReadWriteJson(input, DNDS::Euler::getNVars(DNDS::Euler::NS), false);
    auto &meshInput = input.at("dataIOControl");
    meshInput.erase("meshAuditPersistentIdentity");
    meshInput.erase("meshAuditClosure");
    Configuration restored(DNDS::Euler::getNVars(DNDS::Euler::NS));
    restored.ReadWriteJson(input, DNDS::Euler::getNVars(DNDS::Euler::NS), true);
    CHECK(restored.dataIOControl.meshAuditPersistentIdentity);
    CHECK(restored.dataIOControl.meshAuditClosure);
    meshInput["meshAuditPersistentIdentity"] = false;
    meshInput["meshAuditClosure"] = false;
    restored.ReadWriteJson(input, DNDS::Euler::getNVars(DNDS::Euler::NS), true);
    CHECK_FALSE(restored.dataIOControl.meshAuditPersistentIdentity);
    CHECK_FALSE(restored.dataIOControl.meshAuditClosure);
}

using namespace DNDS;
using namespace DNDS::Euler;

static MPIInfo g_mpi;
static constexpr real GOLDEN_NOT_ACQUIRED = 1e300;

// ===================================================================
// Helper: resolve path relative to project root
// ===================================================================
static std::string projectRoot()
{
    std::string f(__FILE__);
    for (int i = 0; i < 4; i++) // test/cpp/Euler/file -> root
    {
        auto pos = f.rfind('/');
        if (pos == std::string::npos)
            pos = f.rfind('\\');
        if (pos != std::string::npos)
            f = f.substr(0, pos);
    }
    return f;
}

// ===================================================================
// Helper: write a temp JSON config file with mesh path fixup
// ===================================================================
/// Write a temp config by: (1) writing defaults via ConfigureFromJson(read=false),
/// (2) reading the case JSON and merge-patching, (3) applying test overrides.
/// This ensures all required config fields exist (filled with defaults).
template <EulerModel model>
static std::string writeTempConfig(const std::string &caseJsonPath, const std::string &tag,
                                   const nlohmann::ordered_json &overrides = {})
{
    int writerPid = int(getpid());
    MPI_Bcast(&writerPid, 1, MPI_INT, 0, g_mpi.comm);
    auto tmpDir = std::filesystem::temp_directory_path() / "dndsr_euler_test" / std::to_string(writerPid);
    std::filesystem::create_directories(tmpDir);
    std::string defaultPath = (tmpDir / (tag + "_default.json")).string();
    std::string cfgPath = (tmpDir / (tag + ".json")).string();

    // Step 1: Write defaults
    {
        EulerSolver<model> solverTmp(g_mpi);
        solverTmp.ConfigureFromJson(defaultPath, false); // writes all defaults
    }
    MPI::Barrier(g_mpi.comm);

    // Step 2: Read defaults, merge case JSON, merge overrides
    auto fDefault = std::ifstream(defaultPath);
    DNDS_assert(fDefault);
    auto j = nlohmann::ordered_json::parse(fDefault, nullptr, true, true);
    fDefault.close();

    auto fCase = std::ifstream(caseJsonPath);
    DNDS_assert_info(fCase, ("config file not found: " + caseJsonPath).c_str());
    auto jCase = nlohmann::ordered_json::parse(fCase, nullptr, true, true);
    fCase.close();

    j.merge_patch(jCase);
    j.merge_patch(overrides);

    // Fix mesh path to be absolute
    std::string meshRel = j["dataIOControl"]["meshFile"];
    std::filesystem::path meshAbs = std::filesystem::weakly_canonical(
        std::filesystem::path(caseJsonPath).parent_path() / meshRel);
    j["dataIOControl"]["meshFile"] = meshAbs.string();

    // Disable all output
    j["outputControl"]["nDataOutC"] = 1000000;
    j["outputControl"]["nDataOut"] = 1000000;
    j["outputControl"]["nDataOutCInternal"] = 1000000;
    j["outputControl"]["nDataOutInternal"] = 1000000;
    j["outputControl"]["dataOutAtInit"] = false;
    j["timeMarchControl"]["nTimeStep"] = 1;

    // Write final config
    if (g_mpi.rank == 0)
    {
        auto fOut = std::ofstream(cfgPath);
        DNDS_assert(fOut);
        fOut << std::setw(4) << j;
        fOut.close();
    }
    MPI::Barrier(g_mpi.comm);

    return cfgPath;
}

// ===================================================================
// Template: one Newton-like step for a given EulerModel
// Returns: {rhsL1Norm_density, rhsL1Norm_energy, incrementL2Norm}
// ===================================================================
template <EulerModel model>
std::array<real, 3> runOneNewtonStep(const std::string &cfgPath)
{
    using TSolver = EulerSolver<model>;
    using TEval = EulerEvaluator<model>;
    constexpr int nVarsFixed = TEval::nVarsFixed;
    constexpr int gDim = TEval::gDim;
    int nVars = getNVars(model);

    // --- Setup ---
    TSolver solver(g_mpi);
    solver.ConfigureFromJson(cfgPath, true);
    solver.ReadMeshAndInitialize();
    CHECK_FALSE(solver.hasLimitedVRStorage());
    CHECK_FALSE(solver.getReconstructionMask().father);
    const auto inactive = solver.getOutputOverlapDataRefs();
    CHECK_FALSE(inactive.lvrAlpha.father);
    CHECK_FALSE(inactive.lvrBranchAlpha.father);
    CHECK_FALSE(inactive.lvrO2Beta.father);

    auto &eval = solver.getEval();
    auto &u = solver.getU();
    auto &uRec = solver.getURec();
    auto &uRecNew = solver.getURecNew();
    auto vfv = solver.getVFV();
    auto mesh = solver.getMesh();
    auto &config = solver.getConfiguration();
    auto &JSource = solver.getJSource();
    auto &betaPP = solver.getBetaPP();

    // Build working arrays
    ArrayDOFV<nVarsFixed> rhs, xinc, xincNew;
    ArrayDOFV<1> dTau, alphaPP_tmp;
    JacobianDiagBlock<nVarsFixed> JD;

    vfv->BuildUDof(rhs, nVars);
    vfv->BuildUDof(xinc, nVars);
    vfv->BuildUDof(xincNew, nVars);
    vfv->BuildUDof(dTau, 1);
    vfv->BuildUDof(alphaPP_tmp, 1);
    alphaPP_tmp.setConstant(1.0);

    JD.SetModeAndInit(
        config.eulerSettings.useScalarJacobian ? 0 : 1,
        nVars, u);

    // --- Initialize DOF (not done by ReadMeshAndInitialize) ---
    eval.InitializeUDOF(u);
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();

    eval.FixUMaxFilter(u);
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
    std::vector<Eigen::VectorXd> initialU, initialBeta;
    std::vector<Eigen::MatrixXd> initialRec, initialRecNew;
    for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
    {
        initialU.emplace_back(u[cell]);
        initialBeta.emplace_back(betaPP[cell]);
        initialRec.emplace_back(uRec[cell]);
        initialRecNew.emplace_back(uRecNew[cell]);
    }
    struct PipelineResult
    {
        std::array<real, 3> norms;
        std::vector<real> owned;
        real minDt;
    };
    auto run = [&]() -> PipelineResult
    {
        // Restore the complete initial halo as well as owned cells; neither
        // run may consume solution or reconstruction values left by the other.
        for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
        {
            u[cell] = initialU.at(cell);
            betaPP[cell] = initialBeta.at(cell);
            uRec[cell] = initialRec.at(cell);
            uRecNew[cell] = initialRecNew.at(cell);
        }
        alphaPP_tmp.setConstant(1.0);
        JSource.clearValues();
        JD.SetModeAndInit(config.eulerSettings.useScalarJacobian ? 0 : 1, nVars, u);
        // --- STEP 1: fdt (compute local pseudo-timestep) ---
        const real CFL = config.implicitCFLControl.CFL;
        real curDtMin = 1e100;

        eval.EvaluateDt(dTau, u, uRec, CFL, curDtMin, 1e100,
                        config.implicitCFLControl.useLocalDt, 0.0);

        // --- STEP 2: skip reconstruction (first-order, uRec stays zero) ---
        // For a proper high-order test we'd need the boundary callback from
        // frhsOuter, which requires significant setup. A first-order RHS
        // evaluation is a valid regression test of the evaluator pipeline.

        // --- STEP 3: frhs (evaluate spatial RHS, first-order) ---
        eval.EvaluateRHS(rhs, JSource, u, uRec, uRec,
                         betaPP, alphaPP_tmp, false, 0.0);
        rhs.trans.startPersistentPull();
        rhs.trans.waitPersistentPull();

        // --- Measure RHS L1 norm (volume-weighted) ---
        Eigen::VectorXd resNorm(nVars);
        eval.EvaluateNorm(resNorm, rhs, 1, true);

        if (g_mpi.rank == 0)
        {
            std::cout << "  RHS L1 norms:";
            for (int i = 0; i < nVars; i++)
                std::cout << " " << std::scientific << std::setprecision(10) << resNorm(i);
            std::cout << std::endl;
        }

        // --- STEP 4: fsolve (one Jacobi iteration -- MPI-deterministic) ---
        // Jacobi pattern: Forward(uInc, uIncNew); Backward(uInc, uIncNew); pull uIncNew; swap.
        // Since uInc starts at zero, this is a block-diagonal solve: xincNew = D^{-1} * rhs.
        real dt = 1e100; // pseudo-steady
        real alphaDiag = 1.0;

        eval.LUSGSMatrixInit(JD, JSource, dTau, dt, alphaDiag, u, uRec, 0, 0.0);

        xinc.setConstant(0.0);
        xincNew.setConstant(0.0);
        eval.UpdateLUSGSForward(alphaDiag, 0.0, rhs, u, xinc, JD, xincNew);
        eval.UpdateLUSGSBackward(alphaDiag, 0.0, rhs, u, xinc, JD, xincNew);
        xincNew.trans.startPersistentPull();
        xincNew.trans.waitPersistentPull();
        xinc = xincNew;

        // --- Measure increment L2 norm ---
        Eigen::VectorXd incNorm(nVars);
        eval.EvaluateNorm(incNorm, xinc, 2, false);
        real incL2 = incNorm.norm();

        if (g_mpi.rank == 0)
            std::cout << "  Inc L2 norm: " << std::scientific << std::setprecision(10) << incL2 << std::endl;

        PipelineResult result{{resNorm(0), resNorm(nVars > 4 ? 4 : nVars - 1), incL2}, {}, curDtMin};
        for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
        {
            result.owned.push_back(dTau[cell](0));
            for (int variable = 0; variable < nVars; ++variable)
                result.owned.push_back(rhs[cell](variable));
            for (int variable = 0; variable < nVars; ++variable)
                result.owned.push_back(xinc[cell](variable));
        }
        return result;
    };
    REQUIRE(mesh->HasNativeState());
    auto checkNativeOnly = [&]
    {
        CHECK_NOTHROW(mesh->CurrentNativeState()->Validate());
    };
    checkNativeOnly();
    const auto baseline = run();
    auto requirements = Geom::MeshReadRequirements::CellFlux();
    requirements.allFaceCellsLocal = true;
    requirements.wallDistance = vfv->getSettings().functionalSettings.anisotropicType ==
                                CFV::VRSettings::FunctionalSettings::AnisotropicType::WallDist;
    CHECK_NOTHROW(mesh->CurrentNativeState()->CheckRequirements(requirements));
    // Poison surviving reader-configuration metadata before rebuilding metrics.
    // Current native geometry/topology must remain the query authority.
    mesh->isPeriodic = !mesh->isPeriodic;
    mesh->periodicInfo = Geom::Periodicity{};
    TEval::InitializeFV(mesh, vfv, solver.getBCHandler());
    eval.GetWallDist();
    const auto native = run();
    checkNativeOnly();
    REQUIRE(native.owned.size() == baseline.owned.size());
    CHECK(native.minDt == doctest::Approx(baseline.minDt).epsilon(1e-11));
    for (size_t value = 0; value < baseline.owned.size(); ++value)
    {
        CAPTURE(value);
        CHECK(std::isfinite(native.owned[value]));
        CHECK(native.owned[value] == doctest::Approx(baseline.owned[value]).epsilon(1e-11));
    }
    return baseline.norms;
}

// ===================================================================
// TEST 1: Isentropic Vortex (NS, 2D, inviscid)
// ===================================================================

TEST_CASE("EulerEvaluator pipeline: IV (NS, P1, 2D)")
{
    std::string root = projectRoot();
    std::string cfgPath = writeTempConfig<NS>(
        root + "/cases/euler/euler_config_IV.json", "iv_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
         {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});

    if (g_mpi.rank == 0)
        std::cout << "=== IV (NS, P1) ===" << std::endl;

    auto [rhsDensity, rhsEnergy, incL2] = runOneNewtonStep<NS>(cfgPath);

    CHECK(std::isfinite(rhsDensity));
    CHECK(std::isfinite(rhsEnergy));
    CHECK(std::isfinite(incL2));
    CHECK(rhsDensity > 0);

    // Golden values (Jacobi iteration is MPI-deterministic)
    real goldenRhsDensity = 1.0524823780e+01;
    real goldenIncL2 = 3.1557730986e+00;

    if (g_mpi.rank == 0)
        std::cout << "  golden candidates: rhsDensity=" << std::scientific
                  << std::setprecision(10) << rhsDensity
                  << " incL2=" << incL2 << std::endl;

    CHECK(rhsDensity == doctest::Approx(goldenRhsDensity).epsilon(1e-6));
    CHECK(incL2 == doctest::Approx(goldenIncL2).epsilon(1e-6));
}

// ===================================================================
// TEST 2: NACA0012 (NS_SA, 2D, viscous + SA)
// ===================================================================

TEST_CASE("EulerEvaluator pipeline: NACA0012 (NS_SA, P1)" * doctest::test_suite("large-mesh"))
{
    std::string root = projectRoot();
    std::string cfgPath = writeTempConfig<NS_SA>(
        root + "/cases/eulerSA/eulerSA_config.json", "naca_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
         {"restartState", {{"iStep", 0}, {"iStepInternal", 0}}},
         {"dataIOControl", {{"meshFile", root + "/data/mesh/NACA0012_H2.cgns"}}}});

    if (g_mpi.rank == 0)
        std::cout << "=== NACA0012 (NS_SA, P1) ===" << std::endl;

    auto [rhsDensity, rhsEnergy, incL2] = runOneNewtonStep<NS_SA>(cfgPath);

    CHECK(std::isfinite(rhsDensity));
    CHECK(std::isfinite(rhsEnergy));
    CHECK(std::isfinite(incL2));
    CHECK(rhsDensity > 0);

    real goldenRhsDensity = 2.4001929638e-01;
    real goldenIncL2 = 9.7941775117e+01;

    if (g_mpi.rank == 0)
        std::cout << "  golden candidates: rhsDensity=" << std::scientific
                  << std::setprecision(10) << rhsDensity
                  << " incL2=" << incL2 << std::endl;

    CHECK(rhsDensity == doctest::Approx(goldenRhsDensity).epsilon(1e-6));
    CHECK(incL2 == doctest::Approx(goldenIncL2).epsilon(1e-6));
}

TEST_CASE("EulerEvaluator pipeline: SA IDDES with runtime ft2 disabled" * doctest::test_suite("large-mesh"))
{
    const std::string root = projectRoot();
    const std::string cfgPath = writeTempConfig<NS_SA>(
        root + "/cases/eulerSA/eulerSA_config.json", "naca_iddes_noft2_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
         {"restartState", {{"iStep", 0}, {"iStepInternal", 0}}},
         {"eulerSettings", {{"SADESScale", .65}, {"SADESMode", 1}, {"SAConfig", {{"useFt2", false}}}}},
         {"dataIOControl", {{"meshFile", root + "/data/mesh/NACA0012_H2.cgns"}}}});
    const auto norms = runOneNewtonStep<NS_SA>(cfgPath);
    for (DNDS::real norm : norms)
        CHECK(std::isfinite(norm));
    CHECK(norms[0] > 0);
    CHECK(norms[2] > 0);
}

// ===================================================================
// TEST 3: 3D Box (NS_3D, periodic, inviscid)
// ===================================================================
TEST_CASE("EulerEvaluator pipeline: Box (NS_3D, P1)" * doctest::test_suite("large-mesh"))
{
    std::string root = projectRoot();
    std::string cfgPath = writeTempConfig<NS_3D>(
        root + "/cases/euler3D/euler3D_config_Box.json", "box3d_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}, {"cacheDiffBase", false}}},
         {"dataIOControl", {{"meshFile", root + "/data/mesh/Uniform32_3D_Periodic.cgns"}}}});

    if (g_mpi.rank == 0)
        std::cout << "=== Box (NS_3D, P1) ===" << std::endl;

    auto [rhsDensity, rhsEnergy, incL2] = runOneNewtonStep<NS_3D>(cfgPath);

    CHECK(std::isfinite(rhsDensity));
    CHECK(std::isfinite(rhsEnergy));
    CHECK(std::isfinite(incL2));
    CHECK(rhsDensity > 0);

    real goldenRhsDensity = 5.8855408029e-01;
    real goldenIncL2 = 3.4532977412e+01;

    if (g_mpi.rank == 0)
        std::cout << "  golden candidates: rhsDensity=" << std::scientific
                  << std::setprecision(10) << rhsDensity
                  << " incL2=" << incL2 << std::endl;

    CHECK(rhsDensity == doctest::Approx(goldenRhsDensity).epsilon(1e-6));
    CHECK(incL2 == doctest::Approx(goldenIncL2).epsilon(1e-6));
}

TEST_CASE("RANS wall cache rejects changed geometry and metric publications")
{
    const auto root = projectRoot();
    const auto cfgPath = writeTempConfig<NS_SA>(
        root + "/cases/eulerSA/eulerSA_config.json", "wall_cache_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
         {"restartState", {{"iStep", 0}, {"iStepInternal", 0}}},
         {"eulerSettings", {{"wallDistScheme", 0}}},
         {"dataIOControl", {{"meshFile", root + "/data/mesh/Uniform_3x3_wall.cgns"}, {"meshDirectBisect", 0}, {"meshScale", 1}, {"meshRotZ", 0}}}});
    EulerSolver<NS_SA> solver(g_mpi);
    solver.ConfigureFromJson(cfgPath, true);
    solver.ReadMeshAndInitialize();
    auto mesh = solver.getMesh();
    auto vfv = solver.getVFV();
    auto &eval = solver.getEval();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());
    const auto before = eval.dWall;
    DNDS::real total = 0;
    for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
        total += before.at(cell).mean();
    MPI::AllreduceOneReal(total, MPI_SUM, g_mpi);
    REQUIRE(total > 0);

    mesh->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2);
    vfv->ConstructMetrics();
    vfv->ConstructBaseAndWeight();
    vfv->ConstructRecCoeff();
    CHECK_NOTHROW(vfv->ValidateMeshDependencies());
    CHECK_THROWS(eval.ValidateWallDistanceDependencies());
    ArrayDOFV<1> dt;
    vfv->BuildUDof(dt, 1);
    DNDS::real minDt = 1e100;
    CHECK_THROWS(eval.EvaluateDt(dt, solver.getU(), solver.getURec(), 1, minDt, 1e100, true, 0));
    eval.GetWallDist();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());
    REQUIRE(eval.dWall.size() == before.size());
    for (size_t cell = 0; cell < before.size(); ++cell)
    {
        REQUIRE(eval.dWall[cell].size() == before[cell].size());
        for (Eigen::Index point = 0; point < before[cell].size(); ++point)
            CHECK(eval.dWall[cell](point) == doctest::Approx(2 * before[cell](point)).epsilon(1e-11));
    }

    // Rebuilding quadrature/metrics in the same mesh epoch also requires new
    // distances, even when the number of rows remains unchanged.
    vfv->FinalizeMetrics();
    vfv->ConstructBaseAndWeight();
    vfv->ConstructRecCoeff();
    CHECK_THROWS(eval.ValidateMeshDependencies());
    eval.GetWallDist();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());

    const auto scheme = eval.settings.wallDistScheme;
    eval.settings.wallDistScheme = -123;
    CHECK_THROWS(eval.GetWallDist());
    eval.settings.wallDistScheme = scheme;
    CHECK_THROWS(eval.ValidateWallDistanceDependencies());
    eval.GetWallDist();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());
}

// Basic end-to-end LVR fixture: a two-dimensional state, target PP, all
// branches and two halo layers, compared cellwise against MPI_COMM_SELF.
// Keep this small; it is an integration regression, not a CFD benchmark.
using LVRCellKey = std::array<long long, 2>;
using LVRCellValues = std::vector<DNDS::real>;
using LVRCells = std::map<LVRCellKey, LVRCellValues>;

TEST_CASE("Euler two-equation turbulent normal stress follows the physical face normal")
{
    using Eval = EulerEvaluator<NS_2EQ>;
    const auto root = projectRoot();
    const auto path = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "rans_normal_stress",
                                          {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                           {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});
    EulerSolver<NS> fixture(g_mpi);
    fixture.ConfigureFromJson(path, true);
    fixture.ReadMeshAndInitialize();
    EulerEvaluatorSettings<NS_2EQ> settings(7);
    auto bc = std::make_shared<BoundaryHandler<NS_2EQ>>(7);
    Eval eval(fixture.getMesh(), fixture.getVFV(), bc, settings);
    constexpr int nPoints = 6;
    Eval::TVec_Batch normals, gridVelocity;
    normals.resize(3, nPoints);
    normals << 1, 0, 0, -1, .6, 2. / 3.,
        0, 1, 0, 0, -.8, -1. / 3.,
        0, 0, 1, 0, 0, 2. / 3.;
    gridVelocity.setZero(3, nPoints);
    Eval::TDiffU_Batch gradient, primitiveGradient;
    gradient.setZero(3 * nPoints, 7);
    primitiveGradient.setZero(3 * nPoints, 7);
    for (auto rans : {RANSModel::RANS_KOWilcox, RANSModel::RANS_KOSST, RANSModel::RANS_RKE})
        for (bool moving : {false, true})
        {
            CAPTURE(rans);
            CAPTURE(moving);
            eval.settings.ransModel = rans;
            Eval::TU_Batch left, right, leftFix, rightFix, flux;
            left.resize(7, nPoints);
            const Eigen::Vector3d velocity = moving ? Eigen::Vector3d(.4, -.3, .2) : Eigen::Vector3d::Zero();
            for (int q = 0; q < nPoints; ++q)
            {
                const real rho = 1 + .1 * q;
                left.col(q) << rho, rho * velocity, 1 / .4 + .5 * rho * velocity.squaredNorm(), .1 + .01 * q, 1;
                gridVelocity.col(q) = moving ? Eigen::Vector3d(.1, .2, -.05) : Eigen::Vector3d::Zero();
            }
            right = left;
            Eval::TU meanLeft = left.col(0), meanRight = right.col(0);
            Eval::TVec meanNormal = normals.col(0), meanGridVelocity = gridVelocity.col(0);
            Eval::TReal_Batch lam0, lam123, lam4;
            lam0.resize(nPoints), lam123.resize(nPoints), lam4.resize(nPoints);
            eval.fluxFace(left, right, meanLeft, meanRight, gradient, primitiveGradient,
                          normals, gridVelocity, meanNormal, meanGridVelocity, leftFix, rightFix, flux,
                          lam0, lam123, lam4, Geom::BC_ID_INTERNAL, Gas::Roe_M1, 0, true);
            for (int q = 0; q < nPoints; ++q)
            {
                const real rho = left(0, q), turbulentPressure = (2. / 3.) * left(5, q);
                const real normalVelocity = velocity.dot(normals.col(q));
                const real relativeNormalVelocity = (velocity - gridVelocity.col(q)).dot(normals.col(q));
                Eval::TU expected;
                expected << rho * relativeNormalVelocity,
                    rho * velocity * relativeNormalVelocity + (1 + turbulentPressure) * normals.col(q),
                    left(4, q) * relativeNormalVelocity + (1 + turbulentPressure) * normalVelocity,
                    left(5, q) * relativeNormalVelocity, left(6, q) * relativeNormalVelocity;
                // fluxFace returns a negative physical flux increment.
                CHECK((-flux.col(q) - expected).norm() < 1e-11);
            }
        }
}

TEST_CASE("Viscous wall perturbations match prescribed turbulence ghost values")
{
    using Eval = EulerEvaluator<NS_2EQ>;
    const auto root = projectRoot();
    const auto path = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "rans_wall_jacobian",
                                          {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                           {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});
    EulerSolver<NS> fixture(g_mpi);
    fixture.ConfigureFromJson(path, true);
    fixture.ReadMeshAndInitialize();
    EulerEvaluatorSettings<NS_2EQ> settings(7);
    Eval eval(fixture.getMesh(), fixture.getVFV(), std::make_shared<BoundaryHandler<NS_2EQ>>(7), settings);
    eval.dWall[0].setConstant(1);
    Eval::TU state;
    state << 1.2, .3, .2, .1, 10, .4, 2;
    const Eval::TVec normal = Eval::TVec::UnitX();
    const Eval::TMat base = Eval::TMat::Identity();
    for (auto model : {RANSModel::RANS_KOSST, RANSModel::RANS_KOWilcox})
    {
        eval.settings.ransModel = model;
        for (int j : {5, 6})
        {
            Eval::TU direction = Eval::TU::Unit(j);
            auto perturbation = direction;
            const auto linear = eval.generateBoundaryValue(perturbation, state, 0, 0, -1,
                                                           normal, base, Geom::tPoint::Zero(), 0,
                                                           Geom::BC_ID_DEFAULT_WALL, false, 0, 1);
            constexpr real h = 1e-5;
            Eval::TU plus = state + h * direction, minus = state - h * direction;
            // Hold the cell mean (wall target, viscosity, distance) fixed.
            const auto rightPlus = eval.generateBoundaryValue(plus, state, 0, 0, -1,
                                                              normal, base, Geom::tPoint::Zero(), 0,
                                                              Geom::BC_ID_DEFAULT_WALL, false);
            const auto rightMinus = eval.generateBoundaryValue(minus, state, 0, 0, -1,
                                                               normal, base, Geom::tPoint::Zero(), 0,
                                                               Geom::BC_ID_DEFAULT_WALL, false);
            CHECK((linear - (rightPlus - rightMinus) / (2 * h)).norm() < 1e-9);
            CHECK(linear(j) == -1);
        }
    }
    // The same early-return branch is shared by SA. Keep its wall derivative
    // consistent too, holding fixUL=false as in the implicit matrix assembly.
    EulerEvaluatorSettings<NS_SA> saSettings(6);
    EulerEvaluator<NS_SA> saEval(fixture.getMesh(), fixture.getVFV(), std::make_shared<BoundaryHandler<NS_SA>>(6), saSettings);
    EulerEvaluator<NS_SA>::TU saState = state.head<6>(), direction = EulerEvaluator<NS_SA>::TU::Unit(5);
    auto perturbation = direction;
    const auto linear = saEval.generateBoundaryValue(perturbation, saState, 0, 0, -1,
                                                     normal, base, Geom::tPoint::Zero(), 0,
                                                     Geom::BC_ID_DEFAULT_WALL, false, 0, 1);
    constexpr real h = 1e-5;
    EulerEvaluator<NS_SA>::TU plus = saState + h * direction, minus = saState - h * direction;
    const auto rightPlus = saEval.generateBoundaryValue(plus, saState, 0, 0, -1, normal, base,
                                                        Geom::tPoint::Zero(), 0, Geom::BC_ID_DEFAULT_WALL, false);
    const auto rightMinus = saEval.generateBoundaryValue(minus, saState, 0, 0, -1, normal, base,
                                                         Geom::tPoint::Zero(), 0, Geom::BC_ID_DEFAULT_WALL, false);
    CHECK((linear - (rightPlus - rightMinus) / (2 * h)).norm() < 1e-9);
}

TEST_CASE("SA GG boundaries preserve means, face targets and traversal independence")
{
    using Eval = EulerEvaluator<NS_SA>;
    const auto root = projectRoot();
    const auto path = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "sa_gg_targets",
                                          {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}, {"subs2ndOrderGGScheme", 0}}},
                                           {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/Uniform_3x3_wall.cgns"}}}});
    EulerSolver<NS> fixture(g_mpi);
    fixture.ConfigureFromJson(path, true);
    fixture.ReadMeshAndInitialize();
    auto mesh = fixture.getMesh();
    auto vr = fixture.getVFV();
    REQUIRE_FALSE(mesh->isPeriodic);
    EulerEvaluatorSettings<NS_SA> settings(6);
    settings.useSourceGradFixGG = 1;
    Eval eval(mesh, vr, std::make_shared<BoundaryHandler<NS_SA>>(6), settings);
    ArrayDOFV<6> u, rhs;
    ArrayDOFV<1> beta, alpha;
    ArrayRECV<6> rec;
    vr->BuildUDof(u, 6);
    vr->BuildUDof(rhs, 6);
    vr->BuildUDof(beta, 1);
    vr->BuildUDof(alpha, 1);
    vr->BuildURec(rec, 6);
    for (DNDS::index iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
        rec[iCell].setZero();
    beta.setConstant(1);
    alpha.setConstant(1);
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        const auto center = vr->GetCellBary(iCell);
        u[iCell] << 1, .4, -.2, 0, 3.25, .1 + .02 * center(0) + .03 * center(1);
    }
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
    std::vector<Eval::TU> original;
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        original.emplace_back(u[iCell]);
        eval.gradUFix[iCell].setZero(3, 6);
    }
    std::vector<Eigen::Vector3d> expected(mesh->NumCell(), Eigen::Vector3d::Zero());
    DNDS::index wallCount = 0;
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        auto faces = mesh->CellFaces(iCell);
        for (DNDS::rowsize j = 0; j < faces.size(); ++j)
        {
            const auto iFace = faces[j];
            const auto other = mesh->CellFaceOther(iCell, iFace, j);
            const int side = mesh->CellIsFaceBack(iCell, iFace, j) ? 0 : 1;
            const Eigen::Vector3d normal = vr->GetFaceNormFromCell(iFace, iCell, side, -1) * (side ? -1 : 1);
            DNDS::real faceScalar = 0;
            if (other != UnInitIndex)
                faceScalar = .5 * (u[other](5) + u[iCell](5));
            else
            {
                REQUIRE(eval.pBCHandler->GetTypeFromID(mesh->GetFaceZone(iFace)) == BCWall);
                // VR deliberately returns the raw right value from a fresh
                // mutable trace; the mean and subsequent callbacks stay intact.
                Eval::TU left = u[iCell];
                const Eval::TVec unitNormal = normal.head<Eval::dim>();
                const auto rawRight = eval.generateBoundaryValue(left, u[iCell], iCell, iFace, -1,
                                                                 unitNormal, Geom::NormBuildLocalBaseV<Eval::dim>(unitNormal),
                                                                 vr->GetFaceQuadraturePPhys(iFace, -1), 0,
                                                                 mesh->GetFaceZone(iFace), true, 1);
                CHECK(left(5) == 0);
                CHECK(rawRight(5) == 0);
                CHECK((u[iCell] - original[iCell]).isZero(0));
                ++wallCount;
            }
            expected[iCell] += vr->GetFaceArea(iFace) / vr->GetCellVol(iCell) * normal * (faceScalar - u[iCell](5));
        }
        const auto forward = eval.EvaluateSourceGradientGG(u[iCell], u, iCell, 0);
        CHECK((forward.col(5) - expected[iCell]).norm() < 1e-12);
        std::vector<DNDS::index> reverseOrder(faces.begin(), faces.end());
        std::reverse(reverseOrder.begin(), reverseOrder.end());
        const auto evaluateInOrder = [&](const std::vector<DNDS::index> &order)
        {
            Eval::TDiffU gradient;
            gradient.setZero(3, 6);
            const auto canonical = mesh->CellFaces(iCell);
            for (const auto iFace : order)
            {
                const auto found = std::find(canonical.begin(), canonical.end(), iFace);
                REQUIRE(found != canonical.end());
                const DNDS::rowsize slot = DNDS::rowsize(found - canonical.begin());
                const auto other = mesh->CellFaceOther(iCell, iFace, slot);
                const int side = mesh->CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
                const Eval::TVec normal = (vr->GetFaceNormFromCell(iFace, iCell, side, -1) * (side ? -1 : 1)).head<Eval::dim>();
                Eval::TU right;
                if (other != UnInitIndex)
                    right = u[other]; // This fixture is nonperiodic.
                else
                    right = eval.generateBoundaryValueForGradient(u[iCell], iCell, iFace, -1, normal,
                                                                  vr->GetFaceQuadraturePPhys(iFace, -1), 0,
                                                                  mesh->GetFaceZone(iFace));
                gradient += .5 * vr->GetFaceArea(iFace) * normal * (right - u[iCell]).transpose();
            }
            return gradient / vr->GetCellVol(iCell);
        };
        const auto reversed = evaluateInOrder(reverseOrder);
        CHECK((forward - reversed).norm() < 1e-12);
        for (int sourceAll : {0, 1})
        {
            eval.settings.source2nd = sourceAll;
            eval.settings.ransSource2nd = !sourceAll;
            Eval::TU before = Eval::TU::Zero(), after = Eval::TU::Zero();
            Eval::TJacobianU jac;
            Eval::TDiffU unused = Eval::TDiffU::Zero();
            auto source = [&](auto &value)
            {
                eval.EvaluateCellSource(value, jac, u[iCell], unused, iCell, 0, SourceFilter::All, 1,
                                        true, OptionalRef(u), OptionalRef(rec), OptionalRef(rec), false, 0);
            };
            source(before);
            source(after);
            CHECK((before - after).norm() < 1e-11 * (1 + before.norm()));
        }
        CHECK((u[iCell] - original[iCell]).isZero(0));
    }
    MPI::AllreduceOneIndex(wallCount, MPI_SUM, g_mpi);
    CHECK(wallCount > 0);

    // Independent zero-wall target oracle for all direct-gradient schemes.
    auto reflected = [](const auto &, const auto &mean, auto, auto, auto, const auto &, const auto &, auto) -> Eval::TU
    {
        Eval::TU ghost = mean;
        ghost.template segment<3>(1) *= -1;
        ghost(5) *= -1;
        return ghost;
    };
    CFV::tUGrad<6, 2> actual, reference;
    vr->BuildUGrad(actual, 6);
    vr->BuildUGrad(reference, 6);
    for (int method : {1, 11, 2})
    {
        vr->DoReconstruction2ndGrad<6>(actual, u, eval.GetGradientBoundary(0), method);
        vr->DoReconstruction2ndGrad<6>(reference, u, reflected, method);
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            CHECK((actual[iCell] - reference[iCell]).norm() < 1e-12);
    }

    // A constant polynomial has zero base gradient. Its corrected source GG
    // must therefore equal the face-target oracle, including a full wall jump.
    eval.settings.ignoreSourceTerm = true;
    JacobianDiagBlock<6> sourceJac;
    sourceJac.SetModeAndInit(1, 6, u);
    eval.EvaluateRHS(rhs, sourceJac, u, rec, rec, beta, alpha, false, 0,
                     Eval::RHS_Ignore_Viscosity | Eval::RHS_Dont_Update_Integration | Eval::RHS_Dont_Record_Bud_Flux);
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        CHECK((eval.gradUFix[iCell].col(5) / vr->GetCellVol(iCell) - expected[iCell]).norm() < 1e-12);
        CHECK((u[iCell] - original[iCell]).isZero(0));
    }

    // Reproduce the affine mean field exactly. Interior polynomial traces
    // agree, leaving only wall jumps measured from the nonconstant trace.
    for (DNDS::index iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
    {
        actual[iCell].setZero();
        actual[iCell].col(5) << .02, .03;
    }
    vr->ConvertUGradToURec<6>(rec, actual);
    rec.trans.startPersistentPull();
    rec.trans.waitPersistentPull();
    eval.EvaluateRHS(rhs, sourceJac, u, rec, rec, beta, alpha, false, 0,
                     Eval::RHS_Ignore_Viscosity | Eval::RHS_Dont_Update_Integration | Eval::RHS_Dont_Record_Bud_Flux);
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        Eigen::Vector3d correction = Eigen::Vector3d::Zero();
        const auto faces = mesh->CellFaces(iCell);
        for (DNDS::rowsize j = 0; j < faces.size(); ++j)
        {
            const auto iFace = faces[j];
            if (mesh->CellFaceOther(iCell, iFace, j) != UnInitIndex)
                continue;
            const auto point = vr->GetFaceQuadraturePPhys(iFace, -1);
            const DNDS::real wallTrace = .1 + .02 * point(0) + .03 * point(1);
            correction -= vr->GetFaceArea(iFace) * vr->GetFaceNorm(iFace, -1) * wallTrace / vr->GetCellVol(iCell);
        }
        CHECK((eval.gradUFix[iCell].col(5) / vr->GetCellVol(iCell) - correction).norm() < 1e-12);
        CHECK((u[iCell] - original[iCell]).isZero(0));
    }
}

TEST_CASE("EulerP scalar GG reads neighbour cells and face-indexed boundary buffers")
{
    namespace EP = DNDS::EulerP;
    constexpr auto host = DeviceBackend::Host;
    const auto root = projectRoot();
    const auto path = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "eulerp_scalar_gg",
                                          {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                           {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/Uniform_3x3_wall.cgns"}}}});
    EulerSolver<NS> fixture(g_mpi);
    fixture.ConfigureFromJson(path, true);
    fixture.ReadMeshAndInitialize();
    auto mesh = fixture.getMesh();
    auto fv = std::static_pointer_cast<CFV::FiniteVolume>(fixture.getVFV());
    EP::TUDof flow, boundaryFlow;
    EP::TUGrad flowGradient;
    EP::TUScalar scalars[2], boundaryScalars[2];
    EP::TUScalarGrad gradients[2];
    fv->BuildUDof(flow, 5);
    fv->BuildUDof(boundaryFlow, 5, true, true, Geom::MeshLoc::Face);
    fv->BuildUGradD<5, 3>(flowGradient, 5);
    for (int j = 0; j < 2; ++j)
    {
        fv->BuildUDof(scalars[j], 1);
        fv->BuildUDof(boundaryScalars[j], 1, true, true, Geom::MeshLoc::Face);
        fv->BuildUGradD<1, 3>(gradients[j], 1);
    }
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        const auto x = fv->GetCellBary(iCell);
        scalars[0][iCell](0) = 1 + x(0) * x(0) + .3 * x(1);
        scalars[1][iCell](0) = -2 + .2 * x(0) - x(1) * x(1);
        flow[iCell].setConstant(1);
        flow[iCell](0) = scalars[0][iCell](0);
        flow[iCell](1) = scalars[1][iCell](0);
    }
    flow.trans.startPersistentPull();
    flow.trans.waitPersistentPull();
    for (auto &scalar : scalars)
    {
        scalar.trans.startPersistentPull();
        scalar.trans.waitPersistentPull();
    }
    for (DNDS::index iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
    {
        boundaryScalars[0][iFace](0) = 100 + .7 * iFace;
        boundaryScalars[1][iFace](0) = -30 - .9 * iFace;
        boundaryFlow[iFace].setConstant(1);
        boundaryFlow[iFace](0) = boundaryScalars[0][iFace](0);
        boundaryFlow[iFace](1) = boundaryScalars[1][iFace](0);
    }
    using ScalarView = EP::TUScalar::t_deviceView<host>;
    using GradientView = EP::TUScalarGrad::t_deviceView<host>;
    deviceViewVector<ScalarView, host> scalarViews(2, [&](int j)
                                                   { return scalars[j].deviceView<host>(); });
    deviceViewVector<ScalarView, host> boundaryViews(2, [&](int j)
                                                     { return boundaryScalars[j].deviceView<host>(); });
    deviceViewVector<GradientView, host> gradientViews(2, [&](int j)
                                                       { return gradients[j].deviceView<host>(); });
    EP::Evaluator_impl<host>::RecGradient_Arg::Portable args{};
    args.u = flow.deviceView<host>();
    args.uGrad = flowGradient.deviceView<host>();
    args.faceBCBuffer = boundaryFlow.deviceView<host>();
    args.uScalar = scalarViews.deviceView();
    args.uScalarGrad = gradientViews.deviceView();
    args.faceBCScalarBuffer = boundaryViews.deviceView();
    auto fvRead = fv->AcquireDeviceRead<host>();
    EP::EvaluatorDeviceView<host> view(fvRead.View(), EP::BCHandlerDeviceView<host>({}), EP::PhysicsDeviceView<host>{{}, {}});
    DNDS::index interiorCount = 0, mismatchedFaceCount = 0;
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        EP::RecGradient_GGRec_Kernel_GG<host>(view, args, iCell, mesh->NumCell(), 7, 2);
        for (int j = 0; j < 2; ++j)
        {
            Eigen::Vector3d expected = Eigen::Vector3d::Zero();
            auto faces = mesh->CellFaces(iCell);
            for (DNDS::rowsize k = 0; k < faces.size(); ++k)
            {
                const auto iFace = faces[k];
                const auto other = mesh->CellFaceOther(iCell, iFace, k);
                const int side = mesh->CellIsFaceBack(iCell, iFace, k) ? 0 : 1;
                const Eigen::Vector3d normal = fv->GetFaceNormFromCell(iFace, iCell, side, -1) * (side ? -1 : 1);
                const DNDS::real right = other == UnInitIndex ? boundaryScalars[j][iFace](0) : scalars[j][other](0);
                expected += .5 * fv->GetFaceArea(iFace) / fv->GetCellVol(iCell) * normal * (right - scalars[j][iCell](0));
                interiorCount += other != UnInitIndex;
                mismatchedFaceCount += other == UnInitIndex && iFace != iCell;
            }
            CHECK((gradients[j][iCell] - expected).norm() < 1e-11);
            CHECK((flowGradient[iCell].col(j) - expected).norm() < 1e-11);
        }
    }
    MPI::AllreduceOneIndex(interiorCount, MPI_SUM, g_mpi);
    MPI::AllreduceOneIndex(mismatchedFaceCount, MPI_SUM, g_mpi);
    CHECK(interiorCount > 0);
    CHECK(mismatchedFaceCount > 0);
}

template <EulerModel model>
static void checkReconstructionPPSeparation()
{
    using Eval = EulerEvaluator<model>;
    constexpr int nVars = Eval::nVarsFixed;
    const auto root = projectRoot();
    const auto path = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "rans_pp_" + std::to_string(model),
                                          {{"vfvSettings", {{"maxOrder", 3}, {"intOrder", 5}}},
                                           {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});
    EulerSolver<NS> fixture(g_mpi);
    fixture.ConfigureFromJson(path, true);
    fixture.ReadMeshAndInitialize();
    auto vr = fixture.getVFV();
    auto mesh = fixture.getMesh();
    EulerEvaluatorSettings<model> settings(nVars);
    EulerEvaluatorSettings<NS> flowSettings(5);
    Eval eval(mesh, vr, std::make_shared<BoundaryHandler<model>>(nVars), settings);
    EulerEvaluator<NS> flowEval(mesh, vr, std::make_shared<BoundaryHandler<NS>>(5), flowSettings);
    ArrayDOFV<nVars> mean;
    ArrayDOFV<5> flowMean;
    ArrayRECV<nVars> rec;
    ArrayRECV<5> flowRec;
    ArrayDOFV<1> beta, flowBeta;
    vr->BuildUDof(mean, nVars);
    vr->BuildUDof(flowMean, 5);
    vr->BuildURec(rec, nVars);
    vr->BuildURec(flowRec, 5);
    vr->BuildUDof(beta, 1);
    vr->BuildUDof(flowBeta, 1);
    for (bool compressToMean : {true, false})
        for (int scenario = 0; scenario < 3; ++scenario)
        {
            // 0: tiny constant scalar; 1: negative scalar extrapolation;
            // 2: flow needs fallback, but admissible RANS modes must survive.
            CAPTURE(model);
            CAPTURE(compressToMean);
            CAPTURE(scenario);
            std::vector<Eigen::MatrixXd> original;
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                mean[iCell].setOnes();
                mean[iCell].template head<5>() << 1, .1, .2, 0, 2.6;
                mean[iCell](5) = scenario == 2 ? 1 : 1e-14;
                flowMean[iCell] = mean[iCell].template head<5>();
                rec[iCell].setZero();
                rec[iCell](0, 0) = scenario == 2 ? 100 : .01;
                rec[iCell](rec[iCell].rows() - 1, 0) = .002;
                if (scenario != 0)
                    for (int v = 5; v < nVars; ++v)
                    {
                        rec[iCell](0, v) = .001;
                        rec[iCell](rec[iCell].rows() - 1, v) = .0001;
                    }
                flowRec[iCell] = rec[iCell].template leftCols<5>();
                original.emplace_back(rec[iCell]);
            }
            DNDS::index nLim = 0, nFlowLim = 0;
            real minBeta = 1, minFlowBeta = 1;
            const int flag = Eval::EvaluateURecBeta_LOCAL_STATS |
                             (compressToMean ? Eval::EvaluateURecBeta_COMPRESS_TO_MEAN : Eval::EvaluateURecBeta_DEFAULT);
            eval.EvaluateURecBeta(mean, rec, beta, nLim, minBeta, flag);
            flowEval.EvaluateURecBeta(flowMean, flowRec, flowBeta, nFlowLim, minFlowBeta, flag);
            CHECK(nLim == nFlowLim);
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                CHECK((rec[iCell].template leftCols<5>() - flowRec[iCell]).norm() < 1e-12);
                CHECK(beta[iCell](0) == doctest::Approx(flowBeta[iCell](0)).epsilon(1e-12));
                CHECK((mean[iCell].template head<5>() - flowMean[iCell]).norm() == 0);
                CHECK(mean[iCell](5) == (scenario == 2 ? 1 : 1e-14));
                if (scenario != 2)
                {
                    CHECK(beta[iCell](0) == 1);
                    CHECK((rec[iCell].template leftCols<5>() - original[iCell].leftCols(5)).norm() == 0);
                }
                else
                    CHECK(beta[iCell](0) < 1);
                constexpr bool scalarPP = Eval::Traits::has2EQ
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
                                          || Eval::Traits::hasSA
#endif
                    ;
                if (scenario != 1 || !scalarPP)
                    CHECK((rec[iCell].rightCols(nVars - 5) - original[iCell].rightCols(nVars - 5)).norm() == 0);
                if constexpr (scalarPP)
                {
                    auto checkPoint = [&](const auto &basis)
                    {
                        const auto value = (mean[iCell] + (basis * rec[iCell]).transpose()).eval();
                        for (int v = 5; v < nVars; ++v)
                            CHECK(value(v) >= 0);
                    };
                    for (int q = 0; q < vr->GetCellQuad(iCell).GetNumPoints(); ++q)
                        checkPoint(vr->GetIntPointDiffBaseValue(iCell, -1, -1, q, 0, 1));
                    const auto faces = mesh->CellFaces(iCell);
                    for (DNDS::rowsize j = 0; j < faces.size(); ++j)
                    {
                        const auto face = faces[j];
                        const int side = mesh->CellIsFaceBack(iCell, face, j) ? 0 : 1;
                        for (int q = 0; q < vr->GetFaceQuad(face).GetNumPoints(); ++q)
                            checkPoint(vr->GetIntPointDiffBaseValue(iCell, face, side, q, 0, 1));
                    }
                }
            }
        }
}

TEST_CASE("Euler reconstruction PP keeps RANS out of flow beta and order fallback")
{
    checkReconstructionPPSeparation<NS_SA>();
    checkReconstructionPPSeparation<NS_2EQ>();
}

TEST_CASE("Euler gradients: EvaluateDt prepares limited buffers and direct FV explicitly reuses them")
{
    using Eval = EulerEvaluator<NS>;
    const auto root = projectRoot();
    const auto path = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "gradient_reuse",
                                          {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                           {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/UniformSquare_10.cgns"}}}});
    EulerSolver<NS> solver(g_mpi);
    solver.ConfigureFromJson(path, true);
    solver.ReadMeshAndInitialize();
    auto &eval = solver.getEval();
    auto &u = solver.getU();
    auto mesh = solver.getMesh();
    auto vr = solver.getVFV();
    // The native mesh facade is immutable here. Boundary-time provenance has a
    // separate adapter test; this case isolates gradient-buffer preparation and
    // explicit reuse without rewriting face metadata after solver setup.
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        const auto x = vr->GetCellQuadraturePPhys(iCell, -1);
        const real rho = x(0) < .5 ? 1 : 2;
        u[iCell] << rho, .2 * rho, .1 * rho, 0, 2.5 + .025 * rho;
    }
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
    ArrayGRADV<5, 2> unlimited, limited;
    vr->BuildUGrad(unlimited, 5);
    vr->BuildUGrad(limited, 5);
    ArrayDOFV<5> expectedRHS, reusedRHS;
    ArrayDOFV<1> dt, alpha;
    vr->BuildUDof(expectedRHS, 5);
    vr->BuildUDof(reusedRHS, 5);
    vr->BuildUDof(dt, 1);
    vr->BuildUDof(alpha, 1);
    alpha.setConstant(1);
    solver.getBetaPP().setConstant(1);
    auto compare = [](const auto &a, const auto &b)
    {
        REQUIRE(a.Size() == b.Size());
        for (DNDS::index iCell = 0; iCell < a.Size(); ++iCell)
            CHECK((a[iCell] - b[iCell]).norm() <= 1e-12 * (1 + b[iCell].norm()));
    };
    const real stageTime = -4.5;
    real minDt = 0;
    const uint64_t both = Eval::RHS_Direct_2nd_Rec_already_have_uGradBufNoLim |
                          Eval::RHS_Direct_2nd_Rec_already_have_uGradBuf;
    for (bool shockLimiter : {false, true})
    {
        CAPTURE(shockLimiter);
        CFV::VariationalReconstruction<2>::TFBoundary<5> boundary =
            [&](const Eval::TU &ul, const Eval::TU &mean, DNDS::index iCell, DNDS::index iFace, int q,
                const Geom::tPoint &n, const Geom::tPoint &x, Geom::t_index zone) -> Eval::TU
        {
            bool compressed = false;
            Eval::TU fixed = eval.CompressRecPart(mean, ul - mean, compressed);
            const auto basis = Geom::NormBuildLocalBaseV<3>(n);
            return eval.generateBoundaryValue(fixed, mean, iCell, iFace, q, n, basis, x, stageTime, zone, true, 1);
        };
        // Independent operator sequence is the oracle, including owned/ghost rows.
        vr->DoReconstruction2ndGrad(unlimited, u, boundary, eval.settings.direct2ndRecMethod);
        eval.LimiterUGrad(u, unlimited, limited, shockLimiter ? Eval::LIMITER_UGRAD_No_Flags : Eval::LIMITER_UGRAD_Disable_Shock_Limiter);
        unlimited.trans.startPersistentPull();
        limited.trans.startPersistentPull();
        unlimited.trans.waitPersistentPull();
        limited.trans.waitPersistentPull();
        eval.uGradBufNoLim.setConstant(real(17));
        eval.uGradBuf.setConstant(real(-23));
        const auto dtFlags = shockLimiter ? Eval::DT_No_Flags : Eval::DT_Disable_Gradient_Shock_Limiter;
        eval.EvaluateDt(dt, u, solver.getURec(), 1, minDt, 1e100, true, stageTime, dtFlags);
        compare(eval.uGradBufNoLim, unlimited);
        compare(eval.uGradBuf, limited);
        if (shockLimiter)
        {
            real localDifference = 0, globalDifference = 0;
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                localDifference += (limited[iCell] - unlimited[iCell]).squaredNorm();
            MPI::Allreduce(&localDifference, &globalDifference, 1, DNDS_MPI_REAL, MPI_SUM, g_mpi.comm);
            CHECK(globalDifference > 1e-6); // The fixture really exercises limiting.
        }
        const auto rhsFlags = Eval::RHS_Direct_2nd_Rec | Eval::RHS_Dont_Update_Integration | Eval::RHS_Dont_Record_Bud_Flux |
                              (shockLimiter ? Eval::RHS_Direct_2nd_Rec_use_limiter : Eval::RHS_No_Flags);
        auto rhs = [&](auto &output, real t, uint64_t reuse, bool firstOrderConvection = false)
        {
            eval.EvaluateRHS(output, solver.getJSource(), u, solver.getURec(), solver.getURec(),
                             solver.getBetaPP(), alpha, false, t, rhsFlags | reuse | (firstOrderConvection ? Eval::RHS_Direct_2nd_Rec_1st_Conv : Eval::RHS_No_Flags));
        };
        rhs(reusedRHS, stageTime, both);
        compare(eval.uGradBufNoLim, unlimited);
        compare(eval.uGradBuf, limited);
        rhs(expectedRHS, stageTime, 0);
        compare(reusedRHS, expectedRHS);
        eval.EvaluateDt(dt, u, solver.getURec(), 1, minDt, 1e100, true, stageTime, dtFlags);
        rhs(reusedRHS, stageTime, Eval::RHS_Direct_2nd_Rec_already_have_uGradBufNoLim);
        compare(reusedRHS, expectedRHS);
        // Omitting reuse must recompute the gradients even when U is unchanged.
        // This native-mesh fixture has a static boundary: boundary-time
        // provenance is covered separately because the published native mesh
        // facade deliberately does not permit relabelling face zones in place.
        eval.EvaluateDt(dt, u, solver.getURec(), 1, minDt, 1e100, true, stageTime, dtFlags);
        rhs(reusedRHS, -4, 0);
        compare(eval.uGradBufNoLim, unlimited);
        rhs(expectedRHS, -4, 0);
        compare(reusedRHS, expectedRHS);
        // The pMG first-order-convection branch still copies unlimited gradients.
        rhs(expectedRHS, stageTime, 0, true);
        eval.EvaluateDt(dt, u, solver.getURec(), 1, minDt, 1e100, true, stageTime, dtFlags);
        rhs(reusedRHS, stageTime, both, true);
        compare(eval.uGradBuf, unlimited);
        compare(reusedRHS, expectedRHS);
    }
}

static LVRCells runLVRFixture(const std::string &cfgPath, const MPIInfo &mpi,
                              const std::string &state, O2ReferenceLimiter referenceLimiter, int extraShockLayers,
                              uint64_t reuseFlags = 0)
{
    EulerSolver<NS> solver(mpi);
    solver.ConfigureFromJson(cfgPath, true, "",
                             {"/limiterControl/limitedVR/alphaMax", "/limiterControl/limitedVR/contactAlphaMax", "/limiterControl/limitedVR/preset"},
                             {"0.5", "0.5", "v3"});
    solver.ReadMeshAndInitialize();
    REQUIRE(solver.hasLimitedVRStorage());
    auto vfv = solver.getVFV();
    auto mesh = solver.getMesh();
    auto &u = solver.getU();
    auto &target = solver.getURecO2();
    auto data = solver.getOutputOverlapDataRefs();
    auto settings = solver.getConfiguration().limiterControl.limitedVR;
    CHECK(settings.gateMode == LimitedVRGateMode::PressureExtremum);
    CHECK(settings.contactExtremumHaloLayers == 2);
    CHECK(settings.alphaMax == 0.5);
    settings.o2ReferenceLimiter = referenceLimiter;
    settings.gateHaloLayers = extraShockLayers;
    settings.weakAlphaMax = state == "shock" ? 0.5 : 0;
    using TU = EulerEvaluator<NS>::TU;
    for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
    {
        const auto x = vfv->GetCellQuadraturePPhys(i, -1);
        DNDS::real rho = 1, p = 1, vx = 0.2;
        // Variation in y makes this a genuinely multidimensional velocity field.
        DNDS::real vy = state == "uniform" ? 0.1 : 0.07 * std::sin(2 * M_PI * x(1) / 10);
        if (state == "shock")
        {
            p = x(0) < 5 ? 3 : 1;
            vx = x(0) < 5 ? 0.8 : 0;
            if (x(0) > 3 && x(0) < 4)
                p = 3.08; // a supported pressure-extremum stripe
        }
        if (state == "contact")
        {
            rho = x(0) < 5 ? 10 : 1;
            if (x(0) > 5 && x(0) < 6)
                rho = 11; // a supported density extremum
        }
        u[i] << rho, rho * vx, rho * vy, 0,
            p / 0.4 + 0.5 * rho * (vx * vx + vy * vy);
    }
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
    CFV::tScalarPair indicator;
    vfv->BuildScalar(indicator);
    CFV::VariationalReconstruction<2>::TFBoundary<5> boundary =
        [](const TU &ul, const TU &, DNDS::index, DNDS::index, int,
           const Geom::tPoint &, const Geom::tPoint &, Geom::t_index)
    { return ul; };
    if (reuseFlags)
    {
        ArrayDOFV<1> dt;
        vfv->BuildUDof(dt, 1);
        real minDt = 0;
        solver.getEval().EvaluateDt(dt, u, solver.getURec(), 1, minDt, 1e100, true, 0);
    }
    solver.getEval().BuildLimitedVRO2(
        u, target, solver.getURecLimited(), solver.getURecNew(), indicator,
        data.lvrO2Beta, data.lvrAlpha, data.lvrBranchAlpha, data.lvrDensityJump,
        data.lvrPressureJump, data.lvrCompression, settings, true, true, boundary, {}, reuseFlags);
    LVRCells cells;
    // Total alpha, shock, contact, weak, and contact halo beyond the base gate.
    std::array<DNDS::real, 5> totals{};
    for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
    {
        const auto x = vfv->GetCellQuadraturePPhys(i, -1);
        CHECK(data.lvrAlpha[i](0) >= 0);
        CHECK(data.lvrAlpha[i](0) <= 0.5);
        CHECK(data.lvrAlpha[i](0) == doctest::Approx(
                                         std::max({data.lvrBranchAlpha[i](0), data.lvrBranchAlpha[i](1), data.lvrBranchAlpha[i](3)})));
        CHECK(data.lvrO2Beta[i](0) > 0);
        CHECK(data.lvrO2Beta[i](0) <= 1);
        CHECK(target[i].allFinite());
        CHECK(target[i].bottomRows(target[i].rows() - 2).norm() == 0);
        totals[0] += data.lvrAlpha[i](0);
        totals[1] += data.lvrBranchAlpha[i](0);
        totals[2] += data.lvrBranchAlpha[i](1);
        totals[3] += data.lvrBranchAlpha[i](3);
        if (data.lvrDensityJump[i](0) < settings.contactDensityJumpStart && data.lvrBranchAlpha[i](1) > 0)
            totals[4] += 1;
        LVRCellValues values{data.lvrAlpha[i](0), data.lvrBranchAlpha[i](0),
                             data.lvrBranchAlpha[i](1), data.lvrBranchAlpha[i](2),
                             data.lvrBranchAlpha[i](3), data.lvrO2Beta[i](0)};
        // Compare physical gradients rather than basis coefficients, whose
        // local basis orientation is an implementation detail.
        const Eigen::Matrix<DNDS::real, 2, 2> d1 =
            vfv->GetIntPointDiffBaseValue(i, -1, -1, -1,
                                          std::array<int, 2>{1, 2}, 3)(EigenAll, Eigen::seqN(0, 2));
        Eigen::Matrix<DNDS::real, 2, 5> gradient = d1 * target[i].topRows(2);
        values.insert(values.end(), gradient.data(), gradient.data() + gradient.size());
        cells[{std::llround(x(0) * 1e8), std::llround(x(1) * 1e8)}] = values;
    }
    auto globalTotals = totals;
    MPI::Allreduce(totals.data(), globalTotals.data(), 5, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
    if (state == "uniform")
        CHECK(globalTotals[0] == doctest::Approx(0).epsilon(1e-10).scale(1));
    if (state == "shock")
    {
        CHECK(globalTotals[1] > 0);
        CHECK(globalTotals[3] > 0);
    }
    if (state == "contact")
    {
        CHECK(globalTotals[2] > 0);
        CHECK(globalTotals[4] > 0);
    }

    if (state == "uniform" && referenceLimiter == O2ReferenceLimiter::Barth)
    {
        // Force a negative O2 face state, then exercise the mean-based PP path.
        for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
            target[i](0, 0) = 100;
        DNDS::index nLimited = 0;
        DNDS::real minBeta = 1;
        solver.getEval().EvaluateURecBeta(u, target, data.lvrO2Beta, nLimited, minBeta,
                                          EulerEvaluator<NS>::EvaluateURecBeta_COMPRESS_TO_MEAN |
                                              EulerEvaluator<NS>::EvaluateURecBeta_LOCAL_STATS);
        if (mesh->NumCell())
        {
            CHECK(nLimited > 0);
            CHECK(minBeta < 1);
        }
        for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
        {
            const auto faces = mesh->CellFaces(i);
            for (DNDS::rowsize j = 0; j < faces.size(); ++j)
            {
                const auto face = faces[j];
                const int side = mesh->CellIsFaceBack(i, face, j) ? 0 : 1;
                const auto quad = vfv->GetFaceQuad(face);
                for (int q = 0; q < quad.GetNumPoints(); ++q)
                {
                    TU uq = u[i] + (vfv->GetIntPointDiffBaseValue(i, face, side, q, 0, 1) * target[i]).transpose();
                    CHECK(uq(0) > 0);
                    CHECK(uq(4) - 0.5 * uq.segment<3>(1).squaredNorm() / uq(0) > 0);
                }
            }
        }
        settings.alphaMax = settings.contactAlphaMax = settings.weakAlphaMax = 0;
        solver.getEval().BuildLimitedVRO2(
            u, target, solver.getURecLimited(), solver.getURecNew(), indicator,
            data.lvrO2Beta, data.lvrAlpha, data.lvrBranchAlpha, data.lvrDensityJump,
            data.lvrPressureJump, data.lvrCompression, settings, true, true, boundary);
        for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
        {
            CHECK(data.lvrAlpha[i](0) == 0);
            CHECK(data.lvrBranchAlpha[i].isZero());
            CHECK(target[i].isZero());
            CHECK(data.lvrO2Beta[i](0) == 1);
        }
    }
    if (state == "uniform")
    {
        // Inactive output requests must remain defined without any LVR layouts.
        ArrayDOFV<1> emptyAlpha, emptyJump, emptyPressure, emptyCompression, emptyBeta;
        ArrayDOFV<4> emptyBranches;
        ArrayDOFV<1> unused;
        OutputPicker picker;
        solver.getEval().InitializeOutputPicker(picker,
                                                {u, solver.getURec(), solver.getBetaPP(), solver.getAlphaPP(),
                                                 emptyAlpha, emptyBranches, emptyJump, emptyPressure, emptyCompression, emptyBeta},
                                                {unused, unused, unused, unused, unused});
        auto fields = picker.getSubsetList({"lvrAlpha", "lvrWeakAlpha", "lvrO2Beta"});
        if (mesh->NumCell())
        {
            CHECK(std::get<1>(fields[0])(0) == 0);
            CHECK(std::get<1>(fields[1])(0) == 0);
            CHECK(std::get<1>(fields[2])(0) == 1);
        }
    }
    return cells;
}

TEST_CASE("LVR target and branch halos agree across 2-D MPI partitions")
{
    const std::string root = projectRoot();
    const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "lvr_2d",
                                         {{"vfvSettings", {{"maxOrder", 3}, {"intOrder", 5}, {"subs2ndOrder", 1}, {"cacheDiffBase", true}}},
                                          {"limiterControl", {{"useLimiter", true}, {"limiterProcedure", "LVR"}}},
                                          {"implicitReconstructionControl", {{"recLinearScheme", 0}}},
                                          {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});
    // The fixture also exercises CLI preset expansion and explicit overrides.
    for (const std::string state : {"uniform", "shock", "contact"})
        for (const auto limiter : {O2ReferenceLimiter::Barth, O2ReferenceLimiter::WBAP})
            for (const int extraShockLayers : {0, 2})
            {
                CAPTURE(state);
                CAPTURE(nlohmann::ordered_json(limiter).dump());
                CAPTURE(extraShockLayers);
                // Exercise both the three-round H2 schedule and extra shock halos.
                // Compare cellwise against an independently built serial evaluator.
                auto serial = runLVRFixture(cfg, MPIInfo(MPI_COMM_SELF), state, limiter, extraShockLayers);
                auto parallel = runLVRFixture(cfg, g_mpi, state, limiter, extraShockLayers);
                for (const auto &[key, values] : parallel)
                {
                    REQUIRE(serial.count(key) == 1);
                    const auto &expected = serial.at(key);
                    REQUIRE(values.size() == expected.size());
                    for (size_t i = 0; i < values.size(); ++i)
                        CHECK(values[i] == doctest::Approx(expected[i]).epsilon(2e-10).scale(1));
                }
            }
}

TEST_CASE("LVR explicit gradient reuse matches a fresh Barth or WBAP reference")
{
    using Eval = EulerEvaluator<NS>;
    const auto root = projectRoot();
    const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "lvr_gradient_reuse",
                                         {{"vfvSettings", {{"maxOrder", 3}, {"intOrder", 5}, {"subs2ndOrder", 1}}},
                                          {"limiterControl", {{"useLimiter", true}, {"limiterProcedure", "LVR"}}},
                                          {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});
    for (auto limiter : {O2ReferenceLimiter::Barth, O2ReferenceLimiter::WBAP})
    {
        const auto expected = runLVRFixture(cfg, g_mpi, "shock", limiter, 2);
        for (auto flags : {Eval::LVR_Already_Have_UGradBufNoLim,
                           Eval::LVR_Already_Have_UGradBufNoLim | Eval::LVR_Already_Have_UGradBuf})
        {
            const auto reused = runLVRFixture(cfg, g_mpi, "shock", limiter, 2, flags);
            REQUIRE(reused.size() == expected.size());
            for (const auto &[key, values] : reused)
            {
                const auto &reference = expected.at(key);
                REQUIRE(values.size() == reference.size());
                for (DNDS::index j = 0; j < static_cast<DNDS::index>(values.size()); ++j)
                    CHECK(values[j] == doctest::Approx(reference[j]).epsilon(2e-10).scale(1));
            }
        }
    }
}

TEST_CASE("EulerSolver static reconstruction mask owns selectors and setup storage")
{
    using json = nlohmann::ordered_json;
    using Control = EulerSolver<NS>::Configuration::ImplicitReconstructionControl;
    CHECK(Control::schema().at("properties").contains("reconstructionMask"));
    CHECK_FALSE(CFV::VRSettings::schema().at("properties").contains("reconstructionMask"));
    const auto root = projectRoot();
    const std::vector<std::pair<json, int>> selectors{
        {json::object(), 0},
        {{{"expression", "x[0] < 0"}}, 3},
        {{{"boundaryNames", {"WALL"}}}, 8},
        {{{"boundaryIDs", {Geom::BC_ID_DEFAULT_WALL}}}, 8},
        {{{"expression", "x[0] > 0 and x[0] < 1 and x[1] > 0 and x[1] < 1"}, {"boundaryNames", {"WALL"}}}, 9},
        {{{"boundaryNames", {"misspelled-outlet"}}}, -1},
        {{{"boundaryIDs", {-1}}}, -1},
        {{{"expression", "unknown_variable > 0"}}, -1},
        {{{"expression", "sqrt(x[0])"}}, -1}};
    int testID = 0;
    for (const auto &[selector, expectedCount] : selectors)
    {
        auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "mask_setup_" + std::to_string(testID++),
                                       {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                        {"implicitReconstructionControl", {{"reconstructionMask", selector}}},
                                        {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/Uniform_3x3_wall.cgns"}, {"outAtPointData", false}, {"outPltMode", 1}}}});
        EulerSolver<NS> solver(g_mpi);
        solver.ConfigureFromJson(cfg, true);
        if (expectedCount < 0)
        {
            CHECK_THROWS(solver.ReadMeshAndInitialize());
            continue;
        }
        solver.ReadMeshAndInitialize();
        const auto &mask = solver.getReconstructionMask();
        CHECK(bool(mask.father) == (expectedCount != 0));
        if (!mask.father)
            continue;
        DNDS::index count = 0, countAll = 0;
        for (DNDS::index i = 0; i < solver.getMesh()->NumCell(); ++i)
            count += mask[i](0) != 0;
        MPI::Allreduce(&count, &countAll, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(countAll == expectedCount);
        for (DNDS::index i = 0; i < mask.Size(); ++i)
            CHECK((mask[i](0) == 0 || mask[i](0) == 1));
    }
}

TEST_CASE("EulerSolver static reconstruction mask builds a 3D coordinate selection" * doctest::test_suite("large-mesh"))
{
    const auto root = projectRoot();
    auto cfg = writeTempConfig<NS_3D>(root + "/cases/euler3D/euler3D_config_Box.json", "mask_setup_3d",
                                      {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}, {"cacheDiffBase", false}}},
                                       {"implicitReconstructionControl", {{"reconstructionMask", {{"expression", "x[0] > 0.6 and x[2] < 0.4"}}}}},
                                       {"dataIOControl", {{"meshFile", root + "/data/mesh/Uniform32_3D_Periodic.cgns"}}}});
    EulerSolver<NS_3D> solver(g_mpi);
    solver.ConfigureFromJson(cfg, true);
    solver.ReadMeshAndInitialize();
    auto vr = solver.getVFV();
    const auto &mask = solver.getReconstructionMask();
    REQUIRE(mask.father);
    for (DNDS::index i = 0; i < mask.Size(); ++i)
    {
        auto p = vr->GetCellBary(i);
        CHECK(bool(mask[i](0)) == (p(0) > 0.6 && p(2) < 0.4));
    }
}

TEST_CASE("EulerSolver static reconstruction mask survives limiter paths and time updates")
{
    const auto root = projectRoot();
    // A small periodic fixture exercises the production RHS callback. With every
    // cell masked, all limiter paths must produce the same first-order advance.
    std::vector<Eigen::VectorXd> reference;
    for (const std::string method : {"3WBAP", "CWBAP", "LVR", "directGG"})
    {
        const bool direct = method == "directGG";
        const std::string tag = "static_mask_" + method + "_np" + std::to_string(g_mpi.size);
        auto path = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", tag,
                                        {{"timeMarchControl", {{"odeCode", 2}, {"dtImplicit", 1e-3}, {"useDtPPLimit", false}, {"useRestart", false}, {"tEnd", 1}}},
                                         {"convergenceControl", {{"nTimeStepInternal", 2}}},
                                         {"restartState", {{"iStep", 0}, {"iStepInternal", 0}}},
                                         {"vfvSettings", {{"maxOrder", 3}, {"intOrder", 5}}},
                                         {"bcSettings", {{{"name", "UnusedCustomOutlet"}, {"type", "BCOut"}, {"value", {1, 0, 0, 0, 3}}}}},
                                         {"implicitReconstructionControl", {{"useExplicit", direct}, {"recLinearScheme", 0}, {"nInternalRecStep", 2}, {"reconstructionMask", {{"expression", "1"}, {"boundaryNames", {"UnusedCustomOutlet"}}}}}},
                                         {"limiterControl", {{"useLimiter", true}, {"limiterProcedure", method == "LVR" ? "LVR" : method == "CWBAP" ? "CWBAP"
                                                                                                                                                    : "3WBAP"}}},
                                         {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}, {"outAtPointData", false}, {"outPltMode", 1}, {"outPltName", (std::filesystem::temp_directory_path() / "dndsr_euler_test" / tag).string()}}}});
        EulerSolver<NS> solver(g_mpi);
        solver.ConfigureFromJson(path, true);
        solver.ReadMeshAndInitialize();
        solver.getConfiguration().timeMarchControl.nTimeStep = 3;
        auto vr = solver.getVFV();
        REQUIRE(solver.getReconstructionMask().father);
        solver.RunImplicitEuler();
        auto &rec = solver.getURec();
        auto &limited = solver.getURecLimited();
        auto &u = solver.getU();
        for (DNDS::index i = 0; i < rec.Size(); ++i)
        {
            CHECK(solver.getReconstructionMask()[i](0) == 1);
            CHECK(rec[i].isZero(0));
            CHECK(limited[i].isZero(0));
        }
        if (reference.empty())
            for (DNDS::index i = 0; i < u.Size(); ++i)
                reference.emplace_back(u[i]);
        else
        {
            REQUIRE(reference.size() == size_t(u.Size()));
            for (DNDS::index i = 0; i < u.Size(); ++i)
                CHECK((u[i] - reference[i]).norm() < 1e-12);
        }
        OutputPicker picker;
        ArrayDOFV<1> unused;
        solver.getEval().InitializeOutputPicker(picker, solver.getOutputOverlapDataRefs(),
                                                {unused, unused, unused, unused, unused});
        auto field = picker.getSubsetList({"reconstructionMask"});
        for (DNDS::index i = 0; i < solver.getMesh()->NumCell(); ++i)
            CHECK(std::get<1>(field[0])(i) == 1);
    }
}

static std::vector<double> OutputDataset(const std::string &fileName, const char *datasetName)
{
    auto file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    DNDS_check_throw_info(file >= 0, "Euler output file missing");
    auto dataset = H5Dopen2(file, datasetName, H5P_DEFAULT);
    DNDS_check_throw_info(dataset >= 0, "Euler output dataset missing");
    auto space = H5Dget_space(dataset);
    auto count = H5Sget_simple_extent_npoints(space);
    DNDS_check_throw_info(count > 0, "Euler output dataset empty");
    std::vector<double> result(static_cast<size_t>(count));
    auto error = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());
    H5Sclose(space);
    H5Dclose(dataset);
    H5Fclose(file);
    DNDS_check_throw_info(error >= 0, "Euler output dataset read failed");
    return result;
}

TEST_CASE("Euler distributed volume output reads native authority after detached facade metadata edits")
{
    const auto root = projectRoot();
    const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "native_output",
                                         {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                          {"limiterControl", {{"useLimiter", false}}},
                                          {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}, {"outPltMode", 1}, {"outBndData", false}, {"outVolumeData", true}, {"outPltTecplotFormat", false}, {"outPltVTKFormat", false}, {"outPltVTKHDFFormat", true}, {"outAtCellData", true}, {"outAtPointData", true}, {"allowAsyncPrintData", false}, {"outCellScalarNames", nlohmann::ordered_json::array()}}}});
    EulerSolver<NS> solver(g_mpi);
    solver.ConfigureFromJson(cfg, true);
    solver.ReadMeshAndInitialize();
    auto &eval = solver.getEval();
    eval.InitializeUDOF(solver.getU());
    auto mesh = solver.getMesh();
    for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
    {
        solver.getURec()[cell].setZero();
        solver.getURecNew()[cell].setZero();
    }
    int rootPid = int(getpid());
    MPI_Bcast(&rootPid, 1, MPI_INT, 0, g_mpi.comm);
    auto base = (std::filesystem::temp_directory_path() /
                 ("dndsr-euler-native-output-" + std::to_string(rootPid)))
                    .string();
    EulerSolver<NS>::tAdditionalCellScalarList cells, boundaries;
    auto output = [&](const std::string &suffix)
    { solver.PrintData(base + suffix, "", [](DNDS::index)
                       { return 0.0; }, cells, boundaries, eval, 0); };
    REQUIRE(mesh->HasNativeState());
    CHECK_NOTHROW(mesh->CurrentNativeState()->Validate());
    output("-baseline");
    auto requirements = Geom::MeshReadRequirements::CellFlux();
    requirements.allFaceCellsLocal = true;
    CHECK_NOTHROW(mesh->CurrentNativeState()->CheckRequirements(requirements));
    mesh->isPeriodic = !mesh->IsPeriodic();
    mesh->periodicInfo = Geom::Periodicity{};
    EulerEvaluator<NS>::InitializeFV(mesh, solver.getVFV(), solver.getBCHandler());
    output("-native");
    CHECK(mesh->HasNativeState());
    MPI_Barrier(g_mpi.comm);
    if (g_mpi.rank == 0)
    {
        for (const auto *name : {"/VTKHDF/Points", "/VTKHDF/CellData/R", "/VTKHDF/CellData/P", "/VTKHDF/CellData/Velo",
                                 "/VTKHDF/PointData/R", "/VTKHDF/PointData/P", "/VTKHDF/PointData/Velo"})
        {
            CAPTURE(name);
            const auto baseline = OutputDataset(base + "-baseline.vtkhdf", name);
            const auto native = OutputDataset(base + "-native.vtkhdf", name);
            REQUIRE(native.size() == baseline.size());
            for (size_t i = 0; i < native.size(); ++i)
                CHECK(native[i] == doctest::Approx(baseline[i]).epsilon(1e-11));
        }
        std::filesystem::remove(base + "-baseline.vtkhdf");
        std::filesystem::remove(base + "-native.vtkhdf");
    }
    MPI_Barrier(g_mpi.comm);
}

static std::vector<double> GatherOutputOracle(const std::vector<double> &local)
{
    const int size = int(local.size());
    std::vector<int> counts(g_mpi.size), offsets(g_mpi.size + 1, 0);
    MPI_Allgather(&size, 1, MPI_INT, counts.data(), 1, MPI_INT, g_mpi.comm);
    for (int rank = 0; rank < g_mpi.size; ++rank)
        offsets[rank + 1] = offsets[rank] + counts[rank];
    // Small test-only oracle. Production mesh setup never gathers this data.
    std::vector<double> global(offsets.back());
    MPI_Allgatherv(local.data(), size, MPI_DOUBLE, global.data(), counts.data(), offsets.data(), MPI_DOUBLE, g_mpi.comm);
    return global;
}

static std::vector<double> ReadVTKCellDensity(const std::string &fileName)
{
    std::ifstream file(fileName);
    DNDS_check_throw_info(file.good(), "serial VTU output missing");
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto cells = text.find("<CellData");
    const auto name = text.find("Name=\"R\"", cells);
    const auto begin = text.find('>', name), end = text.find("</DataArray>", begin);
    DNDS_check_throw_info(cells != std::string::npos && name != std::string::npos &&
                              begin != std::string::npos && end != std::string::npos,
                          "serial VTU density array missing");
    std::istringstream values(text.substr(begin + 1, end - begin - 1));
    std::vector<double> result;
    double value = 0;
    while (values >> value)
        result.push_back(value);
    return result;
}

TEST_CASE("Native Euler ingress restart and serial boundary output retain field association")
{
    const auto root = projectRoot();
    int rootPid = int(getpid());
    MPI_Bcast(&rootPid, 1, MPI_INT, 0, g_mpi.comm);
    const auto directory = std::filesystem::temp_directory_path() / ("dndsr-native-euler-io-" + std::to_string(rootPid));
    if (g_mpi.rank == 0)
        std::filesystem::create_directories(directory);
    MPI_Barrier(g_mpi.comm);
    const auto archive = (directory / "mesh").string();
    std::vector<double> originalGeometry;
    for (int readMode : {0, 1, 2})
        for (int outputMode : {0, 1})
        {
            CAPTURE(readMode);
            CAPTURE(outputMode);
            const auto tag = std::to_string(readMode) + "-" + std::to_string(outputMode);
            const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "native_io_" + tag,
                                                 {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                                  {"limiterControl", {{"useLimiter", false}}},
                                                  {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/Uniform_3x3_wall.cgns"}, {"readMeshMode", readMode}, {"meshFilePartitionedInput", archive + ".dnds.h5"}, {"outPltMode", outputMode}, {"outBndData", true}, {"outVolumeData", true}, {"outPltTecplotFormat", true}, {"outPltVTKFormat", true}, {"outPltVTKHDFFormat", true}, {"outAtCellData", true}, {"outAtPointData", true}, {"vtuFloatEncodeMode", "ascii"}, {"nASCIIPrecision", 17}, {"allowAsyncPrintData", false}, {"outCellScalarNames", nlohmann::ordered_json::array()}}}});
            EulerSolver<NS> solver(g_mpi);
            solver.ConfigureFromJson(cfg, true);
            solver.ReadMeshAndInitialize();
            const auto mesh = solver.getMesh();
            REQUIRE(mesh->HasNativeState());
            CHECK_NOTHROW(mesh->CurrentNativeState()->Validate());
            CHECK(mesh->NumCellGlobal() == 9);
            if (readMode == 0 && outputMode == 0)
            {
                Serializer::SerializerFactory factory("H5");
                Geom::SerializeMesh(*mesh, archive, factory);
            }
            auto &eval = solver.getEval();
            const auto fv = solver.getVFV();
            std::vector<double> geometry, expectedCell, expectedBoundary;
            for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
            {
                const auto center = fv->GetCellBary(cell);
                const DNDS::real rho = 1 + center[0] / 32 + center[1] / 128;
                solver.getU()[cell] << rho, 0, 0, 0, 2.5;
                solver.getURec()[cell].setZero();
                solver.getURecNew()[cell].setZero();
                if (cell < mesh->NumCell())
                {
                    expectedCell.push_back(rho);
                    geometry.push_back(mesh->OriginalEntityId(Geom::EntityKind::Cell, cell) * 1024 + center[0] * 16 + center[1]);
                    CHECK(fv->GetCellVol(cell) == doctest::Approx(1));
                }
            }
            for (auto &flux : eval.fluxBnd)
                flux.setConstant(7);
            for (auto &force : eval.fluxBndForceT)
                force.setConstant(11);
            {
                auto read = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
                const auto view = read.HostView();
                for (DNDS::index boundary = 0; boundary < view.NumBnd(); ++boundary)
                    expectedBoundary.push_back(solver.getU()[view.BoundaryCells(boundary)[0]](0));
            }
            geometry = GatherOutputOracle(geometry);
            std::sort(geometry.begin(), geometry.end());
            if (readMode == 0 && outputMode == 0)
                originalGeometry = geometry;
            CHECK(geometry == originalGeometry);
            expectedCell = GatherOutputOracle(expectedCell);
            expectedBoundary = GatherOutputOracle(expectedBoundary);
            const auto output = (directory / tag).string();
            EulerSolver<NS>::tAdditionalCellScalarList cells, boundaries;
            solver.PrintData(output, "", [](DNDS::index)
                             { return 0.; }, cells, boundaries, eval, 0);
            MPI_Barrier(g_mpi.comm);
            if (g_mpi.rank == 0)
            {
                auto actualCell = OutputDataset(output + ".vtkhdf", "/VTKHDF/CellData/R");
                auto actualBoundary = OutputDataset(output + "_bnd.vtkhdf", "/VTKHDF/CellData/R");
                CHECK(actualCell == expectedCell);
                if (!outputMode)
                {
                    const auto serial = ReadVTKCellDensity(output + ".vtu");
                    REQUIRE(serial.size() == expectedCell.size());
                    // ASCII output uses the configured decimal precision.
                    for (size_t row = 0; row < serial.size(); ++row)
                        CHECK(serial[row] == doctest::Approx(expectedCell[row]).epsilon(1e-13));
                }
                std::sort(actualBoundary.begin(), actualBoundary.end());
                std::sort(expectedBoundary.begin(), expectedBoundary.end());
                CHECK(actualBoundary == expectedBoundary);
                for (auto value : OutputDataset(output + "_bnd.vtkhdf", "/VTKHDF/CellData/F0"))
                    CHECK(value == 7);
                const auto suffix = outputMode ? ".pvtu" : ".vtu";
                for (const auto &base : {output, output + "_bnd"})
                {
                    CHECK(std::filesystem::file_size(base + suffix) > 0);
                    if (!outputMode)
                        CHECK(std::filesystem::file_size(base + ".plt") > 0);
                }
            }
        }
    MPI_Barrier(g_mpi.comm);
    if (g_mpi.rank == 0)
        std::filesystem::remove_all(directory);
}

template <int variables>
static void WriteLegacyEulerRestart(const std::string &file, const MPIInfo &mpi,
                                    const std::vector<DNDS::index> &keys,
                                    const std::vector<double> &values,
                                    const std::string &invalid = "")
{
    // Deliberately use the old wire format: u has no origIndex. Rank zero owns
    // reversed rows; other writer ranks are empty. Stored IDs stay full int64.
    const DNDS::index rows = invalid == "global_count" ? 1 : DNDS::index(keys.size());
    const DNDS::index keyRows = rows - (invalid == "row_count" ? 1 : 0);
    ArrayDOFV<variables> data;
    data.InitPair("legacy-restart-values", mpi);
    data.father->Resize(mpi.rank == 0 ? rows : 0, variables, 1);
    data.son->Resize(0, variables, 1);
    Geom::tAdj1Pair original;
    original.InitPair("legacy-restart-keys", mpi);
    original.father->Resize(mpi.rank == 0 ? keyRows : 0);
    original.son->Resize(0);
    for (DNDS::index row = 0; row < data.father->Size(); ++row)
    {
        const auto source = keys.size() - 1 - size_t(row);
        for (int component = 0; component < variables; ++component)
            data[row](component) = component < 5 ? values[source * 5 + size_t(component)] : 37;
        if (row < keyRows)
        {
            original(row, 0) = keys[source];
            if (invalid == "duplicate" && row == 1)
                original(row, 0) = keys.back();
            if (invalid == "negative" && row == 0)
                original(row, 0) = -1;
            if (invalid == "missing" && row == 0)
                original(row, 0) = 0;
        }
    }
    auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
    serializer->OpenFile(file, false);
    data.WriteSerialize(serializer, "u", false, false);
    if (invalid != "no_keys")
        original.WriteSerialize(serializer, "cell2cellOrig", false, false);
    serializer->CloseFile();
}

TEST_CASE("Native Euler H5 restarts preserve sparse high Cell IDs and selected variable dimensions")
{
    const auto root = projectRoot();
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, g_mpi.comm);
    const auto directory = std::filesystem::temp_directory_path() / ("dndsr-euler-high-id-" + std::to_string(pid));
    if (g_mpi.rank == 0)
        std::filesystem::create_directories(directory);
    MPI_Barrier(g_mpi.comm);
    const auto archive = (directory / "mesh").string();
    {
        auto source = Geom::ReadNativeCGNS(g_mpi, root + "/data/mesh/Uniform_3x3_wall.cgns", 2,
                                           [](const std::string &)
                                           { return Geom::BC_ID_DEFAULT_WALL; });
        auto input = Geom::NativeMeshAssembler::CaptureOwned(*source, g_mpi);
        auto ids = input.ids.at(Geom::EntityKind::Cell).Clone();
        const auto cells = input.spaces.at(Geom::EntityKind::Cell);
        {
            auto write = ids.Write();
            for (DNDS::index row = 0; row < cells.NumOwned(); ++row)
                write.View()(size_t(row), 0) = std::numeric_limits<DNDS::index>::max() - 1 - cells.LocalToGlobal(row) * 4099;
        }
        input.ids[Geom::EntityKind::Cell] = std::move(ids);
        input.identity = {}; // This test creates a new ID domain from explicit runtime payloads.
        auto mesh = Geom::UnstructuredMesh::FromNative(g_mpi, Geom::AssembleNativeMesh(g_mpi, std::move(input)));
        Serializer::SerializerFactory factory("H5");
        Geom::SerializeMesh(*mesh, archive, factory);
    }
    const auto checkpoint = (directory / "solution").string();
    const auto checkpointFile = checkpoint + "_p" + std::to_string(g_mpi.size) + ".restart.dnds.h5";
    const auto dynamicFile = (directory / "dynamic.restart.dnds.h5").string();
    const auto legacyFile = (directory / "legacy.restart.dnds.h5").string();
    const auto legacyDynamicFile = (directory / "legacy-dynamic.restart.dnds.h5").string();
    const auto legacySerialFile = (directory / "legacy-serial.restart.dnds.h5").string();
    const auto invalidFile = (directory / "invalid.restart.dnds.h5").string();
    for (int readMode : {1, 2})
    {
        CAPTURE(readMode);
        const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json",
                                             "high_id_" + std::to_string(pid) + "_" + std::to_string(readMode),
                                             {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                              {"limiterControl", {{"useLimiter", false}}},
                                              {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/Uniform_3x3_wall.cgns"}, {"readMeshMode", readMode}, {"meshFilePartitionedInput", archive + ".dnds.h5"}, {"outPltMode", 1}, {"outLogName", (directory / "log").string()}, {"restartWriter", {{"type", "H5"}}}}}});
        EulerSolver<NS> solver(g_mpi);
        solver.ConfigureFromJson(cfg, true);
        solver.ReadMeshAndInitialize();
        const auto mesh = solver.getMesh();
        const auto fv = solver.getVFV();
        REQUIRE(mesh->NumCellGlobal() == 9);
        auto expected = [&](DNDS::index row)
        {
            const auto center = fv->GetCellBary(row);
            Eigen::Matrix<DNDS::real, 5, 1> value;
            value << 1 + center[0] / 32 + center[1] / 128,
                center[0] / 16, center[1] / 16, 0, 2.5 + center[0] / 8;
            return value;
        };
        for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
        {
            CHECK(mesh->OriginalEntityId(Geom::EntityKind::Cell, row) >= std::numeric_limits<DNDS::index>::max() - 1 - 8 * 4099);
            solver.getU()[row] = expected(row);
        }
        if (readMode == 1)
        {
            solver.PrintRestart(checkpoint);
            ArrayDOFV<Eigen::Dynamic> dynamic;
            fv->BuildUDof(dynamic, 7);
            std::vector<DNDS::index> ids;
            for (DNDS::index row = 0; row < mesh->NumCell(); ++row)
            {
                dynamic[row].head<5>() = expected(row);
                dynamic[row].tail<2>().setConstant(37);
                ids.push_back(mesh->OriginalEntityId(Geom::EntityKind::Cell, row));
            }
            auto serializer = std::make_shared<Serializer::SerializerH5>(g_mpi);
            serializer->OpenFile(dynamicFile, false);
            dynamic.WriteSerialize(serializer, "u", ids, false, false);
            serializer->CloseFile();

            const int count = int(ids.size());
            std::vector<int> counts(g_mpi.size), offsets(g_mpi.size + 1, 0);
            MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, g_mpi.comm);
            for (int rank = 0; rank < g_mpi.size; ++rank)
                offsets[rank + 1] = offsets[rank] + counts[rank];
            std::vector<DNDS::index> allKeys(size_t(offsets.back()));
            MPI_Allgatherv(ids.data(), count, DNDS_MPI_INDEX, allKeys.data(), counts.data(), offsets.data(), DNDS_MPI_INDEX, g_mpi.comm);
            std::vector<double> localValues;
            for (DNDS::index row = 0; row < mesh->NumCell(); ++row)
                for (int component = 0; component < 5; ++component)
                    localValues.push_back(expected(row)(component));
            const auto allValues = GatherOutputOracle(localValues);
            WriteLegacyEulerRestart<5>(legacyFile, g_mpi, allKeys, allValues);
            WriteLegacyEulerRestart<7>(legacyDynamicFile, g_mpi, allKeys, allValues);
            if (g_mpi.rank == 0)
                WriteLegacyEulerRestart<5>(legacySerialFile, MPIInfo(MPI_COMM_SELF), allKeys, allValues);
            MPI_Barrier(g_mpi.comm);

            for (const std::string invalid : {"duplicate", "negative", "missing", "row_count", "global_count", "no_keys"})
            {
                CAPTURE(invalid);
                WriteLegacyEulerRestart<5>(invalidFile, g_mpi, allKeys, allValues, invalid);
                solver.getU().setConstant(-905);
                CHECK_THROWS(solver.ReadRestart(invalidFile));
                CHECK_THROWS(solver.ReadRestartOtherSolver(invalidFile, {0, 4}));
                for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                    CHECK((solver.getU()[row].array() == -905).all());
                // A failed collective import must leave no leases or pending
                // operations that prevent a valid retry on the same solver.
                solver.ReadRestart(legacyFile);
                for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                    CHECK(solver.getU()[row] == expected(row));
            }

            const auto jsonDirectory = directory / "legacy.dir";
            if (g_mpi.rank == 0)
                std::filesystem::create_directories(jsonDirectory);
            MPI_Barrier(g_mpi.comm);
            for (bool duplicate : {true, false})
            {
                ArrayDOFV<5> data;
                data.InitPair("legacy-json-values", g_mpi);
                data.father->Resize(mesh->NumCell(), 5, 1);
                data.son->Resize(0, 5, 1);
                Geom::tAdj1Pair original;
                original.InitPair("legacy-json-keys", g_mpi);
                original.father->Resize(mesh->NumCell());
                original.son->Resize(0);
                for (DNDS::index row = 0; row < mesh->NumCell(); ++row)
                {
                    const auto source = mesh->NumCell() - 1 - row;
                    data[row] = expected(source);
                    original(row, 0) = ids[size_t(source)];
                }
                int hasDuplicate = duplicate && mesh->NumCell() > 1;
                if (hasDuplicate)
                    original(1, 0) = original(0, 0);
                MPI_Allreduce(MPI_IN_PLACE, &hasDuplicate, 1, MPI_INT, MPI_MAX, g_mpi.comm);
                auto json = std::make_shared<Serializer::SerializerJSON>();
                json->SetUseCodecOnUint8(true);
                json->OpenFile((jsonDirectory / fmt::format("{:04d}.json", g_mpi.rank)).string(), false);
                data.WriteSerialize(json, "u", false, false);
                original.WriteSerialize(json, "cell2cellOrig", false, false);
                json->CloseFile();
                MPI_Barrier(g_mpi.comm);
                solver.getU().setConstant(-907);
                if (hasDuplicate)
                {
                    CHECK_THROWS(solver.ReadRestart(jsonDirectory.string()));
                    CHECK_THROWS(solver.ReadRestartOtherSolver(jsonDirectory.string(), {0, 4}));
                    for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                        CHECK((solver.getU()[row].array() == -907).all());
                }
                else
                {
                    solver.ReadRestart(jsonDirectory.string());
                    for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                        CHECK(solver.getU()[row] == expected(row));
                    ArrayDOFV<Eigen::Dynamic> wider;
                    fv->BuildUDof(wider, 7);
                    for (DNDS::index row = 0; row < mesh->NumCell(); ++row)
                    {
                        wider[row].head<5>() = expected(mesh->NumCell() - 1 - row);
                        wider[row].tail<2>().setConstant(37);
                    }
                    json->OpenFile((jsonDirectory / fmt::format("{:04d}.json", g_mpi.rank)).string(), false);
                    wider.WriteSerialize(json, "u", false, false);
                    original.WriteSerialize(json, "cell2cellOrig", false, false);
                    json->CloseFile();
                    MPI_Barrier(g_mpi.comm);
                    const auto unchanged = [&](DNDS::index row, int component)
                    { return -909 - expected(row)(0) * 32 - DNDS::real(component) / 16; };
                    for (const auto &file : {jsonDirectory.string(), legacyFile})
                    {
                        CAPTURE(file);
                        for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                            for (int component = 0; component < 5; ++component)
                                solver.getU()[row](component) = unchanged(row, component);
                        CHECK_THROWS(solver.ReadRestartOtherSolver(file, {-1}));
                        CHECK_THROWS(solver.ReadRestartOtherSolver(file, {5}));
                        for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                            for (int component = 0; component < 5; ++component)
                                CHECK(solver.getU()[row](component) == unchanged(row, component));
                        solver.ReadRestartOtherSolver(file, {0, 4});
                        for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                            for (int component = 0; component < 5; ++component)
                                CHECK(solver.getU()[row](component) ==
                                      (component == 0 || component == 4 ? expected(row)(component) : unchanged(row, component)));
                    }
                }
            }
        }
        for (const auto &file : {checkpointFile, legacyFile, legacySerialFile})
        {
            CAPTURE(file);
            solver.getU().setConstant(-901);
            solver.ReadRestart(file);
            for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                for (int component = 0; component < 5; ++component)
                    CHECK(solver.getU()[row](component) == expected(row)(component));
        }
        for (const auto &file : {checkpointFile, dynamicFile, legacyFile, legacyDynamicFile, legacySerialFile})
        {
            CAPTURE(file);
            solver.getU().setConstant(-903);
            solver.ReadRestartOtherSolver(file, {0, 4});
            for (DNDS::index row = 0; row < mesh->NumCellProc(); ++row)
                for (int component = 0; component < 5; ++component)
                    CHECK(solver.getU()[row](component) == (component == 0 || component == 4 ? expected(row)(component) : -903));
        }
    }
    MPI_Barrier(g_mpi.comm);
    if (g_mpi.rank == 0)
        std::filesystem::remove_all(directory);
}

TEST_CASE("Euler archived mesh setup refreshes loaded wall vectors after actual coordinate snapping")
{
    const auto root = projectRoot();
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, g_mpi.comm);
    const auto directory = std::filesystem::temp_directory_path() / ("dndsr-euler-setup-" + std::to_string(pid));
    if (g_mpi.rank == 0)
        std::filesystem::create_directories(directory);
    MPI_Barrier(g_mpi.comm);
    const auto archive = (directory / "mesh").string();
    constexpr DNDS::real epsilon = 1e-4;
    {
        auto source = Geom::ReadNativeCGNS(g_mpi, root + "/data/mesh/Uniform_3x3_wall.cgns", 2,
                                           [](const std::string &)
                                           { return Geom::BC_ID_DEFAULT_WALL; });
        source = Geom::PrepareNativeMeshTopology(*source, g_mpi);
        auto mesh = Geom::UnstructuredMesh::FromNative(g_mpi, source);
        DNDS::real minimum = DNDS::veryLargeReal;
        for (DNDS::index node = 0; node < mesh->NumNode(); ++node)
            minimum = std::min(minimum, mesh->Coord(node)[0]);
        MPI::AllreduceOneReal(minimum, MPI_MIN, g_mpi);
        mesh->ApplyAffineGeometryTransform(Eigen::Matrix3d::Identity(), Geom::tPoint{epsilon - minimum, 0, 0});
        Geom::UnstructuredMesh::WallDistOptions sentinel;
        sentinel.minWallDist = 1234;
        mesh->BuildNodeWallDist([](auto zone)
                                { return zone == Geom::BC_ID_DEFAULT_WALL; }, sentinel);
        Serializer::SerializerFactory factory("H5");
        Geom::SerializeMesh(*mesh, archive, factory);
    }
    auto overrides = nlohmann::ordered_json{
        {"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}, {"functionalSettings", {{"anisotropicType", "WallDist"}}}}},
        {"limiterControl", {{"useLimiter", false}}},
        {"dataIOControl", {{"meshFile", (directory / "partition-output").string()}, {"meshFilePartitionedInput", archive + ".dnds.h5"}, {"readMeshMode", 1}, {"meshDirectBisect", 0}, {"meshElevation", 0}, {"meshReorderCells", 0}, {"meshBuildWallDist", 0}, {"meshScale", 1}, {"meshRotZ", 0}, {"rectifyNearPlane", 1}, {"rectifyNearPlaneThres", 1e-3}, {"meshWallDistOptions", {{"method", 0}, {"minWallDist", 0}}}, {"outPltMode", 1}, {"outLogName", (directory / "log").string()}, {"meshPartitionedWriter", {{"type", "H5"}}}}}};
    for (int mode : {1, 2})
    {
        CAPTURE(mode);
        overrides["dataIOControl"]["readMeshMode"] = mode;
        const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json",
                                             "setup_wall_" + std::to_string(pid) + "_" + std::to_string(mode), overrides);
        EulerSolver<NS> solver(g_mpi);
        solver.ConfigureFromJson(cfg, true);
        solver.ReadMeshAndInitialize();
        auto mesh = solver.getMesh();
        REQUIRE(mesh->HasNodeWallDist());
        CHECK(solver.getConfiguration().dataIOControl.meshBuildWallDist == 0);
        CHECK_NOTHROW(solver.getVFV()->ValidateMeshDependencies());
        CHECK_NOTHROW(solver.getEval().ValidateMeshDependencies());
        Geom::tPoint lower = Geom::tPoint::Constant(DNDS::veryLargeReal), upper = -lower;
        for (DNDS::index node = 0; node < mesh->NumNode(); ++node)
        {
            lower = lower.cwiseMin(mesh->Coord(node));
            upper = upper.cwiseMax(mesh->Coord(node));
        }
        MPI_Allreduce(MPI_IN_PLACE, lower.data(), 3, MPI_DOUBLE, MPI_MIN, g_mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, upper.data(), 3, MPI_DOUBLE, MPI_MAX, g_mpi.comm);
        CHECK(lower[0] == 0); // The archive's minimum was strictly epsilon.
        DNDS::real totalWallDistance = 0;
        auto read = mesh->AcquireRead(mesh->CurrentNativeRequirements());
        const auto view = read.HostView();
        for (DNDS::index node = 0; node < view.NumNodeProc(); ++node)
        {
            const auto point = view.Coord(node);
            CHECK_FALSE((point[0] > 0 && point[0] < 1e-3));
            const auto distance = std::min({point[0] - lower[0], upper[0] - point[0], point[1] - lower[1], upper[1] - point[1]});
            CHECK(view.NodeWallDist(node).norm() == doctest::Approx(distance).epsilon(1e-10).scale(1));
            if (node < view.NumNode())
                totalWallDistance += view.NodeWallDist(node).norm();
        }
        MPI::AllreduceOneReal(totalWallDistance, MPI_SUM, g_mpi);
        CHECK(totalWallDistance > 0);
        CHECK(totalWallDistance < 1234);
    }
    // Partition-only output keeps the archive's unsnapped geometry and loaded
    // wall vectors, before solver transforms or FV construction are entered.
    overrides["dataIOControl"]["readMeshMode"] = 1;
    overrides["dataIOControl"]["meshScale"] = 2;
    overrides["timeMarchControl"] = {{"partitionMeshOnly", true}};
    const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json",
                                         "setup_partition_" + std::to_string(pid), overrides);
    {
        EulerSolver<NS> solver(g_mpi);
        solver.ConfigureFromJson(cfg, true);
        solver.ReadMeshAndInitialize();
        const auto output = Geom::MeshH5Path((directory / "partition-output").string(), g_mpi.size);
        auto mesh = Geom::ReadNativeMeshFromArchive(g_mpi, Serializer::SerializerFactory("H5"), output);
        CHECK(mesh->HasNodeWallDist());
        DNDS::real minimum = DNDS::veryLargeReal, maximumWall = 0;
        for (DNDS::index node = 0; node < mesh->NumNode(); ++node)
        {
            minimum = std::min(minimum, mesh->Coord(node)[0]);
            maximumWall = std::max(maximumWall, mesh->NodeWallDist(node).norm());
        }
        MPI::AllreduceOneReal(minimum, MPI_MIN, g_mpi);
        MPI::AllreduceOneReal(maximumWall, MPI_MAX, g_mpi);
        CHECK(minimum == doctest::Approx(epsilon));
        CHECK(maximumWall == doctest::Approx(1234));
    }
    MPI_Barrier(g_mpi.comm);
    if (g_mpi.rank == 0)
        std::filesystem::remove_all(directory);
}

TEST_CASE("Euler periodic source setup accepts configured near-plane cleanup after rotation")
{
    const auto root = projectRoot();
    const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "setup_periodic_snap",
                                         {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                          {"dataIOControl", {{"meshFile", root + "/data/mesh/IV10_10.cgns"}, {"readMeshMode", 0}, {"meshDirectBisect", 0}, {"meshElevation", 0}, {"meshReorderCells", 0}, {"meshRotZ", 1e-6}, {"meshScale", 1}, {"rectifyNearPlane", 1}, {"rectifyNearPlaneThres", 1e-5}, {"outPltMode", 1}}}});
    EulerSolver<NS> solver(g_mpi);
    solver.ConfigureFromJson(cfg, true);
    solver.ReadMeshAndInitialize();
    auto mesh = solver.getMesh();
    CHECK(mesh->IsPeriodic());
    CHECK_NOTHROW(solver.getVFV()->ValidateMeshDependencies());
    CHECK_NOTHROW(solver.getEval().ValidateMeshDependencies());
    int snapped = 0;
    for (DNDS::index node = 0; node < mesh->NumNode(); ++node)
    {
        const auto point = mesh->Coord(node);
        if (std::abs(point[0]) < 1e-5)
        {
            CHECK(point[0] == 0);
            // Rotation moved every positive-y point on the original x=0
            // plane away from zero, so these zeros prove actual cleanup.
            snapped += point[1] > 1;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &snapped, 1, MPI_INT, MPI_SUM, g_mpi.comm);
    CHECK(snapped > 0);
    const auto angle = 1e-6 * DNDS::pi / 180;
    CHECK(mesh->PeriodicMetadata().translation[1].map().isApprox(Geom::tPoint{10 * std::cos(angle), 10 * std::sin(angle), 0}, 1e-13));
}

// ===================================================================
// main with MPI
// ===================================================================
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    MPI_Finalize();
    return res;
}
