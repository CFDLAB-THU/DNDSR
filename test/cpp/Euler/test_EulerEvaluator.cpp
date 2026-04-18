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
 *
 * This is an MPI test registered at np=1, 2, 4.
 * Golden value sentinel: 1e300 means not yet acquired.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Euler/EulerSolver.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

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
    auto tmpDir = std::filesystem::temp_directory_path() / "dndsr_euler_test";
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

    // --- STEP 1: fdt (compute local pseudo-timestep) ---
    real CFL = config.implicitCFLControl.CFL;
    real curDtMin = 1e100;

    eval.FixUMaxFilter(u);
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();

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

    return {resNorm(0), resNorm(nVars > 4 ? 4 : nVars - 1), incL2};
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
         {"dataIOControl", {{"meshDirectBisect", 0},
                            {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});

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

TEST_CASE("EulerEvaluator pipeline: NACA0012 (NS_SA, P1)")
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

// ===================================================================
// TEST 3: 3D Box (NS_3D, periodic, inviscid)
// ===================================================================

TEST_CASE("EulerEvaluator pipeline: Box (NS_3D, P1)")
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
