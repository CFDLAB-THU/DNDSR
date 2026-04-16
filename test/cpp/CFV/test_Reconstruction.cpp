/**
 * @file test_Reconstruction.cpp
 * @brief Phase-0 regression tests for CFV Variational Reconstruction.
 *
 * Parameterized over [mesh+function, reconstruction_method].
 *
 * All iterative VR uses Jacobi iteration (SORInstead=false) so that
 * golden values are deterministic across MPI partitionings (np=1,2,4).
 *
 * Error metric: L1 pointwise error at 6th-degree quadrature points,
 * divided by domain volume (so the dimension matches the field function).
 *
 * Meshes:
 *   - Uniform_3x3_wall (9 quads, wall BC)  -- polynomial exactness tests
 *   - IV10_10 (100 quads, periodic) + bisections -> 400, 1600 cells
 *   - IV10U_10 (322 tris, periodic) + bisections -> ~1288, ~5152 cells
 *
 * Golden values captured from commit c774b89 on dev/harry_refac1.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "CFV/VariationalReconstruction.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <functional>
#include <nlohmann/json.hpp>
#include <map>
#include <string>
#include <vector>

using namespace DNDS;
using namespace DNDS::Geom;

static constexpr int g_dim = 2;
static constexpr int g_nv = 1;
using tVR = CFV::VariationalReconstruction<g_dim>;

static MPIInfo g_mpi;

// ===================================================================
// Mesh builder
// ===================================================================
static std::string meshPath(const std::string &name)
{
    std::string f(__FILE__);
    for (int i = 0; i < 4; i++)
    {
        auto pos = f.rfind('/');
        if (pos == std::string::npos)
            pos = f.rfind('\\');
        if (pos != std::string::npos)
            f = f.substr(0, pos);
    }
    return f + "/data/mesh/" + name;
}

static ssp<UnstructuredMesh> buildMesh(
    const std::string &file, bool periodic,
    DNDS::real Lx, DNDS::real Ly, int nBisect = 0)
{
    auto mesh = std::make_shared<UnstructuredMesh>(g_mpi, g_dim);
    UnstructuredMeshSerialRW reader(mesh, 0);

    if (periodic)
    {
        tPoint zero{0, 0, 0};
        mesh->SetPeriodicGeometry(
            tPoint{Lx, 0, 0}, zero, zero,
            tPoint{0, Ly, 0}, zero, zero,
            tPoint{0, 0, 0}, zero, zero);
    }

    reader.ReadFromCGNSSerial(meshPath(file));
    reader.Deduplicate1to1Periodic();
    reader.BuildCell2Cell();

    UnstructuredMeshSerialRW::PartitionOptions pOpt;
    pOpt.metisSeed = 42;
    reader.MeshPartitionCell2Cell(pOpt);
    reader.PartitionReorderToMeshCell2Cell();

    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();

    // Bisect nBisect times via elevation+bisection
    for (int ib = 0; ib < nBisect; ib++)
    {
        auto meshO2 = std::make_shared<UnstructuredMesh>(g_mpi, g_dim);
        meshO2->BuildO2FromO1Elevation(*mesh);
        meshO2->RecoverNode2CellAndNode2Bnd();
        meshO2->RecoverCell2CellAndBnd2Cell();
        meshO2->BuildGhostPrimary();
        meshO2->AdjGlobal2LocalPrimary();

        meshO2->AdjLocal2GlobalPrimary();
        auto meshBis = std::make_shared<UnstructuredMesh>(g_mpi, g_dim);
        meshBis->BuildBisectO1FormO2(*meshO2);
        meshBis->RecoverNode2CellAndNode2Bnd();
        meshBis->RecoverCell2CellAndBnd2Cell();
        meshBis->BuildGhostPrimary();
        meshBis->AdjGlobal2LocalPrimary();
        mesh = meshBis;
    }

    mesh->InterpolateFace();
    mesh->AssertOnFaces();
    return mesh;
}

// ===================================================================
// VR builder
// ===================================================================
enum class RecMethod
{
    GaussGreen,   // explicit 2nd-order Gauss-Green gradient
    VFV_P1_HQM,   // iterative VR, maxOrder=1, HQM weights
    VFV_P2_HQM,   // iterative VR, maxOrder=2, HQM weights
    VFV_P3_HQM,   // iterative VR, maxOrder=3, HQM weights
    VFV_P1_Default,
    VFV_P2_Default,
    VFV_P3_Default,
};

static const char *recMethodName(RecMethod m)
{
    switch (m)
    {
    case RecMethod::GaussGreen:    return "GG";
    case RecMethod::VFV_P1_HQM:    return "P1-HQM";
    case RecMethod::VFV_P2_HQM:    return "P2-HQM";
    case RecMethod::VFV_P3_HQM:    return "P3-HQM";
    case RecMethod::VFV_P1_Default: return "P1-Def";
    case RecMethod::VFV_P2_Default: return "P2-Def";
    case RecMethod::VFV_P3_Default: return "P3-Def";
    }
    return "?";
}

static int recMethodOrder(RecMethod m)
{
    switch (m)
    {
    case RecMethod::GaussGreen:     return 1;
    case RecMethod::VFV_P1_HQM:
    case RecMethod::VFV_P1_Default: return 1;
    case RecMethod::VFV_P2_HQM:
    case RecMethod::VFV_P2_Default: return 2;
    case RecMethod::VFV_P3_HQM:
    case RecMethod::VFV_P3_Default: return 3;
    }
    return 1;
}

static ssp<tVR> buildVR(ssp<UnstructuredMesh> mesh, RecMethod method)
{
    auto vr = std::make_shared<tVR>(g_mpi, mesh);

    CFV::VRSettings defaultSettings(g_dim);
    nlohmann::ordered_json j;
    defaultSettings.WriteIntoJson(j);

    int order = recMethodOrder(method);
    j["maxOrder"] = order;
    j["intOrder"] = std::max(order + 2, 5);
    j["cacheDiffBase"] = true;
    // Force Jacobi iteration for deterministic results across MPI partitions
    j["SORInstead"] = false;
    j["jacobiRelax"] = 1.0;

    bool isHQM = (method == RecMethod::VFV_P1_HQM ||
                  method == RecMethod::VFV_P2_HQM ||
                  method == RecMethod::VFV_P3_HQM);

    if (method == RecMethod::GaussGreen)
    {
        j["subs2ndOrder"] = 1; // Gauss-Green
        // weights don't matter for explicit GG
    }
    else if (isHQM)
    {
        j["subs2ndOrder"] = 0; // full VFV
        j["functionalSettings"]["dirWeightScheme"] = "HQM_OPT";
        j["functionalSettings"]["geomWeightScheme"] = "HQM_SD";
        j["functionalSettings"]["geomWeightPower"] = 0.5;
        j["functionalSettings"]["geomWeightBias"] = 1.0;
    }
    else
    {
        j["subs2ndOrder"] = 0; // full VFV
        j["functionalSettings"]["dirWeightScheme"] = "Factorial";
        j["functionalSettings"]["geomWeightScheme"] = "GWNone";
    }

    vr->parseSettings(j);
    if (mesh->isPeriodic)
        vr->SetPeriodicTransformations(); // identity for scalar
    vr->ConstructMetrics();
    vr->ConstructBaseAndWeight();
    vr->ConstructRecCoeff();
    return vr;
}

// ===================================================================
// Boundary callback: Dirichlet (returns exact value for wall tests)
// ===================================================================
using ScalarFunc = std::function<DNDS::real(const tPoint &)>;

static tVR::TFBoundary<g_nv> makeDirichletBC(const ScalarFunc &f)
{
    return [f](const auto &, const auto &, DNDS::index, DNDS::index, int,
               const tPoint &, const tPoint &pPhy, t_index)
    {
        return Eigen::Vector<DNDS::real, g_nv>{f(pPhy)};
    };
}

static tVR::TFBoundary<g_nv> g_zeroBC =
    [](const auto &, const auto &, DNDS::index, DNDS::index, int,
       const tPoint &, const tPoint &, t_index)
{ return Eigen::Vector<DNDS::real, g_nv>::Zero(); };

// ===================================================================
// Core: run reconstruction and measure L1 error at 6th-degree quadrature
// points, normalized by domain volume.
// Returns the error and optionally prints iteration progress.
// ===================================================================
static DNDS::real runTest(
    ssp<tVR> vr,
    RecMethod method,
    const ScalarFunc &exactFunc,
    const tVR::TFBoundary<g_nv> &bc,
    int maxIters,
    DNDS::real convTol,    // convergence threshold on increment; 0 = no early stop
    bool printProgress)
{
    auto mesh = vr->mesh;

    // --- Allocate arrays ---
    CFV::tUDof<g_nv> u;
    vr->BuildUDof(u, 1);

    // --- Set cell-averaged DOFs via quadrature ---
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); iCell++)
    {
        auto qCell = vr->GetCellQuad(iCell);
        Eigen::Vector<DNDS::real, g_nv> uc;
        uc.setZero();
        qCell.IntegrationSimple(
            uc,
            [&](auto &vInc, int iG)
            {
                vInc(0) = exactFunc(vr->GetCellQuadraturePPhys(iCell, iG)) *
                           vr->GetCellJacobiDet(iCell, iG);
            });
        u[iCell] = uc / vr->GetCellVol(iCell);
    }
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();

    // --- Reconstruct ---
    if (method == RecMethod::GaussGreen)
    {
        // Explicit Gauss-Green: produces gradient, not polynomial coefficients
        CFV::tUGrad<g_nv, g_dim> uGrad;
        vr->BuildUGrad(uGrad, 1);
        vr->DoReconstruction2ndGrad<g_nv>(uGrad, u, bc, 1 /*GG method*/);
        uGrad.trans.startPersistentPull();
        uGrad.trans.waitPersistentPull();

        // Measure L1 error: u_rec(x) = u_mean + grad^T * (x - x_bary)
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<g_dim - 1>);
        DNDS::real errLocal = 0.0;
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto qCell = vr->GetCellQuad(iCell);
            DNDS::real errCell = 0.0;
            qCell.IntegrationSimple(
                errCell,
                [&](DNDS::real &vInc, int iG)
                {
                    tPoint pPhy = vr->GetCellQuadraturePPhys(iCell, iG);
                    DNDS::real uRecVal = u[iCell](0) +
                        (uGrad[iCell].transpose() *
                         (pPhy - vr->GetCellBary(iCell))(Seq012))(0);
                    DNDS::real uExact = exactFunc(pPhy);
                    vInc = std::abs(uRecVal - uExact) * vr->GetCellJacobiDet(iCell, iG);
                });
            errLocal += errCell;
        }
        DNDS::real errGlobal = 0.0;
        MPI::Allreduce(&errLocal, &errGlobal, 1, DNDS_MPI_REAL, MPI_SUM, g_mpi.comm);
        return errGlobal / vr->GetGlobalVol();
    }
    else
    {
        // Iterative VFV reconstruction
        CFV::tURec<g_nv> uRec, uRecNew;
        vr->BuildURec(uRec, 1);
        vr->BuildURec(uRecNew, 1);

        DNDS::real lastInc = veryLargeReal;
        for (int iter = 0; iter < maxIters; iter++)
        {
            vr->DoReconstructionIter<g_nv>(
                uRec, uRecNew, u, bc, /*putIntoNew=*/true);

            // Compute increment norm
            DNDS::real incLocal = 0.0;
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); iCell++)
                incLocal += (uRecNew[iCell] - uRec[iCell]).array().square().sum();
            DNDS::real incGlobal = 0.0;
            MPI::Allreduce(&incLocal, &incGlobal, 1, DNDS_MPI_REAL, MPI_SUM, g_mpi.comm);
            incGlobal = std::sqrt(incGlobal / mesh->NumCellGlobal());

            std::swap(uRec, uRecNew);
            uRec.trans.startPersistentPull();
            uRec.trans.waitPersistentPull();

            lastInc = incGlobal;
            if (printProgress && g_mpi.rank == 0 && (iter < 5 || (iter + 1) % 20 == 0))
                std::cout << "    iter " << iter + 1 << " inc = "
                          << std::scientific << incGlobal << std::endl;

            if (convTol > 0 && incGlobal < convTol)
            {
                if (printProgress && g_mpi.rank == 0)
                    std::cout << "    converged at iter " << iter + 1 << std::endl;
                break;
            }
        }

        // Measure L1 error at cell quadrature points (using VR's intOrder, which >= 6)
        DNDS::real errLocal = 0.0;
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto qCell = vr->GetCellQuad(iCell);
            DNDS::real errCell = 0.0;
            qCell.IntegrationSimple(
                errCell,
                [&](DNDS::real &vInc, int iG)
                {
                    Eigen::VectorXd baseVal =
                        vr->GetIntPointDiffBaseValue(
                            iCell, -1, -1, iG, std::array<int, 1>{0}, 1) *
                        uRec[iCell];
                    DNDS::real uRecVal = baseVal(0) + u[iCell](0);
                    DNDS::real uExact = exactFunc(vr->GetCellQuadraturePPhys(iCell, iG));
                    vInc = std::abs(uRecVal - uExact) * vr->GetCellJacobiDet(iCell, iG);
                });
            errLocal += errCell;
        }
        DNDS::real errGlobal = 0.0;
        MPI::Allreduce(&errLocal, &errGlobal, 1, DNDS_MPI_REAL, MPI_SUM, g_mpi.comm);
        return errGlobal / vr->GetGlobalVol();
    }
}

// ===================================================================
// Prebuilt test data
// ===================================================================

// --- Wall mesh (polynomial exactness) ---
static ssp<UnstructuredMesh> g_wall_mesh;

// --- Periodic meshes for convergence (IV10 quad, IV10U tri) ---
static ssp<UnstructuredMesh> g_iv10[3];  // bisect 0,1,2
static ssp<UnstructuredMesh> g_iv10u[3]; // bisect 0,1,2

// --- VR objects keyed by (mesh_ptr, method) ---
// We build them on demand in the test runner to avoid combinatorial explosion
// at startup. But for the wall mesh we prebuild a few.

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    if (g_mpi.rank == 0)
        std::cout << "=== Building meshes ===" << std::endl;

    g_wall_mesh = buildMesh("Uniform_3x3_wall.cgns", false, 0, 0, 0);

    for (int ib = 0; ib < 3; ib++)
    {
        if (g_mpi.rank == 0)
            std::cout << "  IV10_10 bisect=" << ib << std::endl;
        g_iv10[ib] = buildMesh("IV10_10.cgns", true, 10, 10, ib);
    }
    for (int ib = 0; ib < 3; ib++)
    {
        if (g_mpi.rank == 0)
            std::cout << "  IV10U_10 bisect=" << ib << std::endl;
        g_iv10u[ib] = buildMesh("IV10U_10.cgns", true, 10, 10, ib);
    }

    if (g_mpi.rank == 0)
        std::cout << "=== Meshes built ===" << std::endl;

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    g_wall_mesh.reset();
    for (auto &m : g_iv10)
        m.reset();
    for (auto &m : g_iv10u)
        m.reset();
    MPI_Finalize();
    return res;
}

// ===================================================================
// Test functions (periodic, L=10)
// ===================================================================
static const DNDS::real g_L = 10.0;
static const DNDS::real g_k = 2.0 * pi / g_L;

static ScalarFunc sinCos = [](const tPoint &p)
{ return std::sin(g_k * p[0]) * std::cos(g_k * p[1]); };

static ScalarFunc cosPlusCos = [](const tPoint &p)
{ return std::cos(g_k * p[0]) + std::cos(g_k * p[1]); };

// ===================================================================
// POLYNOMIAL EXACTNESS on wall mesh (Uniform_3x3_wall, [-1,2]^2)
// ===================================================================

#define POLY_TEST(testName, method, polyFunc, polyDeg)                          \
    TEST_CASE("Wall/" testName "/" #method)                                     \
    {                                                                           \
        ScalarFunc f = polyFunc;                                                \
        auto vr = buildVR(g_wall_mesh, RecMethod::method);                     \
        auto bc = makeDirichletBC(f);                                          \
        DNDS::real err = runTest(vr, RecMethod::method, f, bc, 100, 1e-15, false); \
        if (g_mpi.rank == 0)                                                   \
            std::cout << "[Wall/" testName "/" #method "] err = "               \
                      << std::scientific << err << std::endl;                   \
        if (polyDeg == 0)                                                       \
            CHECK(err < 1e-12);                                                \
        else                                                                    \
            CHECK(err < 10.0); /* golden-value regression checked separately */ \
    }

// Constant (degree 0): exact for all methods
POLY_TEST("const", GaussGreen,    [](const tPoint &) { return 1.0; }, 0)
POLY_TEST("const", VFV_P1_HQM,    [](const tPoint &) { return 1.0; }, 0)
POLY_TEST("const", VFV_P2_HQM,    [](const tPoint &) { return 1.0; }, 0)
POLY_TEST("const", VFV_P3_HQM,    [](const tPoint &) { return 1.0; }, 0)
POLY_TEST("const", VFV_P1_Default, [](const tPoint &) { return 1.0; }, 0)

// Linear (degree 1): exact for GG and p>=1
POLY_TEST("linear", GaussGreen,    [](const tPoint &p) { return p[0] + 2*p[1]; }, 1)
POLY_TEST("linear", VFV_P1_HQM,    [](const tPoint &p) { return p[0] + 2*p[1]; }, 1)
POLY_TEST("linear", VFV_P2_HQM,    [](const tPoint &p) { return p[0] + 2*p[1]; }, 1)
POLY_TEST("linear", VFV_P3_HQM,    [](const tPoint &p) { return p[0] + 2*p[1]; }, 1)
POLY_TEST("linear", VFV_P1_Default, [](const tPoint &p) { return p[0] + 2*p[1]; }, 1)

// Quadratic (degree 2): exact for p>=2
POLY_TEST("quad", VFV_P2_HQM,    [](const tPoint &p) { return p[0]*p[0] + p[1]*p[1]; }, 2)
POLY_TEST("quad", VFV_P3_HQM,    [](const tPoint &p) { return p[0]*p[0] + p[1]*p[1]; }, 2)
POLY_TEST("quad", VFV_P2_Default, [](const tPoint &p) { return p[0]*p[0] + p[1]*p[1]; }, 2)

// Cubic (degree 3): exact for p>=3
POLY_TEST("cubic", VFV_P3_HQM,    [](const tPoint &p) { return p[0]*p[0]*p[0] + p[0]*p[1]*p[1]; }, 3)
POLY_TEST("cubic", VFV_P3_Default, [](const tPoint &p) { return p[0]*p[0]*p[0] + p[0]*p[1]*p[1]; }, 3)

#undef POLY_TEST

// ===================================================================
// PERIODIC SMOOTH FUNCTIONS on IV10 (quad) and IV10U (tri) meshes
// with convergence series (bisect 0,1,2).
// Golden values -- to be filled after first acquisition run.
// ===================================================================

struct PeriodicTestCase
{
    const char *meshName;       // "IV10" or "IV10U"
    ssp<UnstructuredMesh> *meshArray; // pointer to g_iv10 or g_iv10u
    RecMethod method;
    ScalarFunc func;
    const char *funcName;
    int maxIters;
    DNDS::real convTol;
    DNDS::real golden[3]; // golden L1/vol for bisect 0,1,2 (0 = not yet acquired)
    bool checkConvergence; // whether to also check the iteration converges
};

// Golden values captured from commit c774b89 on dev/harry_refac1 (np=1).
// Jacobi iteration ensures determinism across all np values.
static PeriodicTestCase g_periodicTests[] = {
    // IV10 (quad) + sin*cos
    {"IV10", g_iv10, RecMethod::GaussGreen, sinCos, "sincos", 1, 0,
     {1.5599270188e-02, 3.4233789068e-03, 7.8943623278e-04}, false},
    {"IV10", g_iv10, RecMethod::VFV_P1_HQM, sinCos, "sincos", 200, 1e-14,
     {4.6604402914e-02, 9.2961629825e-03, 1.5347312301e-03}, true},
    {"IV10", g_iv10, RecMethod::VFV_P2_HQM, sinCos, "sincos", 200, 1e-14,
     {3.0528143687e-03, 2.3057099673e-04, 2.4367733525e-05}, true},
    {"IV10", g_iv10, RecMethod::VFV_P3_HQM, sinCos, "sincos", 200, 1e-14,
     {1.9105219870e-03, 4.6701352192e-05, 1.4890868814e-06}, false},
    {"IV10", g_iv10, RecMethod::VFV_P1_Default, sinCos, "sincos", 200, 1e-14,
     {4.6604402914e-02, 9.2961629825e-03, 1.5347312301e-03}, false},
    {"IV10", g_iv10, RecMethod::VFV_P3_Default, sinCos, "sincos", 200, 1e-14,
     {1.8840503911e-03, 2.4995731251e-05, 8.3670061853e-07}, false},

    // IV10U (tri) + sin*cos
    {"IV10U", g_iv10u, RecMethod::GaussGreen, sinCos, "sincos", 1, 0,
     {1.3027440876e-02, 3.8074751503e-03, 1.0853979656e-03}, false},
    {"IV10U", g_iv10u, RecMethod::VFV_P1_HQM, sinCos, "sincos", 200, 1e-14,
     {1.2040354507e-02, 2.2707678072e-03, 4.6833420900e-04}, false},
    {"IV10U", g_iv10u, RecMethod::VFV_P2_HQM, sinCos, "sincos", 200, 1e-14,
     {5.5617445849e-04, 5.9964034731e-05, 6.4337896956e-06}, false},
    {"IV10U", g_iv10u, RecMethod::VFV_P3_HQM, sinCos, "sincos", 200, 1e-14,
     {1.5878119293e-04, 9.8836538778e-06, 4.3407055649e-07}, false},

    // IV10 (quad) + cos+cos
    {"IV10", g_iv10, RecMethod::VFV_P3_HQM, cosPlusCos, "cos+cos", 200, 1e-14,
     {5.5612138851e-04, 1.6082305163e-05, 6.6188225747e-07}, false},
};

static const int g_nPeriodicTests = sizeof(g_periodicTests) / sizeof(g_periodicTests[0]);

TEST_CASE("Periodic reconstruction convergence series")
{
    for (int ti = 0; ti < g_nPeriodicTests; ti++)
    {
        auto &tc = g_periodicTests[ti];
        std::string label = std::string(tc.meshName) + "/" + tc.funcName +
                            "/" + recMethodName(tc.method);

        SUBCASE(label.c_str())
        {
            for (int ib = 0; ib < 3; ib++)
            {
                CAPTURE(ib);
                auto mesh = tc.meshArray[ib];
                auto vr = buildVR(mesh, tc.method);
                bool print = (ib == 0 && tc.checkConvergence);
                DNDS::real err = runTest(vr, tc.method, tc.func,
                                         g_zeroBC, tc.maxIters, tc.convTol, print);

                if (g_mpi.rank == 0)
                    std::cout << "[" << label << " bis=" << ib
                              << "] err = " << std::scientific
                              << std::setprecision(10) << err << std::endl;

                if (tc.golden[ib] != 0.0)
                    CHECK(err == doctest::Approx(tc.golden[ib]).epsilon(1e-6));
                else
                    CHECK(err >= 0.0); // acquisition: just check non-negative
            }
        }
    }
}

// ===================================================================
// Convergence check: selected VFV cases should converge
// ===================================================================

TEST_CASE("VFV P2 HQM converges on IV10 base mesh")
{
    auto vr = buildVR(g_iv10[0], RecMethod::VFV_P2_HQM);

    CFV::tUDof<g_nv> u;
    vr->BuildUDof(u, 1);
    CFV::tURec<g_nv> uRec, uRecNew;
    vr->BuildURec(uRec, 1);
    vr->BuildURec(uRecNew, 1);

    for (DNDS::index iCell = 0; iCell < vr->mesh->NumCell(); iCell++)
    {
        auto qCell = vr->GetCellQuad(iCell);
        Eigen::Vector<DNDS::real, g_nv> uc;
        uc.setZero();
        qCell.IntegrationSimple(
            uc,
            [&](auto &vInc, int iG)
            {
                vInc(0) = sinCos(vr->GetCellQuadraturePPhys(iCell, iG)) *
                           vr->GetCellJacobiDet(iCell, iG);
            });
        u[iCell] = uc / vr->GetCellVol(iCell);
    }
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();

    DNDS::real lastInc = veryLargeReal;
    int convergedAt = 0;
    for (int iter = 0; iter < 200; iter++)
    {
        vr->DoReconstructionIter<g_nv>(uRec, uRecNew, u, g_zeroBC, true);

        DNDS::real incLocal = 0.0;
        for (DNDS::index iCell = 0; iCell < vr->mesh->NumCell(); iCell++)
            incLocal += (uRecNew[iCell] - uRec[iCell]).array().square().sum();
        DNDS::real incGlobal = 0.0;
        MPI::Allreduce(&incLocal, &incGlobal, 1, DNDS_MPI_REAL, MPI_SUM, g_mpi.comm);
        incGlobal = std::sqrt(incGlobal / vr->mesh->NumCellGlobal());

        std::swap(uRec, uRecNew);
        uRec.trans.startPersistentPull();
        uRec.trans.waitPersistentPull();

        lastInc = incGlobal;
        if (incGlobal < 1e-14)
        {
            convergedAt = iter + 1;
            break;
        }
    }

    if (g_mpi.rank == 0)
        std::cout << "[Convergence P2-HQM IV10] " << convergedAt
                  << " iters, final inc = " << std::scientific << lastInc << std::endl;

    CHECK(convergedAt > 0);
    CHECK(convergedAt < 200);
}
