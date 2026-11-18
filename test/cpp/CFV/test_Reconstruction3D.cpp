/**
 * @file test_Reconstruction3D.cpp
 * @brief Regression tests for CFV VariationalReconstruction<3> (3D).
 *
 * Exercises the full 3D VR pipeline on a periodic hex mesh:
 *   - Mesh reading, partitioning, ghost building
 *   - VR construction: metrics, base+weight, rec coeff
 *   - Polynomial exactness: constant (P0) for all methods
 *   - Iterative VFV convergence and error measurement
 *
 * Uses the Uniform32_3D_Periodic.cgns mesh (32^3 = 32768 hex cells,
 * periodic on [0,1]^3).
 *
 * Golden values are acquired on first run and must be captured manually.
 * The test verifies the 3D template instantiation works end-to-end.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "CFV/Legacy/VariationalReconstruction.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "ReconstructionFixture.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <functional>
#include <nlohmann/json.hpp>

using namespace DNDS;
using namespace DNDS::Geom;

static constexpr int g_dim = 3;
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

static ssp<UnstructuredMesh> buildMesh3D(
    const std::string &file,
    DNDS::real Lx, DNDS::real Ly, DNDS::real Lz)
{
    NativePeriodicImportOptions periodic;
    periodic.frames.translation[1].map() = tPoint{Lx, 0, 0};
    periodic.frames.translation[2].map() = tPoint{0, Ly, 0};
    periodic.frames.translation[3].map() = tPoint{0, 0, Lz};
    PartitionOptions partition;
    partition.metisSeed = 42;
    AutoAppendName2ID names;
    auto mesh = ReadNativeMeshFromCGNS(g_mpi, g_dim, meshPath(file), partition, [&](const std::string &name)
                                       { return names(name); }, periodic);
    mesh->AssertOnFaces();
    return mesh;
}

// ===================================================================
// VR builder
// ===================================================================
enum class RecMethod3D
{
    GaussGreen,
    VFV_P1_HQM,
    VFV_P2_HQM,
};

static int recMethodOrder(RecMethod3D m)
{
    switch (m)
    {
    case RecMethod3D::GaussGreen:
        return 1;
    case RecMethod3D::VFV_P1_HQM:
        return 1;
    case RecMethod3D::VFV_P2_HQM:
        return 2;
    }
    return 1;
}

static ssp<tVR> buildVR3D(ssp<UnstructuredMesh> mesh, RecMethod3D method)
{
    auto vr = std::make_shared<tVR>(g_mpi, mesh);

    CFV::VRSettings defaultSettings(g_dim);
    nlohmann::ordered_json j;
    defaultSettings.WriteIntoJson(j);

    int order = recMethodOrder(method);
    j["maxOrder"] = order;
    j["intOrder"] = std::max(order + 2, 5);
    j["cacheDiffBase"] = true;
    j["SORInstead"] = false;
    j["jacobiRelax"] = 1.0;

    if (method == RecMethod3D::GaussGreen)
    {
        j["subs2ndOrder"] = 1;
    }
    else
    {
        j["subs2ndOrder"] = 0;
        j["functionalSettings"]["dirWeightScheme"] = "HQM_OPT";
        j["functionalSettings"]["geomWeightScheme"] = "HQM_SD";
        j["functionalSettings"]["geomWeightPower"] = 0.5;
        j["functionalSettings"]["geomWeightBias"] = 1.0;
    }

    vr->parseSettings(j);
    if (mesh->IsPeriodic())
        vr->SetPeriodicTransformations();
    vr->ConstructMetrics();
    vr->ConstructBaseAndWeight();
    vr->ConstructRecCoeff();
    return vr;
}

// ===================================================================
// Boundary callback (periodic -- not needed, but required by interface)
// ===================================================================
using ScalarFunc3D = std::function<DNDS::real(const tPoint &)>;

static tVR::TFBoundary<g_nv> g_zeroBC =
    [](const auto &, const auto &, DNDS::index, DNDS::index, int,
       const tPoint &, const tPoint &, t_index)
{ return Eigen::Vector<DNDS::real, g_nv>::Zero(); };

// ===================================================================
// Core: run reconstruction and measure L1 error at quadrature points,
// normalized by domain volume.
// ===================================================================
struct ReconstructionRun3D
{
    int iterations = 0;
    DNDS::real lastUpdate = veryLargeReal;
};

static DNDS::real runTest3D(
    ssp<tVR> vr,
    RecMethod3D method,
    const ScalarFunc3D &exactFunc,
    const tVR::TFBoundary<g_nv> &bc,
    int maxIters,
    DNDS::real convTol,
    bool printProgress,
    ReconstructionRun3D *stats = nullptr)
{
    if (stats)
        *stats = {};
    auto mesh = vr->mesh;

    CFV::tUDof<g_nv> u;
    vr->BuildUDof(u, 1);

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

    if (method == RecMethod3D::GaussGreen)
    {
        CFV::tUGrad<g_nv, g_dim> uGrad;
        vr->BuildUGrad(uGrad, 1);
        vr->DoReconstruction2ndGrad<g_nv>(uGrad, u, bc, 1);
        uGrad.trans.startPersistentPull();
        uGrad.trans.waitPersistentPull();

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
        CFV::tURec<g_nv> uRec, uRecNew;
        vr->BuildURec(uRec, 1);
        vr->BuildURec(uRecNew, 1);

        DNDS::real lastInc = veryLargeReal;
        for (int iter = 0; iter < maxIters; iter++)
        {
            vr->DoReconstructionIter<g_nv>(
                uRec, uRecNew, u, bc, true);

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
            if (stats)
                *stats = {iter + 1, incGlobal};
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
// Prebuilt mesh
// ===================================================================
static ssp<UnstructuredMesh> g_mesh3d;
static ssp<UnstructuredMesh> LargeMesh()
{
    // Laziness matters: doctest filters and the small contract registration
    // must not import/partition the 32^3 integration fixture.
    if (!g_mesh3d)
        g_mesh3d = buildMesh3D("Uniform32_3D_Periodic.cgns", 1.0, 1.0, 1.0);
    return g_mesh3d;
}

int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    g_mpi.setWorld();

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    g_mesh3d.reset();
    DNDS::MPI::Finalize();
    return res;
}

// ===================================================================
// Test functions (periodic on [0,1]^3)
// ===================================================================
static const DNDS::real g_L = 1.0;
static const DNDS::real g_k = 2.0 * pi / g_L;

static ScalarFunc3D sinCos3D = [](const tPoint &p)
{ return std::sin(g_k * p[0]) * std::cos(g_k * p[1]) * std::cos(g_k * p[2]); };

static ScalarFunc3D cosPlusCos3D = [](const tPoint &p)
{ return std::cos(g_k * p[0]) + std::cos(g_k * p[1]) + std::cos(g_k * p[2]); };

// ===================================================================
// Sentinel for golden values not yet acquired: use 1e300.
// When golden == 1e300, only print the error (no regression check).
// ===================================================================
static constexpr DNDS::real GOLDEN_NOT_ACQUIRED = 1e300;

// ===================================================================
// CONSTANT EXACTNESS: all methods should reproduce a constant exactly
// ===================================================================

TEST_CASE("3D: constant exactness with GaussGreen" * doctest::test_suite("integration"))
{
    auto vr = buildVR3D(LargeMesh(), RecMethod3D::GaussGreen);
    ScalarFunc3D f = [](const tPoint &)
    { return 1.0; };
    DNDS::real err = runTest3D(vr, RecMethod3D::GaussGreen, f, g_zeroBC, 1, 0, false);
    if (g_mpi.rank == 0)
        std::cout << "[3D/const/GG] err = " << std::scientific << err << std::endl;
    CHECK(err < 1e-12);
}

TEST_CASE("3D static reconstruction mask is exchanged and survives GG and VR updates" * doctest::test_suite("integration"))
{
    auto vr = buildVR3D(LargeMesh(), RecMethod3D::VFV_P2_HQM);
    CFV::tUDof<1> u, mask;
    std::vector<DNDS::index> selectedCells;
    CFV::tURec<1> rec, scratch;
    vr->BuildUDof(u, 1);
    vr->BuildUDof(mask, 1);
    vr->BuildURec(rec, 1);
    vr->BuildURec(scratch, 1);
    for (DNDS::index i = 0; i < LargeMesh()->NumCell(); ++i)
    {
        auto p = vr->GetCellBary(i);
        mask[i](0) = p(0) > 0.6 && p(2) < 0.4;
    }
    mask.trans.pullOnce();
    for (DNDS::index i = 0; i < mask.Size(); ++i)
        if (mask[i](0))
            selectedCells.push_back(i);
    const CFV::ReconstructionMaskView reconstructionMask{&mask, &selectedCells};
    for (DNDS::index i = 0; i < u.Size(); ++i)
    {
        auto p = vr->GetCellBary(i);
        const bool expected = p(0) > 0.6 && p(2) < 0.4;
        CHECK(reconstructionMask.IsMasked(i) == expected); // Includes ghost cells.
        u[i](0) = 1 + std::sin(2 * pi * p(0)) * std::cos(2 * pi * p(2));
    }
    rec.setConstant(7);
    for (int k = 0; k < 3; ++k)
    {
        vr->DoReconstructionIter<1>(rec, scratch, u, g_zeroBC, false, false, false, reconstructionMask);
        rec.trans.pullOnce();
        for (DNDS::index i = 0; i < rec.Size(); ++i)
            if (reconstructionMask.IsMasked(i))
                CHECK(rec[i].isZero(0));
    }
    vr->DoReconstruction2nd<1>(rec, u, g_zeroBC, 1, {}, reconstructionMask);
    rec.trans.pullOnce();
    for (DNDS::index i = 0; i < rec.Size(); ++i)
        if (reconstructionMask.IsMasked(i))
            CHECK(rec[i].isZero(0));
}

TEST_CASE("3D: constant exactness with VFV_P1_HQM" * doctest::test_suite("integration"))
{
    auto vr = buildVR3D(LargeMesh(), RecMethod3D::VFV_P1_HQM);
    ScalarFunc3D f = [](const tPoint &)
    { return 1.0; };
    DNDS::real err = runTest3D(vr, RecMethod3D::VFV_P1_HQM, f, g_zeroBC, 100, 1e-15, false);
    if (g_mpi.rank == 0)
        std::cout << "[3D/const/P1-HQM] err = " << std::scientific << err << std::endl;
    CHECK(err < 1e-12);
}

TEST_CASE("3D: constant exactness with VFV_P2_HQM" * doctest::test_suite("integration"))
{
    auto vr = buildVR3D(LargeMesh(), RecMethod3D::VFV_P2_HQM);
    ScalarFunc3D f = [](const tPoint &)
    { return 1.0; };
    DNDS::real err = runTest3D(vr, RecMethod3D::VFV_P2_HQM, f, g_zeroBC, 200, 1e-15, false);
    if (g_mpi.rank == 0)
        std::cout << "[3D/const/P2-HQM] err = " << std::scientific << err << std::endl;
    CHECK(err < 1e-12);
}

// ===================================================================
// Golden-value regression table for 3D smooth functions.
// Sentinel GOLDEN_NOT_ACQUIRED (1e300) means "not yet captured".
// ===================================================================

struct Test3DCase
{
    RecMethod3D method;
    ScalarFunc3D func;
    const char *funcName;
    int maxIters;
    DNDS::real convTol;
    DNDS::real golden; // golden L1/vol (1e300 = not acquired)
};

static const char *recMethodName3D(RecMethod3D m)
{
    switch (m)
    {
    case RecMethod3D::GaussGreen:
        return "GG";
    case RecMethod3D::VFV_P1_HQM:
        return "P1-HQM";
    case RecMethod3D::VFV_P2_HQM:
        return "P2-HQM";
    }
    return "?";
}

static Test3DCase g_3dTests[] = {
    // Keep equal methods adjacent so they reuse the prepared operator.
    {RecMethod3D::GaussGreen, sinCos3D, "sinCos3D", 1, 0,
     1.5098818118e-03},
    {RecMethod3D::GaussGreen, cosPlusCos3D, "cos+cos3D", 1, 0,
     1.4931684237e-03},
    {RecMethod3D::VFV_P1_HQM, sinCos3D, "sinCos3D", 200, 1e-14,
     3.2475021825e-03},
    {RecMethod3D::VFV_P1_HQM, cosPlusCos3D, "cos+cos3D", 200, 1e-14,
     2.4261881197e-03},
    {RecMethod3D::VFV_P2_HQM, sinCos3D, "sinCos3D", 200, 1e-14,
     7.4823241394e-05},
    {RecMethod3D::VFV_P2_HQM, cosPlusCos3D, "cos+cos3D", 200, 1e-14,
     3.7014576777e-05},
};

static const int g_n3dTests = sizeof(g_3dTests) / sizeof(g_3dTests[0]);

TEST_CASE("3D: smooth-function golden values, convergence and order" * doctest::test_suite("integration"))
{
    DNDS::real p1Error = 0, p2Error = 0;
    ssp<tVR> vr;
    for (int ti = 0; ti < g_n3dTests; ++ti)
    {
        const auto &tc = g_3dTests[ti];
        const std::string label = std::string("3D/") + tc.funcName + "/" + recMethodName3D(tc.method);
        INFO(label);
        if (ti == 0 || g_3dTests[ti - 1].method != tc.method)
        {
            vr.reset();
            vr = buildVR3D(LargeMesh(), tc.method);
        }
        ReconstructionRun3D stats;
        const auto error = runTest3D(vr, tc.method, tc.func, g_zeroBC, tc.maxIters, tc.convTol, false, &stats);
        if (g_mpi.rank == 0)
            std::cout << "[" << label << "] err = " << std::scientific
                      << std::setprecision(10) << error << std::endl;
        CHECK(error > 0);
        CHECK(std::isfinite(error));
        if (tc.golden < GOLDEN_NOT_ACQUIRED)
            CHECK(error == doctest::Approx(tc.golden).epsilon(1e-6));
        if (std::string(tc.funcName) == "sinCos3D")
        {
            if (tc.method == RecMethod3D::VFV_P1_HQM)
            {
                p1Error = error;
                CHECK(stats.iterations > 0);
                CHECK(stats.iterations < tc.maxIters);
                CHECK(stats.lastUpdate < tc.convTol);
            }
            if (tc.method == RecMethod3D::VFV_P2_HQM)
                p2Error = error;
        }
    }
    REQUIRE(p1Error > 0);
    REQUIRE(p2Error > 0);
    CHECK(p2Error < p1Error);
}

// ===================================================================
// Metric sanity: cell volumes sum to domain volume
// ===================================================================

TEST_CASE("3D: cell volumes sum to 1.0 (unit cube)" * doctest::test_suite("integration"))
{
    auto vr = buildVR3D(LargeMesh(), RecMethod3D::GaussGreen);
    DNDS::real globalVol = vr->GetGlobalVol();
    if (g_mpi.rank == 0)
        std::cout << "[3D] globalVol = " << std::setprecision(15) << globalVol << std::endl;
    CHECK(globalVol == doctest::Approx(1.0).epsilon(1e-10));
}

TEST_CASE("Small connected 3D cells preserve constants through GG and P1/P2 VR")
{
    const auto mesh = CFVTest::CartesianHexes(g_mpi, 4);
    const ScalarFunc3D constant = [](const tPoint &)
    { return 1.; };
    const tVR::TFBoundary<g_nv> boundary = [](const auto &, const auto &, DNDS::index, DNDS::index, int,
                                              const tPoint &, const tPoint &, t_index)
    { return Eigen::Vector<DNDS::real, g_nv>::Ones(); };
    for (const auto method : {RecMethod3D::GaussGreen, RecMethod3D::VFV_P1_HQM, RecMethod3D::VFV_P2_HQM})
    {
        CAPTURE(recMethodName3D(method));
        auto vr = buildVR3D(mesh, method);
        const auto error = runTest3D(vr, method, constant, boundary, 200, 1e-15, false);
        CHECK(std::isfinite(error));
        CHECK(error < 1e-12);
    }
}
