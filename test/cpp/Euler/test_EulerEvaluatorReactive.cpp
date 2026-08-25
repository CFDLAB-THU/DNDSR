/**
 * @file test_EulerEvaluatorReactive.cpp
 * @brief Reactive EulerEvaluator cell-mean repair tests.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Euler/EulerSolver.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

using namespace DNDS;
using namespace DNDS::Euler;

static MPIInfo g_mpi;

static std::string projectRoot()
{
    std::string f(__FILE__);
    for (int i = 0; i < 4; ++i)
    {
        auto pos = f.rfind('/');
        if (pos == std::string::npos)
            pos = f.rfind('\\');
        if (pos != std::string::npos)
            f = f.substr(0, pos);
    }
    return f;
}

static std::string writeReactiveConfig()
{
    constexpr int nVars = 14;
    const std::string root = projectRoot();
    const auto tmpDir = std::filesystem::temp_directory_path() / "dndsr_euler_test";
    std::filesystem::create_directories(tmpDir);
    const std::string defaultPath = (tmpDir / "reactive_repair_default.json").string();
    const std::string configPath = (tmpDir / "reactive_repair.json").string();

    {
        EulerSolver<NS_EX> solver(g_mpi, nVars);
        solver.ConfigureFromJson(defaultPath, false);
    }
    MPI::Barrier(g_mpi.comm);

    auto fDefault = std::ifstream(defaultPath);
    REQUIRE(fDefault);
    auto config = nlohmann::ordered_json::parse(fDefault, nullptr, true, true);

    auto fCase = std::ifstream(root + "/cases/eulerEX/react_test.json");
    REQUIRE(fCase);
    auto caseConfig = nlohmann::ordered_json::parse(fCase, nullptr, true, true);
    config.merge_patch(caseConfig);

    config["dataIOControl"]["meshFile"] = root + "/data/mesh/IV10_10.cgns";
    config["eulerSettings"]["reactiveFlow"]["mechanismFile"] = root + "/cases/eulerEX/h2o2.yaml";
    config["outputControl"]["dataOutAtInit"] = false;
    config["outputControl"]["nDataOut"] = 1000000;
    config["outputControl"]["nDataOutC"] = 1000000;

    if (g_mpi.rank == 0)
    {
        auto fOut = std::ofstream(configPath);
        REQUIRE(fOut);
        fOut << std::setw(4) << config;
    }
    MPI::Barrier(g_mpi.comm);
    return configPath;
}

TEST_CASE("EulerEvaluator repairs reactive cell-mean species")
{
    constexpr int nVars = 14;
    EulerSolver<NS_EX> solver(g_mpi, nVars);
    solver.ConfigureFromJson(writeReactiveConfig(), true);
    solver.ReadMeshAndInitialize();

    auto &eval = solver.getEval();
    auto &u = solver.getU();
    auto mesh = solver.getMesh();
    eval.InitializeUDOF(u);

    const int Ns = eval.phys().nSpecies();
    const int Ns1 = Ns - 1;
    const int Isp = nVars - Ns1;
    REQUIRE(Ns == 10);
    REQUIRE(Isp == 5);

    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        auto uCell = u[iCell];
        const real rho = uCell(0);
        uCell(Isp + 1) = -rho * 1e-12;
        uCell(Isp + 8) = rho * (0.75 + 4.3811e-10);
    }

    eval.RepairCellMeanState(u);

    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        auto uCell = u[iCell];
        const real rho = uCell(0);
        real sumRhoY = 0;
        for (int k = 0; k < Ns1; ++k)
        {
            CHECK(std::isfinite(uCell(Isp + k)));
            CHECK(uCell(Isp + k) > 0);
            sumRhoY += uCell(Isp + k);
        }
        CHECK(sumRhoY < rho);
        CHECK((rho - sumRhoY) / rho > 1e-15);

        std::vector<double> Y(static_cast<size_t>(Ns));
        Chemistry::ConstSpeciesBufferView rhoYView{uCell.data() + Isp, Ns1};
        Chemistry::SpeciesBufferView YView{Y.data(), Ns};
        CHECK_NOTHROW(eval.phys().chem().massFractions(
            rho, rhoYView, YView));
        CHECK(Y.back() > 0);

        const real T = eval.phys().temperature(uCell);
        CHECK(std::isfinite(T));
        CHECK(eval.phys().toPhysT(T) >= eval.phys().temperatureFloor());
    }

    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        u[iCell](0) = 0;
    CHECK_THROWS_AS(eval.RepairCellMeanState(u), std::runtime_error);

    eval.InitializeUDOF(u);
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        u[iCell](4) = std::numeric_limits<real>::quiet_NaN();
    CHECK_THROWS_AS(eval.RepairCellMeanState(u), std::runtime_error);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();

    MPI_Finalize();
    return result;
}
