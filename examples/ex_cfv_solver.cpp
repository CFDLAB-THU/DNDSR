/**
 * @file ex_cfv_solver.cpp
 * @brief Compilable example: full CFV pipeline from mesh to RHS.
 *
 * Demonstrates: mesh building, FiniteVolume metrics, VR settings,
 * ConstructBaseAndWeight, ConstructRecCoeff, BuildUDof/URec, iterative
 * reconstruction, and RHS evaluation via ModelEvaluator.
 *
 * Build:  cmake --build build -t ex_cfv_solver -j8
 * Run:    mpirun -np 1 build/examples/ex_cfv_solver
 *
 * Requires: data/mesh/Uniform_3x3.cgns (periodic 3x3 mesh)
 */

#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "CFV/ModelEvaluator.hpp"

#include <fmt/core.h>
#include <cmath>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    {
        using namespace DNDS;
        using namespace DNDS::Geom;
        using namespace DNDS::CFV;

        MPIInfo mpi;
        mpi.setWorld();

        constexpr int nVarsFixed = DynamicSize; // ModelEvaluator uses dynamic nVars
        constexpr int nVars = 1;
        constexpr int dim = 2;

        // ============================================================
        // Phase A: Build mesh
        // ============================================================
        auto mesh = std::make_shared<UnstructuredMesh>(mpi, dim);
        UnstructuredMeshSerialRW reader(mesh, 0);

        std::string meshFile = "data/mesh/Uniform2_Periodic.cgns";
        if (mpi.rank == 0)
            fmt::print("Reading mesh: {}\n", meshFile);

        reader.ReadFromCGNSSerial(meshFile);
        reader.Deduplicate1to1Periodic();
        reader.BuildCell2Cell();

        UnstructuredMeshSerialRW::PartitionOptions opts;
        opts.metisSeed = 42;
        reader.MeshPartitionCell2Cell(opts);
        reader.PartitionReorderToMeshCell2Cell();

        mesh->RecoverNode2CellAndNode2Bnd();
        mesh->RecoverCell2CellAndBnd2Cell();
        mesh->BuildGhostPrimary();
        mesh->AdjGlobal2LocalPrimary();
        mesh->InterpolateFace();
        mesh->AssertOnFaces();

        if (mpi.rank == 0)
            fmt::print("Mesh: {} cells, {} nodes, {} faces\n",
                       mesh->NumCell(), mesh->NumNode(), mesh->NumFaceProc());

        // ============================================================
        // Phase B: VR object + settings
        // ============================================================
        auto vfv = std::make_shared<VariationalReconstruction<dim>>(mpi, mesh);

        // Configure VR settings: start from defaults, override what we need.
        VRSettings vrSettings;
        vrSettings.maxOrder = 2;
        vrSettings.intOrder = 5;
        vrSettings.cacheDiffBase = true;
        vrSettings.SORInstead = false;
        vrSettings.jacobiRelax = 1.0;
        vrSettings.subs2ndOrder = 0;
        vrSettings.functionalSettings.dirWeightScheme =
            VRSettings::FunctionalSettings::DirWeightScheme::HQM_OPT;
        vrSettings.functionalSettings.geomWeightScheme =
            VRSettings::FunctionalSettings::GeomWeightScheme::HQM_SD;
        vrSettings.functionalSettings.geomWeightPower = 0.5;
        vrSettings.functionalSettings.geomWeightBias = 1.0;
        vfv->parseSettings(vrSettings);

        // No vector rotation for scalar fields.
        vfv->SetPeriodicTransformations();

        if (mpi.rank == 0)
            fmt::print("VR settings: maxOrder={}, intOrder={}\n",
                       vfv->getSettings().maxOrder, vfv->getSettings().intOrder);

        // ============================================================
        // Phase C: Geometric metrics
        // ============================================================
        vfv->ConstructMetrics();
        if (mpi.rank == 0)
            fmt::print("Metrics constructed. Cell 0 volume: {:.6f}\n",
                       vfv->GetCellVol(0));

        // ============================================================
        // Phase D: Base and weight (periodic mesh: all weights = 1)
        // ============================================================
        vfv->ConstructBaseAndWeight([](Geom::t_index, int) -> real { return 1.0; });
        if (mpi.rank == 0)
            fmt::print("Base and weight constructed.\n");

        // ============================================================
        // Phase E: Reconstruction coefficients
        // ============================================================
        vfv->ConstructRecCoeff();
        if (mpi.rank == 0)
            fmt::print("Reconstruction coefficients constructed.\n");

        // ============================================================
        // Phase F: Build DOF arrays
        // ============================================================
        tUDof<nVarsFixed> u;
        vfv->BuildUDof(u, nVars);

        tURec<nVarsFixed> uRec, uRecNew;
        vfv->BuildURec(uRec, nVars);
        vfv->BuildURec(uRecNew, nVars);

        tUDof<nVarsFixed> rhs;
        vfv->BuildUDof(rhs, nVars);

        if (mpi.rank == 0)
            fmt::print("DOF arrays: u.Size()={}, uRec father rows={}\n",
                       u.Size(), uRec.father->Size());

        // ============================================================
        // Initialize u = sin(x) * cos(y)
        // ============================================================
        for (DNDS::index iCell = 0; iCell < u.father->Size(); iCell++)
        {
            auto bary = vfv->GetCellBary(iCell);
            u[iCell](0) = std::sin(bary(0)) * std::cos(bary(1));
        }
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
        if (mpi.rank == 0)
            fmt::print("u initialized. u[0] = {:.6f}\n", u[0](0));

        // ============================================================
        // Phase G: ModelEvaluator (advection: ax=1, ay=0, sigma=0)
        // ============================================================
        nlohmann::ordered_json evalJson;
        evalJson["ax"] = 1.0;
        evalJson["ay"] = 0.0;
        evalJson["sigma"] = 0.0;
        ModelEvaluator eval(mesh, vfv, evalJson, nVars);
        if (mpi.rank == 0)
            fmt::print("ModelEvaluator created.\n");

        // ============================================================
        // VR iterations + RHS evaluation
        // ============================================================
        uRec.setConstant(0);
        for (int iVR = 0; iVR < 5; iVR++)
        {
            eval.DoReconstructionIter(uRec, uRecNew, u, 0.0,
                                      /*putIntoNew=*/true,
                                      /*recordInc=*/false,
                                      /*uRecIsZero=*/(iVR == 0));
            uRec.father->SwapData(*uRecNew.father);
            uRec.son->SwapData(*uRecNew.son);
        }
        uRec.trans.startPersistentPull();
        uRec.trans.waitPersistentPull();
        if (mpi.rank == 0)
            fmt::print("VR converged. uRec[0](0,0) = {:.6e}\n", uRec[0](0, 0));

        eval.EvaluateRHS(rhs, u, uRec, 0.0);
        real rhsNorm = rhs.norm2();
        if (mpi.rank == 0)
            fmt::print("RHS L2 norm: {:.6e}\n", rhsNorm);

        DNDS_assert(rhsNorm > 0);
        DNDS_assert(std::isfinite(rhsNorm));

        if (mpi.rank == 0)
            fmt::print("\nAll CFV solver examples passed.\n");
    }
    MPI_Finalize();
    return 0;
}
