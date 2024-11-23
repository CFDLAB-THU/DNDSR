#ifndef __DNDS_REALLY_COMPILING__
#define __DNDS_REALLY_COMPILING__
#define __DNDS_REALLY_COMPILING__HEADER_ON__
#endif
#include "Euler/EulerSolver.hpp"
#ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
#undef __DNDS_REALLY_COMPILING__
#endif

namespace DNDS::Euler
{
    template <class JType, int bDim,
              class enable = std::enable_if_t<
                  std::is_same_v<JType, JacobianLocalLDLT<bDim>> ||
                  std::is_same_v<JType, JacobianLocalLU<bDim>>>>
    void jacobiLUTest(MPIInfo &mpi, bool useMesh, int N)
    {
        DNDS_assert(mpi.size == 1);

        ssp<Geom::UnstructuredMesh> mesh;
        DNDS_MAKE_SSP(mesh, mpi, 2);
        auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
        mesh->periodicInfo.translation[1] *= 10;
        mesh->periodicInfo.translation[2] *= 10;
        reader.ReadFromCGNSSerial("../data/mesh/IV10_10.cgns");
        reader.Deduplicate1to1Periodic();
        reader.BuildCell2Cell();
        reader.MeshPartitionCell2Cell(DNDS::Geom::UnstructuredMeshSerialRW::PartitionOptions{});
        reader.PartitionReorderToMeshCell2Cell();
        reader.BuildSerialOut();
        mesh->BuildGhostPrimary();
        mesh->AdjGlobal2LocalPrimary();
        mesh->InterpolateFace(); // this mesh building is copied from meshSerial_Test

        Direct::DirectPrecControl control;
        control.useDirectPrec = true;
        control.iluCode = -1;
        ssp<Direct::SerialSymLUStructure> symLU;
        std::vector<std::vector<index>> cell2cellFaceV;
        if (useMesh)
        {
            N = mesh->NumCell();
            DNDS_MAKE_SSP(symLU, mesh->getMPI(), mesh->NumCell());
            mesh->ObtainLocalFactFillOrdering(*symLU, control);
            mesh->ObtainSymmetricSymbolicFactorization(*symLU, control);
            cell2cellFaceV = mesh->GetCell2CellFaceVLocal();
        }
        else
        {
            DNDS_MAKE_SSP(symLU, mpi, N);
            cell2cellFaceV.resize(N);
            for (index i = 0; i < cell2cellFaceV.size(); i++)
            {
                if (i < N - 1)
                    cell2cellFaceV[i].push_back((i + 1) % N);
                if (i > 0)
                    cell2cellFaceV[i].push_back((i - 1) % N);
            }
            symLU->ObtainSymmetricSymbolicFactorization(cell2cellFaceV, control.getILUCode()); // no ordering
        }

        JType J(symLU, bDim);

        J.setZero();
        //
        for (index iCell = 0; iCell < N; iCell++)
        {
            index iCellP = symLU->FillingReorderOld2New(iCell);
            J.GetDiag(iCell).setIdentity();
            J.GetDiag(iCell) *= 0.5;
            J.GetDiag(iCell) += decltype(J.GetDiag(iCell))::Constant(0.5);
            for (auto iCO : cell2cellFaceV[iCell])
            {
                index iCOP = symLU->FillingReorderOld2New(iCO);
                if (iCOP < iCellP)
                {
                    auto ret = std::lower_bound(
                        symLU->lowerTriStructureNew[iCellP].begin(),
                        symLU->lowerTriStructureNew[iCellP].end(), iCOP);
                    DNDS_assert(ret != symLU->lowerTriStructureNew[iCellP].end());
                    J.GetLower(iCell, ret - symLU->lowerTriStructureNew[iCellP].begin()).setConstant(.1);
                }
                if constexpr (std::is_same_v<JType, JacobianLocalLU<bDim>>) // LDLT no need upper part
                    if (iCOP > iCellP)
                    {
                        auto ret = std::lower_bound(
                            symLU->upperTriStructureNew[iCellP].begin(),
                            symLU->upperTriStructureNew[iCellP].end(), iCOP);
                        DNDS_assert(ret != symLU->upperTriStructureNew[iCellP].end());
                        J.GetUpper(iCell, ret - symLU->upperTriStructureNew[iCellP].begin()).setConstant(-.1);
                    }
            }
        }

        ArrayDOFV<bDim> solAcc, sol, b;
        DNDS_MAKE_SSP(solAcc.father, mpi);
        DNDS_MAKE_SSP(solAcc.son, mpi);
        DNDS_MAKE_SSP(sol.father, mpi);
        DNDS_MAKE_SSP(sol.son, mpi);
        DNDS_MAKE_SSP(b.father, mpi);
        DNDS_MAKE_SSP(b.son, mpi);

        solAcc.father->Resize(N, bDim, 1);
        sol.father->Resize(N, bDim, 1);
        b.father->Resize(N, bDim, 1);
        // solAcc.setConstant(1.0);
        for (index i = 0; i < b.Size(); i++)
            solAcc[i].setConstant(real(i) / real(mesh->NumCell()));
        // solAcc[i].setConstant(1);
        J.MatMul(solAcc, b);
        // for (index i = 0; i < b.Size(); i++)
        //     std::cout << b[i].transpose() << ", ";
        // std::cout << std::endl;

        double t0 = MPI_Wtime();
        J.InPlaceDecompose();
        std::cout << "Decomposing finished for " << std::scientific << (MPI_Wtime() - t0) << std::endl;
        // J.PrintLog();
        t0 = MPI_Wtime();
        J.Solve(b, sol);
        std::cout << "Solving finished for " << std::scientific << (MPI_Wtime() - t0) << std::endl;
        std::cout << "solved" << std::endl;
        solAcc -= sol;
        std::cout << "Error is " << std::scientific << std::setprecision(10) << solAcc.norm2() << std::endl;

        // for (index i = 0; i < b.Size(); i++)
        //     std::cout << sol[i].transpose() << ", ";
        // std::cout << std::endl;
        std::cout << "\n\n\n"
                  << std::endl;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    DNDS::Euler::jacobiLUTest<DNDS::Euler::JacobianLocalLDLT<5>, 5>(mpi, true, 10);
    DNDS::Euler::jacobiLUTest<DNDS::Euler::JacobianLocalLU<5>, 5>(mpi, true, 10);
    MPI_Finalize();
}
