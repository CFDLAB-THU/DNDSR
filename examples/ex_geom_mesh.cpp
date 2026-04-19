/**
 * @file ex_geom_mesh.cpp
 * @brief Compilable examples from the Geom Usage Guide.
 *
 * Demonstrates: mesh building from CGNS, element access, node
 * coordinates, face traversal, and raw quadrature integration.
 *
 * Build:  cmake --build build -t ex_geom_mesh -j8
 * Run:    mpirun -np 1 build/examples/ex_geom_mesh
 *
 * Requires: data/mesh/Uniform_3x3_wall.cgns
 */

#include "Geom/Mesh.hpp"
#include "Geom/Quadrature.hpp"

#include <fmt/core.h>
#include <iostream>
#include <cmath>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    {
        using namespace DNDS;
        using namespace DNDS::Geom;

        MPIInfo mpi;
        mpi.setWorld();

        int dim = 2;
        auto mesh = std::make_shared<UnstructuredMesh>(mpi, dim);
        UnstructuredMeshSerialRW reader(mesh, 0);

        // ============================================================
        // Phase A: Build the mesh
        // ============================================================
        std::string meshFile = "data/mesh/Uniform_3x3_wall.cgns";
        if (mpi.rank == 0)
            fmt::print("Reading mesh: {}\n", meshFile);

        reader.ReadFromCGNSSerial(meshFile);
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
            fmt::print("Mesh built: {} cells, {} nodes, {} faces\n",
                       mesh->NumCell(), mesh->NumNode(), mesh->NumFaceProc());

        // ============================================================
        // Element access
        // ============================================================
        for (DNDS::index iCell = 0; iCell < std::min(mesh->NumCell(), (DNDS::index)3); iCell++)
        {
            Elem::Element elem = mesh->GetCellElement(iCell);
            fmt::print("  Cell {}: type={}, nNodes={}, dim={}, order={}\n",
                       iCell, (int)elem.type, elem.GetNumNodes(),
                       elem.GetDim(), elem.GetOrder());
        }

        // ============================================================
        // Node coordinates
        // ============================================================
        for (DNDS::index iNode = 0; iNode < std::min(mesh->NumNode(), (DNDS::index)4); iNode++)
        {
            auto p = mesh->coords[iNode];
            fmt::print("  Node {}: ({:.3f}, {:.3f}, {:.3f})\n",
                       iNode, p(0), p(1), p(2));
        }

        // ============================================================
        // Face traversal
        // ============================================================
        DNDS::index nBndFaces = 0;
        for (DNDS::index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            DNDS::index cellL = mesh->face2cell(iFace, 0);
            DNDS::index cellR = mesh->face2cell(iFace, 1);
            if (cellR == UnInitIndex)
                nBndFaces++;
            (void)cellL; // suppress unused
        }
        fmt::print("  Total faces: {}, boundary faces: {}\n",
                   mesh->NumFaceProc(), nBndFaces);

        // ============================================================
        // Raw quadrature: compute cell volumes
        // ============================================================
        real totalVol = 0;
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            Elem::Element elem = mesh->GetCellElement(iCell);
            Elem::Quadrature quad(elem, 3);

            Geom::tSmallCoords cs;
            mesh->GetCoordsOnCell(iCell, cs);

            real cellVol = 0;
            quad.Integration(
                cellVol,
                [&](real &acc, int iG, tPoint pParam, Elem::tD01Nj D01Nj)
                {
                    acc = Elem::CellJacobianDet<2>(cs, D01Nj);
                });
            totalVol += cellVol;

            if (iCell < 3)
                fmt::print("  Cell {} volume: {:.6f}\n", iCell, cellVol);
        }
        fmt::print("  Total volume (rank {}): {:.6f}\n", mpi.rank, totalVol);

        // The 3x3 wall mesh has domain [-1,2]^2 = area 9.
        DNDS_assert(std::abs(totalVol - 9.0) < 1e-10);
        fmt::print("\nAll geom examples passed.\n");
    }
    MPI_Finalize();
    return 0;
}
