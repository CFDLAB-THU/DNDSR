#pragma once
/// @file DOFFactory.hpp
/// @brief Free-function DOF array builders extracted from FiniteVolume.
///
/// These build cell/face/node DOF arrays with ghost communication setup.
/// FiniteVolume::BuildUDof and BuildUGradD delegate to these.

#include "VRDefines.hpp"
#include "Geom/Mesh.hpp"

namespace DNDS::CFV
{
    /// @brief Build a DOF array pair matching a mesh location (cell/face/node).
    ///
    /// Creates father+son arrays, optionally sets up ghost communication
    /// by borrowing the indexing from the mesh's corresponding adjacency.
    template <int nVarsFixed = 1>
    void BuildUDofOnMesh(
        tUDof<nVarsFixed> &u,
        const std::string &name,
        const MPIInfo &mpi,
        const ssp<Geom::UnstructuredMesh> &mesh,
        int nVars,
        bool buildSon = true,
        bool buildTrans = true,
        Geom::MeshLoc varloc = Geom::MeshLoc::Cell)
    {
        u.InitPair(name, mpi);
        DNDS_assert(varloc);
        switch (varloc)
        {
        case Geom::MeshLoc::Cell:
            u.father->Resize(mesh->NumCell(), nVars, 1);
            break;
        case Geom::MeshLoc::Node:
            u.father->Resize(mesh->NumNode(), nVars, 1);
            break;
        case Geom::MeshLoc::Face:
            u.father->Resize(mesh->NumFace(), nVars, 1);
            break;
        default:
            DNDS_assert(false);
        }
        if (buildSon)
            switch (varloc)
            {
            case Geom::MeshLoc::Cell:
                u.son->Resize(mesh->NumCellGhost(), nVars, 1);
                break;
            case Geom::MeshLoc::Node:
                u.son->Resize(mesh->NumNodeGhost(), nVars, 1);
                break;
            case Geom::MeshLoc::Face:
                u.son->Resize(mesh->NumFaceGhost(), nVars, 1);
                break;
            default:
                DNDS_assert(false);
            }

        if (buildTrans)
        {
            DNDS_assert(buildSon);
            u.TransAttach();
            switch (varloc)
            {
            case Geom::MeshLoc::Cell:
                u.trans.BorrowGGIndexing(mesh->cell2node.trans);
                break;
            case Geom::MeshLoc::Node:
                u.trans.BorrowGGIndexing(mesh->coords.trans);
                break;
            case Geom::MeshLoc::Face:
                u.trans.BorrowGGIndexing(mesh->face2node.trans);
                break;
            default:
                DNDS_assert(false);
            }
            u.trans.createMPITypes();
            u.trans.initPersistentPull();
            u.trans.initPersistentPush();
        }

        for (index iCell = 0; iCell < u.Size(); iCell++)
            u[iCell].setZero();
    }

    /// @brief Build a gradient DOF array pair matching a mesh location.
    template <int nVarsFixed, int dim>
    void BuildUGradDOnMesh(
        tUGrad<nVarsFixed, dim> &u,
        const std::string &name,
        const MPIInfo &mpi,
        const ssp<Geom::UnstructuredMesh> &mesh,
        int nVars,
        bool buildSon = true,
        bool buildTrans = true,
        Geom::MeshLoc varloc = Geom::MeshLoc::Cell)
    {
        u.InitPair(name, mpi);
        switch (varloc)
        {
        case Geom::MeshLoc::Cell:
            u.father->Resize(mesh->NumCell(), dim, nVars);
            break;
        case Geom::MeshLoc::Node:
            u.father->Resize(mesh->NumNode(), dim, nVars);
            break;
        case Geom::MeshLoc::Face:
            u.father->Resize(mesh->NumFace(), dim, nVars);
            break;
        default:
            DNDS_assert(false);
        }
        if (buildSon)
            switch (varloc)
            {
            case Geom::MeshLoc::Cell:
                u.son->Resize(mesh->NumCellGhost(), dim, nVars);
                break;
            case Geom::MeshLoc::Node:
                u.son->Resize(mesh->NumNodeGhost(), dim, nVars);
                break;
            case Geom::MeshLoc::Face:
                u.son->Resize(mesh->NumFaceGhost(), dim, nVars);
                break;
            default:
                DNDS_assert(false);
            }
        if (buildTrans)
        {
            DNDS_assert(buildSon);
            u.TransAttach();
            switch (varloc)
            {
            case Geom::MeshLoc::Cell:
                u.trans.BorrowGGIndexing(mesh->cell2node.trans);
                break;
            case Geom::MeshLoc::Node:
                u.trans.BorrowGGIndexing(mesh->coords.trans);
                break;
            case Geom::MeshLoc::Face:
                u.trans.BorrowGGIndexing(mesh->face2node.trans);
                break;
            default:
                DNDS_assert(false);
            }
            u.trans.createMPITypes();
            u.trans.initPersistentPull();
            u.trans.initPersistentPush();
        }

        for (index iCell = 0; iCell < u.Size(); iCell++)
            u[iCell].setZero();
    }
}
