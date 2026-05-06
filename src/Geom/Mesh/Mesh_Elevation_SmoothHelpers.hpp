#pragma once

#include "Mesh.hpp"
#include "Geom/PointCloud.hpp"
#include <nanoflann.hpp>
#include <optional>
#include <unordered_set>

namespace DNDS::Geom
{
    // =================================================================
    // CoordPairDOF: extends tCoordPair with dot/norm2/addTo/setConstant
    // for compatibility with Linear::GMRES_LeftPreconditioned.
    // =================================================================
    // Value-semantic class: all members are value types (ssp, TTrans);
    // = default for all special members per rule of five.
    struct CoordPairDOF : public tCoordPair
    {
        CoordPairDOF() = default;
        ~CoordPairDOF() = default;
        CoordPairDOF(const CoordPairDOF &) = default;
        CoordPairDOF(CoordPairDOF &&) = default;
        CoordPairDOF &operator=(const CoordPairDOF &) = default;
        CoordPairDOF &operator=(CoordPairDOF &&) = default;

        real dot(CoordPairDOF &R)
        {
            real ret = 0;
            for (index i = 0; i < this->father->Size(); i++)
                ret += (*this)[i].dot(R[i]);
            real retSum = UnInitReal;
            MPI::Allreduce(&ret, &retSum, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return retSum;
        }

        real norm2() { return std::sqrt(this->dot(*this)); }

        void addTo(CoordPairDOF &R, real alpha)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] += R[i] * alpha;
        }

        void setConstant(real v)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i].setConstant(v);
        }

        void operator=(CoordPairDOF &R)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] = R[i];
        }

        void operator*=(real r)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] *= r;
        }
    };

    // =================================================================
    // Shared setup result for all smooth solver variants.
    // =================================================================
    struct SmoothSolverSetup
    {
        std::unordered_set<index> nodesBoundInterpolated;
        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
    };

    /**
     * \brief Common preamble for all smooth solver variants.
     *
     * Checks state assertions, identifies boundary-interpolated nodes,
     * gathers boundary coordinates and displacement values, and sets up
     * MPI ghost communication so every rank has a complete copy of the
     * boundary interpolation data.
     *
     * \return std::nullopt if nTotalMoved == 0 (nothing to do).
     */
    inline std::optional<SmoothSolverSetup> PrepareSmoothSolverSetup(
        UnstructuredMesh &mesh)
    {
        DNDS_assert(mesh.elevState == Elevation_O1O2);
        DNDS_assert(mesh.adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(mesh.cell2node.isLocal() && mesh.bnd2node.isLocal());
        DNDS_assert(mesh.adjFacialState == Adj_PointToLocal);
        DNDS_assert(mesh.face2cell.isLocal() && mesh.face2node.isLocal());
        DNDS_assert(mesh.adjC2FState == Adj_PointToLocal);
        DNDS_assert(mesh.cell2face.isLocal() && mesh.bnd2face.isLocal());
        DNDS_assert(mesh.face2node.father);
        DNDS_assert(mesh.nTotalMoved >= 0);
        if (!mesh.nTotalMoved)
        {
            if (mesh.mpi.rank == mesh.mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() "
                         "early exit for no nodes were moved";
            return std::nullopt;
        }

        SmoothSolverSetup setup;

        for (index iN = 0; iN < mesh.coords.father->Size(); iN++)
        {
            if (mesh.coordsElevDisp[iN](0) != largeReal ||
                mesh.coordsElevDisp[iN](2) == 2 * largeReal)
            {
                setup.nodesBoundInterpolated.insert(iN);
            }
        }

        setup.boundInterpCoo.InitPair("SmoothSolverSetup::boundInterpCoo", mesh.mpi);
        setup.boundInterpVal.InitPair("SmoothSolverSetup::boundInterpVal", mesh.mpi);

        setup.boundInterpCoo.father->Resize(setup.nodesBoundInterpolated.size());
        setup.boundInterpVal.father->Resize(setup.nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : setup.nodesBoundInterpolated)
        {
            setup.boundInterpCoo[top] = mesh.coords[iN];
            setup.boundInterpVal[top] =
                (mesh.coordsElevDisp[iN](0) != largeReal)
                    ? tPoint{mesh.coordsElevDisp[iN]}
                    : tPoint::Zero();
            top++;
        }

        setup.boundInterpCoo.father->createGlobalMapping();
        index boundInterpGlobSize = setup.boundInterpCoo.father->globalSize();
        std::vector<index> boundInterpPullIdx(boundInterpGlobSize);
        for (index i = 0; i < boundInterpGlobSize; i++)
            boundInterpPullIdx[i] = i;
        setup.boundInterpCoo.TransAttach();
        setup.boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        setup.boundInterpCoo.trans.createMPITypes();
        setup.boundInterpCoo.trans.pullOnce();

        setup.boundInterpVal.TransAttach();
        setup.boundInterpVal.trans.BorrowGGIndexing(setup.boundInterpCoo.trans);
        setup.boundInterpVal.trans.createMPITypes();
        setup.boundInterpVal.trans.pullOnce();

        return setup;
    }

    // =================================================================
    // KD-tree point cloud adapter for tCoord (shared).
    // =================================================================
    struct PointCloudKDTreeCoordPair
    {
        tCoord ref;
        PointCloudKDTreeCoordPair(tCoord &v) : ref(v) {}

        [[nodiscard]] size_t kdtree_get_point_count() const
        {
            DNDS_assert(ref);
            return ref->Size();
        }
        [[nodiscard]] real kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            DNDS_assert(ref);
            return ref->operator[](idx)(dim);
        }
        template <class BBOX>
        bool kdtree_get_bbox(BBOX &bbox) const { return false; }
    };

} // namespace DNDS::Geom
