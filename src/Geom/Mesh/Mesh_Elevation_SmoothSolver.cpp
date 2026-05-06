#include "Mesh.hpp"
#include "Geom/Quadrature.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "Geom/RadialBasisFunction.hpp"

#include <omp.h>
#include <fmt/core.h>
#include <functional>
#include <unordered_set>
#include <optional>
#include <Solver/Linear.hpp>

#include "Geom/PointCloud.hpp"
#include <nanoflann.hpp>

#include "DNDS/EigenPCH.hpp"
#ifdef DNDS_USE_SUPERLU
#    include <superlu_ddefs.h>
#endif

namespace DNDS::Geom
{
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

        real norm2()
        {
            return std::sqrt(this->dot(*this));
        }

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

    struct PointCloudKDTreeCoordPair
    {
        tCoord ref;
        using coord_t = real; //!< The type of each coordinate
        PointCloudKDTreeCoordPair(tCoord &v)
        {
            ref = v;
        }

        // Must return the number of data points
        [[nodiscard]] size_t
        kdtree_get_point_count() const
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
        bool kdtree_get_bbox(BBOX & /* bb */) const
        {
            return false;
        }
    };

    // =================================================================
    // Shared setup for all smooth solver variants
    // =================================================================

    /**
     * \brief Result of the shared smooth solver setup.
     *
     * Populated by PrepareSmoothSolverSetup() and consumed by each
     * smooth solver variant.
     */
    struct SmoothSolverSetup
    {
        std::unordered_set<index> nodesBoundInterpolated{};
        tCoordPair boundInterpCoo{};
        tCoordPair boundInterpVal{};
    };

    /**
     * \brief Common preamble for all smooth solver variants.
     *
     * Checks state assertions, identifies boundary-interpolated nodes,
     * gathers boundary coordinates and displacement values, and sets up
     * MPI ghost communication so every rank has a complete copy of the
     * boundary interpolation data.
     *
     * \param mesh  The mesh being smoothed (must have elevState == Elevation_O1O2,
     *              adjPrimaryState/adjFacialState/adjC2FState == Adj_PointToLocal,
     *              and nTotalMoved >= 0).
     * \return std::nullopt if nTotalMoved == 0 (nothing to do);
     *         otherwise the populated SmoothSolverSetup.
     */
    static std::optional<SmoothSolverSetup> PrepareSmoothSolverSetup(
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
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return std::nullopt;
        }

        SmoothSolverSetup setup;

        // Identify boundary-interpolated nodes: those with a displacement set
        // (coordsElevDisp != largeReal) or marked as O1 nodes (z == 2*largeReal).
        for (index iN = 0; iN < mesh.coords.father->Size(); iN++)
        {
            if (mesh.coordsElevDisp[iN](0) != largeReal ||
                mesh.coordsElevDisp[iN](2) == 2 * largeReal)
            {
                setup.nodesBoundInterpolated.insert(iN);
            }
        }

        // Gather boundary coordinates and displacement values.
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

        // Build global ghost communication: every rank gets the full set.
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

    void UnstructuredMesh::ElevatedNodesSolveInternalSmooth()
    {
        auto setupOpt = PrepareSmoothSolverSetup(*this);
        if (!setupOpt)
            return;
        auto &setup = *setupOpt;
        auto &nodesBoundInterpolated = setup.nodesBoundInterpolated;
        auto &boundInterpCoo = setup.boundInterpCoo;
        auto &boundInterpVal = setup.boundInterpVal;

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (nodesBoundInterpolated.count(iN))
                continue;

            index nFind = elevationInfo.nSearch;
            tPoint cooC = coords[iN];

            std::vector<index> idxFound;
            Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
            idxFound.resize(nFind);
            outDistancesSqr.resize(nFind);

            index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
            DNDS_assert(nFound >= 1);
            idxFound.resize(nFound);
            outDistancesSqr.resize(nFound);
            tSmallCoords coordBnd;
            tSmallCoords dispBnd;
            coordBnd.resize(3, nFound);
            dispBnd.resize(3, nFound);
            for (index iF = 0; iF < nFound; iF++)
            {
                coordBnd(EigenAll, iF) = (*boundInterpCoo.son)[idxFound[iF]];
                dispBnd(EigenAll, iF) = (*boundInterpVal.son)[idxFound[iF]];
            }
            real RMin = std::sqrt(outDistancesSqr.minCoeff());
            tPoint dispC;
            dispC.setZero();

            if (RMin < sqr(elevationInfo.RBFRadius * (KernelIsCompact(elevationInfo.kernel) ? 1 : 5)))
            {
                tPoint coordBndC = coordBnd.rowwise().mean();
                tSmallCoords coordBndRel = coordBnd.colwise() - coordBndC;
                MatrixXR
                    coefs = RBF::RBFInterpolateSolveCoefsNoPoly(coordBndRel, dispBnd.transpose(), elevationInfo.RBFRadius, elevationInfo.kernel);
                tSmallCoords qs;
                qs.resize(3, 1);
                qs = cooC - coordBndC;
                tPoint dCur =
                    (RBF::RBFCPC2(coordBndRel, qs, elevationInfo.RBFRadius, elevationInfo.kernel).transpose() * coefs.topRows(coordBndRel.cols()) +
                     coefs(EigenLast - 3, EigenAll) +
                     qs.transpose() * coefs.bottomRows(3))
                        .transpose();
                dispC = dCur;
            }

            coordsElevDisp[iN] = dispC;
        }

        coordsElevDisp.trans.startPersistentPull();
        coordsElevDisp.trans.waitPersistentPull();

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            // if(dim == 2)
            //     dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            if (coordsElevDisp[iN](0) != largeReal)
                coords[iN] += coordsElevDisp[iN];
        }
        coords.trans.pullOnce();
    }

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV1Old()
    {
        auto setupOpt = PrepareSmoothSolverSetup(*this);
        if (!setupOpt)
            return;
        auto &setup = *setupOpt;
        auto &nodesBoundInterpolated = setup.nodesBoundInterpolated;
        auto &boundInterpCoo = setup.boundInterpCoo;
        auto &boundInterpVal = setup.boundInterpVal;
        index boundInterpGlobSize = boundInterpCoo.father->globalSize();

        // Build full pull index list (for RBF coefficient ghost mapping)
        std::vector<index> boundInterpPullIdx(boundInterpGlobSize);
        for (index i = 0; i < boundInterpGlobSize; i++)
            boundInterpPullIdx[i] = i;

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        using kdtree_tcoo = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>,
            PointCloudKDTree,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        tCoordPair boundInterpCoef;
        using tScalarPair = DNDS::ArrayPair<DNDS::ParArray<real, 1>>;
        tScalarPair boundInterpR;

        boundInterpCoef.InitPair("ElevatedNodesSolveInternalSmoothV1Old::boundInterpCoef", mpi);
        boundInterpR.InitPair("ElevatedNodesSolveInternalSmoothV1Old::boundInterpR", mpi);
        boundInterpCoef.father->Resize(mpi.rank == mRank ? boundInterpGlobSize : 0);
        boundInterpR.father->Resize(mpi.rank == mRank ? boundInterpGlobSize : 0);

        if (mpi.rank == mRank) // that dirty work
        {

            Eigen::SparseMatrix<real> M(boundInterpGlobSize, boundInterpGlobSize);
            M.reserve(Eigen::Vector<index, Eigen::Dynamic>::Constant(boundInterpGlobSize, elevationInfo.nSearch));
            MatrixXR f;
            f.resize(boundInterpGlobSize, 3);
            for (index iN = 0; iN < boundInterpGlobSize; iN++)
            {
                tPoint cooC = (*boundInterpCoo.son)[iN];

                index nFind = elevationInfo.nSearch;
                std::vector<index> idxFound;
                Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
                idxFound.resize(nFind);
                outDistancesSqr.resize(nFind);
                index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
                DNDS_assert(nFound >= 1);
                idxFound.resize(nFound);
                outDistancesSqr.resize(nFound);

                real RMin = outDistancesSqr.minCoeff();
                real RMax = outDistancesSqr.maxCoeff();
                real RRBF = elevationInfo.RBFRadius * std::sqrt(RMax);
                boundInterpR(iN, 0) = RRBF;
                DNDS_assert(RRBF > 0);
                Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().sqrt() * (1. / RRBF);
                auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
                for (index in2n = 0; in2n < nFound; in2n++)
                    M.insert(iN, idxFound[in2n]) = fBasis(in2n);

                f(iN, EigenAll) = (*boundInterpVal.son)[iN].transpose();
            }

            log() << "RBF assembled: " << boundInterpCoo.son->Size() << std::endl;
            MatrixXR coefs;
            M.makeCompressed();
            Eigen::SparseLU<Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int>> LUSolver;
            LUSolver.analyzePattern(M);
            LUSolver.factorize(M);
            coefs = LUSolver.solve(f);
            for (index iN = 0; iN < boundInterpGlobSize; iN++)
                boundInterpCoef[iN] = coefs(iN, EigenAll).transpose();
        }

        boundInterpCoef.father->createGlobalMapping();
        boundInterpCoef.TransAttach();
        boundInterpCoef.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoef.trans.createMPITypes();
        boundInterpCoef.trans.pullOnce();

        boundInterpR.TransAttach();
        boundInterpR.trans.BorrowGGIndexing(boundInterpCoef.trans);
        boundInterpR.trans.createMPITypes();
        boundInterpR.trans.pullOnce();

        if (mpi.rank == mRank)
        {
            for (index iN = 0; iN < boundInterpGlobSize; iN++)
            {
            }
        }

        PointCloudKDTree insidePts;
        insidePts.pts.reserve(coords.father->Size() - nodesBoundInterpolated.size());
        std::vector<index> insideNodes;
        insideNodes.reserve(insidePts.pts.size());
        for (index iN = 0; iN < coords.father->Size(); iN++)
            if (!nodesBoundInterpolated.count(iN))
                insidePts.pts.push_back(coords[iN]), insideNodes.push_back(iN), coordsElevDisp[iN].setZero();
        kdtree_tcoo nodesDstTree(3, insidePts);
        for (index iN = 0; iN < boundInterpGlobSize; iN++)
        {

            tPoint cooC = (*boundInterpCoo.son)[iN];
#if NANOFLANN_VERSION < 0x150
            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
#else
            std::vector<nanoflann::ResultItem<DNDS::index, DNDS::real>> IndicesDists;
#endif
            IndicesDists.reserve(elevationInfo.nSearch * 5);
            real RRBF = boundInterpR.son->operator()(iN, 0);
#if NANOFLANN_VERSION < 0x150
            nanoflann::SearchParams params{}; // default params
#else
            nanoflann::SearchParameters params{};
#endif
            index nFound = nodesDstTree.radiusSearch(cooC.data(), RRBF, IndicesDists, params);
            Eigen::Vector<real, Eigen::Dynamic> outDists;
            outDists.resize(IndicesDists.size());
            DNDS_assert(RRBF > 0);
            for (index i = 0; i < IndicesDists.size(); i++)
                outDists[i] = std::sqrt(IndicesDists[i].second) / RRBF;

            auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
            for (index i = 0; i < IndicesDists.size(); i++)
            {
                coordsElevDisp[insideNodes[IndicesDists[i].first]] +=
                    fBasis(i, 0) *
                    boundInterpCoef.son->operator[](iN);
            }
        }

        // for (index iN = 0; iN < coords.father->Size(); iN++)
        // {
        //     if (nodesBoundInterpolated.count(iN))
        //         continue;

        //     index nFind = elevationInfo.nSearch * 5; // for safety
        //     tPoint cooC = coords[iN];

        //     std::vector<index> idxFound;
        //     Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
        //     idxFound.resize(nFind);
        //     outDistancesSqr.resize(nFind);

        //     index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
        //     DNDS_assert(nFound >= 1);
        //     idxFound.resize(nFound);
        //     outDistancesSqr.resize(nFound);

        //     Eigen::Vector<real, Eigen::Dynamic> foundRRBFs;
        //     foundRRBFs.resize(nFound);
        //     tSmallCoords coefsC;
        //     coefsC.resize(3, nFound);
        //     for (index iF = 0; iF < nFound; iF++)
        //         foundRRBFs[iF] = (*boundInterpR.son)(idxFound[iF], 0),
        //         coefsC(EigenAll, iF) = (*boundInterpCoef.son)[idxFound[iF]].transpose();
        //     Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().sqrt() / foundRRBFs.array();
        //     auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
        //     tPoint dispC = coefsC * fBasis;
        //     coordsElevDisp[iN] = dispC;
        // }

        coordsElevDisp.trans.startPersistentPull();
        coordsElevDisp.trans.waitPersistentPull();

        for (index iN = nNodeO1; iN < coords.father->Size(); iN++)
        {
            // if(dim == 2)
            //     dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            if (coordsElevDisp[iN](0) != largeReal)
                coords[iN] += coordsElevDisp[iN];
        }
        coords.trans.pullOnce();
    }

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV1()
    {
        auto setupOpt = PrepareSmoothSolverSetup(*this);
        if (!setupOpt)
            return;
        auto &setup = *setupOpt;
        auto &nodesBoundInterpolated = setup.nodesBoundInterpolated;
        auto &boundInterpCoo = setup.boundInterpCoo;
        auto &boundInterpVal = setup.boundInterpVal;

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        index boundInterpLocSize = nodesBoundInterpolated.size();

        // Build full pull index list (for RBF coefficient ghost mapping)
        std::vector<index> boundInterpPullIdx(boundInterpGlobSize);
        for (index i = 0; i < boundInterpGlobSize; i++)
            boundInterpPullIdx[i] = i;

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        using kdtree_tcoo = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>,
            PointCloudKDTree,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        CoordPairDOF boundInterpCoef, boundInterpCoefRHS;
        using tScalarPair = DNDS::ArrayPair<DNDS::ParArray<real, 1>>;
        tScalarPair boundInterpR;
        boundInterpCoef.InitPair("ElevatedNodesSolveInternalSmoothV1::boundInterpCoef", mpi);
        boundInterpCoefRHS.InitPair("ElevatedNodesSolveInternalSmoothV1::boundInterpCoefRHS", mpi);
        boundInterpR.InitPair("ElevatedNodesSolveInternalSmoothV1::boundInterpR", mpi);
        boundInterpCoef.father->Resize(boundInterpCoo.father->Size());
        boundInterpCoefRHS.father->Resize(boundInterpCoo.father->Size());
        boundInterpR.father->Resize(boundInterpCoo.father->Size());
        std::vector<std::vector<std::pair<index, real>>> MatC;
        std::vector<index> boundInterpPullingIndexSolving;
        MatC.resize(boundInterpLocSize);
        for (index iN = 0; iN < boundInterpLocSize; iN++)
        {
            tPoint cooC = (*boundInterpCoo.father)[iN];

            index nFind = elevationInfo.nSearch;
            std::vector<index> idxFound;
            Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;

            /**********************************************/
            idxFound.resize(nFind);
            outDistancesSqr.resize(nFind);
            index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
            DNDS_assert(nFound >= 1);
            idxFound.resize(nFound);
            outDistancesSqr.resize(nFound);

            real RMin = std::sqrt(outDistancesSqr.minCoeff());
            real RMax = std::sqrt(outDistancesSqr.maxCoeff());
            real RRBF = elevationInfo.RBFRadius * RMax;
            /***********************************************/
            // std::vector<std::pair<DNDS::index, DNDS::real>> indicesDists;
            // indicesDists.reserve(nFind);

            // nanoflann::SearchParams searchParams;
            // index nFound = bndInterpTree.radiusSearch(cooC.data(), elevationInfo.RBFRadius, indicesDists, searchParams);
            // idxFound.reserve(nFound);
            // outDistancesSqr.resize(nFound);
            // for (auto v : indicesDists)
            //     outDistancesSqr(idxFound.size()) = v.second, idxFound.push_back(v.first);
            // real RRBF = elevationInfo.RBFRadius;
            /***********************************************/

            boundInterpR(iN, 0) = RRBF;
            DNDS_assert(RRBF > 0);
            Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().pow(0.5 / elevationInfo.RBFPower) * (1. / std::pow(RRBF, 1. / elevationInfo.RBFPower));
            auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
            MatC[iN].resize(nFound);
            for (index in2n = 0; in2n < nFound; in2n++)
            {
                MatC[iN][in2n] = std::make_pair(index(idxFound[in2n]), real(fBasis(in2n)));
                // idxFound[in2n] is a global indexing!
                auto [search_good, rank, val] = boundInterpCoo.trans.pLGlobalMapping->search(idxFound[in2n]);
                DNDS_assert(search_good);
                if (rank != mpi.rank)
                    boundInterpPullingIndexSolving.push_back(idxFound[in2n]);
            }

            boundInterpCoefRHS[iN] = (*boundInterpVal.father)[iN];
            boundInterpCoef[iN].setZero(); // init val
            // if(mpi.rank == 0)
            // {
            //     std::cout << "mat row at " << cooC.transpose() << "R = " << RRBF << "\n";
            //     for (auto idx : idxFound)
            //         std::cout << idx << " ";
            //     std::cout << "\n"
            //               << fBasis.transpose() << "\n"
            //               << outDists.transpose() << "\n"
            //               << boundInterpCoefRHS[iN].transpose() << std::endl;
            // }
        }

        if (false)
        { // use superlu_dist to solve

            // gridinfo_t grid;
            // superlu_gridinit(mpi.comm, 1, mpi.size, &grid);
            // DNDS_assert(grid.iam < mpi.size && grid.iam >= 0);

            // std::vector<double> nzval, b;
            // std::vector<int_t> colind;
            // std::vector<int_t> rowptr;
            // rowptr.resize(boundInterpLocSize + 1);
            // rowptr[0] = 0;
            // for (index i = 0; i < boundInterpLocSize; i++)
            //     rowptr[i + 1] = rowptr[i] + MatC[i].size();
            // colind.resize(rowptr.back());
            // nzval.resize(rowptr.back());
            // for (index i = 0; i < boundInterpLocSize; i++)
            //     for (index i2j = 0; i2j < MatC[i].size(); i2j++)
            //     {
            //         colind[rowptr[i] + i2j] = MatC[i][i2j].first;
            //         nzval[rowptr[i] + i2j] = MatC[i][i2j].second;
            //     }
            // b.resize(boundInterpLocSize * 3);
            // for (index i = 0; i < boundInterpLocSize; i++)
            // {
            //     b[boundInterpLocSize * 0 + i] = boundInterpCoefRHS[i](0);
            //     b[boundInterpLocSize * 1 + i] = boundInterpCoefRHS[i](1);
            //     b[boundInterpLocSize * 2 + i] = boundInterpCoefRHS[i](2);
            // }

            // SuperMatrix Aloc;
            // dCreate_CompRowLoc_Matrix_dist(
            //     &Aloc, boundInterpGlobSize, boundInterpGlobSize, nzval.size(), boundInterpLocSize,
            //     boundInterpCoo.trans.pLGlobalMapping->operator()(mpi.rank, 0), nzval.data(), colind.data(), rowptr.data(),
            //     SLU_NR_loc, SLU_D, SLU_GE);

            // superlu_dist_options_t options;
            // set_default_options_dist(&options);
            // SuperLUStat_t stat;
            // PStatInit(&stat);
            // dScalePermstruct_t ScalePermstruct;
            // dScalePermstructInit(Aloc.nrow, Aloc.ncol, &ScalePermstruct);
            // dLUstruct_t LUstruct;
            // dLUstructInit(Aloc.ncol, &LUstruct);
            // dSOLVEstruct_t SOLVEstruct;

            // tPoint berr;
            // berr.setZero();
            // int info;

            // pdgssvx(&options, &Aloc, &ScalePermstruct,
            //         b.data(), boundInterpLocSize, 3,
            //         &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);
            // for (index i = 0; i < boundInterpLocSize; i++)
            // {
            //     boundInterpCoef[i](0) = b[boundInterpLocSize * 0 + i];
            //     boundInterpCoef[i](1) = b[boundInterpLocSize * 1 + i];
            //     boundInterpCoef[i](2) = b[boundInterpLocSize * 2 + i];
            // }
            // PStatPrint(&options, &stat, &grid);
            // DNDS_assert(info == 0);

            // PStatFree(&stat);
            // // TODO: sanitize with valgrind!
            // // Destroy_CompRowLoc_Matrix_dist(&Aloc); // use vector for strorage, this causes double free
            // SUPERLU_FREE(Aloc.Store);
            // dScalePermstructFree(&ScalePermstruct);
            // dDestroy_LU(Aloc.ncol, &grid, &LUstruct);
            // dLUstructFree(&LUstruct);
            // if (options.SolveInitialized)
            // {
            //     dSolveFinalize(&options, &SOLVEstruct);
            // }

            // superlu_gridexit(&grid);
        }
        if (true)
        { // use GMRES to solve
            boundInterpCoef.father->createGlobalMapping();
            boundInterpCoef.TransAttach();
            boundInterpCoef.trans.createGhostMapping(boundInterpPullingIndexSolving);
            boundInterpCoef.trans.createMPITypes();
            boundInterpCoef.trans.initPersistentPull();

            boundInterpCoefRHS.TransAttach();
            boundInterpCoefRHS.trans.BorrowGGIndexing(boundInterpCoef.trans);
            boundInterpCoefRHS.trans.createMPITypes();
            boundInterpCoefRHS.trans.initPersistentPull();

            for (index iN = 0; iN < boundInterpLocSize; iN++)
            {
                for (index in2n = 0; in2n < MatC[iN].size(); in2n++)
                {
                    // MatC[iN][in2n].first;
                    auto [search_good, rank, val] = boundInterpCoef.trans.pLGhostMapping->search_indexAppend(MatC[iN][in2n].first);
                    DNDS_assert(search_good);
                    DNDS_assert(val < boundInterpCoef.Size());
                    MatC[iN][in2n].first = val; // to local
                }
            }
            Eigen::SparseMatrix<real> M(boundInterpLocSize, boundInterpLocSize);
            M.reserve(Eigen::Vector<index, Eigen::Dynamic>::Constant(boundInterpLocSize, elevationInfo.nSearch));
            for (index iN = 0; iN < boundInterpLocSize; iN++)
            {
                for (index in2n = 0; in2n < MatC[iN].size(); in2n++)
                {
                    if (MatC[iN][in2n].first < boundInterpLocSize)
                        M.insert(iN, MatC[iN][in2n].first) = MatC[iN][in2n].second;
                    // ,
                    //              std::cout << fmt::format("rank {}, inserting {},{}={}", mpi.rank, iN, MatC[iN][in2n].first, MatC[iN][in2n].second) << std::endl;
                }
            }
            Eigen::SparseLU<Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int>> LUSolver;
            if (boundInterpLocSize)
            {
                LUSolver.analyzePattern(M);
                LUSolver.factorize(M); // full LU perconditioner
            }

            Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(
                5,
                [&](CoordPairDOF &v)
                {
                    v.InitPair("ElevatedNodesSolveInternalSmoothV1::v", mpi);
                    v.father->Resize(boundInterpCoef.father->Size());
                    v.TransAttach();
                    v.trans.BorrowGGIndexing(boundInterpCoef.trans);
                    v.trans.createMPITypes();
                    v.trans.initPersistentPull();
                });
            boundInterpCoef.trans.startPersistentPull();
            boundInterpCoef.trans.waitPersistentPull();
            boundInterpCoefRHS.trans.startPersistentPull();
            boundInterpCoefRHS.trans.waitPersistentPull();
            gmres.solve(
                [&](CoordPairDOF &x, CoordPairDOF &Ax)
                {
                    x.trans.startPersistentPull();
                    x.trans.waitPersistentPull();
                    for (index iN = 0; iN < boundInterpLocSize; iN++)
                    {
                        Ax[iN].setZero();
                        for (index in2n = 0; in2n < MatC[iN].size(); in2n++)
                        {
                            Ax[iN] += x[MatC[iN][in2n].first] * MatC[iN][in2n].second;
                        }
                    }
                    Ax.trans.startPersistentPull();
                    Ax.trans.waitPersistentPull();
                },
                [&](CoordPairDOF &x, CoordPairDOF &MLX)
                {
                    x.trans.startPersistentPull();
                    x.trans.waitPersistentPull();
                    Eigen::Matrix<real, Eigen::Dynamic, 3> xVal;
                    xVal.resize(boundInterpLocSize, 3);
                    for (index iN = 0; iN < boundInterpLocSize; iN++)
                        xVal(iN, EigenAll) = x[iN].transpose();
                    Eigen::Matrix<real, Eigen::Dynamic, 3> xValPrec;
                    if (boundInterpLocSize)
                        xValPrec = LUSolver.solve(xVal);
                    for (index iN = 0; iN < boundInterpLocSize; iN++)
                        MLX[iN] = xValPrec(iN, EigenAll).transpose();
                    MLX = x;
                    MLX.trans.startPersistentPull();
                    MLX.trans.waitPersistentPull();
                },
                [&](CoordPairDOF &a, CoordPairDOF &b) -> real
                {
                    return a.dot(b);
                },
                boundInterpCoefRHS,
                boundInterpCoef,
                elevationInfo.nIter,
                [&](int iRestart, real res, real resB)
                {
                    if (mpi.rank == mRank)
                        log() << fmt::format("iRestart {}, res {}, resB {}", iRestart, res, resB) << std::endl;
                    return res < resB * 1e-6;
                }

            );
            boundInterpCoef.trans.startPersistentPull();
            boundInterpCoef.trans.waitPersistentPull();
        }

        boundInterpCoef.father->createGlobalMapping();
        boundInterpCoef.TransAttach();
        boundInterpCoef.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoef.trans.createMPITypes();
        boundInterpCoef.trans.pullOnce();

        boundInterpR.TransAttach();
        boundInterpR.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpR.trans.createMPITypes();
        boundInterpR.trans.pullOnce();

        // if (mpi.rank == mRank)
        // {
        //     for (index iN = 0; iN < boundInterpGlobSize; iN++)
        //     {
        //         std::cout << "pt - coef: R " << boundInterpR.son->operator()(iN, 0)
        //                   << " coo " << boundInterpCoo.son->operator[](iN).transpose()
        //                   << " -> " << boundInterpCoef.son->operator[](iN).transpose() << std::endl;
        //     }
        // }

        PointCloudKDTree insidePts;
        insidePts.pts.reserve(coords.father->Size() - boundInterpLocSize);
        std::vector<index> insideNodes;
        insideNodes.reserve(insidePts.pts.size());
        for (index iN = 0; iN < coords.father->Size(); iN++)
            if (!nodesBoundInterpolated.count(iN))
                insidePts.pts.push_back(coords[iN]), insideNodes.push_back(iN), coordsElevDisp[iN].setZero();
        kdtree_tcoo nodesDstTree(3, insidePts);
        for (index iN = 0; iN < boundInterpGlobSize; iN++)
        {

            tPoint cooC = (*boundInterpCoo.son)[iN];
#if NANOFLANN_VERSION < 0x150
            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
#else
            std::vector<nanoflann::ResultItem<DNDS::index, DNDS::real>> IndicesDists;
#endif
            IndicesDists.reserve(elevationInfo.nSearch * 5);
            real RRBF = boundInterpR.son->operator()(iN, 0);
#if NANOFLANN_VERSION < 0x150
            nanoflann::SearchParams params{}; // default params
#else
            nanoflann::SearchParameters params{};
#endif
            index nFound = nodesDstTree.radiusSearch(cooC.data(), RRBF, IndicesDists, params);
            Eigen::Vector<real, Eigen::Dynamic> outDists;
            outDists.resize(IndicesDists.size());
            DNDS_assert(RRBF > 0);
            for (index i = 0; i < IndicesDists.size(); i++)
                outDists[i] = std::sqrt(IndicesDists[i].second) / std::pow(RRBF, 1. / elevationInfo.RBFPower);

            auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
            for (index i = 0; i < IndicesDists.size(); i++)
            {
                coordsElevDisp[insideNodes[IndicesDists[i].first]] +=
                    fBasis(i, 0) *
                    boundInterpCoef.son->operator[](iN);
            }
        }

        // for (index iN = 0; iN < coords.father->Size(); iN++)
        // {
        //     if (nodesBoundInterpolated.count(iN))
        //         continue;

        //     index nFind = elevationInfo.nSearch * 5; // for safety
        //     tPoint cooC = coords[iN];

        //     std::vector<index> idxFound;
        //     Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
        //     idxFound.resize(nFind);
        //     outDistancesSqr.resize(nFind);

        //     index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
        //     DNDS_assert(nFound >= 1);
        //     idxFound.resize(nFound);
        //     outDistancesSqr.resize(nFound);

        //     Eigen::Vector<real, Eigen::Dynamic> foundRRBFs;
        //     foundRRBFs.resize(nFound);
        //     tSmallCoords coefsC;
        //     coefsC.resize(3, nFound);
        //     for (index iF = 0; iF < nFound; iF++)
        //         foundRRBFs[iF] = (*boundInterpR.son)(idxFound[iF], 0),
        //         coefsC(EigenAll, iF) = (*boundInterpCoef.son)[idxFound[iF]].transpose();
        //     Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().sqrt() / foundRRBFs.array();
        //     auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
        //     tPoint dispC = coefsC * fBasis;
        //     coordsElevDisp[iN] = dispC;
        // }

        coordsElevDisp.trans.startPersistentPull();
        coordsElevDisp.trans.waitPersistentPull();

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            // if(dim == 2)
            //     dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            if (coordsElevDisp[iN](0) != largeReal)
                coords[iN] += coordsElevDisp[iN];
        }
        coords.trans.pullOnce();
    }

    // ElevatedNodesSolveInternalSmoothV2 is in Mesh_Elevation_SmoothV2.cpp

} // namespace DNDS::Geom
