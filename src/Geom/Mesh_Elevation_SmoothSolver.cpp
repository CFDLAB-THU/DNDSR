#include "Mesh.hpp"
#include "Quadrature.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "RadialBasisFunction.hpp"

#include <omp.h>
#include <fmt/core.h>
#include <functional>
#include <unordered_set>
#include <Solver/Linear.hpp>

#include "PointCloud.hpp"
#include <nanoflann.hpp>

#include "DNDS/EigenPCH.hpp"
#ifdef DNDS_USE_SUPERLU
#include <superlu_ddefs.h>
#endif

namespace DNDS::Geom
{
    struct CoordPairDOF : public tCoordPair
    {
        real dot(CoordPairDOF &R)
        {
            real ret = 0;
            for (index i = 0; i < this->father->Size(); i++)
                ret += (*this)[i].dot(R[i]);
            real retSum;
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
        [[nodiscard]] inline size_t
        kdtree_get_point_count() const
        {
            DNDS_assert(ref);
            return ref->Size();
        }

        [[nodiscard]] inline real kdtree_get_pt(const size_t idx, const size_t dim) const
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

    void UnstructuredMesh::ElevatedNodesSolveInternalSmooth()
    {
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        // for (index iF = 0; iF < boundInterpCoo.son->Size(); iF++)
        // {
        //     std::cout << boundInterpVal.sfaon->operator[](iF).transpose() << std::endl;
        // }
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
                coordBnd(Eigen::all, iF) = (*boundInterpCoo.son)[idxFound[iF]];
                dispBnd(Eigen::all, iF) = (*boundInterpVal.son)[idxFound[iF]];
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
                     coefs(Eigen::last - 3, Eigen::all) +
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
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

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

        DNDS_MAKE_SSP(boundInterpCoef.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoef.son, mpi);
        DNDS_MAKE_SSP(boundInterpR.father, mpi);
        DNDS_MAKE_SSP(boundInterpR.son, mpi);
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
                std::cout << iN << std::endl;
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

                f(iN, Eigen::all) = (*boundInterpVal.son)[iN].transpose();
            }

            log() << "RBF assembled: " << boundInterpCoo.son->Size() << std::endl;
            MatrixXR coefs;
            M.makeCompressed();
            Eigen::SparseLU<Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int>> LUSolver;
            LUSolver.analyzePattern(M);
            LUSolver.factorize(M);
            coefs = LUSolver.solve(f);
            for (index iN = 0; iN < boundInterpGlobSize; iN++)
                boundInterpCoef[iN] = coefs(iN, Eigen::all).transpose();
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
                std::cout << "coef: " << iN << " " << boundInterpCoef.son->operator[](iN).transpose() << std::endl;
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
            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
            IndicesDists.reserve(elevationInfo.nSearch * 5);
            real RRBF = boundInterpR.son->operator()(iN, 0);
            nanoflann::SearchParams params{}; // default params
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
        //         coefsC(Eigen::all, iF) = (*boundInterpCoef.son)[idxFound[iF]].transpose();
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
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        index boundInterpLocSize = nodesBoundInterpolated.size();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

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
        std::cout << "HEre-2" << std::endl;
        DNDS_MAKE_SSP(boundInterpCoef.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoef.son, mpi);
        DNDS_MAKE_SSP(boundInterpCoefRHS.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoefRHS.son, mpi);
        DNDS_MAKE_SSP(boundInterpR.father, mpi);
        DNDS_MAKE_SSP(boundInterpR.son, mpi);
        boundInterpCoef.father->Resize(boundInterpCoo.father->Size());
        boundInterpCoefRHS.father->Resize(boundInterpCoo.father->Size());
        boundInterpR.father->Resize(boundInterpCoo.father->Size());
        std::vector<std::vector<std::pair<index, real>>> MatC;
        std::vector<index> boundInterpPullingIndexSolving;
        MatC.resize(boundInterpLocSize);
        std::cout << "HEre-1" << std::endl;
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
            std::cout << "HEre0" << std::endl;
            boundInterpCoef.father->createGlobalMapping();
            boundInterpCoef.TransAttach();
            boundInterpCoef.trans.createGhostMapping(boundInterpPullingIndexSolving);
            boundInterpCoef.trans.createMPITypes();
            boundInterpCoef.trans.initPersistentPull();

            boundInterpCoefRHS.TransAttach();
            boundInterpCoefRHS.trans.BorrowGGIndexing(boundInterpCoef.trans);
            boundInterpCoefRHS.trans.createMPITypes();
            boundInterpCoefRHS.trans.initPersistentPull();

            std::cout << "HEre1" << std::endl;
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
            std::cout << "HEre2" << std::endl;

            Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(
                5,
                [&](CoordPairDOF &v)
                {
                    DNDS_MAKE_SSP(v.father, mpi);
                    DNDS_MAKE_SSP(v.son, mpi);
                    v.father->Resize(boundInterpCoef.father->Size());
                    v.TransAttach();
                    v.trans.BorrowGGIndexing(boundInterpCoef.trans);
                    v.trans.createMPITypes();
                    v.trans.initPersistentPull();
                });
            std::cout << "HEre3" << std::endl;
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
                        xVal(iN, Eigen::all) = x[iN].transpose();
                    Eigen::Matrix<real, Eigen::Dynamic, 3> xValPrec;
                    if (boundInterpLocSize)
                        xValPrec = LUSolver.solve(xVal);
                    for (index iN = 0; iN < boundInterpLocSize; iN++)
                        MLX[iN] = xValPrec(iN, Eigen::all).transpose();
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
            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
            IndicesDists.reserve(elevationInfo.nSearch * 5);
            real RRBF = boundInterpR.son->operator()(iN, 0);
            nanoflann::SearchParams params{}; // default params
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
        //         coefsC(Eigen::all, iF) = (*boundInterpCoef.son)[idxFound[iF]].transpose();
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

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV2()
    {
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        index boundInterpLocSize = nodesBoundInterpolated.size();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

        std::vector<std::unordered_set<index>> node2nodeV;
        node2nodeV.resize(coords.father->Size());
        for (index iCell = 0; iCell < cell2node.Size(); iCell++)
        {
            for (auto iN : cell2node[iCell])
                for (auto iNOther : cell2node[iCell])
                {
                    if (iN == iNOther)
                        continue;
                    if (iN < coords.father->Size())
                    {
                        node2nodeV[iN].insert(iNOther);
                    }
                }
        }

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        DNDS_MPI_InsertCheck(mpi, "Got node2node");
        // std::cout << "hereXXXXXXXXXXXXXXX 0" << std::endl;

        CoordPairDOF dispO2, bO2, dispO2New, dispO2PrecRHS, dispO2Inc, dispO2IncLimited;
        auto initDOF = [&](CoordPairDOF &dispO2)
        {
            DNDS_MAKE_SSP(dispO2.father, mpi);
            DNDS_MAKE_SSP(dispO2.son, mpi);
            dispO2.father->Resize(coords.father->Size()); // O1 part is unused
            dispO2.TransAttach();
            dispO2.trans.BorrowGGIndexing(coords.trans); // should be subset of coord?
            dispO2.trans.createMPITypes();
            dispO2.trans.initPersistentPull();
        };
        initDOF(dispO2);
        initDOF(dispO2Inc);
        initDOF(dispO2New);
        initDOF(dispO2IncLimited);
        initDOF(bO2);
        initDOF(dispO2PrecRHS);

        struct MatElem
        {
            index j;
            Eigen::Matrix<real, 3, 3> m;
        };
        std::vector<std::vector<MatElem>> A; // stiff matrixTo O2 coords
        A.resize(coords.father->Size());
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            A[iN].resize(node2nodeV[iN].size() + 1);
            A[iN][0].j = iN;
            int curRowEnd = 1;
            for (auto iNOther : node2nodeV[iN])
            {
                A[iN][curRowEnd++].j = iNOther;
            }
            for (auto &ME : A[iN])
                ME.m.setConstant(0);
            bO2[iN].setZero();
            dispO2[iN].setZero();
            if (coordsElevDisp[iN](0) != largeReal)
                dispO2[iN] = coordsElevDisp[iN];
            if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal)
                dispO2[iN].setZero();
        }
        dispO2.trans.startPersistentPull();
        dispO2.trans.waitPersistentPull();

        auto AssembleRHSMat = [&](tCoordPair &uCur)
        {
            index nFix{0}, nFixB{0};
            if (false) // fem method
                for (index iCell = 0; iCell < cell2node.Size(); iCell++)
                {
                    auto c2n = cell2node[iCell];
                    rowsize nnLoc = c2n.size();
                    tSmallCoords coordsC;
                    GetCoordsOnCell(iCell, coordsC);
                    tPoint cellCent = coordsC.rowwise().mean();
                    real wDist{0};
                    {
                        std::vector<real> sqrDists(1);
                        std::vector<index> idxFound(1);
                        auto nFound = bndInterpTree.knnSearch(cellCent.data(), 1, idxFound.data(), sqrDists.data());
                        DNDS_assert(nFound == 1);
                        wDist = std::sqrt(sqrDists[0]);
                    }

                    MatrixXR ALoc, mLoc;
                    ALoc.setZero(3 * nnLoc, 3 * nnLoc + 1);
                    mLoc.resize(6, 3 * nnLoc);
                    auto eCell = GetCellElement(iCell);
                    auto qCell = Elem::Quadrature{eCell, 2}; // for O2 FEM
                    qCell.Integration(
                        ALoc,
                        [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                        {
                            // J = dx_i/dxii_j, Jinv = dxii_i/dx_j
                            tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsC, DiNj);
                            tJacobi JInv = tJacobi::Identity();
                            real JDet;
                            if (dim == 2)
                                JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                            else
                                JDet = J.determinant();
                            DNDS_assert(JDet > 0);
                            if (dim == 2)
                                JInv({0, 1}, {0, 1}) = J({0, 1}, {0, 1}).inverse().eval();
                            else
                                JInv = J.inverse().eval();
                            // JInv.transpose() * DiNj({1,2,3}, Eigen::all) * u (size = nnode,3) => du_j/dxi
                            tSmallCoords m3 = JInv.transpose() * DiNj({1, 2, 3}, Eigen::all);
                            mLoc(0, Eigen::seq(nnLoc * 0, nnLoc * 0 + nnLoc - 1)) = m3(0, Eigen::all);
                            mLoc(1, Eigen::seq(nnLoc * 1, nnLoc * 1 + nnLoc - 1)) = m3(1, Eigen::all);
                            mLoc(2, Eigen::seq(nnLoc * 2, nnLoc * 2 + nnLoc - 1)) = m3(2, Eigen::all);

                            mLoc(3, Eigen::seq(nnLoc * 1, nnLoc * 1 + nnLoc - 1)) = 0.5 * m3(2, Eigen::all);
                            mLoc(3, Eigen::seq(nnLoc * 2, nnLoc * 2 + nnLoc - 1)) = 0.5 * m3(1, Eigen::all);
                            mLoc(4, Eigen::seq(nnLoc * 2, nnLoc * 2 + nnLoc - 1)) = 0.5 * m3(0, Eigen::all);
                            mLoc(4, Eigen::seq(nnLoc * 0, nnLoc * 0 + nnLoc - 1)) = 0.5 * m3(2, Eigen::all);
                            mLoc(5, Eigen::seq(nnLoc * 0, nnLoc * 0 + nnLoc - 1)) = 0.5 * m3(1, Eigen::all);
                            mLoc(5, Eigen::seq(nnLoc * 1, nnLoc * 1 + nnLoc - 1)) = 0.5 * m3(0, Eigen::all);

                            Eigen::Matrix<real, 6, 6> DStruct;
                            DStruct.setIdentity();
                            // real lam = std::pow((wDist / 1e-6), -1);
                            real lam = 1;
                            real muu = 1;
                            DStruct *= muu;
                            DStruct({0, 1, 2}, {0, 1, 2}) += muu * tJacobi::Identity();
                            DStruct({0, 1, 2}, {0, 1, 2}).array() += lam;
                            // DStruct *= (1. / JDet);
                            // DStruct *= std::pow((wDist / 1e-6), -10);

                            vInc.resizeLike(ALoc);
                            vInc(Eigen::all, Eigen::seq(0, 3 * nnLoc - 1)) = mLoc.transpose() * DStruct * mLoc;
                            vInc(Eigen::all, Eigen::last).setZero(); // no node force, thus zero
                            vInc *= JDet;
                        });
                    // for (int ic2n = 0; ic2n < nnLoc; ic2n++)
                    //     for (int jc2n = 0; jc2n < nnLoc; jc2n++)
                    //     {

                    //         if (ic2n == jc2n)
                    //             continue;
                    //         real lBA = (coordsC(Eigen::all, jc2n) - coordsC(Eigen::all, ic2n)).norm();
                    //         tPoint nBA = (coordsC(Eigen::all, jc2n) - coordsC(Eigen::all, ic2n)).normalized();
                    //         // real K = std::pow(lBA, 0);
                    //         real K = std::pow((wDist / elevationInfo.refDWall), -1);
                    //         tJacobi mnBA = nBA * nBA.transpose() * K;
                    //         for (int iii = 0; iii < 3; iii++)
                    //             for (int jjj = 0; jjj < 3; jjj++)
                    //             {
                    //                 ALoc(iii * nnLoc + ic2n, jjj * nnLoc + ic2n) += mnBA(iii, jjj);
                    //                 ALoc(iii * nnLoc + ic2n, jjj * nnLoc + jc2n) -= mnBA(iii, jjj);
                    //             }
                    //}
                    if (dim != 3)
                    {
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::all, Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::all).setZero();
                    }

                    for (rowsize ic2n = 0; ic2n < nnLoc; ic2n++)
                    {
                        index iN = c2n[ic2n];
                        if (iN >= coords.father->Size())
                            continue; // not local row
                        for (rowsize jc2n = 0; jc2n < nnLoc; jc2n++)
                        {
                            index jN = c2n[jc2n];
                            int nMatrixFound = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == jN)
                                {
                                    nMatrixFound++;
                                    ME.m(0, 0) += ALoc(0 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                    ME.m(0, 1) += ALoc(0 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                    ME.m(0, 2) += ALoc(0 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                    ME.m(1, 0) += ALoc(1 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                    ME.m(1, 1) += ALoc(1 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                    ME.m(1, 2) += ALoc(1 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                    ME.m(2, 0) += ALoc(2 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                    ME.m(2, 1) += ALoc(2 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                    ME.m(2, 2) += ALoc(2 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                    if (isPeriodic)
                                    {
                                        auto ipbi = cell2nodePbi(iCell, ic2n);
                                        auto jpbi = cell2nodePbi(iCell, jc2n);
                                        tJacobi iTrans = periodicInfo.GetVectorByBits<3, 3>(tJacobi::Identity(), ipbi);
                                        tJacobi jTrans = periodicInfo.GetVectorByBits<3, 3>(tJacobi::Identity(), jpbi);
                                        ME.m = jTrans.transpose() * ME.m * iTrans;
                                    }
                                }
                            DNDS_assert(nMatrixFound == 1);
                        }
                        bO2[iN](0) += ALoc(0 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](1) += ALoc(1 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](2) += ALoc(2 * nnLoc + ic2n, Eigen::last); // last col is b
                        if (coordsElevDisp[iN](0) != largeReal)
                        {
                            nFix++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN] = coordsElevDisp[iN] * nDiag;
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal)
                        {
                            nFixB++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN].setZero();
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (dim != 3)
                        {
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m(2, 2) += nDiag; // the z-direction need a constraint
                        }
                    }
                }

            if (true) // bisect fem method
                for (index iCell = 0; iCell < cell2node.Size(); iCell++)
                {
                    auto c2n = cell2node[iCell];
                    rowsize nnLoc = c2n.size();
                    SmallCoordsAsVector coordsC, coordsCu;
                    GetCoordsOnCell(iCell, coordsC);
                    GetCoordsOnCell(iCell, coordsCu, uCur);
                    tPoint cellCent = coordsC.rowwise().mean();
                    real wDist{0};
                    {
                        std::vector<real> sqrDists(1);
                        std::vector<index> idxFound(1);
                        auto nFound = bndInterpTree.knnSearch(cellCent.data(), 1, idxFound.data(), sqrDists.data());
                        DNDS_assert(nFound == 1);
                        wDist = std::sqrt(sqrDists[0]);
                    }

                    MatrixXR ALoc;
                    ALoc.setZero(3 * nnLoc, 3 * nnLoc + 1);
                    auto localNodeIdx = Eigen::ArrayXi::LinSpaced(c2n.size(), 0, c2n.size() - 1);

                    auto eCell = GetCellElement(iCell);
                    int nBi = eCell.GetO2NumBisect();
                    int iBiVariant = GetO2ElemBisectVariant(eCell, coordsC);
                    for (int iBi = 0; iBi < nBi; iBi++)
                    {
                        auto eCellSub = eCell.ObtainO2BisectElem(iBi);
                        Eigen::ArrayXi c2nSubLocal;
                        c2nSubLocal.resize(eCellSub.GetNumNodes());
                        eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, localNodeIdx, c2nSubLocal);
                        SmallCoordsAsVector coordsCSub, coordsCuSub;
                        coordsCSub.resize(3, eCellSub.GetNumNodes());
                        coordsCuSub.resize(3, eCellSub.GetNumNodes());
                        eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsC, coordsCSub);
                        eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsCu, coordsCuSub);
                        auto qCellSub = Elem::Quadrature{eCellSub, 6}; // for O1 FEM
                        auto nnLocSub = rowsize(c2nSubLocal.size());
                        MatrixXR ALocSub, mLoc;
                        ALocSub.setZero(3 * nnLocSub, 3 * nnLocSub + 1);
                        mLoc.resize(6, 3 * nnLocSub);
                        Eigen::ArrayXi c2nSubLocal3;
                        c2nSubLocal3.resize(c2nSubLocal.size() * 3);
                        c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 0, c2nSubLocal.size() * 1 - 1)) = c2nSubLocal + nnLoc * 0;
                        c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 1, c2nSubLocal.size() * 2 - 1)) = c2nSubLocal + nnLoc * 1;
                        c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 2, c2nSubLocal.size() * 3 - 1)) = c2nSubLocal + nnLoc * 2;

                        qCellSub.Integration(
                            ALocSub,
                            [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                            {
                                // J = dx_i/dxii_j, Jinv = dxii_i/dx_j
                                auto getJ = [&](auto coos)
                                {
                                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coos, DiNj);
                                    tJacobi JInv = tJacobi::Identity();
                                    real JDet;
                                    if (dim == 2)
                                        JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                                    else
                                        JDet = J.determinant();
                                    if (dim == 2)
                                        JInv({0, 1}, {0, 1}) = J({0, 1}, {0, 1}).inverse().eval();
                                    else
                                        JInv = J.inverse().eval();
                                    return std::make_tuple(J, JInv, JDet);
                                };
                                auto [J, JInv, JDet] = getJ(coordsCSub);
                                auto [Jn, JInvn, JDetn] = getJ(coordsCSub + coordsCuSub);
                                DNDS_assert(JDet > 0);
                                // JInv.transpose() * DiNj({1,2,3}, Eigen::all) * u (size = nnode,3) => du_j/dxi
                                tSmallCoords m3 = JInv.transpose() * DiNj({1, 2, 3}, Eigen::all);
                                mLoc.setZero();
                                mLoc(0, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = m3(0, Eigen::all);
                                mLoc(1, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = m3(1, Eigen::all);
                                mLoc(2, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = m3(2, Eigen::all);

                                mLoc(3, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(2, Eigen::all);
                                mLoc(3, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(1, Eigen::all);
                                mLoc(4, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(0, Eigen::all);
                                mLoc(4, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(2, Eigen::all);
                                mLoc(5, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(1, Eigen::all);
                                mLoc(5, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(0, Eigen::all);

                                Eigen::Matrix<real, 6, 6> DStruct;
                                DStruct.setIdentity();
                                // real lam = std::pow((wDist / 1e-6), -1);
                                real lam = 1;
                                real muu = 100;
                                // lam *= std::pow(JDetn / JDet, -1) + 1;
                                DStruct *= muu;
                                DStruct({0, 1, 2}, {0, 1, 2}) += muu * tJacobi::Identity();
                                DStruct({0, 1, 2}, {0, 1, 2}).array() += lam;
                                DStruct *= std::pow(JDetn / JDet, -0.0) + 1;
                                // DStruct *= (1. / JDet);
                                // DStruct *= std::pow((wDist / 1e-6), -10);
                                // DStruct *= std::pow((wDist / elevationInfo.refDWall), -1);

                                vInc.resizeLike(ALocSub);
                                vInc(Eigen::all, Eigen::seq(0, 3 * nnLocSub - 1)) = mLoc.transpose() * DStruct * mLoc;
                                vInc(Eigen::all, Eigen::last).setZero(); // no node force, thus zero
                                vInc *= JDet;
                            });
                        ALoc(c2nSubLocal3, c2nSubLocal3) += ALocSub(Eigen::all, Eigen::seq(0, Eigen::last - 1));
                        ALoc(c2nSubLocal3, Eigen::last) += ALocSub(Eigen::all, Eigen::last);
                    }

                    if (dim != 3)
                    {
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::all, Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::all).setZero();
                    }

                    for (rowsize ic2n = 0; ic2n < nnLoc; ic2n++)
                    {
                        index iN = c2n[ic2n];
                        if (iN >= coords.father->Size())
                            continue; // not local row
                        for (rowsize jc2n = 0; jc2n < nnLoc; jc2n++)
                        {
                            index jN = c2n[jc2n];
                            int nMatrixFound = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == jN)
                                {
                                    nMatrixFound++;
                                    ME.m(0, 0) += ALoc(0 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                    ME.m(0, 1) += ALoc(0 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                    ME.m(0, 2) += ALoc(0 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                    ME.m(1, 0) += ALoc(1 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                    ME.m(1, 1) += ALoc(1 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                    ME.m(1, 2) += ALoc(1 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                    ME.m(2, 0) += ALoc(2 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                    ME.m(2, 1) += ALoc(2 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                    ME.m(2, 2) += ALoc(2 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                    if (isPeriodic)
                                    {
                                        auto ipbi = cell2nodePbi(iCell, ic2n);
                                        auto jpbi = cell2nodePbi(iCell, jc2n);
                                        tJacobi iTrans = periodicInfo.GetVectorByBits<3, 3>(tJacobi::Identity(), ipbi);
                                        tJacobi jTrans = periodicInfo.GetVectorByBits<3, 3>(tJacobi::Identity(), jpbi);
                                        ME.m = jTrans.transpose() * ME.m * iTrans;
                                    }
                                }
                            DNDS_assert(nMatrixFound == 1);
                        }
                        bO2[iN](0) += ALoc(0 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](1) += ALoc(1 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](2) += ALoc(2 * nnLoc + ic2n, Eigen::last); // last col is b
                        if (coordsElevDisp[iN](0) != largeReal)
                        {
                            nFix++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN] = coordsElevDisp[iN] * nDiag;
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal || (iN < nNodeO1 && false))
                        {
                            nFixB++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN].setZero();
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (dim != 3)
                        {
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m(2, 2) += nDiag; // the z-direction need a constraint
                        }
                    }
                }

            if (false)
                for (index iN = 0; iN < A.size(); iN++)
                {
                    for (auto jN : node2nodeV[iN])
                    {
                        real lBA = (coords[jN] - coords[iN]).norm();
                        tPoint nBA = (coords[jN] - coords[iN]).normalized();
                        real K = std::pow(lBA, -0.2);
                        // real K = std::pow((wDist / elevationInfo.refDWall), -1);
                        tJacobi mnBA = nBA * nBA.transpose() * K;
                        for (auto &ME : A[iN])
                            if (ME.j == jN)
                            {
                                A[iN][0].m += mnBA;
                                ME.m -= mnBA;
                            }
                    }
                    if (coordsElevDisp[iN](0) != largeReal)
                    {
                        nFix++;
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        bO2[iN] = coordsElevDisp[iN] * nDiag;
                        // bO2[iN].setZero();
                        // if (coords[iN].norm() < 2)
                        //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m = tJacobi::Identity() * nDiag;
                            else
                                ME.m.setZero();
                    }
                    if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal)
                    {
                        nFixB++;
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        bO2[iN].setZero();
                        // bO2[iN].setZero();
                        // if (coords[iN].norm() < 2)
                        //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m = tJacobi::Identity() * nDiag;
                            else
                                ME.m.setZero();
                    }
                    if (dim != 3)
                    {
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m(2, 2) += nDiag; // the z-direction need a constraint
                    }
                }

            bO2.trans.startPersistentPull();
            bO2.trans.waitPersistentPull();
            // for (index iN = 0; iN < A.size(); iN++)
            //     for (auto &ME : A[iN])
            //         std::cout << "ij: " << iN << " " << ME.j << "\n"
            //                   << ME.m << std::endl;
            MPI::AllreduceOneIndex(nFix, MPI_SUM, mpi);
            MPI::AllreduceOneIndex(nFixB, MPI_SUM, mpi);
            if (mpi.rank == mRank)
            {
                log() << fmt::format("UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): Matrix Assembled, nFix {}, nFixB {} ",
                                     nFix, nFixB)
                      << std::endl;
            }
        };
        dispO2.setConstant(0.0);
        AssembleRHSMat(dispO2);

        auto LimitDisp = [&](tCoordPair &uCur, tCoordPair &duCur, tCoordPair &duCurLim)
        {
            index nFix{0};
            real minLim{1};
            if (true) // bisect fem method
                for (index iCell = 0; iCell < cell2node.Size(); iCell++)
                {
                    auto c2n = cell2node[iCell];
                    rowsize nnLoc = c2n.size();
                    SmallCoordsAsVector coordsC, coordsCu, coordsCuinc;
                    GetCoordsOnCell(iCell, coordsC);
                    GetCoordsOnCell(iCell, coordsCu, uCur);
                    GetCoordsOnCell(iCell, coordsCuinc, duCur);
                    auto localNodeIdx = Eigen::ArrayXi::LinSpaced(c2n.size(), 0, c2n.size() - 1);

                    auto eCell = GetCellElement(iCell);
                    int nBi = eCell.GetO2NumBisect();
                    int iBiVariant = GetO2ElemBisectVariant(eCell, coordsC);

                    auto FJDetRatio = [&](real alpha) -> real
                    {
                        real JDetRatio = 1;
                        for (int iBi = 0; iBi < nBi; iBi++)
                        {
                            auto eCellSub = eCell.ObtainO2BisectElem(iBi);
                            Eigen::ArrayXi c2nSubLocal;
                            c2nSubLocal.resize(eCellSub.GetNumNodes());
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, localNodeIdx, c2nSubLocal);
                            SmallCoordsAsVector coordsCSub, coordsCuSub, coordsCuincSub;
                            coordsCSub.resize(3, eCellSub.GetNumNodes());
                            coordsCuSub.resize(3, eCellSub.GetNumNodes());
                            coordsCuincSub.resize(3, eCellSub.GetNumNodes());
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsC, coordsCSub);
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsCu, coordsCuSub);
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsCuinc, coordsCuincSub);
                            auto qCellSub = Elem::Quadrature{eCellSub, 2}; // for O1 FEM
                            Elem::SummationNoOp NoOp;
                            qCellSub.Integration(
                                NoOp,
                                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                                {
                                    // J = dx_i/dxii_j, Jinv = dxii_i/dx_j
                                    auto getJ = [&](auto coos)
                                    {
                                        tJacobi J = Elem::ShapeJacobianCoordD01Nj(coos, DiNj);
                                        tJacobi JInv = tJacobi::Identity();
                                        real JDet;
                                        if (dim == 2)
                                            JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1))(2); // ! note: for 2D, incompatible with curve surface situation
                                        else
                                            JDet = J.determinant();
                                        if (dim == 2)
                                            JInv({0, 1}, {0, 1}) = J({0, 1}, {0, 1}).inverse().eval();
                                        else
                                            JInv = J.inverse().eval();
                                        return std::make_tuple(J, JInv, JDet);
                                    };
                                    auto [J, JInv, JDet] = getJ(coordsCSub + coordsCuSub);
                                    DNDS_assert(JDet > 0);
                                    auto [Jn, JInvn, JDetn] = getJ(coordsCSub + coordsCuSub + coordsCuincSub * alpha);
                                    JDetRatio = std::min(JDetRatio, JDetn / JDet);
                                });
                        }
                        return JDetRatio;
                    };
                    real aLim = 1;
                    real decay = 0.5;
                    real minJDetRatio = 0.1;
                    for (int i = 0; i < 20; i++)
                        if (FJDetRatio(aLim) < minJDetRatio)
                            aLim *= decay;
                    for (auto iN : c2n)
                        if (iN < duCurLim.father->Size())
                            duCurLim[iN] = duCur[iN].array().sign() * duCurLim[iN].array().abs().min(duCur[iN].array().abs() * aLim);
                    minLim = std::min(minLim, aLim);
                    if (aLim < 1)
                        nFix++;
                }

            duCurLim.trans.startPersistentPull();
            duCurLim.trans.waitPersistentPull();
            // for (index iN = 0; iN < A.size(); iN++)
            //     for (auto &ME : A[iN])
            //         std::cout << "ij: " << iN << " " << ME.j << "\n"
            //                   << ME.m << std::endl;
            MPI::AllreduceOneIndex(nFix, MPI_SUM, mpi);
            MPI::AllreduceOneReal(minLim, MPI_MIN, mpi);
            if (mpi.rank == mRank)
            {
                log() << fmt::format("UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): Disp Limited, nLim {}, minLim {:.3e} ",
                                     nFix, minLim)
                      << std::endl;
            }
            return minLim;
        };

        auto MatVec = [&](tCoordPair &x, tCoordPair &Ax)
        {
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                Ax[iN].setZero();
                for (auto &ME : A[iN])
                    Ax[iN] += ME.m * x[ME.j];
            }
            Ax.trans.startPersistentPull();
            Ax.trans.waitPersistentPull();
        };

        struct InitForEigenSparse
        {
            using value_type = index;
            std::vector<std::vector<MatElem>> &A;
            index operator[](index i) const
            {
                return A[i / 3].size() * 3;
            }
        } initForEigenSparse{A};

        Eigen::SparseMatrix<real> ADiag(dispO2.father->Size() * 3, dispO2.father->Size() * 3);
        ADiag.reserve(initForEigenSparse);
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
        for (index iN = 0; iN < dispO2.father->Size(); iN++)
        {
            for (auto &ME : A[iN])
                if (ME.j < dispO2.father->Size())
                    for (int iii = 0; iii < 3; iii++)
                        for (int jjj = 0; jjj < 3; jjj++)
                            ADiag.insert(iN * 3 + iii, ME.j * 3 + jjj) =
                                ME.m(iii, jjj);
        }
        if (mpi.rank == mRank)
        {
            log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): Eigen Matrix Filled" << std::endl;
        }
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
        Eigen::SparseLU<Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int>> LUSolver;
        if (dispO2.father->Size())
        {
            // LUSolver.analyzePattern(ADiag);
            // LUSolver.factorize(ADiag); // full LU perconditioner
            // std::cout << "det " << LUSolver.absDeterminant() << std::endl;
        }
        if (mpi.rank == mRank)
        {
            log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): LU Factorized" << std::endl;
        }

        auto DiagSolveVec = [&](tCoordPair &x, tCoordPair &ADIx)
        {
            Eigen::Vector<real, Eigen::Dynamic> v;
            v.resize(x.father->Size() * 3);
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                v[iN * 3 + 0] = x[iN](0);
                v[iN * 3 + 1] = x[iN](1);
                v[iN * 3 + 2] = x[iN](2);
            }
            Eigen::Vector<real, Eigen::Dynamic> sol;
            if (dispO2.father->Size())
            {
                sol = LUSolver.solve(v);
            }
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                ADIx[iN](0) = sol[iN * 3 + 0];
                ADIx[iN](1) = sol[iN * 3 + 1];
                ADIx[iN](2) = sol[iN * 3 + 2];
            }
            ADIx.trans.startPersistentPull();
            ADIx.trans.waitPersistentPull();
        };
        auto JacobiPrec = [&](tCoordPair &x, tCoordPair &ADIx)
        {
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                ADIx[iN] = A[iN][0].m.fullPivLu().solve(x[iN]);
            }
            ADIx.trans.startPersistentPull();
            ADIx.trans.waitPersistentPull();
        };
        // for(int iIter = 1; iIter <= 200; iIter ++)
        // {
        //     real incC = UpdateDispO2Jacobi(dispO2, dispO2, false); // JOR iteration
        //     dispO2.trans.startPersistentPull();
        //     dispO2.trans.waitPersistentPull();
        //     if(mpi.rank == mRank)
        //         log() << fmt::format("iIter [{}] Jacobi Increment [{:3e}]", iIter,incC) << std::endl;
        // }

        /**********************************************/
        // Linear GMRES

        Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(5, initDOF);
        gmres.solve(
            [&](CoordPairDOF &x, CoordPairDOF &Ax)
            {
                MatVec(x, Ax);
            },
            [&](CoordPairDOF &x, CoordPairDOF &MLx)
            {
                JacobiPrec(x, MLx);
            },
            [&](CoordPairDOF &a, CoordPairDOF &b) -> real
            {
                return a.dot(b);
            },
            bO2,
            dispO2,
            elevationInfo.nIter,
            [&](int iRestart, real res, real resB)
            {
                // std::cout << mpi.rank << "gmres here" << std::endl;
                if (mpi.rank == mRank)
                    log() << fmt::format("iRestart [{}] res [{:3e}] -> [{:3e}]", iRestart, resB, res) << std::endl;
                return res < resB * 1e-10;
            });

        /**********************************************/
        // Nonlin

        // Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(5, initDOF);
        // real incNorm0{0}, rhsNorm0{0};
        // dispO2.setConstant(0.0);
        // for (int iIter = 1; iIter <= elevationInfo.nIter; iIter++)
        // {
        //     bO2.setConstant(0.0); // remember this
        //     AssembleRHSMat(dispO2);
        //     MatVec(dispO2, dispO2PrecRHS);  // "dispO2PrecRHS" = Ax
        //     bO2.addTo(dispO2PrecRHS, -1.0); // RHS
        //     real rhsNorm = bO2.norm2();
        //     dispO2Inc.setConstant(0.0);

        //     gmres.solve(
        //         [&](CoordPairDOF &x, CoordPairDOF &Ax)
        //         {
        //             MatVec(x, Ax);
        //         },
        //         [&](CoordPairDOF &x, CoordPairDOF &MLx)
        //         {
        //             JacobiPrec(x, MLx);
        //         },
        //         bO2,
        //         dispO2Inc,
        //         2,
        //         [&](int iRestart, real res, real resB)
        //         {
        //             // std::cout << mpi.rank << "gmres here" << std::endl;
        //             if (mpi.rank == mRank)
        //                 log() << fmt::format("step [{}]          iRestart [{}] res [{:3e}] -> [{:3e}]", iIter, iRestart, resB, res) << std::endl;
        //             return res < resB * 1e-10;
        //         });
        //     dispO2IncLimited = dispO2Inc;
        //     real minLim = LimitDisp(dispO2, dispO2Inc, dispO2IncLimited);
        //     dispO2IncLimited = dispO2Inc;
        //     dispO2IncLimited *= minLim;
        //     dispO2.addTo(dispO2IncLimited, 1.0);
        //     real incNorm = dispO2IncLimited.norm2();
        //     if (iIter == 1)
        //         incNorm0 = incNorm, rhsNorm0 = rhsNorm;
        //     if (mpi.rank == mRank)
        //         log() << fmt::format("step [{}] increment [{:3e}] -> [{:3e}]  rhs [{:3e}] -> [{:3e}]", iIter, incNorm0, incNorm, rhsNorm0, rhsNorm) << std::endl;
        // }
        /**********************************************/
        // dispO2 = bO2;

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (dim == 2)
                dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            coords[iN] += dispO2[iN];
        }
        coords.trans.pullOnce();
    }
}