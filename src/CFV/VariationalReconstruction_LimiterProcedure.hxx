#pragma once
#include "VariationalReconstruction.hpp"
#include "Limiters.hpp"

namespace DNDS::CFV
{
    /**
     * @brief Dispatches the biway limiter function selected by limiterBiwayAlter.
     *
     * Consolidates the switch(limiterBiwayAlter) block that appeared identically
     * in both DoLimiterWBAP_C and DoLimiterWBAP_3.
     */
    template <int dim, int nVarsFixed, typename Tin1, typename Tin2, typename Tout>
    inline void DispatchBiwayLimiter(int limiterBiwayAlter,
                                     const Tin1 &uThis, const Tin2 &uOther,
                                     Tout &uOut, real n)
    {
        switch (limiterBiwayAlter)
        {
        case 0:
            FWBAP_L2_Biway(uThis, uOther, uOut, n);
            break;
        case 1:
            FMINMOD_Biway(uThis, uOther, uOut, n);
            break;
        case 2:
            FWBAP_L2_Biway_PolynomialNorm<dim, nVarsFixed>(uThis, uOther, uOut, n);
            break;
        case 3:
            FMEMM_Biway_PolynomialNorm<dim, nVarsFixed>(uThis, uOther, uOut, n);
            break;
        case 4:
            FWBAP_L2_Cut_Biway(uThis, uOther, uOut, n);
            break;
        default:
            DNDS_assert_info(false, "no such limiterBiwayAlter code!");
        }
    }

    template <int dim>
    template <int nVarsFixed, int nVarsSee>
    void VariationalReconstruction<dim>::DoCalculateSmoothIndicator(
        tScalarPair &si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,
        const std::array<int, nVarsSee> &varsSee)
    {
        using namespace Geom;
        static const int maxNDiff = dim == 2 ? 10 : 20;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            // int NRecDOF = cellAtr[iCell].NDOF - 1; // ! not good ! TODO

            auto c2f = mesh->cell2face[iCell];
            Eigen::Matrix<real, nVarsSee, 2> IJIISIsum;
            IJIISIsum.setZero(nVarsSee, 2);
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = this->CellFaceOther(iCell, iFace);
                auto gFace = this->GetFaceQuadO1(iFace);
                decltype(IJIISIsum) IJIISI;
                IJIISI.setZero(nVarsSee, 2);
                gFace.IntegrationSimple(
                    IJIISI,
                    [&](auto &finc, int ig)
                    {
                        int nDiff = GetFaceAtr(iFace).NDIFF;
                        // int nDiff = 1;
                        tPoint unitNorm = faceMeanNorm[iFace];

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsSee, Eigen::DontAlign, maxNDiff, nVarsSee>
                            uRecVal(nDiff, nVarsSee), uRecValL(nDiff, nVarsSee), uRecValR(nDiff, nVarsSee), uRecValJump(nDiff, nVarsSee);
                        uRecVal.setZero(), uRecValJump.setZero();
                        uRecValL = this->GetIntPointDiffBaseValue(iCell, iFace, -1, -1, Eigen::seq(0, nDiff - 1)) *
                                   uRec[iCell](EigenAll, varsSee);
                        uRecValL(0, EigenAll) += u[iCell](varsSee).transpose();

                        if (iCellOther != UnInitIndex)
                        {
                            uRecValR = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, -1, Eigen::seq(0, nDiff - 1)) *
                                       uRec[iCellOther](EigenAll, varsSee);
                            uRecValR(0, EigenAll) += u[iCellOther](varsSee).transpose();
                            uRecVal = (uRecValL + uRecValR) * 0.5;
                            uRecValJump = (uRecValL - uRecValR) * 0.5;
                        }

                        Eigen::Matrix<real, nVarsSee, nVarsSee> IJI, ISI;
                        IJI = FFaceFunctional(uRecValJump, uRecValJump, iFace, iCell, iCellOther);
                        ISI = FFaceFunctional(uRecVal, uRecVal, iFace, iCell, iCellOther);

                        finc(EigenAll, 0) = IJI.diagonal();
                        finc(EigenAll, 1) = ISI.diagonal();

                        finc *= GetFaceArea(iFace); // don't forget this
                    });
                IJIISIsum += IJIISI;
            }
            Eigen::Vector<real, nVarsSee> smoothIndicator =
                (IJIISIsum(EigenAll, 0).array() /
                 (IJIISIsum(EigenAll, 1).array() + verySmallReal))
                    .matrix();
            real sImax = smoothIndicator.array().abs().maxCoeff();
            si(iCell, 0) = std::sqrt(sImax) * sqr(settings.maxOrder);
        }
    }

    template <int dim>
    template <int nVarsFixed>
    void VariationalReconstruction<dim>::DoCalculateSmoothIndicatorV1(
        tScalarPair &si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,
        const Eigen::Vector<real, nVarsFixed> &varsSee,
        const TFPost<nVarsFixed> &FPost)
    {
        using namespace Geom;
        static const int maxNDiff = dim == 2 ? 10 : 20;
        int nVars = u.RowSize();

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            // int NRecDOF = cellAtr[iCell].NDOF - 1; // ! not good ! TODO

            auto c2f = mesh->cell2face[iCell];
            Eigen::Matrix<real, nVarsFixed, 2> IJIISIsum;
            IJIISIsum.setZero(nVars, 2);
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = this->CellFaceOther(iCell, iFace);
                auto gFace = this->GetFaceQuadO1(iFace);
                decltype(IJIISIsum) IJIISI;
                // if (iCellOther != UnInitIndex)
                // {
                //     uRec[iCell].setConstant(1);
                //     uRec[iCellOther].setConstant(0);
                //     u[iCell].setConstant(1);
                //     u[iCellOther].setConstant(0);
                // }
                IJIISI.setZero(nVars, 2);
                gFace.IntegrationSimple(
                    IJIISI,
                    [&](auto &finc, int ig)
                    {
                        tPoint unitNorm = faceMeanNorm[iFace];

                        Eigen::Matrix<real, 1, nVarsFixed>
                            uRecVal(1, nVarsFixed), uRecValL(1, nVarsFixed), uRecValR(1, nVarsFixed), uRecValJump(1, nVarsFixed);
                        uRecVal.setZero(1, nVars), uRecValJump.setZero(1, nVars);
                        uRecValL = this->GetIntPointDiffBaseValue(iCell, iFace, -1, -1, std::array<int, 1>{0}, 1) *
                                   uRec[iCell];
                        uRecValL(0, EigenAll) += u[iCell].transpose();
                        FPost(uRecValL);

                        if (iCellOther != UnInitIndex)
                        {
                            uRecValR = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, -1, std::array<int, 1>{0}, 1) *
                                       uRec[iCellOther];
                            uRecValR(0, EigenAll) += u[iCellOther].transpose();
                            FPost(uRecValR);
                            uRecVal = (uRecValL + uRecValR) * 0.5;
                            uRecValJump = (uRecValL - uRecValR) * 0.5;
                        }

                        for (int i = 0; i < nVars; i++)
                        {
                            finc(i, 0) = FFaceFunctional(uRecValJump(EigenAll, {i}), uRecValJump(EigenAll, {i}), iFace, iCell, iCellOther)(0, 0);
                            finc(i, 1) = FFaceFunctional(uRecVal(EigenAll, {i}), uRecVal(EigenAll, {i}), iFace, iCell, iCellOther)(0, 0);
                        }
                        finc *= GetFaceArea(iFace); // don't forget this
                    });
                IJIISIsum += IJIISI;
            }
            Eigen::Vector<real, nVarsFixed> smoothIndicator =
                (IJIISIsum(EigenAll, 0).array() /
                 (IJIISIsum(EigenAll, 1).array() + verySmallReal))
                    .matrix();
            smoothIndicator.array() *= varsSee.array();
            real sImax = smoothIndicator.array().abs().maxCoeff();
            si(iCell, 0) = std::sqrt(sImax) * sqr(settings.maxOrder);
        }
    }

    template <int dim>
    template <int nVarsFixed>
    void VariationalReconstruction<dim>::DoLimiterWBAP_C(
        tUDof<nVarsFixed> &u,
        tURec<nVarsFixed> &uRec,
        tURec<nVarsFixed> &uRecNew,
        tURec<nVarsFixed> &uRecBuf,
        tScalarPair &si,
        bool ifAll,
        const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,
        bool putIntoNew)
    {
        using namespace Geom;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            if ((!ifAll) &&
                si(iCell, 0) < settings.smoothThreshold)
            {
                uRecNew[iCell] = uRec[iCell]; //! no lim need to copy !!!!
                continue;
            }
            index NRecDOF = GetCellAtr(iCell).NDOF - 1;
            auto c2f = mesh->cell2face[iCell];
            std::vector<Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOF>> uFaces(c2f.size());
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                // * safety initialization
                index iFace = c2f[ic2f];
                index iCellOther = this->CellFaceOther(iCell, iFace);
                if (iCellOther != UnInitIndex)
                {
                    uFaces[ic2f].resizeLike(uRec[iCellOther]);
                }
            }

            int cPOrder = settings.maxOrder;
            for (; cPOrder >= 1; cPOrder--)
            {
                auto [LimStart, LimEnd] = GetRecDOFRange<dim>(cPOrder);

                std::vector<Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>>
                    uOthers;
                Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                    uC = uRec[iCell](
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        EigenAll);
                uOthers.reserve(maxNeighbour);
                uOthers.push_back(uC); // using uC centered
                // DNDS_MPI_InsertCheck(mpi, "HereAAC");
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto f2c = mesh->face2cell[iFace];
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                    if (iCellOther != UnInitIndex)
                    {
                        index NRecDOFOther = GetCellAtr(iCellOther).NDOF - 1;
                        index NRecDOFLim = std::min(NRecDOFOther, NRecDOF);
                        if (NRecDOFLim < (LimEnd + 1))
                            continue; // reserved for p-adaption
                        // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                        //     continue;

                        tPoint unitNorm = faceMeanNorm[iFace];

                        const auto &matrixSecondaryThis =
                            this->GetMatrixSecondary(iCell, iFace, -1);

                        const auto &matrixSecondaryOther =
                            this->GetMatrixSecondary(iCellOther, iFace, -1);

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOF>
                            uOtherOther = uRec[iCellOther](Eigen::seq(0, NRecDOFLim - 1), EigenAll);

                        if (LimEnd < uOtherOther.rows() - 1) // successive SR
                            uOtherOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), EigenAll) =
                                matrixSecondaryOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::seq(LimEnd + 1, NRecDOFLim - 1)) *
                                uFaces[ic2f](Eigen::seq(LimEnd + 1, NRecDOFLim - 1), EigenAll);

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uOtherIn =
                                matrixSecondaryThis(Eigen::seq(LimStart, LimEnd), EigenAll) * uOtherOther;

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uThisIn =
                                uC.matrix();

                        // 2 eig space :
                        auto uR = iCellAtFace ? u[iCell] : u[iCellOther];
                        auto uL = iCellAtFace ? u[iCellOther] : u[iCell];

                        uOtherIn = (FM(uL, uR, unitNorm, uOtherIn));
                        uThisIn = (FM(uL, uR, unitNorm, uThisIn));

                        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uLimOutArray;

                        real n = settings.WBAP_nStd;
                        DispatchBiwayLimiter<dim, nVarsFixed>(
                            settings.limiterBiwayAlter,
                            uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);

                        // to phys space
                        uLimOutArray = (FMI(uL, uR, unitNorm, uLimOutArray.matrix())).array();

                        uFaces[ic2f](Eigen::seq(LimStart, LimEnd), EigenAll) = uLimOutArray.matrix();
                        uOthers.push_back(uLimOutArray);
                    }
                    else
                    {
                    }
                }
                Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                    uLimOutArray;

                real n = settings.WBAP_nStd;
                if (settings.normWBAP)
                    FWBAP_L2_Multiway_Polynomial2D(uOthers, uOthers.size(), uLimOutArray, n); // TODO: add 3D version here!
                else
                    FWBAP_L2_Multiway(uOthers, uOthers.size(), uLimOutArray, n);

                uRecNew[iCell](
                    Eigen::seq(
                        LimStart,
                        LimEnd),
                    EigenAll) = uLimOutArray.matrix();
            }
        }
        uRecNew.trans.startPersistentPull();
        uRecNew.trans.waitPersistentPull();
        if (!putIntoNew)
        {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                uRec[iCell] = uRecNew[iCell];
        }
    }

    template <int dim>
    template <int nVarsFixed>
    void VariationalReconstruction<dim>::DoLimiterWBAP_3(
        tUDof<nVarsFixed> &u,
        tURec<nVarsFixed> &uRec,
        tURec<nVarsFixed> &uRecNew,
        tURec<nVarsFixed> &uRecBuf,
        tScalarPair &si,
        bool ifAll,
        const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,
        bool putIntoNew)
    {
        using namespace Geom;

        int cPOrder = settings.maxOrder;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
            uRecNew[iCell] = uRec[iCell];
        for (; cPOrder >= 1; cPOrder--)
        {
            auto [LimStart, LimEnd] = GetRecDOFRange<dim>(cPOrder);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                uRecBuf[iCell] = uRecNew[iCell];
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if ((!ifAll) &&
                    si(iCell, 0) < settings.smoothThreshold)
                {
                    // uRecNew[iCell] = uRecBuf[iCell]; //! no copy for 3wbap!
                    continue;
                }
                index NRecDOF = GetCellAtr(iCell).NDOF - 1;
                auto c2f = mesh->cell2face[iCell];
                // std::vector<Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOF>> uFaces(c2f.size());
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    // * safety initialization
                    index iFace = c2f[ic2f];
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    if (iCellOther != UnInitIndex)
                    {
                        // uFaces[ic2f].resizeLike(uRec[iCellOther]);
                    }
                }

                std::vector<Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>>
                    uOthers;
                Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                    uC = uRecBuf[iCell](
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        EigenAll);
                uOthers.reserve(maxNeighbour);
                uOthers.push_back(uC); // using uC centered
                // DNDS_MPI_InsertCheck(mpi, "HereAAC");
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto f2c = mesh->face2cell[iFace];
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                    if (iCellOther != UnInitIndex)
                    {
                        index NRecDOFOther = GetCellAtr(iCellOther).NDOF - 1;
                        index NRecDOFLim = std::min(NRecDOFOther, NRecDOF);
                        if (NRecDOFLim < (LimEnd + 1))
                            continue; // reserved for p-adaption
                        // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                        //     continue;

                        tPoint unitNorm = faceMeanNorm[iFace];

                        const auto &matrixSecondaryThis =
                            this->GetMatrixSecondary(iCell, iFace, -1);

                        const auto &matrixSecondaryOther =
                            this->GetMatrixSecondary(iCellOther, iFace, -1);

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOF>
                            uOtherOther = uRecBuf[iCellOther](Eigen::seq(0, NRecDOFLim - 1), EigenAll);

                        // if (LimEnd < uOtherOther.rows() - 1) // successive SR
                        //     uOtherOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), EigenAll) =
                        //         matrixSecondaryOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::seq(LimEnd + 1, NRecDOFLim - 1)) *
                        //         uFaces[ic2f](Eigen::seq(LimEnd + 1, NRecDOFLim - 1), EigenAll);

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uOtherIn =
                                matrixSecondaryThis(Eigen::seq(LimStart, LimEnd), EigenAll) * uOtherOther;

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uThisIn =
                                uC.matrix();

                        // 2 eig space :
                        auto uR = iCellAtFace ? u[iCell] : u[iCellOther];
                        auto uL = iCellAtFace ? u[iCellOther] : u[iCell];

                        uOtherIn = FM(uL, uR, unitNorm, uOtherIn);
                        uThisIn = FM(uL, uR, unitNorm, uThisIn);

                        Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uLimOutArray;

                        real n = settings.WBAP_nStd;

                        DispatchBiwayLimiter<dim, nVarsFixed>(
                            settings.limiterBiwayAlter,
                            uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);

                        // to phys space
                        uLimOutArray = FMI(uL, uR, unitNorm, uLimOutArray.matrix()).array();

                        // uFaces[ic2f](Eigen::seq(LimStart, LimEnd), EigenAll) = uLimOutArray.matrix();
                        uOthers.push_back(uLimOutArray);
                    }
                    else
                    {
                    }
                }
                Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                    uLimOutArray;

                real n = settings.WBAP_nStd;
                if (settings.normWBAP)
                    FWBAP_L2_Multiway_Polynomial2D(uOthers, uOthers.size(), uLimOutArray, n);

                else
                    FWBAP_L2_Multiway(uOthers, uOthers.size(), uLimOutArray, n);

                uRecNew[iCell](
                    Eigen::seq(
                        LimStart,
                        LimEnd),
                    EigenAll) = uLimOutArray.matrix();
            }
            uRecNew.trans.startPersistentPull();
            uRecNew.trans.waitPersistentPull();
        }
        if (!putIntoNew)
        {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                uRec[iCell] = uRecNew[iCell];
        }
    }
}
