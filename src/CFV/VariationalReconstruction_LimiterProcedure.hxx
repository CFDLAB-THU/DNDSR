#pragma once
#include "VariationalReconstruction.hpp"
#include "Limiters.hpp"

namespace DNDS::CFV
{
    template <int dim>
    template <int nVarsFixed, int nVarsSee>
    void VariationalReconstruction<dim>::DoCalculateSmoothIndicator(
        tScalarPair &si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,
        const std::array<int, nVarsSee> &varsSee)
    {
        using namespace Geom;
        static const int maxNDiff = dim == 2 ? 10 : 20;

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
                        int nDiff = faceAtr[iFace].NDIFF;
                        // int nDiff = 1;
                        tPoint unitNorm = faceMeanNorm[iFace];

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsSee, Eigen::DontAlign, maxNDiff, nVarsSee>
                            uRecVal(nDiff, nVarsSee), uRecValL(nDiff, nVarsSee), uRecValR(nDiff, nVarsSee), uRecValJump(nDiff, nVarsSee);
                        uRecVal.setZero(), uRecValJump.setZero();
                        uRecValL = this->GetIntPointDiffBaseValue(iCell, iFace, -1, -1, Eigen::seq(0, nDiff - 1)) *
                                   uRec[iCell](Eigen::all, varsSee);
                        uRecValL(0, Eigen::all) += u[iCell](varsSee).transpose();

                        if (iCellOther != UnInitIndex)
                        {
                            uRecValR = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, -1, Eigen::seq(0, nDiff - 1)) *
                                       uRec[iCellOther](Eigen::all, varsSee);
                            uRecValR(0, Eigen::all) += u[iCellOther](varsSee).transpose();
                            uRecVal = (uRecValL + uRecValR) * 0.5;
                            uRecValJump = (uRecValL - uRecValR) * 0.5;
                        }

                        Eigen::Matrix<real, nVarsSee, nVarsSee> IJI, ISI;
                        IJI = FFaceFunctional(uRecValJump, uRecValJump, iFace, -1, iCell, iCellOther);
                        ISI = FFaceFunctional(uRecVal, uRecVal, iFace, -1, iCell, iCellOther);

                        finc(Eigen::all, 0) = IJI.diagonal();
                        finc(Eigen::all, 1) = ISI.diagonal();

                        finc *= GetFaceArea(iFace); // don't forget this

                        // if (iCell == 12517)
                        // {
                        //     std::cout << "   === Face:   ";
                        //     std::cout << uRecValL << std::endl;
                        //     std::cout << uRecValR << std::endl;
                        //     std::cout << IJI << std::endl;
                        //     std::cout << ISI << std::endl;
                        //     std::cout << uRec[iCell] << std::endl;
                        //     std::cout << uRec[iCellOther] << std::endl;
                        //     std::cout << this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, ig, Eigen::all) << std::endl;
                        // }
                    });
                IJIISIsum += IJIISI;
                // if (iCell == 12517)
                // {
                //     std::cout << "iFace " << iFace << " iCellOther " << iCellOther << std::endl;
                //     std::cout << IJIISI << std::endl;
                // }
            }
            Eigen::Vector<real, nVarsSee> smoothIndicator =
                (IJIISIsum(Eigen::all, 0).array() /
                 (IJIISIsum(Eigen::all, 1).array() + verySmallReal))
                    .matrix();
            real sImax = smoothIndicator.array().abs().maxCoeff();
            si(iCell, 0) = std::sqrt(sImax) * sqr(settings.maxOrder);
            // if (iCell == 12517)
            // {
            //     std::cout << "SUM:\n";
            //     std::cout << IJIISIsum << std::endl;
            //     std::abort();
            // }
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
                        uRecValL(0, Eigen::all) += u[iCell].transpose();
                        FPost(uRecValL);

                        if (iCellOther != UnInitIndex)
                        {
                            uRecValR = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, -1, std::array<int, 1>{0}, 1) *
                                       uRec[iCellOther];
                            uRecValR(0, Eigen::all) += u[iCellOther].transpose();
                            FPost(uRecValR);
                            uRecVal = (uRecValL + uRecValR) * 0.5;
                            uRecValJump = (uRecValL - uRecValR) * 0.5;
                        }

                        for (int i = 0; i < nVars; i++)
                        {
                            finc(i, 0) = FFaceFunctional(uRecValJump(Eigen::all, {i}), uRecValJump(Eigen::all, {i}), iFace, -1, iCell, iCellOther)(0, 0);
                            finc(i, 1) = FFaceFunctional(uRecVal(Eigen::all, {i}), uRecVal(Eigen::all, {i}), iFace, -1, iCell, iCellOther)(0, 0);
                        }
                        finc *= GetFaceArea(iFace); // don't forget this
                    });
                IJIISIsum += IJIISI;
            }
            Eigen::Vector<real, nVarsFixed> smoothIndicator =
                (IJIISIsum(Eigen::all, 0).array() /
                 (IJIISIsum(Eigen::all, 1).array() + verySmallReal))
                    .matrix();
            smoothIndicator.array() *= varsSee.array();
            real sImax = smoothIndicator.array().abs().maxCoeff();
            si(iCell, 0) = std::sqrt(sImax) * sqr(settings.maxOrder);
            // if (iCell == 12517)
            // {
            //     std::cout << "SUM:\n";
            //     std::cout << IJIISIsum << std::endl;
            //     std::abort();
            // }
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

        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            if ((!ifAll) &&
                si(iCell, 0) < settings.smoothThreshold)
            {
                uRecNew[iCell] = uRec[iCell]; //! no lim need to copy !!!!
                continue;
            }
            index NRecDOF = cellAtr[iCell].NDOF - 1;
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
                int LimStart, LimEnd; // End is inclusive
                if constexpr (dim == 2)
                    switch (cPOrder)
                    {
                    case 3:
                        LimStart = 5, LimEnd = 8;
                        break;
                    case 2:
                        LimStart = 2, LimEnd = 4;
                        break;
                    case 1:
                        LimStart = 0, LimEnd = 1;
                        break;
                    default:
                        LimStart = -200, LimEnd = -100;
                        DNDS_assert(false);
                    }
                else
                    switch (cPOrder)
                    {
                    case 3:
                        LimStart = 9, LimEnd = 18;
                        break;
                    case 2:
                        LimStart = 3, LimEnd = 8;
                        break;
                    case 1:
                        LimStart = 0, LimEnd = 2;
                        break;
                    default:
                        LimStart = -200, LimEnd = -100;
                        DNDS_assert(false);
                    }

                std::vector<Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>>
                    uOthers;
                Eigen::Array<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                    uC = uRec[iCell](
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all);
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
                        index NRecDOFOther = cellAtr[iCellOther].NDOF - 1;
                        index NRecDOFLim = std::min(NRecDOFOther, NRecDOF);
                        if (NRecDOFLim < (LimEnd + 1))
                            continue; // reserved for p-adaption
                        // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                        //     continue;

                        tPoint unitNorm = faceMeanNorm[iFace];

                        const auto &matrixSecondary =
                            this->GetMatrixSecondary(iCell, iFace, -1);

                        const auto &matrixSecondaryOther =
                            this->GetMatrixSecondary(iCellOther, iFace, -1);

                        // std::cout << "A"<<std::endl;
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOF>
                            uOtherOther = uRec[iCellOther](Eigen::seq(0, NRecDOFLim - 1), Eigen::all);

                        if (LimEnd < uOtherOther.rows() - 1) // successive SR
                            uOtherOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all) =
                                matrixSecondaryOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::seq(LimEnd + 1, NRecDOFLim - 1)) *
                                uFaces[ic2f](Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all);

                        // std::cout << "B" << std::endl;
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uOtherIn =
                                matrixSecondary(Eigen::seq(LimStart, LimEnd), Eigen::all) * uOtherOther;

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
                        switch (settings.limiterBiwayAlter)
                        {
                        case 0:
                            FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 1:
                            FMINMOD_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 2:
                            FWBAP_L2_Biway_PolynomialNorm<dim, nVarsFixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 3:
                            FMEMM_Biway_PolynomialNorm<dim, nVarsFixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 4:
                            FWBAP_L2_Cut_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        default:
                            DNDS_assert_info(false, "no such limiterBiwayAlter code!");
                        }

                        // to phys space
                        uLimOutArray = (FMI(uL, uR, unitNorm, uLimOutArray.matrix())).array();

                        uFaces[ic2f](Eigen::seq(LimStart, LimEnd), Eigen::all) = uLimOutArray.matrix();
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
                    Eigen::all) = uLimOutArray.matrix();
            }
        }
        uRecNew.trans.startPersistentPull();
        uRecNew.trans.waitPersistentPull();
        if (!putIntoNew)
        {
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
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
            uRecNew[iCell] = uRec[iCell];
        for (; cPOrder >= 1; cPOrder--)
        {
            int LimStart, LimEnd; // End is inclusive
            if constexpr (dim == 2)
                switch (cPOrder)
                {
                case 3:
                    LimStart = 5, LimEnd = 8;
                    break;
                case 2:
                    LimStart = 2, LimEnd = 4;
                    break;
                case 1:
                    LimStart = 0, LimEnd = 1;
                    break;
                default:
                    LimStart = -200, LimEnd = -100;
                    DNDS_assert(false);
                }
            else
                switch (cPOrder)
                {
                case 3:
                    LimStart = 9, LimEnd = 18;
                    break;
                case 2:
                    LimStart = 3, LimEnd = 8;
                    break;
                case 1:
                    LimStart = 0, LimEnd = 2;
                    break;
                default:
                    LimStart = -200, LimEnd = -100;
                    DNDS_assert(false);
                }
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                uRecBuf[iCell] = uRecNew[iCell];
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if ((!ifAll) &&
                    si(iCell, 0) < settings.smoothThreshold)
                {
                    // uRecNew[iCell] = uRecBuf[iCell]; //! no copy for 3wbap!
                    continue;
                }
                index NRecDOF = cellAtr[iCell].NDOF - 1;
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
                        Eigen::all);
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
                        index NRecDOFOther = cellAtr[iCellOther].NDOF - 1;
                        index NRecDOFLim = std::min(NRecDOFOther, NRecDOF);
                        if (NRecDOFLim < (LimEnd + 1))
                            continue; // reserved for p-adaption
                        // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                        //     continue;

                        tPoint unitNorm = faceMeanNorm[iFace];

                        const auto &matrixSecondary =
                            this->GetMatrixSecondary(iCell, iFace, -1);

                        const auto &matrixSecondaryOther =
                            this->GetMatrixSecondary(iCellOther, iFace, -1);

                        // std::cout << "A"<<std::endl;
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOF>
                            uOtherOther = uRecBuf[iCellOther](Eigen::seq(0, NRecDOFLim - 1), Eigen::all);

                        // if (LimEnd < uOtherOther.rows() - 1) // successive SR
                        //     uOtherOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all) =
                        //         matrixSecondaryOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::seq(LimEnd + 1, NRecDOFLim - 1)) *
                        //         uFaces[ic2f](Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all);

                        // std::cout << "B" << std::endl;
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed, 0, maxRecDOFBatch>
                            uOtherIn =
                                matrixSecondary(Eigen::seq(LimStart, LimEnd), Eigen::all) * uOtherOther;

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

                        switch (settings.limiterBiwayAlter)
                        {
                        case 0:
                            FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 1:
                            FMINMOD_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 2:
                            FWBAP_L2_Biway_PolynomialNorm<dim, nVarsFixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 3:
                            FMEMM_Biway_PolynomialNorm<dim, nVarsFixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        case 4:
                            FWBAP_L2_Cut_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            break;
                        default:
                            DNDS_assert_info(false, "no such limiterBiwayAlter code!");
                        }

                        // to phys space
                        uLimOutArray = FMI(uL, uR, unitNorm, uLimOutArray.matrix()).array();

                        // uFaces[ic2f](Eigen::seq(LimStart, LimEnd), Eigen::all) = uLimOutArray.matrix();
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
                    Eigen::all) = uLimOutArray.matrix();
            }
            uRecNew.trans.startPersistentPull();
            uRecNew.trans.waitPersistentPull();
        }
        if (!putIntoNew)
        {
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                uRec[iCell] = uRecNew[iCell];
        }
    }
}