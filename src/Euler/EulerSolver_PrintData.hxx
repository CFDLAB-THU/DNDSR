#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    template <EulerModel model>
    void EulerSolver<model>::PrintData(
        const std::string &fname,
        const std::string &fnameSeries,
        const tCellScalarFGet &odeResidualF,
        tAdditionalCellScalarList &additionalCellScalars,
        TEval &eval, real tSimu, PrintDataMode mode)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        reader->SetASCIIPrecision(config.dataIOControl.nASCIIPrecision);
        reader->SetVTKFloatEncodeMode(config.dataIOControl.vtuFloatEncodeMode);
        const int cDim = dim;

        ArrayDOFV<nVarsFixed> &uOut = mode == PrintDataTimeAverage ? uAveraged : u;

        if (config.dataIOControl.outVolumeData || mode == PrintDataTimeAverage)
        {
            if (config.dataIOControl.outAtCellData || mode == PrintDataTimeAverage)
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    // TU recu =
                    //     vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 1>{0}, 1) *
                    //     uRec[iCell];
                    // recu += uOut[iCell];
                    // recu = EulerEvaluator::CompressRecPart(uOut[iCell], recu);
                    TU recu = uOut[iCell];
                    TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                    real vsqr = velo.squaredNorm();
                    real asqr, p, H;
                    Gas::IdealGasThermal(recu(I4), recu(0), vsqr, eval.settings.idealGasProperty.gamma, p, asqr, H);
                    // DNDS_assert(asqr > 0);
                    real M = std::sqrt(std::abs(vsqr / asqr));
                    real T = p / recu(0) / eval.settings.idealGasProperty.Rgas;

                    (*outDist)[iCell][0] = recu(0);
                    for (int i = 0; i < dim; i++)
                        (*outDist)[iCell][i + 1] = velo(i);
                    (*outDist)[iCell][I4 + 0] = p;
                    (*outDist)[iCell][I4 + 1] = T;
                    (*outDist)[iCell][I4 + 2] = M;
                    // (*outDist)[iCell][7] = (bool)(ifUseLimiter[iCell] & 0x0000000FU);
                    (*outDist)[iCell][I4 + 3] = ifUseLimiter[iCell][0] / (vfv->settings.smoothThreshold + verySmallReal);
                    // std::cout << iCell << ode.rhsbuf[0][iCell] << std::endl;
                    (*outDist)[iCell][I4 + 4] = odeResidualF(iCell);
                    // { // see the cond
                    //     auto A = vfv->GetCellRecMatA(iCell);
                    //     Eigen::MatrixXd AInv = A;
                    //     real aCond = HardEigen::EigenLeastSquareInverse(A, AInv);
                    //     (*outDist)[iCell][I4 + 4] = aCond;
                    // }
                    // (*outDist)[iCell][8] = (*vfv->SOR_iCell2iScan)[iCell];//!using SOR rb seq instead

                    for (int i = I4 + 1; i < nVars; i++)
                    {
                        (*outDist)[iCell][4 + i] = recu(i) / recu(0); // 4 is additional amount offset, not Index of last flow variable (I4)
                    }
                    int iCur = 4 + nVars;
                    for (auto &out : additionalCellScalars)
                    {
                        (*outDist)[iCell][iCur] = std::get<1>(out)(iCell);
                        iCur++;
                    }
                }

            if (config.dataIOControl.outAtPointData)
            {
                if (config.limiterControl.useLimiter)
                {
                    uRecNew.trans.startPersistentPull();
                    uRecNew.trans.waitPersistentPull();
                }
                else
                {
                    uRec.trans.startPersistentPull();
                    uRec.trans.waitPersistentPull();
                }

                uOut.trans.startPersistentPull();
                uOut.trans.waitPersistentPull();

                for (index iN = 0; iN < mesh->NumNodeProc(); iN++)
                    outDistPointPair[iN].setZero();
                std::vector<int> nN2C(mesh->NumNodeProc(), 0);
                DNDS_assert(outDistPointPair.father->Size() == mesh->NumNode());
                DNDS_assert(outDistPointPair.son->Size() == mesh->NumNodeGhost());
                for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) //! all cells
                {
                    for (int ic2n = 0; ic2n < mesh->cell2node.RowSize(iCell); ic2n++)
                    {
                        auto iNode = mesh->cell2node(iCell, ic2n);
                        nN2C.at(iNode)++;
                        auto pPhy = mesh->GetCoordNodeOnCell(iCell, ic2n);

                        Eigen::Matrix<real, 1, Eigen::Dynamic> DiBj;
                        DiBj.resize(1, uRecNew[iCell].rows() + 1);
                        // std::cout << uRecNew[iCell].rows() << std::endl;
                        vfv->FDiffBaseValue(DiBj, pPhy, iCell, -2, -2);

                        TU vRec = (DiBj(Eigen::all, Eigen::seq(1, Eigen::last)) * (config.limiterControl.useLimiter ? uRecNew[iCell] : uRec[iCell])).transpose() +
                                  uOut[iCell];
                        if (mode == PrintDataTimeAverage)
                            vRec = uOut[iCell];
                        if (iNode < mesh->NumNode())
                            outDistPointPair[iNode](Eigen::seq(0, nVars - 1)) += vRec;
                    }
                }

                for (index iN = 0; iN < mesh->NumNode(); iN++)
                {
                    TU recu = outDistPointPair[iN](Eigen::seq(0, nVars - 1)) / (nN2C.at(iN) + verySmallReal);
                    DNDS_assert(nN2C.at(iN) > 0);

                    TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                    real vsqr = velo.squaredNorm();
                    real asqr, p, H;
                    Gas::IdealGasThermal(recu(I4), recu(0), vsqr, eval.settings.idealGasProperty.gamma, p, asqr, H);
                    // DNDS_assert(asqr > 0);
                    real M = std::sqrt(std::abs(vsqr / asqr));
                    real T = p / recu(0) / eval.settings.idealGasProperty.Rgas;

                    outDistPointPair[iN][0] = recu(0);
                    for (int i = 0; i < dim; i++)
                        outDistPointPair[iN][i + 1] = velo(i);
                    outDistPointPair[iN][I4 + 0] = p;
                    outDistPointPair[iN][I4 + 1] = T;
                    outDistPointPair[iN][I4 + 2] = M;

                    for (int i = I4 + 1; i < nVars; i++)
                    {
                        outDistPointPair[iN][2 + i] = recu(i) / recu(0); // 2 is additional amount offset
                    }
                }
                outDistPointPair.trans.startPersistentPull();
                outDistPointPair.trans.waitPersistentPull();
            }

            int NOUTS_C{0}, NOUTSPoint_C{0};
            if (config.dataIOControl.outAtCellData || mode == PrintDataTimeAverage)
                NOUTS_C = nOUTS;
            if (config.dataIOControl.outAtPointData)
                NOUTSPoint_C = nOUTSPoint;

            if (config.dataIOControl.outPltMode == 0)
            {
                if (config.dataIOControl.outAtCellData || mode == PrintDataTimeAverage)
                {
                    outDist2SerialTrans.startPersistentPull();
                    outDist2SerialTrans.waitPersistentPull();
                }
                if (config.dataIOControl.outAtPointData)
                {
                    outDist2SerialTransPoint.startPersistentPull();
                    outDist2SerialTransPoint.waitPersistentPull();
                }
            }

            std::vector<std::string> names, namesPoint;
            if constexpr (dim == 2)
                names = {
                    "R", "U", "V", "P", "T", "M", "ifUseLimiter", "RHSr"};
            else
                names = {
                    "R", "U", "V", "W", "P", "T", "M", "ifUseLimiter", "RHSr"};
            if constexpr (dim == 2)
                namesPoint = {
                    "R", "U", "V", "P", "T", "M"};
            else
                namesPoint = {
                    "R", "U", "V", "W", "P", "T", "M"};
            for (int i = I4 + 1; i < nVars; i++)
            {
                names.push_back("V" + std::to_string(i - I4));
                namesPoint.push_back("V" + std::to_string(i - I4));
            }
            for (auto &out : additionalCellScalars)
            {
                names.push_back(std::get<0>(out));
            }
            if (config.dataIOControl.outAtCellData)
                DNDS_assert(names.size() == NOUTS_C);
            if (config.dataIOControl.outAtPointData)
                DNDS_assert(namesPoint.size() == NOUTSPoint_C);

            if (config.dataIOControl.outPltTecplotFormat)
            {
                if (config.dataIOControl.outPltMode == 0)
                {
                    reader->PrintSerialPartPltBinaryDataArray(
                        fname,
                        NOUTS_C, NOUTSPoint_C,
                        [&](int idata)
                        { return names[idata]; }, // cellNames
                        [&](int idata, index iv)
                        {
                            return (*outSerial)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        { return namesPoint[idata] + "_p"; }, // pointNames
                        [&](int idata, index in)
                        { return (*outSerialPoint)[in][idata]; }, // pointData
                        tSimu,
                        0);
                }
                else if (config.dataIOControl.outPltMode == 1)
                {

                    reader->PrintSerialPartPltBinaryDataArray(
                        fname,
                        NOUTS_C, NOUTSPoint_C,
                        [&](int idata)
                        { return names[idata]; }, // cellNames
                        [&](int idata, index iv)
                        {
                            return (*outDist)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        { return namesPoint[idata] + "_p"; }, // pointNames
                        [&](int idata, index in)
                        { return outDistPointPair[in][idata]; }, // pointData
                        tSimu,
                        1);
                }
            }

            if (config.dataIOControl.outPltVTKFormat)
            {
                if (config.dataIOControl.outPltMode == 0)
                {
                    reader->PrintSerialPartVTKDataArray(
                        fname, fnameSeries,
                        std::max(NOUTS_C - cDim, 0), std::min(NOUTS_C, 1),
                        std::max(NOUTSPoint_C - cDim, 0), std::min(NOUTSPoint_C, 1), //! vectors number is not cDim but 1
                        [&](int idata)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return names[idata]; // cellNames
                        },
                        [&](int idata, index iv)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return (*outSerial)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        {
                            return "Velo"; // cellVecNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            return (*outSerial)[iv][1 + idim]; // cellVecData
                        },
                        [&](int idata)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return namesPoint[idata]; // pointNames
                        },
                        [&](int idata, index iv)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return (*outSerialPoint)[iv][idata]; // pointData
                        },
                        [&](int idata)
                        {
                            return "Velo"; // pointVecNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            idata += 1;
                            return (*outSerialPoint)[iv][1 + idim]; // pointVecData
                        },
                        tSimu,
                        0);
                }
                else if (config.dataIOControl.outPltMode == 1)
                {
                    reader->PrintSerialPartVTKDataArray(
                        fname, fnameSeries,
                        std::max(NOUTS_C - cDim, 0), std::min(NOUTS_C, 1),
                        std::max(NOUTSPoint_C - cDim, 0), std::min(NOUTSPoint_C, 1), //! vectors number is not cDim but 1
                        [&](int idata)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return names[idata]; // cellNames
                        },
                        [&](int idata, index iv)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return (*outDist)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        {
                            return "Velo"; // cellVecNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            return idim < cDim ? (*outDist)[iv][1 + idim] : 0.0; // cellVecData
                        },
                        [&](int idata)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return namesPoint[idata]; // pointNames
                        },
                        [&](int idata, index iv)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return outDistPointPair[iv][idata]; // pointData
                        },
                        [&](int idata)
                        {
                            return "Velo"; // pointVecNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            return idim < cDim ? outDistPointPair[iv][1 + idim] : 0.0; // pointVecData
                        },
                        tSimu,
                        1);
                }
            }
        }

        if (config.dataIOControl.outBndData)
        {
            DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd enter");
            for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
            {
                // TU recu =
                //     vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 1>{0}, 1) *
                //     uRec[iCell];
                // recu += uOut[iCell];
                // recu = EulerEvaluator::CompressRecPart(uOut[iCell], recu);
                index iCell = mesh->bnd2cell[iBnd][0];
                index iFace = mesh->bnd2face.at(iBnd);
                if (iFace == -1)
                {
                    DNDS_assert(mesh->isPeriodic);                                  // only internal bnd is valid, periodic bnd should be omitted
                    (*outDistBnd)[iBnd](nOUTSBnd - 4) = meshBnd->GetCellZone(iBnd); // add this to enable blanking
                    continue;
                }

                TU recu = uOut[iCell];
                TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                real vsqr = velo.squaredNorm();
                real asqr, p, H;
                Gas::IdealGasThermal(recu(I4), recu(0), vsqr, eval.settings.idealGasProperty.gamma, p, asqr, H);
                // DNDS_assert(asqr > 0);
                real M = std::sqrt(std::abs(vsqr / asqr));
                real T = p / recu(0) / eval.settings.idealGasProperty.Rgas;

                (*outDistBnd)[iBnd][0] = recu(0);
                for (int i = 0; i < dim; i++)
                    (*outDistBnd)[iBnd][i + 1] = velo(i);
                (*outDistBnd)[iBnd][I4 + 0] = p;
                (*outDistBnd)[iBnd][I4 + 1] = T;
                (*outDistBnd)[iBnd][I4 + 2] = M;
                for (int i = I4 + 1; i < nVars; i++)
                {
                    (*outDistBnd)[iBnd][2 + i] = recu(i) / recu(0); // 4 is additional amount offset, not Index of last flow variable (I4)
                }
                // if(iFace < 0)
                // {
                //     std::cout << iFace << std::endl;
                //     std::abort();
                // }

                (*outDistBnd)[iBnd](Eigen::seq(nVars + 2, nOUTSBnd - 5)) = eval.fluxBnd.at(iBnd);
                // (*outDistBnd)[iBnd](nOUTSBnd - 4) = mesh->GetFaceZone(iFace);
                (*outDistBnd)[iBnd](nOUTSBnd - 4) = meshBnd->GetCellZone(iBnd);
                (*outDistBnd)[iBnd](Eigen::seq(nOUTSBnd - 3, nOUTSBnd - 1)) = vfv->GetFaceNorm(iFace, 0) * vfv->GetFaceArea(iFace);

                // (*outDist)[iCell][8] = (*vfv->SOR_iCell2iScan)[iCell];//!using SOR rb seq instead
            }

            int NOUTS_C{0}, NOUTSPoint_C{0};
            NOUTS_C = nOUTSBnd;
            DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd transfer done");

            if (config.dataIOControl.outPltMode == 0)
            {
                outDist2SerialTransBnd.startPersistentPull();
                outDist2SerialTransBnd.waitPersistentPull();
            }

            std::vector<std::string> names;
            if constexpr (dim == 2)
                names = {
                    "R", "U", "V", "P", "T", "M"};
            else
                names = {
                    "R", "U", "V", "W", "P", "T", "M"};
            for (int i = I4 + 1; i < nVars; i++)
            {
                names.push_back("V" + std::to_string(i - I4));
            }
            for (int i = 0; i < nVars; i++)
            {
                names.push_back("F" + std::to_string(i));
            }
            names.push_back("FaceZone");
            names.push_back("N0");
            names.push_back("N1");
            names.push_back("N2");

            if (config.dataIOControl.outPltTecplotFormat)
            {
                DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd tecplot start");
                if (config.dataIOControl.outPltMode == 0)
                {
                    readerBnd->PrintSerialPartPltBinaryDataArray(
                        fname + "_bnd",
                        NOUTS_C, 0,
                        [&](int idata)
                        { return names[idata]; }, // cellNames
                        [&](int idata, index iv)
                        {
                            return (*outSerialBnd)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        { return "ERROR"; }, // pointNames
                        [&](int idata, index in)
                        { return std::nan("0"); }, // pointData
                        tSimu,
                        0);
                }
                else if (config.dataIOControl.outPltMode == 1)
                {

                    readerBnd->PrintSerialPartPltBinaryDataArray(
                        fname + "_bnd",
                        NOUTS_C, NOUTSPoint_C,
                        [&](int idata)
                        { return names[idata]; }, // cellNames
                        [&](int idata, index iv)
                        {
                            return (*outDistBnd)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        { return "ERROR"; }, // pointNames
                        [&](int idata, index in)
                        { return std::nan("0"); }, // pointData
                        tSimu,
                        1);
                }
            }

            const int cDim = dim;
            if (config.dataIOControl.outPltVTKFormat)
            {
                DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd vtk start");
                if (config.dataIOControl.outPltMode == 0)
                {
                    readerBnd->PrintSerialPartVTKDataArray(
                        fname + "_bnd",
                        fnameSeries + "_bnd",
                        NOUTS_C - cDim - 3, 2,
                        0, 0, //! vectors number is not cDim but 2
                        [&](int idata)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return names[idata]; // cellNames
                        },
                        [&](int idata, index iv)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return (*outSerialBnd)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        {
                            return idata == 0 ? "Velo" : "Norm"; // cellVecNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            if (idata == 0)
                                return idim < cDim ? (*outSerialBnd)[iv][1 + idim] : 0; // cellVecData
                            else
                                return (*outSerialBnd)[iv][nOUTSBnd - 3 + idim];
                        },
                        [&](int idata)
                        {
                            return "error"; // pointNames
                        },
                        [&](int idata, index iv)
                        {
                            return std::nan("0"); // pointData
                        },
                        [&](int idata)
                        {
                            return "error"; // pointNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            return std::nan("0"); // pointData
                        },
                        tSimu,
                        0);
                }
                else if (config.dataIOControl.outPltMode == 1)
                {
                    readerBnd->PrintSerialPartVTKDataArray(
                        fname + "_bnd",
                        fnameSeries + "_bnd",
                        NOUTS_C - cDim - 3, 2,
                        0, 0, //! vectors number is not cDim but 2
                        [&](int idata)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return names[idata]; // cellNames
                        },
                        [&](int idata, index iv)
                        {
                            idata = idata > 0 ? idata + cDim : 0;
                            return (*outDistBnd)[iv][idata]; // cellData
                        },
                        [&](int idata)
                        {
                            return idata == 0 ? "Velo" : "Norm"; // cellVecNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            if (idata == 0)
                                return idim < cDim ? (*outDistBnd)[iv][1 + idim] : 0; // cellVecData
                            else
                                return (*outDistBnd)[iv][nOUTSBnd - 3 + idim];
                        },
                        [&](int idata)
                        {
                            return "error"; // pointNames
                        },
                        [&](int idata, index iv)
                        {
                            return std::nan("0"); // pointData
                        },
                        [&](int idata)
                        {
                            return "error"; // pointNames
                        },
                        [&](int idata, index iv, int idim)
                        {
                            return std::nan("0"); // pointData
                        },
                        tSimu,
                        1);
                }
            }
            DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd output done");
        }
    }
}