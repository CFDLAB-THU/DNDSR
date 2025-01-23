#pragma once

#include <future>
#include <hdf5.h>
#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    static const auto model = NS;
    DNDS_SWITCH_INTELLISENSE(template <EulerModel model>, )
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
        mesh->SetHDF5OutSetting(config.dataIOControl.hdfChunkSize, config.dataIOControl.hdfDeflateLevel);
        const int cDim = dim;

        ArrayDOFV<nVarsFixed> &uOut = mode == PrintDataTimeAverage ? uAveraged : u;
        // int nBad;
        // do
        // {
        //     nBad = 0;
        //     for (auto &f : outFuture)
        //         if (f.valid() && f.wait_for(std::chrono::microseconds(10)) != std::future_status::ready)
        //             nBad++;
        //     for (auto &f : outBndFuture)
        //         if (f.valid() && f.wait_for(std::chrono::microseconds(10)) != std::future_status::ready)
        //             nBad++;
        // } while (nBad);

        std::vector<std::function<void()>> fOuts;

        if (config.dataIOControl.outVolumeData || mode == PrintDataTimeAverage)
        {
            {
                std::lock_guard<std::mutex> outLock(outArraysMutex);
                if (config.dataIOControl.outAtCellData || mode == PrintDataTimeAverage)
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    {
                        // TU recu =
                        //     vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 1>{0}, 1) *
                        //     uRec[iCell];
                        // recu += uOut[iCell];
                        // recu = EulerEvaluator::CompressRecPart(uOut[iCell], recu);
                        TU recu = uOut[iCell];
                        if (eval.settings.frameConstRotation.enabled)
                            eval.TransformURotatingFrame_ABS_VELO(recu, vfv->GetCellQuadraturePPhys(iCell, -1), -1);
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

                            RowVectorXR DiBj;
                            DiBj.resize(1, uRecNew[iCell].rows() + 1);
                            // std::cout << uRecNew[iCell].rows() << std::endl;
                            vfv->FDiffBaseValue(DiBj, pPhy, iCell, -2, -2);

                            TU vRec = (DiBj(Eigen::all, Eigen::seq(1, Eigen::last)) * (config.limiterControl.useLimiter ? uRecNew[iCell] : uRec[iCell])).transpose() +
                                      uOut[iCell];
                            if (mesh->isPeriodic) // transform velocity to node reference frame
                                vRec(Seq123) = mesh->periodicInfo.GetVectorBackByBits<dim, 1>(vRec(Seq123), mesh->cell2nodePbi(iCell, ic2n));
                            if (mode == PrintDataTimeAverage)
                                vRec = uOut[iCell];
                            if (eval.settings.frameConstRotation.enabled)
                                eval.TransformURotatingFrame_ABS_VELO(vRec, pPhy, -1);
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
            }
            int NOUTS_C{0}, NOUTSPoint_C{0};
            if (config.dataIOControl.outAtCellData || mode == PrintDataTimeAverage)
                NOUTS_C = nOUTS;
            if (config.dataIOControl.outAtPointData)
                NOUTSPoint_C = nOUTSPoint;

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
                    auto outRun = [mesh = mesh, reader = reader,
                                   outDist = outDist, outSerial = outSerial, &outDistPointPair = outDistPointPair,
                                   outSerialPoint = outSerialPoint,
                                   fname, fnameSeries, NOUTS_C, NOUTSPoint_C, cDim,
                                   names, namesPoint, tSimu,
                                   &outArraysMutex = outArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outArraysLock(outArraysMutex);
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
                    };
                    // if (outFuture.at(0).valid())
                    //     outFuture.at(0).wait();
                    // outFuture.at(0) = std::async(std::launch::async, outRun);
                    outRun();
                }
                else if (config.dataIOControl.outPltMode == 1)
                {

                    auto outRun = [mesh = mesh, reader = reader,
                                   outDist = outDist, outSerial = outSerial, &outDistPointPair = outDistPointPair,
                                   outSerialPoint = outSerialPoint,
                                   fname, fnameSeries, NOUTS_C, NOUTSPoint_C, cDim,
                                   names, namesPoint, tSimu,
                                   &outArraysMutex = outArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outArraysLock(outArraysMutex);
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
                    };
                    // if (outFuture.at(0).valid())
                    //     outFuture.at(0).wait();
                    // outFuture.at(0) = std::async(std::launch::async, outRun);
                    outRun();
                }
            }

            if (config.dataIOControl.outPltVTKFormat)
            {
                if (config.dataIOControl.outPltMode == 0)
                {
                    auto outRun = [mesh = mesh, reader = reader,
                                   outDist = outDist, outSerial = outSerial, &outDistPointPair = outDistPointPair,
                                   outSerialPoint = outSerialPoint,
                                   fname, fnameSeries, NOUTS_C, NOUTSPoint_C, cDim,
                                   names, namesPoint, tSimu,
                                   &outArraysMutex = outArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outArraysLock(outArraysMutex);
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
                    };
                    // if (outFuture.at(1).valid())
                    //     outFuture.at(1).wait();
                    // outFuture.at(1) = std::async(std::launch::async, outRun);
                    fOuts.push_back(outRun);
                }
                else if (config.dataIOControl.outPltMode == 1)
                {
                    auto outRun = [mesh = mesh, reader = reader,
                                   outDist = outDist, outSerial = outSerial, &outDistPointPair = outDistPointPair,
                                   outSerialPoint = outSerialPoint,
                                   fname, fnameSeries, NOUTS_C, NOUTSPoint_C, cDim,
                                   names, namesPoint, tSimu,
                                   &outArraysMutex = outArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outArraysLock(outArraysMutex);
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
                    };
                    // if (outFuture.at(1).valid())
                    //     outFuture.at(1).wait();
                    // outFuture.at(1) = std::async(std::launch::async, outRun);
                    fOuts.push_back(outRun);
                }
            }

            if (config.dataIOControl.outPltVTKHDFFormat)
            {
                MPI_Comm commDup = MPI_COMM_NULL;
                MPI_Comm_dup(mpi.comm, &commDup);
                auto outRun = [mesh = mesh, reader = reader, outDist = outDist, &outDistPointPair = outDistPointPair,
                               fname, fnameSeries, NOUTS_C, NOUTSPoint_C, cDim,
                               names, namesPoint, tSimu,
                               &outArraysMutex = outArraysMutex, commDup]()
                {
                    // std::lock_guard<std::mutex> outHdfLock(HDF_mutex);
                    // std::lock_guard<std::mutex> outArraysLock(outArraysMutex);
                    std::scoped_lock lock(outArraysMutex, HDF_mutex);
                    MPI_Comm commDup1 = commDup;
                    mesh->PrintParallelVTKHDFDataArray(
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
                        tSimu, commDup);
                    MPI_Comm_free(&commDup1);
                };

                // outRun();
                // if (outFuture.at(2).valid())
                //     outFuture.at(2).wait();
                // MPI::Barrier(mpi.comm);
                // outFuture.at(2) = std::async(std::launch::async, outRun);
                fOuts.push_back(outRun);
            }
        }

        if (config.dataIOControl.outBndData)
        {
            {
                std::lock_guard<std::mutex> outBndLock(outBndArraysMutex);
                DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd enter");
                for (index iB = 0; iB < meshBnd->NumCell(); iB++)
                {
                    // TU recu =
                    //     vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 1>{0}, 1) *
                    //     uRec[iCell];
                    // recu += uOut[iCell];
                    // recu = EulerEvaluator::CompressRecPart(uOut[iCell], recu);
                    index iBnd = meshBnd->cell2parentCell.at(iB);
                    index iCell = mesh->bnd2cell[iBnd][0];
                    index iFace = mesh->bnd2face.at(iBnd);
                    if (iFace == -1)
                    {
                        DNDS_assert(mesh->isPeriodic);                              // only internal bnd is valid, periodic bnd should be omitted
                        (*outDistBnd)[iB](nOUTSBnd - 4) = meshBnd->GetCellZone(iB); // add this to enable blanking
                        continue;
                    }
                    TU recu = uOut[iCell];
                    if (eval.settings.frameConstRotation.enabled)
                        eval.TransformURotatingFrame_ABS_VELO(recu, vfv->GetCellQuadraturePPhys(iCell, -1), -1);
                    TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                    real vsqr = velo.squaredNorm();
                    real asqr, p, H;
                    Gas::IdealGasThermal(recu(I4), recu(0), vsqr, eval.settings.idealGasProperty.gamma, p, asqr, H);
                    // DNDS_assert(asqr > 0);
                    real M = std::sqrt(std::abs(vsqr / asqr));
                    real T = p / recu(0) / eval.settings.idealGasProperty.Rgas;

                    (*outDistBnd)[iB][0] = recu(0);
                    for (int i = 0; i < dim; i++)
                        (*outDistBnd)[iB][i + 1] = velo(i);
                    (*outDistBnd)[iB][I4 + 0] = p;
                    (*outDistBnd)[iB][I4 + 1] = T;
                    (*outDistBnd)[iB][I4 + 2] = M;
                    for (int i = I4 + 1; i < nVars; i++)
                    {
                        (*outDistBnd)[iB][2 + i] = recu(i) / recu(0); // 4 is additional amount offset, not Index of last flow variable (I4)
                    }
                    // if(iFace < 0)
                    // {
                    //     std::cout << iFace << std::endl;
                    //     std::abort();
                    // }

                    (*outDistBnd)[iB](Eigen::seq(nVars + 2, nVars + 2 + nVars - 1)) = eval.fluxBnd.at(iBnd);
                    Geom::tPoint fluxT;
                    fluxT.setZero();
                    fluxT(Seq012) = eval.fluxBndForceT.at(iBnd);
                    (*outDistBnd)[iB](Eigen::seq(nVars + 2 + nVars, nVars + 2 + nVars + 3 - 1)) = fluxT;
                    // (*outDistBnd)[iB](nOUTSBnd - 4) = mesh->GetFaceZone(iFace);
                    (*outDistBnd)[iB](nOUTSBnd - 4) = meshBnd->GetCellZone(iB);
                    (*outDistBnd)[iB](Eigen::seq(nOUTSBnd - 3, nOUTSBnd - 1)) = vfv->GetFaceNorm(iFace, 0) * vfv->GetFaceArea(iFace);

                    // (*outDist)[iCell][8] = (*vfv->SOR_iCell2iScan)[iCell];//!using SOR rb seq instead
                }

                if (config.dataIOControl.outPltMode == 0)
                {
                    outDist2SerialTransBnd.startPersistentPull();
                    outDist2SerialTransBnd.waitPersistentPull();
                }
            }
            int NOUTS_C{0}, NOUTSPoint_C{0};
            NOUTS_C = nOUTSBnd;
            DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd transfer done");

            std::vector<std::string> names;
            std::vector<std::string> namesScalar;
            std::vector<std::string> namesVector;
            std::vector<int> offsetsScalar;
            std::vector<int> offsetsVector;
            if constexpr (dim == 2)
                names = {
                    "R", "U", "V", "P", "T", "M"};
            else
                names = {
                    "R", "U", "V", "W", "P", "T", "M"};
            namesScalar = {"R", "P", "T", "M"};
            offsetsScalar = {0, dim + 1, dim + 2, dim + 3};
            namesVector = {"Velo"};
            offsetsVector = {1};
            int currentTop = dim + 4;
            for (int i = I4 + 1; i < nVars; i++)
            {
                names.push_back("V" + std::to_string(i - I4));
                namesScalar.push_back("V" + std::to_string(i - I4));
                offsetsScalar.push_back(currentTop++);
            }
            for (int i = 0; i < nVars; i++)
            {
                names.push_back("F" + std::to_string(i));
                namesScalar.push_back("F" + std::to_string(i));
                offsetsScalar.push_back(currentTop++);
            }
            names.push_back("FT1");
            names.push_back("FT2");
            names.push_back("FT3");
            namesVector.push_back("FT");
            offsetsVector.push_back(currentTop), currentTop += 3;
            names.push_back("FaceZone");
            namesScalar.push_back("FaceZone");
            offsetsScalar.push_back(currentTop++);
            names.push_back("N0");
            names.push_back("N1");
            names.push_back("N2");
            namesVector.push_back("Norm");
            offsetsVector.push_back(currentTop), currentTop += 3;

            if (config.dataIOControl.outPltTecplotFormat)
            {
                DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd tecplot start");
                if (config.dataIOControl.outPltMode == 0)
                {
                    auto outBndRun = [meshBnd = meshBnd, readerBnd = readerBnd, outDistBnd = outDistBnd, outSerialBnd = outSerialBnd,
                                      fname, fnameSeries, NOUTS_C, nOUTSBnd = nOUTSBnd, cDim, names, tSimu,
                                      &outBndArraysMutex = outBndArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outBndArraysLock(outBndArraysMutex);
                        readerBnd->PrintSerialPartPltBinaryDataArray(
                            fname + "_bnd",
                            NOUTS_C, 0,
                            [&](int idata)
                            { return names.at(idata); }, // cellNames
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
                    };
                    // if (outBndFuture.at(0).valid())
                    //     outBndFuture.at(0).wait();
                    // outBndFuture.at(0) = std::async(std::launch::async, outBndRun);
                    outBndRun();
                }
                else if (config.dataIOControl.outPltMode == 1)
                {
                    auto outBndRun = [meshBnd = meshBnd, readerBnd = readerBnd, outDistBnd = outDistBnd, outSerialBnd = outSerialBnd,
                                      fname, fnameSeries, NOUTS_C, nOUTSBnd = nOUTSBnd, cDim, names, tSimu,
                                      &outBndArraysMutex = outBndArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outBndArraysLock(outBndArraysMutex);
                        readerBnd->PrintSerialPartPltBinaryDataArray(
                            fname + "_bnd",
                            NOUTS_C, 0,
                            [&](int idata)
                            { return names.at(idata); }, // cellNames
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
                    };
                    // if (outBndFuture.at(0).valid())
                    //     outBndFuture.at(0).wait();
                    // outBndFuture.at(0) = std::async(std::launch::async, outBndRun);
                    outBndRun();
                }
            }

            const int cDim = dim;
            if (config.dataIOControl.outPltVTKFormat)
            {
                DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd vtk start");
                if (config.dataIOControl.outPltMode == 0)
                {
                    auto outBndRun = [meshBnd = meshBnd, readerBnd = readerBnd, outDistBnd = outDistBnd, outSerialBnd = outSerialBnd,
                                      fname, fnameSeries, NOUTS_C, nOUTSBnd = nOUTSBnd, nVars = nVars, cDim,
                                      namesScalar, namesVector, offsetsScalar, offsetsVector, tSimu,
                                      &outBndArraysMutex = outBndArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outBndArraysLock(outBndArraysMutex);
                        readerBnd->PrintSerialPartVTKDataArray(
                            fname + "_bnd",
                            fnameSeries.size() ? fnameSeries + "_bnd" : "",
                            namesScalar.size(), namesVector.size(),
                            0, 0, //! vectors number is not cDim but 3
                            [&](int idata)
                            {
                                return namesScalar.at(idata); // cellNames
                            },
                            [&](int idata, index iv)
                            {
                                return (*outSerialBnd)[iv][offsetsScalar.at(idata)]; // cellData
                            },
                            [&](int idata)
                            {
                                return namesVector.at(idata);
                            },
                            [&](int idata, index iv, int idim)
                            {
                                return (*outSerialBnd)[iv][offsetsVector.at(idata) + idim];
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
                    };
                    // if (outBndFuture.at(1).valid())
                    //     outBndFuture.at(1).wait();
                    // outBndFuture.at(1) = std::async(std::launch::async, outBndRun);
                    fOuts.push_back(outBndRun);
                }
                else if (config.dataIOControl.outPltMode == 1)
                {
                    auto outBndRun = [meshBnd = meshBnd, readerBnd = readerBnd, outDistBnd = outDistBnd, outSerialBnd = outSerialBnd,
                                      fname, fnameSeries, NOUTS_C, nOUTSBnd = nOUTSBnd, cDim,
                                      namesScalar, namesVector, offsetsScalar, offsetsVector, tSimu,
                                      &outBndArraysMutex = outBndArraysMutex]()
                    {
                        std::lock_guard<std::mutex> outBndArraysLock(outBndArraysMutex);
                        readerBnd->PrintSerialPartVTKDataArray(
                            fname + "_bnd",
                            fnameSeries.size() ? fnameSeries + "_bnd" : "",
                            namesScalar.size(), namesVector.size(),
                            0, 0, //! vectors number is not cDim but 2
                            [&](int idata)
                            {
                                return namesScalar.at(idata); // cellNames
                            },
                            [&](int idata, index iv)
                            {
                                return (*outDistBnd)[iv][offsetsScalar.at(idata)]; // cellData
                            },
                            [&](int idata)
                            {
                                return namesVector.at(idata);
                            },
                            [&](int idata, index iv, int idim)
                            {
                                return (*outDistBnd)[iv][offsetsVector.at(idata) + idim];
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
                    };
                    // if (outBndFuture.at(1).valid())
                    //     outBndFuture.at(1).wait();
                    // outBndFuture.at(1) = std::async(std::launch::async, outBndRun);
                    fOuts.push_back(outBndRun);
                }
            }

            if (config.dataIOControl.outPltVTKHDFFormat)
            {
                MPI_Comm commDup = MPI_COMM_NULL;
                MPI_Comm_dup(mpi.comm, &commDup);
                auto outBndRun = [meshBnd = meshBnd, outDistBnd = outDistBnd,
                                  fname, fnameSeries, NOUTS_C, nOUTSBnd = nOUTSBnd, cDim,
                                  namesScalar, namesVector, offsetsScalar, offsetsVector, tSimu,
                                  &outBndArraysMutex = outBndArraysMutex, commDup]()
                {
                    // std::lock_guard<std::mutex> outHdfLock(HDF_mutex);
                    // std::lock_guard<std::mutex> outBndArraysLock(outBndArraysMutex);
                    // std::lock_guard<std::mutex> outBndArraysLock1(outArraysMutex);
                    std::scoped_lock lock(outBndArraysMutex, HDF_mutex);
                    MPI_Comm commDup1 = commDup;
                    meshBnd->PrintParallelVTKHDFDataArray(
                        fname + "_bnd",
                        fnameSeries.size() ? fnameSeries + "_bnd" : "",
                        namesScalar.size(), namesVector.size(),
                        0, 0, //! vectors number is not cDim but 2
                        [&](int idata)
                        {
                            return namesScalar.at(idata); // cellNames
                        },
                        [&](int idata, index iv)
                        {
                            return (*outDistBnd)[iv][offsetsScalar.at(idata)]; // cellData
                        },
                        [&](int idata)
                        {
                            return namesVector.at(idata);
                        },
                        [&](int idata, index iv, int idim)
                        {
                            return (*outDistBnd)[iv][offsetsVector.at(idata) + idim];
                        },
                        [](int idata)
                        {
                            return "error"; // pointNames
                        },
                        [](int idata, index iv)
                        {
                            return std::nan("0"); // pointData
                        },
                        [](int idata)
                        {
                            return "error"; // pointNames
                        },
                        [](int idata, index iv, int idim)
                        {
                            return std::nan("0"); // pointData
                        },
                        tSimu, commDup);
                    MPI_Comm_free(&commDup1);
                };
                // if (outBndFuture.at(2).valid())
                //     outBndFuture.at(2).wait();
                // MPI::Barrier(mpi.comm);
                // outBndFuture.at(2) = std::async(std::launch::async, outBndRun);
                fOuts.push_back(outBndRun);
                // outBndRun();
            }
        }
        auto runFOuts = [fOuts]()
        {
            for (auto &f : fOuts)
                f();
        };
        bool useAsyncOut = config.dataIOControl.allowAsyncPrintData;
#ifndef H5_HAVE_THREADSAFE
        if (config.dataIOControl.outPltVTKHDFFormat)
            useAsyncOut = false;
#endif
        if (config.dataIOControl.outPltVTKHDFFormat)
            if (MPI::GetMPIThreadLevel() < MPI_THREAD_MULTIPLE)
                useAsyncOut = false;

        // std::cout << fOuts.size() << std::endl;
        if (outSeqFuture.valid())
            outSeqFuture.wait();
        if (useAsyncOut)
            outSeqFuture = std::async(std::launch::async, runFOuts);
        else
            runFOuts();

        DNDS_MPI_InsertCheck(mpi, "EulerSolver<model>::PrintData === bnd output done");
    }
}