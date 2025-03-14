#pragma once

#include "EulerSolver.hpp"
#include "SpecialFields.hpp"

namespace DNDS::Euler
{
    static const auto model = NS_SA;
    DNDS_SWITCH_INTELLISENSE(template <EulerModel model>, )
    void EulerSolver<model>::ReadMeshAndInitialize()
    {
        DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize 1 nvars " + std::to_string(nVars));
        output_stamp = getTimeStamp(mpi);
        if (!config.dataIOControl.uniqueStamps)
            output_stamp = "";
        if (mpi.rank == 0)
            log() << "=== Got Time Stamp: [" << output_stamp << "] ===" << std::endl;
        // Debug::MPIDebugHold(mpi);

        int gDimLocal = gDim; //! or else the linker breaks down here (with clang++ or g++, -g -O0,2; c++ non-optimizer bug?)

        auto &BCHandler = *pBCHandler;

        DNDS_MAKE_SSP(mesh, mpi, gDimLocal);
        DNDS_MAKE_SSP(meshBnd, mpi, gDimLocal - 1);

        DNDS_MAKE_SSP(reader, mesh, 0);
        DNDS_MAKE_SSP(readerBnd, meshBnd, 0);
        DNDS_assert(config.dataIOControl.readMeshMode == 0 || config.dataIOControl.readMeshMode == 1);
        DNDS_assert(config.dataIOControl.outPltMode == 0 || config.dataIOControl.outPltMode == 1);
        mesh->periodicInfo.translation[1] = config.boundaryDefinition.PeriodicTranslation1;
        mesh->periodicInfo.translation[2] = config.boundaryDefinition.PeriodicTranslation2;
        mesh->periodicInfo.translation[3] = config.boundaryDefinition.PeriodicTranslation3;
        mesh->periodicInfo.rotationCenter[1] = config.boundaryDefinition.PeriodicRotationCent1;
        mesh->periodicInfo.rotationCenter[2] = config.boundaryDefinition.PeriodicRotationCent2;
        mesh->periodicInfo.rotationCenter[3] = config.boundaryDefinition.PeriodicRotationCent3;
        mesh->periodicInfo.rotation[1] =
            Geom::RotZ(config.boundaryDefinition.PeriodicRotationEulerAngles1[2]) *
            Geom::RotY(config.boundaryDefinition.PeriodicRotationEulerAngles1[1]) *
            Geom::RotX(config.boundaryDefinition.PeriodicRotationEulerAngles1[0]);
        mesh->periodicInfo.rotation[2] =
            Geom::RotZ(config.boundaryDefinition.PeriodicRotationEulerAngles2[2]) *
            Geom::RotY(config.boundaryDefinition.PeriodicRotationEulerAngles2[1]) *
            Geom::RotX(config.boundaryDefinition.PeriodicRotationEulerAngles2[0]);
        mesh->periodicInfo.rotation[3] =
            Geom::RotZ(config.boundaryDefinition.PeriodicRotationEulerAngles3[2]) *
            Geom::RotY(config.boundaryDefinition.PeriodicRotationEulerAngles3[1]) *
            Geom::RotX(config.boundaryDefinition.PeriodicRotationEulerAngles3[0]);

        if (config.dataIOControl.readMeshMode == 0)
        {
            if (config.dataIOControl.meshFormat == 1)
                reader->ReadFromOpenFOAMAndConvertSerial(
                    config.dataIOControl.meshFile,
                    config.bcNameMapping,
                    [&](const std::string &name)
                        -> Geom::t_index
                    { return BCHandler.GetIDFromName(name); });
            else
                reader->ReadFromCGNSSerial(
                    config.dataIOControl.meshFile,
                    [&](const std::string &name) -> Geom::t_index
                    { return BCHandler.GetIDFromName(name); });
            reader->Deduplicate1to1Periodic(config.boundaryDefinition.periodicTolerance);
            reader->BuildCell2Cell();
            reader->MeshPartitionCell2Cell(config.dataIOControl.meshPartitionOptions);
            reader->PartitionReorderToMeshCell2Cell();

            mesh->BuildGhostPrimary();
            mesh->AdjGlobal2LocalPrimary();
            if (config.dataIOControl.meshElevation == 1)
            {
                DNDS::ssp<DNDS::Geom::UnstructuredMesh> meshO2;
                DNDS_MAKE_SSP(meshO2, mpi, gDimLocal);
                meshO2->BuildO2FromO1Elevation(*mesh);
                std::swap(meshO2, mesh);

                reader->mesh = mesh;
                mesh->BuildGhostPrimary();
                mesh->AdjGlobal2LocalPrimary();
            }
            DNDS_assert(config.dataIOControl.meshDirectBisect <= 4);
            for (int iter = 1; iter <= config.dataIOControl.meshDirectBisect; iter++)
            {
                DNDS::ssp<DNDS::Geom::UnstructuredMesh> meshO2;
                DNDS_MAKE_SSP(meshO2, mpi, gDimLocal);
                meshO2->BuildO2FromO1Elevation(*mesh);
                meshO2->BuildGhostPrimary();
                DNDS::ssp<DNDS::Geom::UnstructuredMesh> meshO1B;
                DNDS_MAKE_SSP(meshO1B, mpi, gDimLocal);
                meshO1B->BuildBisectO1FormO2(*meshO2);

                std::swap(meshO1B, mesh);
                reader->mesh = mesh;
                mesh->RecoverNode2CellAndNode2Bnd();
                mesh->RecoverCell2CellAndBnd2Cell();
                mesh->BuildGhostPrimary();
                mesh->AdjGlobal2LocalPrimary();
                mesh->AdjGlobal2LocalN2CB();
                index nCell = mesh->NumCellGlobal();
                index nNode = mesh->NumNodeGlobal();
                if (mesh->getMPI().rank == 0)
                {
                    log() << fmt::format("Mesh Direct Bisect {} done, nCell [{}], nNode [{}]", iter, nCell, nNode) << std::endl;
                }
            }
        }
        else
        {
            using namespace std::literals;
            std::filesystem::path meshPath{config.dataIOControl.meshFile};
            std::string meshOutName = std::string(config.dataIOControl.meshFile) + "_part_" + std::to_string(mpi.size) +
                                      (config.dataIOControl.meshElevation == 1 ? "_elevated"s : ""s) +
                                      (config.dataIOControl.meshDirectBisect > 0 ? "_bisect" + std::to_string(config.dataIOControl.meshDirectBisect) : ""s);
            std::string meshPartPath;
            if (config.dataIOControl.meshPartitionedReaderType == "JSON")
            {
                std::filesystem::path meshOutDir{meshOutName + ".dir"};
                // std::filesystem::create_directories(meshOutDir); // reading not writing
                meshPartPath = getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));
            }
            else if (config.dataIOControl.meshPartitionedReaderType == "H5")
            {
                meshPartPath = meshOutName + ".dnds.h5";
            }
            else
                DNDS_assert_info(false, "serializer is invalid");

            Serializer::SerializerBaseSSP serializerP = Serializer::SerializerFactory(config.dataIOControl.meshPartitionedReaderType).BuildSerializer(mpi);

            if (mpi.rank == 0)
                log() << "EulerSolver === to read via [" << config.dataIOControl.meshPartitionedReaderType << "]" << std::endl;

            serializerP->OpenFile(meshPartPath, true);
            mesh->ReadSerialize(serializerP, "meshPart");
            serializerP->CloseFile();
        }
        if (config.dataIOControl.meshReorderCells == 1)
            mesh->ReorderLocalCells(); // do this early so that faces are natural to cell
        // std::cout << "here" << std::endl;
        mesh->InterpolateFace();
        mesh->AssertOnFaces();

        // todo: make this interpolation optional?
        mesh->AdjLocal2GlobalPrimary();
        mesh->RecoverNode2CellAndNode2Bnd(); // todo: don't do this if already done
        mesh->AdjGlobal2LocalPrimary();
        mesh->BuildGhostN2CB();
        mesh->AdjGlobal2LocalN2CB();

        // mesh->AdjLocal2GlobalN2CB();
        // mesh->AdjGlobal2LocalN2CB();
        for (index iNode = 0; iNode < mesh->NumNode(); iNode++)
            for (index iCell : mesh->node2cell[iNode])
                DNDS_assert(iCell >= 0);

        if (config.dataIOControl.meshElevation == 1 && config.dataIOControl.readMeshMode == 0)
        {
            mesh->elevationInfo.nIter = config.dataIOControl.meshElevationIter;
            mesh->elevationInfo.nSearch = config.dataIOControl.meshElevationNSearch;
            mesh->elevationInfo.RBFRadius = config.dataIOControl.meshElevationRBFRadius;
            mesh->elevationInfo.RBFPower = config.dataIOControl.meshElevationRBFPower;
            mesh->elevationInfo.kernel = config.dataIOControl.meshElevationRBFKernel;
            mesh->elevationInfo.MaxIncludedAngle = config.dataIOControl.meshElevationMaxIncludedAngle;
            mesh->elevationInfo.refDWall = config.dataIOControl.meshElevationRefDWall;
            mesh->ElevatedNodesGetBoundarySmooth(
                [&](Geom::t_index bndId)
                {
                    auto bType = pBCHandler->GetTypeFromID(bndId);
                    if (bType == BCWall)
                        return true;
                    if (config.dataIOControl.meshElevationBoundaryMode == 1 &&
                        (bType == BCWallInvis || bType == BCSym))
                        return true;
                    return false;
                });
            if (config.dataIOControl.meshElevationInternalSmoother == 0)
                mesh->ElevatedNodesSolveInternalSmooth();
            else if (config.dataIOControl.meshElevationInternalSmoother == 1)
                mesh->ElevatedNodesSolveInternalSmoothV1();
            else if (config.dataIOControl.meshElevationInternalSmoother == 2)
                mesh->ElevatedNodesSolveInternalSmoothV2();
            else if (config.dataIOControl.meshElevationInternalSmoother == -1)
            {
                if (mpi.rank == 0)
                    log() << " WARNING !!! Not Smoothing internal, abandoning boundary smooth displacements" << std::endl;
            }
            else
                DNDS_assert(false);
        }

        if (config.dataIOControl.outPltMode == 0)
        {
            mesh->AdjLocal2GlobalPrimary();
            reader->BuildSerialOut();
            mesh->AdjGlobal2LocalPrimary();
        }

        if (config.timeMarchControl.partitionMeshOnly)
        {
            using namespace std::literals;
            std::string meshPartPath;
            std::string meshOutName = std::string(config.dataIOControl.meshFile) + "_part_" + std::to_string(mpi.size) +
                                      (config.dataIOControl.meshElevation == 1 ? "_elevated"s : ""s) +
                                      (config.dataIOControl.meshDirectBisect > 0 ? "_bisect" + std::to_string(config.dataIOControl.meshDirectBisect) : ""s);

            if (config.dataIOControl.meshPartitionedWriter.type == "JSON")
            {
                meshOutName += ".dir";
                std::filesystem::path meshOutDir{meshOutName};
                std::filesystem::create_directories(meshOutDir);
                meshPartPath = DNDS::getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));
            }
            else if (config.dataIOControl.meshPartitionedWriter.type == "H5")
            {
                meshOutName += ".dnds.h5";
                meshPartPath = meshOutName;
                std::filesystem::path outPath = meshPartPath;
                std::filesystem::create_directories(outPath.parent_path() / ".");
            }
            else
                DNDS_assert_info(false, "serializer is invalid");

            Serializer::SerializerBaseSSP serializerP = config.dataIOControl.meshPartitionedWriter.BuildSerializer(mpi);

            serializerP->OpenFile(meshPartPath, false);
            mesh->WriteSerialize(serializerP, "meshPart");
            serializerP->CloseFile();
            return; //** mesh preprocess only (without transformation)
        }

#ifdef DNDS_USE_OMP
        omp_set_num_threads( // note that the meaning is like "omp_set_max_threads()"
            DNDS::MPIWorldSize() == 1
                ? std::min(omp_get_num_procs(), omp_get_max_threads())
                : (get_env_DNDS_DIST_OMP_NUM_THREADS() == 0 ? 1 : DNDS::get_env_DNDS_DIST_OMP_NUM_THREADS()));
#endif
        mesh->ConstructBndMesh(*meshBnd);
        if (config.dataIOControl.outPltMode == 0)
        {
            meshBnd->AdjLocal2GlobalPrimaryForBnd();
            readerBnd->BuildSerialOut();
            meshBnd->AdjGlobal2LocalPrimaryForBnd();
        }

        if (config.dataIOControl.meshRotZ != 0.0)
        {
            real rz = config.dataIOControl.meshRotZ / 180.0 * pi;
            Eigen::Matrix3d Rz{
                {std::cos(rz), -std::sin(rz), 0},
                {std::sin(rz), std::cos(rz), 0},
                {0, 0, 1},
            };
            mesh->TransformCoords(
                [&](const Geom::tPoint &p)
                { return Rz * p; });
            meshBnd->TransformCoords(
                [&](const Geom::tPoint &p)
                { return Rz * p; });

            for (auto &r : mesh->periodicInfo.rotation)
                r = Rz * r * Rz.transpose();
            for (auto &p : mesh->periodicInfo.rotationCenter)
                p = Rz * p;
            for (auto &p : mesh->periodicInfo.translation)
                p = Rz * p;
            // @todo  //! todo: alter the rotation and translation in  periodicInfo mesh->periodicInfo
        }
        if (config.dataIOControl.meshScale != 1.0)
        {
            auto scale = config.dataIOControl.meshScale;
            mesh->TransformCoords(
                [&](const Geom::tPoint &p)
                { return p * scale; });
            meshBnd->TransformCoords(
                [&](const Geom::tPoint &p)
                { return p * scale; });

            for (auto &i : mesh->periodicInfo.translation)
                i *= scale;
            for (auto &i : mesh->periodicInfo.rotationCenter)
                i *= scale;
        }
        if (config.dataIOControl.rectifyNearPlane)
        {
            auto fTrans = [&](const Geom::tPoint &p)
            {
                Geom::tPoint ret = p;
                if (config.dataIOControl.rectifyNearPlane & 1)
                    if (std::abs(ret(0)) < config.dataIOControl.rectifyNearPlaneThres)
                        ret(0) = 0;
                if (config.dataIOControl.rectifyNearPlane & 2)
                    if (std::abs(ret(1)) < config.dataIOControl.rectifyNearPlaneThres)
                        ret(1) = 0;
                if (config.dataIOControl.rectifyNearPlane & 4)
                    if (std::abs(ret(2)) < config.dataIOControl.rectifyNearPlaneThres)
                        ret(2) = 0;
                return ret;
            };
            mesh->TransformCoords(fTrans);
            meshBnd->TransformCoords(fTrans);
        }
        { //* symBnd's rectifying: !  altering mesh
            for (index iB = 0; iB < mesh->NumBnd(); iB++)
            {
                index iFace = mesh->bnd2face.at(iB);
                auto bndID = mesh->bndElemInfo(iB, 0).zone;
                EulerBCType bndType = pBCHandler->GetTypeFromID(bndID);
                if (bndType == BCSym)
                {
                    auto rectifyOpt = pBCHandler->GetFlagFromID(bndID, "rectifyOpt");
                    if (rectifyOpt >= 1 && rectifyOpt <= 3)
                        for (auto iNode : mesh->bnd2node[iB])
                            mesh->coords[iNode](rectifyOpt - 1) = 0.0;
                }
            }
            mesh->coords.trans.pullOnce();
            for (index iB = 0; iB < meshBnd->NumCell(); iB++)
            {
                auto bndID = meshBnd->cellElemInfo(iB, 0).zone;
                EulerBCType bndType = pBCHandler->GetTypeFromID(bndID);
                if (bndType == BCSym)
                {
                    auto rectifyOpt = pBCHandler->GetFlagFromID(bndID, "rectifyOpt");
                    if (rectifyOpt >= 1 && rectifyOpt <= 3)
                        for (auto iNode : meshBnd->cell2node[iB])
                            meshBnd->coords[iNode](rectifyOpt - 1) = 0.0;
                }
            }
        }
        /// @todo //todo: upgrade to optional
        if (config.dataIOControl.outPltMode == 0)
            reader->coordSerialOutTrans.pullOnce(),
                readerBnd->coordSerialOutTrans.pullOnce();

        mesh->RecreatePeriodicNodes();
        mesh->BuildVTKConnectivity();
        meshBnd->RecreatePeriodicNodes();
        meshBnd->BuildVTKConnectivity();

        // *demo code
        // mesh->PrintMeshCGNS("test.cgns", [&](Geom::t_index i)
        //                     { return BCHandler.GetNameFormID(i); }, BCHandler.GetAllNames());

        DNDS_MAKE_SSP(vfv, mpi, mesh);
        vfv->SetPeriodicTransformations(
            [&](auto u, Geom::t_index id)
            {
                DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                u(Eigen::all, Seq123) = mesh->periodicInfo.TransVector<dim, Eigen::Dynamic>(u(Eigen::all, Seq123).transpose(), id).transpose();
            },
            [&](auto u, Geom::t_index id)
            {
                DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                u(Eigen::all, Seq123) = mesh->periodicInfo.TransVectorBack<dim, Eigen::Dynamic>(u(Eigen::all, Seq123).transpose(), id).transpose();
            });
        vfv->settings.ParseFromJson(config.vfvSettings);
        vfv->ConstructMetrics();
        vfv->ConstructBaseAndWeight(
            [&](Geom::t_index id, int iOrder) -> real
            {
                auto type = BCHandler.GetTypeFromID(id);
                if (type == BCSpecial || type == BCOut)
                    return 0;
                if (type == BCFar) // use Dirichlet type
                    return iOrder ? 0. : 1.;
                if (type == BCWallInvis || type == BCSym)
                    return iOrder ? 0. : 1.;
                if (Geom::FaceIDIsPeriodic(id))
                    return iOrder ? 1. : 1.; //! treat as real internal
                // others: use Dirichlet type
                return iOrder ? 0. : 1.;
            });
        vfv->ConstructRecCoeff();

        vfv->BuildUDof(u, nVars);
        vfv->BuildUDof(uIncBufODE, nVars);
        if (config.timeAverageControl.enabled)
        {
            vfv->BuildUDof(wAveraged, nVars);
            vfv->BuildUDof(uAveraged, nVars);
        }
        uPool.resizeInit(3,
                         [&](ArrayDOFV<nVarsFixed> &uu)
                         { vfv->BuildUDof(uu, nVars); });

        vfv->BuildURec(uRec, nVars);
        if (config.timeMarchControl.odeCode == 401)
            vfv->BuildURec(uRec1, nVars);
        vfv->BuildURec(uRecLimited, nVars);
        vfv->BuildURec(uRecNew, nVars);
        vfv->BuildURec(uRecNew1, nVars);
        vfv->BuildURec(uRecB, nVars);
        vfv->BuildURec(uRecB1, nVars);
        vfv->BuildURec(uRecOld, nVars);
        vfv->BuildScalar(ifUseLimiter);
        vfv->BuildUDof(betaPP, 1);
        vfv->BuildUDof(alphaPP, 1);
        vfv->BuildUDof(alphaPP_tmp, 1);
        vfv->BuildUDof(dTauTmp, 1);
        betaPP.setConstant(1.0);
        alphaPP.setConstant(1.0);
        if (config.timeMarchControl.odeCode == 401)
        {
            vfv->BuildUDof(betaPP1, 1);
            vfv->BuildUDof(alphaPP1, 1);
            betaPP1.setConstant(1.0);
            alphaPP1.setConstant(1.0);
        }
        if (config.implicitReconstructionControl.storeRecInc)
        {
            vfv->BuildURec(uRecInc, nVars);
            if (config.timeMarchControl.odeCode == 401)
                vfv->BuildURec(uRecInc1, nVars);
        }

        DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize 2 nvars " + std::to_string(nVars));
        /*******************************/
        // initialize pEval
        DNDS_MAKE_SSP(pEval, mesh, vfv, pBCHandler, config.eulerSettings);
        EulerEvaluator<model> &eval = *pEval;

        JD.SetModeAndInit(eval.settings.useScalarJacobian ? 0 : 1, nVars, u);
        JSource.SetModeAndInit(eval.settings.useScalarJacobian ? 0 : 1, nVars, u);
        if (config.timeMarchControl.odeCode == 401)
        {
            JD1.SetModeAndInit(eval.settings.useScalarJacobian ? 0 : 1, nVars, u);
            JSource1.SetModeAndInit(eval.settings.useScalarJacobian ? 0 : 1, nVars, u);
        }
        JDTmp.SetModeAndInit(eval.settings.useScalarJacobian ? 0 : 1, nVars, u);
        JSourceTmp.SetModeAndInit(eval.settings.useScalarJacobian ? 0 : 1, nVars, u);
        /*******************************/
        // ** initialize output Array

        DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize 3 nvars " + std::to_string(nVars));

        // update output number
        DNDS_assert(config.dataIOControl.outCellScalarNames.size() < 128);
        nOUTS += config.dataIOControl.outCellScalarNames.size();

        DNDS_assert(config.dataIOControl.outAtCellData || config.dataIOControl.outAtPointData);
        DNDS_assert(config.dataIOControl.outPltVTKFormat || config.dataIOControl.outPltTecplotFormat || config.dataIOControl.outPltVTKHDFFormat);
        DNDS_MAKE_SSP(outDistBnd, mpi);
        outDistBnd->Resize(meshBnd->NumCell(), nOUTSBnd);

        if (config.dataIOControl.outAtCellData)
        {
            DNDS_MAKE_SSP(outDist, mpi);
            outDist->Resize(mesh->NumCell(), nOUTS);
        }
        if (config.dataIOControl.outAtPointData)
        {
            DNDS_MAKE_SSP(outDistPoint, mpi);
            outDistPoint->Resize(mesh->NumNode(), nOUTSPoint);
            DNDS_assert(nOUTSPoint >= nVars);

            outDistPointPair.father = outDistPoint;
            DNDS_MAKE_SSP(outDistPointPair.son, mpi);
            outDistPointPair.TransAttach();
            outDistPointPair.trans.BorrowGGIndexing(mesh->coords.trans);
            outDistPointPair.trans.createMPITypes();
            outDistPointPair.trans.initPersistentPull();
        }

        if (config.dataIOControl.outPltMode == 0)
        {
            //! serial mesh specific output method
            DNDS_MAKE_SSP(outSerialBnd, mpi);
            outDist2SerialTransBnd.setFatherSon(outDistBnd, outSerialBnd);
            DNDS_assert(readerBnd->mode == Geom::MeshReaderMode::SerialOutput);
            outDist2SerialTransBnd.BorrowGGIndexing(readerBnd->cell2nodeSerialOutTrans);
            outDist2SerialTransBnd.createMPITypes();
            outDist2SerialTransBnd.initPersistentPull();

            if (config.dataIOControl.outAtCellData)
            {
                DNDS_MAKE_SSP(outSerial, mpi);
                outDist2SerialTrans.setFatherSon(outDist, outSerial);
                DNDS_assert(reader->mode == Geom::MeshReaderMode::SerialOutput);
                outDist2SerialTrans.BorrowGGIndexing(reader->cell2nodeSerialOutTrans);
                outDist2SerialTrans.createMPITypes();
                outDist2SerialTrans.initPersistentPull();
            }
            if (config.dataIOControl.outAtPointData)
            {
                DNDS_MAKE_SSP(outSerialPoint, mpi);
                outDist2SerialTransPoint.setFatherSon(outDistPoint, outSerialPoint);
                DNDS_assert(reader->mode == Geom::MeshReaderMode::SerialOutput);
                outDist2SerialTransPoint.BorrowGGIndexing(reader->coordSerialOutTrans);
                outDist2SerialTransPoint.createMPITypes();
                outDist2SerialTransPoint.initPersistentPull();
            }
        }
        DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize -1 nvars " + std::to_string(nVars));
    }

    DNDS_SWITCH_INTELLISENSE(template <EulerModel model>, )
    bool EulerSolver<model>::functor_fstop(int iter, ArrayDOFV<nVarsFixed> &cres, int iStep, RunningEnvironment &runningEnvironment)
    {
        using namespace std::literals;
        DNDS_EULERSOLVER_RUNNINGENV_GET_REF_LIST
        // auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;

        // auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;

        Eigen::VectorFMTSafe<real, -1> res(nVars);
        eval.EvaluateNorm(res, cres, 1, config.convergenceControl.useVolWiseResidual);
        // if (iter == 1 && iStep == 1) // * using 1st rk step for reference
        if (iter == 1)
            resBaseCInternal = res;
        else
            resBaseCInternal = resBaseCInternal.array().max(res.array()); //! using max !
        Eigen::VectorFMTSafe<real, -1> resRel = (res.array() / (resBaseCInternal.array() + verySmallReal)).matrix();
        bool ifStop = resRel(0) < config.convergenceControl.rhsThresholdInternal; // ! using only rho's residual
        if (iter < config.convergenceControl.nTimeStepInternalMin)
            ifStop = false;
        auto [CLCur, CDCur, AOACur] = eval.CLDriverGetIntegrationUpdate(iter);
        if (iter % config.outputControl.nConsoleCheckInternal == 0 || iter > config.convergenceControl.nTimeStepInternal || ifStop)
        {
            double tWall = MPI_Wtime();
            real telapsed = MPI_Wtime() - tstartInternal;
            bool useCollectiveTimer = config.outputControl.useCollectiveTimer;
            real tcomm = Timer().getTimerColOrLoc(PerformanceTimer::Comm, mpi, useCollectiveTimer);
            real tLimiterA = Timer().getTimerColOrLoc(PerformanceTimer::LimiterA, mpi, useCollectiveTimer);
            real tLimiterB = Timer().getTimerColOrLoc(PerformanceTimer::LimiterB, mpi, useCollectiveTimer);
            real trhs = Timer().getTimerColOrLoc(PerformanceTimer::RHS, mpi, useCollectiveTimer);
            real trec = Timer().getTimerColOrLoc(PerformanceTimer::Reconstruction, mpi, useCollectiveTimer);
            real tLim = Timer().getTimerColOrLoc(PerformanceTimer::Limiter, mpi, useCollectiveTimer);
            real tPP = Timer().getTimerColOrLoc(PerformanceTimer::Positivity, mpi, useCollectiveTimer);
            auto [telapsedM, telapsedS] = tInternalStats["t"].update(telapsed).get();
            auto [tcommM, tcommS] = tInternalStats["c"].update(tcomm).get();
            auto [trhsM, trhsS] = tInternalStats["r"].update(trhs).get();
            auto [trecM, trecS] = tInternalStats["v"].update(trec).get();
            auto [tLimM, tLimS] = tInternalStats["l"].update(tLim).get();
            auto [tPPrM, tPPrS] = tInternalStats["p"].update(tPP).get();

            if (mpi.rank == 0)
            {
                auto fmt = log().flags();
                std::string formatStringMain = "";
                for (auto &s : config.outputControl.consoleMainOutputFormatInternal)
                    formatStringMain += s;
                log() << fmt::format(formatStringMain +
                                         "  "s +
                                         (config.outputControl.consoleOutputMode == 1
                                              ? "WallFlux {termYellow}{wallFlux:.6e}{termReset} CL,CD,AoA [{CLCur:.6e},{CDCur:.6e},{AOACur:.2e}]"s
                                              : ""s),
                                     DNDS_FMT_ARG(step),
                                     DNDS_FMT_ARG(iStep),
                                     DNDS_FMT_ARG(iter),
                                     fmt::arg("resRel", resRel.transpose()),
                                     fmt::arg("wallFlux", eval.fluxWallSum.transpose()),
                                     DNDS_FMT_ARG(tSimu),
                                     DNDS_FMT_ARG(curDtImplicit),
                                     DNDS_FMT_ARG(curDtMin),
                                     DNDS_FMT_ARG(CFLNow),
                                     DNDS_FMT_ARG(nLimInc),
                                     DNDS_FMT_ARG(alphaMinInc),
                                     DNDS_FMT_ARG(nLimBeta),
                                     DNDS_FMT_ARG(minBeta),
                                     DNDS_FMT_ARG(nLimAlpha),
                                     DNDS_FMT_ARG(minAlpha),
                                     DNDS_FMT_ARG(telapsed), DNDS_FMT_ARG(telapsedM),
                                     DNDS_FMT_ARG(trec), DNDS_FMT_ARG(trecM),
                                     DNDS_FMT_ARG(trhs), DNDS_FMT_ARG(trhsM),
                                     DNDS_FMT_ARG(tcomm), DNDS_FMT_ARG(tcommM),
                                     DNDS_FMT_ARG(tLim), DNDS_FMT_ARG(tLimM),
                                     DNDS_FMT_ARG(tPP), DNDS_FMT_ARG(tPPrM),
                                     DNDS_FMT_ARG(tLimiterA),
                                     DNDS_FMT_ARG(tLimiterB),
                                     DNDS_FMT_ARG(tWall),
                                     DNDS_FMT_ARG(CLCur),
                                     DNDS_FMT_ARG(CDCur),
                                     DNDS_FMT_ARG(AOACur),
                                     fmt::arg("termRed", TermColor::Red),
                                     fmt::arg("termBlue", TermColor::Blue),
                                     fmt::arg("termGreen", TermColor::Green),
                                     fmt::arg("termCyan", TermColor::Cyan),
                                     fmt::arg("termYellow", TermColor::Yellow),
                                     fmt::arg("termBold", TermColor::Bold),
                                     fmt::arg("termReset", TermColor::Reset));
                log() << std::endl;
                log().setf(fmt);

                // std::string delimC = " ";
                // logErr
                //     << std::left
                //     << step << delimC
                //     << std::left
                //     << iter << delimC
                //     << std::left
                //     << std::setprecision(config.outputControl.nPrecisionLog) << std::scientific
                //     << res.transpose() << delimC
                //     << tSimu << delimC
                //     << curDtMin << delimC
                //     << real(eval.nFaceReducedOrder) << delimC
                //     << eval.fluxWallSum.transpose() << delimC
                //     << (nLimInc) << delimC << (alphaMinInc) << delimC
                //     << (nLimBeta) << delimC << (minBeta) << delimC
                //     << (nLimAlpha) << delimC << (minAlpha) << delimC
                //     << std::endl;

                // std::vector<std::string> logfileOutputTitles{
                //     "step", "iStep", "iter", "tSimu",
                //     "res", "curDtImplicit", "curDtMin", "CFLNow",
                //     "nLimInc", "alphaMinInc",
                //     "nLimBeta", "minBeta",
                //     "nLimAlpha", "minAlpha",
                //     "tWall", "telapsed", "trec", "trhs", "tcomm", "tLim", "tLimiterA", "tLimiterB",
                //     "fluxWall", "CL", "CD", "AoA"};
                auto &fluxWall = eval.fluxWallSum;
                auto &logErrVal = std::get<1>(logErr);
#define DNDS_FILL_IN_LOG_ERR_VAL(v) FillLogValue(logErrVal, #v, v)
                DNDS_FILL_IN_LOG_ERR_VAL(step);
                DNDS_FILL_IN_LOG_ERR_VAL(iStep);
                DNDS_FILL_IN_LOG_ERR_VAL(iter);
                DNDS_FILL_IN_LOG_ERR_VAL(tSimu);
                DNDS_FILL_IN_LOG_ERR_VAL(res);
                DNDS_FILL_IN_LOG_ERR_VAL(curDtImplicit);
                DNDS_FILL_IN_LOG_ERR_VAL(curDtMin);
                DNDS_FILL_IN_LOG_ERR_VAL(CFLNow);

                DNDS_FILL_IN_LOG_ERR_VAL(nLimInc);
                DNDS_FILL_IN_LOG_ERR_VAL(alphaMinInc);
                DNDS_FILL_IN_LOG_ERR_VAL(nLimBeta);
                DNDS_FILL_IN_LOG_ERR_VAL(minBeta);
                DNDS_FILL_IN_LOG_ERR_VAL(nLimAlpha);
                DNDS_FILL_IN_LOG_ERR_VAL(minAlpha);

                DNDS_FILL_IN_LOG_ERR_VAL(tWall);
                DNDS_FILL_IN_LOG_ERR_VAL(telapsed);
                DNDS_FILL_IN_LOG_ERR_VAL(trec);
                DNDS_FILL_IN_LOG_ERR_VAL(trhs);
                DNDS_FILL_IN_LOG_ERR_VAL(tcomm);
                DNDS_FILL_IN_LOG_ERR_VAL(tLim);
                DNDS_FILL_IN_LOG_ERR_VAL(tLimiterA);
                DNDS_FILL_IN_LOG_ERR_VAL(tLimiterB);

                DNDS_FILL_IN_LOG_ERR_VAL(fluxWall);
                real CL{CLCur}, CD{CDCur}, AoA(AOACur);
                DNDS_FILL_IN_LOG_ERR_VAL(CL);
                DNDS_FILL_IN_LOG_ERR_VAL(CD);
                DNDS_FILL_IN_LOG_ERR_VAL(AoA);
#undef DNDS_FILL_IN_LOG_ERR_VAL

                std::get<0>(logErr)->WriteLine(std::get<1>(logErr), config.outputControl.nPrecisionLog);

                FillLogValue(logErrVal, "step", step);

                eval.ConsoleOutputBndIntegrations();
                eval.BndIntegrationLogWriteLine(
                    config.dataIOControl.getOutLogName() + "_" + output_stamp,
                    step, iStep, iter);
            }
            tstartInternal = MPI_Wtime();
            Timer().clearAllTimer();
        }

        if (iter % config.outputControl.nDataOutInternal == 0)
        {
            eval.FixUMaxFilter(u);
            PrintData(
                config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter),
                config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step), // internal series
                [&](index iCell)
                { return ode->getLatestRHS()[iCell](0); },
                addOutList,
                eval, tSimu);
            eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step),
                                 u, uRec);
        }
        if ((iter % config.outputControl.nDataOutCInternal == 0) &&
            !(config.outputControl.lazyCoverDataOutput && (iter % config.outputControl.nDataOutInternal == 0)))
        {
            eval.FixUMaxFilter(u);
            PrintData(
                config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                "",
                [&](index iCell)
                { return ode->getLatestRHS()[iCell](0); },
                addOutList,
                eval, tSimu);
            eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                                 u, uRec);
        }
        if (iter % config.outputControl.nRestartOutInternal == 0)
        {
            config.restartState.iStep = step;
            config.restartState.iStepInternal = iter;
            PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter));
        }
        if ((iter % config.outputControl.nRestartOutCInternal == 0) &&
            !(config.outputControl.lazyCoverDataOutput && (iter % config.outputControl.nRestartOutInternal == 0)))
        {
            config.restartState.iStep = step;
            config.restartState.iStepInternal = iter;
            PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + "C");
        }
        if (iter >= config.implicitCFLControl.nCFLRampStart && iter <= config.implicitCFLControl.nCFLRampLength + config.implicitCFLControl.nCFLRampStart)
        {
            real inter = real(iter - config.implicitCFLControl.nCFLRampStart) / config.implicitCFLControl.nCFLRampLength;
            real logCFL = std::log(config.implicitCFLControl.CFL) + (std::log(config.implicitCFLControl.CFLRampEnd / config.implicitCFLControl.CFL) * inter);
            CFLNow = std::exp(logCFL);
        }
        if (ifStop || iter > config.convergenceControl.nTimeStepInternal) //! TODO: reconstruct the framework of ODE-top-level-control
        {
            CFLNow = config.implicitCFLControl.CFL;
        }
        // return resRel.maxCoeff() < config.convergenceControl.rhsThresholdInternal;
        return ifStop;
    }

    DNDS_SWITCH_INTELLISENSE(template <EulerModel model>, )
    bool EulerSolver<model>::functor_fmainloop(RunningEnvironment &runningEnvironment)
    {
        using namespace std::literals;
        DNDS_EULERSOLVER_RUNNINGENV_GET_REF_LIST
        auto initUDOF = [&](ArrayDOFV<nVarsFixed> &uu)
        { vfv->BuildUDof(uu, nVars); };
        auto initUREC = [&](ArrayRECV<nVarsFixed> &uu)
        { vfv->BuildURec(uu, nVars); };
#define DNDS_EULER_SOLVER_GET_TEMP_UDOF(name)      \
    auto __p##name = uPool.getAllocInit(initUDOF); \
    auto &name = *__p##name;

        tSimu += curDtImplicit;
        if (ifOutT)
            tSimu = nextTout;
        Eigen::VectorFMTSafe<real, -1> res(nVars);
        eval.EvaluateNorm(res, ode->getLatestRHS(), 1, config.convergenceControl.useVolWiseResidual);
        if (stepCount == 0 && resBaseC.norm() == 0)
            resBaseC = res;

        if (config.timeAverageControl.enabled)
        {
            DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
            eval.MeanValueCons2Prim(u, uTemp); // could use time-step-mean-u instead of latest-u
            eval.TimeAverageAddition(uTemp, wAveraged, curDtImplicit, tAverage);
        }

        real CLCur{0.0}, CDCur{0.0}, AOACur{0.0};

        if (step % config.outputControl.nConsoleCheck == 0)
        {
            double tWall = MPI_Wtime();
            real telapsed = MPI_Wtime() - tstart;
            bool useCollectiveTimer = config.outputControl.useCollectiveTimer;
            real tcomm = Timer().getTimerColOrLoc(PerformanceTimer::Comm, mpi, useCollectiveTimer);
            real tLimiterA = Timer().getTimerColOrLoc(PerformanceTimer::LimiterA, mpi, useCollectiveTimer);
            real tLimiterB = Timer().getTimerColOrLoc(PerformanceTimer::LimiterB, mpi, useCollectiveTimer);
            real trhs = Timer().getTimerColOrLoc(PerformanceTimer::RHS, mpi, useCollectiveTimer);
            real trec = Timer().getTimerColOrLoc(PerformanceTimer::Reconstruction, mpi, useCollectiveTimer);
            real tLim = Timer().getTimerColOrLoc(PerformanceTimer::Limiter, mpi, useCollectiveTimer);

            tcomm = tInternalStats["c"].update(tcomm).getSum();
            trhs = tInternalStats["r"].update(trhs).getSum();
            trec = tInternalStats["v"].update(trec).getSum();
            tLim = tInternalStats["l"].update(tLim).getSum();
            auto tPPr = tInternalStats["p"].getSum() + Timer().getTimerColOrLoc(PerformanceTimer::PositivityOuter, mpi, useCollectiveTimer);
            if (mpi.rank == 0)
            {
                auto format = log().flags();
                std::string formatStringMain = "";
                for (auto &s : config.outputControl.consoleMainOutputFormat)
                    formatStringMain += s;
                log() << fmt::format(formatStringMain +
                                         "  "s +
                                         (config.outputControl.consoleOutputMode == 1
                                              ? "WallFlux {termYellow}{wallFlux:.6e}{termReset}"s
                                              : ""s),
                                     DNDS_FMT_ARG(step),
                                     // DNDS_FMT_ARG(iStep),
                                     // DNDS_FMT_ARG(iter),
                                     fmt::arg("resRel", (res.array() / (resBaseC.array() + verySmallReal)).transpose()),
                                     fmt::arg("wallFlux", eval.fluxWallSum.transpose()),
                                     DNDS_FMT_ARG(tSimu),
                                     DNDS_FMT_ARG(curDtImplicit),
                                     DNDS_FMT_ARG(curDtMin),
                                     DNDS_FMT_ARG(CFLNow),
                                     DNDS_FMT_ARG(nLimInc),
                                     DNDS_FMT_ARG(alphaMinInc),
                                     DNDS_FMT_ARG(nLimBeta),
                                     DNDS_FMT_ARG(minBeta),
                                     DNDS_FMT_ARG(nLimAlpha),
                                     DNDS_FMT_ARG(minAlpha),
                                     DNDS_FMT_ARG(telapsed),
                                     DNDS_FMT_ARG(trec),
                                     DNDS_FMT_ARG(trhs),
                                     DNDS_FMT_ARG(tcomm),
                                     DNDS_FMT_ARG(tLim),
                                     DNDS_FMT_ARG(tPPr),
                                     DNDS_FMT_ARG(tLimiterA),
                                     DNDS_FMT_ARG(tLimiterB),
                                     DNDS_FMT_ARG(tWall),
                                     fmt::arg("termRed", TermColor::Red),
                                     fmt::arg("termBlue", TermColor::Blue),
                                     fmt::arg("termGreen", TermColor::Green),
                                     fmt::arg("termCyan", TermColor::Cyan),
                                     fmt::arg("termYellow", TermColor::Yellow),
                                     fmt::arg("termBold", TermColor::Bold),
                                     fmt::arg("termReset", TermColor::Reset));
                log() << std::endl;
                log().setf(format);
                auto &fluxWall = eval.fluxWallSum;
                auto &logErrVal = std::get<1>(logErr);
                int iStep{-1}, iter{-1};
#define DNDS_FILL_IN_LOG_ERR_VAL(v) FillLogValue(logErrVal, #v, v)
                DNDS_FILL_IN_LOG_ERR_VAL(step);
                DNDS_FILL_IN_LOG_ERR_VAL(iStep);
                DNDS_FILL_IN_LOG_ERR_VAL(iter);
                DNDS_FILL_IN_LOG_ERR_VAL(tSimu);
                DNDS_FILL_IN_LOG_ERR_VAL(res);
                DNDS_FILL_IN_LOG_ERR_VAL(curDtImplicit);
                DNDS_FILL_IN_LOG_ERR_VAL(curDtMin);
                DNDS_FILL_IN_LOG_ERR_VAL(CFLNow);

                DNDS_FILL_IN_LOG_ERR_VAL(nLimInc);
                DNDS_FILL_IN_LOG_ERR_VAL(alphaMinInc);
                DNDS_FILL_IN_LOG_ERR_VAL(nLimBeta);
                DNDS_FILL_IN_LOG_ERR_VAL(minBeta);
                DNDS_FILL_IN_LOG_ERR_VAL(nLimAlpha);
                DNDS_FILL_IN_LOG_ERR_VAL(minAlpha);

                DNDS_FILL_IN_LOG_ERR_VAL(tWall);
                DNDS_FILL_IN_LOG_ERR_VAL(telapsed);
                DNDS_FILL_IN_LOG_ERR_VAL(trec);
                DNDS_FILL_IN_LOG_ERR_VAL(trhs);
                DNDS_FILL_IN_LOG_ERR_VAL(tcomm);
                DNDS_FILL_IN_LOG_ERR_VAL(tLim);
                DNDS_FILL_IN_LOG_ERR_VAL(tLimiterA);
                DNDS_FILL_IN_LOG_ERR_VAL(tLimiterB);

                DNDS_FILL_IN_LOG_ERR_VAL(fluxWall);
                real CL{CLCur}, CD{CDCur}, AoA(AOACur);
                DNDS_FILL_IN_LOG_ERR_VAL(CL);
                DNDS_FILL_IN_LOG_ERR_VAL(CD);
                DNDS_FILL_IN_LOG_ERR_VAL(AoA);
#undef DNDS_FILL_IN_LOG_ERR_VAL
                std::get<0>(logErr)->WriteLine(std::get<1>(logErr), config.outputControl.nPrecisionLog);

                eval.ConsoleOutputBndIntegrations();
                eval.BndIntegrationLogWriteLine(
                    config.dataIOControl.getOutLogName() + "_" + output_stamp,
                    step, -1, -1);
            }
            tstart = MPI_Wtime();
            Timer().clearAllTimer();
            for (auto &s : tInternalStats)
                s.second.clear();
        }
        if (step == nextStepOutC)
        {
            if (!(config.outputControl.lazyCoverDataOutput && (step == nextStepOut)))
            {
                eval.FixUMaxFilter(u);
                PrintData(
                    config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                    "",
                    [&](index iCell)
                    { return ode->getLatestRHS()[iCell](0); },
                    addOutList,
                    eval, tSimu);
                eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                                     u, uRec);
            }
            nextStepOutC += config.outputControl.nDataOutC;
        }
        if (step == nextStepOut)
        {
            eval.FixUMaxFilter(u);
            PrintData(
                config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step),
                config.dataIOControl.outPltName + "_" + output_stamp, // physical ts series
                [&](index iCell)
                { return ode->getLatestRHS()[iCell](0); },
                addOutList,
                eval, tSimu);
            eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step),
                                 u, uRec);
            nextStepOut += config.outputControl.nDataOut;
        }
        if (step == nextStepOutAverageC)
        {
            if (!(config.outputControl.lazyCoverDataOutput && (step == nextStepOutAverage)))
            {
                DNDS_assert(config.timeAverageControl.enabled);
                eval.MeanValuePrim2Cons(wAveraged, uAveraged);
                eval.FixUMaxFilter(uAveraged);
                PrintData(
                    config.dataIOControl.outPltName + "_TimeAveraged_" + output_stamp + "_" + "C",
                    "",
                    [&](index iCell)
                    { return ode->getLatestRHS()[iCell](0); },
                    addOutList,
                    eval, tSimu,
                    PrintDataTimeAverage);
            }
            nextStepOutAverageC += config.outputControl.nTimeAverageOutC;
        }
        if (step == nextStepOutAverage)
        {
            DNDS_assert(config.timeAverageControl.enabled);
            eval.MeanValuePrim2Cons(wAveraged, uAveraged);
            eval.FixUMaxFilter(uAveraged);
            PrintData(
                config.dataIOControl.outPltName + "_TimeAveraged_" + output_stamp + "_" + std::to_string(step),
                config.dataIOControl.outPltName + "_TimeAveraged_" + output_stamp, // time average series
                [&](index iCell)
                { return ode->getLatestRHS()[iCell](0); },
                addOutList,
                eval, tSimu,
                PrintDataTimeAverage);
            nextStepOutAverage += config.outputControl.nTimeAverageOut;
        }
        if (step == nextStepRestartC)
        {
            if (!(config.outputControl.lazyCoverDataOutput && (step == nextStepRestart)))
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = -1;
                PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + "C");
            }
            nextStepRestartC += config.outputControl.nRestartOutC;
        }
        if (step == nextStepRestart)
        {
            config.restartState.iStep = step;
            config.restartState.iStepInternal = -1;
            PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + std::to_string(step));
            nextStepRestart += config.outputControl.nRestartOut;
        }
        if (ifOutT)
        {
            eval.FixUMaxFilter(u);
            PrintData(
                config.dataIOControl.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout),
                config.dataIOControl.outPltName + "_" + output_stamp, // physical ts series
                [&](index iCell)
                { return ode->getLatestRHS()[iCell](0); },
                addOutList,
                eval, tSimu);
            eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout),
                                 u, uRec);
            nextTout += config.outputControl.tDataOut;
            if (nextTout >= config.timeMarchControl.tEnd)
                nextTout = config.timeMarchControl.tEnd;
        }
        if ((eval.settings.specialBuiltinInitializer == 2 ||
             eval.settings.specialBuiltinInitializer == 203) &&
            (step % config.outputControl.nConsoleCheck == 0)) // IV problem special: reduction on solution
        {
            auto FVal = [&](const Geom::tPoint &p, real t)
            {
                switch (eval.settings.specialBuiltinInitializer)
                {
                case 203:
                    return SpecialFields::IsentropicVortex10(eval, p, t, nVars, 10.0828);
                default:
                case 2:
                    return SpecialFields::IsentropicVortex10(eval, p, t, nVars, 5);
                }
            };
            auto FWeight = [&](const Geom::tPoint &p, real t) -> real
            {
                // real xyOrig = t;
                // real xCC = float_mod(p(0) - xyOrig, 10);
                // real yCC = float_mod(p(1) - xyOrig, 10);
                // return std::abs(xCC - 5.0) <= 2 && std::abs(yCC - 5.0) <= 2 ? 1.0 : 0.0;
                return 1.0;
            };
            Eigen::Vector<real, -1> err1, errInf;
            eval.EvaluateRecNorm(
                err1, u, uRec, 1, true,
                FVal, FWeight,
                tSimu);
            eval.EvaluateRecNorm(
                errInf, u, uRec, 1000, true,
                FVal, FWeight,
                tSimu);

            if (mpi.rank == 0)
            {
                log() << "=== Mean Error IV: [" << std::scientific
                      << std::setprecision(config.outputControl.nPrecisionConsole + 4) << err1(0) << ", "
                      << err1(0) / vfv->GetGlobalVol() << ", "
                      << errInf(0)
                      << "]" << std::endl;
            }
        }
        if (config.implicitReconstructionControl.zeroGrads)
            uRec.setConstant(0.0), gradIsZero = true;

        stepCount++;

        return tSimu >= config.timeMarchControl.tEnd;
    }
}