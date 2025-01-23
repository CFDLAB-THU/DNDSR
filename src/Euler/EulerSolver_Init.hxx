#pragma once

#include "EulerSolver.hpp"

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
            auto meshOutName = std::string(config.dataIOControl.meshFile) + "_part_" + std::to_string(mpi.size) +
                               (config.dataIOControl.meshElevation == 1 ? "_elevated"s : ""s) + ".dir";
            std::filesystem::path meshOutDir{meshOutName};
            // std::filesystem::create_directories(meshOutDir); // reading not writing
            std::string meshPartPath = getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

            SerializerJSON serializerJSON;
            serializerJSON.SetUseCodecOnUint8(true);
            SerializerBase *serializer = &serializerJSON;
            serializer->OpenFile(meshPartPath, true);
            mesh->ReadSerialize(serializer, "meshPart");
            serializer->CloseFile();
        }

        // std::cout << "here" << std::endl;
        mesh->InterpolateFace();
        mesh->AssertOnFaces();

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
            auto meshOutName = std::string(config.dataIOControl.meshFile) + "_part_" + std::to_string(mpi.size) +
                               (config.dataIOControl.meshElevation == 1 ? "_elevated"s : ""s) +
                               (config.dataIOControl.meshDirectBisect > 0 ? "_bisect" + std::to_string(config.dataIOControl.meshDirectBisect) : ""s) +
                               ".dir";
            std::filesystem::path meshOutDir{meshOutName};
            std::filesystem::create_directories(meshOutDir);
            std::string meshPartPath = DNDS::getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

            DNDS::SerializerJSON serializerJSON;
            serializerJSON.SetUseCodecOnUint8(true);
            DNDS::SerializerBase *serializer = &serializerJSON;

            serializer->OpenFile(meshPartPath, false);
            mesh->WriteSerialize(serializer, "meshPart");
            serializer->CloseFile();
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
        vfv->BuildUDof(rhsTemp1, nVars);
        vfv->BuildUDof(uTemp, nVars);
        vfv->BuildUDof(uMG1, nVars);
        vfv->BuildUDof(uMG1Init, nVars);
        vfv->BuildUDof(rhsMG1, nVars);
        vfv->BuildUDof(rhsTemp, nVars);
        if (config.timeAverageControl.enabled)
        {
            vfv->BuildUDof(wAveraged, nVars);
            vfv->BuildUDof(uAveraged, nVars);
        }

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
}