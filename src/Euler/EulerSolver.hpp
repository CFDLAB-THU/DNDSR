#pragma once

#include <iomanip>
#include <functional>
#include <tuple>
#include <filesystem>
#include <mutex>
#include <future>

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "Gas.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "EulerEvaluator.hpp"
#include "DNDS/JsonUtil.hpp"
#include "EulerBC.hpp"
#include "DNDS/SerializerJSON.hpp"
#include "Solver/Linear.hpp"
#include "DNDS/CsvLog.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

#define JSON_ASSERT DNDS_assert
#include "json.hpp"

namespace DNDS::Euler
{

    template <EulerModel model>
    class EulerSolver
    {
        int nVars;

    public:
        typedef EulerEvaluator<model> TEval;
        static const int nVarsFixed = TEval::nVarsFixed;

        static const int dim = TEval::dim;
        // static const int gdim = TEval::gdim;
        static const int gDim = TEval::gDim;
        static const int I4 = TEval::I4;

        typedef typename TEval::TU TU;
        typedef typename TEval::TDiffU TDiffU;
        typedef typename TEval::TJacobianU TJacobianU;
        typedef typename TEval::TVec TVec;
        typedef typename TEval::TMat TMat;
        typedef typename TEval::TDof TDof;
        typedef typename TEval::TRec TRec;
        typedef typename TEval::TScalar TScalar;
        typedef typename TEval::TVFV TVFV;
        typedef typename TEval::TpVFV TpVFV;

        using tGMRES_u = Linear::GMRES_LeftPreconditioned<TDof>;
        using tGMRES_uRec = Linear::GMRES_LeftPreconditioned<TRec>;
        using tPCG_uRec = Linear::PCG_PreconditionedRes<TRec, Eigen::Array<real, 1, Eigen::Dynamic>>;

    private:
        MPIInfo mpi;
        ssp<Geom::UnstructuredMesh> mesh, meshBnd;
        TpVFV vfv; // ! gDim -> 3 for intellisense
        ssp<Geom::UnstructuredMeshSerialRW> reader, readerBnd;
        ssp<EulerEvaluator<model>> pEval;

        ArrayDOFV<nVarsFixed> u, uIncBufODE, rhsTemp1, uTemp, uMG1, uMG1Init, rhsMG1, rhsTemp, wAveraged, uAveraged;
        ArrayRECV<nVarsFixed> uRec, uRecLimited, uRecNew, uRecNew1, uRecOld, uRec1, uRecInc, uRecInc1, uRecB, uRecB1;
        JacobianDiagBlock<nVarsFixed> JD, JD1, JDTmp, JSource, JSource1, JSourceTmp;
        ssp<JacobianLocalLU<nVarsFixed>> JLocalLU;
        ArrayDOFV<1> alphaPP, alphaPP1, betaPP, betaPP1, alphaPP_tmp, dTauTmp;

        int nOUTS = {-1};
        int nOUTSPoint{-1};
        int nOUTSBnd{-1};
        // rho u v w p T M ifUseLimiter RHS
        ssp<ArrayEigenVector<Eigen::Dynamic>> outDist;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerial;
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTrans;

        ssp<ArrayEigenVector<Eigen::Dynamic>> outDistPoint;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outGhostPoint;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerialPoint;
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTransPoint;
        ArrayPair<ArrayEigenVector<Eigen::Dynamic>> outDistPointPair;
        static const int maxOutFutures{3};
        std::mutex outArraysMutex;
        std::array<std::future<void>, maxOutFutures> outFuture; // mind the order, relies on the arrays and the mutex

        ssp<ArrayEigenVector<Eigen::Dynamic>> outDistBnd;
        // ssp<ArrayEigenVector<Eigen::Dynamic>> outGhostBnd;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerialBnd;
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTransBnd;
        // ArrayPair<ArrayEigenVector<Eigen::Dynamic>> outDistBndPair;
        std::mutex outBndArraysMutex;
        std::array<std::future<void>, maxOutFutures> outBndFuture; // mind the order, relies on the arrays and the mutex
        std::future<void> outSeqFuture;

        // std::vector<uint32_t> ifUseLimiter;
        CFV::tScalarPair ifUseLimiter;

        ssp<BoundaryHandler<model>> pBCHandler;

    public:
        nlohmann::ordered_json gSetting;
        std::string output_stamp = "";

        struct Configuration
        {

            struct TimeMarchControl
            {
                real dtImplicit = 1e100;
                real dtImplicitMin = 0;
                int nTimeStep = 1000000;
                bool steadyQuit = false;
                bool useRestart = false;
                bool useImplicitPP = false;
                int rhsFPPMode = 0;
                real rhsFPPScale = 1.0;
                real rhsFPPRelax = 0.9;
                real incrementPPRelax = 0.9;
                int odeCode = 0;
                real tEnd = veryLargeReal;
                real odeSetting1 = 0;
                real odeSetting2 = 0;
                real odeSetting3 = 0;
                real odeSetting4 = 0;
                bool partitionMeshOnly = false;
                real dtIncreaseLimit = 2;
                int dtIncreaseAfterCount = 0;
                real dtCFLLimitScale = 1e100;
                bool useDtPPLimit = false;
                real dtPPLimitRelax = 0.8;
                real dtPPLimitScale = 1;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    TimeMarchControl,
                    dtImplicit, dtImplicitMin, nTimeStep,
                    steadyQuit, useRestart,
                    useImplicitPP, rhsFPPMode, rhsFPPScale, rhsFPPRelax, incrementPPRelax,
                    odeCode, tEnd, odeSetting1, odeSetting2, odeSetting3, odeSetting4,
                    partitionMeshOnly,
                    dtIncreaseLimit, dtIncreaseAfterCount,
                    dtCFLLimitScale, dtPPLimitRelax,
                    useDtPPLimit, dtPPLimitScale)
            } timeMarchControl;

            struct ImplicitReconstructionControl
            {

                int nInternalRecStep = 1;
                bool zeroGrads = false;
                int recLinearScheme = 0; // 0 for original SOR, 1 for GMRES
                int nGmresSpace = 5;
                int nGmresIter = 10;
                int gmresRecScale = 1;
                int fpcgResetScheme = 0;
                real fpcgResetThres = 0.6;
                int fpcgResetReport = 0;
                int fpcgMaxPHistory = 20;
                real recThreshold = 1e-5;
                int nRecConsolCheck = 1;
                int nRecMultiplyForZeroedGrad = 1;
                bool storeRecInc = false;
                bool dampRecIncDTau = false;
                int zeroRecForSteps = 0;
                int zeroRecForStepsInternal = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ImplicitReconstructionControl,
                    nInternalRecStep, zeroGrads,
                    recLinearScheme, nGmresSpace, nGmresIter, gmresRecScale,
                    fpcgResetScheme, fpcgResetThres, fpcgResetReport, fpcgMaxPHistory,
                    recThreshold, nRecConsolCheck,
                    nRecMultiplyForZeroedGrad,
                    storeRecInc, dampRecIncDTau,
                    zeroRecForSteps,
                    zeroRecForStepsInternal)
            } implicitReconstructionControl;

            struct OutputControl
            {
                int nConsoleCheck = 1;
                int nConsoleCheckInternal = 1;
                int consoleOutputMode = 0; // 0 for basic, 1 for wall force out
                int consoleOutputEveryFix = 0;
                int nPrecisionConsole = 3;
                std::vector<std::string> consoleMainOutputFormat{
                    "=== Step {termBold}[{step:4d}]   ",
                    "res {termBold}{termRed}{resRel:.3e}{termReset}   ",
                    "t,dT,dTaumin,CFL,nFix {termGreen}[{tSimu:.3e},{curDtImplicit:.3e},{curDtMin:.3e},{CFLNow:.3e},[alphaInc({nLimInc},{alphaMinInc}), betaRec({nLimBeta},{minBeta}), alphaRes({nLimAlpha},{minAlpha})]]{termReset}   ",
                    "Time[{telapsed:.3f}] recTime[{trec:.3f}] rhsTime[{trhs:.3f}] commTime[{tcomm:.3f}] limTime[{tLim:.3f}] limTimeA[{tLimiterA:.3f}] limTimeB[{tLimiterB:.3f}]"};
                std::vector<std::string> consoleMainOutputFormatInternal{
                    "\t Internal === Step [{step:4d},{iStep:2d},{iter:4d}]   ",
                    "res {termRed}{resRel:.3e}{termReset}   ",
                    "t,dT,dTaumin,CFL,nFix {termGreen}[{tSimu:.3e},{curDtImplicit:.3e},{curDtMin:.3e},{CFLNow:.3e},[alphaInc({nLimInc},{alphaMinInc:.3g}), betaRec({nLimBeta},{minBeta:.3g}), alphaRes({nLimAlpha},{minAlpha:.3g})]]{termReset}   ",
                    "Time[{telapsedM:.3f}] recTime[{trecM:.3f}] rhsTime[{trhsM:.3f}] commTime[{tcommM:.3f}] limTime[{tLimM:.3f}] limTimeA[{tLimiterA:.3f}] limTimeB[{tLimiterB:.3f}]"};
                std::vector<std::string> logfileOutputTitles{
                    "step", "iStep", "iter", "tSimu",
                    "res", "curDtImplicit", "curDtMin", "CFLNow",
                    "nLimInc", "alphaMinInc",
                    "nLimBeta", "minBeta",
                    "nLimAlpha", "minAlpha",
                    "tWall", "telapsed", "trec", "trhs", "tcomm", "tLim", "tLimiterA", "tLimiterB",
                    "fluxWall", "CL", "CD", "AoA"};
                int nPrecisionLog = 10;
                bool dataOutAtInit = false;
                int nDataOut = 10000;
                int nDataOutC = 50;
                int nDataOutInternal = 10000;
                int nDataOutCInternal = 1;
                int nRestartOut = INT_MAX;
                int nRestartOutC = INT_MAX;
                int nRestartOutInternal = INT_MAX;
                int nRestartOutCInternal = INT_MAX;
                int nTimeAverageOut = INT_MAX;
                int nTimeAverageOutC = INT_MAX;
                real tDataOut = veryLargeReal;
                bool lazyCoverDataOutput = false;
                bool useCollectiveTimer = false;

                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    OutputControl,
                    nConsoleCheck,
                    nConsoleCheckInternal,
                    consoleOutputMode,
                    consoleOutputEveryFix,
                    consoleMainOutputFormat,
                    consoleMainOutputFormatInternal,
                    logfileOutputTitles, nPrecisionLog,
                    dataOutAtInit,
                    nDataOut, nDataOutC,
                    nDataOutInternal, nDataOutCInternal,
                    nRestartOut, nRestartOutC,
                    nRestartOutInternal, nRestartOutCInternal,
                    nTimeAverageOut, nTimeAverageOutC,
                    tDataOut, lazyCoverDataOutput,
                    useCollectiveTimer)
            } outputControl;

            struct ImplicitCFLControl
            {
                real CFL = 10;
                int nForceLocalStartStep = INT_MAX;
                int nCFLRampStart = INT_MAX;
                int nCFLRampLength = INT_MAX;
                real CFLRampEnd = 0;
                bool useLocalDt = true;
                int nSmoothDTau = 0;
                real RANSRelax = 1;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ImplicitCFLControl,
                    CFL,
                    nForceLocalStartStep,
                    nCFLRampStart,
                    nCFLRampLength,
                    CFLRampEnd,
                    useLocalDt,
                    nSmoothDTau,
                    RANSRelax)
            } implicitCFLControl;

            struct ConvergenceControl
            {
                int nTimeStepInternal = 20;
                int nTimeStepInternalMin = 5;
                int nAnchorUpdate = 1;
                int nAnchorUpdateStart = 0;
                real rhsThresholdInternal = 1e-10;
                real res_base = 0;
                bool useVolWiseResidual = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ConvergenceControl,
                    nTimeStepInternal, nTimeStepInternalMin,
                    nAnchorUpdate, nAnchorUpdateStart,
                    rhsThresholdInternal,
                    res_base,
                    useVolWiseResidual)
            } convergenceControl;

            struct DataIOControl
            {
                bool uniqueStamps = true;
                real meshRotZ = 0;
                real meshScale = 1.0;

                int meshElevation = 0;                 // 0 = noOp, 1 = O1->O2
                int meshElevationInternalSmoother = 0; // 0 = local interpolation, 1 = coupled
                int meshElevationIter = 1000;          // -1 to use handle all the nodes
                int meshElevationNSearch = 30;
                real meshElevationRBFRadius = 1;
                real meshElevationRBFPower = 1;
                Geom::RBF::RBFKernelType meshElevationRBFKernel = Geom::RBF::InversedDistanceA1;
                real meshElevationMaxIncludedAngle = 15;
                real meshElevationRefDWall = 1e-3;
                int meshElevationBoundaryMode = 0; // 0: only wall bc; 1: wall + invis vall

                int meshDirectBisect = 0;

                int meshFormat = 0;
                Geom::UnstructuredMeshSerialRW::PartitionOptions meshPartitionOptions;
                std::string meshFile = "data/mesh/NACA0012_WIDE_H3.cgns";
                std::string outPltName = "data/out/debugData_";
                std::string outLogName = "";
                std::string outRestartName = "";

                int outPltMode = 0;   // 0 = serial, 1 = dist plt
                int readMeshMode = 0; // 0 = serial cgns, 1 = dist json
                bool outPltTecplotFormat = true;
                bool outPltVTKFormat = false;
                bool outPltVTKHDFFormat = false;
                bool outAtPointData = true;
                bool outAtCellData = true;
                int nASCIIPrecision = 5;
                std::string vtuFloatEncodeMode = "binary";
                int hdfChunkSize = 32768;
                int hdfDeflateLevel = 0;
                bool outVolumeData = true;
                bool outBndData = false;

                std::vector<std::string> outCellScalarNames{};

                bool serializerSaveURec = false;

                bool allowAsyncPrintData = false;

                int rectifyNearPlane = 0; // 1: x 2: y 4: z
                real rectifyNearPlaneThres = 1e-10;

                const std::string &getOutLogName()
                {
                    return outLogName.empty() ? outPltName : outLogName;
                }

                const std::string &getOutRestartName()
                {
                    return outRestartName.empty() ? outPltName : outRestartName;
                }

                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    DataIOControl,
                    uniqueStamps,
                    meshRotZ, meshScale,
                    meshElevation, meshElevationInternalSmoother,
                    meshElevationIter,
                    meshElevationRBFRadius, meshElevationRBFPower,
                    meshElevationRBFKernel, meshElevationMaxIncludedAngle, meshElevationNSearch, meshElevationRefDWall,
                    meshElevationBoundaryMode,
                    meshDirectBisect,
                    meshFormat,
                    meshPartitionOptions,
                    meshFile,
                    outPltName,
                    outLogName,
                    outRestartName,
                    outPltMode,
                    readMeshMode,
                    outPltTecplotFormat,
                    outPltVTKFormat,
                    outPltVTKHDFFormat,
                    outAtPointData,
                    outAtCellData,
                    nASCIIPrecision,
                    vtuFloatEncodeMode,
                    hdfChunkSize, hdfDeflateLevel,
                    outVolumeData,
                    outBndData,
                    outCellScalarNames,
                    serializerSaveURec,
                    allowAsyncPrintData,
                    rectifyNearPlane, rectifyNearPlaneThres)
            } dataIOControl;

            struct BoundaryDefinition
            {
                Eigen::Vector<real, -1> PeriodicTranslation1;
                Eigen::Vector<real, -1> PeriodicTranslation2;
                Eigen::Vector<real, -1> PeriodicTranslation3;
                Eigen::Vector<real, -1> PeriodicRotationCent1;
                Eigen::Vector<real, -1> PeriodicRotationCent2;
                Eigen::Vector<real, -1> PeriodicRotationCent3;
                Eigen::Vector<real, -1> PeriodicRotationEulerAngles1;
                Eigen::Vector<real, -1> PeriodicRotationEulerAngles2;
                Eigen::Vector<real, -1> PeriodicRotationEulerAngles3;
                real periodicTolerance = 1e-8;
                BoundaryDefinition()
                {
                    PeriodicTranslation1.resize(3);
                    PeriodicTranslation2.resize(3);
                    PeriodicTranslation3.resize(3);
                    PeriodicTranslation1 << 1, 0, 0;
                    PeriodicTranslation2 << 0, 1, 0;
                    PeriodicTranslation3 << 0, 0, 1;
                    PeriodicRotationCent1.setZero(3);
                    PeriodicRotationCent2.setZero(3);
                    PeriodicRotationCent3.setZero(3);
                    PeriodicRotationEulerAngles1.setZero(3);
                    PeriodicRotationEulerAngles2.setZero(3);
                    PeriodicRotationEulerAngles3.setZero(3);
                }

                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    BoundaryDefinition,
                    PeriodicTranslation1,
                    PeriodicTranslation2,
                    PeriodicTranslation3,
                    PeriodicRotationCent1,
                    PeriodicRotationCent2,
                    PeriodicRotationCent3,
                    PeriodicRotationEulerAngles1,
                    PeriodicRotationEulerAngles2,
                    PeriodicRotationEulerAngles3,
                    periodicTolerance)
            } boundaryDefinition;

            struct LimiterControl
            {
                bool useLimiter = true;
                bool usePPRecLimiter = true;
                bool useViscousLimited = true;
                int smoothIndicatorProcedure = 0;
                int limiterProcedure = 0; // 0 for V2==3WBAP, 1 for V3==CWBAP
                int nPartialLimiterStart = INT_MAX;
                int nPartialLimiterStartLocal = INT_MAX;
                bool preserveLimited = false;
                bool ppRecLimiterCompressToMean = true;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    LimiterControl,
                    useLimiter, usePPRecLimiter, useViscousLimited,
                    smoothIndicatorProcedure, limiterProcedure,
                    nPartialLimiterStart, nPartialLimiterStartLocal,
                    preserveLimited,
                    ppRecLimiterCompressToMean)
            } limiterControl;

            struct LinearSolverControl
            {
                int jacobiCode = 1; // 0 for jacobi, 1 for gs, 2 for ilu
                int sgsIter = 0;
                int sgsWithRec = 0;
                int gmresCode = 0;  // 0 for lusgs, 1 for gmres, 2 for lusgs started gmres
                int gmresScale = 0; // 0 for no scaling, 1 use refU, 2 use mean value
                int nGmresSpace = 10;
                int nGmresIter = 2;
                int nSgsConsoleCheck = 100;
                int nGmresConsoleCheck = 100;
                bool initWithLastURecInc = false;
                int multiGridLP = 0;
                int multiGridLPInnerNIter = 4;
                int multiGridLPStartIter = 0;
                int multiGridLPInnerNSee = 10;
                Direct::DirectPrecControl directPrecControl;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    LinearSolverControl,
                    jacobiCode,
                    sgsIter, sgsWithRec, gmresCode, gmresScale,
                    nGmresSpace, nGmresIter,
                    nSgsConsoleCheck, nGmresConsoleCheck,
                    initWithLastURecInc,
                    multiGridLP, multiGridLPInnerNIter, multiGridLPStartIter, multiGridLPInnerNSee,
                    directPrecControl)
            } linearSolverControl;

            struct RestartState
            {
                int iStep = -1;
                int iStepInternal = -1;
                int odeCodePrev = -1;
                std::string lastRestartFile = "";
                std::string otherRestartFile = "";
                std::vector<int> otherRestartStoreDim;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    RestartState,
                    iStep, iStepInternal, odeCodePrev,
                    lastRestartFile,
                    otherRestartFile, otherRestartStoreDim)
                RestartState()
                {
                    otherRestartStoreDim.resize(1);
                    for (int i = 0; i < otherRestartStoreDim.size(); i++)
                        otherRestartStoreDim[i] = i;
                }
            } restartState;

            struct TimeAverageControl
            {
                bool enabled = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    TimeAverageControl,
                    enabled)
            } timeAverageControl;

            struct Others
            {
                int nFreezePassiveInner = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    Others,
                    nFreezePassiveInner)
            } others;

            nlohmann::ordered_json eulerSettings = nlohmann::ordered_json::object();
            nlohmann::ordered_json vfvSettings = nlohmann::ordered_json::object();
            nlohmann::ordered_json bcSettings = nlohmann::ordered_json::array();
            std::map<std::string, std::string> bcNameMapping;

            void
            ReadWriteJson(nlohmann::ordered_json &jsonObj, int nVars, bool read)
            {

                __DNDS__json_to_config(timeMarchControl);
                __DNDS__json_to_config(implicitReconstructionControl);
                __DNDS__json_to_config(outputControl);
                __DNDS__json_to_config(implicitCFLControl);
                __DNDS__json_to_config(convergenceControl);
                __DNDS__json_to_config(dataIOControl);
                __DNDS__json_to_config(boundaryDefinition);
                __DNDS__json_to_config(limiterControl);
                __DNDS__json_to_config(linearSolverControl);
                __DNDS__json_to_config(timeAverageControl);
                __DNDS__json_to_config(others);
                __DNDS__json_to_config(restartState);

                __DNDS__json_to_config(eulerSettings);
                __DNDS__json_to_config(vfvSettings);
                __DNDS__json_to_config(bcSettings);
                __DNDS__json_to_config(bcNameMapping);
                if (read)
                {
                    DNDS_assert(eulerSettings.is_object());
                    DNDS_assert(vfvSettings.is_object());
                    DNDS_assert(bcSettings.is_array());
                }

                // TODO: BC settings
            }

            Configuration()
            {
            }

            Configuration(int nVars)
            {
                vfvSettings = CFV::VRSettings{gDim};
                EulerEvaluatorSettings<model>().ReadWriteJSON(eulerSettings, nVars, false);
                bcSettings = BoundaryHandler<model>(nVars);
            }

        } config = Configuration{};

    public:
        EulerSolver(const MPIInfo &nmpi) : nVars(getNVars(model)), mpi(nmpi)
        {
            nOUTS = nVars + 4;
            nOUTSPoint = nVars + 2;
            nOUTSBnd = nVars + 2 + nVars + 3 + 1 + 3; // Uprim + (T,M) + F + Ft + faceZone + Norm

            config = Configuration(nVars); //* important to initialize using nVars
        }

        ~EulerSolver()
        {
            int nBad{0};
            do
            {
                nBad = 0;
                for (auto &f : outFuture)
                    if (f.valid() && f.wait_for(std::chrono::microseconds(10)) != std::future_status::ready)
                        nBad++;
                for (auto &f : outBndFuture)
                    if (f.valid() && f.wait_for(std::chrono::microseconds(10)) != std::future_status::ready)
                        nBad++;
                if (outSeqFuture.valid() && outSeqFuture.wait_for(std::chrono::microseconds(10)) != std::future_status::ready)
                    nBad++;
            } while (nBad);
        }

        /**
         */
        void ConfigureFromJson(const std::string &jsonName, bool read = false, const std::string &jsonMergeName = "",
                               const std::vector<std::string> &overwriteKeys = {}, const std::vector<std::string> &overwriteValues = {})
        {
            if (read)
            {
                auto fIn = std::ifstream(jsonName);
                DNDS_assert_info(fIn, "config file not existent");
                gSetting = nlohmann::ordered_json::parse(fIn, nullptr, true, true);

                if (read && !jsonMergeName.empty())
                {
                    fIn = std::ifstream(jsonMergeName);
                    DNDS_assert_info(fIn, "config file patch not existent");
                    auto gSettingAdd = nlohmann::ordered_json::parse(fIn, nullptr, true, true);
                    gSetting.merge_patch(gSettingAdd);
                }
                DNDS_assert(overwriteKeys.size() == overwriteValues.size());
                for (size_t i = 0; i < overwriteKeys.size(); i++)
                {

                    auto key = nlohmann::ordered_json::json_pointer(overwriteKeys[i].c_str());
                    std::string valString =
                        fmt::format(R"({{
"__val_entry": {}
}})",
                                    overwriteValues[i]);
                    try
                    {
                        auto valDoc = nlohmann::ordered_json::parse(valString, nullptr, true, true);
                        if (mpi.rank == 0)
                            log() << "JSON: overwrite key: " << key << std::endl
                                  << "JSON: overwrite val: " << valDoc["__val_entry"] << std::endl;
                        gSetting[key] = valDoc["__val_entry"];
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << e.what() << "\n";
                        std::cerr << overwriteValues[i] << "\n";
                        DNDS_assert(false);
                    }
                }
                config.ReadWriteJson(gSetting, nVars, read);
                DNDS_MAKE_SSP(pBCHandler, nVars);
                from_json(config.bcSettings, *pBCHandler);
                gSetting["bcSettings"] = *pBCHandler;
                PrintConfig(true);
                if (mpi.rank == 0)
                    log() << "JSON: read value:" << std::endl
                          << std::setw(4) << gSetting << std::endl;
            }
            else
            {
                gSetting = nlohmann::ordered_json::object();
                config.ReadWriteJson(gSetting, nVars, read);
                if (pBCHandler) // todo: add example pBCHandler
                    gSetting["bcSettings"] = *pBCHandler;
                if (mpi.rank == 0) // single call for output
                {
                    std::filesystem::path outFile{jsonName};
                    std::filesystem::create_directories(outFile.parent_path() / ".");
                    auto fIn = std::ofstream(jsonName);
                    DNDS_assert(fIn);
                    fIn << std::setw(4) << gSetting;
                }
                MPI::Barrier(mpi.comm); // no go until output done
            }

            if (mpi.rank == 0)
                log() << "JSON: Parse " << (read ? "read" : "write")
                      << " Done ===" << std::endl;
        }

        void ReadMeshAndInitialize();

        void PrintConfig(bool updateCommit = false)
        {
            /***********************************************************/
            // if these objects are existent, extract settings from them
            if (vfv)
                config.vfvSettings = vfv->settings;
            if (pEval)
                pEval->settings.ReadWriteJSON(config.eulerSettings, nVars, false);
            if (pBCHandler)
                config.bcSettings = *pBCHandler;
            /***********************************************************/
            config.ReadWriteJson(gSetting, nVars, false);
            if (mpi.rank == 0)
            {
                std::string logConfigFileName = config.dataIOControl.getOutLogName() + "_" + output_stamp + ".config.json";
                std::filesystem::path outFile{logConfigFileName};
                std::filesystem::create_directories(outFile.parent_path() / ".");
                std::ofstream logConfig(logConfigFileName);
                DNDS_assert(logConfig);
                gSetting["___Compile_Time_Defines"] = DNDS_Defines_state;
                gSetting["___Runtime_PartitionNumber"] = mpi.size;
                gSetting["___Commit_ID"] = DNDS_MACRO_TO_STRING(DNDS_CURRENT_COMMIT_HASH);
                // if (updateCommit)
                // {
                //     std::ifstream commitIDFile("commitID.txt");
                //     if (commitIDFile)
                //     {
                //         std::string commitHash;
                //         commitIDFile >> commitHash;
                //         gSetting["___Commit_ID"] = commitHash;
                //     }
                // }
                logConfig << std::setw(4) << gSetting;
                logConfig.close();
            }
        }
        void ReadRestart(std::string fname)
        {
            if (mpi.rank == 0)
                log() << fmt::format("=== Reading Restart From [{}]", fname) << std::endl;
            std::filesystem::path outPath;
            // outPath = {fname + "_p" + std::to_string(mpi.size) + "_restart.dir"};
            outPath = fname;
            // std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%04d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));

            SerializerJSON serializerJSON;
            serializerJSON.SetUseCodecOnUint8(true);
            SerializerBase *serializer = &serializerJSON;
            serializer->OpenFile(fname, true);
            u.ReadSerialize(serializer, "u");
            serializer->CloseFile();
            // config.restartState.lastRestartFile = outPath;
            // PrintConfig();
            if (mpi.rank == 0)
                log() << fmt::format("=== Read Restart") << std::endl;
        }

        void ReadRestartOtherSolver(std::string fname, const std::vector<int> &dimStore)
        {
            ArrayDOFV<Eigen::Dynamic> readBuf;
            DNDS_MAKE_SSP(readBuf.father, mpi);
            DNDS_MAKE_SSP(readBuf.son, mpi);

            if (mpi.rank == 0)
                log() << fmt::format("=== Reading Restart From [{}]", fname) << std::endl;
            std::filesystem::path outPath;
            // outPath = {fname + "_p" + std::to_string(mpi.size) + "_restart.dir"};
            outPath = fname;
            // std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%04d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));

            SerializerJSON serializerJSON;
            serializerJSON.SetUseCodecOnUint8(true);
            SerializerBase *serializer = &serializerJSON;
            serializer->OpenFile(fname, true);
            readBuf.ReadSerialize(serializer, "u");
            serializer->CloseFile();
            DNDS_assert_info(readBuf.father->Size() == u.father->Size(), fmt::format("{}, {}", readBuf.father->Size(), u.father->Size()));
            DNDS_assert_info(readBuf.son->Size() == u.son->Size(), fmt::format("{}, {}", readBuf.son->Size(), u.son->Size()));
            int iMax = std::min(u.RowSize(), readBuf.RowSize()) - 1; // could use this
            for (auto item : dimStore)
                DNDS_assert(item <= iMax);
            for (index iCell = 0; iCell < u.Size(); iCell++)
            {
                // std::cout << iCell << ": " << readBuf[iCell].transpose() << std::endl;

                u[iCell](dimStore) = readBuf[iCell](dimStore);
            }

            if (mpi.rank == 0)
                log() << fmt::format("=== Read Restart") << std::endl;
        }

        void PrintRestart(std::string fname)
        {
            std::filesystem::path outPath;
            outPath = {fname + "_p" + std::to_string(mpi.size) + "_restart.dir"};
            std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%04d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));

            SerializerJSON serializerJSON;
            serializerJSON.SetUseCodecOnUint8(true);
            SerializerBase *serializer = &serializerJSON;
            serializer->OpenFile(fname, false);
            u.WriteSerialize(serializer, "u");
            serializer->CloseFile();
            config.restartState.lastRestartFile = getStringForcePath(outPath);
            PrintConfig();
        }

        using tAdditionalCellScalarList = tCellScalarList;

        enum PrintDataMode
        {
            PrintDataLatest = 0,
            PrintDataTimeAverage = 1,
        };

        void PrintData(const std::string &fname, const std::string &fnameSeries,
                       const tCellScalarFGet &odeResidualF,
                       tAdditionalCellScalarList &additionalCellScalars,
                       TEval &eval, real TSimu = -1.0, PrintDataMode mode = PrintDataLatest);

        void WriteSerializer(SerializerBase *serializer, const std::string &name) // currently not using
        {
            auto cwd = serializer->GetCurrentPath();
            serializer->CreatePath(name);
            serializer->GoToPath(name);

            u.WriteSerialize(serializer, "meanValue");

            serializer->GoToPath(cwd);

            nlohmann::ordered_json configJson;
            config.ReadWriteJson(configJson, nVars, false);
            serializer->WriteString("lastConfig", configJson.dump());

            if (config.dataIOControl.serializerSaveURec)
            {
                serializer->WriteInt("hasReconstructionValue", 1);
                uRec.WriteSerialize(serializer, "recValue");
            }
            else
                serializer->WriteInt("hasReconstructionValue", 0);
        }

        template <class TVal>
        std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<TVal>>>
        FillLogValue(tLogSimpleDIValueMap &v_map, const std::string &name, TVal &&val)
        {
            v_map[name] = val;
        }

        template <class TVal>
        std::enable_if_t<!std::is_arithmetic_v<std::remove_reference_t<TVal>>>
        FillLogValue(tLogSimpleDIValueMap &v_map, const std::string &name, TVal &&val)
        {
            // std::vector<std::string> logfileOutputTitlses{
            //     "step", "iStep", "iter", "tSimu",
            //     "res", "curDtImplicit", "curDtMin", "CFLNow",
            //     "nLimInc", "alphaMinInc",
            //     "nLimBeta", "minBeta",
            //     "nLimAlpha", "minAlpha",
            //     "tWall", "telapsed", "trec", "trhs", "tcomm", "tLim", "tLimiterA", "tLimiterB",
            //     "fluxWall", "CL", "CD", "AoA"};
            if (name == "res" || name == "fluxWall")
                for (int i = 0; i < nVars; i++)
                    v_map[name + std::to_string(i)] = val[i];
            else
                v_map[name] = 0;
        }

        std::tuple<CsvLog, tLogSimpleDIValueMap> LogErrInitialize()
        {
            tLogSimpleDIValueMap v_map;
            TU initVec;
            initVec.setZero(nVars);
            std::vector<std::string> realNames;
            for (auto name : config.outputControl.logfileOutputTitles)
                if (name == "res" || name == "fluxWall")
                {
                    FillLogValue(v_map, name, initVec);
                    for (int i = 0; i < nVars; i++)
                        realNames.push_back(name + std::to_string(i));
                }
                else
                    FillLogValue(v_map, name, 0.), realNames.push_back(name);

            std::string logErrFileName = config.dataIOControl.getOutLogName() + "_" + output_stamp + ".log";
            std::filesystem::path outFile{logErrFileName};
            std::filesystem::create_directories(outFile.parent_path() / ".");
            auto pOs = std::make_unique<std::ofstream>(logErrFileName);
            DNDS_assert_info(*pOs, "logErr file [" + logErrFileName + "] did not open");
            return std::make_tuple(DNDS::CsvLog(realNames, std::move(pOs)), v_map);
        }

        void RunImplicitEuler();
        void solveLinear(
            real alphaDiag,
            TDof &cres, TDof &cx, TDof &cxInc, TRec &uRecC, TRec uRecIncC,
            JacobianDiagBlock<nVarsFixed> &JDC, tGMRES_u &gmres);
        void doPrecondition(real alphaDiag, TDof &crhs, TDof &cx, TDof &cxInc, TDof &uTemp,
                            JacobianDiagBlock<nVarsFixed> &JDC, TU &sgsRes, bool &inputIsZero, bool &hasLUDone);
    };
}

#define DNDS_EULERSOLVER_INS_EXTERN(model, ext)                   \
    namespace DNDS::Euler                                         \
    {                                                             \
        ext template void EulerSolver<model>::RunImplicitEuler(); \
    }

DNDS_EULERSOLVER_INS_EXTERN(NS, extern);
DNDS_EULERSOLVER_INS_EXTERN(NS_2D, extern);
DNDS_EULERSOLVER_INS_EXTERN(NS_SA, extern);
DNDS_EULERSOLVER_INS_EXTERN(NS_2EQ, extern);
DNDS_EULERSOLVER_INS_EXTERN(NS_3D, extern);
DNDS_EULERSOLVER_INS_EXTERN(NS_SA_3D, extern);
DNDS_EULERSOLVER_INS_EXTERN(NS_2EQ_3D, extern);

#define DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(model, ext)             \
    namespace DNDS::Euler                                             \
    {                                                                 \
        ext template void EulerSolver<model>::PrintData(              \
            const std::string &fname, const std::string &fnameSeries, \
            const tCellScalarFGet &odeResidualF,                      \
            tAdditionalCellScalarList &additionalCellScalars,         \
            TEval &eval, real tSimu,                                  \
            PrintDataMode mode);                                      \
    }

DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_2D, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_SA, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_2EQ, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_3D, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_SA_3D, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_2EQ_3D, extern);

#define DNDS_EULERSOLVER_INIT_INS_EXTERN(model, ext)                   \
    namespace DNDS::Euler                                              \
    {                                                                  \
        ext template void EulerSolver<model>::ReadMeshAndInitialize(); \
    }

DNDS_EULERSOLVER_INIT_INS_EXTERN(NS, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_2D, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_SA, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_2EQ, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_3D, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_SA_3D, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_2EQ_3D, extern);