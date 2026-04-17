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
#include "DNDS/JsonUtil.hpp"
#include "DNDS/ConfigParam.hpp"
#include "DNDS/SerializerFactory.hpp"
#include "DNDS/CsvLog.hpp"
#include "DNDS/ObjectPool.hpp"
#include "Solver/Linear.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "Gas.hpp"
#include "EulerEvaluator.hpp"
#include "EulerBC.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

#include "DNDS/JsonUtil.hpp"

#include "Solver/ODE.hpp"
#include "Solver/Linear.hpp"

namespace DNDS::Euler
{

    template <EulerModel model>
    class EulerSolver
    {
        int nVars = getNVars(model);

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

        ArrayDOFV<nVarsFixed> u, uIncBufODE, wAveraged, uAveraged;
        ObjectPool<ArrayDOFV<nVarsFixed>> uPool; // for temporary u entries
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
                nlohmann::ordered_json odeSettingsExtra;
                bool partitionMeshOnly = false;
                real dtIncreaseLimit = 2;
                int dtIncreaseAfterCount = 0;
                real dtCFLLimitScale = 1e100;
                bool useDtPPLimit = false;
                real dtPPLimitRelax = 0.8;
                real dtPPLimitScale = 1;
                DNDS_DECLARE_CONFIG(TimeMarchControl)
                {
                    DNDS_FIELD(dtImplicit,          "Max implicit time step; 1e100 for steady",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(dtImplicitMin,       "Minimum implicit time step",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(nTimeStep,           "Max number of time steps",
                               DNDS::Config::range(1));
                    DNDS_FIELD(steadyQuit,          "Quit on steady convergence");
                    DNDS_FIELD(useRestart,          "Initialize from restart file");
                    DNDS_FIELD(useImplicitPP,       "Use implicit positivity-preserving");
                    DNDS_FIELD(rhsFPPMode,          "RHS flux-positivity-preserving mode");
                    DNDS_FIELD(rhsFPPScale,         "RHS FPP scaling factor");
                    DNDS_FIELD(rhsFPPRelax,         "RHS FPP relaxation factor",
                               DNDS::Config::range(0.0, 1.0));
                    DNDS_FIELD(incrementPPRelax,    "Increment PP relaxation factor",
                               DNDS::Config::range(0.0, 1.0));
                    DNDS_FIELD(odeCode,             "ODE integrator code");
                    DNDS_FIELD(tEnd,                "End time for unsteady simulation");
                    DNDS_FIELD(odeSetting1,         "ODE parameter 1");
                    DNDS_FIELD(odeSetting2,         "ODE parameter 2");
                    DNDS_FIELD(odeSetting3,         "ODE parameter 3");
                    DNDS_FIELD(odeSetting4,         "ODE parameter 4");
                    config.field_json(&T::odeSettingsExtra, "odeSettingsExtra", "Extra ODE integrator settings (opaque JSON)");
                    DNDS_FIELD(partitionMeshOnly,   "Only partition mesh, then exit");
                    DNDS_FIELD(dtIncreaseLimit,     "Max factor for dt increase per step",
                               DNDS::Config::range(1.0));
                    DNDS_FIELD(dtIncreaseAfterCount,"Steps before dt increase allowed",
                               DNDS::Config::range(0));
                    DNDS_FIELD(dtCFLLimitScale,     "CFL-based dt limit scale");
                    DNDS_FIELD(useDtPPLimit,        "Use positivity-preserving dt limiter");
                    DNDS_FIELD(dtPPLimitRelax,      "PP dt limiter relaxation",
                               DNDS::Config::range(0.0, 1.0));
                    DNDS_FIELD(dtPPLimitScale,      "PP dt limiter scale",
                               DNDS::Config::range(0.0));
                }
                bool timeMarchIsTwoStage()
                {
                    return odeCode == 401 || (odeCode >= 411 && odeCode <= 413);
                }
            } timeMarchControl;

            struct ImplicitReconstructionControl
            {
                bool useExplicit = false;
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
                DNDS_DECLARE_CONFIG(ImplicitReconstructionControl)
                {
                    DNDS_FIELD(useExplicit,                "Use explicit reconstruction (no implicit)");
                    DNDS_FIELD(nInternalRecStep,           "Number of internal reconstruction sub-steps",
                               DNDS::Config::range(1));
                    DNDS_FIELD(zeroGrads,                  "Zero gradients before reconstruction");
                    DNDS_FIELD(recLinearScheme,            "Reconstruction linear solver: 0=SOR, 1=GMRES");
                    DNDS_FIELD(nGmresSpace,                "GMRES Krylov subspace size for reconstruction",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nGmresIter,                 "GMRES iterations for reconstruction",
                               DNDS::Config::range(1));
                    DNDS_FIELD(gmresRecScale,              "GMRES reconstruction scaling mode");
                    DNDS_FIELD(fpcgResetScheme,            "FPCG reset scheme");
                    DNDS_FIELD(fpcgResetThres,             "FPCG reset threshold",
                               DNDS::Config::range(0.0, 1.0));
                    DNDS_FIELD(fpcgResetReport,            "FPCG reset report frequency");
                    DNDS_FIELD(fpcgMaxPHistory,            "FPCG max preconditioner history",
                               DNDS::Config::range(1));
                    DNDS_FIELD(recThreshold,               "Reconstruction convergence threshold",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(nRecConsolCheck,            "Console output interval for reconstruction",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nRecMultiplyForZeroedGrad,  "Rec iteration multiplier when grad is zeroed",
                               DNDS::Config::range(1));
                    DNDS_FIELD(storeRecInc,                "Store reconstruction increment");
                    DNDS_FIELD(dampRecIncDTau,             "Damp reconstruction increment by dTau");
                    DNDS_FIELD(zeroRecForSteps,            "Zero reconstruction for N outer steps",
                               DNDS::Config::range(0));
                    DNDS_FIELD(zeroRecForStepsInternal,    "Zero reconstruction for N internal steps",
                               DNDS::Config::range(0));
                }
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
                    "step", "iStep", "iterAll", "iter", "tSimu",
                    "res", "curDtImplicit", "curDtMin", "CFLNow",
                    "nLimInc", "alphaMinInc",
                    "nLimBeta", "minBeta",
                    "nLimAlpha", "minAlpha",
                    "tWall", "telapsed", "trec", "trhs", "tcomm", "tLim", "tLimiterA", "tLimiterB",
                    "fluxWall", "CL", "CD", "AoA"};
                int nPrecisionLog = 10;
                bool dataOutAtInit = false;
                bool restartOutAtInit = false;
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

                DNDS_DECLARE_CONFIG(OutputControl)
                {
                    DNDS_FIELD(nConsoleCheck,                    "Console output interval (outer steps)",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nConsoleCheckInternal,            "Console output interval (internal steps)",
                               DNDS::Config::range(1));
                    DNDS_FIELD(consoleOutputMode,               "Console output mode: 0=basic, 1=wall force");
                    DNDS_FIELD(consoleOutputEveryFix,            "Force console output every N fix iterations");
                    // nPrecisionConsole intentionally not registered: was not serialized
                    // in the original macro and existing config files lack this key.
                    DNDS_FIELD(consoleMainOutputFormat,         "Format strings for console output lines");
                    DNDS_FIELD(consoleMainOutputFormatInternal, "Format strings for internal-step console output");
                    DNDS_FIELD(logfileOutputTitles,             "Column titles for CSV log file");
                    DNDS_FIELD(nPrecisionLog,                   "Log file floating-point precision",
                               DNDS::Config::range(1));
                    DNDS_FIELD(dataOutAtInit,                   "Output data at initialization");
                    DNDS_FIELD(restartOutAtInit,                "Output restart at initialization");
                    DNDS_FIELD(nDataOut,                        "Data output interval (outer steps)",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nDataOutC,                       "Data output interval (convergence steps)",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nDataOutInternal,                "Data output interval (internal steps)",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nDataOutCInternal,               "Data output interval (internal convergence)");
                    DNDS_FIELD(nRestartOut,                     "Restart output interval (outer steps)");
                    DNDS_FIELD(nRestartOutC,                    "Restart output interval (convergence steps)");
                    DNDS_FIELD(nRestartOutInternal,             "Restart output interval (internal steps)");
                    DNDS_FIELD(nRestartOutCInternal,            "Restart output interval (internal convergence)");
                    DNDS_FIELD(nTimeAverageOut,                 "Time-average output interval (outer steps)");
                    DNDS_FIELD(nTimeAverageOutC,                "Time-average output interval (convergence steps)");
                    DNDS_FIELD(tDataOut,                        "Output data at simulation time interval");
                    DNDS_FIELD(lazyCoverDataOutput,             "Overwrite previous data output files");
                    DNDS_FIELD(useCollectiveTimer,              "Use collective MPI timer for profiling");
                }
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
                DNDS_DECLARE_CONFIG(ImplicitCFLControl)
                {
                    DNDS_FIELD(CFL,                    "CFL number for implicit time stepping",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(nForceLocalStartStep,   "Step to force local time stepping",
                               DNDS::Config::range(0));
                    DNDS_FIELD(nCFLRampStart,          "Step to begin CFL ramping",
                               DNDS::Config::range(0));
                    DNDS_FIELD(nCFLRampLength,         "Number of steps for CFL ramp",
                               DNDS::Config::range(1));
                    DNDS_FIELD(CFLRampEnd,             "CFL value at end of ramp");
                    DNDS_FIELD(useLocalDt,             "Use local (vs uniform) time step");
                    DNDS_FIELD(nSmoothDTau,            "Smoothing passes for local dTau",
                               DNDS::Config::range(0));
                    DNDS_FIELD(RANSRelax,              "RANS equation under-relaxation factor",
                               DNDS::Config::range(0.0, 1.0));
                }
            } implicitCFLControl;

            struct ConvergenceControl
            {
                int nTimeStepInternal = 20;
                int nTimeStepInternalMin = 5;
                int nAnchorUpdate = 1;
                int nAnchorUpdateStart = 0;
                real rhsThresholdInternal = 1e-10;
                real res_base = 0;
                int resBaseType = 0;        // 1 to use rhs as base in unsteady
                int mergeMultiResidual = 0; // for HM3, merge stage residuals
                int normOrd = 1;            // use LN norm
                bool useVolWiseResidual = false;
                bool useCLDriver = false;
                DNDS_DECLARE_CONFIG(ConvergenceControl)
                {
                    DNDS_FIELD(nTimeStepInternal,       "Max internal iterations per time step",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nTimeStepInternalMin,    "Min internal iterations per time step",
                               DNDS::Config::range(0));
                    DNDS_FIELD(nAnchorUpdate,           "Anchor update frequency",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nAnchorUpdateStart,      "Step to start anchor updates",
                               DNDS::Config::range(0));
                    DNDS_FIELD(rhsThresholdInternal,    "RHS convergence threshold for internal loop",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(res_base,                "Residual base value (0=auto)");
                    DNDS_FIELD(resBaseType,             "Residual base type: 0=default, 1=rhs-based for unsteady");
                    DNDS_FIELD(mergeMultiResidual,      "Merge stage residuals for multi-stage: 0=off");
                    DNDS_FIELD(normOrd,                 "Residual norm order (1=L1, 2=L2, etc.)",
                               DNDS::Config::range(1));
                    DNDS_FIELD(useVolWiseResidual,      "Volume-weighted residual");
                    DNDS_FIELD(useCLDriver,             "Enable CL-driven AoA adaptation");
                }
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
                int meshReorderCells = 0; // 0: natural; 1: reorder
                int meshBuildWallDist = 0;
                Geom::UnstructuredMesh::WallDistOptions meshWallDistOptions;

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
                bool hdfCollOnData = false;
                bool hdfCollOnMeta = true;
                bool outVolumeData = true;
                bool outBndData = false;

                std::vector<std::string> outCellScalarNames{};

                bool serializerSaveURec = false;

                bool allowAsyncPrintData = false;

                int rectifyNearPlane = 0; // 1: x 2: y 4: z
                real rectifyNearPlaneThres = 1e-10;

                Serializer::SerializerFactory restartWriter;
                Serializer::SerializerFactory meshPartitionedWriter;
                std::string meshPartitionedReaderType = "JSON";

                const std::string &getOutLogName()
                {
                    return outLogName.empty() ? outPltName : outLogName;
                }

                const std::string &getOutRestartName()
                {
                    return outRestartName.empty() ? outPltName : outRestartName;
                }

                DNDS_DECLARE_CONFIG(DataIOControl)
                {
                    DNDS_FIELD(uniqueStamps,                "Use unique output stamps per run");
                    DNDS_FIELD(meshRotZ,                    "Mesh rotation around Z axis (degrees)");
                    DNDS_FIELD(meshScale,                   "Mesh coordinate scaling factor",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(meshElevation,               "Mesh order elevation: 0=none, 1=O1->O2");
                    DNDS_FIELD(meshElevationInternalSmoother,"Elevation smoother: 0=local, 1=coupled");
                    DNDS_FIELD(meshElevationIter,           "Elevation smoothing iterations");
                    DNDS_FIELD(meshElevationNSearch,        "Elevation RBF neighbor search count",
                               DNDS::Config::range(1));
                    DNDS_FIELD(meshElevationRBFRadius,      "Elevation RBF support radius",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(meshElevationRBFPower,       "Elevation RBF power parameter");
                    DNDS_FIELD(meshElevationRBFKernel,      "Elevation RBF kernel type");
                    DNDS_FIELD(meshElevationMaxIncludedAngle,"Max included angle for elevation (degrees)");
                    DNDS_FIELD(meshElevationRefDWall,       "Reference wall distance for elevation",
                               DNDS::Config::range(0.0));
                    DNDS_FIELD(meshElevationBoundaryMode,   "Elevation boundary: 0=wall only, 1=wall+inviscid");
                    DNDS_FIELD(meshDirectBisect,            "Mesh bisection passes",
                               DNDS::Config::range(0));
                    DNDS_FIELD(meshReorderCells,            "Reorder cells: 0=natural, 1=reorder");
                    DNDS_FIELD(meshBuildWallDist,           "Build wall distance field: 0=no, 1=yes");
                    DNDS_FIELD(meshWallDistOptions,         "Wall distance computation options");
                    DNDS_FIELD(meshFormat,                  "Mesh format code");
                    DNDS_FIELD(meshPartitionOptions,        "Mesh partitioning options");
                    DNDS_FIELD(meshFile,                    "Input mesh file path");
                    DNDS_FIELD(outPltName,                  "Output plot file base name");
                    DNDS_FIELD(outLogName,                  "Output log file base name (empty=use outPltName)");
                    DNDS_FIELD(outRestartName,              "Output restart file base name (empty=use outPltName)");
                    DNDS_FIELD(outPltMode,                  "Output mode: 0=serial, 1=distributed");
                    DNDS_FIELD(readMeshMode,                "Read mesh mode: 0=serial CGNS, 1=distributed JSON");
                    DNDS_FIELD(outPltTecplotFormat,         "Output in Tecplot format");
                    DNDS_FIELD(outPltVTKFormat,             "Output in VTK XML format");
                    DNDS_FIELD(outPltVTKHDFFormat,          "Output in VTK HDF format");
                    DNDS_FIELD(outAtPointData,              "Output point-interpolated data");
                    DNDS_FIELD(outAtCellData,               "Output cell-centered data");
                    DNDS_FIELD(nASCIIPrecision,             "ASCII output floating-point precision",
                               DNDS::Config::range(1));
                    DNDS_FIELD(vtuFloatEncodeMode,          "VTU float encoding: binary or ascii",
                               DNDS::Config::enum_values({"binary", "ascii"}));
                    DNDS_FIELD(hdfChunkSize,                "HDF5 chunk size for output",
                               DNDS::Config::range(0));
                    DNDS_FIELD(hdfDeflateLevel,             "HDF5 deflate compression level",
                               DNDS::Config::range(0, 9));
                    DNDS_FIELD(hdfCollOnMeta,               "HDF5 collective I/O on metadata");
                    DNDS_FIELD(hdfCollOnData,               "HDF5 collective I/O on data arrays");
                    DNDS_FIELD(outVolumeData,               "Output volume data");
                    DNDS_FIELD(outBndData,                  "Output boundary data");
                    DNDS_FIELD(outCellScalarNames,          "Additional cell scalar names to output");
                    DNDS_FIELD(serializerSaveURec,          "Save reconstruction in restart");
                    DNDS_FIELD(allowAsyncPrintData,         "Allow asynchronous data output");
                    DNDS_FIELD(rectifyNearPlane,            "Rectify nodes near planes: bitmask 1=x,2=y,4=z");
                    DNDS_FIELD(rectifyNearPlaneThres,       "Threshold for near-plane rectification",
                               DNDS::Config::range(0.0));
                    config.field_section(&T::restartWriter,         "restartWriter",         "Restart file serializer settings");
                    config.field_section(&T::meshPartitionedWriter, "meshPartitionedWriter", "Partitioned mesh serializer settings");
                    DNDS_FIELD(meshPartitionedReaderType,   "Partitioned mesh reader type",
                               DNDS::Config::enum_values({"JSON", "H5"}));
                }
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
                int limiterProcedure = 0;
                int nPartialLimiterStart = INT_MAX;
                int nPartialLimiterStartLocal = INT_MAX;
                bool preserveLimited = false;
                bool ppRecLimiterCompressToMean = true;

                DNDS_DECLARE_CONFIG(LimiterControl)
                {
                    DNDS_FIELD(useLimiter,                 "Enable slope limiter");
                    DNDS_FIELD(usePPRecLimiter,            "Enable positivity-preserving reconstruction limiter");
                    DNDS_FIELD(useViscousLimited,          "Apply limiter to viscous reconstruction");
                    DNDS_FIELD(smoothIndicatorProcedure,   "Smooth indicator procedure index");
                    DNDS_FIELD(limiterProcedure,           "Limiter variant: 0=WBAP (V2), 1=CWBAP (V3)");
                    DNDS_FIELD(nPartialLimiterStart,       "Time step to begin partial limiting");
                    DNDS_FIELD(nPartialLimiterStartLocal,  "Time step to begin local partial limiting");
                    DNDS_FIELD(preserveLimited,            "Preserve limited reconstruction across steps");
                    DNDS_FIELD(ppRecLimiterCompressToMean, "PP limiter compresses toward cell mean");
                }
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
                struct CoarseGridLinearSolverControl
                {
                    int jacobiCode = 0; // 0 for jacobi, 1 for gs, 2 for ilu
                    int sgsIter = 0;
                    int gmresCode = 0;  // 0 for lusgs, 1 for gmres, 2 for lusgs started gmres
                    int gmresScale = 0; // 0 for no scaling, 1 use refU, 2 use mean value
                    int nGmresIter = 2;
                    int nSgsConsoleCheck = 100;
                    int nGmresConsoleCheck = 100;
                    int multiGridNIter = -1;
                    int multiGridNIterPost = 0;
                    int centralSmoothInputResidual = 0;

                    DNDS_DECLARE_CONFIG(CoarseGridLinearSolverControl)
                    {
                        DNDS_FIELD(jacobiCode,                  "Preconditioner: 0=jacobi, 1=GS, 2=ILU");
                        DNDS_FIELD(sgsIter,                     "SGS iteration count",
                                   DNDS::Config::range(0));
                        DNDS_FIELD(gmresCode,                   "Linear solver: 0=LUSGS, 1=GMRES, 2=LUSGS+GMRES");
                        DNDS_FIELD(gmresScale,                  "GMRES scaling: 0=none, 1=refU, 2=mean");
                        DNDS_FIELD(nGmresIter,                  "GMRES iterations",
                                   DNDS::Config::range(1));
                        DNDS_FIELD(nSgsConsoleCheck,            "Console check interval for SGS",
                                   DNDS::Config::range(1));
                        DNDS_FIELD(nGmresConsoleCheck,          "Console check interval for GMRES",
                                   DNDS::Config::range(1));
                        DNDS_FIELD(multiGridNIter,              "Multi-grid pre-smooth iterations (-1=auto)");
                        DNDS_FIELD(multiGridNIterPost,          "Multi-grid post-smooth iterations",
                                   DNDS::Config::range(0));
                        DNDS_FIELD(centralSmoothInputResidual,  "Smooth input residual on coarse grid");
                    }
                };
                std::map<std::string, CoarseGridLinearSolverControl> coarseGridLinearSolverControlList{
                    {"1", CoarseGridLinearSolverControl{}},
                    {"2", CoarseGridLinearSolverControl{}},
                };
                Direct::DirectPrecControl directPrecControl;

                DNDS_DECLARE_CONFIG(LinearSolverControl)
                {
                    DNDS_FIELD(jacobiCode,               "Preconditioner: 0=jacobi, 1=GS, 2=ILU");
                    DNDS_FIELD(sgsIter,                  "SGS iteration count",
                               DNDS::Config::range(0));
                    DNDS_FIELD(sgsWithRec,               "Couple SGS with reconstruction");
                    DNDS_FIELD(gmresCode,                "Linear solver: 0=LUSGS, 1=GMRES, 2=LUSGS+GMRES");
                    DNDS_FIELD(gmresScale,               "GMRES scaling: 0=none, 1=refU, 2=mean");
                    DNDS_FIELD(nGmresSpace,              "GMRES Krylov subspace size",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nGmresIter,               "GMRES outer iterations",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nSgsConsoleCheck,          "Console check interval for SGS",
                               DNDS::Config::range(1));
                    DNDS_FIELD(nGmresConsoleCheck,        "Console check interval for GMRES",
                               DNDS::Config::range(1));
                    DNDS_FIELD(initWithLastURecInc,       "Initialize GMRES with last uRec increment");
                    DNDS_FIELD(multiGridLP,               "Multi-grid levels (0=off)");
                    DNDS_FIELD(multiGridLPInnerNIter,     "Multi-grid inner iterations",
                               DNDS::Config::range(1));
                    DNDS_FIELD(multiGridLPStartIter,      "Iteration to enable multi-grid",
                               DNDS::Config::range(0));
                    DNDS_FIELD(multiGridLPInnerNSee,      "Multi-grid inner console see interval",
                               DNDS::Config::range(1));
                    config.template field_map_of<CoarseGridLinearSolverControl>(
                        &T::coarseGridLinearSolverControlList,
                        "coarseGridLinearSolverControlList",
                        "Per-level coarse grid linear solver settings");
                    config.field_section(&T::directPrecControl, "directPrecControl",
                                        "Direct preconditioner settings");
                }
            } linearSolverControl;

            struct RestartState
            {
                int iStep = -1;
                int iStepInternal = -1;
                int odeCodePrev = -1;
                std::string lastRestartFile = "";
                std::string otherRestartFile = "";
                std::vector<int> otherRestartStoreDim;
                DNDS_DECLARE_CONFIG(RestartState)
                {
                    DNDS_FIELD(iStep,                   "Restart step index");
                    DNDS_FIELD(iStepInternal,           "Restart internal step index");
                    DNDS_FIELD(odeCodePrev,             "Previous ODE code for restart");
                    DNDS_FIELD(lastRestartFile,         "Path to last restart file");
                    DNDS_FIELD(otherRestartFile,        "Path to alternate restart file");
                    DNDS_FIELD(otherRestartStoreDim,    "Dimension mapping for alternate restart");
                }
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

                DNDS_DECLARE_CONFIG(TimeAverageControl)
                {
                    DNDS_FIELD(enabled, "Enable time-averaging of solution fields");
                }
            } timeAverageControl;

            struct Others
            {
                int nFreezePassiveInner = 0;
                int axisSymmetric = 0;
                bool printRecMatrix = false;
                Serializer::SerializerFactory recMatrixWriter;

                DNDS_DECLARE_CONFIG(Others)
                {
                    DNDS_FIELD(nFreezePassiveInner,  "Freeze passive scalars for N inner steps",
                               DNDS::Config::range(0));
                    DNDS_FIELD(axisSymmetric,        "Axisymmetric mode: 0=off");
                    DNDS_FIELD(printRecMatrix,       "Print reconstruction matrix to file");
                    config.field_section(&T::recMatrixWriter, "recMatrixWriter",
                                        "Serializer for reconstruction matrix output");
                }
            } others;

            EulerEvaluatorSettings<model> eulerSettings;
            CFV::VRSettings vfvSettings;
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

                // In-place deserialization preserves _nVars set by Configuration(nVars).
                if (read)
                    from_json(jsonObj.at("eulerSettings"), eulerSettings);
                else
                    jsonObj["eulerSettings"] = eulerSettings;
                __DNDS__json_to_config(vfvSettings);
                __DNDS__json_to_config(bcSettings);
                __DNDS__json_to_config(bcNameMapping);
                if (read)
                {
                    DNDS_assert(bcSettings.is_array());
                }

                // TODO: BC settings
            }

            Configuration()
            {
            }

            Configuration(int nVars)
                : eulerSettings(nVars), vfvSettings(gDim)
            {
                bcSettings = BoundaryHandler<model>(nVars);
            }

        } config = Configuration{};

    public:
        EulerSolver(const MPIInfo &nmpi, int n_nVars = getNVars(model)) : nVars(n_nVars), mpi(nmpi)
        {
            if (getNVars(model) == DynamicSize)
                DNDS_assert_info(nVars >= getDim_Fixed(model) + 2, "nVars too small");
            else
                DNDS_assert_info(nVars == getNVars(model), "do not change the nVars for this model");
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
                    try
                    {
                        std::string valString =
                            fmt::format(R"({{
    "__val_entry": {}
    }})",
                                        overwriteValues[i]);
                        auto valDoc = nlohmann::ordered_json::parse(valString, nullptr, true, true);
                        if (mpi.rank == 0)
                            log() << "JSON: overwrite key: " << key << std::endl
                                  << "JSON: overwrite val: " << valDoc["__val_entry"] << std::endl;
                        gSetting[key] = valDoc["__val_entry"];
                    }
                    catch (const nlohmann::ordered_json::parse_error &e)
                    {
                        try
                        {
                            std::string valString =
                                fmt::format(R"({{
"__val_entry": "{}"
}})",
                                            overwriteValues[i]);
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
                    catch (const std::exception &e)
                    {
                        std::cerr << e.what() << "\n";
                        std::cerr << overwriteValues[i] << "\n";
                        DNDS_assert(false);
                    }
                }
                config.ReadWriteJson(gSetting, nVars, read);
                // create from json the pBCHandler
                pBCHandler = std::make_shared<BoundaryHandler<model>>(nVars);
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
                config.vfvSettings = static_cast<const CFV::VRSettings &>(vfv->getSettings());
            if (pEval)
                config.eulerSettings = pEval->settings;
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
                gSetting["___Commit_ID"] = GetSetVersionName();
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
        void ReadRestart(std::string fname);

        void ReadRestartOtherSolver(std::string fname, const std::vector<int> &dimStore);

        void PrintRestart(std::string fname);

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

        void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name) // currently not using
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            u.WriteSerialize(serializerP, "meanValue");

            serializerP->GoToPath(cwd);

            nlohmann::ordered_json configJson;
            config.ReadWriteJson(configJson, nVars, false);
            serializerP->WriteString("lastConfig", configJson.dump());

            if (config.dataIOControl.serializerSaveURec)
            {
                serializerP->WriteInt("hasReconstructionValue", 1);
                uRec.WriteSerialize(serializerP, "recValue");
            }
            else
                serializerP->WriteInt("hasReconstructionValue", 0);
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
            // std::vector<std::string> logfileOutputTitles{
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

        std::tuple<std::unique_ptr<CsvLog>, tLogSimpleDIValueMap> LogErrInitialize()
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

            // std::string logErrFileName = config.dataIOControl.getOutLogName() + "_" + output_stamp + ".log";
            // std::filesystem::path outFile{logErrFileName};
            // std::filesystem::create_directories(outFile.parent_path() / ".");
            // auto pOs = std::make_unique<std::ofstream>(logErrFileName);
            // DNDS_assert_info(*pOs, "logErr file [" + logErrFileName + "] did not open");

            return std::make_tuple(std::make_unique<DNDS::CsvLog>(
                                       realNames,
                                       config.dataIOControl.getOutLogName() + "_" + output_stamp,
                                       ".log",
                                       100000),
                                   v_map);
        }

        void RunImplicitEuler();

        struct RunningEnvironment
        {
            ssp<EulerEvaluator<model>> pEval;
            std::tuple<std::unique_ptr<CsvLog>, tLogSimpleDIValueMap> logErr;
            ssp<ODE::ImplicitDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>> ode;
            std::unique_ptr<tGMRES_u> gmres;
            std::unique_ptr<tGMRES_uRec> gmresRec;
            std::unique_ptr<tPCG_uRec> pcgRec;
            std::unique_ptr<tPCG_uRec> pcgRec1;
            double tstart = 0;
            double tstartInternal = 0;
            std::map<std::string, ScalarStatistics> tInternalStats;
            int stepCount = 0;
            Eigen::VectorFMTSafe<real, -1> resBaseC;
            Eigen::VectorFMTSafe<real, -1> resBaseCInternal;
            real tSimu = 0;
            real tAverage = 0;
            real nextTout = 0;
            int nextStepOut = -1;
            int nextStepOutC = -1;
            int nextStepRestart = -1;
            int nextStepRestartC = -1;
            int nextStepOutAverage = -1;
            int nextStepOutAverageC = -1;

            real CFLNow = 0;
            bool ifOutT = false;
            real curDtMin = 0;
            real curDtImplicit = 0;
            std::vector<real> curDtImplicitHistory;
            int step = 0;
            int iterAll = 0;
            bool gradIsZero = true;

            index nLimBeta = 0;
            index nLimAlpha = 0;
            real minAlpha = 1;
            real minBeta = 1;
            index nLimInc = 0;
            real alphaMinInc = 1;

            int dtIncreaseCounter = 0;

            tAdditionalCellScalarList addOutList;

#define DNDS_EULERSOLVER_RUNNINGENV_GET_REF(name) auto &name = runningEnvironment.name

#define DNDS_EULERSOLVER_RUNNINGENV_GET_REF_LIST               \
    auto &eval = *runningEnvironment.pEval;                    \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(logErr);               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(ode);                  \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(gmres);                \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(gmresRec);             \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(pcgRec);               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(pcgRec1);              \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(tstart);               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(tstartInternal);       \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(tInternalStats);       \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(stepCount);            \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(resBaseC);             \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(resBaseCInternal);     \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(tSimu);                \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(tAverage);             \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nextTout);             \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nextStepOut);          \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nextStepOutC);         \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nextStepRestart);      \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nextStepRestartC);     \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nextStepOutAverage);   \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nextStepOutAverageC);  \
                                                               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(CFLNow);               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(ifOutT);               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(curDtMin);             \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(curDtImplicit);        \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(curDtImplicitHistory); \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(step);                 \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(iterAll);              \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(gradIsZero);           \
                                                               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nLimBeta);             \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nLimAlpha);            \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(minAlpha);             \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(minBeta);              \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(nLimInc);              \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(alphaMinInc);          \
                                                               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(dtIncreaseCounter);    \
                                                               \
    DNDS_EULERSOLVER_RUNNINGENV_GET_REF(addOutList);

            RunningEnvironment() {};
        };
        void InitializeRunningEnvironment(RunningEnvironment &env);

        /**
         * \warning explicit inst not existent
         */
        void solveLinear(
            real alphaDiag, real t,
            TDof &cres, TDof &cx, TDof &cxInc, TRec &uRecC, TRec uRecIncC,
            JacobianDiagBlock<nVarsFixed> &JDC, tGMRES_u &gmres, int gridLevel);
        /**
         * \warning explicit inst not existent
         */
        void doPrecondition(real alphaDiag, real t,
                            TDof &crhs, TDof &cx, TDof &cxInc, TDof &uTemp,
                            JacobianDiagBlock<nVarsFixed> &JDC, TU &sgsRes, bool &inputIsZero, bool &hasLUDone, int gridLevel);

        bool functor_fstop(int iter, ArrayDOFV<nVarsFixed> &cres, int iStep, RunningEnvironment &env);
        bool functor_fmainloop(RunningEnvironment &env);

        auto getMPI() const { return mpi; }
        auto getMesh() const { return mesh; }
        auto getVFV() const { return vfv; }
    };
}

#define DNDS_EULERSOLVER_INS_EXTERN(model, ext)                             \
    namespace DNDS::Euler                                                   \
    {                                                                       \
        ext template void EulerSolver<model>::RunImplicitEuler();           \
        ext template void EulerSolver<model>::InitializeRunningEnvironment( \
            EulerSolver<model>::RunningEnvironment &env);                   \
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
        ext template void EulerSolver<model>::PrintRestart(           \
            std::string fname);                                       \
        ext template void EulerSolver<model>::ReadRestart(            \
            std::string fname);                                       \
        ext template void EulerSolver<model>::ReadRestartOtherSolver( \
            std::string fname, const std::vector<int> &dimStore);     \
    }

DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_2D, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_SA, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_2EQ, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_3D, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_SA_3D, extern);
DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN(NS_2EQ_3D, extern);

#define DNDS_EULERSOLVER_INIT_INS_EXTERN(model, ext)                                    \
    namespace DNDS::Euler                                                               \
    {                                                                                   \
        ext template void EulerSolver<model>::ReadMeshAndInitialize();                  \
        ext template bool EulerSolver<model>::functor_fstop(                            \
            int iter, ArrayDOFV<nVarsFixed> &cres, int iStep, RunningEnvironment &env); \
        ext template bool EulerSolver<model>::functor_fmainloop(                        \
            RunningEnvironment &env);                                                   \
    }

DNDS_EULERSOLVER_INIT_INS_EXTERN(NS, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_2D, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_SA, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_2EQ, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_3D, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_SA_3D, extern);
DNDS_EULERSOLVER_INIT_INS_EXTERN(NS_2EQ_3D, extern);