/**
 * @file EulerSolver.hpp
 * @brief Top-level solver orchestrator for compressible Navier-Stokes / Euler simulations.
 *
 * Provides the EulerSolver class template which owns and coordinates all solver
 * components: mesh infrastructure, variational reconstruction, the EulerEvaluator
 * spatial discretization, DOF arrays, ODE integrators, linear solvers (LU-SGS / GMRES),
 * and I/O subsystems.
 *
 * Responsibilities include:
 * - JSON-based configuration loading, merging, and validation
 * - Mesh reading (serial CGNS or distributed), partitioning, order elevation, and bisection
 * - Solver initialization (evaluator, DOF allocation, restart loading)
 * - Implicit time-marching loop (dual time stepping, CFL ramping, convergence monitoring)
 * - VTK/HDF5/Tecplot data output and restart I/O
 * - Time-averaging for unsteady statistics
 *
 * Supported model specializations (via EulerModel enum):
 *   NS, NS_2D, NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D, NS_3D
 *
 * @see EulerEvaluator.hpp  Spatial discretization and flux evaluation
 * @see Solver/ODE.hpp      ODE integrator interface
 * @see Solver/Linear.hpp   GMRES and preconditioner interface
 */
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
#include "DNDS/Serializer/JsonUtil.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Serializer/SerializerFactory.hpp"
#include "DNDS/CsvLog.hpp"
#include "DNDS/ObjectPool.hpp"
#include "Solver/Linear.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "Gas.hpp"
#include "EulerEvaluator.hpp"
#include "EulerBC.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

#include "DNDS/Serializer/JsonUtil.hpp"

#include "Solver/ODE.hpp"
#include "Solver/Linear.hpp"

namespace DNDS::Euler
{

    /**
     * @brief Top-level solver orchestrator for compressible Navier-Stokes / Euler equations.
     *
     * Owns the complete solver pipeline: mesh, variational reconstruction, spatial
     * evaluator, DOF arrays, Jacobian data, ODE integrator, linear solvers, and I/O.
     * Provides the main time-marching loop (RunImplicitEuler) which drives steady-state
     * or unsteady simulations with implicit dual-time stepping, CFL ramping, and
     * convergence monitoring.
     *
     * @tparam model  EulerModel enum selecting the equation set and spatial dimension
     *                (NS, NS_2D, NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D, NS_3D).
     */
    template <EulerModel model>
    class EulerSolver
    {
        int nVars = getNVars(model); ///< Runtime number of conserved variables.

    public:
        typedef EulerEvaluator<model> TEval;             ///< Evaluator type for this model.
        static const int nVarsFixed = TEval::nVarsFixed; ///< Compile-time number of conserved variables.

        static const int dim = TEval::dim; ///< Spatial dimension (2 or 3).
        // static const int gdim = TEval::gdim;
        static const int gDim = TEval::gDim; ///< Geometric dimension of the mesh.
        static const int I4 = TEval::I4;     ///< Energy equation index (= dim + 1).

        typedef typename TEval::TU TU;                 ///< Conservative variable vector type.
        typedef typename TEval::TDiffU TDiffU;         ///< Gradient of conserved variables type.
        typedef typename TEval::TJacobianU TJacobianU; ///< Flux Jacobian matrix type.
        typedef typename TEval::TVec TVec;             ///< Spatial vector type.
        typedef typename TEval::TMat TMat;             ///< Spatial matrix type.
        typedef typename TEval::TDof TDof;             ///< Cell-centered DOF array type.
        typedef typename TEval::TRec TRec;             ///< Reconstruction coefficient array type.
        typedef typename TEval::TScalar TScalar;       ///< Scalar reconstruction coefficient array type.
        typedef typename TEval::TVFV TVFV;             ///< Variational reconstruction type.
        typedef typename TEval::TpVFV TpVFV;           ///< Shared pointer to VFV type.

        using tGMRES_u = Linear::GMRES_LeftPreconditioned<TDof>;                                      ///< GMRES solver type for conservative DOFs.
        using tGMRES_uRec = Linear::GMRES_LeftPreconditioned<TRec>;                                   ///< GMRES solver type for reconstruction coefficients.
        using tPCG_uRec = Linear::PCG_PreconditionedRes<TRec, Eigen::Array<real, 1, Eigen::Dynamic>>; ///< PCG solver type for reconstruction.

    private:
        MPIInfo mpi;                                           ///< MPI communicator and rank information.
        ssp<Geom::UnstructuredMesh> mesh, meshBnd;             ///< Volume mesh and (optional) boundary surface mesh.
        TpVFV vfv;                                             // ! gDim -> 3 for intellisense          ///< Variational reconstruction object.
        ssp<Geom::UnstructuredMeshSerialRW> reader, readerBnd; ///< Mesh reader for volume and boundary meshes.
        ssp<EulerEvaluator<model>> pEval;                      ///< Spatial evaluator instance.

        ArrayDOFV<nVarsFixed> u, uIncBufODE, wAveraged, uAveraged;                                                    ///< DOF arrays: solution, ODE increment buffer, time-averaged fields.
        ObjectPool<ArrayDOFV<nVarsFixed>> uPool;                                                                      ///< Object pool for temporary DOF arrays (used by ODE integrators).
        ArrayRECV<nVarsFixed> uRec, uRecLimited, uRecNew, uRecNew1, uRecOld, uRec1, uRecInc, uRecInc1, uRecB, uRecB1; ///< Reconstruction arrays (current, limited, new, old, increment, etc.).
        JacobianDiagBlock<nVarsFixed> JD, JD1, JDTmp, JSource, JSource1, JSourceTmp;                                  ///< Diagonal Jacobian blocks for implicit methods.
        ssp<JacobianLocalLU<nVarsFixed>> JLocalLU;                                                                    ///< Local LU factorization for direct preconditioner.
        ArrayDOFV<1> alphaPP, alphaPP1, betaPP, betaPP1, alphaPP_tmp, dTauTmp;                                        ///< Positivity-preserving limiter scalars and time-step buffer.

        int nOUTS = {-1};   ///< Number of output scalars per cell in volume output.
        int nOUTSPoint{-1}; ///< Number of output scalars per node in point output.
        int nOUTSBnd{-1};   ///< Number of output scalars per face in boundary output.
        // rho u v w p T M ifUseLimiter RHS
        ssp<ArrayEigenVector<Eigen::Dynamic>> outDist;                                    ///< Distributed cell output array.
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerial;                                  ///< Serial (gathered) cell output array.
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTrans; ///< Transformer for distributed-to-serial cell output.

        ssp<ArrayEigenVector<Eigen::Dynamic>> outDistPoint;                                    ///< Distributed node output array.
        ssp<ArrayEigenVector<Eigen::Dynamic>> outGhostPoint;                                   ///< Ghost-node output array.
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerialPoint;                                  ///< Serial (gathered) node output array.
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTransPoint; ///< Transformer for distributed-to-serial node output.
        ArrayPair<ArrayEigenVector<Eigen::Dynamic>> outDistPointPair;                          ///< Array pair for async node output.
        static const int maxOutFutures{3};                                                     ///< Maximum number of concurrent async output futures.
        std::mutex outArraysMutex;                                                             ///< Mutex protecting output arrays during async writes.
        std::array<std::future<void>, maxOutFutures> outFuture;                                ///< Futures for async volume output. Mind the order, relies on the arrays and the mutex.

        ssp<ArrayEigenVector<Eigen::Dynamic>> outDistBnd; ///< Distributed boundary output array.
        // ssp<ArrayEigenVector<Eigen::Dynamic>> outGhostBnd;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerialBnd;                                  ///< Serial (gathered) boundary output array.
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTransBnd; ///< Transformer for distributed-to-serial boundary output.
        // ArrayPair<ArrayEigenVector<Eigen::Dynamic>> outDistBndPair;
        std::mutex outBndArraysMutex;                              ///< Mutex protecting boundary output arrays during async writes.
        std::array<std::future<void>, maxOutFutures> outBndFuture; ///< Futures for async boundary output. Mind the order, relies on the arrays and the mutex.
        std::future<void> outSeqFuture;                            ///< Future for sequential (non-parallel) output operations.

        // std::vector<uint32_t> ifUseLimiter;
        CFV::tScalarPair ifUseLimiter; ///< Per-cell flag indicating whether the limiter was active.

        ssp<BoundaryHandler<model>> pBCHandler; ///< Boundary condition handler (shared with evaluator).

    public:
        nlohmann::ordered_json gSetting; ///< Full JSON configuration (read from file, may be modified at runtime).
        std::string output_stamp = "";   ///< Unique stamp appended to output filenames for this run.

        /**
         * @brief Complete solver configuration, serializable to/from JSON.
         *
         * Aggregates all sub-configurations: time marching, reconstruction, output,
         * CFL control, convergence, data I/O, boundary definitions, limiters, linear
         * solver, restart state, time averaging, evaluator settings, VFV settings,
         * and boundary condition definitions. Each sub-struct uses DNDS_DECLARE_CONFIG
         * for automatic JSON serialization.
         */
        struct Configuration
        {

            /**
             * @brief Time marching control parameters.
             *
             * Controls physical/pseudo time step sizes, ODE integrator selection,
             * positivity-preserving options, and dt ramping strategies.
             */
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
                    // clang-format off
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
                    // clang-format on
                }
                bool timeMarchIsTwoStage()
                {
                    return odeCode == 401 || (odeCode >= 411 && odeCode <= 413);
                }
            } timeMarchControl;

            /**
             * @brief Implicit reconstruction control parameters.
             *
             * Controls the iterative reconstruction solve within each time step:
             * explicit vs implicit reconstruction, linear solver (SOR or GMRES/PCG),
             * convergence thresholds, and gradient zeroing strategies.
             */
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
                    // clang-format off
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
                    // clang-format on
                }
            } implicitReconstructionControl;

            /**
             * @brief Output control parameters.
             *
             * Controls console output frequency and formatting, data file output
             * intervals (VTK/HDF5), restart checkpoint intervals, time-average output,
             * and logging precision.
             */
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
                    // clang-format off
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
                    // clang-format on
                }
            } outputControl;

            /**
             * @brief Implicit CFL number control parameters.
             *
             * Controls the CFL number, local vs global time stepping, CFL ramping
             * schedule, local dTau smoothing, and RANS under-relaxation.
             */
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
                    // clang-format off
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
                    // clang-format on
                }
            } implicitCFLControl;

            /**
             * @brief Convergence monitoring parameters.
             *
             * Controls inner-loop iteration counts, convergence thresholds,
             * residual norm type, and CL-driver (lift-coefficient) adaptation.
             */
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
                    // clang-format off
                    DNDS_FIELD(nTimeStepInternal,       "Max internal iterations per time step (0 = unlimited)",
                               DNDS::Config::range(0));
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
                    // clang-format on
                }
            } convergenceControl;

            /**
             * @brief Data I/O control parameters.
             *
             * Controls mesh file paths and formats, mesh preprocessing (elevation,
             * bisection, wall distance), output formats (Tecplot, VTK, HDF5),
             * restart serialization, and coordinate transformations.
             */
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
                    // clang-format off
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
                    // clang-format on
                }
            } dataIOControl;

            /**
             * @brief Periodic boundary geometry definitions.
             *
             * Defines up to 3 periodic translation vectors and rotation parameters
             * for periodic boundary matching.
             */
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

                DNDS_DECLARE_CONFIG(BoundaryDefinition)
                {
                    // clang-format off
                    DNDS_FIELD(PeriodicTranslation1,          "Periodic translation vector for pair 1");
                    DNDS_FIELD(PeriodicTranslation2,          "Periodic translation vector for pair 2");
                    DNDS_FIELD(PeriodicTranslation3,          "Periodic translation vector for pair 3");
                    DNDS_FIELD(PeriodicRotationCent1,         "Rotation center for periodic pair 1");
                    DNDS_FIELD(PeriodicRotationCent2,         "Rotation center for periodic pair 2");
                    DNDS_FIELD(PeriodicRotationCent3,         "Rotation center for periodic pair 3");
                    DNDS_FIELD(PeriodicRotationEulerAngles1,  "Rotation Euler angles (deg) for periodic pair 1");
                    DNDS_FIELD(PeriodicRotationEulerAngles2,  "Rotation Euler angles (deg) for periodic pair 2");
                    DNDS_FIELD(PeriodicRotationEulerAngles3,  "Rotation Euler angles (deg) for periodic pair 3");
                    DNDS_FIELD(periodicTolerance,             "Tolerance for periodic node matching",
                               DNDS::Config::range(0.0));
                    // clang-format on
                }
            } boundaryDefinition;

            /**
             * @brief Slope limiter control parameters.
             *
             * Controls whether limiters are active, which limiter variant to use
             * (WBAP, CWBAP), positivity-preserving reconstruction limiting, and
             * partial/preserved limiting strategies.
             */
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
                    // clang-format off
                    DNDS_FIELD(useLimiter,                 "Enable slope limiter");
                    DNDS_FIELD(usePPRecLimiter,            "Enable positivity-preserving reconstruction limiter");
                    DNDS_FIELD(useViscousLimited,          "Apply limiter to viscous reconstruction");
                    DNDS_FIELD(smoothIndicatorProcedure,   "Smooth indicator procedure index");
                    DNDS_FIELD(limiterProcedure,           "Limiter variant: 0=WBAP (V2), 1=CWBAP (V3)");
                    DNDS_FIELD(nPartialLimiterStart,       "Time step to begin partial limiting");
                    DNDS_FIELD(nPartialLimiterStartLocal,  "Time step to begin local partial limiting");
                    DNDS_FIELD(preserveLimited,            "Preserve limited reconstruction across steps");
                    DNDS_FIELD(ppRecLimiterCompressToMean, "PP limiter compresses toward cell mean");
                    // clang-format on
                }
            } limiterControl;

            /**
             * @brief Linear solver control parameters.
             *
             * Controls the implicit linear solver: preconditioner type (Jacobi/GS/ILU),
             * SGS iterations, GMRES settings, multi-grid options, and per-level
             * coarse-grid solver configurations.
             */
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
                        // clang-format off
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
                        // clang-format on
                    }
                };
                std::map<std::string, CoarseGridLinearSolverControl> coarseGridLinearSolverControlList{
                    {"1", CoarseGridLinearSolverControl{}},
                    {"2", CoarseGridLinearSolverControl{}},
                };
                Direct::DirectPrecControl directPrecControl;

                DNDS_DECLARE_CONFIG(LinearSolverControl)
                {
                    // clang-format off
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
                    // clang-format on
                }
            } linearSolverControl;

            /**
             * @brief Restart checkpoint state.
             *
             * Records the step indices and file paths for restart continuation,
             * including support for reading restart files from a different solver
             * (with dimension remapping).
             */
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
                    // clang-format off
                    DNDS_FIELD(iStep,                   "Restart step index");
                    DNDS_FIELD(iStepInternal,           "Restart internal step index");
                    DNDS_FIELD(odeCodePrev,             "Previous ODE code for restart");
                    DNDS_FIELD(lastRestartFile,         "Path to last restart file");
                    DNDS_FIELD(otherRestartFile,        "Path to alternate restart file");
                    DNDS_FIELD(otherRestartStoreDim,    "Dimension mapping for alternate restart");
                    // clang-format on
                }
                RestartState()
                {
                    otherRestartStoreDim.resize(1);
                    for (int i = 0; i < otherRestartStoreDim.size(); i++)
                        otherRestartStoreDim[i] = i;
                }
            } restartState;

            /// @brief Time-averaging control for unsteady simulations.
            struct TimeAverageControl
            {
                bool enabled = false;

                DNDS_DECLARE_CONFIG(TimeAverageControl)
                {
                    // clang-format off
                    DNDS_FIELD(enabled, "Enable time-averaging of solution fields");
                    // clang-format on
                }
            } timeAverageControl;

            /// @brief Miscellaneous solver options (axisymmetric mode, passive scalar freezing, rec matrix output).
            struct Others
            {
                int nFreezePassiveInner = 0;
                int axisSymmetric = 0;
                bool printRecMatrix = false;
                Serializer::SerializerFactory recMatrixWriter;

                DNDS_DECLARE_CONFIG(Others)
                {
                    // clang-format off
                    DNDS_FIELD(nFreezePassiveInner,  "Freeze passive scalars for N inner steps",
                               DNDS::Config::range(0));
                    DNDS_FIELD(axisSymmetric,        "Axisymmetric mode: 0=off");
                    DNDS_FIELD(printRecMatrix,       "Print reconstruction matrix to file");
                    config.field_section(&T::recMatrixWriter, "recMatrixWriter",
                                        "Serializer for reconstruction matrix output");
                    // clang-format on
                }
            } others;

            EulerEvaluatorSettings<model> eulerSettings;                         ///< Physics settings passed to the EulerEvaluator.
            CFV::VRSettings vfvSettings;                                         ///< Variational reconstruction settings.
            nlohmann::ordered_json bcSettings = nlohmann::ordered_json::array(); ///< Boundary condition definitions (JSON array).
            std::map<std::string, std::string> bcNameMapping;                    ///< Mapping from mesh BC names to solver BC type names.

            DNDS_DECLARE_CONFIG(Configuration)
            {
                // clang-format off
                config.field_section(&T::timeMarchControl,               "timeMarchControl",               "Time marching settings");
                config.field_section(&T::implicitReconstructionControl,   "implicitReconstructionControl",   "Implicit reconstruction settings");
                config.field_section(&T::outputControl,                  "outputControl",                  "Output settings");
                config.field_section(&T::implicitCFLControl,             "implicitCFLControl",             "Implicit CFL settings");
                config.field_section(&T::convergenceControl,             "convergenceControl",             "Convergence monitoring settings");
                config.field_section(&T::dataIOControl,                  "dataIOControl",                  "Data I/O and restart settings");
                config.field_section(&T::boundaryDefinition,             "boundaryDefinition",             "Periodic boundary geometry");
                config.field_section(&T::limiterControl,                 "limiterControl",                 "Slope limiter settings");
                config.field_section(&T::linearSolverControl,            "linearSolverControl",            "Linear solver settings");
                config.field_section(&T::timeAverageControl,             "timeAverageControl",             "Time averaging settings");
                config.field_section(&T::others,                         "others",                         "Miscellaneous settings");
                config.field_section(&T::restartState,                   "restartState",                   "Restart checkpoint state");
                config.field_section(&T::eulerSettings,                  "eulerSettings",                  "Euler evaluator physics settings");
                config.field_section(&T::vfvSettings,                    "vfvSettings",                    "Variational reconstruction settings");
                config.field_json_schema(&T::bcSettings,                  "bcSettings",
                    "Boundary condition settings (per-BC array)",
                    []() { return bcSettingsSchema(); });
                DNDS_FIELD(bcNameMapping,                                "Boundary name to type mapping");

                config.check("bcSettings must be a JSON array", [](const T &s)
                {
                    return s.bcSettings.is_array();
                });
                // clang-format on
            }

            /// @brief Backward-compatible bidirectional JSON read/write.
            ///
            /// Delegates to the auto-generated from_json / to_json from
            /// DNDS_DECLARE_CONFIG.  Kept for call-site compatibility.
            void
            ReadWriteJson(nlohmann::ordered_json &jsonObj, int nVars, bool read)
            {
                (void)nVars; // nVars is already stored in eulerSettings via Configuration(nVars)
                if (read)
                    from_json(jsonObj, *this);
                else
                    to_json(jsonObj, *this);
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
        /**
         * @brief Construct an EulerSolver with MPI communicator information.
         *
         * Initializes the runtime variable count, output field sizes, and default
         * configuration. Does not read mesh or allocate solver arrays.
         *
         * @param nmpi    MPI communicator information.
         * @param n_nVars Runtime number of conserved variables (default from model).
         */
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

        /// @brief Destructor. Waits for all async output futures to complete.
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
         * @brief Load or write solver configuration from/to a JSON file.
         *
         * When read=true, parses the JSON file, optionally merges a patch file,
         * applies key/value overrides, populates the Configuration struct and
         * creates the boundary condition handler. When read=false, serializes the
         * current configuration to a JSON file.
         *
         * @param jsonName       Path to the JSON configuration file.
         * @param read           true=read from file, false=write to file.
         * @param jsonMergeName  Optional path to a JSON patch file to merge.
         * @param overwriteKeys  JSON pointer paths to override (e.g., "/timeMarchControl/CFL").
         * @param overwriteValues Values corresponding to overwriteKeys.
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

        /**
         * @brief Read the mesh and initialize the full solver pipeline.
         *
         * Performs the complete initialization sequence:
         * 1. Read mesh (serial CGNS or distributed)
         * 2. Apply coordinate transformations (rotation, scaling, rectification)
         * 3. Partition the mesh across MPI ranks (Metis/ParMetis)
         * 4. Apply order elevation (O1->O2) and mesh bisection
         * 5. Build wall distance field (if requested)
         * 6. Construct the variational reconstruction (VFV)
         * 7. Create the EulerEvaluator and initialize FV infrastructure
         * 8. Allocate DOF, reconstruction, and Jacobian arrays
         * 9. Load restart data (if configured)
         * 10. Set initial conditions (if not restarting)
         */
        void ReadMeshAndInitialize();

        /**
         * @brief Write the current configuration to a JSON log file.
         *
         * Extracts live settings from the VFV, evaluator, and BC handler,
         * serializes the full configuration to JSON, and writes it alongside
         * compile-time defines and commit information.
         *
         * @param updateCommit If true, include the git commit hash in the output.
         */
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
        /// @brief Read a restart file and populate u (and optionally uRec) from it.
        void ReadRestart(std::string fname);

        /// @brief Read a restart file from a different solver/model, remapping variable dimensions.
        void ReadRestartOtherSolver(std::string fname, const std::vector<int> &dimStore);

        /// @brief Write the current solution state to a restart file.
        void PrintRestart(std::string fname);

        using tAdditionalCellScalarList = tCellScalarList; ///< Type alias for additional output scalar list.

        /// @brief Output mode selector for PrintData.
        enum PrintDataMode
        {
            PrintDataLatest = 0,      ///< Output the current (latest) solution.
            PrintDataTimeAverage = 1, ///< Output the time-averaged solution.
        };

        /**
         * @brief Write solution data to VTK/HDF5/Tecplot output files.
         *
         * Gathers distributed data to serial (or writes distributed), converts
         * conservative to primitive variables, appends additional scalars (residual,
         * limiter flags), and writes the output in the configured format(s).
         *
         * @param fname                   Output file base name.
         * @param fnameSeries             Series file name (for VTK time series).
         * @param odeResidualF             Callback returning the ODE residual for each cell.
         * @param additionalCellScalars    Additional per-cell scalar fields to output.
         * @param eval                     Reference to the evaluator (for output field computation).
         * @param TSimu                    Current simulation time (-1 for steady).
         * @param mode                     PrintDataLatest or PrintDataTimeAverage.
         */
        void PrintData(const std::string &fname, const std::string &fnameSeries,
                       const tCellScalarFGet &odeResidualF,
                       tAdditionalCellScalarList &additionalCellScalars,
                       TEval &eval, real TSimu = -1.0, PrintDataMode mode = PrintDataLatest);

        /// @brief Serialize the solution to a SerializerBase (currently unused standalone path).
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

        /// @brief Fill a log value map entry for arithmetic types (scalars).
        template <class TVal>
        std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<TVal>>>
        FillLogValue(tLogSimpleDIValueMap &v_map, const std::string &name, TVal &&val)
        {
            v_map[name] = val;
        }

        /// @brief Fill a log value map entry for non-arithmetic types (vectors → per-component entries).
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

        /// @brief Initialize the CSV error logger and value map for convergence monitoring.
        /// @return Tuple of (CsvLog writer, value map with all column names initialized).
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

        /**
         * @brief Run the main implicit time-marching loop.
         *
         * Executes the outer time-step loop with configurable ODE integrators
         * (backward Euler, BDF2, SDIRK, etc.). Each step involves:
         * 1. CFL ramping and time-step computation
         * 2. Variational reconstruction (implicit or explicit)
         * 3. Slope limiting and positivity-preserving enforcement
         * 4. RHS evaluation
         * 5. Implicit linear solve (LU-SGS, GMRES, or direct)
         * 6. Solution update with increment compression
         * 7. Convergence monitoring and data output
         *
         * Supports steady-state convergence checks, CL-driver AoA adaptation,
         * time-averaging, and restart checkpointing.
         */
        void RunImplicitEuler();

        /**
         * @brief Mutable state bundle for the time-marching loop.
         *
         * Holds all transient state that persists across time steps within
         * RunImplicitEuler: evaluator, ODE integrator, linear solvers, timing
         * statistics, residual bases, convergence trackers, output counters,
         * and CFL/dt history. The DNDS_EULERSOLVER_RUNNINGENV_GET_REF_LIST
         * macro provides convenient local aliases for all members.
         */
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

            RunningEnvironment(){};
        };
        /// @brief Populate a RunningEnvironment with allocated solvers, loggers, and initial state.
        void InitializeRunningEnvironment(RunningEnvironment &env);

        /**
         * @brief Solve the implicit linear system at a given grid level.
         *
         * Dispatches to LU-SGS, GMRES, or LU-SGS-preconditioned GMRES depending on
         * configuration. Used within the ODE integrator's inner loop.
         *
         * @warning Explicit template instantiation does not exist; inlined only.
         */
        void solveLinear(
            real alphaDiag, real t,
            TDof &cres, TDof &cx, TDof &cxInc, TRec &uRecC, TRec uRecIncC,
            JacobianDiagBlock<nVarsFixed> &JDC, tGMRES_u &gmres, int gridLevel);
        /**
         * @brief Apply the preconditioner (SGS sweeps or ILU) to a right-hand side vector.
         *
         * @warning Explicit template instantiation does not exist; inlined only.
         */
        void doPrecondition(real alphaDiag, real t,
                            TDof &crhs, TDof &cx, TDof &cxInc, TDof &uTemp,
                            JacobianDiagBlock<nVarsFixed> &JDC, TU &sgsRes, bool &inputIsZero, bool &hasLUDone, int gridLevel);

        /// @brief Convergence/termination check functor called after each inner iteration.
        /// @return true if the inner loop should stop (converged or max iterations reached).
        bool functor_fstop(int iter, ArrayDOFV<nVarsFixed> &cres, int iStep, RunningEnvironment &env);
        /// @brief Main outer-loop functor: performs one full time step (reconstruction, RHS, linear solve, update).
        /// @return true if the outer loop should continue.
        bool functor_fmainloop(RunningEnvironment &env);

        /// @name Accessors
        /// @{
        auto getMPI() const { return mpi; }   ///< Get MPI communicator info.
        auto getMesh() const { return mesh; } ///< Get shared pointer to the mesh.
        auto getVFV() const { return vfv; }   ///< Get shared pointer to the VFV reconstruction.
        /// @}

        /// @name Test accessors (for unit testing the evaluator pipeline)
        /// @{
        auto &getU() { return u; }                       ///< Get mutable reference to the DOF array.
        auto &getURec() { return uRec; }                 ///< Get mutable reference to the reconstruction array.
        auto &getURecNew() { return uRecNew; }           ///< Get mutable reference to the new reconstruction array.
        auto &getURecLimited() { return uRecLimited; }   ///< Get mutable reference to the limited reconstruction array.
        auto &getEval() { return *pEval; }               ///< Get mutable reference to the evaluator.
        auto &getConfiguration() { return config; }      ///< Get mutable reference to the configuration.
        auto &getJSource() { return JSource; }           ///< Get mutable reference to the source Jacobian.
        auto &getBetaPP() { return betaPP; }             ///< Get mutable reference to the PP beta array.
        auto &getAlphaPP() { return alphaPP; }           ///< Get mutable reference to the PP alpha array.
        auto &getDTauTmp() { return dTauTmp; }           ///< Get mutable reference to the dTau temporary.
        auto &getIfUseLimiter() { return ifUseLimiter; } ///< Get mutable reference to the limiter flag array.
        auto &getBCHandler() { return pBCHandler; }      ///< Get mutable reference to the BC handler.
        /// @}
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