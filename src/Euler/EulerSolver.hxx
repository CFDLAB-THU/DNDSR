#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "EulerSolver.hpp"
#include "DNDS/EigenUtil.hpp"
#include "Solver/ODE.hpp"
#include "Solver/Linear.hpp"
#include "SpecialFields.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif
// #include "fmt/ranges.h"

namespace DNDS::Euler
{
    // /***************/ // IDE mode:
    // static const auto model = NS_SA;
    // template <>
    // void EulerSolver<model>::RunImplicitEuler()
    // /***************/ // IDE mode;
    static const auto model = NS_SA;
    DNDS_SWITCH_INTELLISENSE(
        // the real definition
        template <EulerModel model>
        ,
        // the intellisense friendly definition
        template <>
    )
    void EulerSolver<model>::RunImplicitEuler()
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        using namespace std::literals;

        auto runningEnvironment = RunningEnvironment();
        InitializeRunningEnvironment(runningEnvironment);
        DNDS_EULERSOLVER_RUNNINGENV_GET_REF_LIST

        DNDS_MPI_InsertCheck(mpi, "Implicit 1 nvars " + std::to_string(nVars));
        /*******************************************************/
        /*                 CHECK MESH                          */
        /*******************************************************/
        auto hashCoord = mesh->coords.hash();
        if (mpi.rank == 0)
        {
            log() << "Mesh coord hash is: [" << std::hex << hashCoord << std::dec << "]" << std::endl
                  << std::scientific;
        }
        if (config.timeMarchControl.partitionMeshOnly)
        {
            if (mpi.rank == 0)
            {
                log() << "Mesh Is not altered; partitioning done" << std::endl;
            }
            return;
        }

        /*******************************************************/
        /*              DIRECT FACTORIZATION                   */
        /*******************************************************/

        mesh->ObtainLocalFactFillOrdering(*eval.symLU, config.linearSolverControl.directPrecControl);
        mesh->ObtainSymmetricSymbolicFactorization(*eval.symLU, config.linearSolverControl.directPrecControl);
        if (config.linearSolverControl.jacobiCode == 2) // do lu on mean-value jacobian
        {
            DNDS_MAKE_SSP(JLocalLU, eval.symLU, nVars);
        }

        // fmt::print("pEval is {}", (void*)(pEval.get()));

        /*******************************************************/
        /*                 INITIALIZE U                        */
        /*******************************************************/

        eval.InitializeUDOF(u);
        if (config.timeAverageControl.enabled)
            wAveraged.setConstant(0.0);
        if (config.timeMarchControl.useRestart)
        {

            if (!config.restartState.lastRestartFile.empty())
            {
                DNDS_assert(config.restartState.iStep >= 1);
                ReadRestart(config.restartState.lastRestartFile);
            }
            if (!config.restartState.otherRestartFile.empty())
                ReadRestartOtherSolver(config.restartState.otherRestartFile, config.restartState.otherRestartStoreDim);
        }
        OutputPicker outputPicker;
        eval.InitializeOutputPicker(outputPicker, {u, uRec, betaPP, alphaPP});
        addOutList = outputPicker.getSubsetList(config.dataIOControl.outCellScalarNames);

        /*******************************************************/
        /*                 TEMPORARY Us                        */
        /*******************************************************/

        auto initUDOF = [&](ArrayDOFV<nVarsFixed> &uu)
        { vfv->BuildUDof(uu, nVars); };
        auto initUREC = [&](ArrayRECV<nVarsFixed> &uu)
        { vfv->BuildURec(uu, nVars); };

#define DNDS_EULER_SOLVER_GET_TEMP_UDOF(name)      \
    auto __p##name = uPool.getAllocInit(initUDOF); \
    auto &name = *__p##name;

        /*******************************************************/
        /*                 SOLVER MAJOR START                  */
        /*******************************************************/
        tstart = MPI_Wtime();
        tstartInternal = tstart;

        Timer().clearAllTimer();

        DNDS_MPI_InsertCheck(mpi, "Implicit 2 nvars " + std::to_string(nVars));

        /*******************************************************/
        /*                   DEFINE LAMBDAS                    */
        /*******************************************************/

        auto frhsOuter =
            [&](
                ArrayDOFV<nVarsFixed> &crhs,
                ArrayDOFV<nVarsFixed> &cx,
                ArrayDOFV<1> &dTau,
                int iter, real ct, int uPos, int reconstructionFlag)
        {
            cx.trans.startPersistentPull();
            cx.trans.waitPersistentPull(); // for hermite3
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            auto &uRecIncC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRecInc1 : uRecInc;
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            // if (mpi.rank == 0)
            //     std::cout << uRecC.father.get() << std::endl;
            typename TVFV::template TFBoundary<nVarsFixed>
                FBoundary = [&](const TU &UL, const TU &UMean, index iCell, index iFace, int ig,
                                const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
                bool compressed = false;
                TU ULfixed = eval.CompressRecPart(
                    UMean,
                    UL - UMean,
                    compressed);
                return eval.generateBoundaryValue(ULfixed, UMean, iCell, iFace, ig, normOutV, normBase, pPhy, tSimu + ct * curDtImplicit, bType, true, 1);
            };
            typename TVFV::template TFBoundaryDiff<nVarsFixed>
                FBoundaryDiff = [&](const TU &UL, const TU &dU, const TU &UMean, index iCell, index iFace, int ig,
                                    const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
                bool compressed = false;
                TU ULfixed = eval.CompressRecPart(
                    UMean,
                    UL - UMean,
                    compressed);
                TU ULfixedPlus = eval.CompressRecPart(
                    UMean,
                    UL - UMean + dU,
                    compressed);
                return eval.generateBoundaryValue(ULfixedPlus, UMean, iCell, iFace, ig, normOutV, normBase, pPhy, tSimu + ct * curDtImplicit, bType, true, 1) -
                       eval.generateBoundaryValue(ULfixed, UMean, iCell, iFace, ig, normOutV, normBase, pPhy, tSimu + ct * curDtImplicit, bType, true, 1);
            };

            eval.FixUMaxFilter(cx);
            if ((iter > config.convergenceControl.nAnchorUpdateStart &&
                 (iter - config.convergenceControl.nAnchorUpdateStart - 1) % config.convergenceControl.nAnchorUpdate == 0))
            {
                eval.updateBCAnchors(cx, uRecC);
                eval.updateBCProfiles(cx, uRecC);
                eval.updateBCProfilesPressureRadialEq();
            }
            // cx.trans.startPersistentPull();
            // cx.trans.waitPersistentPull();

            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uOld[iCell].m() = uRec[iCell].m();
            if (!reconstructionFlag)
            {
                betaPPC.setConstant(1.0);
                alphaPP_tmp.setConstant(1.0);
                uRecNew.setConstant(0.0);
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecNew, uRecNew, betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit, TEval::RHS_Ignore_Viscosity); // TODO: test with viscosity
                // vfv->DoReconstruction2nd(uRecOld, cx, FBoundary, 1, std::vector<int>());
                // eval.EvaluateRHS(crhs, JSourceC, cx, uRecOld, uRecNew, betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit,
                //                  0); // TEval::RHS_Ignore_Viscosity
                return;
            }

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: StartRec");
            int nRec = (gradIsZero ? config.implicitReconstructionControl.nRecMultiplyForZeroedGrad : 1) *
                       config.implicitReconstructionControl.nInternalRecStep;
            if (step <= config.implicitReconstructionControl.zeroRecForSteps ||
                iter <= config.implicitReconstructionControl.zeroRecForStepsInternal)
            {
                nRec = 0;
                uRec.setConstant(0.0);
            }
            real recIncBase = 0;
            Timer().StartTimer(PerformanceTimer::Reconstruction);
            if (config.implicitReconstructionControl.storeRecInc)
                uRecOld = uRecC;
            if (config.implicitReconstructionControl.recLinearScheme == 0)
            {
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    if (nRec > 1)
                        uRecNew1 = uRecC;

                    vfv->DoReconstructionIter(
                        uRecC, uRecNew, cx,
                        FBoundary,
                        false);

                    uRecC.trans.startPersistentPull();
                    uRecC.trans.waitPersistentPull();

                    if (nRec > 1)
                    {
                        uRecNew1 -= uRecC;
                        real recInc = uRecNew1.norm2();
                        if (iRec == 1)
                            recIncBase = recInc;

                        if (iRec % config.implicitReconstructionControl.nRecConsolCheck == 0)
                        {
                            if (mpi.rank == 0)
                                log() << iRec << " Rec inc: " << recIncBase << " -> " << recInc << std::endl;
                        }
                        if (recInc < recIncBase * config.implicitReconstructionControl.recThreshold)
                            break;
                    }
                }
            }
            else if (config.implicitReconstructionControl.recLinearScheme == 1)
            {
                Eigen::VectorXd meanScale;
                if (config.implicitReconstructionControl.gmresRecScale == 1)
                {
                    meanScale = eval.settings.refU;
                    meanScale(Seq123).setConstant(std::sqrt(meanScale(0) * meanScale(I4))); //! using consistent rho U scale
                }
                else
                    meanScale.setConstant(nVars, 1.0);
                // meanScale(0) = 10;
                TU meanScaleInv = (meanScale.array() + verySmallReal).inverse();

                int nGMRESrestartAll{0};
                real gmresResidualB = 0;
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    vfv->DoReconstructionIter(
                        uRecC, uRecNew, cx,
                        FBoundary,
                        true, true);
                    uRecNew.trans.startPersistentPull();
                    uRecNew.trans.waitPersistentPull();
                    // uRec.setConstant(0.0);
                    uRecNew1 = uRecNew;
                    if (iRec == 1)
                        gmresResidualB = uRecNew1.norm2();

                    bool gmresConverge =
                        gmresRec->solve(
                            [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &Ax)
                            {
                                vfv->DoReconstructionIterDiff(uRec, x, Ax, cx, FBoundaryDiff);
                                Ax.trans.startPersistentPull();
                                Ax.trans.waitPersistentPull();
                            },
                            [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &MLx)
                            {
                                MLx = x; // initial value; for the input is mostly a good estimation
                                // MLx no need to comm
                                // vfv->DoReconstructionIterSOR(uRecC, x, MLx, cx, FBoundaryDiff, false); //! causes loss of accuracy; why?????
                            },
                            [&](ArrayRECV<nVarsFixed> &a, ArrayRECV<nVarsFixed> &b) -> real
                            {
                                return (a.dotV(b).array() * meanScaleInv.transpose().array().square()).sum(); //! dim balancing here
                            },
                            uRecNew, uRecNew1, config.implicitReconstructionControl.nGmresIter,
                            [&](uint32_t i, real res, real resB) -> bool
                            {
                                if (i > 0)
                                {
                                    nGMRESrestartAll++;
                                    if (mpi.rank == 0 &&
                                        (nGMRESrestartAll % config.implicitReconstructionControl.nRecConsolCheck == 0))
                                    {
                                        auto fmt = log().flags();
                                        log() << std::scientific;
                                        log() << "GMRES for Rec: " << iRec << " Restart " << i << " " << resB << " -> " << res << std::endl;
                                        log().setf(fmt);
                                    }
                                }
                                return res < gmresResidualB * config.implicitReconstructionControl.recThreshold;
                            });
                    uRecC.addTo(uRecNew1, -1);
                    if (gmresConverge)
                        break;
                }
                uRecC.trans.startPersistentPull();
                uRecC.trans.waitPersistentPull();
            }
            else if (config.implicitReconstructionControl.recLinearScheme == 2)
            {
                Eigen::Array<real, 1, Eigen::Dynamic> resB;
                int nPCGIterAll{0};
                auto &pcgRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? pcgRec1 : pcgRec;
                auto &uRecBC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRecB1 : uRecB;

                if (iter <= 2 //! consecutive pcg is bad in 0012, using separate pcg
                    || pcgRecC->getPHistorySize() >= config.implicitReconstructionControl.fpcgMaxPHistory)
                    pcgRecC->reset();
                uRecNew = uRecBC; // last version of B
                vfv->DoReconstructionIter(
                    uRecC, uRecBC, cx,
                    FBoundary, true, true, true); // puts rhs of reconstruction A^-1b into uRecBC
                if (iter > 2)
                {
                    if (config.implicitReconstructionControl.fpcgResetScheme == 0)
                    {
                        Eigen::RowVectorXd bPrevSqr = uRecNew.dotV(uRecNew);
                        uRecNew.addTo(uRecBC, -1.0);
                        Eigen::RowVectorXd bIncSqr = uRecNew.dotV(uRecNew);
                        real maxPortion = (bIncSqr.array() / (bPrevSqr.array() + smallReal)).sqrt().maxCoeff();
                        if (maxPortion >= config.implicitReconstructionControl.fpcgResetThres)
                        {
                            if (mpi.rank == 0 && config.implicitReconstructionControl.fpcgResetReport > 0)
                                log() << "FPCG force reset at portion " << fmt::format("{:.4g}", maxPortion) << std::endl;
                            pcgRecC->reset();
                        }
                    }
                    else if (config.implicitReconstructionControl.fpcgResetScheme == 1)
                        pcgRecC->reset();
                    else if (config.implicitReconstructionControl.fpcgResetScheme == 2)
                        ;
                    else
                        DNDS_assert_info(false, "invalid fpcgResetScheme");
                }
                // pcgRecC->reset(); // using separate pcg
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    bool pcgConverged = pcgRecC->solve(
                        [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &Ax)
                        {
                            vfv->DoReconstructionIterDiff(uRec, x, Ax, cx, FBoundaryDiff);
                            Ax.trans.startPersistentPull();
                            Ax.trans.waitPersistentPull();
                            vfv->MatrixAMult(Ax, Ax);
                        },
                        [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &Mx)
                        {
                            vfv->MatrixAMult(x, Mx);
                        },
                        [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &res)
                        {
                            vfv->DoReconstructionIter(
                                x, res, cx,
                                FBoundary, true, true);
                            res.trans.startPersistentPull();
                            res.trans.waitPersistentPull();
                            res *= -1;
                        },
                        [&](ArrayRECV<nVarsFixed> &a, ArrayRECV<nVarsFixed> &b)
                        {
                            return (a.dotV(b)).array();
                        },
                        uRecC, config.implicitReconstructionControl.nGmresIter,
                        [&](uint32_t i, const Eigen::Array<real, 1, Eigen::Dynamic> &res, const Eigen::Array<real, 1, Eigen::Dynamic> &res1) -> bool
                        {
                            if (i == 1 && iRec == 1)
                                resB = res1;
                            if (i > 0)
                            {
                                nPCGIterAll++;
                                if (mpi.rank == 0 &&
                                    (nPCGIterAll % config.implicitReconstructionControl.nRecConsolCheck == 0))
                                {
                                    auto fmt = log().flags();
                                    log() << std::scientific << std::setprecision(3);
                                    log() << "PCG for Rec: " << iRec << " Restart " << i << " [" << resB << "] -> [" << res1 << "]" << std::endl;
                                    log().setf(fmt);
                                }
                            }
                            return (res1 / resB).maxCoeff() < config.implicitReconstructionControl.recThreshold;
                        });
                    if (pcgConverged)
                        break;
                }
                uRecC.trans.startPersistentPull();
                uRecC.trans.waitPersistentPull();
            }
            else
                DNDS_assert_info(false, "no such recLinearScheme");
            if ((model == NS_SA || model == NS_SA_3D) && eval.settings.ransForce2nd)
            {
                std::vector<int> mask;
                mask.resize(1);
                mask[0] = 5;
                vfv->DoReconstruction2nd(uRec, u, FBoundary, 1, mask);
            }
            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                std::vector<int> mask;
                mask.resize(2);
                mask[0] = 5;
                mask[1] = 6;
                vfv->DoReconstruction2nd(uRec, u, FBoundary, 1, mask);
            }
            Timer().StopTimer(PerformanceTimer::Reconstruction);
            if (gradIsZero)
            {
                uRec = uRecC;
                if (config.timeMarchControl.odeCode == 401)
                    uRec1 = uRecC;
                gradIsZero = false;
            }

            Timer().StartTimer(PerformanceTimer::Limiter);
            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uRecC[iCell].m() -= uOld[iCell].m();

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: StartLim");
            if (config.limiterControl.useLimiter)
            {
                // vfv->ReconstructionWBAPLimitFacial(
                //     cx, uRecC, uRecNew, uF0, uF1, ifUseLimiter,

                using tLimitBatch = typename TVFV::template tLimitBatch<nVarsFixed>;

                auto fML = [&](const TU &UL, const TU &UR, const Geom::tPoint &n,
                               const Eigen::Ref<tLimitBatch> &data) -> tLimitBatch
                {
                    Timer().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234) * 0.5;
                    Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasLeftEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Eigen::all, Seq123) *= normBase.transpose();

                    Eigen::Matrix<real, nVarsFixed, nVarsFixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;
                    Timer().StopTimer(PerformanceTimer::LimiterA);
                    return (ret * data.transpose()).transpose();
                    // return real(1);
                };
                auto fMR = [&](const TU &UL, const TU &UR, const Geom::tPoint &n,
                               const Eigen::Ref<tLimitBatch> &data) -> tLimitBatch
                {
                    Timer().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234) * 0.5;
                    Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasRightEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Seq123, Eigen::all) = normBase * M(Seq123, Eigen::all);

                    Eigen::Matrix<real, nVarsFixed, nVarsFixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;

                    Timer().StopTimer(PerformanceTimer::LimiterA);
                    return (ret * data.transpose()).transpose();
                    // return real(1);
                };
                if (config.limiterControl.smoothIndicatorProcedure == 0)
                    vfv->template DoCalculateSmoothIndicator<nVarsFixed, 2>(
                        ifUseLimiter, (uRecC), (u),
                        std::array<int, 2>{0, I4});
                else if (config.limiterControl.smoothIndicatorProcedure == 1)
                    vfv->template DoCalculateSmoothIndicatorV1<nVarsFixed>(
                        ifUseLimiter, (uRecC), (u),
                        TU::Ones(nVars),
                        [&](Eigen::Matrix<real, 1, nVarsFixed> &v)
                        {
                            TU prim;
                            TU cons;
                            cons = v.transpose();
                            if (cons(0) < verySmallReal)
                            {
                                v.setConstant(-veryLargeReal * v(I4));
                                return;
                            }
                            Gas::IdealGasThermalConservative2Primitive<dim>(cons, prim, eval.settings.idealGasProperty.gamma);
                            v.setConstant(prim(I4));
                            return;
                        });

                else
                {
                    DNDS_assert(false);
                }
                if (config.limiterControl.limiterProcedure == 1)
                    vfv->template DoLimiterWBAP_C<nVarsFixed>(
                        (cx),
                        (uRecC),
                        (uRecLimited),
                        (uRecNew),
                        ifUseLimiter,
                        iter < config.limiterControl.nPartialLimiterStartLocal && step < config.limiterControl.nPartialLimiterStart,
                        fML, fMR, true);
                else if (config.limiterControl.limiterProcedure == 0)
                    vfv->template DoLimiterWBAP_3<nVarsFixed>(
                        (cx),
                        (uRecC),
                        (uRecLimited),
                        (uRecNew),
                        ifUseLimiter,
                        iter < config.limiterControl.nPartialLimiterStartLocal && step < config.limiterControl.nPartialLimiterStart,
                        fML, fMR, true);
                else
                {
                    DNDS_assert(false);
                }
                // uRecLimited.trans.startPersistentPull();
                // uRecLimited.trans.waitPersistentPull();
            }
            Timer().StopTimer(PerformanceTimer::Limiter);
            if (config.implicitReconstructionControl.storeRecInc)
            {
                uRecIncC = uRecC;
                uRecIncC -= uRecOld; //! uRecIncC now stores uRecIncrement
            }
            if (config.implicitReconstructionControl.storeRecInc && config.implicitReconstructionControl.dampRecIncDTau)
            {
                ArrayDOFV<1> damper;
                ArrayDOFV<1> damper1;
                vfv->BuildUDof(damper, 1);
                vfv->BuildUDof(damper1, 1);
                damper = dTau;
                damper1 = dTau;
                damper1 += curDtImplicit;
                damper /= damper1;
                // for (auto &v : damper)
                //     v = v / (curDtImplicit + v); //! warning: teleported value here

                uRecIncC *= damper;
                uRecC = uRecOld;
                uRecC += uRecIncC;
            }
            if (config.limiterControl.preserveLimited && config.limiterControl.useLimiter)
                uRecC = uRecLimited;

            // uRecC.trans.startPersistentPull(); //! this also need to update!
            // uRecC.trans.waitPersistentPull();

            // }

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: StartEval");
            eval.setPassiveDiscardSource(iter <= 0);

            if (iter == 1)
                alphaPPC.setConstant(1.0); // make RHS un-disturbed
            alphaPP_tmp.setConstant(1.0);  // make RHS un-disturbed
            if (config.limiterControl.usePPRecLimiter)
            {
                Timer().StartTimer(PerformanceTimer::Positivity);
                nLimBeta = 0;
                minBeta = 1;
                if (!config.limiterControl.useLimiter)
                    uRecLimited = uRecC;
                eval.EvaluateURecBeta(cx, uRecLimited, betaPPC, nLimBeta, minBeta,
                                      config.limiterControl.ppRecLimiterCompressToMean
                                          ? TEval::EvaluateURecBeta_COMPRESS_TO_MEAN
                                          : TEval::EvaluateURecBeta_DEFAULT); //*cx instead of u!
                if (nLimBeta)
                    if (mpi.rank == 0 &&
                        (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 3))
                    {
                        log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole)
                              << "PPRecLimiter: nLimBeta [" << nLimBeta << "]"
                              << " minBeta[" << minBeta << "]" << std::endl;
                    }
                uRecLimited.trans.startPersistentPull();
                betaPPC.trans.startPersistentPull();
                uRecLimited.trans.waitPersistentPull();
                betaPPC.trans.waitPersistentPull();
                Timer().StopTimer(PerformanceTimer::Positivity);
            }

            Timer().StartTimer(PerformanceTimer::RHS);
            if (config.limiterControl.useLimiter || config.limiterControl.usePPRecLimiter) // todo: opt to using limited for uRecUnlim
                eval.EvaluateRHS(crhs, JSourceC, cx, config.limiterControl.useViscousLimited ? uRecLimited : uRecC, uRecLimited,
                                 betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, uRecC,
                                 betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit);

            crhs.trans.startPersistentPull();
            crhs.trans.waitPersistentPull();
            if (getNVars(model) > (I4 + 1) && iter <= config.others.nFreezePassiveInner)
            {
                for (int i = 0; i < crhs.Size(); i++)
                    crhs[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
            Timer().StopTimer(PerformanceTimer::RHS);

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: End");
        };

        auto frhs =
            [&](
                ArrayDOFV<nVarsFixed> &crhs,
                ArrayDOFV<nVarsFixed> &cx,
                ArrayDOFV<1> &dTau,
                int iter, real ct, int uPos)
        {
            return frhsOuter(crhs, cx, dTau, iter, ct, uPos, 1); // reconstructionFlag == 1
        };

        auto fdtau = [&](ArrayDOFV<nVarsFixed> &cx, ArrayDOFV<1> &dTau, real alphaDiag, int uPos)
        {
            eval.FixUMaxFilter(cx);
            cx.trans.startPersistentPull(); //! this also need to update!
            cx.trans.waitPersistentPull();
            // uRec.trans.startPersistentPull();
            // uRec.trans.waitPersistentPull();
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            eval.EvaluateDt(dTau, cx, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            for (int iS = 1; iS <= config.implicitCFLControl.nSmoothDTau; iS++)
            {
                // ArrayDOFV<1> dTauNew = dTau; //TODO: copying is still unusable; consider doing copiers on the level of ArrayDOFV and ArrayRecV
                dTauTmp = dTau;
                dTau.trans.startPersistentPull();
                dTau.trans.waitPersistentPull();
                eval.MinSmoothDTau(dTau, dTauTmp);
                dTau = dTauTmp;
            }

            dTau *= 1. / alphaDiag;
        };

        auto fincrement = [&](
                              ArrayDOFV<nVarsFixed> &cx,
                              ArrayDOFV<nVarsFixed> &cxInc,
                              real alpha, int uPos)
        {
            Timer().StartTimer(PerformanceTimer::Positivity);
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            nLimInc = 0;
            alphaMinInc = 1;
            eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, cxInc, alphaPP_tmp, nLimInc, alphaMinInc, config.timeMarchControl.incrementPPRelax,
                                      2, TEval::EvaluateCellRHSAlpha_DEFAULT);
            if (nLimInc)
                if (mpi.rank == 0 &&
                    (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 2))
                {
                    log() << std::scientific << std::setw(3) << TermColor::Red;
                    log() << "PPIncrementLimiter: nIncrementRes[" << nLimInc << "] minAlpha [" << alphaMinInc << "]"
                          << TermColor::Reset
                          << std::endl;
                }
            auto pUTemp = uPool.getAllocInit(initUDOF);
            auto &uTemp = *pUTemp;

            uTemp = cxInc;
            uTemp *= alphaPP_tmp;
            //*directadd
            // cx += uTemp;
            //*fixing add
            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                if (config.implicitCFLControl.RANSRelax != 1)
                    for (index i = 0; i < uTemp.Size(); i++)
                        uTemp[i]({I4, I4 + 1}) *= config.implicitCFLControl.RANSRelax;
            }
            Timer().StopTimer(PerformanceTimer::Positivity);
            eval.AddFixedIncrement(cx, uTemp, alpha);
            eval.AssertMeanValuePP(cx, true);

            // eval.AddFixedIncrement(cx, cxInc, alpha);
        };

        auto fsolve = [&](ArrayDOFV<nVarsFixed> &cx, ArrayDOFV<nVarsFixed> &cres, ArrayDOFV<nVarsFixed> &resOther, ArrayDOFV<1> &dTau,
                          real dt, real alphaDiag, ArrayDOFV<nVarsFixed> &cxInc, int iter, real ct, int uPos)
        {
            {
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsTemp)
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
                rhsTemp = cres;
                eval.CentralSmoothResidual(rhsTemp, cres, uTemp);
            }

            auto &JDC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JD1 : JD;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &uRecIncC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRecInc1 : uRecInc;
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;

            Timer().StartTimer(PerformanceTimer::Positivity);
            if (config.timeMarchControl.rhsFPPMode == 1 || config.timeMarchControl.rhsFPPMode == 11)
            {
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsTemp)
                // ! experimental: bad now ?
                rhsTemp = cres;
                rhsTemp *= dTau;
                rhsTemp *= config.timeMarchControl.rhsFPPScale;
                index nLimFRes = 0;
                real alphaMinFRes = 1;
                eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, rhsTemp, alphaPP_tmp, nLimFRes, alphaMinFRes, config.timeMarchControl.rhsFPPRelax,
                                          2, config.timeMarchControl.rhsFPPMode == 1 ? TEval::EvaluateCellRHSAlpha_DEFAULT : TEval::EvaluateCellRHSAlpha_MIN_IF_NOT_ONE);
                if (nLimFRes)
                    if (mpi.rank == 0)
                    {
                        log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole) << std::setw(3) << TermColor::Red;
                        log() << "PPFResLimiter: nLimFRes[" << nLimFRes << "] minAlpha [" << alphaMinFRes << "]" << TermColor::Reset << std::endl;
                    }

                cres *= alphaPP_tmp;
            }
            else if (config.timeMarchControl.rhsFPPMode == 2)
            {
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsTemp)
                rhsTemp = cres;
                rhsTemp *= dTau;
                rhsTemp *= config.timeMarchControl.rhsFPPScale;
                index nLimFRes = 0;
                real alphaMinFRes = 1;
                eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, rhsTemp, alphaPP_tmp, nLimFRes, alphaMinFRes, config.timeMarchControl.rhsFPPRelax,
                                          2, TEval::EvaluateCellRHSAlpha_DEFAULT);
                if (nLimFRes)
                    if (mpi.rank == 0)
                    {
                        log() << std::scientific << std::setw(3) << TermColor::Red;
                        log() << "PPFResLimiter: nLimFRes[" << nLimFRes << "] minAlpha [" << alphaMinFRes << "]" << TermColor::Reset << std::endl;
                    }
                alphaPP_tmp.setMaxWith(smallReal); // dTau cannot be zero
                dTauTmp = dTau;
                dTauTmp *= alphaPP_tmp;
            }
            auto &dTauC = config.timeMarchControl.rhsFPPMode == 2 ? dTauTmp : dTau;
            Timer().StopTimer(PerformanceTimer::Positivity);

            if (config.limiterControl.useLimiter) // uses urec value
                eval.LUSGSMatrixInit(JDC, JSourceC,
                                     dTauC, dt, alphaDiag,
                                     cx, uRecLimited,
                                     0,
                                     tSimu);
            else
                eval.LUSGSMatrixInit(JDC, JSourceC,
                                     dTauC, dt, alphaDiag,
                                     cx, uRecC,
                                     0,
                                     tSimu);

            // for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            // {
            //     cres[iCell] = eval.CompressInc(cx[iCell], cres[iCell] * dTau[iCell]) / dTau[iCell];
            // }
            cxInc.setConstant(0.0);
            this->solveLinear(alphaDiag, cres, cx, cxInc, uRecC, uRecIncC,
                              JDC, *gmres, 0);
            // cxInc: in: full increment from previous level; out: full increment form current level
            const auto solve_multigrid = [&](TDof &x_upper, TDof &xIncBuf, TDof &rhsBuf, const TDof &resOther, int mgLevelInit, int mgLevelMax)
            {
                uRecNew.setConstant(0.0);
                // templated lambda recursion:
                // https://stackoverflow.com/questions/2067988/how-to-make-a-recursive-lambda
                // std::function<void(TDof &, TDof &, int, int)> solve_multigrid_impl;
                // solve_multigrid_impl = [&](TDof &x_base, TDof &cxInc, int mgLevel, int mgLevelMax)

                //  [&frhs, &fincrement, &fdtau, &initUDOF,
                //                              &eval, &config = config, &dTauC, &cres, &resOther,
                //                              &mpi = mpi, &JSourceTmp = JSourceTmp, &JDTmp = JDTmp,
                //                              &uPool = uPool, &uRecNew = uRecNew, &uRec = uRec, &betaPP = betaPP, &alphaPP = alphaPP,
                //                              &gmres, tSimu, solveLinear = solveLinear,
                //                              alphaDiag, dt,
                //                              iter, ct, uPos]
                auto solve_multigrid_impl = [&](TDof &x_upper, const TDof &rhs_upper, const TDof &resOther_upper, int mgLevel, int mgLevelMax, auto &solve_multigrid_impl_ref) -> void
                {
                    static const int use_1st_conv = 1;
                    DNDS_assert(mgLevel > 0 && mgLevel <= mgLevelMax);
                    DNDS_assert(config.linearSolverControl.coarseGridLinearSolverControlList.at(mgLevel - 1).multiGridNIterPost >= 0);
                    int curMGIter = config.linearSolverControl.coarseGridLinearSolverControlList.at(mgLevel - 1).multiGridNIter +
                                    config.linearSolverControl.coarseGridLinearSolverControlList.at(mgLevel - 1).multiGridNIterPost;
                    if (curMGIter < 0)
                        curMGIter = config.linearSolverControl.multiGridLPInnerNIter;
                    DNDS_EULER_SOLVER_GET_TEMP_UDOF(uMG1)
                    // DNDS_EULER_SOLVER_GET_TEMP_UDOF(uMG1Init)
                    // DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsInitCurMG1)
                    DNDS_EULER_SOLVER_GET_TEMP_UDOF(resOtherCurMG)
                    DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsTemp)
                    // if (mgLevel == 2)
                    //     return;
                    uMG1 = x_upper; // projection
                    // std::cout << "here0" << std::endl;
                    resOtherCurMG = rhs_upper; // projection
                    resOtherCurMG *= alphaDiag;
                    resOtherCurMG += resOther_upper; // projection
                    // res0CurMG.addTo(uMG1, -1. / dt);
                    // uMG1Init = uMG1;

                    for (int iIterMG = 1; iIterMG <= curMGIter; iIterMG++)
                    {

                        if (mgLevel == 1)
                            eval.EvaluateRHS(rhsTemp, JSourceTmp, uMG1,
                                             config.limiterControl.useViscousLimited ? uRecNew : uRec /*dummy*/, uRec /*dummy*/,
                                             betaPP /*dummy*/, alphaPP /*dummy*/, false, tSimu + dt * ct,
                                             TEval::RHS_Direct_2nd_Rec | TEval::RHS_Dont_Record_Bud_Flux | TEval::RHS_Dont_Update_Integration);
                        else if (mgLevel == 2)
                            eval.EvaluateRHS(rhsTemp, JSourceTmp, uMG1,
                                             config.limiterControl.useViscousLimited ? uRecNew : uRec /*dummy*/, uRec /*dummy*/,
                                             betaPP /*dummy*/, alphaPP /*dummy*/, false, tSimu + dt * ct,
                                             (TEval::RHS_Direct_2nd_Rec_1st_Conv * use_1st_conv) | TEval::RHS_Direct_2nd_Rec | TEval::RHS_Dont_Record_Bud_Flux | TEval::RHS_Dont_Update_Integration);
                        else
                            DNDS_assert(false);
                        rhsTemp.trans.startPersistentPull();
                        rhsTemp.trans.waitPersistentPull();
                        if (iIterMG == 1)
                        {
                            // rhsInitCurMG1 = rhsTemp;
                            resOtherCurMG.addTo(rhsTemp, -alphaDiag);
                        }
                        // if (mgLevel < mgLevelMax && iIterMG == 1) // pre smoother coarser grid call
                        //     solve_multigrid_impl_ref(uMG1, rhsTemp, resOtherCurMG, mgLevel + 1, mgLevelMax, solve_multigrid_impl_ref);
                        rhsTemp *= alphaDiag;
                        rhsTemp += resOtherCurMG;
                        rhsTemp.addTo(uMG1, -1. / dt);
                        // todo: add rhsfpphere
                        // rhsTemp === alphaDiag * rhs(cur) - alphaDiag * rhs(at_step_1) - uMG1 / dt + uMG1Init / dt
                        fdtau(uMG1, dTauC, alphaDiag, uPos); //! warning! dTauC is overwritten
                        eval.LUSGSMatrixInit(JDTmp, JSourceTmp, dTauC, dt, alphaDiag, uMG1, uRecNew, 0, tSimu);

                        if (iIterMG % config.linearSolverControl.multiGridLPInnerNSee == 0)
                        {
                            Eigen::VectorFMTSafe<real, -1> resNorm;
                            eval.EvaluateNorm(resNorm, rhsTemp);
                            if (mpi.rank == 0)
                                log() << fmt::format("MG Level LP [{}] iter [{}] res [{:.3e}]", mgLevel, iIterMG, resNorm.transpose()) << std::endl;
                        }
                        xIncBuf.setConstant(0.0);
                        solveLinear(alphaDiag, rhsTemp, uMG1, xIncBuf, uRecNew, uRecNew,
                                    JDTmp, *gmres, mgLevel);
                        fincrement(uMG1, xIncBuf, 1.0, uPos);
                        // solve_multigrid_impl(x_base, cxInc, mgLevel + 1, mgLevelMax);
                        // cxInc *= -1;
                        // std::cout << "here" << std::endl;

                        if (mgLevel < mgLevelMax &&
                            iIterMG == config.linearSolverControl.coarseGridLinearSolverControlList.at(mgLevel - 1).multiGridNIter) // post smoother coarser grid call
                        {
                            if (mgLevel == 1)
                                eval.EvaluateRHS(rhsTemp, JSourceTmp, uMG1,
                                                 config.limiterControl.useViscousLimited ? uRecNew : uRec /*dummy*/, uRec /*dummy*/,
                                                 betaPP /*dummy*/, alphaPP /*dummy*/, false, tSimu + dt * ct,
                                                 TEval::RHS_Direct_2nd_Rec | TEval::RHS_Dont_Record_Bud_Flux | TEval::RHS_Dont_Update_Integration);
                            else if (mgLevel == 2)
                                eval.EvaluateRHS(rhsTemp, JSourceTmp, uMG1,
                                                 config.limiterControl.useViscousLimited ? uRecNew : uRec /*dummy*/, uRec /*dummy*/,
                                                 betaPP /*dummy*/, alphaPP /*dummy*/, false, tSimu + dt * ct,
                                                 (TEval::RHS_Direct_2nd_Rec_1st_Conv * use_1st_conv) | TEval::RHS_Direct_2nd_Rec | TEval::RHS_Dont_Record_Bud_Flux | TEval::RHS_Dont_Update_Integration);
                            else
                                DNDS_assert(false);
                            solve_multigrid_impl_ref(uMG1, rhsTemp, resOtherCurMG, mgLevel + 1, mgLevelMax, solve_multigrid_impl_ref);
                        }
                    }

                    x_upper = uMG1; // interpolate

                    // alphaDiag *rhsTemp(x + xinc) - xinc / dt == alphaDiag *rhsTemp(x) - res_of_first
                };

                frhs(rhsBuf, x_upper, dTauC, iter, ct, uPos);
                solve_multigrid_impl(x_upper, rhsBuf, resOther, mgLevelInit, mgLevelMax, solve_multigrid_impl);
            };

            if (config.linearSolverControl.multiGridLP >= 1 && iter > config.linearSolverControl.multiGridLPStartIter)
            {
                DNDS_assert(config.linearSolverControl.multiGridLP <= 2);
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(cxTemp)
                cxTemp = cx;
                fincrement(cxTemp, cxInc, 1.0, uPos);
                solve_multigrid(cxTemp, cxInc, cres, resOther, 1, config.linearSolverControl.multiGridLP); //! overwrites cxInc and cres here
                cxInc = cxTemp;
                cxInc -= cx;
            }
            // eval.FixIncrement(cx, cxInc);
            // !freeze something
            if (getNVars(model) > I4 + 1 && iter <= config.others.nFreezePassiveInner)
            {
                for (int i = 0; i < cres.Size(); i++)
                    cxInc[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
        };

        auto fsolveNest = [&](
                              ArrayDOFV<nVarsFixed> &cx,
                              ArrayDOFV<nVarsFixed> &cx1,
                              ArrayDOFV<nVarsFixed> &crhs,
                              ArrayDOFV<1> &dTau,
                              const std::vector<real> &Coefs, // coefs are dU * c[0] + dt * c[1] * (I/(dt * c[2]) - JMid) * (I/(dt * c[3]) - J)
                              real dt, real alphaDiag, ArrayDOFV<nVarsFixed> &cxInc, int iter, int uPos)
        {
            {
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsTemp)

                crhs.trans.startPersistentPull();
                crhs.trans.waitPersistentPull();
                rhsTemp = crhs;
                eval.CentralSmoothResidual(rhsTemp, crhs, uTemp);
            }

            cxInc.setConstant(0.0);
            auto &JDC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JD1 : JD;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            // TODO: use "update spectral radius" procedure? or force update in fsolve
            eval.EvaluateDt(dTau, cx1, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            dTau *= Coefs[2];
            eval.LUSGSMatrixInit(JD1, JSource1,
                                 dTau, dt * Coefs[2], alphaDiag,
                                 cx1, uRec,
                                 0,
                                 tSimu);
            eval.EvaluateDt(dTau, cx, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            dTau *= Coefs[3] * veryLargeReal;
            eval.LUSGSMatrixInit(JD, JSource,
                                 dTau, dt * Coefs[3], alphaDiag,
                                 cx, uRec,
                                 0,
                                 tSimu);

            // for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            // {
            //     crhs[iCell] = eval.CompressInc(cx[iCell], crhs[iCell] * dTau[iCell]) / dTau[iCell];
            // }

            if (config.linearSolverControl.gmresCode == 0 || config.linearSolverControl.gmresCode == 2)
            {
                // //! LUSGS
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)

                eval.UpdateLUSGSForward(alphaDiag, crhs, cx1, cxInc, JD1, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                eval.UpdateLUSGSBackward(alphaDiag, crhs, cx1, cxInc, JD1, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                uTemp = cxInc;
                eval.UpdateLUSGSForward(alphaDiag, uTemp, cx, cxInc, JD, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                eval.UpdateLUSGSBackward(alphaDiag, uTemp, cx, cxInc, JD, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                cxInc *= 1. / (dt * Coefs[1]);
            }

            if (config.linearSolverControl.gmresCode != 0)
            {
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
                // !  GMRES
                // !  for gmres solver: A * uinc = rhsinc, rhsinc is average value insdead of cumulated on vol
                gmres->solve(
                    [&](decltype(u) &x, decltype(u) &Ax)
                    {
                        eval.LUSGSMatrixVec(alphaDiag, cx, x, JD, Ax);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                        uTemp = Ax;
                        eval.LUSGSMatrixVec(alphaDiag, cx1, uTemp, JD1, Ax);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                        Ax *= dt * Coefs[1];
                        Ax.addTo(x, Coefs[0]);
                    },
                    [&](decltype(u) &x, decltype(u) &MLx)
                    {
                        // x as rhs, and MLx as uinc
                        eval.UpdateLUSGSForward(alphaDiag, x, cx1, MLx, JD1, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        eval.UpdateLUSGSBackward(alphaDiag, x, cx1, MLx, JD1, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        uTemp = MLx;
                        eval.UpdateLUSGSForward(alphaDiag, uTemp, cx, MLx, JD, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        eval.UpdateLUSGSBackward(alphaDiag, uTemp, cx, MLx, JD, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        MLx *= 1. / (dt * Coefs[1]);
                    },
                    [&](decltype(u) &a, decltype(u) &b) -> real
                    {
                        return a.dot(b); //! need dim balancing here
                    },
                    crhs, cxInc, config.linearSolverControl.nGmresIter,
                    [&](uint32_t i, real res, real resB) -> bool
                    {
                        if (i > 0)
                        {
                            if (mpi.rank == 0)
                            {
                                // log() << std::scientific;
                                // log() << "GMRES: " << i << " " << resB << " -> " << res << std::endl;
                            }
                        }
                        return false;
                    });
            }
            // eval.FixIncrement(cx, cxInc);
            // !freeze something
            if (getNVars(model) > I4 + 1 && iter <= config.others.nFreezePassiveInner)
            {
                for (int i = 0; i < crhs.Size(); i++)
                    cxInc[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
        };

        auto falphaLimSource = [&](
                                   ArrayDOFV<nVarsFixed> &v,
                                   int uPos)
        {
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            for (index i = 0; i < v.Size(); i++)
                v[i] *= alphaPPC[i](0);
        };

        auto fresidualIncPP = [&](
                                  ArrayDOFV<nVarsFixed> &cx,
                                  ArrayDOFV<nVarsFixed> &xPrev,
                                  ArrayDOFV<nVarsFixed> &crhs,
                                  ArrayDOFV<nVarsFixed> &rhsIncPart,
                                  const std::function<void()> &renewRhsIncPart,
                                  real ct,
                                  int uPos)
        {
            Timer().StartTimer(PerformanceTimer::Positivity);
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            renewRhsIncPart(); // un-fixed now
            // rhsIncPart.trans.startPersistentPull();
            // rhsIncPart.trans.waitPersistentPull(); //seems not needed
            eval.EvaluateCellRHSAlpha(xPrev, uRecC, betaPPC, rhsIncPart, alphaPP_tmp, nLimAlpha, minAlpha, 1.,
                                      2, TEval::EvaluateCellRHSAlpha_MIN_IF_NOT_ONE);
            alphaPP_tmp.trans.startPersistentPull();
            alphaPP_tmp.trans.waitPersistentPull();
            if (nLimAlpha)
                if (mpi.rank == 0 &&
                    (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 4))
                {
                    log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole)
                          << "PPResidualLimiter: nLimAlpha [" << nLimAlpha << "]"
                          << " minAlpha[" << minAlpha << "]" << std::endl;
                }
            alphaPPC = alphaPP_tmp;
            if (config.limiterControl.useLimiter || config.limiterControl.usePPRecLimiter)
                eval.EvaluateRHS(crhs, JSourceC, cx, config.limiterControl.useViscousLimited ? uRecLimited : uRecC, uRecLimited,
                                 betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, uRecC,
                                 betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
            // rhs now last-fixed
            crhs.trans.startPersistentPull();
            crhs.trans.waitPersistentPull();

            /********************************************/
            // first fix
            {
                renewRhsIncPart(); // now last-fixed
                // rhsIncPart.trans.startPersistentPull();
                // rhsIncPart.trans.waitPersistentPull(); //seems not needed
                eval.EvaluateCellRHSAlphaExpansion(xPrev, uRecC, betaPPC, rhsIncPart, alphaPP_tmp, nLimAlpha, minAlpha);
                alphaPP_tmp.trans.startPersistentPull();
                alphaPP_tmp.trans.waitPersistentPull();
                if (nLimAlpha)
                    if (mpi.rank == 0 &&
                        (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 4))
                    {
                        log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole)
                              << "PPResidualLimiter - first expand: nLimAlpha [" << nLimAlpha << "]"
                              << " minAlpha[" << minAlpha << "]" << std::endl;
                    }
                alphaPPC = alphaPP_tmp;
                if (config.limiterControl.useLimiter || config.limiterControl.usePPRecLimiter)
                    eval.EvaluateRHS(crhs, JSourceC, cx, config.limiterControl.useViscousLimited ? uRecLimited : uRecC, uRecLimited,
                                     betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
                else
                    eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, uRecC,
                                     betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
                crhs.trans.startPersistentPull();
                crhs.trans.waitPersistentPull();
            }
            Timer().StopTimer(PerformanceTimer::Positivity);
        };

        auto fstop = [&](int iter, ArrayDOFV<nVarsFixed> &cres, int iStep) -> bool
        {
            return functor_fstop(iter, cres, iStep, runningEnvironment);
        };

        // fmainloop gets the time-variant residual norm,
        // handles the output / log nested loops,
        // integrates physical time tsimu
        // and finally decides if break time loop
        auto fmainloop = [&]() -> bool
        {
            return functor_fmainloop(runningEnvironment);
        };

        /***************************************************************************************************************************************
                                  .___  ___.      ___       __  .__   __.     __        ______     ______   .______
                                  |   \/   |     /   \     |  | |  \ |  |    |  |      /  __  \   /  __  \  |   _  \
                                  |  \  /  |    /  ^  \    |  | |   \|  |    |  |     |  |  |  | |  |  |  | |  |_)  |
                                  |  |\/|  |   /  /_\  \   |  | |  . `  |    |  |     |  |  |  | |  |  |  | |   ___/
                                  |  |  |  |  /  _____  \  |  | |  |\   |    |  `----.|  `--'  | |  `--'  | |  |
                                  |__|  |__| /__/     \__\ |__| |__| \__|    |_______| \______/   \______/  | _|
        ***************************************************************************************************************************************/
        // step 0 extra:
        if (config.outputControl.dataOutAtInit)
        {
            PrintData(
                config.dataIOControl.outPltName + "_" + output_stamp + "_" + "00000",
                "",
                [&](index iCell)
                { return ode->getLatestRHS()[iCell](0); },
                addOutList,
                eval, tSimu);
            eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "00000",
                                 u, uRec);
        }

        for (step = 1; step <= config.timeMarchControl.nTimeStep; step++)
        {
            DNDS_MPI_InsertCheck(mpi, "Implicit Step");
            ifOutT = false;
            real curDtImplicitOld = curDtImplicit;
            curDtImplicit = config.timeMarchControl.dtImplicit;
            CFLNow = config.implicitCFLControl.CFL;
            fdtau(u, dTauTmp, 1., 0);                                                                             // generates a curDtMin / CFLNow value as a CFL=1 dt value
            curDtImplicit = std::min(curDtMin / CFLNow * config.timeMarchControl.dtCFLLimitScale, curDtImplicit); // limits dt by CFL

            if (config.timeMarchControl.useDtPPLimit)
            {
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsTemp)
                Timer().StartTimer(PerformanceTimer::PositivityOuter);
                dTauTmp.setConstant(curDtImplicit * config.timeMarchControl.dtPPLimitScale); //? used as damper here, appropriate?
                frhsOuter(rhsTemp, u, dTauTmp, 1, 0.0, 0, 0);                                //* trick: use 0th order reconstruction RHS for dt PP limiting
                uTemp = u;
                rhsTemp *= curDtImplicit * config.timeMarchControl.dtPPLimitScale;
                index nLim = 0;
                real minLim = 1;
                eval.EvaluateCellRHSAlpha(u, uRec, alphaPP, rhsTemp, alphaPP_tmp, nLim, minLim, config.timeMarchControl.dtPPLimitRelax,
                                          1, TEval::EvaluateCellRHSAlpha_DEFAULT); // using compress = 1, only minLim is used as output
                if (nLim)
                    curDtImplicit = std::min(curDtImplicit, minLim * curDtImplicit);
                if (curDtImplicitHistory.size() && curDtImplicit > curDtImplicitHistory.back() * config.timeMarchControl.dtIncreaseLimit)
                {
                    curDtImplicit = curDtImplicitHistory.back() * config.timeMarchControl.dtIncreaseLimit;
                }
                if (mpi.rank == 0 && nLim)
                {
                    log() << "##################################################################" << std::endl;
                    log() << fmt::format("At Step [{:d}] t [{:.8g}] Changing dt to [{}], using PP", step, tSimu, curDtImplicit) << std::endl;
                    log() << "##################################################################" << std::endl;
                }
                Timer().StopTimer(PerformanceTimer::PositivityOuter);
            }
            curDtImplicit = std::max(curDtImplicit, config.timeMarchControl.dtImplicitMin);

            DNDS_assert(curDtImplicit > 0);
            if (tSimu + curDtImplicit >= nextTout - curDtImplicit * smallReal) // limits dt by output nodes
            {
                ifOutT = true;
                curDtImplicit = std::max(0.0, nextTout - tSimu);
            }

            if (config.timeMarchControl.useImplicitPP)
            {
                switch (config.timeMarchControl.odeCode)
                {
                case 1: // BDF2
                    std::dynamic_pointer_cast<ODE::ImplicitBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(ode)
                        ->StepPP(
                            u, uIncBufODE,
                            frhs,
                            fdtau,
                            fsolve,
                            config.convergenceControl.nTimeStepInternal,
                            fstop, fincrement,
                            curDtImplicit + verySmallReal,
                            falphaLimSource,
                            fresidualIncPP);
                    break;
                case 102: // VBDFPP
                {
                    index nLimAlpha;
                    real minAlpha;
                    auto odeVBDF = std::dynamic_pointer_cast<ODE::ImplicitVBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(ode);
                    odeVBDF->LimitDt_StepPPV2(
                        u, [&](ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &uInc) -> real
                        {
                            eval.EvaluateCellRHSAlpha(u, uRec, alphaPP, uInc, alphaPP_tmp, nLimAlpha, minAlpha, 1.,
                                                    1, TEval::EvaluateCellRHSAlpha_MIN_IF_NOT_ONE); // using compress == 1, only minAlpha is used
                            return minAlpha; },
                        curDtImplicit, 2); // curDtImplicit modified
                    if (curDtImplicit > curDtImplicitOld)
                    {
                        dtIncreaseCounter++;
                        if (dtIncreaseCounter > config.timeMarchControl.dtIncreaseAfterCount)
                            curDtImplicit = std::min(curDtImplicitOld * config.timeMarchControl.dtIncreaseLimit, curDtImplicit);
                        else
                            curDtImplicit = curDtImplicitOld;
                    }
                    else
                    {
                        dtIncreaseCounter = 0;
                    }

                    if (mpi.rank == 0)
                    {
                        log() << "##################################################################" << std::endl;
                        log() << fmt::format("At Step [{:d}] t [{:.8g}] Changing dt to {}", step, tSimu, curDtImplicit) << std::endl;
                        log() << "##################################################################" << std::endl;
                    }
                    odeVBDF->StepPP(
                        u, uIncBufODE,
                        frhs,
                        fdtau,
                        fsolve,
                        config.convergenceControl.nTimeStepInternal,
                        fstop, fincrement,
                        curDtImplicit + verySmallReal,
                        falphaLimSource,
                        fresidualIncPP);
                }
                break;

                default:
                    DNDS_assert_info(false, "unsupported odeCode for PP!");
                    break;
                }
            }
            else if (config.timeMarchControl.odeCode == 401 && false)
                std::dynamic_pointer_cast<ODE::ImplicitHermite3SimpleJacobianDualStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(ode)
                    ->StepNested(
                        u, uIncBufODE,
                        frhs,
                        fdtau,
                        fsolve, fsolveNest,
                        config.convergenceControl.nTimeStepInternal,
                        fstop, fincrement,
                        curDtImplicit + verySmallReal);
            else
                ode
                    ->Step(
                        u, uIncBufODE,
                        frhs,
                        fdtau,
                        fsolve,
                        config.convergenceControl.nTimeStepInternal,
                        fstop, fincrement,
                        curDtImplicit + verySmallReal);
            curDtImplicitHistory.push_back(curDtImplicit);
            if (fmainloop())
                break;
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        // the real definition
        template <EulerModel model>
        ,
        // the intellisense friendly definition
        template <>
    )
    void EulerSolver<model>::solveLinear(
        real alphaDiag,
        TDof &cres, TDof &cx, TDof &cxInc, TRec &uRecC, TRec uRecIncC,
        JacobianDiagBlock<nVarsFixed> &JDC, tGMRES_u &gmres, int gridLevel)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        auto &eval = *pEval;
        auto initUDOF = [&](ArrayDOFV<nVarsFixed> &uu)
        { vfv->BuildUDof(uu, nVars); };

        TU sgsRes(nVars), sgsRes0(nVars);
        bool inputIsZero{true}, hasLUDone{false};

        typename TVFV::template TFBoundary<nVarsFixed>
            FBoundary = [&](const TU &UL, const TU &UMean, index iCell, index iFace, int iG,
                            const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
        {
            TU UR = UL;
            UR.setZero();
            return UR;
        };
        DNDS_assert(gridLevel <= config.linearSolverControl.coarseGridLinearSolverControlList.size());

        auto gmresCode =
            gridLevel > 0
                ? config.linearSolverControl.coarseGridLinearSolverControlList.at(gridLevel - 1).gmresCode
                : config.linearSolverControl.gmresCode;
        bool initWithLastURecInc =
            gridLevel > 0
                ? false
                : config.linearSolverControl.initWithLastURecInc;
        int sgsWithRec =
            gridLevel > 0
                ? 0
                : config.linearSolverControl.sgsWithRec;
        int sgsIter =
            gridLevel > 0
                ? config.linearSolverControl.coarseGridLinearSolverControlList.at(gridLevel - 1).sgsIter
                : config.linearSolverControl.sgsIter;
        int nSgsConsoleCheck =
            gridLevel > 0
                ? config.linearSolverControl.coarseGridLinearSolverControlList.at(gridLevel - 1).nSgsConsoleCheck
                : config.linearSolverControl.nSgsConsoleCheck;
        int gmresScale =
            gridLevel > 0
                ? config.linearSolverControl.coarseGridLinearSolverControlList.at(gridLevel - 1).gmresScale
                : config.linearSolverControl.gmresScale;
        int nGmresIter =
            gridLevel > 0
                ? config.linearSolverControl.coarseGridLinearSolverControlList.at(gridLevel - 1).nGmresIter
                : config.linearSolverControl.nGmresIter;
        int nGmresConsoleCheck =
            gridLevel > 0
                ? config.linearSolverControl.coarseGridLinearSolverControlList.at(gridLevel - 1).nGmresConsoleCheck
                : config.linearSolverControl.nGmresConsoleCheck;
        if (gmresCode == 0 || gmresCode == 2)
        {
            // //! LUSGS

            if (initWithLastURecInc)
            {
                DNDS_assert(config.implicitReconstructionControl.storeRecInc);
                eval.UpdateSGSWithRec(alphaDiag, cres, cx, uRecC, cxInc, uRecIncC, JDC, true, sgsRes);
                // for (index iCell = 0; iCell < uRecIncC.Size(); iCell++)
                //     std::cout << "-------\n"
                //               << uRecIncC[iCell] << std::endl;
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                eval.UpdateSGSWithRec(alphaDiag, cres, cx, uRecC, cxInc, uRecIncC, JDC, false, sgsRes);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
            }
            else
            {
                DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
                doPrecondition(alphaDiag, cres, cx, cxInc, uTemp, JDC, sgsRes, inputIsZero, hasLUDone, gridLevel);
            }

            if (sgsWithRec != 0)
                uRecNew.setConstant(0.0);
            for (int iterSGS = 1; iterSGS <= sgsIter; iterSGS++)
            {
                if (sgsWithRec != 0)
                {
                    vfv->DoReconstructionIter(
                        uRecNew, uRecNew1, cxInc,
                        FBoundary,
                        false);
                    uRecNew.trans.startPersistentPull();
                    uRecNew.trans.waitPersistentPull();
                    eval.UpdateSGSWithRec(alphaDiag, cres, cx, uRecC, cxInc, uRecNew, JDC, true, sgsRes);
                    cxInc.trans.startPersistentPull();
                    cxInc.trans.waitPersistentPull();
                    eval.UpdateSGSWithRec(alphaDiag, cres, cx, uRecC, cxInc, uRecNew, JDC, false, sgsRes);
                    cxInc.trans.startPersistentPull();
                    cxInc.trans.waitPersistentPull();
                }
                else
                {
                    DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
                    doPrecondition(alphaDiag, cres, cx, cxInc, uTemp, JDC, sgsRes, inputIsZero, hasLUDone, gridLevel);
                }
                if (iterSGS == 1)
                    sgsRes0 = sgsRes;

                if (mpi.rank == 0 && iterSGS % nSgsConsoleCheck == 0)
                    log() << std::scientific << "SGS1 " << std::to_string(iterSGS)
                          << " [" << sgsRes0.transpose() << "] ->"
                          << " [" << sgsRes.transpose() << "] " << std::endl;
            }
        }
        Eigen::VectorXd meanScale;
        if (gmresScale == 1)
        {
            meanScale = eval.settings.refU;
            meanScale(Seq123).setConstant(std::sqrt(meanScale(0) * meanScale(I4))); //! using consistent rho U scale
            // meanScale(I4) = sqr(meanScale(1)) / (meanScale(0) + verySmallReal);
            // meanScale(0) = 0.01;
            // meanScale(Seq123).setConstant(0.1);
            // meanScale(I4) = 1;
        }
        else if (gmresScale == 2)
        {
            eval.EvaluateNorm(meanScale, cx, 1, true, true);
            meanScale(Seq123).setConstant(meanScale(Seq123).norm());
            meanScale(I4) = sqr(meanScale(1)) / (meanScale(0) + verySmallReal);
        }
        else
            meanScale.setOnes(nVars);
        // meanScale(0) = 10;
        TU meanScaleInv = (meanScale.array() + verySmallReal).inverse();

        if (gmresCode != 0)
        {
            DNDS_EULER_SOLVER_GET_TEMP_UDOF(uTemp)
            // !  GMRES
            // !  for gmres solver: A * uinc = rhsinc, rhsinc is average value insdead of cumulated on vol
            gmres.solve(
                [&](TDof &x, TDof &Ax)
                {
                    eval.LUSGSMatrixVec(alphaDiag, cx, x, JDC, Ax);
                    Ax.trans.startPersistentPull();
                    Ax.trans.waitPersistentPull();
                },
                [&](TDof &x, TDof &MLx)
                {
                    // x as rhs, and MLx as uinc
                    MLx.setConstant(0.0), inputIsZero = true; //! start as zero
                    doPrecondition(alphaDiag, x, cx, MLx, uTemp, JDC, sgsRes, inputIsZero, hasLUDone, gridLevel);
                    for (int i = 0; i < sgsIter; i++)
                    {
                        doPrecondition(alphaDiag, x, cx, MLx, uTemp, JDC, sgsRes, inputIsZero, hasLUDone, gridLevel);
                    }
                },
                [&](TDof &a, TDof &b) -> real
                {
                    return a.dot(b, meanScaleInv.array(), meanScaleInv.array());
                },
                cres, cxInc, nGmresIter,
                [&](uint32_t i, real res, real resB) -> bool
                {
                    if (i > 0 && i % nGmresConsoleCheck == 0)
                    {
                        if (mpi.rank == 0)
                        {
                            log() << std::scientific;
                            log() << "GMRES: " << i << " " << resB << " -> " << res << std::endl;
                        }
                    }
                    return false;
                });
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        // the real definition
        template <EulerModel model>
        ,
        // the intellisense friendly definition
        template <>
    )
    void EulerSolver<model>::doPrecondition(real alphaDiag, TDof &cres, TDof &cx, TDof &cxInc, TDof &uTemp,
                                            JacobianDiagBlock<nVarsFixed> &JDC, TU &sgsRes, bool &inputIsZero, bool &hasLUDone, int gridLevel)
    {
        DNDS_assert(pEval);
        auto &eval = *pEval;
        auto initUDOF = [&](ArrayDOFV<nVarsFixed> &uu)
        { vfv->BuildUDof(uu, nVars); };
        // static int nCall{0};
        // nCall++;
        // if (mpi.rank == 0)
        //     std::cout << "doPrecondition nCall " << nCall << fmt::format(" {} ", hasLUDone) << std::endl;
        int jacobiCode =
            gridLevel > 0
                ? config.linearSolverControl.coarseGridLinearSolverControlList.at(gridLevel - 1).jacobiCode
                : config.linearSolverControl.jacobiCode;

        if (jacobiCode <= 1)
        {
            bool useJacobi = jacobiCode == 0;
            eval.UpdateSGS(alphaDiag, cres, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, true, sgsRes);
            if (useJacobi)
                cxInc = uTemp;
            cxInc.trans.startPersistentPull();
            cxInc.trans.waitPersistentPull();
            eval.UpdateSGS(alphaDiag, cres, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, false, sgsRes);
            if (useJacobi)
                cxInc = uTemp;
            cxInc.trans.startPersistentPull();
            cxInc.trans.waitPersistentPull();
            // eval.UpdateLUSGSForward(alphaDiag, cres, cx, cxInc, JDC, cxInc);
            // cxInc.trans.startPersistentPull();
            // cxInc.trans.waitPersistentPull();
            // eval.UpdateLUSGSBackward(alphaDiag, cres, cx, cxInc, JDC, cxInc);
            // cxInc.trans.startPersistentPull();
            // cxInc.trans.waitPersistentPull();
            inputIsZero = false;
        }
        else if (jacobiCode == 2)
        {
            DNDS_EULER_SOLVER_GET_TEMP_UDOF(rhsTemp)
            DNDS_assert_info(config.linearSolverControl.directPrecControl.useDirectPrec, "need to use config.linearSolverControl.directPrecControl.useDirectPrec first !");
            if (!hasLUDone)
                eval.LUSGSMatrixToJacobianLU(alphaDiag, cx, JDC, *JLocalLU), hasLUDone = true;
            for (int iii = 0; iii < 2; iii++)
            {
                eval.LUSGSMatrixSolveJacobianLU(alphaDiag, cres, cx, cxInc, uTemp, rhsTemp, JDC, *JLocalLU, sgsRes);
                uTemp.SwapDataFatherSon(cxInc);
                // cxInc = uTemp;
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
            }
            inputIsZero = false;
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        // the real definition
        template <EulerModel model>
        ,
        // the intellisense friendly definition
        template <>
    )
    void EulerSolver<model>::InitializeRunningEnvironment(EulerSolver<model>::RunningEnvironment &runningEnvironment)
    {
        if (config.timeMarchControl.partitionMeshOnly)
        {
            return;
        }
        // mind we need to get ptr-to-actual-eval into env.pEval,
        // before assigning the ref (ptr), or the ptr is null
        runningEnvironment.pEval = pEval;
        DNDS_EULERSOLVER_RUNNINGENV_GET_REF_LIST

        /*******************************************************/
        /*                   LOGERR STREAM                     */
        /*******************************************************/
        logErr = LogErrInitialize();

        /*******************************************************/
        /*                 PICK ODE SOLVER                     */
        /*******************************************************/

        auto buildDOF = [&](ArrayDOFV<nVarsFixed> &data)
        {
            vfv->BuildUDof(data, nVars);
        };
        auto buildScalar = [&](ArrayDOFV<1> &data)
        {
            vfv->BuildUDof(data, 1);
        };

        if (config.timeMarchControl.steadyQuit)
        {
            if (mpi.rank == 0)
                log() << "Using steady!" << std::endl;
            config.timeMarchControl.odeCode = 1; // To bdf;
            config.timeMarchControl.nTimeStep = 1;
        }
        switch (config.timeMarchControl.odeCode)
        {
        case 0: // esdirk4
            if (mpi.rank == 0)
                log() << "=== ODE: ESDIRK4 " << std::endl;
            ode = std::make_shared<ODE::ImplicitSDIRK4DualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                1); // 1 for esdirk
            break;
        case 101: // sdirk4
            if (mpi.rank == 0)
                log() << "=== ODE: SSP-SDIRK4 " << std::endl;
            ode = std::make_shared<ODE::ImplicitSDIRK4DualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                0);
            break;
        case 1: // BDF2 // Backward Euler
        case 103:
            if (mpi.rank == 0 && config.timeMarchControl.odeCode == 1)
                log() << "=== ODE: BDF2 " << std::endl;
            if (mpi.rank == 0 && config.timeMarchControl.odeCode == 103)
                log() << "=== ODE: Backward Euler " << std::endl;
            ode = std::make_shared<ODE::ImplicitVBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                config.timeMarchControl.odeCode == 1 ? 2 : 1);
            break;
        case 102: // VBDF2
            if (mpi.rank == 0)
                log() << "=== ODE: VBDF2 " << std::endl;
            ode = std::make_shared<ODE::ImplicitVBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                2);
            break;
        case 2: // SSPRK
            if (mpi.rank == 0)
                log() << "=== ODE: SSPRK4 " << std::endl;
            ode = std::make_shared<ODE::ExplicitSSPRK3TimeStepAsImplicitDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                false); // TODO: add local stepping options
            break;
        case 401: // H3S
            if (mpi.rank == 0)
                log() << "=== ODE: Hermite3 (simple jacobian) " << std::endl;
            ode = std::make_shared<ODE::ImplicitHermite3SimpleJacobianDualStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                config.timeMarchControl.odeSetting1 == 0 ? 0.55 : config.timeMarchControl.odeSetting1,
                std::round(config.timeMarchControl.odeSetting2),                                         // Backward Euler Starter
                0,                                                                                       // method
                config.timeMarchControl.odeSetting3 == 0 ? 0.9146 : config.timeMarchControl.odeSetting3, // thetaM1
                std::round(config.timeMarchControl.odeSetting4)                                          // mask
            );
            break;
        default:
            DNDS_assert_info(false, "no such ode code");
        }
        if (config.timeMarchControl.useImplicitPP)
        {
            DNDS_assert(config.timeMarchControl.odeCode == 1 || config.timeMarchControl.odeCode == 102);
        }

        /*******************************************************/
        /*                 INIT GMRES AND PCG                  */
        /*******************************************************/

        if (config.linearSolverControl.gmresCode == 1 ||
            config.linearSolverControl.gmresCode == 2)
            gmres = std::make_unique<tGMRES_u>(
                config.linearSolverControl.nGmresSpace,
                [&](decltype(u) &data)
                {
                    vfv->BuildUDof(data, nVars);
                });

        if (config.implicitReconstructionControl.recLinearScheme == 1)
            gmresRec = std::make_unique<tGMRES_uRec>(
                config.implicitReconstructionControl.nGmresSpace,
                [&](decltype(uRec) &data)
                {
                    vfv->BuildURec(data, nVars);
                });

        if (config.implicitReconstructionControl.recLinearScheme == 2)
            pcgRec = std::make_unique<tPCG_uRec>(
                [&](decltype(uRec) &data)
                {
                    vfv->BuildURec(data, nVars);
                });

        if (config.implicitReconstructionControl.recLinearScheme == 2 || config.timeMarchControl.odeCode == 401)
            pcgRec1 = std::make_unique<tPCG_uRec>(
                [&](decltype(uRec) &data)
                {
                    vfv->BuildURec(data, nVars);
                });

        /*******************************************************/
        /*                 SOLVER MAJOR START                  */
        /*******************************************************/
        tstart = MPI_Wtime();
        tstartInternal = tstart;
        stepCount = 0;

        resBaseC.resize(nVars);
        resBaseCInternal.resize(nVars);
        resBaseC.setConstant(config.convergenceControl.res_base);

        tSimu = 0.0;
        tAverage = 0.0;
        nextTout = std::min(config.outputControl.tDataOut, config.timeMarchControl.tEnd); // ensures the destination time output
        nextStepOut = config.outputControl.nDataOut;
        nextStepOutC = config.outputControl.nDataOutC;
        nextStepRestart = config.outputControl.nRestartOut;
        nextStepRestartC = config.outputControl.nRestartOutC;
        nextStepOutAverage = config.outputControl.nTimeAverageOut;
        nextStepOutAverageC = config.outputControl.nTimeAverageOutC;

        CFLNow = config.implicitCFLControl.CFL;
        ifOutT = false;
        curDtMin = veryLargeReal;
        curDtImplicit = config.timeMarchControl.dtImplicit;
        step = 0;
        gradIsZero = true;

        dtIncreaseCounter = 0;
    }
}