#include "../EulerEvaluator.hxx"

#define DNDS_EulerEvaluator_INS(model)                                                        \
    namespace DNDS::Euler                                                                     \
    {                                                                                         \
        template void EulerEvaluator<model>::LUSGSMatrixInit(                                 \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            JacobianDiagBlock<nVarsFixed> &JSource,                                           \
            ArrayDOFV<1> &dTau, real dt, real alphaDiag,                                      \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayRECV<nVarsFixed> &uRec,                                                      \
            int jacobianCode,                                                                 \
            real t);                                                                          \
                                                                                              \
        template void EulerEvaluator<model>::LUSGSMatrixVec(                                  \
            real alphaDiag,                                                                   \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayDOFV<nVarsFixed> &uInc,                                                      \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            ArrayDOFV<nVarsFixed> &AuInc);                                                    \
                                                                                              \
        template void EulerEvaluator<model>::LUSGSMatrixToJacobianLU(                         \
            real alphaDiag,                                                                   \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            JacobianLocalLU<nVarsFixed> &jacLU);                                              \
                                                                                              \
        template void EulerEvaluator<model>::UpdateLUSGSForward(                              \
            real alphaDiag,                                                                   \
            ArrayDOFV<nVarsFixed> &rhs,                                                       \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayDOFV<nVarsFixed> &uInc,                                                      \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            ArrayDOFV<nVarsFixed> &uIncNew);                                                  \
                                                                                              \
        template void EulerEvaluator<model>::UpdateLUSGSBackward(                             \
            real alphaDiag,                                                                   \
            ArrayDOFV<nVarsFixed> &rhs,                                                       \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayDOFV<nVarsFixed> &uInc,                                                      \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            ArrayDOFV<nVarsFixed> &uIncNew);                                                  \
                                                                                              \
        template void EulerEvaluator<model>::UpdateSGS(                                       \
            real alphaDiag,                                                                   \
            ArrayDOFV<nVarsFixed> &rhs,                                                       \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayDOFV<nVarsFixed> &uInc,                                                      \
            ArrayDOFV<nVarsFixed> &uIncNew,                                                   \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            bool forward, TU &sumInc);                                                        \
        template void EulerEvaluator<model>::UpdateSGSWithRec(                                \
            real alphaDiag,                                                                   \
            ArrayDOFV<nVarsFixed> &rhs,                                                       \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayRECV<nVarsFixed> &uRec,                                                      \
            ArrayDOFV<nVarsFixed> &uInc,                                                      \
            ArrayRECV<nVarsFixed> &uRecInc,                                                   \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            bool forward, TU &sumInc);                                                        \
                                                                                              \
        template void EulerEvaluator<model>::LUSGSMatrixSolveJacobianLU(                      \
            real alphaDiag,                                                                   \
            ArrayDOFV<nVarsFixed> &rhs,                                                       \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayDOFV<nVarsFixed> &uInc,                                                      \
            ArrayDOFV<nVarsFixed> &uIncNew,                                                   \
            ArrayDOFV<nVarsFixed> &bBuf,                                                      \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                             \
            JacobianLocalLU<nVarsFixed> &jacLU,                                               \
            TU &sumInc);                                                                      \
                                                                                              \
        template void EulerEvaluator<model>::InitializeUDOF(ArrayDOFV<nVarsFixed> &u);        \
                                                                                              \
        template void EulerEvaluator<model>::FixUMaxFilter(                                   \
            ArrayDOFV<nVarsFixed> &u);                                                        \
                                                                                              \
        template void EulerEvaluator<model>::TimeAverageAddition(                             \
            ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &wAveraged, real dt, real &tCur); \
        template void EulerEvaluator<model>::MeanValueCons2Prim(                              \
            ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &w);                              \
        template void EulerEvaluator<model>::MeanValuePrim2Cons(                              \
            ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &u);                              \
                                                                                              \
        template void EulerEvaluator<model>::EvaluateNorm(                                    \
            Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P, bool volWise); \
                                                                                              \
        template void EulerEvaluator<model>::EvaluateRecNorm(                                 \
            Eigen::Vector<real, -1> &res,                                                     \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayRECV<nVarsFixed> &uRec,                                                      \
            index P,                                                                          \
            bool compare,                                                                     \
            const tFCompareField &FCompareField,                                              \
            const tFCompareFieldWeight &FCompareFieldWeight,                                  \
            real t);                                                                          \
                                                                                              \
        template void EulerEvaluator<model>::EvaluateURecBeta(                                \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayRECV<nVarsFixed> &uRec,                                                      \
            ArrayDOFV<1> &uRecBeta, index &nLim, real &betaMin, int flag);                    \
                                                                                              \
        template bool EulerEvaluator<model>::AssertMeanValuePP(                               \
            ArrayDOFV<nVarsFixed> &u, bool panic);                                            \
                                                                                              \
        template void EulerEvaluator<model>::EvaluateCellRHSAlpha(                            \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayRECV<nVarsFixed> &uRec,                                                      \
            ArrayDOFV<1> &uRecBeta,                                                           \
            ArrayDOFV<nVarsFixed> &rhs,                                                       \
            ArrayDOFV<1> &cellRHSAlpha, index &nLim, real &alphaMin, real relax,              \
            int flag);                                                                        \
                                                                                              \
        template void EulerEvaluator<model>::EvaluateCellRHSAlphaExpansion(                   \
            ArrayDOFV<nVarsFixed> &u,                                                         \
            ArrayRECV<nVarsFixed> &uRec,                                                      \
            ArrayDOFV<1> &uRecBeta,                                                           \
            ArrayDOFV<nVarsFixed> &res,                                                       \
            ArrayDOFV<1> &cellRHSAlpha, index &nLim, real alphaMin);                          \
        template void EulerEvaluator<model>::MinSmoothDTau(                                   \
            ArrayDOFV<1> &dTau, ArrayDOFV<1> &dTauNew);                                       \
    }
