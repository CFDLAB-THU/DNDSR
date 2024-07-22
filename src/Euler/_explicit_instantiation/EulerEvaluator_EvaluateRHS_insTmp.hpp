#include "../EulerEvaluator_EvaluateRHS.hxx"

#define DNDS_EulerEvaluator_EvaluateRHS_INS(model)        \
    namespace DNDS::Euler                                 \
    {                                                     \
        template void EulerEvaluator<model>::EvaluateRHS( \
            ArrayDOFV<nVarsFixed> &rhs,                   \
            JacobianDiagBlock<nVarsFixed> &JSource,       \
            ArrayDOFV<nVarsFixed> &u,                     \
            ArrayRECV<nVarsFixed> &uRecUnlim,             \
            ArrayRECV<nVarsFixed> &uRec,                  \
            ArrayDOFV<1> &uRecBeta,                       \
            ArrayDOFV<1> &cellRHSAlpha,                   \
            bool onlyOnHalfAlpha,                         \
            real t,                                       \
            uint64_t flags);                              \
    }
