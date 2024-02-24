#include "../VariationalReconstruction_LimiterProcedure.hxx"

#define DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS(dim, nVarsFixed)                     \
    namespace DNDS::CFV                                                                          \
    {                                                                                            \
        template void VariationalReconstruction<dim>::DoCalculateSmoothIndicator<nVarsFixed, 2>( \
            tScalarPair & si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,                     \
            const std::array<int, 2> &varsSee);                                                  \
                                                                                                 \
        template void VariationalReconstruction<dim>::DoCalculateSmoothIndicatorV1<nVarsFixed>(  \
            tScalarPair & si, tURec<nVarsFixed> &uRec, tUDof<nVarsFixed> &u,                     \
            const Eigen::Vector<real, nVarsFixed> &varsSee,                                      \
            const TFPost<nVarsFixed> &FPost);                                                    \
                                                                                                 \
        template void VariationalReconstruction<dim>::DoLimiterWBAP_C(                           \
            tUDof<nVarsFixed> &u,                                                                \
            tURec<nVarsFixed> &uRec,                                                             \
            tURec<nVarsFixed> &uRecNew,                                                          \
            tURec<nVarsFixed> &uRecBuf,                                                          \
            tScalarPair &si,                                                                     \
            bool ifAll,                                                                          \
            const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,                         \
            bool putIntoNew);                                                                    \
                                                                                                 \
        template void VariationalReconstruction<dim>::DoLimiterWBAP_3(                           \
            tUDof<nVarsFixed> &u,                                                                \
            tURec<nVarsFixed> &uRec,                                                             \
            tURec<nVarsFixed> &uRecNew,                                                          \
            tURec<nVarsFixed> &uRecBuf,                                                          \
            tScalarPair &si,                                                                     \
            bool ifAll,                                                                          \
            const tFMEig<nVarsFixed> &FM, const tFMEig<nVarsFixed> &FMI,                         \
            bool putIntoNew);                                                                    \
    }
