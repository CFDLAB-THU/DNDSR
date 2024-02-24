#include "../VariationalReconstruction_Reconstruction.hxx"

#define DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS(dim, nVarsFixed)                  \
    namespace DNDS::CFV                                                                     \
    {                                                                                       \
        template void VariationalReconstruction<dim>::DoReconstruction2nd<nVarsFixed>(      \
            tURec<nVarsFixed> & uRec,                                                       \
            tUDof<nVarsFixed> &u,                                                           \
            const TFBoundary<nVarsFixed> &FBoundary,                                        \
            int method);                                                                    \
                                                                                            \
        template void VariationalReconstruction<dim>::DoReconstructionIter<nVarsFixed>(     \
            tURec<nVarsFixed> & uRec,                                                       \
            tURec<nVarsFixed> &uRecNew,                                                     \
            tUDof<nVarsFixed> &u,                                                           \
            const TFBoundary<nVarsFixed> &FBoundary,                                        \
            bool putIntoNew,                                                                \
            bool recordInc);                                                                \
                                                                                            \
        template void VariationalReconstruction<dim>::DoReconstructionIterDiff<nVarsFixed>( \
            tURec<nVarsFixed> & uRec,                                                       \
            tURec<nVarsFixed> &uRecDiff,                                                    \
            tURec<nVarsFixed> &uRecNew,                                                     \
            tUDof<nVarsFixed> &u,                                                           \
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff);                               \
                                                                                            \
        template void VariationalReconstruction<dim>::                                      \
            DoReconstructionIterSOR<nVarsFixed>(                                            \
                tURec<nVarsFixed> & uRec,                                                   \
                tURec<nVarsFixed> &uRecInc,                                                 \
                tURec<nVarsFixed> &uRecNew,                                                 \
                tUDof<nVarsFixed> &u,                                                       \
                const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,                            \
                bool reverse);                                                              \
    }
