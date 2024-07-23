#include "../EulerEvaluator_EvaluateDt.hxx"

#define DNDS_EulerEvaluator_EvaluateDt_INS(model)                                                                      \
    namespace DNDS::Euler                                                                                              \
    {                                                                                                                  \
        template void EulerEvaluator<model>::GetWallDist();                                                            \
        template void EulerEvaluator<model>::EvaluateDt(                                                               \
            ArrayDOFV<1> &dt,                                                                                          \
            ArrayDOFV<nVarsFixed> &u,                                                                                  \
            ArrayRECV<nVarsFixed> &uRec,                                                                               \
            real CFL, real &dtMinall, real MaxDt,                                                                      \
            bool UseLocaldt);                                                                                          \
        template                                                                                                       \
            typename EulerEvaluator<model>::TU                                                                         \
            EulerEvaluator<model>::fluxFace(                                                                           \
                const TU &ULxy,                                                                                        \
                const TU &URxy,                                                                                        \
                const TU &ULMeanXy,                                                                                    \
                const TU &URMeanXy,                                                                                    \
                const TDiffU &DiffUxy,                                                                                 \
                const TDiffU &DiffUxyPrim,                                                                             \
                const TVec &unitNorm,                                                                                  \
                const TVec &vg,                                                                                        \
                const TMat &normBase,                                                                                  \
                TU &FLfix,                                                                                             \
                TU &FRfix,                                                                                             \
                Geom::t_index btype,                                                                                   \
                typename Gas::RiemannSolverType rsType,                                                                \
                index iFace, int ig);                                                                                  \
        template                                                                                                       \
            typename EulerEvaluator<model>::TU                                                                         \
            EulerEvaluator<model>::source(                                                                             \
                const TU &UMeanXy,                                                                                     \
                const TDiffU &DiffUxy,                                                                                 \
                const Geom::tPoint &pPhy,                                                                              \
                TJacobianU &jacobian,                                                                                  \
                index iCell,                                                                                           \
                index ig,                                                                                              \
                int Mode);                                                                                             \
        template                                                                                                       \
            typename EulerEvaluator<model>::TU                                                                         \
            EulerEvaluator<model>::generateBoundaryValue(                                                              \
                TU &ULxy,                                                                                              \
                const TU &ULMeanXy,                                                                                    \
                index iCell, index iFace, int iG,                                                                      \
                const TVec &uNorm,                                                                                     \
                const TMat &normBase,                                                                                  \
                const Geom::tPoint &pPhysics,                                                                          \
                real t,                                                                                                \
                Geom::t_index btype,                                                                                   \
                bool fixUL,                                                                                            \
                int geomMode);                                                                                         \
        template void EulerEvaluator<model>::InitializeOutputPicker(OutputPicker &op, OutputOverlapDataRefs dataRefs); \
    }
