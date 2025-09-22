#pragma once

#include <nlohmann/json.hpp>

#include "VariationalReconstruction.hpp"

namespace DNDS::CFV
{
    struct ModelSettings
    {
        real ax = 1.0;
        real ay = 0.0;
        real sigma = 0.0;

        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
            ModelSettings,
            ax, ay,
            sigma)
    };

    class ModelEvaluator
    {
    public:
        static const int nVarsFixed = 1;
        static const int dim = 2;
        using tvfv = CFV::VariationalReconstruction<dim>;

    private:
        ssp<Geom::UnstructuredMesh> mesh;
        ssp<tvfv> vfv;
        ModelSettings settings;

        tUGrad<nVarsFixed, dim> uGradBuf;

    public:
        using TVec = Eigen::Matrix<real, dim, 1>;
        using TMat = Eigen::Matrix<real, dim, dim>;
        using TU = Eigen::Matrix<real, nVarsFixed, 1>;
        using TDiffU = Eigen::Matrix<real, dim, nVarsFixed>;
        using Tvfv_FBoundary = typename tvfv::TFBoundary<1>;

        ModelEvaluator(decltype(mesh) mesh_, decltype(vfv) vfv_,
                       const ModelSettings &settings_)
            : mesh(mesh_), vfv(vfv_), settings(settings_)
        {
            vfv->BuildUGrad(uGradBuf, nVarsFixed);
        }

        ModelSettings &get_settings() { return settings; }

        Tvfv_FBoundary get_FBoundary(real t)
        {

            return [this, t](const auto &u, const auto &uMean, index iCell, index iFace, int iG,
                             const Geom::tPoint &norm, const Geom::tPoint &pPhy, Geom::t_index fType)
            {
                static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
                TU uF = u;
                TVec normV = norm(Seq012);
                return generateBoundaryValue(
                    uF, uMean,
                    iCell, iFace, iG,
                    normV, Geom::NormBuildLocalBaseV<dim>(normV),
                    pPhy,
                    t, fType, false, 1);
            };
        }

        TU generateBoundaryValue(TU &ULxy, //! warning, possible that UL is also modified
                                 const TU &ULMeanXy,
                                 index iCell, index iFace, int iG,
                                 const TVec &uNorm,
                                 const TMat &normBase,
                                 const Geom::tPoint &pPhysics,
                                 real t,
                                 Geom::t_index btype,
                                 bool fixUL = false,
                                 int geomMode = 0,
                                 int linMode = 0)
        {
            return TU{0.0};
        }

        struct EvaluateRHSOptions
        {
            bool direct2ndRec = false;
            bool direct2ndRec1stConv = false;
            EvaluateRHSOptions() {}
        };

        void EvaluateRHS(tUDof<nVarsFixed> &rhs, tUDof<nVarsFixed> &u, tURec<nVarsFixed> &uRec, real t,
                         const EvaluateRHSOptions &options = EvaluateRHSOptions{});

        void DoReconstructionIter(tURec<nVarsFixed> &uRec,
                                  tURec<nVarsFixed> &uRecNew,
                                  tUDof<nVarsFixed> &u,
                                  real t,
                                  bool putIntoNew = false,
                                  bool recordInc = false,
                                  bool uRecIsZero = false)
        {
            vfv->DoReconstructionIter(uRec, uRecNew, u,
                                      get_FBoundary(t),
                                      putIntoNew, recordInc, uRecIsZero);
        }
    };
}