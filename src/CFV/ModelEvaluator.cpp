#include "ModelEvaluator.hpp"

namespace DNDS::CFV
{
    void ModelEvaluator::EvaluateRHS(
        tUDof<nVarsFixed> &rhs, tUDof<nVarsFixed> &u, tURec<nVarsFixed> &uRec, real t,
        const ModelEvaluator::EvaluateRHSOptions &options)
    {
        using namespace Geom;
        int cnvars = nVarsFixed;
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto SeqG012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

        bool direct2ndRec = options.direct2ndRec;
        bool direct2ndRec1stConv = options.direct2ndRec1stConv;

        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            rhs[iCell].setZero();

        if (direct2ndRec)
        {
            vfv->DoReconstruction2ndGrad(uGradBuf, u, this->get_FBoundary(t), 1 /* green gauss*/);
            uGradBuf.trans.startPersistentPull();
            uGradBuf.trans.waitPersistentPull();
        }

        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            Elem::Quadrature gFace = direct2ndRec ? vfv->GetFaceQuadO1(iFace) : vfv->GetFaceQuad(iFace);
            Eigen::Matrix<real, nVarsFixed, 1, Eigen::ColMajor> fluxEs(cnvars, 1);
            fluxEs.setZero();

            auto faceBndID = mesh->GetFaceZone(iFace);
            gFace.IntegrationSimple(
                fluxEs,
                [&](decltype(fluxEs) &finc, int iG, real w)
                {
                    int iGQ = direct2ndRec ? -1 : iG;
                    TVec unitNorm = vfv->GetFaceNorm(iFace, iGQ)(Seq012);
                    TMat normBase = Geom::NormBuildLocalBaseV<dim>(unitNorm);

                    TU ULxy = u[f2c[0]];
                    if (direct2ndRec && !direct2ndRec1stConv)
                        ULxy += uGradBuf[f2c[0]].transpose() * (vfv->GetFaceQuadraturePPhysFromCell(iFace, f2c[0], 0, -1) - vfv->GetCellQuadraturePPhys(f2c[0], -1))(SeqG012);
                    else if (!direct2ndRec1stConv)
                        ULxy += (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iGQ, std::array<int, 1>{0}, 1) *
                                 uRec[f2c[0]])
                                    .transpose();
                    TU URxy;
                    TDiffU GradULxy, GradURxy;
                    GradULxy.setZero(), GradURxy.setZero();

                    if (direct2ndRec && !direct2ndRec1stConv)
                        GradULxy(SeqG012, Eigen::all) = uGradBuf[f2c[0]];
                    else if (!direct2ndRec1stConv)
                        GradULxy({0, 1}, Eigen::all) =
                            vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iGQ, std::array<int, 2>{1, 2}, 3) *
                            uRec[f2c[0]];

                    real minVol = vfv->GetCellVol(f2c[0]);
                    real distBary = veryLargeReal;
                    if (f2c[1] != UnInitIndex)
                    {
                        URxy = u[f2c[1]];
                        if (direct2ndRec && !direct2ndRec1stConv)
                            URxy += uGradBuf[f2c[1]].transpose() * (vfv->GetFaceQuadraturePPhysFromCell(iFace, f2c[1], 1, -1) - vfv->GetCellQuadraturePPhys(f2c[1], -1))(SeqG012);
                        else if (!direct2ndRec1stConv)
                            URxy += (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iGQ, std::array<int, 1>{0}, 1) *
                                     uRec[f2c[1]])
                                        .transpose();

                        if (direct2ndRec && !direct2ndRec1stConv)
                            GradURxy(SeqG012, Eigen::all) = uGradBuf[f2c[1]];
                        else if (!direct2ndRec1stConv)
                            GradURxy({0, 1}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iGQ, std::array<int, 2>{1, 2}, 3) *
                                uRec[f2c[1]];
                        distBary = (vfv->GetOtherCellBaryFromCell(f2c[0], f2c[1], iFace) - vfv->GetCellBary(f2c[0])).norm();
                        minVol = std::min(minVol, vfv->GetCellVol(f2c[1]));
                    }
                    else
                    {
                        GradURxy = GradULxy;
                        URxy = generateBoundaryValue(
                            ULxy, u[f2c[0]], f2c[0], iFace, iGQ,
                            unitNorm,
                            normBase,
                            vfv->GetFaceQuadraturePPhys(iFace, iGQ),
                            t,
                            mesh->GetFaceZone(iFace), true, 0);
                        distBary = (vfv->GetFaceQuadraturePPhysFromCell(iFace, f2c[0], 0, -1) - vfv->GetCellBary(f2c[0])).norm() * 2.;
                    }

                    real distGRP = minVol / vfv->GetFaceArea(iFace) * 2;

                    if (direct2ndRec1stConv)
                        distGRP = distBary;
                    TDiffU GradUMeanXy = (GradURxy + GradULxy) * 0.5;
                    GradUMeanXy += (1.0 / distGRP) *
                                   (unitNorm * (URxy - ULxy).transpose());
                    real an = unitNorm(0) * settings.ax + unitNorm(1) * settings.ay;
                    finc = -(an * URxy + an * ULxy) * 0.5 + 0.5 * std::abs(an) * (URxy - ULxy);
                    // viscous:
                    finc += GradUMeanXy.transpose() * unitNorm * settings.sigma;
                    finc *= vfv->GetFaceJacobiDet(iFace, iG);
                });
            rhs[f2c[0]] += fluxEs / vfv->GetCellVol(f2c[0]);
            if (f2c[1] != UnInitIndex)
                rhs[f2c[1]] += -fluxEs / vfv->GetCellVol(f2c[1]);
        }
    }
}