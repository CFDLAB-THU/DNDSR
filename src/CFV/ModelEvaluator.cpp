#include "ModelEvaluator.hpp"


namespace DNDS::CFV
{
    void ModelEvaluator::EvaluateRHS(tUDof<nVarsFixed> &rhs, tUDof<nVarsFixed> &u, tURec<nVarsFixed> &uRec, real t)
    {
        using namespace Geom;
        int cnvars = nVarsFixed;
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto SeqG012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            rhs[iCell].setZero();

        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            Elem::Quadrature gFace = vfv->GetFaceQuad(iFace);
            Eigen::Matrix<real, nVarsFixed, 1, Eigen::ColMajor> fluxEs(cnvars, 1);
            fluxEs.setZero();

            auto faceBndID = mesh->GetFaceZone(iFace);
            gFace.IntegrationSimple(
                fluxEs,
                [&](decltype(fluxEs) &finc, int iG, real w)
                {
                    int iGQ = iG;
                    TVec unitNorm = vfv->GetFaceNorm(iFace, iGQ)(Seq012);
                    TMat normBase = Geom::NormBuildLocalBaseV<dim>(unitNorm);

                    TU ULxy = u[f2c[0]];
                    ULxy += (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iGQ, std::array<int, 1>{0}, 1) *
                             uRec[f2c[0]])
                                .transpose();
                    TU URxy;
                    TDiffU GradULxy, GradURxy;
                    GradULxy.setZero(), GradURxy.setZero();

                    real minVol = vfv->GetCellVol(f2c[0]);
                    real distBary = veryLargeReal;
                    if (f2c[1] != UnInitIndex)
                    {
                        URxy = u[f2c[1]];
                        URxy += (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iGQ, std::array<int, 1>{0}, 1) *
                                 uRec[f2c[1]])
                                    .transpose();

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
                    TDiffU GradUMeanXy = (GradURxy + GradULxy) * 0.5;
                    GradUMeanXy += (1.0 / distGRP) *
                                   (unitNorm * (URxy - ULxy).transpose());
                    real an = unitNorm(0) * settings.ax + unitNorm(1) * settings.ay;
                    finc = -(an * URxy + an * ULxy) * 0.5 + 0.5 * std::abs(an) * (URxy - ULxy);
                    // TODO: viscous
                    finc *= vfv->GetFaceJacobiDet(iFace, iG);
                });
            rhs[f2c[0]] += fluxEs / vfv->GetCellVol(f2c[0]);
            if (f2c[1] != UnInitIndex)
                rhs[f2c[1]] += -fluxEs / vfv->GetCellVol(f2c[1]);
        }
    }
}