#include "EulerP_Evaluator.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "DNDS/DeviceStorageHelper.hpp"
#include "EulerP/EulerP.hpp"
#include <cstddef>
#include <string>

namespace DNDS::EulerP
{
    template <>
    void Evaluator::RecGradient_impl<DeviceBackend::Host>(
        const ssp<TUDof> &u,
        const ssp<TUGrad> &uGrad,
        const std::vector<ssp<TUScalar>> &uScalar,
        const std::vector<ssp<TUScalarGrad>> &uScalarGrad)
    {
        using namespace Geom;
        static const DeviceBackend B = DeviceBackend::Host;
        deviceViewVector<TUScalar::t_deviceView<B>, B>
            uScalar_v(uScalar.size(), [&](int i)
                      { return uScalar.at(i)->deviceView<B>(); });
        deviceViewVector<TUScalarGrad::t_deviceView<B>, B>
            uScalarGrad_v(uScalarGrad.size(), [&](int i)
                          { return uScalarGrad.at(i)->deviceView<B>(); });

        auto u_view = u->deviceView<B>();
        auto uGrad_view = uGrad->deviceView<B>();
        auto uScalar_view = uScalar_v.deviceView();
        auto uScalarGrad_view = uScalarGrad_v.deviceView();
        auto this_v = this->deviceView<B>(); //! must keep this alive
        EvaluatorDeviceView<B> self_view = this_v;

        auto u_face_bufferL_view = u_face_bufferL.deviceView<B>();
        deviceViewVector<TUScalar::t_deviceView<B>, B>
            uScalar_face_bufferL_v(uScalar_face_bufferL.size(), [&](int i)
                                   { return uScalar_face_bufferL.at(i).deviceView<B>(); });
        auto uScalar_face_bufferL_view = uScalar_face_bufferL_v.deviceView();

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        {
            auto &mesh = self_view.fv.mesh;
            auto &fv = self_view.fv;
            auto &bcHandler = self_view.bcHandler;
            auto &phy = self_view.physics;
            u->trans.waitPersistentPull();
            for (auto &uS : uScalar)
                uS->trans.waitPersistentPull();

            /*********************** */
            // bc handling
            VectorXR fullUOther;
            VectorXR fullU;
            fullU.resize(nVars);
            fullUOther.resize(nVars);
            for (index iBnd = 0; iBnd < mesh.NumBnd(); iBnd++)
            {
                index iFace = mesh.bnd2face(iBnd, 0);
                index iCell = mesh.bnd2cell(iBnd, 0);
                fullU(Seq01234) = u_view[iCell];
                for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                    fullU(iVarS + nVarsFlow) = uScalar_view[iVarS](iCell);
                auto bc = bcHandler.id2bc(mesh.GetFaceZone(iFace));
                bc.apply(fullU.data(), fullUOther.data(), nVars,
                         fv.GetFaceQuadraturePPhys(iFace, -1).data(),
                         fv.GetFaceNorm(iFace, -1).data(),
                         phy);
                u_face_bufferL_view[iCell] = fullUOther(Seq01234);
                for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                    uScalar_face_bufferL_view[iVarS](iFace) = fullUOther(iVarS + nVarsFlow);
            }

            /*********************** */
            // rec
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
            {
                auto grad_flow = uGrad_view.father[iCell];
                for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                    uScalarGrad_view[iVarS].father[iCell].setZero();
                grad_flow.setZero();
                auto c2f = mesh.cell2face[iCell];
                TU uI = u_view[iCell];

                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = mesh.CellFaceOther(iCell, iFace);
                    rowsize if2c = mesh.CellIsFaceBack(iCell, iFace) ? 0 : 1;
                    tPoint norm = fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) *
                                  ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace));

                    TU uOther;
                    uOther.setZero();
                    if (iCellOther != UnInitIndex)
                        uOther = u_view[iCellOther];
                    else
                        uOther = u_face_bufferL_view.father[iFace];

                    grad_flow.noalias() += norm * (uOther - uI).transpose();
                    for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                    {
                        real uI = uScalar_view[iVarS].father(iCell);
                        real uOther = 0.;
                        if (iCellOther != UnInitIndex)
                            uOther = uScalar_view[iVarS](iCell);
                        else
                            uOther = uScalar_face_bufferL_view[iVarS](iCell);
                        uScalarGrad_view[iVarS].father[iCell].noalias() += norm * (uOther - uI);
                    }
                }
            }

            /*********************** */
            // limit
            // TODO
        }
    }

    void Evaluator::RecGradient(const ssp<TUDof> &u,
                                const ssp<TUGrad> &uGrad,
                                const std::vector<ssp<TUScalar>> &uScalar,
                                const std::vector<ssp<TUScalarGrad>> &uScalarGrad)
    {
        DeviceBackend B = this->device();
        checkValidUDof(u, "u", 0, true, B);
        checkValidUDof(uGrad, "uGrad", 0, true, B);
        for (size_t i = 0; i < uScalar.size(); i++)
            checkValidUDof(uScalar[i], "uScalar_" + std::to_string(i), 0, true, B);
        for (size_t i = 0; i < uScalarGrad.size(); i++)
            checkValidUDof(uScalarGrad[i], "uScalarGrad_" + std::to_string(i), 0, true, B);

        // DNDS_assert(u_face_bufferL.father);
        // DNDS_assert(uScalar_face_bufferL.size() >= uScalar.size());
        // for (int i = 0; i < uScalar.size(); i++)
        //     DNDS_assert(uScalar_face_bufferL[i].father);
        PrepareFaceBuffer(uScalar.size());

        if (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
        {
            this->RecGradient_impl<DeviceBackend::Host>(
                u, uGrad, uScalar, uScalarGrad);
        }
        else
            DNDS_assert(false);
    }

    void Evaluator::EstEigenDt(const ssp<TUDof> &u,
                               const ssp<TUGrad> &uGrad,
                               const std::vector<ssp<TUScalar>> &uScalar,
                               const std::vector<ssp<TUScalarGrad>> &uScalarGrad,
                               const ssp<TUScalar> &faceLamEst,
                               const ssp<TUScalar> &faceLamVisEst,
                               const ssp<TUScalar> &dt)
    {
    }

    void Evaluator::RHS2nd(const ssp<TUDof> &u,
                           const ssp<TUGrad> &uGrad,
                           const std::vector<ssp<TUScalar>> &uScalar,
                           const std::vector<ssp<TUScalarGrad>> &uScalarGrad,
                           const ssp<TUScalar> &faceLamEst,
                           const ssp<TUScalar> &faceLamVisEst,
                           const ssp<TUDof> &rhs,
                           const ssp<TUDof> &flux_face,
                           const ssp<TUDof> &source)
    {
    }
}