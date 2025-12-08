#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "EulerP.hpp"
#include "DNDS/JsonUtil.hpp"
#include <cmath>
#include "Geom/Geometric.hpp"

namespace DNDS::EulerP
{

    struct PhysicsParams
    {
        //! only simple data here allowed
        real gamma = 1.4;
        real mu0 = 1e-100;
        real cp = 1.;
        real cv = 1.;
        real Rg = 1.;
        real TRef = 273.15;

        int muModel = 0;
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(
            PhysicsParams,
            gamma, mu0, cp, Rg, TRef,
            muModel);
    };

    template <DeviceBackend B>
    struct PhysicsDeviceView
    {
        vector_DeviceView<B, real, int32_t> reference_values;
        PhysicsParams params;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(PhysicsDeviceView, PhysicsDeviceView)

        template <class tUPrim, class tDiffUPrim>
        DNDS_DEVICE_CALLABLE real getMuTot(
            tUPrim &&UPrim, tDiffUPrim &&DiffUPrim, int nVars,
            real p, real T) const
        {
            // ! no rans model
            real muPhy = params.mu0;
            return muPhy;
        }

        template <class tUPrim, class tU>
        DNDS_DEVICE_CALLABLE void Cons2Prim(
            tU &&U, tUPrim &&UPrim, int nVars) const
        {
            real rho = U(0);
            real vSqr = 0.0;
#ifdef __CUDA_ARCH__
#    pragma unroll
#endif
            for (int i = 1; i < nVarsFlow - 1; i++)
                UPrim(i) = U(i) / rho, vSqr += sqr(UPrim(i));
            real E = U(I4);
            real e = (E - rho * 0.5 * vSqr);
            UPrim(0) = rho;
            UPrim(I4) = e;
            for (int i = nVarsFlow; i < nVars; i++)
                UPrim(i) = U(i) / rho;
        }

        template <class tU, class tUPrim, class tDiffU, class tDiffUPrim>
        DNDS_DEVICE_CALLABLE void Cons2PrimDiff(
            tU &&U, tUPrim &&UPrim, tDiffU &&DiffU, tDiffUPrim &&DiffUPrim, int nVars) const
        {

            for (int d = 0; d < 3; d++)
                DiffUPrim(d, 0) = DiffU(d, 0);
            for (int i = 1; i < I4; i++)
                for (int d = 0; d < 3; d++)
                    DiffUPrim(d, i) = (DiffU(d, i) - DiffUPrim(d, 0) * UPrim(i)) / UPrim(0);
            real vSqr = 0.0;
            real rho = UPrim(0);
            for (int i = 1; i < nVarsFlow - 1; i++)
                vSqr += sqr(UPrim(i));

            // nabla(E - rho * 0.5 * dot(v,v)) =
            // nabla(E) - 0.5 nabla(rho) * dot(v,v) -
            // rho * dot(nabla(v), v)
            for (int d = 0; d < 3; d++)
            {
                DiffUPrim(d, I4) = DiffU(d, I4) - 0.5 * DiffUPrim(d, 0) * vSqr;
                for (int i = 1; i < nVarsFlow - 1; i++)
                    DiffUPrim(d, I4) -= rho * DiffUPrim(d, i) * UPrim(i);
            }
            for (int i = nVarsFlow; i < nVars; i++)
                for (int d = 0; d < 3; d++)
                    DiffUPrim(d, i) = (DiffU(d, i) - DiffUPrim(d, 0) * UPrim(i)) / UPrim(0);
        }

        template <class tUPrim, class tU>
        DNDS_DEVICE_CALLABLE void Prim2Cons(
            tUPrim &&UPrim, tU &&U, int nVars) const
        {
            real rho = UPrim(0);
            real vSqr = 0.0;
#ifdef __CUDA_ARCH__
#    pragma unroll
#endif
            for (int i = 1; i < nVarsFlow - 1; i++)
                vSqr += sqr(UPrim(i)), U(i) = UPrim(i) * rho;
            real e = UPrim(I4);
            real E = (e + rho * 0.5 * vSqr);
            U(0) = rho;
            U(I4) = E;
            for (int i = nVarsFlow; i < nVars; i++)
                U(i) = UPrim(i) * rho;
        }

        template <class tU>
        DNDS_DEVICE_CALLABLE real Cons2EInternal(
            tU &&U, int nVars) const
        {
            real rho = U(0);
            real rvSqr = 0.0;
#ifdef __CUDA_ARCH__
#    pragma unroll
#endif
            for (int i = 1; i < nVarsFlow - 1; i++)
                rvSqr += sqr(U(i));
            rvSqr /= rho;
            real E = U(I4);
            real e = (E - 0.5 * rvSqr);
            return e;
        }

        template <class tUPrim>
        DNDS_DEVICE_CALLABLE real Prim2Pressure(
            tUPrim &&UPrim, int nVars, real T) const
        {
            //! perfect gas here
            return UPrim(I4) * (params.gamma - 1);
        }

        template <class tUPrim>
        DNDS_DEVICE_CALLABLE auto Prim2GammaAcousticSpeed(
            tUPrim &&UPrim, int nVars, real p) const
        {
            return std::make_tuple(params.gamma, std::sqrt(params.gamma * p / UPrim(0)));
        }

        template <class tUPrim>
        DNDS_DEVICE_CALLABLE real Prim2Temperature(
            tUPrim &&UPrim, int nVars) const
        {
            //! perfect gas here
            return UPrim(I4) / UPrim(0) / params.cp;
        }

        template <class tU>
        DNDS_DEVICE_CALLABLE real Pressure2Enthalpy(
            tU &&U, int nVars, real p) const
        {
            //! perfect gas here
            return (U(I4) + p) / U(0);
        }

        template <class tU>
        DNDS_DEVICE_CALLABLE real Enthalpy2Pressure(
            tU &&U, int nVars, real H) const
        {
            //! perfect gas here
            return H * U(0) - U(I4);
        }
    };

    struct Physics
    {
        host_device_vector<real> reference_values;
        PhysicsParams params;
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(
            Physics,
            reference_values,
            params)

        void to_host()
        {
            reference_values.to_host();
        }

        void to_device(DeviceBackend B)
        {
            reference_values.to_device(B);
        }

        DeviceBackend device()
        {
            return reference_values.device();
        }

        template <DeviceBackend B>
        using t_deviceView = PhysicsDeviceView<B>;

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DNDS_assert(reference_values.device() == B ||
                        (B == DeviceBackend::Host && reference_values.device() == DeviceBackend::Unknown));
            return t_deviceView<B>{
                reference_values.deviceView<B, int32_t>(),
                params};
        }

        void setPerfectGas(real Rg, real gamma)
        {
            params.Rg = Rg;
            params.gamma = gamma;
            params.cp = Rg * params.gamma / (params.gamma - 1);
            params.cv = params.cp / params.gamma;
        }
    };

    template <typename TU, typename TF, class TVecN>
    DNDS_DEVICE_CALLABLE inline void GasInviscidFlux_XY(TU &&U, int nVars, real vn, real vgn, TVecN &&n,
                                                        real p, TF &F)
    {
        for (int i = 0; i < nVars; i++)
            F(i) += U(i) * (vn - vgn);
        for (int d = 0; d < 3; d++)
            F(d + 1) += p * n(d);
        F(I4) += vn * p;
    }
}