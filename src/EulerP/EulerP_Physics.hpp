#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "EulerP.hpp"
#include "DNDS/JsonUtil.hpp"
#include <cmath>

namespace DNDS::EulerP
{

    struct PhysicsParams
    {
        //! only simple data here allowed
        real gamma = 1.4;
        real mu0 = 1e-100;
        real cp = 1.;
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

        DNDS_DEVICE_CALLABLE real getMuPhy() const
        {
            switch (params.muModel)
            {
            case 0:
                return params.mu0;
            default:
                DNDS_HD_assert(false);
                return 1e-100;
            }
        }

        DNDS_DEVICE_CALLABLE real getMuTot(
            const real *U, const real *DiffU, int nVars,
            const real *UPrim = nullptr, const real *DiffUPrim = nullptr) const
        {
            // ! no rans model
            return getMuPhy();
        }

        DNDS_DEVICE_CALLABLE void Cons2Prim(
            const real *U, real *UPrim, int nVars) const
        {
            real rho = U[0];
            real vSqr = 0.0;
#ifdef __CUDA_ARCH__
#    pragma unroll
#endif
            for (int i = 1; i < nVarsFlow - 1; i++)
                UPrim[i] = U[i] / rho, vSqr += sqr(UPrim[i]);
            real E = U[I4];
            real e = (E - rho * 0.5 * vSqr);
            UPrim[0] = rho;
            UPrim[I4] = e;
            for (int i = nVarsFlow; i < nVars; i++)
                UPrim[i] = U[i] / rho;
        }

        DNDS_DEVICE_CALLABLE void Prim2Cons(
            const real *UPrim, real *U, int nVars) const
        {
            real rho = UPrim[0];
            real vSqr = 0.0;
#ifdef __CUDA_ARCH__
#    pragma unroll
#endif
            for (int i = 1; i < nVarsFlow - 1; i++)
                vSqr += sqr(UPrim[i]), U[i] = UPrim[i] * rho;
            real e = UPrim[I4];
            real E = (e + rho * 0.5 * vSqr);
            U[0] = rho;
            U[I4] = e;
            for (int i = nVarsFlow; i < nVars; i++)
                U[i] = UPrim[i] * rho;
        }

        DNDS_DEVICE_CALLABLE real Prim2Pressure(
            const real *UPrim, int nVars) const
        {
            //! perfect gas here
            return UPrim[I4] / (params.gamma - 1);
        }

        DNDS_DEVICE_CALLABLE real Prim2AcousticSpeed(
            const real *UPrim, int nVars) const
        {
            real p = Prim2Pressure(UPrim, nVars);
            return std::sqrt(params.gamma * p / UPrim[0]);
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
    };

}