/**
 * @file EulerP_Physics.hpp
 * @brief Physics model definitions for the EulerP module: gas properties, state conversions,
 *        and inviscid flux computation.
 *
 * Provides:
 * - `PhysicsParams:` POD struct of gas parameters (gamma, viscosity, specific heats), JSON-serializable.
 * - `PhysicsDeviceView:` Device-callable view with conservative/primitive conversions and thermodynamic relations.
 * - `Physics:` Host-side physics object managing device transfer of reference values.
 * - `GasInviscidFlux_XY:` Device-callable inviscid flux projected onto a face normal.
 *
 * @note In EulerP, the primitive state stores @b internal @b energy (rho*e) at index I4,
 *       NOT pressure. This differs from the Euler module convention.
 */
#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "DNDS/IdealGasPhysics.hpp"
#include "EulerP.hpp"
#include "DNDS/Serializer/JsonUtil.hpp"
#include <cmath>
#include "Geom/Geometric.hpp"

namespace DNDS::EulerP
{

    /**
     * @brief POD struct holding gas physical properties for the ideal gas model.
     *
     * All members are simple scalar types suitable for device transfer.
     * JSON-serializable via nlohmann_json intrusive macros.
     */
    struct PhysicsParams
    {
        //! only simple data here allowed
        real gamma = 1.4;    ///< Ratio of specific heats (Cp/Cv). Default: 1.4 for air.
        real mu0 = 1e-100;   ///< Dynamic viscosity coefficient. Default: effectively inviscid.
        real cp = 1.;        ///< Specific heat at constant pressure.
        real cv = 1.;        ///< Specific heat at constant volume.
        real Rg = 1.;        ///< Specific gas constant (Cp - Cv).
        real TRef = 273.15;  ///< Reference temperature [K] for viscosity models.

        int muModel = 0;     ///< Viscosity model selector (0 = constant viscosity).
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(
            PhysicsParams,
            gamma, mu0, cp, Rg, TRef,
            muModel);
    };

    /**
     * @brief Device-callable view of physics parameters providing thermodynamic operations.
     *
     * Wraps a device-resident view of reference values and a copy of `PhysicsParams.`
     * All methods are `DNDS_DEVICE_CALLABLE` for use in both Host and CUDA kernels.
     *
     * @note Primitive state convention: prim[0] = rho, prim[1..3] = velocity, prim[I4] = internal energy (rho*e).
     *       This differs from the Euler module where prim[I4] stores pressure.
     *
     * @tparam B Device backend (Host or CUDA).
     */
    template <DeviceBackend B>
    struct PhysicsDeviceView
    {
        vector_DeviceView<B, real, int32_t> reference_values; ///< Device-resident reference values (e.g., freestream quantities).
        PhysicsParams params;                                  ///< Gas physical parameters (copied to device).

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(PhysicsDeviceView, PhysicsDeviceView)

        /**
         * @brief Computes the total dynamic viscosity.
         *
         * Currently returns only the physical viscosity (no RANS turbulence model contribution).
         *
         * @tparam tUPrim Primitive state vector type (deduced).
         * @tparam tDiffUPrim Primitive state gradient type (deduced).
         * @param UPrim Primitive state vector (unused in current constant-viscosity model).
         * @param DiffUPrim Primitive gradient (unused in current constant-viscosity model).
         * @param nVars Total number of variables (flow + turbulence).
         * @param p Pressure (unused in current model).
         * @param T Temperature (unused in current model).
         * @return Total dynamic viscosity mu_total = mu_physical.
         */
        template <class tUPrim, class tDiffUPrim>
        DNDS_DEVICE_CALLABLE real getMuTot(
            tUPrim &&UPrim, tDiffUPrim &&DiffUPrim, int nVars,
            real p, real T) const
        {
            // ! no rans model
            real muPhy = params.mu0;
            return muPhy;
        }

        /**
         * @brief Converts conservative state to primitive state.
         *
         * Computes primitive variables from conservative variables:
         * - prim[0] = rho (density, copied directly)
         * - prim[1..3] = velocity components (U_i / rho)
         * - prim[I4] = internal energy e = E - 0.5 * rho * v^2
         * - prim[i >= nVarsFlow] = U_i / rho (specific turbulence quantities)
         *
         * @note prim[I4] stores internal energy (rho*e), NOT pressure.
         *
         * @tparam tUPrim Primitive state vector type (deduced).
         * @tparam tU Conservative state vector type (deduced).
         * @param U Input conservative state vector.
         * @param[out] UPrim Output primitive state vector.
         * @param nVars Total number of variables (flow + turbulence).
         */
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

        /**
         * @brief Converts conservative variable gradients to primitive variable gradients.
         *
         * Given conservative state @p U, primitive state @p UPrim, and conservative gradient @p DiffU,
         * computes the corresponding primitive gradient @p DiffUPrim using the chain rule:
         * - nabla(v_i) = (nabla(rho*v_i) - nabla(rho) * v_i) / rho
         * - nabla(e) = nabla(E) - 0.5 * nabla(rho) * v^2 - rho * dot(nabla(v), v)
         *
         * @tparam tU Conservative state type (deduced).
         * @tparam tUPrim Primitive state type (deduced).
         * @tparam tDiffU Conservative gradient type (deduced), 3 x nVars.
         * @tparam tDiffUPrim Primitive gradient type (deduced), 3 x nVars.
         * @param U Conservative state vector.
         * @param UPrim Primitive state vector (must be pre-computed via Cons2Prim).
         * @param DiffU Conservative variable spatial gradients (3 rows = x,y,z directions).
         * @param[out] DiffUPrim Output primitive variable spatial gradients.
         * @param nVars Total number of variables (flow + turbulence).
         */
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

        /**
         * @brief Converts primitive state to conservative state.
         *
         * Inverse of Cons2Prim:
         * - cons[0] = rho
         * - cons[1..3] = rho * v_i
         * - cons[I4] = E = e + 0.5 * rho * v^2 (total energy)
         * - cons[i >= nVarsFlow] = rho * prim_i
         *
         * @tparam tUPrim Primitive state vector type (deduced).
         * @tparam tU Conservative state vector type (deduced).
         * @param UPrim Input primitive state vector.
         * @param[out] U Output conservative state vector.
         * @param nVars Total number of variables (flow + turbulence).
         */
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

        /**
         * @brief Extracts internal energy from a conservative state vector.
         *
         * Computes e = E - 0.5 * (rhoU^2 + rhoV^2 + rhoW^2) / rho.
         *
         * @tparam tU Conservative state vector type (deduced).
         * @param U Conservative state vector.
         * @param nVars Total number of variables (flow + turbulence).
         * @return Internal energy e = rho * cv * T for an ideal gas.
         */
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

        /**
         * @brief Computes pressure from the primitive state using the ideal gas law.
         *
         * Delegates to `IdealGas::Pressure_From_InternalEnergy` using prim[I4] (internal energy)
         * and the ratio of specific heats gamma.
         *
         * @tparam tUPrim Primitive state vector type (deduced).
         * @param UPrim Primitive state vector (prim[I4] = internal energy).
         * @param nVars Total number of variables.
         * @param T Temperature (unused; pressure is computed from internal energy directly).
         * @return Pressure p = (gamma - 1) * e.
         */
        template <class tUPrim>
        DNDS_DEVICE_CALLABLE real Prim2Pressure(
            tUPrim &&UPrim, int nVars, real T) const
        {
            //! perfect gas here — prim[I4] stores internal energy e
            return IdealGas::Pressure_From_InternalEnergy(UPrim(I4), params.gamma);
        }

        /**
         * @brief Computes the ratio of specific heats and the acoustic speed of sound.
         *
         * @tparam tUPrim Primitive state vector type (deduced).
         * @param UPrim Primitive state vector.
         * @param nVars Total number of variables.
         * @param p Pressure.
         * @return A std::tuple of (gamma, speed_of_sound) where a = sqrt(gamma * p / rho).
         */
        template <class tUPrim>
        DNDS_DEVICE_CALLABLE auto Prim2GammaAcousticSpeed(
            tUPrim &&UPrim, int nVars, real p) const
        {
            return std::make_tuple(params.gamma,
                                   std::sqrt(IdealGas::SpeedOfSoundSqr(params.gamma, p, UPrim(0))));
        }

        /**
         * @brief Computes temperature from the primitive state.
         *
         * For a perfect gas: T = e / (rho * cp), where e = prim[I4] is internal energy.
         *
         * @tparam tUPrim Primitive state vector type (deduced).
         * @param UPrim Primitive state vector.
         * @param nVars Total number of variables.
         * @return Temperature T.
         */
        template <class tUPrim>
        DNDS_DEVICE_CALLABLE real Prim2Temperature(
            tUPrim &&UPrim, int nVars) const
        {
            //! perfect gas here
            return UPrim(I4) / UPrim(0) / params.cp;
        }

        /**
         * @brief Computes specific total enthalpy from conservative state and pressure.
         *
         * H = (E + p) / rho, using `IdealGas::Enthalpy.`
         *
         * @tparam tU Conservative state vector type (deduced).
         * @param U Conservative state vector.
         * @param nVars Total number of variables.
         * @param p Pressure.
         * @return Specific total enthalpy H.
         */
        template <class tU>
        DNDS_DEVICE_CALLABLE real Pressure2Enthalpy(
            tU &&U, int nVars, real p) const
        {
            //! perfect gas here
            return IdealGas::Enthalpy(U(I4), U(0), p);
        }

        /**
         * @brief Recovers pressure from specific total enthalpy and conservative state.
         *
         * p = H * rho - E (inverse of Pressure2Enthalpy).
         *
         * @tparam tU Conservative state vector type (deduced).
         * @param U Conservative state vector.
         * @param nVars Total number of variables.
         * @param H Specific total enthalpy.
         * @return Pressure p.
         */
        template <class tU>
        DNDS_DEVICE_CALLABLE real Enthalpy2Pressure(
            tU &&U, int nVars, real H) const
        {
            //! perfect gas here
            return H * U(0) - U(I4);
        }
    };

    /**
     * @brief Host-side physics object managing gas parameters and device-transferable reference values.
     *
     * Stores a `host_device_vector` of reference values (e.g., freestream quantities) and
     * `PhysicsParams.` Supports JSON serialization and host/device transfer for GPU offloading.
     * Use `deviceView<B>()` to obtain a `PhysicsDeviceView` for kernel invocation.
     */
    struct Physics
    {
        host_device_vector<real> reference_values; ///< Reference values (e.g., freestream state) for non-dimensionalization.
        PhysicsParams params;                       ///< Gas physical parameters.
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(
            Physics,
            reference_values,
            params)

        /// @brief Transfers reference values to host memory.
        void to_host()
        {
            reference_values.to_host();
        }

        /// @brief Transfers reference values to the specified device backend.
        /// @param B Target device backend (Host or CUDA).
        void to_device(DeviceBackend B)
        {
            reference_values.to_device(B);
        }

        /// @brief Returns the current device backend where reference values reside.
        DeviceBackend device()
        {
            return reference_values.device();
        }

        template <DeviceBackend B>
        using t_deviceView = PhysicsDeviceView<B>; ///< Device view type alias parameterized by backend.

        /**
         * @brief Creates a device-callable view of this Physics object.
         *
         * Asserts that the reference values reside on backend @p B (or Host with Unknown).
         *
         * @tparam B Target device backend.
         * @return A `PhysicsDeviceView<B>` suitable for device kernel invocation.
         */
        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DNDS_assert(reference_values.device() == B ||
                        (B == DeviceBackend::Host && reference_values.device() == DeviceBackend::Unknown));
            return t_deviceView<B>{
                reference_values.deviceView<B, int32_t>(),
                params};
        }

        /**
         * @brief Configures the physics for a calorically perfect ideal gas.
         *
         * Computes cp = Rg * gamma / (gamma - 1) and cv = cp / gamma.
         *
         * @param Rg Specific gas constant R = Cp - Cv.
         * @param gamma Ratio of specific heats Cp/Cv.
         */
        void setPerfectGas(real Rg, real gamma)
        {
            params.Rg = Rg;
            params.gamma = gamma;
            params.cp = Rg * params.gamma / (params.gamma - 1);
            params.cv = params.cp / params.gamma;
        }
    };

    /**
     * @brief Computes the inviscid (Euler) flux projected onto a face normal direction.
     *
     * Accumulates the inviscid flux contribution into @p F:
     * - F_i += U_i * (vn - vgn) for all variables (advection with grid velocity correction)
     * - F_{d+1} += p * n_d for d = 0,1,2 (pressure contribution to momentum)
     * - F_{I4} += vn * p (pressure work on energy)
     *
     * @note This function @b adds to @p F (does not zero it first). Caller must initialize F.
     *
     * @tparam TU Conservative state vector type (deduced).
     * @tparam TF Flux vector type (deduced).
     * @tparam TVecN Normal vector type (deduced).
     * @param U Conservative state vector (rho, rhoU, rhoV, rhoW, E, ...).
     * @param nVars Total number of variables.
     * @param vn Normal velocity v dot n.
     * @param vgn Grid normal velocity (for ALE/moving mesh; 0 for static mesh).
     * @param n Outward unit face normal vector (3 components).
     * @param p Pressure at the face.
     * @param[in,out] F Flux vector to accumulate into.
     */
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