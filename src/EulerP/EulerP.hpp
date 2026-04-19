/**
 * @file EulerP.hpp
 * @brief Core type definitions and utilities for the EulerP alternative Navier-Stokes evaluator module.
 *
 * The EulerP module is an alternative compressible Navier-Stokes evaluator with CUDA GPU support.
 * Unlike the Euler module (which uses Eigen compile-time nVars and template-heavy dispatch),
 * EulerP uses scalar loops, is device-callable, and supports both Host and CUDA backends
 * via the @c DeviceBackend template parameter.
 *
 * This header defines:
 * - Fixed flow variable count and index constants
 * - Eigen vector/matrix type aliases for state and gradient storage
 * - ArrayDof-based distributed array type aliases for MPI-parallel data
 * - Helper functions for extracting momentum and coordinate sub-vectors
 * - CRTP base class for packed kernel argument structs with MPI ghost exchange
 */
#pragma once
#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Defines.hpp"
#include "Eigen/Core"

/**
 * @brief Namespace for the EulerP alternative evaluator module with GPU support.
 *
 * Provides device-callable (Host + CUDA) compressible Navier-Stokes solvers using
 * scalar loops instead of Eigen compile-time templates, enabling GPU offloading.
 */
namespace DNDS::EulerP
{
    static constexpr int nVarsFlow = 5; ///< Number of flow variables: rho, rhoU, rhoV, rhoW, E.
    static constexpr int I4 = 4;        ///< Index of the energy variable in the conservative state vector.

    using TU = Eigen::Vector<real, nVarsFlow>;         ///< Fixed-size 5-component conservative state vector (rho, rhoU, rhoV, rhoW, E).
    using TDiffU = Eigen::Matrix<real, 3, nVarsFlow>;  ///< Fixed-size 3x5 spatial gradient of the conservative state (one row per spatial dimension).

    using TUFull = Eigen::Vector<real, Eigen::Dynamic>;        ///< Dynamic-size state vector for extended variables (flow + turbulence model variables).
    using TDiffUFull = Eigen::Matrix<real, 3, Eigen::Dynamic>; ///< Dynamic-size 3xN spatial gradient for extended variables.

    using TUFullMap = Eigen::Map<TUFull>;        ///< Eigen::Map wrapper for non-owning access to a dynamic-size state vector.
    using TDiffUFullMap = Eigen::Map<TDiffUFull>; ///< Eigen::Map wrapper for non-owning access to a dynamic-size gradient matrix.

    using TUDof = ArrayDof<nVarsFlow, 1>;  ///< Distributed array of 5-component flow state vectors (one per DOF).
    using TUGrad = ArrayDof<3, nVarsFlow>; ///< Distributed array of 3x5 gradient matrices (one per DOF).

    using TUScalar = ArrayDof<1, 1>;     ///< Distributed array of scalar values (e.g., pressure, temperature).
    using TUScalarGrad = ArrayDof<3, 1>; ///< Distributed array of 3-component scalar gradients.

    using TUScalar2 = ArrayDof<2, 1>; ///< Distributed array of 2-component scalar values (e.g., paired quantities).

    using TUVec = ArrayDof<3, 1>;     ///< Distributed array of 3-component vectors (e.g., velocity, coordinates).
    using TUVecGrad = ArrayDof<3, 3>; ///< Distributed array of 3x3 vector gradient tensors.

    /**
     * @brief Extracts the momentum components (indices 1,2,3) from a state vector as a 3x1 block.
     * @tparam TU Eigen vector type (deduced).
     * @param v State vector of at least 4 components (rho, rhoU, rhoV, rhoW, ...).
     * @return A 3x1 Eigen block expression referencing (rhoU, rhoV, rhoW).
     */
    template <class TU>
    DNDS_DEVICE_CALLABLE DNDS_FORCEINLINE auto U123(TU &&v)
    {
        return v.template block<3, 1>(1, 0);
    }

    /**
     * @brief Extracts the first 3 components (indices 0,1,2) from a vector as a 3x1 block.
     * @tparam TU Eigen vector type (deduced).
     * @param v Vector of at least 3 components.
     * @return A 3x1 Eigen block expression referencing components (0, 1, 2).
     */
    template <class TU>
    DNDS_DEVICE_CALLABLE DNDS_FORCEINLINE auto U012(TU &&v)
    {
        return v.template block<3, 1>(0, 0);
    }

    /**
     * @brief CRTP base class for packed kernel argument structs used by the Evaluator.
     *
     * Kernel argument structs (e.g., RecGradient_Arg, Flux2nd_Arg) inherit from this
     * base using CRTP. Each derived class must provide a static @c member_list() function
     * returning a list of named pointers to its ArrayDof shared_ptr members.
     *
     * The base provides @c WaitAllPull(B), which iterates over all ArrayDof members
     * and completes their MPI persistent ghost-exchange pull operations on the
     * specified device backend.
     *
     * @tparam TDerived The derived packed argument struct type (CRTP pattern).
     */
    template <class TDerived>
    class EvaluatorArgBase
    {
    public:
        /**
         * @brief Completes MPI ghost exchange for all ArrayDof members on the specified backend.
         *
         * Iterates over every ArrayDof shared_ptr registered in @c TDerived::member_list(),
         * validates that both father and son arrays exist and reside on backend @p B,
         * then calls @c waitPersistentPull(B) on each transformer to finalize
         * non-blocking MPI receives.
         *
         * @param B The device backend (Host or CUDA) on which to perform the wait.
         * @throws std::runtime_error If father/son arrays are missing or on the wrong device.
         */
        void WaitAllPull(DeviceBackend B)
        {
            auto *dThis = static_cast<TDerived *>(this);

            auto wait_all = [&](std::string name, auto &v)
            {
                auto do_wait = [B, name](auto &vv)
                {
                    DNDS_check_throw_info(vv->father && vv->son, name + " needs father and son");
                    //! this assertion should be provided by ArrayTransformer
                    DNDS_check_throw_info(vv->trans.father.get() == vv->father.get(), name + " needs father == trans.father");
                    DNDS_check_throw_info(vv->trans.son.get() == vv->son.get(), name + " needs son == trans.son");
                    DNDS_check_throw_info(vv->father->device() == B,
                                          name +
                                              " father on " + device_backend_name(vv->father->device()) +
                                              " wait on " + device_backend_name(B));
                    DNDS_check_throw_info(vv->son->device() == B,
                                          name +
                                              " son on " + device_backend_name(vv->father->device()) +
                                              " wait on " + device_backend_name(B));

                    vv->trans.waitPersistentPull(B);
                };
                if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                    do_wait(v);
                else
                    for (size_t i = 0; i < v.size(); i++)
                        do_wait(v[i]);
            };
            for_each_member_ptr_list(*dThis, TDerived::member_list(), wait_all);
        }
    };
}