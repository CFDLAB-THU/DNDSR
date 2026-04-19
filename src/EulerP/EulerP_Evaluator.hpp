/** @file EulerP_Evaluator.hpp
 *  @brief Main evaluator class for the EulerP compressible Navier-Stokes solver module.
 *
 *  Defines the Evaluator class and its associated argument structs for dispatching
 *  CFD kernels (gradient reconstruction, face reconstruction, conservative-to-primitive
 *  conversion, eigenvalue estimation, and flux computation) to Host or CUDA backends.
 *  Also provides the EvaluatorDeviceView for device-callable access and the
 *  EvaluatorConfig for JSON-based runtime configuration.
 */
#pragma once

#include "CFV/FiniteVolume.hpp"
#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "EulerP.hpp"
#include "EulerP_BC.hpp"
#include "EulerP_Physics.hpp"
#include <vector>
#include "DNDS/DeviceStorageHelper.hpp"
#include "DNDS/ObjectUtils.hpp"

namespace DNDS::EulerP
{

    using TFiniteVolume = CFV::FiniteVolume;

    /**
     * @brief Device-callable view of the Evaluator, combining finite volume, BC handler, and physics views.
     *
     * This trivially-copyable struct bundles the device views needed by GPU/Host kernels
     * so they can access mesh geometry, boundary conditions, and gas physics without
     * indirecting through shared pointers.
     *
     * @tparam B The DeviceBackend (Host or CUDA).
     */
    template <DeviceBackend B>
    class EvaluatorDeviceView
    {
    public:
        using t_fv = TFiniteVolume::t_deviceView<B>;
        t_fv fv; /**< @brief Device view of the finite volume / VFV reconstruction object. */
        using t_bcHandler = BCHandlerDeviceView<B>;
        t_bcHandler bcHandler; /**< @brief Device view of the boundary condition handler. */
        using t_physics = PhysicsDeviceView<B>;
        t_physics physics; /**< @brief Device view of the gas physics model. */

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(EvaluatorDeviceView, EvaluatorDeviceView)
        DNDS_DEVICE_CALLABLE EvaluatorDeviceView(
            t_fv n_fv,
            t_bcHandler n_bcHandler,
            t_physics n_physics) : fv(n_fv), bcHandler(n_bcHandler), physics(n_physics) {}
    };

    /**
     * @brief JSON-configurable settings for the Evaluator.
     *
     * Stores runtime configuration such as CUDA threads-per-block, whether to pull
     * all input arguments before kernel dispatch, and serialization of CUDA execution.
     * Configuration is applied via JSON merge-patch semantics.
     */
    class EvaluatorConfig
    {
        t_jsonconfig config_data;

    public:
        /** @brief Constructs the config with default values (threadsPerBlock=128, etc.). */
        EvaluatorConfig();

        /**
         * @brief Validates that all keys in @p config_in exist in the current defaults.
         * @param config_in JSON patch to validate.
         * @throws std::runtime_error if an unrecognized key is found.
         */
        void valid_patch_keys(const t_jsonconfig &config_in);

        /**
         * @brief Validates and applies a JSON merge-patch to the configuration.
         * @param config_in JSON patch to merge.
         */
        void merge_patch(const t_jsonconfig &config_in)
        {
            valid_patch_keys(config_in);
            config_data.merge_patch(config_in);
        }

        /** @brief Returns a const reference to the underlying JSON configuration. */
        const t_jsonconfig &config() const { return config_data; }
    };

    /**
     * @brief Main EulerP evaluator orchestrating CFD kernel dispatch for compressible N-S solvers.
     *
     * Holds the finite volume reconstruction object, boundary condition handler, gas physics
     * model, and internal face buffers. Provides methods for each stage of a 2nd-order finite
     * volume time step: gradient reconstruction (Green-Gauss + Barth-Jespersen limiter),
     * face value reconstruction, conservative-to-primitive conversion, eigenvalue/timestep
     * estimation, and inviscid+viscous flux evaluation.
     *
     * Each method accepts a packed argument struct (e.g., RecGradient_Arg) that bundles
     * all input/output arrays. The dispatch logic selects Host or CUDA backend at runtime
     * and delegates to Evaluator_impl<B>.
     */
    class Evaluator
    {
    public:
        using t_self = Evaluator;
        using t_fv = ssp<CFV::FiniteVolume>;
        t_fv fv; /**< @brief Shared pointer to the Compact Finite Volume (VFV) reconstruction object. */
        using t_bcHandler = ssp<BCHandler>;
        t_bcHandler bcHandler; /**< @brief Shared pointer to the boundary condition handler. */
        using t_physics = ssp<Physics>;
        t_physics physics; /**< @brief Shared pointer to the gas physics model. */

        using t_config = EvaluatorConfig;

        t_config config; /**< @brief Runtime configuration (Riemann solver, CUDA settings, etc.). */

        /** @brief Returns a const reference to the JSON configuration. */
        const t_jsonconfig &get_config() { return config.config(); }

        /**
         * @brief Merges a JSON patch into the evaluator configuration.
         * @tparam Tc Forwarding reference type for the JSON config.
         * @param config_in JSON config patch to apply.
         */
        template <typename Tc>
        void set_config(Tc &&config_in)
        {
            config.merge_patch(config_in);
        }

        // internal buffers
        TUDof u_face_bufferL;                         /**< @brief Left-side face DOF buffer for boundary value storage. */
        TUDof u_face_bufferR;                         /**< @brief Right-side face DOF buffer for boundary value storage. */
        std::vector<TUScalar> uScalar_face_bufferL;   /**< @brief Left-side face scalar buffers for additional transported scalars. */
        std::vector<TUScalar> uScalar_face_bufferR;   /**< @brief Right-side face scalar buffers for additional transported scalars. */

        /**
         * @brief Allocates or resizes the face DOF buffer @p u to match the current mesh face count.
         *
         * Creates the father (owned faces) and son (ghost faces) arrays if they do not exist,
         * or resizes them if the mesh topology has changed. Also transfers to the active device
         * backend if one is set.
         *
         * @param u Face DOF buffer to build (in-out).
         */
        void BuildFaceBufferDof(TUDof &u)
        {
            // TODO: upgrade to matching face quad instead
            DNDS_check_throw(fv);
            bool father_changed{false}, son_changed{false};
            if (!u.father)
                u.father = make_ssp<decltype(u.father)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDof::u.father"}, fv->mesh->getMPI()), father_changed = true;
            if (u.father->Size() != fv->mesh->NumFace())
                u.father->Resize(fv->mesh->NumFace(), nVarsFlow, 1), father_changed = true;

            if (!u.son)
                u.son = make_ssp<decltype(u.son)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDof::u.son"}, fv->mesh->getMPI()), son_changed = true;
            if (u.son->Size() != fv->mesh->NumFaceGhost())
                u.son->Resize(fv->mesh->NumFaceGhost(), nVarsFlow, 1), son_changed = true;
            auto B = fv->device();
            if (B != DeviceBackend::Unknown && father_changed)
                u.father->to_device(B);
            if (B != DeviceBackend::Unknown && son_changed)
                u.son->to_device(B);
        }

        /**
         * @brief Allocates or resizes a scalar face buffer to match the current mesh face count.
         *
         * Similar to BuildFaceBufferDof but for single-component scalar quantities.
         *
         * @param u Scalar face buffer to build (in-out).
         */
        void BuildFaceBufferDofScalar(TUScalar &u)
        {
            // TODO: upgrade to matching face quad instead
            DNDS_check_throw(fv);
            bool father_changed{false}, son_changed{false};
            if (!u.father)
                u.father = make_ssp<decltype(u.father)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDofScalar::u.father"}, fv->mesh->getMPI()), father_changed = true;
            if (u.father->Size() != fv->mesh->NumFace())
                u.father->Resize(fv->mesh->NumFace(), 1, 1), father_changed = true;

            if (!u.son)
                u.son = make_ssp<decltype(u.son)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDofScalar::u.son"}, fv->mesh->getMPI()), son_changed = true;
            if (u.son->Size() != fv->mesh->NumFaceGhost())
                u.son->Resize(fv->mesh->NumFaceGhost(), 1, 1), son_changed = true;
            auto B = fv->device();
            // because ArrayPair's to_device is father and son to device:
            if (B != DeviceBackend::Unknown && father_changed)
                u.father->to_device(B);
            if (B != DeviceBackend::Unknown && son_changed)
                u.son->to_device(B);
        }

        /**
         * @brief Prepares all face buffers (DOF + scalars) for left and right states.
         * @param nVarsScalar Number of additional transported scalar variables.
         */
        void PrepareFaceBuffer(int nVarsScalar)
        {
            BuildFaceBufferDof(u_face_bufferL);
            BuildFaceBufferDof(u_face_bufferR);

            uScalar_face_bufferL.resize(nVarsScalar);
            uScalar_face_bufferR.resize(nVarsScalar);
            for (auto &uS : uScalar_face_bufferL)
                BuildFaceBufferDofScalar(uS);
            for (auto &uS : uScalar_face_bufferR)
                BuildFaceBufferDofScalar(uS);
        }

        /** @brief Returns a tuple of member pointers to face buffer arrays for device transfer. */
        static auto device_array_list_Buffer()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_PTR_SELF(u_face_bufferL),
                DNDS_MAKE_1_MEMBER_PTR_SELF(u_face_bufferR),
                DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar_face_bufferL),
                DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar_face_bufferR));
        }

        /**
         * @brief Constructs the Evaluator with the given finite volume, BC handler, and physics objects.
         * @param n_fv Shared pointer to the finite volume reconstruction object.
         * @param n_bcHandler Shared pointer to the boundary condition handler.
         * @param n_physics Shared pointer to the gas physics model.
         */
        Evaluator(
            t_fv n_fv,
            t_bcHandler n_bcHandler,
            t_physics n_physics) : fv(n_fv), bcHandler(n_bcHandler), physics(n_physics)
        {
        }

        /**
         * @brief Writes cell-centered and node-centered field data to a VTK-HDF5 file.
         *
         * Outputs primitive variables (density, velocity) along with user-specified scalar
         * and vector fields at both cell centers and nodes.
         *
         * @param fname Output file path.
         * @param series_name Name for the VTK time series.
         * @param arrCellCentScalar Cell-centered scalar field arrays.
         * @param arrCellCentScalar_names_in Names for each cell-centered scalar field.
         * @param arrCellCentVec Cell-centered vector field arrays.
         * @param arrCellCentVec_names_in Names for each cell-centered vector field.
         * @param arrNodeScalar Node-centered scalar field arrays.
         * @param arrNodeScalar_names_in Names for each node-centered scalar field.
         * @param arrNodeVec Node-centered vector field arrays.
         * @param arrNodeVec_names_in Names for each node-centered vector field.
         * @param uPrimCell Cell-centered primitive variable DOF (density + velocity); may be null.
         * @param uPrimNode Node-centered primitive variable DOF; may be null.
         * @param t Physical simulation time.
         */
        void PrintDataVTKHDF(std::string fname, std::string series_name,
                             std::vector<ssp<TUScalar>> &arrCellCentScalar,
                             const std::vector<std::string> &arrCellCentScalar_names_in,
                             std::vector<ssp<TUVec>> &arrCellCentVec,
                             const std::vector<std::string> &arrCellCentVec_names_in,
                             std::vector<ssp<TUScalar>> &arrNodeScalar,
                             const std::vector<std::string> &arrNodeScalar_names_in,
                             std::vector<ssp<TUVec>> &arrNodeVec,
                             const std::vector<std::string> &arrNodeVec_names_in,
                             ssp<TUDof> uPrimCell,
                             ssp<TUDof> uPrimNode,
                             double t);

        /**
         * @brief Queries and validates the device backend across all sub-objects.
         *
         * Checks that the finite volume, mesh, BC handler, and physics objects all agree
         * on the active device backend.
         *
         * @return The active DeviceBackend (Host, CUDA, or Unknown).
         * @throws std::runtime_error if backends are inconsistent.
         */
        DeviceBackend device()
        {
            DeviceBackend B = fv->device();
            DeviceBackend B_mesh = fv->mesh->device();
            DeviceBackend B_bcHandler = bcHandler->device();
            DeviceBackend B_physics = physics->device();
            DNDS_check_throw(B == B_mesh);
            DNDS_check_throw(B == B_bcHandler);
            DNDS_check_throw(B == B_physics);

            return B;
        }

        /**
         * @brief Non-trivially-copyable device view holding shared_ptr to fv and device views of BC/physics.
         *
         * Unlike EvaluatorDeviceView (trivially copyable for kernels), this struct retains
         * ownership of the fv shared_ptr and is move-only to prevent accidental copies
         * that would interfere with device memory management.
         *
         * @tparam B The DeviceBackend.
         */
        template <DeviceBackend B>
        struct t_deviceView
        {
            ssp<CFV::FiniteVolume> fv;
            t_bcHandler::element_type::t_deviceView<B> bcHandler;
            t_physics::element_type::t_deviceView<B> physics;

            //! only permit moving to avoid host_device_vector to change
            // also avoids accidentally copying this to device...
            t_deviceView(t_deviceView &&R) noexcept = default;
            t_deviceView(const t_deviceView &R) = delete;
            t_deviceView &operator=(t_deviceView &&R) = delete;
            t_deviceView &operator=(const t_deviceView &R) = delete;

            operator EvaluatorDeviceView<B>() const
            {
                static_assert(std::is_trivially_copyable_v<remove_cvref_t<decltype(fv->deviceView<B>())>>);
                return {fv->deviceView<B>(), bcHandler, physics};
            }
        };

        /**
         * @brief Creates a device view for backend B from the current evaluator state.
         * @tparam B The target DeviceBackend.
         * @return A move-only t_deviceView<B> combining fv, bcHandler, and physics views.
         * @throws std::runtime_error if the fv device backend does not match B.
         */
        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DeviceBackend B_fv = fv->device();
            DNDS_check_throw_info(B_fv == B || (B == DeviceBackend::Host && B_fv == DeviceBackend::Unknown),
                                  fmt::format("B_fv is  {} ", device_backend_name(B_fv)) +
                                      fmt::format("B is  {} ", device_backend_name(B)));

            return t_deviceView<B>{
                fv,
                bcHandler->deviceView<B>(),
                physics->deviceView<B>()};
        }

        /** @brief Transfers all sub-objects and face buffers from device memory back to host. */
        void to_host()
        {
            fv->to_host();
            bcHandler->to_host();
            physics->to_host();

            u_face_bufferL.to_host();
            u_face_bufferR.to_host();
            for (auto &a : uScalar_face_bufferL)
                a.to_host();
            for (auto &a : uScalar_face_bufferR)
                a.to_host();
        }

        /**
         * @brief Transfers all sub-objects and face buffers to the specified device backend.
         * @param B Target device backend (e.g., DeviceBackend::CUDA).
         */
        void to_device(DeviceBackend B)
        {
            fv->to_device(B);
            bcHandler->to_device(B);
            physics->to_device(B);
            u_face_bufferL.to_device(B);
            u_face_bufferR.to_device(B);
            for (auto &a : uScalar_face_bufferL)
                a.to_device(B);
            for (auto &a : uScalar_face_bufferR)
                a.to_device(B);
        }

        /****************************** */

        /**
         * @brief Validates that a DOF array pair has the expected sizes and device placement.
         *
         * Checks father/son existence, device backend match, and array sizes against the mesh
         * topology for the specified variable location.
         *
         * @tparam TPair ArrayPair type (e.g., TUDof, TUScalar).
         * @param u The DOF array pair to validate.
         * @param name Human-readable name for error messages.
         * @param varloc Variable location: 0=cell, 1=face, 2=node, -1=skip size check.
         * @param includeSon Whether to also validate the son (ghost) array.
         * @param B Expected device backend.
         * @param optional If true, a null @p u is allowed without error.
         */
        template <class TPair>
        void checkValidUDof(
            const ssp<TPair> &u,
            const std::string &name = "unknown",
            int varloc = 0,
            bool includeSon = true,
            DeviceBackend B = DeviceBackend::Unknown,
            bool optional = false)
        {
            std::string emit_info = fmt::format("{} varloc {} check failure", name, varloc);
            if (optional && !u)
                return;

            DNDS_check_throw_info(u, emit_info);
            DNDS_check_throw_info(u->father, emit_info);
            if (includeSon)
                DNDS_check_throw_info(u->son, emit_info);
            DNDS_check_throw_info(u->father->device() == B, emit_info);
            if (includeSon)
                DNDS_check_throw_info(u->son->device() == B, emit_info);
            if (varloc == -1)
            {
            }
            else if (varloc == 0)
            {
                DNDS_check_throw_info(u->father->Size() == fv->mesh->NumCell(), emit_info);
                if (includeSon)
                    DNDS_check_throw_info(u->son->Size() == fv->mesh->NumCellGhost(), emit_info);
            }
            else if (varloc == 1)
            {
                DNDS_check_throw_info(u->father->Size() == fv->mesh->NumFace(), emit_info);
                if (includeSon)
                    DNDS_check_throw_info(u->son->Size() == fv->mesh->NumFaceGhost(), emit_info);
            }
            else if (varloc == 2)
            {
                DNDS_check_throw_info(u->father->Size() == fv->mesh->NumNode(), emit_info);
                if (includeSon)
                    DNDS_check_throw_info(u->son->Size() == fv->mesh->NumNodeGhost(), emit_info);
            }
            else
                DNDS_check_throw_info(false, "varloc should be -1, 0, 1 or 2");
        }

        /**
         * @brief Packed argument struct for Green-Gauss gradient reconstruction with Barth-Jespersen limiting.
         *
         * Inherits EvaluatorArgBase for CRTP-based device transfer and validation.
         * Contains the conservative state and its gradient, plus optional transported scalars
         * and their gradients. The gradient fields are both input (initial guess) and output
         * (reconstructed + limited gradients).
         */
        struct RecGradient_Arg : public EvaluatorArgBase<RecGradient_Arg>
        {
            ssp<TUDof> u;      /**< @brief Conservative state vector (cell-centered, nVarsFlow components). */
            ssp<TUGrad> uGrad; /**< @brief Gradient of conservative state (in-out: reconstructed by Green-Gauss). */
            // out
            std::vector<ssp<TUScalar>> uScalar;         /**< @brief Additional transported scalar fields (cell-centered). */
            std::vector<ssp<TUScalarGrad>> uScalarGrad;  /**< @brief Gradients of transported scalars (in-out). */

            using t_self = RecGradient_Arg;

            /** @brief Returns a tuple of member-to-pointer mappings for CRTP iteration / device transfer. */
            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad));
            }
            /**
             * @brief Validates all member arrays against the mesh topology and device backend.
             * @param self Reference to the owning Evaluator.
             */
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                self.checkValidUDof(u, "u", 0, true, B);
                self.checkValidUDof(uGrad, "uGrad", 0, true, B);
                for (size_t i = 0; i < uScalar.size(); i++)
                    self.checkValidUDof(uScalar[i], "uScalar_" + std::to_string(i), 0, true, B);
                for (size_t i = 0; i < uScalarGrad.size(); i++)
                    self.checkValidUDof(uScalarGrad[i], "uScalarGrad_" + std::to_string(i), 0, true, B);
            }
        };

        /**
         * @brief Performs Green-Gauss gradient reconstruction with Barth-Jespersen limiting.
         *
         * Computes cell gradients via the Green-Gauss theorem (face-area-weighted averaging),
         * then applies the Barth-Jespersen monotonicity limiter to prevent spurious oscillations.
         * Dispatches to Host or CUDA backend based on the current device().
         *
         * @param arg Packed input/output arguments (conservative state, gradients, scalars).
         */
        void RecGradient(RecGradient_Arg &arg);

        /**
         * @brief Packed argument struct for conservative-to-primitive conversion with viscosity computation.
         *
         * Holds conservative and primitive states (both flow DOF and scalars), their gradients,
         * and output thermodynamic quantities (pressure, temperature, speed of sound, gamma, viscosity).
         */
        struct Cons2PrimMu_Arg : public EvaluatorArgBase<Cons2PrimMu_Arg>
        {
            using t_self = Cons2PrimMu_Arg;

            ssp<TUDof> u;                                    /**< @brief Conservative state (input). */
            ssp<TUGrad> uGrad;                               /**< @brief Gradient of conservative state (input). */
            std::vector<ssp<TUScalar>> uScalar;              /**< @brief Transported scalar fields (input). */
            std::vector<ssp<TUScalarGrad>> uScalarGrad;      /**< @brief Gradients of transported scalars (input). */
            // out
            ssp<TUDof> uPrim;                                /**< @brief Primitive state (output). */
            ssp<TUGrad> uGradPrim;                           /**< @brief Gradient of primitive state (output). */
            std::vector<ssp<TUScalar>> uScalarPrim;          /**< @brief Primitive transported scalars (output). */
            std::vector<ssp<TUScalarGrad>> uScalarGradPrim;  /**< @brief Gradients of primitive scalars (output). */
            ssp<TUScalar> p;                                 /**< @brief Pressure (output). */
            ssp<TUScalar> T;                                 /**< @brief Temperature (output). */
            ssp<TUScalar> a;                                 /**< @brief Speed of sound (output). */
            ssp<TUScalar> gamma;                             /**< @brief Ratio of specific heats (output). */
            ssp<TUScalar> mu;                                /**< @brief Total (laminar + turbulent) viscosity (output). */
            std::vector<ssp<TUScalar>> muComp;               /**< @brief Component-wise viscosity contributions (output). */

            /** @brief Returns a tuple of member-to-pointer mappings for CRTP iteration / device transfer. */
            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(p),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(T),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(a),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(gamma),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(mu),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(muComp));
            }

            /**
             * @brief Validates all member arrays against the mesh topology and device backend.
             * @param self Reference to the owning Evaluator.
             */
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;
                int varloc = -1;
                // TODO: improve to check if all sizes are compatible (varloc = -1 does not check size now)

                auto validate_member = [&](std::string name, auto &v)
                {
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                    {
                        self.checkValidUDof(v, name,
                                            varloc, true, B);
                    }
                    else
                    {
                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                    }
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        /**
         * @brief Converts conservative variables to primitive variables and computes viscosity.
         *
         * For each cell, converts (rho, rho*u, rho*E, ...) to (rho, u, T, ...) and
         * computes primitive gradients, pressure, temperature, speed of sound, gamma, and
         * total viscosity (laminar + turbulent).
         *
         * @param arg Packed input/output arguments.
         */
        void Cons2PrimMu(Cons2PrimMu_Arg &arg);

        /**
         * @brief Packed argument struct for conservative-to-primitive conversion without gradients or viscosity.
         *
         * A simpler variant of Cons2PrimMu_Arg that only computes primitive variables and
         * thermodynamic scalars (p, T, a, gamma) without gradient transformation or viscosity.
         */
        struct Cons2Prim_Arg : public EvaluatorArgBase<Cons2Prim_Arg>
        {
            using t_self = Cons2Prim_Arg;

            ssp<TUDof> u;                           /**< @brief Conservative state (input). */
            std::vector<ssp<TUScalar>> uScalar;     /**< @brief Transported scalar fields (input). */
            // out
            ssp<TUDof> uPrim;                       /**< @brief Primitive state (output). */
            std::vector<ssp<TUScalar>> uScalarPrim;  /**< @brief Primitive transported scalars (output). */
            ssp<TUScalar> p;                         /**< @brief Pressure (output). */
            ssp<TUScalar> T;                         /**< @brief Temperature (output). */
            ssp<TUScalar> a;                         /**< @brief Speed of sound (output). */
            ssp<TUScalar> gamma;                     /**< @brief Ratio of specific heats (output). */

            /** @brief Returns a tuple of member-to-pointer mappings for CRTP iteration / device transfer. */
            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(p),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(T),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(a),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(gamma));
            }
            /**
             * @brief Validates all member arrays against the mesh topology and device backend.
             * @param self Reference to the owning Evaluator.
             */
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;
                int varloc = -1;

                auto validate_member = [&](std::string name, auto &v)
                {
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                    {
                        self.checkValidUDof(v, name,
                                            varloc, true, B);
                    }
                    else
                    {
                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                    }
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        /**
         * @brief Converts conservative variables to primitive variables (no gradients or viscosity).
         *
         * Simplified version of Cons2PrimMu that computes only primitive state, pressure,
         * temperature, speed of sound, and gamma.
         *
         * @param arg Packed input/output arguments.
         */
        void Cons2Prim(Cons2Prim_Arg &arg);

        /**
         * @brief Packed argument struct for eigenvalue estimation and local time-step computation.
         *
         * Provides the conservative state, per-cell viscosity and speed of sound, and outputs
         * per-face eigenvalue estimates (acoustic waves), per-face viscous eigenvalue estimates,
         * face/cell delta-lambda for H-correction, and the local CFL time step.
         */
        struct EstEigenDt_Arg : public EvaluatorArgBase<EstEigenDt_Arg>
        {
            using t_self = EstEigenDt_Arg;
            ssp<TUDof> u;                    /**< @brief Conservative state (cell-centered, input). */
            ssp<TUScalar> muCell;            /**< @brief Per-cell total viscosity (input). */
            ssp<TUScalar> aCell;             /**< @brief Per-cell speed of sound (input). */
            ssp<TUScalarGrad> faceLamEst;    /**< @brief Per-face eigenvalue estimates [lam-, lam0, lam+] (output). */
            ssp<TUScalar> faceLamVisEst;     /**< @brief Per-face viscous eigenvalue estimate (output). */
            ssp<TUScalar> deltaLamFace;      /**< @brief Per-face eigenvalue difference for H-correction (output). */
            ssp<TUScalar> deltaLamCell;      /**< @brief Per-cell maximum eigenvalue difference (output). */
            ssp<TUScalar> dt;                /**< @brief Per-cell local time step (output). */

            /** @brief Returns a tuple of member-to-pointer mappings for CRTP iteration / device transfer. */
            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(muCell),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(aCell),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(faceLamEst),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(faceLamVisEst),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(deltaLamFace),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(deltaLamCell),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(dt));
            }
            /**
             * @brief Validates all member arrays, distinguishing cell-located from face-located fields.
             * @param self Reference to the owning Evaluator.
             */
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();

                auto validate_member = [&](std::string name, auto &v)
                {
                    int varloc = 0;
                    if (std::set<std::string>{
                            "faceLamEst",
                            "faceLamVisEst",
                            "deltaLamFace",
                        }
                            .count(name))
                        varloc = 1;
                    self.checkValidUDof(v, name, varloc, true, B);
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        /**
         * @brief Estimates per-face eigenvalues and computes per-cell local time steps.
         *
         * First pass: computes face-level convective and viscous eigenvalue estimates.
         * Second pass: accumulates face eigenvalues to cells and computes dt = Vol / sum(lambda * area).
         *
         * @param arg Packed input/output arguments.
         */
        void EstEigenDt(EstEigenDt_Arg &arg);

        /**
         * @brief Packed argument struct for 2nd-order face value reconstruction.
         *
         * Takes cell-centered conservative state and gradients, and outputs left/right
         * reconstructed face values (uFL, uFR) and face-averaged gradients (uGradFF).
         * Boundary faces receive ghost values via the BC handler.
         */
        struct RecFace2nd_Arg : public EvaluatorArgBase<RecFace2nd_Arg>
        {
            using t_self = RecFace2nd_Arg;
            ssp<TUDof> u;                                       /**< @brief Conservative state (cell-centered, input). */
            ssp<TUGrad> uGrad;                                  /**< @brief Cell gradient of conservative state (input). */
            std::vector<ssp<TUScalar>> uScalar;                 /**< @brief Cell-centered transported scalars (input). */
            std::vector<ssp<TUScalarGrad>> uScalarGrad;         /**< @brief Gradients of transported scalars (input). */
            // out
            ssp<TUDof> uFL;                                     /**< @brief Left reconstructed face values (face-located, output). */
            ssp<TUDof> uFR;                                     /**< @brief Right reconstructed face values (face-located, output). */
            ssp<TUGrad> uGradFF;                                /**< @brief Face-averaged gradient (face-located, output). */
            std::vector<ssp<TUScalar>> uScalarFL;               /**< @brief Left reconstructed scalar face values (output). */
            std::vector<ssp<TUScalar>> uScalarFR;               /**< @brief Right reconstructed scalar face values (output). */
            std::vector<ssp<TUScalarGrad>> uScalarGradFF;       /**< @brief Face-averaged scalar gradients (output). */

            /** @brief Returns a tuple of member-to-pointer mappings for CRTP iteration / device transfer. */
            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad),
                    // out
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradFF),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradFF));
            }
            /**
             * @brief Validates member arrays, auto-detecting cell vs. face location by name suffix.
             * @param self Reference to the owning Evaluator.
             */
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;

                auto validate_member = [&](std::string name, auto &v)
                {
                    int varloc = 0;
                    auto name_last2 = name.substr(std::max(name.size(), 2ul) - 2, 2);
                    if (name_last2 == "FL" || name_last2 == "FR" || name_last2 == "FF")
                        varloc = 1;
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                        self.checkValidUDof(v, name,
                                            varloc, true, B);

                    else

                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        /**
         * @brief Performs 2nd-order face value reconstruction from cell-centered data.
         *
         * Extrapolates cell values and gradients to face quadrature points to produce left/right
         * face states. Applies boundary conditions at domain boundaries via the BC handler.
         *
         * @param arg Packed input/output arguments.
         */
        void RecFace2nd(RecFace2nd_Arg &arg);

        /**
         * @brief Packed argument struct for 2nd-order inviscid (and eventually viscous) flux evaluation.
         *
         * The largest argument struct, combining cell-centered state and gradients,
         * face-reconstructed left/right values, thermodynamic scalars, viscosity, eigenvalue
         * corrections, face flux output, and accumulated RHS output.
         */
        struct Flux2nd_Arg : public EvaluatorArgBase<Flux2nd_Arg>
        {
            using t_self = Flux2nd_Arg;
            ssp<TUDof> u;                                       /**< @brief Conservative state (cell-centered, input). */
            ssp<TUGrad> uGrad;                                  /**< @brief Gradient of conservative state (input). */
            std::vector<ssp<TUScalar>> uScalar;                 /**< @brief Transported scalar fields (input). */
            std::vector<ssp<TUScalarGrad>> uScalarGrad;         /**< @brief Gradients of transported scalars (input). */
            ssp<TUDof> uPrim;                                   /**< @brief Primitive state (cell-centered, input). */
            ssp<TUGrad> uGradPrim;                              /**< @brief Gradient of primitive state (input). */
            std::vector<ssp<TUScalar>> uScalarPrim;             /**< @brief Primitive transported scalars (input). */
            std::vector<ssp<TUScalarGrad>> uScalarGradPrim;     /**< @brief Gradients of primitive scalars (input). */
            ssp<TUScalar> p;                                    /**< @brief Pressure (cell-centered, input). */
            ssp<TUScalar> T;                                    /**< @brief Temperature (cell-centered, input). */
            ssp<TUScalar> a;                                    /**< @brief Speed of sound (cell-centered, input). */
            ssp<TUScalar> mu;                                   /**< @brief Total viscosity (cell-centered, input). */
            std::vector<ssp<TUScalar>> muComp;                  /**< @brief Component-wise viscosity contributions (input). */
            ssp<TUScalar> gamma;                                /**< @brief Ratio of specific heats (cell-centered, input). */
            ssp<TUScalar> deltaLamCell;                         /**< @brief Per-cell eigenvalue difference for H-correction (input). */

            ssp<TUDof> uFL;                                     /**< @brief Left reconstructed face values (face-located, input). */
            ssp<TUDof> uFR;                                     /**< @brief Right reconstructed face values (face-located, input). */
            ssp<TUGrad> uGradFF;                                /**< @brief Face-averaged gradient (face-located, input). */
            std::vector<ssp<TUScalar>> uScalarFL;               /**< @brief Left reconstructed scalar face values (input). */
            std::vector<ssp<TUScalar>> uScalarFR;               /**< @brief Right reconstructed scalar face values (input). */
            std::vector<ssp<TUScalarGrad>> uScalarGradFF;       /**< @brief Face-averaged scalar gradients (input). */

            ssp<TUScalar> pFL;                                  /**< @brief Left face pressure (face-located, input). */
            ssp<TUScalar> pFR;                                  /**< @brief Right face pressure (face-located, input). */

            // out
            ssp<TUDof> fluxFF;                                  /**< @brief Numerical flux at each face (face-located, output). */
            std::vector<ssp<TUScalar>> fluxScalarFF;            /**< @brief Scalar numerical flux at each face (output). */
            // out added
            ssp<TUDof> rhs;                                     /**< @brief Accumulated right-hand-side residual (cell-centered, in-out). */
            std::vector<ssp<TUScalar>> rhsScalar;               /**< @brief Accumulated scalar RHS residual (cell-centered, in-out). */

            /** @brief Returns a tuple of member-to-pointer mappings for CRTP iteration / device transfer. */
            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(p),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(T),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(a),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(gamma),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(mu),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(muComp),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(deltaLamCell),

                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradFF),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradFF),

                    DNDS_MAKE_1_MEMBER_PTR_SELF(pFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(pFR),
                    // out
                    DNDS_MAKE_1_MEMBER_PTR_SELF(fluxFF),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(fluxScalarFF),
                    // out added
                    DNDS_MAKE_1_MEMBER_PTR_SELF(rhs),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(rhsScalar));
            }
            /**
             * @brief Validates member arrays, auto-detecting cell vs. face location by name suffix.
             * @param self Reference to the owning Evaluator.
             */
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;

                auto validate_member = [&](std::string name, auto &v)
                {
                    int varloc = 0;
                    auto name_last2 = name.substr(std::max(name.size(), 2ul) - 2, 2);
                    if (name_last2 == "FL" || name_last2 == "FR" || name_last2 == "FF")
                        varloc = 1;
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                        self.checkValidUDof(v, name,
                                            varloc, true, B);

                    else

                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        /**
         * @brief Evaluates 2nd-order inviscid (Roe) flux and accumulates into the cell RHS residual.
         *
         * First pass: computes face-level numerical flux using Roe-averaged states with
         * eigenvalue fixing (H-correction). Second pass: scatters face fluxes to cell RHS
         * weighted by face area / cell volume.
         *
         * @param arg Packed input/output arguments (state, face values, flux, RHS).
         */
        void Flux2nd(Flux2nd_Arg &arg);
    };

    /********************************************************************************** */

    /********************************************************************************** */

}