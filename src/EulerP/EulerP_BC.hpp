/**
 * @file EulerP_BC.hpp
 * @brief Boundary condition types, implementations, and handler for the EulerP module.
 *
 * Provides:
 * - `BCType:` Enumeration of supported boundary condition types (Far, Wall, Sym, etc.)
 * - `BCFunc_Impl:` Template specializations for each BC type (currently stubs, TODO)
 * - `BC_DeviceView` / `BC:` Device-callable and host-side single BC objects
 * - `BCHandlerDeviceView` / `BCHandler:` Device-callable and host-side BC managers
 * - `BCInput:` JSON-deserializable input struct for BC configuration
 *
 * The BC system uses a runtime switch-dispatch in `BC_DeviceView::apply()` to route
 * to the appropriate `BCFunc_Impl` specialization, enabling device-callable BC evaluation
 * on both Host and CUDA backends.
 */
#pragma once
#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/Config/ConfigEnum.hpp"
#include "EulerP.hpp"
#include "EulerP_Physics.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "Geom/Geometric.hpp"

namespace DNDS::EulerP
{
    /**
     * @brief Enumeration of boundary condition types for the EulerP module.
     *
     * Mirrors the BC types in the Euler module. JSON-serializable via DNDS_DEFINE_ENUM_JSON.
     */
    enum class BCType : uint8_t
    {
        Unknown = 0,      ///< Uninitialized or unrecognized BC type.
        Far,              ///< Farfield BC (characteristic-based).
        Wall,             ///< No-slip viscous wall (adiabatic).
        WallInvis,        ///< Inviscid (slip) wall BC.
        WallIsothermal,   ///< No-slip viscous wall with fixed temperature.
        Out,              ///< Supersonic/subsonic outflow BC.
        OutP,             ///< Outflow with specified back-pressure.
        In,               ///< Supersonic/subsonic inflow BC.
        InPsTs,           ///< Inflow with specified stagnation pressure and temperature.
        Sym,              ///< Symmetry plane BC.
        Special,          ///< Special-purpose BC for benchmark cases (e.g., DMR, Riemann).
    };

    DNDS_DEFINE_ENUM_JSON(
        BCType,
        {
            {BCType::Unknown, nullptr},
            {BCType::Far, "Far"},
            {BCType::Wall, "Wall"},
            {BCType::WallInvis, "WallInvis"},
            {BCType::WallIsothermal, "WallIsothermal"},
            {BCType::Out, "Out"},
            {BCType::OutP, "OutP"},
            {BCType::In, "In"},
            {BCType::InPsTs, "InPsTs"},
            {BCType::Sym, "Sym"},
            {BCType::Special, "Special"},
        })

    /// @brief Type alias for the device-resident BC parameter storage vector.
    template <DeviceBackend B>
    using BCStorageVecDeviceView = vector_DeviceView<B, real, int32_t>;

    /**
     * @brief Primary template for boundary condition function implementations.
     *
     * Each specialization provides a static `apply()` method implementing the
     * specific BC logic for one `BCType.` All current specializations are stubs
     * that assert false (implementations are TODO).
     *
     * @tparam B Device backend (Host or CUDA).
     * @tparam T The BCType this specialization implements.
     */
    template <DeviceBackend B, BCType T>
    struct BCFunc_Impl;

    /**
     * @brief Macro defining the standard BC apply interface signature.
     *
     * Expands to a static device-callable apply function taking:
     * - `U:` Input conservative state
     * - `UOut:` Output (ghost) conservative state
     * - `uSiz:` Number of variables
     * - `x:` Face centroid coordinates
     * - `n:` Outward face normal
     * - `id:` Boundary zone ID
     * - `value:` BC parameter storage view
     * - `phy:` Physics device view
     */
#define DNDS_EULERP_BC_INTERFACE_APPLY                   \
    template <class tU, class tUOut, class tx, class tn> \
    DNDS_DEVICE_CALLABLE static void apply(              \
        tU &&U, tUOut &&UOut, int32_t uSiz,              \
        tx &&x, tn &&n,                                  \
        Geom::t_index id,                                \
        BCStorageVecDeviceView<B> value, PhysicsDeviceView<B> &phy)

    /// @brief Farfield BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Far>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief No-slip viscous wall BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Wall>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief Isothermal viscous wall BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::WallIsothermal>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief Outflow BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Out>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief Outflow with back-pressure BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::OutP>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief Inflow with stagnation pressure/temperature BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::InPsTs>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief Symmetry plane BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Sym>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief Special-purpose BC implementation for benchmark cases (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Special>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief General inflow BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::In>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /// @brief Inviscid (slip) wall BC implementation (TODO: not yet implemented).
    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::WallInvis>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    /**
     * @brief Device-callable view of a single boundary condition.
     *
     * Holds a device-resident view of BC parameter values, the boundary zone ID,
     * and the BC type. The `apply()` method uses a runtime switch to dispatch
     * to the appropriate `BCFunc_Impl` specialization.
     *
     * @tparam B Device backend (Host or CUDA).
     */
    template <DeviceBackend B>
    class BC_DeviceView
    {
        BCStorageVecDeviceView<B> values; ///< Device-resident BC parameter values.
        Geom::t_index id = Geom::BC_ID_NULL; ///< Boundary zone identifier.
        BCType type = BCType::Unknown;        ///< Type of boundary condition.

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(BC_DeviceView, BC_DeviceView)

        /**
         * @brief Constructs a BC device view from values, zone ID, and type.
         * @param n_values Device view of BC parameter values.
         * @param n_id Boundary zone identifier.
         * @param n_type Boundary condition type.
         */
        DNDS_DEVICE_CALLABLE BC_DeviceView(
            BCStorageVecDeviceView<B> n_values,
            Geom::t_index n_id,
            BCType n_type)
            : values(n_values), id(n_id), type(n_type) {}

        DNDS_DEVICE_CALLABLE Geom::t_index getId() const { return id; }   ///< Returns the boundary zone ID.
        DNDS_DEVICE_CALLABLE BCType getType() const { return type; }       ///< Returns the boundary condition type.
        DNDS_DEVICE_CALLABLE int32_t getNValues() const { return values.size(); } ///< Returns the number of BC parameter values.
        DNDS_DEVICE_CALLABLE real value(int32_t i) const { return values[i]; }    ///< Returns the i-th BC parameter value.

        /**
         * @brief Applies this boundary condition to compute the ghost state.
         *
         * Dispatches to the appropriate `BCFunc_Impl` specialization based on the runtime
         * `type` field using a switch statement. Asserts on unknown types.
         *
         * @tparam tU Input conservative state type (deduced).
         * @tparam tUOut Output ghost state type (deduced).
         * @tparam tx Coordinate vector type (deduced).
         * @tparam tn Normal vector type (deduced).
         * @param U Input conservative state at the boundary face.
         * @param[out] UOut Output ghost conservative state.
         * @param uSiz Number of variables.
         * @param x Face centroid coordinates.
         * @param n Outward face unit normal.
         * @param phy Physics device view for thermodynamic computations.
         */
        template <class tU, class tUOut, class tx, class tn>
        DNDS_DEVICE_CALLABLE void apply(tU &&U, tUOut &&UOut, int32_t uSiz,
                                        tx &&x, tn &&n,
                                        PhysicsDeviceView<B> &phy)
        {
#define DNDS_EULERP_CASE_BC_APPLY(type)                                    \
    case type:                                                             \
    {                                                                      \
        BCFunc_Impl<B, type>::apply(U, UOut, uSiz, x, n, id, values, phy); \
    }                                                                      \
    break;
            switch (type)
            {
                DNDS_EULERP_CASE_BC_APPLY(BCType::Far);
                DNDS_EULERP_CASE_BC_APPLY(BCType::Wall);
                DNDS_EULERP_CASE_BC_APPLY(BCType::WallInvis);
                DNDS_EULERP_CASE_BC_APPLY(BCType::WallIsothermal);
                DNDS_EULERP_CASE_BC_APPLY(BCType::Out);
                DNDS_EULERP_CASE_BC_APPLY(BCType::OutP);
                DNDS_EULERP_CASE_BC_APPLY(BCType::In);
                DNDS_EULERP_CASE_BC_APPLY(BCType::InPsTs);
                DNDS_EULERP_CASE_BC_APPLY(BCType::Sym);
                DNDS_EULERP_CASE_BC_APPLY(BCType::Special);
            default:
                DNDS_HD_assert(false);
            }
        }
    };

    /**
     * @brief Host-side boundary condition object managing parameter values and device transfer.
     *
     * Stores BC parameters in a `host_device_vector` for transparent host/device transfer.
     * Provides property accessors for zone ID, type, and values. Use `deviceView<B>()`
     * to obtain a `BC_DeviceView` for kernel invocation.
     */
    class BC
    {
        host_device_vector<real> values; ///< BC parameter values (e.g., freestream state, wall temperature).
        Geom::t_index id = Geom::BC_ID_NULL; ///< Boundary zone identifier.
        BCType type = BCType::Unknown;        ///< Boundary condition type.

    public:
        [[nodiscard]] Geom::t_index getId() const { return id; }            ///< Returns the boundary zone ID.
        void setId(Geom::t_index n_id) { id = n_id; }                      ///< Sets the boundary zone ID.
        [[nodiscard]] BCType getType() const { return type; }               ///< Returns the BC type.
        void setType(BCType n_type) { type = n_type; }                      ///< Sets the BC type.
        [[nodiscard]] int32_t getNValues() const { return size_t_to_signed<int32_t>(values.size()); } ///< Returns the number of parameter values.
        [[nodiscard]] real value(int i) const { return values.at(i); }      ///< Returns the i-th parameter value (bounds-checked).
        void setValues(const std::vector<real> &v) { values = v; }          ///< Sets BC parameter values from a vector.
        [[nodiscard]] auto getValues() const { return std::vector<real>(values); } ///< Returns a copy of BC parameter values as std::vector.

        /// @brief Transfers BC values to host memory.
        void to_host()
        {
            values.to_host();
        }

        /// @brief Transfers BC values to the specified device backend.
        /// @param B Target device backend.
        void to_device(DeviceBackend B)
        {
            values.to_device(B);
        }

        /// @brief Returns the current device backend where BC values reside.
        DeviceBackend device()
        {
            return values.device();
        }

        template <DeviceBackend B>
        using t_deviceView = BC_DeviceView<B>; ///< Device view type alias.

        /**
         * @brief Creates a device-callable view of this BC object.
         * @tparam B Target device backend.
         * @return A `BC_DeviceView<B>` for use in device kernels.
         */
        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DNDS_assert_info(values.device() == B || (B == DeviceBackend::Host || B == DeviceBackend::Unknown),
                             "not on this device: " + std::string(device_backend_name(B)));
            return t_deviceView<B>(values.deviceView<B, int32_t>(), id, type);
        }
    };

    /**
     * @brief Device-callable view of the BC handler providing BC lookup by zone ID.
     *
     * Holds a device-resident array of `BC_DeviceView` objects indexed by boundary zone ID.
     *
     * @tparam B Device backend (Host or CUDA).
     */
    template <DeviceBackend B>
    class BCHandlerDeviceView
    {
        vector_DeviceView<B, BC_DeviceView<B>, int32_t> bcs; ///< Device array of BC views indexed by zone ID.

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(BCHandlerDeviceView, BCHandlerDeviceView)

        /// @brief Constructs from a device view of BC_DeviceView array.
        DNDS_DEVICE_CALLABLE BCHandlerDeviceView(vector_DeviceView<B, BC_DeviceView<B>, int32_t> n_bcs) : bcs(n_bcs) {}

        /**
         * @brief Looks up a boundary condition by zone ID.
         * @param id Boundary zone identifier.
         * @return Reference to the `BC_DeviceView` for the given zone.
         */
        DNDS_DEVICE_CALLABLE BC_DeviceView<B> &id2bc(Geom::t_index id)
        {
            DNDS_HD_assert(id < bcs.size() && id >= 0);
            return bcs[id];
        }
    };

    /**
     * @brief Simple struct for JSON-deserialized boundary condition input specification.
     *
     * Used to read BC definitions from JSON configuration files before constructing
     * the `BCHandler.` JSON-serializable via nlohmann_json intrusive macros.
     */
    struct BCInput
    {
        BCType type;              ///< Boundary condition type.
        std::string name;         ///< Boundary zone name (must match mesh zone names).
        std::vector<real> value;  ///< BC parameter values (meaning depends on BC type).
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(
            BCInput,
            type, name, value)
    };

    /**
     * @brief Host-side boundary condition handler managing all BC objects for a simulation.
     *
     * Constructed from a list of `BCInput` specifications and a name-to-ID mapping.
     * Automatically assigns default BC types for standard mesh zones (WALL, WALL_INVIS,
     * FAR, and SPECIAL benchmark zones). Supports host/device transfer of all managed BCs.
     *
     * Use `id2bc(id)` to look up a BC by zone ID, and `deviceView<B>()` to obtain a
     * `BCHandlerDeviceView` for kernel invocation.
     */
    class BCHandler
    {
        std::vector<BC> bcs; ///< Array of BC objects indexed by zone ID.

    public:
        /**
         * @brief Constructs the BC handler from JSON-deserialized inputs and a name-to-ID map.
         *
         * Allocates BC slots for all zone IDs, applies user-specified BC inputs, then
         * sets default types for standard zones:
         * - `BC_ID_DEFAULT_WALL` → BCType::Wall
         * - `BC_ID_DEFAULT_WALL_INVIS` → BCType::WallInvis
         * - `BC_ID_DEFAULT_FAR` → BCType::Far
         * - Special benchmark zone IDs → BCType::Special
         *
         * @param bc_inputs Vector of BCInput specifications from JSON configuration.
         * @param name2id Name-to-ID mapping from mesh zone names to integer IDs.
         * @throws std::runtime_error If a BC input name is not found in name2id.
         */
        BCHandler(const std::vector<BCInput> &bc_inputs, Geom::AutoAppendName2ID &name2id)
        {
            bcs.resize(name2id.id_cap);
            for (auto &[name, id] : name2id.n2id_map)
            {
                if (id >= 0)
                    bcs.at(id).setId(id);
            }
            for (const auto &input : bc_inputs)
            {
                DNDS_assert_info(name2id.n2id_map.count(input.name), fmt::format("bc input [{}] name not found in name2id", input.name));
                auto id = name2id.n2id_map.at(input.name);
                bcs.at(id).setType(input.type);
                bcs.at(id).setId(id);
                bcs.at(id).setValues(input.value);
            }
            {
                auto &bcC = bcs.at(Geom::BC_ID_DEFAULT_WALL);
                DNDS_assert_info(bcC.getType() == BCType::Unknown, "you should not input WALL as custom bc");
                bcC.setType(BCType::Wall);
            }
            {
                auto &bcC = bcs.at(Geom::BC_ID_DEFAULT_WALL_INVIS);
                DNDS_assert_info(bcC.getType() == BCType::Unknown, "you should not input WALL_INVIS as custom bc");
                bcC.setType(BCType::WallInvis);
            }
            {
                auto &bcC = bcs.at(Geom::BC_ID_DEFAULT_FAR);
                DNDS_assert_info(bcC.getType() == BCType::Unknown, "you should not input FAR as custom bc");
                bcC.setType(BCType::Far);
            }
            for (auto id : std::vector<Geom::t_index>{Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR,
                                                      Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR,
                                                      Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR,
                                                      Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR})
            {
                auto &bcC = bcs.at(id);
                DNDS_assert_info(bcC.getType() == BCType::Unknown, "you should set default BC id" + std::to_string(id));
                bcC.setType(BCType::Special);
            }
        }

        /**
         * @brief Looks up a boundary condition by zone ID.
         * @param id Boundary zone identifier.
         * @return Reference to the `BC` object for the given zone.
         */
        BC &id2bc(Geom::t_index id)
        {
            DNDS_HD_assert(id < bcs.size());
            return bcs[id];
        }

        /// @brief Transfers all BC values to host memory.
        void to_host()
        {
            for (auto &bc : bcs)
                bc.to_host();
        }

        /// @brief Transfers all BC values to the specified device backend.
        /// @param B Target device backend.
        void to_device(DeviceBackend B)
        {
            for (auto &bc : bcs)
                bc.to_device(B);
        }

        /// @brief Returns the device backend where BCs reside (asserts all BCs are on the same device).
        DeviceBackend device()
        {
            DeviceBackend B = DeviceBackend::Unknown;
            if (bcs.size())
                B = bcs[0].device();
            for (auto &bc : bcs)
            {
                DNDS_assert(bc.device() == B);
            }
            return B;
        }

        /**
         * @brief Move-only device view wrapper owning the BC device view storage.
         *
         * Wraps a `host_device_vector` of `BC_DeviceView` and the resulting
         * `BCHandlerDeviceView.` Move-only to prevent accidental copies that
         * would invalidate device pointers.
         *
         * @tparam B Device backend.
         */
        template <DeviceBackend B>
        struct t_deviceView
        {
            host_device_vector<BC_DeviceView<B>> bcs_device_view; ///< Owned device storage of BC views.
            BCHandlerDeviceView<B> view;                           ///< The handler device view referencing bcs_device_view.

            t_deviceView(host_device_vector<BC_DeviceView<B>> &&n_bcs_device_view)
                : bcs_device_view(n_bcs_device_view),
                  view(bcs_device_view.template deviceView<B, int32_t>())
            {
            }

            //! only permit moving to avoid host_device_vector to change
            // also avoids accidentally copying this to device...
            t_deviceView(t_deviceView &&R) noexcept = default;
            t_deviceView(const t_deviceView &R) = delete;
            t_deviceView &operator=(t_deviceView &&R) = delete;
            t_deviceView &operator=(const t_deviceView &R) = delete;

            operator BCHandlerDeviceView<B>() const
            {
                return view;
            }
        };

        /**
         * @brief Creates a device view of all managed BCs for kernel invocation.
         *
         * Builds a `host_device_vector` of `BC_DeviceView` from each managed BC,
         * transfers it to the specified device backend, and returns an owning
         * `t_deviceView` wrapper. The returned object is move-only.
         *
         * @tparam B Target device backend.
         * @return A `t_deviceView<B>` owning the device BC array.
         */
        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            host_device_vector<BC_DeviceView<B>> bcs_device_view;
            bcs_device_view.resize(bcs.size());
            index i = 0;
            for (auto &bc : bcs)
            {
                DNDS_assert(bc.device() == B ||
                            (B == DeviceBackend::Host && bc.device() == DeviceBackend::Unknown));
                bcs_device_view[i++] = bc.template deviceView<B>();
            }
            bcs_device_view.to_device(B);

            return t_deviceView<B>{std::move(bcs_device_view)};
        }
    };
}

namespace DNDS
{
    //     DNDS_DEVICE_STORAGE_BASE_DELETER_INST(EulerP::BC_DeviceView<DeviceBackend::Host>, extern)
    //     DNDS_DEVICE_STORAGE_INST(EulerP::BC_DeviceView<DeviceBackend::Host>, DeviceBackend::Host, extern)
    // #ifdef DNDS_USE_CUDA
    //     DNDS_DEVICE_STORAGE_BASE_DELETER_INST(EulerP::BC_DeviceView<DeviceBackend::CUDA>, extern)
    //     DNDS_DEVICE_STORAGE_INST(EulerP::BC_DeviceView<DeviceBackend::CUDA>, DeviceBackend::CUDA, extern)
    // #endif
}
