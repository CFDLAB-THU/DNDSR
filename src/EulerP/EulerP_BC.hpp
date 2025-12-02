#pragma once
#include "DNDS/DeviceStorage.hpp"
#include "EulerP.hpp"
#include "EulerP_Physics.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "Geom/Geometric.hpp"

namespace DNDS::EulerP
{
    enum class BCType : uint8_t
    {
        Unknown = 0,
        Far,
        Wall,
        WallInvis,
        WallIsothermal,
        Out,
        OutP,
        In,
        InPsTs,
        Sym,
        Special,
    };

#define DNDS_JSON_ENUM_CLASS_BCType_ADD(v) {BCType::v, #v}

    NLOHMANN_JSON_SERIALIZE_ENUM(
        BCType,
        {
            DNDS_JSON_ENUM_CLASS_BCType_ADD(Unknown),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(Wall),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(WallInvis),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(WallIsothermal),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(Out),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(OutP),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(In),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(InPsTs),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(Sym),
            DNDS_JSON_ENUM_CLASS_BCType_ADD(Special),
        })

    template <DeviceBackend B>
    using BCStorageVecDeviceView = vector_DeviceView<B, real, int32_t>;

    template <DeviceBackend B, BCType T>
    struct BCFunc_Impl;

#define DNDS_EULERP_BC_INTERFACE_APPLY                   \
    template <class tU, class tUOut, class tx, class tn> \
    DNDS_DEVICE_CALLABLE static void apply(              \
        tU &&U, tUOut &&UOut, int32_t uSiz,              \
        tx &&x, tn &&n,                                  \
        Geom::t_index id,                                \
        BCStorageVecDeviceView<B> value, PhysicsDeviceView<B> &phy)

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Far>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Wall>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::WallIsothermal>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Out>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::OutP>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::InPsTs>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Sym>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::Special>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::In>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    struct BCFunc_Impl<B, BCType::WallInvis>
    {
        DNDS_EULERP_BC_INTERFACE_APPLY
        {
            DNDS_HD_assert(false); // TODO
        }
    };

    template <DeviceBackend B>
    class BC_DeviceView
    {
        BCStorageVecDeviceView<B> values;
        Geom::t_index id = Geom::BC_ID_NULL;
        BCType type = BCType::Unknown;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(BC_DeviceView, BC_DeviceView)

        DNDS_DEVICE_CALLABLE BC_DeviceView(
            BCStorageVecDeviceView<B> n_values,
            Geom::t_index n_id,
            BCType n_type)
            : values(n_values), id(n_id), type(n_type) {}

        DNDS_DEVICE_CALLABLE Geom::t_index getId() const { return id; }
        DNDS_DEVICE_CALLABLE BCType getType() const { return type; }
        DNDS_DEVICE_CALLABLE int32_t getNValues() const { return values.size(); }
        DNDS_DEVICE_CALLABLE real value(int32_t i) const { return values[i]; }

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

    class BC
    {
        host_device_vector<real> values;
        Geom::t_index id = Geom::BC_ID_NULL;
        BCType type = BCType::Unknown;

    public:
        [[nodiscard]] Geom::t_index getId() const { return id; }
        void setId(Geom::t_index n_id) { id = n_id; }
        [[nodiscard]] BCType getType() const { return type; }
        void setType(BCType n_type) { type = n_type; }
        [[nodiscard]] int32_t getNValues() const { return size_t_to_signed<int32_t>(values.size()); }
        [[nodiscard]] real value(int i) const { return values.at(i); }
        void setValues(const std::vector<real> &v) { values = v; }
        [[nodiscard]] auto getValues() const { return std::vector<real>(values); }

        void to_host()
        {
            values.to_host();
        }

        void to_device(DeviceBackend B)
        {
            values.to_device(B);
        }

        DeviceBackend device()
        {
            return values.device();
        }

        template <DeviceBackend B>
        using t_deviceView = BC_DeviceView<B>;

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DNDS_assert_info(values.device() == B || (B == DeviceBackend::Host || B == DeviceBackend::Unknown),
                             "not on this device: " + std::string(device_backend_name(B)));
            return t_deviceView<B>(values.deviceView<B, int32_t>(), id, type);
        }
    };

    template <DeviceBackend B>
    class BCHandlerDeviceView
    {
        vector_DeviceView<B, BC_DeviceView<B>, int32_t> bcs;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(BCHandlerDeviceView, BCHandlerDeviceView)

        DNDS_DEVICE_CALLABLE BCHandlerDeviceView(vector_DeviceView<B, BC_DeviceView<B>, int32_t> n_bcs) : bcs(n_bcs) {}

        DNDS_DEVICE_CALLABLE BC_DeviceView<B> &id2bc(Geom::t_index id)
        {
            DNDS_HD_assert(id < bcs.size() && id >= 0);
            return bcs[id];
        }
    };

    struct BCInput
    {
        BCType type;
        std::string name;
        std::vector<real> value;
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(
            BCInput,
            type, name, value)
    };

    class BCHandler
    {
        std::vector<BC> bcs;

    public:
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

        BC &id2bc(Geom::t_index id)
        {
            DNDS_HD_assert(id < bcs.size());
            return bcs[id];
        }

        void to_host()
        {
            for (auto &bc : bcs)
                bc.to_host();
        }

        void to_device(DeviceBackend B)
        {
            for (auto &bc : bcs)
                bc.to_device(B);
        }

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

        template <DeviceBackend B>
        struct t_deviceView
        {
            host_device_vector<BC_DeviceView<B>> bcs_device_view;
            BCHandlerDeviceView<B> view;

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
