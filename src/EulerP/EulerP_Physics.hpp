#pragma once
#include "EulerP.hpp"
#include "DNDS/JsonUtil.hpp"

namespace DNDS::EulerP
{

    struct PhysicsParams
    {
        //! only simple data here allowed
        real gamma = -1;
        real mu0 = -1;
        real cp = -1;
        real TRef = -1;
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_AND_UNORDERED_JSON(
            PhysicsParams,
            gamma, mu0, cp, TRef);
    };

    template <DeviceBackend B>
    struct PhysicsDeviceView
    {
        vector_DeviceView<B, int32_t> reference_values;
        PhysicsParams params;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(PhysicsDeviceView, PhysicsDeviceView)
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

        template <DeviceBackend B>
        using t_deviceView = PhysicsDeviceView<B>;

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            return t_deviceView<B>{
                reference_values.deviceView<B, int32_t>(),
                params};
        }
    };

}