#pragma once
/// @file DeviceTransferable.hpp
/// @brief CRTP mixin that provides uniform to_device/to_host/device/getDeviceArrayBytes
///        for classes that enumerate their device-managed arrays via
///        for_each_device_member(F&&).

#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "DNDS/ObjectUtils.hpp"

#include <fmt/format.h>

namespace DNDS
{
    /**
     * @brief CRTP mixin giving a class uniform `to_device` / `to_host` / `device` /
     * `getDeviceArrayBytes` methods.
     *
     * @details The derived class only has to expose its device-managed members
     * via a `for_each_device_member(F&&)` method template that invokes `F(m)`
     * once per `MemberRef<ArrayPairType>`:
     *
     * ```cpp
     * template <typename F>
     * void for_each_device_member(F&& f)
     * {
     *     for_each_member_list(device_array_list(), std::forward<F>(f));
     * }
     * ```
     *
     * Once that is in place, all four member functions below iterate every
     * registered pair and forward the device operation.
     */
    template <typename Derived>
    class DeviceTransferable
    {
    public:
        /// @brief Mirror every registered @ref DNDS::ArrayPair "ArrayPair" to the given device backend.
        void to_device(DeviceBackend B)
        {
            auto op = [B](auto &v)
            {
                v.ref.to_device(B);
            };
            self().for_each_device_member(op);
        }

        /// @brief Pull every registered pair back to host memory.
        void to_host()
        {
            auto op = [](auto &v)
            {
                v.ref.to_host();
            };
            self().for_each_device_member(op);
        }

        /// @brief Consistent device backend across all registered pairs.
        /// @details Asserts that every pair's father and son live on the same
        /// backend; throws a descriptive message if not.
        DeviceBackend device()
        {
            DeviceBackend B = DeviceBackend::Unknown;
            auto getB = [&B](auto &v)
            {
                if (v.ref.father)
                    B = v.ref.father->device();
            };
            self().for_each_device_member(getB);

            auto check = [&B](auto &v)
            {
                if (v.ref.father)
                    DNDS_assert_info(
                        B == v.ref.father->device(),
                        fmt::format("member [{}.father] expected to be on device {} but on {}",
                                    v.name,
                                    device_backend_name(B),
                                    device_backend_name(v.ref.father->device())));
                if (v.ref.son)
                    DNDS_assert_info(
                        B == v.ref.son->device(),
                        fmt::format("member [{}.son] expected to be on device {} but on {}",
                                    v.name,
                                    device_backend_name(B),
                                    device_backend_name(v.ref.son->device())));
            };
            self().for_each_device_member(check);

            return B;
        }

        /// @brief Total footprint of every registered father+son pair in bytes.
        index getDeviceArrayBytes()
        {
            index bytes = 0;
            auto accumulate = [&bytes](auto &v)
            {
                if (v.ref.father)
                    bytes += v.ref.father->FullSizeBytes();
                if (v.ref.son)
                    bytes += v.ref.son->FullSizeBytes();
            };
            self().for_each_device_member(accumulate);
            return bytes;
        }

    private:
        Derived &self() { return static_cast<Derived &>(*this); }
    };
}
