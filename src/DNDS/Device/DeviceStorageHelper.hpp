#pragma once
/// @file DeviceStorageHelper.hpp
/// @brief Helpers for shipping an array-of-views (e.g., `ArrayDeviceView`)
/// to the device in one contiguous buffer.

#include "DeviceStorage.hpp"
#include <type_traits>
#include "DNDS/Vector.hpp"

namespace DNDS
{
    /**
     * @brief Contiguous #host_device_vector of non-owning views, mirrored on a device.
     *
     * @details Many CUDA kernels need "a vector of array views" -- e.g., the
     * per-cell adjacency views for a mesh, bundled into one flat array on the
     * GPU. This helper takes a factory callback that produces the `i`-th view,
     * stores all views in a `host_device_vector`, and pushes the result to
     * the chosen backend in one call.
     *
     * @tparam TView  View type to store (must be trivially copyable).
     * @tparam B      Target device backend.
     */
    template <class TView, DeviceBackend B>
    struct deviceViewVector
    {
        static_assert(std::is_trivially_copyable_v<TView> && std::is_default_constructible_v<TView>,
                      "view elements must be trivially_copyable and default_constructible");
        /// @brief Vector of per-entry views, with a device mirror.
        host_device_vector<TView> views;

        /// @brief Build `nViews` views via `GetView(i)` and transfer them to device.
        template <class TGetView>
        deviceViewVector(int32_t nViews, TGetView &&GetView)
        {
            views.resize(nViews);
            for (int32_t i = 0; i < nViews; i++)
                views[i] = GetView(i);
            views.to_device(B);
        }

        /// @brief Device-side span over the stored views.
        auto deviceView()
        {
            return views.template deviceView<B>();
        }
    };
}