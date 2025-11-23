#pragma once
#include "DeviceStorage.hpp"
#include <type_traits>

namespace DNDS
{
    template <class TView, DeviceBackend B>
    struct deviceViewVector
    {
        static_assert(std::is_trivially_copyable_v<TView> && std::is_default_constructible_v<TView>,
                      "view elements must be trivially_copyable and default_constructible");
        host_device_vector<TView> views;

        template <class TGetView>
        deviceViewVector(int32_t nViews, TGetView &&GetView)
        {
            views.reserve(nViews);
            for (int32_t i = 0; i < nViews; i++)
                views.emplace_back(GetView(i));
            views.to_device(B);
        }

        auto deviceView()
        {
            return views.template deviceView<B>();
        }
    };
}