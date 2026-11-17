#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::Runtime
{
    /** @brief Project a uniform borrowed field once at a C++ kernel boundary.
     * @details This changes only the access type. It does not acquire access,
     * retain storage, move values or infer stream readiness. The submitting
     * operator retains those responsibilities. Width and physical pitch are
     * distinct: FixedRows<W> accepts padded AoS, FixedRows<W,S> checks S too.
     * Empty optional operands may remain unbound. Residency is checked with
     * ValidateViewDevices by the enclosing kernel wrapper, never per element.
     */
    template <class Policy, class T, bool Checked, class SourcePolicy>
    ArrayView<T, Checked, Policy> ProjectKernelView(ArrayView<T, Checked, SourcePolicy> source)
    {
        const auto view = source.DynamicView();
        using Traits = Detail::RowPolicyTraits<Policy>;
        static_assert(!std::is_same_v<Policy, RowLayout>, "kernel projection needs an explicit uniform access policy");
        DNDS_check_throw_info(view.rows >= 0 && view.width >= 0 && !view.offsets && !view.sizes,
                              "kernel view projection requires uniform row metadata");
        if (view.rows)
        {
            DNDS_check_throw_info(!view.width || view.data, "nonempty kernel field has no data");
            DNDS_check_throw_info(view.kind == RowLayoutKind::Table || view.kind == RowLayoutKind::ColMajorTable ||
                                      view.kind == RowLayoutKind::TiledColMajorTable,
                                  "kernel view projection requires a physical table");
            DNDS_check_throw_info(view.kind == RowLayoutKind::Table ? view.stride >= view.width : view.stride > 0,
                                  "kernel view projection has invalid physical pitch");
            if constexpr (!Traits::is_dynamic)
                DNDS_check_throw_info(view.kind == Traits::layout_kind, "kernel view physical layout differs");
            if constexpr (Traits::row_width != DynamicExtent)
                DNDS_check_throw_info(view.width == Traits::row_width, "kernel view fixed width differs");
            if constexpr (Traits::row_stride != DynamicExtent)
                DNDS_check_throw_info(view.stride == Traits::row_stride, "kernel view fixed pitch differs");
            if constexpr (Traits::layout_kind == RowLayoutKind::TiledColMajorTable && Traits::component_stride != DynamicExtent)
                DNDS_check_throw_info(view.stride == Traits::component_stride, "kernel view tile size differs");
        }
        return {view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
    }

    /// Same projection for disjoint owned/ghost storage, preserving each pitch.
    template <class Policy, class T, bool Checked, class SourcePolicy>
    PairReadView<T, Checked, Policy> ProjectKernelPair(PairReadView<T, Checked, SourcePolicy> view)
    {
        DNDS_check_throw_info(!view.ghost.rows || view.owned.width == view.ghost.width,
                              "kernel owned/ghost logical widths differ");
        return {ProjectKernelView<Policy>(view.owned), ProjectKernelView<Policy>(view.ghost)};
    }
}
