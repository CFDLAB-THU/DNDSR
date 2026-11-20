#pragma once

#include "DNDS/Runtime/Execution/KernelViews.hpp"

namespace DNDS::Cauchy
{
    /// Borrow the optional owned-plus-ghost scalar field of cell viscosities.
    /// The host operator retains the whole Array for the duration of the flux.
    struct CellCenterViscosityView
    {
        using Values = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
        Values values;

        template <class View>
        static CellCenterViscosityView From(const View &view)
        {
            if (!view.RowCount())
                return {};
            return {Runtime::ProjectKernelView<Runtime::FixedRows<1>>(view.DynamicView())};
        }

        /// A negative return selects pointwise transport. Boundary samples
        /// have no adjacent cell on the exterior side.
        DNDS_DEVICE_CALLABLE real AtFace(index cell, index other) const
        {
            return !values.RowCount() || other == UnInitIndex ? -1 : .5 * values(cell, 0) + .5 * values(other, 0);
        }
    };
}
