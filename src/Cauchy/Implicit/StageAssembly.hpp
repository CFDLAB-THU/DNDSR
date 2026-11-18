#pragma once

#include "Cauchy/Physics/EquationSchema.hpp"
#include <type_traits>

namespace DNDS::Cauchy
{
    /// Defect emits an integrated correction RHS. The equation operations emit
    /// the volume-normalized time operator; EquationAndRate also emits R_int/V.
    enum class StageAssembly
    {
        Defect,
        Equation,
        EquationAndRate
    };

    /// Select physical mass once per operator invocation. Identity retains all
    /// physical state components. ACM omits pressure from M U but preserves
    /// pressure-equation spatial rates and history.
    template <class Launch>
    decltype(auto) DispatchStageAccumulation(PhysicalAccumulationKind kind, Launch &&launch)
    {
        if (kind == PhysicalAccumulationKind::ConstantDensityVelocityPressure)
            return launch(std::integral_constant<PhysicalAccumulationKind, PhysicalAccumulationKind::ConstantDensityVelocityPressure>{});
        DNDS_assert(kind == PhysicalAccumulationKind::Identity);
        return launch(std::integral_constant<PhysicalAccumulationKind, PhysicalAccumulationKind::Identity>{});
    }

    /// Select supported compact equation widths once before host loops or
    /// device launches. Width zero preserves the runtime component count.
    template <class Launch>
    decltype(auto) DispatchStageShape(PhysicalAccumulationKind kind, index width, Launch &&launch)
    {
        return DispatchStageAccumulation(kind, [&](auto mass) -> decltype(auto)
                                         {
            if constexpr (decltype(mass)::value == PhysicalAccumulationKind::ConstantDensityVelocityPressure)
            {
                if (width == 4) return launch(mass, std::integral_constant<int, 4>{});
            }
            else if (width == 7) return launch(mass, std::integral_constant<int, 7>{});
            if (width == 5) return launch(mass, std::integral_constant<int, 5>{});
            if (width == 6) return launch(mass, std::integral_constant<int, 6>{});
            return launch(mass, std::integral_constant<int, 0>{}); });
    }

}
