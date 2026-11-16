#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Operators/Source/ChemicalSource.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Core/Span.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus AdvanceChemicalRows(SpanView<const EquationBlockArrayView<const real>> from,
                                        SpanView<const EquationBlockArrayView<real>> to, EquationBlockArrayView<const real> flow,
                                        EquationBlockArrayView<const real> species, EquationBlockArrayView<real> target,
                                        Physics::Thermodynamics::IdealGasMixture::PointEvaluator &evaluator, int independent,
                                        real dt, const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale) noexcept;
    /// Exclusive pre-reserved reactor contexts; rows partition contiguously.
    NumericalStatus AdvanceChemicalRows(SpanView<const EquationBlockArrayView<const real>>,
                                        SpanView<const EquationBlockArrayView<real>>, EquationBlockArrayView<const real>,
                                        EquationBlockArrayView<const real>, EquationBlockArrayView<real>,
                                        SpanView<Physics::Thermodynamics::IdealGasMixture::PointEvaluator>, int,
                                        real, const Physics::Chemistry::ReactorStepSettings &, real) noexcept;
    NumericalStatus EvaluateChemicalRows(EquationBlockArrayView<const real> flow, EquationBlockArrayView<const real> rhoY,
                                         EquationBlockArrayView<real> sourceSpecies, SpanView<const EquationBlockArrayView<real>> nonchemical,
                                         SpanView<const index> columns, index components, index speciesFirst, int species, real sourceScale,
                                         std::optional<Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>> matrix,
                                         std::optional<EquationBlockArrayView<const real>> oldT, PointEvaluator &evaluator) noexcept;
}
