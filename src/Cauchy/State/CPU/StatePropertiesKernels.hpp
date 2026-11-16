#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/State/DerivedFields.hpp"
#include "Cauchy/State/EquationViews.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
namespace DNDS::Cauchy::CPU
{
    /// Point-sized private state; the operator reserves contexts before teams.
    struct StatePropertiesWorker
    {
        PointEvaluator evaluator;
        NumericalStatus status;
        std::array<real, 14> diagnostics{};
        explicit StatePropertiesWorker(PointEvaluator value) : evaluator(std::move(value)) {}
    };
    /// Same complete-equation mean predicate for implicit and explicit acceptance.
    /// Empty activeRows checks every owned row; otherwise zero entries skip
    /// unused rows (e.g. physical boundaries in an interface-only packet).
    NumericalStatus ValidateMeanState(EquationLayoutView<const real>, AdmissibilityView,
                                      Runtime::ArrayView<const uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> activeRows = {}) noexcept;
    NumericalStatus RefreshDerivedRows(EquationBlockArrayView<const real> flow, EquationBlockArrayView<const real> species,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> output,
                                       std::array<int, DerivedQuantityNumber> columns, bool useTemperatureGuess, PointEvaluator &,
                                       index begin = 0, index end = -1) noexcept;
    /// One private evaluator per disjoint contiguous row range; no leases/MPI.
    NumericalStatus RefreshDerivedRows(EquationBlockArrayView<const real>, EquationBlockArrayView<const real>,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>,
                                       std::array<int, DerivedQuantityNumber>, bool, SpanView<StatePropertiesWorker>) noexcept;
    NumericalStatus AccumulateStateDiagnostics(EquationBlockReadView flow, std::optional<EquationBlockReadView> species,
                                               Algebra::DofComponentView<const real> rhs, CFV::FiniteVolumeHostView metric,
                                               PointEvaluator &, SpanView<real> result, index begin = 0, index end = -1) noexcept;
    /// Deterministic range-local accumulation followed by worker-order merge.
    NumericalStatus AccumulateStateDiagnostics(EquationBlockReadView, std::optional<EquationBlockReadView>,
                                               Algebra::DofComponentView<const real>, CFV::FiniteVolumeHostView,
                                               SpanView<StatePropertiesWorker>, SpanView<real>) noexcept;
}
