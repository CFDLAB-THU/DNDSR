#include "Cauchy/State/CPU/StatePropertiesKernels.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus ValidateMeanState(EquationLayoutView<const real> layout, AdmissibilityView bounds,
                                      Runtime::ArrayView<const uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> activeRows) noexcept
    {
        return DispatchCPUEquationLayouts([&](auto values) -> NumericalStatus
                                          { return OMP::ForRowsStatus(0, values.RowCount(), [&](index iCell) -> NumericalStatus
                                                                      {
                if (activeRows.RowCount() && !activeRows(iCell, 0)) return {};
                if (!bounds.Admissible(ThermalAdmissibilityDetail::PointRow<decltype(values)>{values, iCell}))
                    return {NumericalError::InvalidState, 1, iCell};
                return {}; }); }, layout);
    }
    NumericalStatus RefreshDerivedRows(EquationBlockArrayView<const real> rows, EquationBlockArrayView<const real> species,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> values,
                                       std::array<int, DerivedQuantityNumber> columns, bool useTemperatureGuess, PointEvaluator &evaluator,
                                       index begin, index end) noexcept
    {
        auto rhoY = evaluator.SpeciesInput();
        if (end < 0)
            end = values.RowCount();
        for (index iRow = begin; iRow < end; ++iRow)
        {
            FlowState point;
            for (int component = 0; component < 5; ++component)
                point[component] = rows(iRow, component);
            for (int k = 0; k < rhoY.nSpecies; ++k)
                rhoY[k] = species(iRow, k);
            const real temperatureGuess = useTemperatureGuess ? values(iRow, columns[int(DerivedQuantity::Temperature)]) : 0;
            const auto properties = TryEvaluateDerivedPoint(evaluator, point, rhoY, temperatureGuess);
            if (!properties)
                return properties.status.At(iRow);
            for (int quantity = 0; quantity < DerivedQuantityNumber; ++quantity)
                if (columns[quantity] >= 0)
                    values(iRow, columns[quantity]) = properties.value[quantity];
        }
        return {};
    }
    NumericalStatus RefreshDerivedRows(EquationBlockArrayView<const real> flow, EquationBlockArrayView<const real> species,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> output,
                                       std::array<int, DerivedQuantityNumber> columns, bool guess, SpanView<StatePropertiesWorker> workers) noexcept
    {
        if (workers.empty())
            return {NumericalError::InvalidInput, 1, -1};
        return OMP::ForRowsStatus(0, workers.size(), [&](index part) -> NumericalStatus
                                  {
            const index begin = part * (output.RowCount() / workers.size()) + std::min(part, output.RowCount() % workers.size());
            const index end = begin + output.RowCount() / workers.size() + (part < output.RowCount() % workers.size());
            return RefreshDerivedRows(flow, species, output, columns, guess, workers[part].evaluator, begin, end); });
    }
    NumericalStatus AccumulateStateDiagnostics(EquationBlockReadView flow, std::optional<EquationBlockReadView> species,
                                               Algebra::DofComponentView<const real> rhs, CFV::FiniteVolumeHostView metric,
                                               PointEvaluator &evaluator, SpanView<real> result, index begin, index end) noexcept
    {
        const index width = rhs.ComponentNumber();
        const int independent = species ? int(species->ComponentNumber()) : 0;
        auto rhoY = evaluator.SpeciesInput();
        if (end < 0)
            end = rhs.RowCount();
        for (index iCell = begin; iCell < end; ++iCell)
        {
            FlowState value;
            for (int j = 0; j < 5; ++j)
                value[j] = flow(iCell, j);
            if (independent)
            {
                for (int k = 0; k < independent; ++k)
                    rhoY[k] = (*species)(iCell, k);
            }
            PointProperties point;
            if (const auto status = evaluator.TryEvaluate(value, rhoY, 0, point); !status)
                return status.At(iCell);
            const real volume = metric.GetCellVol(iCell);
            result[0] += volume * value[0];
            result[1] += volume * value[4];
            result[2] = std::min(result[2], value[0]);
            result[3] = std::min(result[3], point.thermo.pressure);
            result[13] = std::min(result[13], point.thermo.temperature);
            for (index j = 0; j < width; ++j)
                result[4] += rhs(iCell, j) * rhs(iCell, j) / volume;
        }
        return {};
    }
    NumericalStatus AccumulateStateDiagnostics(EquationBlockReadView flow, std::optional<EquationBlockReadView> species,
                                               Algebra::DofComponentView<const real> rhs, CFV::FiniteVolumeHostView metric,
                                               SpanView<StatePropertiesWorker> workers, SpanView<real> result) noexcept
    {
        if (workers.empty())
            return {NumericalError::InvalidInput, 1, -1};
        if (workers.size() == 1)
            return AccumulateStateDiagnostics(flow, species, rhs, metric, workers[0].evaluator, result);
        const auto status = OMP::ForRowsStatus(0, workers.size(), [&](index part) -> NumericalStatus
                                               {
            auto &values = workers[part].diagnostics;
            values.fill(0);
            values[2] = values[3] = values[13] = std::numeric_limits<real>::max();
            const index begin = part * (rhs.RowCount() / workers.size()) + std::min(part, rhs.RowCount() % workers.size());
            const index end = begin + rhs.RowCount() / workers.size() + (part < rhs.RowCount() % workers.size());
            return AccumulateStateDiagnostics(flow, species, rhs, metric, workers[part].evaluator,
                                              {values.data(), index(values.size())}, begin, end); });
        if (!status)
            return status;
        for (index part = 0; part < workers.size(); ++part)
            for (index j = 0; j < result.size(); ++j)
                if (j == 2 || j == 3 || j == 13)
                    result[j] = std::min(result[j], workers[part].diagnostics[j]);
                else
                    result[j] += workers[part].diagnostics[j];
        return {};
    }
}
