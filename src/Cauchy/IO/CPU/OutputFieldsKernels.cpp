#include "Cauchy/IO/CPU/OutputFieldsKernels.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    void PrepareOutputFields(EquationLayoutView<const real> layout, int species,
                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> output,
                             Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> status,
                             SpanView<OutputFieldsWorker> workers) noexcept
    {
        DNDS_assert(!workers.empty());
        DispatchCPUEquationLayouts([&](auto values)
                                   {
            const index rows = values.RowCount(), components = values.ComponentNumber();
            if (output.Width() == components)
            {
                OMP::ForRows(0, rows, [&](index iRow) {
                    NumericalStatus result;
                    for (index j = 0; j < components; ++j)
                    {
                        output(iRow, j) = values(iRow, j);
                        if (!std::isfinite(output(iRow, j))) result = NumericalStatus{NumericalError::InvalidState, int(j)}.At(iRow);
                    }
                    status(iRow, 0) = result;
                });
                return;
            }
            OMP::ForRows(0, workers.size(), [&](index iWorker)
            {
                auto &evaluator = workers[iWorker].evaluator;
                auto rhoY = evaluator.SpeciesInput();
                const index begin = iWorker * (rows / workers.size()) + std::min(iWorker, rows % workers.size());
                const index end = begin + rows / workers.size() + (iWorker < rows % workers.size());
                for (index iRow = begin; iRow < end; ++iRow)
                {
                    FlowState flow{};
                    for (int j = 0; j < 5; ++j) flow[j] = values(iRow, values.flowFirst + j);
                    for (int j = 0; j < rhoY.nSpecies; ++j) rhoY[j] = values(iRow, values.offsets[values.species] + j);
                    PointProperties properties;
                    status(iRow, 0) = evaluator.TryEvaluate(flow, rhoY, 0, properties);
                    if (!status(iRow, 0))
                    {
                        status(iRow, 0) = status(iRow, 0).At(iRow);
                        continue;
                    }
                    for (index j = 0; j < components; ++j) output(iRow, j) = values(iRow, j);
                    output(iRow, components) = properties.thermo.pressure;
                    output(iRow, components + 1) = properties.thermo.temperature;
                    real speedSquared = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        const real velocity = flow[d + 1] / flow[0];
                        output(iRow, components + 2 + d) = velocity;
                        speedSquared += velocity * velocity;
                    }
                    output(iRow, components + 5) = std::sqrt(speedSquared) / properties.thermo.soundSpeed;
                    for (int j = 0; j < species; ++j) output(iRow, components + 6 + j) = properties.fractions[j];
                }
            }); }, layout);
    }
}
