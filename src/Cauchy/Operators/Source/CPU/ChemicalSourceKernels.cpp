#include "Cauchy/Operators/Source/CPU/ChemicalSourceKernels.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    namespace
    {
        NumericalStatus AdvanceChemicalRange(SpanView<const EquationBlockArrayView<const real>> from,
                                             SpanView<const EquationBlockArrayView<real>> to, EquationBlockArrayView<const real> flow,
                                             EquationBlockArrayView<const real> species, EquationBlockArrayView<real> target,
                                             Physics::Thermodynamics::IdealGasMixture::PointEvaluator &evaluator, int independent,
                                             real dt, const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale,
                                             index begin, index end) noexcept
        {
            auto rhoY = evaluator.SpeciesInput();
            for (index iCell = begin; iCell < end; ++iCell)
            {
                for (index iBlock = 0; iBlock < index(from.size()); ++iBlock)
                    for (index j = 0; j < from[iBlock].ComponentNumber(); ++j)
                        to[iBlock](iCell, j) = from[iBlock](iCell, j);
                if (dt == 0 || sourceScale == 0)
                    continue;
                FlowState stateFlow;
                for (int j = 0; j < 5; ++j)
                    stateFlow[j] = flow(iCell, j);
                for (int k = 0; k < independent; ++k)
                    rhoY[k] = species(iCell, k);
                const auto updated = evaluator.TryAdvanceConstVolume(stateFlow[0], Physics::Thermodynamics::ConservativeSpecificInternalEnergy<3>(stateFlow),
                                                                     {rhoY.data, independent}, dt, controls, sourceScale);
                if (!updated)
                    return updated.status.At(iCell);
                for (int k = 0; k < independent; ++k)
                    target(iCell, k) = updated.value.rhoY[k];
            }
            return {};
        }
    }
    NumericalStatus AdvanceChemicalRows(SpanView<const EquationBlockArrayView<const real>> from,
                                        SpanView<const EquationBlockArrayView<real>> to, EquationBlockArrayView<const real> flow,
                                        EquationBlockArrayView<const real> species, EquationBlockArrayView<real> target,
                                        Physics::Thermodynamics::IdealGasMixture::PointEvaluator &evaluator, int independent,
                                        real dt, const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale) noexcept
    {
        // A caller explicitly lending one mutable reactor context requests one
        // serial range. The pool overload supplies actual private OMP workers.
        return AdvanceChemicalRange(from, to, flow, species, target, evaluator, independent, dt, controls, sourceScale, 0, flow.RowCount());
    }
    NumericalStatus AdvanceChemicalRows(SpanView<const EquationBlockArrayView<const real>> from,
                                        SpanView<const EquationBlockArrayView<real>> to, EquationBlockArrayView<const real> flow,
                                        EquationBlockArrayView<const real> species, EquationBlockArrayView<real> target,
                                        SpanView<Physics::Thermodynamics::IdealGasMixture::PointEvaluator> workers, int independent,
                                        real dt, const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale) noexcept
    {
        if (workers.empty())
            return {NumericalError::InvalidInput, 1, -1};
        return OMP::ForRowsStatus(0, workers.size(), [&](index part) -> NumericalStatus
                                  {
            const index begin = part * (flow.RowCount() / workers.size()) + std::min(part, flow.RowCount() % workers.size());
            const index end = begin + flow.RowCount() / workers.size() + (part < flow.RowCount() % workers.size());
            return AdvanceChemicalRange(from, to, flow, species, target, workers[part], independent, dt, controls, sourceScale, begin, end); });
    }
    NumericalStatus EvaluateChemicalRows(EquationBlockArrayView<const real> flow, EquationBlockArrayView<const real> rhoY,
                                         EquationBlockArrayView<real> sourceSpecies, SpanView<const EquationBlockArrayView<real>> nonchemical,
                                         SpanView<const index> columns, index components, index speciesFirst, int species, real sourceScale,
                                         std::optional<Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>> matrix,
                                         std::optional<EquationBlockArrayView<const real>> oldT, PointEvaluator &evaluator) noexcept
    {
        auto composition = evaluator.SpeciesInput();
        // Any evaluation failure can leave scratch partial. Do not cancel
        // leases or publish that failed call as a valid physical result.
        for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            for (const auto &block : nonchemical)
                for (index j = 0; j < block.ComponentNumber(); ++j)
                    block(iCell, j) = 0;
            if (matrix)
                for (index j = 0; j < components * components; ++j)
                    (*matrix)(iCell, j) = 0;
            if (sourceScale == 0)
            {
                for (int k = 0; k < species; ++k)
                    sourceSpecies(iCell, k) = 0;
                continue;
            }
            FlowState pointFlow;
            for (int j = 0; j < 5; ++j)
                pointFlow[j] = flow(iCell, j);
            for (int k = 0; k < species; ++k)
                composition[k] = rhoY(iCell, k);
            PointProperties point;
            if (const auto status = evaluator.TryEvaluate(pointFlow, {composition.data, composition.nSpecies}, oldT ? (*oldT)(iCell, 0) : 0, point); !status)
                return status.At(iCell);
            for (int k = 0; k < species; ++k)
            {
                const real rate = sourceScale * point.chemicalSource[k];
                if (!(std::isfinite(rate)))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                sourceSpecies(iCell, k) = rate;
                if (matrix)
                    for (index j = 0; j < index(columns.size()); ++j)
                    {
                        const real derivative = sourceScale * point.sourceJacobian(k, int(j));
                        if (!(std::isfinite(derivative)))
                            return NumericalStatus{NumericalError::InvalidState, 2, iCell};
                        (*matrix)(iCell, speciesFirst + k + components * columns[j]) = derivative;
                    }
            }
        }
        return {};
    }
}
