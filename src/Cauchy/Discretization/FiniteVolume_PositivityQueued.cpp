#include "Cauchy/Discretization/FiniteVolumePositivityInternal.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        template <class T>
        auto ComponentLayout(const EquationDofs &shape,
                             SpanView<const R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows>> views,
                             index firstField, index rows)
        {
            std::vector<Algebra::DofComponentColumn<T>> columns;
            for (index iBlock = 0; iBlock < shape.BlockNumber(); ++iBlock)
            {
                const auto location = shape.LocateBlock(iBlock);
                for (index j = 0; j < location.componentNumber; ++j)
                    columns.push_back({views[firstField + location.physicalField], {}, location.firstComponent + j});
            }
            return Algebra::DofComponentLayout<T>(std::move(columns), rows, 0);
        }
        template <class T>
        auto AoS(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
        {
            return R::ArrayView<T, R::AccessBoundsChecks, R::TableRows>{view.data, view.offsets, view.rows,
                                                                        view.width, view.stride, view.sizes, view.kind, view.device};
        }
    }
    void FiniteVolume::PositivityWorkspace::Apply(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                  const EquationDofs &input, const EquationDofs &unlimited, EquationDofs &flux, bool refresh)
    {
        const TimelineScope timeline("Cauchy/FV/PolynomialPP");
        const auto pin = state;
        auto &data = *pin;
        const auto device = data.owner->metrics.Device();
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == device && data.owner->reconstruction,
                              "queued modal PP stream, placement or plan differs");
        input.RequireCompatible(data.prototype);
        unlimited.RequireCompatible(flux);
        EquationDofOperations::RequireDistinct(input, flux);
        DNDS_check_throw_info(flux.PhysicalFieldNumber() == input.PhysicalFieldNumber() &&
                                  unlimited.PhysicalFieldNumber() == flux.PhysicalFieldNumber(),
                              "modal PP physical field grouping differs");
        for (index iBlock = 0; iBlock < input.BlockNumber(); ++iBlock)
        {
            const auto mean = input.LocateBlock(iBlock), prototype = data.prototype.LocateBlock(iBlock);
            const auto coefficient = flux.LocateBlock(iBlock), source = unlimited.LocateBlock(iBlock);
            DNDS_check_throw_info(mean.physicalField == prototype.physicalField && mean.firstComponent == prototype.firstComponent &&
                                      mean.componentNumber == prototype.componentNumber && coefficient.physicalField == mean.physicalField &&
                                      coefficient.firstComponent == mean.firstComponent * data.policy.modes &&
                                      coefficient.componentNumber == mean.componentNumber * data.policy.modes &&
                                      source.physicalField == coefficient.physicalField && source.firstComponent == coefficient.firstComponent &&
                                      source.componentNumber == coefficient.componentNumber,
                                  "modal PP logical/physical component mapping differs");
        }
        bool same = true;
        for (index iField = 0; iField < flux.PhysicalFieldNumber(); ++iField)
            same = same && flux.PhysicalField(iField).Owned().SharedStorage() == unlimited.PhysicalField(iField).Owned().SharedStorage();
        const bool useInitial = refresh && !same;
        if (useInitial)
            EquationDofOperations::RequireDistinct(unlimited, flux);
        const index cells = data.owner->metrics.NumCell();
        if (device == R::Device::Host())
        {
            struct Operands
            {
                EquationDofs outputShape;
                index fields;
                bool refresh;
                Operands(const EquationDofs &output, bool fresh) : outputShape(output.NumericAlias()), fields(output.PhysicalFieldNumber()), refresh(fresh) {}
            };
            const auto operands = std::make_shared<Operands>(flux, useInitial);
            std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> reads;
            std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> writes;
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
            {
                reads.push_back(R::ReadAs<R::UniformTableRows>(input.PhysicalField(iField).Owned()));
                writes.push_back(R::ReadWriteAs<R::UniformTableRows>(flux.PhysicalField(iField).Owned()));
            }
            if (useInitial)
                for (index iField = 0; iField < unlimited.PhysicalFieldNumber(); ++iField)
                    reads.push_back(R::ReadAs<R::UniformTableRows>(unlimited.PhysicalField(iField).Owned()));
            R::SubmitArrayBatch(engine, stream, [pin, operands, cells](R::ExecutionContext, auto reads, auto writes, auto beta, auto degree)
                                {
                const auto means = ComponentLayout(pin->prototype, reads, 0, cells);
                const auto output = ComponentLayout(operands->outputShape, writes, 0, cells);
                std::optional<Algebra::DofComponentLayout<const real>> initial;
                if (operands->refresh)
                    initial.emplace(ComponentLayout(operands->outputShape, reads, operands->fields, cells));
                const CPU::ModalPositivityRequest request{pin->geometry, pin->policy, means.View(),
                    initial ? initial->View() : Algebra::DofComponentView<const real>{}, output.View(), beta, degree, bool(initial)};
                const auto result = RequireNumericalSuccess(CPU::LimitPositiveModalCoefficients(request,
                    {pin->cpuWorkers.data(), index(pin->cpuWorkers.size())}), "queued cell-local modal PP");
                pin->limited = result.limited;
                pin->minimum = result.minimum; }, std::move(reads), std::move(writes), std::make_tuple(R::WriteAs<R::FixedRows<1>>(data.beta), R::WriteAs<R::FixedRows<1>>(data.degree)));
            return;
        }
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(data.device && input.PhysicalFieldNumber() == 1 &&
                                  input.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                  flux.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                  (!useInitial || unlimited.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table),
                              "CUDA modal PP requires combined AoS fields");
        data.pending = true;
        auto &scratch = *data.device;
        R::SubmitArrays(engine, stream, [pin, useInitial](R::ExecutionContext context, auto mean, auto initial, auto coefficients, auto beta, auto degree, auto status, auto count, auto minimum)
                        {
            const CUDA::ModalPositivityRequest request{pin->geometry, pin->policy, AoS(mean), AoS(initial),
                                                       AoS(coefficients), beta, degree, useInitial};
            CUDA::LimitPositiveModalCoefficients(context, request, {status, count, minimum}, pin->launch); }, R::ReadAs<R::UniformTableRows>(input.CoupledField().Owned()), R::ReadAs<R::UniformTableRows>(useInitial ? unlimited.CoupledField().Owned() : data.owner->deviceEmpty), R::ReadWriteAs<R::UniformTableRows>(flux.CoupledField().Owned()), R::WriteAs<R::FixedRows<1>>(data.beta), R::WriteAs<R::FixedRows<1>>(data.degree), R::WriteAs<R::FixedRows<1>>(scratch.partialStatus), R::WriteAs<R::FixedRows<1>>(scratch.partialCount), R::WriteAs<R::FixedRows<1>>(scratch.partialMinimum));
        R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto partialStatus, auto partialCount, auto partialMinimum, auto status, auto count, auto minimum)
                        {
            // Only compact per-CTA summaries are merged. No cell status/flag or
            // beta field is rescanned, and the controller observes no trial.
            CUDA::FinalizeModalPositivity(context,
                {partialStatus, partialCount, partialMinimum}, {status, count, minimum}); }, R::ReadWriteAs<R::FixedRows<1>>(scratch.partialStatus), R::ReadWriteAs<R::FixedRows<1>>(scratch.partialCount), R::ReadWriteAs<R::FixedRows<1>>(scratch.partialMinimum), R::WriteAs<R::FixedRows<1>>(scratch.status), R::WriteAs<R::FixedRows<1>>(scratch.count), R::WriteAs<R::FixedRows<1>>(scratch.minimum));
#else
        DNDS_check_throw_info(false, "CUDA modal PP is not built");
#endif
    }
}
