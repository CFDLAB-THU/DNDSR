#include "Cauchy/Discretization/FiniteVolumePositivityInternal.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumePositivityKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        enum class PPPhase
        {
            Validate,
            Mark,
            Truncate,
            Select,
            Compress,
            MarkRecovery,
            Recover,
            Verify
        };
        using ReadRows = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        using WriteRows = R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>;
        template <class T>
        auto ComponentLayout(const EquationDofs &shape,
                             SpanView<const R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows>> views,
                             const std::vector<index> &slots, index rows)
        {
            std::vector<Algebra::DofComponentColumn<T>> columns;
            for (index iBlock = 0; iBlock < shape.BlockNumber(); ++iBlock)
            {
                const auto location = shape.LocateBlock(iBlock);
                for (index j = 0; j < location.componentNumber; ++j)
                    columns.push_back({views[slots[location.physicalField]], {}, location.firstComponent + j});
            }
            return Algebra::DofComponentLayout<T>(std::move(columns), rows, 0);
        }
        template <class T>
        auto AoS(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
        {
            return R::ArrayView<T, R::AccessBoundsChecks, R::TableRows>{view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
        }
    }

    void FiniteVolume::PositivityWorkspace::Apply(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &input, const EquationDofs &unlimited, EquationDofs &flux, bool refresh)
    {
        const TimelineScope timeline("Cauchy/FV/PolynomialPP");
        const auto pin = state;
        auto &data = *pin;
        const auto &n = data.owner->settings.numerics;
        const auto device = data.owner->metrics.Device();
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == device && data.owner->reconstruction,
                              "queued polynomial PP stream, placement or geometry differs");
        input.RequireCompatible(data.prototype);
        unlimited.RequireCompatible(flux);
        EquationDofOperations::RequireDistinct(input, flux);
        DNDS_check_throw_info(input.PhysicalFieldNumber() == index(data.higherValues.size()) &&
                                  flux.PhysicalFieldNumber() == input.PhysicalFieldNumber() &&
                                  unlimited.PhysicalFieldNumber() == flux.PhysicalFieldNumber(),
                              "polynomial PP field grouping differs");
        const bool cuda = device != R::Device::Host();
        const index cells = data.owner->metrics.NumCell(), points = data.owner->quadrature.positivity->Size();
        CPU::PolynomialPositivityRequest base;
        base.bounds = data.policy.View();
        base.rans = data.rans;
        base.ransFirst = data.ransFirst;
        base.modes = data.owner->reconstruction->NRecDOF();
        base.maxOrder = n.cfv.maxOrder;
        for (index iBlock = 0; iBlock < input.BlockNumber(); ++iBlock)
        {
            const auto mean = input.LocateBlock(iBlock), prototype = data.prototype.LocateBlock(iBlock);
            const auto coefficient = flux.LocateBlock(iBlock), source = unlimited.LocateBlock(iBlock);
            DNDS_check_throw_info(mean.physicalField == prototype.physicalField && mean.firstComponent == prototype.firstComponent &&
                                      mean.componentNumber == prototype.componentNumber && coefficient.physicalField == mean.physicalField &&
                                      coefficient.firstComponent == mean.firstComponent * base.modes && coefficient.componentNumber == mean.componentNumber * base.modes &&
                                      source.physicalField == coefficient.physicalField && source.firstComponent == coefficient.firstComponent &&
                                      source.componentNumber == coefficient.componentNumber,
                                  "polynomial PP logical-to-physical field mapping differs");
        }
        base.betaPower = n.positivityBetaPower;
        base.betaCutoff = n.positivityBetaCutoff;
        base.clippedPolicy = n.positivityClippedReferencePolicy == "backtracking"
                                 ? ReconstructionClippedReferencePolicy::Backtracking
                                 : ReconstructionClippedReferencePolicy::FallbackToBase;
        if (n.positivityReconstruction == "Segment")
        {
            DNDS_check_throw_info(!data.owner->equation.HasMixture() && !data.owner->equation.HasRANS() && input.ComponentNumber() == 5,
                                  "Segment reconstruction PP is a flow5 compatibility policy");
            base.segmentBounds = data.owner->FixedFlowThermalBounds();
        }
        CUDA::PolynomialPositivityParameters gpuPolicy;
        if (cuda)
        {
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(input.PhysicalFieldNumber() == 1 && !data.owner->equation.HasMixture(),
                                  "CUDA polynomial PP needs combined perfect-gas fields");
            input.CoupledField();
            flux.CoupledField();
            gpuPolicy.bounds = base.bounds;
            gpuPolicy.rans = base.rans;
            gpuPolicy.components = int(input.ComponentNumber());
            gpuPolicy.modes = base.modes;
            gpuPolicy.maxOrder = base.maxOrder;
            gpuPolicy.betaPower = n.positivityBetaPower;
            gpuPolicy.betaCutoff = base.betaCutoff;
            gpuPolicy.clippedPolicy = base.clippedPolicy;
            gpuPolicy.segment = bool(base.segmentBounds);
            if (base.segmentBounds)
                gpuPolicy.segmentBounds = *base.segmentBounds;
            gpuPolicy.Validate();
            if (!data.device)
            {
                const index partials = CUDA::StateReductionScratchRows(cells);
                data.device.emplace(State::DeviceStorage{
                    R::Array<NumericalStatus>(R::RowLayout::Dense(cells, 1), device),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(cells + 1), 1), device),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), device),
                    R::Array<index>(R::RowLayout::Dense(partials, 1), device), R::Array<index>(R::RowLayout::Dense(1, 1), device),
                    R::Array<real>(R::RowLayout::Dense(partials, 1), device), R::Array<real>(R::RowLayout::Dense(1, 1), device)});
                data.device->status.SetName("Cauchy/modalPP/status", DNDS_ALLOCATION_SITE());
                data.device->statusPartial.SetName("Cauchy/modalPP/statusPartials", DNDS_ALLOCATION_SITE());
                data.device->statusResult.SetName("Cauchy/modalPP/statusResult", DNDS_ALLOCATION_SITE());
                data.device->partialCount.SetName("Cauchy/modalPP/countPartials", DNDS_ALLOCATION_SITE());
                data.device->count.SetName("Cauchy/modalPP/count", DNDS_ALLOCATION_SITE());
                data.device->partialMinimum.SetName("Cauchy/modalPP/minimumPartials", DNDS_ALLOCATION_SITE());
                data.device->minimum.SetName("Cauchy/modalPP/minimum", DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto status)
                            { CUDA::ClearNumericalStatus(context, status); }, R::WriteAs<R::FixedRows<1>>(data.device->statusResult));
#else
            DNDS_check_throw_info(false, "CUDA polynomial PP is not built");
#endif
        }
        const auto copy = [&](EquationDofs &to, const EquationDofs &from)
        {
            bool same = true;
            for (index iField = 0; iField < to.PhysicalFieldNumber(); ++iField)
                same = same && to.PhysicalField(iField).Owned().SharedStorage() == from.PhysicalField(iField).Owned().SharedStorage();
            if (same)
                return;
            EquationDofOperations::RequireDistinct(to, from);
            for (index iField = 0; iField < to.PhysicalFieldNumber(); ++iField)
                Algebra::Copy(engine, stream, to.PhysicalField(iField), from.PhysicalField(iField));
        };
        if (refresh)
            copy(flux, unlimited);
        const auto appendStatus = [&]
        {
#ifdef DNDS_USE_CUDA
            if (cuda)
                R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto partial, auto result)
                                { CUDA::ReduceNumericalStatus(context, rows, result.ConstView(), partial, result); }, R::ReadAs<R::FixedRows<1>>(data.device->status), R::WriteAs<R::FixedRows<1>>(data.device->statusPartial), R::ReadWriteAs<R::FixedRows<1>>(data.device->statusResult));
#endif
        };
        const auto submit = [&](PPPhase phase, int order = 0)
        {
            struct Operands
            {
                std::vector<index> means, higher, base, reference, destination;
                R::Array<index>::ReadLease offsets;
                EquationDofs destinationShape;
                Operands(const R::Array<index> &values, EquationDofs shape, R::Device device)
                    : offsets(values.Read({device, nullptr})), destinationShape(std::move(shape)) {}
            };
            const bool writeReference = phase == PPPhase::Truncate;
            const bool writeFlux = phase == PPPhase::Select || phase == PPPhase::Compress || phase == PPPhase::Recover;
            auto slots = std::make_shared<Operands>(data.owner->stencil.positivityOffsets,
                                                    writeReference ? *data.reference : flux, device);
            std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> reads;
            std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> writes;
            const auto read = [&](const R::Array<real> &array)
            { const index slot = index(reads.size()); reads.push_back(R::ReadAs<R::UniformTableRows>(array)); return slot; };
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
                slots->means.push_back(read(input.PhysicalField(iField).Owned()));
            if (phase == PPPhase::Mark || phase == PPPhase::Compress || phase == PPPhase::MarkRecovery || phase == PPPhase::Verify)
                for (const auto &array : data.higherValues)
                    slots->higher.push_back(read(array));
            if (phase == PPPhase::Select || (phase == PPPhase::Compress && data.reference))
                for (const auto &array : data.baseValues)
                    slots->base.push_back(read(array));
            if (data.reference && (phase == PPPhase::Select || phase == PPPhase::Compress || phase == PPPhase::Recover))
                for (index iField = 0; iField < data.reference->PhysicalFieldNumber(); ++iField)
                    slots->reference.push_back(read(data.reference->PhysicalField(iField).Owned()));
            if (writeReference || writeFlux)
                for (index iField = 0; iField < slots->destinationShape.PhysicalFieldNumber(); ++iField)
                {
                    slots->destination.push_back(index(writes.size()));
                    writes.push_back(R::ReadWriteAs<R::UniformTableRows>(slots->destinationShape.PhysicalField(iField).Owned()));
                }
            auto &status = cuda ? data.device->status : data.emptyStatus;
            R::SubmitArrayBatch(engine, stream, [pin, slots, base, gpuPolicy, phase, order, cells, points](R::ExecutionContext context, auto reads, auto writes, auto degree, auto beta, auto recover, auto scale, auto status)
                                {
                                    if (context.device == R::Device::Host())
                                    {
                                        const auto means = ComponentLayout(pin->prototype, reads, slots->means, cells);
                                        auto request = base;
                                        request.means = means.View();
                                        request.offsets = slots->offsets.View<R::FixedRows<1>>();
                                        using ReadLayout = Algebra::DofComponentLayout<const real>;
                                        std::optional<ReadLayout> higher, lower, reference;
                                        std::optional<Algebra::DofComponentLayout<real>> destination;
                                        if (!slots->higher.empty())
                                            higher.emplace(ComponentLayout(pin->prototype, reads, slots->higher, points));
                                        if (!slots->base.empty())
                                            lower.emplace(ComponentLayout(pin->prototype, reads, slots->base, points));
                                        if (!slots->reference.empty())
                                            reference.emplace(ComponentLayout(*pin->reference, reads, slots->reference, cells));
                                        if (!slots->destination.empty())
                                            destination.emplace(ComponentLayout(slots->destinationShape, writes, slots->destination, cells));
                                        const auto optionalReference = reference ? std::make_optional(reference->View()) : std::nullopt;
                                        switch (phase)
                                        {
                                        case PPPhase::Validate:
                                            RequireNumericalSuccess(CPU::ValidateReconstructionMeans(request), "queued PP means");
                                            break;
                                        case PPPhase::Mark:
                                            pin->decision = RequireNumericalSuccess(CPU::MarkAdmissibleDegrees(request, higher->View(), degree), "queued PP degree");
                                            break;
                                        case PPPhase::Truncate:
                                            CPU::TruncateReferenceModes(request, degree.ConstView(), destination->View(), order);
                                            break;
                                        case PPPhase::Select:
                                            CPU::SelectReferenceDegree(request, lower->View(), reference->View(), degree, destination->View(), order);
                                            break;
                                        case PPPhase::Compress:
                                            RequireNumericalSuccess(CPU::CompressPolynomial(request, higher->View(), lower ? std::make_optional(lower->View()) : std::nullopt,
                                                                                            optionalReference, degree.ConstView(), destination->View(), beta),
                                                                    "queued PP compression");
                                            break;
                                        case PPPhase::MarkRecovery:
                                            pin->decision = RequireNumericalSuccess(CPU::MarkPolynomialRecovery(request, higher->View(), beta, recover, scale), "queued PP recovery decision");
                                            break;
                                        case PPPhase::Recover:
                                            CPU::RecoverPolynomial(request, optionalReference, destination->View(), recover.ConstView(), scale.ConstView());
                                            break;
                                        case PPPhase::Verify:
                                            RequireNumericalSuccess(CPU::VerifyReconstructedPoints(request, higher->View()), "queued PP verification");
                                            break;
                                        }
                                        return;
                                    }
#ifdef DNDS_USE_CUDA
                                    CUDA::PolynomialPositivityRequest request{AoS(reads[slots->means.front()]), slots->offsets.View<R::FixedRows<1>>(), status, gpuPolicy};
                                    const auto higher = slots->higher.empty() ? CUDA::PPInput{} : AoS(reads[slots->higher.front()]);
                                    const auto lower = slots->base.empty() ? CUDA::PPInput{} : AoS(reads[slots->base.front()]);
                                    const auto reference = slots->reference.empty() ? CUDA::PPInput{} : AoS(reads[slots->reference.front()]);
                                    const auto destination = slots->destination.empty() ? CUDA::PPOutput{} : AoS(writes[slots->destination.front()]);
                                    switch (phase)
                                    {
                                    case PPPhase::Validate:
                                        CUDA::ValidateReconstructionMeans(context, request);
                                        break;
                                    case PPPhase::Mark:
                                        CUDA::MarkAdmissibleDegrees(context, request, higher, degree, recover);
                                        break;
                                    case PPPhase::Truncate:
                                        CUDA::TruncateReferenceModes(context, request, degree.ConstView(), destination, order);
                                        break;
                                    case PPPhase::Select:
                                        CUDA::SelectReferenceDegree(context, request, lower, reference, degree, destination, order);
                                        break;
                                    case PPPhase::Compress:
                                        CUDA::CompressPolynomial(context, request, higher, lower, reference, !slots->base.empty(), !slots->reference.empty(), degree.ConstView(), destination, beta);
                                        break;
                                    case PPPhase::MarkRecovery:
                                        CUDA::MarkPolynomialRecovery(context, request, higher, beta, recover, scale);
                                        break;
                                    case PPPhase::Recover:
                                        CUDA::RecoverPolynomial(context, request, reference, !slots->reference.empty(), destination, recover.ConstView(), scale.ConstView());
                                        break;
                                    case PPPhase::Verify:
                                        CUDA::VerifyReconstructedPoints(context, request, higher);
                                        break;
                                    }
#endif
                                },
                                std::move(reads), std::move(writes), std::make_tuple(R::ReadWriteAs<R::FixedRows<1>>(data.degree), R::ReadWriteAs<R::FixedRows<1>>(data.beta), R::ReadWriteAs<R::FixedRows<1>>(data.recover), R::ReadWriteAs<R::FixedRows<1>>(data.recoveryScale), R::WriteAs<R::FixedRows<1>>(status)));
            appendStatus();
        };
        const auto observe = [&]
        {
            // Explicit numerical decision boundary, shared by every backend.
            // TODO(sync-free-iteration-control): a future device controller can
            // replace this host observation; it must preserve dynamic stopping.
            engine.Synchronize();
            if (cuda)
            {
                const auto result = data.device->statusResult.To(R::Device::Host());
                RequireNumericalSuccess(result.Read().View()(0, 0), "queued polynomial positivity");
            }
        };
        const auto decision = [&](bool all)
        {
#ifdef DNDS_USE_CUDA
            if (cuda)
                R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto flags, auto partial, auto result)
                                { CUDA::CountFlags(context, flags, partial, result); }, R::ReadAs<R::FixedRows<1>>(data.recover), R::WriteAs<R::FixedRows<1>>(data.device->partialCount), R::WriteAs<R::FixedRows<1>>(data.device->count));
#endif
            observe();
            if (!cuda)
                return data.decision;
            const auto count = data.device->count.To(R::Device::Host());
            const auto value = count.Read().View()(0, 0);
            return all ? value == cells : value != 0;
        };
        submit(PPPhase::Validate);
        data.Sample(engine, stream, input, flux, data.higherValues);
        submit(PPPhase::Mark);
        if (decision(true) && !base.segmentBounds)
        {
            data.limited = 0;
            data.minimum = 1;
            return;
        }
        if (data.reference)
        {
            copy(*data.reference, flux);
            for (int order = n.cfv.maxOrder - 1; order >= 0; --order)
            {
                const int keep = order ? CFV::Math::PolynomialSize(data.owner->metrics.Dimension(), order) - 1 : 0;
                submit(PPPhase::Truncate, keep);
                data.Sample(engine, stream, input, *data.reference, data.baseValues);
                submit(PPPhase::Select, order);
            }
            data.Sample(engine, stream, input, flux, data.higherValues);
        }
        submit(PPPhase::Compress);
        for (int iteration = 0; iteration < 1000; ++iteration)
        {
            data.Sample(engine, stream, input, flux, data.higherValues);
            submit(PPPhase::MarkRecovery);
            if (!decision(false))
                break;
            submit(PPPhase::Recover);
        }
        data.Sample(engine, stream, input, flux, data.higherValues);
        submit(PPPhase::Verify);
        if (!cuda)
            R::SubmitArrays(engine, stream, [pin](R::ExecutionContext, auto beta)
                            {
                const auto statistics = CPU::PolynomialPositivityStatistics(beta);
                pin->limited = statistics.first; pin->minimum = statistics.second; }, R::ReadAs<R::FixedRows<1>>(data.beta));
#ifdef DNDS_USE_CUDA
        else
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto beta, auto counts, auto minima, auto count, auto minimum)
                            { CUDA::ReduceLimiterStatistics(context, beta, counts, minima, count, minimum); }, R::ReadAs<R::TableRows>(data.beta), R::WriteAs<R::FixedRows<1>>(data.device->partialCount), R::WriteAs<R::FixedRows<1>>(data.device->partialMinimum), R::WriteAs<R::FixedRows<1>>(data.device->count), R::WriteAs<R::FixedRows<1>>(data.device->minimum));
#endif
        observe();
        if (cuda)
        {
            const auto count = data.device->count.To(R::Device::Host());
            const auto minimum = data.device->minimum.To(R::Device::Host());
            data.limited = count.Read().View()(0, 0);
            data.minimum = minimum.Read().View()(0, 0);
        }
    }
}
