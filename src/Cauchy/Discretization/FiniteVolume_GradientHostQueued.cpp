#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeGradientKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        using ReadRows = CPU::GradientReadView;
        using WriteRows = CPU::GradientWriteView;
        using ReadOperand = R::ArrayOperand<real, false, R::UniformTableRows>;
        using WriteOperand = R::ArrayOperand<real, true, R::UniformTableRows>;
        auto TableRows(WriteRows view)
        {
            return R::ArrayView<real, R::AccessBoundsChecks, R::TableRows>{
                view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
        }
        CPU::GradientScratchView ScratchWindow(WriteRows source, index first, index count)
        {
            auto view = TableRows(source);
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && first >= 0 && count >= 0 &&
                                      first <= view.rows && count <= view.rows - first,
                                  "gradient worker scratch slice is outside its table");
            if (first)
                view.data += first * view.stride;
            view.rows = count;
            return view;
        }
        void CheckDisjoint(const std::vector<R::Array<real>> &inputs, const std::vector<R::Array<real>> &outputs)
        {
            for (index i = 0; i < index(outputs.size()); ++i)
            {
                for (const auto &input : inputs)
                    DNDS_check_throw_info(!outputs[i].Storage().Overlaps(input.Storage()), "gradient output aliases a read operand");
                for (index j = 0; j < i; ++j)
                    DNDS_check_throw_info(!outputs[i].Storage().Overlaps(outputs[j].Storage()), "gradient outputs overlap");
            }
        }
        std::vector<Algebra::DofComponentColumn<const real>> MeanColumns(
            SpanView<const ReadRows> reads, const std::vector<std::array<index, 2>> &slots,
            SpanView<const index> logicalField, SpanView<const index> logicalComponent)
        {
            std::vector<Algebra::DofComponentColumn<const real>> columns;
            columns.reserve(size_t(logicalField.size()));
            for (index j = 0; j < logicalField.size(); ++j)
            {
                const auto field = slots[logicalField[j]];
                columns.push_back({reads[field[0]], field[1] >= 0 ? reads[field[1]] : ReadRows{}, logicalComponent[j]});
            }
            return columns;
        }
    }

    void FiniteVolume::Workspace::SubmitGradientLimiter(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state, bool barth)
    {
        FiniteVolume(owner).ValidateWorkspace(state, *this);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  nativeDirectGradient && directGradients.size() == fields.size(),
                              "gradient limiter stream, placement or workspace differs");
        if (owner->metrics.Device() != R::Device::Host())
        {
            SubmitDeviceGradientLimiter(engine, stream, state, barth, CFV::ReconstructionRegion::All);
            return;
        }
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            Geom::MeshReadAccess geometry;
            R::MetadataBuffer<index> logicalField, logicalComponent, vectorStarts;
            std::vector<std::array<index, 2>> means;
            std::vector<index> gradients;
            index components, owned, ghost;
            explicit Inputs(std::shared_ptr<const Data> data) : owner(std::move(data)),
                                                                geometry(owner->metrics.AcquireMeshRead()) {}
        };
        auto pin = std::make_shared<Inputs>(owner);
        pin->logicalField = logicalField;
        pin->logicalComponent = logicalComponent;
        pin->vectorStarts = vectorStarts;
        pin->components = state.ComponentNumber();
        pin->owned = owner->metrics.NumCell();
        pin->ghost = owner->metrics.NumCellProc() - pin->owned;
        DNDS_check_throw_info(directLimiterScratch.RowCount() == 6 * cpuGradientWorkers && directLimiterScratch.Layout().Width() == size_t(pin->components) &&
                                  directLimiterScratch.Layout().Kind() == R::RowLayoutKind::Table,
                              "gradient limiter scratch shape differs");
        std::vector<ReadOperand> reads;
        std::vector<WriteOperand> writes;
        std::vector<R::Array<real>> inputArrays, outputArrays;
        const auto addRead = [&](const R::Array<real> &array)
        {
            const index slot = index(reads.size());
            inputArrays.push_back(array);
            reads.push_back(R::ReadAs<R::UniformTableRows>(array));
            return slot;
        };
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            pin->means.push_back({addRead(state.PhysicalField(iField).Owned()), addRead(state.PhysicalField(iField).Ghost())});
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            DNDS_check_throw_info(fields[iField].gradient, "direct limiter has no unlimited gradients");
            pin->gradients.push_back(addRead(fields[iField].gradient->Owned()));
            outputArrays.push_back(directGradients[iField].Owned());
            writes.push_back(R::WriteAs<R::UniformTableRows>(directGradients[iField].Owned()));
        }
        outputArrays.push_back(directLimiterScratch);
        writes.push_back(R::WriteAs<R::UniformTableRows>(directLimiterScratch));
        CheckDisjoint(inputArrays, outputArrays);
        R::SubmitArrayBatch(engine, stream, [pin, barth, count = cpuGradientWorkers](R::ExecutionContext, auto reads, auto writes, auto samples, auto offsets, auto cells)
                            {
            const auto columns = MeanColumns(reads, pin->means, pin->logicalField.HostView(), pin->logicalComponent.HostView());
            std::vector<ReadRows> source;
            std::vector<WriteRows> destination;
            for (const index slot : pin->gradients) source.push_back(reads[slot]);
            for (index iField = 0; iField < index(pin->gradients.size()); ++iField) destination.push_back(writes[iField]);
            CPU::GradientLimiterRequest request;
            request.stencil.samples = samples;
            request.stencil.offsets = offsets;
            request.metric.cells = cells;
            request.means = {{columns.data(), index(columns.size())}, pin->owned, pin->ghost, pin->components};
            request.source = {source.data(), index(source.size())}; request.destination = {destination.data(), index(destination.size())};
            request.logicalField = pin->logicalField.HostView(); request.logicalComponent = pin->logicalComponent.HostView();
            request.vectorStarts = pin->vectorStarts.HostView();
            request.scratch = TableRows(writes[index(destination.size())]);
            request.components = pin->components; request.dimension = pin->owner->metrics.Dimension(); request.barth = barth;
            std::vector<CPU::GradientScratchView> workers;
            workers.reserve(size_t(count));
            for (index iWorker = 0; iWorker < count; ++iWorker)
                workers.push_back(ScratchWindow(writes[index(destination.size())], 6 * iWorker, 6));
            CPU::LimitPhysicalGradients(request, {workers.data(), index(workers.size())}); }, std::move(reads), std::move(writes), std::make_tuple(R::ReadAs<R::FixedRows<1>>(owner->stencil.samples), R::ReadAs<R::FixedRows<1>>(owner->stencil.offsets), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics())));
    }

    std::shared_ptr<FiniteVolume::Workspace::GradientPositivitySubmission> FiniteVolume::Workspace::SubmitGradientPositivity(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state)
    {
        FiniteVolume(owner).ValidateWorkspace(state, *this);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() && nativeDirectGradient &&
                                  owner->settings.numerics.positivity && directGradients.size() == fields.size() &&
                                  owner->stencil.positivityOffsets.RowCount() == owner->metrics.NumCell() + 1,
                              "gradient PP stream, placement or workspace differs");
        DNDS_check_throw_info(!pendingGradientStatistics || pendingGradientStatistics->published,
                              "publish the previous gradient PP diagnostics before resubmission");
        const auto result = std::make_shared<GradientPositivitySubmission>();
        if (owner->metrics.Device() != R::Device::Host())
        {
#ifdef DNDS_USE_CUDA
            SubmitDeviceGradientPositivity(engine, stream, state);
            const auto device = owner->metrics.Device();
            if (!gradientStatistics)
            {
                const index partials = CUDA::StateReductionScratchRows(owner->metrics.NumCell());
                gradientStatistics.emplace(LimiterStatisticsStorage{
                    R::Array<index>(R::RowLayout::Dense(partials, 1), device), R::Array<index>(R::RowLayout::Dense(1, 1), device),
                    R::Array<real>(R::RowLayout::Dense(partials, 1), device), R::Array<real>(R::RowLayout::Dense(1, 1), device)});
                gradientStatistics->partialCount.SetName("Cauchy/gradientPP/countPartials", DNDS_ALLOCATION_SITE());
                gradientStatistics->count.SetName("Cauchy/gradientPP/count", DNDS_ALLOCATION_SITE());
                gradientStatistics->partialMinimum.SetName("Cauchy/gradientPP/minimumPartials", DNDS_ALLOCATION_SITE());
                gradientStatistics->minimum.SetName("Cauchy/gradientPP/minimum", DNDS_ALLOCATION_SITE());
            }
            auto &statistics = *gradientStatistics;
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto beta, auto counts, auto minima, auto count, auto minimum)
                            { CUDA::ReduceLimiterStatistics(context, beta, counts, minima, count, minimum); }, R::ReadAs<R::TableRows>(preparationStatus->gradientBeta), R::WriteAs<R::FixedRows<1>>(statistics.partialCount), R::WriteAs<R::FixedRows<1>>(statistics.partialMinimum), R::WriteAs<R::FixedRows<1>>(statistics.count), R::WriteAs<R::FixedRows<1>>(statistics.minimum));
            result->deviceCount = statistics.count;
            result->deviceMinimum = statistics.minimum;
            result->countEpoch = statistics.count.MutationEpoch();
            result->minimumEpoch = statistics.minimum.MutationEpoch();
#else
            DNDS_check_throw_info(false, "CUDA gradient PP is not built");
#endif
            pendingGradientStatistics = result;
            return result;
        }
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            Geom::MeshReadAccess geometry;
            ThermalPolicy thermal;
            R::MetadataBuffer<index> logicalField, logicalComponent;
            std::vector<std::array<index, 2>> means;
            index components, ransFirst;
            RANSReconstructionView rans;
            Inputs(std::shared_ptr<const Data> data, const ThermalPolicy &policy)
                : owner(std::move(data)), geometry(owner->metrics.AcquireMeshRead()), thermal(policy) {}
        };
        auto pin = std::make_shared<Inputs>(owner, thermal);
        pin->logicalField = logicalField;
        pin->logicalComponent = logicalComponent;
        pin->components = state.ComponentNumber();
        pin->ransFirst = ransFirst;
        pin->rans = owner->equation.HasRANS() ? ResolveRANSReconstructionPolicy(owner->equation) : RANSReconstructionView{};
        DNDS_check_throw_info(directPPSamples.Layout().Kind() == R::RowLayoutKind::Table &&
                                  directPPSamples.RowCount() == owner->maximumPositivityPoints * cpuGradientWorkers &&
                                  directPPSamples.Layout().Width() == size_t(pin->components),
                              "gradient PP scratch shape differs");
        std::vector<ReadOperand> reads;
        std::vector<WriteOperand> writes;
        std::vector<R::Array<real>> inputArrays, outputArrays;
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            pin->means.push_back({index(reads.size()), -1});
            inputArrays.push_back(state.PhysicalField(iField).Owned());
            reads.push_back(R::ReadAs<R::UniformTableRows>(state.PhysicalField(iField).Owned()));
            outputArrays.push_back(directGradients[iField].Owned());
            writes.push_back(R::ReadWriteAs<R::UniformTableRows>(directGradients[iField].Owned()));
        }
        outputArrays.push_back(directPPSamples);
        writes.push_back(R::WriteAs<R::UniformTableRows>(directPPSamples));
        CheckDisjoint(inputArrays, outputArrays);
        R::SubmitArrayBatch(engine, stream, [pin, result, count = cpuGradientWorkers](R::ExecutionContext, auto reads, auto writes, auto offsets, auto displacements)
                            {
            const auto columns = MeanColumns(reads, pin->means, pin->logicalField.HostView(), pin->logicalComponent.HostView());
            std::vector<WriteRows> gradients;
            for (index iField = 0; iField < index(pin->means.size()); ++iField) gradients.push_back(writes[iField]);
            CPU::GradientPositivityRequest request;
            request.stencil.positivityOffsets = offsets;
            request.stencil.positivityDisplacements = displacements;
            request.means = {{columns.data(), index(columns.size())}, pin->owner->metrics.NumCell(), 0, pin->components};
            request.gradients = {gradients.data(), index(gradients.size())};
            request.logicalField = pin->logicalField.HostView(); request.logicalComponent = pin->logicalComponent.HostView();
            request.samples = TableRows(writes[index(gradients.size())]); request.bounds = pin->thermal.View(); request.rans = pin->rans;
            request.components = pin->components; request.ransFirst = pin->ransFirst; request.dimension = pin->owner->metrics.Dimension();
            const auto &numerics = pin->owner->settings.numerics;
            request.betaPower = numerics.positivityBetaPower; request.betaCutoff = numerics.positivityBetaCutoff;
            request.clippedPolicy = numerics.positivityClippedReferencePolicy == "backtracking"
                ? ReconstructionClippedReferencePolicy::Backtracking : ReconstructionClippedReferencePolicy::FallbackToBase;
            std::vector<CPU::GradientPositivityWorkerScratch> workers(size_t(count), CPU::GradientPositivityWorkerScratch{});
            const index points = pin->owner->maximumPositivityPoints;
            for (index iWorker = 0; iWorker < count; ++iWorker)
                workers[iWorker].samples = ScratchWindow(writes[index(gradients.size())], points * iWorker, points);
            const auto value = RequireNumericalSuccess(CPU::LimitPositivePhysicalGradients(request,
                SpanView<CPU::GradientPositivityWorkerScratch>{workers.data(), index(workers.size())}), "queued gradient positivity");
            result->limited = value.limited; result->minimum = value.minimum;
            result->completed.store(true, std::memory_order_release); }, std::move(reads), std::move(writes), std::make_tuple(R::ReadAs<R::FixedRows<1>>(owner->stencil.positivityOffsets), R::ReadAs<R::FixedRows<3>>(owner->stencil.positivityDisplacements)));
        pendingGradientStatistics = result;
        return result;
    }

    void FiniteVolume::Workspace::PublishGradientPositivity(const std::shared_ptr<GradientPositivitySubmission> &result)
    {
        DNDS_check_throw_info(result && result == pendingGradientStatistics && !result->published,
                              "gradient PP diagnostics are foreign or already published");
        if (result->deviceCount)
        {
            DNDS_check_throw_info(result->deviceMinimum && result->deviceCount->Storage().IsIdle() && result->deviceMinimum->Storage().IsIdle() &&
                                      result->countEpoch == result->deviceCount->MutationEpoch() && result->minimumEpoch == result->deviceMinimum->MutationEpoch(),
                                  "gradient PP statistics need a drained unchanged segment");
            // Only two scalars cross the reporting boundary, never the
            // cell-sized beta field. The caller has checked numerical status.
            const auto count = result->deviceCount->To(R::Device::Host());
            const auto minimum = result->deviceMinimum->To(R::Device::Host());
            result->limited = count.Read().View()(0, 0);
            result->minimum = minimum.Read().View()(0, 0);
        }
        else
            DNDS_check_throw_info(result->completed.load(std::memory_order_acquire), "gradient PP kernels have not completed");
        directPPLimited = result->limited;
        directPPMinimum = result->minimum;
        result->published = true;
    }
}
