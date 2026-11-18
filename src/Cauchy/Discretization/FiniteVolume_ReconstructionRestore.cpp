#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/State/CUDA/FieldKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        /// Retain the execution runtime before acquiring immediate reservations.
        /// It outlives all reserved scopes, including an abandoned restore.
        /// No hidden device/default-stream engine is constructed. Standalone
        /// synchronous CPU callers may omit a lane; production supplies one.
        struct RestoreExecution
        {
            std::optional<R::ExecutionStream> stream;
            std::shared_ptr<R::EagerEngine> engine;
            R::ExecutionContext context;
            RestoreExecution(std::optional<R::ExecutionStream> lane, R::Device device)
                : stream(std::move(lane)), context(stream ? stream->Context() : R::ExecutionContext::Host())
            {
                DNDS_check_throw_info(context.device == device, "reconstruction restoration requires an explicit matching device stream");
                if (stream)
                {
                    // LockEngine returns a retaining runtime handle. C++17
                    // guaranteed elision constructs it without copying engine.
                    engine.reset(new R::EagerEngine(stream->LockEngine()));
                    engine->Synchronize();
                }
            }
            void Run(R::StreamLauncher::Work work) const
            {
                if (engine)
                {
                    engine->Submit(*stream, std::move(work));
                    engine->Synchronize();
                }
                else
                    work(context);
            }
            R::Completion Completed() const { return R::Completion::FromExternalCompletion(context); }
        };
        /// Validation scratch is temporary, bounded by 256 partial rows and
        /// reused across all restored blocks. Only a scalar status crosses D2H.
        struct RestoreValidation
        {
            const RestoreExecution &execution;
            R::Array<NumericalStatus> partial, result;
            RestoreValidation(const RestoreExecution &run, index rows) : execution(run)
            {
                if (run.context.device == R::Device::Host())
                    return;
#ifdef DNDS_USE_CUDA
                partial = R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::FieldCheckScratchRows(rows), 1), run.context.device);
                result = R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), run.context.device);
#else
                DNDS_check_throw_info(false, "CUDA reconstruction restore is not compiled");
#endif
            }
            void Check(EquationBlockArrayView<const real> values, bool capped = false, std::array<real, 2> caps = {})
            {
                values.ValidateDevice(execution.context.device);
                if (execution.context.device == R::Device::Host())
                {
                    const auto status = std::make_shared<NumericalStatus>();
                    execution.Run([values, capped, caps, status](auto)
                                  { *status = capped ? CPU::RequireCappedRows(values, {caps.data(), index(caps.size())})
                                                     : CPU::RequireFiniteRows(values, values.ComponentNumber()); });
                    RequireNumericalSuccess(*status, "reconstruction restore validation");
                }
                else
                {
#ifdef DNDS_USE_CUDA
                    {
                        auto scratch = partial.Write(execution.context), output = result.Write(execution.context);
                        const auto p = scratch.View<R::FixedRows<1>>(), r = output.View<R::FixedRows<1>>();
                        execution.Run([values, capped, caps, p, r](auto context)
                                      { CUDA::CheckFieldRows(context, values, capped, caps, p, r); });
                        scratch.Complete(execution.Completed());
                        output.Complete(execution.Completed());
                    }
                    const auto host = result.To(R::Device::Host(), execution.context);
                    const auto read = host.Read();
                    RequireNumericalSuccess(read.View<R::FixedRows<1>>()(0, 0), "reconstruction restore validation");
#else
                    DNDS_check_throw_info(false, "CUDA reconstruction restore is not compiled");
#endif
                }
            }
        };
    }
    void FiniteVolume::RestoreReconstructionGuess(ReconstructionState &state, const EquationDofs &guess,
                                                  OptionalRef<const Runtime::Array<real>> limitedVRHistory, std::optional<R::ExecutionStream> stream) const
    {
        auto restore = ReserveReconstructionRestoreImpl(state, guess, limitedVRHistory, {}, true, {}, stream);
        restore.Commit();
        if (_data->settings.numerics.limiter == "WBAP" || _data->limitedVRGeometry ||
            _data->settings.numerics.positivity)
        {
            state.finalizedFrom.reset();
            state.finalizedPublication.reset();
        }
        else
            state.PublishFinalized();
    }
    void FiniteVolume::CopyReconstructionState(ReconstructionState &destination,
                                               const ReconstructionState &source, std::optional<R::ExecutionStream> stream) const
    {
        DNDS_check_throw_info(&destination != &source && destination.owner == _data && source.owner == _data,
                              "reconstruction state copy requires distinct states from this finite-volume plan");
        DNDS_check_throw_info(source.FinalizedCurrent(),
                              "source reconstruction has no current finalized polynomial");
        const auto history = source.limitedVRHistory
                                 ? OptionalRef<const R::Array<real>>(*source.limitedVRHistory)
                                 : OptionalRef<const R::Array<real>>{};
        auto restore = ReserveReconstructionRestoreImpl(destination, source.WarmValues(), history, {}, true,
                                                        OptionalRef<const EquationDofs>(source.FinalizedValues()), stream);
        restore.Commit();
        destination.PublishFinalized();
    }
    void FiniteVolume::RestoreReconstructionState(ReconstructionState &state,
                                                  const EquationDofs &warm, const EquationDofs &finalized,
                                                  OptionalRef<const Runtime::Array<real>> limitedVRHistory, std::optional<R::ExecutionStream> stream) const
    {
        auto restore = ReserveReconstructionRestoreImpl(state, warm, limitedVRHistory, {}, true,
                                                        OptionalRef<const EquationDofs>(finalized), stream);
        restore.Commit();
        state.PublishFinalized();
    }
    std::optional<Runtime::Array<real>> FiniteVolume::LimitedVRHistory(const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
        return state.limitedVRHistory;
    }
    std::optional<Runtime::Array<real>> FiniteVolume::LimitedVRAlpha(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.limitedVR ? std::optional<R::Array<real>>(work.limitedVR->alpha) : std::nullopt;
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestore(std::shared_ptr<ReconstructionState> state, const EquationDofs &guess,
                                                                                   OptionalRef<const Runtime::Array<real>> limitedVRHistory,
                                                                                   bool collectiveChecks, std::optional<R::ExecutionStream> stream) const
    {
        const auto validate = [&]
        { DNDS_check_throw_info(bool(state), "null equation reconstruction state"); };
        R::CommunicationDetail::LocalStage(Metrics().Communicator(), collectiveChecks && _data->settings.implicit.collectiveChecks,
                                           validate, "reconstruction workspace validation failed");
        auto &reference = *state;
        return ReserveReconstructionRestoreImpl(reference, guess, limitedVRHistory, std::move(state), collectiveChecks, {}, stream);
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestoreImpl(ReconstructionState &state, const EquationDofs &guess,
                                                                                       OptionalRef<const Runtime::Array<real>> limitedVRHistory,
                                                                                       std::shared_ptr<ReconstructionState> retained, bool collectiveChecks,
                                                                                       OptionalRef<const EquationDofs> finalized, std::optional<R::ExecutionStream> stream) const
    {
        const auto *historyInput = limitedVRHistory.Get();
        const auto stage = [&](const auto &run)
        {
            R::CommunicationDetail::LocalStage(Metrics().Communicator(), collectiveChecks && _data->settings.implicit.collectiveChecks,
                                               run, "reconstruction restore preparation failed");
        };
        struct Reservation
        {
            // Declared first: scopes retire before their runtime/stream owner.
            std::shared_ptr<RestoreExecution> execution;
            std::shared_ptr<ReconstructionState> owner;
            R::SessionAccess::Lease session;
            std::optional<EquationDofs::ReadScope> source, fluxSource;
            std::optional<EquationDofs::WriteScope> destination, fluxDestination;
            std::optional<R::Array<real>::ReadLease> historySource;
            std::optional<R::Array<real>::WriteLease> historyDestination;
            R::MetadataBuffer<EquationBlockArrayView<const real>> input, fluxInput;
            R::MetadataBuffer<EquationBlockArrayView<real>> output, fluxOutput;
            bool started = false;
            ~Reservation()
            {
                if (!started && destination)
                    destination->CancelUnmodified();
                if (!started && fluxDestination)
                    fluxDestination->CancelUnmodified();
                if (!started && historyDestination)
                    historyDestination->CancelUnmodified();
            }
        };
        std::shared_ptr<Reservation> reservation;
        stage([&]
              {
                reservation = std::make_shared<Reservation>();
                reservation->execution = std::make_shared<RestoreExecution>(stream, Device());
                const auto context = reservation->execution->context;
                RestoreValidation validation(*reservation->execution, Metrics().NumCell());
                reservation->owner = std::move(retained);
                reservation->session = state.session.Acquire();
                DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
                // Direct P0/P1 has no modal storage. Reject before reserving
                // write access; means and physical gradients are separate data.
                DNDS_check_throw_info(_data->reconstruction,
                                      "direct P0/P1 has no modal reconstruction to restore");
                state.WarmValues().RequireCompatible(guess);
                if (context.device != R::Device::Host())
                {
                    (void)state.WarmValues().CoupledField();
                    (void)guess.CoupledField();
                    if (finalized) (void)finalized->CoupledField();
                }
                EquationDofOperations::RequireDistinct(state.WarmValues(), guess);
                const auto &finalInput = state.distinctFinalized && finalized ? *finalized : guess;
                if (state.distinctFinalized)
                {
                    state.MutableFinalized().RequireCompatible(guess);
                    EquationDofOperations::RequireDistinct(state.MutableFinalized(), guess);
                    state.MutableFinalized().RequireCompatible(finalInput);
                    EquationDofOperations::RequireDistinct(state.WarmValues(), finalInput);
                    EquationDofOperations::RequireDistinct(state.MutableFinalized(), finalInput);
                }
                DNDS_check_throw_info(state.limitedVRHistory || !historyInput,
                                      "LVR history requires an active matching reconstruction state");
                if (historyInput)
                {
                    DNDS_check_throw_info(historyInput->GetDevice() == context.device &&
                                              !historyInput->Layout().IsVariable() &&
                                              historyInput->RowCount() == Metrics().NumCell() &&
                                              historyInput->Layout().Width() == 2,
                                          "LVR history shape or placement differs");
                    for (index iField = 0; iField < guess.PhysicalFieldNumber(); ++iField)
                    {
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(),
                                                                         state.WarmValues().PhysicalField(iField).Owned().Storage()),
                                              "LVR history aliases reconstruction output");
                        if (state.distinctFinalized)
                            DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(),
                                                                             state.MutableFinalized().PhysicalField(iField).Owned().Storage()),
                                                  "LVR history aliases finalized reconstruction output");
                    }
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(), state.limitedVRHistory->Storage()),
                                          "LVR history aliases its destination");
                    reservation->historySource.emplace(historyInput->Read(context));
                    const auto values = reservation->historySource->View<R::UniformTableRows>();
                    const std::array<real, 2> caps{_data->settings.numerics.limitedVR.alphaMax,
                                                   _data->settings.numerics.limitedVR.contactAlphaMax};
                    validation.Check({values, 0, 2}, true, caps);
                }
                std::vector<EquationBlockArrayView<const real>> inputRows, fluxInputRows;
                std::vector<EquationBlockArrayView<real>> outputRows, fluxOutputRows;
                reservation->source.emplace(guess.ReadOwned(context));
                if (state.distinctFinalized && finalized)
                    reservation->fluxSource.emplace(finalInput.ReadOwned(context));
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                {
                    const auto input = reservation->source->Block(iBlock);
                    validation.Check(input.owned);
                    inputRows.push_back(input.owned);
                    if (state.distinctFinalized)
                    {
                        const auto fluxInput = finalized ? reservation->fluxSource->Block(iBlock) : input;
                        validation.Check(fluxInput.owned);
                        fluxInputRows.push_back(fluxInput.owned);
                    }
                }
                reservation->destination.emplace(state.WarmValues().WriteOwned(context));
                if (state.distinctFinalized)
                    reservation->fluxDestination.emplace(state.MutableFinalized().WriteOwned(context));
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                {
                    outputRows.push_back(reservation->destination->Block(iBlock));
                    if (state.distinctFinalized)
                        fluxOutputRows.push_back(reservation->fluxDestination->Block(iBlock));
                }
                reservation->input = R::MetadataBuffer<EquationBlockArrayView<const real>>(inputRows);
                reservation->fluxInput = R::MetadataBuffer<EquationBlockArrayView<const real>>(fluxInputRows);
                reservation->output = R::MetadataBuffer<EquationBlockArrayView<real>>(outputRows);
                reservation->fluxOutput = R::MetadataBuffer<EquationBlockArrayView<real>>(fluxOutputRows);
                if (state.limitedVRHistory)
                    reservation->historyDestination.emplace(state.limitedVRHistory->Write(context)); });
        std::optional<ReconstructionRestore> restore;
        stage([&]
              { restore.emplace(ReconstructionRestore([reservation]()
                                                      {
                  reservation->started = true;
                  reservation->execution->Run([reservation](R::ExecutionContext context)
                  {
                      if (context.device == R::Device::Host())
                      {
                          for (index iBlock = 0; iBlock < index(reservation->input.size()); ++iBlock)
                          {
                              const auto input = reservation->input[iBlock];
                              const auto output = reservation->output[iBlock];
                              CPU::CopyRows(input, output, output.ComponentNumber());
                              if (!reservation->fluxOutput.empty())
                                  CPU::CopyRows(reservation->fluxInput[iBlock], reservation->fluxOutput[iBlock],
                                                reservation->fluxOutput[iBlock].ComponentNumber());
                          }
                      }
                      else
                      {
                          // Combined device fields preserve all modal/variable
                          // columns; logical blocks share this one parent.
                          Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Copy,
                              reservation->output[0].parent.DynamicView(), reservation->input[0].parent.DynamicView());
                          if (!reservation->fluxOutput.empty())
                              Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Copy,
                                  reservation->fluxOutput[0].parent.DynamicView(), reservation->fluxInput[0].parent.DynamicView());
                      }
                      if (reservation->historyDestination)
                      {
                          const auto output = reservation->historyDestination->DynamicView();
                          if (reservation->historySource)
                              Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Copy, output, reservation->historySource->DynamicView());
                          else
                              Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Fill, output, {}, {}, 0);
                      }
                  });
                  const auto complete = reservation->execution->Completed();
                  reservation->source->Complete(complete);
                  reservation->destination->Complete(complete);
                  if (reservation->fluxSource) reservation->fluxSource->Complete(complete);
                  if (reservation->fluxDestination) reservation->fluxDestination->Complete(complete);
                  if (reservation->historySource) reservation->historySource->Complete(complete);
                  if (reservation->historyDestination) reservation->historyDestination->Complete(complete); })); });
        return std::move(*restore);
    }
    void FiniteVolume::ResetReconstruction(ReconstructionState &state, Workspace &work, std::optional<R::ExecutionStream> stream) const
    {
        RestoreExecution execution(stream, Device());
        const auto context = execution.context;
        auto stateSession = state.session.Acquire();
        auto session = work.session.Acquire();
        DNDS_check_throw_info(state.owner == _data && work.owner == _data, "foreign equation reconstruction state or workspace");
        if (context.device != R::Device::Host() && _data->reconstruction)
            (void)state.WarmValues().CoupledField();
        // Reserve every destination before changing any value. In particular,
        // a retained LVR history reader must not allow coefficients to be
        // cleared first and only then report the conflict.
        std::optional<EquationDofs::WriteScope> coefficients, fluxCoefficients;
        std::vector<R::Array<real>::WriteLease> temporary;
        std::optional<R::Array<real>::WriteLease> previous, alpha;
        try
        {
            if (_data->reconstruction)
                coefficients.emplace(state.WarmValues().WriteOwned(context));
            if (state.distinctFinalized)
                fluxCoefficients.emplace(state.MutableFinalized().WriteOwned(context));
            temporary.reserve(work.fields.size());
            for (auto &field : work.fields)
                if (field.reconstruction)
                    temporary.emplace_back(field.reconstruction->temporary.Owned().Write(context));
            if (work.limitedVR)
            {
                previous.emplace(state.limitedVRHistory->Write(context));
                alpha.emplace(work.limitedVR->alpha.Write(context));
            }
        }
        catch (...)
        {
            if (coefficients)
                coefficients->CancelUnmodified();
            if (fluxCoefficients)
                fluxCoefficients->CancelUnmodified();
            for (auto &write : temporary)
                write.CancelUnmodified();
            if (previous)
                previous->CancelUnmodified();
            if (alpha)
                alpha->CancelUnmodified();
            throw;
        }
        execution.Run([&](R::ExecutionContext where)
                      {
            const auto fillState = [&](auto &scope)
            {
                if (!scope) return;
                if (where.device == R::Device::Host())
                {
                    const auto layout = scope->ComponentLayout();
                    const auto values = layout.View();
                    CPU::FillRows(values, values.ComponentNumber(), 0);
                }
                else
                    Algebra::ApplyRowViews(where, Algebra::PointwiseOperation::Fill, scope->Block(0).parent.DynamicView(), {}, {}, 0);
            };
            fillState(coefficients); fillState(fluxCoefficients);
            for (auto &write : temporary)
                Algebra::ApplyRowViews(where, Algebra::PointwiseOperation::Fill, write.DynamicView(), {}, {}, 0);
            if (previous) Algebra::ApplyRowViews(where, Algebra::PointwiseOperation::Fill, previous->DynamicView(), {}, {}, 0);
            if (alpha) Algebra::ApplyRowViews(where, Algebra::PointwiseOperation::Fill, alpha->DynamicView(), {}, {}, 0); });
        const auto completed = execution.Completed();
        if (coefficients)
            coefficients->Complete(completed);
        if (fluxCoefficients)
            fluxCoefficients->Complete(completed);
        for (auto &write : temporary)
            write.Complete(completed);
        if (previous)
            previous->Complete(completed);
        if (alpha)
            alpha->Complete(completed);
        work.rhsState.reset();
        work.rhsCoefficients.reset();
        state.finalizedFrom.reset();
        state.finalizedPublication.reset();
    }
}
