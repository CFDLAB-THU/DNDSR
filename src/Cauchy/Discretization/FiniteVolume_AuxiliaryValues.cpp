#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Discretization/FiniteVolumeAuxiliaryInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using AuxiliaryDetail::EquationAuxiliaryArrays;
    MeanGradients FiniteVolume::BuildMeanGradients(EquationDofs &state) const
    {
        MeanGradients result;
        const MPIInfo mpi(Metrics().Communicator());
        Metrics().Communication()->LocalStage([&]
                                              {
            state.RequireSchema(_data->equation.Schema());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && state.Device() == Device(), "mean gradient state space/backend differs");
            result._data = std::make_shared<MeanGradients::Data>(_data, state);
            auto &work = *result._data;
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                const auto &field = state.PhysicalField(iField);
                result.fields.push_back(Metrics().BuildField(FiniteVolumeDetail::FieldLayout(field.Owned().Layout(), Metrics().NumCell(), field.Width() * Metrics().Dimension()), Device()));
                work.boundary.push_back(_data->gradient.BuildBoundaryValues(field.Width()));
                work.vectors.emplace_back();
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = state.LocateBlock(iBlock);
                    if (location.physicalField == iField)
                        for (auto first : state.Schema().blocks[iBlock].vectorStarts)
                            work.vectors.back().push_back(location.firstComponent + first);
                }
                work.execution.push_back(_data->gradient.BuildWorkspace(field.Width(), work.vectors.back()));
            }
            work.fields = result.fields;
            if (Device().space == R::MemorySpace::CUDA)
            {
#ifdef DNDS_USE_CUDA
                const index rows = _data->boundaryFaces.RowCount();
                work.deviceBoundaryStatus.emplace(MeanGradients::Data::DeviceBoundaryStatus{
                    R::Array<NumericalStatus>(R::RowLayout::Dense(rows, 1), Device()),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(rows), 1), Device()),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), Device()),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(0, 1), Device())});
                work.deviceBoundaryStatus->rows.SetName("Cauchy/meanGradientBoundaryStatus", DNDS_ALLOCATION_SITE());
                work.deviceBoundaryStatus->partial.SetName("Cauchy/meanGradientBoundaryStatusPartials", DNDS_ALLOCATION_SITE());
                work.deviceBoundaryStatus->result.SetName("Cauchy/meanGradientBoundaryStatusResult", DNDS_ALLOCATION_SITE());
#else
                DNDS_check_throw_info(false, "CUDA mean gradients are not built");
#endif
            } });
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            result._data->meanHalo.push_back(BindHalo(state.PhysicalField(iField)));
            result._data->gradientHalo.push_back(BindHalo(result.fields[iField]));
        }
        return result;
    }
    void FiniteVolume::PrepareAuxValues(EquationDofs &state, AuxiliaryValues &output, MeanGradients &gradients,
                                        Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time,
                                        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                        AuxiliaryInputs inputs) const
    {
        PrepareAuxValues(state, output, gradients, engine, stream, stream,
                         time, std::move(workers), std::move(inputs));
    }
    void FiniteVolume::PrepareAuxValues(EquationDofs &state, AuxiliaryValues &output, MeanGradients &gradients,
                                        Runtime::EagerEngine &engine,
                                        const Runtime::ExecutionStream &stream,
                                        const Runtime::ExecutionStream &communication,
                                        real time,
                                        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                        AuxiliaryInputs inputs) const
    {
        std::optional<AuxiliaryPreparation> preparation;
        OperatorSchedule schedule(engine, stream, communication, _data->settings.execution.overlap);
        preparation.emplace(QueueMeanGradientAuxValues(state, output, gradients, engine, stream, communication,
                                                       time, std::move(workers), std::move(inputs), true));
        Metrics().Communication()->LocalStage([&]
                                              {
            schedule.Join();
            FinishMeanGradientAuxValues(state, output, gradients, *preparation); });
    }
    FiniteVolume::AuxiliaryPreparation FiniteVolume::QueueMeanGradientAuxValues(
        EquationDofs &state, AuxiliaryValues &output, MeanGradients &gradients,
        R::EagerEngine &engine, const R::ExecutionStream &stream, const R::ExecutionStream &communication,
        real time, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
        AuxiliaryInputs inputs, bool exchangeMeans) const
    {
        using namespace EquationFVDetail;
        const MPIInfo mpi(Metrics().Communicator());
        const bool overlap = _data->settings.execution.overlap;
        const auto stage = [&](auto &&action)
        {
            Metrics().Communication()->LocalStage(std::forward<decltype(action)>(action));
        };
        std::optional<BoundaryInputs> boundary;
        OptionalRef<const R::Array<real>> incoming;
        // This entry preflight follows the same optional validation policy as
        // the later rank-local stages. Halo launches keep their own contract;
        // unchecked failures must reach the application's fail-fast boundary.
        stage([&]
              {
            DNDS_check_throw_info(gradients._data && gradients._data->owner.lock() == _data &&
                                      engine.Owns(stream) && engine.Owns(communication) &&
                                      stream.Context().device == Device() && communication.Context().device == Device(),
                                  "foreign mean gradient workspace or execution streams");
            gradients._data->binding.Check(state);
            state.RequireSchema(_data->equation.Schema());
            DNDS_check_throw_info(gradients.fields.size() == gradients._data->fields.size(), "mean gradient fields replaced");
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                const auto &a = gradients.fields[iField], &b = gradients._data->fields[iField];
                Metrics().ValidateField(a, state.PhysicalField(iField).Width() * Metrics().Dimension());
                DNDS_check_throw_info(a.Owned().SharedStorage() == b.Owned().SharedStorage() && a.Ghost().SharedStorage() == b.Ghost().SharedStorage(), "mean gradient binding changed");
            }
            DNDS_check_throw_info(!HasInterfaces() || inputs.boundary,
                                  "mean-gradient preparation needs an incoming interface packet");
            if (inputs.boundary)
            {
                boundary = *inputs.boundary;
                if (inputs.deferBoundary) ValidateBoundaryLayout(*boundary);
                else ValidateBoundaryInputs(*boundary, time);
                incoming.Emplace(boundary->FaceCenterValues().CoupledField().Owned());
                inputs.boundary.Emplace(*boundary);
            } });
        auto &work = *gradients._data;
        work.origin.reset();
        work.ghostsReady = false;
        const auto gradient = [&](CFV::ReconstructionRegion region)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                _data->gradient.Evaluate(engine, stream, state.PhysicalField(iField), work.boundary[iField],
                                         gradients.fields[iField], work.execution[iField], region);
        };
        std::optional<AuxiliaryPreparation> preparation;
        // These closures and the retained preparation must outlive exceptional
        // schedule retirement as well as ordinary worker completion.
        OperatorSchedule schedule(engine, stream, communication, overlap);
        try
        {
            std::vector<R::TransferOperation> means;
            if (exchangeMeans)
                means = schedule.Begin(work.meanHalo);
            SubmitMeanGradientBoundary(engine, stream, state, gradients, time, workers, incoming);
            if (overlap)
                gradient(CFV::ReconstructionRegion::Core);
            schedule.Await(means);
            gradient(schedule.DependentRegion());
            work.RecordQueued(state, time, false);
            DNDS_check_throw_info(!inputs.meanGradients, "mean gradient input is owned by this preparation call");
            inputs.meanGradients.Emplace(gradients);
            const bool pending = inputs.deferBoundary;
            if (pending)
                inputs.boundary = {};
            preparation.emplace(BeginAuxValues(state, output, time, workers, inputs));
            if (pending)
                preparation->BindBoundary(*boundary);
            auto halos = schedule.BeginAfterCompute(work.gradientHalo);
            work.RecordQueued(state, time, true);
            if (overlap)
                preparation->Apply(engine, stream, CFV::ReconstructionRegion::Core);
            schedule.Await(halos);
            preparation->Apply(engine, stream, schedule.DependentRegion());
            return std::move(*preparation);
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy equation auxiliary preparation failed");
            try
            {
                schedule.Join();
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy equation auxiliary preparation failed");
            }
            work.origin.reset();
            work.ghostsReady = false;
            work.queuedSource.reset();
            work.queuedOwned.clear();
            work.queuedGhost.clear();
            throw;
        }
    }
    void FiniteVolume::FinishMeanGradientAuxValues(const EquationDofs &state, AuxiliaryValues &output,
                                                   MeanGradients &gradients, AuxiliaryPreparation &preparation) const
    {
        DNDS_check_throw_info(gradients._data && gradients._data->owner.lock() == _data,
                              "foreign mean-gradient publication");
        auto &work = *gradients._data;
        if (work.deviceBoundaryStatus)
        {
            const auto status = work.deviceBoundaryStatus->result.To(R::Device::Host());
            RequireNumericalSuccess(status.Read().View()(0, 0), "mean-gradient boundary");
        }
        work.PublishQueued(state);
        preparation.Finish(output);
    }
    real FiniteVolume::StableStep(const EquationDofs &state, const AuxiliaryValues &values, real cfl, real time) const
    {
        real dt = std::numeric_limits<real>::max();
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            DNDS_check_throw_info(std::isfinite(cfl) && cfl > 0, "positive CFL required");
            ValidateAuxValues(state, values, time);
            DNDS_check_throw_info(values.cellSpectralSum, "CFL selection requires prepared cell spectral sums");
            const auto geometry = Metrics().AcquireRead(); const auto metric = geometry.HostView();
            const auto read = values.cellSpectralSum->Owned().Read(); const auto sums = read.View<R::UniformTableRows>();
            dt = RequireNumericalSuccess(CPU::MinimumStableStep(sums, metric, cfl), "MinimumStableStep"); });
        dt = MPI::AllreduceValue<MPI::Reduction::Min>(dt, Metrics().Communicator());
        return dt;
    }
    FiniteVolume::AuxiliaryPreparation FiniteVolume::BeginAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time,
                                                                    std::optional<Physics::Chemistry::ChemicalWorkerPool> workers, AuxiliaryInputs inputs) const
    {
        Metrics().Validate();
        _data->partition.Validate();
        state.RequireSchema(_data->equation.Schema());
        if (state.Device() == R::Device::Host())
            _data->equation.ResolveKernelProfile(state.Device());
        else
        {
            if (!_data->equation.IsACM())
                PerfectGasSpatialParameters::FromEquation(_data->equation, Metrics().Dimension()).Validate(int(state.ComponentNumber()));
            const auto &field = state.CoupledField();
            DNDS_check_throw_info(field.Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                      field.Ghost().Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA auxiliary preparation requires combined AoS fields");
            DNDS_check_throw_info(!output.derivedOwned && !output.derivedGhost,
                                  "queued CUDA derived-field publication is not connected yet");
        }
        DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && state.Device() == Device() && std::isfinite(time), "equation auxiliary row space/device/time differs");
        DNDS_check_throw_info(_data->equation.HasMixture() == bool(workers), "equation auxiliary needs exactly its matching worker backend");
        DNDS_check_throw_info(!HasInterfaces() || inputs.boundary || inputs.deferBoundary,
                              "equation auxiliary preparation needs an incoming interface packet");
        DNDS_check_throw_info(!inputs.deferBoundary || !inputs.boundary,
                              "deferred auxiliary boundary already supplies a packet");
        if (inputs.boundary)
            ValidateBoundaryInputs(*inputs.boundary, time);
        _data->boundary.ValidateEquationProducers(_data->boundaryZones);
        const auto arrays = EquationAuxiliaryArrays(output);
        const index faces = _data->stencil.stencilFaces.RowCount();
        const index rows[9]{Metrics().NumCellProc(), faces, faces, faces, Metrics().NumCell(), Metrics().NumCell(), Metrics().NumCell(), Metrics().NumCell(), Metrics().NumCellProc() - Metrics().NumCell()};
        const index widths[7]{_data->equation.HasMixture() ? 5 : 3, 2, 1, 1, 1, 1, 1};
        for (int k = 0; k < 9; ++k)
            if (arrays[k])
            {
                DNDS_check_throw_info(arrays[k]->GetDevice() == Device() && arrays[k]->RowCount() == rows[k] && !arrays[k]->Layout().IsVariable() && (k >= 7 || arrays[k]->Layout().Width() == widths[k]), "equation auxiliary output shape/device differs");
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    for (const auto *input : {&state.PhysicalField(iField).Owned(), &state.PhysicalField(iField).Ghost()})
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), arrays[k]->Storage()), "equation auxiliary output aliases state");
                if (inputs.boundary)
                    for (const auto *packet : {
                             &inputs.boundary->QuadratureValues(),
                             &inputs.boundary->FaceCenterValues()})
                        for (index iField = 0;
                             iField < packet->PhysicalFieldNumber(); ++iField)
                            DNDS_check_throw_info(
                                !Algebra::Detail::Overlaps(
                                    packet->PhysicalField(iField).Owned().Storage(),
                                    arrays[k]->Storage()),
                                "equation auxiliary output aliases interface input");
                if (inputs.meanGradients)
                    for (const auto &field : inputs.meanGradients->fields)
                        for (const auto *input : {&field.Owned(), &field.Ghost()})
                            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), arrays[k]->Storage()), "equation auxiliary output aliases mean gradients");
                for (int j = 0; j < k; ++j)
                    if (arrays[j])
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(arrays[k]->Storage(), arrays[j]->Storage()), "equation auxiliary outputs overlap");
            }
        for (const auto *field : {&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum})
            if (*field)
                Metrics().ValidateField(**field, 1);
        for (int part = 0; part < 2; ++part)
        {
            const auto &derived = part ? output.derivedGhost : output.derivedOwned;
            if (derived)
            {
                DNDS_check_throw_info(derived->GhostPart() == bool(part), "equation derived row part differs");
                (void)derived->Current(state, _data->equation);
            }
        }
        if (output._derivedThermodynamics)
        {
            DNDS_check_throw_info(output.derivedOwned && output.derivedGhost, "derived thermodynamics needs both row parts");
            for (const auto *field : {&*output.derivedOwned, &*output.derivedGhost})
            {
                for (auto q : {DerivedQuantity::Pressure, DerivedQuantity::SoundSpeed, DerivedQuantity::DiffusionBound})
                    DNDS_check_throw_info(field->Request().Has(q), "derived thermodynamic column missing");
                if (_data->equation.HasMixture())
                    for (auto q : {DerivedQuantity::TotalEnthalpy, DerivedQuantity::PressureInternalEnergyDerivative})
                        DNDS_check_throw_info(field->Request().Has(q), "mixture characteristic thermodynamic column missing");
            }
        }
        const bool needScratch = (arrays[4] && !arrays[1]) || (arrays[5] && !arrays[2]) || (arrays[6] && !arrays[3]);
        if (needScratch && (!output._faceScratch || output._faceScratch->RowCount() != faces))
            output._faceScratch.emplace(R::RowLayout::Dense(faces, 3), Device());
        output._equationOrigin.reset();
        if (Device() != R::Device::Host())
            return AuxiliaryPreparation::BuildQueuedState(*this, state, output, time, inputs);
        return AuxiliaryPreparation::BuildState(
            *this, state, output, time, std::move(workers), inputs);
    }
    void FiniteVolume::PrepareAuxValues(const EquationDofs &state, AuxiliaryValues &output, real time,
                                        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers, AuxiliaryInputs inputs) const
    {
        auto scope = BeginAuxValues(state, output, time, std::move(workers), inputs);
        if (Device() == R::Device::Host())
            scope.Apply();
        else
        {
#ifdef DNDS_USE_CUDA
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            scope.Apply(engine, stream);
            engine.Synchronize();
#else
            DNDS_check_throw_info(false, "CUDA auxiliary preparation is not built");
#endif
        }
        scope.Finish(output);
    }
    void FiniteVolume::ValidateAuxValues(const EquationDofs &state, const AuxiliaryValues &values, real time) const
    {
        Metrics().Validate();
        state.RequireSchema(_data->equation.Schema());
        DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && values._equationOrigin,
                              "equation auxiliary publication is missing or belongs to another path");
        const R::ExecutionContext context{state.Device(), nullptr};
        const auto read = state.ReadWithGhost(context);
        const auto &origin = *values._equationOrigin;
        DNDS_check_throw_info(origin.owner.lock().get() == _data.get() && origin.time == time, "equation auxiliary plan/time is stale");
        if (origin.boundary)
            ValidateBoundaryInputs(*origin.boundary, time);
        else
            DNDS_check_throw_info(!HasInterfaces(), "equation auxiliary interface publication is missing");
        origin.source.Check(state, true);
        for (index k = 0; k < index(origin.gradientInputs.size()); ++k)
        {
            const auto read = origin.gradientInputs[k].Read(context);
            DNDS_check_throw_info(origin.gradientInputs[k].Version() == origin.gradientVersions[k][0] &&
                                      origin.gradientInputs[k].ShapeVersion() == origin.gradientVersions[k][1],
                                  "equation auxiliary gradient input changed");
        }
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(origin.ghostVersions[iField] == state.PhysicalField(iField).Ghost().Version(), "equation auxiliary ghosts changed");
        const auto arrays = EquationAuxiliaryArrays(values);
        for (int k = 0; k < 9; ++k)
        {
            const auto &stamp = origin.outputs[k];
            DNDS_check_throw_info(bool(stamp.storage) == bool(arrays[k]), "equation auxiliary selection changed");
            if (arrays[k])
            {
                const auto pin = arrays[k]->Read(context);
                DNDS_check_throw_info(stamp.storage == arrays[k]->SharedStorage() && stamp.version == arrays[k]->Version() && stamp.shape == arrays[k]->ShapeVersion(), "equation auxiliary output changed");
            }
        }
        int k = 0;
        for (const auto *field : {&values.cellSpectralSum, &values.cellWaveJump, &values.cellPressureMinimum})
        {
            DNDS_check_throw_info(!*field || (*field)->Space().Same(state.Space()), "equation auxiliary cell space changed");
            DNDS_check_throw_info(origin.ghosts[k++] == (*field ? (*field)->Ghost().SharedStorage() : nullptr), "equation auxiliary cell ghost rebound");
        }
        for (const auto *field : {&values.derivedOwned, &values.derivedGhost})
            if (*field)
            {
                const auto pin = (*field)->ReadCurrent(state, _data->equation);
            }
    }
    std::optional<BoundaryInputs> FiniteVolume::PreparedBoundaryInputs(const AuxiliaryValues &values) const
    {
        if (values._equationOrigin)
            return values._equationOrigin->boundary;
        return std::nullopt;
    }
    bool FiniteVolume::AuxiliaryBoundaryCurrent(
        const AuxiliaryValues &values, const Workspace &workspace) const
    {
        const auto prepared = PreparedBoundaryInputs(values);
        const auto bound = PreparedBoundaryInputs(workspace);
        return prepared ? bound && prepared->SamePublication(*bound) : !bound;
    }
}
