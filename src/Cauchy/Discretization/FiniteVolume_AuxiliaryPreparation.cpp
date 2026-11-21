#include "Cauchy/Discretization/BoundaryFluxInput.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeAuxiliaryInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeAuxiliaryKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Core/OMP.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#include <atomic>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using AuxiliaryDetail::EquationAuxiliaryArrays;

    struct FiniteVolume::AuxiliaryPreparation::State : std::enable_shared_from_this<State>
    {
        enum class Mode
        {
            Unselected,
            Immediate,
            Queued
        };
        using Part = AuxiliaryRegion;
        FiniteVolume owner;
        std::shared_ptr<const Data> data;
        EquationDofs source;
        AuxiliaryValues buffers;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        CFV::FiniteVolumeRead geometry;
        FiniteVolumeStencilStorage::ReadAccess stencilRead;
        std::optional<EquationDofs::ReadScope> owned;
        std::optional<EquationDofs::ReadScope> complete;
        std::optional<MeanGradients> gradients;
        std::optional<BoundaryInputs> boundary;
        std::optional<R::Array<real>::ReadLease> boundaryRead;
        RHSScalarView incoming;
        std::vector<R::Array<real>::ReadLease> gradientOwned, gradientGhost;
        R::MetadataBuffer<RHSFieldView> gradientViews;
        R::MetadataBuffer<EquationBlockLocation> locations;
        std::optional<WallDistanceRead> walls;
        WallDistanceView wallValues;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        std::array<std::optional<R::Array<real>::WriteLease>, 8> writes;
        std::array<RHSOutputView, 8> values;
        std::optional<R::Array<real>::WriteLease> scratchWrite;
        RHSOutputView scratch;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int independent;
        R::MetadataBuffer<index> vectorStarts;
        real time;
        RigidFrame frame;
        int phase = 0;
        bool started = false;
        bool boundaryPending = false, independentDone = false;
        bool useScratch = false;
        Mode mode = Mode::Unselected;
        std::atomic<int> completed{0};
        int issued = 0;
        struct Stamp
        {
            R::Array<real> array;
            uint64_t shape, epoch;
            explicit Stamp(const R::Array<real> &value)
                : array(value), shape(value.ShapeVersion()), epoch(value.MutationEpoch()) {}
            void Check() const
            {
                DNDS_check_throw_info(array.IsValid() && array.ShapeVersion() == shape && array.MutationEpoch() == epoch,
                                      "queued auxiliary operand changed after submission");
            }
        };
        std::vector<Stamp> inputStamps, outputStamps;

        State(const FiniteVolume &plan, const EquationDofs &state, const AuxiliaryValues &output, real atTime,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, AuxiliaryInputs inputs)
            : owner(plan), data(plan._data), source(state.NumericAlias()), buffers(output), workers(std::move(pool)),
              geometry(data->metrics.AcquireRead()), stencilRead(data->stencil.Read()),
              independent(data->equation.HasMixture() ? data->equation.Mixture().SpeciesNumber() - 1 : 0),
              time(atTime), frame(data->motion.At(time)), boundaryPending(inputs.deferBoundary)
        {
            if (inputs.meanGradients)
            {
                gradients = *inputs.meanGradients;
                DNDS_check_throw_info(gradients->_data, "mean gradient input is uninitialized");
            }
            if (inputs.boundary)
            {
                boundary = *inputs.boundary;
            }
            if (data->equation.HasRANS() && data->equation.RANS().ComponentNumber() == 2)
            {
                DNDS_check_throw_info(gradients, "two-equation auxiliary transport requires explicit mean gradients");
                walls.emplace(data->wallDistance->AcquireRead());
                wallValues = walls->HostView();
            }
            index first = 0;
            locations = R::MetadataBuffer<EquationBlockLocation>::Generate(source.BlockNumber(),
                                                                           [&](index iBlock)
                                                                           { return source.LocateBlock(iBlock); });
            std::vector<index> vectors;
            for (const auto &block : source.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                if (block.name == "rans")
                    ransFirst = first;
                for (const auto vector : block.vectorStarts)
                    vectors.push_back(first + vector);
                first += index(block.components.size());
            }
            vectorStarts = R::MetadataBuffer<index>(vectors);
            const auto arrays = EquationAuxiliaryArrays(buffers);
            useScratch = (arrays[4] && !arrays[1]) || (arrays[5] && !arrays[2]) || (arrays[6] && !arrays[3]);
            // Begin retains handles and immutable geometry only. The first
            // Apply selects immediate or queued access; a queued producer may
            // already be pending here, and chemistry workers must be acquired
            // in execution order rather than exhausted on the controller.
        }
        void AcquireImmediate()
        {
            DNDS_check_throw_info(mode != Mode::Queued, "cannot mix synchronous and queued auxiliary execution");
            if (mode == Mode::Immediate)
                return;
            const auto arrays = EquationAuxiliaryArrays(buffers);
            try
            {
                owned.emplace(source.ReadOwned());
                if (gradients)
                {
                    gradients->_data->Check(source, gradients->fields, data, time, false);
                    for (const auto &field : gradients->fields)
                        gradientOwned.push_back(field.Owned().Read());
                    gradientViews = R::MetadataBuffer<RHSFieldView>::Generate(gradientOwned.size(), [&](index iField)
                                                                              { return RHSFieldView{gradientOwned[iField].View<R::UniformTableRows>(), {}}; });
                }
                if (boundary)
                {
                    owner.ValidateBoundaryInputs(*boundary, time);
                    boundaryRead.emplace(FiniteVolumeDetail::BoundaryFluxInput(data->stencil, *boundary).Read());
                    incoming = boundaryRead->View<R::UniformTableRows>();
                }
                for (const auto *derived : {&buffers.derivedOwned, &buffers.derivedGhost})
                    if (*derived)
                        (*derived)->PreflightRefresh();
                for (int k = 0; k < 8; ++k)
                    if (arrays[k])
                    {
                        auto handle = *arrays[k];
                        writes[k].emplace(handle.Write());
                        values[k] = writes[k]->View<R::UniformTableRows>();
                    }
                if (useScratch)
                {
                    scratchWrite.emplace(buffers._faceScratch->Write());
                    scratch = scratchWrite->View<R::UniformTableRows>();
                }
                mode = Mode::Immediate;
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy equation auxiliary preparation failed");
                Cancel();
                for (auto &write : writes)
                    write.reset();
                scratchWrite.reset();
                owned.reset();
                gradientOwned.clear();
                boundaryRead.reset();
                throw;
            }
        }
        void Cancel()
        {
            for (auto &write : writes)
                if (write)
                    write->CancelUnmodified();
            if (scratchWrite)
                scratchWrite->CancelUnmodified();
        }
        ~State()
        {
            if (!started)
                Cancel();
            else
            {
                for (auto &write : writes)
                    if (write)
                        write->Fail();
                if (scratchWrite)
                    scratchWrite->Fail();
            }
        }
        PointEvaluator Evaluator() const
        {
            return data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {true, true, false});
        }
        std::vector<CPU::AuxiliaryWorkerScratch> MakeKernelWorkers() const
        {
            // Reserve private point contexts before numerical work. Mixture
            // team size is bounded atomically by the available pool capacity.
            return MakeCPUPointWorkers<CPU::AuxiliaryWorkerScratch>(data->equation, workers,
                                                                    {true, true, false}, source.ComponentNumber());
        }
        void RefreshDerived(bool ghost)
        {
            auto &field = ghost ? buffers.derivedGhost : buffers.derivedOwned;
            if (!field)
                return;
            if (workers)
                RefreshDerivedFields(source, data->equation, *field, workers->Acquire(), true);
            else
                RefreshDerivedFields(source, data->equation, *field, true);
            (ghost ? derivedGhost : derivedOwned).emplace(field->ReadCurrent(source, data->equation));
        }
        CPU::AuxiliaryKernelRequest KernelInput(
            SpanView<const EquationBlockReadView> blocks, SpanView<const RHSFieldView> gradient,
            const std::array<RHSOutputView, 8> &outputs, RHSOutputView privateFaces,
            std::optional<MeanThermodynamicView> derived, RHSScalarView packet) const
        {
            CPU::AuxiliaryKernelRequest request;
            request.stencil = stencilRead.View();
            request.metric = geometry.HostView();
            request.blocks = blocks;
            request.gradients = gradient;
            request.locations = locations.HostView();
            request.vectorStarts = vectorStarts.HostView();
            request.conditions = {data->boundaryConditions.data(), index(data->boundaryConditions.size())};
            request.wall = wallValues;
            request.hasWall = bool(walls);
            request.derived = derived;
            const auto arrays = EquationAuxiliaryArrays(buffers);
            for (int k = 0; k < 8; ++k)
                request.enabled[k] = bool(arrays[k]);
            request.values = outputs;
            request.scratch = privateFaces;
            request.useScratch = useScratch;
            request.incoming = packet;
            request.frame = frame;
            request.components = source.ComponentNumber();
            request.flowFirst = flowFirst;
            request.speciesFirst = speciesFirst;
            request.ransFirst = ransFirst;
            request.dimension = data->metrics.Dimension();
            request.independentSpecies = independent;
            request.time = time;
            return request;
        }
        CPU::AuxiliaryKernelRequest ImmediateInput(SpanView<const EquationBlockReadView> blocks) const
        {
            std::optional<MeanThermodynamicView> derived;
            if (buffers._derivedThermodynamics)
                derived.emplace(MeanThermodynamicView::Derived(*derivedOwned,
                                                               derivedGhost ? OptionalRef<const DerivedFields::ReadScope>(*derivedGhost) : OptionalRef<const DerivedFields::ReadScope>{}));
            return KernelInput(blocks, gradientViews.HostView(), values, scratch, derived, incoming);
        }
        void Run(const CPU::AuxiliaryKernelRequest &request, Part part) const
        {
            auto kernelWorkers = MakeKernelWorkers();
            const SpanView<CPU::AuxiliaryWorkerScratch> evaluator{kernelWorkers.data(), index(kernelWorkers.size())};
            if (part != Part::InterfaceCore)
                RequireNumericalSuccess(CPU::PrepareAuxiliaryMeans(request, data->equation, evaluator,
                                                                   part == Part::Dependent ? data->metrics.NumCell() : 0,
                                                                   part == Part::Dependent ? data->metrics.NumCellProc() : data->metrics.NumCell()),
                                        "auxiliary means");
            const auto faces = part == Part::Independent ? request.stencil.auxiliaryInterfaceIndependentFaces : part == Part::InterfaceCore ? request.stencil.auxiliaryInterfaceFaces
                                                                                                            : part == Part::Dependent       ? request.stencil.auxiliaryDependentFaces
                                                                                                                                            : request.stencil.auxiliaryCoreFaces;
            const auto cells = part == Part::Independent ? request.stencil.interfaceIndependentCells : part == Part::InterfaceCore ? request.stencil.interfaceCoreCells
                                                                                                   : part == Part::Dependent       ? request.stencil.dependentCells
                                                                                                                                   : request.stencil.coreCells;
            const auto &selection = data->stencil.auxiliaryRegions[size_t(part)];
            RequireNumericalSuccess(CPU::PrepareAuxiliaryFaces(request, data->equation, data->boundaryPoint, evaluator, {selection.faces, faces}), "auxiliary faces");
            CPU::ReduceAuxiliaryCells(request, {selection.cells, cells});
        }
        void InterfaceIndependent()
        {
            DNDS_check_throw_info(phase == 0 && boundaryPending && !independentDone,
                                  "auxiliary independent phase requires a pending boundary");
            AcquireImmediate();
            started = true;
            RefreshDerived(false);
            const auto blocks = EquationFVDetail::BlockViews(source, *owned);
            Run(ImmediateInput(blocks.HostView()), Part::Independent);
            independentDone = true;
        }
        void BindBoundary(const BoundaryInputs &input)
        {
            DNDS_check_throw_info(phase == 0 && boundaryPending,
                                  "auxiliary boundary is not pending");
            if (mode == Mode::Immediate)
                owner.ValidateBoundaryInputs(input, time);
            else
                owner.ValidateBoundaryLayout(input);
            const auto arrays = EquationAuxiliaryArrays(buffers);
            for (const auto *array : arrays)
                if (array)
                    FiniteVolumeDetail::RequireDistinctBoundaryOutput(*array, input);
            boundary = input;
            if (mode == Mode::Immediate)
            {
                boundaryRead.emplace(FiniteVolumeDetail::BoundaryFluxInput(data->stencil, *boundary).Read());
                incoming = boundaryRead->View<R::UniformTableRows>();
            }
            boundaryPending = false;
        }
        void Core()
        {
            DNDS_check_throw_info(phase == 0 && !boundaryPending,
                                  "auxiliary core already executed or interface is pending");
            AcquireImmediate();
            started = true;
            if (!independentDone)
                RefreshDerived(false);
            const auto blocks = EquationFVDetail::BlockViews(source, *owned);
            Run(ImmediateInput(blocks.HostView()), independentDone ? Part::InterfaceCore : Part::Core);
            phase = 1;
        }
        void Dependent()
        {
            DNDS_check_throw_info(phase == 1, "equation auxiliary dependent phase requires core");
            DNDS_check_throw_info(mode == Mode::Immediate, "cannot mix synchronous and queued auxiliary execution");
            complete.emplace(source.ReadWithGhost()); // An unfinished halo is retriable before any dependent write.
            if (gradients)
            {
                gradients->_data->Check(source, gradients->fields, data, time, true);
                for (const auto &field : gradients->fields)
                    gradientGhost.push_back(field.Ghost().Read());
                gradientViews = R::MetadataBuffer<RHSFieldView>::Generate(gradientViews.size(), [&](index iField)
                                                                          { return RHSFieldView{gradientViews[iField].owned, gradientGhost[iField].View<R::UniformTableRows>()}; });
            }
            RefreshDerived(true);
            const auto blocks = EquationFVDetail::BlockViews(source, *complete);
            Run(ImmediateInput(blocks.HostView()), Part::Dependent);
            phase = 2;
        }
        void Submit(R::EagerEngine &engine, const R::ExecutionStream &stream, Part part)
        {
            DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host() && mode != Mode::Immediate,
                                  "queued auxiliary requires its CPU stream and cannot mix execution modes");
            const bool ghost = part == Part::Dependent;
            const bool independent = part == Part::Independent;
            DNDS_check_throw_info(ghost ? phase == 1 : phase == 0,
                                  "queued auxiliary phases are out of order");
            DNDS_check_throw_info(independent ? boundaryPending && !independentDone : !boundaryPending,
                                  "bind the packet before submitting boundary-consuming auxiliary work");
            DNDS_check_throw_info(part != Part::InterfaceCore || independentDone,
                                  "interface-core auxiliary work requires its independent prefix");
            if (gradients)
                gradients->_data->CheckQueued(source, gradients->fields, data, time, ghost);
            mode = Mode::Queued;
            started = true;
            if (part != Part::InterfaceCore)
            {
                auto &derived = ghost ? buffers.derivedGhost : buffers.derivedOwned;
                if (derived)
                    RefreshDerivedFields(engine, stream, source, data->equation, *derived, workers, true);
            }
            using Input = R::ArrayOperand<real, false, R::UniformTableRows>;
            using Output = R::ArrayOperand<real, true, R::UniformTableRows>;
            struct Bindings
            {
                std::vector<std::array<index, 2>> source, gradient;
                std::array<index, 8> output{-1, -1, -1, -1, -1, -1, -1, -1};
                index scratch = -1, packet = -1, derivedOwned = -1, derivedGhost = -1;
                std::optional<BoundaryInputs> boundary;
                std::array<int, 5> derivedColumns{0, 1, 2, -1, -1};
                index derivedWidth = 3;
            };
            auto slots = std::make_shared<Bindings>();
            std::vector<Input> reads;
            std::vector<Output> writes;
            const auto addRead = [&](const R::Array<real> &array)
            {
                inputStamps.emplace_back(array);
                const index slot = index(reads.size());
                reads.push_back(R::ReadAs<R::UniformTableRows>(array));
                return slot;
            };
            const auto addField = [&](const CFV::FVField &field)
            {
                return std::array<index, 2>{addRead(field.Owned()), ghost ? addRead(field.Ghost()) : -1};
            };
            for (index iField = 0; iField < source.PhysicalFieldNumber(); ++iField)
                slots->source.push_back(addField(source.PhysicalField(iField)));
            if (gradients)
                for (const auto &field : gradients->fields)
                    slots->gradient.push_back(addField(field));
            if (!independent && boundary)
            {
                slots->boundary = boundary;
                slots->packet = addRead(FiniteVolumeDetail::BoundaryFluxInput(data->stencil, *boundary));
                // The completed packet includes both point sets. Its metadata
                // validator checks both, so declare both even though this
                // numerical operation uses only the section selected by the stencil.
                const auto &other = data->stencil.quadratureBoundaryValues ? boundary->FaceCenterValues() : boundary->QuadratureValues();
                addRead(other.CoupledField().Owned());
            }
            if (buffers._derivedThermodynamics)
            {
                DNDS_check_throw_info(buffers.derivedOwned && (!ghost || buffers.derivedGhost), "derived auxiliary row part is missing");
                const auto &derived = *buffers.derivedOwned;
                slots->derivedOwned = addRead(derived.Values());
                slots->derivedColumns = {derived.Column(DerivedQuantity::Pressure), derived.Column(DerivedQuantity::SoundSpeed),
                                         derived.Column(DerivedQuantity::DiffusionBound), -1, -1};
                if (derived.Request().Has(DerivedQuantity::TotalEnthalpy) && derived.Request().Has(DerivedQuantity::PressureInternalEnergyDerivative))
                {
                    slots->derivedColumns[3] = derived.Column(DerivedQuantity::TotalEnthalpy);
                    slots->derivedColumns[4] = derived.Column(DerivedQuantity::PressureInternalEnergyDerivative);
                    slots->derivedWidth = 5;
                }
                if (ghost)
                {
                    slots->derivedGhost = addRead(buffers.derivedGhost->Values());
                    for (int q = 0; q < DerivedQuantityNumber; ++q)
                        DNDS_check_throw_info(derived.Request().enabled[q] == buffers.derivedGhost->Request().enabled[q],
                                              "derived auxiliary owned/ghost projections differ");
                }
            }
            const auto arrays = EquationAuxiliaryArrays(buffers);
            for (int k = 0; k < 8; ++k)
                if (arrays[k])
                {
                    slots->output[k] = index(writes.size());
                    auto array = *arrays[k];
                    writes.push_back(R::ReadWriteAs<R::UniformTableRows>(array));
                }
            if (useScratch)
            {
                slots->scratch = index(writes.size());
                writes.push_back(R::ReadWriteAs<R::UniformTableRows>(*buffers._faceScratch));
            }
            const auto self = shared_from_this();
            R::SubmitArrayBatch(engine, stream, [self, slots, part, ghost](R::ExecutionContext, auto reads, auto writes)
                                {
                    const TimelineScope execution("Cauchy/CPU/Auxiliary", int(part));
                    if (slots->boundary) self->owner.ValidateBoundaryPublication(*slots->boundary, self->time);
                    std::vector<EquationBlockReadView> blocks;
                    blocks.reserve(size_t(self->source.BlockNumber()));
                    for (index iBlock = 0; iBlock < self->source.BlockNumber(); ++iBlock)
                    {
                        const auto location = self->source.LocateBlock(iBlock);
                        const auto field = slots->source[location.physicalField];
                        blocks.push_back({{reads[field[0]], location.firstComponent, location.componentNumber},
                                          {field[1] >= 0 ? reads[field[1]] : RHSScalarView{}, location.firstComponent, location.componentNumber}});
                    }
                    std::vector<RHSFieldView> gradients;
                    for (const auto &field : slots->gradient)
                        gradients.push_back({reads[field[0]], field[1] >= 0 ? reads[field[1]] : RHSScalarView{}});
                    std::array<RHSOutputView, 8> outputs{};
                    for (int k = 0; k < 8; ++k)
                        if (slots->output[k] >= 0) outputs[k] = writes[slots->output[k]];
                    std::optional<MeanThermodynamicView> derived;
                    if (slots->derivedOwned >= 0)
                    {
                        DNDS_check_throw_info(self->buffers.derivedOwned->Current(self->source, self->data->equation) &&
                                                  (!ghost || self->buffers.derivedGhost->Current(self->source, self->data->equation)),
                                              "queued auxiliary derived properties are stale or unpublished");
                        derived.emplace(RHSFieldView{reads[slots->derivedOwned],
                                                     slots->derivedGhost >= 0 ? reads[slots->derivedGhost] : RHSScalarView{}},
                                        slots->derivedColumns, slots->derivedWidth);
                    }
                    const auto request = self->KernelInput({blocks.data(), index(blocks.size())},
                        {gradients.data(), index(gradients.size())}, outputs,
                        slots->scratch >= 0 ? writes[slots->scratch] : RHSOutputView{}, derived,
                        slots->packet >= 0 ? reads[slots->packet] : RHSScalarView{});
                    self->Run(request, part);
                    self->completed.fetch_add(1, std::memory_order_release); }, std::move(reads), std::move(writes));
            ++issued;
            // Replace output stamps after each accepted write reservation; the
            // latest region legitimately advances the same allocation's epoch.
            outputStamps.clear();
            for (const auto *array : EquationAuxiliaryArrays(buffers))
                if (array)
                    outputStamps.emplace_back(*array);
            if (useScratch)
                outputStamps.emplace_back(*buffers._faceScratch);
            if (independent)
                independentDone = true;
            else
                phase = ghost ? 2 : 1;
        }
        void CheckQueuedCompletion() const
        {
            DNDS_check_throw_info(issued > 0 && completed.load(std::memory_order_acquire) == issued,
                                  "queued auxiliary kernels have not completed");
            for (const auto &stamp : inputStamps)
                stamp.Check();
            for (const auto &stamp : outputStamps)
            {
                stamp.Check();
                DNDS_check_throw_info(stamp.array.Storage().IsIdle(), "queued auxiliary publication needs a drained segment");
            }
        }
    };
    FiniteVolume::AuxiliaryPreparation
    FiniteVolume::AuxiliaryPreparation::BuildState(
        const FiniteVolume &plan, const EquationDofs &state,
        const AuxiliaryValues &output, real time,
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
        AuxiliaryInputs inputs)
    {
        return AuxiliaryPreparation(std::make_unique<State>(
            plan, state, output, time, std::move(workers), inputs));
    }

    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(std::unique_ptr<State> state) : _state(std::move(state)) {}
    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation &FiniteVolume::AuxiliaryPreparation::operator=(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation::~AuxiliaryPreparation() = default;
    void FiniteVolume::AuxiliaryPreparation::ApplyInterfaceIndependent()
    {
        DNDS_check_throw_info(_state, "equation auxiliary scope is finished or moved");
        _state->InterfaceIndependent();
    }
    void FiniteVolume::AuxiliaryPreparation::BindBoundary(const BoundaryInputs &input)
    {
        if (_queued)
        {
            BindQueuedBoundary(input);
            return;
        }
        DNDS_check_throw_info(_state, "equation auxiliary scope is finished or moved");
        _state->BindBoundary(input);
    }
    void FiniteVolume::AuxiliaryPreparation::Apply(CFV::ReconstructionRegion region)
    {
        DNDS_check_throw_info(_state, "equation auxiliary scope is finished or moved");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core || region == CFV::ReconstructionRegion::HaloDependent, "invalid auxiliary region");
        if (region != CFV::ReconstructionRegion::HaloDependent)
            _state->Core();
        if (region != CFV::ReconstructionRegion::Core)
            _state->Dependent();
    }
    void FiniteVolume::AuxiliaryPreparation::Finish(AuxiliaryValues &output)
    {
        if (_queued)
        {
            FinishQueued(output);
            return;
        }
        DNDS_check_throw_info(_state && _state->phase == 2, "cannot publish incomplete equation auxiliary preparation");
        if (_state->mode == State::Mode::Queued)
            _state->CheckQueuedCompletion();
        const auto a = EquationAuxiliaryArrays(_state->buffers), b = EquationAuxiliaryArrays(output);
        for (int k = 0; k < int(a.size()); ++k)
            DNDS_check_throw_info(bool(a[k]) == bool(b[k]) && (!a[k] || a[k]->SharedStorage() == b[k]->SharedStorage()), "equation auxiliary buffers were replaced");
        int k = 0;
        for (const auto *field : {&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum})
        {
            const std::optional<CFV::FVField> *original[]{&_state->buffers.cellSpectralSum, &_state->buffers.cellWaveJump, &_state->buffers.cellPressureMinimum};
            DNDS_check_throw_info(!*field || ((*field)->Space().Same(_state->source.Space()) && (*field)->Ghost().SharedStorage() == (*original[k])->Ghost().SharedStorage()), "equation auxiliary ghost/space changed");
            ++k;
        }
        for (auto &write : _state->writes)
            write.reset();
        _state->scratchWrite.reset();
        auto origin = std::make_shared<AuxiliaryValues::EquationOrigin>(_state->data, _state->source, output, _state->time,
                                                                        _state->boundary);
        // FV-owned gradients are reusable scratch. The queued reads and their
        // completion checks protect this evaluation; the frozen auxiliary
        // outputs can survive subsequent scratch uses by another stage.
        if (_state->gradients && _state->gradients->_data->stateBinding)
            for (const auto &field : _state->gradients->fields)
                for (const auto *array : {&field.Owned(), &field.Ghost()})
                {
                    origin->gradientInputs.push_back(*array);
                    origin->gradientVersions.push_back({array->Version(), array->ShapeVersion()});
                }
        output._equationOrigin = std::move(origin);
        _state.reset();
    }

    void FiniteVolume::AuxiliaryPreparation::Apply(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                   CFV::ReconstructionRegion region)
    {
        const TimelineScope timeline("Cauchy/FV/Auxiliary/AllBackends");
        if (_queued)
        {
            ApplyQueued(engine, stream, region);
            return;
        }
        DNDS_check_throw_info(_state && stream.Context().device == R::Device::Host(), "auxiliary preparation/backend mismatch");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                  region == CFV::ReconstructionRegion::HaloDependent,
                              "invalid auxiliary region");
        if (region != CFV::ReconstructionRegion::HaloDependent)
            _state->Submit(engine, stream, _state->independentDone ? State::Part::InterfaceCore : State::Part::Core);
        if (region != CFV::ReconstructionRegion::Core)
            _state->Submit(engine, stream, State::Part::Dependent);
    }
    void FiniteVolume::AuxiliaryPreparation::ApplyInterfaceIndependent(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
        if (_queued)
        {
            ApplyIndependentQueued(engine, stream);
            return;
        }
        DNDS_check_throw_info(_state && stream.Context().device == R::Device::Host(), "auxiliary preparation/backend mismatch");
        _state->Submit(engine, stream, State::Part::Independent);
    }

}
