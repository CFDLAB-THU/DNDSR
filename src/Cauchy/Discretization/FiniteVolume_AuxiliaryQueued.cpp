#include "Cauchy/Discretization/FiniteVolumeAuxiliaryInternal.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeAuxiliaryKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(std::shared_ptr<QueuedState> state) : _queued(std::move(state)) {}

#ifdef DNDS_USE_CUDA
    namespace
    {
        /// An issued write epoch identifies queued input content without
        /// requiring completion. Access ordering/readiness remain the engine's
        /// contract; this stamp only detects later mutation/rebinding at Finish.
        struct QueuedArrayStamp
        {
            R::ArrayStorage::Shared storage;
            uint64_t shape, epoch;
            template <class T>
            explicit QueuedArrayStamp(const R::Array<T> &array)
                : storage(array.SharedStorage()), shape(array.ShapeVersion()), epoch(array.MutationEpoch()) {}
            void Check() const
            {
                DNDS_check_throw_info(storage->IsValid() && storage->ShapeVersion() == shape &&
                                          storage->Values().MutationEpoch() == epoch,
                                      "queued auxiliary operand changed after submission");
            }
        };
    }
    struct AuxiliaryValues::QueuedStorage
    {
        R::SessionAccess session;
        R::Array<real> empty;
        R::Array<NumericalStatus> meanStatus, faceStatus, partial, result;
        QueuedStorage(R::Device device, index means, index faces)
            : empty(R::RowLayout::Dense(0, 0), device),
              meanStatus(R::RowLayout::Dense(means, 1), device), faceStatus(R::RowLayout::Dense(faces, 1), device),
              partial(R::RowLayout::Dense(CUDA::StateReductionScratchRows(means + faces), 1), device),
              result(R::RowLayout::Dense(1, 1), device)
        {
            meanStatus.SetName("Cauchy/auxMeanStatus", DNDS_ALLOCATION_SITE());
            faceStatus.SetName("Cauchy/auxFaceStatus", DNDS_ALLOCATION_SITE());
            partial.SetName("Cauchy/auxStatusPartials", DNDS_ALLOCATION_SITE());
            result.SetName("Cauchy/auxStatusResult", DNDS_ALLOCATION_SITE());
        }
    };
    struct FiniteVolume::AuxiliaryPreparation::QueuedState : std::enable_shared_from_this<QueuedState>
    {
        FiniteVolume owner;
        std::shared_ptr<const Data> data;
        EquationDofs source;
        AuxiliaryValues buffers;
        std::shared_ptr<AuxiliaryValues::QueuedStorage> scratch;
        R::SessionAccess::Lease session;
        Geom::MeshReadAccess geometry;
        bool hasWall = false;
        std::optional<MeanGradients> gradients;
        std::optional<BoundaryInputs> boundary;
        RigidFrameParameters frame;
        real time;
        int phase = 0;
        bool boundaryPending = false, independentDone = false, needFaces = false;
        std::vector<QueuedArrayStamp> inputs, outputs;

        QueuedState(const FiniteVolume &plan, const EquationDofs &state, AuxiliaryValues output, real atTime, AuxiliaryInputs input)
            : owner(plan), data(plan._data), source(state.NumericAlias()), buffers(std::move(output)), scratch(buffers._queuedStorage),
              session(scratch->session.Acquire()), geometry(data->metrics.AcquireMeshRead(bool(data->wallDistance))),
              frame(data->motion.At(atTime).KernelParameters()), time(atTime), boundaryPending(input.deferBoundary)
        {
            if (input.meanGradients)
            {
                gradients = *input.meanGradients;
                DNDS_check_throw_info(gradients->_data && gradients->fields.size() == 1,
                                      "CUDA auxiliary gradient publication must contain one combined field");
                gradients->_data->CheckQueued(source, gradients->fields, data, time, false);
                data->metrics.ValidateField(gradients->fields.front(), source.ComponentNumber() * data->metrics.Dimension());
            }
            if (input.boundary)
                boundary = *input.boundary;
            if (data->equation.HasRANS() && data->equation.RANS().ComponentNumber() == 2)
            {
                DNDS_check_throw_info(gradients && data->wallDistance, "two-equation auxiliary preparation requires current gradients and wall distance");
                hasWall = true;
            }
            needFaces = bool(buffers.faceSpectralRadius || buffers.faceWaveJump || buffers.facePressureMinimum ||
                             buffers.cellSpectralSum || buffers.cellWaveJump || buffers.cellPressureMinimum);
        }
        R::Array<real> &Output(index i)
        {
            switch (i)
            {
            case 0:
                if (buffers.meanThermodynamics)
                    return *buffers.meanThermodynamics;
                break;
            case 1:
                if (buffers.faceSpectralRadius)
                    return *buffers.faceSpectralRadius;
                break;
            case 2:
                if (buffers.faceWaveJump)
                    return *buffers.faceWaveJump;
                break;
            case 3:
                if (buffers.facePressureMinimum)
                    return *buffers.facePressureMinimum;
                break;
            case 4:
                if (buffers.cellSpectralSum)
                    return buffers.cellSpectralSum->Owned();
                break;
            case 5:
                if (buffers.cellWaveJump)
                    return buffers.cellWaveJump->Owned();
                break;
            case 6:
                if (buffers.cellPressureMinimum)
                    return buffers.cellPressureMinimum->Owned();
                break;
            case 7:
                if (buffers._faceScratch)
                    return *buffers._faceScratch;
                break;
            }
            return scratch->empty;
        }
        auto StateInput(CUDA::AuxiliaryInput owned, CUDA::AuxiliaryInput ghost,
                        CUDA::AuxiliaryInput gradOwned, CUDA::AuxiliaryInput gradGhost, CUDA::AuxiliaryInput wall) const
        {
            CUDA::AuxiliaryStateInput result;
            result.parameters = R::MetadataArrayView(data->devicePhysics, owner.Device());
            result.mean = {owned, ghost};
            result.gradient = {gradOwned, gradGhost};
            result.components = int(source.ComponentNumber());
            result.dimension = data->metrics.Dimension();
            result.hasGradient = bool(gradients);
            result.hasWall = hasWall;
            if (hasWall)
                result.wallMean = {wall.data, wall.offsets, wall.rows, wall.width, wall.stride, wall.sizes, wall.kind, wall.device};
            return result;
        }
        void PinInput(bool ghost)
        {
            const auto &field = source.CoupledField();
            inputs.emplace_back(ghost ? field.Ghost() : field.Owned());
            if (gradients)
                inputs.emplace_back(ghost ? gradients->fields.front().Ghost() : gradients->fields.front().Owned());
        }
        void Submit(R::EagerEngine &engine, const R::ExecutionStream &stream,
                    bool ghost, bool means, const R::Array<index> &faces, const R::Array<index> &cells)
        {
            DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner.Device(), "auxiliary stream belongs to another backend/runtime");
            const auto &field = source.CoupledField();
            const auto &uGhost = ghost ? field.Ghost() : scratch->empty;
            const auto &gOwned = gradients ? gradients->fields.front().Owned() : scratch->empty;
            const auto &gGhost = gradients && ghost ? gradients->fields.front().Ghost() : scratch->empty;
            const auto pin = shared_from_this();
            const index first = ghost ? data->metrics.NumCell() : 0;
            const index end = ghost ? data->metrics.NumCellProc() : data->metrics.NumCell();
            if (means && buffers.meanThermodynamics)
                R::SubmitArrays(engine, stream, [pin, first, end](R::ExecutionContext context, auto u, auto ug, auto g, auto gg, auto out, auto status, auto wall)
                                { CUDA::PrepareAuxiliaryMeans(context, pin->StateInput(u, ug, g, gg, wall), out, status, first, end); }, R::ReadAs<R::TableRows>(field.Owned()), R::ReadAs<R::TableRows>(uGhost), R::ReadAs<R::TableRows>(gOwned), R::ReadAs<R::TableRows>(gGhost), R::WriteAs<R::TableRows>(*buffers.meanThermodynamics), R::WriteAs<R::FixedRows<1>>(scratch->meanStatus), R::ReadAs<R::TableRows>(hasWall ? data->wallDistance->CellMeans() : scratch->empty));
            if (!needFaces)
                return;
            const auto &incoming = boundary ? boundary->QuadratureValues().CoupledField().Owned() : scratch->empty;
            if (boundary)
                inputs.emplace_back(incoming);
            R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto u, auto ug, auto g, auto gg, auto prepared, auto incoming, auto spectral, auto wave, auto pressure, auto scratchValues, auto status, auto selected, auto samples, auto entries, auto stencilFaces, auto quadratureOffsets, auto geometry, auto area, auto wall, auto wallFaces)
                            {
                    CUDA::AuxiliaryFaceRequest request;
                    request.state = pin->StateInput(u, ug, g, gg, wall);
                    request.samples = samples;
                    request.entries = entries;
                    request.stencilFaces = stencilFaces;
                    request.quadratureOffsets = quadratureOffsets;
                    request.geometry = geometry;
                    request.area = area;
                    request.conditions = R::MetadataArrayView(pin->data->deviceBoundaryKinds, context.device);
                    if (pin->hasWall) request.wallFaces = {wallFaces.data, wallFaces.offsets, wallFaces.rows, wallFaces.width, wallFaces.stride, wallFaces.sizes, wallFaces.kind, wallFaces.device};
                    request.preparedMeans = prepared;
                    request.incoming = incoming;
                    request.spectral = spectral; request.wave = wave; request.pressure = pressure; request.scratch = scratchValues;
                    request.status = status; request.frame = pin->frame;
                    request.needSpectral = bool(pin->buffers.faceSpectralRadius || pin->buffers.cellSpectralSum);
                    request.needWave = bool(pin->buffers.faceWaveJump || pin->buffers.cellWaveJump);
                    request.needPressure = bool(pin->buffers.facePressureMinimum || pin->buffers.cellPressureMinimum);
                    CUDA::PrepareAuxiliaryFaces(context, request, selected); }, R::ReadAs<R::TableRows>(field.Owned()), R::ReadAs<R::TableRows>(uGhost), R::ReadAs<R::TableRows>(gOwned), R::ReadAs<R::TableRows>(gGhost), R::ReadAs<R::TableRows>(Output(0)), R::ReadAs<R::TableRows>(incoming), R::WriteAs<R::TableRows>(Output(1)), R::WriteAs<R::TableRows>(Output(2)), R::WriteAs<R::TableRows>(Output(3)), R::WriteAs<R::TableRows>(Output(7)), R::WriteAs<R::FixedRows<1>>(scratch->faceStatus), R::ReadAs<R::FixedRows<1>>(faces), R::ReadAs<R::FixedRows<1>>(data->stencil.samples), R::ReadAs<R::FixedRows<1>>(data->stencil.faceEntries), R::ReadAs<R::FixedRows<1>>(data->stencil.stencilFaces), R::ReadAs<R::FixedRows<1>>(data->metrics.FaceOffsets()), R::ReadAs<R::FixedRows<8>>(data->metrics.CellMetrics()), R::ReadAs<R::FixedRows<1>>(data->stencil.auxiliaryFaceArea), R::ReadAs<R::TableRows>(hasWall ? data->wallDistance->CellMeans() : scratch->empty), R::Read(hasWall ? data->wallDistance->FaceValues() : scratch->empty));
            if (!buffers.cellSpectralSum && !buffers.cellWaveJump && !buffers.cellPressureMinimum)
                return;
            R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto spectral, auto wave, auto pressure, auto scratchValues, auto sum, auto jump, auto minimum, auto selected, auto offsets, auto faces, auto area, auto geometry)
                            {
                    CUDA::AuxiliaryCellRequest request;
                    request.offsets = offsets; request.faces = faces;
                    request.area = area; request.geometry = geometry;
                    request.spectral = spectral; request.wave = wave; request.pressure = pressure; request.scratch = scratchValues;
                    request.spectralSum = sum; request.waveMaximum = jump; request.pressureMinimum = minimum;
                    const auto &w = pin->frame.angularVelocity;
                    request.angularSpeed = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
                    CUDA::ReduceAuxiliaryCells(context, request, selected); }, R::ReadAs<R::TableRows>(Output(1)), R::ReadAs<R::TableRows>(Output(2)), R::ReadAs<R::TableRows>(Output(3)), R::ReadAs<R::TableRows>(Output(7)), R::WriteAs<R::TableRows>(Output(4)), R::WriteAs<R::TableRows>(Output(5)), R::WriteAs<R::TableRows>(Output(6)), R::ReadAs<R::FixedRows<1>>(cells), R::ReadAs<R::FixedRows<1>>(data->stencil.auxiliaryCellOffsets), R::ReadAs<R::FixedRows<1>>(data->stencil.auxiliaryCellFaces), R::ReadAs<R::FixedRows<1>>(data->stencil.auxiliaryFaceArea), R::ReadAs<R::FixedRows<8>>(data->metrics.CellMetrics()));
        }
        void ReduceStatus(R::EagerEngine &engine, const R::ExecutionStream &stream)
        {
            const auto pin = shared_from_this();
            R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto means, auto faces, auto partial, auto result)
                            { CUDA::ReduceNumericalStatus(context, means, faces, partial, result); }, R::ReadAs<R::FixedRows<1>>(scratch->meanStatus), R::ReadAs<R::FixedRows<1>>(scratch->faceStatus), R::WriteAs<R::FixedRows<1>>(scratch->partial), R::WriteAs<R::FixedRows<1>>(scratch->result));
            for (const auto *array : AuxiliaryDetail::EquationAuxiliaryArrays(buffers))
                if (array)
                    outputs.emplace_back(*array);
        }
    };

    FiniteVolume::AuxiliaryPreparation FiniteVolume::AuxiliaryPreparation::BuildQueuedState(
        const FiniteVolume &owner, const EquationDofs &state, AuxiliaryValues &output, real time, AuxiliaryInputs inputs)
    {
        DNDS_check_throw_info(owner.Device().space == R::MemorySpace::CUDA, "queued auxiliary backend is unavailable");
        const index means = output.meanThermodynamics ? owner.Metrics().NumCellProc() : 0;
        const bool needFaces = bool(output.faceSpectralRadius || output.faceWaveJump || output.facePressureMinimum ||
                                    output.cellSpectralSum || output.cellWaveJump || output.cellPressureMinimum);
        const index faces = needFaces ? owner._data->stencil.stencilFaces.RowCount() : 0;
        if (!output._queuedStorage)
            output._queuedStorage = std::make_shared<AuxiliaryValues::QueuedStorage>(owner.Device(), means, faces);
        DNDS_check_throw_info(output._queuedStorage->meanStatus.RowCount() == means && output._queuedStorage->faceStatus.RowCount() == faces &&
                                  output._queuedStorage->result.GetDevice() == owner.Device(),
                              "auxiliary queued scratch selection/device changed; rebuild the bundle");
        return AuxiliaryPreparation(std::make_shared<QueuedState>(owner, state, output, time, inputs));
    }
    void FiniteVolume::AuxiliaryPreparation::ApplyIndependentQueued(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
        auto &s = *_queued;
        DNDS_check_throw_info(s.phase == 0 && s.boundaryPending && !s.independentDone, "auxiliary independent phase is unavailable");
        s.PinInput(false);
        s.Submit(engine, stream, false, true, s.data->stencil.auxiliaryInterfaceIndependentFaces, s.data->stencil.interfaceIndependentCells);
        s.independentDone = true;
    }
    void FiniteVolume::AuxiliaryPreparation::ApplyQueued(R::EagerEngine &engine, const R::ExecutionStream &stream, CFV::ReconstructionRegion region)
    {
        const TimelineScope timeline("Cauchy/FV/Auxiliary/Submit");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                  region == CFV::ReconstructionRegion::HaloDependent,
                              "invalid auxiliary region");
        auto &s = *_queued;
        if (region != CFV::ReconstructionRegion::HaloDependent)
        {
            DNDS_check_throw_info(s.phase == 0 && !s.boundaryPending, "auxiliary core was already submitted or boundary is pending");
            if (!s.independentDone)
                s.PinInput(false);
            s.Submit(engine, stream, false, !s.independentDone,
                     s.independentDone ? s.data->stencil.auxiliaryInterfaceFaces : s.data->stencil.auxiliaryCoreFaces,
                     s.independentDone ? s.data->stencil.interfaceCoreCells : s.data->stencil.coreCells);
            s.phase = 1;
        }
        if (region != CFV::ReconstructionRegion::Core)
        {
            DNDS_check_throw_info(s.phase == 1, "auxiliary dependent phase requires the core submission");
            if (s.gradients)
                s.gradients->_data->CheckQueued(s.source, s.gradients->fields, s.data, s.time, true);
            s.PinInput(true);
            s.Submit(engine, stream, true, true, s.data->stencil.auxiliaryDependentFaces, s.data->stencil.dependentCells);
            s.ReduceStatus(engine, stream);
            s.phase = 2;
        }
    }
    void FiniteVolume::AuxiliaryPreparation::BindQueuedBoundary(const BoundaryInputs &input)
    {
        auto &s = *_queued;
        DNDS_check_throw_info(s.phase == 0 && s.boundaryPending, "auxiliary boundary is not pending");
        s.owner.ValidateBoundaryInputs(input, s.time);
        DNDS_check_throw_info(input.QuadratureValues().Device() == s.owner.Device(), "auxiliary boundary device differs");
        for (const auto *array : AuxiliaryDetail::EquationAuxiliaryArrays(s.buffers))
            if (array)
                DNDS_check_throw_info(!array->Storage().Overlaps(input.QuadratureValues().CoupledField().Owned().Storage()),
                                      "auxiliary output aliases deferred boundary input");
        s.boundary = input;
        s.boundaryPending = false;
    }
    void FiniteVolume::AuxiliaryPreparation::FinishQueued(AuxiliaryValues &output)
    {
        const auto s = _queued;
        DNDS_check_throw_info(s && s->phase == 2 && s->scratch->result.Storage().IsIdle(),
                              "queued auxiliary publication requires a completed full preparation");
        const auto original = AuxiliaryDetail::EquationAuxiliaryArrays(s->buffers), current = AuxiliaryDetail::EquationAuxiliaryArrays(output);
        for (index k = 0; k < index(original.size()); ++k)
            DNDS_check_throw_info(bool(original[k]) == bool(current[k]) && (!original[k] || original[k]->SharedStorage() == current[k]->SharedStorage()),
                                  "queued auxiliary output was rebound");
        const std::array<const std::optional<CFV::FVField> *, 3> before{&s->buffers.cellSpectralSum, &s->buffers.cellWaveJump, &s->buffers.cellPressureMinimum};
        const std::array<const std::optional<CFV::FVField> *, 3> after{&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum};
        for (index i = 0; i < 3; ++i)
            if (*after[i])
                DNDS_check_throw_info((*after[i])->Space().Same(s->source.Space()) &&
                                          (*after[i])->Ghost().SharedStorage() == (*before[i])->Ghost().SharedStorage(),
                                      "queued auxiliary cell row space or ghost binding changed");
        for (const auto &stamp : s->inputs)
            stamp.Check();
        for (const auto &stamp : s->outputs)
            stamp.Check();
        // This is the explicit reporting/publication boundary after the
        // caller's drain. Only one mandatory status record crosses to the CPU.
        const auto status = s->scratch->result.To(R::Device::Host());
        RequireNumericalSuccess(status.Read().View<R::FixedRows<1>>()(0, 0), "queued auxiliary preparation");
        auto origin = std::make_shared<AuxiliaryValues::EquationOrigin>(s->data, s->source, output, s->time, s->boundary);
        if (s->gradients)
            for (const auto &field : s->gradients->fields)
                for (const auto *array : {&field.Owned(), &field.Ghost()})
                {
                    origin->gradientInputs.push_back(*array);
                    origin->gradientVersions.push_back({array->Version(), array->ShapeVersion()});
                }
        output._equationOrigin = std::move(origin);
        _queued.reset();
    }
#else
    FiniteVolume::AuxiliaryPreparation FiniteVolume::AuxiliaryPreparation::BuildQueuedState(
        const FiniteVolume &, const EquationDofs &, AuxiliaryValues &, real, AuxiliaryInputs)
    {
        DNDS_check_throw_info(false, "CUDA auxiliary preparation is not built");
        return AuxiliaryPreparation(std::shared_ptr<QueuedState>{});
    }
    void FiniteVolume::AuxiliaryPreparation::ApplyQueued(R::EagerEngine &, const R::ExecutionStream &, CFV::ReconstructionRegion)
    {
        DNDS_check_throw_info(false, "CUDA auxiliary preparation is not built");
    }
    void FiniteVolume::AuxiliaryPreparation::ApplyIndependentQueued(R::EagerEngine &, const R::ExecutionStream &)
    {
        DNDS_check_throw_info(false, "CUDA auxiliary preparation is not built");
    }
    void FiniteVolume::AuxiliaryPreparation::BindQueuedBoundary(const BoundaryInputs &)
    {
        DNDS_check_throw_info(false, "CUDA auxiliary preparation is not built");
    }
    void FiniteVolume::AuxiliaryPreparation::FinishQueued(AuxiliaryValues &)
    {
        DNDS_check_throw_info(false, "CUDA auxiliary preparation is not built");
    }
#endif
}
