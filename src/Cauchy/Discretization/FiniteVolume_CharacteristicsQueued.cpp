#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeReconstructionKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeReconstructionKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    struct FiniteVolume::Workspace::CharacteristicDeviceStorage
    {
        R::Array<NumericalStatus> status, partial, result, empty;
        R::Array<int> flags;
        R::Array<index> countPartial, count;
    };

    void FiniteVolume::Workspace::BuildCharacteristicStorage()
    {
        if ((!flowCharacteristic && !gradientFlowCharacteristic) || owner->metrics.Device() == R::Device::Host())
            return;
#ifdef DNDS_USE_CUDA
        const auto device = owner->metrics.Device();
        const index faces = owner->metrics.NumFaceProc();
        characteristicDevice = std::make_shared<CharacteristicDeviceStorage>();
        auto &storage = *characteristicDevice;
        storage.status = R::Array<NumericalStatus>(R::RowLayout::Dense(faces, 1), device);
        storage.partial = R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(faces), 1), device);
        storage.result = R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), device);
        storage.empty = R::Array<NumericalStatus>(R::RowLayout::Dense(0, 1), device);
        storage.flags = R::Array<int>(R::RowLayout::Dense(faces, 1), device);
        storage.countPartial = R::Array<index>(R::RowLayout::Dense(std::max<index>(1, std::min<index>(256, (faces + 127) / 128)), 1), device);
        storage.count = R::Array<index>(R::RowLayout::Dense(1, 1), device);
        storage.status.SetName("Cauchy/characteristicStatus", DNDS_ALLOCATION_SITE());
        storage.partial.SetName("Cauchy/characteristicStatusPartials", DNDS_ALLOCATION_SITE());
        storage.result.SetName("Cauchy/characteristicStatusResult", DNDS_ALLOCATION_SITE());
        storage.flags.SetName("Cauchy/characteristicFallbackFlags", DNDS_ALLOCATION_SITE());
        storage.countPartial.SetName("Cauchy/characteristicFallbackPartials", DNDS_ALLOCATION_SITE());
        storage.count.SetName("Cauchy/characteristicFallbackCount", DNDS_ALLOCATION_SITE());
#else
        DNDS_check_throw_info(false, "CUDA characteristics are not built");
#endif
    }

    std::shared_ptr<FiniteVolume::Workspace::CharacteristicSubmission>
    FiniteVolume::Workspace::SubmitCharacteristics(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state,
        OptionalRef<const AuxiliaryValues> auxiliary)
    {
        FiniteVolume(owner).ValidateWorkspace(state, *this);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device(),
                              "characteristic stream or placement differs");
        DNDS_check_throw_info(bool(flowCharacteristic) != bool(gradientFlowCharacteristic),
                              "characteristic submission needs exactly one matrix target");
        const auto flow = state.LocateBlock("flow");
        auto &forward = flowCharacteristic ? flowCharacteristic->forward : gradientFlowCharacteristic->forward;
        auto &inverse = flowCharacteristic ? flowCharacteristic->inverse : gradientFlowCharacteristic->inverse;
        const index first = flowCharacteristic ? flowCharacteristic->first : gradientFlowCharacteristic->first;
        const index count = flowCharacteristic ? flowCharacteristic->count : gradientFlowCharacteristic->count;
        DNDS_check_throw_info(flow.componentNumber == 5 && flow.firstComponent == first && count == 5 &&
                                  flow.physicalField == flowCharacteristicField,
                              "characteristic component range differs from flow5");
        for (const auto *array : {&forward, &inverse})
            DNDS_check_throw_info(array->GetDevice() == owner->metrics.Device() && array->Layout().Kind() == R::RowLayoutKind::Table &&
                                      array->RowCount() == owner->metrics.NumFaceProc() && array->Layout().Width() == 25,
                                  "characteristic output shape or placement differs");
        auto record = std::make_shared<CharacteristicSubmission>();
        if (owner->metrics.Device() != R::Device::Host())
        {
#ifdef DNDS_USE_CUDA
            DNDS_check_throw_info(characteristicDevice && !owner->equation.HasMixture() && first == 0 && state.PhysicalFieldNumber() == 1,
                                  "CUDA characteristics require combined perfect-gas flow5/SA/2EQ");
            struct Inputs
            {
                Geom::MeshReadAccess geometry;
                explicit Inputs(const Data &data)
                    : geometry(data.mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux())) {}
            };
            const auto pin = std::make_shared<Inputs>(*owner);
            auto &storage = *characteristicDevice;
            const auto &field = state.CoupledField();
            R::SubmitArrays(engine, stream, [pin, gas = owner->equation.Gas()](R::ExecutionContext context, auto owned, auto ghost, auto left, auto right, auto status, auto flags, auto samples, auto entries, auto quadrature)
                            { CUDA::PrepareFaceCharacteristicMatrices(context,
                                                                      {samples, entries, quadrature},
                                                                      {owned, ghost}, gas, left, right, status, flags); }, R::ReadAs<R::TableRows>(field.Owned()), R::ReadAs<R::TableRows>(field.Ghost()), R::WriteAs<R::TableRows>(forward), R::WriteAs<R::TableRows>(inverse), R::WriteAs<R::FixedRows<1>>(storage.status), R::WriteAs<R::FixedRows<1>>(storage.flags), R::ReadAs<R::FixedRows<1>>(owner->stencil.samples), R::ReadAs<R::FixedRows<1>>(owner->stencil.characteristicFaceEntries), R::ReadAs<R::FixedRows<7>>(owner->metrics.FaceQuadrature()));
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto empty, auto partial, auto result)
                            { CUDA::ReduceNumericalStatus(context, rows, empty, partial, result); }, R::ReadAs<R::FixedRows<1>>(storage.status), R::ReadAs<R::FixedRows<1>>(storage.empty), R::WriteAs<R::FixedRows<1>>(storage.partial), R::WriteAs<R::FixedRows<1>>(storage.result));
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto flags, auto partial, auto result)
                            { CUDA::ReduceCharacteristicFallbacks(context, flags, partial, result); }, R::ReadAs<R::FixedRows<1>>(storage.flags), R::WriteAs<R::FixedRows<1>>(storage.countPartial), R::WriteAs<R::FixedRows<1>>(storage.count));
            record->device = characteristicDevice;
            record->resultEpoch = storage.result.MutationEpoch();
            if (owner->settings.numerics.characteristicDiagnostics)
            {
                if (flowCharacteristic)
                    owner->reconstruction->SubmitValidateFaceTransform(engine, stream, *flowCharacteristic, *fields[flowCharacteristicField].reconstruction);
                else
                    owner->gradient.SubmitValidateFaceTransform(engine, stream, *gradientFlowCharacteristic, *fields[flowCharacteristicField].gradientExecution);
            }
#else
            DNDS_check_throw_info(false, "CUDA characteristics are not built");
#endif
            return record;
        }
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            EquationDofs state;
            Geom::MeshReadAccess mesh;
            std::optional<PerfectGas> gas;
            std::optional<DerivedFields> derivedOwned, derivedGhost;
            std::array<int, 5> columns{};
            index ownedSlot = -1, ghostSlot = -1, thermalSlot = -1, thermalGhostSlot = -1;
            explicit Inputs(std::shared_ptr<const Data> data, const EquationDofs &values)
                : owner(std::move(data)), state(values.NumericAlias()), mesh(owner->metrics.AcquireMeshRead()) {}
        };
        auto pin = std::make_shared<Inputs>(owner, state);
        std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> inputs;
        const auto add = [&](const R::Array<real> &array)
        {
            const index slot = index(inputs.size());
            inputs.push_back(R::ReadAs<R::UniformTableRows>(array));
            return slot;
        };
        // Retain every mean field: cached mixture properties depend on the
        // species allocation too, even though only momentum is transformed.
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const index owned = add(state.PhysicalField(iField).Owned());
            const index ghost = add(state.PhysicalField(iField).Ghost());
            if (iField == flow.physicalField)
                pin->ownedSlot = owned, pin->ghostSlot = ghost;
        }
        if (!owner->equation.HasMixture())
            pin->gas.emplace(owner->equation.Gas());
        else
        {
            DNDS_check_throw_info(auxiliary, "mixture characteristics need explicit cached thermodynamics");
            if (auxiliary->meanThermodynamics)
                pin->thermalSlot = add(*auxiliary->meanThermodynamics);
            else
            {
                DNDS_check_throw_info(auxiliary->derivedOwned && auxiliary->derivedGhost,
                                      "mixture characteristic cache requires both row parts");
                pin->derivedOwned = auxiliary->derivedOwned;
                pin->derivedGhost = auxiliary->derivedGhost;
                const std::array<DerivedQuantity, 5> quantities{DerivedQuantity::Pressure, DerivedQuantity::SoundSpeed,
                                                                DerivedQuantity::DiffusionBound, DerivedQuantity::TotalEnthalpy, DerivedQuantity::PressureInternalEnergyDerivative};
                for (index i = 0; i < 5; ++i)
                {
                    pin->columns[i] = pin->derivedOwned->Column(quantities[i]);
                    DNDS_check_throw_info(pin->columns[i] == pin->derivedGhost->Column(quantities[i]),
                                          "derived characteristic column projections differ");
                }
                pin->thermalSlot = add(pin->derivedOwned->Values());
                pin->thermalGhostSlot = add(pin->derivedGhost->Values());
            }
        }
        std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> outputs{
            R::ReadWriteAs<R::UniformTableRows>(forward), R::ReadWriteAs<R::UniformTableRows>(inverse)};
        R::SubmitArrayBatch(engine, stream, [pin, record, first](R::ExecutionContext context, auto inputs, auto outputs, auto faces, auto cells, auto offsets, auto quadrature)
                            {
            std::optional<MeanThermodynamicView> thermal;
            if (pin->derivedOwned)
            {
                DNDS_check_throw_info(pin->derivedOwned->Current(pin->state, pin->owner->equation) &&
                                          pin->derivedGhost->Current(pin->state, pin->owner->equation),
                                      "queued characteristic cache is stale");
                thermal.emplace(RHSFieldView{inputs[pin->thermalSlot], inputs[pin->thermalGhostSlot]}, pin->columns, 5);
            }
            else if (pin->thermalSlot >= 0)
                thermal.emplace(inputs[pin->thermalSlot]);
            const RHSFieldView means{inputs[pin->ownedSlot], inputs[pin->ghostSlot]};
            const auto table = [](const auto &view)
            {
                return R::ArrayView<real, R::AccessBoundsChecks, R::TableRows>{view.data, view.offsets, view.rows,
                    view.width, view.stride, view.sizes, view.kind, view.device};
            };
            const auto left = table(outputs[0]), right = table(outputs[1]);
            R::ValidateViewDevices(context.device, means, left, right);
            if (thermal) thermal->values.ValidateDevice(context.device);
            FiniteVolumeStencilView stencil;
            stencil.stencilFaces = faces;
            CFV::FiniteVolumeHostView metric;
            metric.cells = cells; metric.faceOffsets = offsets; metric.quadrature = quadrature;
            record->fallbacks = RequireNumericalSuccess(CPU::PrepareFaceCharacteristicMatrices(
                stencil, pin->mesh.HostView(), metric, means, first,
                left, right, pin->gas, thermal), "queued characteristics");
            if (pin->owner->settings.numerics.characteristicDiagnostics)
                RequireNumericalSuccess(CPU::ValidateFaceCharacteristicMatrices(left.ConstView(), right.ConstView()),
                                        "characteristic inverse diagnostic");
            record->completed.store(true, std::memory_order_release); }, std::move(inputs), std::move(outputs), std::make_tuple(R::ReadAs<R::FixedRows<1>>(owner->stencil.stencilFaces), R::ReadAs<R::FixedRows<8>>(owner->metrics.CellMetrics()), R::ReadAs<R::FixedRows<1>>(owner->metrics.FaceOffsets()), R::ReadAs<R::FixedRows<7>>(owner->metrics.FaceQuadrature())));
        return record;
    }

    void FiniteVolume::Workspace::PublishCharacteristics(const std::shared_ptr<CharacteristicSubmission> &record)
    {
        DNDS_check_throw_info(record && !record->published, "characteristic result missing or already published");
        if (record->device)
        {
            const auto &storage = *record->device;
            DNDS_check_throw_info(storage.result.Storage().IsIdle() && storage.count.Storage().IsIdle() &&
                                      storage.result.MutationEpoch() == record->resultEpoch,
                                  "characteristic result incomplete or reused before publication");
            const auto status = storage.result.To(R::Device::Host());
            RequireNumericalSuccess(status.Read().View()(0, 0), "queued characteristics");
            const auto count = storage.count.To(R::Device::Host());
            record->fallbacks = count.Read().View()(0, 0);
            if (owner->settings.numerics.characteristicDiagnostics)
            {
                if (flowCharacteristic)
                    owner->reconstruction->CheckStatus(*fields[flowCharacteristicField].reconstruction);
                else
                    owner->gradient.CheckStatus(*fields[flowCharacteristicField].gradientExecution);
            }
        }
        else
            DNDS_check_throw_info(record->completed.load(std::memory_order_acquire), "characteristic CPU operation incomplete");
        characteristicFallbacks = record->fallbacks;
        record->published = true;
    }
}
