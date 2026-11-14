#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy
{
#ifdef DNDS_USE_CUDA
    /// Long-lived immutable descriptors and scratch for one FV workspace.
    /// The mesh-generation scopes deliberately prevent geometry edits while
    /// this workspace is usable. A new geometry requires rebuilding the plan.
    struct FiniteVolume::Workspace::DeviceFluxData
    {
        std::shared_ptr<const Data> owner;
        CFV::FiniteVolumeRead metric;
        FiniteVolumeStencilStorage::ReadAccess stencil;
        R::ReadAccess conditions;
        std::optional<CFV::ReconstructionResidentRead> basis;
        std::optional<WallDistanceRead> walls;
        std::optional<CUDA::RHSGeometryPublication> geometry;
        R::MetadataBuffer<CUDA::RHSKernelParameters> parameters;
        std::optional<R::ReadAccess> parameterRead;
        R::Array<NumericalStatus> faceStatus, cellStatus, partial, result;
        R::Array<index> fallbackCounts, allCells, allFaces;
        int submittedRegions = 0;
        bool pending = false;

        DeviceFluxData(std::shared_ptr<const Data> data, ThermalView thermal)
            : owner(std::move(data)), metric(owner->metrics.AcquireRead({owner->metrics.Device(), nullptr})),
              stencil(owner->stencil.Read({owner->metrics.Device(), nullptr})),
              conditions(owner->deviceBoundaryConditions.Read({owner->metrics.Device(), nullptr}))
        {
            const auto device = owner->metrics.Device();
            const R::ExecutionContext context{device, nullptr};
            const auto metrics = metric.View();
            const auto incidence = stencil.View();
            CUDA::RHSGeometry view;
            view.samples = incidence.samples;
            view.offsets = incidence.offsets;
            view.faceEntries = incidence.faceEntries;
            view.faceOffsets = metrics.faceOffsets;
            view.cells = metrics.cells;
            view.quadrature = metrics.quadrature;
            view.conditions = {conditions.View<PerfectGasBoundaryParameters>().data, nullptr,
                               index(owner->deviceBoundaryConditions.size()), 1, 1, nullptr, R::RowLayoutKind::Table, device};
            if (owner->reconstruction)
            {
                basis.emplace(owner->reconstruction->AcquireResidentRead());
                view.basis = basis->PointBasisView();
                view.hasBasis = true;
            }
            if (owner->wallDistance)
            {
                walls.emplace(owner->wallDistance->AcquireRead(context));
                const auto wall = walls->View();
                view.wallMean = {wall.cellMean.data, wall.cellMean.offsets, wall.cellMean.rows, wall.cellMean.width,
                                 wall.cellMean.stride, wall.cellMean.sizes, wall.cellMean.kind, wall.cellMean.device};
                view.wallFaces = {wall.faces.data, wall.faces.offsets, wall.faces.rows, wall.faces.width,
                                  wall.faces.stride, wall.faces.sizes, wall.faces.kind, wall.faces.device};
                view.hasWall = true;
            }
            geometry.emplace(view, context);
            CUDA::RHSKernelParameters physics;
            physics.physics = PerfectGasSpatialParameters::FromEquation(owner->equation, owner->metrics.Dimension());
            physics.thermal = thermal;
            physics.Validate(int(thermal.components));
            parameters = R::MetadataBuffer<CUDA::RHSKernelParameters>({physics}).CopyTo(device, context);
            parameters.SetDiagnosticName("Cauchy/rhsPolicy", DNDS_ALLOCATION_SITE());
            parameterRead.emplace(parameters.Read(context));
            const index cells = owner->metrics.NumCell(), faces = incidence.faceEntries.RowCount();
            faceStatus = R::Array<NumericalStatus>(R::RowLayout::Dense(faces, 1), device);
            cellStatus = R::Array<NumericalStatus>(R::RowLayout::Dense(cells, 1), device);
            partial = R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(cells + 1), 1), device);
            result = R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), device);
            fallbackCounts = R::Array<index>(R::RowLayout::Dense(faces, 1), device);
            const auto sequence = [&](index rows)
            {
                R::Array<index> host(R::RowLayout::Dense(rows, 1));
                {
                    auto write = host.Write();
                    const auto values = write.View<R::FixedRows<1>>();
                    OMP::ForRows(0, rows, [&](index iRow) noexcept
                                 { values(iRow, 0) = iRow; });
                }
                return host.To(device, context);
            };
            allCells = sequence(cells);
            allFaces = sequence(faces);
            faceStatus.SetName("Cauchy/fluxFaceStatus", DNDS_ALLOCATION_SITE());
            cellStatus.SetName("Cauchy/fluxCellStatus", DNDS_ALLOCATION_SITE());
            partial.SetName("Cauchy/fluxStatusPartials", DNDS_ALLOCATION_SITE());
            result.SetName("Cauchy/fluxStatusResult", DNDS_ALLOCATION_SITE());
            fallbackCounts.SetName("Cauchy/fluxFaceFallbackCounts", DNDS_ALLOCATION_SITE());
            allCells.SetName("Cauchy/fluxAllCells", DNDS_ALLOCATION_SITE());
            allFaces.SetName("Cauchy/fluxAllFaces", DNDS_ALLOCATION_SITE());
        }
        auto Parameters() const
        {
            return R::ArrayView<const CUDA::RHSKernelParameters, R::AccessBoundsChecks, R::FixedRows<1>>{
                parameterRead->View<CUDA::RHSKernelParameters>().data, nullptr, 1, 1, 1, nullptr, R::RowLayoutKind::Table, owner->metrics.Device()};
        }
    };
#endif

    void FiniteVolume::Workspace::BuildDeviceFlux()
    {
        if (owner->metrics.Device() == R::Device::Host())
            return;
#ifdef DNDS_USE_CUDA
        deviceFlux = std::make_shared<DeviceFluxData>(owner, thermal.View());
#else
        DNDS_check_throw_info(false, "CUDA flux execution is not built");
#endif
    }
    const R::Array<index> &FiniteVolume::Workspace::DeviceCellSelection(CFV::ReconstructionRegion region) const
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(deviceFlux && (region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                             region == CFV::ReconstructionRegion::HaloDependent),
                              "invalid device cell selection");
        return region == CFV::ReconstructionRegion::All ? deviceFlux->allCells : region == CFV::ReconstructionRegion::Core ? owner->stencil.coreCells
                                                                                                                           : owner->stencil.dependentCells;
#else
        DNDS_check_throw_info(false, "CUDA cell selection is not built");
        return owner->stencil.coreCells;
#endif
    }
    void FiniteVolume::Workspace::ResetDeviceFluxStatus(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(deviceFlux && !deviceFlux->pending, "device flux status is unavailable or pending");
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto cells, auto result)
                        { CUDA::ClearNumericalStatus(context, cells); CUDA::ClearNumericalStatus(context, result); }, R::WriteAs<R::FixedRows<1>>(deviceFlux->cellStatus), R::WriteAs<R::FixedRows<1>>(deviceFlux->result));
        deviceFlux->pending = true;
        deviceFlux->submittedRegions = 0;
#else
        DNDS_check_throw_info(false, "CUDA flux execution is not built");
#endif
    }
    void FiniteVolume::Workspace::CheckDeviceFluxStatus()
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(deviceFlux && deviceFlux->pending && deviceFlux->submittedRegions == 3 && deviceFlux->result.Storage().IsIdle(),
                              "device flux status requires a completed full evaluation");
        const auto result = deviceFlux->result.To(R::Device::Host());
        deviceFlux->pending = false;
        RequireNumericalSuccess(result.Read().View()(0, 0), "queued spatial flux");
#else
        DNDS_check_throw_info(false, "CUDA flux execution is not built");
#endif
    }

    void FiniteVolume::Workspace::SubmitDeviceRHS(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                  const EquationDofs &state, EquationDofs &output, CFV::ReconstructionRegion region, real time,
                                                  OptionalRef<const AuxiliaryValues> auxiliary, OptionalRef<const R::Array<real>> incoming)
    {
        const TimelineScope timeline("Cauchy/FV/RHS/Submit");
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(deviceFlux && deviceFlux->pending && engine.Owns(stream) && stream.Context().device == owner->metrics.Device(),
                              "device flux needs an initialized status segment and matching stream");
        state.RequireSchema(owner->equation.Schema());
        state.RequireCompatible(output);
        EquationDofOperations::RequireDistinct(state, output);
        DNDS_check_throw_info(state.Space().Same(owner->metrics.Space()) && state.Device() == owner->metrics.Device(), "device flux row space differs");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core || region == CFV::ReconstructionRegion::HaloDependent,
                              "invalid device flux region");
        const int mask = region == CFV::ReconstructionRegion::All ? 3 : region == CFV::ReconstructionRegion::Core ? 1
                                                                                                                  : 2;
        DNDS_check_throw_info(!(deviceFlux->submittedRegions & mask), "device flux region already submitted");
        const bool ghosts = region != CFV::ReconstructionRegion::Core;
        const auto &n = owner->settings.numerics;
        const auto &empty = owner->deviceEmpty;
        const auto &mean = state.CoupledField();
        OptionalRef<const CFV::FVField> rec, viscous;
        CUDA::RHSKernelControls controls;
        controls.fluxPolicy = owner->fluxPolicy;
        controls.solver = owner->flux.Kind();
        controls.components = int(state.ComponentNumber());
        controls.dimension = owner->metrics.Dimension();
        controls.modes = owner->reconstruction ? owner->reconstruction->NRecDOF() : 0;
        controls.direct = n.reconstruction != "variational";
        controls.linear = n.reconstruction == "gradient";
        controls.firstOrder = meanOnly || n.firstOrderConvection;
        controls.viscous = !meanOnly && owner->equation.Viscous();
        controls.primitiveGradient = bool(primitiveGradient);
        controls.positivity = n.positivity;
        controls.faceFallback = n.faceFallback;
        controls.useStencil = n.riemann.useStencil;
        controls.produceLow = bool(lowOrderResidual);
        controls.time = time;
        controls.frame = owner->motion.At(time).KernelParameters();
        if (!controls.firstOrder && (!controls.direct || controls.linear))
        {
            if (controls.direct)
            {
                DNDS_check_throw_info(directGradients.size() == 1, "device direct gradient is missing");
                rec.Emplace(directGradients.front());
            }
            else
                rec.Emplace(FluxCoefficients().CoupledField());
        }
        if (primitiveGradient)
            viscous.Emplace(*primitiveGradient);
        else if (controls.viscous && !n.viscousLimited && (n.limiter == "WBAP" || limitedVR || n.positivity) && rec)
        {
            if (controls.direct)
            {
                DNDS_check_throw_info(directViscousGradients.size() == 1, "device unlimited viscous gradient is missing");
                viscous.Emplace(directViscousGradients.front());
            }
            else
                viscous.Emplace(Coefficients().CoupledField());
            controls.distinctViscous = true;
        }
        const auto &recOwned = rec ? rec->Owned() : empty, &recGhost = rec && ghosts ? rec->Ghost() : empty;
        const auto &visOwned = viscous ? viscous->Owned() : empty, &visGhost = viscous && ghosts ? viscous->Ghost() : empty;
        const auto &prepared = auxiliary && auxiliary->meanThermodynamics ? *auxiliary->meanThermodynamics : empty;
        const auto &wave = auxiliary && auxiliary->cellWaveJump ? auxiliary->cellWaveJump->Owned() : empty;
        const auto &waveGhost = auxiliary && auxiliary->cellWaveJump && ghosts ? auxiliary->cellWaveJump->Ghost() : empty;
        const auto &pressure = auxiliary && auxiliary->cellPressureMinimum ? auxiliary->cellPressureMinimum->Owned() : empty;
        const auto &pressureGhost = auxiliary && auxiliary->cellPressureMinimum && ghosts ? auxiliary->cellPressureMinimum->Ghost() : empty;
        const auto &packet = incoming ? *incoming : empty;
        for (const auto *input : {&mean.Owned(), &mean.Ghost(), &recOwned, &recGhost, &visOwned, &visGhost,
                                  &prepared, &wave, &waveGhost, &pressure, &pressureGhost, &packet})
        {
            DNDS_check_throw_info(!output.CoupledField().Owned().Storage().Overlaps(input->Storage()),
                                  "device residual aliases a flux input");
            if (lowOrderResidual)
                DNDS_check_throw_info(!lowOrderResidual->CoupledField().Owned().Storage().Overlaps(input->Storage()),
                                      "device low-order residual aliases a flux input");
        }
        const bool hasMean = auxiliary && bool(auxiliary->meanThermodynamics), hasIncoming = bool(incoming);
        const bool hasWave = auxiliary && bool(auxiliary->cellWaveJump), hasPressure = auxiliary && bool(auxiliary->cellPressureMinimum);
        auto &lowFace = lowOrderFaceFlux ? *lowOrderFaceFlux : faceFlux;
        auto &lowOutput = lowOrderResidual ? lowOrderResidual->CoupledField().Owned() : output.CoupledField().Owned();
        const auto &faces = region == CFV::ReconstructionRegion::All ? deviceFlux->allFaces : region == CFV::ReconstructionRegion::Core ? owner->stencil.rhsCoreFaces
                                                                                                                                        : owner->stencil.rhsDependentFaces;
        const auto &cells = DeviceCellSelection(region);
        const auto pin = deviceFlux;
        // Inactive low-order outputs alias their primary arrays deliberately;
        // the leaf never accesses them. Tracking merges their access footprints.
        R::SubmitArrays(engine, stream, [pin, controls, ghosts, hasMean, hasIncoming, hasWave, hasPressure](R::ExecutionContext context, auto u, auto ug, auto rec, auto recg, auto vis, auto visg, auto prepared, auto packet, auto wave, auto waveg, auto pressure, auto pressureg, auto high, auto low, auto force, auto status, auto counts, auto selected)
                        {
                CUDA::RHSFields fields;
                fields.mean = {u, ug}; fields.reconstruction = {rec, recg}; fields.viscousData = {vis, visg};
                fields.preparedMean = prepared; fields.incoming = packet;
                fields.waveJump = {wave, waveg}; fields.pressureMinimum = {pressure, pressureg};
                fields.hasPreparedMean = hasMean; fields.hasIncoming = hasIncoming; fields.hasWave = hasWave; fields.hasPressure = hasPressure;
                CUDA::EvaluateFluxFaces(context, *pin->geometry, pin->Parameters(), controls, fields, {high, low, force, status, counts}, selected, ghosts); }, R::ReadAs<R::TableRows>(mean.Owned()), R::ReadAs<R::TableRows>(ghosts ? mean.Ghost() : empty), R::ReadAs<R::TableRows>(recOwned), R::ReadAs<R::TableRows>(recGhost), R::ReadAs<R::TableRows>(visOwned), R::ReadAs<R::TableRows>(visGhost), R::ReadAs<R::TableRows>(prepared), R::ReadAs<R::TableRows>(packet), R::ReadAs<R::TableRows>(wave), R::ReadAs<R::TableRows>(waveGhost), R::ReadAs<R::TableRows>(pressure), R::ReadAs<R::TableRows>(pressureGhost), R::WriteAs<R::TableRows>(faceFlux), R::WriteAs<R::TableRows>(lowFace), R::WriteAs<R::FixedRows<3>>(boundaryMomentumFlux), R::WriteAs<R::FixedRows<1>>(pin->faceStatus), R::WriteAs<R::FixedRows<1>>(pin->fallbackCounts), R::ReadAs<R::FixedRows<1>>(faces));
        R::SubmitArrays(engine, stream, [pin, low = controls.produceLow](R::ExecutionContext context, auto highFaces, auto lowFaces, auto faceStatus, auto high, auto lowValues, auto status, auto cells)
                        { CUDA::GatherFluxCells(context, *pin->geometry, highFaces, lowFaces, faceStatus, high, lowValues, status, cells, low); }, R::ReadAs<R::TableRows>(faceFlux), R::ReadAs<R::TableRows>(lowFace), R::ReadAs<R::FixedRows<1>>(pin->faceStatus), R::WriteAs<R::TableRows>(output.CoupledField().Owned()), R::WriteAs<R::TableRows>(lowOutput), R::WriteAs<R::FixedRows<1>>(pin->cellStatus), R::ReadAs<R::FixedRows<1>>(cells));
        R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto status, auto partial, auto result)
                        { CUDA::ReduceNumericalStatus(context, status, result.ConstView(), partial, result); }, R::ReadAs<R::FixedRows<1>>(pin->cellStatus), R::WriteAs<R::FixedRows<1>>(pin->partial), R::ReadWriteAs<R::FixedRows<1>>(pin->result));
        pin->submittedRegions |= mask;
#else
        DNDS_check_throw_info(false, "CUDA flux execution is not built");
#endif
    }
}
