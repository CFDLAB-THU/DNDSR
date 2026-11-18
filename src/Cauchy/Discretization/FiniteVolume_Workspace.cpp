#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;

    std::vector<FiniteVolume::Workspace::Field> FiniteVolume::Workspace::BuildFields(const Data &data, const EquationDofs &state, bool meanOnly)
    {
        std::vector<Field> result;
        result.reserve(size_t(state.PhysicalFieldNumber()));
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto &physical = state.PhysicalField(iField);
            std::vector<index> vectors;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto location = state.LocateBlock(iBlock);
                if (location.physicalField == iField)
                    for (const auto first : state.Schema().blocks[iBlock].vectorStarts)
                        vectors.push_back(location.firstComponent + first);
            }
            std::sort(vectors.begin(), vectors.end());
            Field field;
            field.vectorStarts = vectors;
            field.boundaryValues = R::Array<real>(R::RowLayout::Dense(
                                                      data.BoundaryValueRows(), index(physical.Width())),
                                                  data.metrics.Device());
            if (data.reconstruction)
            {
                field.reconstruction.emplace(data.reconstruction->BuildWorkspace(
                    index(physical.Width()),
                    FiniteVolumeDetail::FieldLayout(
                        physical.Owned().Layout(), data.metrics.NumCell(),
                        index(physical.Width()) * data.reconstruction->NRecDOF()),
                    vectors));
                const auto name = "Cauchy/reconstruction/workspace/" + state.Fields().Name(iField);
                field.reconstruction->temporary.Owned().SetName(name + "/temporary/owned", DNDS_ALLOCATION_SITE());
                field.reconstruction->temporary.Ghost().SetName(name + "/temporary/ghost", DNDS_ALLOCATION_SITE());
                field.reconstruction->scratch.SetName(name + "/scratch", DNDS_ALLOCATION_SITE());
                field.reconstructionBoundary.emplace(
                    data.reconstruction->BuildBoundaryValues(index(physical.Width())));
                field.boundaryValues = field.reconstructionBoundary->values;
                field.boundaryValues.SetName(name + "/boundaryValues", DNDS_ALLOCATION_SITE());
                Algebra::Fill(field.reconstruction->temporary, 0.);
                if (data.reconstructionAMG && !meanOnly)
                    field.reconstructionAMG = data.reconstructionAMG->BuildWorkspace(index(physical.Width()));
            }
            // A mean-only forecast suppresses face reconstruction, not the
            // physical derivatives in the RANS source. Keep those gradients
            // and their frozen boundary values in the P0 workspace as well.
            const auto &numerics = data.settings.numerics;
            const bool meanSource = data.equation.HasRANS() &&
                                    (meanOnly || numerics.reconstruction == "constant" ||
                                     (numerics.reconstruction == "variational" && data.settings.implicit.sourceJacobianMode == "center"));
            if (meanSource || (!meanOnly && (numerics.reconstruction == "gradient" || data.limitedVRGeometry)))
            {
                field.gradient.emplace(data.metrics.BuildField(index(physical.Width()) * data.metrics.Dimension()));
                field.gradientBoundary.emplace(numerics.reconstruction == "variational"
                                                   ? data.gradient.BuildBoundaryValues(index(physical.Width()))
                                                   : field.boundaryValues);
            }
            if (field.gradient)
                field.gradientExecution.emplace(data.gradient.BuildWorkspace(index(physical.Width()), vectors,
                                                                             !meanOnly && data.settings.numerics.limiter == "WBAP"));
            result.push_back(std::move(field));
        }
        return result;
    }
    void FiniteVolume::Workspace::BuildSourceSampling()
    {
        if (!owner->quadrature.source)
            return;
        // Geometry is frozen once by the plan before placement; workspaces
        // share its immutable handles. No host mesh access
        // or reconstruction setup is required for a CUDA workspace.
        const auto &prepared = *owner->quadrature.source;
        sourceOffsets = prepared.offsets;
        sourceWeights = prepared.weights;
        if (!meanOnly)
            sourceSampling = prepared.sampling;
    }
    EquationDofs FiniteVolume::Workspace::BindBoundary(const EquationDofs &state, const std::vector<Field> &fields, index rows)
    {
        const Algebra::DofSpace space(size_t(rows), 0, "equation boundary quadrature");
        Algebra::DofBundle bundle(space);
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            bundle.Add(state.Fields().Name(iField), CFV::FVField(space, fields[iField].boundaryValues,
                                                                 R::Array<real>(R::RowLayout::Dense(0, state.PhysicalField(iField).Width()), state.Device())));
        return FiniteVolumeDetail::BindFieldSchema(state, std::move(bundle), 1);
    }
    FiniteVolume::Workspace::Workspace(std::shared_ptr<const Data> data, EquationDofs &state,
                                       std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, bool meanTrial, bool capture)
        : owner(std::move(data)), meanOnly(meanTrial), captureBoundaryFlux(capture), componentNumber(state.ComponentNumber()), physicalFieldNumber(state.PhysicalFieldNumber()),
          fields(BuildFields(*owner, state, meanOnly)),
          boundary(BindBoundary(state, fields, owner->BoundaryValueRows())),
          thermal(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                  owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                  pool ? pool->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}),
          diagnostics(owner->forceZones),
          workers(std::move(pool)),
          boundaryMomentumFlux(R::RowLayout::Dense(owner->BoundaryValueRows(), 3), owner->metrics.Device()),
          faceFlux(R::RowLayout::Dense(owner->stencil.stencilFaces.RowCount(), state.ComponentNumber()), owner->metrics.Device()),
          chemical(owner->settings.equation.source == "chemical")
    {
        index first = 0;
        std::vector<index> blockOffsets, vectorOffsets;
        std::vector<index> physicalFields(size_t(state.ComponentNumber()), -1),
            physicalComponents(size_t(state.ComponentNumber()), -1);
        for (const auto &block : state.Schema().blocks)
        {
            blockOffsets.push_back(first);
            const auto location = state.LocateBlock(index(blockOffsets.size() - 1));
            for (index j = 0; j < location.componentNumber; ++j)
            {
                physicalFields[size_t(first + j)] = location.physicalField;
                physicalComponents[size_t(first + j)] = location.firstComponent + j;
            }
            if (block.name == "flow")
                flowFirst = first;
            if (block.name == "species")
                speciesFirst = first;
            if (block.name == "rans")
                ransFirst = first;
            for (const auto vector : block.vectorStarts)
                vectorOffsets.push_back(first + vector);
            first += index(block.components.size());
        }
        logicalFirst = R::MetadataBuffer<index>(blockOffsets);
        vectorStarts = R::MetadataBuffer<index>(vectorOffsets);
        logicalField = R::MetadataBuffer<index>(physicalFields);
        logicalComponent = R::MetadataBuffer<index>(physicalComponents);
        if (!meanOnly && owner->settings.numerics.positivity)
        {
            if (owner->settings.numerics.reconstruction == "variational")
            {
                // Modal PP owns bounded point scratch and compact decisions.
                // It borrows the supplied coefficient fields at submission;
                // no dormant coefficient prototype or halo binding is needed.
                positivity = std::make_shared<PositivityWorkspace>(owner, state, workers);
            }
            // Conservative explicit flux blending is currently the
            // established pure-flow policy. Extended RANS/species stages
            // reject and retry their raw update instead of constructing a
            // false independently blended scalar system.
            if (!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1)
            {
                lowOrderResidual.emplace(EquationDofOperations::NewVector(state));
                lowOrderFaceFlux.emplace(R::RowLayout::Dense(owner->stencil.stencilFaces.RowCount(), state.ComponentNumber()), owner->metrics.Device());
            }
        }
        if (!meanOnly && owner->limitedVRGeometry)
        {
            DNDS_check_throw_info(!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                                      state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                  "LVR requires one coupled perfect-gas flow5 field");
            DNDS_check_throw_info(fields.front().reconstruction,
                                  "LVR requires a variational reconstruction workspace");
            limitedVR = std::make_unique<LimitedVRWorkspace>(*owner, *fields.front().reconstruction);
        }
        nativeDirectGradient = !meanOnly && owner->settings.numerics.reconstruction == "gradient" &&
                               (owner->settings.numerics.limiter == "none" ||
                                owner->settings.numerics.limiter == "Barth" ||
                                owner->settings.numerics.limiter == "WBAP");
        if (nativeDirectGradient)
        {
            // These are CPU row-local temporaries. CUDA leaves use local
            // fixed-width values and per-cell result/status arrays instead.
            if (owner->metrics.Device() == R::Device::Host())
            {
                cpuGradientWorkers = OMP::DistributedThreads();
                directLimiterScratch = R::Array<real>(R::RowLayout::Dense(6 * cpuGradientWorkers, componentNumber));
                if (owner->settings.numerics.positivity)
                    directPPSamples = R::Array<real>(R::RowLayout::Dense(owner->maximumGradientPoints * cpuGradientWorkers, componentNumber));
            }
        }
        if (!meanOnly && (owner->settings.numerics.limiter == "WBAP" || limitedVR) && owner->settings.numerics.characteristicLimiter)
        {
            const auto flow = state.LocateBlock("flow");
            DNDS_check_throw_info(flow.componentNumber == 5, "characteristic limiter requires one five-component flow block");
            flowCharacteristicField = flow.physicalField;
            if (nativeDirectGradient)
                gradientFlowCharacteristic.emplace(owner->gradient.BuildQueuedFaceTransform(flow.firstComponent, 5));
            else if (owner->reconstruction)
                flowCharacteristic.emplace(owner->reconstruction->BuildQueuedFaceTransform(flow.firstComponent, 5));
        }
        if (owner->settings.numerics.reconstruction != "variational" || (meanOnly && owner->equation.HasRANS()))
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                if (owner->settings.numerics.reconstruction == "constant" &&
                    !fields[iField].gradient)
                    continue;
                if (nativeDirectGradient)
                {
                    DNDS_check_throw_info(fields[iField].gradient,
                                          "explicit P1 has no prepared physical-gradient publication");
                    if (owner->settings.numerics.limiter == "none" &&
                        !owner->settings.numerics.positivity)
                        directGradients.push_back(*fields[iField].gradient);
                    else
                    {
                        directGradients.push_back(owner->metrics.BuildField(
                            FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(),
                                                            owner->metrics.NumCell(),
                                                            state.PhysicalField(iField).Width() * owner->metrics.Dimension()),
                            owner->metrics.Device()));
                        if (owner->settings.numerics.limiter == "WBAP")
                            directGradientInputHalo.push_back(owner->BindHalo(
                                *fields[iField].gradient, "cauchy-direct-gradient-input"));
                    }
                    directGradientHalo.push_back(owner->BindHalo(
                        directGradients.back(), "cauchy-direct-gradient"));
                    if (owner->equation.Viscous() &&
                        owner->settings.numerics.viscousGradient == "polynomial" &&
                        !owner->settings.numerics.viscousLimited &&
                        (owner->settings.numerics.limiter != "none" ||
                         owner->settings.numerics.positivity))
                    {
                        directViscousGradients.push_back(*fields[iField].gradient);
                        directViscousGradientHalo.push_back(owner->BindHalo(
                            directViscousGradients.back(),
                            "cauchy-direct-unlimited-viscous-gradient"));
                    }
                }
                else if ((meanOnly || owner->settings.numerics.reconstruction == "constant") && fields[iField].gradient)
                    // P0/mean-only RANS sources use this same unlimited
                    // physical gradient. No limiter writes a second version,
                    // so retain the field instead of allocating/copying it.
                    directGradients.push_back(*fields[iField].gradient);
                else
                    directGradients.push_back(owner->metrics.BuildField(
                        FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owner->metrics.NumCell(),
                                                        state.PhysicalField(iField).Width() * owner->metrics.Dimension()),
                        owner->metrics.Device()));
                if (!meanOnly && owner->equation.Viscous() && owner->settings.numerics.viscousGradient == "polynomial" &&
                    !owner->settings.numerics.viscousLimited && !nativeDirectGradient && owner->reconstruction)
                    directViscousGradients.push_back(owner->metrics.BuildField(
                        FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owner->metrics.NumCell(),
                                                        state.PhysicalField(iField).Width() * owner->metrics.Dimension()),
                        owner->metrics.Device()));
            }
        }
        if (!meanOnly && owner->equation.Viscous() && owner->settings.numerics.viscousGradient == "primitive")
        {
            DNDS_check_throw_info(!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                                      state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                  "primitive viscosity requires one perfect-gas flow5 field");
            primitive.emplace(owner->metrics.BuildField(5));
            primitiveGradient.emplace(owner->metrics.BuildField(5 * owner->metrics.Dimension()));
            primitiveBoundary.emplace(owner->gradient.BuildBoundaryValues(5));
            primitiveGradientExecution.emplace(owner->gradient.BuildWorkspace(5, {1}));
            primitiveGradientHalo.emplace(owner->BindHalo(*primitiveGradient, "cauchy-primitive-gradient"));
        }
        BuildSourceSampling();
        BuildPreparationStatus();
        BuildCharacteristicStorage();
        BuildDeviceFlux();
        boundaryMomentumFlux.SetName("Cauchy/workspace/boundaryMomentumFlux", DNDS_ALLOCATION_SITE());
        faceFlux.SetName("Cauchy/workspace/faceFlux", DNDS_ALLOCATION_SITE());
        if (lowOrderFaceFlux)
            lowOrderFaceFlux->SetName("Cauchy/workspace/lowOrderFaceFlux", DNDS_ALLOCATION_SITE());
        if (nativeDirectGradient || ransFirst >= 0)
            directGradientPublications.resize(directGradients.size());
    }
    void FiniteVolume::Workspace::BindHalos(EquationDofs &state)
    {
        (void)state;
    }
}
