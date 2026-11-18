#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;

    void FiniteVolume::Workspace::ApplyFixedFlowRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                                                    real time, OptionalRef<const AuxiliaryValues> auxiliary,
                                                    OptionalRef<R::Array<real>> faceOutput,
                                                    RHSScalarView incoming)
    {
        DNDS_assert_info(fixedFlowRHS && state.PhysicalFieldNumber() == 1 && result.PhysicalFieldNumber() == 1 &&
                             state.ComponentNumber() == 5 && result.ComponentNumber() == 5,
                         "fixed-flow RHS leaf requires one flow5 physical field");
        const bool ghosts = region != CFV::ReconstructionRegion::Core;
        RHSAccessScope scope(ghosts);
        auto &input = scope.input;
        input.mean = scope.ReadFlowField(state.PhysicalField(0));
        input.includeViscosity = !meanOnly;
        input.meanOnly = meanOnly;
        if (owner->settings.numerics.reconstruction == "variational")
            input.coefficients = scope.ReadField(FluxCoefficients().PhysicalField(0));
        else if (!meanOnly && !owner->settings.numerics.firstOrderConvection &&
                 owner->settings.numerics.reconstruction == "gradient")
        {
            DNDS_assert_info(directGradients.size() == 1, "fixed-flow direct gradient is unavailable");
            input.conservativeGradient = scope.ReadField(directGradients.front());
        }
        if (!owner->settings.numerics.viscousLimited &&
            (owner->settings.numerics.limiter == "WBAP" || limitedVR || owner->settings.numerics.positivity))
        {
            if (owner->settings.numerics.reconstruction == "variational")
                input.viscousCoefficients = scope.ReadField(Coefficients().PhysicalField(0));
            else if (!directViscousGradients.empty())
                input.viscousConservativeGradient = scope.ReadField(directViscousGradients.front());
        }
        if (primitiveGradient)
            input.primitiveGradient = scope.ReadField(*primitiveGradient);
        if (auxiliary)
        {
            const auto &values = *auxiliary;
            if (values.meanThermodynamics)
            {
                scope.thermodynamics.emplace(values.meanThermodynamics->Read());
                input.auxiliary.meanThermodynamics.emplace(scope.thermodynamics->View<R::UniformTableRows>());
            }
            else if (values.derivedOwned)
            {
                const auto typed = EquationDofs::Coupled(owner->equation.Schema(), state.PhysicalField(0));
                scope.derivedOwned.emplace(values.derivedOwned->ReadCurrent(typed, owner->equation));
                if (ghosts && values.derivedGhost)
                    scope.derivedGhost.emplace(values.derivedGhost->ReadCurrent(typed, owner->equation));
                input.auxiliary.meanThermodynamics.emplace(MeanThermodynamicView::Derived(
                    *scope.derivedOwned,
                    scope.derivedGhost
                        ? OptionalRef<const DerivedFields::ReadScope>(*scope.derivedGhost)
                        : OptionalRef<const DerivedFields::ReadScope>{}));
            }
            if (values.cellWaveJump)
                input.auxiliary.cellWaveJump = scope.ReadField(*values.cellWaveJump);
            if (values.cellPressureMinimum)
                input.auxiliary.cellPressureMinimum = scope.ReadField(*values.cellPressureMinimum);
        }
        if (incoming.data)
            input.boundary = incoming;
        scope.result.emplace(result.PhysicalField(0).Owned().Write());
        scope.output.residual = scope.result->View<RHSFlowRows>();
        if (lowOrderResidual)
        {
            scope.lowOrder.emplace(lowOrderResidual->PhysicalField(0).Owned().Write());
            scope.output.lowOrder = scope.lowOrder->View<RHSFlowRows>();
        }
        if (faceOutput || captureBoundaryFlux)
        {
            scope.face.emplace((faceOutput ? *faceOutput : faceFlux).Write());
            scope.output.faceFlux = scope.face->View<RHSFlowRows>();
            scope.output.boundaryFacesOnly = !faceOutput;
        }
        if (captureBoundaryFlux)
        {
            scope.boundaryMomentum.emplace(boundaryMomentumFlux.Write());
            scope.output.boundaryMomentum = scope.boundaryMomentum->View<R::FixedRows<3>>();
        }
        diagnostics.Merge(fixedFlowRHS->ApplyFluxBorrowedFatal(input, scope.output, region, time));
    }
    void FiniteVolume::Workspace::ApplyRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                                           PointEvaluator &evaluator, real time,
                                           OptionalRef<const AuxiliaryValues> auxiliary,
                                           OptionalRef<R::Array<real>> faceOutput,
                                           RHSScalarView incoming)
    {
        const PhaseTime timing(*this, RHSPhase);
        if (fixedFlowRHS)
        {
            ApplyFixedFlowRHS(state, result, region, time, auxiliary, faceOutput, incoming);
            return;
        }
        const auto mean = region != CFV::ReconstructionRegion::Core ? state.ReadWithGhost() : state.ReadOwned();
        auto output = result.WriteOwned();
        const EquationViewLayout<const real> owned(state, mean), remote(state, mean, true);
        const EquationViewLayout<real> destination(result, output);
        if (lowOrderResidual)
        {
            auto lowOutput = lowOrderResidual->WriteOwned();
            const EquationViewLayout<real> lowDestination(*lowOrderResidual, lowOutput);
            ApplyEquationRHS(state, region, evaluator, time, auxiliary, mean,
                             owned.Descriptor(), remote.Descriptor(), destination.Descriptor(),
                             lowDestination.Descriptor(), true, faceOutput, incoming);
            return;
        }
        // The disabled low-order view deliberately aliases the high-order
        // descriptor. ApplyRHSRows receives produceLow=false and never
        // reads or writes it, avoiding a dummy allocation in non-PP runs.
        ApplyEquationRHS(state, region, evaluator, time, auxiliary, mean,
                         owned.Descriptor(), remote.Descriptor(), destination.Descriptor(),
                         destination.Descriptor(), false, faceOutput, incoming);
    }
    void FiniteVolume::Workspace::ApplyEquationRHS(const EquationDofs &state, CFV::ReconstructionRegion region,
                                                   PointEvaluator &evaluator, real time,
                                                   OptionalRef<const AuxiliaryValues> auxiliary,
                                                   const EquationDofs::ReadScope &mean, EquationLayoutView<const real> owned, EquationLayoutView<const real> remote,
                                                   EquationLayoutView<real> destination, EquationLayoutView<real> lowDestination, bool produceLow,
                                                   OptionalRef<R::Array<real>> faceOutput, RHSScalarView incoming)
    {
        auto geometry = owner->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        std::optional<CFV::ReconstructionBasisRead> basisRead;
        const bool direct = owner->settings.numerics.reconstruction != "variational";
        if (!direct)
            basisRead.emplace(owner->reconstruction->AcquireBasisRead());
        const bool ghosts = region != CFV::ReconstructionRegion::Core;
        // Preparation owns writes; the point kernel holds only numeric
        // read views. These means are not reconstructed thermodynamics.
        std::optional<R::Array<real>::ReadLease> meanProperties;
        std::optional<RHSScalarView> preparedMean;
        if (auxiliary && auxiliary->meanThermodynamics)
        {
            meanProperties.emplace(auxiliary->meanThermodynamics->Read());
            preparedMean.emplace(meanProperties->View<R::UniformTableRows>());
        }
        std::array<std::optional<R::Array<real>::ReadLease>, 4> stencilReads;
        RHSFieldView waveJump, pressureMinimum;
        const auto readStencil = [&](const std::optional<CFV::FVField> &field, int slot, RHSFieldView &view)
        {
            if (!field)
                return;
            stencilReads[slot].emplace(field->Owned().Read());
            if (ghosts)
                stencilReads[slot + 1].emplace(field->Ghost().Read());
            view = {stencilReads[slot]->View<R::UniformTableRows>(),
                    ghosts ? stencilReads[slot + 1]->View<R::UniformTableRows>() : RHSScalarView{}};
        };
        if (auxiliary)
        {
            readStencil(auxiliary->cellWaveJump, 0, waveJump);
            readStencil(auxiliary->cellPressureMinimum, 2, pressureMinimum);
        }
        std::optional<EquationDofs::ReadScope> rec, viscousRec;
        R::MetadataBuffer<EquationBlockReadView> recBlocks, viscousRecBlocks;
        if (!direct)
        {
            rec.emplace(ghosts ? FluxCoefficients().ReadWithGhost() : FluxCoefficients().ReadOwned());
            recBlocks = BlockViews(FluxCoefficients(), *rec);
            if (owner->equation.Viscous() && !owner->settings.numerics.viscousLimited &&
                (owner->settings.numerics.limiter == "WBAP" || limitedVR || owner->settings.numerics.positivity))
            {
                viscousRec.emplace(ghosts ? Coefficients().ReadWithGhost() : Coefficients().ReadOwned());
                viscousRecBlocks = BlockViews(Coefficients(), *viscousRec);
            }
        }
        const auto meanBlocks = BlockViews(state, mean);
        std::vector<EquationBlockLocation> directLocations;
        for (index iBlock = 0; direct && iBlock < state.BlockNumber(); ++iBlock)
            directLocations.push_back(state.LocateBlock(iBlock));
        std::vector<R::Array<real>::ReadLease> directOwned, directGhost, directViscousOwned, directViscousGhost;
        std::vector<RHSFieldView> directViews, directViscousViews;
        if (!meanOnly && direct && owner->settings.numerics.reconstruction == "gradient" && !owner->settings.numerics.firstOrderConvection)
        {
            DNDS_check_throw_info(directGradients.size() == size_t(state.PhysicalFieldNumber()), "direct FV requires prepared physical gradients");
            for (const auto &gradient : directGradients)
            {
                directOwned.emplace_back(gradient.Owned().Read());
                if (ghosts)
                    directGhost.emplace_back(gradient.Ghost().Read());
                directViews.push_back({directOwned.back().View<R::UniformTableRows>(), ghosts ? directGhost.back().View<R::UniformTableRows>() : RHSScalarView{}});
            }
            for (const auto &gradient : directViscousGradients)
            {
                directViscousOwned.emplace_back(gradient.Owned().Read());
                if (ghosts)
                    directViscousGhost.emplace_back(gradient.Ghost().Read());
                directViscousViews.push_back({directViscousOwned.back().View<R::UniformTableRows>(),
                                              ghosts ? directViscousGhost.back().View<R::UniformTableRows>() : RHSScalarView{}});
            }
        }
        std::optional<R::Array<real>::ReadLease> primitiveOwned, primitiveGhost;
        RHSFieldView primitiveGradientValues;
        if (primitiveGradient)
        {
            primitiveOwned.emplace(primitiveGradient->Owned().Read());
            if (ghosts)
                primitiveGhost.emplace(primitiveGradient->Ghost().Read());
            primitiveGradientValues = {primitiveOwned->View<R::UniformTableRows>(),
                                       primitiveGhost ? primitiveGhost->View<R::UniformTableRows>() : RHSScalarView{}};
        }
        auto boundaryWrite = boundaryMomentumFlux.Write();
        const auto wallFlux = boundaryWrite.View<R::FixedRows<3>>();
        auto faceWrite = (faceOutput ? *faceOutput : faceFlux).Write();
        const auto faceValues = faceWrite.View<R::UniformTableRows>();
        std::optional<R::Array<real>::WriteLease> lowFaceWrite;
        decltype(faceWrite.View<R::UniformTableRows>()) lowFaceValues;
        if (produceLow)
        {
            DNDS_check_throw_info(lowOrderFaceFlux && lowOrderResidual &&
                                      state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                  "low-order equation flux requires the coupled flow5 PP profile");
            lowFaceWrite.emplace(lowOrderFaceFlux->Write());
            lowFaceValues = lowFaceWrite->View<R::UniformTableRows>();
        }

        const auto stencilRead = owner->stencil.Read();
        std::optional<WallDistanceRead> wallRead;
        if (owner->wallDistance)
            wallRead.emplace(owner->wallDistance->AcquireRead());
        CPU::EquationRHSRequest request;
        request.stencil = stencilRead.View();
        request.metric = metric;
        if (basisRead)
            request.basis = basisRead->EvaluationView();
        request.conditions = {owner->boundaryConditions.data(), index(owner->boundaryConditions.size())};
        const R::MetadataBuffer<EquationBlockReadView> meanMetadata(meanBlocks), recMetadata(recBlocks), viscousMetadata(viscousRecBlocks);
        const R::MetadataBuffer<EquationBlockLocation> locationMetadata(directLocations);
        const R::MetadataBuffer<RHSFieldView> directMetadata(directViews), viscousDirectMetadata(directViscousViews);
        request.meanBlocks = meanMetadata.HostView();
        request.recBlocks = recMetadata.HostView();
        request.viscousRecBlocks = viscousMetadata.HostView();
        request.directLocations = locationMetadata.HostView();
        request.directViews = directMetadata.HostView();
        request.directViscousViews = viscousDirectMetadata.HostView();
        request.logicalFirst = logicalFirst.HostView();
        request.vectorStarts = vectorStarts.HostView();
        if (owner->equation.HasRANS())
            request.ransMean = mean.Block("rans");
        request.preparedMean = preparedMean;
        request.waveJump = waveJump;
        request.pressureMinimum = pressureMinimum;
        request.primitiveGradientValues = primitiveGradientValues;
        request.incoming = incoming;
        request.faceValues = faceValues;
        request.lowFaceValues = lowFaceValues;
        request.wallFlux = wallFlux;
        request.captureBoundaryFlux = captureBoundaryFlux;
        request.genericFluxOracle = genericFluxOracle;
        request.wall = wallRead ? wallRead->HostView() : WallDistanceView{};
        request.thermal = thermal.View();
        request.frame = owner->motion.At(time);
        request.fluxPolicy = owner->fluxPolicy;
        request.ownedCells = owner->metrics.NumCell();
        request.width = state.ComponentNumber();
        request.flowFirst = flowFirst;
        request.speciesFirst = speciesFirst;
        request.ransFirst = ransFirst;
        request.dimension = owner->metrics.Dimension();
        request.modes = owner->reconstruction ? owner->reconstruction->NRecDOF() : 0;
        request.species = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() : 0;
        const auto &settings = owner->settings.numerics;
        request.direct = direct;
        request.linear = settings.reconstruction == "gradient";
        request.firstOrder = meanOnly || settings.firstOrderConvection;
        request.viscous = !meanOnly && owner->equation.Viscous();
        request.primitiveGradient = bool(primitiveGradient);
        request.hasWall = bool(wallRead);
        request.positivity = settings.positivity;
        request.faceFallback = settings.faceFallback;
        request.useStencil = settings.riemann.useStencil;
        request.preparedWaveJump = auxiliary && bool(auxiliary->cellWaveJump);
        request.preparedPressureMinimum = auxiliary && bool(auxiliary->cellPressureMinimum);
        request.produceLow = produceLow;
        request.time = time;
        const auto result = CPU::EvaluateEquationFlux(request, owner->equation, owner->boundaryPoint, owner->flux,
                                                      evaluator, owned, remote, destination, lowDestination, region);
        RequireNumericalSuccess(result.status, "equation RHS");
        const index slot = region == CFV::ReconstructionRegion::HaloDependent ? 1 : 0;
        faceFallbacks[slot] = result.value;
        if (region == CFV::ReconstructionRegion::All)
            faceFallbacks[1] = 0;
    }
}
