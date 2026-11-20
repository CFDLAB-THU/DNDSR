#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/PointWorkerContexts.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;

    namespace
    {
        NumericalStatus PrepareSourceJacobianCPU(const CPU::SourceKernelRequest &request,
                                                 const EquationSystem &equation, PointEvaluator &evaluator)
        {
            // The caller already holds the mixture lease. Reuse it rather than
            // reserving another bounded-pool worker and risking self-starvation.
            if (equation.HasMixture() || OMP::DistributedThreads() == 1)
                return CPU::AssembleCenterSourceJacobian(request, equation, evaluator);
            auto workers = MakeCPUPointWorkers<CPU::SourceWorkerScratch>(equation, std::nullopt,
                                                                         {request.ransFirst >= 0, false, bool(request.linearization), request.chemical, request.chemical}, request.components);
            const SpanView<CPU::SourceWorkerScratch> slots{workers.data(), index(workers.size())};
            return CPU::AssembleCenterSourceJacobian(request, equation, slots);
        }
    }

    /// Assemble the centered source approximation from means and their common
    /// unlimited physical gradient, independent of the modal or limited RHS.
    /// Compressible mean properties
    /// can also be published for the flux matrix in this pass.
    bool FiniteVolume::Workspace::AddMeanSourceJacobian(const EquationDofs &state, const R::Array<real>::WriteLease &diagonal,
                                                        DiagonalStorage storage, real scale, PointEvaluator &evaluator,
                                                        OptionalRef<const MeanLinearizationOutput> linearization, bool rowMajorBlocks) const
    {
        if (sourceOffsets.RowCount() == 0)
            return false;
        std::vector<std::optional<R::Array<real>::ReadLease>> gradientReads;
        using CPU::SourceGradientView;
        std::vector<std::array<SourceGradientView, 3>> gradients;
        if (ransFirst >= 0)
        {
            DNDS_check_throw_info(meanGradients, "centered source Jacobian has no mean gradient");
            gradientReads.resize(size_t(state.PhysicalFieldNumber()));
            gradients.resize(size_t(state.PhysicalFieldNumber()));
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                const auto &gradient = meanGradients->fields[iField];
                gradientReads[iField].emplace(gradient.Owned().Read());
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    gradients[iField][d] = {gradientReads[iField]->View<R::UniformTableRows>(), owner->metrics.Dimension(), d};
            }
        }
        const auto means = state.ReadOwned();
        const auto blocks = BlockViews(state, means);
        const auto geometry = owner->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        std::optional<WallDistanceRead> wallRead;
        std::optional<R::Array<real>::ReadLease> lengthRead;
        if (ransFirst >= 0)
        {
            wallRead.emplace(owner->wallDistance->AcquireRead());
            lengthRead.emplace(owner->ransCellLength.Read());
        }
        const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
        const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>() : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
        const auto matrix = diagonal.View<R::UniformTableRows>();
        const index nv = state.ComponentNumber();
        const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
        std::vector<EquationBlockLocation> locations;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            locations.push_back(state.LocateBlock(iBlock));
        const auto offsetRead = sourceOffsets.Read();
        CPU::SourceKernelRequest request;
        request.offsets = offsetRead.View<R::FixedRows<1>>();
        request.rows = R::RowSelectionShape::AllRows(owner->metrics.NumCell());
        request.components = state.ComponentNumber();
        request.dimension = owner->metrics.Dimension();
        request.flowFirst = flowFirst;
        request.speciesFirst = speciesFirst;
        request.ransFirst = ransFirst;
        request.independentSpecies = independent;
        request.chemical = chemical;
        request.walls = walls;
        request.lengths = lengths;
        const R::MetadataBuffer<EquationBlockLocation> locationMetadata(locations);
        request.locations = locationMetadata.HostView();
        request.logicalFirst = logicalFirst.HostView();
        const R::MetadataBuffer<std::array<CPU::SourceGradientView, 3>> gradientMetadata(gradients);
        request.gradients = gradientMetadata.HostView();
        request.metric = metric;
        request.matrix = matrix;
        request.rowMajorBlocks = rowMajorBlocks;
        request.scale = scale;
        request.fullBlock = storage == DiagonalStorage::Block;
        if (linearization)
            request.linearization = *linearization;
        const R::MetadataBuffer<EquationBlockReadView> blockMetadata(blocks);
        request.blocks = blockMetadata.HostView();
        RequireNumericalSuccess(PrepareSourceJacobianCPU(request, owner->equation, evaluator), "mean source Jacobian");
        return true;
    }

    void FiniteVolume::Workspace::PrepareVolumeSource(const EquationDofs &state,
                                                      OptionalRef<EquationDofs> result,
                                                      PointEvaluator &evaluator, real time,
                                                      OptionalRef<EquationDofs> lowOrder)
    {
        const PhaseTime timing(*this, SourcePhase);
        // Kinematic and constitutive sources share this additive entrance;
        // face-only calls never include either kind of source.
        const auto frame = owner->motion.At(time);
        if (result && frame.moving)
        {
            const auto read = state.ReadOwned();
            auto write = result->WriteOwned();
            std::optional<EquationDofs::WriteScope> lowWrite;
            if (lowOrder)
            {
                state.RequireCompatible(*lowOrder);
                EquationDofOperations::RequireDistinct(*result, *lowOrder);
                lowWrite.emplace(lowOrder->WriteOwned());
            }
            const auto flow = read.Block("flow");
            const auto output = write.Block("flow");
            const auto lowOutput = lowWrite ? lowWrite->Block("flow") : EquationBlockArrayView<real>{};
            const auto geometry = owner->metrics.AcquireRead();
            const auto metric = geometry.HostView();
            CPU::AddEquationRotationSource(metric, frame, flow, output,
                                           lowWrite ? std::make_optional(lowOutput) : std::nullopt, owner->equation.IsACM() ? 0 : 1, R::RowSelectionShape::AllRows(owner->metrics.NumCell()));
        }
        if (sourceOffsets.RowCount() == 0)
            return;
        if (!sourceSampling && ransFirst >= 0)
            RefreshDirectGradients(state, false, time);
        if (!result)
            return;
        CPU::SourceKernelRequest request;
        std::vector<R::Array<real>::ReadLease> sources, coefficientReads, gradientReads, basisReads;
        std::vector<RHSScalarView> samples, coefficientViews;
        std::vector<std::array<CPU::SourceGradientView, 3>> gradients;
        for (index f = 0; f < state.PhysicalFieldNumber(); ++f)
        {
            sources.push_back(state.PhysicalField(f).Owned().Read());
            samples.push_back(sources.back().View<R::UniformTableRows>());
            if (sourceSampling)
            {
                coefficientReads.push_back(FluxCoefficients().PhysicalField(f).Owned().Read());
                coefficientViews.push_back(coefficientReads.back().View<R::UniformTableRows>());
            }
            else if (ransFirst >= 0)
            {
                gradientReads.push_back(directGradients[f].Owned().Read());
                gradients.emplace_back();
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    gradients.back()[d] = {gradientReads.back().View<R::UniformTableRows>(), owner->metrics.Dimension(), d};
            }
        }
        if (sourceSampling)
        {
            sourceSampling->Validate();
            auto bind = [&](const R::Array<real> &array)
            {
                basisReads.push_back(array.Read());
                return basisReads.back().View<R::UniformTableRows>();
            };
            request.basis = {bind(sourceSampling->PointStorage()), bind(sourceSampling->BasisStorage()),
                             bind(sourceSampling->MomentStorage()), bind(sourceSampling->CellGeometryStorage()),
                             sourceSampling->BasisParameters(), sourceSampling->CachedDerivativeNumber()};
            request.coefficients = {coefficientViews.data(), index(coefficientViews.size())};
        }
        std::optional<WallDistanceRead> wallRead;
        if (owner->wallDistance)
            wallRead.emplace(owner->wallDistance->AcquireRead());
        const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
        const auto weightRead = sourceWeights.Read();
        const auto weights = weightRead.View<R::FixedRows<1>>();
        std::optional<R::Array<real>::ReadLease> lengthRead;
        if (ransFirst >= 0)
            lengthRead.emplace(owner->ransCellLength.Read());
        const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>() : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
        std::optional<EquationDofs::WriteScope> output;
        if (result)
            output.emplace(result->WriteOwned());
        const auto speciesOutput = result && speciesFirst >= 0 ? output->Block("species") : EquationBlockArrayView<real>{};
        const auto ransOutput = result && ransFirst >= 0 ? output->Block("rans") : EquationBlockArrayView<real>{};
        const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
        const auto mean = state.ReadOwned();
        const auto ransMean = ransFirst >= 0 ? mean.Block("rans") : EquationBlockReadView{};
        std::vector<EquationBlockLocation> locations;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            locations.push_back(state.LocateBlock(iBlock));
        const auto offsetRead = sourceOffsets.Read();
        request.offsets = offsetRead.View<R::FixedRows<1>>();
        request.rows = R::RowSelectionShape::AllRows(owner->metrics.NumCell());
        request.components = state.ComponentNumber();
        request.dimension = owner->metrics.Dimension();
        request.flowFirst = flowFirst;
        request.speciesFirst = speciesFirst;
        request.ransFirst = ransFirst;
        request.independentSpecies = independent;
        request.cellInputs = !sourceSampling;
        request.chemical = chemical;
        request.walls = walls;
        request.lengths = lengths;
        const R::MetadataBuffer<EquationBlockLocation> locationMetadata(locations);
        request.locations = locationMetadata.HostView();
        request.logicalFirst = logicalFirst.HostView();
        const R::MetadataBuffer<std::array<CPU::SourceGradientView, 3>> gradientMetadata(gradients);
        request.gradients = gradientMetadata.HostView();
        request.weights = weights;
        request.ransMean = ransMean;
        const R::MetadataBuffer<RHSScalarView> sampleMetadata(samples);
        request.samples = sampleMetadata.HostView();
        request.speciesOutput = speciesOutput;
        request.ransOutput = ransOutput;
        RequireNumericalSuccess(CPU::EvaluateVolumeSource(request, owner->equation, evaluator), "volume source");
    }
}
