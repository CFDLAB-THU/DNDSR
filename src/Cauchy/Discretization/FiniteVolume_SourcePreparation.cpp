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
                                                 const EquationSystem &equation, PointEvaluator &evaluator, bool quadrature)
        {
            // The caller already holds the mixture lease. Reuse it rather than
            // reserving another bounded-pool worker and risking self-starvation.
            if (equation.HasMixture() || OMP::DistributedThreads() == 1)
                return quadrature ? CPU::AddQuadratureSourceJacobian(request, equation, evaluator)
                                  : CPU::AssembleCenterSourceJacobian(request, equation, evaluator);
            auto workers = MakeCPUPointWorkers<CPU::SourceWorkerScratch>(equation, std::nullopt,
                                                                         {request.ransFirst >= 0, false, bool(request.linearization), request.chemical, request.chemical}, request.components);
            const SpanView<CPU::SourceWorkerScratch> slots{workers.data(), index(workers.size())};
            return quadrature ? CPU::AddQuadratureSourceJacobian(request, equation, slots)
                              : CPU::AssembleCenterSourceJacobian(request, equation, slots);
        }
    }

    void FiniteVolume::Workspace::SampleSourceState(const EquationDofs &state, real time)
    {
        const PhaseTime timing(*this, SamplingPhase);
        if (sourceOffsets.RowCount() == 0)
            return;
        // P0 face convection/viscosity does not disable these derivatives.
        // Direct-FV sources consume physical gradients without a basis.
        if (ransFirst >= 0 && owner->settings.numerics.reconstruction != "variational")
            RefreshDirectGradients(state, false, time);
        if (!sourceSampling)
            return;
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            if (ransFirst < 0)
                sourceSampling->Evaluate(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), sourceValues[iField]);
            else
            {
                CFV::PolynomialSampleOutputs outputs;
                outputs.values = &sourceValues[iField];
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    outputs.gradients[d] = &sourceGradients[iField][d];
                sourceSampling->EvaluateValueGradient(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), outputs);
            }
        }
    }
    /// Add the approximate source matrix to caller-owned output.
    /// The default is one mean point per cell; rhsQuadrature selects the
    /// explicit same-quadrature source linearization.
    /// The controller retains current means/coefficients and ready source
    /// gradients. Optional linearization outputs consume this same mean
    /// property evaluation; false means no samples existed and no outputs
    /// were filled. This entry neither samples RHS quadrature nor publishes
    /// source freshness; the production wrapper validates its provenance.
    bool FiniteVolume::Workspace::AddMeanSourceJacobian(const EquationDofs &state, const R::Array<real>::WriteLease &diagonal,
                                                        DiagonalStorage storage, real scale, PointEvaluator &evaluator,
                                                        OptionalRef<const MeanLinearizationOutput> linearization, bool rowMajorBlocks) const
    {
        if (sourceOffsets.RowCount() == 0)
            return false;
        // Resolve the synchronous nullable borrow once; property sampling
        // below must not pay a checked OptionalRef dereference per cell.
        // The default approximate source matrix is a one-point,
        // mean-state operator. The opt-in quadrature path is a separate
        // branch selected by the explicit implicit source mode.
        // RANS still consumes the current polynomial's center gradients.
        std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> gradientReads;
        // Direct gradients already contain every direction in one physical
        // field. Retain that allocation's read lease and select the direction
        // through indexing, rather than copying whole fields into scratch.
        // Polynomial center sampling keeps its separate per-direction arrays.
        using CPU::SourceGradientView;
        std::vector<std::array<SourceGradientView, 3>> gradients;
        if (ransFirst >= 0)
        {
            gradientReads.resize(size_t(state.PhysicalFieldNumber()));
            gradients.resize(size_t(state.PhysicalFieldNumber()));
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                if (owner->settings.numerics.reconstruction != "variational")
                {
                    gradientReads[iField][0].emplace(directGradients[iField].Owned().Read());
                    const auto view = gradientReads[iField][0]->View<R::UniformTableRows>();
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        gradients[iField][d] = {view, owner->metrics.Dimension(), d};
                    continue;
                }
                CFV::PolynomialSampleOutputs outputs;
                // Workspace identity is const; shallow mutable handles
                // retain its existing scratch allocations for the write.
                auto scratch = sourceJacobianGradients[iField];
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    outputs.gradients[d] = &scratch[d];
                // One center, gradients only. The sampling facility owns
                // bounded layout specialization and generic fallback; no
                // unused sampled-value field is allocated for the matrix.
                sourceJacobianSampling->EvaluateValueGradient(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), outputs);
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                {
                    gradientReads[iField][d].emplace(sourceJacobianGradients[iField][d].Read());
                    gradients[iField][d] = {gradientReads[iField][d]->View<R::UniformTableRows>(), 1, 0};
                }
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
        request.cells = owner->metrics.NumCell();
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
        RequireNumericalSuccess(PrepareSourceJacobianCPU(request, owner->equation, evaluator, false), "mean source Jacobian");
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
                                           lowWrite ? std::make_optional(lowOutput) : std::nullopt);
        }
        if (sourceOffsets.RowCount() == 0)
            return;
        SampleSourceState(state, time);
        if (!result)
            return;
        std::vector<R::Array<real>::ReadLease> sources;
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            sources.push_back((sourceSampling ? sourceValues[iField] : state.PhysicalField(iField).Owned()).Read());
        std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> derivatives(ransFirst >= 0 ? sources.size() : 0);
        for (index iField = 0; iField < index(derivatives.size()); ++iField)
            if (!sourceSampling)
                derivatives[iField][0].emplace(directGradients[iField].Owned().Read());
            else
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    derivatives[iField][d].emplace(sourceGradients[iField][d].Read());
        // Resolve physical layouts once, outside the quadrature loop.
        using SampleView = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        std::vector<SampleView> samples;
        using CPU::SourceGradientView;
        std::vector<std::array<SourceGradientView, 3>> gradients(derivatives.size());
        for (const auto &read : sources)
            samples.push_back(read.View<R::UniformTableRows>());
        for (index iField = 0; iField < index(derivatives.size()); ++iField)
            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                gradients[iField][d] = !sourceSampling
                                           ? SourceGradientView{derivatives[iField][0]->View<R::UniformTableRows>(), owner->metrics.Dimension(), d}
                                           : SourceGradientView{derivatives[iField][d]->View<R::UniformTableRows>(), 1, 0};
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
        CPU::SourceKernelRequest request;
        request.offsets = offsetRead.View<R::FixedRows<1>>();
        request.cells = owner->metrics.NumCell();
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
