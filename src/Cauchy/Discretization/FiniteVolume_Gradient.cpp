#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeGradientKernels.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;

    void FiniteVolume::Workspace::RefreshDirectGradients(const EquationDofs &state, bool ghost,
                                                         real time)
    {
        if (owner->settings.numerics.reconstruction == "variational")
            return;
        // A mean forecast has no face polynomial. Only RANS source terms
        // consume its physical gradients, prepared directly from cell means.
        if (meanOnly && !owner->equation.HasRANS())
            return;
        if (owner->settings.numerics.reconstruction == "constant" && directGradients.empty())
            return;
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            if (nativeDirectGradient)
            {
                const auto &mean = ghost
                                       ? state.PhysicalField(iField).Ghost()
                                       : state.PhysicalField(iField).Owned();
                const auto &output = ghost
                                         ? directGradients[iField].Ghost()
                                         : directGradients[iField].Owned();
                auto &published = directGradientPublications[iField][ghost ? 1 : 0];
                published.Publish(*owner, mean, mean, output, incoming, time);
                continue;
            }
            const bool fromGradientField = (meanOnly || owner->settings.numerics.reconstruction == "constant") && fields[iField].gradient;
            // Constant face fluxes never read remote source gradients.
            if (fromGradientField && ghost)
                continue;
            // Pure P0 has no polynomial or physical-gradient publication.
            // Its directGradients slot is retained only for the common
            // workspace shape; no modal zero field is manufactured.
            if (owner->settings.numerics.reconstruction == "constant" && !fromGradientField)
                continue;
            const auto refresh = [&]
            {
                if (fromGradientField)
                {
                    // This CPU operation already executes on the caller's
                    // compute lane. Retain compatible reads and the distinct
                    // output write directly; do not create a nested runtime
                    // merely to bypass immediate algebra's idle-field check.
                    const auto &input = *fields[iField].gradient;
                    auto &output = directGradients[iField];
                    input.Validate();
                    output.Validate();
                    if (input.Owned().SharedStorage() == output.Owned().SharedStorage())
                        return; // P0/mean-only sources share their ready gradient.
                    DNDS_check_throw_info(input.Space().Same(output.Space()) &&
                                              input.Width() == output.Width() &&
                                              input.Device() == R::Device::Host() &&
                                              output.Device() == R::Device::Host(),
                                          "direct gradient copy requires matching host fields");
                    auto read = input.Owned().Read();
                    auto write = output.Owned().Write();
                    CPU::CopyRows(read.View<R::UniformTableRows>(),
                                  write.View<R::UniformTableRows>(), output.Width());
                }
                else
                {
                    DNDS_check_throw_info(owner->reconstruction,
                                          "modal gradient extraction requires variational reconstruction");
                    owner->gradient.ExtractLinearGradients(FluxCoefficients().PhysicalField(iField), *owner->reconstruction,
                                                           directGradients[iField], ghost);
                }
            };
            // Only RANS source sampling repeats this pass. Flow-only and
            // chemical-only paths retain their original preparation path.
            if (ransFirst < 0)
            {
                refresh();
                continue;
            }
            const auto &source = fromGradientField ? fields[iField].gradient->Owned()
                                 : ghost           ? FluxCoefficients().PhysicalField(iField).Ghost()
                                                   : FluxCoefficients().PhysicalField(iField).Owned();
            const auto &mean = ghost ? state.PhysicalField(iField).Ghost()
                                     : state.PhysicalField(iField).Owned();
            const auto &destination = ghost ? directGradients[iField].Ghost() : directGradients[iField].Owned();
            // Pin the input through both the check and a possible refresh:
            // an unfinished writer has not published its next generation.
            auto &published = directGradientPublications[iField][ghost ? 1 : 0];
            if (published.Current(*owner, mean, source, destination,
                                  incoming, time))
            {
                owner->metrics.Validate();
                continue;
            }
            // The independently callable source path still executes this
            // on a miss. P0 sources keep their physical gradients, and P1
            // extracts the final limiter/PP coefficients, never raw slopes.
            refresh();
            published.Publish(*owner, mean, source, destination, incoming,
                              time);
        }
        if (!nativeDirectGradient && !directViscousGradients.empty())
        {
            DNDS_check_throw_info(directViscousGradients.size() == fields.size(),
                                  "unlimited direct viscous-gradient fields are incomplete");
            for (index iField = 0; iField < index(fields.size()); ++iField)
            {
                DNDS_check_throw_info(owner->reconstruction,
                                      "modal viscous-gradient extraction requires variational reconstruction");
                owner->gradient.ExtractLinearGradients(Coefficients().PhysicalField(iField), *owner->reconstruction,
                                                       directViscousGradients[iField], ghost);
            }
        }
    }

    /// Cauchy-owned explicit-P1 componentwise Barth stage. CFV supplies
    /// the unlimited physical gradients; equation block layout, periodic
    /// vector components and conservative state semantics remain here.
    void FiniteVolume::Workspace::PrepareDirectGradientLimiter(const EquationDofs &state, bool barth)
    {
        DNDS_check_throw_info(nativeDirectGradient && directGradients.size() == fields.size(),
                              "direct-P1 limiter workspace is incomplete");
        const int dim = owner->metrics.Dimension();
        const auto meanScope = state.ReadWithGhost();
        const auto meanLayout = meanScope.ComponentLayout();
        const auto means = meanLayout.View();
        std::vector<R::Array<real>::ReadLease> sourceReads;
        std::vector<R::Array<real>::WriteLease> destinationWrites;
        sourceReads.reserve(fields.size());
        destinationWrites.reserve(fields.size());
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            DNDS_check_throw_info(fields[iField].gradient,
                                  "direct-P1 limiter has no unlimited gradient");
            DNDS_check_throw_info(
                fields[iField].gradient->Owned().SharedStorage() !=
                    directGradients[iField].Owned().SharedStorage(),
                "direct-P1 limiter needs distinct input and output fields");
            sourceReads.emplace_back(fields[iField].gradient->Owned().Read());
            destinationWrites.emplace_back(directGradients[iField].Owned().Write());
        }
        using ReadView = R::ArrayView<const real, R::AccessBoundsChecks,
                                      R::UniformTableRows>;
        using WriteView = R::ArrayView<real, R::AccessBoundsChecks,
                                       R::UniformTableRows>;
        std::vector<ReadView> source;
        std::vector<WriteView> destination;
        source.reserve(fields.size());
        destination.reserve(fields.size());
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            source.push_back(sourceReads[iField].View<R::UniformTableRows>());
            destination.push_back(destinationWrites[iField].View<R::UniformTableRows>());
        }
        const index components = state.ComponentNumber();
        DNDS_check_throw_info(
            directLimiterScratch.RowCount() == 6 * cpuGradientWorkers && directLimiterScratch.Layout().Width() == size_t(components),
            "direct-P1 limiter scratch extent differs");
        const auto stencilRead = owner->stencil.Read();
        const auto metricRead = owner->metrics.AcquireRead();
        auto scratchWrite = directLimiterScratch.Write();
        const R::MetadataBuffer<ReadView> sourceMetadata(source);
        const R::MetadataBuffer<WriteView> destinationMetadata(destination);
        CPU::LimitPhysicalGradients({stencilRead.View(), metricRead.HostView(), means,
                                     sourceMetadata.HostView(), destinationMetadata.HostView(), logicalField.HostView(),
                                     logicalComponent.HostView(), vectorStarts.HostView(), scratchWrite.View<R::TableRows>(),
                                     components, dim, barth});
    }

    /// Equation-aware P1 positivity stage. One beta scales the flow5 and
    /// transported-species gradients; RANS uses its independent scalar
    /// policy. This method deliberately lives in Cauchy rather than CFV.
    void FiniteVolume::Workspace::ApplyDirectGradientPositivity(const EquationDofs &state)
    {
        DNDS_check_throw_info(nativeDirectGradient &&
                                  owner->settings.GradientPositivity() &&
                                  owner->maximumGradientPoints >= 0,
                              "direct-P1 positivity workspace is incomplete");
        const int dim = owner->metrics.Dimension();
        const index components = state.ComponentNumber();
        const auto meanScope = state.ReadOwned();
        // DofComponentView borrows its descriptor table. Retain the
        // owning layout for the complete pointwise PP traversal.
        const auto meanLayout = meanScope.ComponentLayout();
        const auto means = meanLayout.View();
        std::vector<R::Array<real>::WriteLease> gradientWrites;
        gradientWrites.reserve(directGradients.size());
        for (auto &field : directGradients)
            gradientWrites.emplace_back(field.Owned().Write());
        using WriteView = R::ArrayView<real, R::AccessBoundsChecks,
                                       R::UniformTableRows>;
        std::vector<WriteView> gradients;
        for (auto &write : gradientWrites)
            gradients.push_back(write.View<R::UniformTableRows>());
        const auto bounds = thermal.View();
        const auto rans = owner->equation.HasRANS()
                              ? ResolveRANSReconstructionPolicy(owner->equation)
                              : RANSReconstructionView{};
        const auto stencilRead = owner->stencil.Read();
        auto sampleWrite = directPPSamples.Write();
        const R::MetadataBuffer<WriteView> gradientMetadata(gradients);
        const auto clippedPolicy = owner->settings.numerics.positivityClippedReferencePolicy == "backtracking"
                                       ? ReconstructionClippedReferencePolicy::Backtracking
                                       : ReconstructionClippedReferencePolicy::FallbackToBase;
        const auto result = RequireNumericalSuccess(CPU::LimitPositivePhysicalGradients({stencilRead.View(), means, gradientMetadata.HostView(),
                                                                                         logicalField.HostView(), logicalComponent.HostView(), sampleWrite.View<R::TableRows>(),
                                                                                         bounds, rans, components, ransFirst, dim,
                                                                                         real(owner->settings.numerics.positivityBetaPower), owner->settings.numerics.positivityBetaCutoff, clippedPolicy}),
                                                    "LimitPositivePhysicalGradients");
        directPPLimited = result.limited;
        directPPMinimum = result.minimum;
    }

    void FiniteVolume::Workspace::PreparePrimitiveGradient(const EquationDofs &state, CFV::ReconstructionRegion region)
    {
        if (!primitiveGradient)
            return;
        DNDS_assert_info(primitive && primitiveBoundary && state.PhysicalFieldNumber() == 1,
                         "primitive-gradient workspace is incomplete");
        const auto fill = [&](const R::Array<real> &source, R::Array<real> &destination)
        {
            const auto read = source.Read();
            auto write = destination.Write();
            const auto from = read.View<R::UniformTableRows>();
            const auto to = write.View<R::UniformTableRows>();
            CPU::FillPrimitiveMeans(from, to, owner->equation.Gas());
        };
        if (region != CFV::ReconstructionRegion::HaloDependent)
            fill(state.PhysicalField(0).Owned(), primitive->Owned());
        if (region != CFV::ReconstructionRegion::Core)
            fill(state.PhysicalField(0).Ghost(), primitive->Ghost());
        owner->gradient.Evaluate(*primitive, *primitiveBoundary, *primitiveGradient, {1}, region);
    }

}
