#include "Cauchy/Discretization/CUDA/FiniteVolumeACMFluxKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Inputs.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Reconstruction.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "Physics/Transport/NewtonianFourier.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA::Detail
{
    using ACMGeometryView = Runtime::ArrayView<const RHSGeometry, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    template <int C, int M, int D>
    struct ACMPolynomialFields
    {
        static constexpr int static_modes = M, static_dimension = D;
        static constexpr bool separate_roles = false;
        template <int Width>
        using Pair = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<Width, Width>>;
        Pair<C> mean;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<C, C>> incoming;
        Pair<C *(M ? M : D)> reconstruction, viscousData;
        CellCenterViscosityView centerViscosity;
    };

    template <int C, class Fields, class Output>
    __device__ NumericalStatus ACMFacePoint(const RHSGeometry &geometry, const RHSKernelControls &controls,
                                            const Fields &fields, const Output &output,
                                            index iFace, index first, index entry, std::array<real, C> &value)
    {
        const auto &physics = geometry.acm(0, 0);
        index unusedFallbacks = 0;
        const auto sample = geometry.FaceSample(iFace, entry);
        ACM::KernelPoint<C> left, right;
        ReconstructPoint<C>(geometry, physics, controls, fields, sample, false, left, unusedFallbacks);
        BoundaryKind kind = BoundaryKind::Interior;
        if (sample.other != UnInitIndex)
            ReconstructPoint<C>(geometry, physics, controls, fields, sample, true, right, unusedFallbacks);
        else
        {
            right = left;
            const auto &bc = geometry.acmBC(sample.condition, 0);
            kind = bc.kind;
            real omega = 0;
            if constexpr (C == 6)
                if (kind == BoundaryKind::NoSlipWall)
                {
                    const real distance = geometry.wallMean(sample.cell, 0);
                    if (!(distance > 0))
                        return NumericalStatus{NumericalError::InvalidInput}.At(sample.cell, entry);
                    omega = physics.flow.kinematicViscosity * Physics::RANS::kWallOmegaCoeff / sqr(distance);
                }
            if (kind == BoundaryKind::Interface)
            {
                if (sample.boundaryPoint < 0 || sample.boundaryPoint >= fields.incoming.RowCount())
                    return NumericalStatus{NumericalError::InvalidInput}.At(sample.cell, entry);
                for (int j = 0; j < C; ++j)
                    right.values[j] = fields.incoming(sample.boundaryPoint, j);
            }
            else if (!ACM::ExteriorAt(physics.flow, kind, left.values, bc, sample.normal, sample.ownPoint, controls.frame, right.values, C, omega))
                return NumericalStatus{NumericalError::UnsupportedRequest}.At(sample.cell, entry);
            // Keep the reflected SA scalar for the wall-normal viscous jump.
        }
        std::array<real, C> flux{};
        const auto grid = controls.frame.GridVelocity(sample.ownPoint);
        ACM::NumericalFlux(physics.flow, left.values, right.values, sample.normal, flux, C, ACM::NormalSpeed(grid, sample.normal));
        // Slip walls keep only the inviscid flux, even with a nonzero interior gradient.
        if (controls.viscous && kind != BoundaryKind::SlipWall)
        {
            ACM::KernelPoint<C> face;
            for (int j = 0; j < C; ++j)
            {
                face.values[j] = .5 * (left.values[j] + right.values[j]);
                for (int d = 0; d < 3; ++d)
                    face.gradients[j][d] = .5 * (left.gradients[j][d] + right.gradients[j][d]) +
                                           sample.normal[d] * (right.values[j] - left.values[j]) / sample.viscousDistance;
            }
            real viscosity = physics.flow.kinematicViscosity;
            if constexpr (C > 4)
            {
                const auto at = face.Turbulence(physics, geometry.wallFaces(sample.face, entry - first));
                const auto transport = Physics::RANS::TryTransport(physics.rans, at, fields.centerViscosity.AtFace(sample.cell, sample.other));
                if (!transport)
                    return transport.status.At(sample.cell, entry);
                const Eigen::Vector3d normal(sample.normal[0], sample.normal[1], sample.normal[2]);
                const auto diffusion = Physics::RANS::TryDiffusionFlux(physics.rans, at, normal, transport.value.viscosity);
                if (!diffusion)
                    return diffusion.status.At(sample.cell, entry);
                for (int j = 4; j < C; ++j)
                    flux[j] -= diffusion.value[j - 4];
                viscosity += transport.value.viscosity;
            }
            const auto traction = Physics::Transport::NewtonianTraction(
                [&](int j, int d)
                { return face.gradients[j][d]; }, sample.normal, viscosity);
            // Cauchy omits the former isotropic 2*rho*k/3 normal stress.
            for (int d = 0; d < 3; ++d)
                flux[d] -= traction[d];
        }
        const real weight = geometry.quadrature(entry, 6);
        if (output.wallForce.RowCount() && sample.boundaryPoint != UnInitIndex)
            for (int d = 0; d < 3; ++d)
                output.wallForce(sample.boundaryPoint, d) =
                    controls.captureBoundaryFlux || kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall ? weight * flux[d] : 0;
        for (int j = 0; j < C; ++j)
        {
            if (!std::isfinite(flux[j]))
                return NumericalStatus{NumericalError::InvalidState, j, sample.cell, entry};
            value[j] = -weight * flux[j];
        }
        return {};
    }

    template <int C, class Fields, class Output>
    __device__ NumericalStatus ACMFace(const RHSGeometry &geometry, const RHSKernelControls &controls,
                                       const Fields &fields, const Output &output, index iFace)
    {
        const auto anchor = geometry.FaceAnchor(iFace);
        const index first = geometry.faceOffsets(anchor.face, 0), end = geometry.faceOffsets(anchor.face + 1, 0);
        std::array<real, C> integrated{};
        for (index entry = first; entry < end; ++entry)
        {
            std::array<real, C> value{};
            const auto status = ACMFacePoint<C>(geometry, controls, fields, output, iFace, first, entry, value);
            if (!status)
                return status;
            for (int j = 0; j < C; ++j)
                integrated[j] += value[j];
        }
        for (int j = 0; j < C; ++j)
            output.high(iFace, j) = integrated[j];
        return {};
    }

    template <int C, ReconstructionForm Form, class Fields, class Output>
    __global__ void ACMFluxFaceThreads(ACMGeometryView geometry, RHSKernelControls controls, Fields fields,
                                       Output output, RHSIndices selected)
    {
        if constexpr (Form != ReconstructionForm::Generic)
        {
            controls.direct = Form != ReconstructionForm::Modal;
            controls.linear = Form == ReconstructionForm::DirectGradient;
            controls.firstOrder = Form == ReconstructionForm::MeanOnly;
        }

        selected.Visit([&](auto rowAt)
                       {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
        {
            const index iFace = rowAt(iRow);
            output.status(iFace, 0) = ACMFace<C>(geometry(0, 0), controls, fields, output, iFace);
            output.fallbackCounts(iFace, 0) = 0;
        } });
    }

    // The prototype uses the same complete-face packages as the compressible
    // mapping. Each lane evaluates one quadrature point; its segment leader
    // integrates in rule order, preserving the thread-per-face sum.
    template <int C, class Fields, class Output>
    __global__ void ACMFluxCTASegments(ACMGeometryView publication, RHSKernelControls controls, Fields fields,
                                       Output output, RHSFacePackageView prepared)
    {
        extern __shared__ real shared[];
        real *values = shared;
        static_assert(sizeof(NumericalStatus) == 2 * sizeof(index) + 2 * sizeof(int));
        // Publish scalar fields explicitly. Keep the status aggregate in
        // registers and avoid assigning it through a cast of the lane scratch.
        auto *rows = reinterpret_cast<index *>(values + C * blockDim.x);
        auto *points = rows + blockDim.x;
        auto *codes = reinterpret_cast<int *>(points + blockDim.x);
        auto *details = codes + blockDim.x;
        const auto &geometry = publication(0, 0);
        for (index iPackage = blockIdx.x; iPackage < prepared.packages.RowCount(); iPackage += gridDim.x)
        {
            const auto package = prepared.packages(iPackage, 0);
            RHSFaceSegment segment{};
            int lane = -1;
            for (int slot = 0; slot < package.faces; ++slot)
            {
                const auto candidate = prepared.segments(package.firstSegment + slot, 0);
                const int offset = int(threadIdx.x) - candidate.firstLane;
                if (offset >= 0 && offset < candidate.lanes)
                {
                    segment = candidate;
                    lane = offset;
                    break;
                }
            }
            index first = 0;
            if (lane >= 0)
                first = geometry.faceOffsets(geometry.FaceAnchor(segment.auxiliaryFace).face, 0);
            std::array<real, C> total{};
            NumericalStatus faceStatus;
            for (int tile = 0; tile < package.tiles; ++tile)
            {
                std::array<real, C> value{};
                NumericalStatus pointStatus;
                const index q = index(tile) * segment.lanes + lane;
                if (lane >= 0 && q < segment.points)
                    pointStatus = ACMFacePoint<C>(geometry, controls, fields, output,
                                                  segment.auxiliaryFace, first, first + q, value);
                for (int j = 0; j < C; ++j)
                    values[C * threadIdx.x + j] = value[j];
                rows[threadIdx.x] = pointStatus.row;
                points[threadIdx.x] = pointStatus.point;
                codes[threadIdx.x] = int(pointStatus.code);
                details[threadIdx.x] = pointStatus.detail;
                __syncthreads();
                if (lane == 0)
                    for (int offset = 0; offset < segment.lanes; ++offset)
                    {
                        const int source = segment.firstLane + offset;
                        if (faceStatus && codes[source] != int(NumericalError::None))
                            faceStatus = {NumericalError(codes[source]), details[source], rows[source], points[source]};
                        for (int j = 0; j < C; ++j)
                            total[j] += values[C * source + j];
                    }
                __syncthreads();
            }
            if (lane == 0)
            {
                const index iFace = segment.auxiliaryFace;
                output.status(iFace, 0) = faceStatus;
                output.fallbackCounts(iFace, 0) = 0;
                if (faceStatus)
                    for (int j = 0; j < C; ++j)
                        output.high(iFace, j) = total[j];
            }
        }
    }

    template <int Components, RHSFaceMapping Mapping, ReconstructionForm Form = ReconstructionForm::Generic, class Fields, class Output>
    void LaunchACMMapping(const RHSFaceLaunch &launch, const Fields &fields, const Output &output)
    {
        const auto &controls = launch.controls;
        const auto stream = Runtime::CUDAStream(launch.context);
        if constexpr (Mapping == RHSFaceMapping::CTASegments)
        {
            Runtime::CheckKernelArgumentBudget<ACMGeometryView, RHSKernelControls, Fields, Output, RHSFacePackageView>();
            ACMFluxCTASegments<Components><<<launch.segmentBlocks, controls.blockSize, launch.packages.sharedBytes, stream>>>(
                launch.geometry.View(), controls, fields, output, launch.packages);
        }
        else
        {
            Runtime::CheckKernelArgumentBudget<ACMGeometryView, RHSKernelControls, Fields, Output, RHSIndices>();
            const unsigned blocks = unsigned(std::min<index>((launch.selected.RowCount() + controls.blockSize - 1) / controls.blockSize,
                                                             controls.gridBlocks ? controls.gridBlocks : 1024));
            ACMFluxFaceThreads<Components, Form><<<blocks, controls.blockSize, 0, stream>>>(
                launch.geometry.View(), controls, fields, output, launch.selected);
        }
    }

    template <int Components>
    void LaunchACMFaces(const RHSFaceLaunch &launch)
    {
        if (launch.controls.mapping == RHSFaceMapping::CTASegments)
            LaunchACMMapping<Components, RHSFaceMapping::CTASegments>(launch, launch.fields, launch.output);
        else
            LaunchACMMapping<Components, RHSFaceMapping::Thread>(launch, launch.fields, launch.output);
    }

    template <int C, int M, int D, RHSFaceMapping Mapping>
    void LaunchACMRHSShape(const RHSFaceLaunch &launch)
    {
        static_assert(Mapping == RHSFaceMapping::Thread || Mapping == RHSFaceMapping::CTASegments);
        const auto &controls = launch.controls;
        const bool modal = !controls.firstOrder && !controls.direct;
        DNDS_check_throw_info(controls.mechanical && controls.components == C && controls.dimension == D &&
                                  controls.mapping == Mapping && (modal ? M > 0 && controls.modes == M : M == 0),
                              "ACM flux launch differs from its prepared profile");
        constexpr int W = C * (M ? M : D);
        using State = Runtime::FixedRows<C, C>;
        using Coefficients = Runtime::FixedRows<W, W>;
        ACMPolynomialFields<C, M, D> fields{};
        fields.mean = ProjectRHSField<State>(launch.fields.mean);
        if (launch.fields.hasIncoming)
            fields.incoming = Runtime::ProjectKernelView<State>(launch.fields.incoming.DynamicView());
        fields.centerViscosity = launch.fields.centerViscosity;
        if (modal || (!controls.firstOrder && controls.linear))
        {
            fields.reconstruction = ProjectRHSField<Coefficients>(launch.fields.reconstruction);
            if (controls.distinctViscous)
                fields.viscousData = ProjectRHSField<Coefficients>(launch.fields.viscousData);
        }
        const auto output = ProjectRHSOutputs<C>(launch.output);
        if constexpr (M > 0)
            LaunchACMMapping<C, Mapping, ReconstructionForm::Modal>(launch, fields, output);
        else if (!controls.firstOrder && controls.linear)
            LaunchACMMapping<C, Mapping, ReconstructionForm::DirectGradient>(launch, fields, output);
        else
            LaunchACMMapping<C, Mapping, ReconstructionForm::MeanOnly>(launch, fields, output);
    }
}
