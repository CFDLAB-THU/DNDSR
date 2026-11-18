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
        Pair<C * M> reconstruction, viscousData;
    };

    template <int C, class Fields, class Output>
    __device__ NumericalStatus ACMFacePoint(const RHSGeometry &geometry, const RHSKernelControls &controls,
                                            const Fields &fields, const Output &output, RHSWallOutput wallShear,
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
                    if (!(distance > 0)) return NumericalStatus{NumericalError::InvalidInput}.At(sample.cell, entry);
                    omega = physics.flow.kinematicViscosity * Physics::RANS::kWallOmegaCoeff / sqr(distance);
                }
            if (!ACM::Exterior(physics.flow, kind, left.values, bc.acmState, bc.velocity,
                               sample.normal, right.values, C, omega))
                return NumericalStatus{NumericalError::UnsupportedRequest}.At(sample.cell, entry);
            // Keep the reflected SA scalar for the wall-normal viscous jump.
        }
        std::array<real, C> flux{};
        ACM::NumericalFlux(physics.flow, left.values, right.values, sample.normal, flux, C);
        if (controls.viscous)
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
                const auto transport = Physics::RANS::TryTransport(physics.rans, at);
                if (!transport) return transport.status.At(sample.cell, entry);
                const Eigen::Vector3d normal(sample.normal[0], sample.normal[1], sample.normal[2]);
                const auto diffusion = Physics::RANS::TryDiffusionFlux(physics.rans, at, normal, transport.value.viscosity);
                if (!diffusion) return diffusion.status.At(sample.cell, entry);
                for (int j = 4; j < C; ++j) flux[j] -= diffusion.value[j - 4];
                viscosity += transport.value.viscosity;
            }
            const auto traction = Physics::Transport::NewtonianTraction(
                [&](int j, int d) { return face.gradients[j][d]; }, sample.normal, viscosity);
            if (wallShear.RowCount() && sample.boundaryPoint != UnInitIndex)
            {
                const auto shear = Physics::Transport::WallShear(traction, sample.normal);
                for (int d = 0; d < 3; ++d)
                    wallShear(sample.boundaryPoint, d) = kind == BoundaryKind::NoSlipWall ? shear[d] : 0;
            }
            for (int d = 0; d < 3; ++d)
            {
                flux[d] -= traction[d];
                if constexpr (C == 6) flux[d] += (2. / 3.) * face.values[4] * sample.normal[d];
            }
        }
        const real weight = geometry.quadrature(entry, 6);
        if (output.wallForce.RowCount() && sample.boundaryPoint != UnInitIndex)
            for (int d = 0; d < 3; ++d)
                output.wallForce(sample.boundaryPoint, d) =
                    kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall ? weight * flux[d] : 0;
        for (int j = 0; j < C; ++j)
        {
            if (!std::isfinite(flux[j])) return NumericalStatus{NumericalError::InvalidState}.At(sample.cell, entry);
            value[j] = -weight * flux[j];
        }
        return {};
    }

    template <int C, class Fields, class Output>
    __device__ NumericalStatus ACMFace(const RHSGeometry &geometry, const RHSKernelControls &controls,
                                       const Fields &fields, const Output &output, RHSWallOutput wallShear, index iFace)
    {
        const auto anchor = geometry.FaceAnchor(iFace);
        const index first = geometry.faceOffsets(anchor.face, 0), end = geometry.faceOffsets(anchor.face + 1, 0);
        std::array<real, C> integrated{};
        for (index entry = first; entry < end; ++entry)
        {
            std::array<real, C> value{};
            const auto status = ACMFacePoint<C>(geometry, controls, fields, output, wallShear, iFace, first, entry, value);
            if (!status)
                return status;
            for (int j = 0; j < C; ++j)
                integrated[j] += value[j];
        }
        for (int j = 0; j < C; ++j)
            output.high(iFace, j) = integrated[j];
        return {};
    }

    template <int C, class Fields, class Output>
    __global__ void ACMFaces(ACMGeometryView geometry, RHSKernelControls controls, Fields fields,
                             Output output, RHSIndices selected, RHSWallOutput wallShear)
    {
        for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
        {
            const index iFace = selected(iRow, 0);
            output.status(iFace, 0) = ACMFace<C>(geometry(0, 0), controls, fields, output, wallShear, iFace);
            output.fallbackCounts(iFace, 0) = 0;
        }
    }

    // The prototype uses the same complete-face packages as the compressible
    // mapping. Each lane evaluates one quadrature point; its segment leader
    // integrates in rule order, preserving the thread-per-face sum.
    template <int C, class Fields, class Output>
    __global__ void ACMSegments(ACMGeometryView publication, RHSKernelControls controls, Fields fields,
                                Output output, RHSWallOutput wallShear, RHSFacePackageView prepared)
    {
        extern __shared__ real shared[];
        real *values = shared;
        auto *status = reinterpret_cast<NumericalStatus *>(values + C * blockDim.x);
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
                    pointStatus = ACMFacePoint<C>(geometry, controls, fields, output, wallShear,
                                                 segment.auxiliaryFace, first, first + q, value);
                for (int j = 0; j < C; ++j)
                    values[C * threadIdx.x + j] = value[j];
                status[threadIdx.x] = pointStatus;
                __syncthreads();
                if (lane == 0)
                    for (int offset = 0; offset < segment.lanes; ++offset)
                    {
                        const int source = segment.firstLane + offset;
                        if (faceStatus && !status[source])
                            faceStatus = status[source];
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

    template <int Dimension>
    constexpr RHSFaceMapping ACMPolynomialMapping = Dimension == 3 ? RHSFaceMapping::CTASegments : RHSFaceMapping::Thread;

    template <int Components, RHSFaceMapping Mapping, class Fields, class Output>
    void LaunchACMMapping(const RHSFaceLaunch &launch, RHSWallOutput wallShear, const Fields &fields, const Output &output)
    {
        const auto &controls = launch.controls;
        const auto stream = Runtime::CUDAStream(launch.context);
        if constexpr (Mapping == RHSFaceMapping::CTASegments)
        {
            Runtime::CheckKernelArgumentBudget<ACMGeometryView, RHSKernelControls, Fields, Output, RHSWallOutput, RHSFacePackageView>();
            ACMSegments<Components><<<launch.segmentBlocks, controls.blockSize, launch.packages.sharedBytes, stream>>>(
                launch.geometry.View(), controls, fields, output, wallShear, launch.packages);
        }
        else
        {
            Runtime::CheckKernelArgumentBudget<ACMGeometryView, RHSKernelControls, Fields, Output, RHSIndices, RHSWallOutput>();
            const unsigned blocks = unsigned(std::min<index>((launch.selected.RowCount() + controls.blockSize - 1) / controls.blockSize,
                                                             controls.gridBlocks ? controls.gridBlocks : 1024));
            ACMFaces<Components><<<blocks, controls.blockSize, 0, stream>>>(
                launch.geometry.View(), controls, fields, output, launch.selected, wallShear);
        }
    }

    template <int Components>
    void LaunchACMFaces(const RHSFaceLaunch &launch, RHSWallOutput wallShear)
    {
        if (launch.controls.mapping == RHSFaceMapping::CTASegments)
            LaunchACMMapping<Components, RHSFaceMapping::CTASegments>(launch, wallShear, launch.fields, launch.output);
        else
            LaunchACMMapping<Components, RHSFaceMapping::Thread>(launch, wallShear, launch.fields, launch.output);
    }

    template <int C, int M, int D>
    void LaunchACMRHSShape(const RHSFaceLaunch &launch, RHSWallOutput wallShear)
    {
        DNDS_check_throw_info(launch.controls.mechanical && !launch.controls.direct && !launch.controls.firstOrder &&
                                  launch.controls.components == C && launch.controls.modes == M && launch.controls.dimension == D &&
                                  launch.controls.mapping == ACMPolynomialMapping<D>,
                              "ACM RHS static C/M/D launch differs from its prepared profile");
        using Width = Runtime::FixedRows<C, C>;
        using Coefficients = Runtime::FixedRows<C * M, C * M>;
        const ACMPolynomialFields<C, M, D> fields{
            ProjectRHSField<Width>(launch.fields.mean),
            ProjectRHSField<Coefficients>(launch.fields.reconstruction),
            launch.controls.distinctViscous ? ProjectRHSField<Coefficients>(launch.fields.viscousData)
                                            : typename ACMPolynomialFields<C, M, D>::template Pair<C * M>{}};
        const auto output = ProjectRHSOutputs<C>(launch.output);
        LaunchACMMapping<C, ACMPolynomialMapping<D>>(launch, wallShear, fields, output);
    }
}
