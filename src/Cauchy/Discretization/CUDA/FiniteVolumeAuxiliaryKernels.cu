#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeAuxiliaryKernels.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <int Components, bool Mechanical = false>
        __device__ std::conditional_t<Mechanical, ACM::KernelPoint<Components>, PerfectGasKernelPoint<Components>> Load(const AuxiliaryStateInput &input, index iCell)
        {
            std::conditional_t<Mechanical, ACM::KernelPoint<Components>, PerfectGasKernelPoint<Components>> point;
            point.Load(iCell, input.mean, input.gradient, input.dimension, input.hasGradient);
            return point;
        }
        template <int Components, bool Mechanical = false>
        __device__ NumericalResult<std::array<real, 4>> Properties(const AuxiliaryStateInput &input,
                                                                   const std::conditional_t<Mechanical, ACM::KernelPoint<Components>, PerfectGasKernelPoint<Components>> &point, index iCell)
        {
            if constexpr (Mechanical)
                return point.Auxiliary(input.acm(0, 0), input.hasWall ? input.wallMean(iCell, 0) : 1);
            else
                return point.Auxiliary(input.parameters(0, 0), input.hasWall ? input.wallMean(iCell, 0) : 1);
        }
        template <int Components, bool Mechanical = false>
        __global__ void Means(AuxiliaryStateInput input, AuxiliaryOutput output, AuxiliaryOutput viscosity, AuxiliaryStatus status, index first, index end)
        {
            for (index iCell = first + index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < end; iCell += index(gridDim.x) * blockDim.x)
            {
                const auto result = Properties<Components, Mechanical>(input, Load<Components, Mechanical>(input, iCell), iCell);
                status(iCell, 0) = result.status.At(iCell);
                if (result)
                {
                    if (output.RowCount())
                        for (int j = 0; j < 3; ++j)
                            output(iCell, j) = result.value[j];
                    if (viscosity.RowCount())
                        viscosity(iCell, 0) = result.value[3];
                }
            }
        }
        template <int Components, bool Mechanical = false>
        __device__ NumericalResult<std::array<real, 4>> Prepared(const AuxiliaryFaceRequest &request,
                                                                 const std::conditional_t<Mechanical, ACM::KernelPoint<Components>, PerfectGasKernelPoint<Components>> &point, index iCell)
        {
            if (request.preparedMeans.RowCount())
                return std::array<real, 4>{request.preparedMeans(iCell, 0), request.preparedMeans(iCell, 1), request.preparedMeans(iCell, 2)};
            return Properties<Components, Mechanical>(request.state, point, iCell);
        }
        template <int Components, bool Mechanical = false>
        __device__ NumericalStatus Face(const AuxiliaryFaceRequest &request, index iAux)
        {
            const auto &stencil = request.stencil(0, 0);
            const auto &metric = stencil.metric;
            const auto iFace = stencil.stencilFaces(iAux, 0), first = metric.FaceQuadratureBegin(iFace);
            const auto anchor = stencil.RHSFaceSample(iFace, 0, first);
            const auto left = Load<Components, Mechanical>(request.state, anchor.cell);
            const auto leftResult = Prepared<Components, Mechanical>(request, left, anchor.cell);
            if (!leftResult)
                return leftResult.status.At(anchor.cell);
            Riemann::AcousticState leftData{};
            if constexpr (!Mechanical)
                leftData = {{left.Flow(), leftResult.value[0]}, leftResult.value[1]};
            real convection = 0, diffusion = 0, jump = 0, minimum = leftResult.value[0];
            const auto count = metric.FaceQuadratureEnd(iFace) - first;
            for (index q = 0; q < count; ++q)
            {
                const auto &sample = stencil.RHSFaceSample(iFace, 0, first + q);
                auto right = left;
                bool interface = false;
                if (sample.other == UnInitIndex)
                {
                    // Match CPU EvaluateDt semantics: physical BCs use the
                    // owned mean; only an interface uses supplied remote data.
                    interface = request.conditions(sample.condition, 0) == BoundaryKind::Interface;
                    if (interface)
                    {
                        if (sample.boundaryPoint < 0 || sample.boundaryPoint >= request.incoming.RowCount())
                            return NumericalStatus{NumericalError::InvalidInput}.At(sample.cell, sample.point);
                        for (int j = 0; j < Components; ++j)
                            right.values[j] = request.incoming(sample.boundaryPoint, j);
                    }
                }
                else
                {
                    right = Load<Components, Mechanical>(request.state, sample.other);
                    if (sample.changesFrame)
                        right.Rotate(sample.rotation);
                }
                const auto &physics = [&]() -> const auto &
                { if constexpr (Mechanical) return request.state.acm(0,0); else return request.state.parameters(0,0); }();
                const auto rightResult = sample.other == UnInitIndex
                                             ? (interface ? right.Auxiliary(physics, 1) : leftResult)
                                             : Prepared<Components, Mechanical>(request, right, sample.other);
                if (!rightResult)
                    return rightResult.status.At(sample.cell, sample.point);
                Riemann::AcousticState rightData{};
                if constexpr (!Mechanical)
                    rightData = {{right.Flow(), rightResult.value[0]}, rightResult.value[1]};
                if (request.needSpectral)
                {
                    const auto grid = request.frame.GridVelocity(sample.ownPoint);
                    if constexpr (Mechanical)
                    {
                        const real gridNormal = ACM::NormalSpeed(grid, sample.normal);
                        convection = std::max(convection, std::max(ACM::SpectralRadius(physics.flow, ACM::NormalSpeed(left.values, sample.normal), gridNormal),
                                                                   ACM::SpectralRadius(physics.flow, ACM::NormalSpeed(right.values, sample.normal), gridNormal)));
                    }
                    else
                        convection = std::max(convection, std::max(Riemann::SignalSpeed(leftData, sample.normal, grid), Riemann::SignalSpeed(rightData, sample.normal, grid)));
                    real bound = std::max(leftResult.value[2], rightResult.value[2]);
                    if (request.state.hasWall)
                    {
                        std::conditional_t<Mechanical, ACM::KernelPoint<Components>, PerfectGasKernelPoint<Components>> face;
                        for (int j = 0; j < Components; ++j)
                        {
                            face.values[j] = sample.other == UnInitIndex ? left.values[j] : .5 * (left.values[j] + right.values[j]);
                            for (int d = 0; d < 3; ++d)
                                face.gradients[j][d] = .5 * (left.gradients[j][d] + right.gradients[j][d]);
                        }
                        const auto faceResult = face.Auxiliary(physics, request.wallFaces(sample.face, q));
                        if (!faceResult)
                            return faceResult.status.At(sample.cell, sample.point);
                        bound = std::max(bound, faceResult.value[2]);
                    }
                    diffusion = std::max(diffusion, bound * metric.GetFaceArea(stencil.stencilFaces(iAux, 0)) *
                                                        (1 / metric.GetCellVol(sample.cell) + 1 / metric.GetCellVol(sample.other == UnInitIndex ? sample.cell : sample.other)));
                }
                if (!Mechanical && request.needWave)
                    jump = std::max(jump, Riemann::TransverseSignalJump(leftData, rightData, sample.normal));
                if (request.needPressure)
                    minimum = std::min(minimum, rightResult.value[0]);
            }
            if (request.spectral.RowCount())
            {
                request.spectral(iAux, 0) = convection;
                request.spectral(iAux, 1) = diffusion;
            }
            if (request.wave.RowCount())
                request.wave(iAux, 0) = jump;
            if (request.pressure.RowCount())
                request.pressure(iAux, 0) = minimum;
            if (request.scratch.RowCount())
            {
                request.scratch(iAux, 0) = convection + diffusion;
                request.scratch(iAux, 1) = jump;
                request.scratch(iAux, 2) = minimum;
            }
            return {};
        }
        template <int Components, bool Mechanical, class Rows>
        __global__ void Faces(AuxiliaryFaceRequest request, index count, Rows faceAt)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < count; iRow += index(gridDim.x) * blockDim.x)
            {
                const auto iAux = faceAt(iRow);
                request.status(iAux, 0) = Face<Components, Mechanical>(request, iAux);
            }
        }
        template <class Rows>
        __global__ void Cells(AuxiliaryCellRequest request, index count, Rows cellAt)
        {
            const auto &stencil = request.stencil(0, 0);
            const auto &metric = stencil.metric;
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < count; iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = cellAt(iRow);
                real sum = 0, wave = 0, pressure = std::numeric_limits<real>::max();
                for (index entry = stencil.auxiliaryCellOffsets(iCell, 0); entry < stencil.auxiliaryCellOffsets(iCell + 1, 0); ++entry)
                {
                    const index iAux = stencil.auxiliaryCellFaces(entry, 0);
                    if (request.spectralSum.RowCount())
                        sum += metric.GetFaceArea(stencil.stencilFaces(iAux, 0)) * (request.spectral.RowCount() ? request.spectral(iAux, 0) + request.spectral(iAux, 1) : request.scratch(iAux, 0));
                    if (request.waveMaximum.RowCount())
                        wave = std::max(wave, request.wave.RowCount() ? request.wave(iAux, 0) : request.scratch(iAux, 1));
                    if (request.pressureMinimum.RowCount())
                        pressure = std::min(pressure, request.pressure.RowCount() ? request.pressure(iAux, 0) : request.scratch(iAux, 2));
                }
                if (request.spectralSum.RowCount())
                    request.spectralSum(iCell, 0) = sum + metric.GetCellVol(iCell) * request.angularSpeed;
                if (request.waveMaximum.RowCount())
                    request.waveMaximum(iCell, 0) = wave;
                if (request.pressureMinimum.RowCount())
                    request.pressureMinimum(iCell, 0) = pressure;
            }
        }
        unsigned Blocks(index count) { return unsigned(std::min<index>((count + 127) / 128, 1024)); }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA auxiliary launch failed"); }
        template <class View>
        void CheckRows(View view, index rows, index width, bool required)
        {
            DNDS_check_throw_info((!required && !view.RowCount()) || (view.RowCount() == rows && view.Width() == width), "CUDA auxiliary field shape mismatch");
        }
    }
    void AuxiliaryStateInput::Validate(R::Device device) const
    {
        DNDS_check_throw_info(device.space == R::MemorySpace::CUDA && (acm.RowCount() ? components >= 4 && components <= 6 : components >= 5 && components <= 7) && (dimension == 2 || dimension == 3), "invalid CUDA auxiliary profile");
        R::ValidateViewDevices(device, mean, gradient, wallMean);
        if (acm.RowCount())
            acm.ValidateDevice(device);
        else
            parameters.ValidateDevice(device);
        DNDS_check_throw_info((acm.RowCount() == 1 || parameters.RowCount() == 1) && mean.owned.Width() == components &&
                                  (!mean.ghost.RowCount() || mean.ghost.Width() == components),
                              "CUDA auxiliary state shape mismatch");
        if (hasGradient)
        {
            CheckRows(gradient.owned, mean.owned.RowCount(), components * dimension, true);
            CheckRows(gradient.ghost, mean.ghost.RowCount(), components * dimension, bool(mean.ghost.RowCount()));
        }
        if (hasWall)
            DNDS_check_throw_info(wallMean.RowCount() >= mean.RowCount(), "CUDA auxiliary wall mean is incomplete");
    }
    void PrepareAuxiliaryMeans(R::ExecutionContext context, const AuxiliaryStateInput &state,
                               AuxiliaryOutput output, AuxiliaryOutput viscosity, AuxiliaryStatus status, index first, index end)
    {
        state.Validate(context.device);
        R::ValidateViewDevices(context.device, output, viscosity, status);
        R::CheckKernelArgumentBudget<AuxiliaryStateInput, AuxiliaryOutput, AuxiliaryOutput, AuxiliaryStatus, index, index>();
        DNDS_check_throw_info(first >= 0 && first <= end && end <= state.mean.RowCount() && status.RowCount() >= end &&
                                  (!end || output.RowCount() || viscosity.RowCount()) &&
                                  (!output.RowCount() || (output.RowCount() >= end && output.Width() == 3)) &&
                                  (!viscosity.RowCount() || (viscosity.RowCount() >= end && viscosity.Width() == 1)),
                              "CUDA auxiliary mean range/shape mismatch");
        if (first == end)
            return;
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        const unsigned blocks = Blocks(end - first);
        if (state.acm.RowCount())
        {
            if (state.components == 4)
                Means<4, true><<<blocks, 128, 0, stream>>>(state, output, viscosity, status, first, end);
            else if (state.components == 5)
                Means<5, true><<<blocks, 128, 0, stream>>>(state, output, viscosity, status, first, end);
            else
                Means<6, true><<<blocks, 128, 0, stream>>>(state, output, viscosity, status, first, end);
            CheckLaunch();
            return;
        }
        switch (state.components)
        {
        case 5:
            Means<5><<<blocks, 128, 0, stream>>>(state, output, viscosity, status, first, end);
            break;
        case 6:
            Means<6><<<blocks, 128, 0, stream>>>(state, output, viscosity, status, first, end);
            break;
        case 7:
            Means<7><<<blocks, 128, 0, stream>>>(state, output, viscosity, status, first, end);
            break;
        }
        CheckLaunch();
    }
    void PrepareAuxiliaryFaces(R::ExecutionContext context, const AuxiliaryFaceRequest &request, AuxiliaryIndices faces)
    {
        request.state.Validate(context.device);
        R::ValidateViewDevices(context.device, request.stencil, request.conditions, request.wallFaces, request.preparedMeans, request.incoming,
                               request.spectral, request.wave, request.pressure, request.scratch, request.status, faces);
        R::CheckKernelArgumentBudget<AuxiliaryFaceRequest, AuxiliaryIndices>();
        const auto rows = request.faceNumber;
        DNDS_check_throw_info(request.stencil.RowCount() == 1 && request.status.RowCount() == rows,
                              "CUDA auxiliary face shape mismatch");
        CheckRows(request.spectral, rows, 2, false);
        CheckRows(request.wave, rows, 1, false);
        CheckRows(request.pressure, rows, 1, false);
        CheckRows(request.scratch, rows, 3, false);
        if (request.preparedMeans.RowCount())
            CheckRows(request.preparedMeans, request.allCells, 3, true);
        DNDS_check_throw_info(!request.incoming.RowCount() || request.incoming.Width() == request.state.components,
                              "CUDA auxiliary interface width mismatch");
        DNDS_check_throw_info(!request.state.hasWall || request.wallFaces.RowCount() >= rows,
                              "CUDA auxiliary wall-face table is incomplete");
        DNDS_check_throw_info((!request.spectral.RowCount() || request.needSpectral) && (!request.wave.RowCount() || request.needWave) &&
                                  (!request.pressure.RowCount() || request.needPressure),
                              "CUDA auxiliary output requested without its computation");
        if (!faces.RowCount() || (!request.needSpectral && !request.needWave && !request.needPressure))
            return;
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        const unsigned blocks = Blocks(faces.RowCount());
        faces.Visit([&](auto faceAt)
                    {
        if (request.state.acm.RowCount())
        {
            if (request.state.components == 4)
                Faces<4, true><<<blocks, 128, 0, stream>>>(request, faces.RowCount(), faceAt);
            else if (request.state.components == 5)
                Faces<5, true><<<blocks, 128, 0, stream>>>(request, faces.RowCount(), faceAt);
            else
                Faces<6, true><<<blocks, 128, 0, stream>>>(request, faces.RowCount(), faceAt);
            return;
        }
        switch (request.state.components)
        {
        case 5:
            Faces<5, false><<<blocks, 128, 0, stream>>>(request, faces.RowCount(), faceAt);
            break;
        case 6:
            Faces<6, false><<<blocks, 128, 0, stream>>>(request, faces.RowCount(), faceAt);
            break;
        case 7:
            Faces<7, false><<<blocks, 128, 0, stream>>>(request, faces.RowCount(), faceAt);
            break;
        } });
        CheckLaunch();
    }
    void ReduceAuxiliaryCells(R::ExecutionContext context, const AuxiliaryCellRequest &request, AuxiliaryIndices cells)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && request.angularSpeed >= 0 && std::isfinite(request.angularSpeed), "invalid CUDA cell reduction context");
        R::ValidateViewDevices(context.device, request.stencil,
                               request.spectral, request.wave, request.pressure, request.scratch, request.spectralSum, request.waveMaximum, request.pressureMinimum, cells);
        R::CheckKernelArgumentBudget<AuxiliaryCellRequest, AuxiliaryIndices>();
        const index rows = request.ownedCells;
        DNDS_check_throw_info(rows >= 0 && request.stencil.RowCount() == 1, "CUDA auxiliary cell descriptor is incomplete");
        for (const auto output : {request.spectralSum, request.waveMaximum, request.pressureMinimum})
            CheckRows(output, rows, 1, false);
        const auto faceRows = request.faceNumber;
        CheckRows(request.spectral, faceRows, 2, bool(request.spectralSum.RowCount()) && !request.scratch.RowCount());
        CheckRows(request.wave, faceRows, 1, bool(request.waveMaximum.RowCount()) && !request.scratch.RowCount());
        CheckRows(request.pressure, faceRows, 1, bool(request.pressureMinimum.RowCount()) && !request.scratch.RowCount());
        CheckRows(request.scratch, faceRows, 3, false);
        if (!cells.RowCount())
            return;
        cells.Visit([&](auto cellAt)
                    { Cells<<<Blocks(cells.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(request, cells.RowCount(), cellAt); });
        CheckLaunch();
    }
}
