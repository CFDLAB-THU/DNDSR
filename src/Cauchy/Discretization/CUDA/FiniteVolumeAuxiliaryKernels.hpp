#pragma once

#include "Cauchy/Discretization/PerfectGasKernelPoint.hpp"
#include "Cauchy/Discretization/ACMKernelPoint.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::Cauchy::CUDA
{
    using AuxiliaryIndices = Runtime::RowSelection;
    using AuxiliaryInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using AuxiliaryOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using AuxiliaryStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /// Common numerical operands. The C++ operator owns the frozen metadata,
    /// field leases and status buffers. Empty optional views must carry no
    /// active use. Component ordering is flow5 followed by zero/one/two RANS.
    struct AuxiliaryStateInput
    {
        Runtime::ArrayView<const PerfectGasSpatialParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> parameters;
        Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> mean, gradient;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> wallMean;
        int components = 5, dimension = 3;
        bool hasGradient = false, hasWall = false;
        Runtime::ArrayView<const ACM::SpatialParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> acm;
        void Validate(Runtime::Device device) const;
    };

    /// Face work is indexed by canonical auxiliary-face IDs. Optional outputs
    /// preserve the public auxiliary bundle: spectral [convection,diffusion],
    /// wave jump and pressure minimum may be omitted. Scratch [lambda,jump,p]
    /// serves cell reductions when a corresponding public face field is absent.
    /// Status is one record per auxiliary face, not an atomic global counter.
    struct AuxiliaryFaceRequest
    {
        AuxiliaryStateInput state;
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        index faceNumber = 0, allCells = 0;
        Runtime::ArrayView<const BoundaryKind, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> conditions;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::CSRRows> wallFaces;
        AuxiliaryInput preparedMeans, incoming;
        AuxiliaryOutput spectral, wave, pressure, scratch;
        AuxiliaryStatus status;
        RigidFrameParameters frame;
        bool needSpectral = false, needWave = false, needPressure = false;
    };
    /// One row per owned cell. Face inputs must be ready on this stream; cell
    /// subsets are already classified by the mesh work-partition facility.
    struct AuxiliaryCellRequest
    {
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        index ownedCells = 0, faceNumber = 0;
        AuxiliaryInput spectral, wave, pressure, scratch;
        AuxiliaryOutput spectralSum, waveMaximum, pressureMinimum;
        real angularSpeed = 0;
    };
    /// Host launchers validate residency and shape once and enqueue on context.
    /// No host wait/copy, lease acquisition, MPI, recovery or logging is hidden
    /// here. Caller retains all operands through completion, then checks status
    /// before publication. Mean status uses local-cell IDs; face status uses
    /// auxiliary-face IDs. Untouched rows retain their previous contents.
    void PrepareAuxiliaryMeans(Runtime::ExecutionContext context, const AuxiliaryStateInput &state,
                               AuxiliaryOutput output, AuxiliaryOutput viscosity, AuxiliaryStatus status, index first, index end);
    void PrepareAuxiliaryFaces(Runtime::ExecutionContext context, const AuxiliaryFaceRequest &request, AuxiliaryIndices faces);
    void ReduceAuxiliaryCells(Runtime::ExecutionContext context, const AuxiliaryCellRequest &request, AuxiliaryIndices cells);
}
