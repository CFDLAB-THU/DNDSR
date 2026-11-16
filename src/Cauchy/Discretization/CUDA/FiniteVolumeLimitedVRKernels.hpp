#pragma once
#include "Cauchy/Discretization/LimitedVREvaluation.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "CFV/Math/LimitedVR.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy::CUDA
{
    using LVRInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using LVROutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using LVRMeans = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using LVRIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using LVRStencil = Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using LVRStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Pure-flow5 O2 target guard, matching the current CPU LVR contract.
    /// It is separate from final equation-aware reconstruction PP.
    void LimitLVRTarget(Runtime::ExecutionContext, LVRStencil, LVRInput means, LVROutput coefficients,
                        int dimension, ThermalBounds bounds, int betaPower, real betaCutoff, LVRStatus);
    void FillLVRScalars(Runtime::ExecutionContext, LVRInput means, LVROutput scalars, PerfectGas, LVRStatus);
    /// On-site accepted affine target, sharing canonical face geometry and
    /// periodic maps. No global sampled values or LVR-only geometry payload.
    void BuildLVRFaceSensors(Runtime::ExecutionContext, LVRStencil, LVRMeans means,
                             LVRMeans targetCoefficients, int dimension, PerfectGas,
                             ThermalBounds, CFV::LimitedVRKernelParameters, LVROutput sensors, LVRStatus);
    /// Deterministic cell gather using the shared face routing and incidences.
    void GatherLVRBranches(Runtime::ExecutionContext, LVRStencil, LVRInput sensors,
                           Runtime::ArrayView<const NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faceStatus,
                           LVROutput branches, Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base, LVRStatus);
    void MergeLVRBase(Runtime::ExecutionContext, Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base,
                      LVROutput seeds, int column);
    void PublishLVRPenalty(Runtime::ExecutionContext, LVRInput branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> history,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> penalty);
}
