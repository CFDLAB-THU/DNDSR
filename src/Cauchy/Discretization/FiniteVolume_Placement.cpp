#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Discretization/PerfectGasKernelPoint.hpp"
#include "Cauchy/Config/CUDAKernelDefaults.hpp"

namespace DNDS::Cauchy
{
    FiniteVolume FiniteVolume::Placed(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        namespace R = Runtime;
        DNDS_check_throw_info(_data && context.device == device,
                              "spatial placement needs a live plan and matching context");
        Metrics().Validate();
        if (device == Device())
            return *this;
        DNDS_check_throw_info(Device() == R::Device::Host() && device.space == R::MemorySpace::CUDA,
                              "spatial placement starts from CPU setup and supports CUDA only");
        // Keep the source generation pinned while publishing all its caches.
        // A sequence of individual generation checks would leave gaps between
        // independent placements in which a mesh edit could otherwise enter.
        const auto geometry = Metrics().AcquireRead();
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(false, "CUDA spatial placement is unavailable in this build");
#endif
        // Resolve physics and every physical boundary before allocating any
        // resident publication. This is not a permissive capability fallback.
        const auto physics = PerfectGasSpatialParameters::FromEquation(_data->equation, Metrics().Dimension());
        physics.Validate(5 + physics.rans.ComponentNumber());
        std::vector<PerfectGasBoundaryParameters> conditions;
        conditions.reserve(_data->boundaryConditions.size());
        for (const auto &condition : _data->boundaryConditions)
        {
            if (condition.kind == BoundaryKind::Interior || condition.kind == BoundaryKind::Interface)
            {
                // These entries have no prescribed thermodynamic state. The
                // face kernel takes its neighboring mean or interface packet.
                PerfectGasBoundaryParameters parameters;
                parameters.kind = condition.kind;
                conditions.push_back(parameters);
            }
            else
                conditions.push_back(condition.PerfectGasKernelParameters());
        }
        // Copy the semantic facade, then replace numerical owners. No second
        // backend-specific public plan/workspace class is introduced. This
        // source plan stays usable. Runtime topology/control metadata remain
        // host-owned; replaced numerical payloads do not retain CPU duplicates.
        auto placed = std::make_shared<Data>(*_data);
        ResolveAutomaticCUDAKernels(placed->settings, IsA100Device(device), !_data->equation.HasRANS());
        placed->metrics = _data->metrics.Placed(device, context);
        placed->gradient = _data->gradient.Placed(placed->metrics, context);
        CFV::SamplingPlacementMemo samplingPlacement(device, context);
        if (_data->reconstruction)
        {
            CFV::ReconstructionResidencyOptions options;
            const auto &solver = placed->settings.numerics.reconstructionSolver;
            if (solver.UsesAMG())
            {
                solver.Validate();
                Algebra::AmgXSettings settings;
                settings.cycles = solver.amgxCycles;
                placed->reconstructionAMG = CFV::ReconstructionAMGPlan::Prepare(
                    *_data->reconstruction, context,
                    solver.amgxStorage == "native" ? CFV::ReconstructionAMGStorage::NativeStorage
                                                   : CFV::ReconstructionAMGStorage::PublicUpload,
                    settings);
                // AMG owns the projected matrix. Keep only polynomial sampling
                // and optional WBAP data in the ordinary resident publication.
                options.sweeps = false;
            }
            options.retainHostPayloads = false;
            // Centered boundary consumers evaluate independently of the face
            // quadrature cache and therefore retain central moments/scale.
            options.arbitraryPointSampling = _data->hasInterfaces ||
                                             _data->settings.numerics.reconstructionBoundaryFromPolynomial;
            options.basis = true;
            options.basisDerivativeOrder = 1;
            placed->reconstructionKernel = placed->settings.execution.ReconstructionKernel();
            if (placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::FP32 ||
                placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::FP32Pipelined ||
                placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::TF32Pipelined ||
                placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::FP32FullAccumulation ||
                placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::FP32FloatFloatAccumulation ||
                placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::TF32TensorCore)
                options.matrixPrecision = CFV::ReconstructionMatrixPrecision::FP32;
            DNDS_check_throw_info(_data->settings.numerics.limiter != "LVR" ||
                                      (placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::FP64 ||
                                       placed->reconstructionKernel == CFV::ReconstructionDeviceKernel::FP64Pipelined),
                                  "LVR requires general fp64 or production fp64Pipelined reconstruction");
            placed->reconstruction = _data->reconstruction->Place(device, context, options);
            placed->quadrature.faceBasis = placed->reconstruction->SamplingFaceBasis();
            samplingPlacement.Seed(_data->quadrature.faceBasis, placed->quadrature.faceBasis);
            const auto &cache = _data->reconstruction->Settings();
            const bool directBasis = !cache.cacheDiffBase || cache.cacheDiffBaseOrder < options.basisDerivativeOrder;
            if (directBasis || options.arbitraryPointSampling)
            {
                samplingPlacement.Seed(_data->reconstruction->SamplingMoments(), placed->reconstruction->SamplingMoments());
                samplingPlacement.Seed(_data->reconstruction->SamplingCellGeometry(), placed->reconstruction->SamplingCellGeometry());
            }
            DNDS_check_throw_info(solver.UsesAMG() || placed->reconstruction->SupportsKernel(placed->reconstructionKernel),
                                  "requested CFV reconstruction kernel is unsupported on this CUDA device");
        }
        placed->stencil = _data->stencil.Placed(placed->metrics, context);
        const auto moveField = [&](const auto &source, const char *name)
        {
            auto target = source.To(device, context);
            target.SetName(std::string("Cauchy/") + name, DNDS_ALLOCATION_SITE());
            return target;
        };
        placed->boundaryFaces = moveField(_data->boundaryFaces, "boundary/faces");
        placed->ransCellLength = moveField(_data->ransCellLength, "ransCellLength");
        placed->deviceBoundaryConditions = R::MetadataBuffer<PerfectGasBoundaryParameters>(conditions).CopyTo(device, context);
        placed->deviceBoundaryConditions.SetDiagnosticName("Cauchy/boundaryConditions", DNDS_ALLOCATION_SITE());
        placed->devicePhysics = R::MetadataBuffer<PerfectGasSpatialParameters>({physics}).CopyTo(device, context);
        placed->devicePhysics.SetDiagnosticName("Cauchy/pointPhysics", DNDS_ALLOCATION_SITE());
        placed->deviceBoundaryKinds = R::MetadataBuffer<BoundaryKind>::Generate(conditions.size(),
                                                                                [&](index iCondition)
                                                                                { return conditions[iCondition].kind; })
                                          .CopyTo(device, context);
        placed->deviceBoundaryKinds.SetDiagnosticName("Cauchy/boundaryKinds", DNDS_ALLOCATION_SITE());
        const auto &numerics = _data->settings.numerics;
        const ThermalPolicy thermal(_data->equation, numerics.densityFloor, numerics.temperatureFloor,
                                    numerics.sensibleEnergyFloor, numerics.pressureFloor);
        CUDA::BoundaryKernelParameters boundaryPolicy;
        boundaryPolicy.gas = _data->equation.Gas();
        boundaryPolicy.rans = physics.rans.kind;
        boundaryPolicy.thermal = thermal.View();
        boundaryPolicy.allowPointFallback = numerics.positivity || numerics.faceFallback;
#ifdef DNDS_USE_CUDA
        boundaryPolicy.Validate(5 + physics.rans.ComponentNumber());
#endif
        placed->deviceBoundaryPolicy = R::MetadataBuffer<CUDA::BoundaryKernelParameters>({boundaryPolicy}).CopyTo(device, context);
        placed->deviceBoundaryPolicy.SetDiagnosticName("Cauchy/boundaryPolicy", DNDS_ALLOCATION_SITE());
        placed->deviceEmpty = R::Array<real>(R::RowLayout::Dense(0, 0), device);
        if (_data->wallDistance)
            placed->wallDistance = _data->wallDistance->Placed(device, context);
        if (_data->quadrature.positivity)
            placed->quadrature.positivity = _data->quadrature.positivity->Placed(samplingPlacement);
        if (_data->quadrature.source)
        {
            auto source = std::make_shared<Detail::SourceQuadrature>(*_data->quadrature.source);
            // PP and source must share row boundaries on the placed backend,
            // just as their volume point/basis storage does.
            source->offsets = samplingPlacement.Place(source->offsets);
            source->weights = moveField(source->weights, "sourceWeights");
            if (source->sampling)
                source->sampling = source->sampling->Placed(samplingPlacement);
            if (source->jacobianSampling)
                source->jacobianSampling = source->jacobianSampling->Placed(samplingPlacement);
            placed->quadrature.source = std::move(source);
        }
        if (_data->limitedVRGeometry)
        {
            auto limited = std::make_shared<LimitedVRGeometry>(*_data->limitedVRGeometry);
            limited->gates = _data->limitedVRGeometry->gates.Placed(placed->metrics, context);
            placed->limitedVRGeometry = std::move(limited);
        }
        placed->settings.execution.device = "cuda";
        placed->settings.execution.deviceOrdinal = device.ordinal;
        return FiniteVolume(std::move(placed));
    }
}
