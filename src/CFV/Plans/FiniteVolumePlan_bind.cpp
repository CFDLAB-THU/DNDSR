#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Plans/InterfaceSamplingPlan.hpp"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "CFV/Plans/GradientPlan.hpp"
#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace DNDS::CFV
{
    void pybind11_FiniteVolumePlan_define(pybind11::module_ &m)
    {
        namespace py = pybind11;
        py::enum_<ReconstructionOperation>(m, "ReconstructionOperation")
            .value("Sweep", ReconstructionOperation::Sweep)
            .value("WBAP", ReconstructionOperation::WBAP)
            .value("LinearSystem", ReconstructionOperation::LinearSystem)
            .value("LocalPreconditioner", ReconstructionOperation::LocalPreconditioner)
            .value("InverseDiagnostics", ReconstructionOperation::InverseDiagnostics);
        py::class_<ReconstructionCapabilities>(m, "ReconstructionCapabilities")
            .def(py::init<>())
            .def_static("ForSweeps", &ReconstructionCapabilities::ForSweeps, py::arg("withWBAP") = true)
            .def_readwrite("sweeps", &ReconstructionCapabilities::sweeps)
            .def_readwrite("wbap", &ReconstructionCapabilities::wbap)
            .def_readwrite("linearSystem", &ReconstructionCapabilities::linearSystem)
            .def_readwrite("localPreconditioner", &ReconstructionCapabilities::localPreconditioner)
            .def_readwrite("inverseDiagnostics", &ReconstructionCapabilities::inverseDiagnostics)
            .def("Supports", &ReconstructionCapabilities::Supports);
        py::class_<ReconstructionTableStorage>(m, "ReconstructionTableStorage")
            .def_readonly("name", &ReconstructionTableStorage::name)
            .def_readonly("allocated", &ReconstructionTableStorage::allocated)
            .def_readonly("payloadBytes", &ReconstructionTableStorage::payloadBytes)
            .def_readonly("allocationBytes", &ReconstructionTableStorage::allocationBytes);
        py::class_<ReconstructionStorageReport>(m, "ReconstructionStorageReport")
            .def_readonly("tables", &ReconstructionStorageReport::tables)
            .def_readonly("retainedPayloadBytes", &ReconstructionStorageReport::retainedPayloadBytes)
            .def_readonly("retainedAllocationBytes", &ReconstructionStorageReport::retainedAllocationBytes)
            .def_readonly("preparationRuntimeScratchBytes", &ReconstructionStorageReport::preparationRuntimeScratchBytes)
            .def_readonly("preparationRuntimePeakBytes", &ReconstructionStorageReport::preparationRuntimePeakBytes);
        py::enum_<ReconstructionRegion>(m, "ReconstructionRegion")
            .value("All", ReconstructionRegion::All)
            .value("Core", ReconstructionRegion::Core)
            .value("HaloDependent", ReconstructionRegion::HaloDependent);
        using F = ReconstructionFunctionalSettings;
        py::enum_<F::DirectionalWeight>(m, "ReconstructionDirectionalWeight")
            .value("Factorial", F::DirectionalWeight::Factorial)
            .value("HQM_OPT", F::DirectionalWeight::HQM_OPT)
            .value("TEST_OPT", F::DirectionalWeight::TEST_OPT)
            .value("ManualDirWeight", F::DirectionalWeight::ManualDirWeight);
        py::enum_<F::GeometricWeight>(m, "ReconstructionGeometricWeight")
            .value("GWNone", F::GeometricWeight::GWNone)
            .value("HQM_SD", F::GeometricWeight::HQM_SD)
            .value("SD_Power", F::GeometricWeight::SD_Power);
        py::enum_<F::LengthScale>(m, "ReconstructionLengthScale")
            .value("BaryDiff", F::LengthScale::BaryDiff)
            .value("MeanAACBB", F::LengthScale::MeanAACBB)
            .value("CellMax", F::LengthScale::CellMax);
        py::class_<F>(m, "ReconstructionFunctionalSettings")
            .def(py::init<>())
            .def_readwrite("dirWeightScheme", &F::dirWeightScheme)
            .def_readwrite("geomWeightScheme", &F::geomWeightScheme)
            .def_readwrite("scaleType", &F::scaleType)
            .def_readwrite("manualDirWeights", &F::manualDirWeights)
            .def_readwrite("scaleMultiplier", &F::scaleMultiplier)
            .def_readwrite("geomWeightBias", &F::geomWeightBias)
            .def_readwrite("geomWeightPower", &F::geomWeightPower)
            .def_readwrite("geomWeightPower1", &F::geomWeightPower1)
            .def_readwrite("geomWeightPower2", &F::geomWeightPower2)
            .def_readwrite("dirWeightCombPowV", &F::dirWeightCombPowV)
            .def("Validate", &F::Validate);
        py::class_<ReconstructionStorageSettings>(m, "ReconstructionStorageSettings")
            .def(py::init<>())
            .def_readwrite("representation", &ReconstructionStorageSettings::representation)
            .def_readwrite("inverseTriangle", &ReconstructionStorageSettings::inverseTriangle)
            .def_readwrite("faceCoupling", &ReconstructionStorageSettings::faceCoupling)
            .def_readwrite("aosRowPadding", &ReconstructionStorageSettings::aosRowPadding)
            .def_readwrite("allocationAlignmentBytes", &ReconstructionStorageSettings::allocationAlignmentBytes)
            .def("Validate", &ReconstructionStorageSettings::Validate);
        py::class_<ReconstructionSweepLaunch>(m, "ReconstructionSweepLaunch")
            .def(py::init<>())
            .def_readwrite("threads", &ReconstructionSweepLaunch::threads)
            .def_readwrite("gridBlocks", &ReconstructionSweepLaunch::gridBlocks)
            .def("Validate", &ReconstructionSweepLaunch::Validate);
        py::class_<ReconstructionSettings>(m, "ReconstructionSettings")
            .def(py::init<>())
            .def_readwrite("storage", &ReconstructionSettings::storage)
            .def_readwrite("cudaSweep", &ReconstructionSettings::cudaSweep)
            .def_readwrite("maxOrder", &ReconstructionSettings::maxOrder)
            .def_readwrite("intOrder", &ReconstructionSettings::intOrder)
            .def_readwrite("jacobiRelax", &ReconstructionSettings::jacobiRelax)
            .def_readwrite("SORInstead", &ReconstructionSettings::SORInstead)
            .def_readwrite("coreFirst", &ReconstructionSettings::coreFirst)
            .def_readwrite("HQMWeights", &ReconstructionSettings::HQMWeights)
            .def_readwrite("cacheDiffBase", &ReconstructionSettings::cacheDiffBase)
            .def_readwrite("cacheDiffBaseOrder", &ReconstructionSettings::cacheDiffBaseOrder)
            .def_readwrite("functional", &ReconstructionSettings::functional)
            .def_readwrite("bcWeight", &ReconstructionSettings::bcWeight)
            .def_readwrite("boundaryDerivativeWeights", &ReconstructionSettings::boundaryDerivativeWeights)
            .def_readwrite("boundaryZoneWeights", &ReconstructionSettings::boundaryZoneWeights)
            .def_readwrite("svdTolerance", &ReconstructionSettings::svdTolerance)
            .def_readwrite("svdAbsoluteTolerance", &ReconstructionSettings::svdAbsoluteTolerance)
            .def_readwrite("WBAP_nStd", &ReconstructionSettings::WBAP_nStd)
            .def("Validate", &ReconstructionSettings::Validate);
        // This is the DNDS numerical field, not a second Python registration
        // or a CFV-specific payload owner.
        m.attr("FVField") = py::type::of<FVField>();
        py::class_<ReconstructionInverseDiagnostics>(m, "ReconstructionInverseDiagnostics")
            .def_readonly("cells", &ReconstructionInverseDiagnostics::cells)
            .def_readonly("filteredCells", &ReconstructionInverseDiagnostics::filteredCells)
            .def_readonly("discardedModes", &ReconstructionInverseDiagnostics::discardedModes)
            .def_readonly("rankZeroCells", &ReconstructionInverseDiagnostics::rankZeroCells)
            .def_readonly("maxCondition", &ReconstructionInverseDiagnostics::maxCondition)
            .def_readonly("maxConditionCell", &ReconstructionInverseDiagnostics::maxConditionCell)
            .def_readonly("rankCounts", &ReconstructionInverseDiagnostics::rankCounts);
        py::class_<HardEigen::SVDInverseDiagnostics>(m, "SVDInverseDiagnostics")
            .def_readonly("modes", &HardEigen::SVDInverseDiagnostics::modes)
            .def_readonly("rank", &HardEigen::SVDInverseDiagnostics::rank)
            .def_readonly("relativeCutoff", &HardEigen::SVDInverseDiagnostics::relativeCutoff)
            .def_readonly("rawCondition", &HardEigen::SVDInverseDiagnostics::rawCondition)
            .def_readonly("retainedCondition", &HardEigen::SVDInverseDiagnostics::retainedCondition);
        py::class_<FiniteVolumeMetricOptions>(m, "FiniteVolumeMetricOptions")
            .def(py::init<>())
            .def_readwrite("integrationOrder", &FiniteVolumeMetricOptions::integrationOrder)
            .def_readwrite("faceIntegrationOrder", &FiniteVolumeMetricOptions::faceIntegrationOrder)
            .def_readwrite("axisymmetric", &FiniteVolumeMetricOptions::axisymmetric)
            .def_readwrite("collectivePreflight", &FiniteVolumeMetricOptions::collectivePreflight);
        py::class_<FiniteVolumePlan>(m, "FiniteVolumePlan")
            .def_static("ConstructMetrics", py::overload_cast<ssp<Geom::UnstructuredMesh>, FiniteVolumeMetricOptions>(&FiniteVolumePlan::ConstructMetrics),
                        py::arg("mesh"), py::arg("options"), "Collective metric preparation with named options.")
            .def_static("ConstructMetrics", [](ssp<Geom::UnstructuredMesh> mesh, int intOrder, bool axisymmetric, bool collectivePreflight, int faceIntOrder)
                        {
                            FiniteVolumeMetricOptions options;
                            options.integrationOrder = intOrder;
                            options.axisymmetric = axisymmetric;
                            options.collectivePreflight = collectivePreflight;
                            options.faceIntegrationOrder = faceIntOrder;
                            return FiniteVolumePlan::ConstructMetrics(std::move(mesh), options); }, py::arg("mesh"), py::arg("intOrder") = 5, py::arg("axisymmetric") = false, py::kw_only(), py::arg("collective_preflight") = true, py::arg("faceIntOrder") = 0, "Collective CPU preparation. All ranks must call with matching settings.")
            .def("Validate", &FiniteVolumePlan::Validate)
            .def("Device", &FiniteVolumePlan::Device)
            .def("Placed", &FiniteVolumePlan::Placed)
            .def("NumCell", &FiniteVolumePlan::NumCell)
            .def("NumCellProc", &FiniteVolumePlan::NumCellProc)
            .def("NumFaceProc", &FiniteVolumePlan::NumFaceProc)
            .def("Axisymmetric", &FiniteVolumePlan::Axisymmetric)
            .def("BuildField", py::overload_cast<index>(&FiniteVolumePlan::BuildField, py::const_))
            .def("BuildFieldWithLayout", py::overload_cast<Runtime::RowLayout>(&FiniteVolumePlan::BuildField, py::const_), py::arg("prototype"))
            .def("BuildFieldWithLayout", py::overload_cast<Runtime::RowLayout, Runtime::Device>(&FiniteVolumePlan::BuildField, py::const_), py::arg("prototype"), py::arg("device"))
            .def("BindHalo", &FiniteVolumePlan::BindHalo, py::arg("field"), py::arg("options") = Runtime::TransferOptions{}, "Collective persistent direct MPI binding; explicitly begin/wait on each field stage.")
            .def("GetCellVol", &FiniteVolumePlan::GetCellVol)
            .def("GetCellCent", &FiniteVolumePlan::GetCellCent)
            .def("BuildCellMaxLenScale", &FiniteVolumePlan::BuildCellMaxLenScale, "CPU setup: derive Euler-compatible principal-axis maximum lengths for local metric rows; rebuild after mesh changes.");
        py::class_<ReconstructionBoundaryValues>(m, "ReconstructionBoundaryValues")
            .def_property_readonly("values", [](ReconstructionBoundaryValues &value)
                                   { return Runtime::PythonArray::Share(value.values); });
        py::class_<BoundaryQuadraturePoint>(m, "BoundaryQuadraturePoint")
            .def_readonly("cell", &BoundaryQuadraturePoint::cell)
            .def_readonly("face", &BoundaryQuadraturePoint::face)
            .def_readonly("quadrature", &BoundaryQuadraturePoint::quadrature)
            .def_readonly("point", &BoundaryQuadraturePoint::point)
            .def_readonly("normal", &BoundaryQuadraturePoint::normal);
        py::class_<FaceCharacteristicTransform>(m, "FaceCharacteristicTransform")
            .def_readonly("first", &FaceCharacteristicTransform::first)
            .def_readonly("count", &FaceCharacteristicTransform::count)
            .def_property_readonly("forward", [](FaceCharacteristicTransform &value)
                                   { return Runtime::PythonArray::Share(value.forward); })
            .def_property_readonly("inverse", [](FaceCharacteristicTransform &value)
                                   { return Runtime::PythonArray::Share(value.inverse); });
        py::class_<ReconstructionWorkspace>(m, "ReconstructionWorkspace")
            .def_readonly("temporary", &ReconstructionWorkspace::temporary);
        py::enum_<ReconstructionDeviceKernel>(m, "ReconstructionDeviceKernel")
            .value("FP64", ReconstructionDeviceKernel::FP64)
            .value("FP64MatrixShared", ReconstructionDeviceKernel::FP64MatrixShared)
            .value("FP64Pipelined", ReconstructionDeviceKernel::FP64Pipelined)
            .value("FloatFloatPipelined", ReconstructionDeviceKernel::FloatFloatPipelined)
            .value("FP32Pipelined", ReconstructionDeviceKernel::FP32Pipelined)
            .value("TF32Pipelined", ReconstructionDeviceKernel::TF32Pipelined)
            .value("FP64TensorCoreAsync", ReconstructionDeviceKernel::FP64TensorCoreAsync)
            .value("FP32", ReconstructionDeviceKernel::FP32)
            .value("FP32FullAccumulation", ReconstructionDeviceKernel::FP32FullAccumulation)
            .value("FP32FloatFloatAccumulation", ReconstructionDeviceKernel::FP32FloatFloatAccumulation)
            .value("TF32TensorCore", ReconstructionDeviceKernel::TF32TensorCore);
        py::enum_<ReconstructionMatrixPrecision>(m, "ReconstructionMatrixPrecision")
            .value("FP64", ReconstructionMatrixPrecision::FP64)
            .value("FP32", ReconstructionMatrixPrecision::FP32);
        py::class_<ReconstructionResidencyOptions>(m, "ReconstructionResidencyOptions")
            .def(py::init<>())
            .def_readwrite("mesh", &ReconstructionResidencyOptions::mesh)
            .def_readwrite("metrics", &ReconstructionResidencyOptions::metrics)
            .def_readwrite("basis", &ReconstructionResidencyOptions::basis)
            .def_readwrite("retain_host_payloads", &ReconstructionResidencyOptions::retainHostPayloads)
            .def_readwrite("sweeps", &ReconstructionResidencyOptions::sweeps)
            .def_readwrite("arbitrary_point_sampling", &ReconstructionResidencyOptions::arbitraryPointSampling)
            .def_readwrite("basis_derivative_order", &ReconstructionResidencyOptions::basisDerivativeOrder)
            .def_readwrite("matrix_precision", &ReconstructionResidencyOptions::matrixPrecision);
        py::class_<ResidentBufferInfo>(m, "ResidentBufferInfo")
            .def_readonly("name", &ResidentBufferInfo::name)
            .def_readonly("bytes", &ResidentBufferInfo::bytes)
            .def_readonly("device", &ResidentBufferInfo::device);
        py::class_<DistributedSamplingWorkspace>(m, "DistributedSamplingWorkspace")
            .def_property_readonly("donorValues", [](DistributedSamplingWorkspace &work)
                                   { return Runtime::PythonArray::Share(work.donorValues); });
        py::class_<PolynomialSamplingPlan>(m, "PolynomialSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, const Runtime::Array<index> &, const Runtime::Array<real> &>())
            .def("Size", &PolynomialSamplingPlan::Size)
            .def("Validate", &PolynomialSamplingPlan::Validate)
            .def("Device", &PolynomialSamplingPlan::Device)
            .def("Placed", py::overload_cast<Runtime::Device, Runtime::ExecutionContext>(&PolynomialSamplingPlan::Placed, py::const_))
            .def("Evaluate", py::overload_cast<const FVField &, const FVField &, Runtime::Array<real> &, int>(&PolynomialSamplingPlan::Evaluate, py::const_), py::arg("mean"), py::arg("coefficients"),
                 py::arg("output"), py::arg("derivative") = 0)
            .def("EvaluateQueued", py::overload_cast<Runtime::EagerEngine &, const Runtime::ExecutionStream &, const FVField &, const FVField &, Runtime::Array<real> &, int>(&PolynomialSamplingPlan::Evaluate, py::const_),
                 py::arg("engine"), py::arg("stream"), py::arg("mean"), py::arg("coefficients"), py::arg("output"), py::arg("derivative") = 0);
        py::class_<VariationalReconstructionPlan>(m, "VariationalReconstructionPlan")
            .def_static("ConstructRecCoeff", [](FiniteVolumePlan metrics, ReconstructionSettings settings, ReconstructionCapabilities capabilities)
                        { return VariationalReconstructionPlan::ConstructRecCoeff(std::move(metrics), std::move(settings), capabilities); }, py::arg("metrics"), py::arg("settings") = ReconstructionSettings{}, py::arg("capabilities") = ReconstructionCapabilities{})
            .def("NRecDOF", &VariationalReconstructionPlan::NRecDOF)
            .def("Capabilities", &VariationalReconstructionPlan::Capabilities)
            .def("Require", &VariationalReconstructionPlan::Require)
            .def("StorageReport", &VariationalReconstructionPlan::StorageReport)
            .def("GetInverseDiagnostics", &VariationalReconstructionPlan::GetInverseDiagnostics)
            .def("GetCellInverseDiagnostics", &VariationalReconstructionPlan::GetCellInverseDiagnostics)
            .def("Validate", &VariationalReconstructionPlan::Validate)
            .def("Place", &VariationalReconstructionPlan::Place, py::arg("device"), py::arg("context"), py::arg("options") = ReconstructionResidencyOptions{}, "Return the same logical CFV plan placed on the requested backend. CUDA setup is explicit and synchronous.")
            .def("Device", &VariationalReconstructionPlan::Device)
            .def("RetainsHostPayloads", &VariationalReconstructionPlan::RetainsHostPayloads)
            .def("SupportsKernel", &VariationalReconstructionPlan::SupportsKernel)
            .def("ResidentBuffers", &VariationalReconstructionPlan::ResidentBuffers)
            .def("BuildWorkspace", py::overload_cast<index, std::vector<index>>(&VariationalReconstructionPlan::BuildWorkspace, py::const_), py::arg("components"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("BuildWorkspaceWithLayout", py::overload_cast<index, Runtime::RowLayout, std::vector<index>>(&VariationalReconstructionPlan::BuildWorkspace, py::const_), py::arg("components"), py::arg("prototype"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("BuildCoefficients", py::overload_cast<index>(&VariationalReconstructionPlan::BuildCoefficients, py::const_), py::arg("components"))
            .def("BuildCoefficientsWithLayout", py::overload_cast<index, Runtime::RowLayout>(&VariationalReconstructionPlan::BuildCoefficients, py::const_), py::arg("components"), py::arg("prototype"))
            .def("BuildBoundaryValues", &VariationalReconstructionPlan::BuildBoundaryValues)
            .def("PlaceBoundaryValues", &VariationalReconstructionPlan::PlaceBoundaryValues)
            .def("NumBoundaryQuadrature", &VariationalReconstructionPlan::NumBoundaryQuadrature)
            .def("GetBoundaryQuadrature", &VariationalReconstructionPlan::GetBoundaryQuadrature)
            .def("BuildFaceTransform", &VariationalReconstructionPlan::BuildFaceTransform)
            .def("PlaceFaceTransform", &VariationalReconstructionPlan::PlaceFaceTransform)
            .def("FinalizeFaceTransform", &VariationalReconstructionPlan::FinalizeFaceTransform, py::arg("transform"), py::arg("checkInverses") = false, "Publish matrices after filling from cell means; optional inverse diagnostics are off by default. Any later write requires re-finalization.")
            .def("DoReconstructionIter", &VariationalReconstructionPlan::DoReconstructionIter, py::arg("mean"), py::arg("coefficients"), py::arg("workspace"), py::arg("boundary"), py::arg("kernel") = ReconstructionDeviceKernel::FP64, "One synchronous backend-resolved sweep and commit. Release Python aliases and refresh input halos first.")
            .def("DoReconstructionIterTo", &VariationalReconstructionPlan::DoReconstructionIterTo, py::arg("mean"), py::arg("input"), py::arg("workspace"), py::arg("boundary"), py::arg("region") = ReconstructionRegion::All)
            .def("SubmitReconstructionIterTo", &VariationalReconstructionPlan::SubmitReconstructionIterTo, py::arg("engine"), py::arg("stream"), py::arg("mean"), py::arg("input"), py::arg("workspace"), py::arg("boundary"), py::arg("region") = ReconstructionRegion::All, py::arg("kernel") = ReconstructionDeviceKernel::FP64, "Enqueue one backend-resolved reconstruction sweep. The operation retains all Array owners through completion.")
            .def("DoReconstructionIterLimitedTo", &VariationalReconstructionPlan::DoReconstructionIterLimitedTo, py::arg("mean"), py::arg("input"), py::arg("workspace"), py::arg("boundary"), py::arg("target"), py::arg("alpha"), py::arg("region") = ReconstructionRegion::All, "CPU A-weighted penalty sweep. Freeze finite target and alpha in [0,1]; refresh input halos before dependent work and commit only after all rows finish.")
            .def("CommitReconstruction", &VariationalReconstructionPlan::CommitReconstruction)
            .def("SubmitCommitReconstruction", &VariationalReconstructionPlan::SubmitCommitReconstruction, py::arg("engine"), py::arg("stream"), py::arg("coefficients"), py::arg("workspace"), "Enqueue temporary-to-coefficient publication on the selected backend stream.")
            .def("DoLimiterWBAP_3Order", [](const VariationalReconstructionPlan &plan, FVField &coefficients, ReconstructionWorkspace &work, int order, const FaceCharacteristicTransform *transform)
                 { plan.DoLimiterWBAP_3Order(coefficients, work, order, transform ? FaceCharacteristicTransformRef(std::cref(*transform)) : std::nullopt); }, py::arg("coefficients"), py::arg("workspace"), py::arg("order"), py::arg("transform") = nullptr, "Call in descending order with a coefficient halo exchange after each order.")
            .def("DoLimiterWBAP_3OrderTo", [](const VariationalReconstructionPlan &plan, const FVField &coefficients, ReconstructionWorkspace &work, int order, const FaceCharacteristicTransform *transform, ReconstructionRegion region, const Runtime::Array<real> *indicator, real threshold)
                 { plan.DoLimiterWBAP_3OrderTo(coefficients, work, order,
                                               transform ? FaceCharacteristicTransformRef(std::cref(*transform)) : std::nullopt,
                                               region, indicator ? LimiterIndicatorRef(std::cref(*indicator)) : std::nullopt,
                                               threshold); }, py::arg("coefficients"), py::arg("workspace"), py::arg("order"), py::arg("transform") = nullptr, py::arg("region") = ReconstructionRegion::All, py::arg("indicator") = nullptr, py::arg("threshold") = 0)
            .def("SubmitLimiterWBAP_3OrderTo", [](const VariationalReconstructionPlan &plan, Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, const FVField &coefficients, ReconstructionWorkspace &work, int order, const FaceCharacteristicTransform *transform, ReconstructionRegion region, const Runtime::Array<real> *indicator, real threshold)
                 { plan.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order,
                                                   transform ? FaceCharacteristicTransformRef(std::cref(*transform)) : std::nullopt,
                                                   region, indicator ? LimiterIndicatorRef(std::cref(*indicator)) : std::nullopt,
                                                   threshold); }, py::arg("engine"), py::arg("stream"), py::arg("coefficients"), py::arg("workspace"), py::arg("order"), py::arg("transform") = nullptr, py::arg("region") = ReconstructionRegion::All, py::arg("indicator") = nullptr, py::arg("threshold") = 0, "Enqueue one descending limiter order through the common CFV plan.")
            .def("BindHalo", &VariationalReconstructionPlan::BindHalo, py::arg("field"), py::arg("transport") = Runtime::TransferTransport::Direct, py::arg("options") = Runtime::TransferOptions{}, "Collective payload binding for this plan placement; explicitly order consumers after transfer completion.")
            .def("CheckStatus", &VariationalReconstructionPlan::CheckStatus)
            .def("Evaluate", &VariationalReconstructionPlan::Evaluate)
            .def("EvaluateDerivative", &VariationalReconstructionPlan::EvaluateDerivative)
            .def("GetCachedFaceBasis", &VariationalReconstructionPlan::GetCachedFaceBasis)
            .def("ApplyLinearOperator", &VariationalReconstructionPlan::ApplyLinearOperator, py::arg("coefficients"), py::arg("output"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("BuildRightHandSide", &VariationalReconstructionPlan::BuildRightHandSide, py::arg("mean"), py::arg("boundary"), py::arg("output"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("ComputeResidual", &VariationalReconstructionPlan::ComputeResidual, py::arg("rhs"), py::arg("coefficients"), py::arg("output"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("ApplyLocalPreconditioner", &VariationalReconstructionPlan::ApplyLocalPreconditioner);
        py::class_<InterfaceSamplingPlan>(m, "InterfaceSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, Geom::InterfacePointMap>())
            .def("Size", &InterfaceSamplingPlan::Size)
            .def("Evaluate", &InterfaceSamplingPlan::Evaluate,
                 "Sample donor-frame values in map query order. The equation adapter transforms vector components.");
        py::enum_<GradientMethod>(m, "GradientMethod")
            .value("GreenGauss", GradientMethod::GreenGauss)
            .value("LeastSquares", GradientMethod::LeastSquares)
            .value("GGMP", GradientMethod::GGMP);
        py::class_<GradientWorkspace>(m, "GradientWorkspace");
        py::class_<GradientFaceCharacteristicTransform>(m, "GradientFaceCharacteristicTransform")
            .def_readonly("first", &GradientFaceCharacteristicTransform::first)
            .def_readonly("count", &GradientFaceCharacteristicTransform::count)
            .def_property_readonly("forward", [](GradientFaceCharacteristicTransform &value)
                                   { return Runtime::PythonArray::Share(value.forward); })
            .def_property_readonly("inverse", [](GradientFaceCharacteristicTransform &value)
                                   { return Runtime::PythonArray::Share(value.inverse); });
        py::class_<GradientPlan>(m, "GradientPlan")
            .def_static("Prepare", [](Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics, GradientMethod method, real cutoff, real areaPower, real distancePower, const std::vector<int> &extrapolatedZones)
                        { return GradientPlan::Prepare(mesh, std::move(metrics), method, cutoff, {}, {areaPower, distancePower}, extrapolatedZones); }, py::arg("mesh"), py::arg("metrics"), py::arg("method") = GradientMethod::LeastSquares, py::arg("relativeCutoff") = 1e-13, py::arg("areaPower") = 0, py::arg("distancePower") = -3, py::arg("extrapolatedZones") = std::vector<int>{})
            .def("NumBoundaryQuadrature", &GradientPlan::NumBoundaryQuadrature)
            .def("GetBoundaryQuadrature", &GradientPlan::GetBoundaryQuadrature)
            .def("Device", &GradientPlan::Device)
            .def("Placed", py::overload_cast<Runtime::Device, Runtime::ExecutionContext>(&GradientPlan::Placed, py::const_))
            .def("BuildWorkspace", &GradientPlan::BuildWorkspace, py::arg("components"), py::arg("periodicVectorStarts") = std::vector<index>{}, py::arg("withLimiter") = false)
            .def("BuildFaceTransform", &GradientPlan::BuildFaceTransform)
            .def("FinalizeFaceTransform", &GradientPlan::FinalizeFaceTransform, py::arg("transform"), py::arg("checkInverses") = false)
            .def("PlaceFaceTransform", &GradientPlan::PlaceFaceTransform)
            .def("BuildBoundaryValues", [](const GradientPlan &plan, index components)
                 { return Runtime::PythonArray::Share(plan.BuildBoundaryValues(components)); })
            .def("Evaluate", py::overload_cast<const FVField &, const Runtime::Array<real> &, FVField &, const std::vector<index> &, ReconstructionRegion>(&GradientPlan::Evaluate, py::const_), py::arg("mean"), py::arg("boundaryValues"), py::arg("gradient"), py::arg("periodicVectorStarts") = std::vector<index>{}, py::arg("region") = ReconstructionRegion::All)
            .def("EvaluateQueued", py::overload_cast<Runtime::EagerEngine &, const Runtime::ExecutionStream &, const FVField &, const Runtime::Array<real> &, FVField &, const GradientWorkspace &, ReconstructionRegion>(&GradientPlan::Evaluate, py::const_), py::arg("engine"), py::arg("stream"), py::arg("mean"), py::arg("boundaryValues"), py::arg("gradient"), py::arg("workspace"), py::arg("region") = ReconstructionRegion::All)
            .def("ConvertToCoefficients", py::overload_cast<const FVField &, const VariationalReconstructionPlan &, FVField &>(&GradientPlan::ConvertToCoefficients, py::const_))
            .def("ConvertToCoefficientsQueued", py::overload_cast<Runtime::EagerEngine &, const Runtime::ExecutionStream &, const FVField &, const VariationalReconstructionPlan &, FVField &>(&GradientPlan::ConvertToCoefficients, py::const_))
            .def("ExtractLinearGradients", py::overload_cast<const FVField &, const VariationalReconstructionPlan &, FVField &, bool>(&GradientPlan::ExtractLinearGradients, py::const_), py::arg("coefficients"), py::arg("reconstruction"), py::arg("gradient"), py::arg("ghost") = false)
            .def("ExtractLinearGradientsQueued", py::overload_cast<Runtime::EagerEngine &, const Runtime::ExecutionStream &, const FVField &, const VariationalReconstructionPlan &, FVField &, bool>(&GradientPlan::ExtractLinearGradients, py::const_), py::arg("engine"), py::arg("stream"), py::arg("coefficients"), py::arg("reconstruction"), py::arg("gradient"), py::arg("ghost") = false)
            .def("LimitWBAP3", [](const GradientPlan &plan, const FVField &input, FVField &output, const std::vector<index> &vectors, const std::optional<GradientFaceCharacteristicTransform> &transform, ReconstructionRegion region, const std::optional<Runtime::Array<real>> &indicator, real threshold, real centerWeight)
                 { plan.LimitWBAP3(input, output, vectors, transform ? GradientFaceCharacteristicTransformRef(std::cref(*transform)) : std::nullopt,
                                   region, indicator ? LimiterIndicatorRef(std::cref(*indicator)) : std::nullopt, threshold, centerWeight); }, py::arg("input"), py::arg("output"), py::arg("periodicVectorStarts") = std::vector<index>{}, py::arg("transform") = std::nullopt, py::arg("region") = ReconstructionRegion::All, py::arg("indicator") = std::nullopt, py::arg("threshold") = 0, py::arg("centerWeight") = 10)
            .def("LimitWBAP3Queued", [](const GradientPlan &plan, Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, const FVField &input, FVField &output, GradientWorkspace &workspace, const std::optional<GradientFaceCharacteristicTransform> &transform, ReconstructionRegion region, const std::optional<Runtime::Array<real>> &indicator, real threshold, real centerWeight)
                 { plan.LimitWBAP3(engine, stream, input, output, workspace, transform ? GradientFaceCharacteristicTransformRef(std::cref(*transform)) : std::nullopt,
                                   region, indicator ? LimiterIndicatorRef(std::cref(*indicator)) : std::nullopt, threshold, centerWeight); }, py::arg("engine"), py::arg("stream"), py::arg("input"), py::arg("output"), py::arg("workspace"), py::arg("transform") = std::nullopt, py::arg("region") = ReconstructionRegion::All, py::arg("indicator") = std::nullopt, py::arg("threshold") = 0, py::arg("centerWeight") = 10);
        py::class_<DistributedInterfaceSamplingPlan>(m, "DistributedInterfaceSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, Geom::DistributedInterfacePointMap>())
            .def(py::init<std::optional<VariationalReconstructionPlan>, Geom::DistributedInterfacePointMap>(),
                 "Coupling ranks outside the donor domain pass None; they retain query/transfer state only.")
            .def("Size", &DistributedInterfaceSamplingPlan::Size)
            .def("Validate", &DistributedInterfaceSamplingPlan::Validate)
            .def("BuildWorkspace", &DistributedInterfaceSamplingPlan::BuildWorkspace)
            .def("Evaluate", &DistributedInterfaceSamplingPlan::Evaluate,
                 "Collective owner evaluation and direct return transfer. Release views and finish all producers first.")
            .def("EvaluateCoupled", [](const DistributedInterfaceSamplingPlan &plan, const std::optional<FVField> &mean, const std::optional<FVField> &coefficients, DistributedSamplingWorkspace &workspace, Runtime::Array<real> &values)
                 {
                     // Preserve Python's two optional arguments, but cross the
                     // native operator boundary with one owning donor bundle.
                     plan.Communication()->Preflight(bool(mean) == bool(coefficients), "donor mean and coefficients must both be present or absent");
                     plan.EvaluateCoupled(mean ? std::make_optional(DonorInputs(*mean, *coefficients)) : std::nullopt, workspace, values); }, py::arg("mean"), py::arg("coefficients"), py::arg("workspace"), py::arg("values"), "Collective coupling evaluation. Both inputs must be None outside the donor domain; all ranks participate and retain buffers through completion.");
    }
}
