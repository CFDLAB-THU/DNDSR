#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "CFV/Detail/CharacteristicKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include <set>

namespace DNDS::CFV
{
    namespace R = Runtime;
    namespace Detail
    {
        struct ReconstructionPlacementData
        {
            std::shared_ptr<const ReconstructionIdentity> identity;
            R::Device device;
            ReconstructionResidencyOptions options;
            ReconstructionSettings settings;
            ReconstructionCapabilities capabilities;
            R::VersionStamp geometryStamp;
            ReconstructionInverseDiagnostics inverseDiagnostics;
            int n = 0, dimension = 0;
            index boundaryPoints = 0, faces = 0;
            bool fp64TensorCore = false;
            bool tf32TensorCore = false;
            std::optional<Algebra::DofSpace> dofs;
            std::shared_ptr<R::CommunicationContext> communication;
            std::optional<R::RowTransferPlan> halo;
            std::vector<ResidentBufferInfo> buffers;
            std::vector<R::Array<real>> realArrays;
            std::vector<R::Array<float>> floatArrays;
            std::vector<R::Array<index>> indexArrays;
            std::vector<R::Array<uint8_t>> byteArrays;
            // Immutable device read leases are retained with the publication;
            // scheduled commands retain the publication through completion.
            std::vector<R::Array<real>::ReadLease> realReads;
            std::vector<R::Array<float>::ReadLease> floatReads;
            std::vector<R::Array<index>::ReadLease> indexReads;
            std::vector<R::Array<uint8_t>::ReadLease> byteReads;
            R::Array<real> empty;
            ReconstructionDeviceView view;
            FiniteVolumeHostView metricView;
            ReconstructionBasisHostView basisView;
            ReconstructionPointBasisView pointBasisView;
            std::optional<R::Array<real>> samplingMoments, samplingCellGeometry;
            std::optional<R::Array<real>> samplingFaceBasis;
            std::shared_ptr<Geom::UnstructuredMeshSnapshot> mesh;
            ReconstructionPlacementData(const std::shared_ptr<const ReconstructionData> &source, R::Device placement,
                                        ReconstructionResidencyOptions requested)
                : identity(source->identity), device(placement), options(requested), settings(source->settings),
                  capabilities(source->capabilities), inverseDiagnostics(source->inverseDiagnostics), n(source->n),
                  boundaryPoints(source->boundaryPoints.RowCount()),
                  empty(R::RowLayout::Dense(0, 0), placement)
            {
                // The placed publication contains sweep/limiter tables only;
                // retaining a host preparation object is not device support.
                capabilities.linearSystem = capabilities.localPreconditioner = false;
            }
            template <class T>
            R::ArrayView<const T> Add(const char *name, const R::Array<T> &input, R::ExecutionContext context, R::AllocationOptions allocation = {})
            {
                // Explicit placement preserves requested base alignment; To()
                // retains shape/pitch but intentionally uses provider defaults.
                R::Array<T> array(input.Layout(), device, allocation);
                array.CopyFrom(input, context).Wait();
                array.SetName(std::string("CFV/") + name, DNDS_ALLOCATION_SITE());
                buffers.push_back({name, array.Storage().Bytes(), device});
                if constexpr (std::is_same_v<T, real>)
                {
                    realArrays.push_back(std::move(array));
                    realReads.push_back(realArrays.back().Read({device, nullptr}));
                    return realReads.back().View();
                }
                else if constexpr (std::is_same_v<T, float>)
                {
                    floatArrays.push_back(std::move(array));
                    floatReads.push_back(floatArrays.back().Read({device, nullptr}));
                    return floatReads.back().View();
                }
                else if constexpr (std::is_same_v<T, index>)
                {
                    indexArrays.push_back(std::move(array));
                    indexReads.push_back(indexArrays.back().Read({device, nullptr}));
                    return indexReads.back().View();
                }
                else
                {
                    static_assert(std::is_same_v<T, uint8_t>);
                    byteArrays.push_back(std::move(array));
                    byteReads.push_back(byteArrays.back().Read({device, nullptr}));
                    return byteReads.back().View();
                }
            }
            R::ArrayView<const float> AddQuantized(const char *name, const R::Array<real> &input, R::ExecutionContext context, R::AllocationOptions allocation = {})
            {
                // Setup-only conversion after FP64 assembly/SVD/inversion. A
                // quantized placement does not keep a second device FP64 copy.
                R::Array<float> converted(input.Layout(), R::Device::Host(), allocation);
                {
                    const auto read = input.Read();
                    auto write = converted.Write();
                    const auto source = read.View();
                    const auto target = write.View();
                    // CPU setup rows are disjoint; invalid casts are reported
                    // on the controller only after every worker has joined.
                    OMP::ForSetupRows(0, input.RowCount(), [&](index iRow)
                                      {
                        for (index j = 0; j < source.RowSize(iRow); ++j)
                        {
                            const float value = float(source(iRow, j));
                            DNDS_check_throw_info(std::isfinite(value), "reconstruction matrix cannot be represented in FP32");
                            target(iRow, j) = value;
                        } });
                }
                return Add(name, converted, context, allocation);
            }
        };
    }
    namespace
    {
        void ValidateDeviceField(const VariationalReconstructionPlan &plan, const FVField &field, index width)
        {
            field.Validate();
            DNDS_check_throw_info(field.Device() == plan.Device() && index(field.Width()) == width, "device FV field width/placement mismatch");
        }
        void ValidateDeviceWorkspace(const std::shared_ptr<const Detail::ReconstructionPlacementData> &data, const ReconstructionWorkspace &work)
        {
            const auto n = data->view.n;
            DNDS_check_throw_info(work.devicePlan == data && work.components > 0 && work.components <= INT_MAX / n &&
                                      work.deviceBlocks > 0 && work.deviceBlocks <= 256,
                                  "device workspace belongs to another publication or has an invalid shape");
            work.temporary.Validate();
            DNDS_check_throw_info(work.temporary.Device() == data->device &&
                                      work.temporary.Width() == n * work.components &&
                                      work.temporary.Space().Same(*data->dofs),
                                  "device workspace fields invalid");
            const index scratchRows = data->capabilities.wbap ? work.deviceBlocks * 6 : 0;
            DNDS_check_throw_info(work.scratch.GetDevice() == data->device && work.scratch.Layout().Kind() == R::RowLayoutKind::Table && work.scratch.RowCount() == scratchRows &&
                                      work.scratch.Layout().Width() == n * work.components && work.deviceStatus.GetDevice() == data->device && work.deviceStatus.Layout().Kind() == R::RowLayoutKind::Table &&
                                      work.deviceStatus.RowCount() == 1 && work.deviceStatus.Layout().Width() == 1 &&
                                      work.devicePeriodicVectorStarts.Identity() == work.devicePeriodicIdentity &&
                                      work.devicePeriodicVectorStarts.Version() == work.devicePeriodicVersion &&
                                      work.devicePeriodicVectorStarts.GetDevice() == data->device &&
                                      work.devicePeriodicVectorStarts.Layout().Kind() == R::RowLayoutKind::Table &&
                                      work.devicePeriodicVectorStarts.Layout().Width() == 1,
                                  "device workspace output/scratch/status mismatch");
        }
        void ValidateDeviceWorkspace(const std::shared_ptr<const Detail::ReconstructionPlacementData> &data,
                                     const ReconstructionWorkspace &work, const FVField &coefficients)
        {
            ValidateDeviceWorkspace(data, work);
            coefficients.Validate();
            DNDS_check_throw_info(coefficients.Device() == data->device &&
                                      coefficients.Width() == data->view.n * work.components &&
                                      coefficients.Space().Same(work.temporary.Space()) &&
                                      !Algebra::Detail::Overlaps(coefficients.Owned().Storage(), work.temporary.Owned().Storage()),
                                  "device coefficient state and scratch are incompatible or alias");
        }
        Detail::DeviceWorkSelection Selection(const Detail::ReconstructionPlacementData &data, ReconstructionRegion region)
        {
            if (region == ReconstructionRegion::All)
                return {{}, data.view.owned, false};
            DNDS_check_throw_info(region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent, "invalid device work region");
            const auto rows = region == ReconstructionRegion::Core ? data.view.core : data.view.dependent;
            return {rows, rows.rows, true};
        }
        void ValidateDeviceStream(const VariationalReconstructionPlan &plan, const R::ExecutionStream &stream)
        {
            plan.Validate();
            DNDS_check_throw_info(stream.Context().device == plan.Device(), "device operator stream placement mismatch");
        }
        void ValidateArray(const R::Array<real> &array, R::Device device, index rows, index width)
        {
            DNDS_check_throw_info(array.GetDevice() == device && array.IsValid() && !array.Layout().IsVariable() && array.RowCount() == rows &&
                                      array.Layout().Width() == width,
                                  "device operator array shape/placement mismatch");
        }
    }
    VariationalReconstructionPlan::VariationalReconstructionPlan(
        std::shared_ptr<const Detail::ReconstructionData> data,
        std::shared_ptr<const Detail::ReconstructionPlacementData> device)
        : _data(std::move(data)), _device(std::move(device))
    {
        _identity = _data ? _data->identity : (_device ? _device->identity : nullptr);
        DNDS_check_throw_info(!_data || !_device || _data->identity == _device->identity,
                              "host and device reconstruction publications differ");
    }
    const ReconstructionSettings &VariationalReconstructionPlan::PlacementSettings() const
    {
        DNDS_check_throw_info(_device, "reconstruction plan has no device placement");
        return _device->settings;
    }
    ReconstructionCapabilities VariationalReconstructionPlan::PlacementCapabilities() const
    {
        DNDS_check_throw_info(_device, "reconstruction plan has no device placement");
        return _device->capabilities;
    }
    ReconstructionStorageReport VariationalReconstructionPlan::PlacementStorageReport() const
    {
        DNDS_check_throw_info(_device, "reconstruction plan has no device placement");
        ReconstructionStorageReport result;
        for (const auto &buffer : _device->buffers)
        {
            result.tables.push_back({buffer.name, true, buffer.bytes, buffer.bytes});
            result.retainedPayloadBytes += buffer.bytes;
            result.retainedAllocationBytes += buffer.bytes;
        }
        result.preparationRuntimePeakBytes = result.retainedAllocationBytes;
        return result;
    }
    ReconstructionInverseDiagnostics VariationalReconstructionPlan::PlacementInverseDiagnostics() const
    {
        DNDS_check_throw_info(_device, "reconstruction plan has no device placement");
        return _device->inverseDiagnostics;
    }
    int VariationalReconstructionPlan::PlacementNRecDOF() const
    {
        DNDS_check_throw_info(_device && _device->n > 0, "reconstruction plan has no valid device placement");
        return _device->n;
    }
    index VariationalReconstructionPlan::PlacementOwnedRows() const
    {
        DNDS_check_throw_info(_device && _device->dofs, "device reconstruction row space is unavailable");
        return index(_device->dofs->OwnedRows());
    }
    index VariationalReconstructionPlan::PlacementBoundaryPoints() const
    {
        DNDS_check_throw_info(_device, "reconstruction plan has no device placement");
        return _device->boundaryPoints;
    }
    FVField VariationalReconstructionPlan::BuildDeviceField(R::RowLayout prototype) const
    {
        DNDS_check_throw_info(_device && _device->dofs && !prototype.IsVariable() && prototype.Width() > 0,
                              "invalid device reconstruction field prototype");
        return FVField(*_device->dofs, std::move(prototype), _device->device);
    }
    void VariationalReconstructionPlan::ValidatePlacement() const
    {
        DNDS_check_throw_info(_device && _identity == _device->identity && _device->device != R::Device::Host() &&
                                  _device->dofs && _device->communication && _device->halo &&
                                  _device->n > 0 && (_device->dimension == 2 || _device->dimension == 3) && _device->faces >= 0 &&
                                  _device->view.n == _device->n &&
                                  _device->view.owned == index(_device->dofs->OwnedRows()) &&
                                  _device->realArrays.size() == _device->realReads.size() &&
                                  _device->indexArrays.size() == _device->indexReads.size(),
                              "device reconstruction publication is incomplete or inconsistent");
        // Dropping host payloads must not drop cache invalidation. Device
        // storage is immutable, but still represents this source generation.
        _device->geometryStamp.Validate();
        for (const auto &array : _device->realArrays)
            DNDS_check_throw_info(array.IsValid() && array.GetDevice() == _device->device,
                                  "device reconstruction real payload is invalid");
        for (const auto &array : _device->floatArrays)
            DNDS_check_throw_info(array.IsValid() && array.GetDevice() == _device->device,
                                  "device reconstruction quantized payload is invalid");
        for (const auto &array : _device->indexArrays)
            DNDS_check_throw_info(array.IsValid() && array.GetDevice() == _device->device,
                                  "device reconstruction index payload is invalid");
    }
    VariationalReconstructionPlan VariationalReconstructionPlan::Place(R::Device device, R::ExecutionContext context,
                                                                       ReconstructionResidencyOptions options) const
    {
        DNDS_check_throw_info(_data, "placing a reconstruction plan requires retained host payloads");
        if (device == R::Device::Host())
        {
            DNDS_check_throw_info(context.device == device, "host reconstruction placement/context differs");
            Validate();
            return *this;
        }
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(false, "CUDA reconstruction is unavailable in this build");
#endif
        Require(ReconstructionOperation::Sweep);
        Validate();
        DNDS_check_throw_info(device.space == R::MemorySpace::CUDA && context.device == device, "device reconstruction requires explicit CUDA placement/context");
        DNDS_check_throw_info(!Settings().SORInstead, "CUDA SOR requires a logical parallel sweep design and is unsupported");
        DNDS_check_throw_info(!options.basis || (options.basisDerivativeOrder >= 0 && options.basisDerivativeOrder <= 3),
                              "invalid resident basis derivative order");
        DNDS_check_throw_info(options.matrixPrecision == ReconstructionMatrixPrecision::FP64 ||
                                  options.matrixPrecision == ReconstructionMatrixPrecision::FP32,
                              "unknown reconstruction matrix precision");
        _data->workPartition->Validate();
        auto geometry = Metrics().AcquireRead();
        auto result = std::make_shared<Detail::ReconstructionPlacementData>(_data, device, options);
        result->fp64TensorCore = Detail::FP64TensorCoreAvailable(device);
        result->tf32TensorCore = Detail::TF32TensorCoreAvailable(device);
        const auto &host = *_data;
        auto &view = result->view;
        view.owned = Metrics().NumCell();
        view.n = NRecDOF();
        view.relax = Settings().jacobiRelax;
        view.wbapCenterWeight = Settings().WBAP_nStd;
        view.rotated = host.hasRotatedPeriodicity;
        const auto allocation = host.settings.storage.Allocation();
        view.quantized = options.matrixPrecision == ReconstructionMatrixPrecision::FP32;
        view.raw = host.settings.storage.Raw();
        view.inverseTriangle = host.settings.storage.inverseTriangle;
        view.faceCoupling = host.settings.storage.faceCoupling;
        DNDS_check_throw_info(!view.raw || !view.quantized, "raw VR sweeps currently require FP64 matrix placement");
#define DNDS_REC_ADD(member) view.member = result->Add(#member, host.member, context)
        DNDS_REC_ADD(offsets);
        DNDS_REC_ADD(neighbors);
        DNDS_REC_ADD(faces);
        DNDS_REC_ADD(boundaryOffsets);
        if (view.quantized)
        {
            view.matrixAInvB32 = result->AddQuantized("matrixAInvB32", *host.matrixAInvB, context, allocation);
            view.vectorAInvB32 = result->AddQuantized("vectorAInvB32", *host.vectorAInvB, context, allocation);
            view.boundaryInjection32 = result->AddQuantized("boundaryInjection32", *host.boundaryInjection, context, allocation);
        }
        else
        {
            view.matrixAInvB = result->Add(view.raw ? "rawMatrixB" : "matrixAInvB", host.SweepMatrices(), context, allocation);
            view.vectorAInvB = result->Add(view.raw ? "rawVectorB" : "vectorAInvB", host.SweepVectors(), context, allocation);
            view.boundaryInjection = result->Add(view.raw ? "rawBoundaryInjection" : "boundaryInjection", host.SweepInjection(), context, allocation);
        }
        if (view.raw)
        {
            view.localInverse = result->Add("localInverse", *host.localInverse, context, allocation);
            if (view.faceCoupling)
                view.faceSide = result->Add("rawFaceSide", *host.rawFaceSide, context);
        }
        if (host.capabilities.wbap)
            view.secondary = result->Add("secondary", *host.secondary, context);
#undef DNDS_REC_ADD
        // Allocation accounting follows the actual selected payload, including
        // physical pitch. No alternate full maps/inverse are retained here.
        // Every sweep/limiter kernel checks the publication-wide rotated flag
        // before reading either table. Translational/nonperiodic meshes need
        // neither identity matrices nor per-entry activity flags on device.
        if (view.rotated)
        {
            view.rotation = result->Add("neighborRotation", host.neighborRotation, context);
            view.rotationActive = result->Add("neighborRotationActive", host.neighborRotationActive, context);
        }
        for (bool core : {true, false})
        {
            const auto &list = core ? host.workPartition->Core() : host.workPartition->HaloDependent();
            R::Array<index> rows(R::RowLayout::Dense(list.size(), 1));
            {
                auto write = rows.Write();
                const auto output = write.View();
                OMP::ForRows(0, index(list.size()), [&](index iRow)
                             { output(iRow, 0) = list[size_t(iRow)]; });
            }
            const auto selected = result->Add(core ? "core" : "dependent", rows, context);
            if (core)
                view.core = selected;
            else
                view.dependent = selected;
        }
        const auto &metric = *Metrics()._data;
        result->geometryStamp = metric.stamp;
        result->dofs = metric.dofs;
        result->communication = metric.communication;
        result->halo = metric.halo;
        result->dimension = metric.dim;
        result->faces = metric.faceOffsets.RowCount() - 1;
        if (options.metrics)
        {
            const auto cell = result->Add("quadrature/cellGeometry", metric.cell, context);
            result->samplingCellGeometry = result->realArrays.back();
            const auto offsets = result->Add("faceOffsets", metric.faceOffsets, context);
            const auto quadrature = result->Add("faceQuadrature", metric.faceQuadrature, context);
            const auto radial = result->Add("radialMoment", metric.radialMoment, context);
            const auto faceMean = result->Add("faceMean", metric.faceMean, context);
            result->metricView = {R::Detail::ProjectView<const real, false, R::FixedRows<8>>(cell, metric.cell.Layout()),
                                  R::Detail::ProjectView<const index, false, R::FixedRows<1>>(offsets, metric.faceOffsets.Layout()),
                                  R::Detail::ProjectView<const real, false, R::FixedRows<7>>(quadrature, metric.faceQuadrature.Layout()),
                                  R::Detail::ProjectView<const real, false, R::FixedRows<3>>(radial, metric.radialMoment.Layout()),
                                  R::Detail::ProjectView<const real, false, R::FixedRows<7>>(faceMean, metric.faceMean.Layout())};
        }
        if (options.arbitraryPointSampling)
        {
            result->Add("quadrature/cellMoments", host.moments, context);
            result->samplingMoments = result->realArrays.back();
            if (!result->samplingCellGeometry)
            {
                result->Add("quadrature/cellGeometry", metric.cell, context);
                result->samplingCellGeometry = result->realArrays.back();
            }
        }
        if (options.basis)
        {
            auto &pointBasis = result->pointBasisView;
            const Math::TaylorBasis polynomial(metric.dim, Settings().maxOrder);
            pointBasis.polynomial = polynomial.KernelParameters();
            pointBasis.availableDerivatives = polynomial.DerivativeNumber(options.basisDerivativeOrder);
            pointBasis.cachedDerivatives = Settings().cacheDiffBase ? host.cachedDerivatives : 0;
            if (pointBasis.cachedDerivatives)
            {
                const auto basis = result->Add("quadrature/face/basis", host.faceBasis, context);
                result->samplingFaceBasis = result->realArrays.back();
                pointBasis.cached = R::Detail::ProjectView<const real, false, R::TableRows>(basis, host.faceBasis.Layout());
                const auto offsets = options.metrics ? result->metricView.faceOffsets
                                                     : R::Detail::ProjectView<const index, false, R::FixedRows<1>>(result->Add("faceOffsets", metric.faceOffsets, context), metric.faceOffsets.Layout());
                result->basisView = {pointBasis.cached, offsets, host.n, host.cachedDerivatives};
            }
            if (pointBasis.cachedDerivatives < pointBasis.availableDerivatives)
            {
                if (!result->samplingMoments)
                {
                    result->Add("quadrature/cellMoments", host.moments, context);
                    result->samplingMoments = result->realArrays.back();
                }
                if (!result->samplingCellGeometry)
                {
                    result->Add("quadrature/cellGeometry", metric.cell, context);
                    result->samplingCellGeometry = result->realArrays.back();
                }
                // Add retains immutable read leases with this publication;
                // these handles also seed the PP/source placement memo.
                const auto moments = result->samplingMoments->Read(context);
                const auto cells = result->samplingCellGeometry->Read(context);
                pointBasis.moments = moments.View<R::TableRows, false>();
                pointBasis.cells = cells.View<R::FixedRows<8>, false>();
            }
        }
        if (options.mesh)
        {
            result->mesh = std::make_shared<Geom::UnstructuredMeshSnapshot>(Geom::UnstructuredMeshSnapshot::Capture(*metric.mesh, Geom::MeshReadRequirements::CellFlux()).To(device, context));
            std::set<uint64_t> allocations;
            result->mesh->VisitAllocations([&](uint64_t identity, size_t capacity, R::Device placement)
                                           {
                if(allocations.insert(identity).second)
                    result->buffers.push_back({"mesh.allocation."+std::to_string(allocations.size()),capacity,placement}); });
        }
        return VariationalReconstructionPlan(options.retainHostPayloads ? _data : nullptr, std::move(result));
    }
    const R::Array<real> &VariationalReconstructionPlan::SamplingFaceBasis() const
    {
        Validate();
        if (!_device)
            return _data->faceBasis;
        if (!_device->settings.cacheDiffBase)
            return _device->empty;
        DNDS_check_throw_info(_device->samplingFaceBasis, "resident face sampling basis was not requested at reconstruction placement");
        return *_device->samplingFaceBasis;
    }
    const R::Array<real> &VariationalReconstructionPlan::SamplingMoments() const
    {
        Validate();
        if (!_device)
            return _data->moments;
        DNDS_check_throw_info(_device->samplingMoments, "resident moments require direct-basis or arbitrary-point sampling placement");
        return *_device->samplingMoments;
    }
    const R::Array<real> &VariationalReconstructionPlan::SamplingCellGeometry() const
    {
        Validate();
        if (!_device)
            return _data->metrics._data->cell;
        DNDS_check_throw_info(_device->samplingCellGeometry,
                              "resident cell geometry requires direct-basis, metric or arbitrary-point sampling placement");
        return *_device->samplingCellGeometry;
    }
    R::Device VariationalReconstructionPlan::Device() const
    {
        DNDS_check_throw_info(_identity, "reconstruction plan was moved/released");
        return _device ? _device->device : R::Device::Host();
    }
    const Algebra::DofSpace &VariationalReconstructionPlan::Space() const
    {
        Validate();
        return _device ? *_device->dofs : Metrics().Space();
    }
    std::vector<ResidentBufferInfo> VariationalReconstructionPlan::ResidentBuffers() const
    {
        Validate();
        return _device ? _device->buffers : std::vector<ResidentBufferInfo>{};
    }
    ReconstructionResidentRead VariationalReconstructionPlan::AcquireResidentRead() const
    {
        Validate();
        DNDS_check_throw_info(_device, "reconstruction plan has no non-host placement");
        return ReconstructionResidentRead(_device);
    }
    Detail::ReconstructionDeviceView ReconstructionResidentRead::DeviceView() const
    {
        DNDS_check_throw_info(_data, "resident reconstruction read was moved/released");
        return _data->view;
    }
    FiniteVolumeHostView ReconstructionResidentRead::MetricView() const
    {
        DNDS_check_throw_info(_data && _data->options.metrics,
                              "resident reconstruction metrics were not requested");
        return _data->metricView;
    }
    ReconstructionBasisHostView ReconstructionResidentRead::BasisView() const
    {
        DNDS_check_throw_info(_data && _data->options.basis && _data->settings.cacheDiffBase,
                              "resident cached basis was not requested; use PointBasisView for on-demand evaluation");
        return _data->basisView;
    }
    ReconstructionPointBasisView ReconstructionResidentRead::PointBasisView() const
    {
        DNDS_check_throw_info(_data && _data->options.basis, "resident point basis was not requested");
        return _data->pointBasisView;
    }
    const Geom::UnstructuredMeshSnapshot &VariationalReconstructionPlan::MeshSnapshot() const
    {
        Validate();
        DNDS_check_throw_info(_device && _device->mesh, "device mesh was not requested");
        return *_device->mesh;
    }
    ReconstructionWorkspace VariationalReconstructionPlan::BuildDeviceWorkspace(index components, R::RowLayout prototype,
                                                                                std::vector<index> periodic) const
    {
        Validate();
        DNDS_check_throw_info(_device && components > 0 && components <= INT_MAX / _device->view.n &&
                                  !prototype.IsVariable() && prototype.Width() == components * _device->view.n,
                              "invalid device reconstruction shape");
        Detail::ValidateVectorBlocks(components, periodic);
        const index blocks = std::max<index>(1, std::min<index>(256, _device->view.owned));
        R::Array<index> starts(R::RowLayout::Dense(periodic.size(), 1));
        {
            auto write = starts.Write();
            for (index i = 0; i < index(periodic.size()); ++i)
                write.View()(i, 0) = periodic[size_t(i)];
        }
        auto deviceStarts = starts.To(Device());
        const auto periodicIdentity = deviceStarts.Identity();
        const auto periodicVersion = deviceStarts.Version();
        const auto width = prototype.Width();
        return {BuildDeviceField(std::move(prototype)),
                R::Array<real>(R::RowLayout::Dense(_device->capabilities.wbap ? blocks * 6 : 0, width), Device()),
                components,
                {},
                std::move(periodic),
                std::move(deviceStarts),
                R::Array<int>(R::RowLayout::Dense(1, 1), Device()),
                blocks,
                periodicIdentity,
                periodicVersion,
                _device};
    }
    R::TransferBinding VariationalReconstructionPlan::BindHalo(FVField &field, R::TransferTransport transport, R::TransferOptions options) const
    {
        Validate();
        if (!_device)
        {
            const auto &metrics = _data->metrics;
            const auto &data = *metrics._data;
            data.communication->LocalStage([&]
                                           { metrics.ValidateField(field, field.Width()); });
            return R::PayloadPlan(*data.halo, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<real>())
                .Bind(field.Owned(), field.Ghost(), transport, std::move(options));
        }
        _device->communication->LocalStage([&]
                                           {Validate();ValidateDeviceField(*this,field,field.Width());
            DNDS_check_throw_info(field.Space().Same(*_device->dofs),"device field belongs to another metric space"); });
        return R::PayloadPlan(*_device->halo, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<real>()).Bind(field.Owned(), field.Ghost(), transport, std::move(options));
    }
    ReconstructionBoundaryValues VariationalReconstructionPlan::PlaceBoundaryValues(const ReconstructionBoundaryValues &source) const
    {
        Validate();
        DNDS_check_throw_info(_device && source.plan == _identity, "boundary values belong to another plan or placement is not device-resident");
        ValidateArray(source.values, R::Device::Host(), NumBoundaryQuadrature(), source.values.Layout().Width());
        return {source.values.To(Device()), _identity};
    }
    FaceCharacteristicTransform VariationalReconstructionPlan::PlaceFaceTransform(const FaceCharacteristicTransform &source) const
    {
        Validate();
        DNDS_check_throw_info(_device && source.plan == _identity && source.first >= 0 && source.count >= 0 && source.count <= 64,
                              "device characteristic source belongs to another plan or has an invalid range");
        if (source.count == 0)
        {
            FaceCharacteristicTransform result;
            result.first = source.first;
            result.plan = _identity;
            return result;
        }
        DNDS_check_throw_info(
            source.validation == CharacteristicValidation::Frozen && source.forwardVersion == source.forward.Version() && source.inverseVersion == source.inverse.Version() && source.forwardVersion &&
                source.forwardIdentity == source.forward.Identity() && source.inverseIdentity == source.inverse.Identity(),
            "unfinalized device characteristic source");
        FaceCharacteristicTransform result{source.first, source.count, source.forward.To(Device()), source.inverse.To(Device()), _identity};
        result.forwardVersion = result.forward.Version();
        result.inverseVersion = result.inverse.Version();
        result.forwardIdentity = result.forward.Identity();
        result.inverseIdentity = result.inverse.Identity();
        result.plan = _identity;
        return result;
    }
    FaceCharacteristicTransform VariationalReconstructionPlan::BuildQueuedFaceTransform(index first, index count) const
    {
        Require(ReconstructionOperation::WBAP);
        Validate();
        DNDS_check_throw_info(first >= 0 && count >= 0 && count <= 64,
                              "queued characteristic buffers require a valid component range");
        if (!_device)
        {
            auto result = BuildFaceTransform(first, count);
            result.validation = CharacteristicValidation::Queued;
            return result;
        }
        FaceCharacteristicTransform result{first, count, {}, {}, _identity};
        result.validation = CharacteristicValidation::Queued;
        if (!count)
            return result;
        result.forward = R::Array<real>(R::RowLayout::Dense(_device->faces, count * count), Device());
        result.inverse = R::Array<real>(R::RowLayout::Dense(_device->faces, count * count), Device());
        result.forwardIdentity = result.forward.Identity();
        result.inverseIdentity = result.inverse.Identity();
        return result;
    }
    void VariationalReconstructionPlan::SubmitValidateFaceTransform(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                                    const FaceCharacteristicTransform &transform, ReconstructionWorkspace &work) const
    {
        ValidateDeviceStream(*this, stream);
        ValidateDeviceWorkspace(_device, work);
        DNDS_check_throw_info(transform.plan == _identity && transform.validation == CharacteristicValidation::Queued &&
                                  transform.first >= 0 && transform.first <= work.components && transform.count >= 0 &&
                                  transform.count <= 64 && transform.count <= work.components - transform.first,
                              "queued characteristic publication belongs to another plan/range");
        if (!transform.count)
            return;
        DNDS_check_throw_info(transform.forwardIdentity == transform.forward.Identity() && transform.inverseIdentity == transform.inverse.Identity(),
                              "queued characteristic buffers were replaced");
        for (const auto *array : {&transform.forward, &transform.inverse})
        {
            ValidateArray(*array, Device(), _device->faces, transform.count * transform.count);
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(array->Storage(), work.deviceStatus.Storage()), "characteristic matrices alias validation status");
        }
        R::SubmitArrays(engine, stream, [first = transform.first, count = transform.count](R::ExecutionContext c, auto left, auto right, auto status)
                        { Detail::LaunchValidateCharacteristics(c, {left, right, first, count}, status); }, R::Read(transform.forward), R::Read(transform.inverse), R::ReadWrite(work.deviceStatus));
    }
    void VariationalReconstructionPlan::SubmitDeviceReconstructionIterTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean,
        const FVField &input, ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary,
        ReconstructionRegion region, ReconstructionDeviceKernel kernel) const
    {
        ValidateDeviceStream(*this, stream);
        DNDS_check_throw_info(SupportsKernel(kernel), "requested CUDA reconstruction kernel is unavailable on this device/build");
        ValidateDeviceWorkspace(_device, work, input);
        ValidateDeviceField(*this, mean, work.components);
        ValidateDeviceField(*this, input, work.components * _device->view.n);
        for (const auto *array : {&mean.Owned(), &mean.Ghost(), &input.Owned(), &input.Ghost(), &boundary.values})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(array->Storage(), work.temporary.Owned().Storage()), "device Jacobi output aliases an input");
        DNDS_check_throw_info(mean.Space().Same(input.Space()) && input.Space().Same(work.temporary.Space()), "device fields belong to different spaces");
        DNDS_check_throw_info(boundary.plan == _identity, "device boundary plan mismatch");
        ValidateArray(boundary.values, Device(), NumBoundaryQuadrature(), work.components);
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input.Owned().Storage(), work.temporary.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(input.Ghost().Storage(), work.temporary.Owned().Storage()),
                              "device Jacobi aliases its input");
        const auto selection = Selection(*_device, region);
        auto pin = _device;
        if (kernel == ReconstructionDeviceKernel::FP32FullAccumulation || kernel == ReconstructionDeviceKernel::FP32FloatFloatAccumulation ||
            kernel == ReconstructionDeviceKernel::FloatFloatPipelined || kernel == ReconstructionDeviceKernel::FP32Pipelined ||
            kernel == ReconstructionDeviceKernel::TF32Pipelined)
        {
            // Only these policies validate narrow-range arithmetic results.
            // Declare the retained status mutation, including the separate
            // stream-ordered result check, to the Runtime dependency tracker.
            auto launchAccum = [pin, selection, kernel](R::ExecutionContext context, auto mo, auto mg, auto ro, auto rg, auto bc, auto output, auto blocks, auto status)
            {
                if (kernel == ReconstructionDeviceKernel::FloatFloatPipelined || kernel == ReconstructionDeviceKernel::FP32Pipelined || kernel == ReconstructionDeviceKernel::TF32Pipelined)
                {
                    Detail::LaunchStagedJacobi(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status.ConstView(),
                                               pin->settings.cudaSweep, kernel == ReconstructionDeviceKernel::FloatFloatPipelined ? Detail::ReconstructionSweepArithmetic::FloatFloat : kernel == ReconstructionDeviceKernel::FP32Pipelined ? Detail::ReconstructionSweepArithmetic::FP32
                                                                                                                                                                                                                                            : Detail::ReconstructionSweepArithmetic::TF32);
                    Detail::LaunchValidateJacobiResult(context, selection, output.ConstView(), status);
                }
                else
                    Detail::LaunchJacobiFullAccumulation(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status,
                                                         kernel == ReconstructionDeviceKernel::FP32FloatFloatAccumulation);
            };
            if (region == ReconstructionRegion::Core)
                R::SubmitArrays(engine, stream, [launchAccum](R::ExecutionContext c, auto mo, auto ro, auto bc, auto output, auto blocks, auto status)
                                { launchAccum(c, mo, R::ArrayView<const real>{}, ro, R::ArrayView<const real>{}, bc, output, blocks, status); }, R::Read(mean.Owned()), R::Read(input.Owned()), R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.devicePeriodicVectorStarts), R::ReadWrite(work.deviceStatus));
            else
                R::SubmitArrays(engine, stream, launchAccum, R::Read(mean.Owned()), R::Read(mean.Ghost()), R::Read(input.Owned()), R::Read(input.Ghost()),
                                R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.devicePeriodicVectorStarts), R::ReadWrite(work.deviceStatus));
            return;
        }
        auto launch = [pin, selection, kernel](R::ExecutionContext context, auto mo, auto mg, auto ro, auto rg, auto bc, auto output, auto blocks, auto status)
        {
            if (kernel == ReconstructionDeviceKernel::FP64MatrixShared)
                Detail::LaunchJacobiMatrixShared(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status);
            else if (kernel == ReconstructionDeviceKernel::TF32TensorCore)
                Detail::LaunchJacobiTF32(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status);
            else if (kernel == ReconstructionDeviceKernel::FP64Pipelined || kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync)
                Detail::LaunchStagedJacobi(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status, pin->settings.cudaSweep,
                                           kernel == ReconstructionDeviceKernel::FP64Pipelined ? Detail::ReconstructionSweepArithmetic::FP64 : Detail::ReconstructionSweepArithmetic::FP64TensorCore);
            else
                Detail::LaunchJacobi(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status);
        };
        if (region == ReconstructionRegion::Core)
            R::SubmitArrays(engine, stream, [launch](R::ExecutionContext c, auto meanOwned, auto recOwned, auto bc, auto output, auto blocks, auto status)
                            { launch(c, meanOwned, R::ArrayView<const real>{}, recOwned, R::ArrayView<const real>{}, bc, output, blocks, status); }, R::Read(mean.Owned()), R::Read(input.Owned()), R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.devicePeriodicVectorStarts), R::Read(work.deviceStatus));
        else
            R::SubmitArrays(engine, stream, launch, R::Read(mean.Owned()), R::Read(mean.Ghost()), R::Read(input.Owned()), R::Read(input.Ghost()),
                            R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.devicePeriodicVectorStarts), R::Read(work.deviceStatus));
    }
    void VariationalReconstructionPlan::SubmitDeviceReconstructionIterLimitedTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean,
        const FVField &input, ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary,
        const FVField &target, const R::Array<real> &alpha, ReconstructionRegion region) const
    {
        ValidateDeviceStream(*this, stream);
        ValidateDeviceWorkspace(_device, work, input);
        ValidateDeviceField(*this, mean, work.components);
        ValidateDeviceField(*this, input, work.components * _device->view.n);
        ValidateDeviceField(*this, target, work.components * _device->view.n);
        DNDS_check_throw_info(mean.Space().Same(input.Space()) && input.Space().Same(target.Space()) &&
                                  input.Space().Same(work.temporary.Space()),
                              "limited Jacobi field spaces differ");
        DNDS_check_throw_info(boundary.plan == _identity, "limited Jacobi boundary plan mismatch");
        ValidateArray(boundary.values, Device(), NumBoundaryQuadrature(), work.components);
        ValidateArray(alpha, Device(), input.Owned().RowCount(), 1);
        for (const auto *array : {&mean.Owned(), &mean.Ghost(), &input.Owned(), &input.Ghost(),
                                  &boundary.values, &target.Owned(), &target.Ghost(), &alpha})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(array->Storage(), work.temporary.Owned().Storage()),
                                  "limited Jacobi output aliases an input");
        const auto selection = Selection(*_device, region);
        auto pin = _device;
        auto launch = [pin, selection](R::ExecutionContext context, auto mo, auto mg, auto ro, auto rg,
                                       auto bc, auto wanted, auto penalty, auto output, auto blocks, auto status)
        {
            Detail::LaunchLimitedJacobi(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg},
                                        bc, wanted, penalty, output, status);
        };
        if (region == ReconstructionRegion::Core)
            R::SubmitArrays(engine, stream, [launch](R::ExecutionContext c, auto mo, auto ro, auto bc, auto wanted, auto penalty, auto output, auto blocks, auto status)
                            { launch(c, mo, R::ArrayView<const real>{}, ro, R::ArrayView<const real>{}, bc,
                                     wanted, penalty, output, blocks, status); }, R::Read(mean.Owned()), R::Read(input.Owned()), R::Read(boundary.values), R::Read(target.Owned()), R::Read(alpha), R::Write(work.temporary.Owned()), R::Read(work.devicePeriodicVectorStarts), R::Write(work.deviceStatus));
        else
            R::SubmitArrays(engine, stream, launch, R::Read(mean.Owned()), R::Read(mean.Ghost()),
                            R::Read(input.Owned()), R::Read(input.Ghost()), R::Read(boundary.values),
                            R::Read(target.Owned()), R::Read(alpha), R::Write(work.temporary.Owned()),
                            R::Read(work.devicePeriodicVectorStarts), R::Write(work.deviceStatus));
    }
    void VariationalReconstructionPlan::SubmitDeviceCommitReconstruction(
        R::EagerEngine &engine, const R::ExecutionStream &stream, FVField &coefficients,
        ReconstructionWorkspace &work) const
    {
        // Stable owned-only publication: coefficient ghosts remain stale until
        // exchange. Keep leases and the device-status guard in any future copy
        // optimization; Runtime::Array::CopyFrom alone does not provide the guard.
        ValidateDeviceStream(*this, stream);
        ValidateDeviceWorkspace(_device, work, coefficients);
        auto pin = _device;
        R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto input, auto output, auto status)
                        { Detail::LaunchCommit(context, input, output, status); }, R::Read(work.temporary.Owned()), R::Write(coefficients.Owned()), R::Read(work.deviceStatus));
    }
    void VariationalReconstructionPlan::SubmitDeviceLimiterWBAP_3OrderTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &coefficients,
        ReconstructionWorkspace &work, int order,
        FaceCharacteristicTransformRef transform, ReconstructionRegion region, const R::Array<real> *indicator,
        real threshold) const
    {
        Require(ReconstructionOperation::WBAP);
        ValidateDeviceStream(*this, stream);
        ValidateDeviceWorkspace(_device, work, coefficients);
        DNDS_check_throw_info(order >= 1 && order <= Settings().maxOrder && std::isfinite(threshold), "invalid device limiter order/threshold");
        for (const auto *input : std::array<const R::Array<real> *, 3>{&coefficients.Owned(), &coefficients.Ghost(), &work.scratch})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), work.temporary.Owned().Storage()), "device limiter output aliases its input/scratch");
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(coefficients.Owned().Storage(), work.scratch.Storage()) &&
                                  !Algebra::Detail::Overlaps(coefficients.Ghost().Storage(), work.scratch.Storage()),
                              "device limiter scratch aliases coefficients");
        const auto selection = Selection(*_device, region);
        const auto n = _device->view.n;
        const index begin = order == 1 ? 0 : Math::PolynomialSize(_device->dimension, order - 1) - 1;
        const index modes = Math::PolynomialSize(_device->dimension, order) - 1 - begin;
        if (indicator)
        {
            ValidateArray(*indicator, Device(), _device->view.owned, 1);
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(indicator->Storage(), work.temporary.Owned().Storage()) &&
                                      !Algebra::Detail::Overlaps(indicator->Storage(), work.scratch.Storage()),
                                  "device indicator aliases mutable output/scratch");
        }
        const auto *selectedTransform = transform && transform->get().count ? &transform->get() : nullptr;
        if (transform && !selectedTransform)
            DNDS_check_throw_info(transform->get().plan == _identity && transform->get().first >= 0 && transform->get().first <= work.components,
                                  "disabled device characteristic transform belongs to another plan or has an invalid offset");
        if (selectedTransform)
        {
            DNDS_check_throw_info(selectedTransform->plan == _identity && selectedTransform->first >= 0 && selectedTransform->count > 0 && selectedTransform->count <= 64 && selectedTransform->count <= work.components - selectedTransform->first &&
                                      selectedTransform->forwardIdentity == selectedTransform->forward.Identity() && selectedTransform->inverseIdentity == selectedTransform->inverse.Identity() &&
                                      (selectedTransform->validation == CharacteristicValidation::Queued ||
                                       (selectedTransform->validation == CharacteristicValidation::Frozen && selectedTransform->forwardVersion != 0 &&
                                        selectedTransform->forwardVersion == selectedTransform->forward.Version() && selectedTransform->inverseVersion == selectedTransform->inverse.Version())),
                                  "device characteristic data changed/unfinalized");
            ValidateArray(selectedTransform->forward, Device(), _device->faces, selectedTransform->count * selectedTransform->count);
            ValidateArray(selectedTransform->inverse, Device(), _device->faces, selectedTransform->count * selectedTransform->count);
            for (const auto *input : {&selectedTransform->forward, &selectedTransform->inverse})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), work.temporary.Owned().Storage()) &&
                                          !Algebra::Detail::Overlaps(input->Storage(), work.scratch.Storage()),
                                      "device transform aliases mutable output/scratch");
        }
        auto pin = _device;
        const auto first = selectedTransform ? selectedTransform->first : 0, count = selectedTransform ? selectedTransform->count : 0,
                   blocksCount = work.deviceBlocks;
        // Optional read operands are empty owned arrays on the right device,
        // retained in the closure. No fabricated pointers or implicit transfer.
        const auto &forward = selectedTransform ? selectedTransform->forward : _device->empty,
                   &inverse = selectedTransform ? selectedTransform->inverse : _device->empty,
                   &activation = indicator ? *indicator : _device->empty;
        auto launch = [pin, selection, first, count, begin, modes, threshold, blocksCount](R::ExecutionContext context, auto owned, auto ghost, auto output, auto scratch, auto blocks, auto left, auto right, auto active, auto status)
        {
            const Detail::DeviceCharacteristicView matrices{left, right, first, count};
            Detail::LaunchWBAP(context, pin->view, selection, {blocks}, {owned, ghost}, output, scratch, matrices, active, status, begin, modes, threshold, blocksCount);
        };
        if (region == ReconstructionRegion::Core)
            R::SubmitArrays(engine, stream, [launch](R::ExecutionContext c, auto owned, auto output, auto scratch, auto blocks, auto left, auto right, auto active, auto status)
                            { launch(c, owned, R::ArrayView<const real>{}, output, scratch, blocks, left, right, active, status); }, R::Read(coefficients.Owned()), R::Write(work.temporary.Owned()), R::ReadWrite(work.scratch), R::Read(work.devicePeriodicVectorStarts), R::Read(forward), R::Read(inverse), R::Read(activation), R::ReadWrite(work.deviceStatus));
        else
            R::SubmitArrays(engine, stream, launch, R::Read(coefficients.Owned()), R::Read(coefficients.Ghost()), R::Write(work.temporary.Owned()),
                            R::ReadWrite(work.scratch), R::Read(work.devicePeriodicVectorStarts), R::Read(forward), R::Read(inverse), R::Read(activation), R::ReadWrite(work.deviceStatus));
    }
    void VariationalReconstructionPlan::CheckStatus(const ReconstructionWorkspace &work) const
    {
        Validate();
        if (!_device)
            return;
        ValidateDeviceWorkspace(_device, work);
        const auto host = work.deviceStatus.To(R::Device::Host());
        DNDS_check_throw_info(host.Read().View()(0, 0) == 0, "device reconstruction validation failed (indicator, LVR penalty, characteristics or nonfinite narrow accumulation); do not publish the queued result");
    }
    bool VariationalReconstructionPlan::SupportsKernel(ReconstructionDeviceKernel kernel) const
    {
        DNDS_check_throw_info(_identity, "reconstruction plan is empty");
        if (!_device)
            return kernel == ReconstructionDeviceKernel::FP64;
        if (_device->view.quantized)
            return kernel == ReconstructionDeviceKernel::FP32 ||
                   kernel == ReconstructionDeviceKernel::FP32Pipelined ||
                   kernel == ReconstructionDeviceKernel::FP32FullAccumulation ||
#ifdef DNDS_USE_CCCL_COMPUTE
                   kernel == ReconstructionDeviceKernel::FP32FloatFloatAccumulation ||
#endif
                   ((kernel == ReconstructionDeviceKernel::TF32TensorCore || kernel == ReconstructionDeviceKernel::TF32Pipelined) && _device->tf32TensorCore);
        const auto n = _device->view.n;
        const bool staticModes = n == 2 || n == 3 || n == 5 || n == 9 || n == 19;
        return kernel == ReconstructionDeviceKernel::FP64 ||
               (staticModes && (kernel == ReconstructionDeviceKernel::FP64MatrixShared || kernel == ReconstructionDeviceKernel::FP64Pipelined ||
#ifdef DNDS_USE_CCCL_COMPUTE
                                kernel == ReconstructionDeviceKernel::FloatFloatPipelined ||
#endif
                                (kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync && _device->fp64TensorCore)));
    }
#ifndef DNDS_USE_CUDA
    namespace Detail
    {
        ReconstructionSweepResources InspectStagedJacobi(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                                         R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>,
                                                         R::ArrayView<const int>, const ReconstructionSweepLaunch &, ReconstructionSweepArithmetic)
        {
            DNDS_check_throw_info(false, "stage-once CUDA VR resource inspection unavailable in this build");
            return {};
        }
        void LaunchStagedJacobi(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>,
                                R::ArrayView<const int>, const ReconstructionSweepLaunch &, ReconstructionSweepArithmetic)
        {
            DNDS_check_throw_info(false, "stage-once CUDA VR unavailable in this build");
        }
        void LaunchValidateCharacteristics(R::ExecutionContext, DeviceCharacteristicView, R::ArrayView<int>)
        {
            DNDS_check_throw_info(false, "queued CUDA characteristic validation unavailable");
        }
        bool FP64TensorCoreAvailable(R::Device) { return false; }
        bool TF32TensorCoreAvailable(R::Device) { return false; }
        void LaunchJacobiTF32(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                              R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>)
        {
            DNDS_check_throw_info(false, "CUDA TF32 reconstruction unavailable");
        }
        void LaunchJacobi(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks, R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>)
        {
            DNDS_check_throw_info(false, "CUDA Jacobi unavailable");
        }
        void LaunchJacobiMatrixShared(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks, R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>)
        {
            DNDS_check_throw_info(false, "CUDA matrix-shared Jacobi unavailable");
        }
        void LaunchValidateJacobiResult(R::ExecutionContext, DeviceWorkSelection, R::ArrayView<const real>, R::ArrayView<int>)
        {
            DNDS_check_throw_info(false, "CUDA finite-result check unavailable");
        }
        void LaunchJacobiFullAccumulation(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                          R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<int>, bool)
        {
            DNDS_check_throw_info(false, "CUDA full accumulation Jacobi unavailable");
        }
        void LaunchLimitedJacobi(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                 R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<const real>,
                                 R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<int>)
        {
            DNDS_check_throw_info(false, "CUDA limited Jacobi unavailable");
        }
        void LaunchWBAP(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks, R::PairReadView<real>, R::ArrayView<real>, R::ArrayView<real>, DeviceCharacteristicView, R::ArrayView<const real>, R::ArrayView<int>, index, index, real, index)
        {
            DNDS_check_throw_info(false, "CUDA WBAP unavailable");
        }
        void LaunchCommit(R::ExecutionContext, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>)
        {
            DNDS_check_throw_info(false, "CUDA reconstruction commit unavailable");
        }
    }
#endif
}
