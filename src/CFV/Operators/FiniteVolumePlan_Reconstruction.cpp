#include "CFV/Detail/ReconstructionData.hpp"
#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "CFV/CPU/ReconstructionKernels.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using Matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    using ConstMap = Eigen::Map<const Matrix>;
    using Map = Eigen::Map<Matrix>;
    namespace
    {
        void CheckArray(const R::Array<real> &array, index rows, index width)
        {
            DNDS_check_throw_info(array.GetDevice() == R::Device::Host() && array.IsValid() &&
                                      !array.Layout().IsVariable() && array.RowCount() == rows && index(array.Layout().Width()) == width,
                                  "FV operand has wrong shape, validity or CPU placement");
        }
        void CheckWorkspace(const Detail::ReconstructionData *data, const ReconstructionWorkspace &work)
        {
            DNDS_check_throw_info(work.plan.get() == data && work.components > 0, "reconstruction workspace belongs to another plan");
            data->metrics.ValidateField(work.temporary, data->n * work.components);
            CheckArray(work.scratch, data->capabilities.wbap ? 6 * data->hostThreads : 0, data->n * work.components);
            DNDS_check_throw_info(OMP::DistributedThreads() <= data->hostThreads,
                                  "CPU reconstruction team exceeds prepared scratch capacity");
        }
        void CheckWorkspace(const Detail::ReconstructionData *data, const ReconstructionWorkspace &work,
                            const FVField &coefficients)
        {
            CheckWorkspace(data, work);
            data->metrics.ValidateField(coefficients, data->n * work.components);
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(coefficients.Owned().Storage(), work.temporary.Owned().Storage()),
                                  "reconstruction input and temporary storage overlap");
        }
        void CheckRegion(const Detail::ReconstructionData &data, ReconstructionRegion region)
        {
            data.workPartition->Validate();
            DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                                  "invalid reconstruction work region");
        }
        void CheckSweep(const Detail::ReconstructionData &data, const FVField &mean, const FVField &input,
                        const ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary,
                        ReconstructionRegion region, const FVField *target, const R::Array<real> *alpha)
        {
            CheckWorkspace(&data, work, input);
            DNDS_check_throw_info(!data.settings.storage.Raw() || work.components <= 20,
                                  "raw CPU VR sweep supports at most 20 components in bounded row scratch");
            Detail::ValidateVectorBlocks(work.components, work.periodicVectorStarts);
            data.metrics.ValidateField(mean, work.components);
            DNDS_check_throw_info(boundary.plan == data.identity, "boundary quadrature values belong to another plan");
            CheckArray(boundary.values, data.boundaryPoints.RowCount(), work.components);
            R::TableRows::Validate(boundary.values.Layout());
            CheckRegion(data, region);
            for (const auto *source : {&input.Owned(), &input.Ghost(), &mean.Owned(), &mean.Ghost(), &boundary.values})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), work.temporary.Owned().Storage()),
                                      "reconstruction output aliases input");
            DNDS_check_throw_info(bool(target) == bool(alpha), "limited reconstruction requires both target and alpha");
            if (target)
            {
                data.metrics.ValidateField(*target, data.n * work.components);
                CheckArray(*alpha, data.metrics.NumCell(), 1);
                for (const auto *source : {&target->Owned(), &target->Ghost(), alpha})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), work.temporary.Owned().Storage()),
                                          "reconstruction output aliases penalty input");
            }
        }
        void CheckLimiter(const Detail::ReconstructionData &data, const FVField &coefficients,
                          const ReconstructionWorkspace &work, int order, FaceCharacteristicTransformRef transform,
                          ReconstructionRegion region, const R::Array<real> *indicator, real threshold)
        {
            CheckWorkspace(&data, work, coefficients);
            R::TableRows::Validate(work.scratch.Layout());
            CheckRegion(data, region);
            DNDS_check_throw_info(order >= 1 && order <= data.settings.maxOrder && std::isfinite(threshold),
                                  "invalid limiter order or threshold");
            Detail::ValidateVectorBlocks(work.components, work.periodicVectorStarts);
            if (indicator)
                CheckArray(*indicator, data.metrics.NumCell(), 1);
            if (!transform)
                return;
            const auto &value = transform->get();
            DNDS_check_throw_info(value.plan == data.identity && value.first >= 0 && value.count >= 0 &&
                                      value.first <= work.components && value.count <= work.components - value.first,
                                  "invalid face characteristic transform range/plan");
            if (!value.count)
                return;
            CheckArray(value.forward, data.metrics.NumFaceProc(), value.count * value.count);
            CheckArray(value.inverse, data.metrics.NumFaceProc(), value.count * value.count);
            R::TableRows::Validate(value.forward.Layout());
            R::TableRows::Validate(value.inverse.Layout());
            DNDS_check_throw_info(value.forwardIdentity == value.forward.Identity() && value.inverseIdentity == value.inverse.Identity(),
                                  "face characteristic buffers replaced");
            if (value.validation == CharacteristicValidation::Frozen)
                DNDS_check_throw_info(value.forwardVersion == value.forward.Version() && value.inverseVersion == value.inverse.Version() && value.forwardVersion != 0,
                                      "face characteristic matrices changed; finalize them before limiting");
            else
                DNDS_check_throw_info(value.validation == CharacteristicValidation::Queued, "unknown characteristic publication policy");
        }

        /// Owner/generation and small work-selection metadata. Queued array
        /// operands are reserved separately by SubmitArrayBatch; this object
        /// does not acquire immediate payload leases or fence at retirement.
        struct HostReconstructionContext
        {
            std::shared_ptr<const Detail::ReconstructionData> data;
            Geom::MeshReadAccess mesh;
            std::vector<index> vectorStarts;
            CPU::ReconstructionWorkSelection selection;

            HostReconstructionContext(std::shared_ptr<const Detail::ReconstructionData> values, Geom::MeshReadAccess meshRead,
                                      const ReconstructionWorkspace &work, ReconstructionRegion region)
                : data(std::move(values)), mesh(std::move(meshRead)), vectorStarts(work.periodicVectorStarts)
            {
                const auto &rows = region == ReconstructionRegion::Core ? data->workPartition->Core() : data->workPartition->HaloDependent();
                selection.rows = {rows.data(), index(rows.size())};
                selection.coreFirstPosition = {data->coreFirstPosition.data(), index(data->coreFirstPosition.size())};
                selection.partitionStarts = data->hostPartitionStarts.HostView();
                selection.vectorStarts = {vectorStarts.data(), index(vectorStarts.size())};
                selection.owned = data->metrics.NumCell();
                selection.coreCount = index(data->workPartition->Core().size());
                selection.workers = data->hostThreads;
                selection.all = region == ReconstructionRegion::All;
                selection.core = region == ReconstructionRegion::Core;
                selection.coreFirst = !selection.all || data->settings.coreFirst;
            }
            CPU::ReconstructionSweepView SweepPolicy(index components) const
            {
                CPU::ReconstructionSweepView result;
                result.modes = data->n;
                result.components = components;
                result.relaxation = data->settings.jacobiRelax;
                result.sor = data->settings.SORInstead;
                result.rotated = data->hasRotatedPeriodicity;
                result.raw = data->settings.storage.Raw();
                result.inverseTriangle = data->settings.storage.inverseTriangle;
                result.faceCoupling = data->settings.storage.faceCoupling;
                return result;
            }
            CPU::ReconstructionLimiterView LimiterPolicy(index components, int order) const
            {
                CPU::ReconstructionLimiterView result;
                result.modes = data->n;
                result.components = components;
                result.begin = order == 1 ? 0 : Math::PolynomialSize(data->metrics.Dimension(), order - 1) - 1;
                result.orderModes = Math::PolynomialSize(data->metrics.Dimension(), order) - 1 - result.begin;
                result.centralWeight = data->settings.WBAP_nStd;
                result.rotated = data->hasRotatedPeriodicity;
                return result;
            }
        };
        /// Synchronous convenience APIs retain immediate leases. Their queued
        /// counterparts use HostReconstructionContext and declared operands.
        struct HostReconstructionRead : HostReconstructionContext
        {
            R::Array<index>::ReadLease offsets, neighbors;
            R::Array<real>::ReadLease rotations;
            R::Array<uint8_t>::ReadLease rotationActive;
            std::optional<R::Array<index>::ReadLease> boundaryOffsets, faces;
            std::optional<R::Array<real>::ReadLease> matrices, vectors, injection, inverse, secondary;
            std::optional<R::Array<uint8_t>::ReadLease> faceSides;
            HostReconstructionRead(std::shared_ptr<const Detail::ReconstructionData> values, Geom::MeshReadAccess meshRead,
                                   const ReconstructionWorkspace &work, ReconstructionRegion region, bool limiter)
                : HostReconstructionContext(std::move(values), std::move(meshRead), work, region),
                  offsets(data->offsets.Read()), neighbors(data->neighbors.Read()), rotations(data->neighborRotation.Read()),
                  rotationActive(data->neighborRotationActive.Read())
            {
                if (limiter)
                {
                    faces.emplace(data->faces.Read());
                    secondary.emplace(data->secondary->Read());
                }
                else
                {
                    boundaryOffsets.emplace(data->boundaryOffsets.Read());
                    matrices.emplace(data->SweepMatrices().Read());
                    vectors.emplace(data->SweepVectors().Read());
                    injection.emplace(data->SweepInjection().Read());
                    if (data->settings.storage.Raw())
                    {
                        inverse.emplace(data->localInverse->Read());
                        if (data->settings.storage.faceCoupling)
                        {
                            faces.emplace(data->faces.Read());
                            faceSides.emplace(data->rawFaceSide->Read());
                        }
                    }
                }
            }
            CPU::ReconstructionSweepView Sweep(index components) const
            {
                auto result = SweepPolicy(components);
                result.offsets = offsets.View<R::FixedRows<1>>();
                result.neighbors = neighbors.View<R::FixedRows<1>>();
                result.boundaryOffsets = boundaryOffsets->View<R::FixedRows<1>>();
                result.matrices = matrices->View<R::UniformTableRows>();
                result.vectors = vectors->View<R::UniformTableRows>();
                result.injection = injection->View<R::UniformTableRows>();
                if (inverse)
                    result.inverse = inverse->View<R::UniformTableRows>();
                if (result.faceCoupling)
                {
                    result.faces = faces->View<R::FixedRows<1>>();
                    result.faceSides = faceSides->View<R::FixedRows<1>>();
                }
                result.rotations = rotations.View<R::FixedRows<9>>();
                result.rotationActive = rotationActive.View<R::FixedRows<1>>();
                return result;
            }
            CPU::ReconstructionLimiterView Limiter(index components, int order) const
            {
                auto result = LimiterPolicy(components, order);
                result.offsets = offsets.View<R::FixedRows<1>>();
                result.neighbors = neighbors.View<R::FixedRows<1>>();
                result.faces = faces->View<R::FixedRows<1>>();
                result.secondary = secondary->View<R::TableRows>();
                result.rotations = rotations.View<R::FixedRows<9>>();
                result.rotationActive = rotationActive.View<R::FixedRows<1>>();
                return result;
            }
        };
        template <class T>
        R::ArrayView<T, R::AccessBoundsChecks, R::TableRows> TableView(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
        {
            // TableRows physical-layout validity is checked before submission.
            return {view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
        }
        using ReadRows = CPU::ReconstructionReadRows;
        using WriteRows = CPU::ReconstructionWriteRows;
        using ReadOperand = R::ArrayOperand<real, false, R::UniformTableRows>;
        using WriteOperand = R::ArrayOperand<real, true, R::UniformTableRows>;

        void SubmitHostSweep(R::EagerEngine &engine, const R::ExecutionStream &stream,
                             std::shared_ptr<const Detail::ReconstructionData> data, Geom::MeshReadAccess mesh,
                             const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                             const ReconstructionBoundaryValues &boundary, ReconstructionRegion region,
                             const FVField *target, const R::Array<real> *alpha)
        {
            CheckSweep(*data, mean, input, work, boundary, region, target, alpha);
            auto retained = std::make_shared<HostReconstructionContext>(data, std::move(mesh), work, region);
            auto view = retained->SweepPolicy(work.components);
            std::vector<ReadOperand> reads{R::ReadAs<R::UniformTableRows>(mean.Owned()),
                                           R::ReadAs<R::UniformTableRows>(input.Owned()),
                                           R::ReadAs<R::UniformTableRows>(boundary.values)};
            index meanGhost = -1, inputGhost = -1, targetSlot = -1, alphaSlot = -1;
            const auto add = [&](const R::Array<real> &array)
            {
                const index slot = index(reads.size());
                reads.push_back(R::ReadAs<R::UniformTableRows>(array));
                return slot;
            };
            if (region != ReconstructionRegion::Core)
            {
                meanGhost = add(mean.Ghost());
                inputGhost = add(input.Ghost());
            }
            if (target)
            {
                targetSlot = add(target->Owned());
                alphaSlot = add(*alpha);
                view.limited = true;
            }
            const index inverseSlot = view.raw ? add(*data->localInverse) : -1;
            // Split SOR consumes core results already written to this output.
            // Its dependent pass is therefore a read/write operand, not merely
            // a write reservation, even though the input coefficients stay frozen.
            std::vector<WriteOperand> writes{view.sor && region == ReconstructionRegion::HaloDependent
                                                 ? R::ReadWriteAs<R::UniformTableRows>(work.temporary.Owned())
                                                 : R::WriteAs<R::UniformTableRows>(work.temporary.Owned())};
            const auto launch = [retained, view, meanGhost, inputGhost, targetSlot, alphaSlot, inverseSlot](R::ExecutionContext, SpanView<const ReadRows> inputs, SpanView<const WriteRows> outputs, auto offsets, auto neighbors, auto boundaryOffsets, auto matrices, auto vectors, auto injection, auto rotations, auto rotationActive, auto... rawMetadata) mutable
            {
                                    view.offsets = offsets;
                                    view.neighbors = neighbors;
                                    view.boundaryOffsets = boundaryOffsets;
                                    view.matrices = matrices;
                                    view.vectors = vectors;
                                    view.injection = injection;
                                    view.rotations = rotations;
                                    view.rotationActive = rotationActive;
                                    if (view.raw) view.inverse = inputs[inverseSlot];
                                    if constexpr (sizeof...(rawMetadata) == 2)
                                    {
                                        const auto raw = std::make_tuple(rawMetadata...);
                                        view.faces = std::get<0>(raw);
                                        view.faceSides = std::get<1>(raw);
                                    }
                                    view.means = {inputs[0], meanGhost >= 0 ? inputs[meanGhost] : ReadRows{}};
                                    view.current = inputs[1];
                                    view.boundary = TableView(inputs[2]);
                                    view.ghost = inputGhost >= 0 ? inputs[inputGhost] : ReadRows{};
                                    view.output = outputs[0];
                                    if (view.limited)
                                    {
                                        view.target = inputs[targetSlot];
                                        view.penalty = inputs[alphaSlot];
                                    }
                                    const auto status = CPU::ReconstructionSweep(view, retained->selection);
                                    DNDS_check_throw_info(status, "reconstruction penalty alpha must be finite and in [0,1]"); };
            auto operands = std::make_tuple(R::ReadAs<R::FixedRows<1>>(data->offsets), R::ReadAs<R::FixedRows<1>>(data->neighbors),
                                            R::ReadAs<R::FixedRows<1>>(data->boundaryOffsets), R::ReadAs<R::UniformTableRows>(data->SweepMatrices()),
                                            R::ReadAs<R::UniformTableRows>(data->SweepVectors()), R::ReadAs<R::UniformTableRows>(data->SweepInjection()),
                                            R::ReadAs<R::FixedRows<9>>(data->neighborRotation), R::ReadAs<R::FixedRows<1>>(data->neighborRotationActive));
            if (view.faceCoupling)
                R::SubmitArrayBatch(engine, stream, launch, std::move(reads), std::move(writes), std::tuple_cat(std::move(operands), std::make_tuple(R::ReadAs<R::FixedRows<1>>(data->faces), R::ReadAs<R::FixedRows<1>>(*data->rawFaceSide))));
            else
                R::SubmitArrayBatch(engine, stream, launch, std::move(reads), std::move(writes), std::move(operands));
        }
    }
    const FiniteVolumePlan &VariationalReconstructionPlan::Metrics() const
    {
        DNDS_check_throw_info(_data, "device-only reconstruction placement has no host metric publication");
        return _data->metrics;
    }
    const ReconstructionSettings &VariationalReconstructionPlan::Settings() const
    {
        return _data ? _data->settings : PlacementSettings();
    }
    int VariationalReconstructionPlan::NRecDOF() const { return _data ? _data->n : PlacementNRecDOF(); }
    ReconstructionCapabilities VariationalReconstructionPlan::Capabilities() const
    {
        DNDS_check_throw_info(_device || _data, "reconstruction plan is empty");
        return _device ? PlacementCapabilities() : _data->capabilities;
    }
    void VariationalReconstructionPlan::Require(ReconstructionOperation operation) const
    {
        if (Capabilities().Supports(operation))
            return;
        const char *name = "unknown";
        switch (operation)
        {
        case ReconstructionOperation::Sweep:
            name = "reconstruction sweep";
            break;
        case ReconstructionOperation::WBAP:
            name = "WBAP limiter";
            break;
        case ReconstructionOperation::LinearSystem:
            name = "original linear system";
            break;
        case ReconstructionOperation::LocalPreconditioner:
            name = "local preconditioner";
            break;
        case ReconstructionOperation::InverseDiagnostics:
            name = "inverse diagnostics";
            break;
        }
        DNDS_check_throw_info(false, std::string("VR operation was not prepared: ") + name);
    }
    ReconstructionStorageReport VariationalReconstructionPlan::StorageReport() const
    {
        Validate();
        if (!_data)
            return PlacementStorageReport();
        ReconstructionStorageReport result;
        std::vector<uint64_t> identities;
        const auto add = [&](const char *name, const auto &array)
        {
            using Value = typename std::decay_t<decltype(array)>::value_type;
            const uint64_t payload = uint64_t(array.RowCount()) * array.Layout().Width() * sizeof(Value);
            uint64_t allocation = 0;
            std::vector<uint64_t> tableIdentities;
            array.SharedStorage()->VisitAllocations([&](uint64_t identity, size_t bytes, auto)
                                                    {
                if (std::find(tableIdentities.begin(), tableIdentities.end(), identity) == tableIdentities.end())
                {
                    tableIdentities.push_back(identity);
                    allocation += bytes;
                }
                if (std::find(identities.begin(), identities.end(), identity) == identities.end())
                {
                    identities.push_back(identity);
                    result.retainedAllocationBytes += bytes;
                } });
            result.tables.push_back({name, true, payload, allocation});
            result.retainedPayloadBytes += payload;
        };
        const auto optional = [&](const char *name, const auto &array)
        {
            if (array)
                add(name, *array);
            else
                result.tables.push_back({name, false, 0, 0});
        };
#define DNDS_REC_STORAGE(name) add(#name, _data->name)
        DNDS_REC_STORAGE(moments);
        if (_data->settings.cacheDiffBase)
            DNDS_REC_STORAGE(faceBasis);
        else
            result.tables.push_back({"faceBasis", false, 0, 0});
        DNDS_REC_STORAGE(offsets);
        DNDS_REC_STORAGE(neighbors);
        DNDS_REC_STORAGE(faces);
        DNDS_REC_STORAGE(boundaryOffsets);
        DNDS_REC_STORAGE(neighborRotation);
        DNDS_REC_STORAGE(neighborRotationActive);
        DNDS_REC_STORAGE(boundaryPoints);
#undef DNDS_REC_STORAGE
#define DNDS_REC_OPTIONAL_STORAGE(name) optional(#name, _data->name)
        DNDS_REC_OPTIONAL_STORAGE(rawFaceSide);
        DNDS_REC_OPTIONAL_STORAGE(matrixAInvB);
        DNDS_REC_OPTIONAL_STORAGE(vectorAInvB);
        DNDS_REC_OPTIONAL_STORAGE(boundaryInjection);
        DNDS_REC_OPTIONAL_STORAGE(secondary);
        DNDS_REC_OPTIONAL_STORAGE(localMatrix);
        DNDS_REC_OPTIONAL_STORAGE(localInverse);
        DNDS_REC_OPTIONAL_STORAGE(rawMatrixB);
        DNDS_REC_OPTIONAL_STORAGE(rawVectorB);
        DNDS_REC_OPTIONAL_STORAGE(rawBoundaryInjection);
        DNDS_REC_OPTIONAL_STORAGE(inverseConditions);
#undef DNDS_REC_OPTIONAL_STORAGE
        result.preparationRuntimeScratchBytes = _data->preparationRuntimeScratchBytes;
        result.preparationRuntimePeakBytes = result.retainedAllocationBytes + result.preparationRuntimeScratchBytes;
        return result;
    }
    ReconstructionInverseDiagnostics VariationalReconstructionPlan::GetInverseDiagnostics() const
    {
        Require(ReconstructionOperation::InverseDiagnostics);
        Validate();
        return _data ? _data->inverseDiagnostics : PlacementInverseDiagnostics();
    }
    HardEigen::SVDInverseDiagnostics VariationalReconstructionPlan::GetCellInverseDiagnostics(index iCell) const
    {
        Require(ReconstructionOperation::InverseDiagnostics);
        Validate();
        DNDS_check_throw_info(_data, "per-cell inverse diagnostics require retained host payloads");
        DNDS_check_throw_info(iCell >= 0 && iCell < Metrics().NumCell(), "inverse diagnostic cell is not owned");
        auto read = _data->inverseConditions->Read();
        const auto values = read.View<Runtime::FixedRows<4>>();
        return {_data->n, Eigen::Index(values(iCell, 0)), values(iCell, 3), values(iCell, 1), values(iCell, 2)};
    }
    void VariationalReconstructionPlan::Validate() const
    {
        DNDS_check_throw_info(_identity, "reconstruction plan is empty");
        if (_data)
            Metrics().Validate();
        if (_device)
            ValidatePlacement();
    }
    ReconstructionWorkspace VariationalReconstructionPlan::BuildWorkspace(index components, std::vector<index> periodicVectorStarts) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n, "invalid reconstruction component count");
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        if (_device)
            return BuildDeviceWorkspace(components, R::RowLayout::Dense(PlacementOwnedRows(), n * components),
                                        std::move(periodicVectorStarts));
        return {Metrics().BuildField(n * components),
                R::Array<real>(R::RowLayout::Dense(_data->capabilities.wbap ? 6 * _data->hostThreads : 0, n * components)), components, _data, std::move(periodicVectorStarts)};
    }
    ReconstructionWorkspace VariationalReconstructionPlan::BuildWorkspace(index components, R::RowLayout prototype,
                                                                          std::vector<index> periodicVectorStarts) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n && !prototype.IsVariable() &&
                                  prototype.Width() == n * components,
                              "reconstruction physical prototype width mismatch");
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        if (_device)
            return BuildDeviceWorkspace(components, std::move(prototype), std::move(periodicVectorStarts));
        return {Metrics().BuildField(prototype),
                R::Array<real>(R::RowLayout::Dense(_data->capabilities.wbap ? 6 * _data->hostThreads : 0, n * components)), components, _data, std::move(periodicVectorStarts)};
    }
    FVField VariationalReconstructionPlan::BuildCoefficients(index components) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n, "invalid reconstruction component count");
        if (_device)
            return BuildDeviceField(R::RowLayout::Dense(PlacementOwnedRows(), n * components));
        return Metrics().BuildField(R::RowLayout::Dense(Metrics().NumCell(), n * components), Device());
    }
    FVField VariationalReconstructionPlan::BuildCoefficients(index components, R::RowLayout prototype) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n && !prototype.IsVariable() &&
                                  prototype.Width() == n * components,
                              "reconstruction physical prototype width mismatch");
        if (_device)
            return BuildDeviceField(std::move(prototype));
        return Metrics().BuildField(std::move(prototype), Device());
    }
    ReconstructionBoundaryValues VariationalReconstructionPlan::BuildBoundaryValues(index components) const
    {
        Validate();
        DNDS_check_throw_info(components > 0, "boundary value component count must be positive");
        return {R::Array<real>(R::RowLayout::Dense(NumBoundaryQuadrature(), components), Device()), _identity};
    }
    index VariationalReconstructionPlan::NumBoundaryQuadrature() const
    {
        return _data ? _data->boundaryPoints.RowCount() : PlacementBoundaryPoints();
    }
    BoundaryQuadraturePoint VariationalReconstructionPlan::GetBoundaryQuadrature(index iPoint) const
    {
        Validate();
        DNDS_check_throw_info(!_device, "boundary quadrature geometry is a host setup query");
        auto indices = _data->boundaryPoints.Read();
        const auto entry = indices.View<R::FixedRows<4>, true>()[iPoint];
        auto geometry = Metrics()._data->faceQuadrature.Read();
        const auto values = geometry.View<R::FixedRows<7>>()[entry[3]];
        return {entry[0], entry[1], int(entry[2]), {values[0], values[1], values[2]}, {values[3], values[4], values[5]}};
    }
    FaceCharacteristicTransform VariationalReconstructionPlan::BuildFaceTransform(index first, index count) const
    {
        Require(ReconstructionOperation::WBAP);
        Validate();
        DNDS_check_throw_info(!_device,
                              "build mutable characteristic transforms on the host plan, then place them on the selected backend");
        DNDS_check_throw_info(first >= 0 && count >= 0 && count <= 64, "invalid characteristic component range");
        if (count == 0)
            return FaceCharacteristicTransform{first, 0, {}, {}, _identity};
        FaceCharacteristicTransform result{first, count,
                                           R::Array<real>(R::RowLayout::Dense(Metrics().NumFaceProc(), count * count)),
                                           R::Array<real>(R::RowLayout::Dense(Metrics().NumFaceProc(), count * count)), _identity};
        for (auto *array : {&result.forward, &result.inverse})
        {
            auto write = array->Write();
            auto view = write.View<R::TableRows>();
            OMP::ForRows(0, view.RowCount(), [&](index iFace)
                         { Map(view[iFace].data(), count, count).setIdentity(); });
        }
        FinalizeFaceTransform(result);
        return result;
    }
    void VariationalReconstructionPlan::FinalizeFaceTransform(FaceCharacteristicTransform &transform, bool checkInverses) const
    {
        Require(ReconstructionOperation::WBAP);
        Validate();
        DNDS_check_throw_info(!_device, "finalize characteristic transforms before backend placement");
        DNDS_check_throw_info(transform.validation == CharacteristicValidation::Frozen, "queued characteristics do not use host version finalization");
        transform.forwardVersion = transform.inverseVersion = 0;
        DNDS_check_throw_info(transform.plan == _identity && transform.first >= 0 && transform.count >= 0 && transform.count <= 64,
                              "characteristic transform belongs to another plan or has an invalid range");
        const index count = transform.count;
        if (count == 0)
        {
            transform.forwardVersion = transform.inverseVersion = 0;
            transform.forwardIdentity = transform.inverseIdentity = 0;
            return;
        }
        CheckArray(transform.forward, Metrics().NumFaceProc(), count * count);
        CheckArray(transform.inverse, Metrics().NumFaceProc(), count * count);
        if (checkInverses)
        {
            auto forward = transform.forward.Read(), inverse = transform.inverse.Read();
            const auto left = forward.View<R::TableRows>(), right = inverse.View<R::TableRows>();
            OMP::ForSetupRows(0, Metrics().NumFaceProc(), [&](index iFace)
                              {
                Matrix product(count, count);
                const ConstMap L(left[iFace].data(), count, count), R(right[iFace].data(), count, count);
                product.noalias() = L * R;
                DNDS_check_throw_info(L.allFinite() && R.allFinite() && product.isApprox(Matrix::Identity(count, count), 1e-10),
                                      "face characteristic matrices must be finite mutual inverses"); });
        }
        transform.forwardVersion = transform.forward.Version();
        transform.inverseVersion = transform.inverse.Version();
        transform.forwardIdentity = transform.forward.Identity();
        transform.inverseIdentity = transform.inverse.Identity();
    }
    void VariationalReconstructionPlan::DoReconstructionIterTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                                               const ReconstructionBoundaryValues &boundary, ReconstructionRegion region) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceReconstructionIterTo(engine, stream, mean, input, work, boundary, region,
                                             ReconstructionDeviceKernel::FP64);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        DoReconstructionIterImpl(mean, input, work, boundary, region, nullptr, nullptr);
    }
    void VariationalReconstructionPlan::DoReconstructionIterLimitedTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                                                      const ReconstructionBoundaryValues &boundary, const FVField &target,
                                                                      const R::Array<real> &alpha, ReconstructionRegion region) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceReconstructionIterLimitedTo(engine, stream, mean, input, work, boundary, target, alpha, region, ReconstructionDeviceKernel::FP64);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        DoReconstructionIterImpl(mean, input, work, boundary, region, &target, &alpha);
    }
    void VariationalReconstructionPlan::SubmitReconstructionIterTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean, const FVField &input,
        ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary, ReconstructionRegion region,
        ReconstructionDeviceKernel kernel) const
    {
        if (_device)
        {
            SubmitDeviceReconstructionIterTo(engine, stream, mean, input, work, boundary, region, kernel);
            return;
        }
        DNDS_check_throw_info(kernel != ReconstructionDeviceKernel::FP32FullAccumulation &&
                                  kernel != ReconstructionDeviceKernel::FP32FloatFloatAccumulation &&
                                  kernel != ReconstructionDeviceKernel::FloatFloatPipelined &&
                                  kernel != ReconstructionDeviceKernel::FP32Pipelined && kernel != ReconstructionDeviceKernel::TF32Pipelined,
                              "reduced arithmetic reconstruction requires explicit CUDA placement");
        Require(ReconstructionOperation::Sweep);
        Validate();
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU reconstruction submission needs a matching host stream");
        SubmitHostSweep(engine, stream, _data,
                        Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux()),
                        mean, input, work, boundary, region, nullptr, nullptr);
    }
    void VariationalReconstructionPlan::SubmitReconstructionIterLimitedTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean, const FVField &input,
        ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary, const FVField &target,
        const R::Array<real> &alpha, ReconstructionRegion region, ReconstructionDeviceKernel kernel) const
    {
        if (_device)
        {
            SubmitDeviceReconstructionIterLimitedTo(engine, stream, mean, input, work, boundary, target, alpha, region, kernel);
            return;
        }
        Require(ReconstructionOperation::Sweep);
        Validate();
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU limited reconstruction submission needs a matching host stream");
        SubmitHostSweep(engine, stream, _data,
                        Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux()),
                        mean, input, work, boundary, region, &target, &alpha);
    }
    void VariationalReconstructionPlan::DoReconstructionIterImpl(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                                                 const ReconstructionBoundaryValues &boundary, ReconstructionRegion region,
                                                                 const FVField *target, const R::Array<real> *alpha) const
    {
        Require(ReconstructionOperation::Sweep);
        Validate();
        CheckSweep(*_data, mean, input, work, boundary, region, target, alpha);
        HostReconstructionRead retained(_data,
                                        Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux()),
                                        work, region, false);
        auto view = retained.Sweep(work.components);
        auto meanOwned = mean.Owned().Read(), recOwned = input.Owned().Read(), boundaryRead = boundary.values.Read();
        std::optional<R::Array<real>::ReadLease> meanGhost, recGhost, targetRead, alphaRead;
        if (region != ReconstructionRegion::Core)
        {
            meanGhost.emplace(mean.Ghost().Read());
            recGhost.emplace(input.Ghost().Read());
        }
        view.means = {meanOwned.View<R::UniformTableRows>(), meanGhost ? meanGhost->View<R::UniformTableRows>() : ReadRows{}};
        view.current = recOwned.View<R::UniformTableRows>();
        view.ghost = recGhost ? recGhost->View<R::UniformTableRows>() : ReadRows{};
        view.boundary = boundaryRead.View<R::TableRows>();
        if (target)
        {
            targetRead.emplace(target->Owned().Read());
            alphaRead.emplace(alpha->Read());
            view.target = targetRead->View<R::UniformTableRows>();
            view.penalty = alphaRead->View<R::UniformTableRows>();
            view.limited = true;
        }
        // A failed input lease is a rejected launch, so acquire the output
        // only after every read has been admitted.
        auto next = work.temporary.Owned().Write();
        view.output = next.View<R::UniformTableRows>();
        const auto status = CPU::ReconstructionSweep(view, retained.selection);
        DNDS_check_throw_info(status, "reconstruction penalty alpha must be finite and in [0,1]");
    }
    void VariationalReconstructionPlan::DoReconstructionIter(const FVField &mean, FVField &coefficients,
                                                             ReconstructionWorkspace &work,
                                                             const ReconstructionBoundaryValues &boundary,
                                                             ReconstructionDeviceKernel kernel) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceReconstructionIterTo(engine, stream, mean, coefficients, work, boundary,
                                             ReconstructionRegion::All, kernel);
            SubmitDeviceCommitReconstruction(engine, stream, coefficients, work);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        DNDS_check_throw_info(kernel != ReconstructionDeviceKernel::FP32FullAccumulation &&
                                  kernel != ReconstructionDeviceKernel::FP32FloatFloatAccumulation &&
                                  kernel != ReconstructionDeviceKernel::FloatFloatPipelined &&
                                  kernel != ReconstructionDeviceKernel::FP32Pipelined && kernel != ReconstructionDeviceKernel::TF32Pipelined,
                              "reduced arithmetic reconstruction requires explicit CUDA placement");
        DoReconstructionIterTo(mean, coefficients, work, boundary);
        CommitReconstruction(coefficients, work);
    }
    void VariationalReconstructionPlan::CommitReconstruction(FVField &coefficients, ReconstructionWorkspace &work) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceCommitReconstruction(engine, stream, coefficients, work);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        CheckWorkspace(_data.get(), work, coefficients);
        auto source = work.temporary.Owned().Read();
        auto destination = coefficients.Owned().Write();
        const auto src = source.View<R::UniformTableRows>();
        const auto dst = destination.View<R::UniformTableRows>();
        CPU::CommitReconstruction(src, dst);
    }
    void VariationalReconstructionPlan::SubmitCommitReconstruction(
        R::EagerEngine &engine, const R::ExecutionStream &stream, FVField &coefficients,
        ReconstructionWorkspace &work) const
    {
        if (_device)
        {
            SubmitDeviceCommitReconstruction(engine, stream, coefficients, work);
            return;
        }
        Validate();
        CheckWorkspace(_data.get(), work, coefficients);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU reconstruction commit needs a matching host stream");
        auto mesh = std::make_shared<Geom::MeshReadAccess>(
            Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux()));
        R::SubmitArrays(engine, stream, [plan = _data, mesh](R::ExecutionContext, auto source, auto destination)
                        { CPU::CommitReconstruction(source, destination); }, R::ReadAs<R::UniformTableRows>(work.temporary.Owned()), R::WriteAs<R::UniformTableRows>(coefficients.Owned()));
    }
    void VariationalReconstructionPlan::DoLimiterWBAP_3OrderTo(const FVField &coefficients,
                                                               ReconstructionWorkspace &work, int order,
                                                               FaceCharacteristicTransformRef transform, ReconstructionRegion region,
                                                               LimiterIndicatorRef indicator, real threshold) const
    {
        const auto *indicatorArray = indicator ? &indicator->get() : nullptr;
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, transform, region, indicatorArray, threshold);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        Require(ReconstructionOperation::WBAP);
        Validate();
        CheckLimiter(*_data, coefficients, work, order, transform, region, indicatorArray, threshold);
        HostReconstructionRead retained(_data,
                                        Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux()),
                                        work, region, true);
        auto view = retained.Limiter(work.components, order);
        auto owned = coefficients.Owned().Read();
        std::optional<R::Array<real>::ReadLease> ghost, forward, inverse, activation;
        if (region != ReconstructionRegion::Core)
            ghost.emplace(coefficients.Ghost().Read());
        view.coefficients = {owned.View<R::UniformTableRows>(), ghost ? ghost->View<R::UniformTableRows>() : ReadRows{}};
        if (transform && transform->get().count)
        {
            const auto &selected = transform->get();
            forward.emplace(selected.forward.Read());
            inverse.emplace(selected.inverse.Read());
            view.forward = forward->View<R::TableRows>();
            view.inverse = inverse->View<R::TableRows>();
            view.characteristicFirst = selected.first;
            view.characteristicCount = selected.count;
        }
        if (indicatorArray)
        {
            activation.emplace(indicatorArray->Read());
            view.indicator = activation->View<R::UniformTableRows>();
            view.hasIndicator = true;
        }
        auto output = work.temporary.Owned().Write(), scratch = work.scratch.Write();
        view.output = output.View<R::UniformTableRows>();
        view.scratch = scratch.View<R::TableRows>();
        view.threshold = threshold;
        const auto status = CPU::ReconstructionLimiterWBAP(view, retained.selection);
        DNDS_check_throw_info(status, "limiter indicator must be finite");
    }
    void VariationalReconstructionPlan::SubmitLimiterWBAP_3OrderTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &coefficients,
        ReconstructionWorkspace &work, int order,
        FaceCharacteristicTransformRef transform, ReconstructionRegion region,
        LimiterIndicatorRef indicator, real threshold) const
    {
        const auto *indicatorArray = indicator ? &indicator->get() : nullptr;
        if (_device)
        {
            SubmitDeviceLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, transform, region, indicatorArray, threshold);
            return;
        }
        Require(ReconstructionOperation::WBAP);
        Validate();
        CheckLimiter(*_data, coefficients, work, order, transform, region, indicatorArray, threshold);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU WBAP submission needs a matching host stream");
        auto retained = std::make_shared<HostReconstructionContext>(_data,
                                                                    Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux()), work, region);
        auto view = retained->LimiterPolicy(work.components, order);
        view.threshold = threshold;
        std::vector<ReadOperand> reads{R::ReadAs<R::UniformTableRows>(coefficients.Owned())};
        index ghostSlot = -1, forwardSlot = -1, inverseSlot = -1, indicatorSlot = -1;
        const auto add = [&](const R::Array<real> &array)
        {
            const index slot = index(reads.size());
            reads.push_back(R::ReadAs<R::UniformTableRows>(array));
            return slot;
        };
        if (region != ReconstructionRegion::Core)
            ghostSlot = add(coefficients.Ghost());
        std::optional<FaceCharacteristicTransform> publication;
        if (transform && transform->get().count)
        {
            publication = transform->get();
            forwardSlot = add(publication->forward);
            inverseSlot = add(publication->inverse);
            view.characteristicFirst = publication->first;
            view.characteristicCount = publication->count;
        }
        if (indicatorArray)
        {
            indicatorSlot = add(*indicatorArray);
            view.hasIndicator = true;
        }
        std::vector<WriteOperand> writes{R::WriteAs<R::UniformTableRows>(work.temporary.Owned()),
                                         R::WriteAs<R::UniformTableRows>(work.scratch)};
        R::SubmitArrayBatch(engine, stream, [retained, view, publication, ghostSlot, forwardSlot, inverseSlot, indicatorSlot](R::ExecutionContext, SpanView<const ReadRows> inputs, SpanView<const WriteRows> outputs, auto offsets, auto neighbors, auto faces, auto secondary, auto rotations, auto rotationActive) mutable
                            {
                                view.offsets = offsets;
                                view.neighbors = neighbors;
                                view.faces = faces;
                                view.secondary = secondary;
                                view.rotations = rotations;
                                view.rotationActive = rotationActive;
                                // Frozen publications retain their completed-version
                                // contract even if an earlier queued producer changed
                                // them after this operation was submitted.
                                if (publication && publication->validation == CharacteristicValidation::Frozen)
                                    DNDS_check_throw_info(publication->forwardVersion == publication->forward.Version() &&
                                                              publication->inverseVersion == publication->inverse.Version(),
                                                          "frozen characteristic matrices changed before queued limiting");
                                view.coefficients = {inputs[0], ghostSlot >= 0 ? inputs[ghostSlot] : ReadRows{}};
                                if (publication)
                                {
                                    view.forward = TableView(inputs[forwardSlot]);
                                    view.inverse = TableView(inputs[inverseSlot]);
                                }
                                if (view.hasIndicator)
                                    view.indicator = inputs[indicatorSlot];
                                view.output = outputs[0];
                                view.scratch = TableView(outputs[1]);
                                const auto status = CPU::ReconstructionLimiterWBAP(view, retained->selection);
                                DNDS_check_throw_info(status, "limiter indicator must be finite"); }, std::move(reads), std::move(writes), std::make_tuple(R::ReadAs<R::FixedRows<1>>(_data->offsets), R::ReadAs<R::FixedRows<1>>(_data->neighbors), R::ReadAs<R::FixedRows<1>>(_data->faces), R::ReadAs<R::TableRows>(*_data->secondary), R::ReadAs<R::FixedRows<9>>(_data->neighborRotation), R::ReadAs<R::FixedRows<1>>(_data->neighborRotationActive)));
    }
    void VariationalReconstructionPlan::DoLimiterWBAP_3Order(FVField &coefficients,
                                                             ReconstructionWorkspace &work, int order,
                                                             FaceCharacteristicTransformRef transform) const
    {
        DoLimiterWBAP_3OrderTo(coefficients, work, order, transform);
        CommitReconstruction(coefficients, work);
    }
    void VariationalReconstructionPlan::Evaluate(const FVField &mean, const FVField &coefficients,
                                                 const R::Array<index> &cells, const R::Array<real> &points, R::Array<real> &values) const
    {
        EvaluateDerivative(mean, coefficients, cells, points, values, 0);
    }
    void VariationalReconstructionPlan::EvaluateDerivative(const FVField &mean, const FVField &coefficients,
                                                           const R::Array<index> &cells, const R::Array<real> &points,
                                                           R::Array<real> &values, int derivative) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const index nv = mean.Width(), n = NRecDOF();
        DNDS_check_throw_info(derivative >= 0 && derivative <= n, "polynomial derivative ordinal outside basis");
        Metrics().ValidateField(mean, nv), Metrics().ValidateField(coefficients, n * nv);
        CheckArray(points, cells.RowCount(), 3), CheckArray(values, cells.RowCount(), nv);
        DNDS_check_throw_info(cells.GetDevice() == R::Device::Host() && !cells.Layout().IsVariable() && cells.Layout().Width() == 1, "sampling cell index shape");
        auto cellRead = cells.Read();
        const auto ids = cellRead.View<R::FixedRows<1>>();
        OMP::ForSetupRows(0, cells.RowCount(), [&](index iPoint)
                          { DNDS_check_throw_info(ids(iPoint, 0) >= 0 && ids(iPoint, 0) < Metrics().NumCellProc(), "sampling cell outside prepared closure"); });
        auto mo = mean.Owned().Read(), mg = mean.Ghost().Read(), co = coefficients.Owned().Read(), cg = coefficients.Ghost().Read();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{mo.View<R::UniformTableRows>(), mg.View<R::UniformTableRows>()}, rec{co.View<R::UniformTableRows>(), cg.View<R::UniformTableRows>()};
        auto pRead = points.Read(), geomRead = Metrics()._data->cell.Read(), momentRead = _data->moments.Read();
        auto output = values.Write();
        const auto p = pRead.View<R::FixedRows<3>>();
        const auto geom = geomRead.View<R::FixedRows<8>>();
        const auto moment = momentRead.View<R::TableRows>();
        const auto out = output.View<R::UniformTableRows>();
        OMP::ForRows(0, cells.RowCount(), [&](index iPoint)
                     {
            Math::SmallMatrix basis;
            const auto cell = ids(iPoint, 0);
            Math::PolynomialDerivatives(basis, Metrics().Dimension(), Settings().maxOrder, {p(iPoint, 0), p(iPoint, 1), p(iPoint, 2)},
                                        {geom(cell, 1), geom(cell, 2), geom(cell, 3)}, geom(cell, 7));
            for (index v = 0; v < nv; ++v)
            {
                real value = derivative == 0 ? u(cell, v) : 0;
                for (index mode = 0; mode < n; ++mode)
                    value += (basis(derivative, mode + 1) - (derivative == 0 ? moment(cell, mode + 1) : 0)) * rec(cell, v * n + mode);
                out(iPoint, v) = value;
            } });
    }
    Eigen::MatrixXd VariationalReconstructionPlan::GetCachedFaceBasis(index iFace, int side, int iQuadrature) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        DNDS_check_throw_info(Settings().cacheDiffBase && iFace >= 0 && iFace < Metrics().NumFaceProc() && (side == 0 || side == 1),
                              "basis cache disabled or face/side outside prepared data");
        DNDS_check_throw_info(meshRead.HostView().FaceCells(iFace)[side] != UnInitIndex, "exterior face parent has no basis cache");
        const auto parents = meshRead.HostView().FaceCells(iFace);
        DNDS_check_throw_info(parents[0] < Metrics().NumCell() || (parents[1] != UnInitIndex && parents[1] < Metrics().NumCell()),
                              "face basis outside owned cell stencils was not prepared");
        auto offsets = Metrics()._data->faceOffsets.Read();
        const auto offset = offsets.View<R::FixedRows<1>>();
        const index begin = offset(iFace, 0), end = offset(iFace + 1, 0);
        DNDS_check_throw_info(iQuadrature >= 0 && iQuadrature < end - begin, "basis cache quadrature outside face");
        auto read = _data->faceBasis.Read();
        const auto view = read.View<R::TableRows>();
        return ConstMap(view[2 * (begin + iQuadrature) + side].data(), _data->cachedDerivatives, NRecDOF());
    }
    ReconstructionBasisRead::ReconstructionBasisRead(std::shared_ptr<const Detail::ReconstructionData> data)
        : _data(std::move(data)), _geometry(_data->metrics.AcquireRead()), _moments(_data->moments.Read()),
          _meshRead(std::make_shared<Geom::MeshReadAccess>(_data->metrics._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux())))
    {
        if (_data->settings.cacheDiffBase)
        {
            _basis.emplace(_data->faceBasis.Read());
            _basisView = _basis->View<R::TableRows>();
        }
        // Freeze plain pointers once while the enclosing read scopes pin both
        // metadata and allocations. Face queries acquire no new Array views.
        _momentView = _moments.View<R::TableRows>();
        _evaluation = {_basisView, _momentView, _geometry.HostView(),
                       _meshRead->HostView(), _data->metrics.Dimension(),
                       _data->settings.maxOrder, _data->n,
                       _data->cachedDerivatives};
    }
    ReconstructionBasisHostView ReconstructionBasisRead::HostView() const
    {
        DNDS_check_throw_info(bool(_basis), "face basis caching was disabled");
        return {_basisView, _geometry.HostView().faceOffsets, _data->n, _data->cachedDerivatives};
    }
    Math::ReconstructionBasisView ReconstructionBasisRead::FaceBasis(index face, int side, index point,
                                                                     int derivativeOrder, Math::SmallMatrix &scratch) const
    {
        return CPU::EvaluateFaceBasis(EvaluationView(), face, side, point,
                                      derivativeOrder, scratch);
    }
    const ReconstructionBasisEvaluationView &ReconstructionBasisRead::EvaluationView() const
    {
        return _evaluation;
    }
    ReconstructionBasisRead VariationalReconstructionPlan::AcquireBasisRead() const
    {
        Validate();
        return ReconstructionBasisRead(_data);
    }
}
