#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "Cauchy/Interface/CPU/InterfacePacketKernels.hpp"
#include "Cauchy/Interface/CPU/InterfaceSamplingKernels.hpp"
#include "Cauchy/Interface/InterfacePacketOperator.hpp"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <limits>
#include <atomic>
#include <map>
#include <nlohmann/json.hpp>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        using SchemaJson = nlohmann::ordered_json;

        std::shared_ptr<R::CommunicationContext> CouplingContext(
            const SimulationConfig &config,
            std::shared_ptr<R::CommunicationContext> context)
        {
            DNDS_check_throw_info(context, "interface coupling requires an owning communication context");
            // Every control rank has the complete resolved domain list,
            // including dislocated nonmembers. Any requesting domain enables
            // agreement for this shared protocol; local membership must never
            // decide whether a collective is entered. WithPreflight shares
            // the communicator and tag namespace, without duplication.
            bool collectiveChecks = false;
            for (const auto &domain : config.Domains())
                collectiveChecks |= domain.settings.implicit.collectiveChecks;
            return context->ChecksCollectivePreflight() == collectiveChecks
                       ? std::move(context)
                       : context->WithPreflight(collectiveChecks);
        }

        std::string DescribeSchema(const EquationSchema &schema)
        {
            SchemaJson blocks = SchemaJson::array();
            for (const auto &block : schema.blocks)
            {
                SchemaJson components = SchemaJson::array();
                for (const auto &component : block.components)
                    components.push_back(
                        {{"name", component.name},
                         {"units", component.units},
                         {"conservation_group", component.conservationGroup}});
                blocks.push_back({{"name", block.name},
                                  {"components", std::move(components)},
                                  {"vector_starts", block.vectorStarts}});
            }
            return SchemaJson{{"family", schema.family},
                              {"energy_convention", schema.energyConvention},
                              {"physical_accumulation", schema.physicalAccumulation},
                              {"blocks", std::move(blocks)}}
                .dump();
        }

        EquationSchema ParseSchema(const std::string &description)
        {
            const auto json = SchemaJson::parse(description);
            EquationSchema schema;
            schema.family = json.at("family").get<std::string>();
            schema.energyConvention =
                json.at("energy_convention").get<std::string>();
            schema.physicalAccumulation =
                json.at("physical_accumulation").get<std::string>();
            for (const auto &encodedBlock : json.at("blocks"))
            {
                EquationBlockSchema block;
                block.name = encodedBlock.at("name").get<std::string>();
                block.vectorStarts =
                    encodedBlock.at("vector_starts").get<std::vector<index>>();
                for (const auto &encodedComponent :
                     encodedBlock.at("components"))
                    block.components.push_back(
                        {encodedComponent.at("name").get<std::string>(),
                         encodedComponent.at("units").get<std::string>(),
                         encodedComponent.at("conservation_group")
                             .get<std::string>()});
                schema.blocks.push_back(std::move(block));
            }
            return schema;
        }

        void BroadcastString(std::string &value, int root, MPI_Comm communicator)
        {
            index size = index(value.size());
            MPI::Bcast(&size, 1, MPI_INT64_T, root, communicator);
            DNDS_check_throw_info(
                size >= 0 && size <= std::numeric_limits<int>::max(),
                "interface schema descriptor is too large");
            value.resize(size_t(size));
            if (size)
                MPI::Bcast(value.data(), int(size), MPI_CHAR, root,
                           communicator);
        }
    }
    struct InterfaceCoupling::Data
    {
        /// Exclusivity belongs to an operation, not a CPU thread. Submission
        /// and publication may occur on different workers; std::mutex ownership
        /// must not be moved across that boundary.
        class OperationLease
        {
            std::shared_ptr<std::atomic<bool>> _busy;

        public:
            OperationLease(std::shared_ptr<std::atomic<bool>> busy, bool terminal)
                : _busy(std::move(busy))
            {
                bool expected = false;
                const bool acquired = _busy->compare_exchange_strong(expected, true);
                if (terminal)
                    DNDS_assert_info(acquired, "concurrent interface publication");
                else
                    DNDS_check_throw_info(acquired, "interface coupling already has an active operation");
            }
            OperationLease(const OperationLease &) = delete;
            OperationLease &operator=(const OperationLease &) = delete;
            OperationLease(OperationLease &&) noexcept = default;
            OperationLease &operator=(OperationLease &&) noexcept = delete;
            ~OperationLease() { Release(); }
            void Release() noexcept
            {
                if (_busy)
                {
                    _busy->store(false, std::memory_order_release);
                    _busy.reset();
                }
            }
        };
        struct Connection
        {
            struct DonorField
            {
                R::MetadataBuffer<index> logicalComponents;
                std::optional<CFV::DistributedSamplingWorkspace> workspace;
                R::Array<real> values;
                std::optional<R::TransferBinding> result;
            };
            index donor, receiver, protocol = 0;
            InterfaceSettings settings;
            std::shared_ptr<R::CommunicationContext> context;
            std::optional<Geom::BoundaryPatch> patch;
            // A single distributed match/sample batch, with explicit output
            // destinations in the quadrature and face-centered packet tables.
            R::Array<InterfacePacketRow> receiverRows{R::RowLayout::Dense(0, 1)};
            R::Array<real> receiverReferencePoints{R::RowLayout::Dense(0, 3)};
            struct StageRoute
            {
                std::optional<std::pair<Geom::RigidPose, Geom::RigidPose>> poses;
                std::optional<CFV::DistributedInterfaceSamplingPlan> sampling;
                std::vector<DonorField> fields;
                std::optional<InterfacePacketOperator> packet;
                std::optional<StateEpoch> epoch;
                uint64_t mapGeneration = 0;
            };
            // DITR needs both poses and persistent bindings alive throughout
            // its nonlinear loop. Ordinary DIRK reuses slot zero between stages.
            std::map<uint64_t, StageRoute> stageRoutes;
            /// Collective, immutable donor physical grouping: mode, logical
            /// component count, then one logical-component map per field.
            std::vector<index> donorDescriptor;
            /// Exact logical schema and donor-ordinal to receiver-ordinal map.
            /// The latter is broadcast from a real receiver rank, so all
            /// coupling ranks share one setup-time compatibility decision.
            std::string donorSchema;
            R::MetadataBuffer<index> receiverComponents;
            R::MetadataBuffer<index> vectorStarts;
            // Initial history is empty but retains the same fixed width as
            // every later publication, including nonmember/zero-query ranks.
            R::Array<index> lastDonors{R::RowLayout::Dense(0, 1)};
        };
        std::shared_ptr<std::atomic<bool>> operationBusy =
            std::make_shared<std::atomic<bool>>(false);
        // One controller operation at a time; do not wait behind an MPI path.
        // Commit/discard remain allocation-free, nonthrowing publication.
        OperationLease Enter(bool publication = false) const
        {
            return OperationLease(operationBusy, publication);
        }
        std::shared_ptr<R::CommunicationContext> control;
        MPIInfo mpi;
        // A receiver's queued accesses must use its execution.checkAccess
        // policy, including while its interface writer is still pending.
        // The Runtime deliberately rejects overlapping checked/unchecked
        // reservations. Share one lane pair per policy, not per connection;
        // both run the identical transfer/mapping sequence below.
        struct ValueLane
        {
            R::EagerEngine engine;
            R::ExecutionStream compute, communication;
            explicit ValueLane(bool checked, R::Device device, size_t workers)
                : engine(workers, checked ? std::make_shared<R::AccessTrackingContext>() : nullptr)
            {
                if (device == R::Device::Host())
                {
                    compute = engine.CreateCPUStream();
                    communication = engine.CreateCPUStream();
                }
                else
                {
                    compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal));
                    communication = engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal));
                }
            }
        };
        std::unique_ptr<ValueLane> checkedValues, uncheckedValues;
        std::vector<bool> receiverChecks;
        ValueLane &Lane(index receiver)
        {
            return receiverChecks.at(size_t(receiver)) ? *checkedValues : *uncheckedValues;
        }
        std::vector<std::optional<FiniteVolume>> discrete;
        std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>> workers;
        std::vector<RigidMotion> motion;
        std::vector<std::optional<BoundaryInputs>> incoming;
        std::vector<std::optional<BoundaryInputs>> staging;
        std::vector<std::optional<BoundaryInputs>> accepted;
        bool useAccepted = false;
        bool refreshEveryNonlinearIteration = false;
        std::vector<Connection> connections;
        index changedDonors = 0;
        uint64_t mapBuilds = 0;
        real maximumGap = 0;
        explicit Data(std::shared_ptr<R::CommunicationContext> owner)
            : control(std::move(owner))
        {
            mpi = MPIInfo(control->Communicator());
        }
    };
    struct InterfaceCoupling::RefreshOperation::State
    {
        std::shared_ptr<InterfaceCoupling::Data> data;
        InterfaceCoupling::Data::OperationLease operation;
        real time = 0;
        StateEpoch epoch;
        // Keep sampled local inputs alive independently of the caller's
        // temporary descriptor vector. Receiver packets may retain these
        // snapshots after completion; there is no back-reference to this
        // operation, its engine, or the interface controller.
        std::vector<std::shared_ptr<const InterfaceDonorPublication>> donors;
        std::vector<std::vector<BoundaryInterfaceSource>> sources;
        std::optional<std::vector<std::optional<BoundaryInputs>>> published;
        std::vector<std::optional<BoundaryInputs>> pending;
        R::StreamEvent completion;
        bool finished = false;

        State(std::shared_ptr<InterfaceCoupling::Data> owner,
              InterfaceCoupling::Data::OperationLease lock, real physicalTime,
              StateEpoch stateEpoch)
            : data(std::move(owner)), operation(std::move(lock)),
              time(physicalTime), epoch(stateEpoch),
              donors(data->discrete.size()), sources(data->discrete.size())
        {
        }
    };
    InterfaceCoupling::InterfaceCoupling(
        const SimulationConfig &config,
        std::shared_ptr<Runtime::CommunicationContext> context,
        const std::vector<std::optional<FiniteVolume>> &localDomains,
        const std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>> &workers)
        : _data(std::make_shared<Data>(CouplingContext(config, std::move(context))))
    {
        auto &data = *_data;
        for (const auto &domain : config.Domains())
            data.receiverChecks.push_back(domain.settings.execution.checkAccess);
        const auto coupling = data.mpi.comm;
        if (!config.Interfaces().empty())
            data.refreshEveryNonlinearIteration =
                config.Interfaces().front().refreshCadence ==
                "nonlinearIteration";
        data.control->LocalStage([&]
                                 {
            DNDS_check_throw_info(localDomains.size()==config.Domains().size(),"interface domain descriptor count differs");
            data.discrete.resize(localDomains.size());
            data.workers = workers;
            if (data.workers.empty())
                data.workers.resize(localDomains.size());
            DNDS_check_throw_info(data.workers.size() == localDomains.size(),
                                  "interface chemistry-worker descriptor count differs");
            data.incoming.resize(localDomains.size());
            data.staging.resize(localDomains.size());
            data.accepted.resize(localDomains.size());
            std::optional<R::Device> localDevice;
            size_t hostThreads = 0;
            for(index iDomain=0;iDomain<index(localDomains.size());++iDomain)
                if(localDomains[iDomain])
                {
                    const auto device = localDomains[iDomain]->Device();
                    DNDS_check_throw_info(!localDevice || *localDevice == device,
                        "interface coupling uses one execution device per rank; local domains must share it");
                    localDevice = device;
                    hostThreads = std::max(hostThreads, size_t(config.Domains()[size_t(iDomain)].settings.execution.hostThreads));
                }
            data.checkedValues = std::make_unique<Data::ValueLane>(true, localDevice.value_or(R::Device::Host()), hostThreads);
            data.uncheckedValues = std::make_unique<Data::ValueLane>(false, localDevice.value_or(R::Device::Host()), hostThreads);
            for(const auto &domain:config.Domains()) data.motion.emplace_back(domain.settings.motion,domain.settings.mesh.dimension);
            for(index iDomain=0;iDomain<index(localDomains.size());++iDomain)
                if(localDomains[iDomain])
                {
                    data.discrete[iDomain]=*localDomains[iDomain];
                    if(localDomains[iDomain]->HasInterfaces())
                    {
                        data.incoming[iDomain]=localDomains[iDomain]->BuildBoundaryInputs(data.workers[iDomain], data.Lane(iDomain).compute);
                        data.staging[iDomain]=localDomains[iDomain]->BuildBoundaryInputs(data.workers[iDomain], data.Lane(iDomain).compute);
                    }
                } });
        for (const auto &settings : config.Interfaces())
        {
            Data::Connection connection;
            connection.settings = settings;
            connection.protocol = index(data.connections.size());
            const auto domainIndex = [&](const std::string &name)
            {
                const auto found = std::find_if(config.Domains().begin(), config.Domains().end(), [&](const auto &domain)
                                                { return domain.placement.name == name; });
                return index(found - config.Domains().begin());
            };
            connection.donor = domainIndex(settings.donor);
            connection.receiver = domainIndex(settings.receiver);
            // Connections share the coupling communicator owner. Persistent
            // result bindings reserve distinct agreed channels below; transient
            // geometric matching retains its own short-lived search namespace.
            connection.context = data.control;
            // Patch construction is domain-collective. A local exception must
            // propagate to the fatal boundary, not to a coupling reduction
            // while another donor rank is still constructing the patch.
            {
                if (auto &fv = data.discrete[connection.donor])
                    connection.patch = fv->BuildBoundaryPatch(settings.donorZone);
                if (auto &fv = data.discrete[connection.receiver])
                {
                    std::vector<InterfacePacketRow> selectedRows;
                    const auto pointRead = fv->ReadBoundaryPoints();
                    const auto points = pointRead.View<R::FixedRows<1>>();
                    for (index iPoint = 0; iPoint < points.RowCount(); ++iPoint)
                        if (points(iPoint, 0).interface && points(iPoint, 0).zone == settings.receiverZone)
                            selectedRows.push_back({iPoint, false});
                    const auto faceRead = fv->ReadBoundaryFaces();
                    const auto faces = faceRead.View<R::FixedRows<1>>();
                    for (index iFace = 0; iFace < faces.RowCount(); ++iFace)
                        if (faces(iFace, 0).interface && faces(iFace, 0).zone == settings.receiverZone)
                            selectedRows.push_back({iFace, true});
                    connection.receiverRows = R::Array<InterfacePacketRow>(
                        R::RowLayout::Dense(selectedRows.size(), 1));
                    connection.receiverReferencePoints = R::Array<real>(
                        R::RowLayout::Dense(selectedRows.size(), 3));
                    auto write = connection.receiverRows.Write();
                    auto pointWrite = connection.receiverReferencePoints.Write();
                    const auto rows = write.View<R::FixedRows<1>>();
                    const auto locations = pointWrite.View<R::FixedRows<3>>();
                    for (index iRow = 0; iRow < rows.RowCount(); ++iRow)
                    {
                        rows(iRow, 0) = selectedRows[size_t(iRow)];
                        const auto row = selectedRows[size_t(iRow)];
                        const auto point = row.faceCenter
                                               ? faces(row.row, 0).point
                                               : points(row.row, 0).point;
                        for (int d = 0; d < 3; ++d)
                            locations(iRow, d) = point[size_t(d)];
                    }
                    std::vector<index> vectors;
                    index first = 0;
                    for (const auto &block : fv->Equation().Schema().blocks)
                    {
                        for (const auto start : block.vectorStarts)
                            vectors.push_back(first + start);
                        first += index(block.components.size());
                    }
                    connection.vectorStarts = R::MetadataBuffer<index>(vectors);
                }
            }
            index counts[2]{connection.patch ? connection.patch->Size() : 0, connection.receiverRows.RowCount()};
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<index>{counts, 2}, coupling);
            DNDS_check_throw_info(counts[0] > 0 && counts[1] > 0, "interface donor patch or receiver boundary quadrature is globally empty; check zone IDs");
            data.connections.push_back(std::move(connection));
        }
    }
    bool InterfaceCoupling::RefreshEveryNonlinearIteration() const
    {
        return _data->refreshEveryNonlinearIteration;
    }
    std::optional<BoundaryInputs> InterfaceCoupling::Inputs(index domain) const
    {
        auto session = _data->Enter();
        DNDS_check_throw_info(domain >= 0 && domain < index(_data->incoming.size()), "invalid interface receiver domain");
        const auto &packets = _data->useAccepted ? _data->accepted : _data->incoming;
        return packets[domain];
    }
    void InterfaceCoupling::Commit() noexcept
    {
        auto session = _data->Enter(true);
        _data->incoming.swap(_data->accepted);
        _data->useAccepted = true;
    }
    void InterfaceCoupling::Discard() noexcept
    {
        auto session = _data->Enter(true);
        _data->useAccepted = true;
    }
    InterfaceDiagnostics InterfaceCoupling::Diagnostics() const
    {
        auto session = _data->Enter();
        InterfaceDiagnostics result;
        for (const auto &connection : _data->connections)
            result.queryPoints += connection.receiverRows.RowCount();
        result.changedDonors = _data->changedDonors;
        index counts[2]{result.queryPoints, result.changedDonors};
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(
            MPI::ContiguousView<index>{counts, 2}, _data->mpi.comm);
        result.queryPoints = counts[0];
        result.changedDonors = counts[1];
        result.maximumGap = _data->maximumGap;
        result.maximumGap = MPI::AllreduceValue<MPI::Reduction::Max>(
            result.maximumGap, _data->mpi.comm);
        result.mapBuilds = _data->mapBuilds;
        return result;
    }
    InterfaceCoupling::RefreshOperation InterfaceCoupling::BeginRefreshImpl(
        const std::vector<InterfaceDomainInput> &domains, real time,
        StateEpoch epoch, bool valuesOnly)
    {
        auto state = std::make_shared<RefreshOperation::State>(
            _data, _data->Enter(), time, epoch);
        auto &data = *_data;
        std::vector<std::pair<index, R::TransferOperation>> transfers;
        data.control->LocalStage([&]
                                 {
            DNDS_check_throw_info(domains.size()==data.discrete.size()&&std::isfinite(time),"invalid interface refresh descriptors/time");
            for(index iDomain=0;iDomain<index(domains.size());++iDomain)
            {
                const auto &input=domains[iDomain];
                DNDS_check_throw_info(bool(input.values)==bool(data.discrete[iDomain]),
                                      "interface refresh has incomplete or changed domain participation");
                if (input.values)
                {
                    input.values->Validate();
                    state->donors[size_t(iDomain)] =
                        std::make_shared<const InterfaceDonorPublication>(*input.values);
                }
                if(data.discrete[iDomain]&&data.discrete[iDomain]->HasInterfaces())
                {
                    if(!data.staging[iDomain])
                        data.staging[iDomain]=data.discrete[iDomain]->BuildBoundaryInputs(data.workers[iDomain], data.Lane(iDomain).compute);
                    else data.discrete[iDomain]->BeginBoundaryUpdate(*data.staging[iDomain], data.Lane(iDomain).compute);
                }
            } });
        for (auto &connection : data.connections)
        {
            auto &donor = data.discrete[connection.donor];
            auto &receiver = data.discrete[connection.receiver];
            auto &route = connection.stageRoutes[epoch.coupledStage];
            const auto donorPose = data.motion[connection.donor].At(time).pose;
            const auto receiverPose = data.motion[connection.receiver].At(time).pose;
            const auto samePose = [](const auto &a, const auto &b)
            { return a.rotation == b.rotation && a.translation == b.translation; };
            const auto describe = [](const InterfaceDonorPublication &publication)
            {
                const auto &values = publication._means;
                std::vector<std::vector<index>> logical(
                    size_t(values.PhysicalFieldNumber()));
                for (index iField = 0; iField < values.PhysicalFieldNumber();
                     ++iField)
                    logical[size_t(iField)].assign(
                        values.PhysicalField(iField).Width(), -1);
                index first = 0;
                for (index iBlock = 0; iBlock < values.BlockNumber(); ++iBlock)
                {
                    const auto location = values.LocateBlock(iBlock);
                    for (index j = 0; j < location.componentNumber; ++j)
                        logical[size_t(location.physicalField)]
                               [size_t(location.firstComponent + j)] = first + j;
                    first += location.componentNumber;
                }
                std::vector<index> result{first, values.PhysicalFieldNumber()};
                for (const auto &field : logical)
                    result.push_back(index(field.size()));
                for (const auto &field : logical)
                    result.insert(result.end(), field.begin(), field.end());
                return result;
            };
            const auto &publication = state->donors[size_t(connection.donor)];
            std::vector<index> localDescriptor;
            if (publication)
                localDescriptor = describe(*publication);
            std::vector<index> descriptor;
            if (!valuesOnly)
            {
                int descriptorRoot = publication ? data.mpi.rank : data.mpi.size;
                MPI::Allreduce(MPI_IN_PLACE, &descriptorRoot, 1, MPI_INT, MPI_MIN,
                               data.mpi.comm);
                DNDS_check_throw_info(descriptorRoot >= 0 && descriptorRoot < data.mpi.size,
                                      "interface connection has no donor publication");
                // Resolve semantic component compatibility once, independently of
                // physical field grouping. Same-width equations are not sufficient:
                // species names, units, conservation groups and vector-frame roles
                // must match. The exact equation/material configuration was already
                // required equal in the constructor; this runtime map proves that
                // the published schema still matches the receiver.
                std::string localSchema;
                if (publication)
                    localSchema = DescribeSchema(publication->Schema());
                if (connection.receiverComponents.empty())
                {
                    std::string donorSchema = data.mpi.rank == descriptorRoot
                                                  ? localSchema
                                                  : std::string{};
                    BroadcastString(donorSchema, descriptorRoot, data.mpi.comm);
                    data.control->LocalStage([&]
                                             {
                    if(publication)
                        DNDS_check_throw_info(localSchema==donorSchema,
                                              "interface donor ranks publish different equation schemas"); });
                    EquationSchema schema;
                    data.control->LocalStage([&]
                                             { schema = ParseSchema(donorSchema); });
                    int receiverRoot = receiver ? data.mpi.rank : data.mpi.size;
                    MPI::Allreduce(MPI_IN_PLACE, &receiverRoot, 1, MPI_INT, MPI_MIN,
                                   data.mpi.comm);
                    DNDS_check_throw_info(
                        receiverRoot >= 0 && receiverRoot < data.mpi.size,
                        "interface connection has no receiver equation schema");
                    std::vector<index> mapping;
                    data.control->LocalStage([&]
                                             {
                    if(receiver)
                        mapping=schema.MapComponentsTo(receiver->Equation().Schema()); });
                    index mappingSize = data.mpi.rank == receiverRoot
                                            ? index(mapping.size())
                                            : 0;
                    MPI::Bcast(&mappingSize, 1, MPI_INT64_T, receiverRoot,
                               data.mpi.comm);
                    DNDS_check_throw_info(
                        mappingSize > 0 &&
                            mappingSize <= std::numeric_limits<int>::max(),
                        "invalid interface equation component mapping size");
                    if (data.mpi.rank != receiverRoot)
                        mapping.resize(size_t(mappingSize));
                    MPI::Bcast(mapping.data(), int(mappingSize), MPI_INT64_T,
                               receiverRoot, data.mpi.comm);
                    data.control->LocalStage([&]
                                             {
                    if(receiver)
                        DNDS_check_throw_info(
                            mapping==schema.MapComponentsTo(receiver->Equation().Schema()),
                            "interface receiver ranks resolve different equation mappings"); });
                    connection.donorSchema = std::move(donorSchema);
                    connection.receiverComponents = R::MetadataBuffer<index>(mapping);
                }
                else
                    data.control->LocalStage([&]
                                             {
                    if(publication)
                        DNDS_check_throw_info(
                            localSchema==connection.donorSchema,
                            "interface donor equation schema changed after setup"); });
                int descriptorSize = data.mpi.rank == descriptorRoot
                                         ? int(localDescriptor.size())
                                         : 0;
                MPI::Bcast(&descriptorSize, 1, MPI_INT, descriptorRoot, data.mpi.comm);
                DNDS_check_throw_info(descriptorSize >= 2,
                                      "invalid interface donor field descriptor");
                descriptor.assign(static_cast<size_t>(descriptorSize), index{});
                if (data.mpi.rank == descriptorRoot)
                    descriptor = localDescriptor;
                MPI::Bcast(descriptor.data(), descriptorSize, MPI_INT64_T,
                           descriptorRoot, data.mpi.comm);
                data.control->LocalStage([&]
                                         {
                if (publication)
                    DNDS_check_throw_info(
                        localDescriptor == descriptor,
                        "interface donor ranks use different physical field grouping");
                const index fields = descriptor[1];
                DNDS_check_throw_info(
                    fields > 0 && descriptor.size() >= size_t(2 + fields),
                    "invalid interface donor field count");
                index described = 0;
                for (index iField = 0; iField < fields; ++iField)
                {
                    const index width = descriptor[size_t(2 + iField)];
                    DNDS_check_throw_info(width > 0 &&
                                              described <= descriptor[0] - width,
                                          "invalid interface donor field width");
                    described += width;
                }
                DNDS_check_throw_info(
                    described == descriptor[0] &&
                        descriptor.size() == size_t(2 + fields + described),
                    "interface donor descriptor does not cover its equation");
                if (receiver)
                    DNDS_check_throw_info(
                        data.staging[connection.receiver]
                                ->QuadratureValues()
                                .ComponentNumber() == descriptor[0] &&
                            index(connection.receiverComponents.size()) ==
                                descriptor[0],
                        "interface donor and receiver equation schemas differ");
                if (!connection.donorDescriptor.empty())
                    DNDS_check_throw_info(
                        connection.donorDescriptor == descriptor,
                        "interface donor physical grouping changed after setup"); });
                if (connection.donorDescriptor.empty())
                    connection.donorDescriptor = descriptor;
            }
            else
            {
                data.control->LocalStage([&]
                                         {
                    DNDS_check_throw_info(
                        route.epoch &&
                            route.epoch->acceptedStep == epoch.acceptedStep &&
                            route.epoch->trial == epoch.trial &&
                            route.epoch->stage == epoch.stage,
                        "interface value refresh has no route for this physical stage");
                    DNDS_check_throw_info(
                        route.poses && route.sampling &&
                            !route.fields.empty() &&
                            !connection.donorDescriptor.empty() &&
                            !connection.receiverComponents.empty(),
                        "interface value refresh has no frozen topology");
                    DNDS_check_throw_info(
                        samePose(route.poses->first, donorPose) &&
                            samePose(route.poses->second, receiverPose),
                        "interface pose changed inside a frozen nonlinear stage");
                    if(publication)
                    {
                        DNDS_check_throw_info(
                            localDescriptor == connection.donorDescriptor,
                            "interface donor grouping changed inside a frozen stage");
                        DNDS_check_throw_info(
                            DescribeSchema(publication->Schema()) ==
                                connection.donorSchema,
                            "interface donor schema changed inside a frozen stage");
                    } });
                descriptor = connection.donorDescriptor;
            }
            if (!valuesOnly &&
                (!route.poses ||
                 !samePose(route.poses->first, donorPose) ||
                 !samePose(route.poses->second, receiverPose)))
            {
                R::Array<real> points;
                R::Array<index> ids;
                std::optional<Geom::BoundaryPatch> patch;
                std::optional<CFV::FiniteVolumePlan> metrics;
                std::optional<CFV::VariationalReconstructionPlan> reconstruction;
                data.control->LocalStage([&]
                                         {
                    points=R::Array<real>(R::RowLayout::Dense(connection.receiverRows.RowCount(),3));
                    ids=R::Array<index>(R::RowLayout::Dense(connection.receiverRows.RowCount(),1));
                    if(donor)
                    {
                        patch=connection.patch->WithPose(donorPose);
                        metrics=donor->Metrics();
                        const auto polynomial=donor->PolynomialReconstruction();
                        if(polynomial) reconstruction=*polynomial;
                    }
                    if(receiver)
                    {
                        const auto frame=receiver->Motion().At(time);
                        auto p=points.Write(); auto key=ids.Write();
                        const auto reference = connection.receiverReferencePoints.Read();
                        CPU::InterfaceQueryRequest request;
                        request.referencePoints = reference.View<R::FixedRows<3>>();
                        request.worldPoints = p.View<R::FixedRows<3>>();
                        request.ids = key.View<R::FixedRows<1>>();
                        request.moving = frame.moving;
                        for (int d = 0; d < 3; ++d)
                        {
                            request.translation[size_t(d)] = frame.pose.translation[d];
                            for (int e = 0; e < 3; ++e)
                                request.rotation[size_t(3*d+e)] = frame.pose.rotation(d,e);
                        }
                        CPU::BuildInterfaceQueries(request);
                    } });
                Geom::InterfaceSearchSettings search;
                search.maxGap = connection.settings.maxGap;
                search.coarseBoxesPerRank = connection.settings.coarseBoxesPerRank;
                auto map = Geom::DistributedInterfacePointMap::Match(connection.context, std::move(patch), points, ids,
                                                                     connection.settings.routing == "coarseBoxes" ? Geom::InterfaceRoutingMode::CoarseBoxes : Geom::InterfaceRoutingMode::AllGatherBoundary, search);
                data.control->LocalStage([&]
                                         {
                    R::Array<index> donors(R::RowLayout::Dense(map.Size(), 1));
                    auto write = donors.Write();
                    const auto donorRows = write.View<R::FixedRows<1>>();
                    const auto previousRead = connection.lastDonors.Read();
                    const auto previous = previousRead.View<R::FixedRows<1>>();
                    for(index iPoint=0;iPoint<map.Size();++iPoint)
                    {
                        const auto match=map.Get(iPoint).geometry;
                        donorRows(iPoint,0)=match.donorOriginalId;
                        data.maximumGap=std::max(data.maximumGap,match.distance);
                        if(previous.RowCount()&&previous(iPoint,0)!=donorRows(iPoint,0)) ++data.changedDonors;
                    }
                    connection.lastDonors=std::move(donors);
                    ++data.mapBuilds;
                    ++route.mapGeneration; });
                route.fields.clear();
                route.sampling.reset();
                route.sampling.emplace(std::move(metrics),
                                       std::move(reconstruction),
                                       std::move(map), data.Lane(connection.receiver).compute.Context());
                data.control->LocalStage([&]
                                         {
                    const index fields = connection.donorDescriptor[1];
                    index offset = 2 + fields;
                    route.fields.resize(size_t(fields));
                    for (index iField = 0; iField < fields; ++iField)
                    {
                        auto &field = route.fields[size_t(iField)];
                        const index width =
                            connection.donorDescriptor[size_t(2 + iField)];
                        std::vector<index> logicalComponents(size_t(width), index{});
                        for (index j = 0; j < width; ++j)
                        {
                            const index donorComponent =
                                connection.donorDescriptor[size_t(offset + j)];
                            DNDS_check_throw_info(
                                donorComponent >= 0 &&
                                    donorComponent <
                                        index(connection.receiverComponents.size()),
                                "interface donor field contains an invalid logical component");
                            logicalComponents[size_t(j)] =
                                connection.receiverComponents[size_t(donorComponent)];
                        }
                        field.logicalComponents = R::MetadataBuffer<index>(logicalComponents);
                        offset += width;
                        field.workspace =
                            route.sampling->BuildWorkspace(width);
                        field.values = R::Array<real>(R::RowLayout::Dense(
                            connection.receiverRows.RowCount(), width), route.sampling->Device());
                    }
                    if (receiver)
                    {
                        const auto &packet = *data.staging[connection.receiver];
                        std::vector<R::MetadataBuffer<index>> componentMaps;
                        for (const auto &field : route.fields) componentMaps.push_back(field.logicalComponents);
                        route.packet.emplace(connection.receiverRows, std::move(componentMaps), connection.vectorStarts,
                            packet.QuadratureValues().ComponentNumber(),
                            packet.QuadratureValues().CoupledField().Owned().RowCount(),
                            packet.FaceCenterValues().CoupledField().Owned().RowCount(),
                            data.Lane(connection.receiver).compute.Context());
                    } });
                for (index iField = 0;
                     iField < index(route.fields.size()); ++iField)
                {
                    auto &field = route.fields[size_t(iField)];
                    field.result = route.sampling->BindResult(
                        *field.workspace, field.values,
                        R::TransferMode::Persistent,
                        "cauchy-interface-" +
                            std::to_string(connection.protocol) + "-slot-" +
                            std::to_string(epoch.coupledStage) + "-field-" +
                            std::to_string(iField));
                }
                route.poses = std::make_pair(donorPose, receiverPose);
            }
            route.epoch = epoch;
            data.control->LocalStage([&]
                                     {
                for (index iField = 0;
                     iField < index(route.fields.size()); ++iField)
                {
                    std::optional<CFV::DonorInputs> input;
                    if (publication)
                    {
                        const auto &mean = publication->_means.PhysicalField(iField);
                        switch (publication->_kind)
                        {
                        case InterfaceDonorPublication::Kind::Constant:
                            input.emplace(mean);
                            break;
                        case InterfaceDonorPublication::Kind::Gradient:
                            input.emplace(CFV::DonorInputs::Gradient(
                                mean, publication->_gradients[size_t(iField)]));
                            break;
                        case InterfaceDonorPublication::Kind::Variational:
                            input.emplace(
                                mean,
                                publication->_coefficients->PhysicalField(iField));
                            break;
                        }
                    }
                    route.sampling->EvaluateDonors(
                        data.Lane(connection.receiver).engine, data.Lane(connection.receiver).compute, input,
                        *route.fields[size_t(iField)].workspace);
                } });
            for (auto &field : route.fields)
            {
                auto &lane = data.Lane(connection.receiver);
                // Transfer owns the provider-specific producer readiness bridge.
                // Portable MPI observes CUDA/worker completion on this controller;
                // a stream-native binding would import the same event directly.
                transfers.emplace_back(connection.receiver,
                                       field.result->SubmitAfter(lane.engine, lane.communication, lane.compute.Record()));
            }
        }
        // Post all connections before waiting for any receiver. On current
        // CUDA-aware MPI, EnqueueWait progresses on the permitted host thread;
        // CPU workers and native providers consume the completion event normally.
        for (const auto &transfer : transfers)
            transfer.second.EnqueueWait(data.Lane(transfer.first).compute);
        for (auto &connection : data.connections)
        {
            auto &route = connection.stageRoutes.at(state->epoch.coupledStage);
            auto &receiver = data.discrete[connection.receiver];
            const auto donorPose =
                data.motion[connection.donor].At(state->time).pose;
            const auto receiverPose =
                data.motion[connection.receiver].At(state->time).pose;
            // Reserve receiver writes at submission, before any downstream
            // consumer reserves reads. The completion event includes mapping;
            // publication must never acquire a late immediate write lease.
            if (receiver)
            {
                InterfacePacketRotation rotation;
                // Translation moves query points but leaves vector components unchanged.
                // Identical stage axes also cancel exactly, including two rotating frames.
                const bool differentAxes = donorPose.rotation != receiverPose.rotation;
                rotation.donorActive = differentAxes && !donorPose.rotation.isIdentity(0);
                rotation.receiverActive = differentAxes && !receiverPose.rotation.isIdentity(0);
                for (int d = 0; d < 3; ++d)
                    for (int e = 0; e < 3; ++e)
                    {
                        rotation.donor[size_t(3 * d + e)] = donorPose.rotation(d, e);
                        rotation.receiver[size_t(3 * d + e)] = receiverPose.rotation(e, d);
                    }
                auto &packet = *data.staging[connection.receiver];
                // BuildBoundaryInputs owns canonical coupled receiver rows;
                // donor fields may still be segmented and use their mapping.
                auto &quadrature = packet.QuadratureValues().CoupledField().Owned();
                auto &faces = packet.FaceCenterValues().CoupledField().Owned();
                std::vector<R::Array<real>> inputs;
                for (const auto &field : route.fields)
                {
                    inputs.push_back(field.values);
                }
                // Several connections may fill disjoint rows of one packet.
                // Preserve rows written by earlier connections on this lane.
                auto &lane = data.Lane(connection.receiver);
                DNDS_check_throw_info(route.packet, "interface receiver has no prepared packet operator");
                route.packet->Submit(lane.engine, lane.compute, inputs, quadrature, faces, rotation);
                DNDS_check_throw_info(
                    route.mapGeneration > 0 && !route.fields.empty() &&
                        route.fields.front().workspace &&
                        route.fields.front().workspace->plan,
                    "interface publication lacks a retained matching route");
                state->sources[size_t(connection.receiver)].push_back(
                    {connection.donor, connection.receiver,
                     route.mapGeneration, donorPose, receiverPose,
                     route.fields.front().workspace->plan,
                     state->donors[size_t(connection.donor)]});
            }
        }
        state->pending = data.staging;
        data.checkedValues->compute.WaitFor(data.uncheckedValues->compute.Record());
        state->completion = data.checkedValues->compute.Record();
        return RefreshOperation(std::move(state));
    }
    InterfaceCoupling::RefreshOperation InterfaceCoupling::BeginRefresh(
        const std::vector<InterfaceDomainInput> &domains, real time,
        StateEpoch epoch)
    {
        return BeginRefreshImpl(domains, time, epoch, false);
    }
    InterfaceCoupling::RefreshOperation InterfaceCoupling::BeginValueRefresh(
        const std::vector<InterfaceDomainInput> &domains, real time,
        StateEpoch epoch)
    {
        return BeginRefreshImpl(domains, time, epoch, true);
    }

    InterfaceCoupling::RefreshOperation::~RefreshOperation()
    {
        if (_state && !_state->finished)
        {
            MPI::AbortIfSolverFailure(
                "unfinished Cauchy interface refresh operation");
            DNDS_assert_info(false,
                             "interface refresh must be completed or discarded");
        }
    }
    R::StreamEvent
    InterfaceCoupling::RefreshOperation::CompletionEvent() const
    {
        DNDS_check_throw_info(_state && !_state->finished,
                              "interface refresh operation is not active");
        return _state->completion;
    }
    bool InterfaceCoupling::RefreshOperation::Finished() const
    {
        return !_state || _state->finished;
    }
    std::optional<BoundaryInputs>
    InterfaceCoupling::RefreshOperation::Inputs(index domain) const
    {
        DNDS_check_throw_info(_state && _state->finished && _state->published,
                              "interface refresh has no published packets");
        const auto &packets = *_state->published;
        DNDS_check_throw_info(domain >= 0 && domain < index(packets.size()),
                              "invalid interface receiver domain");
        return packets[size_t(domain)];
    }
    void InterfaceCoupling::RefreshOperation::Complete()
    {
        Finish(true);
    }
    std::optional<BoundaryInputs>
    InterfaceCoupling::RefreshOperation::PendingInputs(index domain) const
    {
        DNDS_check_throw_info(_state && domain >= 0 && domain < index(_state->pending.size()),
                              "invalid pending interface receiver domain");
        return _state->pending[size_t(domain)];
    }
    void InterfaceCoupling::RefreshOperation::PublishReady()
    {
        Finish(false);
    }
    void InterfaceCoupling::RefreshOperation::Finish(bool wait)
    {
        DNDS_check_throw_info(_state && !_state->finished,
                              "interface refresh operation is not active");
        auto &state = *_state;
        if (wait)
            state.completion.WaitHost();
        else
            DNDS_check_throw_info(state.completion.Ready(),
                                  "interface refresh publication preceded its completion event");
        auto &data = *state.data;
        for (index iDomain = 0; iDomain < index(data.discrete.size()); ++iDomain)
            if (data.staging[iDomain])
                data.discrete[iDomain]->PublishBoundaryInputs(
                    *data.staging[iDomain], state.time, state.epoch,
                    std::move(state.sources[size_t(iDomain)]), data.Lane(iDomain).compute);
        data.incoming.swap(data.staging);
        data.useAccepted = false;
        state.published = data.incoming;
        state.finished = true;
        state.operation.Release();
    }
    void InterfaceCoupling::RefreshOperation::Discard()
    {
        DNDS_check_throw_info(_state && !_state->finished,
                              "interface refresh operation is not active");
        _state->completion.WaitHost();
        _state->finished = true;
        _state->operation.Release();
    }
    void InterfaceCoupling::Refresh(
        const std::vector<InterfaceDomainInput> &domains, real time,
        StateEpoch epoch)
    {
        auto operation = BeginRefresh(domains, time, epoch);
        operation.Complete();
    }
}
