#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Connectivity/NativePeriodicMatch.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"

namespace DNDS::Geom
{
    namespace
    {
        /// Agree on recoverable local errors before the next MPI stage. This
        /// includes local relation construction, which validates retained rows.
        template <class F>
        auto LocalStage(const EntitySpace &space, const char *stage, F &&action)
        {
            using Result = decltype(action());
            std::optional<Result> result;
            const auto message = std::string(stage) + ": native mesh setup failed on a rank";
            space.LocalStage([&]
                             { result.emplace(action()); }, message.c_str());
            return std::move(*result);
        }

        template <class T>
        void CheckRows(const Runtime::Array<T> &values, index count, std::optional<size_t> width = {})
        {
            DNDS_check_throw_info(values.GetDevice() == Runtime::Device::Host() && values.RowCount() == size_t(count),
                                  "native input host placement or owned row count mismatch");
            auto read = values.Read();
            if (width)
                for (size_t row = 0; row < read.View().RowCount(); ++row)
                    DNDS_check_throw_info(read.View().RowSize(row) == *width, "native input logical row width mismatch");
        }

        /// Uniform semantic components use a table view at publication. A
        /// metadata-only reinterpretation preserves the original value buffer
        /// and access controller, including aliases held by Python callers.
        template <class T>
        Runtime::Array<T> UniformTable(Runtime::Array<T> values, index width)
        {
            const auto layout = values.Layout();
            if (!layout.IsVariable())
            {
                DNDS_check_throw_info(layout.Width() == width, "native uniform input table width mismatch");
                return values;
            }
            for (index row = 0; row < layout.RowCount(); ++row)
            {
                DNDS_check_throw_info(layout.RowSize(row) == width, "native uniform input logical width mismatch");
                if (layout.IsRagged())
                    DNDS_check_throw_info(layout.Offset(row) == CheckedSize::Multiply(row, width),
                                          "native uniform CSR input is not tightly contiguous");
            }
            if (layout.IsRagged())
                DNDS_check_throw_info(layout.Elements() == CheckedSize::Multiply(layout.RowCount(), width),
                                      "native uniform CSR input has an invalid terminal offset");
            const auto stride = layout.IsPadded() && layout.RowCount() ? layout.Stride() : width;
            return Runtime::Array<T>::FromStorage(Runtime::RowLayout::Dense(layout.RowCount(), width, stride), values.Storage());
        }

        void CheckCone(const Runtime::Array<index> &cone, const Runtime::Array<ElemInfo> &info,
                       const EntitySpace &source, const EntitySpace &nodes, int dimension, bool boundary,
                       const Runtime::Array<NodePeriodicBits> *periodic = nullptr)
        {
            CheckRows(cone, source.NumOwned());
            CheckRows(info, source.NumOwned(), 1);
            auto entries = cone.Read();
            auto elements = info.Read();
            std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> bits;
            if (periodic)
            {
                CheckRows(*periodic, source.NumOwned());
                bits.emplace(periodic->Read());
            }
            std::array<std::pair<index, uint8_t>, Elem::CellNumNodeMax> seen{};
            for (size_t row = 0; row < entries.View().RowCount(); ++row)
            {
                const auto elementInfo = elements.View()(row, 0);
                DNDS_check_throw_info(elementInfo.type > Elem::UnknownElem && elementInfo.type < Elem::ElemType_NUM,
                                      "native input contains an unsupported element type");
                const Elem::Element element{elementInfo.getElemType()};
                DNDS_check_throw_info(element.GetDim() == dimension &&
                                          entries.View().RowSize(row) == size_t(element.GetNumNodes()),
                                      "native input cone differs from its element type/dimension");
                DNDS_check_throw_info(!boundary || FaceIDIsExternalBC(elementInfo.zone) || (periodic && FaceIDIsPeriodic(elementInfo.zone)),
                                      "native boundary requires an external or periodic zone");
                DNDS_check_throw_info(!bits || bits->View().RowSize(row) == entries.View().RowSize(row), "periodic cone slots differ from node slots");
                const auto count = entries.View().RowSize(row);
                DNDS_check_throw_info(count <= seen.size(), "native input cone exceeds the supported element size");
                for (size_t slot = 0; slot < count; ++slot)
                {
                    const auto node = entries.View()(row, slot);
                    const auto frame = bits ? uint8_t(bits->View()(row, slot)) : uint8_t(0);
                    DNDS_check_throw_info(node >= 0 && node < nodes.NumGlobal() && frame < 8,
                                          "native input node/image is invalid in an ordered cone");
                    seen[slot] = {node, frame};
                }
                std::sort(seen.begin(), seen.begin() + count);
                DNDS_check_throw_info(std::adjacent_find(seen.begin(), seen.begin() + count) == seen.begin() + count,
                                      "native input node/image is repeated in an ordered cone");
            }
        }

        Runtime::Array<index> IndexRows(const std::vector<std::vector<index>> &rows, std::optional<size_t> width = {})
        {
            return Runtime::BuildArrayRows<index>(rows.size(), [&](size_t row)
                                                  { return rows[row].size(); }, [&](size_t row, Runtime::RowSpan<index> output)
                                                  {
                if (!rows[row].empty())
                    std::copy(rows[row].begin(), rows[row].end(), output.data()); }, width);
        }

        std::pair<MeshAdjacency, MeshAdjacency> PeriodicBoundaryRelations(
            const NativeMeshInput &input, const MeshAdjacencyRead &candidates, const AdjacencyDescriptor &exactKind)
        {
            const auto mpi = input.mpi;
            std::vector<Runtime::RowRequest> requests;
            std::vector<index> candidateCells;
            std::vector<size_t> starts;
            std::optional<Runtime::Array<index>> types, bits;
            LocalStage(input.cells, "periodic boundary candidate preparation", [&]
                       {
                starts.push_back(0);
                for (index boundary = 0; boundary < candidates.NumOwned(); ++boundary)
                {
                    for (auto cell : candidates.GlobalRow(boundary))
                    {
                        requests.push_back({input.cells.Owner(cell), input.cells.OwnerLocalRow(cell), index(candidateCells.size())});
                        candidateCells.push_back(cell);
                    }
                    starts.push_back(candidateCells.size());
                }
                std::vector<std::vector<index>> cellTypes(size_t(input.cells.NumOwned())), cellBits(cellTypes.size());
                auto info = input.cellInfo.Read();
                auto pbi = input.cellNodesPbi->Read();
                for (size_t row = 0; row < cellTypes.size(); ++row)
                {
                    cellTypes[row] = {info.View()(row, 0).type};
                    for (auto value : pbi.View()[row]) cellBits[row].push_back(uint8_t(value));
                }
                types = IndexRows(cellTypes, 1);
                bits = IndexRows(cellBits);
                return true; });
            auto context = input.cells.Communication() ? input.cells.Communication() : std::make_shared<Runtime::CommunicationContext>(mpi.comm);
            Runtime::RowTransferPlan route(context, size_t(input.cells.NumOwned()), candidateCells.size(), requests);
            auto fetch = [&](const Runtime::Array<index> &values)
            {
                Runtime::PayloadPlan payload(route, values.Layout(), Runtime::ElementSchema::Arithmetic<index>());
                auto output = LocalStage(input.cells, "periodic boundary payload allocation", [&]
                                         { return Runtime::Array<index>(payload.DestinationLayout()); });
                Runtime::TransferOptions options;
                options.mode = Runtime::TransferMode::Transient;
                options.context = context;
                payload.Bind(values, output, Runtime::TransferTransport::Direct, options)
                    .Begin()
                    .Wait();
                return output;
            };
            auto cones = fetch(input.cellNodes), metadata = fetch(*types), frames = fetch(*bits);
            return LocalStage(input.cells, "periodic physical boundary occurrences", [&]
                              {
                auto nodes = cones.Read(), info = metadata.Read(), pbi = frames.Read();
                auto boundaryNodes = input.boundaryNodes.Read();
                auto boundaryFrames = input.boundaryNodesPbi->Read();
                auto boundaryInfo = input.boundaryInfo.Read();
                std::vector<std::vector<index>> exact(size_t(input.boundaries.NumOwned())), physical(exact.size());
                for (size_t boundary = 0; boundary < exact.size(); ++boundary)
                {
                    NativePeriodicNodes expected;
                    for (size_t slot = 0; slot < boundaryNodes.View().RowSize(boundary); ++slot)
                        expected.emplace_back(boundaryNodes.View()(boundary, slot), uint8_t(boundaryFrames.View()(boundary, slot)));
                    struct Occurrence { index cell; t_index slot; uint8_t frame; };
                    std::vector<Occurrence> matches;
                    for (size_t candidate = starts[boundary]; candidate < starts[boundary + 1]; ++candidate)
                    {
                        Elem::Element element{static_cast<Elem::ElemType>(info.View()(candidate, 0))};
                        const auto values = nodes.View()[candidate], labels = pbi.View()[candidate];
                        std::vector<index> parent(values.begin(), values.end()), parentFrames(labels.begin(), labels.end());
                        for (t_index face = 0; face < element.GetNumFaces(); ++face)
                        {
                            const auto sub = element.ObtainFace(face);
                            if (sub.type != boundaryInfo.View()(boundary, 0).type) continue;
                            std::vector<index> subNodes(size_t(sub.GetNumNodes())), subFrames(subNodes.size());
                            element.ExtractFaceNodes(face, parent, subNodes);
                            element.ExtractFaceNodes(face, parentFrames, subFrames);
                            NativePeriodicNodes actual;
                            for (size_t slot = 0; slot < subNodes.size(); ++slot) actual.emplace_back(subNodes[slot], uint8_t(subFrames[slot]));
                            auto frame = NativePeriodicFrame(std::move(actual), expected);
                            if (frame) matches.push_back({candidateCells[candidate], face, *frame});
                        }
                    }
                    auto side = std::find_if(matches.begin(), matches.end(), [](const auto &entry) { return entry.frame == 0; });
                    DNDS_check_throw_info(side != matches.end() && std::count_if(matches.begin(), matches.end(), [](const auto &entry) { return entry.frame == 0; }) == 1,
                                          "boundary must match exactly one full element face in its physical frame");
                    exact[boundary] = {side->cell};
                    physical[boundary] = {side->cell, UnInitIndex};
                    const auto zone = boundaryInfo.View()(boundary, 0).zone;
                    if (FaceIDIsPeriodic(zone))
                    {
                        const uint8_t mask = uint8_t(1U << ((-int(zone) - 1) % 3));
                        DNDS_check_throw_info(matches.size() == 2, "periodic boundary must have exactly two physical face occurrences");
                        const auto &other = matches[side == matches.begin() ? 1 : 0];
                        DNDS_check_throw_info(other.frame == mask, "periodic partner frame differs from its boundary group");
                        physical[boundary][1] = other.cell;
                    }
                }
                return std::make_pair(MeshAdjacency(exactKind, input.boundaries, input.cells, IndexRows(exact)),
                                      MeshAdjacency(Adj::Bnd2CellCoupled, input.boundaries, input.cells, IndexRows(physical, 2))); });
        }

        void ValidateBoundaryFaces(const NativeMeshInput &input, const MeshAdjacencyRead &containment)
        {
            if (!input.boundaries.NumGlobal())
                return;
            const auto mpi = input.mpi;
            std::vector<Runtime::RowRequest> requests;
            Runtime::Array<index> types(Runtime::RowLayout::Dense(size_t(input.cells.NumOwned()), 1));
            LocalStage(input.cells, "boundary face request preparation", [&]
                       {
                auto metadata = input.cellInfo.Read();
                auto write = types.Write();
                for (index cell = 0; cell < input.cells.NumOwned(); ++cell)
                    write.View()(size_t(cell), 0) = metadata.View()(size_t(cell), 0).type;
                for (index boundary = 0; boundary < containment.NumOwned(); ++boundary)
                {
                    DNDS_check_throw_info(containment.RowSize(boundary) == 1,
                                          "external boundary must have exactly one all-node containing cell");
                    const auto cell = containment.ReadEntry(boundary, 0).global;
                    requests.push_back({input.cells.Owner(cell), input.cells.OwnerLocalRow(cell), index(boundary)});
                }
                return true; });
            // Containment may name a remote cell that is not in the initial
            // cell closure. Fetch just those candidate rows in global node IDs;
            // no complete remote mesh or additional entity map is needed.
            auto context = input.cells.Communication() ? input.cells.Communication() : std::make_shared<Runtime::CommunicationContext>(mpi.comm);
            Runtime::RowTransferPlan route(context, size_t(input.cells.NumOwned()), size_t(input.boundaries.NumOwned()),
                                           requests, {input.cells.Stamp(), input.boundaries.Stamp()});
            auto fetch = [&](const Runtime::Array<index> &source)
            {
                Runtime::PayloadPlan payload(route, source.Layout(), Runtime::ElementSchema::Arithmetic<index>());
                Runtime::Array<index> received(payload.DestinationLayout());
                Runtime::TransferOptions options;
                options.mode = Runtime::TransferMode::Transient;
                options.context = context;
                payload.Bind(source, received, Runtime::TransferTransport::Direct, options)
                    .Begin()
                    .Wait();
                return received;
            };
            auto candidateNodes = fetch(input.cellNodes);
            auto candidateTypes = fetch(types);
            LocalStage(input.cells, "exact boundary face validation", [&]
                       {
                auto nodes = candidateNodes.Read(), elementTypes = candidateTypes.Read();
                auto boundaryNodes = input.boundaryNodes.Read();
                auto boundaryInfo = input.boundaryInfo.Read();
                for (size_t boundary = 0; boundary < size_t(input.boundaries.NumOwned()); ++boundary)
                {
                    Elem::Element cell{static_cast<Elem::ElemType>(elementTypes.View()(boundary, 0))};
                    const auto source = nodes.View()[boundary];
                    std::vector<index> parent(source.begin(), source.end());
                    const auto target = boundaryNodes.View()[boundary];
                    std::vector<index> expected(target.begin(), target.end());
                    std::sort(expected.begin(), expected.end());
                    bool matched = false;
                    for (t_index face = 0; face < cell.GetNumFaces(); ++face)
                    {
                        const auto element = cell.ObtainFace(face);
                        if (element.type != boundaryInfo.View()(boundary, 0).type)
                            continue;
                        std::vector<index> actual(size_t(element.GetNumNodes()));
                        cell.ExtractFaceNodes(face, parent, actual);
                        std::sort(actual.begin(), actual.end());
                        matched = matched || actual == expected;
                    }
                    DNDS_check_throw_info(matched, "external boundary nodes do not form a face of their containing cell");
                }
                return true; });
        }

        std::vector<index> ReferencedGhostNodes(const NativeMeshInput &input)
        {
            std::vector<index> ghosts;
            for (const auto *array : {&input.cellNodes, &input.boundaryNodes})
            {
                auto read = array->Read();
                ghosts.reserve(ghosts.size() + read.View().RowCount() * 2);
                for (size_t row = 0; row < read.View().RowCount(); ++row)
                    for (auto global : read.View()[row])
                        if (input.nodes.Owner(global) != input.mpi.rank)
                            ghosts.push_back(global);
            }
            std::sort(ghosts.begin(), ghosts.end());
            ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
            return ghosts;
        }

        Runtime::Array<index> PadBoundaryCells(const MeshAdjacencyRead &relation,
                                               const Runtime::Array<ElemInfo> *boundaryInfo = nullptr)
        {
            Runtime::Array<index> rows(Runtime::RowLayout::Dense(size_t(relation.NumOwned()), 2));
            std::optional<Runtime::Array<ElemInfo>::ReadLease> info;
            if (boundaryInfo)
                info.emplace(boundaryInfo->Read());
            auto write = rows.Write();
            for (index boundary = 0; boundary < relation.NumOwned(); ++boundary)
            {
                DNDS_check_throw_info(relation.RowSize(boundary) == 1,
                                      "external boundary must have exactly one containing cell");
                write.View()(size_t(boundary), 0) = relation.ConvertedIndex(boundary, 0, Adj_PointToGlobal);
                // Legacy boundary incidence is own/self for a periodic side;
                // the physical partner belongs to its separate coupled key.
                write.View()(size_t(boundary), 1) = info && FaceIDIsPeriodic(info->View()(size_t(boundary), 0).zone)
                                                        ? write.View()(size_t(boundary), 0)
                                                        : UnInitIndex;
            }
            return rows;
        }
    }

    NativeOwnedMeshData NativeMeshBuilder::Build(NativeMeshInput input, NativeMeshBuildOptions options)
    {
        const auto mpi = input.mpi;
        const auto audit = options.audit.value_or(MeshAuditOptions{});
        const auto communication = options.communication ? options.communication : input.cells.Communication();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
        if (communication)
            communication->LocalStage([&]
                                      {
                input.cells = input.cells.WithCommunication(communication);
                input.nodes = input.nodes.WithCommunication(communication);
                input.boundaries = input.boundaries.WithCommunication(communication); });
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native mesh builder requires a communicator");
        LocalStage(input.cells, "input validation", [&]
                   {
            int actualRank = -1, actualSize = 0;
            MPI_Comm_rank(mpi.comm, &actualRank);
            MPI_Comm_size(mpi.comm, &actualSize);
            DNDS_check_throw_info(mpi.rank == actualRank && mpi.size == actualSize &&
                                      (input.dimension >= 1 && input.dimension <= 3) && options.cellLayers >= 1,
                                  "invalid native mesh MPI/dimension/layer options");
            DNDS_check_throw_info(input.periodic ? bool(input.cellNodesPbi && input.boundaryNodesPbi) : !input.cellNodesPbi && !input.boundaryNodesPbi,
                                  "native periodic flag and ordered cone companions disagree");
            for (const auto *space : {&input.cells, &input.nodes, &input.boundaries})
            {
                space->Validate();
                DNDS_check_throw_info(space->MeshIdentity() == input.nodes.MeshIdentity() && !space->NumGhost(),
                                      "native input spaces must be sibling owned-only families");
            }
            DNDS_check_throw_info(input.cells.Kind() == EntityKind::Cell && input.nodes.Kind() == EntityKind::Node &&
                                      input.boundaries.Kind() == EntityKind::Bnd, "native input entity kinds are invalid");
            DNDS_check_throw_info(input.dimension != 1 || input.boundaries.NumGlobal() == 0,
                                  "native one-dimensional primary meshes require an empty boundary family; point elements are unsupported");
            CheckRows(input.coordinates, input.nodes.NumOwned(), 3);
            CheckCone(input.cellNodes, input.cellInfo, input.cells, input.nodes, input.dimension, false, input.cellNodesPbi ? &*input.cellNodesPbi : nullptr);
            CheckCone(input.boundaryNodes, input.boundaryInfo, input.boundaries, input.nodes, input.dimension - 1, true, input.boundaryNodesPbi ? &*input.boundaryNodesPbi : nullptr);
            input.coordinates = UniformTable(std::move(input.coordinates), 3);
            input.cellInfo = UniformTable(std::move(input.cellInfo), 1);
            input.boundaryInfo = UniformTable(std::move(input.boundaryInfo), 1);
            // Register validates each space's communicator against the caller
            // before WithGhosts enters that descriptor's own collectives.
            AdjacencyRegistry validate(mpi);
            validate.Register(MeshAdjacencyRead(Adj::Cell2Node, input.cells, input.nodes, Adj_PointToGlobal, input.cellNodes));
            validate.Register(MeshAdjacencyRead(Adj::Bnd2Node, input.boundaries, input.nodes, Adj_PointToGlobal, input.boundaryNodes));
            for (const auto &[kind, ids] : input.ids)
            {
                DNDS_check_throw_info(kind == EntityKind::Cell || kind == EntityKind::Node || kind == EntityKind::Bnd,
                                      "native input IDs name an unsupported family");
                const auto &space = kind == EntityKind::Cell ? input.cells : kind == EntityKind::Node ? input.nodes : input.boundaries;
                CheckRows(ids, space.NumOwned(), 1);
                auto read = ids.Read();
                std::vector<index> listed;
                if (audit.persistentIdentity)
                    listed.reserve(ids.RowCount());
                for (size_t row = 0; row < ids.RowCount(); ++row)
                {
                    const auto id = read.View()(row, 0);
                    DNDS_check_throw_info(id >= 0 && id != UnInitIndex, "native input ID is invalid");
                    if (audit.persistentIdentity)
                        listed.push_back(id);
                }
                if (audit.persistentIdentity)
                {
                    std::sort(listed.begin(), listed.end());
                    DNDS_check_throw_info(std::adjacent_find(listed.begin(), listed.end()) == listed.end(),
                                          "native input IDs are locally duplicated");
                }
            }
            return true; });
        int idMask = 0;
        for (const auto &[kind, unused] : input.ids)
            idMask |= 1 << int(kind);
        int settings[]{input.dimension, options.cellLayers, int(options.completeNodeStar), idMask, int(input.periodic),
                       int(audit.persistentIdentity), int(audit.closure)};
        int minimum[7], maximum[7];
        MPI_Allreduce(settings, minimum, 7, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(settings, maximum, 7, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::equal(minimum, minimum + 7, maximum), "native mesh build options differ between ranks");
        if (input.periodic)
        {
            std::array<real, 60> values{}, root{};
            LocalStage(input.cells, "periodic geometry validation", [&]
                       {
                size_t offset = 0;
                for (size_t frame = 0; frame < 4; ++frame)
                {
                    const auto rotation = input.periodicInfo.rotation[frame].map();
                    DNDS_check_throw_info(rotation.allFinite() && (rotation.transpose() * rotation - tGPoint::Identity()).norm() < 1e-10 &&
                                              std::abs(rotation.determinant() - 1) < 1e-10 && input.periodicInfo.translation[frame].map().allFinite() &&
                                              input.periodicInfo.rotationCenter[frame].map().allFinite(), "invalid native rigid periodic frame");
                    for (int i = 0; i < 9; ++i) values[offset++] = rotation.data()[i];
                    for (int i = 0; i < 3; ++i) values[offset++] = input.periodicInfo.translation[frame].map()[i];
                    for (int i = 0; i < 3; ++i) values[offset++] = input.periodicInfo.rotationCenter[frame].map()[i];
                }
                return true; });
            root = values;
            MPI_Bcast(root.data(), int(root.size()), DNDS_MPI_REAL, 0, mpi.comm);
            LocalStage(input.cells, "periodic geometry agreement", [&]
                       { DNDS_check_throw_info(values == root, "periodic geometry differs between ranks"); return true; });
        }

        auto referenced = LocalStage(input.cells, "initial node closure", [&]
                                     { return ReferencedGhostNodes(input); });
        const auto nodes = input.nodes.WithGhosts(std::move(referenced));
        auto cellNodes = LocalStage(input.cells, "cell cone", [&]
                                    { return MeshAdjacency(Adj::Cell2Node, input.cells, nodes, input.cellNodes); });
        auto boundaryNodes = LocalStage(input.cells, "boundary cone", [&]
                                        { return MeshAdjacency(Adj::Bnd2Node, input.boundaries, nodes, input.boundaryNodes); });
        auto nodeCells = AdjacencyRegistry::Inverse(cellNodes.Read(), mpi);
        auto nodeBoundaries = AdjacencyRegistry::Inverse(boundaryNodes.Read(), mpi);
        int anyNodeGhost = nodes.NumGhost() > 0;
        MPI_Allreduce(MPI_IN_PLACE, &anyNodeGhost, 1, MPI_INT, MPI_MAX, mpi.comm);
        if (anyNodeGhost)
        {
            EntityGhostMapping initialNodeRoute(nodes);
            const auto schema = Runtime::ElementSchema::Arithmetic<index>();
            auto pendingCells = initialNodeRoute.BeginRows(nodes, nodeCells.Owned(), schema);
            auto pendingBoundaries = initialNodeRoute.BeginRows(nodes, nodeBoundaries.Owned(), schema);
            pendingCells.operation.Wait();
            pendingBoundaries.operation.Wait();
            nodeCells = MeshAdjacency::WithGhostValues(nodeCells.Kind(), nodes, input.cells, nodeCells.Owned(),
                                                       std::move(pendingCells.ghost));
            nodeBoundaries = MeshAdjacency::WithGhostValues(nodeBoundaries.Kind(), nodes, input.boundaries,
                                                            nodeBoundaries.Owned(), std::move(pendingBoundaries.ghost));
        }
        auto cellCells = LocalStage(input.cells, "cell neighbor composition", [&]
                                    { return AdjacencyRegistry::Compose(cellNodes.Read(), nodeCells.Read(), true); });
        auto containment = LocalStage(input.cells, "boundary cell containment", [&]
                                      { return AdjacencyRegistry::ComposeAll(boundaryNodes.Read(), nodeCells.Read()); });
        const auto boundaryCellKind = containment.Kind();
        std::optional<MeshAdjacency> periodicExact, physicalBoundary;
        if (input.periodic)
        {
            const auto exactKind = AdjacencyDescriptor::Composition(boundaryNodes.Kind(), nodeCells.Kind(), RelationSelection::Exact, false,
                                                                    "geom.boundary.physical-frame-face", 1, RelationPeriodicPolicy::FrameMatched);
            auto matched = PeriodicBoundaryRelations(input, containment.Read(), exactKind);
            periodicExact.emplace(std::move(matched.first));
            physicalBoundary.emplace(std::move(matched.second));
        }
        else
        {
            LocalStage(input.cells, "external boundary incidence", [&]
                       {
            auto read = containment.Read();
            for (index row = 0; row < read.NumOwned(); ++row)
                DNDS_check_throw_info(read.RowSize(row) == 1, "external boundary must have exactly one all-node containing cell");
            return true; });
            ValidateBoundaryFaces(input, containment.Read());
        }
        auto data = LocalStage(input.cells, "owned topology output", [&]
                               {
            NativeOwnedMeshData result;
            result.communication = communication;
            result.audit = audit;
            result.identity = std::move(input.identity);
            result.provenance = std::move(input.provenance);
            result.dimension = input.dimension;
            result.periodic = input.periodic;
            result.periodicInfo = input.periodicInfo;
            auto relation = [&](MeshRelation slot, const MeshAdjacency &relation)
            {
                result.namedRelations.emplace(relation.Kind(), relation.Owned());
                result.relations[size_t(slot)] = relation.Owned();
            };
            relation(MeshRelation::CellNodes, cellNodes);
            relation(MeshRelation::BoundaryNodes, boundaryNodes);
            relation(MeshRelation::CellCells, cellCells);
            result.cellCellsComplete = true;
            // The native containment relation stays sorted/set-valued. Only
            // this explicit facade slot adds the external second-cell sentinel.
            result.relations[size_t(MeshRelation::BoundaryCells)] = physicalBoundary
                ? PadBoundaryCells(periodicExact->Read(), &input.boundaryInfo) : PadBoundaryCells(containment.Read());
            result.namedRelations.emplace(boundaryCellKind, containment.Owned());
            if (periodicExact)
            {
                result.namedRelations.emplace(periodicExact->Kind(), periodicExact->Owned());
                result.namedRelations.emplace(physicalBoundary->Kind(), physicalBoundary->Owned());
                result.namedRelations.emplace(Adj::Bnd2Cell, *result.relations[size_t(MeshRelation::BoundaryCells)]);
            }
            relation(MeshRelation::NodeCells, nodeCells);
            relation(MeshRelation::NodeBoundaries, nodeBoundaries);
            return result; });
        data.coords = input.coordinates;
        if (input.periodic)
        {
            data.cellNodesPbi = input.cellNodesPbi;
            data.bndNodesPbi = input.boundaryNodesPbi;
        }
        data.cellInfo = input.cellInfo;
        data.bndInfo = input.boundaryInfo;
        data.spaces = {{EntityKind::Cell, input.cells}, {EntityKind::Node, input.nodes}, {EntityKind::Bnd, input.boundaries}};
        std::map<EntityKind, index> maxima;
        for (auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd})
        {
            const auto &space = data.spaces.at(kind);
            auto owned = LocalStage(input.cells, "persistent ID initialization", [&]
                                    {
                const auto supplied = input.ids.find(kind);
                if (supplied != input.ids.end()) return supplied->second;
                Runtime::Array<index> generated(Runtime::RowLayout::Dense(size_t(space.NumOwned()), 1));
                auto write = generated.Write();
                for (index row = 0; row < space.NumOwned(); ++row)
                    write.View()(size_t(row), 0) = space.LocalToGlobal(row);
                return generated; });
            LocalStage(input.cells, "owned persistent IDs", [&]
                       {
                index maximum = -1;
                auto read = owned.Read();
                for (index row = 0; row < space.NumOwned(); ++row)
                    maximum = std::max(maximum, read.View()(size_t(row), 0));
                maxima.emplace(kind, maximum);
                data.ids.emplace(kind, std::move(owned));
                return true; });
        }
        data.identity = PrepareMeshIdentity(mpi, std::move(data.identity), maxima);
        data.requirements.inverse = true;
        return data;
    }
}
