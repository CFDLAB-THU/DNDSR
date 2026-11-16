#include "Geom/Refinement/NativeMeshExtraction.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Connectivity/NativePeriodicMatch.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "DNDS/Core/OMP.hpp"
#include <algorithm>
#include <optional>

namespace DNDS::Geom
{
    namespace
    {
        using Array = Runtime::Array<index>;
        Array BuildIndexRows(const std::vector<std::vector<index>> &rows)
        {
            Runtime::RowBuilder builder(rows.size(), sizeof(index));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                std::copy(rows[row].begin(), rows[row].end(), builder.Row<index>(row).data);
            }
            return Array::FromSharedStorage(builder.Commit());
        }
        template <class T>
        Runtime::Array<T> Fetch(const Runtime::RowTransferPlan &route, const Runtime::Array<T> &source, const MPIInfo &mpi,
                                const std::shared_ptr<Runtime::CommunicationContext> &context)
        {
            Runtime::PayloadPlan payload(route, source.Layout(), Runtime::ElementSchema::For<T>());
            std::optional<Runtime::Array<T>> result;
            FieldTransferDetail::LocalStage(mpi, context, [&]
                                            { result.emplace(payload.DestinationLayout()); });
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = context;
            payload.Bind(source, *result, Runtime::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            return std::move(*result);
        }
    }

    NativeMeshDerivedResult NativeMeshExtraction::ExtractBoundary(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                                  NativeMeshBuildOptions options)
    {
        if (!options.communication)
            options.communication = source.PreparationCommunication();
        auto context = options.communication ? options.communication : std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, context->ChecksCollectivePreflight());
        std::optional<MeshReadAccess> sourceRead;
        FieldTransferDetail::LocalStage(mpi, context, [&]
                                        { sourceRead.emplace(source.Snapshot(Runtime::Device::Host()).AcquireRead()); });
        const auto sourceView = sourceRead->HostView();
        if (!options.audit)
            options.audit = source.AuditOptions();
        FieldTransferDetail::LocalStage(mpi, context, [&]
                                        {
            DNDS_check_throw_info(sourceView.Dimension() >= 2 && sourceView.Dimension() <= 3,
                                  "native boundary extraction requires two- or three-dimensional input");
            for (index row = 0; row < source.NumOwned(EntityKind::Bnd); ++row)
                DNDS_check_throw_info(FaceIDIsExternalBC(sourceView.GetBndInfo(row).zone) ||
                                          (sourceView.IsPeriodic() && FaceIDIsPeriodic(sourceView.GetBndInfo(row).zone)),
                                      "native boundary extraction requires external or declared periodic boundary zones"); });
        const auto oldNodes = source.Entities(EntityKind::Node);
        const auto oldBoundaries = source.Entities(EntityKind::Bnd);
        std::vector<std::vector<index>> cones;
        std::vector<std::vector<NodePeriodicBits>> images;
        std::vector<size_t> selectedBoundaries;
        selectedBoundaries.reserve(size_t(oldBoundaries.NumOwned()));
        cones.reserve(size_t(oldBoundaries.NumOwned()));
        images.reserve(size_t(oldBoundaries.NumOwned()));
        FieldTransferDetail::LocalStage(mpi, context, [&]
                                        {
            for (size_t row = 0; row < size_t(oldBoundaries.NumOwned()); ++row)
            {
                if (FaceIDIsPeriodic(sourceView.GetBndInfo(index(row)).zone)) continue;
                selectedBoundaries.push_back(row);
                cones.emplace_back();
                for (const auto node : sourceView.BoundaryNodes(index(row)))
                    cones.back().push_back(oldNodes.LocalToGlobal(node));
                images.emplace_back();
                if (sourceView.IsPeriodic())
                {
                    const auto values = sourceView.BoundaryNodePeriodicBits(index(row));
                    DNDS_check_throw_info(values.size() == cones.back().size(),
                                          "boundary periodic metadata has the wrong row width");
                    images.back().assign(values.begin(), values.end());
                }
            } });
        if (source.Requirements().faces)
        {
            // Published boundary-to-face indices already address the retained
            // owned/ghost face pair. Read its oriented cone directly; fetching
            // the same face from its owner adds a route and payload transfers
            // without supplying any information absent from this snapshot.
            FieldTransferDetail::LocalStage(mpi, context, [&]
                                            {
                for (index iRow = 0; iRow < index(cones.size()); ++iRow)
                {
                    const auto boundary = selectedBoundaries[iRow];
                    const auto mappedFaces = sourceView.BoundaryFaces(index(boundary));
                    DNDS_check_throw_info(mappedFaces.size() <= 1, "boundary extraction requires at most one mapped physical face");
                    if (mappedFaces.size() == 0 || mappedFaces[0] == UnInitIndex) continue;
                    const auto iFace = mappedFaces[0];
                    DNDS_check_throw_info(iFace >= 0 && iFace < sourceView.NumFaceProc(),
                                          "boundary extraction mapped face is absent from the source closure");
                    const auto faceNodes = sourceView.FaceNodes(iFace);
                    std::vector<index> values;
                    values.reserve(faceNodes.size());
                    for (const auto iNode : faceNodes) values.push_back(oldNodes.LocalToGlobal(iNode));
                    std::vector<index> expected = cones[iRow], actual = values;
                    std::sort(expected.begin(), expected.end());
                    std::sort(actual.begin(), actual.end());
                    DNDS_check_throw_info(actual == expected, "mapped boundary face has different nodes");
                    if (sourceView.IsPeriodic())
                    {
                        const auto frames = sourceView.FaceNodePeriodicBits(iFace);
                        DNDS_check_throw_info(frames.size() == index(values.size()),
                                              "face periodic metadata has the wrong row width");
                        NativePeriodicNodes facePattern, boundaryPattern;
                        for (index slot = 0; slot < index(values.size()); ++slot)
                        {
                            facePattern.emplace_back(values[slot], uint8_t(frames[slot]));
                            boundaryPattern.emplace_back(cones[iRow][slot], uint8_t(images[iRow][slot]));
                        }
                        const auto delta = NativePeriodicFrame(std::move(facePattern), std::move(boundaryPattern));
                        DNDS_check_throw_info(delta, "mapped boundary face images disagree with physical boundary");
                        for (index slot = 0; slot < index(values.size()); ++slot)
                            images[iRow][slot] = NodePeriodicBits{uint8_t(uint8_t(frames[slot]) ^ uint8_t(*delta))};
                    }
                    cones[iRow] = std::move(values);
                } });
        }
        std::vector<index> requested;
        std::vector<Runtime::RowRequest> requests;
        FieldTransferDetail::LocalStage(mpi, context, [&]
                                        {
            for (const auto &row : cones) requested.insert(requested.end(), row.begin(), row.end());
            std::sort(requested.begin(), requested.end());
            requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
            for (size_t row = 0; row < requested.size(); ++row)
                requests.push_back({oldNodes.Owner(requested[row]), oldNodes.OwnerLocalRow(requested[row]), index(row)}); });
        Runtime::RowTransferPlan discovery(context, size_t(oldNodes.NumOwned()), requested.size(), requests);
        std::vector<index> selected;
        std::vector<std::vector<index>> send;
        FieldTransferDetail::LocalStage(mpi, context, [&]
                                        {
            for (const auto &peer : discovery.SendRows()) selected.insert(selected.end(), peer.begin(), peer.end());
            std::sort(selected.begin(), selected.end());
            selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
            send = discovery.SendRows();
            for (auto &peer : send)
                for (auto &row : peer)
                    row = index(std::lower_bound(selected.begin(), selected.end(), row) - selected.begin()); });
        const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, index(selected.size()));
        const auto cells = nodes.CreateSibling(EntityKind::Cell, index(selectedBoundaries.size()));
        const auto boundaries = nodes.CreateSibling(EntityKind::Bnd, 0);
        std::optional<Array> labels;
        std::optional<Runtime::Array<real>> coordinates;
        NativeMeshDerivedResult result;
        FieldTransferDetail::LocalStage(mpi, context, [&]
                                        {
            labels.emplace(Runtime::RowLayout::Dense(selected.size(), 1));
            coordinates.emplace(Runtime::RowLayout::Dense(selected.size(), 3));
            auto writeLabels = labels->Write();
            auto writeCoords = coordinates->Write();
            const auto originalIds = source.ReadOriginalEntityIds(EntityKind::Node);
            Runtime::Array<index> parentNodes(Runtime::RowLayout::Dense(selected.size(), 1));
            auto writeParentNodes = parentNodes.Write();
            OMP::ForSetupRows(0, index(selected.size()), [&](index iRow)
            {
                const auto row = size_t(iRow);
                writeLabels.View()(row, 0) = nodes.LocalToGlobal(index(row));
                for (size_t component = 0; component < 3; ++component)
                    writeCoords.View()(row, component) = sourceView.Coord(index(selected[row]))[component];
                writeParentNodes.View()(row, 0) = originalIds->View()(selected[row], 0);
            });
            result.provenance.sourceNamespace = source.MeshNamespace();
            result.provenance.sourceKinds = {{EntityKind::Node, EntityKind::Node}, {EntityKind::Cell, EntityKind::Bnd}};
            result.provenance.sourceIds.emplace(EntityKind::Node, std::move(parentNodes));
            const auto boundaryIds = source.ReadOriginalEntityIds(EntityKind::Bnd);
            Runtime::Array<index> parentBoundaries(Runtime::RowLayout::Dense(selectedBoundaries.size(), 1));
            auto writeParentBoundaries = parentBoundaries.Write();
            OMP::ForSetupRows(0, index(selectedBoundaries.size()), [&](index iRow)
                              { writeParentBoundaries.View()(iRow, 0) = boundaryIds->View()(selectedBoundaries[size_t(iRow)], 0); });
            result.provenance.sourceIds.emplace(EntityKind::Cell, std::move(parentBoundaries)); });
        // Discovery names original owner rows. Remap only its send rows into
        // compact owned order; the per-peer correspondence remains exact.
        auto route = Runtime::RowTransferPlan::Known(context, selected.size(), requested.size(), std::move(send), discovery.ReceiveRows());
        auto newGlobals = Fetch(route, *labels, mpi, context);
        std::optional<NativeMeshInput> input;
        FieldTransferDetail::LocalStage(mpi, context, [&]
                                        {
            auto read = newGlobals.Read();
            for (auto &row : cones)
                for (auto &global : row)
                {
                    const auto found = std::lower_bound(requested.begin(), requested.end(), global);
                    DNDS_check_throw_info(found != requested.end() && *found == global, "native extraction node mapping is incomplete");
                    global = read.View()(size_t(found - requested.begin()), 0);
                }
            Runtime::Array<ElemInfo> info(Runtime::RowLayout::Dense(selectedBoundaries.size(), 1));
            { auto write = info.Write(); for (size_t row = 0; row < selectedBoundaries.size(); ++row) write.View()(row, 0) = sourceView.GetBndInfo(index(selectedBoundaries[row])); }
            input.emplace(NativeMeshInput{mpi, sourceView.Dimension() - 1, cells, nodes, boundaries, BuildIndexRows(cones), BuildIndexRows({}),
                                         std::move(*coordinates), std::move(info),
                                         Runtime::Array<ElemInfo>(Runtime::RowLayout::Dense(0, 1))});
            input->provenance = result.provenance;
            if (sourceView.IsPeriodic())
            {
                Runtime::RowBuilder builder(images.size(), sizeof(NodePeriodicBits));
                for (size_t row = 0; row < images.size(); ++row)
                {
                    builder.ResizeRow(row, images[row].size());
                    std::copy(images[row].begin(), images[row].end(), builder.Row<NodePeriodicBits>(row).data);
                }
                input->periodic = true;
                input->periodicInfo = sourceView.Periodic();
                input->cellNodesPbi = Runtime::Array<NodePeriodicBits>::FromSharedStorage(builder.Commit());
                input->boundaryNodesPbi = Runtime::Array<NodePeriodicBits>(Runtime::RowLayout::Ragged({}));
            } });
        options.communication = context;
        result.state = ConstructNativeMesh(std::move(*input), options);
        return result;
    }
}
