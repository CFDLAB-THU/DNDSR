#include "Geom/Refinement/NativeMeshRefinement.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "DNDS/Core/OMP.hpp"
#include "DNDS/Core/CheckedSize.hpp"

namespace DNDS::Geom
{
    namespace
    {
        Runtime::Array<index> IndexRows(const std::vector<std::vector<index>> &rows)
        {
            // Exclusive CSR shape mutation/commit remains a controller phase;
            // the parent-child numerical fill above has already joined.
            Runtime::RowBuilder builder(rows.size(), sizeof(index));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                auto output = builder.Row<index>(row);
                for (size_t slot = 0; slot < rows[row].size(); ++slot)
                    output[slot] = rows[row][slot];
            }
            return Runtime::Array<index>::FromSharedStorage(builder.Commit());
        }
        template <class T>
        Runtime::Array<T> Column(const std::vector<T> &values)
        {
            Runtime::Array<T> result(Runtime::RowLayout::Dense(values.size(), 1));
            auto write = result.Write();
            OMP::ForSetupRows(0, index(values.size()), [&](index iRow)
                              { write.View()(iRow, 0) = values[size_t(iRow)]; });
            return result;
        }
    }

    NativeOwnedMeshDerivedResult NativeMeshRefinement::BisectOwned(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                                   NativeMeshBuildOptions options)
    {
        NativeOwnedMeshDerivedResult result;
        auto owned = source;
        if (!options.audit)
            options.audit = owned.audit;
        if (!options.communication)
            options.communication = owned.communication;
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !options.communication || options.communication->ChecksCollectivePreflight());
        if (options.communication)
        {
            owned.communication = options.communication;
            owned.LocalStage(mpi, [&]
                             {
                for (auto &[kind, space] : owned.spaces)
                    space = space.WithCommunication(options.communication); });
        }

        AdjacencyRegistry closure(mpi);
        owned.LocalStage(mpi, [&]
                         {
            for (const auto &[kind, space] : owned.spaces)
                DNDS_check_throw_info(space.Valid() && space.NumGhost() == 0, "owned bisection requires owned-only entity spaces");
            for (const auto &[key, slot] : std::initializer_list<std::pair<AdjacencyDescriptor, MeshRelation>>{
                     {Adj::Cell2Node, MeshRelation::CellNodes}, {Adj::Bnd2Node, MeshRelation::BoundaryNodes}})
                closure.Register(MeshAdjacencyRead(key, owned.spaces.at(key.from), owned.spaces.at(key.to),
                    Adj_PointToGlobal, *owned.relations[size_t(slot)])); });
        const GhostSpec needed{{{EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
                                {EntityKind::Bnd, {Adj::Bnd2Node}, EntityKind::Node}}};
        auto scratch = closure.BuildGhosted(closure.Evaluate(CompiledGhostTree::compile(needed)));
        const auto expandedNodes = scratch.spaces.at(EntityKind::Node);
        auto ghostCoords = scratch.routes.at(EntityKind::Node).TransferRows(expandedNodes, *owned.coords, Runtime::ElementSchema::Arithmetic<real>());
        Runtime::ArrayPair<real> coordinates{*owned.coords, std::move(ghostCoords)};
        auto coordinateRead = coordinates.ReadCombined();
        std::vector<std::vector<index>> cellNodes, boundaryNodes;
        std::vector<std::vector<NodePeriodicBits>> cellBits, boundaryBits;
        std::vector<ElemInfo> cellInfo, boundaryInfo;
        std::vector<index> cellParents, boundaryParents, nodeParents;
        owned.LocalStage(mpi, [&]
                         {
            result.provenance.sourceNamespace = source.identity.meshNamespace;
            auto nodeIds = owned.ids.at(EntityKind::Node).Read();
            nodeParents.resize(size_t(source.spaces.at(EntityKind::Node).NumOwned()));
            OMP::ForSetupRows(0, index(nodeParents.size()), [&](index iRow)
                              { nodeParents[size_t(iRow)] = nodeIds.View()(iRow, 0); });
            auto split = [&](EntityKind kind, const Runtime::Array<index> &cones,
                             const Runtime::Array<ElemInfo> &metadata, auto &children, auto &childInfo, auto &parents, auto &childBits)
            {
                auto cone = cones.Read();
                auto info = metadata.Read();
                auto ids = owned.ids.at(kind).Read();
                std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> images;
                if (owned.periodic) images.emplace((kind == EntityKind::Cell ? *owned.cellNodesPbi : *owned.bndNodesPbi).Read());
                // Stable parent-prefix numbering lets every parent populate its
                // own children, metadata and ancestry without shared appends.
                std::vector<index> childStarts(size_t(cones.RowCount()) + 1, 0);
                OMP::ForSetupRows(0, cones.RowCount(), [&](index iRow)
                {
                    const Elem::Element element{info.View()(iRow, 0).getElemType()};
                    DNDS_check_throw_info(element.GetOrder() == 2 && element.GetO2NumBisect() > 0 &&
                                              cone.View().RowSize(iRow) == size_t(element.GetNumNodes()),
                                          "native O2 bisection requires supported complete quadratic elements");
                    childStarts[size_t(iRow) + 1] = element.GetO2NumBisect();
                });
                for (index iRow = 0; iRow < cones.RowCount(); ++iRow)
                    childStarts[size_t(iRow) + 1] = CheckedSize::Add(childStarts[size_t(iRow)], childStarts[size_t(iRow) + 1]);
                children.resize(size_t(childStarts.back()));
                childInfo.resize(size_t(childStarts.back()));
                parents.resize(size_t(childStarts.back()));
                if (owned.periodic)
                    childBits.resize(size_t(childStarts.back()));
                OMP::ForSetupRows(0, cones.RowCount(), [&](index iRow)
                {
                    const auto row = size_t(iRow);
                    auto parentInfo = info.View()(row, 0);
                    const Elem::Element element{parentInfo.getElemType()};
                    DNDS_check_throw_info(element.GetOrder() == 2 && element.GetO2NumBisect() > 0 &&
                                              cone.View().RowSize(row) == size_t(element.GetNumNodes()),
                                          "native O2 bisection requires supported complete quadratic elements");
                    tSmallCoords coordinates;
                    coordinates.resize(Eigen::NoChange, element.GetNumNodes());
                    std::vector<index> globals(size_t(element.GetNumNodes()));
                    std::vector<NodePeriodicBits> bits(globals.size());
                    for (size_t slot = 0; slot < globals.size(); ++slot)
                    {
                        globals[slot] = cone.View()(row, slot);
                        const auto local = expandedNodes.GlobalToLocal(globals[slot]);
                        DNDS_check_throw_info(local, "native bisection source node closure is incomplete");
                        if (images) bits[slot] = images->View()(row, slot);
                        const auto point = coordinateRead.View()[size_t(*local)];
                        DNDS_check_throw_info(point.size() == 3, "native bisection coordinate width differs");
                        const tPoint value{point[0], point[1], point[2]};
                        coordinates.col(Eigen::Index(slot)) = owned.periodic ? owned.periodicInfo.GetCoordByBits(value, bits[slot]) : value;
                    }
                    DNDS_check_throw_info(coordinates.allFinite(), "native bisection source coordinates are not finite");
                    const auto variant = Elem::GetO2ElemBisectVariant(element, coordinates);
                    for (int child = 0; child < element.GetO2NumBisect(); ++child)
                    {
                        const auto output = size_t(childStarts[row] + child);
                        const auto childElement = element.ObtainO2BisectElem(child);
                        DNDS_check_throw_info(childElement.GetOrder() == 1, "native bisection child is not first order");
                        std::vector<index> nodes(size_t(childElement.GetNumNodes()));
                        auto mutableElement = element;
                        mutableElement.ExtractO2BisectElemNodes(child, variant, globals, nodes);
                        children[output] = std::move(nodes);
                        if (owned.periodic)
                        {
                            std::vector<NodePeriodicBits> labels(size_t(childElement.GetNumNodes()));
                            mutableElement.ExtractO2BisectElemNodes(child, variant, bits, labels);
                            childBits[output] = std::move(labels);
                        }
                        auto value = parentInfo;
                        value.setElemType(childElement.type);
                        childInfo[output] = value;
                        parents[output] = ids.View()(row, 0);
                    }
                });
            };
            split(EntityKind::Cell, *owned.relations[size_t(MeshRelation::CellNodes)], *owned.cellInfo, cellNodes, cellInfo, cellParents, cellBits);
            split(EntityKind::Bnd, *owned.relations[size_t(MeshRelation::BoundaryNodes)], *owned.bndInfo, boundaryNodes, boundaryInfo, boundaryParents, boundaryBits); });
        const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, source.spaces.at(EntityKind::Node).NumOwned());
        const auto cells = nodes.CreateSibling(EntityKind::Cell, index(cellNodes.size()));
        const auto boundaries = nodes.CreateSibling(EntityKind::Bnd, index(boundaryNodes.size()));
        std::optional<NativeMeshInput> input;
        owned.LocalStage(mpi, [&]
                         {
            input.emplace(NativeMeshInput{mpi, owned.dimension, cells, nodes, boundaries,
                                         IndexRows(cellNodes), IndexRows(boundaryNodes), *owned.coords,
                                         Column(cellInfo), Column(boundaryInfo), {}});
            result.provenance.sourceIds.emplace(EntityKind::Node, Column(nodeParents));
            result.provenance.sourceIds.emplace(EntityKind::Cell, Column(cellParents));
            result.provenance.sourceIds.emplace(EntityKind::Bnd, Column(boundaryParents));
            result.provenance.sourceKinds = {{EntityKind::Node, EntityKind::Node}, {EntityKind::Cell, EntityKind::Cell}, {EntityKind::Bnd, EntityKind::Bnd}};
            input->provenance = result.provenance; });
        if (owned.periodic)
        {
            auto bitRows = [](const auto &rows)
            {
                Runtime::RowBuilder builder(rows.size(), sizeof(NodePeriodicBits));
                for (size_t row = 0; row < rows.size(); ++row)
                {
                    builder.ResizeRow(row, rows[row].size());
                    std::copy(rows[row].begin(), rows[row].end(), builder.Row<NodePeriodicBits>(row).data);
                }
                return Runtime::Array<NodePeriodicBits>::FromSharedStorage(builder.Commit());
            };
            owned.LocalStage(mpi, [&]
                             {
                input->periodic = true;
                input->periodicInfo = owned.periodicInfo;
                input->cellNodesPbi = bitRows(cellBits);
                input->boundaryNodesPbi = bitRows(boundaryBits); });
        }
        result.owned = NativeMeshBuilder::Build(std::move(*input), options);
        return result;
    }

    NativeMeshDerivedResult NativeMeshRefinement::Bisect(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                         NativeMeshBuildOptions options)
    {
        // This entry orchestrates capture, owned refinement and one final ghost
        // assembly. Teams are inside independent fill leaves, never around MPI.
        auto result = BisectOwned(NativeMeshAssembler::CaptureOwned(source, mpi), mpi, options);
        GhostSpec recipe;
        result.owned.LocalStage(mpi, [&]
                                { recipe = DefaultNativeMeshGhostSpec(result.owned, options.cellLayers, options.completeNodeStar); });
        return {NativeMeshAssembler::Build(mpi, std::move(result.owned), recipe), std::move(result.provenance)};
    }

}
