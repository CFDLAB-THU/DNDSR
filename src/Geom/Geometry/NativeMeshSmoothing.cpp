#include "Geom/Geometry/NativeMeshSmoothing.hpp"
#include "Geom/Geometry/NativeMeshSmoothingMath.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "Geom/Spatial/PointCloud.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "DNDS/Core/OMP.hpp"
#include <nanoflann.hpp>
#include <set>
#include <sstream>
#include "Solver/Linear.hpp"

namespace DNDS::Geom
{
    namespace
    {
        template <class T>
        Runtime::ArrayPair<T> Halo(const EntitySpace &space, const Runtime::Array<T> &owned, const Runtime::ElementSchema &schema)
        {
            EntityGhostMapping route(space);
            auto ghosts = route.TransferRows(space, owned, schema);
            if (!owned.Layout().IsVariable())
                ghosts = Runtime::Array<T>::FromStorage(Runtime::RowLayout::Dense(ghosts.RowCount(), owned.Layout().Width()), ghosts.Storage());
            return {owned, std::move(ghosts)};
        }
        void Options(const MPIInfo &mpi, const NativeSmoothingOptions &options,
                     const std::shared_ptr<Runtime::CommunicationContext> &communication)
        {
            DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native smoothing requires a communicator");
            Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
            std::string signature;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                int rank, size;
                MPI_Comm_rank(mpi.comm, &rank); MPI_Comm_size(mpi.comm, &size);
                DNDS_check_throw_info(rank == mpi.rank && size == mpi.size && options.nSearch > 0 && options.nIter >= 0 &&
                                          std::isfinite(options.RBFRadius) && options.RBFRadius > 0 && std::isfinite(options.RBFPower) && options.RBFPower > 0 &&
                                          std::isfinite(options.MaxIncludedAngle) && options.MaxIncludedAngle >= 0 && options.MaxIncludedAngle <= 90 &&
                                          options.kernel >= RBF::Distance && options.kernel <= RBF::CPC0,
                                      "invalid native smoothing options");
                std::ostringstream out;
                out << std::hexfloat << options.RBFRadius << ";" << options.RBFPower << ";" << options.MaxIncludedAngle << ";" << options.refDWall
                    << ";" << options.nSearch << ";" << options.nIter << ";" << int(options.kernel);
                signature = out.str(); });
            int length = int(signature.size());
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            std::string expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0');
            MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { DNDS_check_throw_info(expected == signature, "native smoothing options differ across ranks"); });
        }
        Runtime::ElementSchema ElementSchema()
        {
            return Runtime::ElementSchema::For<ElemInfo>();
        }
        Runtime::ElementSchema BitSchema()
        {
            return Runtime::ElementSchema::For<NodePeriodicBits>();
        }
        struct BoundaryCloud
        {
            Runtime::Array<real> values;
            std::vector<index> nodes, offsets;
            std::shared_ptr<Runtime::CommunicationContext> context;
            std::optional<Runtime::RowTransferPlan> route;
        };
        BoundaryCloud Constraints(const NativeSmoothingWorkspace &workspace, const MPIInfo &mpi)
        {
            const auto communication = workspace.Source()->PreparationCommunication();
            // Keep ascending owned-node order. Legacy unordered_set iteration
            // ordered the same constraints differently, affecting tied-neighbor
            // order and floating-point reductions/accumulation in global RBF.
            // Native ordering is intentional; see GlobalRBF's parity note.
            const auto nodes = workspace.Source()->Entities(EntityKind::Node);
            auto displacement = workspace.Displacement().ReadCombined();
            auto roles = workspace.Constraints().ReadCombined();
            const auto &view = workspace.Source()->BorrowHostView();
            std::vector<std::vector<real>> selected;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
            for (index node = 0; node < nodes.NumOwned(); ++node)
            {
                const auto role = NativeSmoothingConstraint(roles.View()(size_t(node),0));
                if (role != NativeSmoothingConstraint::Prescribed && role != NativeSmoothingConstraint::OriginalBoundary) continue;
                const auto point = view.Coord(node);
                std::vector<real> row{point[0],point[1],point[2],0,0,0};
                if (role == NativeSmoothingConstraint::Prescribed)
                    for (size_t component = 0; component < 3; ++component) row[component+3] = displacement.View()(size_t(node),component);
                selected.push_back(std::move(row));
            } });
            index localCount = index(selected.size());
            std::vector<index> counts(size_t(mpi.size)), offsets(size_t(mpi.size) + 1);
            MPI_Allgather(&localCount, 1, DNDS_MPI_INDEX, counts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
            for (int rank = 0; rank < mpi.size; ++rank)
            {
                DNDS_check_throw_info(counts[size_t(rank)] >= 0 && counts[size_t(rank)] <= INT64_MAX-offsets[size_t(rank)],"native smoothing boundary point count overflows");
                offsets[size_t(rank)+1] = offsets[size_t(rank)]+counts[size_t(rank)];
            }
            DNDS_check_throw_info(offsets.back() > 0,"native smoothing has no interpolation constraints"); });
            // Explicit replicated spatial dataset, matching the established RBF
            // algorithm. This is not a mesh topology or remapping allocation.
            std::optional<Runtime::Array<real>> boundary;
            std::vector<Runtime::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
            boundary.emplace(Runtime::RowLayout::Dense(selected.size(),6));
            auto write = boundary->Write();
            for (size_t row = 0; row < selected.size(); ++row)
                for (size_t component = 0; component < 6; ++component) write.View()(row,component) = selected[row][component];
            requests.reserve(size_t(offsets.back()));
            for (int rank = 0; rank < mpi.size; ++rank)
                for (index row = 0; row < counts[size_t(rank)]; ++row)
                    requests.push_back({rank,row,offsets[size_t(rank)] + row}); });
            // Separate payload namespace, with the source safety policy. The
            // replicated constraint dataset remains the established RBF method.
            auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm,
                                                                           !communication || communication->ChecksCollectivePreflight());
            Runtime::RowTransferPlan route(context, selected.size(), size_t(offsets.back()), requests);
            Runtime::Array<real> points(Runtime::RowLayout::Dense(size_t(offsets.back()), 6));
            Runtime::TransferOptions transfer;
            transfer.context = context;
            transfer.mode = Runtime::TransferMode::Transient;
            Runtime::PayloadPlan(route, boundary->Layout(), points.Layout(), Runtime::ElementSchema::Arithmetic<real>())
                .Bind(*boundary, points, Runtime::TransferTransport::Direct, transfer)
                .Begin()
                .Wait();
            BoundaryCloud result;
            result.values = std::move(points);
            result.offsets = std::move(offsets);
            result.context = std::move(context);
            result.route.emplace(std::move(route));
            for (index node = 0; node < nodes.NumOwned(); ++node)
            {
                const auto role = NativeSmoothingConstraint(roles.View()(size_t(node), 0));
                if (role == NativeSmoothingConstraint::Prescribed || role == NativeSmoothingConstraint::OriginalBoundary)
                    result.nodes.push_back(node);
            }
            return result;
        }
        Runtime::Array<real> Replicate(const BoundaryCloud &cloud, const Runtime::Array<real> &owned)
        {
            Runtime::Array<real> global(Runtime::RowLayout::Dense(cloud.values.RowCount(), owned.Layout().Width()));
            Runtime::TransferOptions options;
            options.context = cloud.context;
            options.mode = Runtime::TransferMode::Transient;
            Runtime::PayloadPlan(*cloud.route, owned.Layout(), global.Layout(), Runtime::ElementSchema::Arithmetic<real>())
                .Bind(owned, global, Runtime::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            return global;
        }
        struct SmoothingField
        {
            Runtime::Array<real> values;
            MPIInfo mpi;
            void Init(size_t rows, const MPIInfo &comm, const std::shared_ptr<Runtime::CommunicationContext> &communication)
            {
                mpi = comm;
                FieldTransferDetail::LocalStage(mpi, communication, [&]
                                                { values = Runtime::Array<real>(Runtime::RowLayout::Dense(rows,3)); SetConstant(0); });
            }
            void SetConstant(real value)
            {
                auto write = values.Write();
                OMP::ForRows(0, values.RowCount(), [&](index row)
                             {
                    for (size_t component = 0; component < 3; ++component)
                        write.View()(row, component) = value; });
            }
            void operator=(SmoothingField &other)
            {
                if (this == &other)
                    return;
                auto read = other.values.Read();
                auto write = values.Write();
                OMP::ForRows(0, values.RowCount(), [&](index row)
                             {
                    for (size_t component = 0; component < 3; ++component)
                        write.View()(row, component) = read.View()(row, component); });
            }
            void addTo(SmoothingField &other, real factor)
            {
                auto read = other.values.Read();
                auto write = values.Write();
                OMP::ForRows(0, values.RowCount(), [&](index row)
                             {
                    for (size_t component = 0; component < 3; ++component)
                        write.View()(row, component) += factor * read.View()(row, component); });
            }
            void operator*=(real factor)
            {
                auto write = values.Write();
                OMP::ForRows(0, values.RowCount(), [&](index row)
                             {
                    for (size_t component = 0; component < 3; ++component)
                        write.View()(row, component) *= factor; });
            }
            real dot(SmoothingField &other)
            {
                // Keep native componentwise accumulation. Legacy added an Eigen
                // three-vector dot per row, with different rounding/grouping.
                // Global-RBF GMRES coefficients can therefore differ in bits.
                auto a = values.Read();
                auto b = other.values.Read();
                real result = 0;
                const index workers = OMP::DistributedThreads();
                std::vector<real> partials(size_t(workers > 1 ? workers : 0));
                OMP::SumRows(values.RowCount(), {&result, 1}, {partials.data(), index(partials.size())}, [&](index row, auto sum) noexcept -> NumericalStatus
                             {
                    for (size_t component = 0; component < 3; ++component)
                        sum[0] += a.View()(row, component) * b.View()(row, component);
                    return {}; });
                // The team is joined before the controller's distributed dot.
                MPI_Allreduce(MPI_IN_PLACE, &result, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                return result;
            }
        };

    }

    std::shared_ptr<const NativeSmoothingWorkspace> NativeMeshSmoothing::PrepareBoundary(
        std::shared_ptr<const UnstructuredMeshState> source, const MPIInfo &mpi, NativeSmoothingOptions options,
        const std::function<bool(t_index)> &selected)
    {
        // Selection invokes a user callback and several faces contribute to one
        // nodal normal cluster/displacement. Preserve ordered merge semantics;
        // independent RBF/query rows are parallelized in the solve entries.
        const auto communication = source ? source->PreparationCommunication() : nullptr;
        Options(mpi, options, communication);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            DNDS_check_throw_info(source && selected, "native smoothing requires a source and boundary selector");
            source->CheckRequirements(MeshReadRequirements::CellFlux()); });
        auto owned = NativeMeshAssembler::CaptureOwned(*source, mpi);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info((owned.dimension == 2 || owned.dimension == 3) && owned.faceInfo && owned.relations[size_t(MeshRelation::FaceNodes)],
                                                                "native smoothing requires two- or three-dimensional face topology"); });
        const auto sourceNodes = source->Entities(EntityKind::Node);
        MeshAdjacency faces(Adj::Face2Node, owned.spaces.at(EntityKind::Face), owned.spaces.at(EntityKind::Node), *owned.relations[size_t(MeshRelation::FaceNodes)]);
        auto nodeFaces = AdjacencyRegistry::Inverse(faces.Read(), mpi);
        AdjacencyRegistry setup(mpi);
        setup.Register(faces.Read());
        setup.Register(nodeFaces.Read());
        GhostSpec spec{{{EntityKind::Node, {nodeFaces.Kind()}, EntityKind::Face},
                        {EntityKind::Node, {nodeFaces.Kind(), Adj::Face2Node}, EntityKind::Node}}};
        auto closure = setup.BuildGhosted(setup.Evaluate(CompiledGhostTree::compile(spec)));
        const auto nodes = closure.spaces.at(EntityKind::Node), faceSpace = closure.spaces.at(EntityKind::Face);
        auto faceNodes = closure.relations.at(Adj::Face2Node).Read();
        auto coordinates = Halo(nodes, *owned.coords, Runtime::ElementSchema::Arithmetic<real>());
        auto metadata = Halo(faceSpace, *owned.faceInfo, ElementSchema());
        std::optional<Runtime::ArrayPair<NodePeriodicBits>> images;
        if (owned.periodic)
            images = Halo(faceSpace, *owned.faceNodesPbi, BitSchema());
        auto coordRead = coordinates.ReadCombined();
        auto infoRead = metadata.ReadCombined();
        std::optional<Runtime::ArrayPair<NodePeriodicBits>::ReadLease> bitRead;
        if (images)
            bitRead.emplace(images->ReadCombined());
        std::vector<uint8_t> selectedFaces(size_t(faceSpace.NumLocal())), relevant(selectedFaces.size());
        std::vector<std::vector<tPoint>> clusters(size_t(nodes.NumOwned()));
        auto physical = [&](index face)
        {
            tSmallCoords result(3, Eigen::Index(faceNodes.RowSize(face)));
            for (size_t slot = 0; slot < faceNodes.RowSize(face); ++slot)
            {
                const auto local = nodes.GlobalToLocal(faceNodes.ReadEntry(face, slot).global);
                DNDS_check_throw_info(local, "native smoothing face node closure is incomplete");
                tPoint point;
                for (int component = 0; component < 3; ++component)
                    point[component] = coordRead.View()(size_t(*local), size_t(component));
                if (bitRead)
                    point = owned.periodicInfo.GetCoordByBits(point, bitRead->View()(size_t(face), slot));
                result.col(Eigen::Index(slot)) = point;
            }
            return result;
        };
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            for (index face = 0; face < faceSpace.NumLocal(); ++face)
            {
                for (auto global : faceNodes.GlobalRow(face)) relevant[size_t(face)] |= nodes.Owner(global) == mpi.rank;
                if (!relevant[size_t(face)]) continue;
                auto element = Elem::Element{infoRead.View()(size_t(face),0).getElemType()};
                DNDS_check_throw_info(element.GetOrder() == 2, "native boundary smoothing requires quadratic face topology");
                selectedFaces[size_t(face)] = selected(infoRead.View()(size_t(face),0).zone);
                if (!selectedFaces[size_t(face)]) continue;
                const auto coords = physical(face);
                tPoint normal = tPoint::Zero();
                real area = 0;
                Elem::Quadrature{element,1}.Integration(area,[&](auto &increment,int,const tPoint &,const Elem::tD01Nj &shape)
                {
                    const auto jacobian = Elem::ShapeJacobianCoordD01Nj(coords,shape);
                    normal = owned.dimension == 2 ? FacialJacobianToNormVec<2>(jacobian) : FacialJacobianToNormVec<3>(jacobian);
                    increment = normal.norm();
                });
                DNDS_check_throw_info(normal.allFinite() && normal.norm() > 0, "native smoothing face normal is degenerate");
                const auto original = element.ObtainO1Elem();
                for (size_t slot = 0; slot < size_t(original.GetNumNodes()); ++slot)
                {
                    const auto global = faceNodes.ReadEntry(face,slot).global;
                    if (nodes.Owner(global) != mpi.rank) continue;
                    const auto local = nodes.OwnerLocalRow(global);
                    const tPoint canonical = bitRead ? owned.periodicInfo.GetVectorBackByBits<3,1>(normal,bitRead->View()(size_t(face),slot)) : normal;
                    clusters[size_t(local)].push_back(canonical);
                }
            } });
        std::optional<Runtime::Array<real>> normalRows;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            Runtime::RowBuilder builder(clusters.size(),sizeof(real));
            for (size_t node = 0; node < clusters.size(); ++node)
            {
                builder.ResizeRow(node,3*clusters[node].size());
                auto values = builder.Row<real>(node);
                for (size_t normal = 0; normal < clusters[node].size(); ++normal)
                    for (size_t component = 0; component < 3; ++component) values[3*normal+component] = clusters[node][normal][Eigen::Index(component)];
            }
            normalRows = Runtime::Array<real>::FromSharedStorage(builder.Commit()); });
        auto normals = Halo(nodes, *normalRows, Runtime::ElementSchema::Arithmetic<real>());
        auto normalRead = normals.ReadCombined();
        Runtime::Array<real> displacement(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
        Runtime::Array<uint8_t> roles(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 1));
        index moved = 0;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto output = displacement.Write(); auto mask = roles.Write();
            for (size_t node = 0; node < size_t(nodes.NumOwned()); ++node)
            { mask.View()(node,0) = uint8_t(NativeSmoothingConstraint::Interior); for (int i = 0; i < 3; ++i) output.View()(node,size_t(i)) = 0; }
            std::set<index> processed;
            for (index face = 0; face < faceSpace.NumLocal(); ++face)
            {
                if (!relevant[size_t(face)]) continue;
                auto element = Elem::Element{infoRead.View()(size_t(face),0).getElemType()};
                auto firstOrder = element.ObtainO1Elem();
                const auto globals = faceNodes.GlobalRow(face);
                if (!selectedFaces[size_t(face)])
                {
                    if (!FaceIDIsInternal(infoRead.View()(size_t(face),0).zone))
                        for (auto global : globals) if (nodes.Owner(global) == mpi.rank)
                        {
                            auto &role = mask.View()(size_t(nodes.OwnerLocalRow(global)),0);
                            if (role == uint8_t(NativeSmoothingConstraint::Interior)) role = uint8_t(NativeSmoothingConstraint::OtherBoundary);
                        }
                    continue;
                }
                const auto coords = physical(face);
                for (size_t slot = 0; slot < size_t(firstOrder.GetNumNodes()); ++slot)
                    if (nodes.Owner(globals[slot]) == mpi.rank) mask.View()(size_t(nodes.OwnerLocalRow(globals[slot])),0) = uint8_t(NativeSmoothingConstraint::OriginalBoundary);
                for (size_t slot = size_t(firstOrder.GetNumNodes()); slot < globals.size(); ++slot)
                {
                    if (nodes.Owner(globals[slot]) != mpi.rank) continue;
                    const auto row = size_t(nodes.OwnerLocalRow(globals[slot]));
                    const t_index span = t_index(slot-size_t(firstOrder.GetNumNodes()));
                    const auto desc = firstOrder.ObtainElevNodeSpan(span);
                    std::vector<index> spanNodes(size_t(desc.GetNumNodes()));
                    SmallCoordsAsVector spanCoords;
                    spanCoords.resize(3,desc.GetNumNodes());
                    firstOrder.ExtractElevNodeSpanNodes(span,globals,spanNodes);
                    std::vector<int> sequence(globals.size()), spanSlots(spanNodes.size());
                    std::iota(sequence.begin(),sequence.end(),0);
                    firstOrder.ExtractElevNodeSpanNodes(span,sequence,spanSlots);
                    for (size_t vertex = 0; vertex < spanNodes.size(); ++vertex) spanCoords.col(Eigen::Index(vertex)) = coords.col(spanSlots[vertex]);
                    std::vector<tPoint> edges;
                    DNDS_check_throw_info(spanNodes.size() == 2 || spanNodes.size() == 4,"unsupported native boundary smoothing span");
                    if (spanNodes.size() == 2) edges.push_back((spanCoords.col(1)-spanCoords.col(0)).stableNormalized());
                    else for (int edge = 0; edge < 4; ++edge) edges.push_back((spanCoords.col((edge+1)%4)-spanCoords.col(edge)).stableNormalized());
                    Eigen::Matrix<real,3,4> accepted = Eigen::Matrix<real,3,4>::Zero();
                    for (size_t vertex = 0; vertex < spanNodes.size(); ++vertex)
                    {
                        const auto local = nodes.GlobalToLocal(spanNodes[vertex]);
                        DNDS_check_throw_info(local,"native smoothing normal node is absent");
                        const auto values = normalRead.View()[size_t(*local)];
                        for (size_t at = 0; at < values.size(); at += 3)
                        {
                            tPoint normal{values[at],values[at+1],values[at+2]};
                            // Preserve the established new-node image basis.
                            if (bitRead) normal = owned.periodicInfo.GetVectorByBits<3,1>(normal,bitRead->View()(size_t(face),slot));
                            real sine = 0;
                            for (const auto &edge : edges) sine = std::max(sine,std::abs(edge.dot(normal.stableNormalized())));
                            if (sine > std::sin(pi/180*options.MaxIncludedAngle)) continue;
                            accepted.col(Eigen::Index(vertex)) += normal.stableNormalized();
                        }
                        if (accepted.col(Eigen::Index(vertex)).norm() != 0) accepted.col(Eigen::Index(vertex)).normalize();
                    }
                    const tPoint point = spanNodes.size() == 2 ? NativeSmoothingDetail::LineMidpoint(spanCoords.col(0),spanCoords.col(1),accepted.col(0),accepted.col(1)) :
                        NativeSmoothingDetail::QuadMidpoint(spanCoords.col(0),spanCoords.col(1),spanCoords.col(2),spanCoords.col(3),accepted.col(0),accepted.col(1),accepted.col(2),accepted.col(3));
                    tPoint increment = point - coords.col(Eigen::Index(slot));
                    if (bitRead) increment = owned.periodicInfo.GetVectorBackByBits<3,1>(increment,bitRead->View()(size_t(face),slot));
                    DNDS_check_throw_info(increment.allFinite(),"native boundary smoothing produced nonfinite displacement");
                    for (size_t component = 0; component < 3; ++component) output.View()(row,component) = increment[Eigen::Index(component)];
                    mask.View()(row,0) = uint8_t(NativeSmoothingConstraint::Prescribed);
                    processed.insert(index(row));
                }
            }
            moved = index(processed.size()); });
        MPI_Allreduce(MPI_IN_PLACE, &moved, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        auto result = std::make_shared<NativeSmoothingWorkspace>();
        result->_source = std::move(source);
        result->_nodes = sourceNodes;
        result->_mpi = mpi;
        result->_totalMoved = moved;
        result->_displacement = Halo(sourceNodes, displacement, Runtime::ElementSchema::Arithmetic<real>());
        result->_constraints = Halo(sourceNodes, roles, Runtime::ElementSchema::Arithmetic<uint8_t>());
        result->_ownedRead = std::make_shared<Runtime::Array<real>::ReadLease>(result->_displacement.owned.Read());
        result->_ghostRead = std::make_shared<Runtime::Array<real>::ReadLease>(result->_displacement.ghost.Read());
        result->_ownedRoles = std::make_shared<Runtime::Array<uint8_t>::ReadLease>(result->_constraints.owned.Read());
        result->_ghostRoles = std::make_shared<Runtime::Array<uint8_t>::ReadLease>(result->_constraints.ghost.Read());
        return result;
    }

    std::shared_ptr<const UnstructuredMeshState> NativeMeshSmoothing::LocalRBF(
        const NativeSmoothingWorkspace &workspace, const MPIInfo &mpi, NativeSmoothingOptions options)
    {
        const auto communication = workspace._source ? workspace._source->PreparationCommunication() : nullptr;
        Options(mpi, options, communication);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { workspace.Validate(mpi); });
        const auto source = workspace.Source();
        if (!workspace.TotalMoved())
            return source;
        const auto nodes = source->Entities(EntityKind::Node);
        auto displacement = workspace.Displacement().ReadCombined();
        auto roles = workspace.Constraints().ReadCombined();
        const auto &view = source->BorrowHostView();
        auto boundaryData = Constraints(workspace, mpi);
        const auto &points = boundaryData.values;
        auto data = points.Read();
        PointCloudKDTree cloud;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            cloud.pts.resize(points.RowCount());
            OMP::ForSetupRows(0, points.RowCount(), [&](index row)
                              { cloud.pts[size_t(row)] = tPoint{data.View()(row,0),data.View()(row,1),data.View()(row,2)}; }); });
        using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>, PointCloudKDTree, 3, index>;
        Tree tree(3, cloud);
        Runtime::Array<real> coordinates(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto write = coordinates.Write();
            // nanoflann eagerly builds its index in the constructor; knnSearch
            // is const and each row owns all query/RBF/Eigen scratch.
            OMP::ForSetupRows(0, nodes.NumOwned(), [&](index node)
            {
                const tPoint point = view.Coord(node);
                tPoint increment = tPoint::Zero();
                const auto role = NativeSmoothingConstraint(roles.View()(size_t(node),0));
                if (role == NativeSmoothingConstraint::Prescribed)
                    for (size_t component = 0; component < 3; ++component) increment[Eigen::Index(component)] = displacement.View()(size_t(node),component);
                else if (role != NativeSmoothingConstraint::OriginalBoundary)
                {
                    const auto count = std::min<size_t>(size_t(options.nSearch),cloud.pts.size());
                    std::vector<index> nearest(count);
                    Eigen::VectorXd distances{Eigen::Index(count)};
                    const auto found = tree.knnSearch(point.data(),count,nearest.data(),distances.data());
                    DNDS_check_throw_info(found > 0,"native smoothing nearest-boundary search is empty");
                    tSmallCoords boundaryCoords(3,Eigen::Index(found)), boundaryValues(3,Eigen::Index(found));
                    for (size_t row = 0; row < found; ++row)
                    {
                        boundaryCoords.col(Eigen::Index(row)) = cloud.pts[size_t(nearest[row])];
                        for (size_t component = 0; component < 3; ++component) boundaryValues(Eigen::Index(component),Eigen::Index(row)) = data.View()(size_t(nearest[row]),component+3);
                    }
                    const real minimum = std::sqrt(distances.head(Eigen::Index(found)).minCoeff());
                    // Preserve the legacy radius/units convention exactly.
                    if (minimum < sqr(options.RBFRadius*(RBF::KernelIsCompact(options.kernel)?1:5)))
                    {
                        const tPoint center = boundaryCoords.rowwise().mean();
                        const tSmallCoords relative = boundaryCoords.colwise()-center;
                        const MatrixXR coefficients = RBF::RBFInterpolateSolveCoefsNoPoly(relative,boundaryValues.transpose(),options.RBFRadius,options.kernel);
                        tSmallCoords query(3,1); query = point-center;
                        increment = (RBF::RBFCPC2(relative,query,options.RBFRadius,options.kernel).transpose()*coefficients.topRows(relative.cols())+
                                     coefficients(EigenLast-3,EigenAll)+query.transpose()*coefficients.bottomRows(3)).transpose();
                    }
                }
                DNDS_check_throw_info(increment.allFinite(),"native RBF smoothing produced nonfinite displacement");
                for (size_t component = 0; component < 3; ++component) write.View()(size_t(node),component) = point[Eigen::Index(component)]+increment[Eigen::Index(component)];
            }); });
        auto prepared = Halo(nodes, coordinates, Runtime::ElementSchema::Arithmetic<real>());
        std::shared_ptr<const UnstructuredMeshState> result;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { result = source->WithGeometry(std::move(prepared), {}, view.Periodic()); });
        return result;
    }

    std::shared_ptr<const UnstructuredMeshState> NativeMeshSmoothing::GlobalRBF(
        const NativeSmoothingWorkspace &workspace, const MPIInfo &mpi, NativeSmoothingOptions options)
    {
        // Accepted native/legacy bit difference (2026-10-27): constraint order,
        // division versus reciprocal multiplication, and dot-product grouping
        // change rounding without changing the RBF equations. In the two-rank
        // NACA diagnostic, matching all three restored exact coefficients,
        // displacements and downstream Euler fields; matching any two did not.
        // Retain the native choices, rather than emulate unordered_set order.
        // This is scoped reproducibility evidence, not an accuracy guarantee.
        // Full causes/ablation evidence: dndsr-next-plan repository,
        // experiments/euler-migration-parity/RBF_CAUSE.md.
        const auto communication = workspace._source ? workspace._source->PreparationCommunication() : nullptr;
        Options(mpi, options, communication);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { workspace.Validate(mpi); });
        if (!workspace.TotalMoved())
            return workspace.Source();
        const auto source = workspace.Source();
        const auto nodes = source->Entities(EntityKind::Node);
        const auto &view = source->BorrowHostView();
        auto problem = Constraints(workspace, mpi);
        auto data = problem.values.Read();
        PointCloudKDTree cloud;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            cloud.pts.resize(problem.values.RowCount());
            OMP::ForSetupRows(0, problem.values.RowCount(), [&](index row)
                              { cloud.pts[size_t(row)] = tPoint{data.View()(row,0),data.View()(row,1),data.View()(row,2)}; }); });
        using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>, PointCloudKDTree, 3, index>;
        Tree tree(3, cloud);
        std::vector<std::vector<std::pair<index, real>>> matrix(problem.nodes.size());
        Runtime::Array<real> radii(Runtime::RowLayout::Dense(problem.nodes.size(), 1));
        SmoothingField coefficients, rhs;
        coefficients.Init(problem.nodes.size(), mpi, communication);
        rhs.Init(problem.nodes.size(), mpi, communication);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto radius = radii.Write(); auto b = rhs.values.Write();
            OMP::ForSetupRows(0, index(problem.nodes.size()), [&](index iRow)
            {
                const auto row = size_t(iRow);
                const tPoint point = view.Coord(problem.nodes[row]);
                const auto count = std::min(size_t(options.nSearch),cloud.pts.size());
                std::vector<index> nearest(count); Eigen::VectorXd distances{Eigen::Index(count)};
                const auto found = tree.knnSearch(point.data(),count,nearest.data(),distances.data());
                DNDS_check_throw_info(found > 0,"native global RBF has no neighboring constraints");
                const real support = options.RBFRadius*std::sqrt(distances.head(Eigen::Index(found)).maxCoeff());
                DNDS_check_throw_info(std::isfinite(support) && support > 0,"native global RBF radius is zero; include another distinct constraint");
                radius.View()(row,0) = support;
                // Keep native division. Legacy multiplied by a rounded
                // reciprocal; different matrix-weight bits can propagate to
                // GMRES, mesh coordinates/metrics and the solver RHS.
                const Eigen::VectorXd normalized = distances.head(Eigen::Index(found)).array().pow(.5/options.RBFPower)/std::pow(support,1/options.RBFPower);
                const auto basis = RBF::FRBFBasis(normalized,options.kernel);
                for (size_t slot = 0; slot < found; ++slot) matrix[row].push_back({nearest[slot],basis(Eigen::Index(slot))});
                for (size_t component = 0; component < 3; ++component) b.View()(row,component) = data.View()(size_t(problem.offsets[size_t(mpi.rank)])+row,component+3);
            }); });
        std::vector<Runtime::RowRequest> coefficientRequests;
        std::map<index, size_t> compactColumns;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            for (const auto &row : matrix)
                for (auto [column,value] : row) compactColumns.emplace(column,0);
            for (auto &[global,local] : compactColumns)
            {
                local = coefficientRequests.size();
                const int owner = int(std::upper_bound(problem.offsets.begin(),problem.offsets.end(),global)-problem.offsets.begin())-1;
                DNDS_check_throw_info(owner >= 0 && owner < mpi.size,"native RBF coefficient owner is unavailable");
                coefficientRequests.push_back({owner,global-problem.offsets[size_t(owner)],index(local)});
            }
            for (auto &row : matrix) for (auto &[column,value] : row) column = index(compactColumns.at(column)); });
        Runtime::RowTransferPlan coefficientRoute(problem.context, problem.nodes.size(), coefficientRequests.size(), coefficientRequests);
        const auto coefficientLayout = Runtime::RowLayout::Dense(coefficientRequests.size(), 3);
        Runtime::PayloadPlan coefficientPayload(coefficientRoute, coefficients.values.Layout(), coefficientLayout, Runtime::ElementSchema::Arithmetic<real>());
        Linear::GMRES_LeftPreconditioned<SmoothingField> solver(5, [&](auto &field)
                                                                { field.Init(problem.nodes.size(), mpi, communication); });
        solver.solve(
            [&](SmoothingField &x, SmoothingField &result)
            {
                Runtime::Array<real> global(coefficientLayout);
                Runtime::TransferOptions transfer;
                transfer.context = problem.context;
                transfer.mode = Runtime::TransferMode::Transient;
                coefficientPayload.Bind(x.values, global, Runtime::TransferTransport::Direct, transfer).Begin().Wait();
                auto values = global.Read();
                auto output = result.values.Write();
                OMP::ForRows(0, index(matrix.size()), [&](index iRow)
                             {
                    const auto row = size_t(iRow);
                    for (size_t component = 0; component < 3; ++component)
                    {
                        real sum = 0;
                        for (auto [column, value] : matrix[row])
                            sum += value * values.View()(size_t(column), component);
                        output.View()(row, component) = sum;
                    } });
            },
            // The legacy path overwrites its LU-preconditioned value with x;
            // preserve that effective identity preconditioner deliberately.
            [&](SmoothingField &x, SmoothingField &result)
            { result = x; },
            [&](SmoothingField &x, SmoothingField &y)
            { return x.dot(y); }, rhs, coefficients, uint32_t(options.nIter),
            [](int, real residual, real reference)
            { return residual < reference * 1e-6; });
        auto allCoefficients = Replicate(problem, coefficients.values);
        auto allRadii = Replicate(problem, radii);
        auto coefficientRead = allCoefficients.Read();
        auto radiusRead = allRadii.Read();
        auto roles = workspace.Constraints().ReadCombined();
        auto prescribed = workspace.Displacement().ReadCombined();
        Runtime::Array<real> increments(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
        std::vector<index> insideNodes;
        PointCloudKDTree interior;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto write = increments.Write();
            for (index node = 0; node < nodes.NumOwned(); ++node)
            {
                const auto role = NativeSmoothingConstraint(roles.View()(size_t(node),0));
                for (size_t component = 0; component < 3; ++component)
                    write.View()(size_t(node),component) = role == NativeSmoothingConstraint::Prescribed ? prescribed.View()(size_t(node),component) : 0;
                if (role != NativeSmoothingConstraint::Prescribed && role != NativeSmoothingConstraint::OriginalBoundary)
                { insideNodes.push_back(node); interior.pts.push_back(view.Coord(node)); }
            } });
        Tree destinations(3, interior);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto write = increments.Write();
            // Different constraint supports overlap in the destination nodes.
            // Keep this accumulation in its prescribed constraint order; a
            // parallel boundary loop would race and reorder FP summation.
            for (size_t boundary = 0; boundary < cloud.pts.size(); ++boundary)
            {
                const real support = radiusRead.View()(boundary,0);
#if NANOFLANN_VERSION < 0x150
                std::vector<std::pair<index,real>> matches;
                nanoflann::SearchParams parameters{};
#else
                std::vector<nanoflann::ResultItem<index,real>> matches;
                nanoflann::SearchParameters parameters{};
#endif
                // Preserve the legacy squared-distance/radius convention.
                destinations.radiusSearch(cloud.pts[boundary].data(),support,matches,parameters);
                Eigen::VectorXd distances{Eigen::Index(matches.size())};
                for (size_t found = 0; found < matches.size(); ++found) distances[Eigen::Index(found)] = std::sqrt(matches[found].second)/std::pow(support,1/options.RBFPower);
                const auto basis = RBF::FRBFBasis(distances,options.kernel);
                for (size_t found = 0; found < matches.size(); ++found)
                    for (size_t component = 0; component < 3; ++component)
                        write.View()(size_t(insideNodes[size_t(matches[found].first)]),component) += basis(Eigen::Index(found))*coefficientRead.View()(boundary,component);
            } });
        Runtime::Array<real> coordinates(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto read = increments.Read(); auto write = coordinates.Write();
            OMP::ForSetupRows(0, nodes.NumOwned(), [&](index node)
            {
                for (size_t component = 0; component < 3; ++component)
                {
                    const real value = view.Coord(node)[Eigen::Index(component)]+read.View()(size_t(node),component);
                    DNDS_check_throw_info(std::isfinite(value),"native global RBF produced nonfinite coordinates");
                    write.View()(size_t(node),component) = value;
                }
            }); });
        auto prepared = Halo(nodes, coordinates, Runtime::ElementSchema::Arithmetic<real>());
        std::shared_ptr<const UnstructuredMeshState> result;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { result = source->WithGeometry(std::move(prepared), {}, view.Periodic()); });
        return result;
    }

    std::shared_ptr<const UnstructuredMeshState> NativeMeshSmoothing::Elasticity(
        const NativeSmoothingWorkspace &workspace, const MPIInfo &mpi, NativeSmoothingOptions options)
    {
        const auto communication = workspace._source ? workspace._source->PreparationCommunication() : nullptr;
        Options(mpi, options, communication);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { workspace.Validate(mpi); });
        if (!workspace.TotalMoved())
            return workspace.Source();
        auto source = workspace.Source();
        auto owned = NativeMeshAssembler::CaptureOwned(*source, mpi);
        const auto sourceNodes = source->Entities(EntityKind::Node);
        MeshAdjacency cells(Adj::Cell2Node, owned.spaces.at(EntityKind::Cell), owned.spaces.at(EntityKind::Node), *owned.relations[size_t(MeshRelation::CellNodes)]);
        auto support = AdjacencyRegistry::Inverse(cells.Read(), mpi);
        AdjacencyRegistry setup(mpi);
        setup.Register(cells.Read());
        setup.Register(support.Read());
        GhostSpec spec{{{EntityKind::Node, {support.Kind()}, EntityKind::Cell},
                        {EntityKind::Node, {support.Kind(), Adj::Cell2Node}, EntityKind::Node}}};
        auto publication = setup.BuildGhosted(setup.Evaluate(CompiledGhostTree::compile(spec)));
        const auto nodes = publication.spaces.at(EntityKind::Node), cellSpace = publication.spaces.at(EntityKind::Cell);
        auto cones = publication.relations.at(Adj::Cell2Node).Read();
        auto metadata = Halo(cellSpace, *owned.cellInfo, ElementSchema());
        auto coordinateRows = Halo(nodes, *owned.coords, Runtime::ElementSchema::Arithmetic<real>());
        std::optional<Runtime::ArrayPair<NodePeriodicBits>> images;
        if (owned.periodic)
            images = Halo(cellSpace, *owned.cellNodesPbi, BitSchema());
        auto coordinateRead = coordinateRows.ReadCombined();
        auto infoRead = metadata.ReadCombined();
        std::optional<Runtime::ArrayPair<NodePeriodicBits>::ReadLease> bits;
        if (images)
            bits.emplace(images->ReadCombined());
        auto roles = workspace.Constraints().ReadCombined();
        auto prescribed = workspace.Displacement().ReadCombined();
        std::vector<std::map<index, tJacobi>> matrix(size_t(nodes.NumOwned()));
        SmoothingField rhs, solution;
        rhs.Init(size_t(nodes.NumOwned()), mpi, communication);
        solution.Init(size_t(nodes.NumOwned()), mpi, communication);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto b = rhs.values.Write();
            // Elements share nodal matrix/RHS destinations. Preserve ordered
            // assembly, including the accumulated periodic block transforms.
            // Per-row actions after assembly are independent below.
            for (index cell = 0; cell < cellSpace.NumLocal(); ++cell)
            {
                const auto globals = cones.GlobalRow(cell);
                if (std::none_of(globals.begin(),globals.end(),[&](index global) { return nodes.Owner(global) == mpi.rank; })) continue;
                SmallCoordsAsVector coordinates,displacements;
                coordinates.resize(3,Eigen::Index(globals.size())); displacements.resizeLike(coordinates);
                std::vector<index> localNodes;
                for (size_t slot = 0; slot < globals.size(); ++slot)
                {
                    const auto local = nodes.GlobalToLocal(globals[slot]);
                    DNDS_check_throw_info(local,"native elasticity cell-node closure is incomplete");
                    localNodes.push_back(*local);
                    tPoint point;
                    for (size_t component = 0; component < 3; ++component) point[Eigen::Index(component)] = coordinateRead.View()(size_t(*local),component);
                    tPoint zero = tPoint::Zero();
                    if (bits)
                    {
                        point = owned.periodicInfo.GetCoordByBits(point,bits->View()(size_t(cell),slot));
                        // Legacy GetCoordsOnCell(uCur) applies coordinate frames
                        // to its initially zero field. Its JDet ratio exponent
                        // remains -0.0 below, so preserve that calculation.
                        zero = owned.periodicInfo.GetCoordByBits(zero,bits->View()(size_t(cell),slot));
                    }
                    coordinates.col(Eigen::Index(slot)) = point; displacements.col(Eigen::Index(slot)) = zero;
                }
                const auto element = Elem::Element{infoRead.View()(size_t(cell),0).getElemType()};
                DNDS_check_throw_info(element.GetOrder() == 2,"native elasticity smoothing requires quadratic cells");
                auto stiffness = NativeSmoothingDetail::Stiffness(element,coordinates,displacements,owned.dimension);
                DNDS_check_throw_info(stiffness.allFinite(),"native elasticity stiffness is nonfinite");
                const index count = index(globals.size());
                for (size_t i = 0; i < globals.size(); ++i)
                {
                    const index node = localNodes[i];
                    if (node >= nodes.NumOwned()) continue;
                    auto &row = matrix[size_t(node)];
                    for (size_t j = 0; j < globals.size(); ++j)
                    {
                        auto found = row.try_emplace(localNodes[j],tJacobi::Zero()).first;
                        auto &block = found->second;
                        for (int a = 0; a < 3; ++a)
                            for (int c = 0; c < 3; ++c) block(a,c) += stiffness(a*count+index(i),c*count+index(j));
                        if (bits)
                        {
                            const tJacobi iFrame = owned.periodicInfo.GetVectorByBits<3,3>(tJacobi::Identity(),bits->View()(size_t(cell),i));
                            const tJacobi jFrame = owned.periodicInfo.GetVectorByBits<3,3>(tJacobi::Identity(),bits->View()(size_t(cell),j));
                            // Deliberately preserve the legacy accumulated-block
                            // transform, pending a separate numerical review.
                            block = jFrame.transpose()*block*iFrame;
                        }
                    }
                    for (size_t component = 0; component < 3; ++component) b.View()(size_t(node),component) += stiffness(index(component)*count+index(i),EigenLast);
                    const auto role = NativeSmoothingConstraint(roles.View()(size_t(node),0));
                    if (role != NativeSmoothingConstraint::Interior)
                    {
                        const real diagonal = row.at(node).array().abs().maxCoeff();
                        for (size_t component = 0; component < 3; ++component)
                            b.View()(size_t(node),component) = role == NativeSmoothingConstraint::Prescribed ? prescribed.View()(size_t(node),component)*diagonal : 0;
                        for (auto &[column,block] : row)
                        {
                            block.setZero();
                            if (column == node) block = tJacobi::Identity()*diagonal;
                        }
                    }
                    if (owned.dimension != 3) row.at(node)(2,2) += row.at(node).array().abs().maxCoeff();
                }
            }
            for (index node = 0; node < nodes.NumOwned(); ++node)
            {
                // Unused canonical points have no stiffness support. Retain
                // them unchanged rather than introducing a singular work row.
                auto &row = matrix[size_t(node)];
                if (row.empty()) row.emplace(node,tJacobi::Identity());
            } });
        EntityGhostMapping nodeRoute(nodes);
        Linear::GMRES_LeftPreconditioned<SmoothingField> solver(5, [&](auto &field)
                                                                { field.Init(size_t(nodes.NumOwned()), mpi, communication); });
        solver.solve(
            [&](SmoothingField &x, SmoothingField &result)
            {
                Runtime::ArrayPair<real> values{x.values, nodeRoute.TransferRows(nodes, x.values, Runtime::ElementSchema::Arithmetic<real>())};
                auto read = values.ReadCombined();
                auto output = result.values.Write();
                OMP::ForRows(0, index(matrix.size()), [&](index iRow)
                             {
                    const auto row = size_t(iRow);
                    tPoint sum = tPoint::Zero();
                    for (const auto &[column, block] : matrix[row])
                        sum += block * tPoint{read.View()(size_t(column), 0), read.View()(size_t(column), 1), read.View()(size_t(column), 2)};
                    for (size_t component = 0; component < 3; ++component)
                        output.View()(row, component) = sum[Eigen::Index(component)]; });
            },
            [&](SmoothingField &x, SmoothingField &result)
            {
                auto input = x.values.Read();
                auto output = result.values.Write();
                OMP::ForSetupRows(0, index(matrix.size()), [&](index iRow)
                                  {
                    const auto row = size_t(iRow);
                    const tPoint value = matrix[row].at(index(row)).fullPivLu().solve(tPoint{input.View()(row, 0), input.View()(row, 1), input.View()(row, 2)});
                    for (size_t component = 0; component < 3; ++component)
                        output.View()(row, component) = value[Eigen::Index(component)]; });
            },
            [&](SmoothingField &x, SmoothingField &y)
            { return x.dot(y); }, rhs, solution, uint32_t(options.nIter),
            [](int, real residual, real reference)
            { return residual < reference * 1e-10; });
        Runtime::Array<real> coordinates(Runtime::RowLayout::Dense(size_t(sourceNodes.NumOwned()), 3));
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            auto read = solution.values.Read(); auto write = coordinates.Write();
            OMP::ForSetupRows(0, sourceNodes.NumOwned(), [&](index node)
            {
                for (size_t component = 0; component < 3; ++component)
                {
                    const real displacement = owned.dimension == 2 && component == 2 ? 0 : read.View()(size_t(node),component);
                    const real value = source->BorrowHostView().Coord(node)[Eigen::Index(component)]+displacement;
                    DNDS_check_throw_info(std::isfinite(value),"native elasticity smoothing produced nonfinite coordinates");
                    write.View()(size_t(node),component) = value;
                }
            }); });
        auto prepared = Halo(sourceNodes, coordinates, Runtime::ElementSchema::Arithmetic<real>());
        std::shared_ptr<const UnstructuredMeshState> result;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { result = source->WithGeometry(std::move(prepared), {}, source->BorrowHostView().Periodic()); });
        return result;
    }
}
