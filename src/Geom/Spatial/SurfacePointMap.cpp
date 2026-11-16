#include "Geom/Spatial/SurfacePointMap.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Communication/SparseBatchExchange.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Geom
{
    namespace R = Runtime;
    namespace Detail
    {
        struct SurfacePatchData
        {
            std::vector<SurfaceElementRecord> elements;
            R::Array<real> coordinates;
            std::shared_ptr<const R::Array<real>::ReadLease> read;
            RigidPose pose;
            std::shared_ptr<const BoundingBoxTree> tree;
            real diameter = 0;
            ElementGeometryView Geometry(index iElement) const
            {
                DNDS_check_throw_info(iElement >= 0 && iElement < index(elements.size()), "surface element outside snapshot");
                const auto row = read->View()[iElement];
                return {elements[size_t(iElement)].type, row.data(), row.size() / 3};
            }
            void BuildIndex(BoundingBoxBackend backend)
            {
                std::vector<SpatialBox> boxes(elements.size());
                diameter = 0;
                OMP::ForSetupRows(0, index(elements.size()), [&](index iElement)
                                  { boxes[size_t(iElement)] = Geometry(iElement).Bounds(); });
                for (const auto &box : boxes)
                    diameter = std::max(diameter, (box.upper - box.lower).norm());
                tree = std::make_shared<BoundingBoxTree>(boxes, backend);
            }
        };
        struct SurfacePointMapData
        {
            SurfacePatch surface;
            std::vector<SurfacePointMatch> matches;
            explicit SurfacePointMapData(SurfacePatch s) : surface(std::move(s)) {}
        };
        struct DistributedSurfacePointMapData : SurfacePointMapData
        {
            MPIInfo mpi;
            std::shared_ptr<R::CommunicationContext> context;
            DistributedSurfacePointMapData(SurfacePatch s, MPIInfo m, std::shared_ptr<R::CommunicationContext> c)
                : SurfacePointMapData(std::move(s)), mpi(m), context(std::move(c)) {}
        };
    }
    namespace
    {
        void CheckQueries(const R::Array<real> &points, const R::Array<index> &ids)
        {
            DNDS_check_throw_info(points.GetDevice() == R::Device::Host() && ids.GetDevice() == R::Device::Host() &&
                                      points.Layout().Kind() == R::RowLayoutKind::Table && points.Layout().Width() == 3 &&
                                      ids.Layout().Kind() == R::RowLayoutKind::Table && ids.Layout().Width() == 1 &&
                                      points.RowCount() == ids.RowCount(),
                                  "surface queries require host points[N,3] and IDs[N,1]");
            auto p = points.Read();
            auto k = ids.Read();
            const auto coordinates = p.View<R::FixedRows<3>>();
            const auto queryKeys = k.View<R::FixedRows<1>>();
            std::vector<index> keys(size_t(ids.RowCount()));
            OMP::ForSetupRows(0, points.RowCount(), [&](index iQuery)
                              {
                DNDS_check_throw_info(tPoint(coordinates(iQuery, 0), coordinates(iQuery, 1), coordinates(iQuery, 2)).allFinite(), "nonfinite surface query");
                keys[size_t(iQuery)] = queryKeys(iQuery, 0); });
            std::sort(keys.begin(), keys.end());
            DNDS_check_throw_info(std::adjacent_find(keys.begin(), keys.end()) == keys.end(), "duplicate local surface query ID");
        }
    }
    void SurfaceSearchSettings::Validate() const
    {
        PointProjectionSettings::Validate();
        DNDS_check_throw_info(coarseBoxesPerRank >= 1 && coarseBoxesPerRank <= 10, "invalid surface coarse-box count");
    }
    SurfacePatch SurfacePatch::Create(std::vector<SurfaceElementRecord> elements, R::Array<real> coordinates, RigidPose pose, BoundingBoxBackend backend)
    {
        pose.Validate();
        DNDS_check_throw_info(coordinates.GetDevice() == R::Device::Host() && coordinates.RowCount() == index(elements.size()), "surface coordinate row count or placement differs");
        auto data = std::make_shared<Detail::SurfacePatchData>();
        data->elements = std::move(elements);
        data->coordinates = std::move(coordinates);
        data->pose = pose;
        data->read = std::make_shared<const R::Array<real>::ReadLease>(data->coordinates.Read());
        std::vector<index> keys(data->elements.size());
        OMP::ForSetupRows(0, index(data->elements.size()), [&](index iElement)
                          {
            const auto row = data->read->View()[iElement];
            DNDS_check_throw_info(row.size() % 3 == 0, "surface coordinate row is not a node triple sequence");
            data->Geometry(iElement).Validate();
            keys[size_t(iElement)] = data->elements[size_t(iElement)].id; });
        std::sort(keys.begin(), keys.end());
        DNDS_check_throw_info(std::adjacent_find(keys.begin(), keys.end()) == keys.end(), "duplicate local surface element ID");
        data->BuildIndex(backend);
        return SurfacePatch(std::move(data));
    }
    SurfacePatch SurfacePatch::WithPose(RigidPose pose) const
    {
        pose.Validate();
        auto data = std::make_shared<Detail::SurfacePatchData>(*_data);
        data->pose = pose;
        // Rigid placement preserves reference geometry and point distances.
        // Keep its immutable local BVH; transform queries instead of rebuilding.
        return SurfacePatch(std::move(data));
    }
    SurfacePatch SurfacePatch::WithBackend(BoundingBoxBackend backend) const
    {
        auto data = std::make_shared<Detail::SurfacePatchData>(*_data);
        data->BuildIndex(backend);
        return SurfacePatch(std::move(data));
    }
    BoundingBoxBackend SurfacePatch::Backend() const { return _data->tree->Backend(); }
    index SurfacePatch::Size() const { return index(_data->elements.size()); }
    SurfaceElementRecord SurfacePatch::Element(index iElement) const
    {
        DNDS_check_throw_info(iElement >= 0 && iElement < Size(), "surface element outside snapshot");
        return _data->elements[size_t(iElement)];
    }
    ElementGeometryView SurfacePatch::Geometry(index iElement) const { return _data->Geometry(iElement); }
    RigidPose SurfacePatch::Pose() const { return _data->pose; }

    std::optional<SurfacePointMatch> ResolveSurfaceMatches(const std::vector<SurfaceMatchProposal> &proposals, real distanceTolerance)
    {
        DNDS_check_throw_info(std::isfinite(distanceTolerance) && distanceTolerance >= 0, "invalid donor arbitration tolerance");
        real closest = std::numeric_limits<real>::infinity();
        for (const auto &proposal : proposals)
        {
            DNDS_check_throw_info(std::isfinite(proposal.match.distance) && proposal.match.distance >= 0, "invalid donor proposal distance");
            closest = std::min(closest, proposal.match.distance);
        }
        std::optional<SurfacePointMatch> best;
        index interiors = 0;
        for (const auto &[candidate, interior] : proposals)
        {
            if (candidate.distance > closest + distanceTolerance)
                continue;
            interiors += interior;
            DNDS_check_throw_info(interiors <= 1, "ambiguous surface element interiors");
            if (!best || candidate.elementId < best->elementId)
                best = candidate;
        }
        return best;
    }
    SurfacePointMap SurfacePointMap::Match(SurfacePatch surface, const R::Array<real> &points, const R::Array<index> &queryIds, SurfaceSearchSettings settings)
    {
        settings.Validate();
        CheckQueries(points, queryIds);
        auto pointRead = points.Read();
        auto idRead = queryIds.Read();
        const auto p = pointRead.View<R::FixedRows<3>>();
        const auto ids = idRead.View<R::FixedRows<1>>();
        auto data = std::make_shared<Detail::SurfacePointMapData>(surface);
        data->matches.resize(size_t(points.RowCount()));
        // The spatial index is eagerly built; each query owns its proposal
        // scratch and output row. No lazy BVH mutation or shared append occurs.
        OMP::ForSetupRows(0, points.RowCount(), [&](index iQuery)
                          {
            const tPoint world{p(iQuery, 0), p(iQuery, 1), p(iQuery, 2)};
            const tPoint referenceQuery = surface.Pose().Inverse(world);
            std::vector<SurfaceMatchProposal> proposals;
            surface._data->tree->Query(referenceQuery, settings.maxGap + ElementGeometryView::ReferencePadding(surface._data->diameter, settings.referenceTolerance), [&](index iElement)
                                       {
                const auto geometry = surface.Geometry(iElement);
                const auto projection = geometry.Project(referenceQuery,settings);
                if (!projection.Accepted())
                    return;
                proposals.push_back({{ids(iQuery,0),iElement,surface.Element(iElement).id,0,projection.reference,
                                      surface.Pose().Apply(projection.point),projection.point,surface.Pose().rotation * projection.normal,projection.distance},
                                     geometry.ContainsReference(projection.reference,-std::max(settings.referenceTolerance,32*std::numeric_limits<real>::epsilon()))}); });
            const auto match = ResolveSurfaceMatches(proposals, settings.projectionTolerance);
            DNDS_check_throw_info(match, "uncovered surface query " + std::to_string(ids(iQuery, 0)));
            data->matches[size_t(iQuery)] = *match; });
        return SurfacePointMap(std::move(data));
    }
    index SurfacePointMap::Size() const { return index(_data->matches.size()); }
    SurfacePointMatch SurfacePointMap::Get(index iQuery) const
    {
        DNDS_check_throw_info(iQuery >= 0 && iQuery < Size(), "surface query outside map");
        return _data->matches[size_t(iQuery)];
    }
    void SurfacePointMap::ValidateFor(const SurfacePatch &active) const { DNDS_check_throw_info(active.Same(_data->surface), "active surface snapshot or pose differs from map"); }

    DistributedSurfacePointMap DistributedSurfacePointMap::Match(MPIInfo mpi, SurfacePatch surface, const R::Array<real> &points,
                                                                 const R::Array<index> &queryIds, SurfaceRoutingMode mode, SurfaceSearchSettings settings,
                                                                 std::shared_ptr<R::CommunicationContext> communication)
    {
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        // Search owns a distinct tag namespace: the caller may retain bindings
        // on its coupling context. Construction agrees policy before local
        // stages; inherit the mode, never borrow occupied tags.
        auto context = std::make_shared<R::CommunicationContext>(mpi.comm, checks);
        context->LocalStage([&]
                            {
            int relation = MPI_UNEQUAL;
            DNDS_check_throw_info(MPI_Comm_compare(mpi.comm, communication ? communication->Communicator() : context->Communicator(), &relation) == MPI_SUCCESS &&
                                      (relation == MPI_IDENT || relation == MPI_CONGRUENT),
                                  "surface communication context has different ranks/order"); });
        std::optional<R::Array<real>::ReadLease> pointRead;
        std::optional<R::Array<index>::ReadLease> idRead;
        context->LocalStage([&]
                            {
            settings.Validate();
            CheckQueries(points, queryIds);
            DNDS_check_throw_info(mode == SurfaceRoutingMode::CoarseBoxes || mode == SurfaceRoutingMode::AllGatherSurface,"unknown surface routing mode");
            pointRead.emplace(points.Read());
            idRead.emplace(queryIds.Read()); });
        real controls[]{real(int(mode)), settings.maxGap, settings.referenceTolerance, settings.projectionTolerance, real(settings.maxIterations), real(settings.coarseBoxesPerRank)};
        real low[6], high[6];
        MPI_Allreduce(controls, low, 6, DNDS_MPI_REAL, MPI_MIN, mpi.comm);
        MPI_Allreduce(controls, high, 6, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        for (int i = 0; i < 6; ++i)
            DNDS_check_throw_info(low[i] == high[i], "distributed surface settings differ across ranks");
        const auto queries = pointRead->View<R::FixedRows<3>>();
        const auto ids = idRead->View<R::FixedRows<1>>();
        R::Array<index> elements;
        R::Array<real> geometry;
        std::vector<SpatialBox> localBounds;
        SpatialBox coarse;
        context->LocalStage([&]
                            {
            if (mode == SurfaceRoutingMode::CoarseBoxes)
            {
                // Sparse search keeps geometry and its reference BVH on the
                // owning rank. Only coarse world boxes need updating for pose.
                localBounds.resize(size_t(surface.Size()));
                for (index iElement = 0; iElement < surface.Size(); ++iElement)
                {
                    localBounds[size_t(iElement)] = surface.Geometry(iElement).Bounds(surface.Pose(), settings.referenceTolerance);
                    coarse.Add(localBounds[size_t(iElement)]);
                }
                return;
            }
            // Phase 1: fixed records for the current <=4-node family.
            // Integer columns: owner element row, persistent ID, type, nodes,
            // geometry rank. Real columns: coordinates[12], rotation[9],
            // translation[3]. These contain no volume-cell or equation state.
            elements = R::Array<index>(R::RowLayout::Dense(surface.Size(),5));
            geometry = R::Array<real>(R::RowLayout::Dense(surface.Size(),24));
            auto e = elements.Write();
            auto g = geometry.Write();
            for (index iElement = 0; iElement < surface.Size(); ++iElement)
            {
                const auto shape = surface.Geometry(iElement);
                const auto coords = shape.Coordinates();
                const auto pose = surface.Pose();
                const index key[]{iElement,surface.Element(iElement).id,index(shape.Type()),coords.cols(),mpi.rank};
                std::copy_n(key,5,e.View()[iElement].data());
                std::fill(g.View()[iElement].begin(),g.View()[iElement].end(),0);
                std::copy_n(coords.data(),3*coords.cols(),g.View()[iElement].data());
                for (int row = 0; row < 3; ++row)
                {
                    g.View()(iElement,21+row) = pose.translation[row];
                    for (int col = 0; col < 3; ++col) g.View()(iElement,12+3*row+col) = pose.rotation(row,col);
                }
            } });
        // Phase 2: route geometry to query owners for the dense baseline, or
        // route only candidate queries to geometry owners for coarse routing.
        R::Array<real> candidatePoints;
        R::Array<index> candidateKeys;
        std::vector<int> origins;
        if (mode == SurfaceRoutingMode::AllGatherSurface)
        {
            std::vector<std::vector<index>> all;
            context->LocalStage([&]
                                {
                all.resize(size_t(mpi.size));
                for (auto &peer : all)
                {
                    peer.resize(size_t(surface.Size()));
                    std::iota(peer.begin(), peer.end(), index(0));
                } });
            const auto route = R::SparseBatchExchange::Build(context, surface.Size(), std::move(all));
            elements = route.Exchange(elements);
            geometry = route.Exchange(geometry);
            context->LocalStage([&]
                                {
                candidatePoints = points;
                candidateKeys = R::Array<index>(R::RowLayout::Dense(points.RowCount(),2));
                auto write = candidateKeys.Write();
                for (index iQuery = 0; iQuery < points.RowCount(); ++iQuery)
                {
                    write.View()(iQuery,0) = iQuery;
                    write.View()(iQuery,1) = ids(iQuery,0);
                }
                origins.assign(size_t(points.RowCount()),mpi.rank); });
        }
        else
        {
            const int count = settings.coarseBoxesPerRank;
            std::vector<real> boxes, local;
            context->LocalStage([&]
                                {
                boxes.resize(size_t(mpi.size)*6*count);
                local.resize(size_t(6*count));
                std::vector<index> order(size_t(surface.Size()));
                std::iota(order.begin(),order.end(),index(0));
                int axis = 0;
                (coarse.upper-coarse.lower).maxCoeff(&axis);
                std::sort(order.begin(),order.end(),[&](index a,index b)
                {
                    const real x = localBounds[size_t(a)].lower[axis]+localBounds[size_t(a)].upper[axis],y = localBounds[size_t(b)].lower[axis]+localBounds[size_t(b)].upper[axis];
                    return x == y ? a < b : x < y;
                });
                for (int iBox = 0; iBox < count; ++iBox)
                {
                    SpatialBox box;
                    const auto [begin,end] = EvenSplitRange(iBox,count,surface.Size());
                    for (index i = begin; i < end; ++i)
                        box.Add(localBounds[size_t(order[size_t(i)])]);
                    for (int d = 0; d < 3; ++d)
                    {
                        local[6*iBox+d] = box.lower[d];
                        local[6*iBox+d+3] = box.upper[d];
                    }
                } });
            MPI_Allgather(local.data(), 6 * count, DNDS_MPI_REAL, boxes.data(), 6 * count, DNDS_MPI_REAL, mpi.comm);
            std::vector<std::vector<index>> destinations;
            R::Array<index> keys;
            context->LocalStage([&]
                                {
                destinations.resize(size_t(mpi.size));
                keys = R::Array<index>(R::RowLayout::Dense(points.RowCount(),2));
                auto write = keys.Write();
                for (index iQuery = 0; iQuery < points.RowCount(); ++iQuery)
                {
                    write.View()(iQuery,0) = iQuery;
                    write.View()(iQuery,1) = ids(iQuery,0);
                    const tPoint point{queries(iQuery,0),queries(iQuery,1),queries(iQuery,2)};
                    for (int peer = 0; peer < mpi.size; ++peer)
                        for (int iBox = 0; iBox < count; ++iBox)
                        {
                            SpatialBox box;
                            const int offset = 6*(peer*count+iBox);
                            for (int d = 0; d < 3; ++d)
                            {
                                box.lower[d] = boxes[offset+d];
                                box.upper[d] = boxes[offset+d+3];
                            }
                            if ((box.lower.array() <= box.upper.array()).all() && box.SquaredDistance(point) <= sqr(settings.maxGap))
                            {
                                destinations[size_t(peer)].push_back(iQuery);
                                break; // One query per candidate rank, even if its boxes overlap.
                            }
                        }
                } });
            const auto route = R::SparseBatchExchange::Build(context, points.RowCount(), std::move(destinations));
            candidatePoints = route.Exchange(points);
            candidateKeys = route.Exchange(keys);
            context->LocalStage([&]
                                {
                origins.resize(size_t(candidatePoints.RowCount()));
                for (int peer = 0; peer < mpi.size; ++peer)
                    std::fill(origins.begin()+route.ReceiveOffsets()[peer],origins.begin()+route.ReceiveOffsets()[peer+1],peer); });
        }
        // Phase 3: exact geometry on candidate ranks. Proposal keys identify
        // the origin query and selected owner row; values carry parameters,
        // world/reference projected points, gap and oriented world normal.
        struct Proposal
        {
            std::array<index, 6> keys;
            std::array<real, 13> values;
            int origin;
        };
        std::vector<Proposal> proposals;
        context->LocalStage([&]
                            {
            auto qr = candidatePoints.Read();
            auto kr = candidateKeys.Read();
            const auto q = qr.View<R::FixedRows<3>>();
            const auto key = kr.View<R::FixedRows<2>>();
            auto append = [&](const ElementGeometryView &shape, const RigidPose &pose, index element, index elementId,
                              int geometryRank, index iQuery, const tPoint &referenceQuery)
            {
                const auto p = shape.Project(referenceQuery, settings);
                if (!p.Accepted())
                    return;
                const auto projected = pose.Apply(p.point);
                const tPoint normal = pose.rotation * p.normal;
                const bool interior = shape.ContainsReference(p.reference, -std::max(settings.referenceTolerance, 32 * std::numeric_limits<real>::epsilon()));
                proposals.push_back({{key(iQuery,0),key(iQuery,1),element,elementId,geometryRank,index(interior)},
                    {p.reference[0],p.reference[1],p.reference[2],projected[0],projected[1],projected[2],p.point[0],p.point[1],p.point[2],p.distance,normal[0],normal[1],normal[2]},origins[size_t(iQuery)]});
            };
            if (mode == SurfaceRoutingMode::CoarseBoxes)
            {
                const auto pose = surface.Pose();
                const real gap = settings.maxGap + ElementGeometryView::ReferencePadding(surface._data->diameter, settings.referenceTolerance);
                for (index iQuery = 0; iQuery < candidatePoints.RowCount(); ++iQuery)
                {
                    const tPoint referenceQuery = pose.Inverse({q(iQuery,0),q(iQuery,1),q(iQuery,2)});
                    surface._data->tree->Query(referenceQuery, gap, [&](index iElement)
                    {
                        append(surface.Geometry(iElement), pose, iElement, surface.Element(iElement).id, mpi.rank, iQuery, referenceQuery);
                    });
                }
                return;
            }
            // Dense baseline geometry is replicated in different owner poses;
            // use one temporary world-space index over those received rows.
            auto er = elements.Read();
            auto gr = geometry.Read();
            const auto e = er.View<R::FixedRows<5>>();
            const auto g = gr.View<R::FixedRows<24>>();
            std::vector<SpatialBox> bounds(size_t(elements.RowCount()));
            auto poseAt = [&](index iElement)
            {
                RigidPose pose;
                for (int row = 0; row < 3; ++row)
                {
                    pose.translation[row] = g(iElement,21+row);
                    for (int col = 0; col < 3; ++col)
                        pose.rotation(row,col) = g(iElement,12+3*row+col);
                }
                return pose;
            };
            for (index iElement = 0; iElement < elements.RowCount(); ++iElement)
                bounds[size_t(iElement)] = ElementGeometryView(Elem::ElemType(e(iElement,2)),g[iElement].data(),e(iElement,3)).Bounds(poseAt(iElement),settings.referenceTolerance);
            BoundingBoxTree tree(bounds, surface.Backend());
            for (index iQuery = 0; iQuery < candidatePoints.RowCount(); ++iQuery)
            {
                const tPoint world{q(iQuery,0),q(iQuery,1),q(iQuery,2)};
                tree.Query(world,settings.maxGap,[&](index iElement)
                {
                    const auto pose = poseAt(iElement);
                    const ElementGeometryView shape(Elem::ElemType(e(iElement,2)),g[iElement].data(),e(iElement,3));
                    append(shape, pose, e(iElement,0), e(iElement,1), int(e(iElement,4)), iQuery, pose.Inverse(world));
                });
            } });
        // Phase 4: return proposals and arbitrate on each query origin.
        R::Array<index> proposalKeys;
        R::Array<real> proposalValues;
        std::vector<std::vector<index>> replies;
        context->LocalStage([&]
                            {
            proposalKeys = R::Array<index>(R::RowLayout::Dense(proposals.size(),6));
            proposalValues = R::Array<real>(R::RowLayout::Dense(proposals.size(),13));
            replies.resize(size_t(mpi.size));
            auto k = proposalKeys.Write();
            auto v = proposalValues.Write();
            for (index i = 0; i < index(proposals.size()); ++i)
            {
                std::copy(proposals[size_t(i)].keys.begin(),proposals[size_t(i)].keys.end(),k.View()[i].data());
                std::copy(proposals[size_t(i)].values.begin(),proposals[size_t(i)].values.end(),v.View()[i].data());
                replies[size_t(proposals[size_t(i)].origin)].push_back(i);
            } });
        const auto replyRoute = R::SparseBatchExchange::Build(context, index(proposals.size()), std::move(replies));
        auto returnedKeys = replyRoute.Exchange(proposalKeys);
        auto returnedValues = replyRoute.Exchange(proposalValues);
        std::shared_ptr<Detail::DistributedSurfacePointMapData> data;
        context->LocalStage([&]
                            {
            data = std::make_shared<Detail::DistributedSurfacePointMapData>(surface,mpi,context);
            std::vector<std::vector<SurfaceMatchProposal>> byQuery(size_t(points.RowCount()));
            auto kr = returnedKeys.Read();
            auto vr = returnedValues.Read();
            const auto k = kr.View<R::FixedRows<6>>();
            const auto v = vr.View<R::FixedRows<13>>();
            for (index i = 0; i < returnedKeys.RowCount(); ++i)
            {
                const index iQuery = k(i,0);
                DNDS_check_throw_info(iQuery >= 0 && iQuery < points.RowCount() && k(i,1) == ids(iQuery,0),"surface proposal/query key mismatch");
                byQuery[size_t(iQuery)].push_back({{ids(iQuery,0),k(i,2),k(i,3),int(k(i,4)),{v(i,0),v(i,1),v(i,2)},
                    {v(i,3),v(i,4),v(i,5)},{v(i,6),v(i,7),v(i,8)},{v(i,10),v(i,11),v(i,12)},v(i,9)},k(i,5) != 0});
            }
            data->matches.resize(size_t(points.RowCount()));
            OMP::ForSetupRows(0, points.RowCount(), [&](index iQuery)
            {
                const auto match = ResolveSurfaceMatches(byQuery[size_t(iQuery)],settings.projectionTolerance);
                DNDS_check_throw_info(match,"uncovered distributed surface query " + std::to_string(ids(iQuery,0)));
                data->matches[size_t(iQuery)] = *match;
            }); });
        return DistributedSurfacePointMap(std::move(data));
    }
    index DistributedSurfacePointMap::Size() const { return index(_data->matches.size()); }
    SurfacePointMatch DistributedSurfacePointMap::Get(index iQuery) const
    {
        DNDS_check_throw_info(iQuery >= 0 && iQuery < Size(), "distributed surface query outside map");
        return _data->matches[size_t(iQuery)];
    }
    void DistributedSurfacePointMap::ValidateFor(const SurfacePatch &active) const { DNDS_check_throw_info(active.Same(_data->surface), "active surface snapshot or pose differs from map"); }
    R::RowTransferPlan DistributedSurfacePointMap::BuildElementDataRoute() const
    {
        std::vector<R::RowRequest> requests;
        _data->context->LocalStage([&]
                                   {
            requests.reserve(size_t(Size()));
            for (index iQuery = 0; iQuery < Size(); ++iQuery)
            {
                const auto &match = _data->matches[size_t(iQuery)];
                requests.push_back({match.geometryRank,match.element,iQuery});
            } });
        return R::RowTransferPlan(_data->context, _data->surface.Size(), Size(), requests);
    }
}
