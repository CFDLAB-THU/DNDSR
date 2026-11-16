#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Geom
{
    namespace
    {
        // Coordinate edits are collective setup operations, not kernel reads.
        // Agree the selected mode before optional validation; required option
        // broadcasts, mutation epochs and numerical decisions remain separate.
        std::shared_ptr<Runtime::CommunicationContext> CheckGeometryEdit(UnstructuredMesh &mesh)
        {
            auto communication = mesh.PreparationCommunication();
            const bool checks = !communication || communication->ChecksCollectivePreflight();
            Runtime::CommunicationDetail::AgreePolicy(mesh.mpi.comm, checks);
            mesh.CheckCollectiveMutationAvailable(checks);
            return communication;
        }
        template <class F>
        Runtime::ArrayPair<real> GeometryRows(const UnstructuredMeshView &view, F &&value)
        {
            Runtime::ArrayPair<real> result{
                Runtime::Array<real>(Runtime::RowLayout::Dense(size_t(view.NumNode()), 3)),
                Runtime::Array<real>(Runtime::RowLayout::Dense(size_t(view.NumNodeGhost()), 3))};
            auto owned = result.owned.Write(), ghost = result.ghost.Write();
            // Internal callers supply immutable affine/mask operations. The
            // arbitrary user callback entry below deliberately stays serial.
            OMP::ForSetupRows(0, view.NumNodeProc(), [&](index node)
                              {
                const tPoint point = value(node);
                auto output = node < view.NumNode() ? owned.View()[size_t(node)] : ghost.View()[size_t(node - view.NumNode())];
                for (size_t component = 0; component < 3; ++component)
                    output[component] = point[component]; });
            return result;
        }
    }

    void UnstructuredMesh::PublishNativeGeometry(std::shared_ptr<const UnstructuredMeshState> prepared,
                                                 Runtime::VersionStamp sourceStamp, Runtime::Change changes)
    {
        const auto communication = prepared ? prepared->PreparationCommunication() : PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        std::shared_ptr<UnstructuredMeshState> next;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            sourceStamp.Validate();
            DNDS_check_throw_info(_nativeState && prepared, "native geometry publication requires source and destination");
            prepared->Validate();
            next = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*prepared)); });
        MeshMutationToken mutation(this, versions, changes);
        mutation.Commit();
        // No fallible array construction remains after committing generations.
        next->BindPreparedEpoch(versions);
        _nativeState = std::move(next);
        _nativeSmoothing.reset();
    }

    void UnstructuredMesh::TransformNativeCoords(const std::function<tPoint(const tPoint &)> &transform)
    {
        // Arbitrary callbacks may carry mutable user state and are invoked in
        // owned-node order. Built-in affine transforms use GeometryRows teams.
        const auto communication = CheckGeometryEdit(*this);
        const auto stamp = versions.Stamp();
        std::shared_ptr<const UnstructuredMeshState> prepared;
        {
            std::optional<MeshReadAccess> sourceRead;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { sourceRead.emplace(AcquireRead(_nativeState->Requirements())); });
            const auto view = sourceRead->HostView();
            std::optional<Runtime::Array<real>> coordinates;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                DNDS_check_throw_info(!view.IsPeriodic(), "arbitrary native coordinate transforms require nonperiodic topology; use ApplyGeometryTransform for periodic rigid transforms");
                coordinates.emplace(Runtime::RowLayout::Dense(size_t(view.NumNode()), 3));
                auto write = coordinates->Write();
                for (index node = 0; node < view.NumNode(); ++node)
                {
                    const auto point = transform(view.Coord(node));
                    DNDS_check_throw_info(point.allFinite(), "native coordinate callback produced nonfinite values");
                    for (size_t component = 0; component < 3; ++component) write.View()(size_t(node), component) = point[component];
                } });
            const auto nodes = Entities(EntityKind::Node);
            EntityGhostMapping route(nodes);
            auto ghost = route.TransferRows(nodes, *coordinates, Runtime::ElementSchema::Arithmetic<real>());
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                ghost = Runtime::Array<real>::FromStorage(Runtime::RowLayout::Dense(ghost.RowCount(), 3), ghost.Storage());
                prepared = _nativeState->WithGeometry({std::move(*coordinates), std::move(ghost)}, {}, view.Periodic()); });
        }
        PublishNativeGeometry(std::move(prepared), stamp);
    }

    void UnstructuredMesh::ApplyGeometryTransform(const Eigen::Matrix3d &rotation, real scale)
    {
        const auto communication = CheckGeometryEdit(*this);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(HasNativeState(), "geometry transform requires native mesh authority");
                DNDS_check_throw_info(rotation.allFinite() && std::isfinite(scale) && scale > 0 &&
                                                                    (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-10 &&
                                                                    std::abs(rotation.determinant() - 1.0) < 1e-10,
                                                                "geometry transform requires a proper rotation and positive scale"); });
        // Transforming every owned/ghost row locally is valid only if all
        // ranks use the same affine map. Detect disagreement before writes.
        std::array<real, 10> parameters{};
        std::copy(rotation.data(), rotation.data() + 9, parameters.begin());
        parameters[9] = scale;
        auto expected = parameters;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_DOUBLE, 0, mpi.comm);
        std::shared_ptr<const UnstructuredMeshState> prepared;
        const auto stamp = versions.Stamp();
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            DNDS_check_throw_info(parameters == expected, "native geometry transform differs across ranks");
            const auto view = _nativeState->BorrowHostView();
            const bool rotate = !(rotation.array() == Eigen::Matrix3d::Identity().array()).all();
            auto transform = [&](const tPoint &point) -> tPoint
            {
                // Preserve the legacy pure scale/rotation operation order.
                // Identity matrix products can erase signed zero; a combined
                // transform explicitly rotates first, then scales the result.
                const tPoint rotated = rotate ? tPoint((rotation * point).eval()) : point;
                return scale == 1.0 ? rotated : tPoint((scale * rotated).eval());
            };
            auto coordinates = GeometryRows(view, [&](index node) { return transform(view.Coord(node)); });
            std::optional<Runtime::ArrayPair<real>> walls;
            if (view.HasNodeWallDist())
                walls = GeometryRows(view, [&](index node) { return transform(view.NodeWallDist(node)); });
            auto frame = view.Periodic();
            if (rotate)
                for (auto &r : frame.rotation) r.map() = (rotation * r.map() * rotation.transpose()).eval();
            for (auto &center : frame.rotationCenter) center.map() = transform(center.map());
            for (auto &translation : frame.translation) translation.map() = transform(translation.map());
            prepared = _nativeState->WithGeometry(std::move(coordinates), std::move(walls), frame); });
        PublishNativeGeometry(std::move(prepared), stamp);
    }

    void UnstructuredMesh::ApplyAffineGeometryTransform(const Eigen::Matrix3d &linear, const tPoint &offset)
    {
        const auto communication = CheckGeometryEdit(*this);
        Eigen::Matrix3d inverse;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            DNDS_check_throw_info(HasNativeState(), "affine geometry transform requires native mesh authority");
            DNDS_check_throw_info(linear.allFinite() && offset.allFinite(), "affine geometry transform must be finite");
            const auto determinant = linear.determinant();
            DNDS_check_throw_info(std::isfinite(determinant) && determinant > 0,
                                  "affine geometry transform must be invertible and orientation preserving");
            inverse = linear.inverse();
            DNDS_check_throw_info(inverse.allFinite(), "affine geometry inverse is not finite"); });
        std::array<real, 12> parameters{};
        std::copy(linear.data(), linear.data() + 9, parameters.begin());
        std::copy(offset.data(), offset.data() + 3, parameters.begin() + 9);
        auto expected = parameters;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_DOUBLE, 0, mpi.comm);
        const auto stamp = versions.Stamp();
        std::shared_ptr<const UnstructuredMeshState> prepared;
        {
            // Keep source geometry, topology and metadata alive through staging;
            // publication occurs only after this operation releases its read.
            std::unique_ptr<MeshReadAccess> sourceRead;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                DNDS_check_throw_info(parameters == expected, "affine geometry transform differs across ranks");
                sourceRead = std::make_unique<MeshReadAccess>(AcquireRead(_nativeState->Requirements()));
                const auto view = sourceRead->HostView();
                auto frame = view.Periodic();
                for (size_t image = 0; image < frame.rotation.size(); ++image)
                {
                    const Eigen::Matrix3d rotation = linear * frame.rotation[image].map() * inverse;
                    DNDS_check_throw_info(rotation.allFinite() &&
                                              (!view.IsPeriodic() ||
                                               ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-10 &&
                                                std::abs(rotation.determinant() - 1.0) < 1e-10)),
                                          "affine geometry transform makes a periodic rotation nonrigid");
                    frame.rotation[image].map() = rotation;
                    frame.rotationCenter[image].map() = (linear * frame.rotationCenter[image].map() + offset).eval();
                    frame.translation[image].map() = (linear * frame.translation[image].map()).eval();
                }
                for (size_t image = 0; image < frame.rotation.size(); ++image)
                    DNDS_check_throw_info(frame.translation[image].map().allFinite() && frame.rotationCenter[image].map().allFinite(),
                                          "affine geometry frame overflowed");
                auto coordinates = GeometryRows(view, [&](index node)
                {
                    const tPoint point = linear * view.Coord(node) + offset;
                    DNDS_check_throw_info(point.allFinite(), "affine geometry coordinate overflowed");
                    return point;
                });
                // General affine maps do not preserve nearest-wall directions or
                // distances. Recompute wall geometry after this publication.
                prepared = _nativeState->WithGeometry(std::move(coordinates), {}, frame); });
        }
        PublishNativeGeometry(std::move(prepared), stamp);
    }

    void UnstructuredMesh::RectifyNearPlaneCoordinates(int components, real threshold)
    {
        // The exact changed flag controls collective publication. Its ordered
        // detection/fill stays serial; there are no expensive point operations.
        const auto communication = CheckGeometryEdit(*this);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            DNDS_check_throw_info(HasNativeState(), "near-plane rectification requires native mesh authority");
            DNDS_check_throw_info(components >= 0 && components <= 7 && std::isfinite(threshold) && threshold >= 0,
                                  "near-plane rectification requires component mask 0..7 and finite nonnegative threshold"); });
        int expectedComponents = components;
        real expectedThreshold = threshold;
        MPI_Bcast(&expectedComponents, 1, MPI_INT, 0, mpi.comm);
        MPI_Bcast(&expectedThreshold, 1, MPI_DOUBLE, 0, mpi.comm);
        const auto stamp = versions.Stamp();
        std::shared_ptr<const UnstructuredMeshState> prepared;
        {
            std::optional<MeshReadAccess> sourceRead;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { sourceRead.emplace(AcquireRead(_nativeState->Requirements())); });
            const auto view = sourceRead->HostView();
            std::optional<Runtime::Array<real>> coordinates;
            int changed = 0;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                DNDS_check_throw_info(components == expectedComponents && threshold == expectedThreshold,
                                      "near-plane rectification options differ across ranks");
                coordinates.emplace(Runtime::RowLayout::Dense(size_t(view.NumNode()), 3));
                auto write = coordinates->Write();
                for (index node = 0; node < view.NumNode(); ++node)
                    for (size_t component = 0; component < 3; ++component)
                    {
                        real value = view.Coord(node)[component];
                        if ((components & (1 << component)) && std::abs(value) < threshold)
                        {
                            changed |= value != 0;
                            value = 0;
                        }
                        write.View()(size_t(node), component) = value;
                    } });
            MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, mpi.comm);
            if (!changed)
                return;
            // Canonical coordinates have one authority per node. Images are
            // evaluated using the unchanged PBI/frame after owner-row cleanup.
            const auto nodes = Entities(EntityKind::Node);
            EntityGhostMapping route(nodes);
            auto ghost = route.TransferRows(nodes, *coordinates, Runtime::ElementSchema::Arithmetic<real>());
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                ghost = Runtime::Array<real>::FromStorage(Runtime::RowLayout::Dense(ghost.RowCount(), 3), ghost.Storage());
                prepared = _nativeState->WithGeometry({std::move(*coordinates), std::move(ghost)}, {}, view.Periodic()); });
        }
        PublishNativeGeometry(std::move(prepared), stamp);
    }

    void UnstructuredMesh::RectifyBoundaryCoordinates(std::function<int(t_index)> componentForZone,
                                                      bool volumeBoundary)
    {
        // Zone callbacks plus repeated-node bit-mask scatter are controller
        // work. Masks are exchanged before the independent GeometryRows fill.
        const auto communication = CheckGeometryEdit(*this);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(HasNativeState(), "boundary rectification requires native mesh authority"); });
        const auto stamp = versions.Stamp();
        const auto nodes = Entities(EntityKind::Node);
        std::optional<Runtime::Array<uint8_t>> owned, ghost;
        std::shared_ptr<const UnstructuredMeshState> prepared;
        {
            std::optional<MeshReadAccess> sourceRead;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { sourceRead.emplace(AcquireRead(_nativeState->Requirements())); });
            const auto view = sourceRead->HostView();
            int selectedMode = volumeBoundary ? 1 : 0, expectedMode = selectedMode;
            int selected = 0;
            MPI_Bcast(&expectedMode, 1, MPI_INT, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                DNDS_check_throw_info(selectedMode == expectedMode, "native rectification mode differs across ranks");
                owned.emplace(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 1));
                ghost.emplace(Runtime::RowLayout::Dense(size_t(nodes.NumGhost()), 1));
                auto local = owned->Write(), remote = ghost->Write();
                for (size_t row = 0; row < owned->RowCount(); ++row) local.View()(row, 0) = 0;
                for (size_t row = 0; row < ghost->RowCount(); ++row) remote.View()(row, 0) = 0;
                const auto count = volumeBoundary ? view.NumBnd() : view.NumCell();
                for (index row = 0; row < count; ++row)
                {
                    const auto zone = volumeBoundary ? view.GetBndInfo(row).zone : view.GetCellInfo(row).zone;
                    const auto component = componentForZone(zone);
                    DNDS_check_throw_info(component >= 0 && component <= 3, "rectification component must be zero or one through three");
                    if (!component) continue;
                    selected = 1;
                    const auto cone = volumeBoundary ? view.BoundaryNodes(row) : view.CellNodes(row);
                    for (auto node : cone)
                    {
                        DNDS_check_throw_info(node >= 0 && node < nodes.NumLocal(), "native rectification node is unavailable");
                        auto &value = node < nodes.NumOwned() ? local.View()(size_t(node), 0) : remote.View()(size_t(node - nodes.NumOwned()), 0);
                        value |= uint8_t(1u << (component - 1));
                    }
                } });
            // Preserve callback validation on every boundary, but a globally
            // empty selection has no masks to communicate or geometry to edit.
            MPI_Allreduce(MPI_IN_PLACE, &selected, 1, MPI_INT, MPI_MAX, mpi.comm);
            if (!selected)
                return;
            const auto route = nodes.BuildRowTransferPlan();
            Runtime::PayloadPlan reverse(route.Reverse(), ghost->Layout(), owned->Layout(), Runtime::ElementSchema::Arithmetic<uint8_t>());
            reverse.Bind(*ghost, *owned, Runtime::Reducer::Builtin<uint8_t>(Runtime::Reduction::BitOr),
                         Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient})
                .Begin()
                .Wait();
            Runtime::PayloadPlan forward(route, owned->Layout(), ghost->Layout(), Runtime::ElementSchema::Arithmetic<uint8_t>());
            forward.Bind(*owned, *ghost, Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient}).Begin().Wait();
            int changed = 0;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                auto local = owned->Read(), remote = ghost->Read();
                for (index node = 0; node < view.NumNodeProc(); ++node)
                {
                    const auto mask = node < nodes.NumOwned() ? local.View()(size_t(node), 0) : remote.View()(size_t(node - nodes.NumOwned()), 0);
                    for (size_t component = 0; component < 3; ++component)
                        if (mask & (1u << component))
                            changed |= view.Coord(node)[component] != 0;
                } });
            MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, mpi.comm);
            if (!changed)
                return;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                auto local = owned->Read(), remote = ghost->Read();
                auto coordinates = GeometryRows(view, [&](index node)
                {
                    tPoint point = view.Coord(node);
                    const auto mask = node < nodes.NumOwned() ? local.View()(size_t(node), 0) : remote.View()(size_t(node - nodes.NumOwned()), 0);
                    for (size_t component = 0; component < 3; ++component)
                        if (mask & (1u << component))
                        {
                            point[component] = 0;
                        }
                    return point;
                });
                prepared = _nativeState->WithGeometry(std::move(coordinates), {}, view.Periodic()); });
        }
        PublishNativeGeometry(std::move(prepared), stamp);
    }
}
