#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include <limits>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace Detail
    {
        struct WallDistanceData
        {
            ssp<Geom::UnstructuredMesh> mesh;
            R::VersionStamp stamp;
            int intOrder;
            real localMinimumCellCenter = std::numeric_limits<real>::infinity();
            real localMinimumFirstLayer = std::numeric_limits<real>::infinity();
            R::Array<real> cells, faces, mean;
            std::optional<R::Array<WallCellDistance>> firstLayerCells;
        };
    }
    namespace
    {
        Geom::MeshReadRequirements Requirements()
        {
            auto requirements = Geom::MeshReadRequirements::CellFlux();
            requirements.wallDistance = true;
            return requirements;
        }
    }
    R::ArrayPair<uint8_t> WallDistanceSamples::BuildFirstLayerMask(
        Geom::UnstructuredMesh &mesh, std::shared_ptr<R::CommunicationContext> communication,
        const std::function<bool(int)> &isWallZone)
    {
        R::ArrayPair<uint8_t> flags;
        communication->LocalStage([&]
                                  {
            auto access = mesh.AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto view = access.HostView();
            flags.owned = R::Array<uint8_t>(R::RowLayout::Dense(view.NumCell(), 1));
            flags.ghost = R::Array<uint8_t>(R::RowLayout::Dense(view.NumCellGhost(), 1));
            auto write = flags.owned.Write();
            const auto marked = write.View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < view.NumCell(); ++iCell)
            {
                const auto faces = view.CellFaces(iCell);
                marked(iCell, 0) = std::any_of(faces.begin(), faces.end(), [&](index iFace)
                                              {
                    const auto zone = view.GetFaceZone(iFace);
                    return Geom::FaceIDIsExternalBC(zone) && isWallZone(zone); });
            } });
        const auto route = mesh.Entities(Geom::EntityKind::Cell).BuildRowTransferPlan(communication);
        R::PayloadPlan(route, flags.owned.Layout(), flags.ghost.Layout(), R::ElementSchema::For<uint8_t>())
            .Bind(flags.owned, flags.ghost, R::TransferTransport::Direct,
                  {R::TransferMode::Transient, communication})
            .Begin()
            .Wait();
        return flags;
    }
    WallDistanceSamples WallDistanceSamples::Construct(ssp<Geom::UnstructuredMesh> mesh, int intOrder,
                                                       real minimumDistance,
                                                       OptionalRef<const R::ArrayPair<uint8_t>> firstLayer,
                                                       FirstLayerWallDistance method, bool retainFirstLayer, bool replaceSamples,
                                                       bool centerQuadratureSamples)
    {
        DNDS_check_throw_info(mesh && intOrder >= 1 && intOrder <= Geom::Elem::INT_ORDER_MAX &&
                                  std::isfinite(minimumDistance) && minimumDistance >= 0,
                              "invalid wall-distance sampling setup");
        auto read = mesh->AcquireRead(Requirements());
        const auto view = read.HostView();
        std::optional<R::ArrayPair<uint8_t>::ReadLease> firstLayerRead;
        if (firstLayer)
        {
            DNDS_check_throw_info(firstLayer->owned.RowCount() == view.NumCell() &&
                                      firstLayer->ghost.RowCount() == view.NumCellGhost(),
                                  "wall-distance first-layer mask is not aligned to cell rows");
            firstLayerRead.emplace(firstLayer->Read());
        }
        using MaskView = R::PairReadView<uint8_t, R::AccessBoundsChecks, R::FixedRows<1>>;
        const MaskView selected = firstLayerRead ? firstLayerRead->View<R::FixedRows<1>>() : MaskView{};
        auto data = std::make_shared<Detail::WallDistanceData>();
        data->mesh = std::move(mesh);
        data->stamp = data->mesh->versions.Stamp(R::Change::All);
        data->intOrder = intOrder;
        std::vector<size_t> cellSizes(size_t(view.NumCellProc())), faceSizes(size_t(view.NumFaceProc()));
        for (index iCell = 0; iCell < view.NumCellProc(); ++iCell)
            cellSizes[size_t(iCell)] = Geom::Elem::Quadrature(view.GetCellElement(iCell), intOrder).GetNumPoints();
        for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
            faceSizes[size_t(iFace)] = Geom::Elem::Quadrature(view.GetFaceElement(iFace), intOrder).GetNumPoints();
        // Allocate every extent before filling, with only local and ghost rows.
        data->cells = R::Array<real>(R::RowLayout::Ragged(cellSizes));
        data->faces = R::Array<real>(R::RowLayout::Ragged(faceSizes));
        data->mean = R::Array<real>(R::RowLayout::Dense(view.NumCellProc(), 1));
        data->cells.SetName("Cauchy/wallDistance/cellQuadrature", DNDS_ALLOCATION_SITE());
        data->faces.SetName("Cauchy/wallDistance/faceQuadrature", DNDS_ALLOCATION_SITE());
        data->mean.SetName("Cauchy/wallDistance/cellMean", DNDS_ALLOCATION_SITE());
        auto cellWrite = data->cells.Write(), faceWrite = data->faces.Write(), meanWrite = data->mean.Write();
        const auto cells = cellWrite.View<R::CSRRows>(), faces = faceWrite.View<R::CSRRows>();
        const auto mean = meanWrite.View<R::FixedRows<1>>();
        std::vector<WallCellDistance> marked;
        const auto fill = [&](const auto &nodes, Geom::Elem::Element element, auto row, OptionalRef<real> center, OptionalRef<real> firstDistance)
        {
            // At most one element's nodal scratch; reuse the shared reference
            // shape cache rather than caching shape matrices per mesh entity.
            std::array<real, Geom::Elem::CellNumNodeMax> distances{};
            for (index iNode = 0; iNode < index(nodes.size()); ++iNode)
            {
                const real distance = view.NodeWallDist(nodes[iNode]).norm();
                DNDS_check_throw_info(std::isfinite(distance), "nonfinite nodal wall distance");
                distances[size_t(iNode)] = distance;
            }
            const auto sample = [&](int order, index iPoint)
            {
                const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iPoint);
                real distance = 0;
                for (index iNode = 0; iNode < index(nodes.size()); ++iNode)
                    distance += shape(0, iNode) * distances[size_t(iNode)];
                DNDS_check_throw_info(std::isfinite(distance), "nonfinite sampled wall distance");
                return distance;
            };
            // Order-one volume quadrature uses the same reference center as
            // FiniteVolumePlan::GetCellCent, including higher-order geometry.
            if (center)
                *center = sample(1, 0);
            const bool uniformVolume = center && centerQuadratureSamples;
            if (firstDistance)
            {
                const real rawDistance = method == FirstLayerWallDistance::HalfMaxNodal
                                             ? .5 * *std::max_element(distances.begin(), distances.end())
                                             : *center;
                const real distance = std::max(minimumDistance, rawDistance);
                *firstDistance = distance;
                if (replaceSamples)
                {
                    std::fill(row.begin(), row.end(), uniformVolume ? std::max(minimumDistance, *center) : distance);
                    return distance;
                }
            }
            if (uniformVolume)
            {
                const real distance = std::max(minimumDistance, *center);
                std::fill(row.begin(), row.end(), distance);
                return distance;
            }
            real sum = 0;
            for (index iPoint = 0; iPoint < index(row.size()); ++iPoint)
            {
                const real distance = center && intOrder == 1 && iPoint == 0 ? *center : sample(intOrder, iPoint);
                row[iPoint] = std::max(minimumDistance, distance);
                sum += row[iPoint];
            }
            return std::max(minimumDistance, sum / real(row.size()));
        };
        for (index iCell = 0; iCell < view.NumCellProc(); ++iCell)
        {
            real center = 0, firstDistance = 0;
            const bool replace = firstLayerRead && selected(iCell, 0);
            mean(iCell, 0) = fill(view.CellNodes(iCell), view.GetCellElement(iCell), cells[iCell], std::ref(center), replace ? OptionalRef<real>(firstDistance) : OptionalRef<real>{});
            if (iCell < view.NumCell())
            {
                data->localMinimumCellCenter = std::min(data->localMinimumCellCenter, center);
                if (replace)
                {
                    data->localMinimumFirstLayer = std::min(data->localMinimumFirstLayer, firstDistance);
                    if (retainFirstLayer)
                    {
                        DNDS_check_throw_info(firstDistance > 0, "resolved cell omega requires positive first-layer wall distance");
                        marked.push_back({iCell, firstDistance});
                    }
                }
            }
        }
        for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
        {
            auto row = faces[iFace];
            if (centerQuadratureSamples)
            {
                const auto parents = view.FaceCells(iFace);
                const index left = parents[0], right = parents[1];
                // CellFlux guarantees local parents on all flux faces. Extra
                // outer ghost faces may have a parent beyond the cell halo.
                if (left >= 0 && (right >= 0 || right == UnInitIndex))
                {
                    const real leftCenter = cells(left, 0);
                    const real distance = right == UnInitIndex ? leftCenter : .5 * leftCenter + .5 * cells(right, 0);
                    std::fill(row.begin(), row.end(), distance);
                    continue;
                }
            }
            fill(view.FaceNodes(iFace), view.GetFaceElement(iFace), row, {}, {});
        }
        if (retainFirstLayer)
        {
            data->firstLayerCells = R::Array<WallCellDistance>(R::RowLayout::Dense(index(marked.size()), 1));
            data->firstLayerCells->SetName("Cauchy/wallDistance/firstLayerCells", DNDS_ALLOCATION_SITE());
            auto write = data->firstLayerCells->Write();
            const auto rows = write.View<R::FixedRows<1>>();
            for (index i = 0; i < rows.RowCount(); ++i)
                rows(i, 0) = marked[i];
        }
        return WallDistanceSamples(std::move(data));
    }
    const R::Array<WallCellDistance> &WallDistanceSamples::FirstLayerCells() const
    {
        _data->stamp.Validate();
        return _data->firstLayerCells.value();
    }
    real WallDistanceSamples::LocalMinimumCellCenterDistance() const
    {
        _data->stamp.Validate();
        return _data->localMinimumCellCenter;
    }
    real WallDistanceSamples::LocalMinimumFirstLayerDistance() const
    {
        _data->stamp.Validate();
        return _data->localMinimumFirstLayer;
    }
    const R::Array<real> &WallDistanceSamples::CellValues() const
    {
        _data->stamp.Validate();
        return _data->cells;
    }
    const R::Array<real> &WallDistanceSamples::FaceValues() const
    {
        _data->stamp.Validate();
        return _data->faces;
    }
    const R::Array<real> &WallDistanceSamples::CellMeans() const
    {
        _data->stamp.Validate();
        return _data->mean;
    }
    WallDistanceRead::WallDistanceRead(std::shared_ptr<const Detail::WallDistanceData> data, R::ExecutionContext context)
        : _data(std::move(data)), _mesh(_data->mesh->AcquireRead(Requirements())),
          _cells(_data->cells.Read(context)), _faces(_data->faces.Read(context)), _mean(_data->mean.Read(context))
    {
        // Pin first, then check the cache stamp: mutation cannot race this check.
        _data->stamp.Validate();
    }
    WallDistanceView WallDistanceRead::HostView() const
    {
        DNDS_check_throw_info(_data->cells.GetDevice() == R::Device::Host(), "host wall-distance view requires host placement");
        return View();
    }
    WallDistanceView WallDistanceRead::View() const
    {
        return {_cells.View<R::CSRRows>(), _faces.View<R::CSRRows>(), _mean.View<R::FixedRows<1>>()};
    }
    WallDistanceRead WallDistanceSamples::AcquireRead(R::ExecutionContext context) const { return WallDistanceRead(_data, context); }
    R::Device WallDistanceSamples::Device() const { return _data->cells.GetDevice(); }
    WallDistanceSamples WallDistanceSamples::Placed(R::Device device, R::ExecutionContext context) const
    {
        const auto guard = _data->mesh->AcquireRead(Requirements());
        _data->stamp.Validate();
        if (device == Device())
            return *this;
        auto copy = std::make_shared<Detail::WallDistanceData>(*_data);
        copy->cells = _data->cells.To(device, context);
        copy->faces = _data->faces.To(device, context);
        if (_data->firstLayerCells)
            copy->firstLayerCells = _data->firstLayerCells->To(device, context);
        copy->mean = _data->mean.To(device, context);
        return WallDistanceSamples(std::move(copy));
    }
    int WallDistanceSamples::IntegrationOrder() const { return _data->intOrder; }
}
