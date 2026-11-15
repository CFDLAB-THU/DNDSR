#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

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
            R::Array<real> cells, faces, mean;
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
    WallDistanceSamples WallDistanceSamples::Construct(ssp<Geom::UnstructuredMesh> mesh, int intOrder,
                                                       real minimumDistance)
    {
        DNDS_check_throw_info(mesh && intOrder >= 1 && intOrder <= Geom::Elem::INT_ORDER_MAX &&
                                  std::isfinite(minimumDistance) && minimumDistance >= 0,
                              "invalid wall-distance sampling setup");
        auto read = mesh->AcquireRead(Requirements());
        const auto view = read.HostView();
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
        auto cellWrite = data->cells.Write(), faceWrite = data->faces.Write(), meanWrite = data->mean.Write();
        const auto cells = cellWrite.View<R::CSRRows>(), faces = faceWrite.View<R::CSRRows>();
        const auto mean = meanWrite.View<R::FixedRows<1>>();
        const auto fill = [&](const auto &nodes, Geom::Elem::Element element, auto row)
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
            real sum = 0;
            for (index iPoint = 0; iPoint < index(row.size()); ++iPoint)
            {
                const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, intOrder, iPoint);
                real distance = 0;
                for (index iNode = 0; iNode < index(nodes.size()); ++iNode)
                    distance += shape(0, iNode) * distances[size_t(iNode)];
                DNDS_check_throw_info(std::isfinite(distance), "nonfinite sampled wall distance");
                row[iPoint] = std::max(minimumDistance, distance);
                sum += row[iPoint];
            }
            return sum / real(row.size());
        };
        for (index iCell = 0; iCell < view.NumCellProc(); ++iCell)
            mean(iCell, 0) = fill(view.CellNodes(iCell), view.GetCellElement(iCell), cells[iCell]);
        for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
            fill(view.FaceNodes(iFace), view.GetFaceElement(iFace), faces[iFace]);
        return WallDistanceSamples(std::move(data));
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
        copy->mean = _data->mean.To(device, context);
        return WallDistanceSamples(std::move(copy));
    }
    int WallDistanceSamples::IntegrationOrder() const { return _data->intOrder; }
}
