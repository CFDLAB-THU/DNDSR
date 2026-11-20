#include "Cauchy/IO/PreparedVisualization.hpp"
#include "Cauchy/IO/OutputFields.hpp"
#include "Cauchy/IO/BoundaryOutputPlan.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;

    struct PreparedVisualization::Data
    {
        std::shared_ptr<Geom::UnstructuredMesh> mesh;
        Geom::MeshReadAccess read;
        OutputFields fields;
        std::vector<std::string> names;
        std::shared_ptr<const BoundaryOutputGeometry> boundary;
        std::optional<R::Array<real>> flux;
        std::vector<std::string> fluxNames;
        R::ExecutionContext copy;
        real time;
        Geom::UnstructuredMesh::HDF5OutSetting settings;
        static auto Requirements(const Geom::UnstructuredMesh &mesh)
        {
            auto request = Geom::MeshReadRequirements::CellFlux();
            request.wallDistance = mesh.HasNodeWallDist();
            return request;
        }
        Data(std::shared_ptr<Geom::UnstructuredMesh> geometry, OutputFields values,
             std::vector<std::string> labels, std::shared_ptr<const BoundaryOutputGeometry> surface,
             std::optional<R::Array<real>> fluxValues, std::vector<std::string> fluxLabels, real when, int deflate)
            : mesh(std::move(geometry)), read(mesh->AcquireRead(Requirements(*mesh))),
              fields(std::move(values)), names(std::move(labels)), boundary(std::move(surface)),
              flux(std::move(fluxValues)), fluxNames(std::move(fluxLabels)), copy{fields.Device(), nullptr}, time(when), settings(mesh->hdf5OutSetting)
        {
            settings.deflateLevel = deflate;
            DNDS_check_throw_info(!flux || (flux->GetDevice() == fields.Device() && flux->RowCount() == boundary->rows.RowCount() &&
                                            flux->Layout().Width() == index(fluxNames.size()) + 3),
                                  "visualization face flux differs from the frozen boundary layout");
        }
    };

    PreparedVisualization::PreparedVisualization(std::shared_ptr<Geom::UnstructuredMesh> mesh, OutputFields fields,
                                                 std::vector<std::string> names, std::shared_ptr<const BoundaryOutputGeometry> boundary,
                                                 std::optional<R::Array<real>> flux, std::vector<std::string> fluxNames, real time, int deflate)
        : _data(std::make_shared<Data>(std::move(mesh), std::move(fields), std::move(names), std::move(boundary),
                                       std::move(flux), std::move(fluxNames), time, deflate))
    {
    }

    void PreparedVisualization::BindWriterCopy(R::ExecutionContext copy)
    {
        DNDS_check_throw_info(_data.unique() && copy.device == _data->fields.Device(), "bind the writer stream before sharing a visualization snapshot");
        _data->copy = std::move(copy);
    }

    void PreparedVisualization::Write(const std::string &prefix, std::shared_ptr<R::CommunicationContext> communication) const
    {
        DNDS_check_throw_info(communication, "visualization writer needs a retained I/O communicator");
        const auto &data = *_data;
        const auto comm = communication->Communicator();
        const auto &copy = data.copy;
        R::Array<real> volume, surface;
        const auto &boundary = *data.boundary;
        const auto meshView = data.read.HostView();
        const bool exportDistance = meshView.HasNodeWallDist();
        const index nFields = index(data.names.size()) + index(exportDistance);
        const index nFlux = index(data.fluxNames.size());
        MPI::LocalStage(MPIInfo(comm), [&]
                        {
            volume = data.fields.CopyToHost(copy);
            auto fieldsRead = volume.Read();
            const auto fields = fieldsRead.View<R::TableRows>();
            DNDS_check_throw_info(fields.RowCount() == data.mesh->NumCell() && fields.Width() == index(data.names.size()), "visualization volume field shape differs");
            std::optional<R::Array<real>> hostFlux;
            std::optional<R::Array<real>::ReadLease> fluxRead;
            if (data.flux)
            {
                hostFlux.emplace(data.flux->GetDevice() == R::Device::Host() ? *data.flux : data.flux->To(R::Device::Host(), copy));
                fluxRead.emplace(hostFlux->Read());
            }
            const auto flux = fluxRead ? fluxRead->View<R::TableRows>() : decltype(fields){};
            surface = R::Array<real>(R::RowLayout::Dense(boundary.rows.RowCount(), nFields + (fluxRead ? nFlux + 3 : 0)));
            surface.SetName("Cauchy/output/boundaryParentRows", DNDS_ALLOCATION_SITE());
            auto rowsRead = boundary.rows.Read();
            const auto rows = rowsRead.View<R::FixedRows<1>>();
            auto write = surface.Write();
            const auto output = write.View<R::TableRows>();
            for (index iRow = 0; iRow < output.RowCount(); ++iRow)
            {
                const index cell = rows(iRow, 0).cell;
                for (index j = 0; j < fields.Width(); ++j)
                    output(iRow, j) = fields(cell, j);
                if (exportDistance)
                {
                    // Geometric center sample, before any SST floor or first-layer override.
                    const auto nodes = meshView.CellNodes(cell);
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(meshView.GetCellElement(cell).type, 1, 0);
                    real distance = 0;
                    for (index j = 0; j < index(nodes.size()); ++j)
                        distance += shape(0, j) * meshView.NodeWallDist(nodes[j]).norm();
                    output(iRow, fields.Width()) = distance;
                }
                if (fluxRead)
                    for (index j = 0; j < nFlux + 3; ++j)
                        output(iRow, nFields + j) = flux(iRow, j);
            } });
        surface = boundary.InFileOrder(std::move(surface), communication);
        const auto read = volume.Read();
        const auto values = read.View<R::TableRows>();
        const auto name = [&](int i)
        { return data.names.at(i); };
        const auto value = [&](int i, index iCell)
        { return values(iCell, i); };
        data.mesh->PrintParallelVTKHDFDataArray(prefix, "", int(data.names.size()), 0, 0, 0,
                                                name, value, {}, {}, {}, {}, {}, {}, data.time, comm, &data.settings);

        const auto faceRead = surface.Read(), normalRead = boundary.normals.Read();
        const auto faces = faceRead.View<R::TableRows>();
        const auto normals = normalRead.View<R::FixedRows<3>>();

        Geom::UnstructuredMesh::VTKHDFExtraFields extra;
        extra.cellIntegers.push_back({"bnd_zone_id", {boundary.zones->data(), index(boundary.zones->size())}});
        extra.fieldIntegers.push_back({"bnd_zone_ids", {}});
        extra.fieldStrings.push_back({"bnd_zone_names", {}});
        for (const auto &[id, label] : *boundary.names)
        {
            extra.fieldIntegers.front().second.push_back(id);
            extra.fieldStrings.front().second.push_back(label);
        }
        const auto scalarName = [&](int i)
        { if (i >= nFields) return data.fluxNames.at(i - nFields);
          if (i == int(data.names.size())) return std::string("wall_distance_cell_center");
          return data.names.at(i); };
        const auto scalarValue = [&](int i, index iFace)
        { return faces(iFace, i); };
        const auto vectorName = [](int i) -> std::string
        { return i == 0 ? "norm" : "force_tangential"; };
        const auto vectorValue = [&](int i, index iFace, int d)
        { return i == 0 ? normals(iFace, d) : faces(iFace, nFields + nFlux + d); };
        boundary.mesh->PrintParallelVTKHDFDataArray(prefix + "_bnd", "", int(nFields + nFlux), data.flux ? 2 : 1, 0, 0,
                                                    scalarName, scalarValue, vectorName, vectorValue,
                                                    {}, {}, {}, {}, data.time, comm, &data.settings, extra);
    }
}
