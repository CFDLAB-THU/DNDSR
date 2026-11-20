#include "Cauchy/IO/PreparedVisualization.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;

    struct PreparedVisualization::Data
    {
        std::shared_ptr<Geom::UnstructuredMesh> mesh;
        Geom::MeshReadAccess read;
        R::Array<real> fields;
        std::vector<std::string> names;
        Boundary boundary;
        real time;
        Geom::UnstructuredMesh::HDF5OutSetting settings;
        Data(std::shared_ptr<Geom::UnstructuredMesh> geometry, R::Array<real> values,
             std::vector<std::string> labels, Boundary surface, real when, int deflate)
            : mesh(std::move(geometry)),
              read(mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux())),
              fields(std::move(values)), names(std::move(labels)), boundary(std::move(surface)),
              time(when), settings(mesh->hdf5OutSetting)
        {
            settings.deflateLevel = deflate;
            const index faces = boundary.mesh->NumCell();
            DNDS_check_throw_info(fields.GetDevice() == R::Device::Host() &&
                                      fields.RowCount() == mesh->NumCell() && fields.Layout().Width() == index(names.size()) &&
                                      boundary.fields.GetDevice() == R::Device::Host() &&
                                      boundary.fields.RowCount() == faces && boundary.fields.Layout().Width() == index(names.size()) + index(mesh->HasNodeWallDist()) &&
                                      boundary.normals.GetDevice() == R::Device::Host() &&
                                      boundary.normals.RowCount() == faces && boundary.normals.Layout().Width() == 3 &&
                                      index(boundary.zones->size()) == faces,
                                  "visualization snapshot fields and mesh differ");
            DNDS_check_throw_info(!boundary.flux ||
                                      (boundary.flux->GetDevice() == R::Device::Host() && boundary.flux->RowCount() == faces &&
                                       boundary.flux->Layout().Width() == index(boundary.fluxNames.size()) + 3),
                                  "visualization snapshot face flux shape differs");
        }
    };

    PreparedVisualization::PreparedVisualization(std::shared_ptr<Geom::UnstructuredMesh> mesh, R::Array<real> fields,
                                                 std::vector<std::string> names, Boundary boundary, real time, int deflate)
        : _data(std::make_shared<Data>(std::move(mesh), std::move(fields), std::move(names), std::move(boundary), time, deflate))
    {
    }

    void PreparedVisualization::Write(const std::string &prefix, std::shared_ptr<R::CommunicationContext> communication) const
    {
        DNDS_check_throw_info(communication, "visualization writer needs a retained I/O communicator");
        const auto &data = *_data;
        const auto comm = communication->Communicator();
        const auto read = data.fields.Read();
        const auto values = read.View<R::TableRows>();
        const auto name = [&](int i)
        { return data.names.at(i); };
        const auto value = [&](int i, index iCell)
        { return values(iCell, i); };
        data.mesh->PrintParallelVTKHDFDataArray(prefix, "", int(data.names.size()), 0, 0, 0,
                                                name, value, {}, {}, {}, {}, {}, {}, data.time, comm, &data.settings);

        const auto &boundary = data.boundary;
        const auto faceRead = boundary.fields.Read(), normalRead = boundary.normals.Read();
        const auto faces = faceRead.View<R::TableRows>();
        const auto normals = normalRead.View<R::FixedRows<3>>();
        std::optional<R::Array<real>::ReadLease> fluxRead;
        if (boundary.flux)
            fluxRead.emplace(boundary.flux->Read());
        const auto fluxValues = fluxRead ? fluxRead->View<R::TableRows>() : decltype(faces){};
        const index nFields = faces.Width(), nFlux = index(boundary.fluxNames.size());

        Geom::UnstructuredMesh::VTKHDFExtraFields extra;
        extra.cellIntegers.push_back({"bnd_zone_id", {boundary.zones->data(), index(boundary.zones->size())}});
        extra.fieldIntegers.push_back({"bnd_zone_ids", {}});
        extra.fieldStrings.push_back({"bnd_zone_names", {}});
        for (const auto &[id, label] : *boundary.zoneNames)
        {
            extra.fieldIntegers.front().second.push_back(id);
            extra.fieldStrings.front().second.push_back(label);
        }
        const auto scalarName = [&](int i)
        { if (i >= nFields) return boundary.fluxNames.at(i - nFields);
          if (i == int(data.names.size())) return std::string("wall_distance_cell_center");
          return data.names.at(i); };
        const auto scalarValue = [&](int i, index iFace)
        { return i < nFields ? faces(iFace, i) : fluxValues(iFace, i - nFields); };
        const auto vectorName = [](int i) -> std::string
        { return i == 0 ? "norm" : "force_tangential"; };
        const auto vectorValue = [&](int i, index iFace, int d)
        { return i == 0 ? normals(iFace, d) : fluxValues(iFace, nFlux + d); };
        boundary.mesh->PrintParallelVTKHDFDataArray(prefix + "_bnd", "", int(nFields + nFlux), boundary.flux ? 2 : 1, 0, 0,
                                                    scalarName, scalarValue, vectorName, vectorValue,
                                                    {}, {}, {}, {}, data.time, comm, &data.settings, extra);
    }
}
