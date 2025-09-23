#pragma once

#include "MPI.hpp"
#include "Defines_bind.hpp"
#include "pybind11_json/pybind11_json.hpp"
namespace py = pybind11;

#include "SerializerBase.hpp"
#include "SerializerH5.hpp"
#include "SerializerJSON.hpp"

#include "SerializerFactory.hpp"

namespace DNDS::Serializer
{
    inline auto pybind11_SerializerBase_declare(py::module_ m)
    {
        return py_class_ssp<SerializerBase>(m, "SerializerBase");
    }

    inline auto pybind11_SerializerBase_get_class(py::module_ m)
    {
        return py_class_ssp<SerializerBase>(m.attr("SerializerBase"));
    }
    inline void pybind11_SerializerBase_define(py::module_ m)
    {
        auto Serializer_ = pybind11_SerializerBase_declare(m);
        using tSerializer = SerializerBase;
        // Serializer_ //! no initializer (ctor) as this is virtual base
        //     .def(py::init<>());
        Serializer_
            .def("OpenFile", &tSerializer::OpenFile, py::arg("fName"), py::arg("read"))
            .def("CloseFile", &tSerializer::CloseFile)
            .def("GoToPath", &tSerializer::GoToPath, py::arg("p"))
            .def("CreatePath", &tSerializer::CreatePath, py::arg("p"))
            .def("GetCurrentPath", &tSerializer::GetCurrentPath)
            .def("ListCurrentPath", &tSerializer::ListCurrentPath)
            .def("IsPerRank", &tSerializer::IsPerRank);
    }

    inline auto pybind11_SerializerJSON_declare(py::module_ m)
    {
        return py_class_ssp<SerializerJSON>(m, "SerializerJSON", pybind11_SerializerBase_get_class(m));
    }
    inline void pybind11_SerializerJSON_define(py::module_ m)
    {
        auto Serializer_ = pybind11_SerializerJSON_declare(m);
        using tSerializer = SerializerJSON;
        Serializer_
            .def(py::init<>());
        Serializer_
            .def("SetDeflateLevel", &tSerializer::SetDeflateLevel)
            .def("SetUseCodecOnUint8", &tSerializer::SetUseCodecOnUint8);
    }

    inline auto pybind11_SerializerH5_declare(py::module_ m)
    {
        return py_class_ssp<SerializerH5>(m, "SerializerH5", pybind11_SerializerBase_get_class(m));
    }
    inline void pybind11_SerializerH5_define(py::module_ m)
    {
        auto Serializer_ = pybind11_SerializerH5_declare(m);
        using tSerializer = SerializerH5;
        Serializer_
            .def(py::init<const MPIInfo &>(), py::arg("mpi"));
        Serializer_
            .def("SetChunkAndDeflate", &tSerializer::SetChunkAndDeflate, py::arg("n_chunksize"), py::arg("n_deflateLevel"))
            .def("SetCollectiveRW", &tSerializer::SetCollectiveRW);
    }

    inline auto pybind11_SerializerFactory_declare(py::module_ m)
    {
        return py_class_ssp<SerializerFactory>(m, "SerializerFactory");
    }
    inline void pybind11_SerializerFactory_define(py::module_ m)
    {
        auto SerializerFactory_ = pybind11_SerializerFactory_declare(m);
        SerializerFactory_
            .def(py::init<>())
            // .def(py::init<const std::string &>(), py::arg("type"))
            .def("BuildSerializer", &SerializerFactory::BuildSerializer, py::arg("mpi"))
            .def("ModifyFilePath", &SerializerFactory::ModifyFilePath,
                 py::arg("fname"), py::arg("mpi"), py::arg("rank_part_fmt") = "%06d",
                 py::arg("read") = false)
            .def(
                "to_dict",
                [](SerializerFactory &self) -> py::object
                {
                    nlohmann::ordered_json j;
                    to_json(j, self);
                    return j;
                })
            .def(
                "from_dict",
                [](SerializerFactory &self, py::object options_in) -> void
                {
                    nlohmann::json j(options_in);
                    from_json(j, self);
                });
    }

    void pybind11_bind_Serializer(py::module_ m);
}