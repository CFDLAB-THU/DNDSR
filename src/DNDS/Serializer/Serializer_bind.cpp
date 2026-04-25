#include "Serializer_bind.hpp"

namespace DNDS::Serializer
{
    void pybind11_bind_Serializer(py::module_ m)
    {
        auto m_Serializer = m.def_submodule("Serializer");

        pybind11_SerializerBase_define(m_Serializer);
        pybind11_SerializerJSON_define(m_Serializer);
        pybind11_SerializerH5_define(m_Serializer);

        pybind11_SerializerFactory_define(m_Serializer);
    }
}