#pragma once
/// @file DeviceStorage_bind.hpp
/// @brief pybind11 binding of the @ref DNDS::DeviceBackend "DeviceBackend" enum into the Python module.

#include "Defines_bind.hpp"
#include "DeviceStorage.hpp"

namespace DNDS
{
    inline void pybind11_bind_deviceStorage(py::module_ &m)
    {
#define DNDS_PY_ENUM_CLASS_DeviceBackend_ADD(v) value(#v, DeviceBackend::v)
        auto DeviceBackend_ = py::enum_<DeviceBackend>(m, "DeviceBackend");
        DeviceBackend_
            .DNDS_PY_ENUM_CLASS_DeviceBackend_ADD(Unknown)
            .DNDS_PY_ENUM_CLASS_DeviceBackend_ADD(Host)
#ifdef DNDS_USE_CUDA
            .DNDS_PY_ENUM_CLASS_DeviceBackend_ADD(CUDA)
#endif
            .DNDS_PY_ENUM_CLASS_DeviceBackend_ADD(Custom1);

        DeviceBackend_
            .def(py::init([](const std::string &name) -> DeviceBackend
                          { return device_backend_name_to_enum(name); }));
    }

    inline void pybind11_bind_device_controls(py::module_ &m)
    {
        // #ifdef DNDS_USE_CUDA
        //         m.def("cudaGetDevice", []()
        //               {

        //               });
        // #endif
    }
}