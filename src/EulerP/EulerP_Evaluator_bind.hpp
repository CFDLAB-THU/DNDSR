#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/ObjectUtils.hpp"
#include "EulerP_Evaluator.hpp"
#include <pybind11/pytypes.h>

namespace DNDS::EulerP
{
    inline void pybind11_Evaluator_define(py::module_ &m)
    {
        using T = Evaluator;
        auto T_ = py_class_ssp<T>(
            m, "Evaluator");

        T_.def(py::init<T::t_fv, T::t_bcHandler, T::t_physics>());
        T_.def("setConfig", [](T &self, const nlohmann::json &c)
               { self.set_config(c); });
        T_.def("getConfig", [](T &self) -> nlohmann::json
               { return self.get_config(); });

        T_
            .def_readwrite("fv", &T::fv)
            .def_readwrite("bcHandler", &T::bcHandler)
            .def_readwrite("physics", &T::physics);
        T_
            .def("to_host", &T::to_host);
        T_
            .def("to_device", [](T &self, const std::string &B)
                 { self.to_device(device_backend_name_to_enum(B)); });
        T_
            .def("device", &T::device);

        T_
            .def("PrintDataVTKHDF", &T::PrintDataVTKHDF,
                 py::arg("fname"), py::arg("series_name"),
                 py::arg("arrCellCentScalar") = py::list(),
                 py::arg("arrCellCentScalar_names") = py::list(),
                 py::arg("arrCellCentVec") = py::list(),
                 py::arg("arrCellCentVec_names") = py::list(),
                 py::arg("arrNodeScalar") = py::list(),
                 py::arg("arrNodeScalar_names") = py::list(),
                 py::arg("arrNodeVec") = py::list(),
                 py::arg("arrNodeVec_names") = py::list(),
                 py::arg("uPrimCell") = py::none(),
                 py::arg("uPrimNode") = py::none(),
                 py::arg("t") = 0.0);

#define DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API(func, func_arg)                    \
                                                                                          \
    {                                                                                     \
        using TArg = T::func_arg;                                                         \
        auto arg_ = py_class_ssp<TArg>(T_, #func_arg);                                    \
        arg_.def(py::init());                                                             \
        for_each_member_ptr_list_raw(TArg::member_list(), [&](auto &mem_ptr)              \
                                     { arg_.def_readwrite(mem_ptr.name, mem_ptr.ptr); }); \
        T_.def(#func, &T::func);                                                          \
    }
        DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API(RecGradient, RecGradient_Arg)
        DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API(RecFace2nd, RecFace2nd_Arg)
        DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API(Cons2PrimMu, Cons2PrimMu_Arg)
        DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API(Cons2Prim, Cons2Prim_Arg)
        DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API(EstEigenDt, EstEigenDt_Arg)
        DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API(Flux2nd, Flux2nd_Arg)

        static auto const a = sizeof(T::Flux2nd_Arg);

#undef DNDS_EULERP_EVALUATOR_BIND_STANDARD_PACKED_API
    }

    inline void pybind11_Evaluator_bind(py::module_ &m)
    {
        pybind11_Evaluator_define(m);
    }
}