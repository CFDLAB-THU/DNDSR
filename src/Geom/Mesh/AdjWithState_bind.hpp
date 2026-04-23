#pragma once
/// @file AdjWithState_bind.hpp
/// @brief pybind11 bindings for AdjWithState<TPair> and AdjIndexInfo.
///
/// AdjWithState<TPair> inherits from TPair, so its pybind11 class is
/// declared as a subclass of the already-registered ArrayAdjacencyPair
/// type.  The three instantiations needed are:
///   AdjWithState<tAdjPair>   (NonUniformSize)
///   AdjWithState<tAdj2Pair>  (row_size=2)
///   AdjWithState<tAdj1Pair>  (row_size=1)

#include "AdjIndexInfo.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency_bind.hpp"

namespace DNDS::Geom
{
    // ================================================================
    // Name helpers (follow ArrayAdjacency_bind.hpp pattern)
    // ================================================================

    template <rowsize _row_size, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    std::string pybind11_AdjWithState_name()
    {
        return "AdjWithState" + pybind11_ArrayAdjacency_name_appends<_row_size, _row_max, _align>();
    }

    template <rowsize _row_size, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using tPy_AdjWithState = py_class_ssp<AdjWithState<ArrayAdjacencyPair<_row_size, _row_max, _align>>>;

    // ================================================================
    // MeshAdjState enum (bound once)
    // ================================================================

    inline void pybind11_MeshAdjState_define(py::module_ &m)
    {
        py::enum_<MeshAdjState>(m, "MeshAdjState")
            .value("Unknown", Adj_Unknown)
            .value("PointToGlobal", Adj_PointToGlobal)
            .value("PointToLocal", Adj_PointToLocal);
    }

    // ================================================================
    // AdjIndexInfo (bound once, all methods)
    // ================================================================

    inline void pybind11_AdjIndexInfo_define(py::module_ &m)
    {
        py::class_<AdjIndexInfo>(m, "AdjIndexInfo")
            .def("state", &AdjIndexInfo::state)
            .def("isLocal", &AdjIndexInfo::isLocal)
            .def("isGlobal", &AdjIndexInfo::isGlobal)
            .def("isBuilt", &AdjIndexInfo::isBuilt)
            .def("isWired", &AdjIndexInfo::isWired);
    }

    // ================================================================
    // AdjWithState<ArrayAdjacencyPair<rs, rm, al>> binding
    // ================================================================

    /// Declare the pybind11 class (creates the type stub in the module).
    /// Must be called AFTER the base ArrayAdjacencyPair type is registered
    /// (typically by importing DNDSR.DNDS).
    template <rowsize _row_size, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    tPy_AdjWithState<_row_size, _row_max, _align>
    pybind11_AdjWithState_declare(py::module_ &m)
    {
        using TPair = ArrayAdjacencyPair<_row_size, _row_max, _align>;

        // Get the already-registered base class from the DNDS module
        auto dnds_m = py::module_::import("DNDSR.DNDS");
        auto baseCls = pybind11_ArrayAdjacencyPair_get_class<_row_size, _row_max, _align>(dnds_m);

        return {m, pybind11_AdjWithState_name<_row_size, _row_max, _align>().c_str(), baseCls};
    }

    /// Define all AdjWithState-specific methods on the pybind11 class.
    template <rowsize _row_size, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    void pybind11_AdjWithState_define(py::module_ &m)
    {
        using TPair = ArrayAdjacencyPair<_row_size, _row_max, _align>;
        using TAdjWS = AdjWithState<TPair>;

        auto cls = pybind11_AdjWithState_declare<_row_size, _row_max, _align>(m);

        cls
            // idx member (read-only reference)
            .def_readonly("idx", &TAdjWS::idx, py::return_value_policy::reference_internal)
            // State queries (convenience, delegate to idx)
            .def("state", &TAdjWS::state)
            .def("isLocal", &TAdjWS::isLocal)
            .def("isGlobal", &TAdjWS::isGlobal)
            .def("isBuilt", &TAdjWS::isBuilt)
            .def("isWired", &TAdjWS::isWired);
    }

} // namespace DNDS::Geom
