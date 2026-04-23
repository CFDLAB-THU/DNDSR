
#include "Boundary_bind.hpp"
#include "Elements_bind.hpp"
#include "Mesh/Mesh_bind.hpp"
#include "Mesh/AdjWithState_bind.hpp"
#include "Mesh/AdjWithState_bind_inst/AdjWithState_bind_fwd.hpp"

PYBIND11_MODULE(geom_pybind11, m)
{
    //! hard coded dependency here
    //! better solution?
    py::module_::import("DNDSR.DNDS");

    using namespace DNDS::Geom;
    auto m_elem = m.def_submodule("Elem");
    Elem::pybind11_ElemType_define(m_elem);

    pybind11_AutoAppendName2ID_define(m);
    pybind11_ElemInfo_define(m);
    pybind11_ArrayElemInfo_define(m);

    // AdjWithState types (must precede UnstructuredMesh which exposes them)
    pybind11_MeshAdjState_define(m);
    pybind11_AdjIndexInfo_define(m);
    pybind11_AdjWithState_I_I_N_define(m);
    pybind11_AdjWithState_2_2_N_define(m);
    pybind11_AdjWithState_1_1_N_define(m);

    pybind11_MeshLocDefine(m);
    pybind11_UnstructuredMesh_define(m);
    pybind11_UnstructuredMeshSerialRW_define(m);
}
