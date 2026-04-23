
#include "Boundary_bind.hpp"
#include "Elements_bind.hpp"
#include "Mesh/Mesh_bind.hpp"

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

    pybind11_MeshLocDefine(m);
    pybind11_UnstructuredMesh_define(m);
    pybind11_UnstructuredMeshSerialRW_define(m);
}
