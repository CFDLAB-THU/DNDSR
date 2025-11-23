import sys, os

col_sizes = [1, 2, 3, 4, 5, 6, 7, 8, "D", "I"]

templates = {
    (
        "ArrayDOF_op_",
        ".cpp",
    ): """
#include "../ArrayDOF.hpp"
#include "../ArrayDOF_op.hxx"
#include "DNDS/DeviceStorage.hxx"
namespace DNDS
{{
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, {offset}, template)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, {offset}, )
}}
    """,
    (
        "ArrayDOF_op_",
        ".cu",
    ): """
#include "../ArrayDOF.hpp"
#include "../ArrayDOF_op_CUDA.cuh"
#include "DNDS/DeviceStorage.hxx"
#include "DNDS/DeviceStorage_CUDA.cuh"
namespace DNDS
{{
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, {offset}, template)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::CUDA, {offset}, )
}}
    """,
}

base_pos = os.path.abspath(os.path.dirname(__file__))

file_done = set()


def col_size_to_offset(v):
    if isinstance(v, str):
        if v == "D":
            return "DynamicSize - 1"
        if v == "I":
            return "NonUniformSize - 1"
    else:
        return v - 1


for (inst_prefix, inst_suffix), template in templates.items():
    for col_size in col_sizes:
        fname = inst_prefix + f"{col_size}" + inst_suffix
        assert fname not in file_done, f"file name clash: {fname}"
        file_done.add(fname)
        template = template.strip()
        print(f"generated: {fname}")
        with open(os.path.join(base_pos, fname), "w") as f:
            f.write(template.format(offset=col_size_to_offset(col_size)))
