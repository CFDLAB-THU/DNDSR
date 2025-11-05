#include "../ArrayDOF.hpp"
#include "../ArrayDOF_op_CUDA.cuh"
namespace DNDS
{
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::CUDA, 7, template)
}