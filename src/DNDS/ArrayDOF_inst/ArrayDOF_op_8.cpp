#include "../ArrayDOF.hpp"
#include "../ArrayDOF_op.hxx"
#include "DNDS/DeviceStorage.hxx"
namespace DNDS
{
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 7, template)
    DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 7, )
}