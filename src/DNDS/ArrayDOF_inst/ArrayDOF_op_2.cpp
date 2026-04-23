#include "../ArrayDOF.hpp"
#include "../ArrayDOF_op.hxx"
#include "DNDS/Device/DeviceStorage.hxx"
namespace DNDS
{
    DNDS_ARRAY_DOF_OP_FUNC_SEQ_INST(DeviceBackend::Host, 1, template)
    // DNDS_ARRAYDOF_INST_STORAGE(DeviceBackend::Host, 1, )
}