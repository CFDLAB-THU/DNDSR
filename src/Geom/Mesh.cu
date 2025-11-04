#include "Mesh.hpp"
#include "DNDS/DeviceStorage_CUDA.cuh"

namespace DNDS
{
    DNDS_DEVICE_STORAGE_INST(Geom::ElemInfo, DeviceBackend::CUDA, )
}