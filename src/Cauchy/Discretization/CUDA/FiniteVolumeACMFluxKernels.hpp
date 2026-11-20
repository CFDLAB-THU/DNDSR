#pragma once
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Shapes.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Integrate the constant-density ALE flux over each face, in its anchor frame.
    /// Externally owned geometry, input and result views live through completion.
    template <int Components>
    void LaunchACMFaces(const RHSFaceLaunch &launch);
}
