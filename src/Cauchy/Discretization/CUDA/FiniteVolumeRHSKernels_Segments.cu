#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Segments.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    namespace R = Runtime;
    unsigned ValidateSegmentedFluxFaces(R::ExecutionContext context, const RHSGeometryPublication &geometry,
                                        RHSKernelControls controls,
                                        RHSIndices selected, RHSFacePackageView packages)
    {
        packages.ValidateDevice(context.device);
        const auto shape = RHSFacePackages::StageShape(controls);
        const size_t sharedBytes = size_t(controls.blockSize) * RHSFacePackages::LaneScratchBytes(controls) +
                                   size_t(packages.maximumFaces) * 2 * shape.width * sizeof(real) * (controls.doubleBuffer ? 2 : 1);
        DNDS_check_throw_info(packages.mapping == controls.mapping && packages.blockSize == controls.blockSize &&
                                  packages.produceLow == controls.produceLow && packages.doubleBuffer == controls.doubleBuffer && packages.selection.data == selected.data &&
                                  packages.selection.RowCount() == selected.RowCount() && packages.selection.stride == selected.stride &&
                                  packages.shape.components == shape.components && packages.shape.reconstruction == shape.reconstruction && packages.shape.viscous == shape.viscous && packages.shape.width == shape.width &&
                                  packages.sharedBytes == sharedBytes && packages.sharedBytes <= 48 * 1024 && packages.maximumFaces > 0 &&
                                  packages.packages.RowCount() > 0 && packages.segments.RowCount() == selected.RowCount(),
                              "RHS segments were not prepared for this exact selection/profile");
        return unsigned(std::min<index>(packages.packages.RowCount(), controls.gridBlocks ? controls.gridBlocks : geometry.MultiprocessorCount() * 4));
    }
}
