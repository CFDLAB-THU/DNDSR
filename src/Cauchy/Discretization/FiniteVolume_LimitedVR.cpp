#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    FiniteVolume::LimitedVRGeometry::LimitedVRGeometry(const Data &data)
        : gates(CFV::LimitedVRPlan::Prepare(data.metrics))
    {
        DNDS_check_throw_info(data.reconstruction,
                              "LVR geometry requires variational reconstruction");
        // Face IDs, parents, center metrics and periodic maps are shared with
        // the ordinary FV publication. The selected gradient operator and its
        // mutable execution workspace belong to the common FV path as well.
    }
    FiniteVolume::LimitedVRWorkspace::LimitedVRWorkspace(const Data &data, const CFV::ReconstructionWorkspace &prototype)
        : target(data.reconstruction->BuildWorkspace(prototype.components, prototype.temporary.Owned().Layout(), prototype.periodicVectorStarts)),
          targetCoefficients(data.reconstruction->BuildCoefficients(prototype.components, prototype.temporary.Owned().Layout())),
          scalars(data.metrics.BuildField(2)),
          branches(data.metrics.BuildField(3)), seeds(data.metrics.BuildField(3)), temporary(data.metrics.BuildField(3)),
          base(R::RowLayout::Dense(data.metrics.NumCell(), 2), data.metrics.Device()),
          alpha(R::RowLayout::Dense(data.metrics.NumCell(), 1), data.metrics.Device()), targetHalo(data.BindHalo(targetCoefficients)),
          branchHalo(data.BindHalo(branches)), seedHalo(data.BindHalo(seeds))
    {
        // Current physical adapter dispatches only compressible flow5. Generic
        // CFV penalties/targets and scalar stencils do not assume this width.
        DNDS_check_throw_info(prototype.components == 5 && data.limitedVRGeometry, "LVR compressible target setup is incomplete");
        Algebra::Fill(targetCoefficients, 0.);
        Algebra::Fill(target.temporary, 0.);
        if (data.metrics.Device() == R::Device::Host())
        {
            auto penalty = alpha.Write();
            for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
                penalty.View()(iCell, 0) = 0;
        }
        // Device allocation is already zero-initialized. Private status and
        // sensor scratch is allocated lazily at the first queued Build.
    }
}
