#pragma once

#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Implicit/FrozenJacobianOperator.hpp"
#include "Cauchy/Implicit/CUDA/FiniteVolumeJacobianKernels.hpp"
#include "DNDS/Algebra/ArrayInputCache.hpp"

namespace DNDS::Cauchy
{
    /** @brief Reusable preparation scratch, separate from frozen publications.
     * @details Usually two numerical slots suffice for evaluated storage: one
     * current publication and one being filled. A retained external publication
     * may keep a slot busy; another slot is allocated rather than overwritten.
     * Released slots are reused. Assembled mode needs only one slot. No engine
     * or stream is owned here; the enclosing domain context owns execution.
     * Persistent immutable read seals use the device's default context, not
     * the borrowed preparation stream, which can expire before this workspace.
     * Every asynchronous numerical use retains this object through completion;
     * the seals are ownership/exclusion guards, not completion proofs for those
     * uses. Placement completes before these immutable views are published.
     */
    struct FiniteVolume::Workspace::DeviceJacobianData
    {
        struct Slot
        {
            FrozenJacobianOperator::Values values;
            std::weak_ptr<const OffDiagonalEvaluator> publication;
            bool roe = false;
        };
        std::shared_ptr<const Data> owner;
        CFV::FiniteVolumeRead metric;
        FiniteVolumeStencilStorage::ReadAccess stencil;
        Runtime::ReadAccess physics, conditions;
        std::optional<WallDistanceRead> walls;
        std::optional<Runtime::Array<real>::ReadLease> acmLengths;
        std::shared_ptr<const JacobianGeometryStorage> geometry;
        std::shared_ptr<const FrozenJacobianOperator::Geometry> publicationGeometry;
        std::optional<JacobianGeometryStorage::ReadAccess> geometryRead;
        std::vector<std::shared_ptr<Slot>> slots;
        Runtime::Array<NumericalStatus> meansStatus, rowsStatus, assemblyStatus, partial, result;
        Runtime::Array<real> waveJump, pressureMinimum;
        std::optional<Algebra::ArrayInputCache> inputOwned, inputGhost;
        index components;

        DeviceJacobianData(std::shared_ptr<const Data>, index components, Runtime::ExecutionContext);
        std::shared_ptr<Slot> Acquire(bool roe);
        void AppendStatus(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const Runtime::Array<NumericalStatus> &);
        void SubmitSource(Workspace &, Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                          const EquationDofs &, Runtime::Array<real> &diagonal, bool fullBlock, real scale);
    };
}
