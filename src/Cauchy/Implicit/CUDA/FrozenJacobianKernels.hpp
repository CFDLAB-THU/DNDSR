#pragma once

#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/JacobianGeometry.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy::CUDA
{
    /** @brief Frozen mathematical choices for the specialized sparse operator.
     * @details Component selection is in the stored equation frame, on both
     * sides of a periodic rotation. kineticComponent is -1 or the rho*k slot
     * of the two-equation profile; SA adds scalar convection only. Roe matrix
     * dissipation is the existing flow5-only choice. Full equation EOS work is
     * performed by the producer, not by sparse application.
     */
    struct FrozenJacobianParameters
    {
        RigidFrameParameters frame;
        PerfectGas gas;
        real scale = 1;
        int components = 5, kineticComponent = -1;
        bool roe = false;
        std::array<uint8_t, 7> active{1, 1, 1, 1, 1, 1, 1};
    };

    /** @brief Read-only linearization snapshot, independent of live means.
     * @details means, pressure and pressureDerivative cover owned+ghost cells
     * in one canonical numbering. radius stores convection/diffusion per
     * auxiliary face; Roe magnitudes are contact/plus/minus per incidence and may
     * be absent for scalar dissipation. Geometry preserves separate periodic
     * images that happen to address the same sparse block. All values are
     * retained by the calling operation; this POD acquires no leases.
     */
    struct FrozenJacobianInput
    {
        JacobianGeometryView geometry;
        VectorReadRows means;
        LinearReadRows pressure, pressureDerivative, radius, magnitudes;
        FrozenJacobianParameters parameters;
    };
    using FrozenJacobianStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /// Fill selected owned off-diagonal block rows, overwriting every selected
    /// block. Status is per owned cell. No diagonal, source or BC work is done:
    /// those contributions belong to the diagonal-preparation producer.
    void AssembleFrozenJacobian(Runtime::ExecutionContext context, const FrozenJacobianInput &snapshot,
                                LinearIndexRows selected, Algebra::BlockMatrixArrayView<real> output,
                                FrozenJacobianStatus status);

    /// Evaluate y=A*x (jacobi=false) or D^-1*(b-Aoff*x) (jacobi=true) directly
    /// from frozen physical values, without allocating dense off-diagonal
    /// blocks. diagonal is D or D^-1 respectively. Core selection permits an
    /// absent vector ghost view; frozen means remain independent read-only
    /// snapshots. Zero-start is handled by JacobiLinearRows and needs no
    /// physical snapshot. Errors are explicit per-row status, never exceptions
    /// or componentwise numerical fallback inside the device kernel.
    /// Status must initially contain success; failures are sticky until an
    /// explicit caller reset. A failed row produces NaNs rather than stale data.
    void ApplyFrozenJacobian(Runtime::ExecutionContext context, const FrozenJacobianInput &snapshot,
                             LinearIndexRows selected, MatrixReadRows diagonal, bool fullDiagonal,
                             VectorReadRows rhs, VectorReadRows owned, VectorReadRows ghost,
                             LinearWriteRows output, FrozenJacobianStatus status, bool jacobi, bool rowMajorBlocks = false);
}
