#pragma once

#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Implicit/CUDA/FrozenJacobianKernels.hpp"
#include "Cauchy/Boundary/PerfectGasBoundary.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy::CUDA
{
    using JacobianStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using JacobianCellMetrics = Runtime::ArrayView<const real, false, Runtime::FixedRows<8>>;

    /** @brief Freeze owned/ghost means and their conservative pressure rows.
     * @details Combined AoS perfect gas, widths 5/6/7. No repair or clipping is
     * applied. RANS pressure derivatives are zero; turbulent normal stress is
     * differentiated independently by the flux Jacobian. The output ordering
     * is owned followed by ghost. Invalid points report status and must not be
     * consumed. Input storage may be reused after the scheduled freeze; all
     * subsequent matrix actions use this independent immutable publication.
     */
    void FreezeJacobianMeans(Runtime::ExecutionContext context, const PerfectGas &gas,
                             VectorReadRows owned, VectorReadRows ghost,
                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> means, LinearWriteRows pressure,
                             LinearWriteRows pressureDerivative, JacobianStatus status);

    /// Freeze a scalar stencil signal in owned-then-ghost numbering. This is
    /// a device copy of an already exchanged field, not a halo operation.
    /// Outputs must be distinct; stream ordering is supplied by the caller.
    void FreezeJacobianSignal(Runtime::ExecutionContext context, LinearReadRows owned,
                              LinearReadRows ghost, LinearWriteRows output);

    /** @brief Initialize M*(1/dTau+physicalMass) and rotating-frame terms.
     * @details Full blocks add the negative source derivative of
     * -Omega cross rho*u; diagonal-entry mode retains the CPU norm bound on
     * flow5 only. No component projection is applied until all source/face
     * contributions have been added. Each selected row is overwritten.
     * RANS source destruction is a separate additive volume-source operator.
     */
    void InitializeJacobianDiagonal(Runtime::ExecutionContext context, JacobianCellMetrics cells,
                                    LinearReadRows inverseTau, MatrixWriteRows diagonal, JacobianStatus status,
                                    LinearIndexRows selected, const RigidFrameParameters &frame,
                                    real physicalMass, real scale, int components, bool fullBlock, bool rowMajorBlocks = false);

    /** @brief Frozen centered face inputs of the approximate diagonal.
     * @details Every incidence is visited once, independently of polynomial
     * degree. The sampled exterior state is local scratch: reconstruction BC
     * buffers are not used. Interface rows use the solver's current centered
     * packet. Conditions are small, immutable resident metadata; all mesh-sized
     * fields are borrowed whole-array views retained by the operator.
     */
    struct JacobianFaceInput
    {
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        index ownedCells = 0, incidences = 0;
        Runtime::ArrayView<const PerfectGasBoundaryParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> conditions;
        VectorReadRows means;
        LinearReadRows pressure, pressureDerivative, radius, incoming;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> wallMean;
        LinearReadRows waveJump, pressureMinimum;
        FrozenJacobianParameters parameters;
        Physics::RANS::ModelKind rans = Physics::RANS::ModelKind::None;
        Riemann::Solver solver = Riemann::Solver::Rusanov;
        Riemann::FluxSettings fluxSettings;
        real time = 0;
        bool physicalBoundary = true;
        bool rowMajorBlocks = false;
    };
    /** @brief Add face dissipation, wall chain rules and self-periodic terms.
     * @details diagonal/status already contain mass and source contributions.
     * Existing failed rows are not consumed. Scalar dissipation supports the
     * physical/eulerScalar choice; Roe always includes its wall chain rule.
     * Roe magnitudes are written per incidence for later evaluated actions.
     * Off-diagonal blocks are filled separately using AssembleFrozenJacobian.
     * Final component projection acts on the complete accumulated diagonal.
     */
    void AddJacobianFaces(Runtime::ExecutionContext context, const JacobianFaceInput &input,
                          LinearIndexRows selected, MatrixWriteRows diagonal, bool fullBlock,
                          LinearWriteRows magnitudes, JacobianStatus status);
}
