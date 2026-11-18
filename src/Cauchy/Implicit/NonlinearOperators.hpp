#pragma once

#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Core/OptionalRef.hpp"

namespace DNDS::Cauchy
{
    struct NonlinearPseudoTimeControls
    {
        real cfl = 1, forecastScale = 1, relax = 1;
        int compressionPower = 1;
        bool positivity = false, compressRHS = false;
    };
    struct NonlinearCorrectionControls
    {
        real scale = 1, relax = 1;
        int compressionPower = 1;
        bool positivity = true;
    };
    struct NonlinearCorrectionResult
    {
        AlphaStatistics alpha;
        bool frozen = true;
    };

    /** @brief Managed nonlinear arithmetic on one ordered execution stream.
     * @details This operator layer owns only immutable metrics/policies and
     * reusable bounded reduction scratch. It never owns the evolving solution
     * or commits a candidate. Submission retains every Array/metadata owner;
     * CPU-inline, CPU-worker and CUDA use the same controller call sequence.
     * Numerical leaves take views and POD policies, without MPI or leases.
     *
     * Methods returning host values explicitly drain the engine before reading
     * compact summaries. These are nonlinear acceptance/convergence boundaries,
     * not per-cell waits. Submit* methods only enqueue, and their consumer must
     * order or observe that prefix. No MPI reduction is performed here; the
     * controller aggregates on the domain communicator. CUDA currently accepts
     * combined AoS perfect-gas flow5/SA/2EQ, while CPU retains generic layouts
     * and mixture policies. A workspace is serialized on its bound stream.
     */
    class NonlinearOperators
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        NonlinearOperators(CFV::FiniteVolumePlan, AdmissibilityPolicy, Runtime::ExecutionContext);
        void ValidateMeans(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &) const;
        /// Observe the volume-weighted domain mean of the ACM pressure.
        real PressureMean(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &) const;
        /// Shift the candidate by target minus its current domain pressure mean.
        /// Only the scalar reduction is observed on the host.
        void ProjectPressureMean(Runtime::EagerEngine &, const Runtime::ExecutionStream &, EquationDofs &, real target) const;
        /// Integrated residual, with raw L1 or volume-weighted L2 contributions
        /// followed by volume and, when measuring ACM, the squared spatial
        /// continuity integral independent of stage scaling/history. An empty selection means
        /// every component; otherwise it has exactly the equation width.
        std::vector<real> AssembleDefect(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                         const EquationDofs &mean, const EquationDofs &residual, const EquationDofs &constant,
                                         EquationDofs &output, real mass, real rhsScale, SpanView<const uint8_t> selection,
                                         int normOrder, bool measureNorm) const;
        /// Raw sums of volume*abs(U_j), then volume. Component normalization
        /// and global agreement are deliberately separate from this operator.
        std::vector<real> MeasureMeanState(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &) const;
        /// Local volume-weighted norm of a rate (or rate-reference). This is
        /// the DITR stopping residual / finest FAS norm, not a new iteration
        /// loop. Return raw L1 or L2-squared integrals and final volume; the
        /// caller performs domain MPI aggregation and any square root. An
        /// optional raw ACM rate adds its squared continuity integral after volume.
        std::vector<real> MeasureRateNorm(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                          const EquationDofs &rate, int normOrder, OptionalRef<const EquationDofs> reference = {},
                                          OptionalRef<const EquationDofs> rawACMRate = {}) const;
        AlphaStatistics PreparePseudoTime(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                          const EquationDofs &mean, EquationDofs &defect, const Runtime::Array<real> &spectral,
                                          Runtime::Array<real> &inverse, OptionalRef<Runtime::Array<real>> alpha,
                                          NonlinearPseudoTimeControls) const;
        void SubmitAgreedRHSAlpha(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                  const Runtime::Array<real> &alpha, Runtime::Array<real> &inverse, EquationDofs &defect,
                                  real minimum, AlphaReduction, bool compressRHS) const;
        NonlinearCorrectionResult ApplyCorrection(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                                  const EquationDofs &mean, const EquationDofs &increment, EquationDofs &candidate,
                                                  SpanView<const uint8_t> selection, NonlinearCorrectionControls) const;
        void SubmitScaleRate(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                             const EquationDofs &input, EquationDofs &output, bool multiplyVolume) const;
        void SubmitEquation(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                            const EquationDofs &mean, const EquationDofs &residual, EquationDofs &output,
                            OptionalRef<EquationDofs> rate, real mass, real rhsScale) const;
    };
}
