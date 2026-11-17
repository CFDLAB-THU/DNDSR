#pragma once
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy
{
    /// Rank-local diagnostics over owned means and a completed spatial residual.
    /// Uses the same submission path on CPU-inline, CPU-worker and CUDA. Evaluate
    /// drains before returning scalar values; the caller owns domain reductions
    /// and normalization. CUDA supports combined AoS perfect-gas flow5/SA/2EQ;
    /// CPU retains general equation layouts and mixture worker ownership.
    class StateDiagnostics
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        StateDiagnostics(CFV::FiniteVolumePlan, EquationSystem, Runtime::ExecutionContext,
                         std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = {});
        /// Slots match the Cauchy progress contract: 0 mass, 1 total energy,
        /// 2 min rho, 3 min p, 4 sum_j R_j^2/volume, 13 min T; other slots zero.
        /// For mechanical flow, mass equals volume, energy is kinetic, and slots
        /// 10/11 are the squared continuity norm and pressure integral. The
        /// domain converts these into an RMS divergence and mean gauge pressure.
        std::array<real, 14> Evaluate(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                      const EquationDofs &mean, const EquationDofs &residual) const;
    };
}
