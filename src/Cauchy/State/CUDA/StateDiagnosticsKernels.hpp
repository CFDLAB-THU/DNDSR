#pragma once
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include <limits>
#include <optional>

namespace DNDS::Cauchy::CUDA
{
    struct StateDiagnosticSummary
    {
        real mass = 0, energy = 0, squaredResidual = 0;
        real squaredContinuity = 0, integratedPressure = 0;
        real densityMinimum = std::numeric_limits<real>::max();
        real pressureMinimum = std::numeric_limits<real>::max();
        real temperatureMinimum = std::numeric_limits<real>::max();
        NumericalStatus status;
    };
    using DiagnosticInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using DiagnosticOutput = Runtime::ArrayView<StateDiagnosticSummary, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using DiagnosticCounter = Runtime::ArrayView<unsigned int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    index StateDiagnosticScratchRows(index rows);
    /// Pure borrowed-view host launcher. Partials and counter are operation-owned
    /// resident buffers. Stream-ordered counter reset precedes one kernel: the
    /// last arriving block performs the fixed-index tail tree without spinning.
    /// No allocation, host observation, MPI or floating-point atomic sum.
    void AccumulateStateDiagnostics(Runtime::ExecutionContext, DiagnosticInput mean, DiagnosticInput residual,
                                    Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells,
                                    std::optional<PerfectGas> gas, DiagnosticOutput partial, DiagnosticOutput result, DiagnosticCounter);
}
