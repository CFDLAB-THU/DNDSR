#pragma once

#include "Cauchy/Physics/PerfectGas.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy::CUDA
{
    void PrepareVelocityPressureOutput(Runtime::ExecutionContext,
                                       Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> state,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> output,
                                       Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> status);
    /// Host launcher checks whole-array residency/shape; one CUDA thread per
    /// output row. No per-element device checks or CPU property fallback.
    void PrepareOutputFields(Runtime::ExecutionContext, PerfectGas,
                             Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> state,
                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> output,
                             Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> status);
}
