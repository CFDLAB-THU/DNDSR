#pragma once

#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/PhysicalAccumulation.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy::CUDA
{
    void ApplyAccumulation(Runtime::ExecutionContext context, PhysicalOperation op, index width,
                           Runtime::ArrayView<real> out, Runtime::ArrayView<const real> x,
                           Runtime::ArrayView<const real> y, Runtime::ArrayView<const real> z, real a, real b);
    using FieldCheckOutput = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Bounded first-invalid-row reduction used by transactional restoration.
    /// Finite mode accepts arbitrary coefficient width. Capped mode is the
    /// two-column LVR history predicate 0<=value<=cap. Caller retains all
    /// borrowed views through completion; this launcher never stages payloads.
    index FieldCheckScratchRows(index rows);
    void CheckFieldRows(Runtime::ExecutionContext, EquationBlockArrayView<const real>,
                        bool capped, std::array<real, 2> caps,
                        FieldCheckOutput partial, FieldCheckOutput result);
}
