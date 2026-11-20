#pragma once

#include "Cauchy/IO/OutputStaging.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy
{
    /** @brief Immutable output-field preparation, independent of file writers.
     * @details Columns are logical conservative components, then p,T,u,v,w,
     * Mach, followed by all species mass fractions when present. Numerical
     * production runs on the supplied lane; CPU supports the equation layouts
     * and mixture workers, CUDA supports combined AoS perfect-gas flow5/SA/2EQ.
     * Constant-density ACM emits its velocity, gauge pressure and optional
     * specific RANS scalars in schema order, with no thermodynamic extras.
     * Input allocations and worker resources are retained until completion.
     * No MPI, serialization, state repair or source mutation is performed.
     */
    class OutputFields
    {
        friend class PreparedVisualization;
        Runtime::Array<real> CopyToHost(Runtime::ExecutionContext, OutputStaging::Batch &) const;
        Runtime::Array<real> _values;
        Runtime::Array<NumericalStatus> _status;

    public:
        static OutputFields Submit(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                   const EquationDofs &, const EquationSystem &,
                                   std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = {});
        Runtime::Device Device() const { return _values.GetDevice(); }
        /// Explicit I/O observation after the producer lane is drained. Checks
        /// numerical status before returning a host field; never evaluates CPU
        /// physics on a downloaded device state. The result owns its payload.
        Runtime::Array<real> CopyToHost(Runtime::ExecutionContext) const;
    };
}
