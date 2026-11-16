#pragma once

#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"

namespace DNDS::Cauchy
{
    /** @brief Rank-local AmgX adapter for the coupled assembled operator.
     * @details One device, no communicator and no distributed AmgX machinery.
     * The initial compatible path requires one rank, a primary native int32
     * pattern and compact column-major cell blocks with separate diagonal.
     * Native DNDS allocations are borrowed directly by the public upload;
     * provider-internal copies remain owned by AmgX. Incompatible layouts or
     * off-rank principal extraction fail explicitly. No DNDS packing, sorting,
     * transposition or index conversion is hidden in this adapter.
     * Pattern preparation is once per solver; values update when the frozen
     * matrix changes. Provider cycles are fixed, zero-started and linear.
     * Update/Apply drain the supplied engine before entering the provider:
     * its device-wide join must not depend on host MPI progress blocked here.
     * Callers serialize use and keep the matrix unchanged throughout Solve.
     */
    class LocalAMGPreconditioner
    {
        struct State;
        std::shared_ptr<State> _state;

    public:
        explicit LocalAMGPreconditioner(const LinearProblemContext &, Algebra::AmgXSettings settings = {});
        void Update(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const LinearProblemContext &);
        void Apply(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                   const Runtime::Array<real> &rhs, Runtime::Array<real> &output);
    };
}
