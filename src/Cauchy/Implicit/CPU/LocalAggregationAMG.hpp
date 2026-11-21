#pragma once

#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Solver/AggregationAMG.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    /// Host adapter for the domain-local principal matrix. Numeric refresh uses the
    /// Cauchy matrix mutation epochs. V-cycle vectors follow equation component order.
    class CPUAMG
    {
        index rows, width;
        std::vector<index> slots;
        R::Array<real> blocks, diagonal, input, output;
        std::unique_ptr<Solver::AggregationAMG> solver;
        uint64_t diagonalEpoch = 0, offEpoch = 0;
        bool current = false;

    public:
        CPUAMG(const LinearProblemContext &problem, int aggregateSize);
        void Update(R::EagerEngine &engine, const LinearProblemContext &problem);

        /// Retire queued field work before the synchronous host V-cycle. The
        /// input and output leases cover only owned rows. Ghosts are unchanged.
        template <class State>
        void Apply(R::EagerEngine &engine, const State &rhs, State &result);
    };
}
