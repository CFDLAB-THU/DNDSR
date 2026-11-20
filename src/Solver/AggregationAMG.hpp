#pragma once
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Solver
{
    struct AggregationAMGSettings
    {
        int aggregateSize = 4, minCoarseRows = 16, maxLevels = 20;
        int preSweeps = 2, postSweeps = 2, coarseSweeps = 8;
        real relaxation = .75;
    };

    /// Rank-local, host-only unsmoothed aggregation with weighted block Jacobi.
    /// Structure is copied at setup. The graph contains local off-diagonal
    /// columns only, unique per row, with nonnegative geometric strengths.
    /// Connected aggregates grow along their strongest available edge, ties
    /// resolved by local row ID. Aggregate size is a target upper bound.
    class AggregationAMG
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        AggregationAMG(const Runtime::Array<int> &offsets, const Runtime::Array<int> &columns,
                       const Runtime::Array<real> &weights, int blockSize,
                       AggregationAMGSettings settings = {});
        ~AggregationAMG();
        index LevelCount() const;
        /// Bind host tables [edges,B*B] and [cells,B*B] of row-major blocks.
        /// Recompute coarse blocks and scaled LU/SVD diagonal inverses. A
        /// subsequent mutation requires another refresh before applying.
        void Refresh(const Runtime::Array<real> &values, const Runtime::Array<real> &diagonal);
        /// One fixed V-cycle from zero. Input/output are distinct host tables
        /// [cells,B]. Calls on the same object are serialized by the caller.
        void Apply(const Runtime::Array<real> &rhs, Runtime::Array<real> &correction);
    };
}
