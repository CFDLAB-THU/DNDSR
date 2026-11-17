#pragma once

#include "Cauchy/Input/StateInput.hpp"

#include <memory>

namespace DNDS::Cauchy
{
    /// Worker-private, reusable setup evaluator. Geometry supplies a world point in
    /// code-length units; velocity components are absolute in world axes.
    /// Boxes -> planes -> expressions, each in config order. Later matches
    /// override earlier values at the point, before conservative averaging.
    /// Owns the compiled expressions and one EOS worker. Use a separate
    /// instance per concurrent CPU worker; no hidden OMP/global state.
    class InitialFieldEvaluator
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        InitialFieldEvaluator(const InitialFieldSettings &settings, const EquationSystem &equation,
                              const Physics::ReferenceScales &scales,
                              Physics::Chemistry::ChemicalWorkerPool::Lease worker = {});
        ~InitialFieldEvaluator();
        InitialFieldEvaluator(const InitialFieldEvaluator &) = delete;
        InitialFieldEvaluator &operator=(const InitialFieldEvaluator &) = delete;
        /// background is a complete conservative world-frame state. t is in
        /// code-time units; iCell is rank-local and for diagnostics only (do not
        /// use it to specify a partition-independent field). Every expression
        /// receives a fresh seed and inRegion=0; stateType never rescales x/t.
        std::vector<real> Evaluate(const Vector3 &point, real time, index iCell, const std::vector<real> &background);
    };
}
