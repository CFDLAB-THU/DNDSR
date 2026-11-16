#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Core/OMP.hpp"
#include <optional>
#include <vector>

namespace DNDS::Cauchy
{
    /// Build private CPU point-evaluator/scratch slots at operator execution,
    /// before entering an OMP region. Kernels receive only the resulting span.
    /// Perfect-gas/RANS slots follow DNDS distributed OMP controls. Mixtures
    /// atomically reserve up to that many available chemistry workers; the
    /// returned span size is the actual parallelism. No worker waits while
    /// holding a partial team or acquires another context inside its loop.
    /// Each Worker owns the evaluator and any mutable point-sized scratch.
    template <class Worker, class... Args>
    std::vector<Worker> MakeCPUPointWorkers(
        const EquationSystem &equation, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &pool,
        Physics::Thermodynamics::MixturePropertyRequest properties, const Args &...args)
    {
        auto leases = equation.HasMixture() && pool ? pool->AcquireUpTo(OMP::DistributedThreads())
                                                    : std::vector<Physics::Chemistry::ChemicalWorkerPool::Lease>{};
        const index count = equation.HasMixture() ? index(leases.size()) : OMP::DistributedThreads();
        DNDS_check_throw_info(count > 0, "mixture point workers require their chemistry pool");
        std::vector<Worker> workers;
        workers.reserve(size_t(count));
        for (index iWorker = 0; iWorker < count; ++iWorker)
            workers.emplace_back(equation.MakePointEvaluator(
                                     equation.HasMixture() ? std::move(leases[iWorker]) : Physics::Chemistry::ChemicalWorkerPool::Lease{}, properties),
                                 args...);
        return workers;
    }
}
