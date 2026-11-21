#pragma once

#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Implicit/NonlinearOperators.hpp"

namespace DNDS::Cauchy
{
    /// One pressure datum for each closed reciprocal ACM interface component.
    /// The simulation owns the placement and participating domains for the
    /// lifetime of this coordinator. Nonmembers hold only component metadata.
    class ConnectedPressureGauge
    {
        struct Component
        {
            std::vector<index> domains;                    ///< First in configuration order owns the target.
            std::optional<NonlinearOperators> measurement; ///< Anchor-domain ranks only.
        };
        const DomainPlacement &_placement;
        const std::vector<std::unique_ptr<Domain>> &_domains;
        std::vector<Component> _components;
        enum class Phase
        {
            Initial,
            Stage,
            RestartMeans
        };
        void Project(Phase, int stage = 0);
        static void ShiftReceiver(Domain &, int stage, real offset);

    public:
        ConnectedPressureGauge(const SimulationConfig &, const DomainPlacement &, const std::vector<std::unique_ptr<Domain>> &);
        void ProjectStage(int stage) { Project(Phase::Stage, stage); }
        void ProjectRestartMeans() { Project(Phase::RestartMeans); }
    };
}
