#include "Cauchy/Core/ConnectedPressureGauge.hpp"
#include "Cauchy/Implicit/ACMPressureGauge.hpp"
#include <set>
#include <tuple>

namespace DNDS::Cauchy
{
    ConnectedPressureGauge::ConnectedPressureGauge(const SimulationConfig &config, const DomainPlacement &placement,
                                                   const std::vector<std::unique_ptr<Domain>> &domains)
        : _placement(placement), _domains(domains)
    {
        std::map<std::string, index> ids;
        for (index i = 0; i < placement.DomainNumber(); ++i)
            ids.emplace(placement.Group(i).name, i);
        std::set<std::tuple<index, int, index, int>> patches;
        for (const auto &edge : config.Interfaces())
        {
            const auto donor = ids.at(edge.donor);
            if (config.Domains()[donor].settings.equation.system == "incompressibleACM")
                patches.emplace(donor, edge.donorZone, ids.at(edge.receiver), edge.receiverZone);
        }
        if (patches.empty())
            return;

        std::vector<std::vector<index>> neighbors(domains.size());
        for (const auto &[a, zoneA, b, zoneB] : patches)
        {
            DNDS_check_throw_info(patches.count({b, zoneB, a, zoneA}), "one-way ACM pressure coupling requires a qualified gauge policy");
            neighbors[a].push_back(b);
        }
        std::vector<int> referenced(domains.size());
        for (auto i : placement.LocalDomains())
            referenced[i] = domains[i]->Discretization().HasPressureBoundary();
        MPI::AllreduceInPlace<MPI::Reduction::Max>(MPI::ContiguousView<int>{referenced.data(), referenced.size()}, placement.Control());

        std::vector<bool> visited(domains.size());
        for (index i = 0; i < placement.DomainNumber(); ++i)
        {
            if (visited[i] || neighbors[i].empty())
                continue;
            Component component{{i}, {}};
            visited[i] = true;
            bool physical = false;
            for (size_t cursor = 0; cursor < component.domains.size(); ++cursor)
            {
                const auto member = component.domains[cursor];
                physical |= bool(referenced[member]);
                for (auto neighbor : neighbors[member])
                    if (!visited[neighbor])
                    {
                        component.domains.push_back(neighbor);
                        visited[neighbor] = true;
                    }
            }
            if (physical)
                continue;
            if (auto &anchor = domains[i])
                component.measurement.emplace(anchor->_problem.finiteVolume.Metrics(), *anchor->_problem.admissibility,
                                              anchor->_execution.ComputeStream().Context());
            _components.push_back(std::move(component));
        }
        Project(Phase::Initial);
    }

    void ConnectedPressureGauge::Project(Phase phase, int stage)
    {
        using NonlinearOperatorDetail::SubmitPressureOffset;
        const auto state = [phase, stage](Domain &domain) -> EquationDofs &
        {
            if (phase == Phase::Stage)
                return domain._state.timeStepper->GroupState(*domain._state.stageGroup, domain._state.trial, stage);
            return phase == Phase::Initial ? domain._state.accepted : domain._state.trial;
        };
        for (const auto &component : _components)
        {
            const auto anchorId = component.domains.front();
            real offset = 0;
            if (auto &anchor = _domains[anchorId])
            {
                const real mean = component.measurement->PressureMean(anchor->_execution.Engine(), anchor->_execution.ComputeStream(), state(*anchor));
                const auto requested = anchor->_problem.finiteVolume.Settings().equation.acmPressureMean;
                if (phase == Phase::Initial)
                    anchor->_state.pressureGauge = std::make_shared<real>(requested.value_or(mean));
                // A means-only load chooses a fresh target unless explicitly configured.
                const real target = phase == Phase::RestartMeans ? requested.value_or(mean) : *anchor->_state.pressureGauge;
                offset = target - mean;
            }
            MPI_Bcast(&offset, 1, MPI_DOUBLE, _placement.SimulationRank(anchorId, 0), _placement.Control());
            if (!offset)
                continue;
            for (auto i : component.domains)
                if (auto &domain = _domains[i])
                {
                    auto &engine = domain->_execution.Engine();
                    const auto &lane = domain->_execution.ComputeStream();
                    SubmitPressureOffset(engine, lane, state(*domain), offset);
                    if (phase == Phase::Initial)
                        SubmitPressureOffset(engine, lane, domain->_state.trial, offset);
                    if (phase == Phase::Stage)
                        ShiftReceiver(*domain, stage, offset);
                    else
                        lane.Record().WaitHost();
                }
        }
    }

    void ConnectedPressureGauge::ShiftReceiver(Domain &domain, int stage, real offset)
    {
        // Preserve the accepted or earlier-stage publication used for rollback.
        // A frozen stage still needs the new datum, but keeps exactly its last
        // sampled velocity, RANS values, epoch and donor provenance.
        const auto &fv = domain._problem.finiteVolume;
        const auto &packets = *domain._execution.interfaces;
        const auto &old = domain._state.stageGroup->coupledStages == 2 ? packets.coupled.at(stage) : packets.active;
        DNDS_check_throw_info(old, "connected pressure projection requires a published stage packet");
        auto &engine = domain._execution.Engine();
        const auto &lane = domain._execution.ComputeStream();
        auto next = fv.CopyBoundaryRows(*old, engine, lane);
        NonlinearOperatorDetail::SubmitPressureOffset(engine, lane, next.FaceCenterValues(), offset);
        NonlinearOperatorDetail::SubmitPressureOffset(engine, lane, next.QuadratureValues(), offset);
        fv.PublishBoundaryInputs(next, old->Time(), old->Epoch(), old->InterfaceSources(), lane);
        domain.BindInterfaceInputs(next, stage);
    }
}
