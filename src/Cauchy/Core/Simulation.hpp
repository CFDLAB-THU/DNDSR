#pragma once

#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Interface/InterfaceDiagnostics.hpp"

namespace DNDS::Cauchy
{
    class InterfaceCoupling;
    class ConnectedPressureGauge;
    /// Controller-local elapsed times, measured without added rank/device sync.
    struct DomainSetupTimings
    {
        real meshIngress = 0, meshPreparation = 0, domainSetup = 0;
    };
    /// Coordinated CPU frontend. Only local domains have runtime objects.
    /// Local domain phases run serially on shared ranks and concurrently on
    /// disjoint communicators. Physical stages share one clock and consume a
    /// frozen interface round. Caller serializes stepping and external mutation.
    /// Unexpected exceptions must escape to a fail-fast MPI application boundary;
    /// catching locally and entering MPI/finalization can strand other ranks.
    /// A rejected StepResult certifies a collectively agreed numerical retry.
    class Simulation
    {
        SimulationConfig _config;
        DomainPlacement _placement;
        std::vector<std::unique_ptr<Domain>> _domains;
        std::vector<DomainSetupTimings> _setupTimings;
        std::unique_ptr<InterfaceCoupling> _interfaces;
        std::unique_ptr<ConnectedPressureGauge> _pressureGauge;
        std::shared_ptr<const UpdatePolicy> _updatePolicy;
        bool _nonlinearCollectiveChecks = false;
        real _time = 0;
        uint64_t _step = 0;
        std::optional<Json> _restoredOutputController;

    public:
        const DomainSetupTimings &SetupTimings(index iDomain) const { return _setupTimings.at(size_t(iDomain)); }
        using MeshFactory = std::function<ssp<Geom::UnstructuredMesh>(index, const DomainConfig &, const MPIInfo &)>;
        Simulation(SimulationConfig config, MPI_Comm comm, const MeshFactory &factory = {});
        ~Simulation();
        const SimulationConfig &Config() const { return _config; }
        const DomainPlacement &Placement() const { return _placement; }
        bool Local(index domain) const;
        Domain &GetDomain(index domain);
        const Domain &GetDomain(index domain) const;
        real Time() const { return _time; }
        uint64_t Step() const { return _step; }
        /// Collective coupling counters and projection gap; no state refresh.
        InterfaceDiagnostics GetInterfaceDiagnostics() const;
        /// Prepare all stage inputs, refresh frozen traces, then solve local
        /// domains. All accepted destinations are reserved before any domain
        /// publishes. Process failure recovery is unsupported. DITR prepares
        /// distinct frozen midpoint/endpoint interface packets, then runs its
        /// registered sequential two-stage nonlinear policy.
        /// If acceptedTime is present, it is the exact controller target for
        /// this step (normally an output time or tEnd). The numerical solve
        /// still uses dt; successful publication snaps every accepted clock to
        /// the supplied target without accumulating an extra floating add.
        [[nodiscard]] StepResult Advance(real dt, std::optional<real> acceptedTime = {});
        /// Parallel persistent-ID checkpoint on each domain communicator.
        /// A multi-domain HDF5 manifest references immutable child files and
        /// is published only after every domain write succeeds. Existing
        /// single-domain archives retain their format and filename.
        void WriteCheckpoint(const std::string &path, bool includeReconstruction = false,
                             Json outputController = Json()) const;
        /// Restore the coordinated clock, all means and compatible method/
        /// reconstruction history. Every domain validates and reserves before
        /// any accepted state changes. Placement may differ; names/order and
        /// physical metadata must match. Interface traces are refreshed at the
        /// stored time from restored polynomials. MeansOnly takes the active
        /// steady means or the accepted means and starts fresh history and
        /// output control. Caller joins work and releases conflicting views.
        void ReadCheckpoint(const std::string &path, CheckpointLoadMode mode = CheckpointLoadMode::Resume);
        const std::optional<Json> &RestoredOutputController() const { return _restoredOutputController; }
        /// One native parallel output per domain, named prefix_domainN for
        /// several domains. The single-domain output name is unchanged.
        void WriteVisualization(const std::string &prefix, int hdfDeflateLevel = 0);
    };
}
