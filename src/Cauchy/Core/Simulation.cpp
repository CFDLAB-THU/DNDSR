#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "Cauchy/IO/ProgressOutput.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "DNDS/Core/OMP.hpp"

#include <fmt/format.h>

namespace DNDS::Cauchy
{
    namespace
    {
        ssp<Geom::UnstructuredMesh> ReadMesh(
            const DomainConfig &domain, const MPIInfo &mpi,
            const std::shared_ptr<Runtime::CommunicationContext> &domainCommunication, DomainSetupTimings &timings,
            Diagnostics::SetupProgress progress)
        {
            // A preceding colocated domain may already have entered the
            // per-rank context. Restore the reader-owned team for this domain's
            // import, serial partitioning and distribution.
            OMP::set_serial_ingress_OMP();
            DNDS_check_throw_info(!domain.settings.mesh.file.empty(),
                                  "Cauchy mesh path is empty");
            Geom::PartitionOptions partition;
            const auto mapper = [ids = domain.settings.mesh.boundaryIds](
                                    const std::string &name)
            {
                const auto found = ids.find(name);
                DNDS_check_throw_info(
                    found != ids.end(),
                    "CGNS boundary name is absent from mesh.boundaryIds: " + name);
                return found->second;
            };
            std::optional<Geom::NativePeriodicImportOptions> periodic;
            if (domain.settings.mesh.periodicLengths !=
                std::array<real, 3>{0, 0, 0})
            {
                periodic.emplace();
                periodic->progress = progress;
                for (int d = 0; d < 3; ++d)
                {
                    periodic->frames.translation[d + 1].map().setZero();
                    periodic->frames.translation[d + 1].map()[d] =
                        domain.settings.mesh.periodicLengths[d];
                }
            }
            auto communication = domainCommunication->WithPreflight(
                domain.settings.implicit.collectiveChecks);
            const real ingressStarted = MPI_Wtime();
            auto owned = Geom::ReadAndDeriveNativeMeshFromCGNSOwned(
                mpi, domain.settings.mesh.dimension, domain.settings.mesh.file,
                partition, mapper, periodic, domain.settings.mesh.elevation,
                domain.settings.mesh.bisect, 0, Geom::MeshAuditOptions{},
                communication, progress);
            timings.meshIngress = MPI_Wtime() - ingressStarted;
            // Native topology, reorder, CFV setup and solver work now belong to
            // distributed ranks and must not inherit the node-wide reader team.
            OMP::set_full_parallel_OMP();
            const real prepareStarted = MPI_Wtime();
            Geom::PrepareMeshOptions prepare;
            prepare.buildSerialOut = false;
            prepare.reorderCells = 1;
            prepare.reorderParts = OMP::DistributedThreads();
            prepare.progress = progress;
            auto mesh = Geom::UnstructuredMesh::FromNative(
                mpi, Geom::PrepareNativeMeshForSolverOwned(
                         std::move(owned), mpi, prepare));
            const auto &settings = domain.settings.mesh;
            if (settings.scale != 1 || settings.rotationZDegrees != 0)
            {
                auto setupStage = progress.Stage("mesh/geometry");
                // Import/matching uses the file frame. The ordinary Geom
                // publication transforms coordinates AND periodic frames
                // together, exactly once before metrics/initial-state sampling.
                const real angle = settings.rotationZDegrees * std::acos(real(-1)) / 180;
                const real c = std::cos(angle), s = std::sin(angle);
                Eigen::Matrix3d rotation;
                rotation << c, -s, 0, s, c, 0, 0, 0, 1;
                mesh->ApplyGeometryTransform(rotation, settings.scale);
            }
            timings.meshPreparation = MPI_Wtime() - prepareStarted;
            return mesh;
        }
    }

    Simulation::Simulation(SimulationConfig config, MPI_Comm comm,
                           const MeshFactory &factory)
        : _config(std::move(config)),
          _placement(comm, _config.RankGroups()),
          _domains(size_t(_placement.DomainNumber())), _setupTimings(_domains.size())
    {
        // Opt-in before mesh/field setup. Do not disable diagnostics belonging
        // to another live simulation in this process.
        if (_config.Output().diagnostics.allocations)
            Runtime::AllocationDiagnostics::Enable();
        if (_config.Time().steady || _config.Time().method != "ssprk3")
        {
            const bool ditr = !_config.Time().steady && _config.Time().method == "ditr";
            const UpdatePolicyRequirements requirements{
                ditr ? 2 : 1,
                ditr ? NonlinearDetail::ResidualObservation::AfterUpdate
                     : NonlinearDetail::ResidualObservation::BeforeUpdate,
                ditr};
            // Resolve over every configured domain on every control rank,
            // including ranks with no local mesh. Membership must not decide
            // whether an optional control-communicator agreement is executed.
            for (const auto &domain : _config.Domains())
            {
                const auto controls = ditr
                                          ? _config.Time().ditr.ResolveControls(domain.settings.implicit)
                                          : domain.settings.implicit;
                auto policy = UpdatePolicyRegistry::Resolve(controls.updatePolicy, requirements);
                if (_updatePolicy)
                    DNDS_check_throw_info(
                        policy->Descriptor().Metadata() == _updatePolicy->Descriptor().Metadata(),
                        "coordinated domains require the same nonlinear update policy");
                else
                    _updatePolicy = std::move(policy);
                _nonlinearCollectiveChecks |= controls.collectiveChecks;
            }
        }
        bool domainProgressWriter = false;
        for (auto local : _placement.LocalDomains())
            domainProgressWriter |= _placement.DomainRank(local) == 0;
        auto setupLogger = std::make_shared<OperationalLogger>(
            _config.Output(), MPIWorldRank(), domainProgressWriter);
        const int requestedWorkers = OMP::DistributedThreads();
        const int observedWorkers = Diagnostics::SetupProgress::ObservedWorkers(requestedWorkers);
        const int ingressWorkers = OMP::IngressThreads();
        const int observedIngress = Diagnostics::SetupProgress::ObservedWorkers(ingressWorkers);
        for (auto iDomain : _placement.LocalDomains())
        {
            const auto &domain = _config.Domains()[iDomain];
            const MPIInfo mpi(_placement.Context(iDomain)->Communicator());
            // An injected/archived mesh already owns its geometry. Reject a
            // second preprocessing request rather than scaling it twice.
            DNDS_check_throw_info(!factory || (domain.settings.mesh.scale == 1 && domain.settings.mesh.rotationZDegrees == 0),
                                  "factory-injected meshes require identity mesh preprocessing");
            const std::string domainPrefix = _domains.size() > 1
                                                 ? "D[" + domain.placement.name + "]/"
                                                 : "";
            Diagnostics::SetupProgress progress(
                [setupLogger, requestedWorkers, observedWorkers,
                 ingressWorkers, observedIngress](const Diagnostics::SetupProgressEvent &event)
                {
                    const char *state = event.state == Diagnostics::SetupProgressState::Starting
                                            ? "starting"
                                        : event.state == Diagnostics::SetupProgressState::Done ? "done"
                                                                                               : "failed";
                    std::string suffix;
                    if (event.state == Diagnostics::SetupProgressState::Starting)
                    {
                        const bool ingress = event.phase.find("mesh/read") != std::string::npos ||
                                             event.phase.find("mesh/periodic") != std::string::npos ||
                                             event.phase.find("mesh/partition") != std::string::npos;
                        suffix = ingress
                                     ? fmt::format("workers reader {}/{}{}{}", ingressWorkers, observedIngress,
                                                   event.detail.empty() ? "" : "  ", event.detail)
                                     : fmt::format("workers rank {}/{}{}{}", requestedWorkers, observedWorkers,
                                                   event.detail.empty() ? "" : "  ", event.detail);
                    }
                    else
                        suffix = fmt::format("{:10.3f} s", event.seconds);
                    const auto line = fmt::format("=== Setup [{:<24}] {:<9} {}", event.phase, state, suffix);
                    setupLogger->Progress(line, line);
                });
            progress = progress.WithPrefix(domainPrefix);
            auto completeStage = progress.Stage("complete");
            const real started = MPI_Wtime();
            if (factory)
                OMP::set_full_parallel_OMP();
            auto mesh = factory ? factory(iDomain, domain, mpi)
                                : ReadMesh(domain, mpi,
                                           _placement.Context(iDomain), _setupTimings[iDomain], progress);
            if (factory)
                _setupTimings[iDomain].meshIngress = MPI_Wtime() - started;
            const real domainStarted = MPI_Wtime();
            _domains[iDomain] = std::make_unique<Domain>(
                domain.placement.name, std::move(mesh), domain.settings,
                _config.Time(), progress);
            _setupTimings[iDomain].domainSetup = MPI_Wtime() - domainStarted;
        }
        if (!_config.Interfaces().empty())
        {
            std::vector<std::optional<FiniteVolume>> local(_domains.size());
            std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>>
                workers(_domains.size());
            for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                if (_domains[iDomain])
                {
                    local[iDomain] = _domains[iDomain]->Discretization();
                    workers[iDomain] = _domains[iDomain]->ChemicalWorkers();
                }
            _interfaces = std::make_unique<InterfaceCoupling>(
                _config, _placement.ControlContext(), local, workers);

            // Establish accepted-time traces before CFL/diagnostics or the
            // first trial. Domain trial fields initially equal accepted means.
            // Separate packet allocations preserve these traces on rejection.
            const auto initialize = [&](bool prepared)
            {
                for (auto &domain : _domains)
                    if (domain)
                    {
                        auto &control = *domain;
                        control.PrepareInterface(0, prepared, 0);
                    }
                std::vector<InterfaceDomainInput> ports(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size());
                     ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        ports[iDomain] = {
                            domain->PublishInterfaceDonor(0, prepared)};
                _interfaces->Refresh(ports, 0, {});
                for (index iDomain = 0; iDomain < index(_domains.size());
                     ++iDomain)
                    if (_domains[iDomain])
                        if (const auto input = _interfaces->Inputs(iDomain))
                            _domains[iDomain]->BindInterfaceInputs(
                                *input);
            };
            initialize(false);
            initialize(true);
            _interfaces->Commit();
        }
    }

    Simulation::~Simulation() = default;

    InterfaceDiagnostics Simulation::GetInterfaceDiagnostics() const
    {
        return _interfaces ? _interfaces->Diagnostics()
                           : InterfaceDiagnostics{};
    }

    bool Simulation::Local(index iDomain) const
    {
        DNDS_check_throw_info(iDomain >= 0 &&
                                  iDomain < index(_domains.size()),
                              "invalid Cauchy domain index");
        return bool(_domains[iDomain]);
    }

    Domain &Simulation::GetDomain(index iDomain)
    {
        DNDS_check_throw_info(Local(iDomain),
                              "Cauchy domain is remote on this rank");
        return *_domains[iDomain];
    }

    const Domain &Simulation::GetDomain(index iDomain) const
    {
        DNDS_check_throw_info(Local(iDomain),
                              "Cauchy domain is remote on this rank");
        return *_domains[iDomain];
    }

}
