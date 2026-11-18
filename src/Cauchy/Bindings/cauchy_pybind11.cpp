#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <pybind11/stl.h>

PYBIND11_MODULE(cauchy_pybind11, m)
{
    namespace py = pybind11;
    namespace C = DNDS::Cauchy;
    namespace R = DNDS::Runtime;
    py::module_::import("DNDSR.CFV");
    // Register the mutually referenced type before emitting method signatures;
    // otherwise pybind records an unbound C++ spelling in Python annotations.
    auto domainClass = py::class_<C::Domain>(
        m, "Domain",
        "CPU perfect-gas or ideal-mixture domain with optional SA, Wilcox/SST or realizable k-epsilon. Collective methods use the mesh communicator. Missing RANS wall distance is built by Geom from no-slip zones. Release all debug views before construction/stepping; no CUDA fallback.");
    py::class_<C::StepResult>(
        m, "StepResult",
        "Collectively agreed numerical outcome; rejection permits smaller-dt retry. Unexpected failures terminate the run.")
        .def_readonly("accepted", &C::StepResult::accepted)
        .def_readonly("reason", &C::StepResult::reason)
        .def("__bool__", [](const C::StepResult &result)
    {
        return result.accepted;
    });
    py::class_<C::InterfaceDiagnostics>(
        m, "InterfaceDiagnostics",
        "Global coupling work counters; donor changes include nonmonotone stage revisits and rejected trials.")
        .def_readonly("query_points", &C::InterfaceDiagnostics::queryPoints)
        .def_readonly("changed_donors", &C::InterfaceDiagnostics::changedDonors)
        .def_readonly("map_builds", &C::InterfaceDiagnostics::mapBuilds)
        .def_readonly("maximum_gap", &C::InterfaceDiagnostics::maximumGap);
    py::enum_<C::DerivedQuantity>(m, "DerivedQuantity")
        .value("Temperature", C::DerivedQuantity::Temperature)
        .value("Pressure", C::DerivedQuantity::Pressure)
        .value("SoundSpeed", C::DerivedQuantity::SoundSpeed)
        .value("MolecularViscosity", C::DerivedQuantity::MolecularViscosity)
        .value("ThermalConductivity", C::DerivedQuantity::ThermalConductivity)
        .value("HeatCapacityP", C::DerivedQuantity::HeatCapacityP)
        .value("HeatCapacityV", C::DerivedQuantity::HeatCapacityV)
        .value("GasConstant", C::DerivedQuantity::GasConstant)
        .value("DiffusionBound", C::DerivedQuantity::DiffusionBound)
        .value("TotalEnthalpy", C::DerivedQuantity::TotalEnthalpy)
        .value("PressureInternalEnergyDerivative", C::DerivedQuantity::PressureInternalEnergyDerivative);
    py::class_<C::DerivedFields>(
        m, "DerivedFields",
        "Optional point properties; no equation components or automatic exchange. Shared arrays retain storage. Raw debug access does not establish freshness.")
        .def_property_readonly("values",
                               [](const C::DerivedFields &fields)
    {
        return R::PythonArray::Share(fields.Values());
    }, "Shared property table, with columns in quantities order. External writes invalidate publication.")
        .def_property_readonly("quantities",
                               [](const C::DerivedFields &fields)
    {
        std::vector<C::DerivedQuantity> quantities;
        for (int q = 0; q < C::DerivedQuantityNumber; ++q)
            if (fields.Request().Has(C::DerivedQuantity(q)))
                quantities.push_back(C::DerivedQuantity(q));
        return quantities;
    })
        .def_property_readonly("ghost_part", &C::DerivedFields::GhostPart)
        .def("current",
             [](const C::DerivedFields &fields, const C::Domain &domain)
    {
        return fields.Current(domain.Dofs(), domain.Equation());
    }, "Provenance snapshot; does not reserve access or refresh.")
        .def(
            "refresh",
            [](C::DerivedFields &fields, const C::Domain &domain, bool onlyIfStale)
    {
        return domain.RefreshDerived(fields, onlyIfStale);
    }, py::arg("domain"), py::arg("only_if_stale") = false, py::call_guard<py::gil_scoped_release>(),
            "Synchronous local pointwise refresh of all enabled properties using the domain worker. No MPI; ghost state must already be ready. Release conflicting views first.");
    py::class_<C::PhysicalStepLimit>(
        m, "PhysicalStepLimit",
        "Explicit physical-step PP result; caller advances with dt. Counts/alpha precede growth capping and retries.")
        .def_readonly("dt", &C::PhysicalStepLimit::dt)
        .def_readonly("limited_cells", &C::PhysicalStepLimit::limitedCells)
        .def_readonly("minimum_alpha", &C::PhysicalStepLimit::minimumAlpha)
        .def_readonly("growth_limited", &C::PhysicalStepLimit::growthLimited);
    py::class_<C::NonlinearIterationReport>(
        m, "NonlinearIterationReport",
        "Read-only snapshot of one implicit stage; norms include every logical equation component.")
        .def_readonly("converged", &C::NonlinearIterationReport::converged)
        .def_readonly("iterations", &C::NonlinearIterationReport::iterations)
        .def_readonly("linear_iterations", &C::NonlinearIterationReport::linearIterations)
        .def_readonly("reconstruction_iterations", &C::NonlinearIterationReport::reconstructionIterations)
        .def_readonly("reconstruction_relative_update", &C::NonlinearIterationReport::reconstructionRelativeUpdate)
        .def_readonly("multigrid_cycles", &C::NonlinearIterationReport::multigridCycles)
        .def_readonly("level_corrections", &C::NonlinearIterationReport::levelCorrections)
        .def_readonly("initial_residual", &C::NonlinearIterationReport::initialResidual)
        .def_readonly("reference_residual", &C::NonlinearIterationReport::referenceResidual)
        .def_readonly("residual_target_ratio", &C::NonlinearIterationReport::residualTargetRatio)
        .def_readonly("residual", &C::NonlinearIterationReport::residual)
        .def_readonly("rhs_limited", &C::NonlinearIterationReport::rhsLimited)
        .def_readonly("increment_limited", &C::NonlinearIterationReport::incrementLimited);
    C::Bindings::BindFiniteVolumeOperators(m);
    domainClass
        .def(
            py::init(
                [](std::shared_ptr<DNDS::Geom::UnstructuredMesh> mesh, const std::string &json, const std::string &name)
    {
        const auto mpi = mesh->getMPI();
        C::Json input;
        DNDS::MPI::LocalStage(mpi, [&]
        {
            input = C::Json::parse(json);
        });
        auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        DNDS_check_throw_info(config.Domains().size() == 1, "Python Domain constructor requires one resolved domain");
        DNDS_check_throw_info(config.Domains()[0].placement.ranks.size() == size_t(mpi.size),
                              "Python Domain requires every mesh-communicator rank");
        DNDS::MPI::LocalStage(mpi, [&]
        {
            DNDS_check_throw_info(!name.empty(), "Python Domain name is empty");
        });
        auto settings = config.Domains()[0].settings;
        return std::make_unique<C::Domain>(name, std::move(mesh), settings, config.Time());
    }),
            py::arg("mesh"), py::arg("config_json") = "{}", py::arg("name") = "fluid",
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("dofs",
                               [](const C::Domain &domain)
    {
        return domain.Dofs();
    }, "Shallow logical state handle. Arrays and row metadata are shared with the accepted state; clone for independent operator input. Release views and serialize mutation with Domain/Simulation work.")
        .def_property_readonly("fields",
                               [](C::Domain &domain)
    {
        std::map<std::string, R::PythonArray> result;
        const auto &state = domain.Dofs();
        for (DNDS::index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            result.emplace(state.Fields().Name(iField), R::PythonArray::Share(state.PhysicalField(iField).Owned()));
        return result;
    }, "Shared physical owned fields, including flow, RANS and species. Retained views follow normal lease/publication rules.")
        .def(
            "make_derived_fields",
            [](const C::Domain &domain, const std::vector<C::DerivedQuantity> &quantities, bool ghost)
    {
        C::DerivedRequest request;
        for (auto quantity : quantities)
        {
            const int q = int(quantity);
            DNDS_check_throw_info(q >= 0 && q < C::DerivedQuantityNumber, "unknown derived quantity");
            request.enabled[q] = true;
        }
        return C::DerivedFields(domain.Dofs().Space(), request, ghost);
    }, py::arg("quantities"), py::arg("ghost") = false,
            "Allocate independent property storage for this domain's rows; call refresh explicitly. This does not alter the domain's solved state or workspace selection.")
        .def_property_readonly("time", &C::Domain::Time)
        .def_property_readonly(
            "previous_dt", &C::Domain::PreviousTimeStep,
            "Last actual committed physical dt; zero for steady/fresh or unknown old restart history.")
        .def_property_readonly("steady", &C::Domain::Steady)
        .def_property_readonly("coupled_iterations", &C::Domain::LastCoupledIterations)
        .def_property_readonly(
            "coupled_target_ratio", &C::Domain::LastCoupledTargetRatio,
            "Selected-component ratio of the DITR I+H stopping residual after both coupled-stage updates; the same value appears in its single nonlinear report. Zero for other methods.")
        .def_property_readonly("step",
                               [](const C::Domain &domain)
    {
        return domain.Epoch().acceptedStep;
    })
        .def_property_readonly(
            "uses_strang_source", &C::Domain::UsesStrangSource,
            "Chemical domains apply the configured constant-volume source half-steps around source-free transport.")
        .def_property_readonly("face_fallbacks", &C::Domain::LastFaceFallbacks)
        .def_property_readonly("implicit_stages",
                               [](const C::Domain &domain)
    {
        return domain.LastImplicitStages();
    }, "Copied diagnostics from the latest implicit attempt; empty for SSPRK3. Inspect converged before interpreting a trial.")
        .def_property_readonly(
            "positivity", &C::Domain::LastPositivity,
            "Last accepted trial: reconstruction cell-stage activations, minimum beta, stage activations, minimum alpha, componentwise characteristic fallbacks.")
        .def("residual_fields",
             [](C::Domain &domain)
    {
        std::optional<C::EquationDofs> value;
        {
            py::gil_scoped_release release;
            value.emplace(domain.Residual().NumericAlias());
        }
        std::map<std::string, R::PythonArray> result;
        for (DNDS::index iField = 0; iField < value->PhysicalFieldNumber(); ++iField)
            result.emplace(value->Fields().Name(iField), R::PythonArray::Share(value->PhysicalField(iField).Owned()));
        return result;
    }, "Collective integrated residual fields; shared overwriteable scratch. Invalidates any prepared candidate; release views before the next collective operation.")
        .def(
            "advance", &C::Domain::Advance, py::arg("dt"), py::call_guard<py::gil_scoped_release>(),
            "Collective configured time method. A rejected StepResult certifies a domain-wide numerical rejection: discard_step and retry with smaller dt. Other exceptions must terminate the MPI run immediately (for example MPI.COMM_WORLD.Abort(1)); do not catch locally and continue to MPI/finalization. SSPRK keeps raw flow5/species increments and separate RANS repair. maxRetries belongs to the C++ frontend.")
        .def(
            "prepare_step", &C::Domain::PrepareStep, py::arg("dt"), py::call_guard<py::gil_scoped_release>(),
            "Prepare all stages without publication. Discard a failed trial before smaller-dt retry; release debug views before preparation and commit.")
        .def("commit_step", &C::Domain::CommitStep, py::call_guard<py::gil_scoped_release>())
        .def("discard_step", &C::Domain::DiscardStep)
        .def("stable_step", &C::Domain::StableStep, py::arg("cfl"), py::call_guard<py::gil_scoped_release>())
        .def(
            "limit_physical_step", &C::Domain::LimitPhysicalStep, py::arg("dt"),
            py::call_guard<py::gil_scoped_release>(),
            "Collective configured Euler PP/growth bound, without advancing state or history. All ranks pass the same positive dt and release conflicting views; advance with result.dt.")
        .def(
            "write_visualization", &C::Domain::WriteVisualization, py::arg("prefix"), py::arg("hdf_deflate_level") = 0,
            py::call_guard<py::gil_scoped_release>(),
            "Collective accepted-state volume and boundary VTKHDF output in code units, including all species fields. Uses separate observation reconstruction. Release debug views before calling; pass a common path/compression setting.")
        .def(
            "write_checkpoint",
            [](const C::Domain &domain, const std::string &path, const std::string &resolved, bool reconstruction)
    {
        C::Json input;
        DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
        {
            input = C::Json::parse(resolved);
        });
        domain.WriteCheckpoint(path, input, reconstruction);
    }, py::arg("path"), py::arg("resolved_json"), py::arg("include_reconstruction") = false,
            py::call_guard<py::gil_scoped_release>(),
            "Collective native checkpoint of all equation fields and BDF history; physical layouts are preserved. Optional reconstruction supports compatible warm continuation. Pass common metadata/path on the domain communicator.")
        .def(
            "read_checkpoint",
            [](C::Domain &domain, const std::string &path, const std::string &resolved)
    {
        C::Json input;
        DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
        {
            input = C::Json::parse(resolved);
        });
        domain.ReadCheckpoint(path, input);
    }, py::arg("path"), py::arg("resolved_json"), py::call_guard<py::gil_scoped_release>(),
            "Collective persistent-cell-ID restart, permitting redistribution and layout changes. Release all retained views, including species. Accepted means/history publish together; missing or incompatible reconstruction cold-starts.");
    py::class_<C::Simulation>(
        m, "Simulation",
        "Coordinated CPU domains. Every control rank calls advance; local meshes follow configured domain-rank order. Release debug views before publication. No process-failure recovery.")
        .def(py::init(
                 [](const std::vector<std::shared_ptr<DNDS::Geom::UnstructuredMesh>> &meshes, const DNDS::MPIInfo &mpi,
                    const std::string &json)
    {
        C::Json input;
        DNDS::MPI::LocalStage(mpi, [&]
        {
            input = C::Json::parse(json);
        });
        auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        DNDS::MPI::LocalStage(mpi, [&]
        {
            DNDS_check_throw_info(meshes.size() == config.Domains().size(),
                                  "one Python mesh slot is required for each configured domain");
            for (DNDS::index iDomain = 0; iDomain < DNDS::index(meshes.size()); ++iDomain)
            {
                const auto &ranks = config.Domains()[iDomain].placement.ranks;
                const bool local = std::find(ranks.begin(), ranks.end(), mpi.rank) != ranks.end();
                DNDS_check_throw_info(bool(meshes[iDomain]) == local,
                                      "Python Simulation mesh must be None exactly on domain nonmembers");
            }
        });
        return std::make_unique<C::Simulation>(std::move(config), mpi.comm,
                                               [&](DNDS::index iDomain, const auto &, const auto &domainMPI)
        {
            DNDS::MPI::LocalStage(domainMPI, [&]
            {
                int comparison = MPI_UNEQUAL;
                MPI_Comm_compare(meshes[iDomain]->getMPI().comm, domainMPI.comm, &comparison);
                DNDS_check_throw_info(comparison == MPI_IDENT || comparison == MPI_CONGRUENT,
                                      "Python mesh communicator differs from configured domain membership/order");
            });
            return meshes[iDomain];
        });
    }),
             py::arg("meshes"), py::arg("mpi"), py::arg("config_json") = "{}", py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("time", &C::Simulation::Time)
        .def_property_readonly("step", &C::Simulation::Step)
        .def("local", &C::Simulation::Local, py::arg("domain"))
        .def(
            "domain", static_cast<C::Domain &(C::Simulation::*)(DNDS::index)>(&C::Simulation::GetDomain),
            py::arg("domain"), py::return_value_policy::reference_internal,
            "Borrow a local domain, retaining Simulation. Advance through Simulation to keep clocks and interface snapshots coordinated.")
        .def(
            "advance", &C::Simulation::Advance, py::arg("dt"), py::arg("accepted_time") = py::none(),
            py::call_guard<py::gil_scoped_release>(),
            "Collective coordinated step with the same dt on all control ranks. accepted_time optionally publishes an exact snapped clock after acceptance. A rejected StepResult certifies an agreed numerical retry; other exceptions must reach a fail-fast MPI application boundary without another collective/finalization. No automatic maxRetries here.")
        .def(
            "write_checkpoint",
            [](const C::Simulation &simulation, const std::string &path, bool reconstruction,
               const py::object &outputController)
    {
        C::Json controller;
        if (!outputController.is_none())
        {
            const auto encoded = py::module_::import("json").attr("dumps")(outputController).cast<std::string>();
            controller = C::Json::parse(encoded);
        }
        py::gil_scoped_release release;
        simulation.WriteCheckpoint(path, reconstruction, controller);
    },
            py::arg("path"), py::arg("reconstruction") = false, py::arg("output_controller") = py::none(),
            "Collective coordinated checkpoint; multi-domain manifests retain per-domain native archives. Optional output-controller JSON preserves frontend cadence/history state. All ranks use the same path and release conflicting views.")
        .def(
            "read_checkpoint", &C::Simulation::ReadCheckpoint, py::arg("path"),
            py::call_guard<py::gil_scoped_release>(),
            "Collectively validate and reserve all domains before restoring accepted states, clocks and compatible history. Release debug views and serialize external mutation.")
        .def("write_visualization", &C::Simulation::WriteVisualization, py::arg("prefix"),
             py::arg("hdf_deflate_level") = 0, py::call_guard<py::gil_scoped_release>(),
             "Collective native domain output; multiple domains append _domainN to the prefix.")
        .def(
            "interface_diagnostics", &C::Simulation::GetInterfaceDiagnostics, py::call_guard<py::gil_scoped_release>(),
            "Collective query/rematch/map counts and maximum physical projection gap. Does not change the solution or traces.")
        .def_property_readonly("resolved_json", [](const C::Simulation &simulation)
    {
        return simulation.Config().Root().dump();
    });
}
