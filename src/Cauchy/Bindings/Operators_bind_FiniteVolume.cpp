#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Bindings/OperatorsBindInternal.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <pybind11/stl.h>

namespace DNDS::Cauchy::Bindings
{
    void BindOperatorFiniteVolume(py::module_ &m)
    {
        py::class_<FiniteVolumeOperators>(
            m, "FiniteVolumeOperators",
            "Configured native equation-state resources. Retains Domain/communicator and shares immutable FV plans plus worker capacity, not time history. Missing profiles still reject. Strang domains provide source-free transport.")
            .def(py::init<const Domain &>(), py::arg("domain"), py::keep_alive<1, 2>())
            .def_property_readonly("spatial_settings_json",
                                   [](const FiniteVolumeOperators &operators)
        {
            return Json(operators.fv.Settings()).dump();
        }, "Read-only resolved spatial controls. Source splitting is already reflected; these are not a new configuration to apply.")
            .def_property_readonly("metrics",
                                   [](const FiniteVolumeOperators &op) -> const CFV::FiniteVolumePlan &
        {
            return op.fv.Metrics();
        }, py::return_value_policy::reference_internal)
            .def("build_workspace",
                 [](const FiniteVolumeOperators &op, EquationDofs &u)
        {
            return Workspace{op.fv.BuildReconstructionState(u), op.fv.BuildWorkspace(u, op.workers)};
        }, py::arg("state"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>(),
                 Collective)
            .def("build_mean_workspace",
                 [](const FiniteVolumeOperators &op, EquationDofs &u)
        {
            return MeanWorkspace{op.fv.BuildMeanWorkspace(u, op.workers)};
        }, py::arg("state"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>(),
                 Collective)
            .def("build_source_derivative",
                 [](const FiniteVolumeOperators &op, const EquationDofs &state) -> std::optional<R::PythonArray>
        {
            const auto record = op.fv.BuildSourceDerivative(state);
            if (record)
                return R::PythonArray::Share(*record);
            return {};
        }, py::arg("state"),
                 "Allocate the cell derivative paired with one spatial rate. Centered linearization returns None.")
            .def("evaluate",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, EquationDofs &r, Workspace &w, R::EagerEngine &e,
                    const R::ExecutionStream &s, real t, const std::optional<R::PythonArray> &sourceDerivative)
        {
            auto writable = sourceDerivative ? std::optional(sourceDerivative->AsArray<real>()) : std::nullopt;
            py::gil_scoped_release release;
            op.fv.Evaluate(u, r, *w.reconstruction, *w.value, e, s, t,
                           writable ? OptionalRef<R::Array<real>>(*writable) : OptionalRef<R::Array<real>>{});
        },
                 py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::arg("source_derivative") = std::nullopt, Collective)
            .def(
                "build_face_flux_buffer",
                [](const FiniteVolumeOperators &op, const EquationDofs &u)
        {
            return R::PythonArray::Share(op.fv.BuildFaceFluxBuffer(u.ComponentNumber()));
        }, py::arg("state"),
                "Allocate caller-owned integrated negative outward face flux, in AuxiliaryFaces order and the local anchor cell frame. No source or halo.")
            .def("prepare_reconstruction",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, Workspace &w, R::EagerEngine &e,
                    const R::ExecutionStream &s, real t)
        {
            op.fv.PrepareReconstruction(u, *w.reconstruction, *w.value, e, s, t);
        },
                 py::arg("state"), py::arg("workspace"), py::arg("engine"), py::arg("stream"), py::arg("time") = 0,
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def(
                "evaluate_flux",
                [](const FiniteVolumeOperators &op, const EquationDofs &u, EquationDofs &r, Workspace &w,
                   R::PythonArray *faces, real t)
        {
            std::optional<R::Array<real>> buffer;
            if (faces)
                buffer.emplace(faces->AsArray<real>());
            py::gil_scoped_release release;
            op.fv.EvaluateFlux(u, r, *w.reconstruction, *w.value, CFV::ReconstructionRegion::All, t,
                               buffer ? OptionalRef<Runtime::Array<real>>(*buffer)
                                      : OptionalRef<Runtime::Array<real>>{});
        },
                py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("face_flux") = nullptr,
                py::arg("time") = 0,
                "Local flux-only overwrite using a completed evaluate publication at the same state/time. Input halos must be ready. Optional face array is shared and overwritten; release views first. No reconstruction or transfer. Caller coordinates failures.")
            .def(
                "add_source",
                [](const FiniteVolumeOperators &op, const EquationDofs &u, EquationDofs &r, Workspace &w, real t)
        {
            op.fv.AddSource(u, r, *w.reconstruction, *w.value, t);
        }, py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("time") = 0,
                py::call_guard<py::gil_scoped_release>(),
                "Local additive volume source using completed evaluate preparation at the same state/time. Includes rotation, RANS and chemistry. P0 retains physical source gradients. Output must be initialized and distinct; no halo exchange or reconstruction. Release views and coordinate failures.")
            .def("evaluate_mean",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, EquationDofs &r, MeanWorkspace &w,
                    R::EagerEngine &e, const R::ExecutionStream &s, real t)
        {
            op.fv.EvaluateMean(u, r, *w.value, e, s, t);
        },
                 py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def("reconstruction_guess",
                 [](const FiniteVolumeOperators &op, const Workspace &w)
        {
            return op.fv.ReconstructionGuess(*w.reconstruction);
        }, py::arg("workspace"), "Shared unlimited coefficient fields. Do not write during workspace/transfer use.")
            .def("flux_reconstruction",
                 [](const FiniteVolumeOperators &op, const Workspace &w)
        {
            return op.fv.FluxReconstruction(*w.reconstruction, *w.value);
        }, py::arg("workspace"))
            .def(
                "reset_reconstruction",
                [](const FiniteVolumeOperators &op, Workspace &w, std::optional<R::ExecutionStream> stream)
        {
            op.fv.ResetReconstruction(*w.reconstruction, *w.value, stream);
        }, py::arg("workspace"), py::arg("stream") = std::nullopt, py::call_guard<py::gil_scoped_release>(),
                "Local synchronous reset. CUDA requires its matching stream; CPU accepts an optional inline/worker stream. Reserves all destinations before writing. No halo exchange.")
            .def(
                "restore_reconstruction",
                [](const FiniteVolumeOperators &op, Workspace &w, const EquationDofs &u,
                   std::optional<R::ExecutionStream> stream)
        {
            op.fv.RestoreReconstructionGuess(*w.reconstruction, u, {}, stream);
        },
                py::arg("workspace"), py::arg("guess"), py::arg("stream") = std::nullopt,
                py::call_guard<py::gil_scoped_release>(),
                "Collective transactional owned-guess restore. Inputs must be distinct and already on the plan device. CUDA requires its matching stream. Validation finishes before destination writes; no payload host staging or halo exchange.")
            .def("build_auxiliary_values",
                 [](const FiniteVolumeOperators &op, AuxiliaryRequest request)
        {
            return op.fv.BuildAuxiliaryValues(request);
        }, py::arg("request"))
            .def("build_mean_gradients",
                 [](const FiniteVolumeOperators &op, EquationDofs &u)
        {
            return op.fv.BuildMeanGradients(u);
        }, py::arg("state"), py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>(), Collective)
            .def("prepare_aux_values",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, AuxiliaryValues &a, MeanGradients &g,
                    R::EagerEngine &e, const R::ExecutionStream &s, real t)
        {
            op.fv.PrepareAuxValues(u, a, g, e, s, t, op.workers);
        },
                 py::arg("state"), py::arg("output"), py::arg("gradients"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def("prepare_correction_aux",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, Workspace &w, AuxiliaryValues &a,
                    R::EagerEngine &e, const R::ExecutionStream &s, real t)
        {
            op.fv.PrepareCorrectionAux(u, *w.reconstruction, *w.value, a, e, s, t);
        },
                 py::arg("state"), py::arg("workspace"), py::arg("output"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def("validate_aux_values",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, const AuxiliaryValues &a, real t)
        {
            op.fv.ValidateAuxValues(u, a, t);
        }, py::arg("state"), py::arg("values"), py::arg("time") = 0, Local)
            .def("build_linear_problem",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, DiagonalStorage d, OffDiagonalStorage o,
                    const LinearMatrixOptions &storage)
        {
            return op.fv.BuildLinearProblem(u, d, o, storage);
        },
                 py::arg("state"), py::arg_v("diagonal", DiagonalStorage::Block, "DiagonalStorage.Block"),
                 py::arg_v("off_diagonal", OffDiagonalStorage::Assembled, "OffDiagonalStorage.Assembled"),
                 py::arg("storage") = LinearMatrixOptions{}, py::keep_alive<0, 1>(),
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def("prepare_jacobian",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, const AuxiliaryValues &a,
                    const R::PythonArray &inverseDt, LinearProblemContext &p, JacobianPreparation c, real t,
                    const Workspace *w, const std::optional<R::PythonArray> &sourceDerivative)
        {
            // Dtype resolution uses Python. Freeze the native sharing
            // handle before releasing the GIL for numerical execution.
            const auto native = inverseDt.AsArray<real>();
            const auto record = sourceDerivative ? std::optional(sourceDerivative->AsArray<real>()) : std::nullopt;
            auto context =
                w ? std::optional(FiniteVolume::JacobianContext{*w->reconstruction, *w->value}) : std::nullopt;
            if (record)
            {
                DNDS_check_throw_info(context, "a source derivative requires its source workspace");
                context->sourceDerivative.Emplace(*record);
            }
            py::gil_scoped_release release;
            op.fv.PrepareJacobian(u, a, native, p, c, t, op.workers, context);
        },
                 py::arg("state"), py::arg("auxiliary"), py::arg("inverse_pseudo_time"), py::arg("output"),
                 py::arg_v("controls", JacobianPreparation{}, "JacobianPreparation()"), py::arg("time") = 0,
                 py::arg("source_workspace") = nullptr, py::arg("source_derivative") = std::nullopt, Local)
            .def("build_linear_solver",
                 [](const FiniteVolumeOperators &op, std::shared_ptr<LinearProblemContext> p, const EquationDofs &u,
                    LinearSolveSettings settings, LinearExecution execution)
        {
            return std::make_unique<DofLinearSolver>(std::move(p), u, [fv = op.fv](Field &field)
            {
                return fv.BindHalo(field);
            }, settings, execution);
        },
                 py::arg("problem"), py::arg("prototype"),
                 py::arg_v("settings", LinearSolveSettings{}, "LinearSolveSettings()"),
                 py::arg_v("execution", LinearExecution{}, "LinearExecution()"), py::keep_alive<0, 1>(),
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def("build_stage_solver",
                 [](const FiniteVolumeOperators &operators, EquationDofs &state, Workspace &workspace,
                    R::EagerEngine &engine, const R::ExecutionStream &stream, const std::string &implicitJson)
        {
            auto solver = std::make_unique<NonlinearStageSolver>();
            const MPIInfo mpi(operators.fv.Metrics().Communicator());
            std::optional<Algebra::ComponentSelection> selection;
            MPI::LocalStage(mpi, [&]
            {
                // Reuse production schema/default/unknown-key validation.
                // Only implicit controls can be overridden; spatial and
                // physical settings come from the immutable native plan.
                Json settings = operators.fv.Settings();
                const auto overrides = Json::parse(implicitJson);
                DNDS_check_throw_info(overrides.is_object(), "implicit overrides must be a JSON object");
                settings["implicit"].merge_patch(overrides);
                solver->settings =
                    SimulationConfig::Resolve({{"defaults", settings}, {"time", {{"method", "bdf1"}}}}, mpi.size)
                        .Domains()
                        .front()
                        .settings;
                solver->owner.emplace(operators);
                solver->state.emplace(state);
                solver->stream = stream;
                selection.emplace(state.ComponentNumber(),
                                  state.Schema().ResolveComponents(solver->settings.implicit.ResolveResidualComponents(
                                      state.Schema().AccumulationKind() ==
                                      PhysicalAccumulationKind::ConstantDensityVelocityPressure)));
            });
            solver->value =
                std::make_unique<NonlinearSolver>(operators.fv, state, workspace.reconstruction, workspace.value,
                                                  engine, stream, solver->settings, *selection, operators.workers);
            return solver;
        },
                 py::arg("state"), py::arg("workspace"), py::arg("engine"), py::arg("stream"),
                 py::arg("implicit_json") = "{}", py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
                 py::keep_alive<0, 3>(), py::keep_alive<0, 4>(), py::call_guard<py::gil_scoped_release>(), Collective)
            .def("build_multigrid",
                 [](const FiniteVolumeOperators &operators, ssp<Geom::UnstructuredMesh> mesh,
                    NonlinearStageSolver &solver)
        {
            MPI::LocalStage(MPIInfo(operators.fv.Metrics().Communicator()), [&]
            {
                DNDS_check_throw_info(mesh && &operators.fv.Mesh() == mesh.get() && solver.owner &&
                                          solver.owner->fv.SharesPlan(operators.fv),
                                      "p-FAS requires its original mesh and finest operator owner");
            });
            auto engine = solver.stream.LockEngine();
            return std::make_unique<PolynomialMultigrid>(std::move(mesh), operators.fv, *solver.state, *solver.value,
                                                         engine, solver.stream, solver.settings, operators.workers);
        },
                 py::arg("mesh"), py::arg("stage_solver"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
                 py::keep_alive<0, 3>(), py::call_guard<py::gil_scoped_release>(), Collective)
            .def("write_visualization",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, const Workspace &w,
                    const std::string &prefix, real t, int deflate)
        {
            op.fv.WriteVisualization(u, *w.reconstruction, *w.value, prefix, t, deflate, op.workers);
        },
                 py::arg("state"), py::arg("workspace"), py::arg("prefix"), py::arg("time") = 0,
                 py::arg("hdf_deflate_level") = 0, py::call_guard<py::gil_scoped_release>(), Collective);
    }
}
