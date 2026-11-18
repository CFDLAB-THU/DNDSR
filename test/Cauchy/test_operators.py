"""Thin Python composition over the same native CPU finite-volume operators.

These are frontend/ownership checks. Independent physical and matrix oracles
remain in the C++ tests; matching wrappers alone are not physics validation.
"""
import gc
import json
import os
import weakref
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from DNDSR import DNDS, Geom
from DNDSR.DNDS import Runtime as R
from DNDSR import Cauchy as C


def block_values(state, physical_values=None):
    if physical_values is None:
        physical_values = {name: field.owned.numpy() for name, field in state.fields.items()}
    values = {}
    for block in state.blocks:
        name, first, width = state.locate_block(block)
        values[block] = physical_values[name][:, first:first + width]
    return values


def assert_state_equal(actual, expected, **tolerance):
    for name, values in block_values(actual).items():
        np.testing.assert_allclose(values, expected[name], **tolerance)


@pytest.mark.mpi
@pytest.mark.parametrize("mixture", [False, True])
def test_finite_volume_operator_frontend_composes_native_residual_and_linear_solve(mixture):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    root = Path(__file__).resolve().parents[2]
    mechanism = Path(os.environ.get("DNDS_MECH_PATH", root / "external/cfd_externals/install/data")) / "h2o2.yaml"
    if mixture and not mechanism.is_file():
        pytest.skip("installed h2o2 mechanism is unavailable")
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(root / "data/mesh/Uniform_3x3_wall.cgns"), 2))
    with mesh.acquire_read(faces=True) as read:
        local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*MPI.COMM_WORLD.allgather(local_zones))
    for threads, overlap in [(0, False), (2, True)]:
        config = {"defaults": {
            "execution": {"layout": "soa", "hostThreads": threads, "overlap": overlap,
                          "transferPreflight": "abortOnFailure" if overlap else "collective"},
            "initialPrimitive": [.8, .1, -.02, .03, 1],
            "numerics": {"flux": "rusanov", "faceFallback": False,
                         "reconstruction": "variational", "cfv": {"maxOrder": 3, "intOrder": 6},
                         "reconstructionIterations": 4, "reconstructionTolerance": 0.,
                         "limiter": "WBAP", "characteristicLimiter": False, "positivity": True},
            "boundaries": {str(zone): {"kind": "extrapolate"} for zone in zones},
        }, "time": {"method": "bdf1"}}
        if mixture:
            config["defaults"].update({
                "equation": {"eos": "idealGasMixture", "source": "chemical", "transport": "NewtonianFourier",
                             "mixture": {"mechanismFile": str(mechanism), "rho0": 1.225, "U0": 340, "L0": 2, "T0": 288.15},
                             "rans": {"model": "SA", "viscosityReference": .01}},
                "initialTemperature": 1100 / 288.15,
                "initialMassFractions": {"H2": .027, "O2": .22, "H2O": .02},
                "initialRANS": {"rhoNuTilde": .02},
                "boundaries": {str(zone): {"kind": "noSlipWall"} for zone in zones},
            })
            config["defaults"]["numerics"].update({"viscousGradient": "polynomial", "ransState": {
                "referenceConservative": {"rho": .8, "rhoNuTilde": .02}}})
            if overlap:
                config["defaults"]["implicit"] = {"sourceJacobianMode": "rhsQuadrature"}
        domain = C.Domain(mesh, json.dumps(config))
        operators = C.FiniteVolumeOperators(domain)
        spatial = json.loads(operators.spatial_settings_json)
        assert spatial["numerics"]["reconstruction"] == "variational"
        assert spatial["numerics"]["cfv"]["maxOrder"] == 3
        prototype = domain.dofs
        state = prototype.clone()
        if mixture:
            # Preserve full parent AoSoA pitch with SA before flow; species have
            # independent padded column-major storage and share the row route.
            algebra = R.Algebra
            packed = algebra.DoubleDofField(state.space, R.RowLayout.tiled_col_major(0, 6, 32))
            species_width = len(state.blocks["species"])
            species = algebra.DoubleDofField(state.space, R.RowLayout.col_major(0, species_width, 16, 1))
            # Registry order deliberately differs from the sorted fields dict.
            state = C.EquationDofs.from_fields(prototype, [("zPacked", packed), ("aSpecies", species)],
                                             [("rans", "zPacked", 0), ("flow", "zPacked", 1), ("species", "aSpecies", 0)])
            assert state.field_names == ["zPacked", "aSpecies"]
            state.copy_from(prototype)
        fields = state.fields
        for i_cell in range(state.space.owned_rows):
            x, y, _ = operators.metrics.GetCellCent(i_cell)
            scale = 1 + .03 * np.cos(x + .7 * y)
            for field in fields.values():
                for j in range(field.components):
                    field.owned[i_cell, j] *= scale
        prototype.copy_from(state)
        # The default Domain uses one physical table for the mixture. Compare
        # logical blocks against the explicitly split/permuted operator state.
        reference = block_values(prototype, {name: array.numpy() for name, array in domain.residual_fields().items()})
        owner = weakref.ref(domain)
        del domain
        gc.collect()
        assert owner() is not None  # Operator retains the Domain/communicator.
        assert state.resolve_components(["flow.rho"]) == [0]
        engine = R.EagerEngine(threads, R.AccessTrackingContext())
        stream = engine.cpu_stream()
        work = operators.build_workspace(state)
        residual = state.zeros_like()
        before = block_values(state)
        with pytest.raises(RuntimeError):
            operators.evaluate(state, state, work, engine, stream)
        assert_state_equal(state, before, rtol=0, atol=0)
        derivative = operators.build_source_derivative(state)
        assert (derivative is not None) == (mixture and overlap)
        operators.evaluate(state, residual, work, engine, stream, source_derivative=derivative)
        assert_state_equal(residual, reference, rtol=3e-10, atol=2e-11)
        flux = state.zeros_like()
        faces = operators.build_face_flux_buffer(state)
        operators.evaluate_flux(state, flux, work, face_flux=faces)
        operators.add_source(state, flux, work)
        assert_state_equal(flux, block_values(residual), rtol=3e-10, atol=2e-11)
        assert np.all(np.isfinite(faces.numpy()))
        assert_state_equal(state, before, rtol=0, atol=0)
        signal = sum(np.sum(np.abs(v)) for v in block_values(residual).values())
        assert MPI.COMM_WORLD.allreduce(signal, op=MPI.SUM) > 1e-6

        request = C.AuxiliaryRequest()
        request.mean_thermodynamics = True
        request.face_spectral_radius = True
        request.cell_spectral_sum = True
        request.derived_quantities = [C.DerivedQuantity.Temperature]
        auxiliary = operators.build_auxiliary_values(request)
        operators.prepare_correction_aux(state, work, auxiliary, engine, stream)
        operators.validate_aux_values(state, auxiliary)
        with pytest.raises(RuntimeError):
            operators.validate_aux_values(state, auxiliary, time=1.)
        # The selected derived bundle supplies thermodynamics without retaining
        # another duplicate p/a/diffusivity table.
        assert auxiliary.mean_thermodynamics is None
        assert auxiliary.face_wave_jump is None
        assert auxiliary.derived_owned is not None
        storage = C.OffDiagonalStorage.SemiMatrixFree if mixture else C.OffDiagonalStorage.Assembled
        matrix_storage = C.LinearMatrixOptions()
        matrix_storage.diagonal.layout = C.MatrixTableLayout.SoA
        matrix_storage.inverse.layout = C.MatrixTableLayout.AoSoA32
        matrix_storage.off_diagonal.layout = C.MatrixTableLayout.AoSoA32
        matrix_storage.indices = C.LinearIndexStorage.Int32
        problem = operators.build_linear_problem(state, off_diagonal=storage, storage=matrix_storage)
        inverse_dt = R.Array(R.RowLayout.dense(state.space.owned_rows, 1), np.dtype("float64"))
        for i_cell in range(state.space.owned_rows):
            inverse_dt[i_cell, 0] = 1e4
        controls = C.JacobianPreparation()
        controls.physical_mass = 10
        operators.prepare_jacobian(state, auxiliary, inverse_dt, problem, controls, source_workspace=work,
                                   source_derivative=derivative)
        problem.invert_diagonal()
        exact = state.zeros_like()
        for i_field, name in enumerate(exact.field_names):
            values = exact.fields[name].owned
            for i_cell in range(exact.space.owned_rows):
                x, y, _ = operators.metrics.GetCellCent(i_cell)
                for j in range(values.shape[1]):
                    values[i_cell, j] = .01 * (1 + i_field + j) * (1 + .1 * np.sin(x + y))
        # The action is a low-level ready-halo operator. Python explicitly uses
        # the canonical native bindings before calling it.
        halo = [operators.metrics.BindHalo(exact.fields[name]) for name in exact.field_names]
        transfers = [binding.begin() for binding in halo]
        for transfer in transfers:
            transfer.wait()
        rhs = exact.zeros_like()
        problem.apply(exact, rhs)
        expected = block_values(exact)
        problem.prepare_ilu(ordering="rcm")
        for method in ["sgs", "gmres", "ilu"]:
            settings = C.LinearSolveSettings()
            settings.method = method
            settings.preconditioner = "sgs"
            settings.ilu_ordering = "rcm" if method == "ilu" else "natural"
            settings.max_iterations = 40
            settings.relative_tolerance = 1e-11
            settings.absolute_tolerance = 0.
            execution = C.LinearExecution()
            execution.overlap = overlap
            solver = operators.build_linear_solver(problem, state, settings, execution)
            report = solver.solve(rhs)
            assert report.converged and report.usable, (method, report.iterations, report.true_residual)
            assert_state_equal(solver.solution, expected, rtol=2e-9, atol=1e-11)
            if method == "ilu" and not mixture:
                settings.stopping = "fixedCount"
                fixed_solver = operators.build_linear_solver(problem, state, settings, execution)
                fixed = fixed_solver.solve(rhs)
                assert fixed.stop == C.LinearStop.FixedCount and fixed.usable and not fixed.converged
                assert fixed.iterations == settings.max_iterations and fixed.inner_product_batches == 0
                assert np.isnan(fixed.initial_true_residual) and np.isnan(fixed.true_residual)
                assert_state_equal(fixed_solver.solution, expected, rtol=2e-9, atol=1e-11)
                del fixed_solver
        # Compare unsplit SGS with its public staged scan, while actual MPI
        # requests own the separate input ghosts during core work.
        old, forward, unsplit, staged = [state.zeros_like() for _ in range(4)]
        problem.apply_sgs(rhs, old, forward, unsplit)
        old_halo = [operators.metrics.BindHalo(old.fields[name]) for name in old.field_names]
        active = [binding.begin() for binding in old_halo]
        # A zero-start scan has no old input at all. It can run while an
        # unrelated old buffer's ghosts are written; no access lease is taken.
        iteration = problem.begin_sgs_zero(rhs, forward, staged)
        iteration.lower(C.LinearRegion.Core)
        for transfer in active:
            transfer.wait()
        iteration.lower(C.LinearRegion.HaloDependent)
        iteration.upper(C.LinearRegion.HaloDependent)
        iteration.upper(C.LinearRegion.Core)
        iteration.finish()
        assert_state_equal(staged, block_values(unsplit), rtol=0, atol=0)
        source = state.clone()
        source.scale(1e6)
        stage_solver = operators.build_stage_solver(state, work, engine, stream, json.dumps({
            "iterations": 1, "minIterations": 1,
            "linear": {"method": "sgs", "iterations": 2},
            "multigrid": {"levels": 1, "preSmooth": 1, "postSmooth": 1, "coarseSmooth": 1}}))
        correction = stage_solver.correct_once(source, inverse_dt=1e6, rhs_scale=1.)
        assert correction.iterations == 1
        assert not correction.converged  # CorrectOnce does not decide convergence.
        multigrid = operators.build_multigrid(mesh, stage_solver)
        cycle = multigrid.apply_cycle(source, inverse_dt=1e6, rhs_scale=1.)
        assert cycle.multigrid_cycles == 1
        assert len(cycle.level_corrections) == 2 and sum(cycle.level_corrections) > 0
        changed = 0.
        for name, values in block_values(state).items():
            assert np.all(np.isfinite(values))
            changed = max(changed, np.max(np.abs(values - before[name]), initial=0.))
        assert MPI.COMM_WORLD.allreduce(changed, op=MPI.MAX) > 1e-14
        assert_state_equal(prototype, before, rtol=0, atol=0)
        engine.synchronize()
        retained = state.fields[state.field_names[0]].owned
        snapshot = retained.numpy()
        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            stage_solver.correct_once(source, inverse_dt=1e6, rhs_scale=1.)
        del state, prototype, operators, work, solver, problem, stage_solver, multigrid
        gc.collect()
        assert owner() is None
        np.testing.assert_array_equal(retained.numpy(), snapshot)
