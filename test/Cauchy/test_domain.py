"""Cauchy frontend on the shared native mesh and Runtime Array backend.

Build/install cauchy_pybind11 and its dependencies before running under MPI.
These tests exercise the real residual/SSPRK3 path, not the placement mock.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS, Geom
from DNDSR.DNDS import Runtime as R
from DNDSR.Cauchy import Domain, Simulation, StepResult, PhysicalStepLimit, DerivedQuantity, advance_domain


@pytest.mark.mpi
@pytest.mark.parametrize("model", ["SA", "KOmegaSST"])
def test_configured_rans_fields_wall_setup_implicit_restart_and_output(tmp_path, model):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    root = Path(__file__).resolve().parents[2]
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(root / "data/mesh/Uniform_3x3_wall.cgns"), 2))
    with mesh.acquire_read(faces=True) as read:
        local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*MPI.COMM_WORLD.allgather(local_zones))
    config = {"defaults": {
        "equation": {"transport": "NewtonianFourier", "viscosity": .001,
                     "rans": {"model": "SA", "viscosityReference": .01}},
        "initialPrimitive": [1, 0, 0, 0, 1],
        "initialRANS": {"rhoNuTilde": .02},
        "execution": {"layout": "soa", "fieldGrouping": "equationBlocks", "hostThreads": 2, "overlap": True},
        "numerics": {"reconstruction": "constant", "limiter": "none", "faceFallback": False,
                     "viscousGradient": "polynomial", "positivity": True},
        # Unit-density BDF residual subtracts O(1/dt)=1e6 terms: allow a
        # few ulps of that arithmetic, while requiring every selected scalar.
        "implicit": {"diagonal": "block", "absoluteTolerance": 1e-9,
                     "residualComponents": ["rho", "rhoNuTilde"],
                     "linear": {"method": "gmres", "preconditioner": "ilu"}},
        "boundaries": {str(zone): {"kind": "noSlipWall"} for zone in zones},
    }, "time": {"method": "bdf1", "positivity": {"enabled": True}}}
    initial = {"rhoNuTilde": .02} if model == "SA" else {"rhoK": .02, "rhoOmega": .1}
    config["defaults"]["equation"]["rans"].update({
        "model": model, "useQCR": True, "desScale": .65, "desMode": 3})
    config["defaults"]["initialRANS"] = initial
    config["defaults"]["numerics"]["ransState"] = {
        "referenceConservative": {"rho": config["defaults"]["initialPrimitive"][0], **initial}}
    config["defaults"]["implicit"]["residualComponents"] = ["rho", *initial]
    if model == "SA":
        config["defaults"]["numerics"].update({
            "reconstruction": "variational", "cfv": {"maxOrder": 3},
            "reconstructionIterations": 4, "reconstructionTolerance": 0.0,
            "limiter": "WBAP", "characteristicLimiter": False})
        config["defaults"]["implicit"]["multigrid"] = {"levels": 1}
    domain = Domain(mesh, json.dumps(config))
    fields = domain.fields
    assert set(fields) == {"flow", "rans"}
    for column, value in enumerate(initial.values()):
        np.testing.assert_array_equal(fields["rans"].numpy()[:, column], value)
    assert domain.stable_step(.5) > 0
    assert domain.limit_physical_step(1e-6).dt == 1e-6
    residual = domain.residual_fields()
    assert set(residual) == set(fields)
    assert all(np.all(np.isfinite(array.numpy())) for array in residual.values())
    assert domain.advance(1e-6).accepted
    for report in domain.implicit_stages:
        assert len(report.residual) == 5 + len(initial)
        assert report.converged, (report.iterations, report.residual_target_ratio,
                                  report.initial_residual, report.residual)
        if model == "SA":
            assert report.multigrid_cycles > 0
    values = fields["rans"].numpy()
    assert np.all(values >= 0)
    changed = np.max(np.abs(values - np.array(list(initial.values()))), initial=0.)
    assert MPI.COMM_WORLD.allreduce(changed, op=MPI.MAX) > 0
    saved = {name: array.numpy().copy() for name, array in fields.items()}
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "sa.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, json.dumps(config))
    assert domain.advance(1e-6).accepted
    domain.read_checkpoint(path, json.dumps(config))
    for name, array in fields.items():
        np.testing.assert_array_equal(array.numpy(), saved[name])
    domain.write_visualization(path + ".output")
    import h5py

    with h5py.File(path + ".output.vtkhdf", "r") as output:
        for component in initial:
            assert component in output["VTKHDF/CellData"]
            assert np.all(np.isfinite(output[f"VTKHDF/CellData/{component}"][:]))


@pytest.mark.mpi
@pytest.mark.parametrize("rans_model", ["none", "SA", "KOmegaSST"])
def test_configured_mixture_fields_implicit_step_and_native_worker_refresh(tmp_path, rans_model):
    import os

    rans = rans_model != "none"
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    root = Path(__file__).resolve().parents[2]
    mechanism = Path(os.environ.get("DNDS_MECH_PATH", root / "external/cfd_externals/install/data")) / "h2o2.yaml"
    if not mechanism.is_file():
        pytest.skip("installed h2o2 mechanism is unavailable")
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(root / "data/mesh/Uniform_3x3_wall.cgns"), 2))
    with mesh.acquire_read(faces=True) as read:
        zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*MPI.COMM_WORLD.allgather(zones))
    config = {"defaults": {
        "equation": {"eos": "idealGasMixture", "source": "chemical", "mixture": {
            "mechanismFile": str(mechanism), "rho0": 1.225, "U0": 340, "L0": 2, "T0": 288.15}},
        "initialPrimitive": [.8, 0, 0, 0, 1], "initialTemperature": 1400 / 288.15,
        "initialMassFractions": {"H2": .027, "O2": .22, "H2O": .02},
        "execution": {"layout": "soa", "fieldGrouping": "equationBlocks", "hostThreads": 2, "overlap": True},
        "numerics": {"reconstruction": "constant", "faceFallback": False, "positivity": True},
        # RHS PP tests residual*dTau, as in Euler. A huge pseudo-CFL makes
        # that hypothetical explicit chemistry update inadmissible even when
        # the physical implicit step is tiny; use the ordinary CFL ramp here.
        "implicit": {"diagonal": "block", "pseudoCFL": 10, "maxPseudoCFL": 1e4, "absoluteTolerance": 1e-8,
                     "residualComponents": ["rhoY:H2", "rhoY:H2O"],
                     "linear": {"method": "ilu", "iterations": 6, "relativeTolerance": 0, "absoluteTolerance": 0}},
        "boundaries": {str(zone): {"kind": "extrapolate"} for zone in zones},
    }, "time": {"method": "bdf1", "positivity": {"enabled": True}}}
    if rans:
        config["defaults"]["equation"].update({"transport": "NewtonianFourier", "rans": {"model": rans_model, "viscosityReference": .01}})
        initial_rans = {"rhoNuTilde": .02} if rans_model == "SA" else {"rhoK": .02, "rhoOmega": .1}
        config["defaults"]["initialRANS"] = initial_rans
        config["defaults"]["numerics"]["ransState"] = {
            "referenceConservative": {"rho": config["defaults"]["initialPrimitive"][0], **initial_rans}}
        config["defaults"]["numerics"]["viscousGradient"] = "polynomial"
    if rans_model == "KOmegaSST":
        # Exercise the combined flow/RANS/species physical input and full
        # reference through production PP; other cases retain old input forms.
        settings = config["defaults"]
        rho = settings["initialPrimitive"][0]
        physical = [rho * 1.225, 0, 0, 0, 1400,
                    initial_rans["rhoK"] / rho * 340**2,
                    initial_rans["rhoOmega"] / rho * 340 / 2,
                    .027, 0, 0, .22, 0, .02, 0, 0, 0]
        settings["initialState"] = {"type": "primRhoT_phy", "state": physical}
        settings["referenceState"] = settings["initialState"]
        for key in ("initialPrimitive", "initialTemperature", "initialRANS", "initialMassFractions"):
            del settings[key]
        del settings["numerics"]["ransState"]
    try:
        domain = Domain(mesh, json.dumps(config))
    except RuntimeError as error:
        if "requires Cantera" in str(error):
            pytest.skip("Cantera is disabled in this build")
        raise
    fields = domain.fields
    assert set(fields) == ({"flow", "rans", "species"} if rans else {"flow", "species"})
    before_flow, before_species = fields["flow"].numpy().copy(), fields["species"].numpy().copy()
    residual = domain.residual_fields()
    assert set(residual) == set(fields)
    derived = domain.make_derived_fields([DerivedQuantity.Temperature, DerivedQuantity.Pressure])
    assert derived.refresh(domain)
    assert derived.current(domain)
    assert domain.stable_step(.5) > 0
    assert domain.limit_physical_step(1e-6).dt == 1e-6
    np.testing.assert_array_equal(fields["flow"].numpy(), before_flow)
    np.testing.assert_array_equal(fields["species"].numpy(), before_species)
    assert domain.advance(1e-6).accepted
    np.testing.assert_allclose(fields["flow"].numpy(), before_flow, rtol=1e-11, atol=1e-12)
    local_change = np.max(np.abs(fields["species"].numpy() - before_species), initial=0.)
    assert MPI.COMM_WORLD.allreduce(local_change, op=MPI.MAX) > 1e-10
    for report in domain.implicit_stages:
        assert len(report.residual) == 14 + (len(initial_rans) if rans else 0)
        if not report.converged:
            print("mixture stage", report.iterations, report.linear_iterations, report.residual_target_ratio,
                  report.rhs_limited, report.increment_limited, report.initial_residual, report.residual)
        assert report.converged, (report.iterations, report.residual_target_ratio,
                                  report.initial_residual, report.residual)
    if rans_model == "SA":
        np.testing.assert_allclose(fields["rans"].numpy(), .02, rtol=1e-11, atol=1e-12)
    elif rans:
        assert np.all(np.isfinite(fields["rans"].numpy()))
        assert np.all(fields["rans"].numpy() >= 0)
    assert not derived.current(domain)
    assert derived.refresh(domain)
    assert derived.current(domain)
    saved = {name: array.numpy().copy() for name, array in fields.items()}
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "mixture.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, json.dumps(config))
    assert domain.advance(1e-6).accepted
    domain.read_checkpoint(path, json.dumps(config))
    assert domain.time == 1e-6 and domain.step == 1
    for name, array in fields.items():
        np.testing.assert_array_equal(array.numpy(), saved[name])
    domain.write_visualization(path + ".output")
    import h5py

    with h5py.File(path + ".output.vtkhdf", "r") as output:
        values = output["VTKHDF/CellData"]
        assert "rhoY:H2" in values and "Y:N2" in values and "T" in values
        assert np.all(np.isfinite(values["T"][:]))
    if rans_model == "none":
        # Exercise the same chemistry through physical Strang and CPU workers.
        # The C++ test supplies the independent nonconstant composition oracle.
        split_config = json.loads(json.dumps(config))
        split_config["time"].update({"method": "esdirk2", "sourceSplitting": "strang"})
        split = Domain(mesh, json.dumps(split_config))
        assert split.uses_strang_source
        split_fields = split.fields
        initial_split_flow = split_fields["flow"].numpy()
        initial_split_species = split_fields["species"].numpy()
        assert split.advance(1e-6).accepted
        np.testing.assert_allclose(split_fields["flow"].numpy(), initial_split_flow, rtol=1e-11, atol=1e-12)
        change = np.max(np.abs(split_fields["species"].numpy() - initial_split_species), initial=0.)
        assert MPI.COMM_WORLD.allreduce(change, op=MPI.MAX) > 1e-10
        assert all(report.converged for report in split.implicit_stages)
        split_path = path + ".strang"
        split.write_checkpoint(split_path, json.dumps(split_config))
        assert split.advance(1e-6).accepted
        expected = {name: array.numpy() for name, array in split_fields.items()}
        split.read_checkpoint(split_path, json.dumps(split_config))
        assert split.advance(1e-6).accepted
        for name, array in split_fields.items():
            np.testing.assert_array_equal(array.numpy(), expected[name])


def build_domain(layout="aos", flux="Roe", workers=0, method="ssprk3", steady=False, time_pp=False, lvr_variant=None, tagged=False, initial_field=None, device="cpu"):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns"
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(path), 2))
    # Zone names/values come from the actual fixture, not rank-local assumptions.
    with mesh.acquire_read(faces=True) as read:
        local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*MPI.COMM_WORLD.allgather(local_zones))
    primitive = [1.2, .3, -.2, .4, 1.1]
    config = {"defaults": {
        "execution": {"layout": layout, "hostThreads": workers},
        "numerics": {"flux": flux, "cfv": {"maxOrder": 1}},
        "initialPrimitive": primitive,
        "boundaries": {str(zone): {"kind": "farfield", "primitive": primitive} for zone in zones},
    }, "time": {"method": method, "steady": steady, "positivity": {"enabled": time_pp}}}
    if device == "cuda":
        config["defaults"]["execution"].update({"device": device, "fieldGrouping": "coupled"})
        config["defaults"]["implicit"] = {"linear": {"method": "jacobi"}}
    if tagged:
        config["defaults"].pop("initialPrimitive")
        config["defaults"]["equation"] = {
            "referenceScales": {"rho0": 2, "U0": 3, "L0": 4, "T0": 5}}
        state = {"type": "primRhoP_phy", "state": [2.4, .9, -.6, 1.2, 19.8]}
        config["defaults"]["initialState"] = state
        config["defaults"]["referenceState"] = state
        config["defaults"]["boundaries"] = {
            str(zone): {"kind": "farfield", "state": state} for zone in zones}
    if initial_field is not None:
        config["defaults"]["initialField"] = initial_field
    if method == "ditr":
        config["time"]["ditr"] = {"minIterations": 1}
    if lvr_variant is not None:
        config["defaults"]["numerics"].update({
            "reconstruction": "variational", "cfv": {"maxOrder": 3},
            "limiter": "LVR", "limitedVR": {"variant": lvr_variant},
            "positivity": True, "faceFallback": False,
        })
    return Domain(mesh, json.dumps(config)), primitive


@pytest.mark.mpi
def test_tagged_physical_inputs_match_code_unit_frontend():
    physical, _ = build_domain(layout="soa", workers=2, tagged=True)
    code, _ = build_domain(layout="soa", workers=2)
    np.testing.assert_allclose(physical.fields["flow"].numpy(), code.fields["flow"].numpy(), rtol=2e-14, atol=2e-14)
    assert physical.advance(.001).accepted
    assert code.advance(.001).accepted
    np.testing.assert_allclose(physical.fields["flow"].numpy(), code.fields["flow"].numpy(), rtol=2e-14, atol=2e-14)


@pytest.mark.mpi
def test_spatial_expression_uses_physical_state_units_and_conservative_output():
    field = {"exprtkInitializers": [{"stateType": "primRhoP_phy", "exprs": [
        "inRegion := 1; UExprtk[1] := UExprtk[1] + 0.3; 0;"
    ]}]}
    domain, primitive = build_domain(layout="aosoa32", tagged=True, initial_field=field)
    rho, u, v, w, p = primitive
    u += .1  # Physical velocity increment .3 / U0=3.
    expected = np.array([rho, rho*u, rho*v, rho*w, p/.4 + .5*rho*(u*u+v*v+w*w)])
    values = domain.fields["flow"].numpy()
    np.testing.assert_allclose(values, np.broadcast_to(expected, values.shape), rtol=2e-13, atol=2e-13)


@pytest.mark.mpi
@pytest.mark.parametrize("split", [False, True])
def test_coordinated_simulation_python_owners_and_reversed_subgroups(split, tmp_path):
    import gc

    world = MPI.COMM_WORLD
    if split and world.size == 1:
        pytest.skip("split-domain fixture needs at least two ranks")
    comm = world.Split(world.rank % 2, -world.rank) if split else world
    mesh_mpi = DNDS.MPIInfo(comm.handle)
    control = DNDS.MPIInfo(world.handle)
    path = Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns"
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mesh_mpi, str(path), 2))
    with mesh.acquire_read(faces=True) as read:
        local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*world.allgather(local_zones))
    ranks = [list(reversed([r for r in range(world.size) if r % 2 == side])) if split
             else list(range(world.size)) for side in range(2)]
    meshes = [mesh if not split or world.rank % 2 == side else None for side in range(2)]
    config = {"domains": [{"name": str(side), "ranks": ranks[side]} for side in range(2)],
              "defaults": {"numerics": {"reconstruction": "constant"},
                           "boundaries": {str(zone): {"kind": "farfield"} for zone in zones}},
              "time": {"method": "bdf2"}}
    simulation = Simulation(meshes, control, json.dumps(config))
    assert simulation.advance(.001).accepted
    assert simulation.advance(.001).accepted
    assert simulation.step == 2
    assert simulation.interface_diagnostics().query_points == 0
    local = world.rank % 2 if split else 1
    assert simulation.local(local)
    domain = simulation.domain(local)
    assert domain.time == simulation.time
    archive = world.bcast(str(tmp_path / "simulation.h5") if world.rank == 0 else None, root=0)
    shared_flow = domain.fields["flow"]
    simulation.write_checkpoint(archive, reconstruction=True,
                                output_controller={"schema": "python-test"})
    assert simulation.advance(.001).accepted
    simulation.read_checkpoint(archive)
    assert simulation.step == domain.step == 2
    assert simulation.time == domain.time == .002
    # Restoring preserves the original Domain/accepted allocation, including
    # handles retained across read. The C++ test uses nonconstant BDF states.
    np.testing.assert_array_equal(shared_flow.numpy(), domain.fields["flow"].numpy())
    simulation.write_visualization(str(Path(archive).with_suffix("")))
    if world.rank == 0:
        for i_domain in range(2):
            assert Path(archive).with_suffix("").with_name(f"simulation_domain{i_domain}.vtkhdf").is_file()
    del shared_flow
    # The receiver zone exists, but the donor selection does not. This reaches
    # the patch guard rather than the earlier unused-BC configuration check.
    bad = json.loads(json.dumps(config))
    receiver_zone = min(zones)
    bad["defaults"]["boundaries"][str(receiver_zone)] = {"kind": "interface"}
    bad["interfaces"] = [{"donor": str(side), "receiver": str(1-side),
                          "donorZone": 999, "receiverZone": receiver_zone} for side in range(2)]
    with pytest.raises(RuntimeError, match="globally empty"):
        Simulation(meshes, control, json.dumps(bad))
    # reference_internal retains the Simulation, while a shared native Array
    # independently retains its allocation after both frontend handles vanish.
    flow = domain.fields["flow"]
    del simulation
    assert domain.step == 2
    del domain
    gc.collect()
    values = flow.numpy()
    np.testing.assert_allclose(values, np.tile([1, 0, 0, 0, 2.5], (values.shape[0], 1)), rtol=0, atol=2e-14)
    del values, flow, meshes, mesh
    gc.collect()
    if split:
        comm.Free()


@pytest.mark.mpi
@pytest.mark.parametrize("layout,flux,workers", [("aos", "Roe", 0), ("soa", "AUSMPlusUp", 2), ("aosoa32", "HLLEP", 2)])
def test_shared_state_real_residual_ssprk3_and_restart(tmp_path, layout, flux, workers):
    domain, primitive = build_domain(layout, flux, workers)
    rho, u, v, w, p = primitive
    exact = [rho, rho*u, rho*v, rho*w, p/.4 + .5*rho*(u*u+v*v+w*w)]
    flow = domain.fields["flow"]
    np.testing.assert_allclose(flow.numpy(), np.tile(exact, (flow.shape[0], 1)), atol=1e-14)
    assert domain.fields["flow"].identity == flow.identity
    assert domain.stable_step(.5) > 0
    np.testing.assert_allclose(domain.residual_fields()["flow"].numpy(), 0, atol=3e-13)
    assert domain.advance(.001).accepted
    assert domain.step == 1
    assert domain.time == .001
    with flow.read():
        assert domain.prepare_step(.001).accepted
        with pytest.raises(RuntimeError):
            domain.commit_step()
    domain.commit_step()
    assert domain.step == 2
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "cauchy.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"python"}')
    with pytest.raises(RuntimeError):
        domain.read_checkpoint(path, "{" if MPI.COMM_WORLD.rank == 0 else '{}')
    saved = flow.numpy().copy()
    assert domain.advance(.001).accepted
    domain.read_checkpoint(path, '{"test":"python"}')
    assert domain.step == 2
    assert domain.time == .002
    np.testing.assert_array_equal(flow.numpy(), saved)
    # A shared Python handle retains its allocation after the domain dies.
    del domain
    np.testing.assert_array_equal(flow.numpy(), saved)


@pytest.mark.mpi
@pytest.mark.parametrize("method,stages", [("bdf2", 1), ("esdirk4", 5), ("ditr", 1)])
def test_configured_implicit_method_and_retained_shared_state(tmp_path, method, stages):
    domain, primitive = build_domain(method=method)
    flow = domain.fields["flow"]
    before = flow.numpy().copy()
    assert domain.advance(.001).accepted
    assert len(domain.implicit_stages) == stages
    if method == "ditr":
        assert domain.coupled_iterations == 1
        assert domain.coupled_target_ratio <= 1
    assert all(report.converged for report in domain.implicit_stages)
    assert all(report.reconstruction_iterations == 0 and report.reconstruction_relative_update == 0
               for report in domain.implicit_stages)  # Direct gradients do not run iterative VR.
    assert domain.step == 1 and domain.time == .001
    np.testing.assert_allclose(flow.numpy(), before, atol=2e-14)
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "implicit.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"implicit-python"}', include_reconstruction=True)
    assert domain.advance(.002).accepted
    domain.read_checkpoint(path, '{"test":"implicit-python"}')
    assert domain.step == 1 and domain.time == .001
    assert domain.advance(.002).accepted
    assert all(report.converged for report in domain.implicit_stages)
    # The same Python Array handle continues to reference the bound allocation.
    assert domain.fields["flow"].identity == flow.identity
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)


@pytest.mark.mpi
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_python_coupled_iteration_matches_native_step_and_restart(tmp_path, device):
    if device == "cuda" and (not R.cuda_enabled or MPI.COMM_WORLD.size != 1):
        pytest.skip("single-device DITR restart comparison requires a CUDA build and one MPI rank")

    def values_on_host(domain):
        field, = domain.fields.values()
        if device == "cuda":
            assert field.device.space == R.MemorySpace.CUDA
            field = field.to(R.Device.host())
        return field.numpy().copy()

    perturb = {"exprtkInitializers": [{"stateType": "primRhoP", "exprs": [
        "inRegion := 1; UExprtk[0] := UExprtk[0] * (1 + 0.01 * x[0]); 0;"
    ]}]}
    native, _ = build_domain(method="ditr", initial_field=perturb, device=device)
    scripted, _ = build_domain(method="ditr", initial_field=perturb, device=device)
    initial = values_on_host(native)
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "ditr.h5") if MPI.COMM_WORLD.rank == 0 else None)
    for step in (1, 2):
        if step == 2:
            scripted.write_checkpoint(path, '{}', include_reconstruction=True)
            scripted, _ = build_domain(method="ditr", initial_field=perturb, device=device)
            scripted.read_checkpoint(path, '{}')
        assert native.advance(.001).accepted
        assert advance_domain(scripted, .001).accepted
        assert native.step == scripted.step == step
        assert native.time == scripted.time
        a, = native.implicit_stages
        b, = scripted.implicit_stages
        assert (a.iterations, a.linear_iterations, a.converged) == (
            b.iterations, b.linear_iterations, b.converged)
        assert a.linear_iterations > 0
        np.testing.assert_allclose(b.residual, a.residual, rtol=1e-12, atol=1e-14)
        values = values_on_host(scripted)
        np.testing.assert_allclose(values, values_on_host(native), rtol=1e-12, atol=1e-14)
        diff = np.max(np.abs(values - initial), initial=0.)
        assert MPI.COMM_WORLD.allreduce(diff, op=MPI.MAX) > 1e-9


@pytest.mark.mpi
@pytest.mark.parametrize("method, split, policy", [
    ("bdf1", False, "pythonSingleStageImplicit"),
    ("ditr", True, "pythonDITRSequential2Stage"),
])
def test_registered_python_policy_matches_native_connected_domains(method, split, policy):
    world = MPI.COMM_WORLD
    if world.size != 2:
        pytest.skip("connected controller oracle uses two ranks")
    comm = world.Split(world.rank if split else 0, world.rank)
    mpi = DNDS.MPIInfo(comm.handle)
    root = Path(__file__).resolve().parents[2]
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(root / "data/mesh/Uniform_3x3_wall.cgns"), 2))
    with mesh.acquire_read(faces=True) as read:
        local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
    zones = set().union(*world.allgather(local_zones))
    coupled = min(zones)
    domains = [{"name": str(i), "ranks": [i] if split else [0, 1]} for i in range(2)]
    # A pressure jump across identical patches makes donor freshness observable.
    domains[1]["overrides"] = {"initialPrimitive": [1., 0., 0., 0., 1.1]}
    meshes = [mesh if world.rank in d["ranks"] else None for d in domains]
    config = {
        "domains": domains,
        "defaults": {
            "mesh": {"dimension": 2},
            "numerics": {"reconstruction": "constant"},
            "initialField": {"exprtkInitializers": [{"stateType": "primRhoP", "exprs": [
                "inRegion := 1; UExprtk[0] := 1 + 0.02*x[0]; UExprtk[1] := 0.1; 0;"]}]},
            "boundaries": {str(zone): {"kind": "interface" if zone == coupled else "farfield"}
                           for zone in zones},
            "implicit": {"iterations": 5, "minIterations": 2},
        },
        "interfaces": [{"donor": str(i), "receiver": str(1-i), "donorZone": coupled,
                        "receiverZone": coupled, "refreshCadence": "nonlinearIteration"}
                       for i in range(2)],
        "time": {"method": method},
    }
    def run():
        sim = Simulation(meshes, DNDS.MPIInfo(world.handle), json.dumps(config))
        assert sim.advance(.001).accepted
        diagnostics = sim.interface_diagnostics()
        assert diagnostics.query_points > 0
        results = []
        for i in range(2):
            if sim.local(i):
                domain = sim.domain(i)
                stage, = domain.implicit_stages
                assert stage.linear_iterations > 0
                field, = domain.fields.values()
                results.append((field.numpy().copy(), stage.iterations, stage.residual))
        return results, (diagnostics.query_points, diagnostics.map_builds)

    expected, native_matching = run()
    config["defaults"]["implicit"]["updatePolicy"] = policy
    actual, matching = run()
    assert matching == native_matching
    for (field, iterations, residual), (ref, count, ref_residual) in zip(actual, expected, strict=True):
        assert iterations == count
        np.testing.assert_allclose(field, ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(residual, ref_residual, rtol=1e-12, atol=1e-14)
    del meshes, mesh, mpi
    comm.Free()


@pytest.mark.mpi
def test_rank_local_invalid_config_is_collectively_rejected():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns"
    mesh = Geom.with_native_topology(Geom.read_native_cgns(mpi, str(path), 2))
    with pytest.raises(RuntimeError):
        Domain(mesh, "{" if MPI.COMM_WORLD.rank == 0 else "{}")


@pytest.mark.mpi
def test_explicit_numerical_rejection_returns_a_status():
    domain, _ = build_domain(layout="soa", flux="rusanov", workers=0)
    flow = domain.fields["flow"]
    if MPI.COMM_WORLD.rank == 0 and flow.shape[0]:
        # A physical pressure jump has a finite RHS but an enormous explicit
        # timestep makes the stage inadmissible. The acceptance is collective.
        flow[0, 4] = float(flow[0, 4]) + 1
    saved = flow.numpy().copy()
    result = domain.advance(1e4)
    assert not result.accepted and result.reason
    assert domain.time == 0 and domain.step == 0
    np.testing.assert_array_equal(flow.numpy(), saved)
    domain.discard_step()
    assert domain.advance(1e-6).accepted
    assert domain.step == 1


@pytest.mark.mpi
def test_explicit_derived_refresh_shared_storage_and_freshness():
    domain, primitive = build_domain(layout="soa")
    fields = domain.make_derived_fields([DerivedQuantity.Pressure, DerivedQuantity.Temperature])
    assert fields.quantities == [DerivedQuantity.Temperature, DerivedQuantity.Pressure]
    assert not fields.current(domain)
    assert fields.refresh(domain)
    rho, _, _, _, p = primitive
    np.testing.assert_allclose(fields.values.numpy(), np.tile([p/rho, p], (domain.fields["flow"].shape[0], 1)))
    assert fields.current(domain)
    assert not fields.refresh(domain, only_if_stale=True)
    retained = fields.values
    before = domain.fields["flow"].numpy().copy()
    if before.shape[0]:
        domain.fields["flow"][0, 4] = before[0, 4] + .1
        assert not fields.current(domain)
    fields.refresh(domain)
    assert retained.identity == fields.values.identity
    saved = retained.numpy().copy()
    del fields, domain
    np.testing.assert_array_equal(retained.numpy(), saved)


@pytest.mark.mpi
def test_steady_bdf1_override_keeps_the_physical_clock_frozen():
    domain, _ = build_domain(method="esdirk4", steady=True)
    before = domain.fields["flow"].numpy().copy()
    assert domain.advance(.03).accepted
    assert domain.steady and domain.time == 0 and domain.step == 1
    assert len(domain.implicit_stages) == 1 and domain.implicit_stages[0].converged
    assert domain.previous_dt == 0
    np.testing.assert_allclose(domain.fields["flow"].numpy(), before, atol=3e-14)


@pytest.mark.mpi
@pytest.mark.parametrize("method", ["ssprk3", "bdf2"])
def test_physical_step_selection_retains_state_and_restores_growth_history(tmp_path, method):
    domain, _ = build_domain(layout="soa", workers=2, method=method, time_pp=True)
    flow = domain.fields["flow"]
    before = flow.numpy().copy()
    assert domain.previous_dt == 0
    initial = domain.limit_physical_step(.01)
    assert isinstance(initial, PhysicalStepLimit)
    assert initial.dt == .01 and initial.minimum_alpha == 1
    assert domain.advance(.001).accepted
    with flow.read():
        bound = domain.limit_physical_step(.1)
    assert bound.dt == .002 and bound.growth_limited
    assert bound.limited_cells == 0 and bound.minimum_alpha == 1
    assert domain.previous_dt == .001 and domain.time == .001 and domain.step == 1
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)
    assert domain.prepare_step(.003).accepted
    domain.discard_step()
    assert domain.previous_dt == .001
    assert domain.limit_physical_step(.1).dt == .002
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "time-step.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"physical-step-python"}')
    assert domain.advance(.004).accepted
    assert domain.previous_dt == .004
    domain.read_checkpoint(path, '{"test":"physical-step-python"}')
    assert domain.previous_dt == .001
    assert domain.limit_physical_step(.1).dt == .002
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)


@pytest.mark.mpi
@pytest.mark.parametrize("variant,layout", [("v2", "soa"), ("v3", "aosoa32")])
def test_native_lvr_config_and_warm_restart_through_shared_python_arrays(tmp_path, variant, layout):
    domain, _ = build_domain(layout=layout, method="bdf1", lvr_variant=variant)
    flow = domain.fields["flow"]
    before = flow.numpy().copy()
    assert domain.advance(.001).accepted
    assert domain.step == 1 and domain.time == .001
    assert domain.implicit_stages[0].converged
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "lvr.h5") if MPI.COMM_WORLD.rank == 0 else None, root=0)
    domain.write_checkpoint(path, '{"test":"lvr-python"}', include_reconstruction=True)
    assert domain.advance(.001).accepted
    domain.read_checkpoint(path, '{"test":"lvr-python"}')
    assert domain.fields["flow"].identity == flow.identity
    assert domain.step == 1 and domain.time == .001
    np.testing.assert_allclose(flow.numpy(), before, atol=3e-14)
