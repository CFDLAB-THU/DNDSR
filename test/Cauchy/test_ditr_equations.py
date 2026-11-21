"""Distinguish physical mass and spatial rate terms in queued DITR arithmetic."""
import json
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from DNDSR import Cauchy, DNDS, Geom
from DNDSR.DNDS import Runtime


@pytest.mark.mpi
def test_queued_acm_stage_sources_and_stopping_residual():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "cases/cauchy/acm_translating_vortex_20.json").read_text())
    mesh = Geom.with_native_topology(
        Geom.read_native_cgns(mpi, str(root / "data/mesh/Uniform_3x3_wall.cgns"), 2))
    settings = config["defaults"]
    for key in ("initialField", "mesh"):
        del settings[key]
    settings["numerics"].update(reconstruction="gradient")
    with mesh.acquire_read(faces=True) as read:
        local_zones = {read.boundary_info(i).zone for i in range(mesh.NumBnd())}
    zones = set().union(*MPI.COMM_WORLD.allgather(local_zones))
    settings["boundaries"] = {str(zone): {"kind": "slipWall"} for zone in zones}
    domain = Cauchy.Domain(mesh, json.dumps(config))
    engine = Runtime.EagerEngine(2, Runtime.AccessTrackingContext())
    stream = engine.cpu_stream()
    equations = Cauchy.DITREquations(stream, .5, .75, "U2R2")
    proto = domain.dofs
    fields = [proto.zeros_like() for _ in range(6)]
    for field, value in zip(fields, (2, 3, 5, 7, 11, 13)):
        field.fill(value)
    initial, middle, endpoint, r0, rm, re = fields
    h_other, i_other, stop = (proto.zeros_like() for _ in range(3))
    inv_dt = 8
    h_scale = equations.midpoint_source(h_other, inv_dt, initial, endpoint, r0, re)
    i_scale = equations.endpoint_source(i_other, inv_dt, initial, r0, rm)
    equations.stopping_residual(stop, inv_dt, initial, middle, endpoint, r0, rm, re)
    del fields, initial, middle, endpoint, r0, rm, re
    stream.record().wait()

    # At a=1/2 integration is Simpson's rule. H uses Hermite interpolation
    # at the midpoint. These full equations are independent of resOther assembly.
    mass = np.ones(proto.components)
    pressure = proto.resolve_components(["p"])[0]
    mass[pressure] = 0
    h = (.5 * 2 + .5 * 5 - 3) * inv_dt * mass + (7 - 13) / 8
    i = (2 - 5) * inv_dt * mass + (7 + 4 * 11 + 13) / 6
    def values(field):
        return next(iter(field.fields.values())).owned.numpy()
    for field, expected in ((h_other, h + .75 * i + inv_dt * 3 * mass - h_scale * 11),
                            (i_other, i + inv_dt * 5 * mass - i_scale * 13),
                            (stop, h + i)):
        actual = values(field)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=3e-14)
    engine.close()
