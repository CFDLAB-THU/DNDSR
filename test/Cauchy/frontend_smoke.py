"""Bounded executable integration: CLI, segmented history, output and restart.

Uses only the Python standard library. CTest launches this controller outside
MPI; its child solver processes use the configured MPI launcher and rank count.
This is a functionality check, not an IV accuracy or performance experiment.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mpi", required=True)
    parser.add_argument("--np", type=int, required=True)
    args = parser.parse_args()

    def run(*arguments, success=True):
        result = subprocess.run(
            [args.mpi, "-np", str(args.np), str(args.exe), *map(str, arguments)],
            cwd=args.source, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60,
        )
        assert (result.returncode == 0) == success, result.stdout
        return result.stdout

    assert "--overwrite_key" in run("--help")
    with tempfile.TemporaryDirectory(prefix="cauchy-frontend-") as directory:
        folder = Path(directory)
        # Independent tiny input, not a rewrite of any maintained case.
        config = folder / "case.json"
        config.write_text(json.dumps({
            "defaults": {
                "mesh": {"file": "data/mesh/IV10_20.cgns", "dimension": 2,
                         "periodicLengths": [10, 10, 0],
                         "boundaryIds": {"PERIODIC_1": -1, "PERIODIC_1_DONOR": -4,
                                         "PERIODIC_2": -2, "PERIODIC_2_DONOR": -5}},
                "initialPrimitive": [1, 1, 1, 0, 1],
                "initialProfile": {"kind": "isentropicVortex", "strength": 1},
                "numerics": {"reconstruction": "constant", "limiter": "none"},
            },
            "domains": [{"name": "fluid", "overrides": {"execution": {"overlap": True}}}],
            "time": {"method": "ssprk3", "steps": 2, "dt": 1e-5},
            "output": {"prefix": str(folder / "first"), "interval": 0,
                       "historyInterval": 1, "historySegmentRows": 2},
        }))
        resolved = json.loads(run(config, "--check-config", "-k", "/time/dt", "-v", "2e-5",
                                  "-k", "/domains/0/settings/execution/overlap", "-v", "false"))
        assert resolved["time"]["dt"] == 2e-5
        assert not resolved["domains"][0]["settings"]["execution"]["overlap"]
        assert "paired" in run(config, "--check-config", "-k", "/time/dt", success=False)
        run(config)
        first = json.loads((folder / "first-summary.json").read_text())
        assert first["steps"] == 2 and math.isclose(first["time"], 2e-5)
        assert first["status"] == "target-reached"
        rows = []
        header = None
        segments = sorted(folder.glob("first-D0-history*.csv"),
                          key=lambda path: ("_" in path.stem, path.stem))
        assert len(segments) > 1
        for segment in segments:
            with segment.open() as stream:
                reader = csv.DictReader(stream)
                if header is None:
                    header = reader.fieldnames
                assert reader.fieldnames == header
                assert {"step", "stage", "coupled_stage", "iteration", "status",
                        "res_rho", "res_rhoE"} <= set(header)
                data = list(reader)
                assert 0 < len(data) <= 2
                rows.extend(data)
        accepted = [int(row["step"]) for row in rows
                    if int(row["stage"]) == -1 and int(row["iteration"]) == -1]
        assert accepted == [0, 1, 2] and len(rows) == len(accepted)
        assert all(None not in row and all(not value or math.isfinite(float(value))
                                          for value in row.values()) for row in rows)
        assert (folder / "first-D0-final.vtkhdf").exists()
        assert (folder / "first-D0-final_bnd.vtkhdf").exists()
        snapshot = json.loads((folder / "first-final-manifest.json").read_text())
        boundary, = (item for item in snapshot["artifacts"] if item["kind"] == "boundary")
        assert Path(boundary["path"]).name == "first-D0-final_bnd.vtkhdf"
        assert not boundary["facialFluxes"]["available"]
        run(config, "--restart", folder / "first-D0-final.h5", "-k", "/output/prefix",
            "-v", json.dumps(str(folder / "resumed")), "-k", "/time/steps", "-v", "1")
        resumed = json.loads((folder / "resumed-summary.json").read_text())
        assert resumed["steps"] == 3 and math.isclose(resumed["time"], 3e-5)
        assert resumed["status"] == "target-reached"
        exact = json.loads(config.read_text())
        exact["time"].update({"method": "bdf2", "steps": 1})
        exact["output"].update({
            "prefix": str(folder / "exact"),
            "implicit": {"checkpointInterval": 1, "visualizationInterval": 1},
        })
        exact_config = folder / "exact.json"
        exact_config.write_text(json.dumps(exact))
        run(exact_config)
        internal = sorted(folder.glob("exact-D0-S1_0_*.h5"))
        assert internal
        assert all(path.with_name(path.stem + "-manifest.json").is_file()
                   for path in internal)
        run(exact_config, "--restart", internal[0], "-k", "/output/prefix", "-v",
            json.dumps(str(folder / "exact_resumed")))
        exact_resumed = json.loads((folder / "exact_resumed-summary.json").read_text())
        assert exact_resumed["steps"] == 1 and math.isclose(exact_resumed["time"], 1e-5)
        assert exact_resumed["status"] == "target-reached"
        # Exercise the same physical frontend with all-rank sharing followed by
        # disjoint rank groups. The existing single-domain files remain valid.
        multi = json.loads(config.read_text())
        multi["domains"] = [{"name": "left"}, {"name": "right"}]
        multi["time"]["method"] = "bdf2"
        multi["time"]["steps"] = 1
        multi["output"]["prefix"] = str(folder / "multi")
        multi["output"]["implicit"] = {"visualizationInterval": 1}
        multi_config = folder / "multi.json"
        multi_config.write_text(json.dumps(multi))
        run(multi_config)
        for i_domain in range(2):
            assert (folder / f"multi-D{i_domain}-final.vtkhdf").exists()
            internal = sorted(path for path in folder.glob(f"multi-D{i_domain}-S1_0_*.vtkhdf")
                              if not path.stem.endswith("_bnd"))
            assert internal
            assert all(path.with_name(path.stem + "-manifest.json").is_file()
                       for path in internal)
            assert all(path.with_name(path.stem + "_bnd.vtkhdf").is_file()
                       for path in internal)
            with (folder / f"multi-D{i_domain}-history.csv").open() as stream:
                assert len(list(csv.DictReader(stream))) == 2
        if args.np > 1:
            multi["domains"][0]["ranks"] = list(range(0, args.np, 2))
            multi["domains"][1]["ranks"] = list(range(1, args.np, 2))
        multi["output"]["prefix"] = str(folder / "multi_resumed")
        multi_config.write_text(json.dumps(multi))
        run(multi_config, "--restart", folder / "multi-final.h5")
        continued = json.loads((folder / "multi_resumed-summary.json").read_text())
        assert continued["steps"] == 2 and math.isclose(continued["time"], 2e-5)
        assert continued["status"] == "target-reached"
        for i_domain in range(2):
            assert (folder / f"multi_resumed-D{i_domain}-final.vtkhdf").exists()
    print(f"Cauchy frontend np={args.np}: CLI, histories, output and single/multi-domain restart passed")


if __name__ == "__main__":
    main()
