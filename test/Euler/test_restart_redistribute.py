"""
Functional test for ArrayPair redistribution via Euler solver restart.

The test verifies that DOF data written in a restart file by one MPI partition
can be correctly read by a different partition (different np), producing
identical solver behaviour.

Steps:
  1. Run the Euler IV solver for 20 time steps with np=2, writing an H5 restart.
  2. Load that restart and immediately write it back at np=2 (reference).
  3. Load that restart and immediately write it back at np=3 (redistributed read).
  4. Compare the two restart H5 files: the DOF data must match to machine precision.

The file contains three test variants: same-np reseed, cross-np redistribution,
and a large-mesh multi-np (np=4 → np=4..8) case.

Usage:
    pytest test/Euler/test_restart_redistribute.py -s
    # or directly:
    python test/Euler/test_restart_redistribute.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.realpath(os.path.join(SCRIPT_DIR, "..", ".."))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
EULER_EXE = os.path.join(BUILD_DIR, "app", "euler.exe")
BASE_CONFIG = os.path.join(PROJECT_ROOT, "cases",
                           "euler", "euler_config_IV.json")
MESH_SMALL = os.path.join(PROJECT_ROOT, "data", "mesh",
                          "IV10_10.cgns")   # 10x10 = 100 cells
MESH_LARGE = os.path.join(PROJECT_ROOT, "data", "mesh",
                          "IV10_20.cgns")   # 20x20 = 400 cells


def _strip_json_comments(text):
    """Remove // comments from JSON-with-comments (DNDS convention)."""
    return re.sub(r"//.*", "", text)


def _load_json(path):
    with open(path) as f:
        return json.loads(_strip_json_comments(f.read()))


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def _run_solver(np_count, config_path, work_dir, overrides=None, timeout=300):
    """Run the Euler solver via mpirun and return the process result."""
    cmd = [
        "mpirun", "--oversubscribe", "-np", str(np_count),
        EULER_EXE, config_path,
    ]
    if overrides:
        for k, v in overrides:
            cmd.extend(["-k", k, "-v", v])

    env = os.environ.copy()
    result = subprocess.run(
        cmd, cwd=work_dir, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env,
    )
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    if result.returncode != 0:
        print(f"=== SOLVER FAILED (np={np_count}) ===")
        print(stdout_text[-4000:])
    return result, stdout_text


def _read_h5_u_data(h5_path):
    """Read the DOF 'u' data from a restart H5 file.

    Returns (origIndex, data) where:
      - origIndex: 1-D array of original cell indices (global key)
      - data: 2-D array of DOF values, shape (nGlobal, nVars)
    """
    with h5py.File(h5_path, "r") as f:
        # The H5 layout from ArrayPair::WriteSerialize with origIndex:
        #   /u/origIndex           -- flat array of original cell indices
        #   /u/father/data         -- for plain ParArray
        #   /u/father/array/data   -- for ArrayEigenMatrix (extra 'array' sub-group)
        orig_idx = f["u"]["origIndex"][:]

        father_grp = f["u"]["father"]
        if "data" in father_grp:
            raw_data = father_grp["data"][:]
        elif "array" in father_grp and "data" in father_grp["array"]:
            raw_data = father_grp["array"]["data"][:]
        else:
            raise KeyError(
                f"Cannot find data in {h5_path}: /u/father/ contains {list(father_grp.keys())}")

    n_global = len(orig_idx)
    assert raw_data.size % n_global == 0, (
        f"data size {raw_data.size} not divisible by nGlobal {n_global}"
    )
    n_vars = raw_data.size // n_global
    data = raw_data.reshape(n_global, n_vars)
    return orig_idx, data


def _gather_by_orig_index(orig_idx, data):
    """Reorder data rows by original index so that global ordering is canonical."""
    order = np.argsort(orig_idx)
    return orig_idx[order], data[order]


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def work_dir():
    """Create a temporary working directory for the test."""
    d = tempfile.mkdtemp(prefix="dnds_test_redist_")
    yield d
    # Cleanup after test (comment out for debugging)
    shutil.rmtree(d, ignore_errors=True)


def _make_step1_config(work_dir, mesh_file=MESH_SMALL):
    """Config for the initial 20-step run (produces the restart)."""
    cfg = _load_json(BASE_CONFIG)

    # Time march: 20 steps, explicit ESDIRK4 (odeCode=0), small dt
    cfg["timeMarchControl"]["nTimeStep"] = 20
    cfg["timeMarchControl"]["tEnd"] = 1e10  # won't hit tEnd, use nTimeStep
    cfg["timeMarchControl"]["dtImplicit"] = 1.25e-3
    cfg["timeMarchControl"]["odeCode"] = 0
    cfg["timeMarchControl"]["steadyQuit"] = False
    cfg["timeMarchControl"]["useRestart"] = False

    # Output: minimal
    cfg["outputControl"]["nDataOut"] = 1000000
    cfg["outputControl"]["nDataOutC"] = 1000000
    cfg["outputControl"]["nConsoleCheck"] = 5
    cfg["outputControl"]["dataOutAtInit"] = False
    # Restart output every 20 steps (i.e. at the end)
    cfg["outputControl"]["nRestartOut"] = 20
    cfg["outputControl"]["nRestartOutC"] = 1000000

    # Mesh
    cfg["dataIOControl"]["meshFile"] = mesh_file
    out_base = os.path.join(work_dir, "step1", "out")
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    cfg["dataIOControl"]["outPltName"] = out_base

    # H5 restart writer
    cfg["dataIOControl"]["restartWriter"] = {
        "type": "H5",
        "hdfDeflateLevel": 0,
        "hdfChunkSize": 0,
        "hdfCollOnMeta": True,
        "hdfCollOnData": False,
        "jsonBinaryDeflateLevel": 5,
        "jsonUseCodecOnUInt8": True,
    }

    # Use 1st order for speed (maxOrder=1)
    cfg["vfvSettings"]["maxOrder"] = 1
    cfg["vfvSettings"]["intOrder"] = 1

    # No bisection
    cfg["dataIOControl"]["meshDirectBisect"] = 0
    cfg["dataIOControl"]["meshReorderCells"] = False

    path = os.path.join(work_dir, "step1_config.json")
    _write_json(path, cfg)
    return path, cfg


def _make_step2_config(work_dir, restart_h5_path, tag, np_count,
                       reorder_cells=False, mesh_bisect=0, mesh_file=MESH_SMALL):
    """Config for the 1-step restart run."""
    cfg = _load_json(BASE_CONFIG)

    # Time march: 1 step from restart -- load, run one step, and write
    cfg["timeMarchControl"]["nTimeStep"] = 1
    cfg["timeMarchControl"]["tEnd"] = 1e10
    cfg["timeMarchControl"]["dtImplicit"] = 1.25e-3
    cfg["timeMarchControl"]["odeCode"] = 0
    cfg["timeMarchControl"]["steadyQuit"] = False
    cfg["timeMarchControl"]["useRestart"] = True

    # Restart state: start from step 20
    cfg["restartState"] = {
        "iStep": 20,
        "iStepInternal": -1,
        "odeCodePrev": 0,
        "lastRestartFile": restart_h5_path,
        "otherRestartFile": "",
        "otherRestartStoreDim": [0],
    }

    # Output: write restart immediately at init, no step-based output
    cfg["outputControl"]["nDataOut"] = 1000000
    cfg["outputControl"]["nDataOutC"] = 1000000
    cfg["outputControl"]["nConsoleCheck"] = 1
    cfg["outputControl"]["dataOutAtInit"] = False
    cfg["outputControl"]["restartOutAtInit"] = True
    cfg["outputControl"]["nRestartOut"] = 1000000
    cfg["outputControl"]["nRestartOutC"] = 1000000

    # Mesh
    cfg["dataIOControl"]["meshFile"] = mesh_file
    out_base = os.path.join(work_dir, f"step2_{tag}", "out")
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    cfg["dataIOControl"]["outPltName"] = out_base

    # H5 restart writer
    cfg["dataIOControl"]["restartWriter"] = {
        "type": "H5",
        "hdfDeflateLevel": 0,
        "hdfChunkSize": 0,
        "hdfCollOnMeta": True,
        "hdfCollOnData": False,
        "jsonBinaryDeflateLevel": 5,
        "jsonUseCodecOnUInt8": True,
    }

    cfg["vfvSettings"]["maxOrder"] = 1
    cfg["vfvSettings"]["intOrder"] = 1
    cfg["dataIOControl"]["meshDirectBisect"] = mesh_bisect
    cfg["dataIOControl"]["meshReorderCells"] = reorder_cells

    path = os.path.join(work_dir, f"step2_{tag}_config.json")
    _write_json(path, cfg)
    return path, cfg


def _find_restart_h5(search_dir, label=""):
    """Find restart H5 files in directory tree."""
    restart_files = []
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".restart.dnds.h5"):
                restart_files.append(os.path.join(root, f))
    assert len(restart_files) >= 1, (
        f"No restart H5 files found in {search_dir} ({label})"
    )
    return restart_files[0]


def _compare_restart_h5(restart_a, restart_b, tol=1e-10):
    """Compare two restart H5 files by origIndex. Returns max_diff, rel_norm."""
    orig_a, data_a = _read_h5_u_data(restart_a)
    orig_b, data_b = _read_h5_u_data(restart_b)

    orig_a_sorted, data_a_sorted = _gather_by_orig_index(orig_a, data_a)
    orig_b_sorted, data_b_sorted = _gather_by_orig_index(orig_b, data_b)

    assert np.array_equal(orig_a_sorted, orig_b_sorted), (
        "Original index sets differ between restarts"
    )

    max_diff = np.max(np.abs(data_a_sorted - data_b_sorted))
    rel_norm = np.linalg.norm(
        data_a_sorted - data_b_sorted) / (np.linalg.norm(data_a_sorted) + 1e-300)

    print(f"  nGlobal cells: {len(orig_a_sorted)}")
    print(f"  nVars per cell: {data_a_sorted.shape[1]}")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Rel L2 diff:   {rel_norm:.6e}")

    assert max_diff < tol, (
        f"DOF data differs too much: max_diff={max_diff:.6e}, rel={rel_norm:.6e}"
    )
    return max_diff, rel_norm


def _run_step1(work_dir, np_write, mesh_file=MESH_SMALL):
    """Run step 1: initial 20-step run producing a restart."""
    step1_config, _ = _make_step1_config(work_dir, mesh_file=mesh_file)
    result, stdout = _run_solver(np_write, step1_config, work_dir)
    assert result.returncode == 0, f"Step 1 solver failed:\n{stdout[-2000:]}"
    restart_h5 = _find_restart_h5(os.path.join(work_dir, "step1"), "step1")
    print(f"  Restart file: {restart_h5}")
    return restart_h5


def _run_step2(work_dir, restart_h5, tag, np_count, overrides=None, mesh_file=MESH_SMALL):
    """Run step 2: load restart and immediately write it back."""
    step2_config, _ = _make_step2_config(
        work_dir, restart_h5, tag, np_count, mesh_file=mesh_file)
    result, stdout = _run_solver(
        np_count, step2_config, work_dir, overrides=overrides)
    assert result.returncode == 0, f"Step 2 ({tag}) solver failed:\n{stdout[-2000:]}"
    restart_out = _find_restart_h5(os.path.join(work_dir, f"step2_{tag}"), tag)
    print(f"  Output restart: {restart_out}")
    return restart_out


# ---------------------------------------------------------------------------
# Test: same np, different Metis partition
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def work_dir():
    """Create a temporary working directory for the test."""
    d = tempfile.mkdtemp(prefix="dnds_test_redist_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def work_dir_large():
    """Create a temporary working directory for the large-mesh test."""
    d = tempfile.mkdtemp(prefix="dnds_test_redist_large_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_restart_redistribute_same_np(work_dir):
    """Write restart at np=2, read at np=2 with different Metis seed."""
    if not os.path.isfile(EULER_EXE):
        pytest.skip(f"euler.exe not found: {EULER_EXE}")
    if not os.path.isfile(MESH_SMALL):
        pytest.skip(f"Mesh file not found: {MESH_SMALL}")

    np_write = 2

    print(f"\n=== Step 1: 20-step run with np={np_write} ===")
    restart_h5 = _run_step1(work_dir, np_write, mesh_file=MESH_SMALL)

    print(f"\n=== Step 2a: load with np=2, same partition (reference) ===")
    restart_a = _run_step2(work_dir, restart_h5, "ref",
                           2, mesh_file=MESH_SMALL)

    print(f"\n=== Step 2b: load with np=2, different Metis seed ===")
    restart_b = _run_step2(
        work_dir, restart_h5, "reseed", 2,
        overrides=[("/dataIOControl/meshPartitionOptions/metisSeed", "42")],
        mesh_file=MESH_SMALL,
    )

    print("\n=== Comparing: same-np redistribution ===")
    _compare_restart_h5(restart_a, restart_b)
    print("=== PASS ===")


def test_restart_redistribute_different_np(work_dir):
    """Write restart at np=2, read at np=3 (cross-np redistribution)."""
    if not os.path.isfile(EULER_EXE):
        pytest.skip(f"euler.exe not found: {EULER_EXE}")

    # Reuse step1 restart from the same work_dir (written by previous test)
    restart_h5 = _find_restart_h5(
        os.path.join(work_dir, "step1"), "step1 (reuse)")
    print(f"\n  Reusing Step 1 restart: {restart_h5}")

    restart_ref = _find_restart_h5(os.path.join(
        work_dir, "step2_ref"), "ref (reuse)")

    print(f"\n=== Step 2c: load with np=3 (cross-np redistribution) ===")
    restart_c = _run_step2(work_dir, restart_h5, "np3",
                           3, mesh_file=MESH_SMALL)

    print("\n=== Comparing: cross-np redistribution ===")
    _compare_restart_h5(restart_ref, restart_c)
    print("=== PASS ===")


def test_restart_redistribute_large_mesh_multi_np(work_dir_large):
    """Write restart at np=4 with 20x20 mesh, read at np=4..8."""
    if not os.path.isfile(EULER_EXE):
        pytest.skip(f"euler.exe not found: {EULER_EXE}")
    if not os.path.isfile(MESH_LARGE):
        pytest.skip(f"Mesh file not found: {MESH_LARGE}")

    np_write = 4

    print(
        f"\n=== Step 1: 20-step run with np={np_write}, mesh=IV10_20 (400 cells) ===")
    restart_h5 = _run_step1(work_dir_large, np_write, mesh_file=MESH_LARGE)

    print(f"\n=== Reference: load with np={np_write} ===")
    restart_ref = _run_step2(work_dir_large, restart_h5,
                             "ref", np_write, mesh_file=MESH_LARGE)

    for np_read in [4, 5, 6, 7, 8]:
        tag = f"np{np_read}"
        overrides = None
        if np_read == np_write:
            # Same np: use different Metis seed to force redistribution
            overrides = [
                ("/dataIOControl/meshPartitionOptions/metisSeed", "99")]
            tag = f"np{np_read}_reseed"

        print(f"\n=== Load with np={np_read} ({tag}) ===")
        restart_out = _run_step2(
            work_dir_large, restart_h5, tag, np_read,
            overrides=overrides, mesh_file=MESH_LARGE,
        )

        print(f"=== Comparing np={np_write} vs np={np_read} ===")
        _compare_restart_h5(restart_ref, restart_out)
        print(f"=== PASS: np={np_read} ===")

    print("\n=== ALL np=4..8 PASS ===")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="dnds_test_redist_") as d:
        test_restart_redistribute_same_np(d)
        test_restart_redistribute_different_np(d)
    with tempfile.TemporaryDirectory(prefix="dnds_test_redist_large_") as d:
        test_restart_redistribute_large_mesh_multi_np(d)
