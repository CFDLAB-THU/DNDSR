"""Tooling regressions, independent of the DNDSR extension modules.

Run: python -m unittest discover -s scripts/tests -v
Set DNDS_HARNESS_CMAKE to test an older CMake binary without installing it.
All generated projects and Git indexes live in temporary directories.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("DNDS_HARNESS_CMAKE", shutil.which("cmake"))
CTEST = str(Path(CMAKE).with_name("ctest"))


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(
            prefix="dndsr-harness-test-")
        self.addCleanup(self.scratch.cleanup)
        self.work = Path(self.scratch.name)
        self.env = os.environ.copy()
        self.env.pop("DNDS_TEST_TIMEOUT", None)
        self.env.pop("CTEST_OUTPUT_ON_FAILURE", None)
        # Never let the caller's Git context select a real worktree or index.
        for key in list(self.env):
            if key.startswith("GIT_"):
                self.env.pop(key)

    def run_command(self, *args, cwd=None, check=True, env=None):
        result = subprocess.run(
            args, cwd=cwd or self.work, env=env or self.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=60)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def write(self, name, text):
        path = self.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def init_git(self):
        self.run_command("git", "init", "-q")

    def stage(self, name, contents):
        self.write(name, contents)
        self.run_command("git", "--literal-pathspecs", "add", "--", name)

    def formatter(self, name="clang-format", exit_code=0):
        path = self.write(
            "venv/bin/" + name,
            '#!/bin/sh\nfor file in "$@"; do\n'
            '  case "$file" in -*) continue ;; esac\n'
            '  printf "formatted\\n" >> "$file"\ndone\n'
            f'exit {exit_code}\n')
        path.chmod(0o755)

    def hook(self, **kwargs):
        return self.run_command("/bin/bash", str(ROOT / "scripts/pre-commit"), **kwargs)

    def test_hook_rejects_partial_file_before_any_formatting(self):
        self.init_git()
        self.formatter()
        self.stage("a.cpp", "staged a\n")
        self.stage("z.cpp", "staged z\n")
        self.write("z.cpp", "staged z\nunstaged\n")
        before = self.run_command("git", "diff", "--cached").stdout
        result = self.hook(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("partially staged", result.stdout)
        self.assertEqual(self.run_command(
            "git", "diff", "--cached").stdout, before)
        self.assertEqual((self.work / "a.cpp").read_text(), "staged a\n")
        self.assertEqual((self.work / "z.cpp").read_text(),
                         "staged z\nunstaged\n")

    def test_hook_preserves_paths_and_restages_formatted_files(self):
        self.init_git()
        self.formatter()
        names = ["space name.cpp", "line\nbreak.cpp",
                 "-option.cpp", "[literal].cpp"]
        for name in names:
            self.stage(name, "staged\n")
        self.hook()
        for name in names:
            self.assertEqual(self.run_command("git", "show", ":" + name).stdout,
                             "staged\nformatted\n")
        self.assertEqual(self.run_command("git", "diff").stdout, "")

    def test_each_formatter_failure_is_nonzero_and_not_restaged(self):
        self.init_git()
        for tool, name in [("clang-format", "file.cpp"),
                           ("autopep8", "file.py"), ("nbstripout", "file.ipynb")]:
            with self.subTest(tool=tool):
                self.stage(name, "staged\n")
                self.formatter(tool, exit_code=42)
                result = self.hook(check=False)
                self.assertEqual(result.returncode, 42, result.stdout)
                self.assertEqual(self.run_command("git", "show", ":" + name).stdout,
                                 "staged\n")
                self.assertEqual((self.work / name).read_text(),
                                 "staged\nformatted\n")
                self.run_command("git", "update-index",
                                 "--force-remove", "--", name)

    def test_missing_optional_formatter_does_not_block(self):
        self.init_git()
        self.stage("file.cpp", "staged\n")
        bindir = self.work / "tools"
        bindir.mkdir()
        (bindir / "git").symlink_to(shutil.which("git"))
        self.env["PATH"] = str(bindir)
        result = self.hook()
        self.assertIn("clang-format not found", result.stdout)
        self.assertEqual(self.run_command(
            "git", "show", ":file.cpp").stdout, "staged\n")

    def summary_fixture(self, exit_code=0, failed=0, summaries=1):
        script = self.write("summary.py", (
            f'for _ in range({summaries}):\n'
            f'    print("[doctest] test cases: 1 | {1 - failed} passed | {failed} failed | 0 skipped")\n'
            f'    print("[doctest] assertions: 2 | {2 - failed} passed | {failed} failed |")\n'
            f'raise SystemExit({exit_code})\n'))
        self.write("CTestTestfile.cmake",
                   f'add_test(summary "{sys.executable}" "{script}")\n')
        self.env["PATH"] = str(Path(CTEST).parent) + \
            os.pathsep + self.env["PATH"]

    def summary(self, *args):
        return self.run_command(sys.executable, str(ROOT / "scripts/ctest_summary.py"),
                                "--test-dir", str(self.work), *args, check=False)

    def test_summary_process_failure_is_not_success(self):
        self.summary_fixture(exit_code=1)
        result = self.summary("--output-on-failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Status: FAILED", result.stdout)
        self.assertNotIn("Status: SUCCESS", result.stdout)
        self.assertIn(
            "Doctest summaries (including MPI ranks): 1", result.stdout)

    def test_summary_does_not_count_failed_output_replay(self):
        self.summary_fixture(exit_code=1, failed=1)
        result = self.summary("--output-on-failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(
            result.stdout, r"Total test cases:\s+1 \|\s+0 passed \| 1 failed")
        self.assertRegex(
            result.stdout, r"Total assertions:\s+2 \|\s+1 passed \| 1 failed")

    def test_summary_counts_identical_rank_summaries(self):
        self.summary_fixture(summaries=2)
        result = self.summary()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("CTest tests with doctest output: 1", result.stdout)
        self.assertIn(
            "Doctest summaries (including MPI ranks): 2", result.stdout)
        self.assertRegex(result.stdout, r"Total test cases:\s+2 \|\s+2 passed")
        self.assertIn("Status: SUCCESS", result.stdout)

    def test_summary_failed_counts_override_zero_exit(self):
        self.summary_fixture(failed=1)
        result = self.summary()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Status: FAILED", result.stdout)

    def test_summary_ignores_summary_text_in_command_description(self):
        self.write("CTestTestfile.cmake", f'add_test(summary "{CMAKE}" -E echo '
                   '"[doctest] test cases: 1 | 1 passed | 0 failed | 0 skipped")\n')
        self.env["PATH"] = str(Path(CTEST).parent) + \
            os.pathsep + self.env["PATH"]
        result = self.summary()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "Doctest summaries (including MPI ranks): 1", result.stdout)

    def test_summary_failure_without_any_doctest_output(self):
        self.write("CTestTestfile.cmake",
                   f'add_test(failure "{CMAKE}" -E false)\n')
        self.env["PATH"] = str(Path(CTEST).parent) + \
            os.pathsep + self.env["PATH"]
        result = self.summary()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Status: FAILED", result.stdout)

    def test_summary_empty_selection_is_not_success(self):
        self.summary_fixture()
        result = self.summary("-R", "does_not_exist")
        self.assertIn("Status: NO DOCTEST RESULTS", result.stdout)
        self.assertNotIn("Status: SUCCESS", result.stdout)

    def timeout_fixture(self):
        self.write("CMakeLists.txt", f'''
cmake_minimum_required(VERSION 3.21)
project(TimeoutProbe NONE)
enable_testing()
include("{ROOT}/cmake/DndsTestTimeout.cmake")
add_test(NAME probe COMMAND "${{CMAKE_COMMAND}}" -E sleep 1)
_dnds_maybe_set_timeout(probe ${{DNDS_TEST_TIMEOUT}})
''')

    def test_default_timeout_respects_ctest_and_explicit_override(self):
        self.timeout_fixture()
        build = self.work / "build"
        self.run_command(CMAKE, "-S", str(self.work), "-B", str(build))
        tests = self.run_command(
            CTEST, "--test-dir", str(build), "--show-only=json-v1")
        props = json.loads(tests.stdout)["tests"][0]["properties"]
        self.assertNotIn("TIMEOUT", {p["name"] for p in props})
        result = self.run_command(CTEST, "--test-dir", str(build), "--timeout", "0.05",
                                  "--output-on-failure", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Timeout", result.stdout)
        self.env["DNDS_TEST_TIMEOUT"] = "7"
        self.run_command(CMAKE, "-S", str(self.work), "-B", str(build))
        tests = self.run_command(
            CTEST, "--test-dir", str(build), "--show-only=json-v1")
        props = {p["name"]: p["value"]
                 for p in json.loads(tests.stdout)["tests"][0]["properties"]}
        self.assertEqual(props["TIMEOUT"], 7)
        self.env.pop("DNDS_TEST_TIMEOUT")
        self.run_command(CMAKE, "-S", str(self.work), "-B", str(build))
        tests = self.run_command(
            CTEST, "--test-dir", str(build), "--show-only=json-v1")
        self.assertNotIn("TIMEOUT", {p["name"] for p in json.loads(
            tests.stdout)["tests"][0]["properties"]})

    def test_explicit_zero_timeout_remains_supported(self):
        self.timeout_fixture()
        self.env["DNDS_TEST_TIMEOUT"] = "0"
        build = self.work / "build"
        self.run_command(CMAKE, "-S", str(self.work), "-B", str(build))
        tests = self.run_command(
            CTEST, "--test-dir", str(build), "--show-only=json-v1")
        props = {p["name"]: p["value"]
                 for p in json.loads(tests.stdout)["tests"][0]["properties"]}
        self.assertEqual(props["TIMEOUT"], 0)

    def test_mpi_release_guard_and_reconfigure(self):
        # This probe supplies its own MPI headers. The caller's MPI compiler
        # wrapper can otherwise put the real header ahead of the fixture.
        self.env.pop("CXX", None)
        self.write("CMakeLists.txt", f'''
cmake_minimum_required(VERSION 3.21)
project(MPIVersionProbe CXX)
add_library(MPI::MPI_CXX INTERFACE IMPORTED)
set_target_properties(MPI::MPI_CXX PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${{CMAKE_CURRENT_SOURCE_DIR}}/headers")
include("{ROOT}/cmake/DndsMPIVersion.cmake")
dnds_check_mpi_version()
''')
        # Reuse one cache: switching MPI must never retain an old success.
        for release in [(4, 1, 1), (4, 1, 0), (4, 1, 10), (4, 0, 7), (5, 0, 9), None]:
            with self.subTest(release=release):
                header = "#define MPI_VERSION 4\n#define MPI_SUBVERSION 1\n"
                if release:
                    header += "".join(f"#define OMPI_{name}_VERSION {value}\n" for name, value
                                      in zip(["MAJOR", "MINOR", "RELEASE"], release))
                self.write("headers/mpi.h", header)
                result = self.run_command(
                    CMAKE, "-S", str(self.work), "-B", str(self.work / "build"), check=False)
                if release == (4, 1, 0):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "Open MPI 4.1.0 is not supported", result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout)

        self.write("headers/mpi.h", "#define OMPI_MAJOR_VERSION 4\n"
                   "#define OMPI_MINOR_VERSION 1\n#define OMPI_RELEASE_VERSION 0\n")
        # Explicit opt-in and subsequent opt-out must both invalidate the cache.
        for enabled in ["ON", "OFF", "ON"]:
            result = self.run_command(
                CMAKE, "-S", str(self.work), "-B", str(self.work / "build"),
                "-DDNDS_ALLOW_OPENMPI_410=" + enabled, check=False)
            self.assertEqual(result.returncode == 0,
                             enabled == "ON", result.stdout)
            if enabled == "ON":
                self.assertIn("overrides the release guard only",
                              " ".join(result.stdout.split()))
        # The escape hatch must not accept unrelated header/compiler failures.
        self.write("headers/mpi.h", '#error "broken MPI header"\n')
        result = self.run_command(
            CMAKE, "-S", str(self.work), "-B", str(self.work / "build"), check=False)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_all_presets_on_selected_cmake(self):
        # Exercise unchanged preset definitions on a tiny independent project.
        shutil.copy2(ROOT / "CMakePresets.json", self.work)
        self.write("CMakeLists.txt", '''
cmake_minimum_required(VERSION 3.21)
project(PresetProbe NONE)
enable_testing()
add_custom_target(dnds_unit_tests ALL COMMAND ${CMAKE_COMMAND} -E true)
add_test(NAME dnds_probe COMMAND ${CMAKE_COMMAND} -E true)
''')
        self.run_command(CMAKE, "--list-presets=all")
        for name in ["default", "debug", "release-test", "cuda", "ci"]:
            self.run_command(CMAKE, "--preset", name)
        for name in ["default", "tests", "debug"]:
            self.run_command(CMAKE, "--build", "--preset",
                             name, "--parallel", "2")
        for name in ["unit", "all"]:
            self.run_command(CTEST, "--preset", name)

    def test_schema_two_cannot_preserve_generator_discovery(self):
        presets = json.loads((ROOT / "CMakePresets.json").read_text())
        presets["version"] = 2
        self.write("CMakePresets.json", json.dumps(presets))
        result = self.run_command(CMAKE, "--list-presets=all", check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_packaging_respects_parallel_environment(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest(
                "Reading pyproject.toml requires Python 3.11+ for this test")
        ninja = shutil.which("ninja")
        if not ninja:
            self.skipTest("Ninja is required for the build-argument probe")
        settings = tomllib.loads(
            (ROOT / "pyproject.toml").read_text())["tool"]["scikit-build"]
        tool_args = settings.get("build", {}).get("tool-args", [])
        self.write("CMakeLists.txt", '''
cmake_minimum_required(VERSION 3.21)
project(ParallelProbe NONE)
add_custom_target(probe ALL COMMAND ${CMAKE_COMMAND} -E true)
''')
        wrapper = self.write("ninja-wrapper", '#!/bin/sh\n'
                             'printf "ninja-argument=<%s>\\n" "$@" >&2\n'
                             f'exec "{ninja}" "$@"\n')
        wrapper.chmod(0o755)
        build = self.work / "build"
        self.run_command(CMAKE, "-S", str(self.work), "-B", str(build), "-G", "Ninja",
                         "-DCMAKE_MAKE_PROGRAM=" + str(wrapper))
        self.env["CMAKE_BUILD_PARALLEL_LEVEL"] = "2"
        result = self.run_command(
            CMAKE, "--build", str(build), "--", *tool_args)
        self.assertIn("ninja-argument=<-j>\nninja-argument=<2>", result.stdout)
        self.assertNotIn("ninja-argument=<-j0>", result.stdout)

    def test_shared_skill_aliases_resolve_to_single_source(self):
        canonical = ROOT / ".opencode/skills"
        expected = {p.parent.name for p in canonical.glob("*/SKILL.md")}
        self.assertTrue(expected)
        for alias in [ROOT / ".agents/skills", ROOT / ".codex/skills"]:
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), canonical.resolve())
            self.assertEqual(
                {p.parent.name for p in alias.glob("*/SKILL.md")}, expected)


if __name__ == "__main__":
    unittest.main()
