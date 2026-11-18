"""Exercise the RHS generator's CMake contract without a CUDA compilation."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rhs_codegen", ROOT / "scripts/generate_cauchy_rhs.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class RHSCodeGenerationTests(unittest.TestCase):
    def test_catalog_and_unchanged_output(self):
        manifest = ROOT / "scripts/manifests/cauchy_rhs_shapes.json"
        shapes = GENERATOR.read_shapes(manifest)
        self.assertEqual(len(shapes), 24)
        acm = GENERATOR.read_shapes(manifest, mechanical=True)
        self.assertEqual(len(acm), 18)
        self.assertEqual({c for c, _, _ in acm}, {4, 5, 6})
        self.assertTrue(all(m > 0 for _, m, _ in acm))
        for components in (5, 6, 7):
            self.assertEqual([(m, d) for c, m, d in shapes if c == components],
                             [(0, 2), (2, 2), (5, 2), (9, 2), (0, 3), (3, 3), (9, 3), (19, 3)])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "generated"
            command = [sys.executable, str(ROOT / "scripts/generate_cauchy_rhs.py"),
                       "--manifest", str(manifest), "--output-dir", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            self.assertEqual(len(before), 47)
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})

    @unittest.skipUnless(shutil.which("cmake"), "CMake is unavailable")
    def test_cmake_tracks_inputs_and_recovers_every_output(self):
        generators = ["Unix Makefiles"] if shutil.which("make") else []
        if shutil.which("ninja"):
            generators.append("Ninja")
        if not generators:
            self.skipTest("Make and Ninja are unavailable")
        for backend in generators:
            with self.subTest(generator=backend), tempfile.TemporaryDirectory() as temporary:
                source, build = Path(temporary) / "source", Path(temporary) / "build"
                for name in ("scripts/generate_cauchy_rhs.py", "scripts/manifests/cauchy_rhs_shapes.json", "cmake/DndsCauchyRHS.cmake"):
                    destination = source / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, destination)
                (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.21)
project(RHSCodeGeneration NONE)
include("${CMAKE_SOURCE_DIR}/cmake/DndsCauchyRHS.cmake")
dnds_generate_cauchy_rhs(rhs_sources rhs_include rhs_codegen)
add_custom_target(static_consumer ALL)
add_custom_target(shared_consumer ALL)
add_dependencies(static_consumer ${rhs_codegen})
add_dependencies(shared_consumer ${rhs_codegen})
file(APPEND "${CMAKE_BINARY_DIR}/configure_count" "configured\\n")
''')

                def run(*arguments):
                    result = subprocess.run(["cmake", *map(str, arguments)], stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout)

                run("-S", source, "-B", build, "-G", backend)
                run("--build", build, "--parallel", "4")
                output = build / "generated/cauchy_rhs"
                stable = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
                count = (build / "configure_count").read_text().count("configured")
                run("--build", build, "--parallel", "4")
                self.assertEqual(count, (build / "configure_count").read_text().count("configured"))
                self.assertEqual(stable, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})
                # Cover the first and secondary custom-command output, plus
                # the include and CMake index needed before source compilation.
                for name in ("FiniteVolumeRHSKernels_Instantiations.cpp", "FiniteVolumeRHSKernels_C7_M19_D3.cu",
                             "FiniteVolumeACMFluxKernels_C4_M19_D3.cu",
                             "FiniteVolumeRHSKernels_Instantiations.hpp", "FiniteVolumeRHSKernels_Sources.cmake"):
                    (output / name).unlink()
                    run("--build", build, "--parallel", "4")
                    self.assertTrue((output / name).is_file())
                manifest = source / "scripts/manifests/cauchy_rhs_shapes.json"
                data = json.loads(manifest.read_text())
                data["component_counts"].append(8)
                manifest.write_text(json.dumps(data))
                run("--build", build, "--parallel", "4")
                self.assertTrue((output / "FiniteVolumeRHSKernels_C8_M19_D3.cu").is_file())
                self.assertIn("case 8:", (output / "FiniteVolumeRHSKernels_Instantiations.cpp").read_text())
                # A generator-only change must also trigger reconfiguration,
                # even when it emits byte-identical numerical wrappers.
                count = (build / "configure_count").read_text().count("configured")
                with (source / "scripts/generate_cauchy_rhs.py").open("a") as stream:
                    stream.write("\n# Dependency tracking probe.\n")
                run("--build", build, "--parallel", "4")
                self.assertGreater((build / "configure_count").read_text().count("configured"), count)


if __name__ == "__main__":
    unittest.main()
