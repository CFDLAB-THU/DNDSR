"""Check device assertion opt-out scope without building the numerical solver."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("cmake"), "CMake is unavailable")
class CUDAAssertionScopeTests(unittest.TestCase):
    def test_source_definitions_are_local_and_default_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.21)
project(AssertionScope NONE)
include("''' + str(ROOT / "cmake/DndsCudaAssertions.cmake") + '''")
dnds_cuda_kernel_assertions(kernel.cu generated_shape.cu)
get_directory_property(directory_definitions COMPILE_DEFINITIONS)
file(WRITE "${CMAKE_BINARY_DIR}/definitions.txt" "directory=${directory_definitions}\\n")
foreach(source kernel.cu generated_shape.cu host.cpp independent_test.cu)
    get_source_file_property(definitions "${source}" COMPILE_DEFINITIONS)
    file(APPEND "${CMAKE_BINARY_DIR}/definitions.txt" "${source}=${definitions}\\n")
endforeach()
''')
            for enabled in ("ON", "OFF", "ON"):
                build = source / "build"
                subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                                "-DDNDS_CUDA_KERNEL_ASSERTS=" + enabled],
                               check=True, capture_output=True, text=True, timeout=30)
                definitions = dict(line.split("=", 1) for line in
                                   (build / "definitions.txt").read_text().splitlines())
                self.assertEqual(definitions["directory"], "")
                for name in ("kernel.cu", "generated_shape.cu"):
                    self.assertEqual(definitions[name], "DNDS_NDEBUG_DEVICE=1" if enabled == "OFF" else "NOTFOUND")
                for name in ("host.cpp", "independent_test.cu"):
                    self.assertEqual(definitions[name], "NOTFOUND")
