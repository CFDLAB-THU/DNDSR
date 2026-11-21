"""Generate the readable test lists using configured CMake targets and pytest collection.

Run with the active build's Python environment and installed extension modules.
Pass additional build directories to include mutually exclusive device profiles.
"""
import argparse
import ast
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/tests"
CPP = ROOT / "test/cpp"
CASE = re.compile(r'\bTEST_CASE(?:_TEMPLATE)?\(\s*"([^"]+)"')
INCLUDE = re.compile(r'^\s*#include "([^"]+)"', re.MULTILINE)


def defined_cases(source, seen):
    if source in seen:
        return
    seen.add(source)
    content = source.read_text()
    for name in CASE.findall(content):
        yield source, name
    for include in INCLUDE.findall(content):
        child = (source.parent / include).resolve()
        if child.is_relative_to(CPP) and child.is_file():
            yield from defined_cases(child, seen)


def cpp_catalog(builds):
    targets = defaultdict(lambda: defaultdict(set))
    sources = defaultdict(set)
    for build in builds:
        for row in (build / "dnds-test-catalog.tsv").read_text().splitlines():
            test, level, owner, files = row.split("\t")
            targets[owner][level].add(test)
            for file in files.split(";"):
                source = CPP / file
                if source.is_file():
                    sources[owner].add(source.resolve())
    out = ["# C++ test catalog", "",
           "Each heading is a build target. The table gives the minimum CI level of each CTest invocation. "
           "The source lists below it give the doctest case names in that binary. "
           "Rank, build features and doctest filters can restrict an invocation to part of its binary. "
           "Template cases appear once per definition. The list combines the configured profiles supplied to the generator.", ""]
    for target, levels in sorted(targets.items(), key=lambda pair: (min(pair[1]), pair[0])):
        out.extend([f"## `{target}`", "", "| Level | CTest registrations |", "|---|---|"])
        for level, names in sorted(levels.items()):
            out.append(f"| {level} | " + ", ".join(f"`{name}`" for name in sorted(names)) + " |")
        cases = defaultdict(list)
        seen = set()
        for file in sorted(sources[target]):
            for source, name in defined_cases(file, seen):
                cases[source].append(name)
        for source in sorted(set(cases) | sources[target]):
            path = source.relative_to(ROOT)
            out.extend(["", f"[{path.name}](../../{path})", ""])
            out.extend(f"- {name}" for name in cases[source])
        if target == "cauchy_solver":
            out.extend(["", "[Frontend test driver](../../test/Cauchy/frontend_smoke.py)"])
        out.append("")
    return "\n".join(out)


def python_nodes(level, lane):
    args = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--with-mpi", "--strict-markers",
            f"--dnds-test-level={level}", f"--dnds-ci-lane={lane}", "-m", f"level{level}", "test/"]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return {line for line in result.stdout.splitlines() if line.startswith("test/") and "::" in line}


def python_catalog():
    out = ["# Python test catalog", "", "Each row is a collected pytest case or parameterization. "
           "The columns show its minimum level and the CI lanes that select it. "
           "Runtime skips still depend on hardware, optional packages, and rank count.", ""]
    cases = defaultdict(list)
    for level in range(3):
        serial = python_nodes(level, "serial")
        mpi = python_nodes(level, "mpi")
        both = serial & mpi
        for node in sorted(serial | mpi):
            file, name = node.split("::", 1)
            lanes = "1, 2" if node in both else "1" if node in serial else "2"
            cases[file].append((level, lanes, name))
    for file, rows in sorted(cases.items()):
        out.extend([f"## [{file}](../../{file})", "", "| Level | MPI ranks | Case |", "|---|---|---|"])
        out.extend(f"| {level} | {lanes} | `{name.replace('|', '&#124;')}` |" for level, lanes, name in rows)
        out.append("")
    out.extend(["## Standalone tooling", "", "CTest runs these unittest methods in `harness_tooling` at level 1.", ""])
    for source in sorted((ROOT / "scripts/tests").glob("test_*.py")):
        path = source.relative_to(ROOT)
        out.extend([f"### [{path.name}](../../{path})", ""])
        for cls in ast.parse(source.read_text()).body:
            if isinstance(cls, ast.ClassDef):
                out.extend(f"- `{cls.name}.{f.name}`" for f in cls.body
                           if isinstance(f, ast.FunctionDef) and f.name.startswith("test_"))
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", nargs="+", type=Path)
    builds = parser.parse_args().build
    (DOCS / "catalog-cpp.md").write_text(cpp_catalog(builds))
    (DOCS / "catalog-python.md").write_text(python_catalog())


if __name__ == "__main__":
    main()
