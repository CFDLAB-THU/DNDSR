#!/usr/bin/env python3
"""
Run CTest and aggregate doctest summary statistics.

Usage:
    python scripts/ctest_summary.py [ctest args...]
    
Example:
    python scripts/ctest_summary.py -R "^dnds_" --output-on-failure
    python scripts/ctest_summary.py -j4
"""

import subprocess
import sys
import re
from pathlib import Path


def main():
    # Build ctest command
    ctest_args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Always need verbose output to capture doctest summaries
    if '-V' not in ctest_args and '--verbose' not in ctest_args:
        ctest_args = ['-V'] + ctest_args

    cmd = ['ctest'] + ctest_args

    # Run ctest and capture output
    print(f"Running: {' '.join(cmd)}\n")
    print("=" * 70)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Pattern to match doctest summary lines
    # [doctest] test cases:    45 |    45 passed | 0 failed | 0 skipped
    # [doctest] assertions: 85242 | 85242 passed | 0 failed |
    test_case_pattern = re.compile(
        r'\[doctest\] test cases:\s*(\d+)\s*\|\s*(\d+)\s*passed\s*\|\s*(\d+)\s*failed'
    )
    assertion_pattern = re.compile(
        r'\[doctest\] assertions:\s*(\d+)\s*\|\s*(\d+)\s*passed\s*\|\s*(\d+)\s*failed'
    )

    total_test_cases = 0
    total_test_cases_passed = 0
    total_test_cases_failed = 0
    total_assertions = 0
    total_assertions_passed = 0
    total_assertions_failed = 0
    test_count = 0

    # Stream output and parse
    for line in process.stdout:
        print(line, end='')

        # Check for doctest summary
        tc_match = test_case_pattern.search(line)
        if tc_match:
            total_test_cases += int(tc_match.group(1))
            total_test_cases_passed += int(tc_match.group(2))
            total_test_cases_failed += int(tc_match.group(3))
            test_count += 1

        assert_match = assertion_pattern.search(line)
        if assert_match:
            total_assertions += int(assert_match.group(1))
            total_assertions_passed += int(assert_match.group(2))
            total_assertions_failed += int(assert_match.group(3))

    process.wait()

    # Print aggregated summary
    print("\n" + "=" * 70)
    print("DOCTEST AGGREGATE SUMMARY")
    print("=" * 70)

    if test_count > 0:
        print(f"Test executables run:  {test_count}")
        print(
            f"Total test cases:      {total_test_cases:6} | {total_test_cases_passed:6} passed | {total_test_cases_failed} failed")
        print(
            f"Total assertions:      {total_assertions:6} | {total_assertions_passed:6} passed | {total_assertions_failed} failed")

        if total_test_cases_failed > 0 or total_assertions_failed > 0:
            print("\nStatus: FAILED")
        else:
            print("\nStatus: SUCCESS")
    else:
        print("No doctest output found (tests may have been filtered out or not built)")

    print("=" * 70)

    return process.returncode


if __name__ == '__main__':
    sys.exit(main())
