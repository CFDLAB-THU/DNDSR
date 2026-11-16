"""Allocation export tooling; no solver build or extension imports required."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "diagnostics/allocation_explorer.py"
SPEC = importlib.util.spec_from_file_location("allocation_explorer", SCRIPT)
EXPLORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPLORER)

FIXTURE = """Runtime allocation ownership: live_bytes=768 peak_bytes=1024 pinned_host_bytes=128 pid=12
10 cuda:0 bytes=512 provider=native name=CFV/VR/localInverse site=/src/prepare.cpp:42 Prepare()
10 cuda:0 bytes=512 provider=native name=solver/sharedInverse
11 host:-1 bytes=128 provider=native pinned_host=1
12 cuda:0 bytes=128 provider=external
"""


class AllocationExplorerTest(unittest.TestCase):
    def test_identity_aliases_and_totals(self):
        snapshot = EXPLORER.parse_snapshots(FIXTURE, "fixture")[0]
        self.assertEqual(snapshot["unique_bytes"], 768)
        self.assertEqual(snapshot["summary_difference_bytes"], 0)
        self.assertEqual(len(snapshot["allocations"]), 3)
        self.assertEqual(snapshot["allocations"][0]["names"],
                         ["CFV/VR/localInverse", "solver/sharedInverse"])
        self.assertIsNone(snapshot["unknown_provider_bytes"])
        with self.assertRaisesRegex(ValueError, "conflicting bytes"):
            EXPLORER.parse_snapshots(FIXTURE + "10 cuda:0 bytes=513 provider=native\n", "conflict")

    def test_snapshots_keep_process_identity_separate(self):
        snapshots = EXPLORER.parse_snapshots(FIXTURE + FIXTURE.replace("pid=12", "pid=13"), "fixture")
        self.assertEqual([snapshot["unique_bytes"] for snapshot in snapshots], [768, 768])

    def test_safe_fragment_and_standalone_boundary(self):
        data = {"synthetic": True, "snapshots": EXPLORER.parse_snapshots(
            FIXTURE.replace("solver/sharedInverse", "</script><unsafe>"), "fixture")}
        fragment = EXPLORER.render_html(data, fragment=True)
        standalone = EXPLORER.render_html(data)
        self.assertNotIn("<!doctype", fragment)
        self.assertNotIn("</script><unsafe>", fragment)
        self.assertIn("\\u003c/script\\u003e", fragment)
        self.assertNotIn("offline-tooltip", fragment)
        self.assertIn("offline-tooltip", standalone)
        self.assertNotIn("fetch(", standalone)
        with self.assertRaisesRegex(ValueError, "budget"):
            EXPLORER.render_html(data, max_embedded_bytes=10)

    def test_gzip_archive_cli_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="dndsr-allocations-") as temporary:
            directory = Path(temporary)
            source = directory / "snapshot.txt.gz"
            with gzip.open(source, "wt") as stream:
                stream.write(FIXTURE)
            archive = directory / "evidence/allocations.json.gz"
            first = directory / "first/explorer.html"
            second = directory / "second/explorer.html"
            for arguments in (("--input", f"fixture={source}", "--output", str(first), "--json", str(archive)),
                              ("--data", str(archive), "--output", str(second))):
                result = subprocess.run([sys.executable, str(SCRIPT), *arguments],
                                        text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
            with gzip.open(archive, "rt") as stream:
                data = json.load(stream)
            self.assertEqual(data["snapshots"][0]["unique_bytes"], 768)
            for output in (first, second):
                html = output.read_text()
                self.assertTrue(html.startswith("<!doctype html>"))
                self.assertIn('class="payload"', html)
                self.assertNotIn('src="http', html)


if __name__ == "__main__":
    unittest.main()
