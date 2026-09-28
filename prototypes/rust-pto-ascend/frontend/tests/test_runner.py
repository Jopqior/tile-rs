"""Offline orchestration tests; no compiler or Ascend runtime required."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from support import FAKE_TILE, FIXTURES, copy_case, fixture_names, frontend, trace


class RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cases = self.root / "cases"
        self.cases.mkdir()
        self.sdk = self.root / "sdk"
        std = self.sdk / "crates/tile_std"
        (std / "src").mkdir(parents=True)
        (std / "Cargo.toml").write_text("[package]\n")
        (std / "src/lib.rs").write_text("// release DSL\n")
        (std / "src/tile.rs").write_text("// release tile\n")
        macros = self.sdk / "crates/tile_std_macros"
        macros.mkdir()
        (macros / "Cargo.toml").write_text("[package]\n")
        self.tile = self.root / "tile"
        shutil.copyfile(FAKE_TILE, self.tile)
        self.tile.chmod(0o755)
        self.output = self.root / "output"
        self.trace = self.root / "trace.jsonl"
        fake_env = patch.dict(os.environ, {
            "FAKE_FIXTURES": str(FIXTURES), "FAKE_TRACE": str(self.trace),
            "RUSTFLAGS": "must-be-removed",
        })
        fake_env.start()
        self.addCleanup(fake_env.stop)
        head = patch.object(frontend, "git_head", return_value=frontend.SDK_COMMIT)
        head.start()
        self.addCleanup(head.stop)
        clean_sdk = patch.object(frontend.subprocess, "check_output", return_value="")
        clean_sdk.start()
        self.addCleanup(clean_sdk.stop)

    def add_cases(self):
        for name in fixture_names():
            copy_case(self.cases, name)

    def run_suite(self):
        return frontend.run_suite(self.tile, self.sdk, self.output, self.cases)

    def test_two_stages_provenance_isolation_and_actual_digests(self):
        self.add_cases()
        self.assertEqual(self.run_suite(), 0)
        invocations = trace(self.trace)
        self.assertEqual([(call["case"], call["stage"]) for call in invocations],
                         [(name, stage) for name in fixture_names() for stage in ("msl", "pto")])
        provenance = (self.output / "PROVENANCE.txt").read_text()
        self.assertIn(frontend.SDK_COMMIT, provenance)
        self.assertIn(frontend.BACKEND_RELEASE, provenance)
        for name in fixture_names():
            with self.subTest(case=name):
                dest = self.output / "cases" / name
                source = self.cases / name / f"{name}.rs"
                mlir = dest / f"{name}.mlir"
                pto = dest / f"{name}.pto.mlir"
                original = FIXTURES / name
                self.assertEqual((dest / "kernel" / f"{name}.2.mlir.mlir").read_bytes(),
                                 (original / "compiler.mlir").read_bytes())
                self.assertEqual(mlir.read_bytes(), (original / "compiler.mlir").read_bytes())
                self.assertEqual(pto.read_bytes(), (original / "emitted.pto.mlir").read_bytes())
                self.assertIn("fake msl", (dest / "kernel.log").read_text())
                self.assertIn("fake pto", (dest / "pto.log").read_text())
                expected_sums = "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                    for path in (mlir, pto)
                )
                self.assertEqual((dest / "SHA256SUMS").read_text(), expected_sums)
                self.assertIn(f"Case {name}: {source} (source SHA256: "
                              f"{hashlib.sha256(source.read_bytes()).hexdigest()})", provenance)
                first, second = (call for call in invocations if call["case"] == name)
                self.assertEqual(first["args"], [str(source), "-t", "msl",
                                                 "--keep-dir", str(dest / "kernel"),
                                                 "-o", str(dest / f"{name}.metal")])
                self.assertEqual(second["args"], [str(mlir), "-t", "pto", "-o", str(pto)])
                self.assertEqual(first["input_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertEqual(second["input_sha256"], hashlib.sha256(mlir.read_bytes()).hexdigest())
                for call in (first, second):
                    self.assertIsNone(call["rustflags"])
                    self.assertEqual(call["tile_std_path"], str(self.sdk / "crates/tile_std"))

    def test_failed_first_case_does_not_stop_later_cases_and_clears_stale_files(self):
        self.add_cases()
        first, later = fixture_names()[:2]
        self.assertLess(first, later)
        self.assertEqual(self.run_suite(), 0)
        self.trace.unlink()
        with patch.dict(os.environ, {"FAKE_FAIL_CASE": first, "FAKE_FAIL_STAGE": "pto"}):
            self.assertEqual(self.run_suite(), 1)
        self.assertEqual([(call["case"], call["stage"]) for call in trace(self.trace)],
                         [(name, stage) for name in fixture_names() for stage in ("msl", "pto")])
        failed = self.output / "cases" / first
        self.assertIn("command exited 9", (failed / "ERROR.txt").read_text())
        self.assertTrue((failed / "pto.log").is_file())
        self.assertFalse((failed / f"{first}.pto.mlir").exists())
        mlir = failed / f"{first}.mlir"
        self.assertEqual((failed / "SHA256SUMS").read_text(),
                         f"{hashlib.sha256(mlir.read_bytes()).hexdigest()}  {mlir.name}\n")
        self.assertTrue((self.output / "cases" / later / f"{later}.pto.mlir").is_file())
        self.trace.unlink()
        with patch.dict(os.environ, {"FAKE_OMIT_CASE": first, "FAKE_OMIT_STAGE": "msl"}):
            self.assertEqual(self.run_suite(), 1)
        self.assertEqual([(call["case"], call["stage"]) for call in trace(self.trace)],
                         [(first, "msl")] +
                         [(name, stage) for name in fixture_names()[1:] for stage in ("msl", "pto")])
        self.assertFalse(mlir.exists())
        self.assertFalse((failed / f"{first}.pto.mlir").exists())
        self.assertEqual((failed / "SHA256SUMS").read_text(), "")
        self.assertIn("missing or empty file", (failed / "ERROR.txt").read_text())
        self.assertTrue((self.output / "cases" / later / f"{later}.pto.mlir").is_file())

    def test_cli_need_not_be_utf8(self):
        self.add_cases()
        # A valid non-UTF-8 executable must not be decoded during setup.
        self.tile.write_bytes(FAKE_TILE.read_text().replace(
            "#!/usr/bin/env python3\n", "#!/usr/bin/env python3\n# coding: latin-1\n# \xff\n"
        ).encode("latin-1"))
        self.assertEqual(self.run_suite(), 0)

    def test_output_cannot_overlap_inputs(self):
        self.add_cases()
        case = self.cases / fixture_names()[0]
        for output in (self.root, self.cases, case, self.sdk, self.tile):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlaps"):
                frontend.run_suite(self.tile, self.sdk, output, self.cases)
        self.assertTrue((case / f"{case.name}.rs").is_file())
        self.assertTrue(self.tile.is_file())

    def test_rejects_empty_suite_and_wrong_sdk_without_stale_artifacts(self):
        self.assertEqual(self.run_suite(), 1)
        self.assertIn("empty frontend suite", (self.output / "ERROR.txt").read_text())
        self.add_cases()
        self.assertEqual(self.run_suite(), 0)
        with patch.object(frontend, "git_head", return_value="unexpected"):
            self.assertEqual(self.run_suite(), 1)
        self.assertIn("release SDK HEAD", (self.output / "ERROR.txt").read_text())
        self.assertFalse((self.output / "PROVENANCE.txt").exists())
        self.assertFalse((self.output / "cases").exists())

    def test_rejects_modified_release_sdk(self):
        self.add_cases()
        with patch.object(frontend.subprocess, "check_output", return_value=" M crates/tile_std/src/lib.rs\n"):
            self.assertEqual(self.run_suite(), 1)
        self.assertIn("release SDK has local changes", (self.output / "ERROR.txt").read_text())

    def test_rejects_incompatible_dsl_and_missing_or_empty_case_input(self):
        self.add_cases()
        (self.sdk / "crates/tile_std/src/tile.rs").write_text("__tile_window_mask_f32")
        self.assertEqual(self.run_suite(), 1)
        self.assertIn("incompatible release DSL", (self.output / "ERROR.txt").read_text())
        (self.sdk / "crates/tile_std/src/tile.rs").write_text("// valid")
        first, later = fixture_names()[:2]
        (self.cases / first / "expectations.json").unlink()
        (self.cases / later / f"{later}.rs").write_text("")
        self.assertEqual(self.run_suite(), 1)
        for name in (first, later):
            with self.subTest(case=name):
                self.assertTrue((self.output / "cases" / name / "ERROR.txt").is_file())
        self.assertIn("missing or empty file", (self.output / "cases" / later / "ERROR.txt").read_text())
        self.assertFalse(self.trace.exists())


if __name__ == "__main__":
    unittest.main()
