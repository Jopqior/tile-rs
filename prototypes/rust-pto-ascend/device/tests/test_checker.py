"""Offline tests for the add case's numeric oracle (no Ascend SDK required)."""

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CHECKER_PATH = Path(__file__).resolve().parents[2] / "cases/add_generated/device/check.py"
spec = importlib.util.spec_from_file_location("add_generated_numeric_checker", CHECKER_PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class NumericCheckerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def host(self, body):
        binary = self.directory / "fake-host"
        binary.write_text("#!/usr/bin/env python3\n" + body)
        binary.chmod(0o755)
        return binary

    def test_normal_cli_result(self):
        binary = self.host(
            "import sys\n"
            "from pathlib import Path\n"
            "input_path, output_path = map(Path, sys.argv[1:])\n"
            "pairs = (line.split() for line in input_path.read_text().splitlines())\n"
            "output_path.write_text(''.join(f'{float(a) + float(b)}\\n' for a, b in pairs))\n"
            "print('host ran')\n"
        )
        result = subprocess.run(
            [sys.executable, str(CHECKER_PATH), "./fake-host", str(self.directory)],
            cwd=self.directory, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("256/256 exact matches", result.stdout)
        self.assertEqual(result.stderr, "")
        inputs = (self.directory / "inputs.txt").read_text().splitlines()
        self.assertEqual(len(inputs), 256)
        self.assertEqual(inputs[:4], ["0.0 0.0", "1.0 -1.0", "-2.0 -0.5", "0.125 0.25"])
        self.assertEqual((self.directory / "runtime-private.log").read_text(), "host ran\n")

        negative = subprocess.run(
            [sys.executable, str(CHECKER_PATH), str(binary), str(self.directory), "--inject-error"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(negative.returncode, 0)
        self.assertIn("NEGATIVE PROBE", negative.stdout)
        self.assertNotIn("PASS:", negative.stdout)
        self.assertIn("FAIL: index=0", negative.stderr)
        self.assertEqual((self.directory / "outputs.txt").read_text().splitlines()[0], "1.0")

    def test_mutated_nonfinite_count_and_malformed_outputs_fail(self):
        pairs = checker.make_pairs()
        self.assertEqual(len(pairs), 256)
        lines = [str(a + b) for a, b in pairs]
        cases = [
            ("mutation", ["1.0", *lines[1:]], "index=0"),
            ("nan", ["nan", *lines[1:]], "index=0"),
            ("infinity", ["inf", *lines[1:]], "index=0"),
            ("too few", lines[:-1], "expected 256 outputs, got 255"),
            ("too many", [*lines, "0.0"], "expected 256 outputs, got 257"),
            ("malformed", ["not-a-float", *lines[1:]], "index=0, invalid numeric output"),
        ]
        output_path = self.directory / "outputs.txt"
        for name, values, message in cases:
            with self.subTest(name=name):
                output_path.write_text("\n".join(values) + "\n")
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    checker.check_outputs(output_path, pairs)

    def test_failing_host_cannot_reuse_stale_output(self):
        output_path = self.directory / "outputs.txt"
        output_path.write_text("stale result\n")
        binary = self.host("import sys\nprint('host failed')\nsys.exit(7)\n")
        with self.assertRaisesRegex(ValueError, "ACL host exit=7"):
            checker.run_check(binary, self.directory)
        self.assertFalse(output_path.exists())
        self.assertEqual((self.directory / "runtime-private.log").read_text(), "host failed\n")

    def test_successful_host_without_output_cannot_reuse_stale_output(self):
        output_path = self.directory / "outputs.txt"
        output_path.write_text("stale result\n")
        binary = self.host("pass\n")
        with self.assertRaisesRegex(ValueError, "ACL host produced no outputs"):
            checker.run_check(binary, self.directory)
        self.assertFalse(output_path.exists())

    def test_host_timeout_reports_failure(self):
        with mock.patch.object(checker.subprocess, "run", side_effect=subprocess.TimeoutExpired("host", 300)) as run:
            with self.assertRaisesRegex(ValueError, "ACL host timed out after 300s"):
                checker.run_check(self.directory / "host", self.directory)
        self.assertEqual(run.call_args.kwargs["timeout"], 300)


if __name__ == "__main__":
    unittest.main()
