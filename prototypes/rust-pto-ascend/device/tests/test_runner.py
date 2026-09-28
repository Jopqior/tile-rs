"""Offline tests for device.run_suite; fake programs replace compiler/Ascend binaries."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[1] / "run.py"
spec = importlib.util.spec_from_file_location("device_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

GENERATED_CPP = "// synthetic generated PTO C++\n"
FAKE_TOOL = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
case = (Path(args[0]).parent.name if name == "ptoas" else
        Path.cwd().name if "--version" not in args else "inventory")
with open(os.environ["FAKE_TRACE"], "a") as trace:
    trace.write(json.dumps({"tool": name, "case": case, "args": args}) + "\n")
print(f"fake {name}: {case}")
if name == os.environ.get("FAKE_FAIL_TOOL") and case == os.environ.get("FAKE_FAIL_CASE"):
    print("deliberate compiler failure", file=sys.stderr)
    sys.exit(19)
if "--version" in args or name == "npu-smi":
    sys.exit(0)
if name == "ptoas":
    target = Path(args[args.index("-o") + 1])
    if case != os.environ.get("FAKE_OMIT_CASE"):
        target.write_text("// synthetic generated PTO C++\n")
elif name == "bisheng":
    Path(args[args.index("-o") + 1]).write_text("fake kernel library\n")
elif name == "g++":
    host = Path(args[args.index("-o") + 1])
    host.write_text('#!/usr/bin/env python3\n'
                    'from pathlib import Path\nimport sys\n'
                    'Path(sys.argv[1], "runtime-private.log").write_text("secret runtime diagnostics\\n")\n')
    host.chmod(0o755)
'''

FAKE_CHECKER = '''import os
from pathlib import Path
import subprocess
import sys

host, work = sys.argv[1:]
assert Path(host).is_file()
subprocess.run([host, work], check=True)
Path(work, "inputs.txt").write_text("1 2\\n")
Path(work, "outputs.txt").write_text("3\\n")
if os.environ.get("FAKE_FAIL_TOOL") == "checker" and os.environ.get("FAKE_FAIL_CASE") == Path(work).name:
    print("deliberate numeric failure", file=sys.stderr)
    sys.exit(17)
print("numeric check passed")
'''


class DeviceRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cases = self.root / "cases"
        self.source = self.root / "frontend"
        self.output = self.root / "output"
        self.tools = self.root / "tools"
        self.cann = self.root / "cann"
        self.bin = self.root / "bin"
        self.trace = self.root / "trace.jsonl"
        for directory in (self.cases, self.source, self.tools, self.bin,
                          self.cann / "tools/simulator/Ascend910B1/lib", self.cann / "lib64"):
            directory.mkdir(parents=True, exist_ok=True)
        (self.tools / "versions.txt").write_text("fake compiler version\n")
        (self.cann / "tools/simulator/Ascend910B1/lib/libruntime_camodel.so").write_text("fake runtime\n")
        (self.tools / "venv/bin").mkdir(parents=True)
        for path in (self.bin / "bisheng", self.bin / "g++", self.bin / "npu-smi",
                     self.tools / "venv/bin/ptoas"):
            path.write_text(FAKE_TOOL)
            path.chmod(0o755)
        env = patch.dict(os.environ, {
            "ASCEND_HOME_PATH": str(self.cann), "ASCEND_DEVICE_ID": "0",
            "ASCEND_SOC": "Ascend910B1", "LD_LIBRARY_PATH": "", "LD_PRELOAD": "",
            "PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE_TRACE": str(self.trace),
            "FAKE_FAIL_TOOL": "", "FAKE_FAIL_CASE": "", "FAKE_OMIT_CASE": "",
        })
        env.start()
        self.addCleanup(env.stop)
        # The mode guards have dedicated tests; orchestration tests run on any host.
        runtime = patch.object(runner, "runtime_environment",
                               side_effect=lambda mode: dict(os.environ, EXECUTION_MODE=mode))
        runtime.start()
        self.addCleanup(runtime.stop)

    def add_case(self, name, *, device=True):
        case = self.cases / name
        case.mkdir()
        (case / f"{name}.rs").write_text("// synthetic source\n")
        (case / "expectations.json").write_text("{}\n")
        if device:
            device_dir = case / "device"
            device_dir.mkdir()
            (device_dir / "kernel.cpp").write_text("// fake kernel\n")
            (device_dir / "host.cpp").write_text("// fake host\n")
            (device_dir / "check.py").write_text(FAKE_CHECKER)
        incoming = self.source / "cases" / name
        incoming.mkdir(parents=True)
        mlir = incoming / f"{name}.mlir"
        pto = incoming / f"{name}.pto.mlir"
        mlir.write_text(f"// {name} frontend MLIR\n")
        pto.write_text(f"// {name} PTO MLIR\n")
        (incoming / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (mlir, pto)
        ))
        return case

    def add_two_cases(self):
        self.add_case("alpha")
        self.add_case("zebra")

    def run_suite(self, mode="sim", *, output=None):
        return runner.run_suite(mode, self.source, output or self.output, self.tools, self.cases)

    def summary(self):
        return json.loads((self.output / "artifacts/summary.json").read_text())

    def calls(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_sim_discovers_two_device_cases_skips_frontend_only_and_publishes_checksums(self):
        self.add_two_cases()
        self.add_case("frontend_only", device=False)
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(self.summary(), {"alpha": True, "zebra": True})
        self.assertEqual([call["case"] for call in self.calls() if call["tool"] == "ptoas"],
                         ["alpha", "zebra"])
        self.assertFalse((self.output / "artifacts/cases/frontend_only").exists())
        for name in ("alpha", "zebra"):
            with self.subTest(case=name):
                artifacts = self.output / "artifacts/cases" / name
                incoming = self.source / "cases" / name
                self.assertEqual((artifacts / "input.pto").read_bytes(),
                                 (incoming / f"{name}.pto.mlir").read_bytes())
                self.assertEqual((artifacts / "generated.cpp").read_text(), GENERATED_CPP)
                self.assertEqual((artifacts / "generated.sha256").read_text(),
                                 f"{hashlib.sha256(GENERATED_CPP.encode()).hexdigest()}  generated.cpp\n")
                self.assertIn("fake ptoas", (artifacts / "ptoas.log").read_text())
                self.assertIn("fake bisheng", (artifacts / "compile.log").read_text())
                self.assertIn("numeric check passed", (artifacts / "result.txt").read_text())
                self.assertEqual((artifacts / "inputs.txt").read_text(), "1 2\n")
                self.assertEqual((artifacts / "outputs.txt").read_text(), "3\n")
                self.assertEqual((self.output / "work" / name / "runtime-private.log").read_text(),
                                 "secret runtime diagnostics\n")
                self.assertFalse((artifacts / "runtime-private.log").exists())
                self.assertFalse((artifacts / "host").exists())
                self.assertFalse((artifacts / "libkernel.so").exists())
        self.assertFalse((self.output / "artifacts/work").exists())
        self.assertIn("MODE=sim", (self.output / "artifacts/inventory.txt").read_text())
        self.assertFalse((self.output / "artifacts/device.txt").exists())

    def test_npu_reuses_verified_generated_cpp_without_ptoas(self):
        self.add_two_cases()
        self.assertEqual(self.run_suite(), 0)
        generated = {}
        for name in ("alpha", "zebra"):
            artifacts = self.output / "artifacts/cases" / name
            incoming = self.source / "cases" / name
            for filename in ("generated.cpp", "generated.sha256"):
                shutil.copyfile(artifacts / filename, incoming / filename)
            generated[name] = (incoming / "generated.cpp").read_bytes()
        self.trace.unlink()
        self.assertEqual(self.run_suite("npu"), 0)
        self.assertEqual(self.summary(), {"alpha": True, "zebra": True})
        self.assertNotIn("ptoas", [call["tool"] for call in self.calls()])
        self.assertEqual([call["tool"] for call in self.calls() if call["tool"] == "npu-smi"],
                         ["npu-smi"])
        self.assertIn("MODE=npu", (self.output / "artifacts/inventory.txt").read_text())
        self.assertIn("fake npu-smi", (self.output / "artifacts/device.txt").read_text())
        for name in generated:
            artifacts = self.output / "artifacts/cases" / name
            self.assertEqual((artifacts / "generated.cpp").read_bytes(), generated[name])
            self.assertFalse((artifacts / "ptoas.log").exists())
            self.assertFalse((artifacts / "input.pto").exists())
            self.assertIn("numeric check passed", (artifacts / "result.txt").read_text())

    def test_empty_device_suite_fails_at_setup(self):
        self.add_case("frontend_only", device=False)
        self.assertEqual(self.run_suite(), 1)
        self.assertIn("no device cases", (self.output / "artifacts/ERROR.txt").read_text())
        self.assertFalse((self.output / "artifacts/summary.json").exists())
        self.assertFalse(self.trace.exists())

    def test_compile_link_and_checker_failures_do_not_stop_later_cases(self):
        self.add_two_cases()
        for tool, log, diagnostic in (("bisheng", "compile.log", "deliberate compiler failure"),
                                      ("g++", "link.log", "deliberate compiler failure"),
                                      ("checker", "result.txt", "deliberate numeric failure")):
            with self.subTest(tool=tool), patch.dict(os.environ, {
                    "FAKE_FAIL_TOOL": tool, "FAKE_FAIL_CASE": "alpha"}):
                self.assertEqual(self.run_suite(), 1)
                self.assertEqual(self.summary(), {"alpha": False, "zebra": True})
                failed = self.output / "artifacts/cases/alpha"
                self.assertIn(diagnostic, (failed / log).read_text())
                self.assertIn("exited", (failed / "ERROR.txt").read_text())
                self.assertIn("numeric check passed", (self.output / "artifacts/cases/zebra/result.txt").read_text())

    def test_missing_and_tampered_frontend_artifacts_fail_only_affected_case(self):
        self.add_two_cases()
        incoming = self.source / "cases/alpha"
        pto = incoming / "alpha.pto.mlir"
        original = pto.read_bytes()
        for state in ("missing", "tampered"):
            with self.subTest(state=state):
                if state == "missing":
                    pto.unlink()
                else:
                    pto.write_text("// malicious modified PTO\n")
                self.assertEqual(self.run_suite(), 1)
                self.assertEqual(self.summary(), {"alpha": False, "zebra": True})
                self.assertIn("alpha.pto.mlir", (self.output / "artifacts/cases/alpha/ERROR.txt").read_text())
                self.assertIn("numeric check passed", (self.output / "artifacts/cases/zebra/result.txt").read_text())
                self.assertEqual([call["case"] for call in self.calls() if call["tool"] == "ptoas"],
                                 ["zebra"])
                pto.write_bytes(original)
                self.trace.unlink()

    def test_npu_rejects_missing_and_tampered_generated_artifacts(self):
        self.add_two_cases()
        for name in ("alpha", "zebra"):
            incoming = self.source / "cases" / name
            generated = incoming / "generated.cpp"
            generated.write_text(GENERATED_CPP)
            (incoming / "generated.sha256").write_text(
                f"{hashlib.sha256(GENERATED_CPP.encode()).hexdigest()}  generated.cpp\n")
        generated = self.source / "cases/alpha/generated.cpp"
        for state in ("missing", "tampered"):
            with self.subTest(state=state):
                if state == "missing":
                    generated.unlink()
                else:
                    generated.write_text("// changed after simulator\n")
                self.assertEqual(self.run_suite("npu"), 1)
                self.assertEqual(self.summary(), {"alpha": False, "zebra": True})
                error = (self.output / "artifacts/cases/alpha/ERROR.txt").read_text()
                self.assertIn("generated.cpp", error)
                self.assertIn("numeric check passed", (self.output / "artifacts/cases/zebra/result.txt").read_text())
                self.assertNotIn("ptoas", [call["tool"] for call in self.calls()])
                generated.write_text(GENERATED_CPP)
                self.trace.unlink()

    def test_missing_generated_output_fails_even_if_compiler_exits_zero(self):
        self.add_two_cases()
        with patch.dict(os.environ, {"FAKE_OMIT_CASE": "alpha"}):
            self.assertEqual(self.run_suite(), 1)
        self.assertEqual(self.summary(), {"alpha": False, "zebra": True})
        self.assertIn("missing or empty file", (self.output / "artifacts/cases/alpha/ERROR.txt").read_text())

    def test_second_run_removes_stale_work_and_public_artifacts(self):
        self.add_two_cases()
        self.assertEqual(self.run_suite(), 0)
        stale_work = self.output / "work/alpha/stale-runtime.log"
        stale_artifact = self.output / "artifacts/cases/alpha/ERROR.txt"
        stale_work.write_text("old private data")
        stale_artifact.write_text("old failure")
        (self.output / "artifacts/old-case").write_text("old public data")
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(self.summary(), {"alpha": True, "zebra": True})
        for path in (stale_work, stale_artifact, self.output / "artifacts/old-case"):
            self.assertFalse(path.exists(), str(path))

    def test_output_overlap_rejected_without_deleting_any_input(self):
        self.add_two_cases()
        for output in (self.root, self.source, self.tools, self.cases, self.cases / "alpha",
                       self.source / "nested-output"):
            with self.subTest(output=output):
                # These look like runner-owned directories, but belong to protected inputs.
                for directory in (output / "work", output / "artifacts"):
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "sentinel").write_text("keep me")
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    self.run_suite(output=output)
                for directory in (output / "work", output / "artifacts"):
                    self.assertEqual((directory / "sentinel").read_text(), "keep me")
                self.assertTrue((self.cases / "alpha/device/kernel.cpp").is_file())
                self.assertTrue((self.source / "cases/alpha/alpha.pto.mlir").is_file())
                self.assertTrue((self.tools / "versions.txt").is_file())


class RealDeviceCaseContractTests(unittest.TestCase):
    def test_discovered_shared_cases_have_nonempty_device_and_frontend_sources(self):
        cases = runner.discover_cases(runner.CASES)
        self.assertGreaterEqual(len(cases), 1, "expected at least one real shared device case")
        for case in cases:
            with self.subTest(case=case.name):
                for path in (case / "device/kernel.cpp", case / "device/host.cpp",
                             case / "device/check.py", case / f"{case.name}.rs",
                             case / "expectations.json"):
                    self.assertTrue(path.is_file() and path.stat().st_size > 0, str(path))
                self.assertIsInstance(json.loads((case / "expectations.json").read_text()), dict)


if __name__ == "__main__":
    unittest.main()
