"""Runtime selection must fail closed before compiling or touching a device."""

import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("device_runtime", Path(__file__).resolve().parents[1] / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RuntimeTests(unittest.TestCase):
    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.runtime = self.home / "tools/simulator/Ascend910B1/lib"
        self.runtime.mkdir(parents=True)
        (self.runtime / "libruntime_camodel.so").write_text("fake runtime")
        self.start_patch(patch.dict(os.environ, {
            "ASCEND_HOME_PATH": str(self.home), "ASCEND_DEVICE_ID": "0",
            "ASCEND_SOC": "Ascend910B1", "LD_LIBRARY_PATH": "/existing/lib",
        }, clear=True))
        self.start_patch(patch.object(runner.os, "uname", return_value=SimpleNamespace(machine="x86_64")))
        self.devices = self.start_patch(patch.object(Path, "glob", return_value=[]))
        self.exists = self.start_patch(patch.object(Path, "exists", return_value=False))

    def test_selects_runtime_without_changing_parent_environment(self):
        sim = runner.runtime_environment("sim")
        self.assertEqual(sim["EXECUTION_MODE"], "sim")
        self.assertEqual(sim["LD_LIBRARY_PATH"], f"{self.runtime}:/existing/lib")
        npu = runner.runtime_environment("npu")
        self.assertEqual(npu["EXECUTION_MODE"], "npu")
        self.assertEqual(npu["LD_LIBRARY_PATH"], "/existing/lib")
        self.assertNotIn("EXECUTION_MODE", os.environ)

    def test_simulator_rejects_real_devices_wrong_host_and_missing_runtime(self):
        with patch.object(runner.os, "uname", return_value=SimpleNamespace(machine="aarch64")):
            with self.assertRaisesRegex(ValueError, "x86_64"):
                runner.runtime_environment("sim")
        self.devices.return_value = [Path("/dev/davinci0")]
        with self.assertRaisesRegex(ValueError, "exposed NPU"):
            runner.runtime_environment("sim")
        self.devices.return_value = []
        self.exists.return_value = True
        with self.assertRaisesRegex(ValueError, "exposed NPU"):
            runner.runtime_environment("sim")
        self.exists.return_value = False
        (self.runtime / "libruntime_camodel.so").unlink()
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            runner.runtime_environment("sim")

    def test_device_id_is_required_and_validated(self):
        for device in ("", "-1", "65536", "not-a-device", "1.0"):
            with self.subTest(device=device), patch.dict(os.environ, {"ASCEND_DEVICE_ID": device}):
                with self.assertRaisesRegex(ValueError, "ASCEND_DEVICE_ID"):
                    runner.runtime_environment("npu")
        with patch.dict(os.environ, {"ASCEND_DEVICE_ID": "1"}):
            with self.assertRaisesRegex(ValueError, "ASCEND_DEVICE_ID=0"):
                runner.runtime_environment("sim")

    def test_npu_rejects_unsupported_soc_and_simulator_libraries(self):
        for override in ({"ASCEND_SOC": ""}, {"ASCEND_SOC": "Ascend310"},
                         {"LD_LIBRARY_PATH": "/sdk/tools/simulator/lib"},
                         {"LD_PRELOAD": "libruntime_camodel.so"}):
            with self.subTest(override=override), patch.dict(os.environ, override):
                with self.assertRaises(ValueError):
                    runner.runtime_environment("npu")


if __name__ == "__main__":
    unittest.main()
