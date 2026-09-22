"""Throwaway independent scalar oracle; standard library only."""
import math
from pathlib import Path
import subprocess
import sys

binary, directory, inject = sys.argv[1:]
if inject not in ("0", "1"):
    raise SystemExit("INJECT_ERROR must be 0 or 1")
directory = Path(directory)
pairs = [((i % 31 - 15) / 8, ((i * 7) % 37 - 18) / 16) for i in range(256)]
pairs[:4] = [(0.0, 0.0), (1.0, -1.0), (-2.0, -0.5), (0.125, 0.25)]
input_path, output_path = directory / "inputs.txt", directory / "outputs.txt"
input_path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
output_path.unlink(missing_ok=True)
# Simulator diagnostics stay private, outside the artifact allowlist.
with (directory / "runtime-private.log").open("w") as log:
    result = subprocess.run([binary, str(input_path), str(output_path)], stdout=log, stderr=log, timeout=300)
if result.returncode:
    raise SystemExit(f"FAIL: ACL/camodel host exit={result.returncode}; no numeric PASS")
actual = [float(line) for line in output_path.read_text().splitlines()]
if len(actual) != 256:
    raise SystemExit(f"FAIL: expected 256 outputs, got {len(actual)}")
if inject == "1":
    actual[0] += 1.0
    output_path.write_text("".join(f"{value}\n" for value in actual))
    print("NEGATIVE PROBE: changed camodel output[0] by +1.0", flush=True)
expected = [a + b for a, b in pairs]  # Exact dyadic binary32 sums, independent of PTO.
for i, (got, want) in enumerate(zip(actual, expected)):
    if not math.isfinite(got) or got != want:
        raise SystemExit(f"FAIL: index={i}, actual={got}, reference={want}, atol=rtol=0")
print("PASS: PTO camodel f32 add, shape=[1,256], 256/256 exact matches, atol=rtol=0")
