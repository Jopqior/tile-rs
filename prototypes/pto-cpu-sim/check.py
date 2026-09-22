"""Throwaway independent scalar oracle; standard library only."""
import math
from pathlib import Path
import subprocess
import sys

binary, directory, inject = sys.argv[1:]
directory = Path(directory)
# Small dyadic values: inputs and sums are exactly representable in binary32.
pairs = [((i % 31 - 15) / 8, ((i * 7) % 37 - 18) / 16) for i in range(256)]
pairs[:4] = [(0.0, 0.0), (1.0, -1.0), (-2.0, -0.5), (0.125, 0.25)]
input_path, output_path = directory / "inputs.txt", directory / "outputs.txt"
input_path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
subprocess.run([binary, str(input_path), str(output_path)], check=True)
actual = [float(line) for line in output_path.read_text().splitlines()]
if inject == "1":
    actual[0] += 1.0
    output_path.write_text("".join(f"{value}\n" for value in actual))
    print("NEGATIVE PROBE: changed CPU-SIM output[0] by +1.0", flush=True)
expected = [a + b for a, b in pairs]  # No PTO code or output used by the oracle.
if len(actual) != len(expected):
    raise SystemExit(f"FAIL: expected 256 outputs, got {len(actual)}")
for i, (got, want) in enumerate(zip(actual, expected)):
    if not math.isfinite(got) or got != want:
        raise SystemExit(f"FAIL: index={i}, actual={got}, reference={want}, atol=rtol=0")
print("PASS: PTO CPU-SIM f32 add, shape=[1,256], 256/256 exact matches, atol=rtol=0")
