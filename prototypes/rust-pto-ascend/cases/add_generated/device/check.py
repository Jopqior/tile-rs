"""Independent scalar oracle for the generated f32 add kernel; standard library only."""

import argparse
import math
from pathlib import Path
import subprocess


def make_pairs():
    pairs = [((i % 31 - 15) / 8, ((i * 7) % 37 - 18) / 16) for i in range(256)]
    pairs[:4] = [(0.0, 0.0), (1.0, -1.0), (-2.0, -0.5), (0.125, 0.25)]
    return pairs


def check_outputs(output_path, pairs, inject_error=False):
    try:
        lines = output_path.read_text().splitlines()
    except FileNotFoundError as exc:
        raise ValueError("ACL host produced no outputs; no numeric PASS") from exc
    if len(lines) != len(pairs):
        raise ValueError(f"expected {len(pairs)} outputs, got {len(lines)}")

    actual = []
    for i, line in enumerate(lines):
        try:
            actual.append(float(line))
        except ValueError as exc:
            raise ValueError(f"index={i}, invalid numeric output: {line!r}") from exc

    if inject_error:
        actual[0] += 1.0
        output_path.write_text("".join(f"{value}\n" for value in actual))
        print("NEGATIVE PROBE: changed device output[0] by +1.0", flush=True)

    for i, (got, (a, b)) in enumerate(zip(actual, pairs)):
        want = a + b  # Exact dyadic binary32 sums, independent of PTO.
        if not math.isfinite(got) or got != want:
            raise ValueError(f"index={i}, actual={got}, reference={want}, atol=rtol=0")


def run_check(binary, directory, inject_error=False):
    pairs = make_pairs()
    input_path, output_path = directory / "inputs.txt", directory / "outputs.txt"
    input_path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    output_path.unlink(missing_ok=True)

    # Simulator diagnostics stay private, outside the artifact allowlist.
    with (directory / "runtime-private.log").open("w") as log:
        try:
            result = subprocess.run(
                [str(binary.resolve()), str(input_path), str(output_path)],
                stdout=log, stderr=log, timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("ACL host timed out after 300s; no numeric PASS") from exc
    if result.returncode:
        raise ValueError(f"ACL host exit={result.returncode}; no numeric PASS")
    check_outputs(output_path, pairs, inject_error=inject_error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, metavar="BINARY")
    parser.add_argument("directory", type=Path, metavar="DIRECTORY")
    parser.add_argument("--inject-error", action="store_true", help="corrupt output for a negative test")
    args = parser.parse_args(argv)
    try:
        run_check(args.binary, args.directory, inject_error=args.inject_error)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print("PASS: f32 add, shape=[1,256], 256/256 exact matches, atol=rtol=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
