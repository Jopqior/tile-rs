#!/usr/bin/env python3
"""Run the release-matched Rust -> compiler MLIR -> current PTO frontend suite."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
CASES = Path(__file__).resolve().parent.parent / "cases"
SDK_COMMIT = "c3c8ec0169bd02757b51370ba9c6ec3b116b833d"
BACKEND_RELEASE = "v0.0.2+nightly-2025-08-04"


def git_head(directory):
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def check_sdk(sdk):
    head = git_head(sdk)
    if head != SDK_COMMIT:
        raise ValueError(f"release SDK HEAD {head} != pinned {SDK_COMMIT}")
    if subprocess.check_output(
        ["git", "-C", str(sdk), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip():
        raise ValueError(f"release SDK has local changes: {sdk}")
    std = sdk / "crates/tile_std"
    for path in (std / "Cargo.toml", sdk / "crates/tile_std_macros/Cargo.toml",
                 std / "src/lib.rs", std / "src/tile.rs"):
        if not path.is_file():
            raise ValueError(f"missing release SDK file: {path}")
    for path in (std / "src/lib.rs", std / "src/tile.rs"):
        if re.search(r"register_tool|__tile_window_mask_f32", path.read_text()):
            raise ValueError(f"incompatible release DSL symbol in {path}")
    return head, std


def expectations(path):
    data = json.loads(path.read_text())
    required = {"function", "pointer_type", "pointer_count", "mlir_intrinsic", "operations"}
    if set(data) != required:
        raise ValueError(f"{path}: expected keys {sorted(required)}, got {sorted(data)}")
    for key in ("function", "pointer_type", "mlir_intrinsic"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"{path}: {key} must be a nonempty string")
    if type(data["pointer_count"]) is not int or data["pointer_count"] < 1:
        raise ValueError(f"{path}: pointer_count must be a positive integer")
    operations = data["operations"]
    if not isinstance(operations, dict) or not operations or any(
        not isinstance(k, str) or not k or type(v) is not int or v < 0
        for k, v in operations.items()
    ):
        raise ValueError(f"{path}: operations must map names to nonnegative integer counts")
    return data


def check_outputs(mlir, pto, expected):
    if expected["mlir_intrinsic"] not in mlir:
        raise ValueError(f"missing MLIR intrinsic {expected['mlir_intrinsic']}")
    func = re.search(r"func\.func @" + re.escape(expected["function"]) + r"\(([^)]*)\)", pto)
    if not func:
        raise ValueError(f"missing {expected['function']} PTO function")
    args = re.findall(r"(%arg\d+): !pto\.ptr<" + re.escape(expected["pointer_type"]) + r">", func.group(1))
    if len(args) != expected["pointer_count"]:
        raise ValueError(f"expected {expected['pointer_count']} ordered {expected['pointer_type']} pointers, got {args}")
    for arg in args:
        if not re.search(r"pto\.make_tensor_view " + re.escape(arg) + r"\b", pto):
            raise ValueError(f"PTO does not make a tensor view from input pointer {arg}")
    active = "\n".join(line for line in pto.splitlines()
                       if not line.lstrip().startswith("//"))
    definitions = set(args) | set(re.findall(r"(%[A-Za-z_]\w*)\s*=", active))
    missing = set(re.findall(r"%[A-Za-z_]\w*", active)) - definitions
    if missing:
        raise ValueError(f"undefined PTO SSA operands: {sorted(missing)}")
    for operation, count in expected["operations"].items():
        got = len(re.findall(r"\b" + re.escape(operation) + r"\b", active))
        if got != count:
            raise ValueError(f"{operation}: expected {count}, got {got}")


def require_nonempty_file(path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty file: {path}")


def nonempty(path):
    require_nonempty_file(path)
    return path.read_text()


def run_command(args, log, env):
    with log.open("w") as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                check=False)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {args} (see {log})")


def write_checksums(case_dir, name):
    # Hash available evidence even if a later stage or structural check failed.
    with (case_dir / "SHA256SUMS").open("w") as sums:
        for filename in (f"{name}.mlir", f"{name}.pto.mlir"):
            path = case_dir / filename
            if path.is_file() and path.stat().st_size:
                sums.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {filename}\n")


def run_case(case, output, tile, env):
    name = case.name
    dest = output / "cases" / name
    # Clear previous outputs before invoking either CLI stage: a failed run cannot pass
    # by reusing old hop-2 MLIR or PTO files.
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    try:
        expected = expectations(case / "expectations.json")
        source = case / f"{name}.rs"
        nonempty(source)
        kernel = dest / "kernel"
        mlir = dest / f"{name}.mlir"
        pto = dest / f"{name}.pto.mlir"
        run_command([str(tile), str(source), "-t", "msl", "--keep-dir", str(kernel),
                     "-o", str(dest / f"{name}.metal")], dest / "kernel.log", env)
        # Default O2 performs tile -> tile first; compiler MLIR is hop 2.
        hop2 = kernel / f"{name}.2.mlir.mlir"
        nonempty(hop2)
        shutil.copyfile(hop2, mlir)
        run_command([str(tile), str(mlir), "-t", "pto", "-o", str(pto)],
                    dest / "pto.log", env)
        check_outputs(nonempty(mlir), nonempty(pto), expected)
        print(f"PASS {name}: structural MLIR/PTO checks", flush=True)
        return True
    except Exception as error:
        (dest / "ERROR.txt").write_text(traceback.format_exc())
        print(f"FAIL {name}: {error} (details: {dest / 'ERROR.txt'})", file=sys.stderr, flush=True)
        return False
    finally:
        write_checksums(dest, name)


def discover_cases(cases_root):
    cases = sorted(p for p in cases_root.iterdir() if p.is_dir())
    if not cases:
        raise ValueError(f"empty frontend suite: {cases_root}")
    return cases


def run_suite(tile, sdk, output, cases_root=CASES):
    tile, sdk, output, cases_root = (Path(p).resolve() for p in
                                     (tile, sdk, output, cases_root))
    # Cleanup must never touch inputs, even with an accidentally overlapping --output.
    for protected in (cases_root, sdk, tile, Path(__file__).resolve().parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f"output overlaps suite inputs: {output} and {protected}")
    output.mkdir(parents=True, exist_ok=True)
    # Do not upload previous case outputs if a fresh run fails during setup.
    if (output / "cases").exists():
        shutil.rmtree(output / "cases")
    for stale in ("ERROR.txt", "PROVENANCE.txt"):
        (output / stale).unlink(missing_ok=True)
    try:
        head, std = check_sdk(sdk)
        # The real CLI is a native binary, not UTF-8 text like the test double.
        require_nonempty_file(tile)
        if not os.access(tile, os.X_OK):
            raise ValueError(f"tile CLI is not executable: {tile}")
        current = git_head(ROOT)
        (output / "PROVENANCE.txt").write_text(
            f"Current checkout: {current}\n"
            f"Suite sources: {cases_root}\n"
            f"CLI and open PTO emitter: {tile} (caller-supplied; CI builds current checkout)\n"
            f"Release DSL tile_std + tile_std_macros: {sdk} at {head}\n"
            f"Expected published backend: {BACKEND_RELEASE} (CI installs via tile install)\n"
            "Metal only saves compiler MLIR; native PTO requires ASCEND_HOME_PATH on macOS.\n"
            "Structural checks only; no full ABI, shape, or numeric verification.\n"
        )
        cases = discover_cases(cases_root)
        with (output / "PROVENANCE.txt").open("a") as provenance:
            for case in cases:
                source = case / f"{case.name}.rs"
                digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else "missing"
                provenance.write(f"Case {case.name}: {source} (source SHA256: {digest})\n")
    except Exception as error:
        (output / "ERROR.txt").write_text(traceback.format_exc())
        print(f"Suite setup failed: {error} (details: {output / 'ERROR.txt'})", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.pop("RUSTFLAGS", None)
    env["TILE_STD_PATH"] = str(std)
    failures = sum(not run_case(case, output, tile, env) for case in cases)
    print(f"Frontend suite: {len(cases) - failures}/{len(cases)} passed")
    return int(failures != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", required=True, help="built tile CLI executable")
    parser.add_argument("--sdk", required=True, help="pinned release-sdk checkout")
    parser.add_argument("--output", required=True, help="artifact directory (PROBE_DIR in CI)")
    args = parser.parse_args()
    return run_suite(args.tile, args.sdk, args.output)


if __name__ == "__main__":
    sys.exit(main())
