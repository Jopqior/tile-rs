#!/usr/bin/env python3
"""Compile and numerically check shared Rust/PTO cases on camodel or a 910B NPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

CASES = Path(__file__).resolve().parents[1] / "cases"
# Only these checker outputs are public. Runtime diagnostics remain in work/.
CHECKER_ARTIFACTS = ("inputs.txt", "outputs.txt")


def nonempty(path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty file: {path}")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(directory, manifest, names):
    """Verify required files without following paths supplied by a manifest."""
    entries = {}
    for line in (directory / manifest).read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if name in entries:
            raise ValueError(f"duplicate checksum for {name}")
        entries[name] = checksum
    for name in names:
        if entries.get(name) != digest(nonempty(directory / name)):
            raise ValueError(f"checksum missing or mismatched: {directory / name}")


def command(args, log, env, *, cwd=None):
    with log.open("w") as stream:
        result = subprocess.run(
            [str(arg) for arg in args], stdout=stream, stderr=subprocess.STDOUT,
            env=env, cwd=cwd, timeout=600, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}; see {log}")


def discover_cases(root):
    cases = sorted(p for p in root.iterdir() if p.is_dir() and (p / "device").is_dir())
    if not cases:
        raise ValueError(f"no device cases in {root}")
    return cases


def runtime_environment(mode):
    env = os.environ.copy()
    home = Path(env["ASCEND_HOME_PATH"])
    device = env.get("ASCEND_DEVICE_ID", "")
    if not device.isdecimal() or not 0 <= int(device) <= 65535:
        raise ValueError("ASCEND_DEVICE_ID must be an allocated logical device ID (0..65535)")
    env["EXECUTION_MODE"] = mode
    if mode == "sim":
        if os.uname().machine != "x86_64" or device != "0":
            raise ValueError("simulation requires x86_64 and ASCEND_DEVICE_ID=0")
        if any(Path("/dev").glob("davinci*")) or any(
            Path(p).exists() for p in ("/dev/devmm_svm", "/dev/hisi_hdc")
        ):
            raise ValueError("simulation must run without exposed NPU devices")
        runtime = home / "tools/simulator/Ascend910B1/lib"
        nonempty(runtime / "libruntime_camodel.so")
        env["LD_LIBRARY_PATH"] = f"{runtime}:{env.get('LD_LIBRARY_PATH', '')}"
    else:
        if not env.get("ASCEND_SOC", "").startswith("Ascend910B"):
            raise ValueError("ASCEND_SOC must identify the allocated Ascend910B SoC")
        if "simulator" in env.get("LD_LIBRARY_PATH", "") or "camodel" in env.get("LD_PRELOAD", ""):
            raise ValueError("NPU execution must not load the simulator runtime")
    return env


def run_case(case, source, output, tools, mode, env):
    name = case.name
    work = output / "work" / name
    artifacts = output / "artifacts/cases" / name
    work.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    try:
        device = case / "device"
        for filename in ("kernel.cpp", "host.cpp", "check.py"):
            nonempty(device / filename)
        incoming = source / "cases" / name
        generated = work / "generated.cpp"
        if mode == "sim":
            pto_name = f"{name}.pto.mlir"
            verify_files(incoming, "SHA256SUMS", [f"{name}.mlir", pto_name])
            pto = artifacts / "input.pto"
            shutil.copyfile(incoming / pto_name, pto)
            command([tools / "venv/bin/ptoas", pto, "--pto-arch=a3", "--enable-insert-sync",
                     "-o", generated], artifacts / "ptoas.log", env)
        else:
            verify_files(incoming, "generated.sha256", ["generated.cpp"])
            shutil.copyfile(incoming / "generated.cpp", generated)
        nonempty(generated)
        shutil.copyfile(generated, artifacts / "generated.cpp")
        (artifacts / "generated.sha256").write_text(f"{digest(generated)}  generated.cpp\n")

        home = Path(env["ASCEND_HOME_PATH"])
        command([
            "bisheng", "-xcce", "-O2", "-std=c++17", "--cce-aicore-arch=dav-c220-vec",
            "-DMEMORY_BASE", "-fPIC", "-shared", "--cce-fatobj-link",
            f"-I{tools / 'pto-isa/include'}", f"-I{work}", f"-I{home / 'include'}",
            device / "kernel.cpp", "-o", work / "libkernel.so",
        ], artifacts / "compile.log", env, cwd=work)
        runtime = home / "tools/simulator/Ascend910B1/lib" if mode == "sim" else home / "lib64"
        runtime_lib = "runtime_camodel" if mode == "sim" else "runtime"
        command([
            "g++", "-std=c++17", device / "host.cpp", f"-I{home / 'include'}",
            f"-L{work}", "-lkernel", f"-L{runtime}", "-Wl,--no-as-needed",
            f"-l{runtime_lib}", f"-L{home / 'lib64'}", "-lascendcl", "-ldl",
            "-Wl,-rpath,$ORIGIN", "-o", work / "host",
        ], artifacts / "link.log", env, cwd=work)
        command([sys.executable, device / "check.py", work / "host", work],
                artifacts / "result.txt", env, cwd=work)
        print(f"PASS {name}: {mode} numeric check", flush=True)
        return True
    except Exception as error:
        (artifacts / "ERROR.txt").write_text(traceback.format_exc())
        print(f"FAIL {name}: {error}", file=sys.stderr, flush=True)
        return False
    finally:
        for filename in CHECKER_ARTIFACTS:
            path = work / filename
            if path.is_file():
                shutil.copyfile(path, artifacts / filename)


def run_suite(mode, source, output, tools, cases_root=CASES):
    source, output, tools, cases_root = (Path(p).resolve() for p in
                                       (source, output, tools, cases_root))
    # Reject overlaps before removing anything; work and artifacts are runner-owned.
    for protected in (source, tools, cases_root, Path(__file__).resolve().parents[1]):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f"output overlaps suite inputs: {output} and {protected}")
    output.mkdir(parents=True, exist_ok=True)
    for directory in (output / "work", output / "artifacts"):
        if directory.is_symlink():
            raise ValueError(f"refusing symlink output directory: {directory}")
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
    artifacts = output / "artifacts"
    try:
        cases = discover_cases(cases_root)
        env = runtime_environment(mode)
        command(["bisheng", "--version"], artifacts / "inventory.txt", env)
        with (artifacts / "inventory.txt").open("a") as inventory:
            compiler = Path(shutil.which("bisheng", path=env.get("PATH")))
            inventory.write(f"\nMODE={mode}\nDEVICE_ID={env['ASCEND_DEVICE_ID']}\n"
                            f"SOC={env.get('ASCEND_SOC') if mode == 'npu' else 'Ascend910B1'}\n"
                            f"CCE_ARCH=dav-c220-vec\nBISHENG_SHA256={digest(compiler)}\n")
            inventory.write((tools / "versions.txt").read_text())
        if mode == "npu":
            command(["npu-smi", "info"], artifacts / "device.txt", env)
    except Exception as error:
        (artifacts / "ERROR.txt").write_text(traceback.format_exc())
        print(f"Suite setup failed: {error}", file=sys.stderr)
        return 1
    results = {case.name: run_case(case, source, output, tools, mode, env) for case in cases}
    (artifacts / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    passed = sum(results.values())
    print(f"Device suite ({mode}): {passed}/{len(results)} passed", flush=True)
    return int(passed != len(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("sim", "npu"))
    parser.add_argument("--input", required=True, help="frontend artifacts (sim) or simulator artifacts (npu)")
    parser.add_argument("--output", required=True, help="isolated directory for work/ and public artifacts/")
    parser.add_argument("--tools", required=True, help="directory prepared by install-tools.sh")
    args = parser.parse_args()
    try:
        return run_suite(args.mode, args.input, args.output, args.tools)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
