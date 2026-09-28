# Rust → PTO execution tests

See the [suite overview](../README.md) for the shared case layout, adding cases, CI triggers, and offline test commands.

## Compilation and execution

The device runner verifies input artifacts, compiles each case's kernel and host, and invokes its numeric checker. All cases currently use the same 910B vector compilation target and CANN host link flags; other architectures require runner changes. The runner does not interpret tensor shapes or operation names.

```text
device/
├── install-camodel.sh  # Pinned CANN Toolkit, no driver/firmware
├── install-tools.sh    # Pinned PTO-ISA headers and PTOAS (sim only)
├── run.sh              # Sources CANN's environment before starting run.py
├── run.py              # Verifies, compiles, runs and reports
└── tests/              # Offline runner, runtime-selection and checker tests
```

Tool installation happens once per job, not once per case. Simulation converts frontend PTO to C++ through PTOAS with automatic synchronization insertion. NPU execution verifies and recompiles the simulator-generated C++; it neither regenerates PTO nor reuses simulator binaries.

## Case contract

A shared case opts into execution by providing these files in its `device/`:

- **`kernel.cpp`**: includes `generated.cpp` and calls the emitted function through an appropriate device launch wrapper.
- **`host.cpp`**: receives `INPUT_PATH OUTPUT_PATH` from the checker, uses `ASCEND_DEVICE_ID` and `EXECUTION_MODE`, and must fail on ACL errors or actual loaded-runtime mismatches. The framework links it against `libkernel.so`.
- **`check.py`**: accepts `BINARY WORK_DIRECTORY`, generates deterministic inputs, invokes the host, and compares outputs against an independent reference. It must exit nonzero on execution errors, missing output, or numeric mismatches. Keep raw runtime logs in the work directory, not stdout/stderr.

These are handwritten per-case files. See [`add_generated/device/`](../cases/add_generated/device/) for an example. Its checker also supports `--inject-error` for an offline negative probe; this is not a special runner mode.

## Running on the simulator

Prepare CANN and the pinned tools. Run the full Toolkit installer **only on a disposable Ubuntu 22.04 runner/container**, with at least 20 GiB free. No NPU device should be exposed to the simulator container. The pinned PTOAS wheel requires x86-64 Linux and Python 3.10. See [the workflow](../../../.github/workflows/prototype-rust-pto-ascend.yml) for OS packages.

From the repository root:

```sh
export BUILD_DIR=/tmp/rust-pto-sim
export INSTALL_ROOT=/tmp/ascend-prototype
export CANN_ENV="$INSTALL_ROOT/cann/set_env.sh"
export ASCEND_DEVICE_ID=0
bash prototypes/rust-pto-ascend/device/install-camodel.sh
bash prototypes/rust-pto-ascend/device/install-tools.sh sim /tmp/rust-pto-tools
bash prototypes/rust-pto-ascend/device/run.sh \
  --mode sim --input /path/to/frontend-artifacts \
  --output "$BUILD_DIR" --tools /tmp/rust-pto-tools
```

[Frontend artifacts](../frontend/README.md#outputs) must contain `cases/<name>/<name>.pto.mlir`, the matching `.mlir`, and `SHA256SUMS`. Both required files are checked before PTOAS runs.

## Running on an NPU

Use a provisioned, authorized runner with a driver and CANN. Set `CANN_ENV`, `ASCEND_DEVICE_ID`, and the actual `ASCEND_SOC` (e.g. `Ascend910B1`). Prepare tools with `install-tools.sh npu /absolute/tools-dir`, then invoke `run.sh` with `--mode npu --input /path/to/simulator-artifacts`, a fresh `--output`, and `--tools /absolute/tools-dir`. The input is the simulator's public `artifacts/` directory, or the downloaded artifact contents.

The manual GitHub Actions NPU job needs repository variables `ASCEND_CANN_ENV`, `ASCEND_DEVICE_ID`, `ASCEND_SOC`, and runner labels `self-hosted`, `linux`, `ascend-910b`. Its concurrency group serializes device jobs without cancelling a running job.

## Results and privacy

Before a run, the runner removes its previous `work/` and `artifacts/` subdirectories under `--output`.

- `work/<name>/`: binaries, intermediate build files, private runtime logs.
- `artifacts/inventory.txt`: execution mode, target, compiler and PTO tool identity.
- `artifacts/summary.json`: per-case pass/fail status after execution.
- `artifacts/cases/<name>/`: generated C++, checksums, input PTO (sim), compilation logs, checker result, optional `inputs.txt` / `outputs.txt`, and failure details.
- `artifacts/ERROR.txt`: suite setup failures; `device.txt` records NPU inventory.

Only `artifacts/` is uploaded. Case work directories, CANN installers/SDKs, and private installation/runtime logs are not uploaded. The runner copies only `inputs.txt` and `outputs.txt` from checker work directories. Artifacts are retained for 14 days.
