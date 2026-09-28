# Rust → PTO → Ascend tests

This suite checks real Rust kernels through frontend compilation and optional execution on CANN camodel or an Ascend 910B NPU:

```text
Rust → compiler MLIR → PTO → generated C++ → simulator → optional NPU
└────── frontend ────────┘   └──────────── device ───────────────────┘
```

The frontend checks emitted structure, not numeric correctness. The device suite compiles the frontend's PTO and compares execution results against an independent reference; there is no handwritten-IR fallback. The initial real case is `add_generated` (256 exact f32 additions).

## Layout

```text
rust-pto-ascend/
├── README.md
├── cases/<name>/
│   ├── <name>.rs            # Rust kernel
│   ├── expectations.json   # Frontend structural expectations
│   └── device/             # Optional: opts this case into execution tests
│       ├── kernel.cpp      # Handwritten device launch wrapper
│       ├── host.cpp        # Handwritten ACL host
│       └── check.py        # Inputs and independent numeric reference
├── frontend/               # Rust → MLIR → PTO runner and offline tests
├── device/                 # Tool installation, execution runner and offline tests
└── evidence/               # Recorded results from the original add prototype
```

Both runners discover cases in name order. The frontend checks every case; the device runner checks only those with a `device/` directory. They continue after individual case failures and exit nonzero if any case fails. An empty suite fails rather than reporting success.

Workflows prepare dependencies, call runners, and upload results even on failure. Use dedicated output directories outside sources, input artifacts and toolchains: the runners remove their previous case outputs before execution to prevent stale results from passing a failed run. Output layouts are documented per runner.

## Adding a case

1. Create `cases/<name>/<name>.rs` and `expectations.json` following the [frontend contract](frontend/README.md#case-contract).
2. To test execution, also provide `device/kernel.cpp`, `host.cpp`, and `check.py` following the [device contract](device/README.md#case-contract). Each case owns its host, launch wrapper, and numeric reference; only `generated.cpp` is automatically produced by PTOAS.
3. Run offline tests, then the real frontend and, where applicable, device suite.

For cases within the supported checks and compilation settings, no runner or YAML edits are needed. Offline fixtures are synthetic framework-test inputs, not real kernel cases; adding a real case does not require duplicating it as a fixture. Add offline tests for any new case-specific checker logic.

## Offline checks

Run commands from the repository root. Python tests use only the standard library and require neither compiler SDKs nor hardware; ShellCheck is a separate tool.

```sh
python3 -m unittest discover -s prototypes/rust-pto-ascend/frontend/tests -v
python3 -m unittest discover -s prototypes/rust-pto-ascend/device/tests -v
shellcheck prototypes/rust-pto-ascend/device/*.sh
```

These tests cover runner behavior, failure handling, output isolation, checker logic and real-case file contracts. Fake-tool tests do not establish actual compiler compatibility or device correctness.

## CI and real execution

[The Ascend workflow](../../.github/workflows/prototype-rust-pto-ascend.yml):

- Related PRs run offline frontend/device tests and ShellCheck only.
- Matching pushes on `prototype/61-rust-pto-ascend-ci` and manual runs execute frontend → simulator after offline tests pass.
- Only a manual run with `run_npu` enabled runs the NPU job, after the simulator passes. It recompiles the same generated C++ sources on the self-hosted runner.

[The frontend workflow](../../.github/workflows/prototype-rust-pto-frontend.yml) can also run independently or be called by the Ascend workflow.

For prerequisites, commands, and artifacts, see:

- [Frontend compilation](frontend/README.md): published macOS ARM backend, release-matched DSL, and structural MLIR/PTO checks.
- [Device execution](device/README.md): CANN/PTO tools, simulator and NPU setup, per-case C++/Python interfaces, and public versus private artifacts.
