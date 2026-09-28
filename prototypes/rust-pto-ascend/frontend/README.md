# Rust → PTO frontend tests

## Architecture

```text
frontend/
├── README.md              # Architecture, usage, and adding cases
├── run.py                 # Case discovery, compilation, checks, and artifacts
├── cases/                 # Real Rust kernels and per-case expectations
└── tests/                 # Offline tests for the framework itself
    ├── test_checker.py    # Structural checker tests with output mutations
    ├── test_runner.py     # CLI orchestration, isolation, and failure tests
    ├── test_real_cases.py # Real-case configuration and source checks
    ├── support.py         # Shared fixture-loading and test helpers
    ├── fake_tile.py       # Fake CLI supplying recorded stage outputs
    └── fixtures/          # Synthetic inputs, outputs, and expectations
```

```text
Workflow: prepare toolchains → run.py → upload artifacts
                                │
                                └─ discover cases/<name>/
                                     Rust → compiler MLIR → PTO → structural checks
```

- `run.py` runs each case through the CLI twice: `-t msl --keep-dir` saves compiler MLIR, then `-t pto` converts that MLIR using the current-source PTO emitter.
- Rust → MLIR uses the published macOS ARM backend and its matching release `tile_std` / `tile_std_macros`, **not the current checkout's DSL**. The Metal route avoids requiring an Ascend SDK on macOS; it does not run a Metal kernel.
- Results go to `<output>/cases/<name>/`: intermediate files, logs, MLIR, PTO, checksums, and error details. The runner continues after case failures and exits nonzero if any case fails. `PROVENANCE.txt` records the sources and toolchain context.

## How testing works

| Layer | Inputs | What it checks |
| --- | --- | --- |
| Frontend tests (`run.py`) | Real Rust kernels in `cases/` | Compilation and emitted MLIR/PTO structure against each case's `expectations.json` |
| Offline framework tests (`tests/`) | Synthetic samples in `tests/fixtures/` | Checker behavior, two-stage CLI orchestration, failure handling, and output isolation |
| Case contract tests (`tests/test_real_cases.py`) | Automatically discovered real cases | Valid configuration and nonempty source files |

Structural checks cover the required MLIR intrinsic, PTO function and pointer parameters, tensor views using those parameters, undefined SSA references, and operation counts. They are **not a full MLIR/PTO verifier and do not prove numeric correctness**. Execution validation belongs to the downstream simulator/device workflow.

Fixtures are independently authored inputs and expected outputs for a fake CLI. They are not executable kernel tests and do not need to be added when adding a real case.

## Running tests

Run commands from the repository root. Offline tests need only Python's standard library:

```sh
python3 -m unittest discover -s prototypes/rust-pto-ascend/frontend/tests -v
```

For real compilation, first prepare the CLI, published backend, pinned nightly, and release SDK checkout as in [the frontend workflow](../../../.github/workflows/prototype-rust-pto-frontend.yml), then run:

```sh
python3 prototypes/rust-pto-ascend/frontend/run.py \
  --tile crates/tile_cli/target/release/tile \
  --sdk release-sdk \
  --output /tmp/rust-pto-frontend
```

Use a dedicated output directory outside the sources and SDK: each run removes previous case outputs. The runner checks the SDK revision and cleanliness; the workflow is responsible for building the current CLI and installing the checksum-verified backend.

## Adding a test case

1. Create `cases/<name>/<name>.rs` with a kernel compatible with the release DSL/backend.
2. Add `cases/<name>/expectations.json`. For example, the existing add case uses:

   ```json
   {
     "function": "add_generated",
     "pointer_type": "f32",
     "pointer_count": 3,
     "mlir_intrinsic": "__tile_add_f32",
     "operations": {
       "pto.tload": 2,
       "pto.tadd": 1,
       "pto.tstore": 1
     }
   }
   ```

   Set these values for the new kernel. `function` is the emitted PTO function name and need not match the directory name. `pointer_count` must be positive; operation counts may be zero to forbid an operation.
3. Run the offline tests to check the case layout/configuration, then run the real suite to verify compilation and output.

The runner discovers new cases automatically: **no YAML, framework-test, or fixture changes are needed** for cases covered by the existing checks. Adding simulator/device coverage is separate; that workflow currently consumes only `add_generated`.

For release compatibility details and historical results, see [the frontend probe notes](../../../docs/research/rust-pto-frontend-probe.md).
