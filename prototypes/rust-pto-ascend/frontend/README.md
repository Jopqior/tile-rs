# Rust → PTO frontend tests

See the [suite overview](../README.md) for the shared case layout, adding cases, CI triggers, and offline test commands.

## Compilation and checks

`run.py` invokes the CLI twice per case:

1. `-t msl --keep-dir` saves compiler MLIR from the real Rust kernel.
2. `-t pto` converts that MLIR using the current-source PTO emitter.

Rust → MLIR uses the published macOS ARM backend and its matching release `tile_std` / `tile_std_macros`, **not the current checkout's DSL**. The Metal route avoids requiring an Ascend SDK on macOS; it does not run a Metal kernel.

Structural checks cover the required MLIR intrinsic, PTO function and pointer parameters, tensor views using those parameters, undefined SSA references, and operation counts. They are **not a full MLIR/PTO verifier**. Numerical execution is covered by the [device suite](../device/README.md).

## Case contract

Each shared `cases/<name>/` directory must contain a nonempty `<name>.rs` kernel compatible with the release DSL/backend and an `expectations.json`. For example:

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

`function` is the emitted PTO function name and need not match the directory name. `pointer_count` must be positive; operation counts may be zero to forbid an operation. An optional case `device/` directory does not affect frontend checks.

## Running real compilation

From the repository root, first prepare the current CLI, published backend, pinned nightly, and release SDK checkout as in [the frontend workflow](../../../.github/workflows/prototype-rust-pto-frontend.yml), then run:

```sh
python3 prototypes/rust-pto-ascend/frontend/run.py \
  --tile crates/tile_cli/target/release/tile \
  --sdk release-sdk \
  --output /tmp/rust-pto-frontend
```

The runner checks the SDK revision and cleanliness. The workflow builds the current CLI and installs the checksum-verified backend.

## Outputs

Each run clears the previous `<output>/cases/` directory. Results are written to `<output>/cases/<name>/`: intermediate files, compilation logs, `<name>.mlir`, `<name>.pto.mlir`, `SHA256SUMS`, and failure details. The device suite consumes the MLIR/PTO files and checksums from this directory.

`PROVENANCE.txt` records sources and toolchain context; a suite setup failure is saved as `<output>/ERROR.txt`. The workflow uploads the output directory.

## Offline test organization

```text
frontend/
├── run.py
└── tests/
    ├── test_checker.py    # Structural checks with output mutations
    ├── test_runner.py     # Two-stage CLI orchestration and failure handling
    ├── test_real_cases.py # Configuration and source contracts
    ├── support.py         # Fixture-loading helpers
    ├── fake_tile.py       # Fake CLI supplying stage outputs
    └── fixtures/          # Synthetic sources, expectations, MLIR and PTO
```

For release compatibility details and historical results, see [the frontend probe notes](../../../docs/research/rust-pto-frontend-probe.md).
