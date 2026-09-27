# Released Rust backend → PTO frontend probe (macOS ARM)

Status: **blocked before compiler-produced MLIR**. Do not treat simulator/NPU jobs that depend on `add_generated.pto.mlir` as validated. No MLIR/PTO fixtures were substituted. Source is `prototypes/rust-pto-ascend/frontend/add_generated.rs`: `add_generated(a, b, out)`, three rank-2 `1 × 256` f32 views, load/load/add/store. The workflow uses stable to build the *current* CLI, installs published `rustc_codegen_tile` v0.0.2+nightly-2025-08-04 through `tile install rustc_codegen_tile` (manifest SHA-256 `70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891`), and compiles the kernel with exactly `nightly-2025-08-04` + rustc-dev + llvm-tools + rust-src on `macos-14` ARM. It requires a nonempty compiler-produced intermediate before invoking the current open PTO emitter; otherwise it exits nonzero.

| Run | Attempt | Actual result |
| --- | --- | --- |
| [36335715988](https://github.com/Jopqior/tile-rs/actions/runs/36335715988) | Initial workflow | Workflow validation failure before jobs: `runner.temp` unavailable in job-level `env`. Moved setup to a step. |
| [36335776434](https://github.com/Jopqior/tile-rs/actions/runs/36335776434) | Pristine checkout, shipped `softmax.rs`, `-t msl` | `tile_std` fails with duplicate `register_tool`/`feature(register_tool)` (E0636): current `lower_rs.rs` injects these attributes into each target crate and current `tile_std/src/lib.rs` already defines them. No kernel MLIR. |
| [36335944644](https://github.com/Jopqior/tile-rs/actions/runs/36335944644), [36336155872](https://github.com/Jopqior/tile-rs/actions/runs/36336155872) | Scratch attribute fix, shipped sample and minimal `add_generated.rs`, `-t msl` | Both reach the released backend, which refuses `mlir_to_msl has no arm for __tile_window_mask_f32`; the add kernel never calls this intrinsic. No kernel MLIR. Full-workflow run 36335944644 skips the dependent simulator/NPU jobs, correctly. |
| [36336323316](https://github.com/Jopqior/tile-rs/actions/runs/36336323316) | Same released backend and add kernel, `-t pto` | `Could not determine ASCEND_HOME_PATH` on macOS. No kernel MLIR. |

The prototype avoids changing checkout sources/private backend binaries. Only a **throwaway copy** of `tile_std` plus its sibling macros is used for compilation, with this exact difference in `tile_std/src/lib.rs` (also uploaded as `tile_std-attr-dedup.diff` with logs in runs 36336155872 and 36336323316):

```diff
@@ -22,7 +22,6 @@
     generic_const_exprs,
-    register_tool
 )]
@@ -34,7 +33,6 @@
 #![rustc_coherence_is_core]
-#![register_tool(tile)]
```

`-t msl` is the future-ready frontend path because `metal` does not require the CANN SDK and its kept `*.1.mlir.mlir` is genuine compiler output; the second CLI call to `-t pto` on that intermediate is the **current open emitter**, not the published backend's own optional `target_source`. This design was not end-to-end validated: first the CLI/core attribute duplication needs a proper fix, then a compatible public backend release must accept the minimal kernel and produce MLIR. The released binary cannot be repaired by editing this repository's `mlir_to_msl.rs`. PTO compiler path on macOS is SDK-gated instead. Until those blockers are removed, artifacts contain failure logs and the exact temporary diff, **not** successful MLIR or PTO output.
