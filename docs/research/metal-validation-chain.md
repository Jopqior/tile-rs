# tile-rs → Metal execution chain and existing verification gaps

Research findings for [Jopqior/tile-rs#4](https://github.com/Jopqior/tile-rs/issues/4)
under map [#2](https://github.com/Jopqior/tile-rs/issues/2).

All claims below are tied to a revision. Nothing here is fixed or changed; this
is an inventory of what exists, what is public, and where the verification
evidence stops. Where a fact could not be established from primary sources it is
marked **Unknown** rather than assumed.

- Repo convention note: there is no `docs/research/` directory before this file
  and no `CONTEXT.md`/`docs/adr/` in the tree, so the research note location is
  chosen here (`docs/research/`).

## 0. Revisions examined

| Thing | Revision | Date | Source |
|---|---|---|---|
| `Jopqior/tile-rs` (fork) `main` | `c3c8ec0169bd02757b51370ba9c6ec3b116b833d` | 2026-09-01 | `git rev-parse HEAD`; fork `pushedAt` 2026-09-06 |
| `yijunyu/tile-rs` (upstream) `main` | `c3c8ec0169bd02757b51370ba9c6ec3b116b833d` | 2026-09-01 | GitHub compare API: `status=identical, ahead_by=0, behind_by=0` |
| Previous upstream commit (last green coverage) | `9b5ae470f4c9ffc9ddb3dc164f5c05bd0c18078a` | 2026-09-01 | `git show` |
| `yijunyu/tile-rs-metal` `main` | `23130eeba6df91c555dbc71135f836eddb92568d` | 2026-08-29 | clone HEAD |
| Release `v0.0.1+nightly-2025-08-04` tag target | `fed207a63cd5f82df32ebbf632ea28850aae4193` | 2026-06-21 | tags API |
| Release `v0.0.2+nightly-2025-08-04` tag target | `c3c8ec0169bd02757b51370ba9c6ec3b116b833d` | 2026-09-06 (asset upload) | tags API |

The fork is **byte-identical to upstream at HEAD** — it has no commits of its
own. "Fork current commit" and "upstream current commit" are the same revision
everywhere below, so a test result for one applies to the other.

## 1. Ownership / repository topology

```
yijunyu/ascend-rs-priv   (PRIVATE, Contents:Read PAT needed)  ── rustc codegen driver source
        │  checked out by
        ▼
yijunyu/tile-rs   (PUBLIC)  ── pure emitters + parser + tile_std/tile_spec/tile_codegen/tile_hal
        │  fork (no commits)
        ├── Jopqior/tile-rs  (PUBLIC, identical HEAD)
        │
        └── publishes Releases ──► tile-rs-codegen-aarch64-apple-darwin.tar.gz
                                     (librustc_codegen_tile.dylib + libtile_std_macros.dylib
                                      + bundled libLLVM/libMLIR/libzstd)

yijunyu/tile-rs-metal (PUBLIC)  ── Metal host runtime glue + examples + benchmarks
                                    (vendors copies of tile_std/tile_hal/tile_kernel_builder;
                                     does NOT depend on tile-rs as a path dependency)
```

Fork metadata: `Jopqior/tile-rs` is `fork: true`, parent/source `yijunyu/tile-rs`
(`gh api repos/Jopqior/tile-rs`). `yijunyu/ascend-rs-priv` is not resolvable
anonymously (404) — no access was attempted; permissions are **Unknown**.

## 2. The actual chain, hop by hop

```
Rust kernel  (tile_std DSL: Tile<ROWS, COLS, T>, 102 pub tile_* intrinsics)
  #[tile_std::tile_kernel]
        │
        ▼
rustc -Zcodegen-backend=<librustc_codegen_tile.dylib>     [PRIVATE source]
        │   MIR → MLIR (LLVM dialect with __tile_* intrinsics)
        ▼
TILERS_CODEGEN_PATH=metal → convert_mlir_to_msl(mlir) -> Result<String,String>
        │   .metal source; PURE TEXT, no LLVM, no xcrun     [PUBLIC emitter]
        ▼
ACLRS_METAL_RUN=1 → driver shells out `xcrun metal` + `xcrun metallib`
        │   .metallib                                        [PRIVATE driver]
        ▼
host launch (metal-rs 0.29): MTLComputePipelineState + MTLBuffer(StorageModeShared)
        │                                                     [PUBLIC example]
        ▼
readback → compare to CPU reference (tile_rsqrt_metal/src/main.rs, max|err| < 1e-4)
```

Per-hop evidence:

| Hop | File / artifact | Availability | Evidence |
|---|---|---|---|
| Rust DSL | `crates/tile_std/src/tile.rs` `pub struct Tile<const ROWS: usize, const COLS: usize, T>` (L44), 102 `pub fn tile_*` | public | `grep -c "pub fn tile_"` = 102 |
| MIR→MLIR driver | `librustc_codegen_tile.dylib` | **private source** | `codegen-release.yml` L22-25: secret `PRIVATE_SOURCE_TOKEN`, scope "Contents: Read on `yijunyu/ascend-rs-priv` (the private codegen source)" |
| emitter | `crates/rustc_codegen_tile/src/mlir_to_msl.rs`, `convert_mlir_to_msl` L431 | public | included by `tile_spec` via `#[path]` with no LLVM (`crates/tile_spec/tests/cucumber.rs`) |
| parser | `crates/rustc_codegen_tile/src/mlir_parse.rs` | public | imported as `crate::mlir_parse` |
| xcrun drive | string `compile_via_metal`, `ACLRS_METAL_RUN`, `xcrun metal`/`metallib` | inside the shipped dylib; source private | `strings librustc_codegen_tile.dylib`; `examples/tile_rsqrt_metal/kernels/src/lib.rs` L6-7 |
| host launch | `examples/tile_rsqrt_metal/src/main.rs` uses the external `metal` crate 0.29 | public (`yijunyu/tile-rs-metal`) | `Cargo.toml` `metal = "0.29"`; `main.rs` L37 |
| readback oracle | 256 f32 elements vs f32 CPU reference `1.0/x.sqrt()` | public | `main.rs` L33-34, L100-116 |

### 2.1 Two things the docs say that the code does not

- `tile-rs-metal` README / `docs/USAGE.md` / `ci.yml` L46 and
  `benchmarks/kernels/manifest.toml` all say the example runs via **wgpu/Metal**.
  The example actually links the `metal` crate directly (`metal = "0.29"`), not
  wgpu. Both statements are in the same public repo.
- `tile-rs-metal` describes itself as providing "the Apple GPU (Metal) runtime
  glue". `crates/tile_hal/src/` in that repo has **no Metal backend**: only
  `cuda.rs` and an `ascend` module (feature `ascend`, file absent →
  `cargo fmt` reports `failed to resolve mod ascend`). `tile_hal::BackendKind`
  maps `"msl"` but there is no MSL device/stream implementation. The working
  Metal launch path is the example's own `main.rs`.

## 3. Source availability and permissions

| Component | Public source? | Binary? | Notes |
|---|---|---|---|
| Pure MLIR→MSL emitter (`mlir_to_msl.rs`) | yes | — | 17,127 lines, 270 `KernelType` variants, 294 in-source tests |
| Shared MLIR parser (`mlir_parse.rs`) | yes | — | 2,023 lines, 28 tests |
| Other 13 emitters | yes | — | `mlir_to_aie/bang/csl/gaudi/gpu/hexagon/linalg/musa/nki/pto/spirv/tpu/ttmetal` |
| rustc codegen driver (MIR→MLIR + xcrun drive) | **no** | shipped as `.dylib` | source in `yijunyu/ascend-rs-priv`, PAT-gated |
| Prebuilt `librustc_codegen_tile.dylib` | — | yes | Releases `v0.0.1`, `v0.0.2`, aarch64-apple-darwin only |
| Metal host runtime | yes | — | external `metal` crate (crates.io) |
| Metal HAL backend in `tile_hal` | **absent** | — | see §2.1 |

Public/private boundary is **not consistent** in the public tree:

- `crates/tile_codegen/src/targets.rs` L61-63 and `emitters.rs` say the
  AscendC/PTO targets are **closed** and live in `crate::ascend`; but
  `crates/rustc_codegen_tile/src/mlir_to_pto.rs` (7,910 lines) is public and is
  `#[path]`-included and tested by `tile_spec`.
- The shipped dylib contains a source path `mlir_to_msl_canned.rs` and guard
  messages (`"has no arm for"`, `"would drop the operation silently"`) that are
  **absent from the public `mlir_to_msl.rs` at c3c8ec0**. So the emitter inside
  the published binary is a *different, newer* revision than the public emitter.
  Reproduce: `strings librustc_codegen_tile.dylib | grep -c "has no arm for"`
  → 4 in the dylib, 0 in the public source.

**Consequence:** "the emitter" is two different objects — public source
(what `tile_spec` tests) and private build (what users actually run).

## 4. Published binaries and their provenance

Releases on `yijunyu/tile-rs` (public):

| Tag | Asset | Size | Uploader | Producing workflow |
|---|---|---|---|---|
| `v0.0.1+nightly-2025-08-04` | `tile-rs-codegen-aarch64-apple-darwin.tar.gz` | 101,724,767 B | `yijunyu` (human) | run 27910459107 (tag `fed207a`) **failed, `release` skipped** |
| `v0.0.2+nightly-2025-08-04` | same name | 103,449,511 B | `yijunyu` (human) | run 34003884887 (`c3c8ec0`) **failed, `release` skipped** |

Both `Codegen Release` workflow runs failed and their `release` job was skipped:

- `aarch64-apple-darwin`: `##[error]a homebrew abspath survived the rpath rewrite`
  (bundle step), plus post-job `fatal: No url found for submodule path
  'benchmarks/Ascend-910B' in .gitmodules` while checking out the private repo.
- `x86_64-unknown-linux-gnu`: `error: could not compile 'tblgen' (lib)` (the
  shared-libLLVM failure the workflow itself warns about at L190-192).

The asset timestamps show a human upload: v0.0.2 asset `created_at`
`2026-09-06T01:26:07Z`, the failing workflow run started `2026-09-06T01:26:14Z`.
Embedded paths in the dylib point at a local scratchpad
(`/private/tmp/claude-502/-Users-yijun-tile-rs/.../scratchpad/be-rebase/...`),
not at a CI checkout path. So the published binary is **not verifiably
CI-produced**; it is a locally built artifact uploaded by hand.

Artifact-interface facts (inspected, not executed):

- ABI pin: `nightly-2025-08-04` = `rustc 1.91.0-nightly (f34ba774c 2025-08-03)`;
  LLVM/MLIR 20 bundled via `@loader_path` (`USAGE.md`).
- v0.0.1 tarball layout: `tile-rs-codegen-aarch64-apple-darwin/{USAGE.md,.cargo/config.toml,lib/*.dylib}`.
- **v0.0.2 tarball layout differs**: top dir is `tile-rs-codegen/` and
  **`.cargo/config.toml` is missing** (only `USAGE.md` + 5 dylibs).
  `install-smoke.yml` L47/L56 asserts that file exists and contains
  `register_tool(tile)`; it passes only because `scripts/install.sh` L19
  defaults to `v0.0.1`.
- `scripts/install.sh` interface: `TILERS_CODEGEN_HOME`, `TILERS_CODEGEN_TAG`,
  hardcoded `REPO="yijunyu/tile-rs"`, `--strip-components=1`, then
  `TILERS_CODEGEN_SO` + `TILERS_CODEGEN_PATH`.

## 5. Test inventory (what is actually checked)

### 5.1 Public tile-rs test surface (all textual — no oracle)

`cargo test` in `crates/tile_spec` at `c3c8ec0`:

- 796 emitter/parser unit tests (`#[test]` per file: msl 294, linalg 77, pto 68,
  tpu 55, gaudi 47, bang 46, nki 44, aie 42, spirv 42, gpu 37, mlir_parse 28,
  csl/hexagon/musa/ttmetal 4 each) + 1 Gherkin driver test.
- 49 Gherkin scenarios across 5 `.feature` files, all green.
- Result without `-D warnings`: **781 passed, 16 failed**.
- The 16 failures are all `mlir_to_msl::tests::test_template_*`, which read
  `crates/deepseek_metal/templates/{inference_kernels.metal,matmul_transposed.metal}`.
  `crates/deepseek_metal/` is a `members` entry in the root `Cargo.toml` but
  **does not exist** in the public tree.

What the MSL tests assert: string containment on emitted source, determinism
(re-emit byte-identical), and a few structural invariants. There is **no test
that compiles MSL, dispatches on a GPU, or compares a number to a reference**.
Grep-confirmed: `mlir_to_msl.rs` has no `xcrun`/`Command::new`; only a comment at
L438.

The Gherkin layer and the coverage gate are **not Metal-correctness checks**:

- `backend_emit_purity.feature`: for each target, output contains one idiom
  (`msl` → `"kernel void"`) and re-emits identically.
- `core_op_lowering.feature`: `msl` appears only for `rms_norm` and `add`, both
  with evidence `"kernel void"`.
- `scripts/coverage.sh --gate 88`: line/region coverage of the open crates; it
  compiles the emitters via `#[path]` and runs their unit tests.
- `install-smoke.yml`: runs `install.sh` on `macos-14`, checks the tarball
  layout + `otool -L` linkage + debranded config. No kernel is executed.
- `toolchain-drift.yml`: `cargo build -p tile_std` on the pinned nightly and
  `cargo check -p tile_std` on the current nightly.

`mlir_to_msl`'s own comments record that pure string checks hid a real bug:
`test_msl_add` (L13634) notes at L13648 that asserting only the expression
"cannot see" a wrong base index, and that "the standalone harness dispatched one
threadgroup, pinning `row` to 0". That test was later strengthened to assert
`uint gid = row * tcount + tid;`.

### 5.2 `tile-rs-metal` test surface

- 30 `#[test]`s total, only in `tile_std_macros` and `tile_hal` (backend
  selection, HAL launch-grid/memory helpers) — nothing that touches MSL output.
- 109 compiletest UI `*.rs` files under `tests/compiletest/ui/`; no harness
  runner is present and CI never compiles them.
- `scripts/mkb_coverage.sh`: a `grep -rqE "fn <name>"` presence check over the UI
  files against `scripts/mkb_kernels.txt` (341 lines). It reports
  "implemented" from a name match; it is not invoked by CI.
  `benchmarks/kernels/manifest.toml` references "311/312" MKB coverage — that is
  a text-presence count.

### 5.3 Existing numerical oracles

| Oracle | Coverage | Tolerance | Runs in CI? |
|---|---|---|---|
| `examples/tile_rsqrt_metal/src/main.rs` | full 256-element vs f32 CPU ref | `max_abs_err < 1e-4` | **no** (`ci.yml` L59 `cargo run ... || true`, plus L51 `continue-on-error`, plus job skipped behind a failing `build`) |
| `benchmarks/apple-mojo-vs-msl/harness/bench_metal_gpu.swift` (413 lines) | 16 kernels vs Float64 CPU ref, but each `checkFn` samples **one output element** (`i=min(100,n-1)`, `p2[5]`, `p3[3]`, `p2[1*N+1]`) | `rel < 1e-3` (matmul `5e-2`) | **no** (needs swift+mojo on a Mac; `run.sh` is manual) |
| `benchmarks/metal/bench_ascendrs_msl.py` | Tier 1 = `xcrun metal` compile-only; Tier 2 device run explicitly marked TODO | n/a | **no** |
| `benchmarks/model_qwen3_1p5b/src/main.rs` | CPU composition of Qwen3-1.5B ops, Ascend `cpp/pto` path, "coherence" self-check | n/a | **no** |

Note: commit `23130ee`'s message claims a "full-output f64 host reference"
check for `attn_gqa` (which found keys past 256 being silently dropped, rel err
6.5e-3 at seq_len 300/512), but the committed Swift harness still only samples
one element per kernel and its fixed shapes are 16 and 256 — the message itself
says "this harness runs 16 and 256, so nothing here crossed the line". The
full-output check is not in the committed harness.

## 6. CI: what is actually guaranteed

### 6.1 Public tile-rs (`Jopqior/tile-rs` == `yijunyu/tile-rs`)

Fork run history (`gh run list`): only 8 runs ever; the only non-PR, HEAD runs
fail.

| Workflow | At `c3c8ec0` | Evidence |
|---|---|---|
| `coverage` | **failure** (both fork and upstream) | run 34035289777 / 33518947624 |
| `toolchain-drift` | **failure** (current-nightly job) | run 34035289773 / 34118426798 |
| `Codegen Release` | **failure** on both matrix legs; `release` skipped | run 34003884887 |
| `install-smoke` | success | run 34003884901 |

Upstream history shows coverage green at `9b5ae47` (33500991547) and failing at
`c3c8ec0` — i.e. the failure was introduced by
"Update from private development repository".

### 6.2 `tile-rs-metal`

Every `CI` run in the repo's history has `build: failure`, `fmt: failure`,
`metal-smoke: skipped` (12/12 runs, `42fde10` → `23130ee`). The Metal smoke step
has **never executed**.

Build failure at `23130ee`:
`error: function 'build' is never used` (`rs_builder.rs:12`, `-D warnings`) and
`error: the feature 'generic_const_exprs' is incomplete` (`tile_std/src/lib.rs:24`).
Independently, the smoke step cannot work as written even if `build` passed:

- `rs_builder.rs` L10 hardcodes `const TARGET = "davinci-huawei-none"` and
  L23-54 resolves it to `--target=davinci-huawei-none` (no
  `davinci-huawei-none.json` exists in that repo; `find . -name "davinci*"` is
  empty). It also L164 requires `TILERS_CODEGEN_SO` (or the dylib on
  `DYLD_FALLBACK_LIBRARY_PATH`), and `ci.yml` never installs the codegen
  tarball. The step is wrapped in `|| true` and the job in
  `continue-on-error: true`, so even a failure would be reported green.

## 7. Failure evidence at HEAD (reproduced locally)

Reproduced in the worktree at `c3c8ec0` with the installed pinned toolchain
(`cargo 1.91.0-nightly`, `nightly-2025-08-04`):

1. `cd crates/tile_spec && RUSTFLAGS="-D warnings" cargo test`
   → `error: could not compile 'tile_spec' (test "cucumber") due to 44 previous errors`.
   Distinct causes include `method 'ptv_type_str' is never used`
   (`mlir_to_pto.rs:416`), `function 'pick_nb' is never used`
   (`mlir_to_pto.rs:3142`), `function 'dump_causal_msl_for_ab' is never used`,
   `variant 'ResidualAdd' is never constructed`, 7 × `unreachable pattern`,
   11 unused `*_MLIR` test constants.
   This matches the CI `RUSTFLAGS: -D warnings` env
   (`actions-rust-lang/setup-rust-toolchain` default).
2. `cd crates/tile_spec && cargo test` (no `-D warnings`)
   → **781 passed; 16 failed**, all 16 for the missing
   `crates/deepseek_metal/templates/*.metal` fixtures.
3. `cd crates/tile_codegen && cargo test --features emitters`
   → `could not compile 'tile_codegen' (lib) due to 18 previous errors`
   (9 × `unresolved import 'crate::mlir_to_pto'`, 1 × `'crate::mlir_to_gpu'`,
   7 × `size for values of type ... cannot be known`).
4. `cargo build -p tile_std` (pinned nightly) → **succeeds** (1 warning).

Root cause of 1-3 is visible in the `c3c8ec0` diff (a single commit):

- `.github/workflows/coverage.yml`: removed the `with: rustflags: ""` override
  (6 lines), re-enabling `-D warnings` for the coverage job.
- `crates/rustc_codegen_tile/src/mlir_to_msl.rs` L14755: `read_template` changed
  from `-> Option<String>` (`.ok()`, callers skip) to `-> String`
  (`.unwrap_or_else(|e| panic!(...))`), so absent fixtures abort.
- `crates/tile_codegen/src/lib.rs`: removed the crate-root `mlir_to_pto` /
  `mlir_to_gpu` declarations (17 lines) that the other emitters import via
  `crate::mlir_to_pto`.
- `rust-toolchain.toml`: removed the "DELIBERATE PIN" comment block (13 lines).

All four removals contradict the repo's own `CHANGELOG.md` at the same revision,
which lists as **Added/Fixed**:
"golden-template tests skip gracefully when the `deepseek_metal` fixture is
absent", "`tile_codegen`'s `emitters` feature now builds and tests standalone",
and "Coverage and toolchain-drift CI no longer force `RUSTFLAGS="-D warnings"`".
The CHANGELOG was not touched by `c3c8ec0`.

`toolchain-drift` current-nightly failure is separate and looks like real drift:
`error[E0405]: cannot find trait 'TrivialClone'`, `E0522 ... unknown lang item
'drop_in_place'`, `E0539 malformed 'rustc_macro_transparency'`,
`E0093 unrecognized intrinsic function: 'expf32'/'logf32'/'fabsf32'/'minnumf32'`.
The `pinned-build` job succeeds, so the pinned nightly still builds `tile_std`.

## 8. MSL surface (bounded inventory — too large to enumerate exhaustively)

`mlir_to_msl.rs` at `c3c8ec0`: 17,127 lines, 200 top-level fns, **270
`KernelType` variants**, 294 tests. Dispatch is by classification of the MLIR
body (`classify_body` L3729) plus a `match name` on `__tile_*` intrinsics; some
kernels are emitted from canned templates, others composed from SSA dataflow.

Bounded groups of the 270 variants:

| Group | Examples |
|---|---|
| Elementwise core | Add/Sub/Mul/Div/Max/Fill/Scale/Copy, ElementwiseChain, Cast f32↔f16↔bf16, Sqrt/Sqr/Neg/Abs/Step/Exp/Log/Sigmoid/Silu/Gelu/HardSigmoid/HardSwish/Softplus/Clamp/Rsqrt |
| Reductions / normalization | ReduceMax, SumRows, LayerNorm, RmsNorm, L2Dist, Absmax, Argmin/ArgMax |
| Linear algebra | MatmulF16, MatmulF16Simdgroup, MatmulTransposed, MatvecChain, GemmChain (partition cells), i8 matvec, GGUF quant matvec/matmul (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS) |
| Attention | Attention, AttentionGqa, AttentionCausal, AttentionDecode*, FlashAttnExt{Pad,Blk,Setup,Score,Out,OutMS}, FlashAttnExtVec*, indexed mixed attention H8/H8Rb4 |
| Positional / cache | Rope, RopeDsv4, RopeInplace, RopeInplaceSplit, RopeSplitBatched, KvCacheUpdate, KvWriteBatched, Dsv4KvFp8Store, Dsv4Fp8KvQuantize, GetRows/SetRows |
| Quant / sampling / sort | Quantize/Dequantize, ArgMax, SampleTopP, DraftVerify, TokenAccept, TopK, TopkMaskScatter, SortI32RowsAsc, Argsort* variants |
| Indexing | Slice, Concat, Scatter, Gather, Embedding, Repeat, Transpose, PartitionCell/PartitionCellStore |
| DS4 / llama.cpp-derived | ~60 `Dsv4*` variants (MoE routing, HC expand/split, unary families per dtype, softmax variants), `BinFuseF32F32F32`, `UnaryF32F32` |

Operand types seen in the emitter and its tests: `f32`, `f16` (`half`), `bf16`,
`i8`, `i32`/`u32`, FP8 E4M3FN (DS4 helpers) and the GGUF quant block formats.
Default codegen strategy is "single workgroup of 256 threads, stride loop", with
a flat `uint gid = row * tcount + tid` index for the pointwise family (the
indexing bug noted in §5.1), and per-kernel grid/threadgroup contracts for the
2D/3D dispatches.

Documented in-emitter shape limits (constants baked into generated MSL):
`MAX_SEQ = 1024` (causal attention), `MAX_SG = 1024`, `MAX_TOPK = 256`,
plus fixed-size scratch arrays `[2048]`/`[4096]`/`[8192]` in specific kernels.
The emitter returns `Err(...)` for a few classes of unrepresentable module
(partition-copy-with-compute, unclaimed cross-lane ops), but its newer shipped
build (not the public source) contains additional "no arm / would silently drop"
guards — see §3.

The public `tile_spec`/registry surface exposes 14 `convert_mlir_to_*` emitters
(`gpu, musa, spirv, msl, nki, aie, bang, gaudi, tpu, csl, hexagon, ttmetal,
linalg, pto`), while `tile_codegen`'s `emitters` feature wires 13 and the
registry adds `debug` (14 names). README's "15 backends" counts 15
`TILERS_CODEGEN_PATH` values (Ascend contributes `cpp` and `pto`). These counts
are internally inconsistent; see §3 for the PTO open/closed contradiction.

## 9. Candidate domain vocabulary (proposal — not an accepted glossary)

No `CONTEXT.md` exists, so these are proposed terms with source anchors, for the
downstream design sessions to accept or reject:

- **emitter** — a pure `convert_mlir_to_<target>(mlir: &str) -> Result<String,
  String>` function (`crates/tile_codegen/src/targets.rs` `EmitterTarget`).
- **codegen driver / private frontend** — the rustc backend that turns MIR into
  MLIR and drives the target toolchain (`librustc_codegen_tile`, private source).
- **shipped backend** — the prebuilt `librustc_codegen_tile.dylib` release asset
  (per-platform, ABI-pinned).
- **codegen-path** — the `TILERS_CODEGEN_PATH` value selecting a target.
- **KernelType classification** — `mlir_to_msl`'s internal classification of an
  MLIR body; the unit of emitter coverage.
- **golden template fixture** — the `deepseek_metal/templates/*.metal` files the
  16 currently-failing tests read; absent publicly.
- **on-HW validation** — README's term for end-to-end runs on real hardware, as
  opposed to the codegen-generality matrix.
- **generality matrix** — the per-backend emitter test set asserted by
  `tile_spec`.
- **artifact contract** — the release tarball layout + `USAGE.md` +
  `.cargo/config.toml` + `TILERS_CODEGEN_SO`.

## 10. Explicit unknowns / do-not-assume

- **Unknown**: whether `yijunyu/ascend-rs-priv` is accessible to this project.
  Anonymous access 404s; no access was attempted.
- **Unknown**: whether the private driver accepts `TILERS_CODEGEN_PATH=metal`
  and/or `msl`. Public code uses `msl` (`tile_hal`, `tile_spec`, registry); the
  Metal example and `USAGE.md` use `metal`; the shipped dylib contains `metal`
  (and `cuda`/`vulkan`/`aie`/`nki`/`pto`), while a standalone `msl` string was
  not observed in `strings`.
- **Unknown**: whether the published `.dylib` can be obtained and run by this
  project's CI (it is public but ~100 MB and its ABI is pinned to one nightly).
- **Unknown**: whether the `mlir_to_msl` variant embedded in the shipped binary
  passes the public tests (they are different revisions, §3).
- **Not established**: any end-to-end Metal number. No CI run of this project
  has ever compiled and dispatched the example; the only full-output numerical
  check exists as source in `tile-rs-metal` and was not executed here (no Metal
  host).
- **Not assumed**: that the Metal path needs private resources. The emitter,
  parser, host example and prebuilt dylib are all public. What is missing is a
  working wiring (target spec, `TILERS_CODEGEN_SO`, fixture set), not
  necessarily permissions.
- **Out of scope here**: runner capability/selection (sibling ticket), fixes to
  any of the above, and the CI design itself.

## 11. Open questions worth a follow-up ticket

1. Which object is the validation target — the public emitter source at a
   revision, or the shipped dylib? They currently differ (evidence in §3).
2. Is `msl` or `metal` the codegen-path contract, and who owns that name?
3. Are the `deepseek_metal/templates`, `tile_metal_py` and the other absent
   concrete `members` — 58 of the root `Cargo.toml` member paths do not exist in
   the tree (plus the `crates/*` glob) — expected to be vendored for validation,
   or is the template-test path to be removed from the public suite?
4. Which reference oracle is authoritative — a per-kernel CPU reference in Rust
   (the `tile_rsqrt_metal` pattern), the Swift Float64 harness, or f64 emulation
   of each emitted kernel — and what tolerance policy per op?
5. Should the v0.0.2 tarball's missing `.cargo/config.toml` be treated as a
   release regression, and does the artifact contract include it?
6. What is the bounded first operator/type/shape set, given 270 emitter variants
   and a mostly textual test surface?

## Sources

Pinned file URLs use the revision SHAs from §0.

- Fork/upstream HEAD: <https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d>
- `mlir_to_msl.rs` L431, L14755: <https://github.com/yijunyu/tile-rs/blob/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/rustc_codegen_tile/src/mlir_to_msl.rs>
- `mlir_to_pto.rs` L416, L3142: <https://github.com/yijunyu/tile-rs/blob/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/rustc_codegen_tile/src/mlir_to_pto.rs>
- `tile_spec` harness + features: <https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/tile_spec>
- `tile_codegen` registry/emitters/targets: <https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/tile_codegen/src>
- `coverage.yml`, `toolchain-drift.yml`, `codegen-release.yml`, `install-smoke.yml`: <https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/.github/workflows>
- `scripts/coverage.sh`, `scripts/install.sh`: <https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/scripts>
- `Cargo.toml` members/exclude, `CHANGELOG.md`: <https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d>
- Previous green commit `9b5ae47`: <https://github.com/yijunyu/tile-rs/commit/9b5ae470f4c9ffc9ddb3dc164f5c05bd0c18078a>
- Release assets: <https://github.com/yijunyu/tile-rs/releases/tag/v0.0.1%2Bnightly-2025-08-04>, <https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04>
- `tile-rs-metal` HEAD: <https://github.com/yijunyu/tile-rs-metal/tree/23130eeba6df91c555dbc71135f836eddb92568d>
- Metal example: <https://github.com/yijunyu/tile-rs-metal/tree/23130eeba6df91c555dbc71135f836eddb92568d/examples/tile_rsqrt_metal>
- `rs_builder.rs`: <https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/crates/tile_kernel_builder/src/kernel_builder/rs_builder.rs>
- Metal CI: <https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/.github/workflows/ci.yml>
- Swift Float64 harness: <https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/benchmarks/apple-mojo-vs-msl/harness/bench_metal_gpu.swift>
- Compile-only notebook: <https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/benchmarks/metal/bench_ascendrs_msl.py>
- Actions histories: fork runs 34035289777 (coverage), 34035289773 (toolchain-drift); upstream 34003884887 (Codegen Release); metal runs 33249745469 / 33194506227 / 33015575938 / 28285691676 / 28142630716.

Local reproduction commands (worktree at `c3c8ec0`, pinned nightly):

```bash
cd crates/tile_spec && RUSTFLAGS="-D warnings" cargo test   # 44 lint errors
cd crates/tile_spec && cargo test                           # 781 passed, 16 failed
cd crates/tile_codegen && cargo test --features emitters     # 18 compile errors
cargo build -p tile_std                                      # ok
```
