# Forward conversion audit evidence

Throwaway experiment assets, not a converter fix. Subject: `28eef7fa72c7e9e3a94e78f5973e1f5ce7dc0f9f`.

- Latest Actions run: https://github.com/Jopqior/tile-rs/actions/runs/36687555598
- Harness commit: `2ede657`
- `run-36687555598.tar.gz` preserves the downloaded Actions artifact, including commands, exit codes, environment, manifests, Cargo.lock files and compiler outputs. A green Actions job means evidence collection finished, not that conversion succeeded.
- Backend: upstream `v0.0.2+nightly-2025-08-04`, macOS ARM64 asset SHA256 `70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891`.
- Alternative libraries: upstream release tag commit `c3c8ec0169bd02757b51370ba9c6ec3b116b833d`, both `tile_std` and adjacent `tile_std_macros`. CLI and kernel input remain at the subject baseline.

## Observations

1. Baseline libraries fail with duplicate `register_tool` attributes. Release-tag libraries remove that failure, but the normal CLI build then fails to locate `ASCEND_HOME_PATH`.
2. Direct Cargo `--emit=llvm-ir` fails in both dev and release profiles because expected `.rcgu.ll` output is absent. No usable MLIR was captured for these invocations.
3. Direct release build with `cdylib + lib` still fails on Ascend settings, after writing intermediate MLIR.
4. Changing only the diagnostic crate's output types from `cdylib + lib` to `rlib`, with release profile retained, succeeds without an Ascend SDK. This is NOT the CLI's current invocation.
5. The resulting backend-specific rlib is a tar archive. Its `.metadata` member is binary Rust metadata, not MLIR. The harness initially tried parsing this member as MLIR; those UTF-8 errors are invalid diagnostic inputs, not a converter finding. Its `.rcgu.o` member is textual LLVM-dialect MLIR. The local extraction below retrieves the correct member.
6. Passing that `.rcgu.o` text through the baseline CLI emits PTO text successfully. `rlib-object.mlir` and `rlib-object.pto.mlir` are the input and output. This is not proof of complete PTO validity or semantic equivalence: the input computes a `get_block_idx() * 1024` pointer offset, while the output uses the base pointer and leaves block-index handling as a comment.
7. Separately, the unoptimized intermediate MLIR from the normal debug build emits PTO text with undefined SSA operands: calls defining pointer values are rendered as `// unhandled`. Exit 0 does not establish valid PTO.

## Reproduce the local extraction and conversion

From an extracted Actions artifact:

```python
from pathlib import Path
import tarfile
p = Path('remote-run4/release-rlib/target/aarch64-apple-darwin/release/libtile_kernel.rlib')
with tarfile.open(p) as t:
    objects = [m for m in t.getmembers() if m.isfile() and m.name.endswith('.o')]
    assert len(objects) == 1
    Path('rlib-object.mlir').write_bytes(t.extractfile(objects[0]).read())
```

Build the CLI from the subject baseline, then:

```sh
tile rlib-object.mlir -f mlir -t pto -O0 --cross ascend \
  --offline --no-install -o rlib-object.pto.mlir
```

The preserved local stdout/stderr refer to `/tmp/tile-audit64/`. No original source or compiler code was modified. No Ascend directory was fabricated, SDK installed, or hardware run performed.
