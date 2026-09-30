#!/usr/bin/env bash
# Diagnostic direct-Cargo probe, NOT a change to tile-cli's execution path.
set -euo pipefail
evidence="$1"
tile="$2"
export TILERS_CODEGEN_SO="$(find "$TILE_HOME" -name librustc_codegen_tile.dylib -print -quit)"
export TILERS_CODEGEN_PATH=pto
original=""
for f in "$evidence"/backend-capture/*/Cargo.toml; do
  if grep -q release-libs "$f"; then original="$(dirname "$f")"; break; fi
done
if [ -z "$original" ]; then echo 'No captured release-library crate'; exit 1; fi
cp "$TILE_HOME"/toolchains/rustc_codegen_tile/*/USAGE.md "$evidence/backend-USAGE.md"
run_probe() {
  label="$1"; shift
  work="$RUNNER_TEMP/direct-$label"
  dst="$evidence/direct-$label"
  mkdir -p "$work" "$dst"
  cp "$original/Cargo.toml" "$original/Cargo.lock" "$work/"
  cp -R "$original/src" "$original/.cargo" "$work/"
  (
    cd "$work"
    printf '%q ' cargo +nightly-2025-08-04 rustc --locked --target aarch64-apple-darwin --lib "$@" -- --emit=llvm-ir > "$dst/command.txt"
    set +e
    cargo +nightly-2025-08-04 rustc --locked --target aarch64-apple-darwin --lib "$@" -- --emit=llvm-ir > "$dst/stdout" 2> "$dst/stderr"
    rc=$?
    set -e
    echo "$rc" > "$dst/exit"
    echo "$label exit=$rc"
    cp Cargo.lock "$dst/"
    find target -type f \( -name '*.mlir' -o -name '*.ll' -o -name '*.tile.*' \) -print > "$dst/artifact-paths.txt"
    while IFS= read -r f; do
      mkdir -p "$dst/$(dirname "$f")"
      cp "$f" "$dst/$f"
    done < "$dst/artifact-paths.txt"
    input="$(find target -name 'libtile_kernel*.mlir' -print -quit)"
    if [ -n "$input" ]; then
      set +e
      "$tile" "$input" -f mlir -t pto -O0 --cross ascend --offline --no-install -o "$dst/cli-output.pto.mlir" > "$dst/cli.stdout" 2> "$dst/cli.stderr"
      echo "$?" > "$dst/cli.exit"
      set -e
    fi
  )
}
run_probe emit-dev
run_probe emit-release --release
