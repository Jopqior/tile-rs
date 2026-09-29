#!/usr/bin/env bash
# Temporary two-arm Release DSL compatibility experiment. No production source edits.
set -euo pipefail
BASE_SHA=2924af350f8843738f276b9a8695b0416a2d3d49
PUBLIC_TAG_SHA=c3c8ec0169bd02757b51370ba9c6ec3b116b833d
TAG='v0.0.2+nightly-2025-08-04'
EXPECTED_SHA=70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891
EXPERIMENT=experiment/rust-kernel-e2e-release-dsl-v002-20260929
OUT="$RUNNER_TEMP/rust-kernel-e2e"
EV="$OUT/evidence"
WORK="$OUT/work"
BASE="$WORK/baseline"
TAG_SOURCE="$WORK/public-tag"
mkdir -p "$EV/logs" "$EV/inputs" "$EV/mlir" "$EV/msl" "$WORK"
trap 'rc=$?; printf "%s\n" "$rc" > "$EV/exit-code.txt"; echo "experiment exit=$rc (evidence: $EV)"' EXIT
record() { printf '%s\t%s\n' "$1" "$2" | tee -a "$EV/stages.tsv"; }
run() {
  local name="$1" rc
  shift
  printf 'cwd=%q ' "$PWD" >> "$EV/commands.txt"
  printf '%q ' "$@" >> "$EV/commands.txt"
  printf '\n' >> "$EV/commands.txt"
  set +e
  "$@" > "$EV/logs/$name.stdout" 2> "$EV/logs/$name.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$EV/logs/$name.exit"
  echo "$name exit=$rc"
  tail -n 35 "$EV/logs/$name.stderr" || true
}
required() {
  local name="$1"
  shift
  run "$name" "$@"
  if [ "$(< "$EV/logs/$name.exit")" != 0 ]; then
    record "$name" setup_failed
    exit 2
  fi
}
printf 'baseline=%s\npublic_tag=%s\npublic_tag_sha=%s\nrelease_digest=%s\nbranch=%s\nrun=%s attempt=%s\nprofile=release\n' \
  "$BASE_SHA" "$TAG" "$PUBLIC_TAG_SHA" "$EXPECTED_SHA" "$EXPERIMENT" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" > "$EV/identity.txt"
if [ "$GITHUB_REF_NAME" != "$EXPERIMENT" ]; then record branch wrong_branch; exit 2; fi
if [ "$(uname -s -m)" != 'Darwin arm64' ]; then record platform wrong_arch; exit 2; fi
{
  date -u; sw_vers; uname -a; echo "ImageOS=${ImageOS:-} RUNNER_IMAGE_VERSION=${RUNNER_IMAGE_VERSION:-} RUNNER_ARCH=${RUNNER_ARCH:-}"
  git rev-parse HEAD; xcodebuild -version; rustup --version
} > "$EV/environment.txt" 2>&1
required baseline git worktree add --detach "$BASE" "$BASE_SHA"
required public_tag git worktree add --detach "$TAG_SOURCE" "$PUBLIC_TAG_SHA"
if [ "$(git -C "$BASE" rev-parse HEAD)" != "$BASE_SHA" ] ||
   [ "$(git -C "$TAG_SOURCE" rev-parse HEAD)" != "$PUBLIC_TAG_SHA" ]; then record source wrong_commit; exit 2; fi
# Verify the upstream public tag ref independently when it exists in the fork clone.
if git show-ref --verify --quiet "refs/tags/$TAG"; then
  [ "$(git rev-list -n 1 "refs/tags/$TAG")" = "$PUBLIC_TAG_SHA" ] || exit 2
fi
{
  git -C "$BASE" rev-parse HEAD
  git -C "$TAG_SOURCE" rev-parse HEAD
  git -C "$TAG_SOURCE" ls-tree HEAD crates/tile_std crates/tile_std_macros
  git -C "$TAG_SOURCE" diff --stat "$BASE_SHA" HEAD -- crates/tile_std crates/tile_std_macros
} > "$EV/source-provenance.txt"
record source public_tag_checked_out
required toolchain rustup toolchain install nightly-2025-08-04 --profile minimal --component rustc-dev --component llvm-tools --component rust-src
required rustc rustc +nightly-2025-08-04 -vV
required cargo_version cargo +nightly-2025-08-04 -V
required components rustup component list --installed --toolchain nightly-2025-08-04

API='https://api.github.com/repos/yijunyu/tile-rs/releases/tags/v0.0.2%2Bnightly-2025-08-04'
URL='https://github.com/yijunyu/tile-rs/releases/download/v0.0.2%2Bnightly-2025-08-04/tile-rs-codegen-aarch64-apple-darwin.tar.gz'
printf '%s\n' "$URL" > "$EV/asset-url.txt"
run release_api curl --fail --silent --show-error --location --max-time 90 "$API" -o "$EV/release-api.json"
if [ "$(< "$EV/logs/release_api.exit")" = 0 ]; then
  python3 - "$EV/release-api.json" "$TAG" "$EXPECTED_SHA" "$URL" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
assert j['tag_name'] == sys.argv[2], 'release tag metadata changed'  # target_commitish may be 'main'; public git ref verified separately
a, = [a for a in j['assets'] if a['name'] == 'tile-rs-codegen-aarch64-apple-darwin.tar.gz']
assert (a['digest'], a['browser_download_url'], a['size']) == ('sha256:' + sys.argv[3], sys.argv[4], 103449511)
PY
  record release_api digest_confirmed
else
  record release_api unavailable_using_prechecked_digest
fi
TARBALL="$WORK/release.tar.gz"
required download curl --fail --silent --show-error --location --max-time 600 "$URL" -o "$TARBALL"
required sha256 shasum -a 256 "$TARBALL"
if [ "$(awk '{print $1}' "$EV/logs/sha256.stdout")" != "$EXPECTED_SHA" ]; then record download checksum_mismatch; exit 2; fi
python3 - "$TARBALL" "$EV/archive-members.txt" <<'PY'
import os, pathlib, sys, tarfile
assert os.path.getsize(sys.argv[1]) == 103449511
with tarfile.open(sys.argv[1], 'r:gz') as archive:
    names = []
    for member in archive:
        p = pathlib.PurePosixPath(member.name)
        assert not p.is_absolute() and '..' not in p.parts and p.parts[0] == 'tile-rs-codegen', member.name
        assert member.isdir() or member.isfile(), 'link/special member: ' + member.name
        names.append(member.name)
    for required in ('tile-rs-codegen/USAGE.md', 'tile-rs-codegen/lib/librustc_codegen_tile.dylib',
                     'tile-rs-codegen/lib/libtile_std_macros.dylib'):
        assert required in names, 'missing bundle member: ' + required
    pathlib.Path(sys.argv[2]).write_text('\n'.join(names) + '\n')
PY
required extract tar -xzf "$TARBALL" -C "$WORK"
BUNDLE="$WORK/tile-rs-codegen"
cp "$BUNDLE/USAGE.md" "$EV/inputs/release-USAGE.md"
if [ -f "$BUNDLE/.cargo/config.toml" ]; then
  cp "$BUNDLE/.cargo/config.toml" "$EV/inputs/release-config.toml"
else
  echo 'Verified Release tarball has no .cargo/config.toml; public workflow specifies equivalent flags.' > "$EV/inputs/release-config-absence.txt"
fi
required macro_file file "$BUNDLE/lib/libtile_std_macros.dylib"
required macro_dependencies otool -L "$BUNDLE/lib/libtile_std_macros.dylib"
required macro_sha shasum -a 256 "$BUNDLE/lib/libtile_std_macros.dylib"
record download verified_sha256

mkdir -p "$WORK/emitter/src"
cp experiments/rust-kernel-e2e/emitter/src/main.rs "$WORK/emitter/src/main.rs"
cp experiments/rust-kernel-e2e/kernel/src/lib.rs "$EV/inputs/kernel-lib.rs"
cp experiments/rust-kernel-e2e/bundled-macro-wrapper.py "$EV/inputs/bundled-macro-wrapper.py"
chmod +x "$EV/inputs/bundled-macro-wrapper.py"
cat > "$WORK/emitter/Cargo.toml" <<EOF
[package]
name = "fixed_source_emitter_probe"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
tile_codegen = { path = "$BASE/crates/tile_codegen", features = ["emitters"] }
EOF
cp "$WORK/emitter/Cargo.toml" "$EV/inputs/emitter-Cargo.toml"
cp "$WORK/emitter/src/main.rs" "$EV/inputs/emitter-main.rs"
required emitter_build env CARGO_TARGET_DIR="$WORK/emitter-target" cargo +nightly-2025-08-04 build --manifest-path "$WORK/emitter/Cargo.toml" -v

# Two independent kernel builds. In bundled arm Cargo still compiles the public-tag
# macro as a declared dependency; the rustc wrapper replaces its --extern argument
# ONLY for tile_std, recording the exact replacement. No false claim of a pure
# bundled-macro build. Every compilation and emitted MLIR is separately retained.
for arm in tag_source_macro bundled_macro; do
  mkdir -p "$WORK/$arm/src" "$WORK/$arm/.cargo" "$EV/$arm/mlir" "$EV/$arm/msl"
  cp "$EV/inputs/kernel-lib.rs" "$WORK/$arm/src/lib.rs"
  cat > "$WORK/$arm/Cargo.toml" <<EOF
[package]
name = "kernel_add_e2e"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
tile_std = { path = "$TAG_SOURCE/crates/tile_std" }
[lib]
crate-type = ["cdylib", "lib"]
EOF
  extra_search=''
  # A re-exported proc macro must also be discoverable when compiling the
  # dependent kernel, not only tile_std's direct --extern invocation.
  if [ "$arm" = bundled_macro ]; then extra_search="  \"-Ldependency=$BUNDLE/lib\","; fi
  cat > "$WORK/$arm/.cargo/config.toml" <<EOF
[target.aarch64-apple-darwin]
rustflags = [
  "-Zcodegen-backend=$BUNDLE/lib/librustc_codegen_tile.dylib",
  "-Zcrate-attr=feature(register_tool)",
  "-Zcrate-attr=register_tool(tile)",
  "-Cpanic=abort",
  "-Clto=off",
$extra_search
]
EOF
  cp "$WORK/$arm/Cargo.toml" "$EV/inputs/$arm-Cargo.toml"
  cp "$WORK/$arm/.cargo/config.toml" "$EV/inputs/$arm-config.toml"
  if [ "$arm" = bundled_macro ]; then
    # Exact digest and link dependencies above; test whether rustc can load it.
    WRAPPER_LOG="$EV/logs/bundled-macro-swaps.txt"
    : > "$WRAPPER_LOG"
    pushd "$WORK/$arm" >/dev/null
    run "$arm.frontend" env CARGO_TARGET_DIR="$WORK/$arm-target" TILERS_CODEGEN_PATH=metal \
      TILERS_CODEGEN_SO="$BUNDLE/lib/librustc_codegen_tile.dylib" \
      BUNDLE_MACRO="$BUNDLE/lib/libtile_std_macros.dylib" WRAPPER_LOG="$WRAPPER_LOG" \
      RUSTC_WRAPPER="$EV/inputs/bundled-macro-wrapper.py" \
      cargo +nightly-2025-08-04 build --target aarch64-apple-darwin --release -v
    popd >/dev/null
    if [ ! -s "$WRAPPER_LOG" ]; then record "$arm.macro" not_applied; fi
  else
    pushd "$WORK/$arm" >/dev/null
    run "$arm.frontend" env CARGO_TARGET_DIR="$WORK/$arm-target" TILERS_CODEGEN_PATH=metal \
      TILERS_CODEGEN_SO="$BUNDLE/lib/librustc_codegen_tile.dylib" \
      cargo +nightly-2025-08-04 build --target aarch64-apple-darwin --release -v
    popd >/dev/null
  fi
  if [ "$(< "$EV/logs/$arm.frontend.exit")" = 0 ]; then record "$arm.frontend" compiled; else record "$arm.frontend" compilation_failed; fi
  find "$WORK/$arm-target" -type f -name '*.mlir' -print | sort > "$EV/$arm/mlir-paths.txt" || true
  selected=''
  while IFS= read -r file; do
    cp "$file" "$EV/$arm/mlir/$(basename "$file")"
    if grep -q 'add_f32_small' "$file"; then
      if [ -n "$selected" ]; then record "$arm.mlir" ambiguous_kernel_files; selected=''; break; fi
      selected="$file"
    fi
  done < "$EV/$arm/mlir-paths.txt"
  if [ -z "$selected" ]; then record "$arm.mlir" absent_kernel_mlir; continue; fi
  cp "$selected" "$EV/$arm/mlir/actual-kernel.mlir"
  shasum -a 256 "$EV/$arm/mlir/actual-kernel.mlir" > "$EV/$arm/mlir/actual-kernel.sha256"
  record "$arm.mlir" exported_actual_kernel
  run "$arm.emitter" "$WORK/emitter-target/debug/fixed_source_emitter_probe" \
    "$EV/$arm/mlir/actual-kernel.mlir" "$EV/$arm/msl/fixed-source.metal"
  if [ "$(< "$EV/logs/$arm.emitter.exit")" != 0 ]; then record "$arm.emitter" rejected_actual_mlir; continue; fi
  shasum -a 256 "$EV/$arm/msl/fixed-source.metal" > "$EV/$arm/msl/fixed-source.sha256"
  record "$arm.emitter" accepted_generated_msl
done
(cd "$EV/inputs" && shasum -a 256 *.toml *.rs *.py > ../input-sha256.txt)
# Preserve all output including failures; Actions marks the experiment run successful
# ONLY if both front end and independent emitter accepted the actual kernel.
for arm in tag_source_macro bundled_macro; do
  [ -f "$EV/logs/$arm.frontend.exit" ] && [ "$(< "$EV/logs/$arm.frontend.exit")" = 0 ] || exit 1
  [ -f "$EV/logs/$arm.emitter.exit" ] && [ "$(< "$EV/logs/$arm.emitter.exit")" = 0 ] || exit 1
done
