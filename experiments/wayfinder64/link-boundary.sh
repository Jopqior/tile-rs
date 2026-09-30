#!/usr/bin/env bash
# Direct Cargo artifact-boundary probe; does not modify audited source or CLI.
set -euo pipefail
evidence="$1"
tile="$2"
export TILERS_CODEGEN_SO="$(find "$TILE_HOME" -name librustc_codegen_tile.dylib -print -quit)"
export TILERS_CODEGEN_PATH=pto
original=""
for f in "$evidence"/backend-capture/*/Cargo.toml; do
  if grep -q release-libs "$f"; then original="$(dirname "$f")"; break; fi
done
for mode in release-link release-rlib; do
  work="$RUNNER_TEMP/$mode"
  dst="$evidence/$mode"
  mkdir -p "$work" "$dst"
  cp "$original/Cargo.toml" "$original/Cargo.lock" "$work/"
  cp -R "$original/src" "$original/.cargo" "$work/"
  if [ "$mode" = release-rlib ]; then
    python3 - "$work/Cargo.toml" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
a='crate-type = ["cdylib", "lib"]'
assert a in s
p.write_text(s.replace(a, 'crate-type = ["rlib"]'))
PY
  fi
  cp "$work/Cargo.toml" "$dst/Cargo.toml"
  (
    cd "$work"
    echo 'cargo +nightly-2025-08-04 build --locked --release --target aarch64-apple-darwin' > "$dst/command.txt"
    set +e
    cargo +nightly-2025-08-04 build --locked --release --target aarch64-apple-darwin > "$dst/stdout" 2> "$dst/stderr"
    echo "$?" > "$dst/exit"
    set -e
    find target -type f \( -name '*.mlir' -o -name '*tile_kernel*.rlib' -o -name '*.tile.*' \) -print > "$dst/artifact-paths.txt"
    while IFS= read -r f; do
      mkdir -p "$dst/$(dirname "$f")"
      cp "$f" "$dst/$f"
    done < "$dst/artifact-paths.txt"
    python3 - "$dst" <<'PY'
from pathlib import Path
import tarfile,sys
r=Path(sys.argv[1])
for p in r.rglob('*.rlib'):
    if not tarfile.is_tarfile(p): continue
    with tarfile.open(p) as t:
        p.with_suffix('.members.txt').write_text('\n'.join(t.getnames()))
        for i,m in enumerate(t.getmembers()):
            if m.isfile() and (m.name.endswith('.metadata') or m.name.endswith('.mlir')):
                p.with_name(p.name+f'.member{i}.mlir').write_bytes(t.extractfile(m).read())
PY
    while IFS= read -r f; do
      set +e
      "$tile" "$f" -f mlir -t pto -O0 --cross ascend --offline --no-install -o "$f.pto-output" > "$f.cli.stdout" 2> "$f.cli.stderr"
      echo "$?" > "$f.cli.exit"
      set -e
    done < <(find "$dst" -name '*.mlir' -type f)
  )
done
