#!/usr/bin/env bash
# Install pinned PTO headers and, for simulation, the PTO-to-C++ compiler.
set -euo pipefail
if [[ $# != 2 || ( $1 != sim && $1 != npu ) || $2 != /* ]]; then
    echo "Usage: $0 sim|npu /absolute/tools-directory" >&2
    exit 2
fi
mode=$1
tools=$2
mkdir -p "$tools"
sha=82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19
if [[ ! -d $tools/pto-isa/.git ]]; then
    git init -q "$tools/pto-isa"
    git -C "$tools/pto-isa" remote add origin https://github.com/hw-native-sys/pto-isa.git
    git -C "$tools/pto-isa" fetch --depth=1 origin "$sha"
    git -C "$tools/pto-isa" checkout --detach FETCH_HEAD
fi
test "$(git -C "$tools/pto-isa" rev-parse HEAD)" = "$sha"
test -z "$(git -C "$tools/pto-isa" status --porcelain --untracked-files=all)"
printf 'PTO_ISA=%s\n' "$sha" > "$tools/versions.txt"
if [[ $mode == sim ]]; then
    python3 -c 'import platform, sys; assert platform.machine() == "x86_64" and sys.version_info[:2] == (3, 10)'
    wheel=ptoas-0.65-cp310-cp310-manylinux_2_34_x86_64.whl
    curl -fL --retry 3 "https://github.com/hw-native-sys/PTOAS/releases/download/v0.65/$wheel" -o "$tools/$wheel"
    echo "28ba50ddc684b3b011262cd26677a2509428beaf5e267cb6bee5b4c1f776712b  $tools/$wheel" | sha256sum -c -
    python3 -m venv "$tools/venv"
    "$tools/venv/bin/python" -m pip install --no-index --no-deps "$tools/$wheel"
    "$tools/venv/bin/ptoas" --version >> "$tools/versions.txt" 2>&1
fi
