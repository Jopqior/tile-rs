#!/usr/bin/env bash
# THROWAWAY: no CPU emulation, no skip-on-missing-software path.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${BUILD_DIR:?Set an isolated absolute BUILD_DIR}"
: "${INSTALL_ROOT:?Set the full Toolkit installation root}"
mode=${PROGRAM_MODE:-handwritten}
[[ $mode == handwritten || $mode == generated ]] || exit 2
[[ ${INJECT_ERROR:-0} == 0 || ${INJECT_ERROR:-0} == 1 ]] || exit 2
mkdir -p "$BUILD_DIR"
work=$(cd "$BUILD_DIR" && pwd)
# CANN's environment script may reference unset variables.
set +u
source "$INSTALL_ROOT/cann/set_env.sh"
set -u
sim="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib"
canonical="$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_2201/lib"
test "$(realpath "$sim")" = "$(realpath "$canonical")"
test -f "$sim/libruntime_camodel.so"
export LD_LIBRARY_PATH="$sim:${LD_LIBRARY_PATH:-}"
python3 - <<'PY'
import glob, platform, sys
assert sys.version_info[:2] == (3, 10) and platform.machine() == 'x86_64'
assert not any(glob.glob(p) for p in ('/dev/davinci*', '/dev/devmm_svm', '/dev/hisi_hdc')), 'Unexpected exposed NPU'
PY
sha=82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19
if [[ ! -d $work/pto-isa/.git ]]; then
    git init -q "$work/pto-isa"
    git -C "$work/pto-isa" remote add origin https://github.com/hw-native-sys/pto-isa.git
fi
if ! git -C "$work/pto-isa" rev-parse --verify HEAD >/dev/null 2>&1; then
    for attempt in 1 2 3; do
        if timeout 120 git -C "$work/pto-isa" fetch --depth=1 origin "$sha"; then break; fi
        [[ $attempt != 3 ]] || exit 1
        sleep 5
    done
    git -C "$work/pto-isa" checkout --detach FETCH_HEAD
fi
test "$(git -C "$work/pto-isa" rev-parse HEAD)" = "$sha"
test -z "$(git -C "$work/pto-isa" status --porcelain)"
flags=()
cp "$here/add.pto" "$work/input.pto"
cp "$here/kernel.cpp" "$here/host.cpp" "$here/check.py" "$work/"
{
    printf 'PROGRAM_MODE=%s\nINJECT_ERROR=%s\nPTO_ISA=%s\n' "$mode" "${INJECT_ERROR:-0}" "$sha"
    echo 'CANN_TOOLKIT=9.1.0'
    echo 'TOOLKIT_SHA256=985b8c7b68a5f85af7c28c3514f3d3f6baec1e0784669f2bba0cce2b502dabe9'
    echo 'SOC=Ascend910B1; manifest mapping=dav_2201; arch=dav-c220-vec'
    echo 'exposed_Ascend_devices=none; driver/firmware_install=none'
    uname -a
    python3 --version
    bisheng --version
    sha256sum "$(command -v bisheng)" "$sim/libruntime_camodel.so"
    dpkg-query -W gcc g++ python3 libc6 libstdc++6
} > "$work/inventory.txt"
if [[ $mode == generated ]]; then
    wheel=ptoas-0.65-cp310-cp310-manylinux_2_34_x86_64.whl
    if [[ ! -f $work/$wheel ]]; then
        curl -fL --retry 3 "https://github.com/hw-native-sys/PTOAS/releases/download/v0.65/$wheel" -o "$work/$wheel" >"$work/ptoas-download-private.log" 2>&1
    fi
    echo "28ba50ddc684b3b011262cd26677a2509428beaf5e267cb6bee5b4c1f776712b  $work/$wheel" | sha256sum -c -
    python3 -m venv "$work/venv"
    "$work/venv/bin/python" -m pip install --no-index --no-deps "$work/$wheel" >"$work/pip-private.log" 2>&1
    "$work/venv/bin/ptoas" --version >> "$work/inventory.txt" 2>&1
    "$work/venv/bin/ptoas" "$work/input.pto" --pto-arch=a3 --enable-insert-sync -o "$work/generated.cpp" >"$work/ptoas-private.log" 2>&1
    sha256sum "$work/$wheel" "$work/input.pto" "$work/generated.cpp" >> "$work/inventory.txt"
    flags+=(-DGENERATED)
fi
cd "$work"
if ! bisheng -xcce -O2 -std=c++17 --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE \
    "${flags[@]}" -fPIC -shared --cce-fatobj-link -I"$work/pto-isa/include" -I"$work" \
    -I"$ASCEND_HOME_PATH/include" "$here/kernel.cpp" -o libadd.so >compile-private.log 2>&1; then
    echo 'FAIL: Bisheng device compilation failed'; exit 1
fi
if ! g++ -std=c++17 "$here/host.cpp" -I"$ASCEND_HOME_PATH/include" -L. -ladd \
    -L"$sim" -Wl,--no-as-needed -lruntime_camodel -L"$ASCEND_HOME_PATH/lib64" \
    -lascendcl -ldl '-Wl,-rpath,$ORIGIN' -o add >link-private.log 2>&1; then
    echo 'FAIL: ACL/camodel host linking failed'; exit 1
fi
# pipefail propagates oracle failure, including deliberately corrupted output.
python3 "$here/check.py" "$work/add" "$work" "${INJECT_ERROR:-0}" 2>&1 | tee "$work/result.txt"
