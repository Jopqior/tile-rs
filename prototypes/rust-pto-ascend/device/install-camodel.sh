#!/usr/bin/env bash
# THROWAWAY: authorized full Toolkit install, no driver/firmware.
# On a developer machine run ONLY inside a disposable Ubuntu 22.04 container.
set -euo pipefail
: "${BUILD_DIR:?Set an isolated absolute BUILD_DIR}"
: "${INSTALL_ROOT:?Set an isolated absolute INSTALL_ROOT}"
mkdir -p "$BUILD_DIR" "$INSTALL_ROOT"
sha=985b8c7b68a5f85af7c28c3514f3d3f6baec1e0784669f2bba0cce2b502dabe9
url='https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-x86_64.run'
# Full installation plus package/extraction needs substantial temporary space.
available=$(df -Pk "$INSTALL_ROOT" | awk 'NR==2 {print $4}')
(( available >= 20 * 1024 * 1024 )) || { echo 'FAIL: need 20 GiB free'; exit 1; }
package=${TOOLKIT_PACKAGE:-$BUILD_DIR/toolkit.run}
if [[ ! -f $package ]]; then
    curl -fL --retry 3 "$url" -o "$package" >"$BUILD_DIR/download-private.log" 2>&1
fi
printf '%s  %s\n' "$sha" "$package" | sha256sum -c -
# Installer output is not a public artifact or Actions console log.
if ! bash "$package" --quiet --full --install-path="$INSTALL_ROOT" >"$BUILD_DIR/install-private.log" 2>&1; then
    echo 'FAIL: full Toolkit installation failed (private log retained locally)'
    exit 1
fi
test -f "$INSTALL_ROOT/cann/set_env.sh"
echo 'Full CANN Toolkit installed; driver/firmware not installed.'
