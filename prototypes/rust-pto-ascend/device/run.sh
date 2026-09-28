#!/usr/bin/env bash
# CANN's environment script must be sourced before starting the Python runner.
set -euo pipefail
: "${CANN_ENV:?Set the installed CANN environment script path}"
set +u
# shellcheck disable=SC1090
source "$CANN_ENV"
set -u
: "${ASCEND_HOME_PATH:?CANN environment did not set ASCEND_HOME_PATH}"
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/run.py" "$@"
