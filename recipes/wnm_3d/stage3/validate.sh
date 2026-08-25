#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$RECIPE_DIR/../../.." && pwd)
PATHS_FILE=${WAM_PATHS_FILE:-$RECIPE_DIR/paths.env}

if [[ ! -f "$PATHS_FILE" ]]; then
    echo "Missing $PATHS_FILE; copy paths.env.example to paths.env and edit it." >&2
    exit 2
fi

source "$PATHS_FILE"
export WAM_REPO_ROOT=${WAM_REPO_ROOT:-$REPO_ROOT}
source "$RECIPE_DIR/algorithm.env.sh"
source "$RECIPE_DIR/resources.env.sh"

exec "${PYTHON_BIN:-python}" \
    "$RECIPE_DIR/verify_config.py" "$@"
