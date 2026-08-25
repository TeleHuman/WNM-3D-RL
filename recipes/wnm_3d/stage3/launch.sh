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

if (( $# != 0 )); then
    echo "Stage-3 launch does not accept positional Hydra overrides." >&2
    echo "Set machine topology and paths through paths.env or WAM_* environment variables." >&2
    exit 2
fi

printf 'Stage-3 resources: nodes=%s GPUs/node=%s total_GPUs=%s prompt_batch=%s rollout_n=%s micro_batch/GPU=%s steps=%s\n' \
    "$WAM_NNODES" "$WAM_NUM_GPUS" "$WAM_TOTAL_GPUS" \
    "$WAM_PROMPT_BATCH_SIZE" "$WAM_ROLLOUT_N" "$WAM_MICRO_BATCH_SIZE" \
    "$WAM_TOTAL_STEPS"

exec bash "$RECIPE_DIR/runtime/cluster_entry.sh"
