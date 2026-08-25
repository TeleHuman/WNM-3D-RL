#!/usr/bin/env bash
# Start an arbitrary-size Ray cluster for wnm-3d-stage3-navigation-grpo.
set -euo pipefail

RUNTIME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(cd "$RUNTIME_DIR/.." && pwd)
REPO_ROOT=${WAM_REPO_ROOT:-$(cd "$RUNTIME_DIR/../../../.." && pwd)}
export WAM_REPO_ROOT=$REPO_ROOT
source "$RECIPE_DIR/algorithm.env.sh"
source "$RECIPE_DIR/resources.env.sh"
"${PYTHON_BIN:-python}" "$WAM_CONFIG_CONTRACT_VERIFIER"
export TRAIN_SCRIPT=$RUNTIME_DIR/train_entry.sh

# algorithm.env.sh is the single source of reward and sampler semantics. It is
# sourced before raylet startup so every worker inherits the same contract.

if [[ "${WAM_REPLACE_EXISTING_RAY:-false}" != "true" ]] \
    && pgrep -x raylet >/dev/null 2>&1; then
    echo "A Ray runtime is already active on this node." >&2
    echo "Stop it explicitly, or set WAM_REPLACE_EXISTING_RAY=true to replace it." >&2
    exit 2
fi

# The Stage2 checkpoint cache lives in node-local /dev/shm. Every node must
# materialize the signature-qualified checkpoint and frozen assets before its
# raylet can place a worker there; head-only staging produces a valid driver
# path that is nonexistent on the worker. WAM_STAGE_ONLY executes the normal
# verified staging path and exits before Ray/training construction.
export WAM_STAGE_ONLY=true
durable_wam_output_dir=${WAM_OUTPUT_DIR:?Set WAM_OUTPUT_DIR in paths.env}
export WAM_OUTPUT_DIR=/tmp/stage3_runtime_stage_node_${NODE_RANK:-unknown}
export WAM_EXPERIMENT_NAME=stage3_runtime_stage_only
bash "$TRAIN_SCRIPT"
unset WAM_STAGE_ONLY WAM_EXPERIMENT_NAME
export WAM_OUTPUT_DIR=$durable_wam_output_dir

exec bash "$RUNTIME_DIR/ray_cluster.sh"
