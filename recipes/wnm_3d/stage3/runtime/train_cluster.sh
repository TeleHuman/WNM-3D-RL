#!/usr/bin/env bash
# Stage-3 official topology-independent data/runtime plumbing.
set -euo pipefail

# Keep standalone launches aligned with GN0's deployed VGGT sampler. The
# checkpoint value (4) is a training default and is intentionally overridden.
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-16}
export ENABLE_DIT_CACHE=${ENABLE_DIT_CACHE:-true}
export ENABLE_CFG=${ENABLE_CFG:-true}
export CFG_SCALE=${CFG_SCALE:-5.0}
export DYNAMIC_CACHE_SCHEDULE=${DYNAMIC_CACHE_SCHEDULE:-false}

RUNTIME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(cd "$RUNTIME_DIR/.." && pwd)
REPO_ROOT=${WAM_REPO_ROOT:-$(cd "$RUNTIME_DIR/../../../.." && pwd)}
export WAM_REPO_ROOT=$REPO_ROOT
source "$RECIPE_DIR/algorithm.env.sh"
source "$RECIPE_DIR/resources.env.sh"

# VGGT model and frozen conditioning assets.
export WNM3D_SOURCE_ROOT=${WNM3D_SOURCE_ROOT:?Set WNM3D_SOURCE_ROOT in paths.env}
export WNM3D_INITIAL_CHECKPOINT=${WNM3D_INITIAL_CHECKPOINT:?Set WNM3D_INITIAL_CHECKPOINT in paths.env}
export WNM_TOKENIZER_SOURCE=${WNM_TOKENIZER_SOURCE:?Set WNM_TOKENIZER_SOURCE in paths.env}
export WNM_TEXT_ENCODER_SOURCE=${WNM_TEXT_ENCODER_SOURCE:?Set WNM_TEXT_ENCODER_SOURCE in paths.env}
export WNM_IMAGE_ENCODER_SOURCE=${WNM_IMAGE_ENCODER_SOURCE:?Set WNM_IMAGE_ENCODER_SOURCE in paths.env}
export WNM_VAE_SOURCE=${WNM_VAE_SOURCE:?Set WNM_VAE_SOURCE in paths.env}
export WNM_VGGT_SOURCE=${WNM_VGGT_SOURCE:?Set WNM_VGGT_SOURCE in paths.env}
export WNM_RUNTIME_CKPT_CACHE_NAME=${WNM_RUNTIME_CKPT_CACHE_NAME:-policy_init_checkpoint}
export WNM_RUNTIME_CKPT_EXCLUDE=${WNM_RUNTIME_CKPT_EXCLUDE:-}

export WAM_MODEL_ARCHITECTURE=WNM3D
export WAM_DATA_ROOT=${WAM_DATA_ROOT:?Set WAM_DATA_ROOT in paths.env}
export WAM_EXPECTED_TRAIN_SIZE=${WAM_EXPECTED_TRAIN_SIZE:?Set WAM_EXPECTED_TRAIN_SIZE in paths.env}
export WAM_EXPECTED_VAL_SIZE=0
# The train-only manifest needs a bootstrap val loader; step-0 validation hot
# switches it atomically to the held-out event parquet below.
export WAM_PROFILE_USE_TRAIN_AS_VAL=true
export WAM_VAL_MAX_SAMPLES=${WAM_VAL_MAX_SAMPLES:?Set WAM_VAL_MAX_SAMPLES in paths.env}
export WAM_OUTPUT_DIR=${WAM_OUTPUT_DIR:-$REPO_ROOT/outputs/stage3_navigation_grpo}
export WAM_EXPERIMENT_NAME=${WAM_EXPERIMENT_NAME:-wnm_3d_stage3_navigation_grpo}
export WAM_RUN_LABEL=${WAM_RUN_LABEL:-"WNM-3D Stage-3 navigation GRPO N=16"}
export WAM_VARIANT_NAME=${WAM_VARIANT_NAME:-wnm-3d-stage3-navigation-grpo}
export WAM_RESUME_MODE=${WAM_RESUME_MODE:-disable}

# Keep startup assets on tmpfs; all sources above remain authoritative.
export WAM_STAGE_RUNTIME_ASSETS_TO_RAM=true
export WAM_RAM_CACHE_ROOT=/dev/shm/wnm_3d_stage3_runtime
export WAM_RUNTIME_CACHE_KIND=tmpfs
export TORCHINDUCTOR_CACHE_DIR="$WAM_RAM_CACHE_ROOT/torchinductor_cache"

# Topology-dependent workers and batch sizes were resolved by resources.env.sh.
# Rollout N=16 and four branches per stratum remain algorithm invariants.
# Keep the native joint-DiT gradient. The former 4x backward-only action gain
# did not improve the usable action policy and obscured optimizer diagnostics.
export WAM_ACTION_BACKBONE_GRAD_GAIN=${WAM_ACTION_BACKBONE_GRAD_GAIN:-1.0}

export WAM_VAL_BEFORE_TRAIN=true
export WAM_TEST_FREQ=50
export WAM_SAVE_FREQ=200
export WAM_VALIDATION_MAX_MEDIA_SAMPLES=4
export WAM_VALIDATION_CHECKPOINT_GUARD=true
export WAM_ROLLOUT_REPLAY_ATOL=1000000000
export WAM_ROLLOUT_REPLAY_FAIL_ON_MISMATCH=false
export WAM_RUNTIME_STATE_DIR=${WAM_RUNTIME_STATE_DIR:-$WAM_OUTPUT_DIR/runtime_controls}
export WAM_VAL_PATH_CONTROL_FILE=${WAM_VAL_PATH_CONTROL_FILE:-$WAM_RUNTIME_STATE_DIR/validation.json}
export WAM_EVENT_VAL_FILE=${WAM_EVENT_VAL_FILE:?Set WAM_EVENT_VAL_FILE in paths.env}
"${PYTHON_BIN:-python}" - <<'PY'
import json
import os
from pathlib import Path
import tempfile

control_path = Path(os.environ["WAM_VAL_PATH_CONTROL_FILE"]).expanduser()
control_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "files": [os.environ["WAM_EVENT_VAL_FILE"]],
    "max_samples": int(os.environ["WAM_VAL_MAX_SAMPLES"]),
}
descriptor, temporary = tempfile.mkstemp(
    dir=control_path.parent,
    prefix=f".{control_path.name}.",
    suffix=".tmp",
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, control_path)
except BaseException:
    Path(temporary).unlink(missing_ok=True)
    raise
PY

# Deployment-aligned goal-entry and STOP semantics.
export WAM_GOAL_SCORE_USE_POTENTIAL_DELTA=true
export WAM_GOAL_ENTRY_BONUS=0.15
export WAM_CHUNK_MOTION_STOP_ENABLED=true
export WAM_CHUNK_MOTION_STOP_THRESHOLD_M=0.15
# Match GN0 exactly within each 8-action block: first sum XY path length and
# STOP before execution when it is strictly below 0.15 m; otherwise STOP at
# the first component-wise <=1e-3 action. Deployment diagnostics consume only
# chunk 0, while policy training applies the same detector independently to
# chunks 0--3 so all four outputs learn the deployed STOP convention.
export WAM_CHUNK_MOTION_STOP_METRIC=path_length
export WAM_DEPLOY_STOP_SEMANTICS_ENABLED=true
export WAM_DEPLOY_ACTION_NUM=8
export WAM_DEPLOY_STOP_EPS=1e-3
export WAM_CORRECT_STOP_BONUS=0.15
export WAM_PREMATURE_STOP_PENALTY=${WAM_PREMATURE_STOP_PENALTY:-0.50}
export WAM_NDTW_WEIGHT=0.0
export WAM_GEODESIC_BACKTRACK_WEIGHT=0.0
export WAM_YAW_ROTATION_FLOOR_RAD=${WAM_YAW_ROTATION_FLOOR_RAD:-1000000}
export WAM_REWARD_METRICS_COMPACT=true

# Stage-3 collision shaping preserves the 2 px diagnostic hard classifier and
# adds a
# quadratic warning band that reaches zero at GN0's 4 px execution margin.
# Collision stays inside the ordinary navigation reward; no separate
# group-normalized collision advantage or actor loss is constructed.
export WAM_OCCUPANCY_MARGIN_PX=${WAM_OCCUPANCY_MARGIN_PX:-2}
export WAM_COLLISION_PENALTY_WEIGHT=${WAM_COLLISION_PENALTY_WEIGHT:-0.40}
export WAM_COLLISION_SOFT_ENABLED=${WAM_COLLISION_SOFT_ENABLED:-true}
export WAM_COLLISION_SOFT_MARGIN_PX=${WAM_COLLISION_SOFT_MARGIN_PX:-4}
export WAM_COLLISION_SOFT_PENALTY_WEIGHT=${WAM_COLLISION_SOFT_PENALTY_WEIGHT:-0.20}
export WAM_COLLISION_CREDIT_ENABLED=${WAM_COLLISION_CREDIT_ENABLED:-false}
export WAM_COLLISION_LOSS_WEIGHT=${WAM_COLLISION_LOSS_WEIGHT:-0.0}
export WAM_COLLISION_USE_CHUNK_WEIGHTS=${WAM_COLLISION_USE_CHUNK_WEIGHTS:-false}

export WAM_LAUNCH_OVERRIDE_PATH=${WAM_LAUNCH_OVERRIDE_PATH:-$0}
mkdir -p "$WAM_OUTPUT_DIR/config_snapshot"
cp "$0" "$WAM_OUTPUT_DIR/config_snapshot/launch_override.sh"

cat <<PARAMETERS
WNM-3D Stage-3 $WAM_VARIANT_NAME N=16 launch
  checkpoint:          $WNM3D_INITIAL_CHECKPOINT
  data:                $WAM_DATA_ROOT
  output:              $WAM_OUTPUT_DIR
  train rows/steps:    $WAM_EXPECTED_TRAIN_SIZE / $WAM_TOTAL_STEPS
  prompt/rollout:      $WAM_PROMPT_BATCH_SIZE / $WAM_ROLLOUT_N
  nodes/GPUs:          $WAM_NNODES / $WAM_TOTAL_GPUS total ($WAM_NUM_GPUS per node)
  server request B:    $WAM_ROLLOUT_MAX_NUM_SEQS
  actor mini/micro:    $WAM_PPO_MINI_BATCH_SIZE / $WAM_MICRO_BATCH_SIZE
  layer credit:        $WAM_LAYER_CONDITIONED_CREDIT_ENABLED, branches=$WAM_LAYER_CONDITIONED_BRANCHES_PER_STRATUM
  action grad gain:    $WAM_ACTION_BACKBONE_GRAD_GAIN (backward-only into shared DiT)
  collision reward:    recovery=${WAM_COLLISION_RECOVERY_ENABLED:-false}; terminal safety=${WAM_TERMINAL_SAFETY_ADVANTAGE_ENABLED:-false}; separate loss=$WAM_COLLISION_LOSS_WEIGHT
  collision execution: chunk0 margin=${WAM_DEPLOYMENT_COLLISION_MARGIN_PX:-4}px; first-hit freeze=${WAM_COLLISION_STOP_ENABLED:-true}
  collision shaping:   first=@$WAM_COLLISION_PENALTY_WEIGHT, repeats=@${WAM_COLLISION_REPEAT_PENALTY_WEIGHT:-0.0} cap=${WAM_COLLISION_REPEAT_PENALTY_CAP_COUNT:-0}; legacy hard=${WAM_OCCUPANCY_MARGIN_PX}px
  recovery shaping:    bonus=@${WAM_COLLISION_RECOVERY_BONUS_WEIGHT:-0.0}; clearance=${WAM_COLLISION_RECOVERY_CLEARANCE_PX:-0}px; escape=${WAM_COLLISION_RECOVERY_MIN_ESCAPE_M:-0}-${WAM_COLLISION_RECOVERY_FULL_ESCAPE_M:-0}m
  soft shaping:        quadratic ${WAM_OCCUPANCY_MARGIN_PX}-${WAM_COLLISION_SOFT_MARGIN_PX}px@$WAM_COLLISION_SOFT_PENALTY_WEIGHT
  validation:          $WAM_VAL_MAX_SAMPLES event-balanced rows at step 0 / every $WAM_TEST_FREQ
  save/validation:     $WAM_SAVE_FREQ / $WAM_TEST_FREQ
PARAMETERS

"${PYTHON_BIN:-python}" "$WAM_CONFIG_CONTRACT_VERIFIER"
exec bash "$RUNTIME_DIR/train_reward.sh" --run "$@" \
  "trainer.nnodes=$WAM_NNODES" \
  "++ray_kwargs.ray_init.address=auto" \
  "ray_kwargs.ray_init.num_cpus=null" \
  "++actor_rollout_ref.actor.diffusion_loss.action_collision_loss_weight=$WAM_COLLISION_LOSS_WEIGHT" \
  "++actor_rollout_ref.actor.diffusion_loss.action_collision_use_chunk_weights=$WAM_COLLISION_USE_CHUNK_WEIGHTS" \
  "algorithm.validation_reference_kl.enabled=true" \
  "algorithm.validation_reference_kl.micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
