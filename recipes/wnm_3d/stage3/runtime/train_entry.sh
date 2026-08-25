#!/usr/bin/env bash
# Stage-3 official distance-scaled STOP, deployment collision clipping, and
# bounded collision-recovery credit.
set -euo pipefail

RUNTIME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(cd "$RUNTIME_DIR/.." && pwd)
REPO_ROOT=${WAM_REPO_ROOT:-$(cd "$RUNTIME_DIR/../../../.." && pwd)}
export WAM_REPO_ROOT=$REPO_ROOT
source "$RECIPE_DIR/algorithm.env.sh"
source "$RECIPE_DIR/resources.env.sh"
REWARD_PATH=$RECIPE_DIR/reward.py
export WAM_VARIANT_NAME=wnm-3d-stage3-navigation-grpo
export WAM_OUTPUT_DIR=${WAM_OUTPUT_DIR:-$REPO_ROOT/outputs/stage3_navigation_grpo}
export WAM_EXPERIMENT_NAME=${WAM_EXPERIMENT_NAME:-wnm_3d_stage3_navigation_grpo}
export WAM_RUN_LABEL="WNM-3D Stage-3 navigation GRPO: distance STOP and GN0 collision recovery"
export WAM_RUNTIME_STATE_DIR=${WAM_RUNTIME_STATE_DIR:-$WAM_OUTPUT_DIR/runtime_controls}
export WAM_VAL_PATH_CONTROL_FILE=${WAM_VAL_PATH_CONTROL_FILE:-$WAM_RUNTIME_STATE_DIR/validation.json}
export WAM_LAUNCH_OVERRIDE_PATH=$0
export WAM_REWARD_FUNCTION_PATH=$REWARD_PATH
export WAM_RESUME_MODE=disable

# Keep the validated a01f sampler and native joint-DiT update semantics.
export WAM_ACTION_NOISE_LEVEL=0.10
export WAM_ACTION_BACKBONE_GRAD_GAIN=1.0
export WAM_FLOW_ACTION_VISUAL_WEIGHT=0.05
export WAM_FLOW_ACTION_ACTION_WEIGHT=0.05

# Strong but bounded yaw guard. Path/rate consistency is translation-only;
# GT-heading consistency also observes pure-yaw actions after their rotation.
export WAM_YAW_CREDIT_ENABLED=true
export WAM_YAW_SCORE_MODE=linear_guard
export WAM_YAW_PATH_CONSISTENCY_WEIGHT=0.16
export WAM_YAW_RATE_CONSISTENCY_WEIGHT=0.10
export WAM_YAW_GROSS_GT_WEIGHT=0.08
export WAM_YAW_FREE_ANGLE_DEG=10
export WAM_YAW_PATH_HARD_ANGLE_DEG=45
export WAM_YAW_RATE_FREE_ANGLE_DEG=8
export WAM_YAW_RATE_HARD_ANGLE_DEG=35
export WAM_YAW_GT_FREE_ANGLE_DEG=12
export WAM_YAW_GROSS_ANGLE_DEG=60
export WAM_YAW_MOTION_FLOOR_M=0.03
export WAM_YAW_ROTATION_FLOOR_RAD=0.01
export WAM_YAW_SPIKE_MAX_MIX=0.50
export WAM_YAW_TOTAL_PENALTY_CAP=0.24

# Premature STOP uses geodesic distance from the executed stop position. The
# 0.25 m deadband suppresses map-boundary noise but does not make the STOP
# correct. STOP-stream magnitude grows -0.75 -> -1.00; navigation anti-evasion
# grows -0.25 -> -1.00.
export WAM_PREMATURE_STOP_PENALTY=0.75
export WAM_STOP_LOSS_WEIGHT=1.00
export WAM_PREMATURE_STOP_NAV_PENALTY_WEIGHT=0.25
export WAM_PREMATURE_STOP_DISTANCE_SCALING_ENABLED=true
export WAM_PREMATURE_STOP_DISTANCE_DEADBAND_M=0.25
export WAM_PREMATURE_STOP_DISTANCE_TAU_M=2.0
export WAM_PREMATURE_STOP_PENALTY_DISTANCE_ADD=0.25
export WAM_PREMATURE_STOP_NAV_DISTANCE_ADD=0.75

# Chunk 0 follows GN0's deployed 4 px last-free-on-segment clipping and keeps
# executing after contact. Chunks 1--3 retain counterfactual plan diagnostics
# but receive no recovery bonus because GN0 replans after eight actions.
export WAM_COLLISION_CREDIT_ENABLED=false
export WAM_COLLISION_LOSS_WEIGHT=0.0
export WAM_COLLISION_USE_CHUNK_WEIGHTS=false
export WAM_TERMINAL_SAFETY_ADVANTAGE_ENABLED=false
export WAM_COLLISION_STOP_ENABLED=false
export WAM_COLLISION_RECOVERY_ENABLED=true
export WAM_DEPLOYMENT_COLLISION_MARGIN_PX=4
export WAM_COLLISION_PENALTY_WEIGHT=0.60
export WAM_COLLISION_REPEAT_PENALTY_WEIGHT=0.10
export WAM_COLLISION_REPEAT_PENALTY_CAP_COUNT=2
export WAM_COLLISION_RECOVERY_BONUS_WEIGHT=0.10
export WAM_COLLISION_RECOVERY_CLEARANCE_PX=6
export WAM_COLLISION_RECOVERY_MIN_ESCAPE_M=0.20
export WAM_COLLISION_RECOVERY_FULL_ESCAPE_M=0.40
export WAM_COLLISION_RECOVERY_TAIL_FREE_STEPS=2
export WAM_COLLISION_RECOVERY_GRACE_ENABLED=true
export WAM_COLLISION_SOFT_ENABLED=true
export WAM_COLLISION_SOFT_MARGIN_PX=6
export WAM_COLLISION_SOFT_PENALTY_WEIGHT=0.25

test -f "$REWARD_PATH"
"${PYTHON_BIN:-python}" "$WAM_CONFIG_CONTRACT_VERIFIER"
mkdir -p "$WAM_OUTPUT_DIR/config_snapshot"
cp "$RECIPE_DIR/algorithm.env.sh" \
  "$WAM_OUTPUT_DIR/config_snapshot/algorithm.env.sh"
cp "$RECIPE_DIR/resources.env.sh" \
  "$WAM_OUTPUT_DIR/config_snapshot/resources.env.sh"
cp "$WAM_CONFIG_CONTRACT_VERIFIER" \
  "$WAM_OUTPUT_DIR/config_snapshot/verify_config.py"
cp "$REWARD_PATH" "$WAM_OUTPUT_DIR/config_snapshot/reward_function_active.py"
sha256sum "$REWARD_PATH" \
  > "$WAM_OUTPUT_DIR/config_snapshot/reward_function_active.sha256"

exec bash "$RUNTIME_DIR/train_cluster.sh" \
  "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16" \
  "$@"
