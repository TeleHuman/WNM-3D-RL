#!/usr/bin/env bash
# Stage-3 official navigation/STOP/yaw/collision reward launch.
set -euo pipefail

RUNTIME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(cd "$RUNTIME_DIR/.." && pwd)
REPO_ROOT=${WAM_REPO_ROOT:-$(cd "$RUNTIME_DIR/../../../.." && pwd)}
export WAM_REPO_ROOT=$REPO_ROOT
source "$RECIPE_DIR/algorithm.env.sh"
source "$RECIPE_DIR/resources.env.sh"

export WAM_DATA_ROOT=${WAM_DATA_ROOT:?Set WAM_DATA_ROOT in paths.env}
export WAM_OUTPUT_DIR=${WAM_OUTPUT_DIR:-$REPO_ROOT/outputs/stage3_navigation_grpo}
export WAM_EXPERIMENT_NAME=${WAM_EXPERIMENT_NAME:-wnm_3d_stage3_navigation_grpo}
export WAM_EXPECTED_TRAIN_SIZE=${WAM_EXPECTED_TRAIN_SIZE:?Set WAM_EXPECTED_TRAIN_SIZE in paths.env}
export WAM_EXPECTED_VAL_SIZE=${WAM_EXPECTED_VAL_SIZE:?Stage-3 data contract is missing WAM_EXPECTED_VAL_SIZE}
export WAM_VAL_MAX_SAMPLES=${WAM_VAL_MAX_SAMPLES:?Set WAM_VAL_MAX_SAMPLES in paths.env}
export WAM_TOTAL_EPOCHS=${WAM_TOTAL_EPOCHS:-1}
export WAM_TOTAL_STEPS=${WAM_TOTAL_STEPS:?resources.env.sh must resolve WAM_TOTAL_STEPS}
export WAM_RESUME_MODE=${WAM_RESUME_MODE:-disable}
export WAM_RUN_LABEL=${WAM_RUN_LABEL:-"Stage-3 official navigation, STOP, yaw, collision, and recovery credit"}

export WNM3D_INITIAL_CHECKPOINT=${WNM3D_INITIAL_CHECKPOINT:?Set WNM3D_INITIAL_CHECKPOINT in paths.env}
export WNM_RUNTIME_CKPT_CACHE_NAME=${WNM_RUNTIME_CKPT_CACHE_NAME:-policy_init_checkpoint}
export WNM_RUNTIME_CKPT_EXCLUDE=${WNM_RUNTIME_CKPT_EXCLUDE:-}
export WAM_RAM_CACHE_ROOT=${WAM_RAM_CACHE_ROOT:-/tmp/wnm_3d_runtime_stage3}
export WAM_RUNTIME_CACHE_KIND=${WAM_RUNTIME_CACHE_KIND:-local-nvme-page-cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$WAM_RAM_CACHE_ROOT/torchinductor_cache_precision_v2}

export WAM_NOISE_LEVEL=${WAM_NOISE_LEVEL:-0.7}
# The outer Stage-3 wrapper is authoritative. Keeping this inheritable avoids
# silently changing the validated 0.10 action SDE noise back to the old 0.20.
export WAM_ACTION_NOISE_LEVEL=${WAM_ACTION_NOISE_LEVEL:-0.10}
export WAM_INIT_SAME_NOISE=${WAM_INIT_SAME_NOISE:-true}
export WAM_NUM_INFERENCE_STEPS=${WAM_NUM_INFERENCE_STEPS:-16}
export WAM_TRUE_CFG_SCALE=${WAM_TRUE_CFG_SCALE:-5.0}
export WAM_LEARNING_RATE=${WAM_LEARNING_RATE:-5e-6}
export WAM_ACTION_LOG_PROB_WEIGHT=${WAM_ACTION_LOG_PROB_WEIGHT:-0.25}
export WAM_ROLLOUT_N=${WAM_ROLLOUT_N:?Stage-3 rollout contract is missing}
export WAM_VAL_ROLLOUT_N=${WAM_VAL_ROLLOUT_N:-8}
export WAM_ROLLOUT_GROUP_SIZE=${WAM_ROLLOUT_GROUP_SIZE:-8}
export WAM_ROLLOUT_MAX_NUM_SEQS="${WAM_ROLLOUT_MAX_NUM_SEQS:-8}"
export WAM_VISUAL_CLIP_RATIO=${WAM_VISUAL_CLIP_RATIO:-0.0005}
export WAM_ACTION_CLIP_RATIO=${WAM_ACTION_CLIP_RATIO:-0.01}
export WAM_SAVE_FREQ=${WAM_SAVE_FREQ:-200}
export WAM_ACTION_BACKBONE_GRAD_GAIN=${WAM_ACTION_BACKBONE_GRAD_GAIN:-1.0}
export WAM_FLOW_ACTION_VISUAL_WEIGHT=${WAM_FLOW_ACTION_VISUAL_WEIGHT:-0.05}
export WAM_FLOW_ACTION_ACTION_WEIGHT=${WAM_FLOW_ACTION_ACTION_WEIGHT:-0.05}
export WAM_FLOW_ACTION_CALIBRATION_PATH="$RECIPE_DIR/flow_action_calibration.json"
export WAM_PATH_EFFICIENCY_POWER=2.0
export WAM_COLLISION_PENALTY_WEIGHT=${WAM_COLLISION_PENALTY_WEIGHT:-0.40}
export WAM_ACTION_REWARD_MIN=-1.0
export WAM_ACTION_REWARD_MAX=1.0

export WAM_ACTION_CHUNK_CREDIT_ENABLED=true
export WAM_ACTION_CHUNK_SIZE=8
export WAM_ACTION_CHUNK_WEIGHTS=1,0.5,0.25,0.125
export WAM_ACTION_CHUNK_REWARD_MODE=signed_progress_length
export WAM_SIGNED_PROGRESS_WEIGHT=0.70
export WAM_SYMMETRIC_LENGTH_WEIGHT=0.15
export WAM_SIGNED_GOAL_WEIGHT=0.15
export WAM_SIGNED_PROGRESS_DENOM_FLOOR_M=0.50
export WAM_SYMMETRIC_LENGTH_SCALE=0.50
export WAM_SYMMETRIC_LENGTH_FLOOR_M=0.10
export WAM_ROUTE_DEVIATION_WEIGHT=0.35
export WAM_ROUTE_DEVIATION_FREE_RADIUS_M=0.75
export WAM_ROUTE_DEVIATION_SCALE_M=1.50
export WAM_REVERSE_DIRECTION_WEIGHT=${WAM_REVERSE_DIRECTION_WEIGHT:-0.30}
export WAM_ROUTE_MOTION_FLOOR_M=0.03

# The 32 actions in one model response share one request/chunk-start frame.
# Credit chunks do not reset heading and do not rotate later dx/dy. The
# cumulative yaw below is used only to compare action yaw with path curvature.
export WAM_YAW_CREDIT_ENABLED=${WAM_YAW_CREDIT_ENABLED:-true}
export WAM_YAW_PATH_CONSISTENCY_WEIGHT=${WAM_YAW_PATH_CONSISTENCY_WEIGHT:-0.12}
export WAM_YAW_RATE_CONSISTENCY_WEIGHT=${WAM_YAW_RATE_CONSISTENCY_WEIGHT:-0.04}
export WAM_YAW_GROSS_GT_WEIGHT=${WAM_YAW_GROSS_GT_WEIGHT:-0.08}
export WAM_YAW_FREE_ANGLE_DEG=${WAM_YAW_FREE_ANGLE_DEG:-15}
export WAM_YAW_GROSS_ANGLE_DEG=${WAM_YAW_GROSS_ANGLE_DEG:-90}
export WAM_YAW_MOTION_FLOOR_M=${WAM_YAW_MOTION_FLOOR_M:-0.03}

# A colliding action is not executed; later positions remain at the last legal
# waypoint. Only the first-collision chunk receives collision credit.
export WAM_COLLISION_STOP_ENABLED=${WAM_COLLISION_STOP_ENABLED:-true}
export WAM_OCCUPANCY_MARGIN_PX=${WAM_OCCUPANCY_MARGIN_PX:-2}

# Navigation credit ends at first entry into the 1.5 m goal region. There is
# no goal-center potential inside the region: any further translation/yaw is
# pure control effort, while GN0 STOP is supervised as a separate event.
export WAM_STOP_WELL_ENABLED=true
export WAM_STOP_WELL_ENERGY_WEIGHT=${WAM_STOP_WELL_ENERGY_WEIGHT:-0.20}
export WAM_STOP_WELL_XY_DEADZONE_M=0.00
export WAM_STOP_WELL_YAW_WEIGHT=0.18
export WAM_STOP_WELL_YAW_DEADZONE_RAD=0.02
export WAM_STOP_WELL_YAW_SCALE_RAD=0.25
export WAM_STOP_REWARD_MIN=-1.0
export WAM_STOP_REWARD_MAX=1.0
export WAM_STOP_LOSS_WEIGHT=${WAM_STOP_LOSS_WEIGHT:-0.50}
export WAM_STOP_LEFT_GOAL_PENALTY=-1.00
export WAM_STOP_CONTINUED_PENALTY=0.0
export WAM_LAYER_CONDITIONED_CREDIT_ENABLED=${WAM_LAYER_CONDITIONED_CREDIT_ENABLED:-true}
export WAM_LAYER_CONDITIONED_BRANCHES_PER_STRATUM=${WAM_LAYER_CONDITIONED_BRANCHES_PER_STRATUM:-2}

"${PYTHON_BIN:-python}" "$WAM_CONFIG_CONTRACT_VERIFIER"
mkdir -p "$WAM_OUTPUT_DIR/config_snapshot"
cp "$0" "$WAM_OUTPUT_DIR/config_snapshot/launch.sh"
if [[ -n "${WAM_LAUNCH_OVERRIDE_PATH:-}" ]]; then
  cp "$WAM_LAUNCH_OVERRIDE_PATH" "$WAM_OUTPUT_DIR/config_snapshot/launch_override.sh"
fi
cp "$RUNTIME_DIR/trainer_base.sh" "$WAM_OUTPUT_DIR/config_snapshot/base_launch.sh"
for relative_path in \
  verl_omni/utils/reward_score/wam_stage3_action.py \
  verl_omni/utils/reward_score/wam_stage3_collision.py \
  verl_omni/utils/reward_score/wam_stage3_metrics.py \
  verl_omni/utils/reward_score/wam_stage3_stop.py \
  verl_omni/utils/action_chunk_credit.py \
  verl_omni/utils/reward_score/wam_navigation_reward.py \
  verl_omni/utils/reward_score/wam_vision_reward.py \
  verl_omni/utils/reward_score/wam_flow_action_reward.py \
  verl_omni/utils/reward_score/wam_terminal_stop_penalty.py \
  recipes/wnm_3d/stage3/verify_data_manifest.py \
  verl_omni/pipelines/schedulers/flow_match_sde.py \
  verl_omni/pipelines/model_base.py \
  verl_omni/pipelines/wnm_shared/action_gradient_gain.py \
  verl_omni/pipelines/wnm_shared/ar_diffusion_runner_adapter.py \
  verl_omni/pipelines/wnm_shared/batch1_equivalent.py \
  verl_omni/pipelines/wnm_shared/rollout_acceleration.py \
  verl_omni/pipelines/wnm_shared/rollout_batching.py \
  verl_omni/pipelines/wnm_shared/rollout_common.py \
  verl_omni/pipelines/wnm_shared/rollout_rng.py \
  verl_omni/pipelines/wnm_shared/wam_dance_sde.py \
  verl_omni/pipelines/wnm_3d/diffusers_training_adapter.py \
  verl_omni/pipelines/wnm_3d/vllm_omni_rollout_adapter.py \
  verl_omni/pipelines/wnm_2d/diffusers_training_adapter.py \
  verl_omni/pipelines/wnm_2d/vllm_omni_rollout_adapter.py \
  verl_omni/trainer/diffusion/diffusion_algos.py \
  verl_omni/trainer/diffusion/diffusion_metric_utils.py \
  verl_omni/trainer/diffusion/ray_diffusion_trainer.py \
  verl_omni/trainer/diffusion/rollout_correction.py \
  verl_omni/trainer/config/algorithm.py \
  verl_omni/trainer/main_diffusion.py \
  verl_omni/agent_loop/diffusion_agent_loop.py \
  verl_omni/workers/config/diffusion/actor.py \
  verl_omni/workers/engine/fsdp/diffusers_impl.py
do
  snapshot_path="$WAM_OUTPUT_DIR/config_snapshot/source/$relative_path"
  mkdir -p "$(dirname "$snapshot_path")"
  cp "$REPO_ROOT/$relative_path" "$snapshot_path"
done
cp "$WAM_FLOW_ACTION_CALIBRATION_PATH" "$WAM_OUTPUT_DIR/config_snapshot/wam_flow_action_calibration.json"
if [[ -n "${WAM_REWARD_FUNCTION_PATH:-}" ]]; then
  cp "$WAM_REWARD_FUNCTION_PATH" "$WAM_OUTPUT_DIR/config_snapshot/reward_function_active.py"
  sha256sum "$WAM_REWARD_FUNCTION_PATH" \
    > "$WAM_OUTPUT_DIR/config_snapshot/reward_function_active.sha256"
fi
env | grep -E '^(WAM_|WNM_|TORCHINDUCTOR_|NUM_INFERENCE_STEPS=|ENABLE_DIT_CACHE=|ENABLE_CFG=|CFG_SCALE=|DYNAMIC_CACHE_SCHEDULE=)' | LC_ALL=C sort \
  > "$WAM_OUTPUT_DIR/config_snapshot/resolved_environment.txt"

cat <<PARAMETERS
Stage-3 official reward parameters
  request coordinates:      all 32 dx/dy use one chunk-start frame
  credit chunk weights:     $WAM_ACTION_CHUNK_WEIGHTS
  yaw path/rate weights:    $WAM_YAW_PATH_CONSISTENCY_WEIGHT / $WAM_YAW_RATE_CONSISTENCY_WEIGHT
  gross GT-yaw weight:      $WAM_YAW_GROSS_GT_WEIGHT
  yaw free/gross angles:    ${WAM_YAW_FREE_ANGLE_DEG}deg / ${WAM_YAW_GROSS_ANGLE_DEG}deg
  yaw motion floor:         ${WAM_YAW_MOTION_FLOOR_M}m
  stop yaw weight:          $WAM_STOP_WELL_YAW_WEIGHT
  stop energy / PG weight:  $WAM_STOP_WELL_ENERGY_WEIGHT / $WAM_STOP_LOSS_WEIGHT
  collision penalty:        first=-$WAM_COLLISION_PENALTY_WEIGHT; repeat=-${WAM_COLLISION_REPEAT_PENALTY_WEIGHT:-0.0}; recovery=+${WAM_COLLISION_RECOVERY_BONUS_WEIGHT:-0.0}
  collision execution:      recovery=${WAM_COLLISION_RECOVERY_ENABLED:-false}; margin=${WAM_DEPLOYMENT_COLLISION_MARGIN_PX:-4}px; first-hit freeze=${WAM_COLLISION_STOP_ENABLED:-true}
  route/reverse weights:    $WAM_ROUTE_DEVIATION_WEIGHT / $WAM_REVERSE_DIRECTION_WEIGHT
  goal potential delta:     ${WAM_GOAL_SCORE_USE_POTENTIAL_DELTA:-false}
  one-shot arrival bonus:   ${WAM_GOAL_ENTRY_BONUS:-0.0}
  layer-conditioned credit: $WAM_LAYER_CONDITIONED_CREDIT_ENABLED
PARAMETERS

exec bash "$RUNTIME_DIR/trainer_base.sh" \
  "$@" \
  "++algorithm.layer_conditioned_credit.enabled=$WAM_LAYER_CONDITIONED_CREDIT_ENABLED" \
  "++algorithm.layer_conditioned_credit.branches_per_stratum=$WAM_LAYER_CONDITIONED_BRANCHES_PER_STRATUM" \
  "++actor_rollout_ref.actor.diffusion_loss.action_stop_loss_weight=$WAM_STOP_LOSS_WEIGHT" \
  "++actor_rollout_ref.actor.diffusion_loss.action_stop_use_chunk_weights=${WAM_STOP_USE_CHUNK_WEIGHTS:-false}"
