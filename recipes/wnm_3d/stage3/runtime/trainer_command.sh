#!/usr/bin/env bash
# DanceGRPO training for the joint WNM video/action model.
#
# "offline" in this filename describes only the static parquet task input.
# The parquet rows provide prompts, context video, state, and reward metadata;
# they do not provide behavior trajectories.  verl-omni repeats every prompt
# `rollout.n` times and vLLM-Omni samples current-policy video/action
# trajectories.  Reward and actor replay then use the standard online
# Diffusers DanceGRPO path.  No environment or collector process is involved.
#
# Required local inputs:
#   WNM3D_SOURCE_ROOT         WNM-3D source checkout
#   WNM_RUNTIME_CHECKPOINT    full local Stage-2 VLN checkpoint directory
#   WNM_TOKENIZER_PATH local tokenizer directory
#   WAM_TRAIN_FILES          training parquet path or Hydra list of paths
#   WAM_VAL_FILES            held-out validation parquet path or Hydra list
#   WAM_REWARD_FUNCTION_PATH optional local Python reward override
#
# Each parquet row must use the regular RLHF dataset schema.  In particular,
# `prompt`/`videos` carry the instruction and 66-frame context video, while
# `extra_info.rollout_extra_args` contains `instruction` and `state`.
# `reward_model.ground_truth` and other reward metadata stay in the row.

set -euo pipefail

RUNTIME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(cd "$RUNTIME_DIR/.." && pwd)
source "$RECIPE_DIR/algorithm.env.sh"
source "$RECIPE_DIR/resources.env.sh"

# WNM-3D's custom actor and rollout adapters both use and validate FA2.
export ATTENTION_BACKEND=FA2
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export WAM_ENFORCE_SDPA=false

: "${WNM3D_SOURCE_ROOT:?Set WNM3D_SOURCE_ROOT to the local WNM-3D checkout}"
: "${WNM_RUNTIME_CHECKPOINT:?Set WNM_RUNTIME_CHECKPOINT to the local full Stage-2 VLN checkpoint}"
: "${WNM_TOKENIZER_PATH:?Set WNM_TOKENIZER_PATH to the local tokenizer directory}"
: "${WAM_TRAIN_FILES:?Set WAM_TRAIN_FILES to a local training parquet path or Hydra list}"
: "${WAM_VAL_FILES:?Set WAM_VAL_FILES to a local held-out parquet path or Hydra list}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WAM_REWARD_FUNCTION_PATH=${WAM_REWARD_FUNCTION_PATH:-"$RECIPE_DIR/reward.py"}

if [[ ! -d "$WNM3D_SOURCE_ROOT" ]]; then
    echo "WNM3D_SOURCE_ROOT is not a directory: $WNM3D_SOURCE_ROOT" >&2
    exit 2
fi
if [[ ! -d "$WNM_RUNTIME_CHECKPOINT" ]]; then
    echo "WNM_RUNTIME_CHECKPOINT is not a directory: $WNM_RUNTIME_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -d "$WNM_TOKENIZER_PATH" ]]; then
    echo "WNM_TOKENIZER_PATH is not a directory: $WNM_TOKENIZER_PATH" >&2
    exit 2
fi
if [[ ! -f "$WAM_REWARD_FUNCTION_PATH" ]]; then
    echo "WAM_REWARD_FUNCTION_PATH is not a file: $WAM_REWARD_FUNCTION_PATH" >&2
    exit 2
fi

# P and N retain the standard GRPO meaning.  train_batch_size and
# ppo_mini_batch_size count prompt groups; the actor receives P*N trajectories.
WAM_PROMPT_BATCH_SIZE=${WAM_PROMPT_BATCH_SIZE:-1}
WAM_ROLLOUT_N=${WAM_ROLLOUT_N:-8}
WAM_VAL_ROLLOUT_N=${WAM_VAL_ROLLOUT_N:-$WAM_ROLLOUT_N}
WAM_PPO_MINI_BATCH_SIZE=${WAM_PPO_MINI_BATCH_SIZE:-$WAM_PROMPT_BATCH_SIZE}
WAM_MICRO_BATCH_SIZE=${WAM_MICRO_BATCH_SIZE:-1}
WAM_ROLLOUT_CORRECTION_BYPASS=${WAM_ROLLOUT_CORRECTION_BYPASS:-true}
WAM_FSDP_RESHARD_AFTER_FORWARD=${WAM_FSDP_RESHARD_AFTER_FORWARD:-true}
WAM_FSDP_FORWARD_PREFETCH=${WAM_FSDP_FORWARD_PREFETCH:-false}
WAM_FSDP_USE_ORIG_PARAMS=${WAM_FSDP_USE_ORIG_PARAMS:-true}
WAM_FSDP_MIN_NUM_PARAMS=${WAM_FSDP_MIN_NUM_PARAMS:-0}
WAM_TORCH_COMPILE=${WAM_TORCH_COMPILE:-true}
WAM_DATALOADER_WORKERS=${WAM_DATALOADER_WORKERS:-0}
WAM_NUM_GPUS=${WAM_NUM_GPUS:-8}
WAM_ROLLOUT_TP=${WAM_ROLLOUT_TP:-1}
WAM_ROLLOUT_WORKERS=${WAM_ROLLOUT_WORKERS:-$((WAM_NUM_GPUS / WAM_ROLLOUT_TP))}
WAM_ROLLOUT_MAX_NUM_SEQS=${WAM_ROLLOUT_MAX_NUM_SEQS:-8}
WAM_REWARD_WORKERS=${WAM_REWARD_WORKERS:-1}
WAM_TOTAL_STEPS=${WAM_TOTAL_STEPS:-300}
WAM_TOTAL_EPOCHS=${WAM_TOTAL_EPOCHS:-1}
WAM_NOISE_LEVEL=${WAM_NOISE_LEVEL:-0.7}
WAM_ACTION_NOISE_LEVEL=${WAM_ACTION_NOISE_LEVEL:-$WAM_NOISE_LEVEL}
WAM_INIT_SAME_NOISE=${WAM_INIT_SAME_NOISE:-false}
WAM_LEARNING_RATE=${WAM_LEARNING_RATE:-1e-6}
WAM_WEIGHT_DECAY=${WAM_WEIGHT_DECAY:-1e-5}
WAM_WARMUP_RATIO=${WAM_WARMUP_RATIO:-0.05}
WAM_VISUAL_LOG_PROB_WEIGHT=${WAM_VISUAL_LOG_PROB_WEIGHT:-1.0}
WAM_ACTION_LOG_PROB_WEIGHT=${WAM_ACTION_LOG_PROB_WEIGHT:-0.25}
WAM_CLIP_RATIO=${WAM_CLIP_RATIO:-0.0001}
WAM_ROLLOUT_REPLAY_ATOL=${WAM_ROLLOUT_REPLAY_ATOL:-0.001}
WAM_ROLLOUT_REPLAY_RTOL=${WAM_ROLLOUT_REPLAY_RTOL:-0.001}
WAM_ACTION_ROLLOUT_REPLAY_ATOL=${WAM_ACTION_ROLLOUT_REPLAY_ATOL:-$WAM_ROLLOUT_REPLAY_ATOL}
WAM_ACTION_ROLLOUT_REPLAY_RTOL=${WAM_ACTION_ROLLOUT_REPLAY_RTOL:-$WAM_ROLLOUT_REPLAY_RTOL}
WAM_MAX_PROMPT_LENGTH=${WAM_MAX_PROMPT_LENGTH:-512}
WAM_NUM_INFERENCE_STEPS=${WAM_NUM_INFERENCE_STEPS:-16}
WAM_TRUE_CFG_SCALE=${WAM_TRUE_CFG_SCALE:-5.0}
WAM_REWARD_FUNCTION_NAME=${WAM_REWARD_FUNCTION_NAME:-compute_score}
WAM_MODEL_ARCHITECTURE=${WAM_MODEL_ARCHITECTURE:-WNM3D}
WAM_OUTPUT_DIR=${WAM_OUTPUT_DIR:-checkpoints/wnm_3d_dance_grpo}
WAM_SAVE_FREQ=${WAM_SAVE_FREQ:-50}
WAM_TEST_FREQ=${WAM_TEST_FREQ:-50}
WAM_SEED=${WAM_SEED:-42}
WAM_CPU_THREADS=${WAM_CPU_THREADS:-1}

if (( WAM_ROLLOUT_TP <= 0 || WAM_NUM_GPUS % WAM_ROLLOUT_TP != 0 )); then
    echo "WAM_ROLLOUT_TP must be positive and divide WAM_NUM_GPUS" >&2
    exit 2
fi
if (( WAM_ROLLOUT_MAX_NUM_SEQS <= 0 )); then
    echo "WAM_ROLLOUT_MAX_NUM_SEQS must be positive" >&2
    exit 2
fi
if (( WAM_NUM_INFERENCE_STEPS != 16 )); then
    echo "WNM deployed-sampler parity requires 16 inference steps; got $WAM_NUM_INFERENCE_STEPS" >&2
    exit 2
fi
if [[ "$WAM_TRUE_CFG_SCALE" != "5" && "$WAM_TRUE_CFG_SCALE" != "5.0" ]]; then
    echo "WNM deployed-sampler parity requires WAM_TRUE_CFG_SCALE=5.0; got $WAM_TRUE_CFG_SCALE" >&2
    exit 2
fi
if [[ "$WAM_ROLLOUT_CORRECTION_BYPASS" != "true" && "$WAM_ROLLOUT_CORRECTION_BYPASS" != "false" ]]; then
    echo "WAM_ROLLOUT_CORRECTION_BYPASS must be true or false; got $WAM_ROLLOUT_CORRECTION_BYPASS" >&2
    exit 2
fi
if [[ "$WAM_INIT_SAME_NOISE" != "true" && "$WAM_INIT_SAME_NOISE" != "false" ]]; then
    echo "WAM_INIT_SAME_NOISE must be true or false; got $WAM_INIT_SAME_NOISE" >&2
    exit 2
fi

# Fail closed on every Hugging Face code path.  All model, tokenizer, reward,
# and dataset assets must already be present on this machine.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
# Ray starts several actor, rollout, reward, and vLLM subprocesses per GPU.
# BLAS/OpenMP otherwise sizes a thread pool from the host CPU count in every
# process and can exhaust the container pids cgroup before NCCL starts its
# watchdog thread (EAGAIN / "Resource temporarily unavailable").
export OMP_NUM_THREADS="$WAM_CPU_THREADS"
export MKL_NUM_THREADS="$WAM_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$WAM_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$WAM_CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$WAM_CPU_THREADS"
export RAYON_NUM_THREADS="$WAM_CPU_THREADS"
# Colocated rollout servers otherwise perform concurrent Inductor warm-ups
# and can exhaust a node's cgroup pids/thread quota before NCCL and TensorBoard
# create their control threads. Eager execution preserves rollout/training
# numerics across both single-node and multi-node layouts.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"
export WNM_ROLLOUT_TOKENIZER_PATH="$WNM_TOKENIZER_PATH"
export NUM_INFERENCE_STEPS="$WAM_NUM_INFERENCE_STEPS"
export ENABLE_DIT_CACHE=true
export DYNAMIC_CACHE_SCHEDULE=false
export ENABLE_CFG=true
export CFG_SCALE="$WAM_TRUE_CFG_SCALE"
export PYTHONPATH="${WNM3D_SOURCE_ROOT}:${PYTHONPATH:-}"

# These two generic VeRL/vLLM config fields remain a matched placeholder;
# WNM3D configures and validates its custom CausalWan FA2 modules.
"${PYTHON_BIN:-python}" -m verl_omni.trainer.main_diffusion \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    algorithm.adv_estimator=dance_grpo \
    algorithm.rollout_correction.bypass_mode="$WAM_ROLLOUT_CORRECTION_BYPASS" \
    algorithm.rollout_log_prob_validation.enabled=true \
    algorithm.rollout_log_prob_validation.validate=true \
    algorithm.rollout_log_prob_validation.atol="$WAM_ROLLOUT_REPLAY_ATOL" \
    algorithm.rollout_log_prob_validation.rtol="$WAM_ROLLOUT_REPLAY_RTOL" \
    algorithm.rollout_log_prob_validation.action_atol="$WAM_ACTION_ROLLOUT_REPLAY_ATOL" \
    algorithm.rollout_log_prob_validation.action_rtol="$WAM_ACTION_ROLLOUT_REPLAY_RTOL" \
    "data.train_files=${WAM_TRAIN_FILES}" \
    "data.val_files=${WAM_VAL_FILES}" \
    data.train_batch_size="$WAM_PROMPT_BATCH_SIZE" \
    data.val_batch_size="$WAM_PROMPT_BATCH_SIZE" \
    data.max_prompt_length="$WAM_MAX_PROMPT_LENGTH" \
    data.shuffle=true \
    data.seed="$WAM_SEED" \
    data.validation_shuffle=false \
    data.dataloader_num_workers="$WAM_DATALOADER_WORKERS" \
    actor_rollout_ref.model.path="$WNM_RUNTIME_CHECKPOINT" \
    actor_rollout_ref.model.architecture="$WAM_MODEL_ARCHITECTURE" \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.model.tokenizer_path="$WNM_TOKENIZER_PATH" \
    actor_rollout_ref.model.load_tokenizer=true \
    actor_rollout_ref.model.local_files_only=true \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="$WAM_PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$WAM_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio="$WAM_CLIP_RATIO" \
    actor_rollout_ref.actor.diffusion_loss.visual_clip_ratio="$WAM_VISUAL_CLIP_RATIO" \
    actor_rollout_ref.actor.diffusion_loss.action_clip_ratio="$WAM_ACTION_CLIP_RATIO" \
    actor_rollout_ref.actor.diffusion_loss.visual_log_prob_weight="$WAM_VISUAL_LOG_PROB_WEIGHT" \
    actor_rollout_ref.actor.diffusion_loss.action_log_prob_weight="$WAM_ACTION_LOG_PROB_WEIGHT" \
    actor_rollout_ref.actor.optim.lr="$WAM_LEARNING_RATE" \
    actor_rollout_ref.actor.optim.weight_decay="$WAM_WEIGHT_DECAY" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$WAM_WARMUP_RATIO" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward="$WAM_FSDP_RESHARD_AFTER_FORWARD" \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch="$WAM_FSDP_FORWARD_PREFETCH" \
    actor_rollout_ref.actor.fsdp_config.use_orig_params="$WAM_FSDP_USE_ORIG_PARAMS" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="$WAM_TORCH_COMPILE" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params="$WAM_FSDP_MIN_NUM_PARAMS" \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n="$WAM_ROLLOUT_N" \
    actor_rollout_ref.rollout.seed="$WAM_SEED" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$WAM_ROLLOUT_TP" \
    actor_rollout_ref.rollout.max_num_seqs="$WAM_ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.agent.num_workers="$WAM_ROLLOUT_WORKERS" \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$WAM_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.pipeline.height=160 \
    actor_rollout_ref.rollout.pipeline.width=320 \
    actor_rollout_ref.rollout.pipeline.num_frames=33 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps="$WAM_NUM_INFERENCE_STEPS" \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale="$WAM_TRUE_CFG_SCALE" \
    actor_rollout_ref.rollout.pipeline.max_sequence_length="$WAM_MAX_PROMPT_LENGTH" \
    actor_rollout_ref.rollout.algo.noise_level="$WAM_NOISE_LEVEL" \
    actor_rollout_ref.rollout.algo.action_noise_level="$WAM_ACTION_NOISE_LEVEL" \
    actor_rollout_ref.rollout.algo.sde_type=dance_sde \
    actor_rollout_ref.rollout.algo.action_sde_type=dance_sde \
    actor_rollout_ref.rollout.algo.init_same_noise="$WAM_INIT_SAME_NOISE" \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.val_kwargs.n="$WAM_VAL_ROLLOUT_N" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps="$WAM_NUM_INFERENCE_STEPS" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale="$WAM_TRUE_CFG_SCALE" \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level="$WAM_NOISE_LEVEL" \
    actor_rollout_ref.rollout.val_kwargs.algo.action_noise_level="$WAM_ACTION_NOISE_LEVEL" \
    actor_rollout_ref.rollout.val_kwargs.algo.sde_type=dance_sde \
    actor_rollout_ref.rollout.val_kwargs.algo.action_sde_type=dance_sde \
    actor_rollout_ref.rollout.val_kwargs.algo.init_same_noise="$WAM_INIT_SAME_NOISE" \
    reward.num_workers="$WAM_REWARD_WORKERS" \
    reward.reward_manager.name=WAMRewardManager \
    reward.reward_model.enable=false \
    reward.custom_reward_function.path="$WAM_REWARD_FUNCTION_PATH" \
    reward.custom_reward_function.name="$WAM_REWARD_FUNCTION_NAME" \
    trainer.logger='["console"]' \
    trainer.project_name=wnm_3d \
    trainer.experiment_name="$WAM_EXPERIMENT_NAME" \
    trainer.log_val_generations=1 \
    trainer.val_before_train=true \
    trainer.test_freq="$WAM_TEST_FREQ" \
    trainer.save_freq="$WAM_SAVE_FREQ" \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$WAM_OUTPUT_DIR" \
    trainer.n_gpus_per_node="$WAM_NUM_GPUS" \
    trainer.nnodes="$WAM_NNODES" \
    trainer.total_epochs="$WAM_TOTAL_EPOCHS" \
    trainer.total_training_steps="$WAM_TOTAL_STEPS" \
    "$@"
