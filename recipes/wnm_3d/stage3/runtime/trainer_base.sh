#!/usr/bin/env bash
# Stage-3 official low-level one-epoch trainer wrapper.
# Safe by default: print the complete resolved configuration; launch only with --run.

set -euo pipefail

RUNTIME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(cd "$RUNTIME_DIR/.." && pwd)
REPO_ROOT=${WAM_REPO_ROOT:-$(cd "$RUNTIME_DIR/../../../.." && pwd)}
export WAM_REPO_ROOT=$REPO_ROOT
source "$RECIPE_DIR/algorithm.env.sh"
source "$RECIPE_DIR/resources.env.sh"
DATA_ROOT=${WAM_DATA_ROOT:?Set WAM_DATA_ROOT in paths.env}
MANIFEST="$DATA_ROOT/manifest.json"

export WNM3D_SOURCE_ROOT=${WNM3D_SOURCE_ROOT:?Set WNM3D_SOURCE_ROOT in paths.env}
export WNM3D_INITIAL_CHECKPOINT=${WNM3D_INITIAL_CHECKPOINT:?Set WNM3D_INITIAL_CHECKPOINT in paths.env}
export WAM_DATA_CONTRACT_CHECKPOINT=${WAM_DATA_CONTRACT_CHECKPOINT:-$WNM3D_INITIAL_CHECKPOINT}
export WNM_TOKENIZER_SOURCE=${WNM_TOKENIZER_SOURCE:?Set WNM_TOKENIZER_SOURCE in paths.env}
export WNM_TEXT_ENCODER_SOURCE=${WNM_TEXT_ENCODER_SOURCE:?Set WNM_TEXT_ENCODER_SOURCE in paths.env}
export WNM_IMAGE_ENCODER_SOURCE=${WNM_IMAGE_ENCODER_SOURCE:?Set WNM_IMAGE_ENCODER_SOURCE in paths.env}
export WNM_VAE_SOURCE=${WNM_VAE_SOURCE:?Set WNM_VAE_SOURCE in paths.env}
export WNM_VGGT_SOURCE=${WNM_VGGT_SOURCE:-}
export WNM_RUNTIME_CKPT_CACHE_NAME=${WNM_RUNTIME_CKPT_CACHE_NAME:-policy_init_checkpoint}
export WNM_RUNTIME_CKPT_EXCLUDE=${WNM_RUNTIME_CKPT_EXCLUDE:-}
export WAM_RUN_LABEL=${WAM_RUN_LABEL:-Stage-3 official low-level trainer}
export WNM_RUNTIME_CHECKPOINT="$WNM3D_INITIAL_CHECKPOINT"
export WNM_TOKENIZER_PATH="$WNM_TOKENIZER_SOURCE"
export WAM_OUTPUT_DIR=${WAM_OUTPUT_DIR:-$REPO_ROOT/outputs/default}
export WAM_EXPERIMENT_NAME=${WAM_EXPERIMENT_NAME:-wnm_3d_stage3_navigation_grpo}
export WAM_EXPECTED_TRAIN_SIZE=${WAM_EXPECTED_TRAIN_SIZE:?Set WAM_EXPECTED_TRAIN_SIZE in paths.env}
export WAM_EXPECTED_VAL_SIZE=${WAM_EXPECTED_VAL_SIZE:?Stage-3 data contract is missing WAM_EXPECTED_VAL_SIZE}
# This launcher starts fresh RL from the Stage2 checkpoint on Stage3 data.
# Never restore a VERL actor/optimizer checkpoint from the output directory.
export WAM_RESUME_MODE=${WAM_RESUME_MODE:-disable}

# Stage immutable runtime assets once into RAM. The 103 GiB DeepSpeed
# optimizer/global-step directory is not used by the HF/FSDP or rollout
# loaders. The nominal 49 GiB "tokenizer" tree also contains unused FP32
# UMT5 model shards; only its roughly 21 MiB tokenizer assets are copied.
# Subsequent launches reuse a source-signature-qualified cache.
export WAM_STAGE_RUNTIME_ASSETS_TO_RAM=true
export WAM_RAM_CACHE_ROOT=${WAM_RAM_CACHE_ROOT:-/dev/shm/wnm_3d_runtime}
export WAM_RUNTIME_CACHE_KIND=${WAM_RUNTIME_CACHE_KIND:-tmpfs}

# Bound every native CPU runtime before Ray creates any workers.  BLIS needs
# its own variable; it does not consistently honor the OpenBLAS limit.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
# OpenCV uses its own parallel backend and does not inherit the BLAS/OpenMP
# limits above.  Its FFmpeg backend has a second independent decode pool.
export OPENCV_FOR_THREADS_NUM=1
export OPENCV_FFMPEG_THREADS=1
export WAM_CPU_THREADS=1
export TOKENIZERS_PARALLELISM=false

# WNM-3D's actor and rollout adapters both validate the GN0-compatible
# FA2 path. Set it before any model package import so the environment, runtime,
# and diagnostics agree rather than reporting a dormant SDPA placeholder.
export ATTENTION_BACKEND=FA2
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export WAM_ENFORCE_SDPA=false

# Restore the original compiled execution path while keeping Inductor's
# compilation pool bounded independently of the numerical-library pools.
export TORCHDYNAMO_DISABLE=0
export TORCH_COMPILE_DISABLE=0
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
# Version the cache after changing Inductor's BF16 cast semantics. Keeping it
# separate guarantees that no kernel produced by the parity-failing run can be
# reused accidentally; the old cache remains intact for postmortem inspection.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$WAM_RAM_CACHE_ROOT/torchinductor_cache_precision_v2}
export TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export MAX_JOBS=1

# The custom WNM pipeline keeps per-request RNG and Dance-SDE state
# outside its deterministic joint DiT. First compile only repeated Causal-WAN
# blocks (the same granularity as upstream vLLM-Omni regional compilation),
# then capture fixed input signatures as eager/compiled CUDA Graphs. The
# legacy whole-_forward_train compiler is explicitly disabled because its
# accumulated BF16 error failed action-output parity.
export WAM_ROLLOUT_TORCH_COMPILE=false
# Real batch=8 parity failed for regional compilation (the accumulated BF16
# action output diverged), so training stays eager until a semantics-preserving
# model optimization is available.
export WAM_ROLLOUT_REGIONAL_COMPILE=false
export WAM_ROLLOUT_REGIONAL_COMPILE_MODE=default
export WAM_ROLLOUT_REGIONAL_COMPILE_FULLGRAPH=false
export WAM_ROLLOUT_REGIONAL_COMPILE_DYNAMIC=true
export WAM_ROLLOUT_REGIONAL_COMPILE_VERIFY=true
export WAM_ROLLOUT_REGIONAL_COMPILE_ATOL=0.002
export WAM_ROLLOUT_REGIONAL_COMPILE_RTOL=0.002
export WAM_ROLLOUT_REGIONAL_COMPILE_EMULATE_PRECISION_CASTS=true
# Whole-DiT capture is implemented and opt-in, but remains disabled for the
# training launch: batch=8 capture was bitwise-identical on its first replay,
# then PyTorch SDPA hit an unspecified launch failure on a later rollout.
# Regional block compilation is evaluated independently below.
export WAM_ROLLOUT_CUDA_GRAPH=false
export WAM_ROLLOUT_CUDA_GRAPH_VERIFY=true
export WAM_ROLLOUT_CUDA_GRAPH_ATOL=0
export WAM_ROLLOUT_CUDA_GRAPH_RTOL=0
export WAM_ROLLOUT_CUDA_GRAPH_WARMUP_ITERS=2
# Cache exact runtime signatures for request batch sizes 1/2/4/8.
export WAM_ROLLOUT_CUDA_GRAPH_MAX_ENTRIES=4
# A rollout batch repeats video/state/text and changes only its RNG seed.
# Transform one copy, expand the normalized tensors, and restore every seed.
# The first real batch is checked bit-for-bit against the legacy batch path.
export WAM_ROLLOUT_DEDUP_TRANSFORM=true
export WAM_ROLLOUT_DEDUP_TRANSFORM_VERIFY=true
export WAM_ROLLOUT_DEDUP_ENCODERS=true
export WAM_ROLLOUT_DEDUP_ENCODERS_VERIFY=true
export WAM_ROLLOUT_DEDUP_ENCODERS_ATOL=0.002
export WAM_ROLLOUT_DEDUP_ENCODERS_RTOL=0.002
# Keep the N trajectories of one prompt on the same rollout replica. This is
# required for the server-side batch to be homogeneous and makes transform/VAE
# conditioning dedup effective; distinct prompt uids remain load-balanced.
export WAM_ROLLOUT_GROUP_STICKY=true
export WAM_ROLLOUT_GROUP_SIZE="${WAM_ROLLOUT_GROUP_SIZE:-8}"
export WAM_ROLLOUT_GROUP_TIMEOUT_S=30
# Allow a burst of eight sticky requests to enter the scheduler before its
# first forward. A full batch runs immediately; only tail batches wait up to
# this bound.
export WAM_ROLLOUT_BATCH_WAIT_MS=2000
export WAM_ROLLOUT_REQUIRE_FULL_BATCH=true
export WAM_ROLLOUT_BULK_CPU_OUTPUT=true
# Actor compilation is intentionally disabled below.  A real batch=64,
# micro=16 run generated eight Dynamo variants per rank, hit
# torch._dynamo.config.recompile_limit, then fell back to eager because the
# block inputs alternate requires_grad state.  Keep these knobs documented for
# a future isolated compiler experiment, but do not pay compilation/cache costs
# in the production training path.
export WAM_ACTOR_COMPILE_MODE=default
export WAM_ACTOR_COMPILE_FULLGRAPH=false
export WAM_ACTOR_COMPILE_EMULATE_PRECISION_CASTS=true

# Streaming reward has already consumed decoded RGB before generate returns.
# Training keeps replay latents/conditions only; validation retains RGB for
# generation logging. This avoids concatenating and dispatching ~9.7 GiB of
# float32 video per step without moving decode/reward into the rollout server.
export WAM_DROP_TRAIN_RESPONSES_AFTER_REWARD=true

# Online GRPO sampling and optimization.
export WAM_NUM_GPUS=${WAM_NUM_GPUS:?resources.env.sh must resolve WAM_NUM_GPUS}
export WAM_ROLLOUT_TP=${WAM_ROLLOUT_TP:?resources.env.sh must resolve WAM_ROLLOUT_TP}
export WAM_ROLLOUT_WORKERS=${WAM_ROLLOUT_WORKERS:?resources.env.sh must resolve WAM_ROLLOUT_WORKERS}
# Request batching keeps a separate RNG stream per sample. Batch=4 reached
# only 35.9 GiB during real generation, so the next launch defaults to 8.
export WAM_ROLLOUT_MAX_NUM_SEQS="${WAM_ROLLOUT_MAX_NUM_SEQS:-8}"
export WAM_REWARD_WORKERS=${WAM_REWARD_WORKERS:?resources.env.sh must resolve WAM_REWARD_WORKERS}
# Runtime assets are RAM-staged and every native/compile pool is bounded.
# Native thread pools are capped above. Resource resolution limits each node's
# concurrent replica initialization while avoiding fully serialized cold start.
export WAM_ROLLOUT_INIT_CONCURRENCY=${WAM_ROLLOUT_INIT_CONCURRENCY:?resources.env.sh must resolve WAM_ROLLOUT_INIT_CONCURRENCY}
export WAM_RAY_NUM_CPUS=${WAM_RAY_NUM_CPUS:?resources.env.sh must resolve WAM_RAY_NUM_CPUS}
export WAM_PROMPT_BATCH_SIZE=${WAM_PROMPT_BATCH_SIZE:?resources.env.sh must resolve WAM_PROMPT_BATCH_SIZE}
export WAM_ROLLOUT_N="${WAM_ROLLOUT_N:?Stage-3 rollout contract is missing}"
# Validation generation is padded to the active rollout worker count before
# distributed old-logprob replay.
export WAM_VAL_ROLLOUT_N="${WAM_VAL_ROLLOUT_N:?Stage-3 validation rollout contract is missing}"
export WAM_PPO_MINI_BATCH_SIZE=${WAM_PPO_MINI_BATCH_SIZE:?resources.env.sh must resolve WAM_PPO_MINI_BATCH_SIZE}
export WAM_MICRO_BATCH_SIZE="${WAM_MICRO_BATCH_SIZE:?resources.env.sh must resolve WAM_MICRO_BATCH_SIZE}"
# Rollout and actor share the explicit FA2 numerical path. Use the rollout
# policy log-probabilities directly as PPO's old-policy anchor during training.
export WAM_ROLLOUT_CORRECTION_BYPASS=true
# Retain full FSDP1 parameters through backward and overlap the next all-gather.
export WAM_FSDP_RESHARD_AFTER_FORWARD=false
export WAM_FSDP_FORWARD_PREFETCH=true
export WAM_FSDP_USE_ORIG_PARAMS=true
# Zero selects the model's _no_split_modules policy.  CausalWanModel declares
# CausalWanAttentionBlock, yielding one FSDP unit per complete transformer
# block instead of recursively wrapping every >=10k-parameter leaf module.
export WAM_FSDP_MIN_NUM_PARAMS=0
export WAM_TORCH_COMPILE=false
export WAM_DATALOADER_WORKERS=0
export WAM_TOTAL_EPOCHS=${WAM_TOTAL_EPOCHS:-1}
export WAM_TOTAL_STEPS=${WAM_TOTAL_STEPS:?resources.env.sh must resolve WAM_TOTAL_STEPS}

# Optimizer and separate visual/action policy losses.
export WAM_LEARNING_RATE=${WAM_LEARNING_RATE:-1e-6}
export WAM_WEIGHT_DECAY=1e-5
export WAM_WARMUP_RATIO=0.05
export WAM_VISUAL_LOG_PROB_WEIGHT=1.0
export WAM_ACTION_LOG_PROB_WEIGHT=${WAM_ACTION_LOG_PROB_WEIGHT:-0.25}
export WAM_CLIP_RATIO=${WAM_CLIP_RATIO:-0.0001}
export WAM_VISUAL_CLIP_RATIO=${WAM_VISUAL_CLIP_RATIO:-$WAM_CLIP_RATIO}
export WAM_ACTION_CLIP_RATIO=${WAM_ACTION_CLIP_RATIO:-$WAM_CLIP_RATIO}

# Match the deployed WNM sampler's 16-transition schedule, fixed
# eight-position DiT mask, and visual-only CFG=5. The action branch remains
# conditional-only, exactly as in inference.
export WAM_NOISE_LEVEL=${WAM_NOISE_LEVEL:-0.7}
export WAM_ACTION_NOISE_LEVEL=${WAM_ACTION_NOISE_LEVEL:-0.1}
export WAM_INIT_SAME_NOISE=${WAM_INIT_SAME_NOISE:-false}
export WAM_NUM_INFERENCE_STEPS=${WAM_NUM_INFERENCE_STEPS:-16}
export WAM_TRUE_CFG_SCALE=${WAM_TRUE_CFG_SCALE:-5.0}
# The rollout server and FSDP replay actor both run in BF16.  Keep the
# fail-closed parity check enabled, but allow the observed sub-0.2% numerical
# drift between the two execution paths.
export WAM_ROLLOUT_REPLAY_ATOL=${WAM_ROLLOUT_REPLAY_ATOL:-0.002}
export WAM_ROLLOUT_REPLAY_RTOL=${WAM_ROLLOUT_REPLAY_RTOL:-0.002}
export WAM_ACTION_ROLLOUT_REPLAY_ATOL=${WAM_ACTION_ROLLOUT_REPLAY_ATOL:-0.004}
export WAM_ACTION_ROLLOUT_REPLAY_RTOL=${WAM_ACTION_ROLLOUT_REPLAY_RTOL:-0.002}
export WAM_MAX_PROMPT_LENGTH=512

# Checkpointing and validation cadence.
export WAM_SAVE_FREQ=${WAM_SAVE_FREQ:-200}
export WAM_TEST_FREQ=${WAM_TEST_FREQ:-5000}
export WAM_VAL_BEFORE_TRAIN=${WAM_VAL_BEFORE_TRAIN:-false}
export WAM_VALIDATION_MAX_MEDIA_SAMPLES=${WAM_VALIDATION_MAX_MEDIA_SAMPLES:-4}
export WAM_VALIDATION_CHECKPOINT_GUARD=${WAM_VALIDATION_CHECKPOINT_GUARD:-false}
export WAM_SEED=42
export WAM_VAL_MAX_SAMPLES=${WAM_VAL_MAX_SAMPLES:?Set WAM_VAL_MAX_SAMPLES in paths.env}

# Base action reward. Vision weights remain fixed at 0.60/0.25/0.15.
# Score only the first receding-horizon chunk (8 actually executed actions).
# OCC geodesic progress permits any free-space route to the goal; GT supplies
# only the corresponding path-length budget. Collision is a small independent
# penalty and never gates or truncates the progress reward.
export WAM_ACTION_COLLISION_CHUNK_SIZE=8
export WAM_OCCUPANCY_THRESHOLD=200
export WAM_OCCUPANCY_MARGIN_PX=${WAM_OCCUPANCY_MARGIN_PX:-2}
export WAM_GEODESIC_OCCUPANCY_MARGIN_PX=0
export WAM_GEODESIC_SNAP_RADIUS_PX=4
export WAM_SOFTSPL_WEIGHT=0.90
export WAM_GOAL_SCORE_WEIGHT=0.10
export WAM_GOAL_SCORE_TEMPERATURE_M=0.75
export WAM_PATH_EFFICIENCY_POWER=${WAM_PATH_EFFICIENCY_POWER:-1.0}
export WAM_COLLISION_PENALTY_WEIGHT=${WAM_COLLISION_PENALTY_WEIGHT:-0.10}
export WAM_ACTION_REWARD_MIN=${WAM_ACTION_REWARD_MIN:-0.0}
export WAM_ACTION_REWARD_MAX=${WAM_ACTION_REWARD_MAX:-1.0}
# Entering the 1.5m goal radius is not enough to trigger a motion penalty.
# Penalize only a subsequent crossing back outside that exact radius.
export WAM_STOP_LEAVE_HYSTERESIS=0.0
export WAM_STOP_MIN_STEP_MOTION=0.05
export WAM_STOP_MAX_TAIL_PATH=0.30
export WAM_STOP_ALLOWED_MOVING_STEPS=1
export WAM_STOP_CONTINUED_PENALTY=0.0
export WAM_STOP_LEFT_GOAL_PENALTY=${WAM_STOP_LEFT_GOAL_PENALTY:--0.50}

stage_runtime_tree() {
    local source_path
    source_path=$(readlink -f "$1")
    local label=$2
    local excluded_dir=${3:-}
    local excluded_glob=${4:-}
    local signature
    if [[ -n "$excluded_dir" ]] && [[ -n "$excluded_glob" ]]; then
        signature=$(
            {
                printf '%s\n' "$source_path"
                find "$source_path" \
                    -path "$source_path/$excluded_dir" -prune -o \
                    -type f ! -name "$excluded_glob" -printf '%P %s %T@\n'
            } | sort | sha256sum | awk '{print $1}'
        )
    elif [[ -n "$excluded_dir" ]]; then
        signature=$(
            {
                printf '%s\n' "$source_path"
                find "$source_path" \
                    -path "$source_path/$excluded_dir" -prune -o \
                    -type f -printf '%P %s %T@\n'
            } | sort | sha256sum | awk '{print $1}'
        )
    elif [[ -n "$excluded_glob" ]]; then
        signature=$(
            {
                printf '%s\n' "$source_path"
                find "$source_path" -type f ! -name "$excluded_glob" -printf '%P %s %T@\n'
            } | sort | sha256sum | awk '{print $1}'
        )
    else
        signature=$(
            {
                printf '%s\n' "$source_path"
                find "$source_path" -type f -printf '%P %s %T@\n'
            } | sort | sha256sum | awk '{print $1}'
        )
    fi

    local target="$WAM_RAM_CACHE_ROOT/${label}-${signature:0:16}"
    local marker="$target/.wam_stage_complete"
    if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$signature" ]]; then
        printf 'Reusing RAM-staged %s: %s\n' "$label" "$target" >&2
        printf '%s\n' "$target"
        return 0
    fi

    mkdir -p "$WAM_RAM_CACHE_ROOT"
    local required_bytes available_bytes
    local -a du_excludes=()
    [[ -n "$excluded_dir" ]] && du_excludes+=(--exclude="$excluded_dir")
    [[ -n "$excluded_glob" ]] && du_excludes+=(--exclude="$excluded_glob")
    required_bytes=$(du -sb "${du_excludes[@]}" "$source_path" | awk '{print $1}')
    available_bytes=$(df --output=avail -B1 "$WAM_RAM_CACHE_ROOT" | tail -n 1 | tr -d ' ')
    if (( required_bytes + 1073741824 > available_bytes )); then
        printf 'Insufficient tmpfs capacity for %s: need=%s available=%s root=%s\n' \
            "$label" "$required_bytes" "$available_bytes" "$WAM_RAM_CACHE_ROOT" >&2
        return 1
    fi

    local staging_dir
    staging_dir=$(mktemp -d "$WAM_RAM_CACHE_ROOT/.${label}.tmp.XXXXXX")
    trap 'rm -rf "$staging_dir"' RETURN
    local stage_started_at
    stage_started_at=$(date +%s)
    printf 'Staging %s into RAM once: %s -> %s\n' "$label" "$source_path" "$target" >&2
    local -a rsync_excludes=()
    [[ -n "$excluded_dir" ]] && rsync_excludes+=(--exclude="/$excluded_dir/")
    [[ -n "$excluded_glob" ]] && rsync_excludes+=(--exclude="$excluded_glob")
    rsync -a "${rsync_excludes[@]}" "$source_path/" "$staging_dir/"
    printf '%s\n' "$signature" > "$staging_dir/.wam_stage_complete"
    mv "$staging_dir" "$target"
    trap - RETURN
    printf 'Completed RAM staging for %s in %ss\n' "$label" "$(( $(date +%s) - stage_started_at ))" >&2
    printf '%s\n' "$target"
}

stage_runtime_file() {
    local source_path
    source_path=$(readlink -f "$1")
    local label=$2
    local signature
    signature=$(
        stat -Lc '%n %s %Y' "$source_path" | sha256sum | awk '{print $1}'
    )
    local target_dir="$WAM_RAM_CACHE_ROOT/${label}-${signature:0:16}"
    local target="$target_dir/$(basename "$source_path")"
    local marker="$target_dir/.wam_stage_complete"
    if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$signature" ]] && [[ -f "$target" ]]; then
        printf 'Reusing RAM-staged %s: %s\n' "$label" "$target" >&2
        printf '%s\n' "$target"
        return 0
    fi

    mkdir -p "$WAM_RAM_CACHE_ROOT"
    local required_bytes available_bytes
    required_bytes=$(stat -Lc '%s' "$source_path")
    available_bytes=$(df --output=avail -B1 "$WAM_RAM_CACHE_ROOT" | tail -n 1 | tr -d ' ')
    if (( required_bytes + 1073741824 > available_bytes )); then
        printf 'Insufficient tmpfs capacity for %s: need=%s available=%s root=%s\n' \
            "$label" "$required_bytes" "$available_bytes" "$WAM_RAM_CACHE_ROOT" >&2
        return 1
    fi

    local staging_dir stage_started_at
    staging_dir=$(mktemp -d "$WAM_RAM_CACHE_ROOT/.${label}.tmp.XXXXXX")
    trap 'rm -rf "$staging_dir"' RETURN
    stage_started_at=$(date +%s)
    printf 'Staging %s into RAM once: %s -> %s\n' "$label" "$source_path" "$target" >&2
    cp -a "$source_path" "$staging_dir/"
    printf '%s\n' "$signature" > "$staging_dir/.wam_stage_complete"
    mv "$staging_dir" "$target_dir"
    trap - RETURN
    printf 'Completed RAM staging for %s in %ss\n' "$label" "$(( $(date +%s) - stage_started_at ))" >&2
    printf '%s\n' "$target"
}

if [[ ! -f "$MANIFEST" ]]; then
    echo "Dataset conversion is not complete: missing $MANIFEST" >&2
    exit 2
fi

# A dataset may be reused with a later Stage2 initialization only when the
# decode/model schema and action normalization contract are exactly the same.
# The weights themselves are intentionally allowed to differ.
if [[ "$(readlink -f "$WAM_DATA_CONTRACT_CHECKPOINT")" != "$(readlink -f "$WNM3D_INITIAL_CHECKPOINT")" ]]; then
    "${PYTHON_BIN:-python}" - \
        "$WAM_DATA_CONTRACT_CHECKPOINT" \
        "$WNM3D_INITIAL_CHECKPOINT" <<'PY'
from pathlib import Path
import sys

contract, source = (Path(value).resolve() for value in sys.argv[1:])
required = (
    Path("config.json"),
    Path("experiment_cfg/metadata.json"),
    Path("action_normalization.json"),
)
for relative in required:
    contract_bytes = (contract / relative).read_bytes()
    source_bytes = (source / relative).read_bytes()
    if contract_bytes != source_bytes:
        raise RuntimeError(
            f"dataset/source checkpoint contract mismatch: {relative}; "
            f"dataset={contract}, source={source}"
        )
print(f"DATA_SOURCE_CHECKPOINT_COMPATIBLE dataset={contract} source={source}")
PY
fi

if ! MANIFEST_OUTPUT=$(
    "${PYTHON_BIN:-python}" "$RECIPE_DIR/verify_data_manifest.py" \
        --manifest "$MANIFEST" \
        --checkpoint "$WAM_DATA_CONTRACT_CHECKPOINT" \
        --expected-train "$WAM_EXPECTED_TRAIN_SIZE" \
        --expected-val "$WAM_EXPECTED_VAL_SIZE"
); then
    exit 2
fi
readarray -t MANIFEST_VALUES <<< "$MANIFEST_OUTPUT"
if (( ${#MANIFEST_VALUES[@]} != 3 )); then
    echo "Dataset verifier returned ${#MANIFEST_VALUES[@]} lines; expected exactly 3" >&2
    exit 2
fi
export WAM_TRAIN_FILES="${MANIFEST_VALUES[0]}"
export WAM_VAL_FILES="${MANIFEST_VALUES[1]}"
if [[ "${WAM_PROFILE_USE_TRAIN_AS_VAL:-false}" == "true" ]] && [[ "$WAM_VAL_FILES" == "[]" ]]; then
    # RLHFDataset rejects an empty val file list even when validation is
    # disabled. Profiling may reuse the train shard only to construct that
    # dormant loader; no validation rollout is scheduled in this mode.
    export WAM_VAL_FILES="$WAM_TRAIN_FILES"
fi

if [[ -n "${WAM_CONFIG_CONTRACT_VERIFIER:-}" ]]; then
    "${PYTHON_BIN:-python}" "$WAM_CONFIG_CONTRACT_VERIFIER"
    mkdir -p "$WAM_OUTPUT_DIR/config_snapshot"
    env | grep -E '^(WAM_|WNM_|TORCHINDUCTOR_|NUM_INFERENCE_STEPS=|ENABLE_DIT_CACHE=|ENABLE_CFG=|CFG_SCALE=|DYNAMIC_CACHE_SCHEDULE=)' | LC_ALL=C sort \
        > "$WAM_OUTPUT_DIR/config_snapshot/resolved_environment_final.txt"
fi

cat <<PARAMETERS
InteriorGS WAM RL $WAM_RUN_LABEL (review gate)
  source manifest:       $MANIFEST
  dataset counts:        ${MANIFEST_VALUES[2]}
  stage2 checkpoint:     $WNM_RUNTIME_CHECKPOINT
  tokenizer:             $WNM_TOKENIZER_PATH
  runtime asset staging: $WAM_STAGE_RUNTIME_ASSETS_TO_RAM ($WAM_RUNTIME_CACHE_KIND: $WAM_RAM_CACHE_ROOT)
  output:                $WAM_OUTPUT_DIR
  experiment / resume:   $WAM_EXPERIMENT_NAME / $WAM_RESUME_MODE
  GPUs / rollout workers:$WAM_NUM_GPUS / $WAM_ROLLOUT_WORKERS
  rollout TP:            $WAM_ROLLOUT_TP
  rollout request batch: $WAM_ROLLOUT_MAX_NUM_SEQS
  reward workers:        $WAM_REWARD_WORKERS
  attention actor/rollout:$ATTENTION_BACKEND / $DIFFUSION_ATTENTION_BACKEND
  rollout regional/graph: $WAM_ROLLOUT_REGIONAL_COMPILE / $WAM_ROLLOUT_CUDA_GRAPH
  graph warmup/cache:     $WAM_ROLLOUT_CUDA_GRAPH_WARMUP_ITERS / $WAM_ROLLOUT_CUDA_GRAPH_MAX_ENTRIES
  transform dedup/verify:$WAM_ROLLOUT_DEDUP_TRANSFORM / $WAM_ROLLOUT_DEDUP_TRANSFORM_VERIFY
  encoder dedup/verify:  $WAM_ROLLOUT_DEDUP_ENCODERS / $WAM_ROLLOUT_DEDUP_ENCODERS_VERIFY
  same-prompt sticky:    $WAM_ROLLOUT_GROUP_STICKY
  rollout group barrier: $WAM_ROLLOUT_GROUP_SIZE (timeout ${WAM_ROLLOUT_GROUP_TIMEOUT_S}s)
  request batch wait:    ${WAM_ROLLOUT_BATCH_WAIT_MS}ms max
  require full batch:    $WAM_ROLLOUT_REQUIRE_FULL_BATCH
  bulk CPU output:       $WAM_ROLLOUT_BULK_CPU_OUTPUT
  rollout init / Ray CPU:$WAM_ROLLOUT_INIT_CONCURRENCY / $WAM_RAY_NUM_CPUS
  prompt batch / N:      $WAM_PROMPT_BATCH_SIZE / $WAM_ROLLOUT_N
  PPO mini / micro:      $WAM_PPO_MINI_BATCH_SIZE / $WAM_MICRO_BATCH_SIZE
  rollout logprob bypass:$WAM_ROLLOUT_CORRECTION_BYPASS
  FSDP reshard/prefetch: $WAM_FSDP_RESHARD_AFTER_FORWARD / $WAM_FSDP_FORWARD_PREFETCH
  FSDP orig / compile:   $WAM_FSDP_USE_ORIG_PARAMS / $WAM_TORCH_COMPILE
  FSDP min params:       $WAM_FSDP_MIN_NUM_PARAMS (0 = model block policy)
  val samples / N:       $WAM_VAL_MAX_SAMPLES / $WAM_VAL_ROLLOUT_N
  total epochs / steps:  $WAM_TOTAL_EPOCHS / $WAM_TOTAL_STEPS
  lr / weight decay:     $WAM_LEARNING_RATE / $WAM_WEIGHT_DECAY
  visual/action weight:  $WAM_VISUAL_LOG_PROB_WEIGHT / $WAM_ACTION_LOG_PROB_WEIGHT
  visual/action clip:    $WAM_VISUAL_CLIP_RATIO / $WAM_ACTION_CLIP_RATIO
  replay visual atol/rtol:$WAM_ROLLOUT_REPLAY_ATOL / $WAM_ROLLOUT_REPLAY_RTOL
  replay action atol/rtol:$WAM_ACTION_ROLLOUT_REPLAY_ATOL / $WAM_ACTION_ROLLOUT_REPLAY_RTOL
  noise visual/action:   $WAM_NOISE_LEVEL / $WAM_ACTION_NOISE_LEVEL
  shared init noise:     $WAM_INIT_SAME_NOISE (only the credited transition RNG varies)
  inference steps:       $WAM_NUM_INFERENCE_STEPS
  save / validate freq:  $WAM_SAVE_FREQ / $WAM_TEST_FREQ
  OCC threshold/margin:  $WAM_OCCUPANCY_THRESHOLD / ${WAM_OCCUPANCY_MARGIN_PX}px
  action eval chunk:     ${WAM_ACTION_COLLISION_CHUNK_SIZE} actions
  geodesic margin/snap:  ${WAM_GEODESIC_OCCUPANCY_MARGIN_PX}px / ${WAM_GEODESIC_SNAP_RADIUS_PX}px
  SoftSPL/goal weights:  $WAM_SOFTSPL_WEIGHT / $WAM_GOAL_SCORE_WEIGHT
  overlength power:      $WAM_PATH_EFFICIENCY_POWER (1 = original SoftSPL)
  goal temperature:      ${WAM_GOAL_SCORE_TEMPERATURE_M}m
  collision penalty:     -$WAM_COLLISION_PENALTY_WEIGHT (soft; no gate/truncation)
  action reward bounds:  [$WAM_ACTION_REWARD_MIN, $WAM_ACTION_REWARD_MAX]
  stop penalties:        $WAM_STOP_CONTINUED_PENALTY / $WAM_STOP_LEFT_GOAL_PENALTY
PARAMETERS

if [[ "${1:-}" != "--run" ]]; then
    echo "Review-only mode. Pass --run after approval to launch."
    exit 0
fi
shift

if [[ "$WAM_STAGE_RUNTIME_ASSETS_TO_RAM" == "true" ]]; then
    WNM_RUNTIME_CHECKPOINT=$(stage_runtime_tree \
        "$WNM3D_INITIAL_CHECKPOINT" \
        "$WNM_RUNTIME_CKPT_CACHE_NAME" \
        "$WNM_RUNTIME_CKPT_EXCLUDE")
    WNM_TOKENIZER_PATH=$(stage_runtime_tree \
        "$WNM_TOKENIZER_SOURCE" umt5_tokenizer '' 'pytorch_model-*.bin')
    WNM_TEXT_ENCODER_PATH=$(stage_runtime_file "$WNM_TEXT_ENCODER_SOURCE" text_encoder)
    WNM_IMAGE_ENCODER_PATH=$(stage_runtime_file "$WNM_IMAGE_ENCODER_SOURCE" image_encoder)
    WNM_VAE_PATH=$(stage_runtime_file "$WNM_VAE_SOURCE" vae)
    if [[ -n "$WNM_VGGT_SOURCE" ]]; then
        WNM_VGGT_PATH=$(stage_runtime_file "$WNM_VGGT_SOURCE" vggt_omega)
    fi

    # Rewrite only the RAM-staged checkpoint metadata. The source checkpoint
    # and the WNM/GN0 source tree remain untouched.
    STAGED_CONF="$WNM_RUNTIME_CHECKPOINT/experiment_cfg/conf.yaml"
    sed -i -E \
        -e "s|^([[:space:]]*text_encoder_pretrained_path:).*|\\1 $WNM_TEXT_ENCODER_PATH|" \
        -e "s|^([[:space:]]*image_encoder_pretrained_path:).*|\\1 $WNM_IMAGE_ENCODER_PATH|" \
        -e "s|^([[:space:]]*vae_pretrained_path:).*|\\1 $WNM_VAE_PATH|" \
        "$STAGED_CONF"
    if [[ -n "${WNM_VGGT_PATH:-}" ]]; then
        sed -i -E \
            -e "s|^([[:space:]]*vggt_checkpoint_path:).*|\\1 $WNM_VGGT_PATH|" \
            "$STAGED_CONF"
    fi
    STAGED_CONFIG_JSON="$WNM_RUNTIME_CHECKPOINT/config.json"
    if [[ -f "$STAGED_CONFIG_JSON" ]]; then
        "${PYTHON_BIN:-python}" - \
            "$STAGED_CONFIG_JSON" \
            "$WNM_TEXT_ENCODER_PATH" \
            "$WNM_IMAGE_ENCODER_PATH" \
            "$WNM_VAE_PATH" \
            "${WNM_VGGT_PATH:-}" <<'PY'
import json
import os
import sys
import tempfile

path, text_encoder, image_encoder, vae, vggt = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    config = json.load(stream)
head = config["action_head_cfg"]["config"]
head["text_encoder_cfg"]["text_encoder_pretrained_path"] = text_encoder
head["image_encoder_cfg"]["image_encoder_pretrained_path"] = image_encoder
head["vae_cfg"]["vae_pretrained_path"] = vae
if vggt:
    head["vggt_checkpoint_path"] = vggt
fd, temporary = tempfile.mkstemp(prefix=".config.", suffix=".json", dir=os.path.dirname(path))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY
    fi
    export WNM_RUNTIME_CHECKPOINT WNM_TOKENIZER_PATH
    printf 'Using staged checkpoint: %s\n' "$WNM_RUNTIME_CHECKPOINT"
    printf 'Using staged tokenizer:  %s\n' "$WNM_TOKENIZER_PATH"
    printf 'Using staged text encoder: %s\n' "$WNM_TEXT_ENCODER_PATH"
    printf 'Using staged image encoder: %s\n' "$WNM_IMAGE_ENCODER_PATH"
    printf 'Using staged VAE:          %s\n' "$WNM_VAE_PATH"
    if [[ -n "${WNM_VGGT_PATH:-}" ]]; then
        printf 'Using staged VGGT-Omega:   %s\n' "$WNM_VGGT_PATH"
    fi
fi

# Multi-node launches must materialize the same signature-qualified tmpfs
# paths on every node before Ray places rollout/FSDP actors there.  This mode
# runs the normal verified staging path and exits before constructing Ray or
# starting training.
if [[ "${WAM_STAGE_ONLY:-false}" == "true" ]]; then
    printf 'Runtime asset staging completed; WAM_STAGE_ONLY=true, not launching training.\n'
    exit 0
fi

cd "$REPO_ROOT"
exec bash "$RUNTIME_DIR/trainer_command.sh" \
    data.val_max_samples="$WAM_VAL_MAX_SAMPLES" \
    ray_kwargs.ray_init.num_cpus="$WAM_RAY_NUM_CPUS" \
    trainer.val_before_train="$WAM_VAL_BEFORE_TRAIN" \
    ++trainer.validation_max_media_samples="$WAM_VALIDATION_MAX_MEDIA_SAMPLES" \
    ++trainer.validation_checkpoint_guard_enabled="$WAM_VALIDATION_CHECKPOINT_GUARD" \
    trainer.max_actor_ckpt_to_keep=5 \
    'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]' \
    'actor_rollout_ref.actor.checkpoint.load_contents=["model","optimizer","extra"]' \
    trainer.experiment_name="$WAM_EXPERIMENT_NAME" \
    trainer.resume_mode="$WAM_RESUME_MODE" \
    'trainer.logger=["console","tensorboard"]' \
    "$@"
