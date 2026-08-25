#!/usr/bin/env bash
# Resolve machine topology without changing Stage-3's rollout semantics.
# Source algorithm.env.sh before this file.

_wam_require_positive_integer() {
    local name=$1
    local value=${!name:-}
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer; got ${value:-<empty>}" >&2
        return 2
    fi
}

_wam_gcd() {
    local left=$1
    local right=$2
    local remainder
    while (( right != 0 )); do
        remainder=$((left % right))
        left=$right
        right=$remainder
    done
    printf '%d\n' "$left"
}

export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CPUS_PER_NODE=${CPUS_PER_NODE:-32}

for key in NNODES GPUS_PER_NODE CPUS_PER_NODE; do
    _wam_require_positive_integer "$key" || return 2 2>/dev/null || exit 2
done
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "NODE_RANK must lie in [0, NNODES); got rank=$NODE_RANK nodes=$NNODES" >&2
    return 2 2>/dev/null || exit 2
fi

if (( NNODES == 1 )); then
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
elif [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR is required when NNODES > 1" >&2
    return 2 2>/dev/null || exit 2
fi

export WAM_NNODES=$NNODES
export WAM_NUM_GPUS=$GPUS_PER_NODE
export WAM_RAY_NUM_CPUS=$CPUS_PER_NODE
export WAM_ROLLOUT_TP=${WAM_ROLLOUT_TP:-1}
_wam_require_positive_integer WAM_ROLLOUT_TP || return 2 2>/dev/null || exit 2

total_gpus=$((NNODES * GPUS_PER_NODE))
if (( total_gpus % WAM_ROLLOUT_TP != 0 )); then
    echo "Total GPUs ($total_gpus) must be divisible by WAM_ROLLOUT_TP=$WAM_ROLLOUT_TP" >&2
    return 2 2>/dev/null || exit 2
fi

# Keep the paper algorithm fixed: four strata and four branches per stratum.
if [[ "${WAM_ROLLOUT_N:-16}" != "16" ]]; then
    echo "Stage-3 requires WAM_ROLLOUT_N=16" >&2
    return 2 2>/dev/null || exit 2
fi
export WAM_ROLLOUT_N=16
export WAM_ROLLOUT_GROUP_SIZE=16
export WAM_ROLLOUT_GROUP_STICKY=true
export WAM_LAYER_CONDITIONED_BRANCHES_PER_STRATUM=4

export WAM_ROLLOUT_WORKERS=$((total_gpus / WAM_ROLLOUT_TP))
export WAM_REWARD_WORKERS=$total_gpus
default_init_concurrency=$(((WAM_ROLLOUT_WORKERS + 1) / 2))
if (( default_init_concurrency > GPUS_PER_NODE )); then
    default_init_concurrency=$GPUS_PER_NODE
fi
export WAM_ROLLOUT_INIT_CONCURRENCY=$default_init_concurrency

# Actor trajectories per optimizer step are prompt_batch * rollout.n. Round
# the 64-prompt reference batch upward only when a new world size requires it.
base_prompt_batch=${WAM_BASE_PROMPT_BATCH_SIZE:-64}
_wam_require_positive_integer base_prompt_batch || return 2 2>/dev/null || exit 2
prompt_divisor=$((total_gpus / $(_wam_gcd "$total_gpus" "$WAM_ROLLOUT_N")))
if [[ -z "${WAM_PROMPT_BATCH_SIZE:-}" ]]; then
    export WAM_PROMPT_BATCH_SIZE=$(
        printf '%d\n' "$((((base_prompt_batch + prompt_divisor - 1) / prompt_divisor) * prompt_divisor))"
    )
fi
_wam_require_positive_integer WAM_PROMPT_BATCH_SIZE || return 2 2>/dev/null || exit 2

export WAM_PPO_MINI_BATCH_SIZE=${WAM_PPO_MINI_BATCH_SIZE:-$WAM_PROMPT_BATCH_SIZE}
_wam_require_positive_integer WAM_PPO_MINI_BATCH_SIZE || return 2 2>/dev/null || exit 2
if (( WAM_PROMPT_BATCH_SIZE % WAM_PPO_MINI_BATCH_SIZE != 0 )); then
    echo "WAM_PPO_MINI_BATCH_SIZE must divide WAM_PROMPT_BATCH_SIZE" >&2
    return 2 2>/dev/null || exit 2
fi

actor_mini_trajectories=$((WAM_PPO_MINI_BATCH_SIZE * WAM_ROLLOUT_N))
if (( actor_mini_trajectories % total_gpus != 0 )); then
    echo "PPO mini trajectories ($actor_mini_trajectories) must divide across $total_gpus GPUs" >&2
    return 2 2>/dev/null || exit 2
fi
actor_mini_per_gpu=$((actor_mini_trajectories / total_gpus))

if [[ -z "${WAM_MICRO_BATCH_SIZE:-}" ]]; then
    micro_limit=$((actor_mini_per_gpu < 16 ? actor_mini_per_gpu : 16))
    for ((candidate = micro_limit; candidate >= 1; candidate--)); do
        if (( actor_mini_per_gpu % candidate == 0 )); then
            export WAM_MICRO_BATCH_SIZE=$candidate
            break
        fi
    done
fi
_wam_require_positive_integer WAM_MICRO_BATCH_SIZE || return 2 2>/dev/null || exit 2
if (( actor_mini_per_gpu % WAM_MICRO_BATCH_SIZE != 0 )); then
    echo "WAM_MICRO_BATCH_SIZE=$WAM_MICRO_BATCH_SIZE must divide per-GPU actor mini-batch $actor_mini_per_gpu" >&2
    return 2 2>/dev/null || exit 2
fi

expected_train_size=${WAM_EXPECTED_TRAIN_SIZE:-}
_wam_require_positive_integer expected_train_size || return 2 2>/dev/null || exit 2
if (( expected_train_size % WAM_PROMPT_BATCH_SIZE != 0 )); then
    echo "WAM_EXPECTED_TRAIN_SIZE=$expected_train_size must be divisible by WAM_PROMPT_BATCH_SIZE=$WAM_PROMPT_BATCH_SIZE" >&2
    return 2 2>/dev/null || exit 2
fi
resolved_total_steps=$((expected_train_size / WAM_PROMPT_BATCH_SIZE))
if [[ -n "${WAM_TOTAL_STEPS:-}" && "$WAM_TOTAL_STEPS" != "$resolved_total_steps" ]]; then
    echo "WAM_TOTAL_STEPS=$WAM_TOTAL_STEPS does not match one full dataset epoch ($resolved_total_steps)" >&2
    return 2 2>/dev/null || exit 2
fi
export WAM_TOTAL_STEPS=$resolved_total_steps
_wam_require_positive_integer WAM_TOTAL_STEPS || return 2 2>/dev/null || exit 2

for key in WAM_ROLLOUT_WORKERS WAM_REWARD_WORKERS WAM_ROLLOUT_INIT_CONCURRENCY; do
    _wam_require_positive_integer "$key" || return 2 2>/dev/null || exit 2
done

export WAM_RESOURCE_CONTRACT_VERSION=topology-v1
export WAM_TOTAL_GPUS=$total_gpus
export WAM_ACTOR_MINI_BATCH_PER_GPU=$actor_mini_per_gpu

unset total_gpus default_init_concurrency base_prompt_batch prompt_divisor actor_mini_trajectories
unset actor_mini_per_gpu expected_train_size resolved_total_steps micro_limit candidate key
