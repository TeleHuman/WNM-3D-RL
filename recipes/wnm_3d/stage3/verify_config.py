#!/usr/bin/env python3
# Copyright 2026 WNM-3D-RL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-closed Stage-3 environment and cross-field contract verifier."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

DEFAULT_CONTRACT = Path(__file__).resolve().parent / "algorithm.env.sh"

CONTRACT_RUNTIME_PATH_KEYS = {
    "WAM_REPO_ROOT",
    "WAM_CONFIG_CONTRACT_VERIFIER",
    "WAM_REWARD_FUNCTION_PATH",
    "WNM3D_INITIAL_CHECKPOINT",
    "WAM_DATA_CONTRACT_CHECKPOINT",
    "WAM_DATA_ROOT",
    "WAM_FLOW_ACTION_CALIBRATION_PATH",
}

RUNTIME_DIRECTORY_KEYS = {
    "WAM_REPO_ROOT",
    "WNM3D_SOURCE_ROOT",
    "WNM3D_INITIAL_CHECKPOINT",
    "WAM_DATA_CONTRACT_CHECKPOINT",
    "WNM_TOKENIZER_SOURCE",
    "WAM_DATA_ROOT",
}

RUNTIME_FILE_KEYS = {
    "WAM_CONFIG_CONTRACT_VERIFIER",
    "WAM_REWARD_FUNCTION_PATH",
    "WAM_FLOW_ACTION_CALIBRATION_PATH",
    "WNM_TEXT_ENCODER_SOURCE",
    "WNM_IMAGE_ENCODER_SOURCE",
    "WNM_VAE_SOURCE",
    "WNM_VGGT_SOURCE",
    "WAM_EVENT_VAL_FILE",
}

REQUIRED_RUNTIME_PATH_KEYS = RUNTIME_DIRECTORY_KEYS | RUNTIME_FILE_KEYS | {"WAM_OUTPUT_DIR"}

REQUIRED_RESOURCE_KEYS = {
    "NNODES",
    "NODE_RANK",
    "MASTER_ADDR",
    "GPUS_PER_NODE",
    "CPUS_PER_NODE",
    "WAM_NNODES",
    "WAM_NUM_GPUS",
    "WAM_RAY_NUM_CPUS",
    "WAM_ROLLOUT_TP",
    "WAM_ROLLOUT_WORKERS",
    "WAM_ROLLOUT_GROUP_STICKY",
    "WAM_REWARD_WORKERS",
    "WAM_ROLLOUT_INIT_CONCURRENCY",
    "WAM_PROMPT_BATCH_SIZE",
    "WAM_PPO_MINI_BATCH_SIZE",
    "WAM_MICRO_BATCH_SIZE",
    "WAM_TOTAL_STEPS",
    "WAM_EXPECTED_TRAIN_SIZE",
    "WAM_VAL_MAX_SAMPLES",
    "WAM_TOTAL_GPUS",
    "WAM_ACTOR_MINI_BATCH_PER_GPU",
    "WAM_RESOURCE_CONTRACT_VERSION",
}


def parse_contract(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or line == "#!/usr/bin/env bash":
            continue
        if line.startswith("CONFIG_DIR="):
            continue
        if not line.startswith("export ") or "=" not in line:
            raise RuntimeError(f"unsupported contract line {path}:{line_number}: {raw!r}")
        key, value = line[len("export ") :].split("=", 1)
        key = key.strip()
        # Runtime paths deliberately contain shell expansion and command
        # substitution. They are verified below as resolved filesystem paths,
        # not compared as literal contract constants.
        if key in CONTRACT_RUNTIME_PATH_KEYS:
            continue
        parts = shlex.split(value, posix=True)
        if len(parts) != 1:
            raise RuntimeError(f"contract value must be one shell word at {path}:{line_number}")
        expected[key] = parts[0]
    if not expected:
        raise RuntimeError(f"empty contract: {path}")
    return expected


def parse_env_snapshot(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            raise RuntimeError(f"invalid environment line {path}:{line_number}: {raw!r}")
        key, value = raw.split("=", 1)
        values[key] = value
    return values


def verify_sticky_rollout_partition(*, prompt_batch: int, rollout_workers: int, enabled: bool) -> None:
    """Require each agent worker chunk to contain complete same-prompt groups."""
    if enabled and prompt_batch % rollout_workers:
        raise RuntimeError(
            "sticky rollout grouping requires the prompt batch to be divisible by "
            f"the rollout worker count: {prompt_batch} vs {rollout_workers}"
        )


def verify(expected: dict[str, str], actual: dict[str, str]) -> None:
    mismatches = [
        f"{key}: expected={value!r} actual={actual.get(key)!r}"
        for key, value in sorted(expected.items())
        if actual.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("Stage-3 configuration drift:\n  " + "\n  ".join(mismatches))

    missing_paths = [key for key in sorted(REQUIRED_RUNTIME_PATH_KEYS) if not actual.get(key)]
    if missing_paths:
        raise RuntimeError(f"missing runtime paths: {missing_paths}")
    missing_resources = [key for key in sorted(REQUIRED_RESOURCE_KEYS) if not actual.get(key)]
    if missing_resources:
        raise RuntimeError(f"missing runtime resources: {missing_resources}")
    for key in sorted(RUNTIME_FILE_KEYS):
        if not Path(actual[key]).is_file():
            raise RuntimeError(f"{key} is not a file: {actual[key]}")
    for key in sorted(RUNTIME_DIRECTORY_KEYS):
        if not Path(actual[key]).is_dir():
            raise RuntimeError(f"{key} is not a directory: {actual[key]}")
    if actual["WAM_ACTION_CHUNK_SIZE"] != actual["WAM_DEPLOY_ACTION_NUM"]:
        raise RuntimeError("action chunk size must equal deployed action count")
    for runtime_key, contract_key in (
        ("NNODES", "WAM_NNODES"),
        ("GPUS_PER_NODE", "WAM_NUM_GPUS"),
        ("CPUS_PER_NODE", "WAM_RAY_NUM_CPUS"),
    ):
        if actual[runtime_key] != actual[contract_key]:
            raise RuntimeError(
                f"topology mismatch: {runtime_key}={actual[runtime_key]!r} but {contract_key}={actual[contract_key]!r}"
            )
    if actual["WAM_RESOURCE_CONTRACT_VERSION"] != "topology-v1":
        raise RuntimeError(f"unsupported resource contract: {actual['WAM_RESOURCE_CONTRACT_VERSION']!r}")
    positive_integer_keys = REQUIRED_RESOURCE_KEYS - {
        "MASTER_ADDR",
        "NODE_RANK",
        "WAM_RESOURCE_CONTRACT_VERSION",
        "WAM_ROLLOUT_GROUP_STICKY",
    }
    try:
        integers = {key: int(actual[key]) for key in positive_integer_keys}
        node_rank = int(actual["NODE_RANK"])
    except ValueError as error:
        raise RuntimeError("runtime resource values must be integers") from error
    non_positive = [key for key, value in integers.items() if value <= 0]
    if non_positive:
        raise RuntimeError(f"runtime resources must be positive: {non_positive}")
    if not 0 <= node_rank < integers["NNODES"]:
        raise RuntimeError(f"NODE_RANK must lie in [0, NNODES): {node_rank} vs {integers['NNODES']}")
    total_gpus = integers["NNODES"] * integers["GPUS_PER_NODE"]
    if integers["WAM_TOTAL_GPUS"] != total_gpus:
        raise RuntimeError(f"WAM_TOTAL_GPUS={integers['WAM_TOTAL_GPUS']} but topology has {total_gpus}")
    expected_rollout_workers = total_gpus // integers["WAM_ROLLOUT_TP"]
    if integers["WAM_ROLLOUT_WORKERS"] != expected_rollout_workers:
        raise RuntimeError(
            f"WAM_ROLLOUT_WORKERS must equal total_gpus / rollout_tp: "
            f"{integers['WAM_ROLLOUT_WORKERS']} vs {expected_rollout_workers}"
        )
    if integers["WAM_REWARD_WORKERS"] != total_gpus:
        raise RuntimeError("WAM_REWARD_WORKERS must equal the total GPU count")
    expected_init_concurrency = min(integers["GPUS_PER_NODE"], (expected_rollout_workers + 1) // 2)
    if integers["WAM_ROLLOUT_INIT_CONCURRENCY"] != expected_init_concurrency:
        raise RuntimeError("WAM_ROLLOUT_INIT_CONCURRENCY does not match the resolved topology")
    prompt_batch = integers["WAM_PROMPT_BATCH_SIZE"]
    ppo_mini_batch = integers["WAM_PPO_MINI_BATCH_SIZE"]
    expected_train_size = integers["WAM_EXPECTED_TRAIN_SIZE"]
    if actual["WAM_ROLLOUT_GROUP_STICKY"] != "true":
        raise RuntimeError("Stage-3 requires sticky same-prompt rollout grouping")
    verify_sticky_rollout_partition(
        prompt_batch=prompt_batch,
        rollout_workers=integers["WAM_ROLLOUT_WORKERS"],
        enabled=True,
    )
    if expected_train_size % prompt_batch:
        raise RuntimeError("training row count must divide evenly into the prompt batch")
    if integers["WAM_TOTAL_STEPS"] != expected_train_size // prompt_batch:
        raise RuntimeError("total steps must cover exactly one full training-data epoch")
    if prompt_batch % ppo_mini_batch:
        raise RuntimeError("PPO mini-batch must divide the prompt batch")
    actor_mini_trajectories = ppo_mini_batch * int(actual["WAM_ROLLOUT_N"])
    if actor_mini_trajectories % total_gpus:
        raise RuntimeError("PPO mini trajectories must divide evenly across the total GPUs")
    actor_mini_per_gpu = actor_mini_trajectories // total_gpus
    if integers["WAM_ACTOR_MINI_BATCH_PER_GPU"] != actor_mini_per_gpu:
        raise RuntimeError("WAM_ACTOR_MINI_BATCH_PER_GPU does not match the resolved topology")
    if actor_mini_per_gpu % integers["WAM_MICRO_BATCH_SIZE"]:
        raise RuntimeError("micro-batch must divide the per-GPU actor mini-batch")
    if actual["WAM_COLLISION_RECOVERY_ENABLED"] == "true":
        forbidden = {
            "WAM_COLLISION_STOP_ENABLED": "true",
            "WAM_COLLISION_CREDIT_ENABLED": "true",
            "WAM_TERMINAL_SAFETY_ADVANTAGE_ENABLED": "true",
        }
        bad = [key for key, value in forbidden.items() if actual[key] == value]
        if bad:
            raise RuntimeError(f"collision recovery conflict: {bad}")
    weights = [float(value) for value in actual["WAM_ACTION_CHUNK_WEIGHTS"].split(",")]
    if len(weights) != 4 or any(a <= b for a, b in zip(weights, weights[1:], strict=False)):
        raise RuntimeError(f"unexpected chunk weights: {weights}")
    if float(actual["WAM_YAW_TOTAL_PENALTY_CAP"]) > sum(
        float(actual[key])
        for key in (
            "WAM_YAW_PATH_CONSISTENCY_WEIGHT",
            "WAM_YAW_RATE_CONSISTENCY_WEIGHT",
            "WAM_YAW_GROSS_GT_WEIGHT",
        )
    ):
        raise RuntimeError("yaw cap cannot exceed the sum of its component weights")
    spike_max_mix = float(actual["WAM_YAW_SPIKE_MAX_MIX"])
    if not 0.0 <= spike_max_mix <= 1.0:
        raise RuntimeError(f"yaw spike max mix must lie in [0, 1]: {spike_max_mix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    expected = parse_contract(args.contract)
    actual = parse_env_snapshot(args.env_file) if args.env_file else dict(os.environ)
    verify(expected, actual)
    print(
        "STAGE3_CONFIG_CONTRACT_OK "
        f"version={expected['WAM_CONFIG_CONTRACT_VERSION']} "
        f"constants={len(expected)} runtime_paths={len(REQUIRED_RUNTIME_PATH_KEYS)} "
        f"nodes={actual['NNODES']} total_gpus={actual['WAM_TOTAL_GPUS']} "
        f"prompt_batch={actual['WAM_PROMPT_BATCH_SIZE']} "
        f"rollout_n={actual['WAM_ROLLOUT_N']} "
        f"micro_batch={actual['WAM_MICRO_BATCH_SIZE']} "
        f"steps={actual['WAM_TOTAL_STEPS']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
