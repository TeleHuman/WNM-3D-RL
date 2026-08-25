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

"""GN0 STOP detection and deployment diagnostics for the Stage-3 reward."""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from verl_omni.utils.reward_score.wam_stage3_metrics import _env_bool, _env_float, _env_int


def freeze_trajectory_at_gn0_stop(
    predicted_local_deltas: np.ndarray,
    predicted_world_xy: np.ndarray,
    *,
    stop_eps: float,
) -> dict[str, np.ndarray | int | bool | float]:
    """Apply GN0's component-wise STOP test and discard the remaining plan."""

    local_deltas = np.asarray(predicted_local_deltas, dtype=np.float32)
    world_xy = np.asarray(predicted_world_xy, dtype=np.float32)
    if local_deltas.ndim != 2 or local_deltas.shape[1] < 3:
        raise ValueError(f"predicted_local_deltas must have shape [H,>=3], got {local_deltas.shape}")
    if world_xy.shape != (len(local_deltas) + 1, 2):
        raise ValueError(
            "predicted_world_xy must align with local deltas: "
            f"got {world_xy.shape}, expected {(len(local_deltas) + 1, 2)}"
        )
    if stop_eps < 0:
        raise ValueError("stop_eps must be non-negative")

    stop_indices = np.flatnonzero(np.all(np.abs(local_deltas[:, :3]) <= stop_eps, axis=1))
    first_stop_step = int(stop_indices[0]) if len(stop_indices) else -1
    frozen_local_deltas = local_deltas.copy()
    frozen_world_xy = world_xy.copy()
    if first_stop_step >= 0:
        # Action i is STOP at position i. GN0 terminates without applying its
        # tiny residual delta and never executes actions i+1...H-1.
        frozen_local_deltas[first_stop_step:] = 0.0
        frozen_world_xy[first_stop_step + 1 :] = frozen_world_xy[first_stop_step]

    result = {
        "trajectory_world_xy": frozen_world_xy,
        "local_deltas": frozen_local_deltas,
        "emitted_stop": first_stop_step >= 0,
        "stop_step": first_stop_step,
        "stop_max_abs": (float(np.max(np.abs(local_deltas[first_stop_step, :3]))) if first_stop_step >= 0 else -1.0),
    }
    return result


def freeze_trajectory_at_chunk_motion_stop(
    predicted_local_deltas: np.ndarray,
    predicted_world_xy: np.ndarray,
    *,
    chunk_size: int,
    xy_path_threshold_m: float,
    motion_metric: str = "net_displacement",
) -> dict[str, np.ndarray | int | bool | float]:
    """Treat the first low-translation action chunk as a terminal STOP.

    The default metric is the norm of the accumulated chunk dx/dy.  It averages
    away zero-mean per-step Dance-SDE jitter and matches the deployed notion of
    a chunk's net translation.  ``path_length`` remains available for explicit
    anti-oscillation experiments.  The selected chunk is not executed; it and
    every later action are frozen at its starting position.
    """

    local_deltas = np.asarray(predicted_local_deltas, dtype=np.float32)
    world_xy = np.asarray(predicted_world_xy, dtype=np.float32)
    if local_deltas.ndim != 2 or local_deltas.shape[1] < 3:
        raise ValueError(f"predicted_local_deltas must have shape [H,>=3], got {local_deltas.shape}")
    if world_xy.shape != (len(local_deltas) + 1, 2):
        raise ValueError(
            "predicted_world_xy must align with local deltas: "
            f"got {world_xy.shape}, expected {(len(local_deltas) + 1, 2)}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(local_deltas) % chunk_size != 0:
        raise ValueError(
            "chunk-motion STOP requires a horizon divisible by chunk_size: "
            f"horizon={len(local_deltas)}, chunk_size={chunk_size}"
        )
    if xy_path_threshold_m < 0:
        raise ValueError("xy_path_threshold_m must be non-negative")
    motion_metric = str(motion_metric).strip().lower()
    if motion_metric not in {"net_displacement", "path_length"}:
        raise ValueError(f"motion_metric must be 'net_displacement' or 'path_length', got {motion_metric!r}")

    chunk_xy = local_deltas[:, :2].reshape(-1, chunk_size, 2)
    chunk_xy_path_lengths = np.linalg.norm(chunk_xy, axis=2).sum(axis=1)
    chunk_xy_net_displacements = np.linalg.norm(chunk_xy.sum(axis=1), axis=1)
    trigger_motion = chunk_xy_net_displacements if motion_metric == "net_displacement" else chunk_xy_path_lengths
    stop_chunks = np.flatnonzero(trigger_motion <= xy_path_threshold_m)
    first_stop_chunk = int(stop_chunks[0]) if len(stop_chunks) else -1
    first_stop_step = first_stop_chunk * chunk_size if first_stop_chunk >= 0 else -1

    frozen_local_deltas = local_deltas.copy()
    frozen_world_xy = world_xy.copy()
    if first_stop_step >= 0:
        frozen_local_deltas[first_stop_step:] = 0.0
        frozen_world_xy[first_stop_step + 1 :] = frozen_world_xy[first_stop_step]

    return {
        "trajectory_world_xy": frozen_world_xy,
        "local_deltas": frozen_local_deltas,
        "emitted_stop": first_stop_step >= 0,
        "stop_step": first_stop_step,
        "stop_chunk": first_stop_chunk,
        "stop_max_abs": (
            float(np.max(np.abs(local_deltas[first_stop_step : first_stop_step + chunk_size, :2])))
            if first_stop_step >= 0
            else -1.0
        ),
        "chunk_xy_path_lengths": chunk_xy_path_lengths.astype(np.float32),
        "chunk_xy_net_displacements": chunk_xy_net_displacements.astype(np.float32),
        "motion_metric": motion_metric,
    }


def freeze_trajectory_at_gn0_execution_stop(
    predicted_local_deltas: np.ndarray,
    predicted_world_xy: np.ndarray,
    *,
    action_num: int = 8,
    chunk_translation_threshold_m: float = 0.15,
    stop_eps: float = 1e-3,
) -> dict[str, np.ndarray | int | bool | float | str]:
    """Apply GN0's exact two-stage STOP rule to the deployment chunk.

    GN0 first sums XY path length over the returned/executed chunk and emits
    STOP before executing anything when that value is strictly below 0.15 m.
    Otherwise it executes the chunk sequentially and stops at the first action
    whose [dx, dy, dyaw] components are all within ``stop_eps``.  Actions after
    the first deployment chunk are hypothetical because GN0 replans after at
    most ``action_num`` actions, so they must not independently trigger STOP.
    """

    local_deltas = np.asarray(predicted_local_deltas, dtype=np.float32)
    world_xy = np.asarray(predicted_world_xy, dtype=np.float32)
    if local_deltas.ndim != 2 or local_deltas.shape[1] < 3:
        raise ValueError(f"predicted_local_deltas must have shape [H,>=3], got {local_deltas.shape}")
    if world_xy.shape != (len(local_deltas) + 1, 2):
        raise ValueError(
            "predicted_world_xy must align with local deltas: "
            f"got {world_xy.shape}, expected {(len(local_deltas) + 1, 2)}"
        )
    if action_num <= 0:
        raise ValueError("action_num must be positive")
    if chunk_translation_threshold_m < 0 or stop_eps < 0:
        raise ValueError("GN0 STOP thresholds must be non-negative")

    execute_horizon = min(int(action_num), len(local_deltas))
    if execute_horizon == 0:
        return {
            "trajectory_world_xy": world_xy.copy(),
            "local_deltas": local_deltas.copy(),
            "emitted_stop": False,
            "stop_step": -1,
            "stop_chunk": -1,
            "stop_max_abs": -1.0,
            "chunk_xy_path_lengths": np.zeros(0, dtype=np.float32),
            "chunk_xy_net_displacements": np.zeros(0, dtype=np.float32),
            "motion_metric": "path_length",
            "stop_reason": "none",
        }

    diagnostic_chunks = [
        local_deltas[start : start + action_num, :2] for start in range(0, len(local_deltas), action_num)
    ]
    chunk_xy_path_lengths = np.asarray(
        [np.linalg.norm(chunk, axis=1).sum() for chunk in diagnostic_chunks],
        dtype=np.float32,
    )
    chunk_xy_net_displacements = np.asarray(
        [np.linalg.norm(chunk.sum(axis=0)) for chunk in diagnostic_chunks],
        dtype=np.float32,
    )

    first_chunk = local_deltas[:execute_horizon]
    first_chunk_path_length = float(np.linalg.norm(first_chunk[:, :2], axis=1).sum())
    if chunk_translation_threshold_m > 0.0 and first_chunk_path_length < chunk_translation_threshold_m:
        first_stop_step = 0
        stop_reason = "chunk_low_translation"
    else:
        stop_indices = np.flatnonzero(np.all(np.abs(first_chunk[:, :3]) <= stop_eps, axis=1))
        first_stop_step = int(stop_indices[0]) if len(stop_indices) else -1
        stop_reason = "componentwise_epsilon" if first_stop_step >= 0 else "none"

    frozen_local_deltas = local_deltas.copy()
    frozen_world_xy = world_xy.copy()
    if first_stop_step >= 0:
        frozen_local_deltas[first_stop_step:] = 0.0
        frozen_world_xy[first_stop_step + 1 :] = frozen_world_xy[first_stop_step]

    return {
        "trajectory_world_xy": frozen_world_xy,
        "local_deltas": frozen_local_deltas,
        "emitted_stop": first_stop_step >= 0,
        "stop_step": first_stop_step,
        "stop_chunk": 0 if first_stop_step >= 0 else -1,
        "stop_max_abs": (float(np.max(np.abs(local_deltas[first_stop_step, :3]))) if first_stop_step >= 0 else -1.0),
        "chunk_xy_path_lengths": chunk_xy_path_lengths.astype(np.float32),
        "chunk_xy_net_displacements": chunk_xy_net_displacements.astype(np.float32),
        "motion_metric": "path_length",
        "stop_reason": stop_reason,
    }


def detect_gn0_stops_per_credit_chunk(
    predicted_local_deltas: np.ndarray,
    *,
    chunk_size: int = 8,
    chunk_translation_threshold_m: float = 0.15,
    stop_eps: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Apply GN0's two-stage STOP detector independently to every chunk.

    Deployment executes only the first chunk before replanning, but all action
    chunks are policy outputs and therefore need STOP supervision.  Each chunk
    is treated as a candidate GN0 execution chunk: first test its accumulated
    XY path length, then (only when that test fails) search for the first
    componentwise [dx, dy, dyaw] STOP action.

    This function deliberately does not freeze later chunks.  A STOP in chunk
    k must not erase the independent STOP target for chunks k+1...K-1.
    """

    local_deltas = np.asarray(predicted_local_deltas, dtype=np.float32)
    if local_deltas.ndim != 2 or local_deltas.shape[1] < 3:
        raise ValueError(f"predicted_local_deltas must have shape [H,>=3], got {local_deltas.shape}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(local_deltas) % chunk_size != 0:
        raise ValueError(
            "Per-chunk GN0 STOP requires a horizon divisible by chunk_size: "
            f"horizon={len(local_deltas)}, chunk_size={chunk_size}"
        )
    if chunk_translation_threshold_m < 0 or stop_eps < 0:
        raise ValueError("GN0 STOP thresholds must be non-negative")

    chunks = local_deltas.reshape(-1, chunk_size, local_deltas.shape[1])
    chunk_xy_path_lengths = np.linalg.norm(chunks[:, :, :2], axis=2).sum(axis=1)
    chunk_xy_net_displacements = np.linalg.norm(chunks[:, :, :2].sum(axis=1), axis=1)
    chunk_count = len(chunks)
    local_stop_steps = np.full(chunk_count, -1, dtype=np.int64)
    stop_reason_ids = np.zeros(chunk_count, dtype=np.int64)
    stop_max_abs = np.full(chunk_count, -1.0, dtype=np.float32)

    for index, chunk in enumerate(chunks):
        if chunk_translation_threshold_m > 0.0 and float(chunk_xy_path_lengths[index]) < chunk_translation_threshold_m:
            local_stop_steps[index] = 0
            stop_reason_ids[index] = 1  # chunk_low_translation
            stop_max_abs[index] = float(np.max(np.abs(chunk[0, :3])))
            continue
        stop_indices = np.flatnonzero(np.all(np.abs(chunk[:, :3]) <= stop_eps, axis=1))
        if len(stop_indices):
            local_step = int(stop_indices[0])
            local_stop_steps[index] = local_step
            stop_reason_ids[index] = 2  # componentwise_epsilon
            stop_max_abs[index] = float(np.max(np.abs(chunk[local_step, :3])))

    emitted = local_stop_steps >= 0
    global_stop_steps = np.where(
        emitted,
        np.arange(chunk_count, dtype=np.int64) * chunk_size + local_stop_steps,
        -1,
    )
    return {
        "chunk_emitted_stop": emitted,
        "chunk_stop_local_steps": local_stop_steps,
        "chunk_stop_steps": global_stop_steps,
        # Private execution-order metadata only. Both detector branches feed
        # the same STOP event/reward and never form separate policy credits.
        "chunk_stop_reason_ids": stop_reason_ids,
        "chunk_stop_max_abs": stop_max_abs,
        "chunk_xy_path_lengths": chunk_xy_path_lengths.astype(np.float32),
        "chunk_xy_net_displacements": chunk_xy_net_displacements.astype(np.float32),
    }


def compute_deployment_stop_diagnostics(
    predicted_local_deltas: np.ndarray,
    predicted_world_xy: np.ndarray,
    ground_truth: Mapping,
) -> dict[str, float]:
    """Evaluate the first deployment chunk with GN0's exact STOP semantics."""

    local_deltas = np.asarray(predicted_local_deltas, dtype=np.float64)
    world_xy = np.asarray(predicted_world_xy, dtype=np.float64)
    execute_actions = _env_int("WAM_DEPLOY_ACTION_NUM", 8)
    stop_eps = _env_float("WAM_DEPLOY_STOP_EPS", 1e-3)
    if execute_actions <= 0 or stop_eps < 0:
        raise ValueError("deployment action count must be positive and stop eps non-negative")
    if local_deltas.ndim != 2 or local_deltas.shape[1] < 3:
        raise ValueError(f"predicted_local_deltas must be [H,>=3], got {local_deltas.shape}")
    if world_xy.shape != (len(local_deltas) + 1, 2):
        raise ValueError(
            "predicted_world_xy must align with local deltas: "
            f"got {world_xy.shape}, expected {(len(local_deltas) + 1, 2)}"
        )

    horizon = min(execute_actions, len(local_deltas))
    goal = np.asarray(ground_truth["goal_world_xy"], dtype=np.float64).reshape(-1)[:2]
    goal_radius = float(ground_truth.get("stop_radius_m", 1.5))
    distances = np.linalg.norm(world_xy[: horizon + 1] - goal[None, :], axis=1)
    chunk_motion_stop_enabled = _env_bool("WAM_CHUNK_MOTION_STOP_ENABLED", False)
    chunk_motion_threshold = _env_float("WAM_CHUNK_MOTION_STOP_THRESHOLD_M", 0.15)
    chunk_motion_metric = os.environ.get("WAM_CHUNK_MOTION_STOP_METRIC", "net_displacement").strip().lower()
    deploy_stop_semantics_enabled = _env_bool("WAM_DEPLOY_STOP_SEMANTICS_ENABLED", False)
    if deploy_stop_semantics_enabled:
        stop_execution = freeze_trajectory_at_gn0_execution_stop(
            local_deltas,
            world_xy,
            action_num=horizon,
            chunk_translation_threshold_m=(chunk_motion_threshold if chunk_motion_stop_enabled else 0.0),
            stop_eps=stop_eps,
        )
    elif chunk_motion_stop_enabled:
        stop_execution = freeze_trajectory_at_chunk_motion_stop(
            local_deltas[:horizon],
            world_xy[: horizon + 1],
            chunk_size=horizon,
            xy_path_threshold_m=chunk_motion_threshold,
            motion_metric=chunk_motion_metric,
        )
    else:
        stop_execution = freeze_trajectory_at_gn0_stop(
            local_deltas[:horizon],
            world_xy[: horizon + 1],
            stop_eps=stop_eps,
        )
    first_stop_step = int(stop_execution["stop_step"])
    executed_position_end = first_stop_step if first_stop_step >= 0 else horizon
    executed_distances = distances[: executed_position_end + 1]
    reached = bool(np.any(executed_distances <= goal_radius))
    emitted_stop = first_stop_step >= 0
    stop_distance = float(distances[first_stop_step]) if emitted_stop else float("nan")
    successful_stop = bool(emitted_stop and stop_distance <= goal_radius)
    premature_stop = bool(emitted_stop and not successful_stop)
    first_stop_max_abs = float(stop_execution["stop_max_abs"])
    final_distance = float(executed_distances[-1])

    return {
        "deployment_stop_eval_horizon": float(horizon),
        "deployment_stop_eps": float(stop_eps),
        "deployment_stop_uses_chunk_motion": float(chunk_motion_stop_enabled),
        "deployment_stop_chunk_motion_threshold_m": float(chunk_motion_threshold),
        "deployment_stop_chunk_motion_metric_id": float(chunk_motion_metric == "path_length"),
        "deployment_stop_start_distance": float(distances[0]),
        "deployment_stop_start_inside_goal": float(distances[0] <= goal_radius),
        "deployment_stop_reached": float(reached),
        "deployment_stop_emitted": float(emitted_stop),
        "deployment_stop_success": float(successful_stop),
        "deployment_stop_premature": float(premature_stop),
        "deployment_stop_reached_without_stop": float(reached and not successful_stop),
        "deployment_stop_first_step": float(first_stop_step),
        "deployment_stop_first_max_abs": float(first_stop_max_abs),
        "deployment_stop_terminal_distance": float(final_distance),
        "stop_val_candidate": float(bool(ground_truth.get("stop_val_candidate", False))),
        "stop_val_collection_reached": float(bool(ground_truth.get("stop_val_collection_reached", False))),
        "stop_val_band_id": float(ground_truth.get("stop_val_band_id", -1)),
    }


def _empty_deployment_stop_diagnostics(ground_truth: Mapping) -> dict[str, float]:
    return {
        "deployment_stop_eval_horizon": 0.0,
        "deployment_stop_eps": float(_env_float("WAM_DEPLOY_STOP_EPS", 1e-3)),
        "deployment_stop_uses_chunk_motion": float(_env_bool("WAM_CHUNK_MOTION_STOP_ENABLED", False)),
        "deployment_stop_chunk_motion_threshold_m": float(_env_float("WAM_CHUNK_MOTION_STOP_THRESHOLD_M", 0.15)),
        "deployment_stop_chunk_motion_metric_id": float(
            os.environ.get("WAM_CHUNK_MOTION_STOP_METRIC", "net_displacement").strip().lower() == "path_length"
        ),
        "deployment_stop_start_distance": -1.0,
        "deployment_stop_start_inside_goal": 0.0,
        "deployment_stop_reached": 0.0,
        "deployment_stop_emitted": 0.0,
        "deployment_stop_success": 0.0,
        "deployment_stop_premature": 0.0,
        "deployment_stop_reached_without_stop": 0.0,
        "deployment_stop_first_step": -1.0,
        "deployment_stop_first_max_abs": -1.0,
        "deployment_stop_terminal_distance": -1.0,
        "stop_val_candidate": float(bool(ground_truth.get("stop_val_candidate", False))),
        "stop_val_collection_reached": float(bool(ground_truth.get("stop_val_collection_reached", False))),
        "stop_val_band_id": float(ground_truth.get("stop_val_band_id", -1)),
    }
