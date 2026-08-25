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

"""Chunked navigation, yaw, collision, and STOP reward composition."""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from verl_omni.utils.action_chunk_credit import (
    action_chunk_size,
    normalized_action_chunk_weights,
)
from verl_omni.utils.reward_score.wam_navigation_reward import (
    freeze_trajectory_at_first_collision,
    normalized_dtw,
    occupancy_geodesic_distances_to_goal,
    trajectory_clearance_risk,
    trajectory_collides,
    trajectory_path_length,
)
from verl_omni.utils.reward_score.wam_stage3_collision import execute_gn0_collision_recovery_chunk
from verl_omni.utils.reward_score.wam_stage3_metrics import (
    _compute_stop_well_chunks,
    _env_bool,
    _env_float,
    _env_int,
    _linear_deadzone_score,
    _masked_mean,
    _masked_mean_max_aggregate,
    _premature_stop_distance_terms,
    _trajectory_route_diagnostics,
    _trajectory_yaw_diagnostics,
    _wrap_angle,
)
from verl_omni.utils.reward_score.wam_stage3_stop import (
    detect_gn0_stops_per_credit_chunk,
    freeze_trajectory_at_chunk_motion_stop,
)
from verl_omni.utils.reward_score.wam_terminal_stop_penalty import (
    TerminalStopConfig,
    compute_terminal_stop_penalty,
)


def _compute_chunked_action_reward_impl(
    predicted_world_xy: np.ndarray,
    ground_truth: Mapping,
    *,
    predicted_local_deltas: np.ndarray | None = None,
    action_reward_min: float,
    action_reward_max: float,
    path_efficiency_power: float,
    chunk0_flow_action_bonus: float = 0.0,
    _trajectory_collides_fn=trajectory_collides,
    _trajectory_clearance_risk_fn=trajectory_clearance_risk,
    _occupancy_geodesic_distances_to_goal_fn=occupancy_geodesic_distances_to_goal,
    _freeze_trajectory_at_first_collision_fn=freeze_trajectory_at_first_collision,
) -> dict:
    """Score four contiguous action chunks without coupling their policy credit."""

    raw_trajectory = np.asarray(predicted_world_xy, dtype=np.float32)
    horizon = len(raw_trajectory) - 1
    chunk_size = action_chunk_size()
    if horizon <= 0 or horizon % chunk_size != 0:
        raise ValueError(
            "Chunk credit requires a positive action horizon divisible by "
            f"WAM_ACTION_CHUNK_SIZE; horizon={horizon}, chunk_size={chunk_size}."
        )
    chunk_count = horizon // chunk_size
    if chunk0_flow_action_bonus < 0 or not np.isfinite(chunk0_flow_action_bonus):
        raise ValueError("chunk0_flow_action_bonus must be finite and non-negative")
    weights = normalized_action_chunk_weights(expected_chunks=chunk_count)
    gt_world_xy = np.asarray(ground_truth["gt_world_xy"], dtype=np.float32)
    if len(gt_world_xy) < horizon + 1:
        raise ValueError(
            "GT trajectory is shorter than the executed action horizon: "
            f"gt_positions={len(gt_world_xy)}, required={horizon + 1}."
        )

    stop_well_enabled = _env_bool("WAM_STOP_WELL_ENABLED", False)
    yaw_credit_enabled = _env_bool("WAM_YAW_CREDIT_ENABLED", False)
    deploy_stop_semantics_enabled = _env_bool("WAM_DEPLOY_STOP_SEMANTICS_ENABLED", False)
    chunk_motion_stop_enabled = _env_bool("WAM_CHUNK_MOTION_STOP_ENABLED", False)
    chunk_motion_stop_threshold = _env_float("WAM_CHUNK_MOTION_STOP_THRESHOLD_M", 0.15)
    chunk_motion_stop_metric = os.environ.get("WAM_CHUNK_MOTION_STOP_METRIC", "net_displacement").strip().lower()
    if chunk_motion_stop_metric not in {"net_displacement", "path_length"}:
        raise ValueError(
            f"WAM_CHUNK_MOTION_STOP_METRIC must be net_displacement or path_length, got {chunk_motion_stop_metric!r}"
        )
    if chunk_motion_stop_threshold < 0:
        raise ValueError("WAM_CHUNK_MOTION_STOP_THRESHOLD_M must be non-negative")
    if stop_well_enabled or yaw_credit_enabled or deploy_stop_semantics_enabled or chunk_motion_stop_enabled:
        if predicted_local_deltas is None:
            raise ValueError("Stop/yaw credit requires decoded predicted_local_deltas")
        predicted_local_deltas = np.asarray(predicted_local_deltas, dtype=np.float32)
        if predicted_local_deltas.shape != (horizon, 3):
            raise ValueError(
                "predicted_local_deltas must match the action horizon: "
                f"got {predicted_local_deltas.shape}, expected {(horizon, 3)}"
            )

    stop_execution: dict[str, np.ndarray | int | bool | float] = {
        "trajectory_world_xy": raw_trajectory,
        "local_deltas": predicted_local_deltas,
        "emitted_stop": False,
        "stop_step": -1,
        "stop_max_abs": -1.0,
        "stop_chunk": -1,
        "chunk_xy_path_lengths": np.zeros(chunk_count, dtype=np.float32),
        "chunk_xy_net_displacements": np.zeros(chunk_count, dtype=np.float32),
    }
    per_chunk_gn0_stop: dict[str, np.ndarray] | None = None
    if deploy_stop_semantics_enabled:
        assert predicted_local_deltas is not None
        per_chunk_gn0_stop = detect_gn0_stops_per_credit_chunk(
            predicted_local_deltas,
            chunk_size=chunk_size,
            chunk_translation_threshold_m=(chunk_motion_stop_threshold if chunk_motion_stop_enabled else 0.0),
            stop_eps=_env_float("WAM_DEPLOY_STOP_EPS", 1e-3),
        )
        emitted_chunks = np.flatnonzero(per_chunk_gn0_stop["chunk_emitted_stop"])
        first_stop_chunk = int(emitted_chunks[0]) if len(emitted_chunks) else -1
        first_stop_step = int(per_chunk_gn0_stop["chunk_stop_steps"][first_stop_chunk]) if first_stop_chunk >= 0 else -1
        stop_execution.update(
            {
                "emitted_stop": first_stop_step >= 0,
                "stop_step": first_stop_step,
                "stop_chunk": first_stop_chunk,
                "stop_max_abs": (
                    float(per_chunk_gn0_stop["chunk_stop_max_abs"][first_stop_chunk]) if first_stop_chunk >= 0 else -1.0
                ),
                "chunk_xy_path_lengths": per_chunk_gn0_stop["chunk_xy_path_lengths"],
                "chunk_xy_net_displacements": per_chunk_gn0_stop["chunk_xy_net_displacements"],
            }
        )
    elif chunk_motion_stop_enabled:
        assert predicted_local_deltas is not None
        stop_execution = freeze_trajectory_at_chunk_motion_stop(
            predicted_local_deltas,
            raw_trajectory,
            chunk_size=chunk_size,
            xy_path_threshold_m=chunk_motion_stop_threshold,
            motion_metric=chunk_motion_stop_metric,
        )
        raw_trajectory = np.asarray(stop_execution["trajectory_world_xy"], dtype=np.float32)
        predicted_local_deltas = np.asarray(stop_execution["local_deltas"], dtype=np.float32)

    stop_trigger_xy_path_lengths = np.asarray(
        stop_execution.get(
            "chunk_xy_path_lengths",
            np.zeros(chunk_count, dtype=np.float32),
        ),
        dtype=np.float64,
    )
    if stop_trigger_xy_path_lengths.shape != (chunk_count,):
        raise ValueError(
            "chunk-motion STOP diagnostics must align with credit chunks: "
            f"got {stop_trigger_xy_path_lengths.shape}, expected {(chunk_count,)}"
        )
    stop_trigger_xy_net_displacements = np.asarray(
        stop_execution.get(
            "chunk_xy_net_displacements",
            np.zeros(chunk_count, dtype=np.float32),
        ),
        dtype=np.float64,
    )
    if stop_trigger_xy_net_displacements.shape != (chunk_count,):
        raise ValueError(
            "chunk-motion STOP net-displacement diagnostics must align with "
            f"credit chunks: got {stop_trigger_xy_net_displacements.shape}, "
            f"expected {(chunk_count,)}"
        )

    occupancy_threshold = _env_int("WAM_OCCUPANCY_THRESHOLD", 200)
    collision_hard_margin_px = _env_int("WAM_OCCUPANCY_MARGIN_PX", 2)
    collision_soft_enabled = _env_bool("WAM_COLLISION_SOFT_ENABLED", False)
    collision_soft_margin_px = _env_int("WAM_COLLISION_SOFT_MARGIN_PX", 4)
    collision_soft_weight = _env_float("WAM_COLLISION_SOFT_PENALTY_WEIGHT", 0.20)
    collision_credit_enabled = _env_bool("WAM_COLLISION_CREDIT_ENABLED", False)
    terminal_safety_enabled = _env_bool("WAM_TERMINAL_SAFETY_ADVANTAGE_ENABLED", False)
    terminal_safety_hard_weight = _env_float(
        "WAM_TERMINAL_SAFETY_HARD_WEIGHT",
        _env_float("WAM_COLLISION_PENALTY_WEIGHT", 0.10),
    )
    terminal_safety_soft_weight = _env_float(
        "WAM_TERMINAL_SAFETY_SOFT_WEIGHT",
        collision_soft_weight,
    )
    terminal_safety_soft_max_mix = _env_float("WAM_TERMINAL_SAFETY_SOFT_MAX_MIX", 0.70)
    if terminal_safety_hard_weight < 0 or terminal_safety_soft_weight < 0:
        raise ValueError("Terminal safety weights must be non-negative")
    if not 0.0 <= terminal_safety_soft_max_mix <= 1.0:
        raise ValueError("WAM_TERMINAL_SAFETY_SOFT_MAX_MIX must lie in [0, 1]")
    if collision_hard_margin_px < 0:
        raise ValueError("WAM_OCCUPANCY_MARGIN_PX must be non-negative")
    if collision_soft_weight < 0:
        raise ValueError("WAM_COLLISION_SOFT_PENALTY_WEIGHT must be non-negative")
    if collision_soft_enabled and collision_soft_margin_px <= collision_hard_margin_px:
        raise ValueError("WAM_COLLISION_SOFT_MARGIN_PX must exceed WAM_OCCUPANCY_MARGIN_PX")

    collision_recovery_enabled = _env_bool("WAM_COLLISION_RECOVERY_ENABLED", False)
    deployment_collision_margin_px = _env_int("WAM_DEPLOYMENT_COLLISION_MARGIN_PX", 4)
    collision_repeat_weight = _env_float("WAM_COLLISION_REPEAT_PENALTY_WEIGHT", 0.10)
    collision_repeat_cap = _env_int("WAM_COLLISION_REPEAT_PENALTY_CAP_COUNT", 2)
    collision_recovery_weight = _env_float("WAM_COLLISION_RECOVERY_BONUS_WEIGHT", 0.10)
    collision_recovery_clearance_px = _env_float("WAM_COLLISION_RECOVERY_CLEARANCE_PX", 6.0)
    collision_recovery_min_escape_m = _env_float("WAM_COLLISION_RECOVERY_MIN_ESCAPE_M", 0.20)
    collision_recovery_full_escape_m = _env_float("WAM_COLLISION_RECOVERY_FULL_ESCAPE_M", 0.40)
    collision_recovery_tail_free_steps = _env_int("WAM_COLLISION_RECOVERY_TAIL_FREE_STEPS", 2)
    collision_recovery_grace_enabled = _env_bool("WAM_COLLISION_RECOVERY_GRACE_ENABLED", True)
    deployment_action_num = _env_int("WAM_DEPLOY_ACTION_NUM", 8)
    if deployment_collision_margin_px < 0:
        raise ValueError("deployment collision margin must be non-negative")
    if collision_repeat_weight < 0 or collision_recovery_weight < 0:
        raise ValueError("collision repeat/recovery weights must be non-negative")
    if collision_repeat_cap < 0:
        raise ValueError("collision repeat cap must be non-negative")
    if collision_recovery_clearance_px <= deployment_collision_margin_px:
        raise ValueError("collision recovery clearance must exceed deployment collision margin")
    if (
        collision_recovery_min_escape_m < 0
        or collision_recovery_full_escape_m < collision_recovery_min_escape_m
        or collision_recovery_tail_free_steps <= 0
    ):
        raise ValueError("collision recovery distance/tail thresholds are invalid")
    if collision_recovery_enabled and chunk_size != deployment_action_num:
        raise ValueError(
            "collision recovery is deployment-only and requires "
            "WAM_ACTION_CHUNK_SIZE == WAM_DEPLOY_ACTION_NUM; "
            f"got {chunk_size} != {deployment_action_num}"
        )
    if collision_recovery_enabled and not deploy_stop_semantics_enabled:
        raise ValueError("collision recovery requires WAM_DEPLOY_STOP_SEMANTICS_ENABLED=true")
    if collision_recovery_enabled and (collision_credit_enabled or terminal_safety_enabled):
        raise ValueError(
            "collision recovery keeps safety inside navigation; separate "
            "collision/terminal-safety advantage streams must remain disabled"
        )

    collision_stop_enabled = _env_bool("WAM_COLLISION_STOP_ENABLED", False)
    if collision_recovery_enabled and collision_stop_enabled:
        raise ValueError(
            "WAM_COLLISION_RECOVERY_ENABLED requires "
            "WAM_COLLISION_STOP_ENABLED=false; first-hit freezing and "
            "continued GN0 execution are mutually exclusive"
        )

    deployment_stop_local_step = -1
    if per_chunk_gn0_stop is not None and bool(per_chunk_gn0_stop["chunk_emitted_stop"][0]):
        deployment_stop_local_step = int(per_chunk_gn0_stop["chunk_stop_local_steps"][0])
    elif int(stop_execution.get("stop_chunk", -1)) == 0:
        deployment_stop_local_step = int(stop_execution["stop_step"])

    deployment_execution: dict[str, np.ndarray | float | int | bool] | None = None
    if collision_recovery_enabled:
        deployment_execution = execute_gn0_collision_recovery_chunk(
            raw_trajectory[: chunk_size + 1],
            ground_truth["scene_dir"],
            action_num=chunk_size,
            stop_step=deployment_stop_local_step,
            occupancy_threshold=occupancy_threshold,
            execution_margin_px=deployment_collision_margin_px,
            recovery_clearance_px=collision_recovery_clearance_px,
            recovery_min_escape_m=collision_recovery_min_escape_m,
            recovery_full_escape_m=collision_recovery_full_escape_m,
            recovery_tail_free_steps=collision_recovery_tail_free_steps,
        )

    # Evaluate the complete deployment-executable plan before collision
    # freezing.  STOP has already frozen its suffix, while collision has not;
    # this makes STOP safe without allowing the first wall contact to hide the
    # rest of the jointly generated trajectory from the terminal event.
    if terminal_safety_enabled and collision_soft_enabled:
        terminal_clearance = _trajectory_clearance_risk_fn(
            raw_trajectory,
            ground_truth["scene_dir"],
            occupancy_threshold=occupancy_threshold,
            hard_margin_px=collision_hard_margin_px,
            soft_margin_px=collision_soft_margin_px,
        )
    else:
        terminal_clearance = {
            "risk": 0.0,
            "max_risk": 0.0,
            "mean_risk": 0.0,
            "min_clearance_px": -1.0,
        }

    if collision_stop_enabled:
        execution = _freeze_trajectory_at_first_collision_fn(
            raw_trajectory,
            ground_truth["scene_dir"],
            occupancy_threshold=occupancy_threshold,
            occupancy_margin_px=collision_hard_margin_px,
        )
        trajectory = np.asarray(execution["trajectory_world_xy"], dtype=np.float32)
    else:
        execution = {
            "trajectory_world_xy": raw_trajectory,
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        }
        trajectory = raw_trajectory
    first_collision_step = int(execution["collision_step"])
    first_collision_chunk = (
        min(
            max((max(first_collision_step, 1) - 1) // chunk_size, 0),
            chunk_count - 1,
        )
        if bool(execution["collided"])
        else -1
    )
    terminal_safety_hard_event = bool(execution["collided"])
    terminal_safety_hard_penalty = (
        -terminal_safety_hard_weight if terminal_safety_enabled and terminal_safety_hard_event else 0.0
    )
    terminal_safety_soft_max_risk = float(terminal_clearance["max_risk"])
    terminal_safety_soft_mean_risk = float(terminal_clearance["mean_risk"])
    terminal_safety_soft_risk = (
        terminal_safety_soft_max_mix * terminal_safety_soft_max_risk
        + (1.0 - terminal_safety_soft_max_mix) * terminal_safety_soft_mean_risk
    )
    terminal_safety_soft_penalty = (
        -terminal_safety_soft_weight * terminal_safety_soft_risk
        if terminal_safety_enabled and collision_soft_enabled
        else 0.0
    )
    terminal_safety_reward = terminal_safety_hard_penalty + terminal_safety_soft_penalty
    # collision_step is a position index. The transition entering that
    # position collided and therefore was not actually executed.
    collision_transition_limit = max(first_collision_step - 1, 0) if bool(execution["collided"]) else horizon
    hard_stop_step = int(stop_execution["stop_step"])
    hard_stop_chunk = min(hard_stop_step // chunk_size, chunk_count - 1) if hard_stop_step >= 0 else -1
    # Under per-chunk GN0 supervision, each chunk remains an independent
    # candidate execution chunk.  A STOP in an earlier hypothetical chunk must
    # not mask STOP learning in later chunks.
    executed_transition_limit = (
        collision_transition_limit
        if deploy_stop_semantics_enabled
        else min(
            collision_transition_limit,
            hard_stop_step if hard_stop_step >= 0 else horizon,
        )
    )
    hard_stop_executed = bool(hard_stop_step >= 0 and hard_stop_step <= collision_transition_limit)

    reward_mode = os.environ.get("WAM_ACTION_CHUNK_REWARD_MODE", "softspl").strip().lower()
    mode_ids = {"softspl": 0.0, "signed_progress_length": 1.0}
    if reward_mode not in mode_ids:
        raise ValueError(
            f"WAM_ACTION_CHUNK_REWARD_MODE must be 'softspl' or 'signed_progress_length', got {reward_mode!r}."
        )

    yaw_path_weight = _env_float("WAM_YAW_PATH_CONSISTENCY_WEIGHT", 0.0)
    yaw_rate_weight = _env_float("WAM_YAW_RATE_CONSISTENCY_WEIGHT", 0.0)
    yaw_gross_gt_weight = _env_float("WAM_YAW_GROSS_GT_WEIGHT", 0.0)
    yaw_score_mode = os.environ.get("WAM_YAW_SCORE_MODE", "cosine").strip().lower()
    yaw_free_angle_deg = _env_float("WAM_YAW_FREE_ANGLE_DEG", 15.0)
    yaw_path_hard_angle_deg = _env_float("WAM_YAW_PATH_HARD_ANGLE_DEG", 45.0)
    yaw_rate_free_angle_deg = _env_float("WAM_YAW_RATE_FREE_ANGLE_DEG", 8.0)
    yaw_rate_hard_angle_deg = _env_float("WAM_YAW_RATE_HARD_ANGLE_DEG", 35.0)
    yaw_gt_free_angle_deg = _env_float("WAM_YAW_GT_FREE_ANGLE_DEG", 12.0)
    yaw_gross_angle_deg = _env_float("WAM_YAW_GROSS_ANGLE_DEG", 90.0)
    yaw_total_penalty_cap = _env_float("WAM_YAW_TOTAL_PENALTY_CAP", 1.0)
    yaw_spike_max_mix = _env_float("WAM_YAW_SPIKE_MAX_MIX", 0.50)
    yaw_motion_floor = _env_float("WAM_YAW_MOTION_FLOOR_M", 0.03)
    yaw_rotation_floor = _env_float("WAM_YAW_ROTATION_FLOOR_RAD", 0.01)
    if yaw_path_weight < 0 or yaw_rate_weight < 0 or yaw_gross_gt_weight < 0:
        raise ValueError("Yaw credit weights must be non-negative")
    if not 0.0 <= yaw_free_angle_deg < 180.0:
        raise ValueError("WAM_YAW_FREE_ANGLE_DEG must be in [0, 180)")
    if not 0.0 < yaw_gross_angle_deg < 180.0:
        raise ValueError("WAM_YAW_GROSS_ANGLE_DEG must be in (0, 180)")
    if yaw_score_mode not in {"cosine", "linear_guard"}:
        raise ValueError(f"WAM_YAW_SCORE_MODE must be 'cosine' or 'linear_guard', got {yaw_score_mode!r}")
    if yaw_total_penalty_cap < 0.0:
        raise ValueError("WAM_YAW_TOTAL_PENALTY_CAP must be non-negative")
    if not 0.0 <= yaw_spike_max_mix <= 1.0:
        raise ValueError("WAM_YAW_SPIKE_MAX_MIX must lie in [0, 1]")
    if yaw_motion_floor < 0 or yaw_rotation_floor < 0:
        raise ValueError("Yaw motion/rotation floors must be non-negative")
    if yaw_credit_enabled:
        assert predicted_local_deltas is not None
        yaw_diagnostics = _trajectory_yaw_diagnostics(
            predicted_local_deltas,
            gt_world_xy[: horizon + 1],
            free_angle_deg=yaw_free_angle_deg,
            gross_angle_deg=yaw_gross_angle_deg,
            motion_floor_m=yaw_motion_floor,
            rotation_floor_rad=yaw_rotation_floor,
            score_mode=yaw_score_mode,
            path_hard_angle_deg=yaw_path_hard_angle_deg,
            rate_free_angle_deg=yaw_rate_free_angle_deg,
            rate_hard_angle_deg=yaw_rate_hard_angle_deg,
            gt_free_angle_deg=yaw_gt_free_angle_deg,
        )
    else:
        yaw_diagnostics = {
            "moving": np.zeros(horizon, dtype=bool),
            "translation_moving": np.zeros(horizon, dtype=bool),
            "rotation_moving": np.zeros(horizon, dtype=bool),
            "pure_yaw": np.zeros(horizon, dtype=bool),
            "score_mode_id": np.zeros(horizon, dtype=np.float64),
            "path_score": np.zeros(horizon, dtype=np.float64),
            "path_error": np.zeros(horizon, dtype=np.float64),
            "rate_score": np.zeros(horizon, dtype=np.float64),
            "rate_error": np.zeros(horizon, dtype=np.float64),
            "gross_gt_score": np.zeros(horizon, dtype=np.float64),
            "gross_gt_error": np.zeros(horizon, dtype=np.float64),
            "gross_gt_active": np.zeros(horizon, dtype=bool),
        }

    geodesic_backtrack_weight = _env_float("WAM_GEODESIC_BACKTRACK_WEIGHT", 0.0)
    geodesic_backtrack_tolerance = _env_float("WAM_GEODESIC_BACKTRACK_TOLERANCE_M", 0.01)
    geodesic_backtrack_denom_floor = _env_float("WAM_GEODESIC_BACKTRACK_DENOM_FLOOR_M", 0.10)
    if geodesic_backtrack_weight < 0:
        raise ValueError("WAM_GEODESIC_BACKTRACK_WEIGHT must be non-negative")
    if geodesic_backtrack_tolerance < 0 or geodesic_backtrack_denom_floor <= 0:
        raise ValueError("Geodesic backtrack tolerance/floor are invalid")

    premature_distance_enabled = _env_bool("WAM_PREMATURE_STOP_DISTANCE_SCALING_ENABLED", False)
    premature_distance_deadband_m = _env_float("WAM_PREMATURE_STOP_DISTANCE_DEADBAND_M", 0.25)
    premature_distance_tau_m = _env_float("WAM_PREMATURE_STOP_DISTANCE_TAU_M", 2.0)
    if premature_distance_deadband_m < 0 or premature_distance_tau_m <= 0:
        raise ValueError("premature-STOP distance deadband must be non-negative and tau positive")

    boundaries = np.arange(0, horizon + 1, chunk_size, dtype=np.int64)
    # Stop-well navigation still needs the exact entry transition so ordinary
    # navigation credit ends at the first 1.5 m crossing. This dense distance
    # sequence is not an in-goal potential reward.
    dense_geodesic = stop_well_enabled or geodesic_backtrack_weight > 0.0 or premature_distance_enabled
    primary_geodesic_queries = trajectory if dense_geodesic else trajectory[boundaries]
    # Per-credit-chunk GN0 STOPs are independent hypothetical executions. A
    # collision in an earlier chunk may freeze ``trajectory`` while a later
    # chunk's STOP still belongs to the original generated plan, so query raw
    # positions in the same Dijkstra solve when distance scaling is enabled.
    geodesic_query_parts = [primary_geodesic_queries]
    if premature_distance_enabled:
        geodesic_query_parts.append(raw_trajectory)
    if deployment_execution is not None:
        geodesic_query_parts.append(np.asarray(deployment_execution["trajectory_world_xy"], dtype=np.float32))
    geodesic_queries = (
        np.concatenate(geodesic_query_parts, axis=0) if len(geodesic_query_parts) > 1 else primary_geodesic_queries
    )
    geodesic = _occupancy_geodesic_distances_to_goal_fn(
        geodesic_queries,
        ground_truth["goal_world_xy"],
        ground_truth["scene_dir"],
        occupancy_threshold=_env_int("WAM_OCCUPANCY_THRESHOLD", 200),
        occupancy_margin_px=_env_int("WAM_GEODESIC_OCCUPANCY_MARGIN_PX", 0),
        snap_radius_px=_env_int("WAM_GEODESIC_SNAP_RADIUS_PX", 4),
    )
    combined_distances = np.asarray(geodesic["distances"], dtype=np.float64)
    combined_snapped = np.asarray(geodesic["snapped"], dtype=bool)
    primary_query_count = len(primary_geodesic_queries)
    query_distances = combined_distances[:primary_query_count]
    query_snapped = combined_snapped[:primary_query_count]
    geodesic_offset = primary_query_count
    raw_stop_distances = (
        combined_distances[geodesic_offset : geodesic_offset + horizon + 1] if premature_distance_enabled else None
    )
    if premature_distance_enabled:
        geodesic_offset += horizon + 1
    if premature_distance_enabled and raw_stop_distances.shape != (horizon + 1,):
        raise ValueError(
            "raw premature-STOP geodesic distances must align with the action "
            f"trajectory, got {raw_stop_distances.shape}"
        )
    deployment_distances = None
    deployment_snapped = None
    if deployment_execution is not None:
        deployment_query_count = len(np.asarray(deployment_execution["trajectory_world_xy"]))
        deployment_distances = combined_distances[geodesic_offset : geodesic_offset + deployment_query_count]
        deployment_snapped = combined_snapped[geodesic_offset : geodesic_offset + deployment_query_count]
        if deployment_distances.shape != (chunk_size + 1,):
            raise ValueError(
                f"deployment collision execution geodesics must align with chunk 0: got {deployment_distances.shape}"
            )
    if dense_geodesic:
        all_distances = query_distances
        all_snapped = query_snapped
        distances = all_distances[boundaries]
        snapped = all_snapped[boundaries]
    else:
        all_distances = None
        distances = query_distances
        snapped = query_snapped

    goal_radius = float(ground_truth.get("stop_radius_m", 1.5))
    left_goal_penalty = _env_float("WAM_STOP_LEFT_GOAL_PENALTY", -0.50)
    stop = compute_terminal_stop_penalty(
        trajectory,
        ground_truth["goal_world_xy"],
        TerminalStopConfig(
            goal_radius=goal_radius,
            leave_hysteresis=_env_float("WAM_STOP_LEAVE_HYSTERESIS", 0.0),
            min_step_motion=_env_float("WAM_STOP_MIN_STEP_MOTION", 0.05),
            max_tail_path_length=_env_float("WAM_STOP_MAX_TAIL_PATH", 0.30),
            allowed_moving_steps=_env_int("WAM_STOP_ALLOWED_MOVING_STEPS", 1),
            continued_motion_penalty=_env_float("WAM_STOP_CONTINUED_PENALTY", 0.0),
            left_goal_penalty=left_goal_penalty,
        ),
    )
    left_step = int(stop.get("left_step", -1))
    stop_chunk = min(max((left_step - 1) // chunk_size, 0), chunk_count - 1) if left_step >= 0 else -1
    reached_step = int(stop.get("reached_step", -1))
    goal_xy = np.asarray(ground_truth["goal_world_xy"], dtype=np.float64).reshape(-1)[:2]
    deployment_reached_step = -1
    deployment_stop_emitted = False
    deployment_stop_executed = False
    deployment_stop_distance = -1.0
    deployment_stop_geodesic_distance = -1.0
    deployment_stop_correct = False
    deployment_stop_premature = False
    deployment_post_collision_premature_stop = False
    deployment_final_distance = -1.0
    deployment_stop_terminal = None
    if deployment_execution is not None:
        assert deployment_distances is not None
        deployment_trajectory = np.asarray(deployment_execution["trajectory_world_xy"], dtype=np.float32)
        deployment_executed_actions = int(deployment_execution["executed_action_count"])
        inside = np.flatnonzero(deployment_distances[: deployment_executed_actions + 1] <= goal_radius)
        deployment_reached_step = int(inside[0]) if len(inside) else -1
        deployment_stop_emitted = deployment_stop_local_step >= 0
        deployment_stop_executed = bool(
            deployment_stop_emitted and deployment_stop_local_step <= deployment_executed_actions
        )
        if deployment_stop_executed:
            deployment_stop_position = deployment_trajectory[deployment_stop_local_step]
            deployment_stop_distance = float(np.linalg.norm(deployment_stop_position - goal_xy))
            deployment_stop_geodesic_distance = float(deployment_distances[deployment_stop_local_step])
            deployment_stop_correct = bool(deployment_stop_distance <= goal_radius)
            deployment_stop_premature = not deployment_stop_correct
            deployment_post_collision_premature_stop = bool(
                deployment_stop_premature
                and int(deployment_execution["first_collision_action"]) < deployment_stop_local_step
                and int(deployment_execution["first_collision_action"]) >= 0
            )
        deployment_final_distance = float(np.linalg.norm(deployment_trajectory[deployment_executed_actions] - goal_xy))
        deployment_stop_terminal = compute_terminal_stop_penalty(
            deployment_trajectory,
            ground_truth["goal_world_xy"],
            TerminalStopConfig(
                goal_radius=goal_radius,
                leave_hysteresis=_env_float("WAM_STOP_LEAVE_HYSTERESIS", 0.0),
                min_step_motion=_env_float("WAM_STOP_MIN_STEP_MOTION", 0.05),
                max_tail_path_length=_env_float("WAM_STOP_MAX_TAIL_PATH", 0.30),
                allowed_moving_steps=_env_int("WAM_STOP_ALLOWED_MOVING_STEPS", 1),
                continued_motion_penalty=_env_float("WAM_STOP_CONTINUED_PENALTY", 0.0),
                left_goal_penalty=left_goal_penalty,
            ),
        )
    hard_stop_distance = float(np.linalg.norm(trajectory[hard_stop_step] - goal_xy)) if hard_stop_executed else -1.0
    hard_stop_geodesic_distance = (
        float(raw_stop_distances[hard_stop_step])
        if premature_distance_enabled and hard_stop_executed and raw_stop_distances is not None
        else -1.0
    )
    hard_stop_correct = bool(hard_stop_executed and hard_stop_distance <= goal_radius)
    correct_stop_bonus = _env_float("WAM_CORRECT_STOP_BONUS", 0.15)
    premature_stop_penalty = _env_float("WAM_PREMATURE_STOP_PENALTY", 0.50)
    premature_stop_nav_penalty_weight = _env_float("WAM_PREMATURE_STOP_NAV_PENALTY_WEIGHT", 0.0)
    premature_stop_penalty_distance_add = _env_float("WAM_PREMATURE_STOP_PENALTY_DISTANCE_ADD", 0.25)
    premature_stop_nav_distance_add = _env_float("WAM_PREMATURE_STOP_NAV_DISTANCE_ADD", 0.75)
    if (
        correct_stop_bonus < 0
        or premature_stop_penalty < 0
        or premature_stop_nav_penalty_weight < 0
        or premature_stop_penalty_distance_add < 0
        or premature_stop_nav_distance_add < 0
    ):
        raise ValueError("STOP bonus/penalty magnitudes must be non-negative")

    if stop_well_enabled:
        stop_well = _compute_stop_well_chunks(
            trajectory,
            predicted_local_deltas,
            reached_step=reached_step,
            left_step=left_step,
            left_goal_penalty=left_goal_penalty,
            goal_radius=goal_radius,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            max_transition_exclusive=executed_transition_limit,
        )
    else:
        stop_well = {
            "reward": np.zeros(chunk_count, dtype=np.float64),
            "active": np.zeros(chunk_count, dtype=bool),
            "energy": np.zeros(chunk_count, dtype=np.float64),
            "xy_motion": np.zeros(chunk_count, dtype=np.float64),
            "yaw_motion": np.zeros(chunk_count, dtype=np.float64),
            "exit_penalty": np.zeros(chunk_count, dtype=np.float64),
        }

    deployment_stop_well = None
    if stop_well_enabled and deployment_execution is not None and deployment_stop_terminal is not None:
        deployment_local_deltas = np.asarray(predicted_local_deltas[:chunk_size], dtype=np.float32).copy()
        if deployment_stop_local_step >= 0:
            deployment_local_deltas[deployment_stop_local_step:] = 0.0
        deployment_stop_well = _compute_stop_well_chunks(
            np.asarray(deployment_execution["trajectory_world_xy"], dtype=np.float32),
            deployment_local_deltas,
            reached_step=deployment_reached_step,
            left_step=int(deployment_stop_terminal.get("left_step", -1)),
            left_goal_penalty=left_goal_penalty,
            goal_radius=goal_radius,
            chunk_size=chunk_size,
            chunk_count=1,
            max_transition_exclusive=int(deployment_execution["executed_action_count"]),
        )

    collision_weight = _env_float("WAM_COLLISION_PENALTY_WEIGHT", 0.10)
    if collision_weight < 0:
        raise ValueError("WAM_COLLISION_PENALTY_WEIGHT must be non-negative")
    deployment_recovery_score = (
        float(deployment_execution["recovery_score"]) if deployment_execution is not None else 0.0
    )
    deployment_recovery_eligible = bool(
        deployment_execution is not None
        and bool(deployment_execution["recovered"])
        and not deployment_post_collision_premature_stop
    )
    deployment_recovery_bonus = (
        collision_recovery_weight * deployment_recovery_score if deployment_recovery_eligible else 0.0
    )
    goal_temperature = _env_float("WAM_GOAL_SCORE_TEMPERATURE_M", 0.75)
    if goal_temperature <= 0:
        raise ValueError("WAM_GOAL_SCORE_TEMPERATURE_M must be positive")
    goal_credit_uses_potential_delta = _env_bool("WAM_GOAL_SCORE_USE_POTENTIAL_DELTA", False)
    goal_entry_bonus_weight = _env_float("WAM_GOAL_ENTRY_BONUS", 0.0)
    if goal_entry_bonus_weight < 0:
        raise ValueError("WAM_GOAL_ENTRY_BONUS must be non-negative")
    # reached_step is the first position index inside the goal region.  The
    # transition that crosses the boundary belongs to (reached_step - 1), so
    # an entry exactly on an 8-action boundary is credited to the earlier
    # chunk. Starting inside the region is not an arrival event.
    goal_entry_chunk = min((reached_step - 1) // chunk_size, chunk_count - 1) if reached_step > 0 else -1
    softspl_weight = _env_float("WAM_SOFTSPL_WEIGHT", 0.90)
    goal_score_weight = _env_float("WAM_GOAL_SCORE_WEIGHT", 0.10)
    signed_weight = _env_float("WAM_SIGNED_PROGRESS_WEIGHT", 0.70)
    length_weight = _env_float("WAM_SYMMETRIC_LENGTH_WEIGHT", 0.15)
    signed_goal_weight = _env_float("WAM_SIGNED_GOAL_WEIGHT", 0.15)
    signed_progress_floor = _env_float("WAM_SIGNED_PROGRESS_DENOM_FLOOR_M", 0.50)
    length_scale = _env_float("WAM_SYMMETRIC_LENGTH_SCALE", 0.50)
    length_floor = _env_float("WAM_SYMMETRIC_LENGTH_FLOOR_M", 0.10)
    route_deviation_weight = _env_float("WAM_ROUTE_DEVIATION_WEIGHT", 0.0)
    route_free_radius = _env_float("WAM_ROUTE_DEVIATION_FREE_RADIUS_M", 0.75)
    route_deviation_scale = _env_float("WAM_ROUTE_DEVIATION_SCALE_M", 1.50)
    reverse_direction_weight = _env_float("WAM_REVERSE_DIRECTION_WEIGHT", 0.0)
    ndtw_weight = _env_float("WAM_NDTW_WEIGHT", 0.0)
    ndtw_success_distance = _env_float("WAM_NDTW_SUCCESS_DISTANCE_M", 0.50)
    route_motion_floor = _env_float("WAM_ROUTE_MOTION_FLOOR_M", 0.03)
    if signed_progress_floor <= 0 or length_scale <= 0 or length_floor < 0:
        raise ValueError("Signed-progress/length reward scales must be positive")
    if route_deviation_weight < 0 or reverse_direction_weight < 0 or ndtw_weight < 0:
        raise ValueError("Route/reverse/nDTW weights must be non-negative")
    if ndtw_success_distance <= 0:
        raise ValueError("WAM_NDTW_SUCCESS_DISTANCE_M must be positive")
    if route_free_radius < 0 or route_deviation_scale <= 0 or route_motion_floor < 0:
        raise ValueError("Route deviation scales are invalid")
    stop_reward_min = _env_float("WAM_STOP_REWARD_MIN", -1.0)
    stop_reward_max = _env_float("WAM_STOP_REWARD_MAX", 1.0)
    if stop_reward_min > stop_reward_max:
        raise ValueError("WAM_STOP_REWARD_MIN must not exceed WAM_STOP_REWARD_MAX")

    chunks: list[dict] = []
    metrics: dict[str, float] = {
        "action_chunk_count": float(chunk_count),
        "action_chunk_reward_mode_id": mode_ids[reward_mode],
        "action_yaw_credit_enabled": float(yaw_credit_enabled),
        "action_goal_credit_uses_potential_delta": float(goal_credit_uses_potential_delta),
        "action_goal_entry_bonus_weight": float(goal_entry_bonus_weight),
        "action_goal_entry_chunk": float(goal_entry_chunk),
        "action_stop_credit_enabled": float(stop_well_enabled),
        "action_deploy_stop_semantics_enabled": float(deploy_stop_semantics_enabled),
        "action_chunk_motion_stop_enabled": float(chunk_motion_stop_enabled),
        "action_chunk_motion_stop_threshold_m": float(chunk_motion_stop_threshold),
        "action_chunk_motion_stop_metric_id": float(chunk_motion_stop_metric == "path_length"),
        "action_chunk_motion_stop_chunk": float(stop_execution.get("stop_chunk", -1)),
        "action_chunk_motion_stop_emitted": float(chunk_motion_stop_enabled and hard_stop_step >= 0),
        "action_hard_stop_emitted": float(hard_stop_step >= 0),
        "action_hard_stop_executed": float(hard_stop_executed),
        "action_hard_stop_correct": float(hard_stop_correct),
        "action_hard_stop_premature": float(hard_stop_executed and not hard_stop_correct),
        "action_hard_stop_step": float(hard_stop_step),
        "action_hard_stop_chunk": float(hard_stop_chunk),
        "action_hard_stop_distance_m": float(hard_stop_distance),
        "action_hard_stop_geodesic_distance_m": float(
            hard_stop_geodesic_distance if np.isfinite(hard_stop_geodesic_distance) else -1.0
        ),
        "action_premature_stop_distance_scaling_enabled": float(premature_distance_enabled),
        "action_premature_stop_distance_deadband_m": float(premature_distance_deadband_m),
        "action_premature_stop_distance_tau_m": float(premature_distance_tau_m),
        "action_premature_stop_penalty_distance_add": float(premature_stop_penalty_distance_add),
        "action_premature_stop_nav_distance_add": float(premature_stop_nav_distance_add),
        "action_ndtw_weight": float(ndtw_weight),
        "action_geodesic_backtrack_weight": float(geodesic_backtrack_weight),
        "action_collision_stop_enabled": float(collision_stop_enabled),
        "action_collision_recovery_enabled": float(collision_recovery_enabled),
        "action_deployment_collision_margin_px": float(deployment_collision_margin_px),
        "action_collision_repeat_penalty_weight": float(collision_repeat_weight),
        "action_collision_repeat_penalty_cap_count": float(collision_repeat_cap),
        "action_collision_recovery_bonus_weight": float(collision_recovery_weight),
        "action_collision_recovery_clearance_px": float(collision_recovery_clearance_px),
        "action_collision_recovery_min_escape_m": float(collision_recovery_min_escape_m),
        "action_collision_recovery_full_escape_m": float(collision_recovery_full_escape_m),
        "action_collision_recovery_tail_free_steps": float(collision_recovery_tail_free_steps),
        "action_collision_recovery_grace_enabled": float(collision_recovery_grace_enabled),
        "action_chunk_0_deployment_collision": float(
            bool(deployment_execution["collided"]) if deployment_execution is not None else False
        ),
        "action_chunk_0_deployment_collision_count": float(
            int(deployment_execution["collision_count"]) if deployment_execution is not None else 0
        ),
        "action_chunk_0_deployment_first_collision_step": float(
            int(deployment_execution["first_collision_step"]) if deployment_execution is not None else -1
        ),
        "action_chunk_0_deployment_collision_out_of_bounds": float(
            bool(deployment_execution["out_of_bounds"]) if deployment_execution is not None else False
        ),
        "action_chunk_0_collision_recovered": float(
            bool(deployment_execution["recovered"]) if deployment_execution is not None else False
        ),
        "action_chunk_0_collision_recovery_eligible": float(deployment_recovery_eligible),
        "action_chunk_0_collision_recovery_score": float(deployment_recovery_score),
        "action_chunk_0_collision_recovery_bonus": float(deployment_recovery_bonus),
        "action_chunk_0_recovery_distance_m": float(
            deployment_execution["recovery_escape_distance_m"] if deployment_execution is not None else 0.0
        ),
        "action_chunk_0_recovery_clearance_px": float(
            deployment_execution["recovery_tail_clearance_px"] if deployment_execution is not None else -1.0
        ),
        "action_chunk_0_post_collision_premature_stop": float(deployment_post_collision_premature_stop),
        "action_any_collision": float(bool(execution["collided"])),
        "action_first_collision_step": float(first_collision_step),
        "action_first_collision_chunk": float(first_collision_chunk),
        "action_collision_out_of_bounds": float(bool(execution["out_of_bounds"])),
        "action_collision_soft_enabled": float(collision_soft_enabled),
        "action_collision_hard_margin_px": float(collision_hard_margin_px),
        "action_collision_soft_margin_px": float(collision_soft_margin_px),
        "action_collision_soft_penalty_weight": float(collision_soft_weight),
        "action_collision_credit_enabled": float(collision_credit_enabled),
        "action_terminal_safety_enabled": float(terminal_safety_enabled),
        "action_terminal_safety_reward": float(terminal_safety_reward),
        "action_terminal_safety_hard_event": float(terminal_safety_hard_event),
        "action_terminal_safety_hard_penalty": float(terminal_safety_hard_penalty),
        "action_terminal_safety_soft_risk": float(terminal_safety_soft_risk),
        "action_terminal_safety_soft_max_risk": float(terminal_safety_soft_max_risk),
        "action_terminal_safety_soft_mean_risk": float(terminal_safety_soft_mean_risk),
        "action_terminal_safety_soft_penalty": float(terminal_safety_soft_penalty),
        "action_terminal_safety_min_clearance_px": float(terminal_clearance["min_clearance_px"]),
    }
    if deployment_execution is not None:
        # Override pre-collision deployment diagnostics so train/val STOP
        # metrics describe the same clipped pose that GN0 would execute.
        metrics.update(
            {
                "deployment_stop_reached": float(deployment_reached_step >= 0),
                "deployment_stop_emitted": float(deployment_stop_emitted),
                "deployment_stop_success": float(deployment_stop_correct),
                "deployment_stop_premature": float(deployment_stop_premature),
                "deployment_stop_reached_without_stop": float(
                    deployment_reached_step >= 0 and not deployment_stop_correct
                ),
                "deployment_stop_terminal_distance": float(deployment_final_distance),
            }
        )
    weighted_reward = 0.0
    weighted_pre_clip = 0.0
    weighted_base = 0.0
    weighted_yaw_path_score = 0.0
    weighted_yaw_rate_score = 0.0
    weighted_yaw_gross_gt_score = 0.0
    weighted_yaw_path_penalty = 0.0
    weighted_yaw_rate_penalty = 0.0
    weighted_yaw_gross_gt_penalty = 0.0
    weighted_yaw_total_penalty = 0.0
    weighted_ndtw = 0.0
    weighted_ndtw_bonus = 0.0
    weighted_geodesic_backtrack_distance = 0.0
    weighted_geodesic_backtrack_score = 0.0
    weighted_geodesic_backtrack_penalty = 0.0
    weighted_collision_hard_penalty = 0.0
    weighted_collision_repeat_penalty = 0.0
    weighted_collision_soft_risk = 0.0
    weighted_collision_soft_penalty = 0.0
    weighted_collision_recovery_bonus = 0.0
    weighted_collision_reward = 0.0
    active_clearances_px: list[float] = []
    for index, weight in enumerate(weights):
        start = index * chunk_size
        end = start + chunk_size
        use_deployment_execution = bool(index == 0 and deployment_execution is not None)
        if use_deployment_execution:
            deployment_executed_actions = int(deployment_execution["executed_action_count"])
            nav_end = deployment_executed_actions
            if stop_well_enabled and deployment_reached_step >= 0:
                nav_end = min(nav_end, deployment_reached_step)
            nav_end = max(0, min(nav_end, chunk_size))
        else:
            nav_end = max(start, min(end, reached_step)) if stop_well_enabled and reached_step >= 0 else end
            nav_end = max(start, min(nav_end, executed_transition_limit))
        nav_has_transition = start < nav_end
        raw_pred_chunk = raw_trajectory[start : end + 1]
        if use_deployment_execution:
            deployment_trajectory = np.asarray(deployment_execution["trajectory_world_xy"], dtype=np.float32)
            pred_chunk = deployment_trajectory[: nav_end + 1]
            executed_pred_chunk = deployment_trajectory
            gt_chunk = gt_world_xy[: nav_end + 1]
            recovery_grace_mask = np.asarray(deployment_execution["recovery_grace_mask"], dtype=bool)
        else:
            pred_chunk = trajectory[start : nav_end + 1]
            executed_pred_chunk = trajectory[start : end + 1]
            gt_chunk = gt_world_xy[start : nav_end + 1]
            recovery_grace_mask = np.zeros(chunk_size, dtype=bool)
        full_gt_chunk = gt_world_xy[start : end + 1]
        raw_collision = _trajectory_collides_fn(
            raw_pred_chunk,
            ground_truth["scene_dir"],
            occupancy_threshold=occupancy_threshold,
            occupancy_margin_px=collision_hard_margin_px,
        )
        if not collision_stop_enabled and bool(raw_collision["collided"]) and metrics["action_any_collision"] == 0.0:
            raw_collision_step = int(raw_collision["collision_step"])
            metrics["action_any_collision"] = 1.0
            metrics["action_first_collision_step"] = float(start + max(raw_collision_step, 0))
            metrics["action_first_collision_chunk"] = float(index)
            metrics["action_collision_out_of_bounds"] = float(bool(raw_collision["out_of_bounds"]))
        if collision_stop_enabled:
            active = not bool(execution["collided"]) or index <= first_collision_chunk
            is_first_collision_chunk = bool(execution["collided"]) and index == first_collision_chunk
            collision = {
                "collided": is_first_collision_chunk,
                "collision_step": (max(first_collision_step - start, 0) if is_first_collision_chunk else -1),
                "out_of_bounds": (bool(execution["out_of_bounds"]) if is_first_collision_chunk else False),
            }
        else:
            active = True
            collision = raw_collision
        if not deploy_stop_semantics_enabled and hard_stop_executed and index > hard_stop_chunk:
            active = False
        if collision_soft_enabled and active:
            clearance = _trajectory_clearance_risk_fn(
                raw_pred_chunk,
                ground_truth["scene_dir"],
                occupancy_threshold=occupancy_threshold,
                hard_margin_px=collision_hard_margin_px,
                soft_margin_px=collision_soft_margin_px,
            )
            collision_soft_risk = float(clearance["risk"])
            collision_min_clearance_px = float(clearance["min_clearance_px"])
            if collision_min_clearance_px >= 0:
                active_clearances_px.append(collision_min_clearance_px)
        else:
            collision_soft_risk = 0.0
            collision_min_clearance_px = -1.0
        # Navigation progress may stop at first goal entry, and the executed
        # trajectory may freeze at first collision. Neither is allowed to hide
        # planned motion from the fixed-horizon length regularizer: otherwise a
        # policy can enter the goal/collide early and move arbitrarily in the
        # remainder of the chunk without paying the path-length cost.
        nav_pred_length = trajectory_path_length(pred_chunk)
        nav_gt_length = trajectory_path_length(gt_chunk)
        executed_pred_length = trajectory_path_length(executed_pred_chunk)
        pred_length = trajectory_path_length(raw_pred_chunk)
        gt_length = trajectory_path_length(full_gt_chunk)
        if gt_length > 1e-8:
            length_ratio = pred_length / gt_length
        elif pred_length > 1e-8:
            length_ratio = pred_length / 1e-8
        else:
            length_ratio = 1.0
        path_efficiency = (
            1.0
            if pred_length <= gt_length or pred_length <= 1e-8
            else (gt_length / pred_length) ** path_efficiency_power
        )

        if use_deployment_execution:
            assert deployment_distances is not None
            assert deployment_snapped is not None
            d_start = float(deployment_distances[0])
            d_end = float(deployment_distances[nav_end])
            d_start_snapped = bool(deployment_snapped[0])
            d_end_snapped = bool(deployment_snapped[nav_end])
        elif stop_well_enabled:
            assert all_distances is not None
            d_start = float(all_distances[start])
            d_end = float(all_distances[nav_end])
            d_start_snapped = bool(all_snapped[start])
            d_end_snapped = bool(all_snapped[nav_end])
        else:
            d_start = float(distances[index])
            d_end = float(distances[index + 1])
            d_start_snapped = bool(snapped[index])
            d_end_snapped = bool(snapped[index + 1])
        if np.isfinite(d_start) and np.isfinite(d_end) and d_start > 1e-8:
            geodesic_progress = float(np.clip((d_start - d_end) / d_start, 0.0, 1.0))
        elif np.isfinite(d_end) and d_end <= 1e-8:
            geodesic_progress = 1.0
        else:
            geodesic_progress = 0.0
        softspl = geodesic_progress * path_efficiency
        goal_potential_start = float(np.exp(-d_start / goal_temperature)) if np.isfinite(d_start) else 0.0
        goal_potential_end = float(np.exp(-d_end / goal_temperature)) if np.isfinite(d_end) else 0.0
        goal_potential_delta = (
            goal_potential_end - goal_potential_start if np.isfinite(d_start) and np.isfinite(d_end) else 0.0
        )
        # Keep the absolute score as a diagnostic. Training credit uses the
        # potential difference, so holding a fixed distance just
        # outside the goal cannot farm the same positive reward every chunk.
        goal_score = goal_potential_end
        goal_credit = goal_potential_delta if goal_credit_uses_potential_delta else goal_score
        if (
            use_deployment_execution
            and collision_recovery_grace_enabled
            and bool(deployment_execution["collided"])
            and nav_end > 0
            and deployment_distances is not None
            and np.isfinite(deployment_distances[: nav_end + 1]).all()
        ):
            step_progress = deployment_distances[:nav_end] - deployment_distances[1 : nav_end + 1]
            grace = recovery_grace_mask[:nav_end]
            # Recovery may require backing away from the goal. Do not charge
            # negative progress during that bounded window; genuine positive
            # progress remains fully credited.
            step_progress = np.where(
                np.logical_and(grace, step_progress < 0.0),
                0.0,
                step_progress,
            )
            signed_progress = float(
                np.clip(
                    float(np.sum(step_progress)) / max(nav_gt_length, signed_progress_floor),
                    -1.0,
                    1.0,
                )
            )
        elif np.isfinite(d_start) and np.isfinite(d_end):
            signed_progress = float(
                np.clip(
                    (d_start - d_end) / max(nav_gt_length, signed_progress_floor),
                    -1.0,
                    1.0,
                )
            )
        elif np.isfinite(d_start) and not np.isfinite(d_end):
            signed_progress = -1.0
        else:
            signed_progress = 0.0
        length_similarity = float(
            np.exp(-((abs(pred_length - gt_length) / (length_scale * gt_length + length_floor)) ** 2))
        )
        if use_deployment_execution:
            deployment_collision_count = int(deployment_execution["collision_count"])
            collision_hard_penalty = -collision_weight if deployment_collision_count > 0 else 0.0
            collision_repeat_penalty = -collision_repeat_weight * min(
                max(deployment_collision_count - 1, 0),
                collision_repeat_cap,
            )
            chunk_collision_recovery_bonus = deployment_recovery_bonus
        else:
            collision_hard_penalty = -collision_weight if collision["collided"] else 0.0
            collision_repeat_penalty = 0.0
            chunk_collision_recovery_bonus = 0.0
        collision_soft_penalty = -collision_soft_weight * collision_soft_risk if collision_soft_enabled else 0.0
        collision_penalty = collision_hard_penalty + collision_repeat_penalty + collision_soft_penalty
        collision_reward = collision_penalty + chunk_collision_recovery_bonus
        # When collision credit is split into its own GRPO stream, remove it
        # from navigation before clipping.  Otherwise the same event would
        # affect both navigation advantage and collision advantage.
        navigation_collision_penalty = 0.0 if collision_credit_enabled or terminal_safety_enabled else collision_reward
        navigation_terms_active = bool(active and nav_has_transition)
        navigation_active = bool(
            active
            and (
                nav_has_transition
                or bool(collision["collided"])
                or (use_deployment_execution and bool(deployment_execution["collided"]))
            )
        )
        goal_entry = bool(
            navigation_terms_active
            and (
                (use_deployment_execution and deployment_reached_step > 0)
                or (not use_deployment_execution and index == goal_entry_chunk)
            )
        )
        goal_entry_bonus = goal_entry_bonus_weight if goal_entry else 0.0
        if navigation_terms_active and (route_deviation_weight > 0.0 or reverse_direction_weight > 0.0):
            route = _trajectory_route_diagnostics(
                pred_chunk,
                gt_world_xy[: horizon + 1],
                free_radius_m=route_free_radius,
                deviation_scale_m=route_deviation_scale,
                motion_floor_m=route_motion_floor,
                reverse_transition_mask=(
                    ~recovery_grace_mask[: len(pred_chunk) - 1]
                    if use_deployment_execution and collision_recovery_grace_enabled
                    else None
                ),
            )
            route_deviation_penalty = -route_deviation_weight * route["deviation_score"]
            reverse_direction_penalty = -reverse_direction_weight * route["reverse_direction_score"]
        else:
            route = {
                "deviation_score": 0.0,
                "deviation_rms_m": 0.0,
                "deviation_max_m": 0.0,
                "reverse_direction_score": 0.0,
            }
            route_deviation_penalty = 0.0
            reverse_direction_penalty = 0.0

        if navigation_terms_active and ndtw_weight > 0.0:
            chunk_ndtw = normalized_dtw(
                executed_pred_chunk,
                full_gt_chunk,
                success_distance=ndtw_success_distance,
            )
            ndtw_bonus = ndtw_weight * chunk_ndtw
        else:
            chunk_ndtw = 0.0
            ndtw_bonus = 0.0

        geodesic_backtrack_distance = 0.0
        geodesic_backtrack_score = 0.0
        if (
            navigation_terms_active
            and geodesic_backtrack_weight > 0.0
            and (all_distances is not None or deployment_distances is not None)
            and start < nav_end
        ):
            if use_deployment_execution:
                assert deployment_distances is not None
                chunk_distances = np.asarray(deployment_distances[: nav_end + 1], dtype=np.float64)
            else:
                assert all_distances is not None
                chunk_distances = np.asarray(all_distances[start : nav_end + 1], dtype=np.float64)
            distance_before = chunk_distances[:-1]
            distance_after = chunk_distances[1:]
            valid_pairs = np.isfinite(distance_before) & np.isfinite(distance_after)
            if use_deployment_execution and collision_recovery_grace_enabled:
                valid_pairs &= ~recovery_grace_mask[: len(valid_pairs)]
            if np.any(valid_pairs):
                geodesic_backtrack_distance = float(
                    np.maximum(
                        distance_after[valid_pairs] - distance_before[valid_pairs] - geodesic_backtrack_tolerance,
                        0.0,
                    ).sum()
                )
                geodesic_backtrack_score = float(
                    np.clip(
                        geodesic_backtrack_distance / (nav_gt_length + geodesic_backtrack_denom_floor),
                        0.0,
                        1.0,
                    )
                )
        geodesic_backtrack_penalty = -geodesic_backtrack_weight * geodesic_backtrack_score

        yaw_end = min(nav_end, executed_transition_limit)
        if yaw_credit_enabled and navigation_terms_active and start < yaw_end:
            yaw_slice = slice(start, yaw_end)
            yaw_moving = yaw_diagnostics["moving"][yaw_slice]
            yaw_translation_moving = yaw_diagnostics["translation_moving"][yaw_slice]
            yaw_path_scores = np.asarray(yaw_diagnostics["path_score"][yaw_slice], dtype=np.float64).copy()
            yaw_path_errors = np.asarray(yaw_diagnostics["path_error"][yaw_slice], dtype=np.float64).copy()
            yaw_gt_mask = yaw_moving.copy()
            if use_deployment_execution and collision_recovery_grace_enabled:
                grace = recovery_grace_mask[:yaw_end]
                absolute_path_error = np.abs(_wrap_angle(yaw_path_errors))
                folded_path_error = np.minimum(
                    absolute_path_error,
                    np.abs(np.pi - absolute_path_error),
                )
                if yaw_score_mode == "linear_guard":
                    folded_path_score = _linear_deadzone_score(
                        folded_path_error,
                        free_angle_deg=yaw_free_angle_deg,
                        hard_angle_deg=yaw_path_hard_angle_deg,
                    )
                else:
                    free_angle = np.deg2rad(yaw_free_angle_deg)
                    folded_after_deadzone = np.maximum(folded_path_error - free_angle, 0.0)
                    folded_path_score = 0.5 * (1.0 - np.cos(folded_after_deadzone))
                yaw_path_scores = np.where(grace, folded_path_score, yaw_path_scores)
                yaw_path_errors = np.where(grace, folded_path_error, yaw_path_errors)
                # A bounded recovery turn may temporarily face away from the
                # GT route. Rate smoothness remains active, while gross GT yaw
                # resumes as soon as the clearance/escape condition is met.
                yaw_gt_mask = np.logical_and(yaw_gt_mask, ~grace)
            (
                yaw_path_score_mean,
                yaw_path_score_max,
                yaw_path_score,
            ) = _masked_mean_max_aggregate(
                yaw_path_scores,
                yaw_translation_moving,
                max_mix=yaw_spike_max_mix,
            )
            yaw_path_error_deg = np.rad2deg(
                _masked_mean(
                    np.abs(yaw_path_errors),
                    yaw_translation_moving,
                )
            )
            (
                yaw_rate_score_mean,
                yaw_rate_score_max,
                yaw_rate_score,
            ) = _masked_mean_max_aggregate(
                yaw_diagnostics["rate_score"][yaw_slice],
                yaw_translation_moving,
                max_mix=yaw_spike_max_mix,
            )
            yaw_rate_error_deg = np.rad2deg(
                _masked_mean(
                    np.abs(yaw_diagnostics["rate_error"][yaw_slice]),
                    yaw_translation_moving,
                )
            )
            (
                yaw_gross_gt_score_mean,
                yaw_gross_gt_score_max,
                yaw_gross_gt_score,
            ) = _masked_mean_max_aggregate(
                yaw_diagnostics["gross_gt_score"][yaw_slice],
                yaw_gt_mask,
                max_mix=yaw_spike_max_mix,
            )
            yaw_gross_gt_fraction = _masked_mean(
                yaw_diagnostics["gross_gt_active"][yaw_slice].astype(np.float64),
                yaw_gt_mask,
            )
            yaw_gross_gt_error_deg = np.rad2deg(
                _masked_mean(
                    np.abs(yaw_diagnostics["gross_gt_error"][yaw_slice]),
                    yaw_gt_mask,
                )
            )
            yaw_moving_steps = int(np.count_nonzero(yaw_moving))
            yaw_pure_yaw_steps = int(np.count_nonzero(yaw_diagnostics["pure_yaw"][yaw_slice]))
        else:
            yaw_path_score = 0.0
            yaw_path_score_mean = 0.0
            yaw_path_score_max = 0.0
            yaw_path_error_deg = 0.0
            yaw_rate_score = 0.0
            yaw_rate_score_mean = 0.0
            yaw_rate_score_max = 0.0
            yaw_rate_error_deg = 0.0
            yaw_gross_gt_score = 0.0
            yaw_gross_gt_score_mean = 0.0
            yaw_gross_gt_score_max = 0.0
            yaw_gross_gt_fraction = 0.0
            yaw_gross_gt_error_deg = 0.0
            yaw_moving_steps = 0
            yaw_pure_yaw_steps = 0
        yaw_path_penalty = -yaw_path_weight * yaw_path_score
        yaw_rate_penalty = -yaw_rate_weight * yaw_rate_score
        yaw_gross_gt_penalty = -yaw_gross_gt_weight * yaw_gross_gt_score
        yaw_total_penalty_pre_cap = yaw_path_penalty + yaw_rate_penalty + yaw_gross_gt_penalty
        yaw_total_penalty = max(
            yaw_total_penalty_pre_cap,
            -yaw_total_penalty_cap,
        )

        if not navigation_terms_active:
            base_reward = navigation_collision_penalty
        elif reward_mode == "softspl":
            base_reward = (
                softspl_weight * softspl
                + goal_score_weight * goal_credit
                + goal_entry_bonus
                + navigation_collision_penalty
                + route_deviation_penalty
                + reverse_direction_penalty
                + ndtw_bonus
                + geodesic_backtrack_penalty
                + yaw_total_penalty
            )
        else:
            base_reward = (
                signed_weight * signed_progress
                + length_weight * length_similarity
                + signed_goal_weight * goal_credit
                + goal_entry_bonus
                + navigation_collision_penalty
                + route_deviation_penalty
                + reverse_direction_penalty
                + ndtw_bonus
                + geodesic_backtrack_penalty
                + yaw_total_penalty
            )
        chunk_flow_action_bonus = float(chunk0_flow_action_bonus) if index == 0 and active else 0.0
        if chunk_flow_action_bonus > 0.0:
            base_reward += chunk_flow_action_bonus
            navigation_active = True
        chunk_hard_stop_emitted = False
        chunk_hard_stop_executed = False
        chunk_hard_stop_step = -1
        chunk_hard_stop_local_step = -1
        chunk_hard_stop_reason_id = 0
        chunk_hard_stop_distance = -1.0
        chunk_hard_stop_geodesic_distance = -1.0
        chunk_hard_stop_correct = False
        if per_chunk_gn0_stop is not None:
            chunk_hard_stop_emitted = bool(per_chunk_gn0_stop["chunk_emitted_stop"][index])
            chunk_hard_stop_step = int(per_chunk_gn0_stop["chunk_stop_steps"][index])
            chunk_hard_stop_local_step = int(per_chunk_gn0_stop["chunk_stop_local_steps"][index])
            chunk_hard_stop_reason_id = int(per_chunk_gn0_stop["chunk_stop_reason_ids"][index])
            if chunk_hard_stop_emitted:
                if use_deployment_execution:
                    # GN0 continues after a clipped collision, so a later
                    # componentwise STOP is physically reached at the actual
                    # clipped pose rather than being masked by first contact.
                    chunk_hard_stop_executed = deployment_stop_executed
                    chunk_hard_stop_distance = deployment_stop_distance
                    chunk_hard_stop_geodesic_distance = deployment_stop_geodesic_distance
                    chunk_hard_stop_correct = deployment_stop_correct
                else:
                    # Later credit chunks remain independent hypothetical
                    # plans and retain the historical 2 px classifier.
                    chunk_collision_step = int(raw_collision["collision_step"])
                    chunk_hard_stop_executed = bool(
                        chunk_hard_stop_reason_id == 1
                        or chunk_collision_step < 0
                        or chunk_collision_step > chunk_hard_stop_local_step
                    )
                    chunk_hard_stop_distance = float(np.linalg.norm(raw_trajectory[chunk_hard_stop_step] - goal_xy))
                    if premature_distance_enabled:
                        assert raw_stop_distances is not None
                        chunk_hard_stop_geodesic_distance = float(raw_stop_distances[chunk_hard_stop_step])
                    chunk_hard_stop_correct = bool(chunk_hard_stop_executed and chunk_hard_stop_distance <= goal_radius)
        elif hard_stop_executed and index == hard_stop_chunk:
            chunk_hard_stop_emitted = True
            chunk_hard_stop_executed = True
            chunk_hard_stop_step = hard_stop_step
            chunk_hard_stop_local_step = hard_stop_step - start
            chunk_hard_stop_distance = hard_stop_distance
            chunk_hard_stop_geodesic_distance = hard_stop_geodesic_distance
            chunk_hard_stop_correct = hard_stop_correct

        premature_stop_distance_excess_m = 0.0
        premature_stop_distance_scale = 0.0
        if premature_distance_enabled and chunk_hard_stop_executed and not chunk_hard_stop_correct:
            (
                premature_stop_distance_excess_m,
                premature_stop_distance_scale,
            ) = _premature_stop_distance_terms(
                chunk_hard_stop_geodesic_distance,
                goal_radius,
                deadband_m=premature_distance_deadband_m,
                tau_m=premature_distance_tau_m,
            )
        premature_stop_penalty_magnitude = (
            premature_stop_penalty + premature_stop_penalty_distance_add * premature_stop_distance_scale
        )
        premature_stop_nav_penalty_magnitude = (
            premature_stop_nav_penalty_weight + premature_stop_nav_distance_add * premature_stop_distance_scale
        )
        hard_stop_event_reward = 0.0
        if chunk_hard_stop_executed:
            hard_stop_event_reward = (
                correct_stop_bonus if chunk_hard_stop_correct else -premature_stop_penalty_magnitude
            )
        # STOP credit is normalized in its own GRPO stream.  Without a
        # navigation-side term, a low-motion premature STOP can avoid future
        # collision/progress costs and still look attractive in navigation.
        # Couple only incorrect executed STOPs back into navigation so correct
        # arrival STOPs remain unaffected.
        premature_stop_nav_penalty = (
            -premature_stop_nav_penalty_magnitude
            if active and chunk_hard_stop_executed and not chunk_hard_stop_correct
            else 0.0
        )
        if premature_stop_nav_penalty != 0.0:
            base_reward += premature_stop_nav_penalty
            navigation_active = True
        active_stop_well = stop_well
        stop_well_index = index
        if stop_well_enabled:
            active_stop_well = (
                deployment_stop_well if use_deployment_execution and deployment_stop_well is not None else stop_well
            )
            stop_well_index = 0 if use_deployment_execution else index
            stop_active = bool(
                (active and active_stop_well["active"][stop_well_index]) or hard_stop_event_reward != 0.0
            )
            stop_pre_clip = (
                float(active_stop_well["reward"][stop_well_index]) + hard_stop_event_reward if stop_active else 0.0
            )
            stop_reward = float(np.clip(stop_pre_clip, stop_reward_min, stop_reward_max)) if stop_active else 0.0
            chunk_stop_penalty = float(active_stop_well["exit_penalty"][stop_well_index])
        else:
            stop_active = False
            stop_pre_clip = 0.0
            stop_reward = 0.0
            chunk_stop_penalty = float(stop["terminal_stop_penalty"]) if index == stop_chunk else 0.0
        if not active:
            collision_hard_penalty = 0.0
            collision_soft_risk = 0.0
            collision_soft_penalty = 0.0
            collision_min_clearance_px = -1.0
            collision_penalty = 0.0
            collision_reward = 0.0
            base_reward = 0.0
            chunk_stop_penalty = 0.0
            navigation_active = False
            nav_reward = 0.0
            if stop_active:
                pre_clip = stop_pre_clip
                reward = float(np.clip(pre_clip, action_reward_min, action_reward_max))
            else:
                stop_pre_clip = 0.0
                stop_reward = 0.0
                pre_clip = 0.0
                reward = 0.0
        else:
            nav_reward = float(np.clip(base_reward, action_reward_min, action_reward_max)) if navigation_active else 0.0
            if stop_well_enabled:
                pre_clip = base_reward + stop_pre_clip
            else:
                pre_clip = base_reward + chunk_stop_penalty
            reward = float(np.clip(pre_clip, action_reward_min, action_reward_max))
        weighted_reward += weight * reward
        weighted_pre_clip += weight * pre_clip
        weighted_base += weight * base_reward
        weighted_yaw_path_score += weight * yaw_path_score
        weighted_yaw_rate_score += weight * yaw_rate_score
        weighted_yaw_gross_gt_score += weight * yaw_gross_gt_score
        weighted_yaw_path_penalty += weight * yaw_path_penalty
        weighted_yaw_rate_penalty += weight * yaw_rate_penalty
        weighted_yaw_gross_gt_penalty += weight * yaw_gross_gt_penalty
        weighted_yaw_total_penalty += weight * yaw_total_penalty
        weighted_ndtw += weight * chunk_ndtw
        weighted_ndtw_bonus += weight * ndtw_bonus
        weighted_geodesic_backtrack_distance += weight * geodesic_backtrack_distance
        weighted_geodesic_backtrack_score += weight * geodesic_backtrack_score
        weighted_geodesic_backtrack_penalty += weight * geodesic_backtrack_penalty
        weighted_collision_hard_penalty += weight * collision_hard_penalty
        weighted_collision_repeat_penalty += weight * collision_repeat_penalty
        weighted_collision_soft_risk += weight * collision_soft_risk
        weighted_collision_soft_penalty += weight * collision_soft_penalty
        weighted_collision_recovery_bonus += weight * chunk_collision_recovery_bonus
        weighted_collision_reward += weight * collision_reward

        chunk = {
            "collision": collision,
            "raw_collision": raw_collision,
            "active": bool(active),
            "navigation_active": navigation_active,
            "stop_active": stop_active,
            "navigation_reward": nav_reward,
            "flow_action_bonus": chunk_flow_action_bonus,
            "stop_reward": stop_reward,
            "post_collision_masked": bool(execution["collided"] and index > first_collision_chunk),
            "softspl": softspl,
            "geodesic_progress": geodesic_progress,
            "signed_progress": signed_progress,
            "path_efficiency": path_efficiency,
            "length_similarity": length_similarity,
            "route_deviation_score": route["deviation_score"],
            "route_deviation_penalty": route_deviation_penalty,
            "reverse_direction_score": route["reverse_direction_score"],
            "reverse_direction_penalty": reverse_direction_penalty,
            "ndtw": chunk_ndtw,
            "ndtw_bonus": ndtw_bonus,
            "geodesic_backtrack_distance_m": geodesic_backtrack_distance,
            "geodesic_backtrack_score": geodesic_backtrack_score,
            "geodesic_backtrack_penalty": geodesic_backtrack_penalty,
            "yaw_path_consistency_score": yaw_path_score,
            "yaw_path_consistency_score_mean": yaw_path_score_mean,
            "yaw_path_consistency_score_max": yaw_path_score_max,
            "yaw_path_error_deg": yaw_path_error_deg,
            "yaw_path_consistency_penalty": yaw_path_penalty,
            "yaw_rate_consistency_score": yaw_rate_score,
            "yaw_rate_consistency_score_mean": yaw_rate_score_mean,
            "yaw_rate_consistency_score_max": yaw_rate_score_max,
            "yaw_rate_error_deg": yaw_rate_error_deg,
            "yaw_rate_consistency_penalty": yaw_rate_penalty,
            "yaw_gross_gt_score": yaw_gross_gt_score,
            "yaw_gross_gt_score_mean": yaw_gross_gt_score_mean,
            "yaw_gross_gt_score_max": yaw_gross_gt_score_max,
            "yaw_spike_max_mix": yaw_spike_max_mix,
            "yaw_gross_gt_fraction": yaw_gross_gt_fraction,
            "yaw_gross_gt_error_deg": yaw_gross_gt_error_deg,
            "yaw_gross_gt_penalty": yaw_gross_gt_penalty,
            "yaw_total_penalty_pre_cap": yaw_total_penalty_pre_cap,
            "yaw_total_penalty": yaw_total_penalty,
            "yaw_moving_steps": yaw_moving_steps,
            "yaw_pure_yaw_steps": yaw_pure_yaw_steps,
            "hard_stop_event_reward": hard_stop_event_reward,
            "premature_stop_nav_penalty": premature_stop_nav_penalty,
            "hard_stop_emitted": chunk_hard_stop_emitted,
            "hard_stop_executed": chunk_hard_stop_executed,
            "hard_stop_step": chunk_hard_stop_step,
            "hard_stop_local_step": chunk_hard_stop_local_step,
            "hard_stop_distance_m": chunk_hard_stop_distance,
            "hard_stop_geodesic_distance_m": chunk_hard_stop_geodesic_distance,
            "hard_stop_correct": chunk_hard_stop_correct,
            "premature_stop_distance_excess_m": premature_stop_distance_excess_m,
            "premature_stop_distance_scale": premature_stop_distance_scale,
            "premature_stop_penalty_magnitude": premature_stop_penalty_magnitude,
            "premature_stop_nav_penalty_magnitude": (premature_stop_nav_penalty_magnitude),
            "stop_trigger_xy_path_length_m": float(stop_trigger_xy_path_lengths[index]),
            "stop_trigger_xy_net_displacement_m": float(stop_trigger_xy_net_displacements[index]),
            "goal_score": goal_score,
            "goal_potential_start": goal_potential_start,
            "goal_potential_end": goal_potential_end,
            "goal_potential_delta": goal_potential_delta,
            "goal_credit": goal_credit,
            "goal_entry": goal_entry,
            "goal_entry_bonus": goal_entry_bonus,
            "geodesic_start_distance": d_start,
            "geodesic_end_distance": d_end,
            "predicted_path_length": pred_length,
            "reference_path_length": gt_length,
            "executed_predicted_path_length": executed_pred_length,
            "navigation_predicted_path_length": nav_pred_length,
            "navigation_reference_path_length": nav_gt_length,
            "path_length_ratio": length_ratio,
            "path_length_overrun_ratio": max(length_ratio - 1.0, 0.0),
            "collision_penalty": collision_penalty,
            "collision_hard_penalty": collision_hard_penalty,
            "collision_repeat_penalty": collision_repeat_penalty,
            "collision_soft_risk": collision_soft_risk,
            "collision_soft_penalty": collision_soft_penalty,
            "collision_recovery_bonus": chunk_collision_recovery_bonus,
            "collision_reward": collision_reward,
            "deployment_collision": bool(use_deployment_execution and bool(deployment_execution["collided"])),
            "deployment_collision_count": int(
                deployment_execution["collision_count"] if use_deployment_execution else 0
            ),
            "recovery_grace_steps": int(
                np.count_nonzero(recovery_grace_mask[: max(nav_end - start, 0)]) if use_deployment_execution else 0
            ),
            "collision_active": bool(active),
            "collision_min_clearance_px": collision_min_clearance_px,
            "geodesic_start_snapped": d_start_snapped,
            "geodesic_end_snapped": d_end_snapped,
            "evaluated_actions": max(nav_end - start, 0),
        }
        chunks.append(chunk)
        prefix = f"action_chunk_{index}"
        metrics.update(
            {
                f"{prefix}_reward": reward,
                f"{prefix}_reward_pre_clip": float(pre_clip),
                f"{prefix}_nav_reward": float(nav_reward),
                f"{prefix}_nav_active": float(navigation_active),
                f"{prefix}_flow_action_bonus": float(chunk_flow_action_bonus),
                f"{prefix}_stop_reward": float(stop_reward),
                f"{prefix}_stop_reward_pre_clip": float(stop_pre_clip),
                f"{prefix}_stop_active": float(stop_active),
                f"{prefix}_stop_energy": float(active_stop_well["energy"][stop_well_index]),
                f"{prefix}_stop_xy_motion": float(active_stop_well["xy_motion"][stop_well_index]),
                f"{prefix}_stop_yaw_motion": float(active_stop_well["yaw_motion"][stop_well_index]),
                f"{prefix}_weight": float(weight),
                f"{prefix}_softspl": float(softspl),
                f"{prefix}_geodesic_progress": float(geodesic_progress),
                f"{prefix}_signed_progress": float(signed_progress),
                f"{prefix}_path_efficiency": float(path_efficiency),
                f"{prefix}_length_similarity": float(length_similarity),
                f"{prefix}_route_deviation_score": float(route["deviation_score"]),
                f"{prefix}_route_deviation_rms_m": float(route["deviation_rms_m"]),
                f"{prefix}_route_deviation_max_m": float(route["deviation_max_m"]),
                f"{prefix}_route_deviation_penalty": float(route_deviation_penalty),
                f"{prefix}_reverse_direction_score": float(route["reverse_direction_score"]),
                f"{prefix}_reverse_direction_penalty": float(reverse_direction_penalty),
                f"{prefix}_ndtw": float(chunk_ndtw),
                f"{prefix}_ndtw_bonus": float(ndtw_bonus),
                f"{prefix}_geodesic_backtrack_distance_m": float(geodesic_backtrack_distance),
                f"{prefix}_geodesic_backtrack_score": float(geodesic_backtrack_score),
                f"{prefix}_geodesic_backtrack_penalty": float(geodesic_backtrack_penalty),
                f"{prefix}_yaw_path_consistency_score": float(yaw_path_score),
                f"{prefix}_yaw_path_consistency_score_mean": float(yaw_path_score_mean),
                f"{prefix}_yaw_path_consistency_score_max": float(yaw_path_score_max),
                f"{prefix}_yaw_path_error_deg": float(yaw_path_error_deg),
                f"{prefix}_yaw_path_consistency_penalty": float(yaw_path_penalty),
                f"{prefix}_yaw_rate_consistency_score": float(yaw_rate_score),
                f"{prefix}_yaw_rate_consistency_score_mean": float(yaw_rate_score_mean),
                f"{prefix}_yaw_rate_consistency_score_max": float(yaw_rate_score_max),
                f"{prefix}_yaw_rate_error_deg": float(yaw_rate_error_deg),
                f"{prefix}_yaw_rate_consistency_penalty": float(yaw_rate_penalty),
                f"{prefix}_yaw_gross_gt_score": float(yaw_gross_gt_score),
                f"{prefix}_yaw_gross_gt_score_mean": float(yaw_gross_gt_score_mean),
                f"{prefix}_yaw_gross_gt_score_max": float(yaw_gross_gt_score_max),
                f"{prefix}_yaw_spike_max_mix": float(yaw_spike_max_mix),
                f"{prefix}_yaw_gross_gt_fraction": float(yaw_gross_gt_fraction),
                f"{prefix}_yaw_gross_gt_error_deg": float(yaw_gross_gt_error_deg),
                f"{prefix}_yaw_gross_gt_penalty": float(yaw_gross_gt_penalty),
                f"{prefix}_yaw_total_penalty_pre_cap": float(yaw_total_penalty_pre_cap),
                f"{prefix}_yaw_total_penalty": float(yaw_total_penalty),
                f"{prefix}_yaw_moving_steps": float(yaw_moving_steps),
                f"{prefix}_yaw_pure_yaw_steps": float(yaw_pure_yaw_steps),
                f"{prefix}_hard_stop_event_reward": float(hard_stop_event_reward),
                f"{prefix}_premature_stop_nav_penalty": float(premature_stop_nav_penalty),
                f"{prefix}_hard_stop_emitted": float(chunk_hard_stop_emitted),
                f"{prefix}_hard_stop_executed": float(chunk_hard_stop_executed),
                f"{prefix}_hard_stop_step": float(chunk_hard_stop_step),
                f"{prefix}_hard_stop_local_step": float(chunk_hard_stop_local_step),
                f"{prefix}_hard_stop_distance_m": float(chunk_hard_stop_distance),
                f"{prefix}_hard_stop_geodesic_distance_m": float(
                    chunk_hard_stop_geodesic_distance if np.isfinite(chunk_hard_stop_geodesic_distance) else -1.0
                ),
                f"{prefix}_hard_stop_correct": float(chunk_hard_stop_correct),
                f"{prefix}_premature_stop_distance_excess_m": float(
                    premature_stop_distance_excess_m if np.isfinite(premature_stop_distance_excess_m) else -1.0
                ),
                f"{prefix}_premature_stop_distance_scale": float(premature_stop_distance_scale),
                f"{prefix}_premature_stop_penalty_magnitude": float(premature_stop_penalty_magnitude),
                f"{prefix}_premature_stop_nav_penalty_magnitude": float(premature_stop_nav_penalty_magnitude),
                f"{prefix}_stop_trigger_xy_path_length_m": float(stop_trigger_xy_path_lengths[index]),
                f"{prefix}_stop_trigger_xy_net_displacement_m": float(stop_trigger_xy_net_displacements[index]),
                f"{prefix}_goal_score": float(goal_score),
                f"{prefix}_goal_potential_start": float(goal_potential_start),
                f"{prefix}_goal_potential_end": float(goal_potential_end),
                f"{prefix}_goal_potential_delta": float(goal_potential_delta),
                f"{prefix}_goal_credit": float(goal_credit),
                f"{prefix}_goal_entry": float(goal_entry),
                f"{prefix}_goal_entry_bonus": float(goal_entry_bonus),
                f"{prefix}_collision": float(bool(collision["collided"])),
                f"{prefix}_raw_collision": float(bool(raw_collision["collided"])),
                f"{prefix}_collision_penalty": float(collision_penalty),
                f"{prefix}_collision_hard_penalty": float(collision_hard_penalty),
                f"{prefix}_collision_repeat_penalty": float(collision_repeat_penalty),
                f"{prefix}_collision_soft_risk": float(collision_soft_risk),
                f"{prefix}_collision_soft_penalty": float(collision_soft_penalty),
                f"{prefix}_collision_recovery_bonus": float(chunk_collision_recovery_bonus),
                f"{prefix}_collision_reward": float(collision_reward),
                f"{prefix}_deployment_collision": float(
                    use_deployment_execution and bool(deployment_execution["collided"])
                ),
                f"{prefix}_deployment_collision_count": float(
                    int(deployment_execution["collision_count"]) if use_deployment_execution else 0
                ),
                f"{prefix}_recovery_grace_steps": float(
                    np.count_nonzero(recovery_grace_mask[: max(nav_end - start, 0)]) if use_deployment_execution else 0
                ),
                f"{prefix}_collision_active": float(active),
                f"{prefix}_collision_min_clearance_px": float(collision_min_clearance_px),
                f"{prefix}_active": float(active),
                f"{prefix}_post_collision_masked": float(bool(execution["collided"]) and index > first_collision_chunk),
                f"{prefix}_post_stop_masked": float(
                    not deploy_stop_semantics_enabled and hard_stop_executed and index > hard_stop_chunk
                ),
                f"{prefix}_geodesic_start_distance": d_start if np.isfinite(d_start) else -1.0,
                f"{prefix}_geodesic_end_distance": d_end if np.isfinite(d_end) else -1.0,
                f"{prefix}_geodesic_reachable": float(np.isfinite(d_start) and np.isfinite(d_end)),
                f"{prefix}_pred_path_length": float(pred_length),
                f"{prefix}_gt_path_length": float(gt_length),
                f"{prefix}_raw_pred_path_length": float(pred_length),
                f"{prefix}_full_gt_path_length": float(gt_length),
                f"{prefix}_executed_pred_path_length": float(executed_pred_length),
                f"{prefix}_nav_pred_path_length": float(nav_pred_length),
                f"{prefix}_nav_gt_path_length": float(nav_gt_length),
                f"{prefix}_path_length_ratio": float(length_ratio),
                f"{prefix}_stop_penalty": float(chunk_stop_penalty),
            }
        )

    metrics.update(
        {
            "action_yaw_path_consistency_score": float(weighted_yaw_path_score),
            "action_yaw_rate_consistency_score": float(weighted_yaw_rate_score),
            "action_yaw_gross_gt_score": float(weighted_yaw_gross_gt_score),
            "action_yaw_path_consistency_penalty": float(weighted_yaw_path_penalty),
            "action_yaw_rate_consistency_penalty": float(weighted_yaw_rate_penalty),
            "action_yaw_gross_gt_penalty": float(weighted_yaw_gross_gt_penalty),
            "action_yaw_total_penalty": float(weighted_yaw_total_penalty),
            "action_yaw_score_mode_id": float(yaw_score_mode == "linear_guard"),
            "action_yaw_spike_max_mix": float(yaw_spike_max_mix),
            "action_yaw_total_penalty_cap": float(yaw_total_penalty_cap),
            "action_premature_stop_nav_penalty_weight": float(premature_stop_nav_penalty_weight),
            "action_ndtw": float(weighted_ndtw),
            "action_ndtw_bonus": float(weighted_ndtw_bonus),
            "action_geodesic_backtrack_distance_m": float(weighted_geodesic_backtrack_distance),
            "action_geodesic_backtrack_score": float(weighted_geodesic_backtrack_score),
            "action_geodesic_backtrack_penalty": float(weighted_geodesic_backtrack_penalty),
            "action_collision_hard_penalty": float(weighted_collision_hard_penalty),
            "action_collision_repeat_penalty": float(weighted_collision_repeat_penalty),
            "action_collision_soft_risk": float(weighted_collision_soft_risk),
            "action_collision_soft_penalty": float(weighted_collision_soft_penalty),
            "action_collision_recovery_bonus": float(weighted_collision_recovery_bonus),
            "action_collision_reward": float(weighted_collision_reward),
            "action_collision_min_clearance_px": float(min(active_clearances_px) if active_clearances_px else -1.0),
        }
    )
    return {
        # Reporting includes terminal safety, while policy credit consumes it
        # as an independently normalized advantage fused into the single
        # action PPO objective.  Keeping it outside per-chunk clipping ensures
        # every collision retains a marginal cost.
        "action_reward": float(weighted_reward + terminal_safety_reward),
        "action_reward_pre_clip": float(weighted_pre_clip + terminal_safety_reward),
        "base_action_reward": float(weighted_base),
        "stop": stop,
        "chunks": chunks,
        "metrics": metrics,
    }


def _empty_flow_action_diagnostics() -> dict[str, float | bool]:
    return {
        "flow_action_score": 0.0,
        "flow_action_raw_score": 0.0,
        "flow_action_valid": False,
        "flow_action_confidence": 0.0,
        "flow_action_translation_cosine": 0.0,
        "flow_action_translation_score": 0.0,
        "flow_action_yaw_score": 0.0,
        "flow_action_standardized_cosine": 0.0,
        "flow_action_actual_dx": 0.0,
        "flow_action_actual_dy": 0.0,
        "flow_action_actual_dyaw": 0.0,
        "flow_action_inferred_dx": 0.0,
        "flow_action_inferred_dy": 0.0,
        "flow_action_inferred_dyaw": 0.0,
        "flow_action_flow_confidence": 0.0,
        "flow_action_fb_error_px": 0.0,
        "flow_action_texture_gradient": 0.0,
        "flow_action_flow_p50_px": 0.0,
        "flow_action_flow_p75_px": 0.0,
        "flow_action_flow_p90_px": 0.0,
    }
