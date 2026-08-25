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

"""Video loading, diagnostics, route, yaw, and stop-well helpers for Stage-3."""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
import torch

from verl_omni.utils.action_chunk_credit import normalized_action_chunk_weights

_GT_VIDEO_CACHE_LOCK = threading.Lock()


def _decode_video_clip(path: str, start: int, count: int) -> np.ndarray:
    with _GT_VIDEO_CACHE_LOCK:
        return _decode_video_clip_cached(path, start, count)


@lru_cache(maxsize=2)
def _decode_video_clip_cached(path: str, start: int, count: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open GT video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    try:
        for frame_index in range(start, start + count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode GT video {path} at frame {frame_index}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames).astype(np.uint8, copy=False)


def _load_gt_video(path: str, start: int, count: int) -> torch.Tensor:
    if start < 0 or count <= 1:
        raise ValueError(f"invalid GT clip start/count: {start}/{count}")
    # Copy is intentionally avoided. Cache is bounded to two uint8 clips; the
    # tensor conversion/resize happens inside the lightweight reward function.
    clip = _decode_video_clip(path, start, count)
    return torch.from_numpy(np.asarray(clip)).permute(0, 3, 1, 2).float().div_(255.0)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _premature_stop_distance_terms(
    distance_m: float,
    goal_radius_m: float,
    *,
    deadband_m: float,
    tau_m: float,
) -> tuple[float, float]:
    """Return excess distance and a smooth [0, 1] premature-STOP scale.

    Distance is measured to the goal centre, while the penalty starts outside
    the goal region plus a small boundary deadband.  A disconnected geodesic
    is treated as maximally far; NaN is rejected so missing map data cannot
    silently weaken STOP supervision.
    """

    if goal_radius_m < 0 or deadband_m < 0 or tau_m <= 0:
        raise ValueError("premature-STOP goal radius/deadband must be non-negative and tau must be positive")
    distance = float(distance_m)
    if np.isnan(distance) or distance < 0:
        raise ValueError(f"premature-STOP distance must be non-negative or +inf, got {distance!r}")
    if np.isposinf(distance):
        return float("inf"), 1.0
    if not np.isfinite(distance):
        raise ValueError(f"premature-STOP distance must be finite or +inf, got {distance!r}")
    excess = max(distance - float(goal_radius_m) - float(deadband_m), 0.0)
    scale = -float(np.expm1(-excess / float(tau_m)))
    return float(excess), float(np.clip(scale, 0.0, 1.0))


def _compact_reward_metrics(metrics: dict) -> dict:
    """Keep policy-relevant metrics while retaining opt-in full diagnostics."""

    if not _env_bool("WAM_REWARD_METRICS_COMPACT", False):
        return metrics
    core_keys = {
        "score",
        "visual_reward",
        "action_reward",
        "event_val_active",
        "event_val_collision_precursor",
        "event_val_premature_stop_risk",
        "event_val_premature_stop_near",
        "event_val_premature_stop_far",
        "event_val_required_stop",
        "event_val_required_stop_core",
        "event_val_required_stop_mid",
        "event_val_required_stop_boundary",
        "event_val_near_goal_continue",
        "event_val_expected_stop",
        "event_val_distance_to_goal_m",
        "vision_reward_base",
        "vision_ms_ssim",
        "vision_degenerate",
        "vision_degradation_factor",
        "vision_pred_motion",
        "vision_gt_motion",
        "flow_action_bonus",
        "flow_action_action_bonus",
        "flow_action_score",
        "flow_action_valid",
        "flow_action_confidence",
        "action_reward_pre_clip",
        "action_invalid",
        "action_pred_path_length",
        "action_gt_path_length",
        "action_path_length_ratio",
        "action_geodesic_reachable",
        "action_any_collision",
        "action_first_collision_step",
        "action_first_collision_chunk",
        "action_collision_out_of_bounds",
        "action_collision_hard_penalty",
        "action_collision_soft_risk",
        "action_collision_soft_penalty",
        "action_collision_min_clearance_px",
        "action_collision_soft_enabled",
        "action_collision_hard_margin_px",
        "action_collision_soft_margin_px",
        "action_collision_soft_penalty_weight",
        "action_collision_recovery_enabled",
        "action_deployment_collision_margin_px",
        "action_collision_repeat_penalty_weight",
        "action_collision_repeat_penalty_cap_count",
        "action_collision_repeat_penalty",
        "action_collision_recovery_bonus_weight",
        "action_collision_recovery_bonus",
        "action_collision_recovery_clearance_px",
        "action_collision_recovery_min_escape_m",
        "action_collision_recovery_full_escape_m",
        "action_collision_recovery_tail_free_steps",
        "action_collision_recovery_grace_enabled",
        "action_chunk_0_deployment_collision",
        "action_chunk_0_deployment_collision_count",
        "action_chunk_0_deployment_first_collision_step",
        "action_chunk_0_deployment_collision_out_of_bounds",
        "action_chunk_0_collision_recovered",
        "action_chunk_0_collision_recovery_eligible",
        "action_chunk_0_collision_recovery_score",
        "action_chunk_0_collision_recovery_bonus",
        "action_chunk_0_recovery_distance_m",
        "action_chunk_0_recovery_clearance_px",
        "action_chunk_0_post_collision_premature_stop",
        # Keep the flag even when false: the trainer uses it to decide whether
        # collision is part of navigation or an independent reward stream.
        "action_collision_credit_enabled",
        "action_collision_reward",
        "action_goal_entry_chunk",
        # This is a policy-control field, not merely a diagnostic.  The trainer
        # uses it to split navigation and STOP rewards into independent GRPO
        # advantages, so compact logging must never discard it.
        "action_stop_credit_enabled",
        "action_yaw_path_consistency_penalty",
        "action_yaw_rate_consistency_penalty",
        "action_yaw_gross_gt_penalty",
        "action_yaw_total_penalty",
        "action_yaw_score_mode_id",
        "action_yaw_spike_max_mix",
        "action_yaw_total_penalty_cap",
        "action_premature_stop_nav_penalty_weight",
        "action_premature_stop_distance_scaling_enabled",
        "action_premature_stop_distance_deadband_m",
        "action_premature_stop_distance_tau_m",
        "action_premature_stop_penalty_distance_add",
        "action_premature_stop_nav_distance_add",
        "action_hard_stop_geodesic_distance_m",
        "action_chunk_motion_stop_enabled",
        "action_chunk_motion_stop_threshold_m",
        "deployment_stop_start_inside_goal",
        "deployment_stop_uses_chunk_motion",
        "deployment_stop_chunk_motion_threshold_m",
        "deployment_stop_chunk_motion_metric_id",
        "deployment_stop_reached",
        "deployment_stop_emitted",
        "deployment_stop_success",
        "deployment_stop_premature",
        "deployment_stop_reached_without_stop",
        "deployment_stop_first_step",
        "deployment_stop_terminal_distance",
        "stop_val_candidate",
        "stop_val_collection_reached",
        "stop_val_band_id",
    }
    chunk_suffixes = {
        "nav_reward",
        "nav_active",
        "flow_action_bonus",
        "stop_reward",
        "stop_active",
        "signed_progress",
        "length_similarity",
        "collision",
        "collision_soft_risk",
        "collision_repeat_penalty",
        "collision_recovery_bonus",
        "deployment_collision",
        "deployment_collision_count",
        "recovery_grace_steps",
        "collision_reward",
        "collision_active",
        "collision_min_clearance_px",
        "goal_entry",
        "goal_entry_bonus",
        "stop_energy",
        "stop_xy_motion",
        "stop_yaw_motion",
        "hard_stop_event_reward",
        "premature_stop_nav_penalty",
        "hard_stop_emitted",
        "hard_stop_executed",
        "hard_stop_step",
        "hard_stop_local_step",
        "hard_stop_distance_m",
        "hard_stop_geodesic_distance_m",
        "hard_stop_correct",
        "premature_stop_distance_excess_m",
        "premature_stop_distance_scale",
        "premature_stop_penalty_magnitude",
        "premature_stop_nav_penalty_magnitude",
        "stop_trigger_xy_path_length_m",
        "yaw_path_consistency_score_mean",
        "yaw_path_consistency_score_max",
        "yaw_rate_consistency_score_mean",
        "yaw_rate_consistency_score_max",
        "yaw_gross_gt_score_mean",
        "yaw_gross_gt_score_max",
        "yaw_spike_max_mix",
    }

    compact = {}
    for key, value in metrics.items():
        if key.startswith("_exploration_") or key in core_keys:
            compact[key] = value
            continue
        if key.startswith("action_chunk_"):
            remainder = key[len("action_chunk_") :]
            _, separator, suffix = remainder.partition("_")
            if separator and suffix in chunk_suffixes:
                compact[key] = value
    return compact


def _empty_chunk_metrics() -> dict[str, float]:
    weights = normalized_action_chunk_weights()
    terminal_safety_enabled = _env_bool("WAM_TERMINAL_SAFETY_ADVANTAGE_ENABLED", False)
    terminal_hard_weight = _env_float(
        "WAM_TERMINAL_SAFETY_HARD_WEIGHT",
        _env_float("WAM_COLLISION_PENALTY_WEIGHT", 0.10),
    )
    metrics: dict[str, float] = {
        "action_chunk_count": float(len(weights)),
        "action_chunk_reward_mode_id": -1.0,
        "action_yaw_credit_enabled": float(_env_bool("WAM_YAW_CREDIT_ENABLED", False)),
        "action_goal_credit_uses_potential_delta": float(_env_bool("WAM_GOAL_SCORE_USE_POTENTIAL_DELTA", False)),
        "action_goal_entry_bonus_weight": float(_env_float("WAM_GOAL_ENTRY_BONUS", 0.0)),
        "action_goal_entry_chunk": -1.0,
        "action_yaw_path_consistency_score": 0.0,
        "action_yaw_rate_consistency_score": 0.0,
        "action_yaw_gross_gt_score": 0.0,
        "action_yaw_path_consistency_penalty": 0.0,
        "action_yaw_rate_consistency_penalty": 0.0,
        "action_yaw_gross_gt_penalty": 0.0,
        "action_yaw_total_penalty": 0.0,
        "action_yaw_score_mode_id": float(
            os.environ.get("WAM_YAW_SCORE_MODE", "cosine").strip().lower() == "linear_guard"
        ),
        "action_yaw_spike_max_mix": float(_env_float("WAM_YAW_SPIKE_MAX_MIX", 0.50)),
        "action_yaw_total_penalty_cap": float(_env_float("WAM_YAW_TOTAL_PENALTY_CAP", 1.0)),
        "action_premature_stop_nav_penalty_weight": float(_env_float("WAM_PREMATURE_STOP_NAV_PENALTY_WEIGHT", 0.0)),
        "action_premature_stop_distance_scaling_enabled": float(
            _env_bool("WAM_PREMATURE_STOP_DISTANCE_SCALING_ENABLED", False)
        ),
        "action_premature_stop_distance_deadband_m": float(_env_float("WAM_PREMATURE_STOP_DISTANCE_DEADBAND_M", 0.25)),
        "action_premature_stop_distance_tau_m": float(_env_float("WAM_PREMATURE_STOP_DISTANCE_TAU_M", 2.0)),
        "action_premature_stop_penalty_distance_add": float(
            _env_float("WAM_PREMATURE_STOP_PENALTY_DISTANCE_ADD", 0.25)
        ),
        "action_premature_stop_nav_distance_add": float(_env_float("WAM_PREMATURE_STOP_NAV_DISTANCE_ADD", 0.75)),
        "action_ndtw": 0.0,
        "action_ndtw_bonus": 0.0,
        "action_ndtw_weight": float(_env_float("WAM_NDTW_WEIGHT", 0.0)),
        "action_geodesic_backtrack_distance_m": 0.0,
        "action_geodesic_backtrack_score": 0.0,
        "action_geodesic_backtrack_penalty": 0.0,
        "action_geodesic_backtrack_weight": float(_env_float("WAM_GEODESIC_BACKTRACK_WEIGHT", 0.0)),
        "action_stop_credit_enabled": float(_env_bool("WAM_STOP_WELL_ENABLED", False)),
        "action_deploy_stop_semantics_enabled": float(_env_bool("WAM_DEPLOY_STOP_SEMANTICS_ENABLED", False)),
        "action_chunk_motion_stop_enabled": float(_env_bool("WAM_CHUNK_MOTION_STOP_ENABLED", False)),
        "action_chunk_motion_stop_threshold_m": float(_env_float("WAM_CHUNK_MOTION_STOP_THRESHOLD_M", 0.15)),
        "action_chunk_motion_stop_metric_id": float(
            os.environ.get("WAM_CHUNK_MOTION_STOP_METRIC", "net_displacement").strip().lower() == "path_length"
        ),
        "action_chunk_motion_stop_chunk": -1.0,
        "action_chunk_motion_stop_emitted": 0.0,
        "action_hard_stop_emitted": 0.0,
        "action_hard_stop_executed": 0.0,
        "action_hard_stop_correct": 0.0,
        "action_hard_stop_premature": 0.0,
        "action_hard_stop_step": -1.0,
        "action_hard_stop_chunk": -1.0,
        "action_hard_stop_distance_m": -1.0,
        "action_hard_stop_geodesic_distance_m": -1.0,
        "action_collision_stop_enabled": float(_env_bool("WAM_COLLISION_STOP_ENABLED", False)),
        "action_collision_recovery_enabled": float(_env_bool("WAM_COLLISION_RECOVERY_ENABLED", False)),
        "action_deployment_collision_margin_px": float(_env_int("WAM_DEPLOYMENT_COLLISION_MARGIN_PX", 4)),
        "action_collision_repeat_penalty_weight": float(_env_float("WAM_COLLISION_REPEAT_PENALTY_WEIGHT", 0.10)),
        "action_collision_repeat_penalty_cap_count": float(_env_int("WAM_COLLISION_REPEAT_PENALTY_CAP_COUNT", 2)),
        "action_collision_repeat_penalty": 0.0,
        "action_collision_recovery_bonus_weight": float(_env_float("WAM_COLLISION_RECOVERY_BONUS_WEIGHT", 0.10)),
        "action_collision_recovery_bonus": 0.0,
        "action_collision_recovery_clearance_px": float(_env_float("WAM_COLLISION_RECOVERY_CLEARANCE_PX", 6.0)),
        "action_collision_recovery_min_escape_m": float(_env_float("WAM_COLLISION_RECOVERY_MIN_ESCAPE_M", 0.20)),
        "action_collision_recovery_full_escape_m": float(_env_float("WAM_COLLISION_RECOVERY_FULL_ESCAPE_M", 0.40)),
        "action_collision_recovery_tail_free_steps": float(_env_int("WAM_COLLISION_RECOVERY_TAIL_FREE_STEPS", 2)),
        "action_collision_recovery_grace_enabled": float(_env_bool("WAM_COLLISION_RECOVERY_GRACE_ENABLED", True)),
        "action_chunk_0_deployment_collision": 0.0,
        "action_chunk_0_deployment_collision_count": 0.0,
        "action_chunk_0_deployment_first_collision_step": -1.0,
        "action_chunk_0_deployment_collision_out_of_bounds": 0.0,
        "action_chunk_0_collision_recovered": 0.0,
        "action_chunk_0_collision_recovery_eligible": 0.0,
        "action_chunk_0_collision_recovery_score": 0.0,
        "action_chunk_0_collision_recovery_bonus": 0.0,
        "action_chunk_0_recovery_distance_m": 0.0,
        "action_chunk_0_recovery_clearance_px": -1.0,
        "action_chunk_0_post_collision_premature_stop": 0.0,
        "action_any_collision": 0.0,
        "action_first_collision_step": -1.0,
        "action_first_collision_chunk": -1.0,
        "action_collision_out_of_bounds": 0.0,
        "action_collision_hard_penalty": 0.0,
        "action_collision_soft_risk": 0.0,
        "action_collision_soft_penalty": 0.0,
        "action_collision_min_clearance_px": -1.0,
        "action_collision_soft_enabled": float(_env_bool("WAM_COLLISION_SOFT_ENABLED", False)),
        "action_collision_hard_margin_px": float(_env_int("WAM_OCCUPANCY_MARGIN_PX", 2)),
        "action_collision_soft_margin_px": float(_env_int("WAM_COLLISION_SOFT_MARGIN_PX", 4)),
        "action_collision_soft_penalty_weight": float(_env_float("WAM_COLLISION_SOFT_PENALTY_WEIGHT", 0.20)),
        "action_collision_credit_enabled": float(_env_bool("WAM_COLLISION_CREDIT_ENABLED", False)),
        # This helper is used for invalid action tensors.  Treat malformed
        # output as a terminal safety failure instead of silently granting the
        # zero-cost value of a valid collision-free trajectory.
        "action_terminal_safety_enabled": float(terminal_safety_enabled),
        "action_terminal_safety_reward": float(-terminal_hard_weight if terminal_safety_enabled else 0.0),
        "action_terminal_safety_hard_event": float(terminal_safety_enabled),
        "action_terminal_safety_hard_penalty": float(-terminal_hard_weight if terminal_safety_enabled else 0.0),
        "action_terminal_safety_soft_risk": 0.0,
        "action_terminal_safety_soft_max_risk": 0.0,
        "action_terminal_safety_soft_mean_risk": 0.0,
        "action_terminal_safety_soft_penalty": 0.0,
        "action_terminal_safety_min_clearance_px": -1.0,
        "action_collision_reward": 0.0,
    }
    for index, weight in enumerate(weights):
        prefix = f"action_chunk_{index}"
        metrics.update(
            {
                f"{prefix}_reward": 0.0,
                f"{prefix}_reward_pre_clip": 0.0,
                f"{prefix}_nav_reward": 0.0,
                f"{prefix}_nav_active": 0.0,
                f"{prefix}_flow_action_bonus": 0.0,
                f"{prefix}_stop_reward": 0.0,
                f"{prefix}_stop_reward_pre_clip": 0.0,
                f"{prefix}_stop_active": 0.0,
                f"{prefix}_stop_energy": 0.0,
                f"{prefix}_stop_xy_motion": 0.0,
                f"{prefix}_stop_yaw_motion": 0.0,
                f"{prefix}_weight": float(weight),
                f"{prefix}_softspl": 0.0,
                f"{prefix}_geodesic_progress": 0.0,
                f"{prefix}_signed_progress": 0.0,
                f"{prefix}_path_efficiency": 0.0,
                f"{prefix}_length_similarity": 0.0,
                f"{prefix}_route_deviation_score": 0.0,
                f"{prefix}_route_deviation_rms_m": 0.0,
                f"{prefix}_route_deviation_max_m": 0.0,
                f"{prefix}_route_deviation_penalty": 0.0,
                f"{prefix}_reverse_direction_score": 0.0,
                f"{prefix}_reverse_direction_penalty": 0.0,
                f"{prefix}_ndtw": 0.0,
                f"{prefix}_ndtw_bonus": 0.0,
                f"{prefix}_geodesic_backtrack_distance_m": 0.0,
                f"{prefix}_geodesic_backtrack_score": 0.0,
                f"{prefix}_geodesic_backtrack_penalty": 0.0,
                f"{prefix}_yaw_path_consistency_score": 0.0,
                f"{prefix}_yaw_path_error_deg": 0.0,
                f"{prefix}_yaw_path_consistency_penalty": 0.0,
                f"{prefix}_yaw_rate_consistency_score": 0.0,
                f"{prefix}_yaw_rate_error_deg": 0.0,
                f"{prefix}_yaw_rate_consistency_penalty": 0.0,
                f"{prefix}_yaw_gross_gt_score": 0.0,
                f"{prefix}_yaw_gross_gt_fraction": 0.0,
                f"{prefix}_yaw_gross_gt_error_deg": 0.0,
                f"{prefix}_yaw_gross_gt_penalty": 0.0,
                f"{prefix}_yaw_total_penalty_pre_cap": 0.0,
                f"{prefix}_yaw_total_penalty": 0.0,
                f"{prefix}_yaw_moving_steps": 0.0,
                f"{prefix}_yaw_pure_yaw_steps": 0.0,
                f"{prefix}_hard_stop_event_reward": 0.0,
                f"{prefix}_premature_stop_nav_penalty": 0.0,
                f"{prefix}_hard_stop_emitted": 0.0,
                f"{prefix}_hard_stop_executed": 0.0,
                f"{prefix}_hard_stop_step": -1.0,
                f"{prefix}_hard_stop_local_step": -1.0,
                f"{prefix}_hard_stop_distance_m": -1.0,
                f"{prefix}_hard_stop_geodesic_distance_m": -1.0,
                f"{prefix}_hard_stop_correct": 0.0,
                f"{prefix}_premature_stop_distance_excess_m": 0.0,
                f"{prefix}_premature_stop_distance_scale": 0.0,
                f"{prefix}_premature_stop_penalty_magnitude": float(_env_float("WAM_PREMATURE_STOP_PENALTY", 0.50)),
                f"{prefix}_premature_stop_nav_penalty_magnitude": float(
                    _env_float("WAM_PREMATURE_STOP_NAV_PENALTY_WEIGHT", 0.0)
                ),
                f"{prefix}_stop_trigger_xy_path_length_m": 0.0,
                f"{prefix}_stop_trigger_xy_net_displacement_m": 0.0,
                f"{prefix}_goal_score": 0.0,
                f"{prefix}_goal_potential_start": 0.0,
                f"{prefix}_goal_potential_end": 0.0,
                f"{prefix}_goal_potential_delta": 0.0,
                f"{prefix}_goal_credit": 0.0,
                f"{prefix}_goal_entry": 0.0,
                f"{prefix}_goal_entry_bonus": 0.0,
                f"{prefix}_collision": 0.0,
                f"{prefix}_raw_collision": 0.0,
                f"{prefix}_collision_penalty": 0.0,
                f"{prefix}_collision_hard_penalty": 0.0,
                f"{prefix}_collision_repeat_penalty": 0.0,
                f"{prefix}_collision_soft_risk": 0.0,
                f"{prefix}_collision_soft_penalty": 0.0,
                f"{prefix}_collision_recovery_bonus": 0.0,
                f"{prefix}_collision_reward": 0.0,
                f"{prefix}_deployment_collision": 0.0,
                f"{prefix}_deployment_collision_count": 0.0,
                f"{prefix}_recovery_grace_steps": 0.0,
                f"{prefix}_collision_active": 0.0,
                f"{prefix}_collision_min_clearance_px": -1.0,
                f"{prefix}_active": 1.0,
                f"{prefix}_post_collision_masked": 0.0,
                f"{prefix}_post_stop_masked": 0.0,
                f"{prefix}_geodesic_start_distance": -1.0,
                f"{prefix}_geodesic_end_distance": -1.0,
                f"{prefix}_geodesic_reachable": 0.0,
                f"{prefix}_pred_path_length": 0.0,
                f"{prefix}_gt_path_length": 0.0,
                f"{prefix}_raw_pred_path_length": 0.0,
                f"{prefix}_full_gt_path_length": 0.0,
                f"{prefix}_executed_pred_path_length": 0.0,
                f"{prefix}_nav_pred_path_length": 0.0,
                f"{prefix}_nav_gt_path_length": 0.0,
                f"{prefix}_path_length_ratio": 1.0,
                f"{prefix}_stop_penalty": 0.0,
            }
        )
    return metrics


def _trajectory_route_diagnostics(
    trajectory: np.ndarray,
    reference_route: np.ndarray,
    *,
    free_radius_m: float,
    deviation_scale_m: float,
    motion_floor_m: float,
    reverse_transition_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure corridor departure and true reversal against GT route tangents."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    reference = np.asarray(reference_route, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"trajectory must have shape [T,2], got {trajectory.shape}")
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise ValueError(f"reference_route must have shape [R>=2,2], got {reference.shape}")
    if free_radius_m < 0 or deviation_scale_m <= 0 or motion_floor_m < 0:
        raise ValueError("Route diagnostic scales are invalid")

    segment_starts = reference[:-1]
    segment_vectors = reference[1:] - reference[:-1]
    segment_norm_sq = np.sum(segment_vectors * segment_vectors, axis=1)
    valid_segments = segment_norm_sq > 1e-12
    if not np.any(valid_segments):
        distances = np.linalg.norm(trajectory - reference[0][None, :], axis=1)
        excess = np.maximum(distances - free_radius_m, 0.0)
        return {
            "deviation_score": float(
                np.clip(
                    np.sqrt(np.mean(np.square(excess))) / deviation_scale_m,
                    0.0,
                    1.0,
                )
            ),
            "deviation_rms_m": float(np.sqrt(np.mean(np.square(distances)))),
            "deviation_max_m": float(np.max(distances)),
            "reverse_direction_score": 0.0,
        }

    starts = segment_starts[valid_segments]
    vectors = segment_vectors[valid_segments]
    norm_sq = segment_norm_sq[valid_segments]

    def nearest_segments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        offsets = points[:, None, :] - starts[None, :, :]
        projection = np.sum(offsets * vectors[None, :, :], axis=-1)
        projection = np.clip(projection / norm_sq[None, :], 0.0, 1.0)
        closest = starts[None, :, :] + projection[..., None] * vectors[None, :, :]
        distances_sq = np.sum(np.square(points[:, None, :] - closest), axis=-1)
        indices = np.argmin(distances_sq, axis=1)
        rows = np.arange(len(points))
        return np.sqrt(distances_sq[rows, indices]), indices

    distances, _ = nearest_segments(trajectory)
    excess = np.maximum(distances - free_radius_m, 0.0)
    deviation_score = float(
        np.clip(
            np.sqrt(np.mean(np.square(excess))) / deviation_scale_m,
            0.0,
            1.0,
        )
    )

    pred_steps = trajectory[1:] - trajectory[:-1]
    pred_norms = np.linalg.norm(pred_steps, axis=1)
    moving = pred_norms > motion_floor_m
    if reverse_transition_mask is not None:
        reverse_mask = np.asarray(reverse_transition_mask, dtype=bool)
        if reverse_mask.shape != moving.shape:
            raise ValueError(
                "reverse_transition_mask must align with trajectory transitions: "
                f"got {reverse_mask.shape}, expected {moving.shape}"
            )
        moving = np.logical_and(moving, reverse_mask)
    reverse_score = 0.0
    if np.any(moving):
        midpoints = 0.5 * (trajectory[1:] + trajectory[:-1])
        _, nearest_indices = nearest_segments(midpoints[moving])
        route_tangents = vectors[nearest_indices]
        tangent_norms = np.linalg.norm(route_tangents, axis=1)
        cosine = np.sum(pred_steps[moving] * route_tangents, axis=1) / np.maximum(
            pred_norms[moving] * tangent_norms, 1e-12
        )
        # Normal turns and alternative headings up to 90 degrees are free.
        # Only motion genuinely opposed to the nearest GT route direction is
        # treated as a U-turn signal.
        reverse_score = float(np.mean(np.maximum(-cosine, 0.0)))

    return {
        "deviation_score": deviation_score,
        "deviation_rms_m": float(np.sqrt(np.mean(np.square(distances)))),
        "deviation_max_m": float(np.max(distances)),
        "reverse_direction_score": reverse_score,
    }


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _linear_deadzone_score(
    error: np.ndarray,
    *,
    free_angle_deg: float,
    hard_angle_deg: float,
) -> np.ndarray:
    """Map angular error to [0, 1] with a true deadzone and linear pressure.

    The historical cosine score is almost flat around the 15--30 degree
    errors seen in deployment.  This score deliberately has a non-zero slope
    immediately outside the deadzone and saturates at ``hard_angle_deg`` so a
    few extreme samples cannot dominate the whole navigation reward.
    """

    if not 0.0 <= free_angle_deg < hard_angle_deg <= 180.0:
        raise ValueError(
            f"linear yaw guard angles must satisfy 0 <= free < hard <= 180, got {free_angle_deg=} {hard_angle_deg=}"
        )
    absolute_error_deg = np.rad2deg(np.abs(_wrap_angle(error)))
    return np.clip(
        (absolute_error_deg - float(free_angle_deg)) / (float(hard_angle_deg) - float(free_angle_deg)),
        0.0,
        1.0,
    )


def _path_heading_sequence(xy_steps: np.ndarray) -> np.ndarray:
    """Reproduce WNM's tangent-derived heading convention."""

    steps = np.asarray(xy_steps, dtype=np.float64)
    if steps.ndim != 2 or steps.shape[1] != 2:
        raise ValueError(f"xy_steps must have shape [T,2], got {steps.shape}")
    if len(steps) == 0:
        return np.zeros(1, dtype=np.float64)

    initial = steps[0].copy()
    if float(np.linalg.norm(initial)) < 1e-6:
        initial = np.asarray([1.0, 0.0], dtype=np.float64)

    heading = np.zeros(len(steps) + 1, dtype=np.float64)
    last_heading = 0.0
    for index, step in enumerate(steps):
        if float(np.linalg.norm(step)) >= 1e-6:
            dot = float(np.dot(initial, step))
            cross = float(initial[0] * step[1] - initial[1] * step[0])
            last_heading = float(np.arctan2(cross, dot))
        heading[index] = last_heading
    heading[-1] = last_heading
    return heading


def _trajectory_yaw_diagnostics(
    predicted_local_deltas: np.ndarray,
    gt_world_xy: np.ndarray,
    *,
    free_angle_deg: float,
    gross_angle_deg: float,
    motion_floor_m: float,
    rotation_floor_rad: float = 0.01,
    score_mode: str = "cosine",
    path_hard_angle_deg: float = 45.0,
    rate_free_angle_deg: float | None = None,
    rate_hard_angle_deg: float = 35.0,
    gt_free_angle_deg: float | None = None,
) -> dict[str, np.ndarray]:
    """Score yaw coupling without treating useful pure-yaw turns as drift."""

    local_deltas = np.asarray(predicted_local_deltas, dtype=np.float64)
    gt_points = np.asarray(gt_world_xy, dtype=np.float64)
    if local_deltas.ndim != 2 or local_deltas.shape[1] != 3:
        raise ValueError(f"predicted_local_deltas must have shape [T,3], got {local_deltas.shape}")
    horizon = len(local_deltas)
    if gt_points.ndim != 2 or gt_points.shape[1] != 2:
        raise ValueError(f"gt_world_xy must have shape [T+1,2], got {gt_points.shape}")
    if len(gt_points) < horizon + 1:
        raise ValueError(f"gt_world_xy is shorter than the action horizon: got {len(gt_points)}, need {horizon + 1}")
    if not 0.0 <= free_angle_deg < 180.0:
        raise ValueError("free_angle_deg must be in [0, 180)")
    if not 0.0 < gross_angle_deg < 180.0:
        raise ValueError("gross_angle_deg must be in (0, 180)")
    if motion_floor_m < 0.0 or rotation_floor_rad < 0.0:
        raise ValueError("motion/rotation floors must be non-negative")
    score_mode = str(score_mode).strip().lower()
    if score_mode not in {"cosine", "linear_guard"}:
        raise ValueError(f"score_mode must be 'cosine' or 'linear_guard', got {score_mode!r}")
    rate_free_angle_deg = float(free_angle_deg) if rate_free_angle_deg is None else float(rate_free_angle_deg)
    gt_free_angle_deg = float(free_angle_deg) if gt_free_angle_deg is None else float(gt_free_angle_deg)

    predicted_steps = local_deltas[:, :2]
    predicted_path_heading = _path_heading_sequence(predicted_steps)
    predicted_action_heading = np.concatenate(
        [
            np.zeros(1, dtype=np.float64),
            np.cumsum(local_deltas[:, 2], dtype=np.float64),
        ]
    )
    gt_steps = gt_points[1 : horizon + 1] - gt_points[:horizon]
    gt_path_heading = _path_heading_sequence(gt_steps)

    path_error = _wrap_angle(predicted_action_heading[:-1] - predicted_path_heading[:-1])
    predicted_path_dyaw = _wrap_angle(np.diff(predicted_path_heading))
    rate_error = _wrap_angle(local_deltas[:, 2] - predicted_path_dyaw)
    gross_gt_error_before = _wrap_angle(predicted_action_heading[:-1] - gt_path_heading[:-1])

    translation_moving = np.linalg.norm(predicted_steps, axis=1) > motion_floor_m
    rotation_moving = np.abs(local_deltas[:, 2]) > rotation_floor_rad
    control_active = np.logical_or(translation_moving, rotation_moving)
    pure_yaw = np.logical_and(~translation_moving, rotation_moving)

    # A pure-yaw action is judged after applying its rotation.  This rewards a
    # turn that reduces GT-heading error and penalizes a turn that drifts away.
    # Path/rate self-consistency remains translation-only downstream, because
    # a stationary action has no well-defined XY tangent to agree with.
    gross_gt_error_after = _wrap_angle(predicted_action_heading[1:] - gt_path_heading[:-1])
    gross_gt_error = np.where(
        pure_yaw,
        gross_gt_error_after,
        gross_gt_error_before,
    )

    free_angle = np.deg2rad(float(free_angle_deg))
    path_error_after_deadzone = np.sign(path_error) * np.maximum(
        np.abs(path_error) - free_angle,
        0.0,
    )
    rate_error_after_deadzone = np.sign(rate_error) * np.maximum(
        np.abs(rate_error) - free_angle,
        0.0,
    )
    if score_mode == "linear_guard":
        path_score = _linear_deadzone_score(
            path_error,
            free_angle_deg=free_angle_deg,
            hard_angle_deg=path_hard_angle_deg,
        )
        rate_score = _linear_deadzone_score(
            rate_error,
            free_angle_deg=rate_free_angle_deg,
            hard_angle_deg=rate_hard_angle_deg,
        )
        gross_score = _linear_deadzone_score(
            gross_gt_error,
            free_angle_deg=gt_free_angle_deg,
            hard_angle_deg=gross_angle_deg,
        )
    else:
        path_score = 0.5 * (1.0 - np.cos(path_error_after_deadzone))
        rate_score = 0.5 * (1.0 - np.cos(rate_error_after_deadzone))
        gross_angle = np.deg2rad(float(gross_angle_deg))
        gross_denominator = 1.0 + float(np.cos(gross_angle))
        gross_score = np.clip(
            (float(np.cos(gross_angle)) - np.cos(np.abs(gross_gt_error))) / max(gross_denominator, 1e-8),
            0.0,
            1.0,
        )
    gross_angle = np.deg2rad(float(gross_angle_deg))
    return {
        # Keep the historical key for downstream callers, but make it represent
        # any active control. Previously pure-yaw actions were silently masked.
        "moving": control_active,
        "translation_moving": translation_moving,
        "rotation_moving": rotation_moving,
        "pure_yaw": pure_yaw,
        "score_mode_id": np.full(horizon, float(score_mode == "linear_guard"), dtype=np.float64),
        "path_score": path_score,
        "path_error": path_error,
        "rate_score": rate_score,
        "rate_error": rate_error,
        "gross_gt_score": gross_score,
        "gross_gt_error": gross_gt_error,
        "gross_gt_active": np.abs(gross_gt_error) > gross_angle,
        "predicted_action_heading": predicted_action_heading,
        "predicted_path_heading": predicted_path_heading,
        "gt_path_heading": gt_path_heading,
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"values/mask shapes differ: {values.shape} vs {mask.shape}")
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _masked_mean_max_aggregate(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    max_mix: float,
) -> tuple[float, float, float]:
    """Return mean, max, and a spike-sensitive convex aggregation."""

    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"values/mask shapes differ: {values.shape} vs {mask.shape}")
    if not 0.0 <= max_mix <= 1.0:
        raise ValueError(f"max_mix must lie in [0, 1], got {max_mix}")
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    active = values[mask]
    mean_score = float(np.mean(active))
    max_score = float(np.max(active))
    aggregate = (1.0 - max_mix) * mean_score + max_mix * max_score
    return mean_score, max_score, float(aggregate)


def _compute_stop_well_chunks(
    trajectory: np.ndarray,
    predicted_local_deltas: np.ndarray,
    *,
    reached_step: int,
    left_step: int,
    left_goal_penalty: float,
    goal_radius: float,
    chunk_size: int,
    chunk_count: int,
    max_transition_exclusive: int,
) -> dict[str, np.ndarray]:
    """Return post-arrival potential/effort credit with uniform chunk semantics."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    local_deltas = np.asarray(predicted_local_deltas, dtype=np.float64)
    horizon = len(trajectory) - 1
    if trajectory.shape != (horizon + 1, 2):
        raise ValueError(f"trajectory must be [T+1,2], got {trajectory.shape}")
    if local_deltas.shape != (horizon, 3):
        raise ValueError(
            "predicted_local_deltas must align with trajectory transitions: "
            f"got {local_deltas.shape}, expected {(horizon, 3)}"
        )
    if goal_radius <= 0:
        raise ValueError("goal_radius must be positive")

    energy_weight = _env_float("WAM_STOP_WELL_ENERGY_WEIGHT", 0.12)
    xy_deadzone = _env_float("WAM_STOP_WELL_XY_DEADZONE_M", 0.02)
    yaw_weight = _env_float("WAM_STOP_WELL_YAW_WEIGHT", 0.10)
    yaw_deadzone = _env_float("WAM_STOP_WELL_YAW_DEADZONE_RAD", 0.02)
    yaw_scale = _env_float("WAM_STOP_WELL_YAW_SCALE_RAD", 0.25)
    if energy_weight < 0 or yaw_weight < 0:
        raise ValueError("Stop-well weights must be non-negative")
    if xy_deadzone < 0 or yaw_deadzone < 0 or yaw_scale <= 0:
        raise ValueError("Stop-well deadzones must be non-negative and yaw scale positive")

    result = {
        "reward": np.zeros(chunk_count, dtype=np.float64),
        "active": np.zeros(chunk_count, dtype=bool),
        "energy": np.zeros(chunk_count, dtype=np.float64),
        "xy_motion": np.zeros(chunk_count, dtype=np.float64),
        "yaw_motion": np.zeros(chunk_count, dtype=np.float64),
        "exit_penalty": np.zeros(chunk_count, dtype=np.float64),
    }
    if reached_step < 0:
        return result

    transition_limit = min(max(int(max_transition_exclusive), 0), horizon)
    for index in range(chunk_count):
        chunk_start = index * chunk_size
        chunk_end = min(chunk_start + chunk_size, transition_limit)
        # Entering the goal at position k latches the stop well for transition
        # k -> k+1 and all subsequent actually executed transitions.
        start = max(chunk_start, reached_step)
        end = chunk_end
        if start < end:
            xy_steps = np.linalg.norm(
                trajectory[start + 1 : end + 1] - trajectory[start:end],
                axis=1,
            )
            yaw_steps = np.abs(local_deltas[start:end, 2])
            translation_effort = np.maximum(xy_steps - xy_deadzone, 0.0) / goal_radius
            yaw_effort = yaw_weight * np.maximum(yaw_steps - yaw_deadzone, 0.0) / yaw_scale
            energy = float(np.sum(translation_effort + yaw_effort))
            result["active"][index] = True
            result["energy"][index] = energy
            result["xy_motion"][index] = float(np.sum(xy_steps))
            result["yaw_motion"][index] = float(np.sum(yaw_steps))
            result["reward"][index] = -energy_weight * energy

        if left_step >= 0 and index == (left_step - 1) // chunk_size:
            result["active"][index] = True
            result["exit_penalty"][index] = left_goal_penalty
            result["reward"][index] += left_goal_penalty
    return result
