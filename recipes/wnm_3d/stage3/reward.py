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

"""Combined visual/action reward for InteriorGS WNM training.

The entry module intentionally remains small because the trainer loads reward
functions by file path. Collision execution, STOP semantics, diagnostics, and
chunked action composition live in focused package modules.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from verl_omni.utils.action_chunk_credit import action_chunk_credit_enabled
from verl_omni.utils.reward_score.wam_flow_action_reward import (
    compute_flow_action_consistency,
)
from verl_omni.utils.reward_score.wam_navigation_reward import (
    freeze_trajectory_at_first_collision,
    occupancy_geodesic_distances_to_goal,
    rollout_actions_to_local_deltas,
    rollout_actions_to_world_xy,
    trajectory_clearance_risk,
    trajectory_collides,
    trajectory_path_length,
)
from verl_omni.utils.reward_score.wam_stage3_action import (
    _compute_chunked_action_reward_impl,
    _empty_flow_action_diagnostics,
)
from verl_omni.utils.reward_score.wam_stage3_collision import (
    _occupancy_pixel_to_world,
    execute_gn0_collision_recovery_chunk,
)
from verl_omni.utils.reward_score.wam_stage3_metrics import (
    _compact_reward_metrics,
    _compute_stop_well_chunks,
    _empty_chunk_metrics,
    _env_bool,
    _env_float,
    _env_int,
    _linear_deadzone_score,
    _load_gt_video,
    _masked_mean,
    _masked_mean_max_aggregate,
    _path_heading_sequence,
    _trajectory_route_diagnostics,
    _trajectory_yaw_diagnostics,
    _wrap_angle,
)
from verl_omni.utils.reward_score.wam_stage3_stop import (
    _empty_deployment_stop_diagnostics,
    compute_deployment_stop_diagnostics,
    detect_gn0_stops_per_credit_chunk,
    freeze_trajectory_at_chunk_motion_stop,
    freeze_trajectory_at_gn0_execution_stop,
    freeze_trajectory_at_gn0_stop,
)
from verl_omni.utils.reward_score.wam_terminal_stop_penalty import (
    TerminalStopConfig,
    apply_terminal_stop_penalty,
    compute_terminal_stop_penalty,
)
from verl_omni.utils.reward_score.wam_vision_reward import (
    compute_vision_behavior_signature,
    compute_vision_reward,
)

__all__ = [
    "_compute_stop_well_chunks",
    "_linear_deadzone_score",
    "_masked_mean",
    "_masked_mean_max_aggregate",
    "_occupancy_pixel_to_world",
    "_path_heading_sequence",
    "_trajectory_route_diagnostics",
    "_trajectory_yaw_diagnostics",
    "_wrap_angle",
    "detect_gn0_stops_per_credit_chunk",
    "execute_gn0_collision_recovery_chunk",
    "freeze_trajectory_at_chunk_motion_stop",
    "freeze_trajectory_at_gn0_execution_stop",
    "freeze_trajectory_at_gn0_stop",
]


def _compute_chunked_action_reward(
    predicted_world_xy: np.ndarray,
    ground_truth: Mapping,
    *,
    predicted_local_deltas: np.ndarray | None = None,
    action_reward_min: float,
    action_reward_max: float,
    path_efficiency_power: float,
    chunk0_flow_action_bonus: float = 0.0,
) -> dict:
    """Delegate chunked scoring while preserving facade-level test overrides."""

    return _compute_chunked_action_reward_impl(
        predicted_world_xy,
        ground_truth,
        predicted_local_deltas=predicted_local_deltas,
        action_reward_min=action_reward_min,
        action_reward_max=action_reward_max,
        path_efficiency_power=path_efficiency_power,
        chunk0_flow_action_bonus=chunk0_flow_action_bonus,
        _trajectory_collides_fn=trajectory_collides,
        _trajectory_clearance_risk_fn=trajectory_clearance_risk,
        _occupancy_geodesic_distances_to_goal_fn=occupancy_geodesic_distances_to_goal,
        _freeze_trajectory_at_first_collision_fn=freeze_trajectory_at_first_collision,
    )


def compute_score(
    responses,
    actions,
    ground_truth,
    extra_info=None,
    data_source=None,
    **kwargs,
):
    """Return independently attributable visual and action rewards."""
    del data_source, kwargs
    if not isinstance(ground_truth, Mapping):
        raise TypeError(f"ground_truth must be a mapping, got {type(ground_truth).__name__}")
    if extra_info is None:
        extra_info = {}
    if not isinstance(extra_info, Mapping):
        raise TypeError(f"extra_info must be a mapping, got {type(extra_info).__name__}")

    event_validation_label = str(ground_truth.get("event_validation_label", "")).strip()
    event_validation_active = bool(event_validation_label)
    event_validation_flags = {
        "event_val_active": float(event_validation_active),
        "event_val_collision_precursor": float(event_validation_label == "collision_precursor"),
        "event_val_premature_stop_risk": float(event_validation_label.startswith("premature_stop_")),
        "event_val_premature_stop_near": float(event_validation_label == "premature_stop_near"),
        "event_val_premature_stop_far": float(event_validation_label == "premature_stop_far"),
        "event_val_required_stop": float(event_validation_label.startswith("required_stop_")),
        "event_val_required_stop_core": float(event_validation_label == "required_stop_core"),
        "event_val_required_stop_mid": float(event_validation_label == "required_stop_mid"),
        "event_val_required_stop_boundary": float(event_validation_label == "required_stop_boundary"),
        "event_val_near_goal_continue": float(event_validation_label == "near_goal_continue"),
        "event_val_expected_stop": float(bool(ground_truth.get("event_validation_expected_stop", False))),
        "event_val_distance_to_goal_m": float(ground_truth.get("event_validation_distance_to_goal_m", -1.0)),
    }

    gt_video = _load_gt_video(
        str(ground_truth["video_path"]),
        int(ground_truth.get("future_start", 33)),
        int(ground_truth.get("future_frames", 33)),
    )
    vision = compute_vision_reward(responses, gt_video)
    base_visual_reward = float(vision["vision_reward"])
    visual_reward = base_visual_reward
    visual_signature = compute_vision_behavior_signature(responses)
    nav_action_scale = float(ground_truth["nav_action_scale"])
    if not np.isfinite(nav_action_scale) or nav_action_scale <= 0:
        raise ValueError(f"ground_truth.nav_action_scale must be positive and finite, got {nav_action_scale}")

    action_invalid = False
    predicted_local_deltas = np.zeros((0, 3), dtype=np.float32)
    try:
        predicted_local_deltas = rollout_actions_to_local_deltas(
            actions,
            ground_truth["q01"],
            ground_truth["q99"],
            nav_action_scale=nav_action_scale,
        )
        predicted_world_xy = rollout_actions_to_world_xy(
            actions,
            ground_truth["start_extrinsic"],
            ground_truth["q01"],
            ground_truth["q99"],
            nav_action_scale=nav_action_scale,
        )
    except (TypeError, ValueError):
        action_invalid = True
        predicted_world_xy = np.asarray(ground_truth["gt_world_xy"], dtype=np.float32)[:1]

    deployment_stop = _empty_deployment_stop_diagnostics(ground_truth)
    if not action_invalid:
        deployment_stop = compute_deployment_stop_diagnostics(
            predicted_local_deltas,
            predicted_world_xy,
            ground_truth,
        )

    flow_action = _empty_flow_action_diagnostics()
    flow_action_visual_weight = _env_float("WAM_FLOW_ACTION_VISUAL_WEIGHT", 0.0)
    flow_action_action_weight = _env_float("WAM_FLOW_ACTION_ACTION_WEIGHT", 0.0)
    if flow_action_visual_weight < 0 or flow_action_action_weight < 0:
        raise ValueError("Flow/action reward weights must be non-negative")
    flow_action_bonus = 0.0
    flow_action_action_bonus = 0.0
    if (flow_action_visual_weight > 0.0 or flow_action_action_weight > 0.0) and not action_invalid:
        calibration_path = os.environ.get("WAM_FLOW_ACTION_CALIBRATION_PATH", "").strip()
        if not calibration_path:
            raise ValueError(
                "WAM_FLOW_ACTION_CALIBRATION_PATH is required when either flow/action reward weight is positive"
            )
        flow_action = compute_flow_action_consistency(
            responses,
            actions,
            ground_truth["q01"],
            ground_truth["q99"],
            calibration_path,
            nav_action_scale=nav_action_scale,
        )
        # The first nine frames and first eight actions describe the same
        # physical event. Give the consistency score to both policies: visual
        # credit teaches the video to depict the action, while chunk-0 action
        # credit teaches the action to agree with the depicted camera motion.
        # Existing degradation detection gates both branches so blur/static
        # output cannot obtain strong cross-modal credit.
        gated_flow_action_score = float(flow_action["flow_action_score"]) * float(vision["degeneration_factor"])
        flow_action_bonus = flow_action_visual_weight * gated_flow_action_score
        flow_action_action_bonus = flow_action_action_weight * gated_flow_action_score
        visual_reward += flow_action_bonus

    collision = {"collided": False, "collision_step": -1, "out_of_bounds": False}
    softspl = 0.0
    geodesic_progress = 0.0
    path_efficiency = 0.0
    goal_score = 0.0
    goal_credit = 0.0
    goal_entry_bonus = 0.0
    geodesic_start_distance = float("inf")
    geodesic_end_distance = float("inf")
    predicted_path_length = 0.0
    reference_path_length = 0.0
    executed_predicted_path_length = 0.0
    navigation_predicted_path_length = 0.0
    navigation_reference_path_length = 0.0
    path_length_ratio = 1.0
    path_length_overrun_ratio = 0.0
    path_efficiency_power = _env_float("WAM_PATH_EFFICIENCY_POWER", 1.0)
    if path_efficiency_power <= 0:
        raise ValueError("WAM_PATH_EFFICIENCY_POWER must be positive")
    collision_penalty = 0.0
    collision_hard_penalty = 0.0
    collision_soft_risk = 0.0
    collision_soft_penalty = 0.0
    collision_min_clearance_px = -1.0
    geodesic_start_snapped = False
    geodesic_end_snapped = False
    base_action_reward = 0.0
    action_reward_pre_clip = 0.0
    action_reward_min = _env_float("WAM_ACTION_REWARD_MIN", 0.0)
    action_reward_max = _env_float("WAM_ACTION_REWARD_MAX", 1.0)
    if action_reward_min > action_reward_max:
        raise ValueError("WAM_ACTION_REWARD_MIN must not exceed WAM_ACTION_REWARD_MAX")
    stop = {
        "terminal_stop_penalty": 0.0,
        "reached_goal": False,
        "failed_to_stop": False,
        "left_goal": False,
        "left_step": -1,
        "tail_path_length": 0.0,
    }
    action_reward = 0.0
    evaluated_actions = 0
    chunk_extra_metrics: dict[str, float] = {}
    if action_chunk_credit_enabled():
        if action_invalid:
            chunk_extra_metrics = _empty_chunk_metrics()
        else:
            chunk_result = _compute_chunked_action_reward(
                predicted_world_xy,
                ground_truth,
                predicted_local_deltas=predicted_local_deltas,
                action_reward_min=action_reward_min,
                action_reward_max=action_reward_max,
                path_efficiency_power=path_efficiency_power,
                chunk0_flow_action_bonus=flow_action_action_bonus,
            )
            chunk_extra_metrics = chunk_result["metrics"]
            action_reward = chunk_result["action_reward"]
            base_action_reward = chunk_result["base_action_reward"]
            action_reward_pre_clip = chunk_result["action_reward_pre_clip"]
            stop = chunk_result["stop"]
            first = chunk_result["chunks"][0]
            collision = first["collision"]
            softspl = first["softspl"]
            geodesic_progress = first["geodesic_progress"]
            path_efficiency = first["path_efficiency"]
            goal_score = first["goal_score"]
            goal_credit = first["goal_credit"]
            goal_entry_bonus = first["goal_entry_bonus"]
            geodesic_start_distance = first["geodesic_start_distance"]
            geodesic_end_distance = first["geodesic_end_distance"]
            predicted_path_length = first["predicted_path_length"]
            reference_path_length = first["reference_path_length"]
            executed_predicted_path_length = first["executed_predicted_path_length"]
            navigation_predicted_path_length = first["navigation_predicted_path_length"]
            navigation_reference_path_length = first["navigation_reference_path_length"]
            path_length_ratio = first["path_length_ratio"]
            path_length_overrun_ratio = first["path_length_overrun_ratio"]
            collision_penalty = first["collision_penalty"]
            collision_hard_penalty = first["collision_hard_penalty"]
            collision_soft_risk = first["collision_soft_risk"]
            collision_soft_penalty = first["collision_soft_penalty"]
            collision_min_clearance_px = first["collision_min_clearance_px"]
            geodesic_start_snapped = first["geodesic_start_snapped"]
            geodesic_end_snapped = first["geodesic_end_snapped"]
            evaluated_actions = int(first["evaluated_actions"])
    elif not action_invalid:
        collision_chunk_actions = _env_int("WAM_ACTION_COLLISION_CHUNK_SIZE", 8)
        if collision_chunk_actions <= 0:
            raise ValueError("WAM_ACTION_COLLISION_CHUNK_SIZE must be positive")
        evaluated_actions = min(collision_chunk_actions, max(0, len(predicted_world_xy) - 1))
        collision_trajectory = predicted_world_xy[: evaluated_actions + 1]
        collision = trajectory_collides(
            collision_trajectory,
            ground_truth["scene_dir"],
            occupancy_threshold=_env_int("WAM_OCCUPANCY_THRESHOLD", 200),
            occupancy_margin_px=_env_int("WAM_OCCUPANCY_MARGIN_PX", 2),
        )
        collision_soft_enabled = _env_bool("WAM_COLLISION_SOFT_ENABLED", False)
        collision_hard_margin_px = _env_int("WAM_OCCUPANCY_MARGIN_PX", 2)
        collision_soft_margin_px = _env_int("WAM_COLLISION_SOFT_MARGIN_PX", 4)
        collision_soft_weight = _env_float("WAM_COLLISION_SOFT_PENALTY_WEIGHT", 0.20)
        if collision_soft_weight < 0:
            raise ValueError("WAM_COLLISION_SOFT_PENALTY_WEIGHT must be non-negative")
        if collision_soft_enabled:
            clearance = trajectory_clearance_risk(
                collision_trajectory,
                ground_truth["scene_dir"],
                occupancy_threshold=_env_int("WAM_OCCUPANCY_THRESHOLD", 200),
                hard_margin_px=collision_hard_margin_px,
                soft_margin_px=collision_soft_margin_px,
            )
            collision_soft_risk = float(clearance["risk"])
            collision_min_clearance_px = float(clearance["min_clearance_px"])
            collision_soft_penalty = -collision_soft_weight * collision_soft_risk
        gt_world_xy = np.asarray(ground_truth["gt_world_xy"], dtype=np.float32)
        reference_trajectory = gt_world_xy[: evaluated_actions + 1]
        predicted_path_length = trajectory_path_length(collision_trajectory)
        reference_path_length = trajectory_path_length(reference_trajectory)
        executed_predicted_path_length = predicted_path_length
        navigation_predicted_path_length = predicted_path_length
        navigation_reference_path_length = reference_path_length
        if reference_path_length > 1e-8:
            path_length_ratio = predicted_path_length / reference_path_length
            path_length_overrun_ratio = max(path_length_ratio - 1.0, 0.0)
        elif predicted_path_length > 1e-8:
            # Keep diagnostics finite for a degenerate zero-length GT chunk.
            path_length_ratio = predicted_path_length / 1e-8
            path_length_overrun_ratio = path_length_ratio - 1.0
        if predicted_path_length <= reference_path_length or predicted_path_length <= 1e-8:
            path_efficiency = 1.0
        else:
            # One-sided regularization: alternative routes that are no longer
            # than GT remain valid, while path overruns are increasingly
            # suppressed. Power=1 is the original SoftSPL behavior.
            path_efficiency = (reference_path_length / predicted_path_length) ** path_efficiency_power

        geodesic = occupancy_geodesic_distances_to_goal(
            [collision_trajectory[0], collision_trajectory[-1]],
            ground_truth["goal_world_xy"],
            ground_truth["scene_dir"],
            occupancy_threshold=_env_int("WAM_OCCUPANCY_THRESHOLD", 200),
            # Collision uses the conservative 2 px safety margin. Geodesic
            # progress defaults to the raw navigable map so narrow valid
            # passages do not become disconnected solely by reward shaping.
            occupancy_margin_px=_env_int("WAM_GEODESIC_OCCUPANCY_MARGIN_PX", 0),
            snap_radius_px=_env_int("WAM_GEODESIC_SNAP_RADIUS_PX", 4),
        )
        geodesic_start_distance, geodesic_end_distance = (
            float(geodesic["distances"][0]),
            float(geodesic["distances"][1]),
        )
        geodesic_start_snapped, geodesic_end_snapped = (
            bool(geodesic["snapped"][0]),
            bool(geodesic["snapped"][1]),
        )
        if np.isfinite(geodesic_start_distance) and geodesic_start_distance > 1e-8:
            geodesic_progress = float(
                np.clip(
                    (geodesic_start_distance - geodesic_end_distance) / geodesic_start_distance,
                    0.0,
                    1.0,
                )
            )
        elif np.isfinite(geodesic_end_distance) and geodesic_end_distance <= 1e-8:
            geodesic_progress = 1.0

        softspl = geodesic_progress * path_efficiency
        goal_temperature = _env_float("WAM_GOAL_SCORE_TEMPERATURE_M", 0.75)
        if goal_temperature <= 0:
            raise ValueError("WAM_GOAL_SCORE_TEMPERATURE_M must be positive")
        if np.isfinite(geodesic_end_distance):
            goal_score = float(np.exp(-geodesic_end_distance / goal_temperature))

        collision_penalty_weight = _env_float("WAM_COLLISION_PENALTY_WEIGHT", 0.10)
        if collision_penalty_weight < 0:
            raise ValueError("WAM_COLLISION_PENALTY_WEIGHT must be non-negative")
        if collision["collided"]:
            collision_hard_penalty = -collision_penalty_weight
        collision_penalty = collision_hard_penalty + collision_soft_penalty

        softspl_weight = _env_float("WAM_SOFTSPL_WEIGHT", 0.90)
        goal_score_weight = _env_float("WAM_GOAL_SCORE_WEIGHT", 0.10)
        base_action_reward = (
            softspl_weight * softspl + goal_score_weight * goal_score + collision_penalty + flow_action_action_bonus
        )
        stop = compute_terminal_stop_penalty(
            predicted_world_xy,
            ground_truth["goal_world_xy"],
            TerminalStopConfig(
                goal_radius=float(ground_truth.get("stop_radius_m", 1.5)),
                leave_hysteresis=_env_float("WAM_STOP_LEAVE_HYSTERESIS", 0.0),
                min_step_motion=_env_float("WAM_STOP_MIN_STEP_MOTION", 0.05),
                max_tail_path_length=_env_float("WAM_STOP_MAX_TAIL_PATH", 0.30),
                allowed_moving_steps=_env_int("WAM_STOP_ALLOWED_MOVING_STEPS", 1),
                continued_motion_penalty=_env_float("WAM_STOP_CONTINUED_PENALTY", 0.0),
                left_goal_penalty=_env_float("WAM_STOP_LEFT_GOAL_PENALTY", -0.50),
            ),
        )
        action_reward_pre_clip = base_action_reward + float(stop["terminal_stop_penalty"])
        action_reward = apply_terminal_stop_penalty(
            base_action_reward,
            stop,
            min_reward=action_reward_min,
            max_reward=action_reward_max,
        )

    exploration_horizon = _env_int("WAM_ACTION_COLLISION_CHUNK_SIZE", 8)
    trajectory_signature = np.full(
        (exploration_horizon + 1, 2),
        np.nan,
        dtype=np.float32,
    )
    if not action_invalid:
        signature_points = min(len(predicted_world_xy), exploration_horizon + 1)
        trajectory_signature[:signature_points] = predicted_world_xy[:signature_points]
        if 0 < signature_points < exploration_horizon + 1:
            trajectory_signature[signature_points:] = predicted_world_xy[signature_points - 1]

    # score is reporting-only. The trainer normalizes and applies visual/action
    # advantages independently, so this sum does not re-couple credit assignment.
    result = {
        "score": visual_reward + action_reward,
        "visual_reward": visual_reward,
        "action_reward": action_reward,
        **event_validation_flags,
        "vision_reward_base": base_visual_reward,
        "vision_ms_ssim": float(vision["ms_ssim"]),
        "vision_charbonnier_error": float(vision["low_res_charbonnier_error"]),
        "vision_temporal_error": float(vision["temporal_gradient_error"]),
        "vision_degenerate": float(bool(vision["degenerate"])),
        "vision_frozen": float(bool(vision["is_frozen"])),
        "vision_prefix_copy": float(bool(vision["is_prefix_copy"])),
        "vision_low_contrast": float(bool(vision["is_low_contrast"])),
        "vision_exposure": float(bool(vision["is_exposure_degenerate"])),
        "vision_flicker": float(bool(vision["is_flickering"])),
        "vision_excessive_motion": float(bool(vision["is_excessive_motion"])),
        "vision_degradation_factor": float(vision["degeneration_factor"]),
        "vision_pred_motion": float(vision["pred_motion"]),
        "vision_gt_motion": float(vision["gt_motion"]),
        "vision_prefix_copy_error": float(vision["prefix_copy_error"]),
        "vision_pred_luma": float(vision["pred_mean_luma"]),
        "vision_saturated_ratio": float(vision["pred_saturated_ratio"]),
        "vision_luma_jump": float(vision["pred_luma_jump"]),
        "flow_action_bonus": float(flow_action_bonus),
        "flow_action_action_bonus": float(flow_action_action_bonus),
        **{
            key: float(value)
            for key, value in flow_action.items()
            if isinstance(value, bool | int | float | np.bool_ | np.number)
        },
        "action_softspl": float(softspl),
        "action_geodesic_progress": float(geodesic_progress),
        "action_path_efficiency": float(path_efficiency),
        "action_goal_score": float(goal_score),
        "action_goal_credit": float(goal_credit),
        "action_goal_entry_bonus": float(goal_entry_bonus),
        # Keep TensorBoard reductions finite; reachability disambiguates the
        # -1 sentinel used for out-of-bounds/disconnected queries.
        "action_geodesic_start_distance": float(
            geodesic_start_distance if np.isfinite(geodesic_start_distance) else -1.0
        ),
        "action_geodesic_end_distance": float(geodesic_end_distance if np.isfinite(geodesic_end_distance) else -1.0),
        "action_geodesic_reachable": float(np.isfinite(geodesic_start_distance) and np.isfinite(geodesic_end_distance)),
        "action_pred_path_length": float(predicted_path_length),
        "action_gt_path_length": float(reference_path_length),
        # In chunk-credit mode the legacy pred/GT names are deliberately the
        # fixed-horizon raw plan and fixed-horizon GT.  The aliases below make
        # the anti-hacking semantics explicit while retaining old dashboards.
        "action_raw_pred_path_length": float(predicted_path_length),
        "action_full_gt_path_length": float(reference_path_length),
        "action_executed_pred_path_length": float(executed_predicted_path_length),
        "action_nav_pred_path_length": float(navigation_predicted_path_length),
        "action_nav_gt_path_length": float(navigation_reference_path_length),
        "action_path_length_ratio": float(path_length_ratio),
        "action_path_length_overrun_ratio": float(path_length_overrun_ratio),
        "action_path_efficiency_power": float(path_efficiency_power),
        "action_collision_penalty": float(collision_penalty),
        "action_collision_hard_penalty": float(collision_hard_penalty),
        "action_collision_soft_risk": float(collision_soft_risk),
        "action_collision_soft_penalty": float(collision_soft_penalty),
        "action_collision_min_clearance_px": float(collision_min_clearance_px),
        "action_reward_pre_clip": float(action_reward_pre_clip),
        "action_reward_min": float(action_reward_min),
        "action_reward_max": float(action_reward_max),
        "action_reward_hit_floor": float(action_reward <= action_reward_min),
        "action_collision_clipped_to_zero": float(
            bool(collision["collided"]) and action_reward_min == 0.0 and action_reward <= 0.0
        ),
        "action_collision_nonpositive": float(bool(collision["collided"]) and action_reward <= 0.0),
        "action_geodesic_start_snapped": float(geodesic_start_snapped),
        "action_geodesic_end_snapped": float(geodesic_end_snapped),
        "action_invalid": float(action_invalid),
        "action_collision": float(bool(collision["collided"])),
        "action_out_of_bounds": float(bool(collision["out_of_bounds"])),
        "action_collision_step": float(collision["collision_step"]),
        "action_collision_checked_steps": float(
            min(_env_int("WAM_ACTION_COLLISION_CHUNK_SIZE", 8), max(0, len(predicted_world_xy) - 1))
        ),
        "stop_reached_goal": float(bool(stop["reached_goal"])),
        "stop_failed": float(bool(stop["failed_to_stop"])),
        "stop_left_goal": float(bool(stop["left_goal"])),
        "stop_penalty": float(stop["terminal_stop_penalty"]),
        "stop_tail_path_length": float(stop["tail_path_length"]),
        **deployment_stop,
        **chunk_extra_metrics,
        # Internal fixed-size payloads are reduced into group exploration
        # metrics on the trainer driver. They are deliberately prefixed with
        # `_exploration_` so generic scalar reducers skip them.
        "_exploration_action_world_xy": trajectory_signature.tolist(),
        "_exploration_visual_luma": visual_signature.tolist(),
    }
    return _compact_reward_metrics(result)
