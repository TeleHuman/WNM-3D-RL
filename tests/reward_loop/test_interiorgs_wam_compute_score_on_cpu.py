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

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from verl_omni.utils.reward_score.wam_navigation_reward import CAM_TO_NAV


def _module():
    path = Path(__file__).resolve().parents[2] / "recipes" / "wnm_3d" / "stage3" / "reward.py"
    name = "_interiorgs_wam_reward_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_score_returns_separate_visual_and_action_rewards(monkeypatch):
    module = _module()
    torch.manual_seed(5)
    video = torch.rand(5, 3, 24, 32)
    monkeypatch.setattr(module, "_load_gt_video", lambda *args: video.clone())
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {"collided": False, "collision_step": -1, "out_of_bounds": False},
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda *args, **kwargs: {
            "distances": np.asarray([4.0, 2.0]),
            "snapped": np.asarray([False, False]),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = CAM_TO_NAV
    actions = torch.zeros(4, 32)
    actions[:, 0] = 0.1
    gt_path = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]
    result = module.compute_score(
        video,
        actions,
        {
            "video_path": "unused.mp4",
            "future_start": 0,
            "future_frames": 5,
            "start_extrinsic": extrinsic.tolist(),
            "q01": [-1, -1, -1],
            "q99": [1, 1, 1],
            "nav_action_scale": 1.0,
            "scene_dir": "unused",
            "gt_world_xy": gt_path,
            "goal_world_xy": [100, 100],
            "stop_radius_m": 1.5,
        },
    )
    assert result["visual_reward"] > 0.99
    assert result["action_reward"] > 0.44
    assert result["action_softspl"] == pytest.approx(0.5)
    assert result["action_geodesic_progress"] == pytest.approx(0.5)
    assert result["action_collision_penalty"] == 0.0
    assert result["score"] == result["visual_reward"] + result["action_reward"]
    assert result["vision_reward_base"] == result["visual_reward"]
    assert result["flow_action_bonus"] == 0.0
    assert result["flow_action_valid"] == 0.0
    assert np.asarray(result["_exploration_action_world_xy"]).shape == (9, 2)
    assert np.asarray(result["_exploration_visual_luma"]).shape == (4, 8, 8)


def test_collision_check_is_limited_to_first_eight_actions(monkeypatch):
    module = _module()
    torch.manual_seed(6)
    video = torch.rand(5, 3, 24, 32)
    monkeypatch.setattr(module, "_load_gt_video", lambda *args: video.clone())
    checked_lengths = []

    def capture_collision(trajectory, *args, **kwargs):
        checked_lengths.append(len(trajectory))
        return {"collided": False, "collision_step": -1, "out_of_bounds": False}

    monkeypatch.setattr(module, "trajectory_collides", capture_collision)
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda *args, **kwargs: {
            "distances": np.asarray([4.0, 4.0]),
            "snapped": np.asarray([False, False]),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = CAM_TO_NAV
    actions = torch.zeros(32, 32)
    gt_path = [[float(index), 0.0] for index in range(33)]
    module.compute_score(
        video,
        actions,
        {
            "video_path": "unused.mp4",
            "future_start": 0,
            "future_frames": 5,
            "start_extrinsic": extrinsic.tolist(),
            "q01": [-1, -1, -1],
            "q99": [1, 1, 1],
            "nav_action_scale": 1.0,
            "scene_dir": "unused",
            "gt_world_xy": gt_path,
            "goal_world_xy": [100, 100],
            "stop_radius_m": 1.5,
        },
    )
    assert checked_lengths == [9]


def test_path_efficiency_power_strengthens_only_overlength_penalty(monkeypatch):
    module = _module()
    torch.manual_seed(16)
    video = torch.rand(9, 3, 24, 32)
    monkeypatch.setattr(module, "_load_gt_video", lambda *args: video.clone())
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {"collided": False, "collision_step": -1, "out_of_bounds": False},
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda *args, **kwargs: {
            "distances": np.asarray([4.0, 2.0]),
            "snapped": np.asarray([False, False]),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    monkeypatch.setenv("WAM_PATH_EFFICIENCY_POWER", "2.0")

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = CAM_TO_NAV
    actions = torch.zeros(8, 32)
    actions[:, 0] = 0.2
    result = module.compute_score(
        video,
        actions,
        {
            "video_path": "unused.mp4",
            "future_start": 0,
            "future_frames": 9,
            "start_extrinsic": extrinsic.tolist(),
            "q01": [-1, -1, -1],
            "q99": [1, 1, 1],
            "nav_action_scale": 1.0,
            "scene_dir": "unused",
            "gt_world_xy": [[0.1 * index, 0.0] for index in range(9)],
            "goal_world_xy": [100, 100],
            "stop_radius_m": 1.5,
        },
    )

    assert result["action_pred_path_length"] == pytest.approx(1.6)
    assert result["action_gt_path_length"] == pytest.approx(0.8)
    assert result["action_path_length_ratio"] == pytest.approx(2.0)
    assert result["action_path_length_overrun_ratio"] == pytest.approx(1.0)
    assert result["action_path_efficiency_power"] == pytest.approx(2.0)
    assert result["action_path_efficiency"] == pytest.approx(0.25)
    assert result["action_softspl"] == pytest.approx(0.125)


def test_flow_action_consistency_is_applied_to_visual_and_action_rewards(monkeypatch):
    module = _module()
    torch.manual_seed(8)
    video = torch.rand(9, 3, 24, 32)
    monkeypatch.setattr(module, "_load_gt_video", lambda *args: video.clone())
    monkeypatch.setattr(
        module,
        "compute_flow_action_consistency",
        lambda *args, **kwargs: {
            "flow_action_score": 0.8,
            "flow_action_valid": True,
        },
    )
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {"collided": False, "collision_step": -1, "out_of_bounds": False},
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda *args, **kwargs: {
            "distances": np.asarray([4.0, 2.0]),
            "snapped": np.asarray([False, False]),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    monkeypatch.setenv("WAM_FLOW_ACTION_VISUAL_WEIGHT", "0.05")
    monkeypatch.setenv("WAM_FLOW_ACTION_ACTION_WEIGHT", "0.05")
    monkeypatch.setenv("WAM_FLOW_ACTION_CALIBRATION_PATH", "unused.json")
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = CAM_TO_NAV
    actions = torch.zeros(8, 32)
    actions[:, 0] = 0.1
    result = module.compute_score(
        video,
        actions,
        {
            "video_path": "unused.mp4",
            "future_start": 0,
            "future_frames": 9,
            "start_extrinsic": extrinsic.tolist(),
            "q01": [-1, -1, -1],
            "q99": [1, 1, 1],
            "nav_action_scale": 1.0,
            "scene_dir": "unused",
            "gt_world_xy": [[0.1 * index, 0.0] for index in range(9)],
            "goal_world_xy": [100, 100],
            "stop_radius_m": 1.5,
        },
    )
    assert result["flow_action_bonus"] == pytest.approx(0.04)
    assert result["flow_action_action_bonus"] == pytest.approx(0.04)
    assert result["visual_reward"] == pytest.approx(result["vision_reward_base"] + 0.04)
    # The same first-chunk consistency event credits both generated modalities.
    assert result["action_reward"] == pytest.approx(0.45 + 0.10 * np.exp(-2.0 / 0.75) + 0.04)


def test_flow_action_action_credit_is_limited_to_first_temporal_chunk(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_SIGNED_PROGRESS_WEIGHT", "0")
    monkeypatch.setenv("WAM_SYMMETRIC_LENGTH_WEIGHT", "0")
    monkeypatch.setenv("WAM_SIGNED_GOAL_WEIGHT", "0")
    monkeypatch.setenv("WAM_ROUTE_DEVIATION_WEIGHT", "0")
    monkeypatch.setenv("WAM_REVERSE_DIRECTION_WEIGHT", "0")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "false")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "false")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda points, *args, **kwargs: {
            "distances": np.linspace(10.0, 6.0, len(points)),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    predicted = np.stack([np.arange(33, dtype=np.float32) * 0.1, np.zeros(33)], axis=1)
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": [10.0, 0.0],
            "gt_world_xy": predicted.copy(),
            "stop_radius_m": 1.5,
        },
        predicted_local_deltas=np.tile([0.1, 0.0, 0.0], (32, 1)),
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
        chunk0_flow_action_bonus=0.04,
    )

    assert result["chunks"][0]["flow_action_bonus"] == pytest.approx(0.04)
    assert result["chunks"][0]["navigation_reward"] == pytest.approx(0.04)
    assert all(chunk["flow_action_bonus"] == 0.0 for chunk in result["chunks"][1:])
    assert result["action_reward"] == pytest.approx((1.0 / 1.875) * 0.04)


def test_collision_is_a_soft_penalty_not_a_reward_gate(monkeypatch):
    module = _module()
    torch.manual_seed(7)
    video = torch.rand(5, 3, 24, 32)
    monkeypatch.setattr(module, "_load_gt_video", lambda *args: video.clone())
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {"collided": True, "collision_step": 2, "out_of_bounds": False},
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda *args, **kwargs: {
            "distances": np.asarray([4.0, 2.0]),
            "snapped": np.asarray([False, True]),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = CAM_TO_NAV
    actions = torch.zeros(8, 32)
    actions[:, 0] = 0.1
    result = module.compute_score(
        video,
        actions,
        {
            "video_path": "unused.mp4",
            "future_start": 0,
            "future_frames": 5,
            "start_extrinsic": extrinsic.tolist(),
            "q01": [-1, -1, -1],
            "q99": [1, 1, 1],
            "nav_action_scale": 1.0,
            "scene_dir": "unused",
            "gt_world_xy": [[0.1 * index, 0.0] for index in range(9)],
            "goal_world_xy": [100, 100],
            "stop_radius_m": 1.5,
        },
    )
    assert result["action_collision"] == 1.0
    assert result["action_collision_penalty"] == pytest.approx(-0.10)
    assert result["action_reward"] > 0.0


def test_chunk_collision_combines_hard_event_and_soft_clearance(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_CREDIT_ENABLED", "true")
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_COLLISION_PENALTY_WEIGHT", "0.40")
    monkeypatch.setenv("WAM_COLLISION_SOFT_ENABLED", "true")
    monkeypatch.setenv("WAM_COLLISION_SOFT_MARGIN_PX", "4")
    monkeypatch.setenv("WAM_COLLISION_SOFT_PENALTY_WEIGHT", "0.20")
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda points, *args, **kwargs: {
            "distances": np.linspace(8.0, 4.0, len(points)),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    collision_calls = 0

    def fake_collision(*args, **kwargs):
        nonlocal collision_calls
        collided = collision_calls == 0
        collision_calls += 1
        return {
            "collided": collided,
            "collision_step": 2 if collided else -1,
            "out_of_bounds": False,
        }

    risks = iter([1.0, 0.25, 0.0, 0.5])
    monkeypatch.setattr(module, "trajectory_collides", fake_collision)
    monkeypatch.setattr(
        module,
        "trajectory_clearance_risk",
        lambda *args, **kwargs: {
            "risk": next(risks),
            "min_clearance_px": 3.0,
            "evaluated_pixels": 8,
            "out_of_bounds": False,
        },
    )
    predicted = np.stack(
        [np.arange(33, dtype=np.float32) * 0.1, np.zeros(33, dtype=np.float32)],
        axis=1,
    )
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": [100.0, 100.0],
            "gt_world_xy": predicted.copy(),
            "stop_radius_m": 1.5,
        },
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    metrics = result["metrics"]
    assert metrics["action_chunk_0_collision_hard_penalty"] == pytest.approx(-0.40)
    assert metrics["action_chunk_0_collision_soft_penalty"] == pytest.approx(-0.20)
    assert metrics["action_chunk_0_collision_penalty"] == pytest.approx(-0.60)
    assert metrics["action_chunk_1_collision_soft_risk"] == pytest.approx(0.25)
    assert metrics["action_chunk_1_collision_soft_penalty"] == pytest.approx(-0.05)
    assert metrics["action_chunk_2_collision_soft_penalty"] == pytest.approx(0.0)
    assert metrics["action_chunk_3_collision_soft_penalty"] == pytest.approx(-0.10)


def test_collision_credit_removes_collision_from_navigation_stream(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_CREDIT_ENABLED", "true")
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_COLLISION_CREDIT_ENABLED", "true")
    monkeypatch.setenv("WAM_COLLISION_PENALTY_WEIGHT", "0.40")
    monkeypatch.setenv("WAM_COLLISION_SOFT_ENABLED", "true")
    monkeypatch.setenv("WAM_COLLISION_SOFT_MARGIN_PX", "4")
    monkeypatch.setenv("WAM_COLLISION_SOFT_PENALTY_WEIGHT", "0.20")
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda points, *args, **kwargs: {
            "distances": np.linspace(8.0, 4.0, len(points)),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    calls = 0

    def fake_collision(*args, **kwargs):
        nonlocal calls
        collided = calls == 0
        calls += 1
        return {
            "collided": collided,
            "collision_step": 2 if collided else -1,
            "out_of_bounds": False,
        }

    monkeypatch.setattr(module, "trajectory_collides", fake_collision)
    monkeypatch.setattr(
        module,
        "trajectory_clearance_risk",
        lambda *args, **kwargs: {
            "risk": 1.0,
            "min_clearance_px": 2.0,
            "evaluated_pixels": 8,
            "out_of_bounds": False,
        },
    )
    predicted = np.stack(
        [np.arange(33, dtype=np.float32) * 0.1, np.zeros(33, dtype=np.float32)],
        axis=1,
    )
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": [100.0, 100.0],
            "gt_world_xy": predicted.copy(),
            "stop_radius_m": 1.5,
        },
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    metrics = result["metrics"]
    assert metrics["action_collision_credit_enabled"] == 1.0
    assert metrics["action_chunk_0_collision_reward"] == pytest.approx(-0.60)
    assert metrics["action_chunk_0_collision_active"] == 1.0
    # Collision remains observable but is no longer double-counted inside the
    # navigation reward that creates action_advantages.
    assert metrics["action_chunk_0_nav_reward"] > -0.60


@pytest.mark.parametrize(
    ("mode", "expected_mode_id"),
    [("softspl", 0.0), ("signed_progress_length", 1.0)],
)
def test_four_chunk_action_reward_scores_full_horizon(
    monkeypatch,
    mode,
    expected_mode_id,
):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_CREDIT_ENABLED", "true")
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", mode)
    monkeypatch.setenv("WAM_ACTION_REWARD_MIN", "-1")
    monkeypatch.setenv("WAM_ACTION_REWARD_MAX", "1")
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda *args, **kwargs: {
            "distances": np.asarray([8.0, 6.0, 5.0, 5.5, 4.0]),
            "snapped": np.asarray([False] * 5),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    collision_calls = []

    def fake_collision(trajectory, *args, **kwargs):
        collision_calls.append(np.asarray(trajectory).copy())
        collided = len(collision_calls) == 3
        return {
            "collided": collided,
            "collision_step": 2 if collided else -1,
            "out_of_bounds": False,
        }

    monkeypatch.setattr(module, "trajectory_collides", fake_collision)
    predicted = np.stack(
        [np.arange(33, dtype=np.float32) * 0.1, np.zeros(33, dtype=np.float32)],
        axis=1,
    )
    gt = np.stack(
        [np.arange(33, dtype=np.float32) * 0.08, np.zeros(33, dtype=np.float32)],
        axis=1,
    )
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": [100.0, 100.0],
            "gt_world_xy": gt,
            "stop_radius_m": 1.5,
        },
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    assert len(collision_calls) == 4
    assert all(len(chunk) == 9 for chunk in collision_calls)
    assert len(result["chunks"]) == 4
    assert result["metrics"]["action_chunk_count"] == 4.0
    assert result["metrics"]["action_chunk_reward_mode_id"] == expected_mode_id
    assert result["metrics"]["action_any_collision"] == 1.0
    assert result["metrics"]["action_first_collision_step"] == 18.0
    assert result["metrics"]["action_first_collision_chunk"] == 2.0
    assert result["metrics"]["action_chunk_2_collision"] == 1.0
    for index in range(4):
        assert f"action_chunk_{index}_reward" in result["metrics"]
    normalized = np.asarray([1.0, 0.5, 0.25, 0.125]) / 1.875
    expected = sum(normalized[index] * result["metrics"][f"action_chunk_{index}_reward"] for index in range(4))
    assert result["action_reward"] == pytest.approx(expected)


def test_v9_freezes_at_first_collision_and_masks_later_chunk_credit(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_CREDIT_ENABLED", "true")
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "true")

    predicted = np.stack(
        [np.arange(33, dtype=np.float32) * 0.1, np.zeros(33, dtype=np.float32)],
        axis=1,
    )
    frozen = predicted.copy()
    frozen[18:] = frozen[17]
    monkeypatch.setattr(
        module,
        "freeze_trajectory_at_first_collision",
        lambda *args, **kwargs: {
            "trajectory_world_xy": frozen.copy(),
            "collided": True,
            "collision_step": 18,
            "out_of_bounds": False,
        },
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda points, *args, **kwargs: {
            "distances": np.asarray([8.0, 6.0, 5.0, 5.0, 5.0]),
            "snapped": np.asarray([False] * 5),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )
    raw_collision_calls = []

    def fake_raw_collision(trajectory, *args, **kwargs):
        raw_collision_calls.append(np.asarray(trajectory).copy())
        collided = len(raw_collision_calls) >= 3
        return {
            "collided": collided,
            "collision_step": 2 if collided else -1,
            "out_of_bounds": False,
        }

    monkeypatch.setattr(module, "trajectory_collides", fake_raw_collision)
    gt = np.stack(
        [np.arange(33, dtype=np.float32) * 0.08, np.zeros(33, dtype=np.float32)],
        axis=1,
    )
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": [100.0, 100.0],
            "gt_world_xy": gt,
            "stop_radius_m": 1.5,
        },
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    metrics = result["metrics"]
    assert len(raw_collision_calls) == 4
    assert metrics["action_collision_stop_enabled"] == 1.0
    assert metrics["action_any_collision"] == 1.0
    assert metrics["action_first_collision_step"] == 18.0
    assert metrics["action_first_collision_chunk"] == 2.0
    assert metrics["action_chunk_2_collision"] == 1.0
    assert metrics["action_chunk_2_active"] == 1.0
    assert metrics["action_chunk_3_raw_collision"] == 1.0
    assert metrics["action_chunk_3_collision"] == 0.0
    assert metrics["action_chunk_3_active"] == 0.0
    assert metrics["action_chunk_3_post_collision_masked"] == 1.0
    assert metrics["action_chunk_3_reward"] == 0.0
    assert result["chunks"][2]["predicted_path_length"] == pytest.approx(0.8)
    assert result["chunks"][2]["executed_predicted_path_length"] == pytest.approx(0.1)
    assert result["chunks"][3]["predicted_path_length"] == pytest.approx(0.8)
    assert result["chunks"][3]["executed_predicted_path_length"] == 0.0


def test_v9_1_length_regularizer_uses_full_chunk_after_goal_entry(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_CREDIT_ENABLED", "true")
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "true")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_ROUTE_DEVIATION_WEIGHT", "0")
    monkeypatch.setenv("WAM_REVERSE_DIRECTION_WEIGHT", "0")
    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "false")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )

    goal = np.asarray([0.0, 0.0], dtype=np.float32)

    def fake_geodesic(points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float32)
        return {
            "distances": np.linalg.norm(points - goal[None, :], axis=1),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        }

    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        fake_geodesic,
    )
    predicted = np.repeat([[1.5, 0.0]], 33, axis=0).astype(np.float32)
    predicted[:9, 0] = np.asarray(
        [2.0, 1.75, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5],
        dtype=np.float32,
    )
    gt = np.stack(
        [2.0 - np.arange(33, dtype=np.float32) * 0.25, np.zeros(33)],
        axis=1,
    ).astype(np.float32)
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": goal,
            "gt_world_xy": gt,
            "stop_radius_m": 1.5,
        },
        predicted_local_deltas=np.zeros((32, 3), dtype=np.float32),
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    chunk = result["chunks"][0]
    metrics = result["metrics"]
    assert chunk["evaluated_actions"] == 2
    assert chunk["navigation_predicted_path_length"] == pytest.approx(0.5)
    assert chunk["navigation_reference_path_length"] == pytest.approx(0.5)
    assert chunk["predicted_path_length"] == pytest.approx(6.5)
    assert chunk["reference_path_length"] == pytest.approx(2.0)
    assert chunk["path_efficiency"] == pytest.approx((2.0 / 6.5) ** 2)
    assert metrics["action_chunk_0_nav_pred_path_length"] == pytest.approx(0.5)
    assert metrics["action_chunk_0_pred_path_length"] == pytest.approx(6.5)


def test_stop_well_only_charges_motion_energy(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_STOP_WELL_ENERGY_WEIGHT", "0.20")
    monkeypatch.setenv("WAM_STOP_WELL_XY_DEADZONE_M", "0.00")
    monkeypatch.setenv("WAM_STOP_WELL_YAW_WEIGHT", "0.06")
    monkeypatch.setenv("WAM_STOP_WELL_YAW_DEADZONE_RAD", "0.02")
    monkeypatch.setenv("WAM_STOP_WELL_YAW_SCALE_RAD", "0.25")

    def score(trajectory, local_deltas=None):
        if local_deltas is None:
            local_deltas = np.zeros((32, 3), dtype=np.float32)
        return module._compute_stop_well_chunks(
            trajectory,
            local_deltas,
            reached_step=0,
            left_step=-1,
            left_goal_penalty=-1.0,
            goal_radius=1.5,
            chunk_size=8,
            chunk_count=4,
            max_transition_exclusive=32,
        )

    still = np.repeat([[1.0, 0.0]], 33, axis=0).astype(np.float32)
    still_result = score(still)
    assert still_result["reward"][0] == pytest.approx(0.0)

    inward = still.copy()
    inward[1:, 0] = 0.5
    inward_result = score(inward)
    assert inward_result["reward"][0] < 0.0

    inner_motion = np.repeat([[0.4, 0.0]], 33, axis=0).astype(np.float32)
    inner_motion[1:, 0] = 0.2
    inner_result = score(inner_motion)
    assert inner_result["reward"][0] < 0.0

    tangent = still.copy()
    tangent[1:, 1] = 0.5
    tangent_result = score(tangent)
    assert tangent_result["reward"][0] < 0.0

    yaw = np.zeros((32, 3), dtype=np.float32)
    yaw[0, 2] = 0.25
    yaw_result = score(still, yaw)
    assert yaw_result["reward"][0] < 0.0


def test_route_corridor_and_reverse_direction_detect_large_failures():
    module = _module()
    reference = np.stack(
        [np.linspace(0.0, 4.0, 33), np.zeros(33)],
        axis=1,
    ).astype(np.float32)
    aligned = np.stack(
        [np.linspace(0.0, 1.0, 9), np.zeros(9)],
        axis=1,
    ).astype(np.float32)
    aligned_diag = module._trajectory_route_diagnostics(
        aligned,
        reference,
        free_radius_m=0.75,
        deviation_scale_m=1.50,
        motion_floor_m=0.03,
    )
    assert aligned_diag["deviation_score"] == pytest.approx(0.0)
    assert aligned_diag["reverse_direction_score"] == pytest.approx(0.0)

    off_route = aligned.copy()
    off_route[:, 1] = 2.25
    off_route_diag = module._trajectory_route_diagnostics(
        off_route,
        reference,
        free_radius_m=0.75,
        deviation_scale_m=1.50,
        motion_floor_m=0.03,
    )
    assert off_route_diag["deviation_score"] == pytest.approx(1.0)

    reversed_route = aligned[::-1].copy()
    reverse_diag = module._trajectory_route_diagnostics(
        reversed_route,
        reference,
        free_radius_m=0.75,
        deviation_scale_m=1.50,
        motion_floor_m=0.03,
    )
    assert reverse_diag["deviation_score"] == pytest.approx(0.0)
    assert reverse_diag["reverse_direction_score"] == pytest.approx(1.0)


def test_v9_1_switches_from_navigation_to_stop_credit_after_goal_entry(
    monkeypatch,
):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_CREDIT_ENABLED", "true")
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "true")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )

    goal = np.asarray([0.0, 0.0], dtype=np.float32)

    def fake_geodesic(points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float32)
        return {
            "distances": np.linalg.norm(points - goal[None, :], axis=1),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        }

    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        fake_geodesic,
    )
    predicted = np.zeros((33, 2), dtype=np.float32)
    predicted[:9, 0] = np.linspace(3.0, 1.0, 9)
    predicted[9:, 0] = 1.0
    gt = np.stack(
        [np.linspace(3.0, -5.0, 33), np.zeros(33)],
        axis=1,
    ).astype(np.float32)
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": goal,
            "gt_world_xy": gt,
            "stop_radius_m": 1.5,
        },
        predicted_local_deltas=np.zeros((32, 3), dtype=np.float32),
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    metrics = result["metrics"]
    assert metrics["action_stop_credit_enabled"] == 1.0
    assert metrics["action_chunk_0_nav_active"] == 1.0
    assert metrics["action_chunk_1_nav_active"] == 0.0
    assert metrics["action_chunk_1_nav_reward"] == 0.0
    assert metrics["action_chunk_1_stop_active"] == 1.0
    assert metrics["action_chunk_1_stop_reward"] == pytest.approx(0.0)


def test_compact_metrics_preserve_stop_credit_training_contract(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_REWARD_METRICS_COMPACT", "true")
    metrics = module._compact_reward_metrics(
        {
            "score": 0.25,
            "action_stop_credit_enabled": 1.0,
            "action_chunk_0_nav_reward": 0.4,
            "action_chunk_0_nav_active": 1.0,
            "action_chunk_0_stop_reward": -0.2,
            "action_chunk_0_stop_active": 1.0,
            "action_chunk_0_hard_stop_executed": 1.0,
            "action_terminal_safety_reward": -0.6,
            "historical_debug_only": 123.0,
        }
    )

    assert metrics["action_stop_credit_enabled"] == 1.0
    assert metrics["action_chunk_0_nav_reward"] == 0.4
    assert metrics["action_chunk_0_stop_reward"] == -0.2
    assert metrics["action_chunk_0_hard_stop_executed"] == 1.0
    assert "action_terminal_safety_reward" not in metrics
    assert "historical_debug_only" not in metrics


def test_yaw_credit_keeps_heading_continuous_across_credit_chunks():
    module = _module()
    horizon = 32
    local_deltas = np.zeros((horizon, 3), dtype=np.float32)
    local_deltas[:8, 0] = 0.1
    local_deltas[8:, 0] = -0.1

    path_heading = module._path_heading_sequence(local_deltas[:, :2])
    local_deltas[:, 2] = module._wrap_angle(np.diff(path_heading))
    gt = np.stack(
        [np.linspace(0.0, 3.2, horizon + 1), np.zeros(horizon + 1)],
        axis=1,
    ).astype(np.float32)

    diagnostics = module._trajectory_yaw_diagnostics(
        local_deltas,
        gt,
        free_angle_deg=0.0,
        gross_angle_deg=90.0,
        motion_floor_m=0.03,
    )
    assert np.max(diagnostics["path_score"]) == pytest.approx(0.0, abs=1e-7)
    assert np.max(diagnostics["rate_score"]) == pytest.approx(0.0, abs=1e-7)
    # The second 8-step credit chunk starts with the accumulated pi-radian
    # heading. Resetting yaw at the chunk boundary would make this assertion fail.
    assert diagnostics["predicted_action_heading"][8] == pytest.approx(-np.pi, abs=1e-6)
    assert np.mean(diagnostics["gross_gt_score"][8:]) == pytest.approx(1.0, abs=1e-6)


def test_v10_pure_yaw_is_not_masked_by_translation_motion_floor():
    module = _module()
    local_deltas = np.zeros((8, 3), dtype=np.float32)
    local_deltas[0, 2] = 0.20
    gt = np.stack([np.arange(9, dtype=np.float32) * 0.1, np.zeros(9)], axis=1)

    diagnostics = module._trajectory_yaw_diagnostics(
        local_deltas,
        gt,
        free_angle_deg=0.0,
        gross_angle_deg=90.0,
        motion_floor_m=0.03,
        rotation_floor_rad=0.01,
    )

    assert diagnostics["translation_moving"][0] == 0
    assert diagnostics["rotation_moving"][0] == 1
    assert diagnostics["pure_yaw"][0] == 1
    assert diagnostics["moving"][0] == 1
    assert diagnostics["rate_score"][0] > 0.0


def test_gn0_stop_supervises_componentwise_stop_in_every_credit_chunk(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_DEPLOY_STOP_SEMANTICS_ENABLED", "true")
    monkeypatch.setenv("WAM_DEPLOY_STOP_EPS", "0.001")
    monkeypatch.setenv("WAM_CORRECT_STOP_BONUS", "0.15")
    monkeypatch.setenv("WAM_PREMATURE_STOP_PENALTY", "0.50")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "true")
    monkeypatch.setenv("WAM_STOP_WELL_ENERGY_WEIGHT", "0")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "false")
    monkeypatch.setenv("WAM_NDTW_WEIGHT", "0")
    monkeypatch.setenv("WAM_GEODESIC_BACKTRACK_WEIGHT", "0")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda points, goal, *args, **kwargs: {
            "distances": np.linalg.norm(np.asarray(points) - np.asarray(goal)[None, :], axis=1),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )

    predicted = np.stack([np.arange(33, dtype=np.float32) * 0.1, np.zeros(33)], axis=1)
    local_deltas = np.zeros((32, 3), dtype=np.float32)
    local_deltas[:, 0] = 0.1
    local_deltas[10] = 0.0
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": [1.0, 0.0],
            "gt_world_xy": predicted.copy(),
            "stop_radius_m": 0.10,
        },
        predicted_local_deltas=local_deltas,
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    metrics = result["metrics"]
    assert metrics["action_hard_stop_step"] == 10.0
    assert metrics["action_hard_stop_correct"] == 1.0
    assert metrics["action_chunk_1_hard_stop_emitted"] == 1.0
    assert metrics["action_chunk_1_hard_stop_local_step"] == 2.0
    assert metrics["action_chunk_1_hard_stop_event_reward"] == pytest.approx(0.15)
    assert metrics["action_chunk_1_active"] == 1.0
    assert metrics["action_chunk_2_active"] == 1.0
    assert metrics["action_chunk_3_active"] == 1.0


def test_gn0_stop_supervises_low_translation_in_later_credit_chunk():
    module = _module()
    local_deltas = np.zeros((32, 3), dtype=np.float32)
    local_deltas[:8, 0] = 0.1
    local_deltas[8:16, 0] = 0.01
    local_deltas[16:, 0] = 0.1

    result = module.detect_gn0_stops_per_credit_chunk(
        local_deltas,
        chunk_size=8,
        chunk_translation_threshold_m=0.15,
        stop_eps=1e-3,
    )

    np.testing.assert_array_equal(result["chunk_emitted_stop"], [False, True, False, False])
    np.testing.assert_array_equal(result["chunk_stop_steps"], [-1, 8, -1, -1])
    np.testing.assert_array_equal(result["chunk_stop_reason_ids"], [0, 1, 0, 0])
    np.testing.assert_allclose(result["chunk_xy_path_lengths"], [0.8, 0.08, 0.8, 0.8])


def test_chunk_motion_stop_supports_path_length_anti_oscillation_mode():
    module = _module()
    local_deltas = np.zeros((16, 3), dtype=np.float32)
    # Chunk 0 returns to its start, but travels 0.8 m and must not be STOP.
    local_deltas[:4, 0] = 0.1
    local_deltas[4:8, 0] = -0.1
    # Chunk 1 only travels 0.08 m and is interpreted as STOP at its start.
    local_deltas[8:, 0] = 0.01
    world_xy = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.cumsum(local_deltas[:, :2], axis=0),
        ],
        axis=0,
    )

    result = module.freeze_trajectory_at_chunk_motion_stop(
        local_deltas,
        world_xy,
        chunk_size=8,
        xy_path_threshold_m=0.15,
        motion_metric="path_length",
    )

    assert result["emitted_stop"]
    assert result["stop_chunk"] == 1
    assert result["stop_step"] == 8
    np.testing.assert_allclose(result["chunk_xy_path_lengths"], [0.8, 0.08])
    np.testing.assert_allclose(result["chunk_xy_net_displacements"], [0.0, 0.08], atol=1e-7)
    np.testing.assert_allclose(result["local_deltas"][8:], 0.0)
    np.testing.assert_allclose(
        result["trajectory_world_xy"][9:],
        np.repeat(world_xy[8][None, :], 8, axis=0),
    )


def test_chunk_motion_stop_defaults_to_accumulated_xy_net_displacement():
    module = _module()
    local_deltas = np.zeros((16, 3), dtype=np.float32)
    local_deltas[:4, 0] = 0.1
    local_deltas[4:8, 0] = -0.1
    local_deltas[8:, 0] = 0.1
    world_xy = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.cumsum(local_deltas[:, :2], axis=0),
        ],
        axis=0,
    )

    result = module.freeze_trajectory_at_chunk_motion_stop(
        local_deltas,
        world_xy,
        chunk_size=8,
        xy_path_threshold_m=0.15,
    )

    assert result["motion_metric"] == "net_displacement"
    assert result["stop_chunk"] == 0
    assert result["stop_step"] == 0
    np.testing.assert_allclose(result["chunk_xy_path_lengths"], [0.8, 0.8])
    np.testing.assert_allclose(result["chunk_xy_net_displacements"], [0.0, 0.8], atol=1e-7)


def test_gn0_stop_checks_only_first_chunk_and_uses_xy_path_length():
    module = _module()
    local_deltas = np.zeros((32, 3), dtype=np.float32)
    # First deployment chunk oscillates back to the origin but travels 0.8 m.
    local_deltas[:4, 0] = 0.1
    local_deltas[4:8, 0] = -0.1
    # A later hypothetical chunk is nearly stationary. GN0 would have replanned
    # before reaching it, so it must not emit STOP for this model response.
    local_deltas[8:16, 0] = 0.01
    local_deltas[16:, 0] = 0.1
    world_xy = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.cumsum(local_deltas[:, :2], axis=0),
        ],
        axis=0,
    )

    result = module.freeze_trajectory_at_gn0_execution_stop(
        local_deltas,
        world_xy,
        action_num=8,
        chunk_translation_threshold_m=0.15,
        stop_eps=1e-3,
    )

    assert not result["emitted_stop"]
    assert result["stop_step"] == -1
    assert result["stop_reason"] == "none"
    np.testing.assert_allclose(result["chunk_xy_path_lengths"][:2], [0.8, 0.08])


def test_gn0_stop_applies_chunk_rule_before_componentwise_rule():
    module = _module()
    local_deltas = np.zeros((32, 3), dtype=np.float32)
    local_deltas[:8, 0] = 0.01
    local_deltas[8:, 0] = 0.1
    world_xy = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.cumsum(local_deltas[:, :2], axis=0),
        ],
        axis=0,
    )

    result = module.freeze_trajectory_at_gn0_execution_stop(
        local_deltas,
        world_xy,
        action_num=8,
        chunk_translation_threshold_m=0.15,
        stop_eps=1e-3,
    )

    assert result["emitted_stop"]
    assert result["stop_step"] == 0
    assert result["stop_reason"] == "chunk_low_translation"
    np.testing.assert_allclose(result["local_deltas"], 0.0)


def test_gn0_stop_falls_back_to_componentwise_rule_inside_first_chunk():
    module = _module()
    local_deltas = np.zeros((32, 3), dtype=np.float32)
    local_deltas[:8, 0] = 0.1
    local_deltas[3] = 0.0
    local_deltas[8:, 0] = 0.1
    world_xy = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.cumsum(local_deltas[:, :2], axis=0),
        ],
        axis=0,
    )

    result = module.freeze_trajectory_at_gn0_execution_stop(
        local_deltas,
        world_xy,
        action_num=8,
        chunk_translation_threshold_m=0.15,
        stop_eps=1e-3,
    )

    assert result["emitted_stop"]
    assert result["stop_step"] == 3
    assert result["stop_reason"] == "componentwise_epsilon"
    np.testing.assert_allclose(result["local_deltas"][3:], 0.0)


def test_chunk_motion_stop_freezes_from_chunk_start(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_CHUNK_MOTION_STOP_ENABLED", "true")
    monkeypatch.setenv("WAM_CHUNK_MOTION_STOP_THRESHOLD_M", "0.15")
    monkeypatch.setenv("WAM_CHUNK_MOTION_STOP_METRIC", "net_displacement")
    monkeypatch.setenv("WAM_DEPLOY_STOP_SEMANTICS_ENABLED", "false")
    monkeypatch.setenv("WAM_CORRECT_STOP_BONUS", "0.15")
    monkeypatch.setenv("WAM_PREMATURE_STOP_PENALTY", "0.50")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "true")
    monkeypatch.setenv("WAM_STOP_WELL_ENERGY_WEIGHT", "0")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "false")
    monkeypatch.setenv("WAM_NDTW_WEIGHT", "0")
    monkeypatch.setenv("WAM_GEODESIC_BACKTRACK_WEIGHT", "0")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda points, goal, *args, **kwargs: {
            "distances": np.linalg.norm(np.asarray(points) - np.asarray(goal)[None, :], axis=1),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )

    local_deltas = np.zeros((32, 3), dtype=np.float32)
    local_deltas[:8, 0] = 0.1
    local_deltas[8:16, 0] = 0.01
    local_deltas[16:, 0] = 0.1
    predicted = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.cumsum(local_deltas[:, :2], axis=0),
        ],
        axis=0,
    )
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": [0.8, 0.0],
            "gt_world_xy": predicted.copy(),
            "stop_radius_m": 0.10,
        },
        predicted_local_deltas=local_deltas,
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    metrics = result["metrics"]
    assert metrics["action_chunk_motion_stop_enabled"] == 1.0
    assert metrics["action_chunk_motion_stop_chunk"] == 1.0
    assert metrics["action_hard_stop_step"] == 8.0
    assert metrics["action_hard_stop_correct"] == 1.0
    assert metrics["action_chunk_0_stop_trigger_xy_path_length_m"] == pytest.approx(0.8)
    assert metrics["action_chunk_1_stop_trigger_xy_path_length_m"] == pytest.approx(0.08)
    assert metrics["action_chunk_0_stop_trigger_xy_net_displacement_m"] == pytest.approx(0.8)
    assert metrics["action_chunk_1_stop_trigger_xy_net_displacement_m"] == pytest.approx(0.08)
    assert metrics["action_chunk_1_hard_stop_event_reward"] == pytest.approx(0.15)
    assert metrics["action_chunk_2_active"] == 0.0
    assert metrics["action_chunk_3_active"] == 0.0


def test_v10_ndtw_and_geodesic_backtrack_use_executed_path(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "false")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "false")
    monkeypatch.setenv("WAM_REVERSE_DIRECTION_WEIGHT", "0")
    monkeypatch.setenv("WAM_NDTW_WEIGHT", "0.20")
    monkeypatch.setenv("WAM_NDTW_SUCCESS_DISTANCE_M", "0.50")
    monkeypatch.setenv("WAM_GEODESIC_BACKTRACK_WEIGHT", "0.25")
    monkeypatch.setenv("WAM_GEODESIC_BACKTRACK_TOLERANCE_M", "0.01")
    monkeypatch.setenv("WAM_GEODESIC_BACKTRACK_DENOM_FLOOR_M", "0.10")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )
    goal = np.asarray([10.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        lambda points, *args, **kwargs: {
            "distances": np.linalg.norm(np.asarray(points) - goal[None, :], axis=1),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        },
    )

    predicted_x = np.asarray([0, 1, 2, 1, 0, 1, 2, 3, 4] + list(range(5, 29)))
    predicted = np.stack([predicted_x, np.zeros(33)], axis=1).astype(np.float32)
    gt = np.stack([np.arange(33), np.zeros(33)], axis=1).astype(np.float32)
    result = module._compute_chunked_action_reward(
        predicted,
        {
            "scene_dir": "unused",
            "goal_world_xy": goal,
            "gt_world_xy": gt,
            "stop_radius_m": 1.5,
        },
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    metrics = result["metrics"]
    assert 0.0 < metrics["action_chunk_0_ndtw"] < 1.0
    assert metrics["action_chunk_0_ndtw_bonus"] > 0.0
    assert metrics["action_chunk_0_geodesic_backtrack_distance_m"] == pytest.approx(1.98)
    assert metrics["action_chunk_0_geodesic_backtrack_penalty"] < 0.0


def test_yaw_penalties_reduce_navigation_reward(monkeypatch):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "false")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_YAW_PATH_CONSISTENCY_WEIGHT", "0.12")
    monkeypatch.setenv("WAM_YAW_RATE_CONSISTENCY_WEIGHT", "0.04")
    monkeypatch.setenv("WAM_YAW_GROSS_GT_WEIGHT", "0.08")
    monkeypatch.setenv("WAM_YAW_FREE_ANGLE_DEG", "0")
    monkeypatch.setenv("WAM_YAW_GROSS_ANGLE_DEG", "90")
    monkeypatch.setenv("WAM_YAW_MOTION_FLOOR_M", "0.03")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )

    def fake_geodesic(points, *args, **kwargs):
        count = len(points)
        return {
            "distances": np.linspace(10.0, 6.0, count),
            "snapped": np.zeros(count, dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        }

    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        fake_geodesic,
    )
    predicted = np.stack(
        [np.linspace(0.0, 3.2, 33), np.zeros(33)],
        axis=1,
    ).astype(np.float32)
    gt = predicted.copy()
    local_deltas = np.zeros((32, 3), dtype=np.float32)
    local_deltas[:, 0] = 0.1
    local_deltas[:, 2] = 0.25
    ground_truth = {
        "scene_dir": "unused",
        "goal_world_xy": [100.0, 0.0],
        "gt_world_xy": gt,
        "stop_radius_m": 1.5,
    }

    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "false")
    baseline = module._compute_chunked_action_reward(
        predicted,
        ground_truth,
        predicted_local_deltas=local_deltas,
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )
    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "true")
    yaw_scored = module._compute_chunked_action_reward(
        predicted,
        ground_truth,
        predicted_local_deltas=local_deltas,
        action_reward_min=-1.0,
        action_reward_max=1.0,
        path_efficiency_power=2.0,
    )

    assert yaw_scored["base_action_reward"] < baseline["base_action_reward"]
    metrics = yaw_scored["metrics"]
    assert metrics["action_yaw_path_consistency_score"] > 0.0
    assert metrics["action_yaw_rate_consistency_score"] > 0.0
    assert metrics["action_yaw_gross_gt_score"] > 0.0
    assert metrics["action_yaw_path_consistency_penalty"] < 0.0


def test_goal_potential_credit_prevents_hover_farming_and_rewards_entry(
    monkeypatch,
):
    module = _module()
    monkeypatch.setenv("WAM_ACTION_CHUNK_SIZE", "8")
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    monkeypatch.setenv("WAM_ACTION_CHUNK_REWARD_MODE", "signed_progress_length")
    monkeypatch.setenv("WAM_STOP_WELL_ENABLED", "true")
    monkeypatch.setenv("WAM_COLLISION_STOP_ENABLED", "false")
    monkeypatch.setenv("WAM_COLLISION_PENALTY_WEIGHT", "0")
    monkeypatch.setenv("WAM_ROUTE_DEVIATION_WEIGHT", "0")
    monkeypatch.setenv("WAM_REVERSE_DIRECTION_WEIGHT", "0")
    monkeypatch.setenv("WAM_YAW_CREDIT_ENABLED", "false")
    monkeypatch.setenv("WAM_SIGNED_PROGRESS_WEIGHT", "0")
    monkeypatch.setenv("WAM_SYMMETRIC_LENGTH_WEIGHT", "0")
    monkeypatch.setenv("WAM_SIGNED_GOAL_WEIGHT", "1")
    monkeypatch.setenv("WAM_GOAL_SCORE_TEMPERATURE_M", "0.75")
    monkeypatch.setenv("WAM_GOAL_SCORE_USE_POTENTIAL_DELTA", "true")
    monkeypatch.setenv("WAM_GOAL_ENTRY_BONUS", "0.15")
    monkeypatch.setattr(
        module,
        "trajectory_collides",
        lambda *args, **kwargs: {
            "collided": False,
            "collision_step": -1,
            "out_of_bounds": False,
        },
    )

    goal = np.asarray([0.0, 0.0], dtype=np.float32)

    def fake_geodesic(points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float32)
        return {
            "distances": np.linalg.norm(points - goal[None, :], axis=1),
            "snapped": np.zeros(len(points), dtype=bool),
            "goal_snapped": False,
            "goal_valid": True,
        }

    monkeypatch.setattr(
        module,
        "occupancy_geodesic_distances_to_goal",
        fake_geodesic,
    )
    local_deltas = np.zeros((32, 3), dtype=np.float32)

    def score(predicted):
        predicted = np.asarray(predicted, dtype=np.float32)
        return module._compute_chunked_action_reward(
            predicted,
            {
                "scene_dir": "unused",
                "goal_world_xy": goal,
                "gt_world_xy": predicted.copy(),
                "stop_radius_m": 1.5,
            },
            predicted_local_deltas=local_deltas,
            action_reward_min=-1.0,
            action_reward_max=1.0,
            path_efficiency_power=2.0,
        )

    # Remaining just outside the radius still has a positive absolute
    # diagnostic score, but receives exactly zero goal credit and no arrival.
    hover = score(np.repeat([[1.5001, 0.0]], 33, axis=0))
    hover_chunk = hover["chunks"][0]
    assert hover_chunk["goal_score"] > 0.0
    assert hover_chunk["goal_potential_delta"] == pytest.approx(0.0)
    assert hover_chunk["goal_credit"] == pytest.approx(0.0)
    assert hover_chunk["goal_entry_bonus"] == pytest.approx(0.0)
    assert hover_chunk["navigation_reward"] == pytest.approx(0.0)
    assert hover["metrics"]["action_goal_entry_chunk"] == -1.0

    # The crossing transition gets the signed potential improvement and one
    # non-repeatable bonus; later chunks do not receive the arrival again.
    crossing_trajectory = np.repeat([[1.4, 0.0]], 33, axis=0)
    crossing_trajectory[0] = [2.0, 0.0]
    crossing = score(crossing_trajectory)
    expected_delta = np.exp(-1.4 / 0.75) - np.exp(-2.0 / 0.75)
    assert crossing["chunks"][0]["goal_credit"] == pytest.approx(expected_delta)
    assert crossing["chunks"][0]["goal_entry"] is True
    assert crossing["chunks"][0]["goal_entry_bonus"] == pytest.approx(0.15)
    assert crossing["chunks"][0]["navigation_reward"] == pytest.approx(expected_delta + 0.15)
    assert crossing["metrics"]["action_goal_entry_chunk"] == 0.0
    assert all(chunk["goal_entry_bonus"] == 0.0 for chunk in crossing["chunks"][1:])

    # Moving away outside the region is explicitly worse than hovering.
    moving_away_trajectory = np.repeat([[2.0, 0.0]], 33, axis=0)
    moving_away_trajectory[0] = [1.6, 0.0]
    moving_away = score(moving_away_trajectory)
    assert moving_away["chunks"][0]["goal_credit"] < 0.0
    assert moving_away["chunks"][0]["navigation_reward"] < 0.0
