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

import json

import numpy as np
import pytest
from PIL import Image

from verl_omni.utils.reward_score.wam_navigation_reward import (
    CAM_TO_NAV,
    freeze_trajectory_at_first_collision,
    normalized_dtw,
    occupancy_geodesic_distances_to_goal,
    rollout_actions_to_world_xy,
    trajectory_clearance_risk,
    trajectory_collides,
    trajectory_path_length,
)


def _extrinsic(yaw=0.0, translation=(0.0, 0.0, 0.0)):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    value = np.eye(4, dtype=np.float32)
    value[:3, :3] = rotation @ CAM_TO_NAV
    value[:3, 3] = translation
    return value


def test_action_decode_rotates_local_path_into_world():
    actions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    world = rollout_actions_to_world_xy(
        actions,
        _extrinsic(np.pi / 2, (2, 3, 0)),
        q01=[-1, -1, -1],
        q99=[1, 1, 1],
        nav_action_scale=1.0,
    )
    np.testing.assert_allclose(world, [[2, 3], [2, 4], [2, 5]], atol=1e-5)


def test_ndtw_is_one_for_identical_paths_and_lower_for_shifted_paths():
    path = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    assert normalized_dtw(path, path) == pytest.approx(1.0)
    assert normalized_dtw(path + 1.0, path) < 1.0


def test_path_length_and_occ_geodesic_are_route_independent(tmp_path):
    occupancy = np.full((9, 9), 255, dtype=np.uint8)
    # A wall blocks the direct route; the only opening is at the top.
    occupancy[1:, 4] = 0
    Image.fromarray(occupancy).save(tmp_path / "occupancy.png")
    (tmp_path / "occupancy.json").write_text(json.dumps({"lower": [-9, -9, 0], "upper": [0, 0, 1]}), encoding="utf-8")

    def world_xy(pixel_x, pixel_y):
        return [-pixel_x - 0.5 - (-9), pixel_y + 0.5 - 0]

    start = world_xy(2, 7)
    endpoint = world_xy(2, 5)
    goal = world_xy(6, 7)
    result = occupancy_geodesic_distances_to_goal(
        [start, endpoint],
        goal,
        tmp_path,
        occupancy_margin_px=0,
        snap_radius_px=0,
    )
    assert np.isfinite(result["distances"]).all()
    assert result["distances"][1] < result["distances"][0]
    assert result["distances"][0] > np.linalg.norm(np.asarray(start) - np.asarray(goal))
    assert trajectory_path_length([start, endpoint]) == pytest.approx(2.0)


def test_collision_checks_segments_inflation_and_continuous_bounds(tmp_path):
    occupancy = np.full((20, 20), 255, dtype=np.uint8)
    occupancy[10, 10] = 0
    Image.fromarray(occupancy).save(tmp_path / "occupancy.png")
    (tmp_path / "occupancy.json").write_text(
        json.dumps({"lower": [-10, -10, 0], "upper": [10, 10, 1]}), encoding="utf-8"
    )

    # In this GN0 mapping world (0,0) maps close to pixel (9.5,9.5).
    result = trajectory_collides([[2, 0], [0, 0]], tmp_path, occupancy_margin_px=0)
    assert result["collided"] is True
    assert result["out_of_bounds"] is False

    result = trajectory_collides([[100, 0], [101, 0]], tmp_path, occupancy_margin_px=0)
    assert result["collided"] is True
    assert result["out_of_bounds"] is True

    resolved = freeze_trajectory_at_first_collision(
        [[2, 0], [0, 0], [-2, 0]],
        tmp_path,
        occupancy_margin_px=0,
    )
    assert resolved["collided"] is True
    assert resolved["collision_step"] == 1
    assert resolved["out_of_bounds"] is False
    np.testing.assert_allclose(
        resolved["trajectory_world_xy"],
        [[2, 0], [2, 0], [2, 0]],
    )

    legal = freeze_trajectory_at_first_collision(
        [[2, 2], [3, 2], [4, 2]],
        tmp_path,
        occupancy_margin_px=0,
    )
    assert legal["collided"] is False
    assert legal["collision_step"] == -1
    np.testing.assert_allclose(
        legal["trajectory_world_xy"],
        [[2, 2], [3, 2], [4, 2]],
    )


def test_clearance_risk_is_quadratic_between_hard_and_soft_margins(tmp_path):
    occupancy = np.full((21, 21), 255, dtype=np.uint8)
    occupancy[10, 10] = 0
    Image.fromarray(occupancy).save(tmp_path / "occupancy.png")
    (tmp_path / "occupancy.json").write_text(
        json.dumps({"lower": [-10, -10, 0], "upper": [11, 11, 1]}),
        encoding="utf-8",
    )

    def world_xy(pixel_x, pixel_y):
        return [9.5 - pixel_x, pixel_y - 10.5]

    hard = trajectory_clearance_risk(
        [world_xy(9, 12), world_xy(10, 12)],
        tmp_path,
        hard_margin_px=2,
        soft_margin_px=4,
    )
    assert hard["min_clearance_px"] == pytest.approx(2.0)
    assert hard["risk"] == pytest.approx(1.0)

    warning = trajectory_clearance_risk(
        [world_xy(9, 13), world_xy(10, 13)],
        tmp_path,
        hard_margin_px=2,
        soft_margin_px=4,
    )
    assert warning["min_clearance_px"] == pytest.approx(3.0)
    assert warning["risk"] == pytest.approx(0.25)

    safe = trajectory_clearance_risk(
        [world_xy(9, 14), world_xy(10, 14)],
        tmp_path,
        hard_margin_px=2,
        soft_margin_px=4,
    )
    assert safe["min_clearance_px"] == pytest.approx(4.0)
    assert safe["risk"] == pytest.approx(0.0)

    stationary = trajectory_clearance_risk(
        [world_xy(10, 3), world_xy(10, 3)],
        tmp_path,
        hard_margin_px=2,
        soft_margin_px=4,
    )
    assert stationary["evaluated_pixels"] == 0
    assert stationary["risk"] == pytest.approx(0.0)
