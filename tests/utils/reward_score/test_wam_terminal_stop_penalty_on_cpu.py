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

import pytest

from verl_omni.utils.reward_score.wam_terminal_stop_penalty import (
    apply_terminal_stop_penalty,
    compute_terminal_stop_penalty,
)


def test_stopping_at_goal_has_no_penalty():
    result = compute_terminal_stop_penalty([[0, 0], [1.6, 0], [1.61, 0], [1.60, 0]], [3, 0])
    assert result["reached_goal"] is True
    assert result["failed_to_stop"] is False
    assert result["terminal_stop_penalty"] == 0.0


def test_continuing_inside_goal_radius_is_not_penalized():
    result = compute_terminal_stop_penalty([[0, 0], [1.6, 0], [1.9, 0], [2.2, 0], [2.5, 0]], [3, 0])
    assert result["failed_to_stop"] is False
    assert result["left_goal"] is False
    assert result["terminal_stop_penalty"] == 0.0
    assert result["terminal_stop_reason"] == "remained_inside_goal"


def test_leaving_goal_after_reaching_gets_stronger_penalty():
    result = compute_terminal_stop_penalty([[0, 0], [1.6, 0], [1.0, 0]], [3, 0])
    assert result["left_goal"] is True
    assert result["terminal_stop_penalty"] == pytest.approx(-0.5)
    assert apply_terminal_stop_penalty(0.4, result) == 0.0


def test_negative_reward_floor_preserves_penalty_separation():
    result = compute_terminal_stop_penalty([[0, 0], [1.6, 0], [1.0, 0]], [3, 0])
    assert apply_terminal_stop_penalty(
        0.2,
        result,
        min_reward=-1.0,
        max_reward=1.0,
    ) == pytest.approx(-0.3)


def test_reentering_after_leaving_is_still_penalized():
    result = compute_terminal_stop_penalty([[0, 0], [1.6, 0], [1.0, 0], [1.7, 0]], [3, 0])
    assert result["left_goal"] is True
    assert result["terminal_stop_penalty"] == pytest.approx(-0.5)


def test_not_reaching_goal_is_left_to_navigation_reward():
    result = compute_terminal_stop_penalty([[0, 0], [0.1, 0]], [3, 0])
    assert result["reached_goal"] is False
    assert result["terminal_stop_penalty"] == 0.0


def test_final_sample_reach_is_not_false_positive():
    result = compute_terminal_stop_penalty([[0, 0], [1.6, 0]], [3, 0])
    assert result["terminal_stop_reason"] == "no_post_reach_observation"
    assert result["failed_to_stop"] is False
