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

import numpy as np
import pytest

from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_wam_exploration_metrics,
)


def test_wam_exploration_metrics_separate_reward_and_behavior_diversity():
    uids = np.array(["a", "a", "a", "b", "b", "b"])
    endpoints = np.array([0, 1, 2, 4, 4, 4], dtype=np.float64)
    trajectories = np.zeros((6, 2, 2), dtype=np.float64)
    trajectories[:, -1, 0] = endpoints
    visual = np.zeros((6, 1, 1, 1), dtype=np.float64)
    visual[:, 0, 0, 0] = endpoints
    extras = {
        "action_reward": np.array([0, 1, 2, 5, 5, 5], dtype=np.float64),
        "visual_reward": np.array([0.2, 0.4, 0.6, 0.1, 0.1, 0.1], dtype=np.float64),
        "action_pred_path_length": endpoints,
        "action_collision": np.array([0, 1, 0, 1, 1, 1], dtype=np.float64),
        "stop_failed": np.array([0, 0, 1, 0, 0, 0], dtype=np.float64),
        "stop_left_goal": np.zeros(6, dtype=np.float64),
        "_exploration_action_world_xy": trajectories,
        "_exploration_visual_luma": visual,
    }

    metrics = compute_wam_exploration_metrics(extras, uids)

    assert metrics["exploration/action_reward/zero_std_ratio"] == pytest.approx(0.5)
    assert metrics["exploration/action_reward/best_of_n_minus_mean"] == pytest.approx(0.5)
    assert metrics["exploration/action/group_pairwise_ade_m"] == pytest.approx(1.0 / 3.0)
    assert metrics["exploration/action/group_endpoint_std_m"] == pytest.approx(np.sqrt(2.0 / 3.0) / 2.0)
    assert metrics["exploration/action/group_path_length_std_m"] == pytest.approx(np.sqrt(2.0 / 3.0) / 2.0)
    assert metrics["exploration/action/group_collision_disagreement"] == pytest.approx(1.0 / 3.0)
    assert metrics["exploration/action/collision_mix_ratio"] == pytest.approx(0.5)
    assert metrics["exploration/visual/group_pairwise_luma_rms"] == pytest.approx(2.0 / 3.0)


def test_collision_mix_ratio_uses_credit_groups_when_provided():
    uids = np.array(["p"] * 8)
    credit_groups = np.array(["p/s0", "p/s0", "p/s1", "p/s1", "p/s2", "p/s2", "p/s3", "p/s3"])
    extras = {"action_any_collision": np.array([0, 1, 1, 1, 0, 0, 0, 1], dtype=np.float64)}

    metrics = compute_wam_exploration_metrics(
        extras,
        uids,
        collision_group_ids=credit_groups,
    )

    assert metrics["exploration/action/collision_mix_ratio"] == pytest.approx(0.5)
