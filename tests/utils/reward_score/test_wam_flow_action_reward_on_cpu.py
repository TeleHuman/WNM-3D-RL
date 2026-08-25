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
import torch

import verl_omni.utils.reward_score.wam_flow_action_reward as flow_reward
from verl_omni.utils.reward_score.wam_flow_action_reward import (
    FlowActionCalibration,
    FlowActionRewardConfig,
    compute_flow_action_consistency,
    extract_video_flow_descriptor,
    flow_descriptor_size,
)


def _constant_motion_calibration(motion):
    feature_dim = flow_descriptor_size((4, 6))
    return FlowActionCalibration(
        horizon=8,
        resize_hw=(64, 128),
        grid_hw=(4, 6),
        l2_normalize_features=True,
        feature_mean=np.zeros(feature_dim),
        feature_scale=np.ones(feature_dim),
        output_mean=np.asarray(motion, dtype=np.float64),
        output_scale=np.ones(3),
        weights=np.zeros((feature_dim + 1, 3)),
        stationary_flow_p90_px=1.0,
        validation={},
    )


def test_flow_action_score_prefers_action_aligned_with_inferred_motion(monkeypatch):
    descriptor = np.zeros(flow_descriptor_size((4, 6)), dtype=np.float64)
    descriptor[-1] = 1.0
    monkeypatch.setattr(
        flow_reward,
        "extract_video_flow_descriptor",
        lambda *args, **kwargs: (
            descriptor,
            {
                "flow_action_flow_confidence": 1.0,
                "flow_action_fb_error_px": 0.0,
                "flow_action_texture_gradient": 20.0,
                "flow_action_flow_p50_px": 1.0,
                "flow_action_flow_p75_px": 1.0,
                "flow_action_flow_p90_px": 1.0,
            },
        ),
    )
    video = torch.rand(9, 3, 16, 24)
    aligned = torch.zeros(8, 32)
    aligned[:, 0] = 0.1
    opposite = aligned.clone()
    opposite[:, 0] = -0.1
    calibration = _constant_motion_calibration([0.8, 0.0, 0.0])

    aligned_score = compute_flow_action_consistency(
        video,
        aligned,
        [-1, -1, -1],
        [1, 1, 1],
        calibration,
        nav_action_scale=1.0,
    )
    opposite_score = compute_flow_action_consistency(
        video,
        opposite,
        [-1, -1, -1],
        [1, 1, 1],
        calibration,
        nav_action_scale=1.0,
    )

    assert aligned_score["flow_action_translation_cosine"] == pytest.approx(1.0)
    assert opposite_score["flow_action_translation_cosine"] == pytest.approx(-1.0)
    assert aligned_score["flow_action_score"] == pytest.approx(1.0)
    assert opposite_score["flow_action_score"] == 0.0


def test_low_flow_confidence_gates_consistency_reward(monkeypatch):
    descriptor = np.zeros(flow_descriptor_size((4, 6)), dtype=np.float64)
    monkeypatch.setattr(
        flow_reward,
        "extract_video_flow_descriptor",
        lambda *args, **kwargs: (
            descriptor,
            {
                "flow_action_flow_confidence": 0.05,
                "flow_action_fb_error_px": 10.0,
                "flow_action_texture_gradient": 0.1,
                "flow_action_flow_p50_px": 0.0,
                "flow_action_flow_p75_px": 0.0,
                "flow_action_flow_p90_px": 0.0,
            },
        ),
    )
    actions = torch.zeros(8, 32)
    actions[:, 0] = 0.1
    result = compute_flow_action_consistency(
        torch.rand(9, 3, 16, 24),
        actions,
        [-1, -1, -1],
        [1, 1, 1],
        _constant_motion_calibration([0.8, 0.0, 0.0]),
        nav_action_scale=1.0,
    )
    assert result["flow_action_valid"] is False
    assert result["flow_action_score"] == 0.0


def test_dis_flow_descriptor_detects_synthetic_horizontal_motion():
    pytest.importorskip("cv2")
    generator = torch.Generator().manual_seed(11)
    base = torch.rand(3, 64, 128, generator=generator)
    video = torch.stack([torch.roll(base, shifts=frame, dims=-1) for frame in range(9)])
    descriptor, diagnostics = extract_video_flow_descriptor(
        video,
        config=FlowActionRewardConfig(horizon=8),
    )
    # Every grid cell stores median [u,v]. Image content shifted right.
    grid_u = descriptor[:-3].reshape(-1, 2)[:, 0]
    assert float(np.median(grid_u)) > 0.5
    assert diagnostics["flow_action_flow_confidence"] > 0.2
    assert diagnostics["flow_action_flow_p90_px"] > 0.5
