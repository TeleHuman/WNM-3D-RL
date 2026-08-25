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
import torch

from verl_omni.utils.reward_score.wam_vision_reward import VisionRewardConfig, compute_vision_reward


def _video(frames=5):
    torch.manual_seed(7)
    return torch.rand(frames, 3, 32, 48)


def test_identical_video_scores_one_and_excludes_prefix():
    video = _video()
    result = compute_vision_reward(video, video.clone())
    assert result["vision_reward"] == pytest.approx(1.0, abs=2e-3)
    assert result["degenerate"] is False


def test_non_finite_and_out_of_range_are_zeroed():
    gt = _video()
    invalid = gt.clone()
    invalid[2, 0, 0, 0] = float("nan")
    assert compute_vision_reward(invalid, gt)["degenerate_reason"] == "non_finite"

    invalid = gt + 2.0
    assert compute_vision_reward(invalid, gt)["degenerate_reason"] == "out_of_range"


def test_constant_video_is_zeroed():
    gt = _video()
    result = compute_vision_reward(torch.full_like(gt, 0.5), gt)
    assert result["vision_reward"] == 0.0
    assert result["degenerate_reason"] == "constant_video"


def test_frozen_prediction_is_penalized_when_gt_moves():
    gt = torch.stack([torch.full((3, 32, 48), i / 4) for i in range(5)])
    pred = gt.clone()
    pred[1:] = pred[1]
    cfg = VisionRewardConfig(min_spatial_std=0.0, frozen_penalty=0.1)
    result = compute_vision_reward(pred, gt, cfg)
    assert result["is_frozen"] is True
    assert "frozen_video" in result["degenerate_reason"]
    assert result["vision_reward"] <= 0.1


def test_copying_prefix_frame_is_detected():
    torch.manual_seed(8)
    base = torch.rand(3, 32, 48) * 0.4
    gt = torch.stack([(base + 0.08 * index).clamp(0, 1) for index in range(5)])
    pred = torch.stack([base for _ in range(5)])
    cfg = VisionRewardConfig(min_spatial_std=0.0)
    result = compute_vision_reward(pred, gt, cfg)
    assert result["is_prefix_copy"] is True
    assert "prefix_copy" in result["degenerate_reason"]
    assert result["degeneration_factor"] == pytest.approx(cfg.frozen_penalty)


def test_extreme_exposure_is_detected_without_being_constant():
    gt = _video()
    pred = torch.rand_like(gt) * 0.01 + 0.985
    cfg = VisionRewardConfig(min_spatial_std=0.0)
    result = compute_vision_reward(pred, gt, cfg)
    assert result["is_exposure_degenerate"] is True
    assert "exposure" in result["degenerate_reason"]


def test_global_luminance_flicker_is_detected():
    torch.manual_seed(9)
    texture = torch.rand(3, 32, 48) * 0.1
    gt = torch.stack([texture + 0.45 + 0.005 * index for index in range(5)])
    pred = torch.stack([texture + value for value in (0.45, 0.1, 0.8, 0.1, 0.8)])
    result = compute_vision_reward(pred, gt, VisionRewardConfig(min_spatial_std=0.0))
    assert result["is_flickering"] is True
    assert "flicker" in result["degenerate_reason"]


def test_cthw_layout_and_spatial_resize_are_supported():
    video = _video()
    prediction = video.permute(1, 0, 2, 3)
    gt = torch.nn.functional.interpolate(video, (40, 64), mode="bilinear", align_corners=False)
    result = compute_vision_reward(prediction, gt)
    assert 0.0 <= result["vision_reward"] <= 1.0
