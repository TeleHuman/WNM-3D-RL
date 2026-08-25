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

"""Lightweight full-reference vision reward for WNM rollouts.

The public entry point is :func:`compute_vision_reward`.  Both videos must be
RGB tensors in ``[0, 1]`` and may use ``TCHW`` or ``CTHW`` layout.  The first
frame is excluded by default because WNM conditions on one target-prefix
frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisionRewardConfig:
    ms_ssim_weight: float = 0.60
    low_res_charbonnier_weight: float = 0.25
    temporal_gradient_weight: float = 0.15
    exclude_prefix_frames: int = 1
    low_res_size: tuple[int, int] = (64, 128)
    charbonnier_epsilon: float = 1e-3
    max_out_of_range_ratio: float = 0.01
    min_spatial_std: float = 0.01
    min_gt_motion: float = 0.01
    min_pred_motion: float = 0.002
    min_motion_ratio: float = 0.10
    frozen_penalty: float = 0.10
    min_contrast_std: float = 0.03
    min_mean_luma: float = 0.03
    max_mean_luma: float = 0.97
    saturation_low: float = 0.02
    saturation_high: float = 0.98
    max_saturated_ratio: float = 0.85
    min_gt_prefix_change: float = 0.02
    max_prefix_copy_error: float = 0.01
    max_luma_jump: float = 0.08
    max_luma_jump_ratio: float = 3.0
    max_pred_motion: float = 0.12
    max_motion_ratio: float = 3.0
    artifact_penalty: float = 0.25


def _as_tchw(video: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(video).__name__}")
    video = video.detach().float()
    while video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"{name} must have TCHW or CTHW layout, got {tuple(video.shape)}")
    if video.shape[1] in (1, 3, 4):
        pass
    elif video.shape[0] in (1, 3, 4):
        video = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"cannot infer channel dimension for {name} with shape {tuple(video.shape)}")
    if video.shape[1] == 4:
        video = video[:, :3]
    if video.shape[1] == 1:
        video = video.expand(-1, 3, -1, -1)
    return video


def _resize(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(video, size=size, mode="bilinear", align_corners=False, antialias=True)


def _ssim_per_frame(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Local-window SSIM. Reflection padding avoids rewarding artificial black borders.
    kernel = min(11, x.shape[-2], x.shape[-1])
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 3:
        return 1.0 - (x - y).abs().flatten(1).mean(1).clamp(0.0, 1.0)
    padding = kernel // 2
    mu_x = F.avg_pool2d(x, kernel, stride=1, padding=padding, count_include_pad=False)
    mu_y = F.avg_pool2d(y, kernel, stride=1, padding=padding, count_include_pad=False)
    sigma_x = F.avg_pool2d(x * x, kernel, 1, padding, count_include_pad=False) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, kernel, 1, padding, count_include_pad=False) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, kernel, 1, padding, count_include_pad=False) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).flatten(1).mean(1).clamp(0.0, 1.0)


def _multi_scale_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scores = []
    for _ in range(4):
        scores.append(_ssim_per_frame(x, y))
        if min(x.shape[-2:]) < 16:
            break
        x = F.avg_pool2d(x, 2, 2)
        y = F.avg_pool2d(y, 2, 2)
    # Equal-scale averaging is deliberately bounded and more robust than a
    # product when generated frames contain local artifacts.
    return torch.stack(scores).mean()


def _charbonnier_error(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt((x - y).square() + epsilon**2).mean()


def _temporal_gradient_error(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_zeros(())
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return torch.sqrt((dx - dy).square() + epsilon**2).mean()


def compute_vision_behavior_signature(
    prediction: torch.Tensor,
    *,
    exclude_prefix_frames: int = 1,
    num_frames: int = 4,
    size: tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """Return a compact luma signature for within-prompt rollout diversity.

    This intentionally measures generated behavior rather than diffusion
    transition entropy.  It reuses already-decoded rollout frames and adds
    only a tiny 4x8x8 payload per sample to the reward diagnostics.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    pred = _as_tchw(prediction, "prediction")
    if exclude_prefix_frames < 0 or exclude_prefix_frames >= pred.shape[0]:
        raise ValueError(f"exclude_prefix_frames must be in [0, {pred.shape[0] - 1}], got {exclude_prefix_frames}")
    pred = pred[exclude_prefix_frames:].clamp(0.0, 1.0)
    indices = (
        torch.linspace(
            0,
            pred.shape[0] - 1,
            steps=num_frames,
            device=pred.device,
        )
        .round()
        .long()
    )
    sampled = _resize(pred.index_select(0, indices), size)
    luma = 0.2126 * sampled[:, 0] + 0.7152 * sampled[:, 1] + 0.0722 * sampled[:, 2]
    if not torch.isfinite(luma).all():
        return torch.full(
            (num_frames, *size),
            float("nan"),
            dtype=torch.float32,
        )
    return luma.detach().to(device="cpu", dtype=torch.float32)


def compute_vision_reward(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    config: VisionRewardConfig | None = None,
) -> dict[str, float | bool]:
    """Return a bounded vision reward and component diagnostics.

    The requested loss form is preserved up to a constant offset::

        0.60 * MS-SSIM - 0.25 * Charbonnier - 0.15 * temporal_error

    Specifically, each error is clipped to ``[0, 1]`` and converted to a
    score with ``1 - error``.  The constant offset does not affect GRPO ranks,
    while keeping the result in ``[0, 1]`` makes degeneration gates safe.
    """
    cfg = config or VisionRewardConfig()
    pred = _as_tchw(prediction, "prediction")
    gt = _as_tchw(ground_truth, "ground_truth").to(pred.device)

    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"video lengths differ: prediction={pred.shape[0]}, ground_truth={gt.shape[0]}")
    prefix_pred = None
    prefix_gt = None
    if cfg.exclude_prefix_frames:
        if pred.shape[0] <= cfg.exclude_prefix_frames:
            raise ValueError("video contains no frames after excluding target-prefix frames")
        prefix_pred = pred[cfg.exclude_prefix_frames - 1 : cfg.exclude_prefix_frames]
        prefix_gt = gt[cfg.exclude_prefix_frames - 1 : cfg.exclude_prefix_frames]
        pred = pred[cfg.exclude_prefix_frames :]
        gt = gt[cfg.exclude_prefix_frames :]
    if pred.shape[-2:] != gt.shape[-2:]:
        gt = _resize(gt, pred.shape[-2:])

    finite = bool(torch.isfinite(pred).all() and torch.isfinite(gt).all())
    if not finite:
        return _zero_result("non_finite")
    out_of_range_ratio = ((pred < 0.0) | (pred > 1.0)).float().mean()
    if float(out_of_range_ratio) > cfg.max_out_of_range_ratio:
        result = _zero_result("out_of_range")
        result["out_of_range_ratio"] = float(out_of_range_ratio)
        return result

    pred = pred.clamp(0.0, 1.0)
    gt = gt.clamp(0.0, 1.0)
    pred_spatial_std = pred.flatten(1).std(dim=1).mean()
    if float(pred_spatial_std) < cfg.min_spatial_std:
        result = _zero_result("constant_video")
        result["pred_spatial_std"] = float(pred_spatial_std)
        return result

    low_pred = _resize(pred, cfg.low_res_size)
    low_gt = _resize(gt, cfg.low_res_size)
    low_prefix_pred = _resize(prefix_pred, cfg.low_res_size) if prefix_pred is not None else None
    low_prefix_gt = _resize(prefix_gt, cfg.low_res_size) if prefix_gt is not None else None
    ms_ssim = _multi_scale_ssim(low_pred, low_gt)
    charbonnier_error = _charbonnier_error(low_pred, low_gt, cfg.charbonnier_epsilon).clamp(0.0, 1.0)
    temporal_error = _temporal_gradient_error(low_pred, low_gt, cfg.charbonnier_epsilon).clamp(0.0, 1.0)
    charbonnier_score = 1.0 - charbonnier_error
    temporal_score = 1.0 - temporal_error

    reward = (
        cfg.ms_ssim_weight * ms_ssim
        + cfg.low_res_charbonnier_weight * charbonnier_score
        + cfg.temporal_gradient_weight * temporal_score
    )

    pred_motion = (low_pred[1:] - low_pred[:-1]).abs().mean() if len(low_pred) > 1 else low_pred.new_zeros(())
    gt_motion = (low_gt[1:] - low_gt[:-1]).abs().mean() if len(low_gt) > 1 else low_gt.new_zeros(())
    pred_mean_luma = low_pred.mean()
    pred_saturated_ratio = ((low_pred <= cfg.saturation_low) | (low_pred >= cfg.saturation_high)).float().mean()
    pred_frame_luma = low_pred.mean(dim=(1, 2, 3))
    gt_frame_luma = low_gt.mean(dim=(1, 2, 3))
    pred_luma_jump = (
        (pred_frame_luma[1:] - pred_frame_luma[:-1]).abs().max() if len(pred_frame_luma) > 1 else low_pred.new_zeros(())
    )
    gt_luma_jump = (
        (gt_frame_luma[1:] - gt_frame_luma[:-1]).abs().max() if len(gt_frame_luma) > 1 else low_gt.new_zeros(())
    )
    prefix_copy_error = low_pred.new_tensor(1.0)
    gt_prefix_change = low_gt.new_zeros(())
    if low_prefix_pred is not None and low_prefix_gt is not None:
        prefix_copy_error = (low_pred - low_prefix_pred).abs().mean()
        gt_prefix_change = (low_gt - low_prefix_gt).abs().mean()

    frozen = bool(
        float(gt_motion) >= cfg.min_gt_motion
        and float(pred_motion) < max(cfg.min_pred_motion, cfg.min_motion_ratio * float(gt_motion))
    )
    prefix_copy = bool(
        float(gt_prefix_change) >= cfg.min_gt_prefix_change and float(prefix_copy_error) <= cfg.max_prefix_copy_error
    )
    low_contrast = float(pred_spatial_std) < cfg.min_contrast_std
    over_or_under_exposed = bool(
        float(pred_mean_luma) < cfg.min_mean_luma
        or float(pred_mean_luma) > cfg.max_mean_luma
        or float(pred_saturated_ratio) > cfg.max_saturated_ratio
    )
    flicker = bool(float(pred_luma_jump) > max(cfg.max_luma_jump, cfg.max_luma_jump_ratio * float(gt_luma_jump)))
    excessive_motion = bool(float(pred_motion) > max(cfg.max_pred_motion, cfg.max_motion_ratio * float(gt_motion)))

    reasons = []
    degeneration_factor = 1.0
    if frozen:
        reasons.append("frozen_video")
        degeneration_factor = min(degeneration_factor, cfg.frozen_penalty)
    if prefix_copy:
        reasons.append("prefix_copy")
        degeneration_factor = min(degeneration_factor, cfg.frozen_penalty)
    if low_contrast:
        reasons.append("low_contrast")
        degeneration_factor = min(degeneration_factor, cfg.artifact_penalty)
    if over_or_under_exposed:
        reasons.append("exposure")
        degeneration_factor = min(degeneration_factor, cfg.artifact_penalty)
    if flicker:
        reasons.append("flicker")
        degeneration_factor = min(degeneration_factor, cfg.artifact_penalty)
    if excessive_motion:
        reasons.append("excessive_motion")
        degeneration_factor = min(degeneration_factor, cfg.artifact_penalty)
    reward = reward * degeneration_factor

    return {
        "vision_reward": float(reward.clamp(0.0, 1.0)),
        "ms_ssim": float(ms_ssim),
        "low_res_charbonnier_error": float(charbonnier_error),
        "low_res_charbonnier_score": float(charbonnier_score),
        "temporal_gradient_error": float(temporal_error),
        "temporal_gradient_score": float(temporal_score),
        "pred_motion": float(pred_motion),
        "gt_motion": float(gt_motion),
        "pred_spatial_std": float(pred_spatial_std),
        "pred_mean_luma": float(pred_mean_luma),
        "pred_saturated_ratio": float(pred_saturated_ratio),
        "pred_luma_jump": float(pred_luma_jump),
        "gt_luma_jump": float(gt_luma_jump),
        "prefix_copy_error": float(prefix_copy_error),
        "gt_prefix_change": float(gt_prefix_change),
        "out_of_range_ratio": float(out_of_range_ratio),
        "is_frozen": frozen,
        "is_prefix_copy": prefix_copy,
        "is_low_contrast": low_contrast,
        "is_exposure_degenerate": over_or_under_exposed,
        "is_flickering": flicker,
        "is_excessive_motion": excessive_motion,
        "degeneration_factor": float(degeneration_factor),
        "degenerate": bool(reasons),
        "degenerate_reason": ",".join(reasons),
    }


def _zero_result(reason: str) -> dict[str, float | bool | str]:
    return {
        "vision_reward": 0.0,
        "ms_ssim": 0.0,
        "low_res_charbonnier_error": 1.0,
        "low_res_charbonnier_score": 0.0,
        "temporal_gradient_error": 1.0,
        "temporal_gradient_score": 0.0,
        "pred_motion": 0.0,
        "gt_motion": 0.0,
        "pred_spatial_std": 0.0,
        "pred_mean_luma": 0.0,
        "pred_saturated_ratio": 0.0,
        "pred_luma_jump": 0.0,
        "gt_luma_jump": 0.0,
        "prefix_copy_error": 0.0,
        "gt_prefix_change": 0.0,
        "out_of_range_ratio": 0.0,
        "is_frozen": reason == "frozen_video",
        "is_prefix_copy": False,
        "is_low_contrast": reason == "constant_video",
        "is_exposure_degenerate": False,
        "is_flickering": False,
        "is_excessive_motion": False,
        "degeneration_factor": 0.0,
        "degenerate": True,
        "degenerate_reason": reason,
    }
