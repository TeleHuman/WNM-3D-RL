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

"""Optical-flow consistency reward for WNM video/action rollouts.

The first 9 generated frames correspond exactly to the first 8 navigation
actions.  Dense flow is pooled into a small spatial descriptor and a ridge
model, fitted only on GT clips, maps that descriptor back to accumulated local
``[dx, dy, dyaw]``.  Reward measures whether this inferred camera motion points
in the same direction as the action emitted by the rollout.

The calibrated score is shared by the visual branch and the first action
chunk. It teaches the generated video to depict the control and, symmetrically,
the action to agree with the depicted camera motion. Low-texture or
forward/backward-inconsistent flow is gated instead of being treated as a
valid stationary prediction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from verl_omni.utils.reward_score.wam_navigation_reward import (
    rollout_actions_to_local_deltas,
)

_CALIBRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FlowActionRewardConfig:
    horizon: int = 8
    resize_hw: tuple[int, int] = (64, 128)
    grid_hw: tuple[int, int] = (4, 6)
    min_flow_confidence: float = 0.20
    min_action_motion_m: float = 0.05
    min_inferred_motion_m: float = 0.01
    yaw_threshold_rad: float = 0.05
    texture_gradient_scale: float = 8.0
    fb_absolute_threshold_px: float = 1.0
    fb_relative_threshold: float = 0.05


@dataclass(frozen=True)
class FlowActionCalibration:
    horizon: int
    resize_hw: tuple[int, int]
    grid_hw: tuple[int, int]
    l2_normalize_features: bool
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    weights: np.ndarray
    stationary_flow_p90_px: float
    validation: dict[str, Any]


def flow_descriptor_size(grid_hw: tuple[int, int]) -> int:
    return int(grid_hw[0]) * int(grid_hw[1]) * 2 + 3


@lru_cache(maxsize=4)
def load_flow_action_calibration(path: str | Path) -> FlowActionCalibration:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", -1)) != _CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported flow/action calibration schema in {resolved}: {payload.get('schema_version')!r}")

    horizon = int(payload["horizon"])
    resize_hw = tuple(int(value) for value in payload["resize_hw"])
    grid_hw = tuple(int(value) for value in payload["grid_hw"])
    feature_dim = flow_descriptor_size(grid_hw)
    feature_mean = _finite_vector(payload["feature_mean"], feature_dim, "feature_mean")
    feature_scale = _finite_vector(payload["feature_scale"], feature_dim, "feature_scale")
    output_mean = _finite_vector(payload["output_mean"], 3, "output_mean")
    output_scale = _finite_vector(payload["output_scale"], 3, "output_scale")
    weights = np.asarray(payload["weights"], dtype=np.float64)
    if weights.shape != (feature_dim + 1, 3) or not np.isfinite(weights).all():
        raise ValueError(f"weights must be finite [{feature_dim + 1},3], got {weights.shape}")
    if np.any(feature_scale <= 0) or np.any(output_scale <= 0):
        raise ValueError("calibration feature/output scales must be positive")
    stationary_flow_p90_px = float(payload["stationary_flow_p90_px"])
    if not math.isfinite(stationary_flow_p90_px) or stationary_flow_p90_px <= 0:
        raise ValueError("stationary_flow_p90_px must be finite and positive")
    if horizon <= 0 or len(resize_hw) != 2 or len(grid_hw) != 2:
        raise ValueError("invalid horizon/resize/grid in flow/action calibration")
    return FlowActionCalibration(
        horizon=horizon,
        resize_hw=resize_hw,
        grid_hw=grid_hw,
        l2_normalize_features=bool(payload.get("l2_normalize_features", True)),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        output_mean=output_mean,
        output_scale=output_scale,
        weights=weights,
        stationary_flow_p90_px=stationary_flow_p90_px,
        validation=dict(payload.get("validation", {})),
    )


def compute_flow_action_consistency(
    video: torch.Tensor,
    actions: torch.Tensor | np.ndarray | Sequence,
    q01: Sequence[float],
    q99: Sequence[float],
    calibration: FlowActionCalibration | str | Path,
    *,
    nav_action_scale: float = 4.0,
    config: FlowActionRewardConfig | None = None,
) -> dict[str, float | bool]:
    """Measure first-chunk generated-video motion versus generated action."""

    model = load_flow_action_calibration(calibration) if isinstance(calibration, str | Path) else calibration
    cfg = config or FlowActionRewardConfig(
        horizon=model.horizon,
        resize_hw=model.resize_hw,
        grid_hw=model.grid_hw,
    )
    if cfg.horizon != model.horizon or cfg.resize_hw != model.resize_hw or cfg.grid_hw != model.grid_hw:
        raise ValueError(
            "flow/action reward config must match calibration: "
            f"config={(cfg.horizon, cfg.resize_hw, cfg.grid_hw)} "
            f"calibration={(model.horizon, model.resize_hw, model.grid_hw)}"
        )

    local_deltas = rollout_actions_to_local_deltas(
        actions,
        q01,
        q99,
        nav_action_scale=nav_action_scale,
    )
    if local_deltas.shape[0] < cfg.horizon:
        raise ValueError(f"flow/action reward needs at least {cfg.horizon} actions, got {local_deltas.shape[0]}")
    action_motion = local_deltas[: cfg.horizon].sum(axis=0, dtype=np.float64)
    descriptor, flow_diagnostics = extract_video_flow_descriptor(video, config=cfg)
    inferred_motion = infer_motion_from_flow_descriptor(descriptor, model)

    action_scaled = action_motion / model.output_scale
    inferred_scaled = inferred_motion / model.output_scale
    action_norm = float(np.linalg.norm(action_scaled))
    inferred_norm = float(np.linalg.norm(inferred_scaled))
    translation_action_norm = float(np.linalg.norm(action_motion[:2]))
    translation_inferred_norm = float(np.linalg.norm(inferred_motion[:2]))

    if translation_action_norm >= cfg.min_action_motion_m and translation_inferred_norm >= cfg.min_inferred_motion_m:
        translation_cosine = float(
            np.clip(
                np.dot(action_motion[:2], inferred_motion[:2]) / (translation_action_norm * translation_inferred_norm),
                -1.0,
                1.0,
            )
        )
        translation_score = max(0.0, translation_cosine)
    elif translation_action_norm < cfg.min_action_motion_m:
        translation_cosine = 0.0
        flow_p90 = float(descriptor[-1])
        stationary_scale = max(model.stationary_flow_p90_px, 1e-6)
        translation_score = float(math.exp(-((flow_p90 / stationary_scale) ** 2)))
    else:
        translation_cosine = 0.0
        translation_score = 0.0

    action_yaw = float(action_motion[2])
    inferred_yaw = float(inferred_motion[2])
    yaw_scale = max(float(model.output_scale[2]), cfg.yaw_threshold_rad)
    if abs(action_yaw) >= cfg.yaw_threshold_rad:
        yaw_score = float(
            0.5 * (1.0 + math.tanh(action_yaw * inferred_yaw / max(cfg.yaw_threshold_rad * yaw_scale, 1e-8)))
        )
    else:
        yaw_score = float(math.exp(-((inferred_yaw / yaw_scale) ** 2)))

    if action_norm > 1e-8 and inferred_norm > 1e-8:
        standardized_cosine = float(
            np.clip(
                np.dot(action_scaled, inferred_scaled) / (action_norm * inferred_norm),
                -1.0,
                1.0,
            )
        )
    else:
        standardized_cosine = 0.0

    confidence = float(flow_diagnostics["flow_action_flow_confidence"])
    valid = bool(
        np.isfinite(descriptor).all() and np.isfinite(inferred_motion).all() and confidence >= cfg.min_flow_confidence
    )
    # Held-out GT calibration validates accumulated XY direction strongly, but
    # dyaw remains ambiguous under monocular flow (lateral camera motion and
    # yaw both induce horizontal image motion). Keep yaw as a diagnostic until
    # it has a separate high-accuracy calibration; do not inject noisy credit.
    raw_score = translation_score
    score = float(np.clip(raw_score * confidence, 0.0, 1.0)) if valid else 0.0
    return {
        "flow_action_score": score,
        "flow_action_raw_score": float(raw_score),
        "flow_action_valid": valid,
        "flow_action_confidence": confidence,
        "flow_action_translation_cosine": translation_cosine,
        "flow_action_translation_score": float(translation_score),
        "flow_action_yaw_score": yaw_score,
        "flow_action_standardized_cosine": standardized_cosine,
        "flow_action_actual_dx": float(action_motion[0]),
        "flow_action_actual_dy": float(action_motion[1]),
        "flow_action_actual_dyaw": action_yaw,
        "flow_action_inferred_dx": float(inferred_motion[0]),
        "flow_action_inferred_dy": float(inferred_motion[1]),
        "flow_action_inferred_dyaw": inferred_yaw,
        **flow_diagnostics,
    }


def infer_motion_from_flow_descriptor(
    descriptor: np.ndarray,
    calibration: FlowActionCalibration,
) -> np.ndarray:
    value = np.asarray(descriptor, dtype=np.float64).reshape(-1)
    if value.shape != calibration.feature_mean.shape or not np.isfinite(value).all():
        raise ValueError(f"flow descriptor must be finite {calibration.feature_mean.shape}, got {value.shape}")
    if calibration.l2_normalize_features:
        value = value / max(float(np.linalg.norm(value)), 1e-8)
    standardized = (value - calibration.feature_mean) / calibration.feature_scale
    design = np.concatenate([standardized, np.ones(1, dtype=np.float64)])
    prediction = (design @ calibration.weights) * calibration.output_scale
    prediction += calibration.output_mean
    return prediction.astype(np.float64)


def extract_video_flow_descriptor(
    video: torch.Tensor,
    *,
    config: FlowActionRewardConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract an averaged first-chunk DIS-flow descriptor and confidence."""

    cfg = config or FlowActionRewardConfig()
    frames = _video_to_gray_u8(video, cfg.resize_hw)
    if frames.shape[0] < cfg.horizon + 1:
        raise ValueError(f"flow/action reward needs at least {cfg.horizon + 1} frames, got {frames.shape[0]}")
    frames = frames[: cfg.horizon + 1]

    import cv2

    # Environment variables are not honored by every OpenCV build. Enforce the
    # reward-worker contract here as well.
    cv2.setNumThreads(1)
    forward_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    backward_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    forward_estimator.setUseSpatialPropagation(True)
    backward_estimator.setUseSpatialPropagation(True)

    descriptors = []
    confidence_values = []
    fb_errors = []
    texture_values = []
    height, width = frames.shape[1:]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    for previous, current in zip(frames, frames[1:], strict=False):
        forward = forward_estimator.calc(previous, current, None)
        backward = backward_estimator.calc(current, previous, None)
        descriptors.append(_flow_grid_descriptor(forward, cfg.grid_hw))

        destination_x = grid_x + forward[..., 0]
        destination_y = grid_y + forward[..., 1]
        in_bounds = (
            (destination_x >= 0) & (destination_x <= width - 1) & (destination_y >= 0) & (destination_y <= height - 1)
        )
        warped_backward = cv2.remap(
            backward,
            destination_x,
            destination_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        magnitude = np.linalg.norm(forward, axis=-1)
        fb_error = np.linalg.norm(forward + warped_backward, axis=-1)
        consistent = fb_error <= (cfg.fb_absolute_threshold_px + cfg.fb_relative_threshold * magnitude)
        fb_confidence = float(np.mean(in_bounds & consistent))
        fb_errors.append(float(np.median(fb_error[in_bounds])) if in_bounds.any() else float("inf"))

        sobel_x = cv2.Sobel(previous, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(previous, cv2.CV_32F, 0, 1, ksize=3)
        texture = float(np.mean(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)))
        texture_confidence = float(np.clip(texture / cfg.texture_gradient_scale, 0.0, 1.0))
        texture_values.append(texture)
        confidence_values.append(fb_confidence * texture_confidence)

    descriptor = np.mean(np.stack(descriptors), axis=0, dtype=np.float64)
    return descriptor.astype(np.float64), {
        "flow_action_flow_confidence": float(np.median(confidence_values)),
        "flow_action_fb_error_px": float(np.median(fb_errors)),
        "flow_action_texture_gradient": float(np.median(texture_values)),
        "flow_action_flow_p50_px": float(descriptor[-3]),
        "flow_action_flow_p75_px": float(descriptor[-2]),
        "flow_action_flow_p90_px": float(descriptor[-1]),
    }


def _flow_grid_descriptor(
    flow: np.ndarray,
    grid_hw: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 2 or not np.isfinite(value).all():
        raise ValueError(f"flow must be finite [H,W,2], got {value.shape}")
    height, width = value.shape[:2]
    rows, columns = grid_hw
    if rows <= 0 or columns <= 0 or height < rows or width < columns:
        raise ValueError(f"invalid flow/grid shapes: {value.shape}/{grid_hw}")
    descriptor = []
    for row in range(rows):
        for column in range(columns):
            cell = value[
                row * height // rows : (row + 1) * height // rows,
                column * width // columns : (column + 1) * width // columns,
            ]
            descriptor.extend(np.median(cell.reshape(-1, 2), axis=0).tolist())
    magnitude = np.linalg.norm(value, axis=-1)
    descriptor.extend(
        [
            float(np.median(magnitude)),
            float(np.percentile(magnitude, 75)),
            float(np.percentile(magnitude, 90)),
        ]
    )
    return np.asarray(descriptor, dtype=np.float64)


def _video_to_gray_u8(
    video: torch.Tensor,
    resize_hw: tuple[int, int],
) -> np.ndarray:
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"video must be a torch.Tensor, got {type(video).__name__}")
    value = video.detach().float()
    while value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"video must have TCHW or CTHW layout, got {tuple(value.shape)}")
    if value.shape[1] in (1, 3, 4):
        pass
    elif value.shape[0] in (1, 3, 4):
        value = value.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"cannot infer channel dimension for video shape {tuple(value.shape)}")
    value = value[:, :3].clamp(0.0, 1.0).to(device="cpu")
    value = F.interpolate(
        value,
        size=resize_hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if value.shape[1] == 1:
        gray = value[:, 0]
    else:
        gray = 0.299 * value[:, 0] + 0.587 * value[:, 1] + 0.114 * value[:, 2]
    return gray.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).numpy()


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite [{size}], got {array.shape}")
    return array
