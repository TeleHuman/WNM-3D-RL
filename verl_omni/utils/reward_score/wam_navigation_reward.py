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

"""Navigation trajectory, nDTW, and occupancy rewards for InteriorGS WAMs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

CAM_TO_NAV = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class NavigationRewardConfig:
    nav_action_scale: float = 4.0
    ndtw_success_distance: float = 0.50
    occupancy_threshold: int = 200
    occupancy_margin_px: int = 4


def inverse_q99(
    normalized: torch.Tensor | np.ndarray | Sequence,
    q01: Sequence[float],
    q99: Sequence[float],
) -> np.ndarray:
    value = np.asarray(_to_numpy(normalized), dtype=np.float32)
    low = np.asarray(q01, dtype=np.float32)
    high = np.asarray(q99, dtype=np.float32)
    if value.shape[-1] != low.size or low.shape != high.shape:
        raise ValueError(f"q99 shape mismatch: value={value.shape}, q01={low.shape}, q99={high.shape}")
    return ((value + 1.0) * 0.5 * (high - low) + low).astype(np.float32)


def rollout_actions_to_world_xy(
    actions: torch.Tensor | np.ndarray | Sequence,
    start_extrinsic: torch.Tensor | np.ndarray | Sequence,
    q01: Sequence[float],
    q99: Sequence[float],
    *,
    nav_action_scale: float = 4.0,
) -> np.ndarray:
    """Decode ``[H,D]`` normalized actions into ``[H+1,2]`` world positions."""
    physical = rollout_actions_to_local_deltas(
        actions,
        q01,
        q99,
        nav_action_scale=nav_action_scale,
    )
    local_xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), np.cumsum(physical[:, :2], axis=0)],
        axis=0,
    )

    extrinsic = np.asarray(_to_numpy(start_extrinsic), dtype=np.float32).reshape(4, 4)
    if not np.isfinite(extrinsic).all():
        raise ValueError("start_extrinsic contains non-finite values")
    rotation_world_agent = extrinsic[:3, :3] @ CAM_TO_NAV.T
    local_xyz = np.pad(local_xy, ((0, 0), (0, 1)))
    world_xyz = (rotation_world_agent @ local_xyz.T).T + extrinsic[:3, 3]
    return world_xyz[:, :2].astype(np.float32)


def rollout_actions_to_local_deltas(
    actions: torch.Tensor | np.ndarray | Sequence,
    q01: Sequence[float],
    q99: Sequence[float],
    *,
    nav_action_scale: float = 4.0,
) -> np.ndarray:
    """Decode normalized rollout actions into local ``[dx, dy, dyaw]``."""
    action_array = np.asarray(_to_numpy(actions), dtype=np.float32)
    if action_array.ndim == 3 and action_array.shape[0] == 1:
        action_array = action_array[0]
    if action_array.ndim != 2 or action_array.shape[1] < 3:
        raise ValueError(f"actions must have shape [H, D>=3], got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise ValueError("actions contain non-finite values")
    if nav_action_scale <= 0:
        raise ValueError("nav_action_scale must be positive")

    # WNM emits a padded [32,32] action latent. Only nav [dx,dy,dyaw]
    # is normalized. Keep all three physical channels available to cross-modal
    # rewards even though navigation collision/progress consumes XY only.
    return (inverse_q99(action_array[:, :3], q01, q99) / float(nav_action_scale)).astype(np.float32)


def normalized_dtw(
    prediction_world_xy: torch.Tensor | np.ndarray | Sequence,
    reference_world_xy: torch.Tensor | np.ndarray | Sequence,
    *,
    success_distance: float = 0.50,
) -> float:
    prediction = _xy_array(prediction_world_xy, "prediction_world_xy")
    reference = _xy_array(reference_world_xy, "reference_world_xy")
    if success_distance <= 0:
        raise ValueError("success_distance must be positive")
    previous = np.full(len(reference) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for pred_point in prediction:
        current = np.full(len(reference) + 1, np.inf, dtype=np.float64)
        for j, ref_point in enumerate(reference, start=1):
            distance = float(np.linalg.norm(pred_point - ref_point))
            current[j] = distance + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    dtw_distance = float(previous[-1])
    return float(math.exp(-dtw_distance / (len(reference) * float(success_distance))))


def trajectory_path_length(
    trajectory_world_xy: torch.Tensor | np.ndarray | Sequence,
) -> float:
    """Return the world-space arc length of an XY trajectory in metres."""
    trajectory = _xy_array(trajectory_world_xy, "trajectory_world_xy")
    if len(trajectory) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())


def occupancy_geodesic_distances_to_goal(
    positions_world_xy: torch.Tensor | np.ndarray | Sequence,
    goal_world_xy: torch.Tensor | np.ndarray | Sequence,
    scene_dir: str | Path,
    *,
    occupancy_threshold: int = 200,
    occupancy_margin_px: int = 0,
    snap_radius_px: int = 4,
) -> dict[str, np.ndarray | bool]:
    """Measure free-space distances from positions to a goal on the OCC grid.

    Distances use an 8-connected minimum-cost path with the physical X/Y pixel
    resolution as edge sampling. A query landing slightly inside an obstacle
    can be snapped to the nearest passable pixel so collision remains a soft,
    independently weighted signal instead of implicitly hard-zeroing progress.
    Out-of-bounds and disconnected queries retain an infinite distance.
    """
    positions = _xy_array(positions_world_xy, "positions_world_xy")
    goal = np.asarray(_to_numpy(goal_world_xy), dtype=np.float32).reshape(-1)
    if goal.size < 2 or not np.isfinite(goal[:2]).all():
        raise ValueError("goal_world_xy must contain at least two finite values")
    if snap_radius_px < 0:
        raise ValueError("snap_radius_px must be non-negative")

    root = str(Path(scene_dir).resolve())
    grid, metadata = _load_safe_occupancy(root, int(occupancy_threshold), int(occupancy_margin_px))
    height, width = grid.shape
    lower = metadata.get("lower", metadata.get("min"))
    upper = metadata.get("upper", metadata.get("max"))
    if lower is None or upper is None:
        raise KeyError("occupancy.json must contain lower/upper or min/max")
    resolution_x = (float(upper[0]) - float(lower[0])) / float(width)
    resolution_y = (float(upper[1]) - float(lower[1])) / float(height)
    if resolution_x <= 0 or resolution_y <= 0:
        raise ValueError("occupancy world bounds must have positive span")

    goal_index, goal_snapped = _passable_grid_index(
        goal[:2],
        grid,
        metadata,
        snap_radius_px=snap_radius_px,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
    )
    distances = np.full(len(positions), np.inf, dtype=np.float64)
    snapped = np.zeros(len(positions), dtype=bool)
    if goal_index is None:
        return {
            "distances": distances,
            "snapped": snapped,
            "goal_snapped": False,
            "goal_valid": False,
        }

    query_indices: list[tuple[int, int] | None] = []
    for index, position in enumerate(positions):
        query_index, query_snapped = _passable_grid_index(
            position,
            grid,
            metadata,
            snap_radius_px=snap_radius_px,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
        )
        query_indices.append(query_index)
        snapped[index] = query_snapped

    ends = sorted({index for index in query_indices if index is not None})
    if ends:
        # scikit-image is already a runtime dependency of the vision reward.
        # MCP_Geometric performs the Dijkstra traversal in compiled code and
        # respects anisotropic world-space pixel resolution.
        from skimage.graph import MCP_Geometric

        costs = np.where(grid, 1.0, np.inf).astype(np.float32, copy=False)
        solver = MCP_Geometric(
            costs,
            fully_connected=True,
            sampling=(resolution_y, resolution_x),
        )
        cumulative_costs, _ = solver.find_costs(
            [goal_index],
            ends=ends,
            find_all_ends=True,
        )
        for output_index, query_index in enumerate(query_indices):
            if query_index is not None:
                distances[output_index] = float(cumulative_costs[query_index])

    return {
        "distances": distances,
        "snapped": snapped,
        "goal_snapped": bool(goal_snapped),
        "goal_valid": True,
    }


def trajectory_collides(
    trajectory_world_xy: torch.Tensor | np.ndarray | Sequence,
    scene_dir: str | Path,
    *,
    occupancy_threshold: int = 200,
    occupancy_margin_px: int = 4,
) -> dict[str, float | int | bool]:
    result = freeze_trajectory_at_first_collision(
        trajectory_world_xy,
        scene_dir,
        occupancy_threshold=occupancy_threshold,
        occupancy_margin_px=occupancy_margin_px,
    )
    return {
        "collided": bool(result["collided"]),
        "collision_step": int(result["collision_step"]),
        "out_of_bounds": bool(result["out_of_bounds"]),
    }


def trajectory_clearance_risk(
    trajectory_world_xy: torch.Tensor | np.ndarray | Sequence,
    scene_dir: str | Path,
    *,
    occupancy_threshold: int = 200,
    hard_margin_px: float = 2.0,
    soft_margin_px: float = 4.0,
) -> dict[str, float | int | bool]:
    """Return a continuous near-obstacle risk for a planned XY trajectory.

    The hard collision check remains an independently configured binary event.
    This helper measures the minimum centre-line clearance on the *raw* OCC map
    and maps the warning band ``(hard_margin_px, soft_margin_px)`` to a smooth
    quadratic penalty in ``(0, 1)``.  The first trajectory point is excluded:
    it is fixed by the environment, so a policy that starts close to a wall is
    rewarded for moving away instead of receiving an unavoidable state cost.
    """
    trajectory = _xy_array(trajectory_world_xy, "trajectory_world_xy")
    if not math.isfinite(hard_margin_px) or hard_margin_px < 0:
        raise ValueError("hard_margin_px must be finite and non-negative")
    if not math.isfinite(soft_margin_px) or soft_margin_px <= hard_margin_px:
        raise ValueError("soft_margin_px must be finite and greater than hard_margin_px")

    root = str(Path(scene_dir).resolve())
    clearance, metadata = _load_occupancy_clearance_px(root, int(occupancy_threshold))
    height, width = clearance.shape
    minimum = math.inf
    sampled_clearances: list[float] = []
    evaluated_pixels = 0
    out_of_bounds = False

    start_px, start_py = _world_to_pixel_continuous(trajectory[0], metadata, width, height)
    if not (0.0 <= start_px <= width - 1 and 0.0 <= start_py <= height - 1):
        return {
            "risk": 1.0,
            "max_risk": 1.0,
            "mean_risk": 1.0,
            "min_clearance_px": 0.0,
            "evaluated_pixels": 0,
            "out_of_bounds": True,
        }
    start = (int(round(start_px)), int(round(start_py)))
    skip_fixed_start = True
    for point in trajectory[1:]:
        end_px, end_py = _world_to_pixel_continuous(point, metadata, width, height)
        if not (0.0 <= end_px <= width - 1 and 0.0 <= end_py <= height - 1):
            out_of_bounds = True
            minimum = 0.0
            break
        end = (int(round(end_px)), int(round(end_py)))
        for x, y in _bresenham(start, end):
            if skip_fixed_start:
                skip_fixed_start = False
                continue
            if not (0 <= x < width and 0 <= y < height):
                out_of_bounds = True
                minimum = 0.0
                break
            sampled_clearance = float(clearance[y, x])
            minimum = min(minimum, sampled_clearance)
            sampled_clearances.append(sampled_clearance)
            evaluated_pixels += 1
        if out_of_bounds:
            break
        start = end

    if out_of_bounds:
        risk = 1.0
        mean_risk = 1.0
        minimum = 0.0
    elif evaluated_pixels == 0:
        # No raster cell was entered (for example an exactly stationary plan).
        # There is no action-induced clearance risk to assign.
        risk = 0.0
        mean_risk = 0.0
        minimum = -1.0
    else:
        normalized_profile = np.clip(
            (float(soft_margin_px) - np.asarray(sampled_clearances)) / (float(soft_margin_px) - float(hard_margin_px)),
            0.0,
            1.0,
        )
        squared_profile = normalized_profile * normalized_profile
        risk = float(np.max(squared_profile))
        mean_risk = float(np.mean(squared_profile))
    return {
        "risk": float(risk),
        # ``risk`` remains the historical max/min-clearance statistic. The
        # trajectory-average barrier additionally makes safety
        # credit is dense instead of being controlled by one raster pixel.
        "max_risk": float(risk),
        "mean_risk": float(mean_risk),
        "min_clearance_px": float(minimum),
        "evaluated_pixels": int(evaluated_pixels),
        "out_of_bounds": bool(out_of_bounds),
    }


def freeze_trajectory_at_first_collision(
    trajectory_world_xy: torch.Tensor | np.ndarray | Sequence,
    scene_dir: str | Path,
    *,
    occupancy_threshold: int = 200,
    occupancy_margin_px: int = 4,
) -> dict[str, np.ndarray | int | bool]:
    """Freeze a planned trajectory at its last legal waypoint after collision.

    The returned trajectory has the same shape as the input. If action ``i``
    would cross an inflated obstacle or leave the mapped world extent,
    positions ``i`` and later remain at position ``i-1``. This discrete
    execution model prevents a planned endpoint on the far side of a wall from
    receiving geodesic progress.
    """
    trajectory = _xy_array(trajectory_world_xy, "trajectory_world_xy")
    grid, metadata = _load_safe_occupancy(
        str(Path(scene_dir).resolve()), int(occupancy_threshold), int(occupancy_margin_px)
    )
    height, width = grid.shape
    resolved = trajectory.copy()
    start_px, start_py = _world_to_pixel_continuous(trajectory[0], metadata, width, height)
    if not (0.0 <= start_px <= width - 1 and 0.0 <= start_py <= height - 1):
        resolved[1:] = resolved[0]
        return {
            "trajectory_world_xy": resolved,
            "collided": True,
            "collision_step": 0,
            "out_of_bounds": True,
        }
    start = (int(round(start_px)), int(round(start_py)))
    if not bool(grid[start[1], start[0]]):
        resolved[1:] = resolved[0]
        return {
            "trajectory_world_xy": resolved,
            "collided": True,
            "collision_step": 0,
            "out_of_bounds": False,
        }

    for segment_index, point in enumerate(trajectory[1:]):
        end_px, end_py = _world_to_pixel_continuous(point, metadata, width, height)
        collision_step = segment_index + 1
        # GN0 clamps after rounding. Reward must fail before clamping so leaving
        # the mapped world extent cannot become a false free-space point.
        if not (0.0 <= end_px <= width - 1 and 0.0 <= end_py <= height - 1):
            resolved[collision_step:] = resolved[collision_step - 1]
            return {
                "trajectory_world_xy": resolved,
                "collided": True,
                "collision_step": collision_step,
                "out_of_bounds": True,
            }
        end = (int(round(end_px)), int(round(end_py)))
        for x, y in _bresenham(start, end):
            if not (0 <= x < width and 0 <= y < height) or not bool(grid[y, x]):
                resolved[collision_step:] = resolved[collision_step - 1]
                return {
                    "trajectory_world_xy": resolved,
                    "collided": True,
                    "collision_step": collision_step,
                    "out_of_bounds": not (0 <= x < width and 0 <= y < height),
                }
        start = end
    return {
        "trajectory_world_xy": resolved,
        "collided": False,
        "collision_step": -1,
        "out_of_bounds": False,
    }


def _passable_grid_index(
    world_xy: np.ndarray,
    grid: np.ndarray,
    metadata: dict,
    *,
    snap_radius_px: int,
    resolution_x: float,
    resolution_y: float,
) -> tuple[tuple[int, int] | None, bool]:
    height, width = grid.shape
    px, py = _world_to_pixel_continuous(world_xy, metadata, width, height)
    if not (0.0 <= px <= width - 1 and 0.0 <= py <= height - 1):
        return None, False
    x, y = int(round(px)), int(round(py))
    if bool(grid[y, x]):
        return (y, x), False
    if snap_radius_px == 0:
        return None, False

    best: tuple[float, int, int] | None = None
    y0, y1 = max(0, y - snap_radius_px), min(height - 1, y + snap_radius_px)
    x0, x1 = max(0, x - snap_radius_px), min(width - 1, x + snap_radius_px)
    for candidate_y in range(y0, y1 + 1):
        for candidate_x in range(x0, x1 + 1):
            if not bool(grid[candidate_y, candidate_x]):
                continue
            dx = (candidate_x - x) * resolution_x
            dy = (candidate_y - y) * resolution_y
            distance_sq = dx * dx + dy * dy
            if best is None or distance_sq < best[0]:
                best = (distance_sq, candidate_y, candidate_x)
    if best is None:
        return None, False
    return (best[1], best[2]), True


@lru_cache(maxsize=64)
def _load_safe_occupancy(scene_dir: str, occupancy_threshold: int, occupancy_margin_px: int) -> tuple[np.ndarray, dict]:
    root = Path(scene_dir)
    with (root / "occupancy.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    occupancy = np.asarray(Image.open(root / "occupancy.png").convert("L"))
    passable = occupancy >= occupancy_threshold
    if occupancy_margin_px > 0:
        obstacle = ~passable
        inflated = obstacle.copy()
        radius_sq = occupancy_margin_px * occupancy_margin_px
        for dy in range(-occupancy_margin_px, occupancy_margin_px + 1):
            for dx in range(-occupancy_margin_px, occupancy_margin_px + 1):
                if dx * dx + dy * dy > radius_sq:
                    continue
                src_y0, src_y1 = max(0, -dy), min(obstacle.shape[0], obstacle.shape[0] - dy)
                src_x0, src_x1 = max(0, -dx), min(obstacle.shape[1], obstacle.shape[1] - dx)
                dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
                dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
                inflated[dst_y0:dst_y1, dst_x0:dst_x1] |= obstacle[src_y0:src_y1, src_x0:src_x1]
        passable = ~inflated
    return passable, metadata


@lru_cache(maxsize=128)
def _load_occupancy_clearance_px(scene_dir: str, occupancy_threshold: int) -> tuple[np.ndarray, dict]:
    """Cache raw-map Euclidean obstacle clearance in OCC pixels."""
    from scipy.ndimage import distance_transform_edt

    passable, metadata = _load_safe_occupancy(scene_dir, int(occupancy_threshold), 0)
    clearance = distance_transform_edt(passable).astype(np.float32, copy=False)
    return clearance, metadata


def _world_to_pixel_continuous(world_xy: np.ndarray, metadata: dict, width: int, height: int) -> tuple[float, float]:
    lower = metadata.get("lower", metadata.get("min"))
    upper = metadata.get("upper", metadata.get("max"))
    if lower is None or upper is None:
        raise KeyError("occupancy.json must contain lower/upper or min/max")
    span_x = float(upper[0]) - float(lower[0])
    span_y = float(upper[1]) - float(lower[1])
    if span_x <= 0 or span_y <= 0:
        raise ValueError("occupancy world bounds must have positive span")
    px = (-float(world_xy[0]) - float(lower[0])) / (span_x / width) - 0.5
    py = (float(upper[1]) + float(world_xy[1])) / (span_y / height) - 0.5
    return px, py


def _bresenham(start: tuple[int, int], end: tuple[int, int]):
    x0, y0 = start
    x1, y1 = end
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _xy_array(value, name: str) -> np.ndarray:
    array = np.asarray(_to_numpy(value), dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [T, >=2], got {array.shape}")
    array = array[:, :2]
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value
