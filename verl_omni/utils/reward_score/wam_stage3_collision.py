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

"""GN0-aligned collision execution and recovery for the Stage-3 reward."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from verl_omni.utils.reward_score.wam_navigation_reward import (
    _bresenham,
    _load_occupancy_clearance_px,
    _load_safe_occupancy,
    _world_to_pixel_continuous,
)


def _occupancy_pixel_to_world(
    pixel: tuple[int, int],
    metadata: Mapping,
    width: int,
    height: int,
) -> np.ndarray:
    """Invert InteriorGS/GN0's occupancy world-to-pixel transform."""

    lower = metadata.get("lower", metadata.get("min"))
    upper = metadata.get("upper", metadata.get("max"))
    if lower is None or upper is None:
        raise KeyError("occupancy.json must contain lower/upper or min/max")
    span_x = float(upper[0]) - float(lower[0])
    span_y = float(upper[1]) - float(lower[1])
    if span_x <= 0 or span_y <= 0 or width <= 0 or height <= 0:
        raise ValueError("occupancy world bounds and raster dimensions must be positive")
    resolution_x = span_x / float(width)
    resolution_y = span_y / float(height)
    x, y = pixel
    world_x = -(float(lower[0]) + (float(x) + 0.5) * resolution_x)
    world_y = (float(y) + 0.5) * resolution_y - float(upper[1])
    return np.asarray([world_x, world_y], dtype=np.float32)


def execute_gn0_collision_recovery_chunk(
    raw_world_xy: np.ndarray,
    scene_dir: str,
    *,
    action_num: int = 8,
    stop_step: int = -1,
    occupancy_threshold: int = 200,
    execution_margin_px: int = 4,
    recovery_clearance_px: float = 6.0,
    recovery_min_escape_m: float = 0.20,
    recovery_full_escape_m: float = 0.40,
    recovery_tail_free_steps: int = 2,
) -> dict[str, np.ndarray | float | int | bool]:
    """Execute one deployed GN0 chunk with last-free-on-segment clipping.

    Each world-space delta is taken from the original request-frame plan, but
    it is applied from the latest *actual* clipped position.  A collision does
    not terminate the chunk.  Out-of-map endpoints fail closed instead of
    inheriting GN0 simulator's raster-boundary clamp, because otherwise a
    policy could turn leaving the known map into a free recovery manoeuvre.
    """

    raw = np.asarray(raw_world_xy, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2:
        raise ValueError(f"raw_world_xy must have shape [T>=2,2], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("raw_world_xy contains non-finite values")
    horizon = min(int(action_num), len(raw) - 1)
    if horizon <= 0:
        raise ValueError("action_num must select at least one action")
    if stop_step < -1 or stop_step > horizon:
        raise ValueError(f"stop_step must lie in [-1,{horizon}], got {stop_step}")
    if occupancy_threshold < 0 or execution_margin_px < 0:
        raise ValueError("occupancy threshold/margin must be non-negative")
    if recovery_clearance_px <= execution_margin_px:
        raise ValueError("recovery clearance must exceed execution margin")
    if recovery_min_escape_m < 0 or recovery_full_escape_m <= 0:
        raise ValueError("recovery escape thresholds are invalid")
    if recovery_full_escape_m < recovery_min_escape_m:
        raise ValueError("full recovery escape must not be below minimum escape")
    if recovery_tail_free_steps <= 0:
        raise ValueError("recovery_tail_free_steps must be positive")

    grid, metadata = _load_safe_occupancy(str(scene_dir), int(occupancy_threshold), int(execution_margin_px))
    clearance_grid, clearance_metadata = _load_occupancy_clearance_px(str(scene_dir), int(occupancy_threshold))
    height, width = grid.shape
    if clearance_grid.shape != grid.shape:
        raise ValueError("clearance and passability rasters must have identical shapes")

    executed = np.repeat(raw[:1], horizon + 1, axis=0).astype(np.float32)
    collision_mask = np.zeros(horizon, dtype=bool)
    out_of_bounds_mask = np.zeros(horizon, dtype=bool)
    endpoint_clearance = np.full(horizon + 1, -1.0, dtype=np.float64)
    recovery_grace_mask = np.zeros(horizon, dtype=bool)

    def pixel(point: np.ndarray) -> tuple[tuple[int, int] | None, bool]:
        px, py = _world_to_pixel_continuous(point, metadata, width, height)
        inside = 0.0 <= px <= width - 1 and 0.0 <= py <= height - 1
        if not inside:
            return None, False
        return (int(round(px)), int(round(py))), True

    def clearance(point: np.ndarray) -> float:
        px, py = _world_to_pixel_continuous(point, clearance_metadata, width, height)
        if not (0.0 <= px <= width - 1 and 0.0 <= py <= height - 1):
            return 0.0
        return float(clearance_grid[int(round(py)), int(round(px))])

    start_pixel, start_inside = pixel(raw[0])
    start_invalid = bool(not start_inside or start_pixel is None or not bool(grid[start_pixel[1], start_pixel[0]]))
    endpoint_clearance[0] = clearance(raw[0]) if start_inside else 0.0
    executed_action_count = horizon if stop_step < 0 else int(stop_step)
    if start_invalid:
        collision_mask[0] = True
        out_of_bounds_mask[0] = not start_inside
        executed_action_count = 0
    else:
        world_deltas = np.diff(raw[: horizon + 1], axis=0)
        in_recovery = False
        recovery_contact = raw[0].astype(np.float64)
        consecutive_free = 0
        recovery_clearance_history: list[float] = []
        for action_index in range(horizon):
            if action_index >= executed_action_count:
                executed[action_index + 1 :] = executed[action_index]
                endpoint_clearance[action_index + 1 :] = endpoint_clearance[action_index]
                break

            if in_recovery:
                recovery_grace_mask[action_index] = True
            current = executed[action_index]
            proposed = current + world_deltas[action_index]
            start_px, current_inside = pixel(current)
            end_px, end_inside = pixel(proposed)
            collided = False
            out_of_bounds = False
            adjusted = proposed
            if not current_inside or start_px is None:
                collided = True
                out_of_bounds = True
                adjusted = current
            elif not end_inside or end_px is None:
                # Deliberate fail-closed divergence from GN0's clamped pixel
                # conversion. Do not reward a map-boundary excursion.
                collided = True
                out_of_bounds = True
                adjusted = current
            elif start_px != end_px:
                last_free = start_px
                for x, y in _bresenham(start_px, end_px):
                    if not (0 <= x < width and 0 <= y < height):
                        collided = True
                        out_of_bounds = True
                        break
                    if not bool(grid[y, x]):
                        collided = True
                        break
                    last_free = (x, y)
                if collided:
                    adjusted = _occupancy_pixel_to_world(last_free, metadata, width, height)

            executed[action_index + 1] = adjusted
            endpoint_clearance[action_index + 1] = clearance(adjusted)
            collision_mask[action_index] = collided
            out_of_bounds_mask[action_index] = out_of_bounds
            if collided:
                in_recovery = True
                recovery_contact = adjusted.astype(np.float64)
                consecutive_free = 0
                recovery_clearance_history = []
            elif in_recovery:
                consecutive_free += 1
                recovery_clearance_history.append(endpoint_clearance[action_index + 1])
                if len(recovery_clearance_history) > recovery_tail_free_steps:
                    recovery_clearance_history.pop(0)
                escaped = float(np.linalg.norm(adjusted.astype(np.float64) - recovery_contact))
                if (
                    consecutive_free >= recovery_tail_free_steps
                    and len(recovery_clearance_history) == recovery_tail_free_steps
                    and min(recovery_clearance_history) >= recovery_clearance_px
                    and escaped >= recovery_min_escape_m
                ):
                    in_recovery = False

    collision_indices = np.flatnonzero(collision_mask[:executed_action_count])
    collision_count = int(len(collision_indices))
    first_collision_action = int(collision_indices[0]) if collision_count else -1
    last_collision_action = int(collision_indices[-1]) if collision_count else -1
    any_out_of_bounds = bool(np.any(out_of_bounds_mask[:executed_action_count]))
    if start_invalid:
        collision_count = 1
        first_collision_action = 0
        last_collision_action = 0
        any_out_of_bounds = bool(out_of_bounds_mask[0])
    tail_free_steps = 0
    escape_distance = 0.0
    tail_clearance = -1.0
    recovered = False
    recovery_score = 0.0
    if collision_count and not any_out_of_bounds and not start_invalid:
        tail_free_steps = max(executed_action_count - last_collision_action - 1, 0)
        contact_position = executed[last_collision_action + 1]
        final_position = executed[executed_action_count]
        escape_distance = float(np.linalg.norm(final_position - contact_position))
        if tail_free_steps >= recovery_tail_free_steps:
            first_tail_position = executed_action_count - recovery_tail_free_steps + 1
            tail_clearance = float(np.min(endpoint_clearance[first_tail_position : executed_action_count + 1]))
            recovered = bool(escape_distance >= recovery_min_escape_m and tail_clearance >= recovery_clearance_px)
        if recovered:
            escape_score = float(np.clip(escape_distance / recovery_full_escape_m, 0.0, 1.0))
            clearance_score = float(
                np.clip(
                    (tail_clearance - execution_margin_px) / (recovery_clearance_px - execution_margin_px),
                    0.0,
                    1.0,
                )
            )
            recovery_score = min(escape_score, clearance_score)

    return {
        "trajectory_world_xy": executed,
        "collision_mask": collision_mask,
        "out_of_bounds_mask": out_of_bounds_mask,
        "endpoint_clearance_px": endpoint_clearance,
        "recovery_grace_mask": recovery_grace_mask,
        "start_invalid": start_invalid,
        "executed_action_count": int(executed_action_count),
        "collision_count": collision_count,
        "collided": bool(collision_count),
        "first_collision_action": first_collision_action,
        "first_collision_step": (
            0 if start_invalid else first_collision_action + 1 if first_collision_action >= 0 else -1
        ),
        "last_collision_action": last_collision_action,
        "out_of_bounds": any_out_of_bounds,
        "tail_free_steps": int(tail_free_steps),
        "recovery_escape_distance_m": float(escape_distance),
        "recovery_tail_clearance_px": float(tail_clearance),
        "recovered": bool(recovered),
        "recovery_score": float(recovery_score),
    }
