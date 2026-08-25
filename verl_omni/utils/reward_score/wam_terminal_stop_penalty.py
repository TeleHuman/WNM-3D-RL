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

"""Penalty for navigation trajectories that reach a goal but fail to stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TerminalStopConfig:
    """Thresholds are expressed in world-space metres."""

    # Matches GN0 remote_nav_agent.DAGGER_STOP_RADIUS_M used to collect this dataset.
    goal_radius: float = 1.50
    # Zero means that crossing back outside goal_radius is immediately a
    # failure. A positive value is retained only as an opt-in noise tolerance.
    leave_hysteresis: float = 0.0
    min_step_motion: float = 0.05
    max_tail_path_length: float = 0.30
    allowed_moving_steps: int = 1
    continued_motion_penalty: float = 0.0
    left_goal_penalty: float = -0.50


def compute_terminal_stop_penalty(
    trajectory_world_xy: torch.Tensor | Sequence[Sequence[float]],
    goal_world_xy: torch.Tensor | Sequence[float],
    config: TerminalStopConfig | None = None,
) -> dict[str, float | int | bool | str]:
    """Penalize a trajectory only when it leaves the goal after first entering.

    ``trajectory_world_xy`` should include every executed position; including
    the initial position is recommended. Motion that remains inside the goal
    radius is explicitly allowed. There is no penalty when the goal is not
    reached, because nDTW/progress reward already handles that failure.
    """
    cfg = config or TerminalStopConfig()
    trajectory = torch.as_tensor(trajectory_world_xy, dtype=torch.float32)
    goal = torch.as_tensor(goal_world_xy, dtype=torch.float32, device=trajectory.device)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError(f"trajectory_world_xy must have shape [T, >=2], got {tuple(trajectory.shape)}")
    if goal.numel() < 2:
        raise ValueError(f"goal_world_xy must contain at least two values, got {tuple(goal.shape)}")
    trajectory = trajectory[:, :2]
    goal = goal.flatten()[:2]
    if len(trajectory) == 0:
        raise ValueError("trajectory_world_xy must not be empty")
    if not bool(torch.isfinite(trajectory).all() and torch.isfinite(goal).all()):
        raise ValueError("trajectory and goal must contain only finite values")

    distances = torch.linalg.vector_norm(trajectory - goal, dim=-1)
    reached_indices = torch.nonzero(distances <= cfg.goal_radius, as_tuple=False).flatten()
    if len(reached_indices) == 0:
        return _result(reason="goal_not_reached")

    reached_step = int(reached_indices[0])
    tail = trajectory[reached_step:]
    if len(tail) < 2:
        result = _result(reason="no_post_reach_observation", reached=True, reached_step=reached_step)
        result["final_goal_distance"] = float(distances[-1])
        return result

    tail_steps = torch.linalg.vector_norm(tail[1:] - tail[:-1], dim=-1)
    tail_path_length = float(tail_steps.sum())
    moving_steps = int((tail_steps > cfg.min_step_motion).sum())
    max_tail_goal_distance = float(distances[reached_step:].max())
    final_goal_distance = float(distances[-1])
    left_indices = torch.nonzero(
        distances[reached_step:] > cfg.goal_radius + cfg.leave_hysteresis,
        as_tuple=False,
    ).flatten()
    left_goal = len(left_indices) > 0
    left_step = reached_step + int(left_indices[0]) if left_goal else -1

    if left_goal:
        penalty = cfg.left_goal_penalty
        reason = "left_goal_after_reaching"
    else:
        penalty = 0.0
        reason = "remained_inside_goal"

    return {
        "terminal_stop_penalty": float(penalty),
        "reached_goal": True,
        "failed_to_stop": bool(left_goal),
        "left_goal": bool(left_goal),
        "reached_step": reached_step,
        "left_step": left_step,
        "tail_path_length": tail_path_length,
        "moving_steps_after_reach": moving_steps,
        "max_tail_goal_distance": max_tail_goal_distance,
        "final_goal_distance": final_goal_distance,
        "terminal_stop_reason": reason,
    }


def apply_terminal_stop_penalty(
    action_reward: float,
    diagnostics: dict,
    *,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
) -> float:
    """Add the penalty and clamp to an explicitly configured bounded range."""
    if min_reward > max_reward:
        raise ValueError(f"min_reward must not exceed max_reward, got {min_reward} > {max_reward}")
    return max(
        float(min_reward),
        min(
            float(max_reward),
            float(action_reward) + float(diagnostics["terminal_stop_penalty"]),
        ),
    )


def _result(*, reason: str, reached: bool = False, reached_step: int = -1) -> dict[str, float | int | bool | str]:
    return {
        "terminal_stop_penalty": 0.0,
        "reached_goal": reached,
        "failed_to_stop": False,
        "left_goal": False,
        "reached_step": reached_step,
        "left_step": -1,
        "tail_path_length": 0.0,
        "moving_steps_after_reach": 0,
        "max_tail_goal_distance": 0.0,
        "final_goal_distance": 0.0,
        "terminal_stop_reason": reason,
    }
