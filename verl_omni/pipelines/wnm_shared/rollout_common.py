# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Contracts and deterministic helpers shared by WNM rollouts."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

_DEFAULT_INFERENCE_STEPS = 16
_DEFAULT_CFG_SCALE = 5.0
_DEPLOYED_ACTION_DIM = 3
_DIT_STEP_MASK = (
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    True,
    True,
    True,
)
_LAYER_CREDIT_FIELDS = (
    "layer_credit_stratum",
    "layer_credit_branch",
    "layer_credit_transition",
)
_PER_REQUEST_TRANSFORM_FIELDS = frozenset(("rollout_seed", "seed", *_LAYER_CREDIT_FIELDS))
_MAX_TORCH_SEED = 2**63 - 1


def derive_rollout_subseed(root_seed: int, stream_id: int) -> int:
    """Derive the stable WNM RNG substream without repo-local helpers."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError(f"root_seed must be an integer, got {type(root_seed).__name__}")
    if not 0 <= root_seed <= _MAX_TORCH_SEED:
        raise ValueError(f"root_seed must be in [0, {_MAX_TORCH_SEED}], got {root_seed}")
    if isinstance(stream_id, bool) or not isinstance(stream_id, int):
        raise TypeError(f"stream_id must be a non-negative integer, got {type(stream_id).__name__}")
    if stream_id < 0:
        raise ValueError(f"stream_id must be non-negative, got {stream_id}")
    return (root_seed * 1_000_003 + stream_id) % _MAX_TORCH_SEED


def build_shifted_schedule(
    *,
    num_inference_steps: int,
    shift: float = 5.0,
    num_train_timesteps: int = 1000,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact fixed SD3/Wan schedule shared by both WAM adapters."""

    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"shift must be finite and positive, got {shift}")
    if num_train_timesteps <= 0:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}")
    base_sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
    sigmas = shift * base_sigmas / (1 + (shift - 1) * base_sigmas)
    sigmas = sigmas.to(device=device)
    return sigmas, sigmas[:-1] * float(num_train_timesteps)


def _dit_prediction_source_steps(step_mask: Sequence[bool]) -> tuple[int, ...]:
    """Map every scheduler transition to the DiT prediction it consumes."""

    if not step_mask or not step_mask[0]:
        raise ValueError("WNM DiT cache mask must run the first transition.")
    source = 0
    result = []
    for step, should_run in enumerate(step_mask):
        if should_run:
            source = step
        result.append(source)
    return tuple(result)


_DIT_PREDICTION_SOURCE_STEPS = _dit_prediction_source_steps(_DIT_STEP_MASK)


def _deployed_action_policy_mask(action_state: torch.Tensor) -> torch.Tensor:
    """Select exactly the InteriorGS action coordinates consumed by GN0."""

    if action_state.ndim < 2 or action_state.shape[-1] < _DEPLOYED_ACTION_DIM:
        raise ValueError(
            "WNM deployment requires at least [dx, dy, dyaw] action "
            f"coordinates, got shape={tuple(action_state.shape)}."
        )
    mask = torch.zeros_like(action_state, dtype=torch.bool)
    mask[..., :_DEPLOYED_ACTION_DIM] = True
    return mask


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")
