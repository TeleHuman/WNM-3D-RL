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

"""Checkout-independent Dance-SDE transition shared by WNM rollouts.

The VGGT inference checkout intentionally does not carry the RL-only
``gammanav...wam_dance_sde`` module that was added to WNM-2D. Keeping the
small transition primitive in VERL makes the rollout/replay contract identical
for both checkpoint families without modifying either inference repository.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


class DanceSDETransition(NamedTuple):
    prev_sample: torch.Tensor
    log_prob: torch.Tensor
    prev_sample_mean: torch.Tensor
    std_dev_t: torch.Tensor


def _reduce_log_prob(log_prob: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    reduce_dims = tuple(range(1, log_prob.ndim))
    if not reduce_dims:
        raise ValueError("Dance-SDE samples require a batch and at least one event dimension")
    if mask is None:
        return log_prob.mean(dim=reduce_dims)
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"log_prob_mask must be a torch.Tensor, got {type(mask).__name__}")
    mask = mask.detach().to(device=log_prob.device)
    try:
        mask = torch.broadcast_to(mask, log_prob.shape)
    except RuntimeError as exc:
        raise ValueError(
            "log_prob_mask must be broadcastable to the elementwise log probability: "
            f"mask={tuple(mask.shape)}, log_prob={tuple(log_prob.shape)}"
        ) from exc
    if mask.dtype != torch.bool:
        if mask.is_complex() or (torch.is_floating_point(mask) and not torch.isfinite(mask).all()):
            raise ValueError("log_prob_mask must contain finite binary values")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("log_prob_mask must contain only binary values")
        mask = mask.to(torch.bool)
    selected = mask.sum(dim=reduce_dims)
    if torch.any(selected == 0):
        raise ValueError("log_prob_mask must select at least one event element per batch item")
    masked = torch.where(
        mask,
        log_prob,
        torch.zeros((), device=log_prob.device, dtype=log_prob.dtype),
    )
    return masked.sum(dim=reduce_dims) / selected.to(dtype=log_prob.dtype)


def sample_dance_sde_previous_step(
    *,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    noise_level: float,
    generator: torch.Generator | None = None,
    prev_sample: torch.Tensor | None = None,
    log_prob_mask: torch.Tensor | None = None,
) -> DanceSDETransition:
    """Sample or replay one float32 Dance-SDE transition."""

    if sample.dtype != torch.float32 or model_output.dtype != torch.float32:
        raise TypeError("Dance-SDE sample and model_output must both be float32")
    if prev_sample is not None and prev_sample.dtype != torch.float32:
        raise TypeError("Dance-SDE prev_sample must be float32")
    if sample.shape != model_output.shape:
        raise ValueError(f"sample/model_output shape mismatch: {sample.shape} vs {model_output.shape}")
    if prev_sample is not None and prev_sample.shape != sample.shape:
        raise ValueError(f"sample/prev_sample shape mismatch: {sample.shape} vs {prev_sample.shape}")
    if not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError(f"Dance-SDE noise_level must be finite and positive, got {noise_level}")

    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=sample.device)
    sigma_prev = torch.as_tensor(sigma_prev, dtype=torch.float32, device=sample.device)
    if sigma.numel() != 1 or sigma_prev.numel() != 1:
        raise ValueError("sigma and sigma_prev must be scalar schedule entries")
    if not bool((sigma > sigma_prev).item()) or not bool((sigma > 0).item()):
        raise ValueError(f"expected sigma > sigma_prev and sigma > 0, got {sigma.item()} -> {sigma_prev.item()}")

    dsigma = sigma_prev - sigma
    delta_t = sigma - sigma_prev
    prev_sample_mean = sample + dsigma * model_output
    pred_original_sample = sample - sigma * model_output
    score_estimate = -(sample - pred_original_sample * (1 - sigma)) / (sigma**2)
    prev_sample_mean = prev_sample_mean + (-0.5 * noise_level**2 * score_estimate) * dsigma
    std_dev_t = noise_level * torch.sqrt(delta_t)

    if prev_sample is None:
        variance_noise = torch.randn(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    elementwise_log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
    elementwise_log_prob = (
        elementwise_log_prob
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device)))
    )
    return DanceSDETransition(
        prev_sample,
        _reduce_log_prob(elementwise_log_prob, log_prob_mask),
        prev_sample_mean,
        std_dev_t,
    )


__all__ = ["DanceSDETransition", "sample_dance_sde_previous_step"]
