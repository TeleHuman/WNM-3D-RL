# Modified by the WNM-3D-RL contributors, 2026.
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

"""Rollout Correction for diffusion training (experimental).

Diffusion-specific notes:
- No ``response_mask`` — log-probs are dense (no padding).  RS rejection is
  expressed as a 0-weight in ``rollout_is_weights`` instead of a mask.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.rollout_corr_helper import (
    compute_offpolicy_metrics,
    compute_rollout_correction_and_rejection_mask,
)

from verl_omni.trainer.diffusion.diffusion_algos import (
    combine_visual_action_log_probs,
)

__all__ = [
    "apply_bypass_mode_to_diffusion_batch",
    "apply_rollout_correction_to_diffusion_batch",
    "compute_validation_reference_kl_metrics",
    "compute_rollout_corr_metrics_from_logprobs",
    "rollout_correction_enabled",
    "validate_rollout_replay_log_probs",
]

_logger = logging.getLogger(__name__)
_warned_experimental = False

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def rollout_correction_enabled(rollout_corr_config) -> bool:
    """Return True if the config requests any IS or RS computation."""
    if rollout_corr_config is None:
        return False
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    return bool(rollout_is) or bool(rollout_rs)


def compute_validation_reference_kl_metrics(
    batch: DataProto,
    *,
    visual_log_prob_weight: float,
    action_log_prob_weight: float,
) -> dict[str, float]:
    """Estimate KL(current || immutable base) on current-policy val trajectories.

    Both likelihoods are FSDP actor replays of the exact same saved trajectory,
    so this diagnostic excludes the known vLLM-vs-actor numerical drift.  The
    ordinary sample mean (k1) is reported alongside the non-negative k3
    estimator, which is more useful when a finite validation sample makes k1
    slightly negative.
    """

    required = ("old_log_probs", "ref_log_prob")
    missing = [key for key in required if key not in batch.batch]
    if missing:
        raise KeyError(f"Validation reference KL is missing likelihood fields: {missing}.")
    current_visual = batch.batch["old_log_probs"].detach().float()
    reference_visual = batch.batch["ref_log_prob"].detach().float()
    if current_visual.shape != reference_visual.shape:
        raise ValueError(
            "Current/reference visual likelihood shapes differ: "
            f"current={tuple(current_visual.shape)}, "
            f"reference={tuple(reference_visual.shape)}."
        )

    current_action = batch.batch.get("old_action_log_probs", None)
    reference_action = batch.batch.get("ref_action_log_prob", None)
    if (current_action is None) != (reference_action is None):
        raise KeyError("Validation reference KL requires both current and reference action likelihoods.")
    if isinstance(current_action, torch.Tensor):
        current_action = current_action.detach().float()
        reference_action = reference_action.detach().float()
        if current_action.shape != reference_action.shape:
            raise ValueError(
                "Current/reference action likelihood shapes differ: "
                f"current={tuple(current_action.shape)}, "
                f"reference={tuple(reference_action.shape)}."
            )

    def summarize(
        prefix: str,
        current: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, float]:
        log_ratio = current - reference
        if not torch.isfinite(log_ratio).all():
            raise RuntimeError(f"Validation reference {prefix} log-ratio contains non-finite values.")
        # k3 = exp(-x) - 1 + x for x=log pi_current/pi_reference,
        # sampled under pi_current. Clamp only the exponential input to avoid a
        # single corrupt tail overflowing the diagnostic.
        k3 = torch.exp(torch.clamp(-log_ratio, min=-60.0, max=60.0)) - 1.0 + log_ratio
        return {
            f"reference_kl/{prefix}_k1": log_ratio.mean().item(),
            f"reference_kl/{prefix}_k3": k3.mean().item(),
            f"reference_kl/{prefix}_log_ratio_abs_mean": log_ratio.abs().mean().item(),
            f"reference_kl/{prefix}_log_ratio_std": log_ratio.std(unbiased=False).item(),
        }

    metrics = summarize("visual", current_visual, reference_visual)
    joint_current, joint_reference = combine_visual_action_log_probs(
        log_prob=current_visual,
        old_log_prob=reference_visual,
        action_log_prob=current_action,
        old_action_log_prob=reference_action,
        visual_log_prob_weight=visual_log_prob_weight,
        action_log_prob_weight=action_log_prob_weight,
    )
    metrics.update(summarize("joint", joint_current, joint_reference))
    metrics["reference_kl/sample_count"] = float(current_visual.shape[0])

    if isinstance(current_action, torch.Tensor):
        aggregated_action, aggregated_reference_action = combine_visual_action_log_probs(
            log_prob=current_visual,
            old_log_prob=reference_visual,
            action_log_prob=current_action,
            old_action_log_prob=reference_action,
            visual_log_prob_weight=0.0,
            action_log_prob_weight=1.0,
        )
        metrics.update(
            summarize(
                "action",
                aggregated_action,
                aggregated_reference_action,
            )
        )
        if current_action.ndim == current_visual.ndim + 1:
            for chunk in range(current_action.shape[-1]):
                metrics.update(
                    summarize(
                        f"action_chunk_{chunk}",
                        current_action[..., chunk],
                        reference_action[..., chunk],
                    )
                )
    return metrics


# ---------------------------------------------------------------------------
# Bypass mode
# ---------------------------------------------------------------------------


def apply_bypass_mode_to_diffusion_batch(batch: DataProto) -> None:
    """Set ``old_log_probs := rollout_log_probs`` (zero-cost substitution).

    Bypass-mode IS/RS is computed per SDE step inside ``diffusion_loss``,
    which reads ``config.rollout_correction`` from ``DiffusionActorConfig``.
    """
    global _warned_experimental
    if not _warned_experimental:
        _warned_experimental = True
        _logger.warning("[verl-omni] Rollout Correction for diffusion is an EXPERIMENTAL feature.")
    if batch.batch is None or "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "rollout_correction.bypass_mode=True requires `rollout_log_probs` in the batch. "
            "Ensure the rollout backend records log probs (calculate_log_probs=true)."
        )
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    has_wam_actions = "actions" in batch.batch or "all_action_latents" in batch.batch
    if has_wam_actions:
        if "rollout_action_log_probs" not in batch.batch:
            raise ValueError(
                "WAM rollout_correction.bypass_mode=True requires `rollout_action_log_probs` in the batch."
            )
        batch.batch["old_action_log_probs"] = batch.batch["rollout_action_log_probs"]


# ---------------------------------------------------------------------------
# Decoupled mode
# ---------------------------------------------------------------------------


def apply_rollout_correction_to_diffusion_batch(
    batch: DataProto,
    rollout_corr_config,
    *,
    visual_log_prob_weight: float = 1.0,
    action_log_prob_weight: float = 1.0,
) -> tuple[DataProto, dict[str, float]]:
    """Compute IS weights / RS mask for decoupled mode (``bypass_mode=False``).

    Uses joint visual/action log-probs for WAM batches (visual only otherwise)
    to compute IS weights and RS keep-mask, then folds both into a single
    ``rollout_is_weights`` tensor.
    Called once per global batch; in bypass mode this is skipped (old == rollout).
    """
    global _warned_experimental
    if not _warned_experimental:
        _warned_experimental = True
        _logger.warning("[verl-omni] Rollout Correction for diffusion is an EXPERIMENTAL feature.")

    if "old_log_probs" not in batch.batch or "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "Rollout Correction requires both 'old_log_probs' and 'rollout_log_probs' in the "
            "batch. Ensure the rollout backend records log probs (calculate_log_probs=true) "
            "and that the trainer runs the old_log_prob recompute step."
        )

    for name, value in (
        ("visual_log_prob_weight", visual_log_prob_weight),
        ("action_log_prob_weight", action_log_prob_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative value, got {value}.")

    old_visual_log_prob: torch.Tensor = batch.batch["old_log_probs"]
    rollout_visual_log_prob: torch.Tensor = batch.batch["rollout_log_probs"]
    old_log_prob: torch.Tensor
    rollout_log_prob: torch.Tensor

    action_pair_keys = ("old_action_log_probs", "rollout_action_log_probs")
    present_action_keys = [key for key in action_pair_keys if key in batch.batch]
    if present_action_keys and len(present_action_keys) != len(action_pair_keys):
        missing_action_keys = [key for key in action_pair_keys if key not in batch.batch]
        raise ValueError(f"WAM Rollout Correction is missing action likelihood fields: {missing_action_keys}.")
    if present_action_keys:
        old_action_log_prob = batch.batch["old_action_log_probs"]
        rollout_action_log_prob = batch.batch["rollout_action_log_probs"]
        old_log_prob, rollout_log_prob = combine_visual_action_log_probs(
            log_prob=old_visual_log_prob,
            old_log_prob=rollout_visual_log_prob,
            action_log_prob=old_action_log_prob,
            old_action_log_prob=rollout_action_log_prob,
            visual_log_prob_weight=visual_log_prob_weight,
            action_log_prob_weight=action_log_prob_weight,
        )
    else:
        old_log_prob, rollout_log_prob = combine_visual_action_log_probs(
            log_prob=old_visual_log_prob,
            old_log_prob=rollout_visual_log_prob,
            visual_log_prob_weight=visual_log_prob_weight,
            action_log_prob_weight=action_log_prob_weight,
        )

    if old_log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            "old_log_probs and rollout_log_probs must have identical shapes; "
            f"got {tuple(old_log_prob.shape)} vs {tuple(rollout_log_prob.shape)}."
        )
    if old_log_prob.dim() != 2:
        raise ValueError(
            "Rollout Correction expects 2D log-prob tensors of shape (batch, sde_window_size); "
            f"got shape {tuple(old_log_prob.shape)}."
        )

    # Diffusion log-probs are dense (no padding) — response_mask is all-ones.
    response_mask = torch.ones_like(old_log_prob)

    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    is_weights_proto: Optional[DataProto]
    modified_mask: torch.Tensor
    metrics: dict[str, float]
    is_weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )

    # Fold IS weights and RS mask into a single per-element multiplier.
    effective_weights: Optional[torch.Tensor] = None
    if is_weights_proto is not None:
        effective_weights = is_weights_proto.batch["rollout_is_weights"]
    if rollout_rs:
        rs_keep_mask = modified_mask  # 1 = keep, 0 = reject
        effective_weights = rs_keep_mask if effective_weights is None else effective_weights * rs_keep_mask
    if effective_weights is not None:
        batch.batch["rollout_is_weights"] = effective_weights.to(dtype=old_log_prob.dtype)

    return batch, metrics


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
) -> dict[str, float]:
    """Off-policy diagnostics from (current, rollout) log probs.

    Diffusion has no ``response_mask`` — all SDE steps are valid (no padding).

    Args:
        log_prob: Current policy log-prob, shape ``(B,)`` or ``(B, T)``.
        rollout_log_prob: Rollout policy log-prob, same shape.

    Returns:
        Dict of ``rollout_corr/`` metrics (KL, PPL, χ², etc.).
    """
    if log_prob.dim() == 1:
        log_prob = log_prob.unsqueeze(-1)
        rollout_log_prob = rollout_log_prob.unsqueeze(-1)

    response_mask = torch.ones_like(log_prob)
    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    metrics_with_prefix: dict[str, float] = {}
    for key, value in offpolicy_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_with_prefix[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_with_prefix[f"rollout_corr/{key}"] = value

    return metrics_with_prefix


def validate_rollout_replay_log_probs(
    batch: DataProto,
    validation_config,
    *,
    expected_global_step: int,
    visual_log_prob_weight: float = 1.0,
    action_log_prob_weight: float = 1.0,
    clip_ratio: Optional[float] = None,
) -> dict[str, float]:
    """Validate that synchronized rollout and actor replay describe one policy.

    vLLM-Omni samples the trajectory and records its transition likelihoods.
    The FSDP actor then replays the exact saved visual/action latents in one
    joint forward.  When the rollout weights were synchronized correctly, both
    likelihood sets may differ only by the configured numerical tolerance.

    This check is deliberately separate from rollout correction: a large
    rollout/replay difference here indicates stale or incorrectly mapped
    weights (or a non-replayable trajectory), not legitimate policy drift.
    """

    if validation_config is None or not validation_config.get("enabled", False):
        return {}

    atol = float(validation_config.get("atol", 1e-3))
    rtol = float(validation_config.get("rtol", 1e-3))
    configured_action_atol = validation_config.get("action_atol", None)
    configured_action_rtol = validation_config.get("action_rtol", None)
    action_atol = atol if configured_action_atol is None else float(configured_action_atol)
    action_rtol = rtol if configured_action_rtol is None else float(configured_action_rtol)
    fail_on_mismatch_raw = validation_config.get("fail_on_mismatch", None)
    if fail_on_mismatch_raw is None:
        fail_on_mismatch_raw = os.getenv("WAM_ROLLOUT_REPLAY_FAIL_ON_MISMATCH", "true")
    if isinstance(fail_on_mismatch_raw, str):
        normalized = fail_on_mismatch_raw.strip().lower()
        if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ValueError(
                f"rollout_log_prob_validation.fail_on_mismatch must be boolean, got {fail_on_mismatch_raw!r}."
            )
        fail_on_mismatch = normalized in {"1", "true", "yes", "on"}
    else:
        fail_on_mismatch = bool(fail_on_mismatch_raw)
    for name, value in (
        ("atol", atol),
        ("rtol", rtol),
        ("action_atol", action_atol),
        ("action_rtol", action_rtol),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"rollout_log_prob_validation.{name} must be finite and non-negative, got {value}.")

    rollout_steps = batch.non_tensor_batch.get("global_steps")
    if rollout_steps is None:
        raise KeyError("Rollout/replay validation requires the vLLM-Omni `global_steps` weight-version marker.")
    observed_steps = set(np.asarray(rollout_steps, dtype=object).reshape(-1).tolist())
    if observed_steps != {expected_global_step}:
        raise RuntimeError(
            "vLLM-Omni rollout used a different actor checkpoint than the replay worker: "
            f"expected global_step={expected_global_step}, observed={sorted(observed_steps, key=str)}."
        )

    metrics: dict[str, float] = {
        "rollout_replay/checkpoint_global_step": float(expected_global_step),
    }

    def compare_pair(
        label: str,
        replay_key: str,
        rollout_key: str,
        *,
        pair_atol: float = atol,
        pair_rtol: float = rtol,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [key for key in (replay_key, rollout_key) if key not in batch.batch]
        if missing:
            raise KeyError(f"Rollout/replay {label} validation is missing likelihood fields: {missing}.")
        replay = batch.batch[replay_key].detach().float()
        rollout = batch.batch[rollout_key].detach().float()
        if replay.shape != rollout.shape:
            raise ValueError(
                f"Rollout/replay {label} likelihood shapes differ: "
                f"replay={tuple(replay.shape)}, rollout={tuple(rollout.shape)}."
            )
        if not torch.isfinite(replay).all() or not torch.isfinite(rollout).all():
            raise RuntimeError(f"Rollout/replay {label} likelihoods contain non-finite values.")

        absolute_error = (replay - rollout).abs()
        signed_error = replay - rollout
        tolerance = pair_atol + pair_rtol * rollout.abs()
        close = absolute_error <= tolerance
        flat_error = absolute_error.reshape(-1)
        metrics[f"rollout_replay/{label}_max_abs_error"] = flat_error.max().item()
        metrics[f"rollout_replay/{label}_mean_abs_error"] = flat_error.mean().item()
        metrics[f"rollout_replay/{label}_rmse"] = signed_error.square().mean().sqrt().item()
        metrics[f"rollout_replay/{label}_p95_abs_error"] = torch.quantile(flat_error, 0.95).item()
        metrics[f"rollout_replay/{label}_p99_abs_error"] = torch.quantile(flat_error, 0.99).item()
        metrics[f"rollout_replay/{label}_mean_signed_error"] = signed_error.mean().item()
        metrics[f"rollout_replay/{label}_close_fraction"] = close.float().mean().item()
        metrics[f"rollout_replay/{label}_exceed_fraction"] = (~close).float().mean().item()
        metrics[f"rollout_replay/{label}_mismatch_detected"] = float(not bool(close.all()))
        if not bool(close.all()):
            flat_index = int(flat_error.argmax().item())
            max_error = flat_error[flat_index].item()
            allowed = tolerance.reshape(-1)[flat_index].item()
            message = (
                f"Rollout/replay {label} log-prob mismatch at flat index {flat_index}: "
                f"absolute_error={max_error:.6g}, allowed={allowed:.6g}, "
                f"atol={pair_atol}, rtol={pair_rtol}."
            )
            if fail_on_mismatch:
                raise RuntimeError(
                    message + " The rollout checkpoint, actor checkpoint, scheduler, and replay conditions must match."
                )
            _logger.warning(
                "%s Continuing because fail_on_mismatch=false; raw drift metrics are retained.",
                message,
            )
        return replay, rollout

    replay_visual, rollout_visual = compare_pair("visual", "old_log_probs", "rollout_log_probs")

    action_keys = {
        "old_action_log_probs",
        "rollout_action_log_probs",
    }
    has_action_policy = bool(action_keys.intersection(batch.batch.keys())) or "actions" in batch.batch
    if has_action_policy:
        replay_action, rollout_action = compare_pair(
            "action",
            "old_action_log_probs",
            "rollout_action_log_probs",
            pair_atol=action_atol,
            pair_rtol=action_rtol,
        )
        replay_joint, rollout_joint = combine_visual_action_log_probs(
            log_prob=replay_visual,
            old_log_prob=rollout_visual,
            action_log_prob=replay_action,
            old_action_log_prob=rollout_action,
            visual_log_prob_weight=visual_log_prob_weight,
            action_log_prob_weight=action_log_prob_weight,
        )
    else:
        if visual_log_prob_weight <= 0:
            raise ValueError("visual_log_prob_weight must be positive for visual-only replay validation.")
        replay_joint = visual_log_prob_weight * replay_visual
        rollout_joint = visual_log_prob_weight * rollout_visual

    correction_metrics = compute_rollout_corr_metrics_from_logprobs(replay_joint, rollout_joint)
    metrics.update(
        {key.replace("rollout_corr/", "rollout_replay/joint_"): value for key, value in correction_metrics.items()}
    )
    joint_log_ratio = replay_joint - rollout_joint
    joint_ratio = torch.exp(joint_log_ratio)
    metrics["rollout_replay/joint_log_ratio_mean"] = joint_log_ratio.mean().item()
    metrics["rollout_replay/joint_ratio_mean"] = joint_ratio.mean().item()
    metrics["rollout_replay/joint_ratio_std"] = joint_ratio.std(unbiased=False).item()
    metrics["rollout_replay/joint_current_behavior_kl"] = (rollout_joint - replay_joint).mean().item()
    if clip_ratio is not None:
        clip_ratio = float(clip_ratio)
        if not math.isfinite(clip_ratio) or clip_ratio < 0:
            raise ValueError(f"clip_ratio must be finite and non-negative, got {clip_ratio}.")
        clipped = (joint_ratio < 1.0 - clip_ratio) | (joint_ratio > 1.0 + clip_ratio)
        metrics["rollout_replay/joint_clip_fraction"] = clipped.float().mean().item()
    return metrics
