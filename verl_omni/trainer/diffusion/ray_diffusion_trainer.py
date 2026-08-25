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
"""
Flow-GRPO / diffusion trainer with a Ray-based single controller.
This trainer supports model-agnostic model initialization with Hugging Face.
"""

import asyncio
import gc
import glob
import json
import logging
import math
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from pprint import pprint
from typing import Any, Literal, Mapping, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import compute_variance_proxy_metrics, process_validation_metrics
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.utils import update_prometheus_config

from verl_omni.trainer.config import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.diffusion_algos import (
    DiffusionAdvantageEstimator,
    get_diffusion_adv_estimator_fn,
    get_diffusion_loss_fn,
)
from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_old_policy_metrics,
    compute_reward_extra_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
    compute_wam_exploration_metrics,
)
from verl_omni.trainer.diffusion.diffusion_trainer_utils import NoOpCheckpointManager, old_policy_decay
from verl_omni.trainer.diffusion.rollout_correction import (
    apply_bypass_mode_to_diffusion_batch,
    apply_rollout_correction_to_diffusion_batch,
    compute_validation_reference_kl_metrics,
    rollout_correction_enabled,
    validate_rollout_replay_log_probs,
)
from verl_omni.utils.action_chunk_credit import (
    action_chunk_credit_enabled,
    action_chunk_weights,
)
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

sys_logger = logging.getLogger(__name__)

_WNM_ACTOR_REQUIRED_KEYS = (
    "all_latents",
    "all_timesteps",
    "all_action_latents",
    "all_action_timesteps",
    # Recorded sampler contract. Actor likelihood replay must consume the
    # exact deployed 8/16 cache/CFG/noise settings used by rollout.
    "dit_prediction_source_steps",
    "num_dit_prediction_steps",
    "num_dit_forwards",
    "num_inference_steps",
    "true_cfg_scale",
    "noise_level",
    "action_noise_level",
    "prompt_embeds",
    "prompt_embeds_mask",
    "clip_feature",
    "y",
    "state",
    "embodiment_id",
    "clean_x",
    "past_clean_x",
    "old_log_probs",
    "advantages",
    "old_action_log_probs",
    "action_advantages",
)


def _clone_validation_media_prefix(outputs: torch.Tensor, count: int) -> torch.Tensor:
    """Copy only the retained validation samples into independent CPU storage."""
    if count < 0 or count > len(outputs):
        raise ValueError(f"invalid retained validation media count: count={count}, batch={len(outputs)}")
    return outputs[:count].detach().cpu().clone()


def _validated_rollout_scalar(
    batch: Mapping[str, torch.Tensor],
    key: str,
    *,
    expected: float,
    batch_size: int,
    context: str,
    atol: float = 1e-7,
) -> float:
    """Return a rollout scalar only after checking the server-side contract.

    The configured sampler value is not sufficient evidence that vLLM-Omni
    consumed it: sampling parameters cross the driver, agent-loop and server
    boundaries before the adapter records them in the returned replay batch.
    Validate that recorded value here so train/validation noise cannot drift
    silently while TensorBoard continues to display the requested setting.
    """

    if key not in batch:
        raise KeyError(f"{context} rollout did not return required sampler field {key!r}")
    values = batch[key].detach().float().reshape(-1)
    if values.numel() != batch_size:
        raise ValueError(
            f"{context} rollout sampler field {key!r} must have one value per sample: "
            f"got {values.numel()}, expected {batch_size}."
        )
    expected_values = torch.full_like(values, float(expected))
    if not torch.allclose(values, expected_values, rtol=0.0, atol=atol):
        raise ValueError(
            f"{context} rollout sampler field {key!r} disagrees with the requested value "
            f"{expected}: min={values.min().item()}, max={values.max().item()}."
        )
    return float(values.mean().item())


def _event_validation_metrics(
    reward_extra_infos: dict[str, list],
) -> dict[str, float]:
    """Aggregate the held-out VGGT event suite without mixing its strata.

    Event membership is emitted by the reward from immutable parquet metadata.
    STOP and collision outcomes are recomputed from the model rollout, so these
    metrics measure policy behavior rather than the collection policy label.
    """

    if "event_val_active" not in reward_extra_infos:
        return {}
    active = np.asarray(reward_extra_infos["event_val_active"], dtype=np.float64)
    if active.ndim != 1 or len(active) == 0 or not np.all(active > 0.5):
        raise ValueError("Event validation batches must contain event_val_active=1 for every row.")
    size = len(active)

    def values(key: str, default: float | None = 0.0) -> np.ndarray:
        if key not in reward_extra_infos and default is None:
            raise KeyError(
                f"Event validation requires reward field {key!r}; refusing to "
                "replace a missing policy outcome with zeros."
            )
        raw = reward_extra_infos.get(key, [default] * size)
        result = np.asarray(raw, dtype=np.float64)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"Event validation metric {key} must be finite [N], got {result.shape}.")
        return result

    group_keys = {
        "collision_precursor": "event_val_collision_precursor",
        "premature_stop_risk": "event_val_premature_stop_risk",
        "premature_stop_near": "event_val_premature_stop_near",
        "premature_stop_far": "event_val_premature_stop_far",
        "required_stop": "event_val_required_stop",
        "required_stop_core": "event_val_required_stop_core",
        "required_stop_mid": "event_val_required_stop_mid",
        "required_stop_boundary": "event_val_required_stop_boundary",
        "near_goal_continue": "event_val_near_goal_continue",
    }
    masks = {name: values(key) > 0.5 for name, key in group_keys.items()}
    primary_sum = sum(
        masks[name].astype(np.int64)
        for name in (
            "collision_precursor",
            "premature_stop_risk",
            "required_stop",
            "near_goal_continue",
        )
    )
    if not np.all(primary_sum == 1):
        raise ValueError("Every event validation row must belong to exactly one high-level event.")

    # Use the compact, deployment-horizon GN0 diagnostics.  The per-credit
    # action_hard_stop_* fields describe all four hypothetical training chunks
    # and are intentionally omitted by compact reward diagnostics; treating a
    # missing one as zero previously made every event-val STOP rate exactly 0.
    predicted_stop = values("deployment_stop_emitted", default=None) > 0.5
    correct_stop = values("deployment_stop_success", default=None) > 0.5
    collision = values("action_any_collision") > 0.5
    chunk0_collision = values("action_chunk_0_collision") > 0.5
    expected_stop = values("event_val_expected_stop") > 0.5
    soft_risk = values("action_collision_soft_risk")
    action_reward = values("action_reward")
    visual_reward = values("visual_reward")
    distance = values("event_val_distance_to_goal_m", -1.0)

    result: dict[str, float] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            continue
        prefix = f"val-event/{name}"
        result[f"{prefix}/count"] = float(count)
        result[f"{prefix}/stop_rate"] = float(predicted_stop[mask].mean())
        result[f"{prefix}/continue_rate"] = float((~predicted_stop[mask]).mean())
        result[f"{prefix}/correct_stop_rate"] = float(correct_stop[mask].mean())
        result[f"{prefix}/collision_rate"] = float(collision[mask].mean())
        result[f"{prefix}/chunk0_collision_rate"] = float(chunk0_collision[mask].mean())
        result[f"{prefix}/collision_avoidance_rate"] = float((~collision[mask]).mean())
        result[f"{prefix}/soft_collision_risk_mean"] = float(soft_risk[mask].mean())
        result[f"{prefix}/action_reward_mean"] = float(action_reward[mask].mean())
        result[f"{prefix}/visual_reward_mean"] = float(visual_reward[mask].mean())
        result[f"{prefix}/distance_to_goal_mean_m"] = float(distance[mask].mean())

    true_positive = int(np.sum(predicted_stop & expected_stop))
    false_positive = int(np.sum(predicted_stop & ~expected_stop))
    false_negative = int(np.sum(~predicted_stop & expected_stop))
    true_negative = int(np.sum(~predicted_stop & ~expected_stop))

    def divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    result.update(
        {
            "val-event/stop/true_positive": float(true_positive),
            "val-event/stop/false_positive": float(false_positive),
            "val-event/stop/false_negative": float(false_negative),
            "val-event/stop/true_negative": float(true_negative),
            "val-event/stop/precision": precision,
            "val-event/stop/recall": recall,
            "val-event/stop/specificity": specificity,
            "val-event/stop/f1": divide(2.0 * precision * recall, precision + recall),
            "val-event/stop/balanced_accuracy": 0.5 * (recall + specificity),
            "val-event/stop/accuracy": divide(true_positive + true_negative, size),
        }
    )
    return result


_WNM_ACTOR_OPTIONAL_KEYS = (
    "action_loss_weights",
    "action_terminal_safety_advantages",
    "action_stop_advantages",
    "action_stop_loss_weights",
    "action_collision_advantages",
    "action_collision_loss_weights",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "layer_credit_replay_transition",
    "ref_log_prob",
    "ref_prev_sample_mean",
    "old_prev_sample_mean",
    "rollout_is_weights",
    "ref_action_log_prob",
    "ref_action_prev_sample_mean",
    "old_action_prev_sample_mean",
    "action_policy_mask",
)

_WNM_LAYER_CREDIT_STRATA = (
    (0, 1, 2),
    (6,),
    (10,),
    (13, 14, 15),
)


def _scheduled_rollout_noise_levels(config: Any, *, global_step: int) -> dict[str, float] | None:
    """Resolve a per-step training-only SDE noise schedule.

    The generated trajectory records the resolved noise tensors, so actor
    replay continues to use the exact rollout variance. Validation retains its
    separately configured fixed noise and is intentionally not overridden.
    """

    schedule_config = config.algorithm.get("rollout_noise_schedule", None)
    if not schedule_config or not bool(schedule_config.get("enabled", False)):
        return None

    start_step = int(schedule_config.start_step)
    end_step = int(schedule_config.end_step)
    if start_step <= 0 or end_step < start_step:
        raise ValueError(f"rollout_noise_schedule requires 0 < start_step <= end_step, got {start_step}..{end_step}.")

    if end_step == start_step:
        progress = 1.0 if global_step >= end_step else 0.0
    else:
        progress = float(np.clip((global_step - start_step) / (end_step - start_step), 0.0, 1.0))
    schedule = str(schedule_config.get("schedule", "linear")).strip().lower()
    if schedule == "linear":
        interpolation = progress
    elif schedule == "cosine":
        interpolation = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"unsupported rollout_noise_schedule.schedule={schedule!r}; expected linear or cosine")

    rollout_algo = config.actor_rollout_ref.rollout.algo

    def resolve(name: str, default: float) -> float:
        start = float(schedule_config.get(f"{name}_start", default))
        end = float(schedule_config.get(f"{name}_end", start))
        for field, value in ((f"{name}_start", start), (f"{name}_end", end)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"rollout_noise_schedule.{field} must be finite and positive, got {value}. "
                    "Deterministic ODE inference must be approximated with a positive training floor."
                )
        return start + interpolation * (end - start)

    visual_default = float(rollout_algo.noise_level)
    action_default = rollout_algo.get("action_noise_level", None)
    action_default = visual_default if action_default is None else float(action_default)
    return {
        "noise_level": resolve("visual", visual_default),
        "action_noise_level": resolve("action", action_default),
        "progress": progress,
    }


def _layer_conditioned_credit_config(config: Any) -> tuple[bool, int]:
    """Return the layer-conditioned credit switch and branches per noise stratum."""

    layer_config = config.algorithm.get("layer_conditioned_credit", None)
    if not layer_config or not bool(layer_config.get("enabled", False)):
        return False, 0
    branches_per_stratum = int(layer_config.get("branches_per_stratum", 4))
    if branches_per_stratum < 2:
        raise ValueError("layer_conditioned_credit.branches_per_stratum must be at least 2.")
    return True, branches_per_stratum


def _full_transition_actor_replay_config(config: Any) -> tuple[bool, int]:
    """Return the fail-closed full-trajectory actor replay contract."""

    replay_config = config.algorithm.get("full_transition_actor_replay", None)
    if not replay_config or not bool(replay_config.get("enabled", False)):
        return False, 0
    expected_transitions = int(replay_config.get("expected_transitions", 16))
    if expected_transitions <= 1:
        raise ValueError(
            f"full_transition_actor_replay.expected_transitions must exceed one, got {expected_transitions}."
        )
    return True, expected_transitions


def _select_layer_credit_transitions(*, global_step: int, seed: int) -> np.ndarray:
    """Select one real DiT source transition from each deployed noise stratum."""

    rng = np.random.default_rng(int(seed) + 1_000_003 * int(global_step))
    return np.asarray(
        [candidates[int(rng.integers(0, len(candidates)))] for candidates in _WNM_LAYER_CREDIT_STRATA],
        dtype=np.int64,
    )


def _assign_layer_conditioned_credit(
    data: DataProto,
    *,
    rollout_n: int,
    branches_per_stratum: int,
    global_step: int,
    seed: int,
) -> np.ndarray:
    """Label repeated rollouts as four counterfactual groups for each prompt."""

    num_strata = len(_WNM_LAYER_CREDIT_STRATA)
    expected_rollouts = num_strata * branches_per_stratum
    if rollout_n != expected_rollouts:
        raise ValueError(
            "Layer-conditioned credit requires rollout.n == "
            f"num_strata * branches_per_stratum = {num_strata} * {branches_per_stratum} "
            f"= {expected_rollouts}, got {rollout_n}."
        )
    if len(data) % rollout_n:
        raise ValueError(f"Repeated rollout batch {len(data)} is not divisible by rollout.n={rollout_n}.")

    slot = np.arange(len(data), dtype=np.int64) % rollout_n
    stratum = slot // branches_per_stratum
    branch = slot % branches_per_stratum
    selected_by_stratum = _select_layer_credit_transitions(global_step=global_step, seed=seed)
    transition = selected_by_stratum[stratum]
    data.non_tensor_batch["layer_credit_stratum"] = stratum
    data.non_tensor_batch["layer_credit_branch"] = branch
    data.non_tensor_batch["layer_credit_transition"] = transition
    return selected_by_stratum


def _layer_conditioned_group_index(data: DataProto) -> np.ndarray:
    """Build the conditional GRPO group key (prompt, noise stratum)."""

    if "uid" not in data.non_tensor_batch:
        raise KeyError("Layer-conditioned credit requires uid for every rollout.")
    if "layer_credit_stratum" not in data.non_tensor_batch:
        raise KeyError("Layer-conditioned credit requires layer_credit_stratum.")
    uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
    strata = np.asarray(data.non_tensor_batch["layer_credit_stratum"], dtype=np.int64)
    if uids.shape != strata.shape:
        raise ValueError(f"uid and layer_credit_stratum shapes differ: {uids.shape} vs {strata.shape}.")
    return np.asarray(
        [f"{uid}\x1fstratum={int(stratum)}" for uid, stratum in zip(uids, strata, strict=True)],
        dtype=object,
    )


def _compact_wnm_layer_credit_batch(data: DataProto) -> DataProto:
    """Keep only the transition whose counterfactual reward produced each advantage.

    The rollout still completes all 16 transitions to obtain a terminal reward.
    The actor, however, must update only the branched DiT source transition.
    Compacting the replay trajectory here changes the actor contract from T=16
    to T=1 without teaching the worker about per-row timestep indices.
    """

    if "layer_credit_transition" not in data.non_tensor_batch:
        return data
    transitions_np = np.asarray(data.non_tensor_batch["layer_credit_transition"], dtype=np.int64)
    batch_size = len(data)
    if transitions_np.shape != (batch_size,):
        raise ValueError(f"layer_credit_transition must have shape {(batch_size,)}, got {transitions_np.shape}.")
    num_transitions = int(data.batch["old_log_probs"].shape[1])
    if np.any(transitions_np < 0) or np.any(transitions_np >= num_transitions):
        raise ValueError(f"layer_credit_transition must lie in [0,{num_transitions}), got {np.unique(transitions_np)}.")
    allowed = {step for stratum in _WNM_LAYER_CREDIT_STRATA for step in stratum}
    invalid = sorted(set(int(step) for step in transitions_np) - allowed)
    if invalid:
        raise ValueError(f"Layer-conditioned credit selected non-DiT-source transitions: {invalid}.")

    transitions = torch.as_tensor(transitions_np, dtype=torch.long, device=data.batch["old_log_probs"].device)
    rows = torch.arange(batch_size, device=transitions.device)
    # T=1 is otherwise indistinguishable from a malformed/truncated rollout at
    # the actor boundary. Preserve the original deployed-schedule index so the
    # WNM adapter can validate this compact replay fail-closed.
    data.batch["layer_credit_replay_transition"] = transitions.clone()
    source_steps = data.batch.get("dit_prediction_source_steps", None)
    if isinstance(source_steps, torch.Tensor):
        selected_sources = source_steps.to(transitions.device)[rows, transitions]
        if not torch.equal(selected_sources, transitions):
            raise ValueError(
                "Layer-conditioned credit can update only transitions that execute DiT in the deployed cache schedule; "
                f"selected={transitions.tolist()}, source={selected_sources.tolist()}."
            )

    transition_keys = (
        "all_timesteps",
        "all_action_timesteps",
        "old_log_probs",
        "advantages",
        "returns",
        "old_action_log_probs",
        "action_advantages",
        "action_loss_weights",
        "action_terminal_safety_advantages",
        "action_stop_advantages",
        "action_stop_loss_weights",
        "action_collision_advantages",
        "action_collision_loss_weights",
        "ref_log_prob",
        "old_prev_sample_mean",
        "ref_prev_sample_mean",
        "rollout_is_weights",
        "ref_action_log_prob",
        "old_action_prev_sample_mean",
        "ref_action_prev_sample_mean",
    )
    for key in transition_keys:
        value = data.batch.get(key, None)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2 or value.shape[0] != batch_size or value.shape[1] != num_transitions:
            continue
        local_rows = rows.to(value.device)
        local_steps = transitions.to(value.device)
        data.batch[key] = value[local_rows, local_steps].unsqueeze(1)

    for key in ("all_latents", "all_action_latents"):
        value = data.batch.get(key, None)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2 or value.shape[:2] != torch.Size((batch_size, num_transitions + 1)):
            raise ValueError(
                f"{key} must contain T+1 replay states before layer-conditioned compaction; got {tuple(value.shape)}."
            )
        local_rows = rows.to(value.device)
        local_steps = transitions.to(value.device)
        data.batch[key] = torch.stack(
            (value[local_rows, local_steps], value[local_rows, local_steps + 1]),
            dim=1,
        )

    action_mask = data.batch.get("action_policy_mask", None)
    if (
        isinstance(action_mask, torch.Tensor)
        and action_mask.ndim >= 2
        and action_mask.shape[0] == batch_size
        and action_mask.shape[1] == num_transitions
    ):
        local_rows = rows.to(action_mask.device)
        local_steps = transitions.to(action_mask.device)
        data.batch["action_policy_mask"] = action_mask[local_rows, local_steps].unsqueeze(1)
    return data


def _layer_credit_metrics(
    *,
    visual_rewards: torch.Tensor,
    action_rewards: torch.Tensor,
    strata: np.ndarray,
    transitions_by_stratum: np.ndarray,
) -> dict[str, float]:
    """Expose raw conditional reward scale so the four standards are auditable."""

    result: dict[str, float] = {}
    for stratum, transition in enumerate(transitions_by_stratum):
        mask = torch.as_tensor(strata == stratum, device=visual_rewards.device)
        prefix = f"critic/layer_credit/stratum_{stratum}"
        result[f"{prefix}/transition"] = float(transition)
        for name, rewards in (("visual", visual_rewards), ("action", action_rewards)):
            selected = rewards.reshape(-1)[mask]
            result[f"{prefix}/{name}_reward_mean"] = float(selected.mean().item())
            result[f"{prefix}/{name}_reward_std"] = float(selected.std(unbiased=False).item())
    return result


def _action_chunk_credit_metrics(
    *,
    action_rewards: torch.Tensor,
    action_advantages: torch.Tensor,
    action_masks: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Expose reward/advantage scale independently for every temporal chunk."""

    if action_rewards.ndim != 2 or action_advantages.ndim != 3:
        raise ValueError(
            "Chunk metrics expect rewards [B,C] and advantages [B,T,C], "
            f"got {tuple(action_rewards.shape)} and {tuple(action_advantages.shape)}."
        )
    if action_rewards.shape[0] != action_advantages.shape[0] or action_rewards.shape[1] != action_advantages.shape[2]:
        raise ValueError(
            "Chunk reward/advantage shapes disagree: "
            f"rewards={tuple(action_rewards.shape)}, advantages={tuple(action_advantages.shape)}."
        )
    result: dict[str, float] = {}
    if action_masks is not None and action_masks.shape != action_rewards.shape:
        raise ValueError(
            "Chunk action masks must match rewards for metrics; "
            f"rewards={tuple(action_rewards.shape)}, masks={tuple(action_masks.shape)}."
        )
    for index in range(action_rewards.shape[1]):
        reward = action_rewards[:, index]
        advantage = action_advantages[:, :, index]
        prefix = f"critic/action_chunk_{index}"
        result[f"{prefix}/reward_mean"] = float(reward.mean().item())
        result[f"{prefix}/reward_std"] = float(reward.std(unbiased=False).item())
        result[f"{prefix}/advantage_mean"] = float(advantage.mean().item())
        result[f"{prefix}/advantage_std"] = float(advantage.std(unbiased=False).item())
        result[f"{prefix}/advantage_abs_mean"] = float(advantage.abs().mean().item())
        if action_masks is not None:
            result[f"{prefix}/active_fraction"] = float(action_masks[:, index].float().mean().item())
    return result


def _action_stop_credit_metrics(
    *,
    stop_rewards: torch.Tensor,
    stop_advantages: torch.Tensor,
    stop_masks: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Expose the independently normalized, uniformly weighted stop credit."""

    if stop_rewards.ndim != 2 or stop_advantages.ndim != 3:
        raise ValueError(
            "Stop metrics expect rewards [B,C] and advantages [B,T,C], "
            f"got {tuple(stop_rewards.shape)} and {tuple(stop_advantages.shape)}."
        )
    if stop_rewards.shape[0] != stop_advantages.shape[0] or stop_rewards.shape[1] != stop_advantages.shape[2]:
        raise ValueError(
            "Stop reward/advantage shapes disagree: "
            f"rewards={tuple(stop_rewards.shape)}, "
            f"advantages={tuple(stop_advantages.shape)}."
        )
    if stop_masks is not None and stop_masks.shape != stop_rewards.shape:
        raise ValueError(
            "Stop masks must match rewards for metrics; "
            f"rewards={tuple(stop_rewards.shape)}, masks={tuple(stop_masks.shape)}."
        )

    result: dict[str, float] = {}
    for index in range(stop_rewards.shape[1]):
        reward = stop_rewards[:, index]
        advantage = stop_advantages[:, :, index]
        prefix = f"critic/action_chunk_{index}/stop"
        result[f"{prefix}_reward_mean"] = float(reward.mean().item())
        result[f"{prefix}_reward_std"] = float(reward.std(unbiased=False).item())
        result[f"{prefix}_advantage_mean"] = float(advantage.mean().item())
        result[f"{prefix}_advantage_std"] = float(advantage.std(unbiased=False).item())
        result[f"{prefix}_advantage_abs_mean"] = float(advantage.abs().mean().item())
        if stop_masks is not None:
            result[f"{prefix}_active_fraction"] = float(stop_masks[:, index].float().mean().item())
    return result


def _action_collision_credit_metrics(
    *,
    collision_rewards: torch.Tensor,
    collision_advantages: torch.Tensor,
    collision_masks: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Expose collision-only reward and advantage scale for every chunk."""

    if collision_rewards.ndim != 2 or collision_advantages.ndim != 3:
        raise ValueError(
            "Collision metrics expect rewards [B,C] and advantages [B,T,C], "
            f"got {tuple(collision_rewards.shape)} and "
            f"{tuple(collision_advantages.shape)}."
        )
    if (
        collision_rewards.shape[0] != collision_advantages.shape[0]
        or collision_rewards.shape[1] != collision_advantages.shape[2]
    ):
        raise ValueError(
            "Collision reward/advantage shapes disagree: "
            f"rewards={tuple(collision_rewards.shape)}, "
            f"advantages={tuple(collision_advantages.shape)}."
        )
    if collision_masks is not None and collision_masks.shape != collision_rewards.shape:
        raise ValueError(
            "Collision masks must match rewards for metrics; "
            f"rewards={tuple(collision_rewards.shape)}, "
            f"masks={tuple(collision_masks.shape)}."
        )

    result: dict[str, float] = {}
    for index in range(collision_rewards.shape[1]):
        reward = collision_rewards[:, index]
        advantage = collision_advantages[:, :, index]
        prefix = f"critic/action_chunk_{index}/collision"
        result[f"{prefix}_reward_mean"] = float(reward.mean().item())
        result[f"{prefix}_reward_std"] = float(reward.std(unbiased=False).item())
        result[f"{prefix}_advantage_mean"] = float(advantage.mean().item())
        result[f"{prefix}_advantage_std"] = float(advantage.std(unbiased=False).item())
        result[f"{prefix}_advantage_abs_mean"] = float(advantage.abs().mean().item())
        if collision_masks is not None:
            result[f"{prefix}_active_fraction"] = float(collision_masks[:, index].float().mean().item())
    return result


def _tensor_nbytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _prepare_wnm_actor_tensordict(
    batch: DataProto,
    *,
    expected_text_len: int,
    architecture: str = "WNM2D",
):
    """Keep the fixed-length rollout context and only actor-consumed tensors.

    WNM's checkpoint text encoder already returns a masked, zero-padded
    ``[B, 512, D]`` context. The generic diffusion path used to strip this into
    512 Python-created jagged rows, only for every actor micro-batch to pad it
    back to dense form. Besides needless copies, that path obscures the exact
    SFT/rollout contract. Preserve the original dense tensor and mask instead.
    """

    tensor_batch = batch.batch
    required_keys = _WNM_ACTOR_REQUIRED_KEYS
    if architecture == "WNM3D":
        required_keys = tuple(
            key for key in required_keys if key not in {"clean_x", "past_clean_x", "embodiment_id"}
        ) + ("past_obs_tokens",)
    elif architecture != "WNM2D":
        raise ValueError(f"Unsupported WNM actor architecture: {architecture}")
    missing = [key for key in required_keys if key not in tensor_batch]
    if missing:
        raise KeyError(f"WNM actor batch is missing required replay tensors: {missing}.")

    prompt_embeds = tensor_batch["prompt_embeds"]
    prompt_mask = tensor_batch["prompt_embeds_mask"]
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[1] != expected_text_len:
        raise ValueError(
            "WNM actor must preserve the rollout's fixed text context: "
            f"expected [B,{expected_text_len},D], got {tuple(prompt_embeds.shape)}."
        )
    if tuple(prompt_mask.shape) != tuple(prompt_embeds.shape[:2]):
        raise ValueError(
            f"WNM prompt mask {tuple(prompt_mask.shape)} does not match embeddings {tuple(prompt_embeds.shape[:2])}."
        )

    selected_keys = list(required_keys)
    selected_keys.extend(key for key in _WNM_ACTOR_OPTIONAL_KEYS if key in tensor_batch)
    selected = tensor_batch.select(*selected_keys, strict=True)
    kept = sum(_tensor_nbytes(value) for value in selected.values() if isinstance(value, torch.Tensor))
    total = sum(_tensor_nbytes(value) for value in tensor_batch.values() if isinstance(value, torch.Tensor))
    return selected, total, kept


class StaggeredLLMServerManager(LLMServerManager):
    """Start rollout replicas in bounded batches to cap transient PID usage.

    VLN rollout initialization imports and compiles several heavyweight modules.
    Starting every single-GPU replica at once can exceed a container PID/thread
    limit even when steady-state CPU and GPU memory are well within budget.
    ``WAM_ROLLOUT_INIT_CONCURRENCY`` bounds only initialization; all replicas are
    live and available to the load balancer after startup completes.
    """

    async def _initialize_llm_servers(self, start_rank: int = None):
        if start_rank is None:
            start_rank = self.start_rank

        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )

        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            initializers = [server.init_hybrid(self.worker_group) for server in self.rollout_replicas]
        elif self.worker_group and self.rollout_config.name == "trtllm":
            initializers = [
                server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                for server in self.rollout_replicas
            ]
        else:
            initializers = [server.init_standalone() for server in self.rollout_replicas]

        requested_concurrency = int(os.environ.get("WAM_ROLLOUT_INIT_CONCURRENCY", "0"))
        init_concurrency = num_replicas if requested_concurrency <= 0 else min(requested_concurrency, num_replicas)
        for batch_start in range(0, num_replicas, init_concurrency):
            batch = initializers[batch_start : batch_start + init_concurrency]
            print(
                "LLMServerManager: initializing rollout replicas "
                f"{batch_start + 1}-{batch_start + len(batch)}/{num_replicas} "
                f"(concurrency={init_concurrency})"
            )
            await asyncio.gather(*batch)

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)


def compute_advantage(
    data: DataProto,
    adv_estimator: str,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[DiffusionAlgoConfig] = None,
    index_override: Optional[np.ndarray] = None,
) -> DataProto:
    """Compute advantage estimates for diffusion policy optimization.

    This function computes advantage estimates for diffusion models using the registered
    advantage estimator (e.g., Flow-GRPO). The advantage estimates are used to guide
    policy optimization across denoising timesteps.

    Args:
        data (DataProto): The data containing batched diffusion model outputs and inputs.
        adv_estimator (str): Name of the advantage estimator to use (e.g., Flow-GRPO).
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard
            deviation in GRPO. Defaults to True.
        global_std (bool, optional): Whether to use global standard deviation for normalization.
            Defaults to True.
        config (DiffusionAlgoConfig, optional): Configuration object for algorithm settings.
            Defaults to None.

    Returns:
        DataProto: The updated data with computed ``advantages`` and ``returns`` in its batch.
    """
    adv_kwargs = {
        "sample_level_rewards": data.batch["sample_level_rewards"],
        "config": config,
    }
    if index_override is not None:
        adv_kwargs["index"] = index_override
    elif "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    if "reward_baselines" in data.batch:
        adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

    adv_estimator_fn = get_diffusion_adv_estimator_fn(adv_estimator)
    if adv_estimator in {
        DiffusionAdvantageEstimator.FLOW_GRPO,
        DiffusionAdvantageEstimator.DANCE_GRPO,
    }:
        adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo
        adv_kwargs["global_std"] = global_std
    advantages, returns = adv_estimator_fn(**adv_kwargs)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def compute_separate_wam_advantages(
    data: DataProto,
    *,
    visual_rewards: torch.Tensor,
    action_rewards: torch.Tensor,
    action_masks: Optional[torch.Tensor] = None,
    action_stop_rewards: Optional[torch.Tensor] = None,
    action_stop_masks: Optional[torch.Tensor] = None,
    action_collision_rewards: Optional[torch.Tensor] = None,
    action_collision_masks: Optional[torch.Tensor] = None,
    action_terminal_safety_rewards: Optional[torch.Tensor] = None,
    adv_estimator: str,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[DiffusionAlgoConfig] = None,
    index_override: Optional[np.ndarray] = None,
) -> DataProto:
    """Compute independent GRPO advantages for WAM visual and action policies."""
    num_timesteps = data.batch["old_log_probs"].shape[1]
    expected = (len(data), 1)
    chunked_action = action_rewards.ndim == 2 and action_rewards.shape[1] > 1
    valid_action_shape = tuple(action_rewards.shape) == expected or (
        chunked_action
        and action_rewards.shape[0] == len(data)
        and len(action_chunk_weights(expected_chunks=action_rewards.shape[1])) == action_rewards.shape[1]
    )
    if tuple(visual_rewards.shape) != expected or not valid_action_shape:
        raise ValueError(
            f"WAM visual rewards must have shape {expected}; action rewards must "
            "have shape [B,1] or [B,C] with configured chunk weights; "
            f"visual={tuple(visual_rewards.shape)}, action={tuple(action_rewards.shape)}."
        )
    if action_masks is not None:
        if not chunked_action or tuple(action_masks.shape) != tuple(action_rewards.shape):
            raise ValueError(
                "WAM action masks require chunk rewards and must match [B,C]; "
                f"rewards={tuple(action_rewards.shape)}, masks={tuple(action_masks.shape)}."
            )
        if not torch.isfinite(action_masks).all():
            raise ValueError("WAM action masks must be finite")
        action_masks = action_masks > 0.5
    if action_stop_rewards is not None:
        if not chunked_action or tuple(action_stop_rewards.shape) != tuple(action_rewards.shape):
            raise ValueError(
                "WAM stop rewards require chunked action rewards and must match "
                f"[B,C]; action={tuple(action_rewards.shape)}, "
                f"stop={tuple(action_stop_rewards.shape)}."
            )
        if not torch.isfinite(action_stop_rewards).all():
            raise ValueError("WAM stop rewards must be finite")
        if action_stop_masks is not None:
            if tuple(action_stop_masks.shape) != tuple(action_stop_rewards.shape):
                raise ValueError(
                    "WAM stop masks must match stop rewards; "
                    f"rewards={tuple(action_stop_rewards.shape)}, "
                    f"masks={tuple(action_stop_masks.shape)}."
                )
            if not torch.isfinite(action_stop_masks).all():
                raise ValueError("WAM stop masks must be finite")
            action_stop_masks = action_stop_masks > 0.5
    elif action_stop_masks is not None:
        raise ValueError("WAM stop masks were provided without stop rewards")
    if action_collision_rewards is not None:
        if not chunked_action or tuple(action_collision_rewards.shape) != tuple(action_rewards.shape):
            raise ValueError(
                "WAM collision rewards require chunked action rewards and must "
                f"match [B,C]; action={tuple(action_rewards.shape)}, "
                f"collision={tuple(action_collision_rewards.shape)}."
            )
        if not torch.isfinite(action_collision_rewards).all():
            raise ValueError("WAM collision rewards must be finite")
        if action_collision_masks is not None:
            if tuple(action_collision_masks.shape) != tuple(action_collision_rewards.shape):
                raise ValueError(
                    "WAM collision masks must match collision rewards; "
                    f"rewards={tuple(action_collision_rewards.shape)}, "
                    f"masks={tuple(action_collision_masks.shape)}."
                )
            if not torch.isfinite(action_collision_masks).all():
                raise ValueError("WAM collision masks must be finite")
            action_collision_masks = action_collision_masks > 0.5
    elif action_collision_masks is not None:
        raise ValueError("WAM collision masks were provided without collision rewards")
    if action_terminal_safety_rewards is not None:
        if tuple(action_terminal_safety_rewards.shape) != expected:
            raise ValueError(
                f"WAM terminal safety rewards must have shape [B,1], got {tuple(action_terminal_safety_rewards.shape)}."
            )
        if not chunked_action:
            raise ValueError("WAM terminal safety advantage requires temporal action chunks")
        if not torch.isfinite(action_terminal_safety_rewards).all():
            raise ValueError("WAM terminal safety rewards must be finite")
    total_rewards = data.batch["sample_level_rewards"]
    reward_steps = 1 if index_override is not None else num_timesteps
    data.batch["sample_level_rewards"] = visual_rewards.expand(-1, reward_steps)
    compute_advantage(
        data,
        adv_estimator,
        norm_adv_by_std_in_grpo,
        global_std,
        config,
        index_override=index_override,
    )
    visual_advantages = data.batch["advantages"].clone()

    def _active_loss_weights(masks: torch.Tensor) -> torch.Tensor:
        """Make a full-batch mean equal the mean over active rows per chunk."""

        mask_values = masks.to(dtype=visual_rewards.dtype)
        active_counts = mask_values.sum(dim=0, keepdim=True)
        # A singleton cannot form a within-prompt GRPO contrast, so treat it as
        # inactive as well. Its advantage is already zero; zeroing the loss
        # weight also keeps the diagnostics honest about effective credit.
        gain = torch.where(
            active_counts > 1,
            mask_values.new_full(active_counts.shape, float(len(data))) / active_counts.clamp_min(1.0),
            torch.zeros_like(active_counts),
        )
        weights = mask_values * gain
        return weights.unsqueeze(1).expand(-1, reward_steps, -1).clone()

    def _compute_chunk_advantages(
        rewards: torch.Tensor,
        masks: Optional[torch.Tensor],
        *,
        credit_name: str,
    ) -> torch.Tensor:
        chunk_advantages = []
        for chunk_index in range(rewards.shape[1]):
            chunk_rewards = rewards[:, chunk_index : chunk_index + 1].expand(-1, reward_steps)
            if masks is None:
                data.batch["sample_level_rewards"] = chunk_rewards
                compute_advantage(
                    data,
                    adv_estimator,
                    norm_adv_by_std_in_grpo,
                    global_std,
                    config,
                    index_override=index_override,
                )
                chunk_advantages.append(data.batch["advantages"].clone())
                continue

            active = masks[:, chunk_index]
            masked_advantages = torch.zeros_like(chunk_rewards)
            if int(active.sum().item()) > 1:
                if index_override is not None:
                    chunk_index_values = np.asarray(index_override)[active.detach().cpu().numpy()]
                elif "uid" in data.non_tensor_batch:
                    chunk_index_values = np.asarray(data.non_tensor_batch["uid"])[active.detach().cpu().numpy()]
                else:
                    raise KeyError(f"Masked WAM {credit_name} advantages require index_override or uid")
                adv_kwargs = {
                    "sample_level_rewards": chunk_rewards[active],
                    "index": chunk_index_values,
                    "config": config,
                }
                if "reward_baselines" in data.batch:
                    adv_kwargs["reward_baselines"] = data.batch["reward_baselines"][active]
                adv_estimator_fn = get_diffusion_adv_estimator_fn(adv_estimator)
                if adv_estimator in {
                    DiffusionAdvantageEstimator.FLOW_GRPO,
                    DiffusionAdvantageEstimator.DANCE_GRPO,
                }:
                    adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo
                    adv_kwargs["global_std"] = global_std
                advantages, _ = adv_estimator_fn(**adv_kwargs)
                masked_advantages[active] = advantages
            chunk_advantages.append(masked_advantages)
        return torch.stack(chunk_advantages, dim=-1)

    if chunked_action:
        data.batch["action_advantages"] = _compute_chunk_advantages(
            action_rewards,
            action_masks,
            credit_name="action",
        )
        if action_masks is not None:
            data.batch["action_loss_weights"] = _active_loss_weights(action_masks)
        if action_stop_rewards is not None:
            data.batch["action_stop_advantages"] = _compute_chunk_advantages(
                action_stop_rewards,
                action_stop_masks,
                credit_name="stop",
            )
            if action_stop_masks is not None:
                data.batch["action_stop_loss_weights"] = _active_loss_weights(action_stop_masks)
        if action_collision_rewards is not None:
            data.batch["action_collision_advantages"] = _compute_chunk_advantages(
                action_collision_rewards,
                action_collision_masks,
                credit_name="collision",
            )
            if action_collision_masks is not None:
                data.batch["action_collision_loss_weights"] = _active_loss_weights(action_collision_masks)
    else:
        data.batch["sample_level_rewards"] = action_rewards.expand(-1, reward_steps)
        compute_advantage(
            data,
            adv_estimator,
            norm_adv_by_std_in_grpo,
            global_std,
            config,
            index_override=index_override,
        )
        data.batch["action_advantages"] = data.batch["advantages"].clone()
    if action_terminal_safety_rewards is not None:
        # One collision/clearance outcome belongs to the jointly generated
        # 32-action plan.  Normalize once inside the same prompt/stratum group
        # as navigation, then let the actor broadcast it to every temporal
        # chunk.  It deliberately has no post-collision mask.
        data.batch["sample_level_rewards"] = action_terminal_safety_rewards.expand(-1, reward_steps)
        compute_advantage(
            data,
            adv_estimator,
            norm_adv_by_std_in_grpo,
            global_std,
            config,
            index_override=index_override,
        )
        data.batch["action_terminal_safety_advantages"] = data.batch["advantages"].clone()
    data.batch["advantages"] = visual_advantages
    data.batch["returns"] = visual_advantages
    data.batch["sample_level_rewards"] = total_rewards
    return data


class BaseRayDiffusionTrainer(ABC):
    """Common Ray trainer infrastructure for diffusion training.

    Paradigm-specific trainers own the training loop while sharing worker
    initialization, validation, checkpointing, and logging behavior.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        if config.algorithm.sample_source == "online":
            assert self.hybrid_engine, "Currently, only support hybrid engine"
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )
        else:
            assert Role.Actor in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_training_reference_policy = need_reference_policy(self.config)
        validation_reference_kl = self.config.algorithm.get("validation_reference_kl", None)
        self.use_validation_reference_policy = bool(
            validation_reference_kl and validation_reference_kl.get("enabled", False)
        )
        self.use_reference_policy = self.use_training_reference_policy or self.use_validation_reference_policy

        self.use_rm = need_reward_model(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, create_rl_sampler, get_collate_fn

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            collate_fn = get_collate_fn(self.config.data)
        self._val_collate_fn = collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        self._hot_val_signature = None

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _maybe_reload_validation_dataloader(self) -> None:
        """Atomically switch held-out parquet files at validation boundaries."""

        control_path = os.environ.get("WAM_VAL_PATH_CONTROL_FILE", "").strip()
        if not control_path:
            return
        try:
            with open(control_path, encoding="utf-8") as handle:
                raw = handle.read().strip()
        except FileNotFoundError:
            return
        except OSError as exc:
            print(f"WARNING: cannot read validation path control file {control_path}: {exc}")
            return
        # An empty file intentionally means "keep the startup validation set".
        if not raw:
            return

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]

        max_samples = int(self.config.data.get("val_max_samples", -1))
        if isinstance(parsed, dict):
            requested = parsed.get("files", parsed.get("path"))
            max_samples = int(parsed.get("max_samples", max_samples))
        else:
            requested = parsed
        if isinstance(requested, str):
            requested = [requested]
        if (
            not isinstance(requested, list)
            or not requested
            or not all(isinstance(item, str) and item.strip() for item in requested)
        ):
            print(
                "WARNING: validation control must contain a path, a JSON path list, "
                f'or {{"files": [...], "max_samples": N}}; got {raw!r}'
            )
            return

        resolved_files: list[str] = []
        for item in requested:
            path = os.path.expanduser(os.path.expandvars(item.strip()))
            if os.path.isdir(path):
                matches = sorted(glob.glob(os.path.join(path, "*.parquet")))
            elif glob.has_magic(path):
                matches = sorted(glob.glob(path))
            elif os.path.isfile(path):
                matches = [path]
            else:
                print(f"WARNING: hot validation path does not exist; keeping current set: {path}")
                return
            if not matches:
                print(f"WARNING: hot validation path contains no parquet files: {path}")
                return
            resolved_files.extend(os.path.realpath(match) for match in matches)

        signature = (tuple(resolved_files), max_samples)
        if signature == self._hot_val_signature:
            return

        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset

        try:
            new_dataset = create_rl_dataset(
                resolved_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=max_samples,
            )
            if len(new_dataset) < 1:
                raise ValueError("hot validation dataset is empty")
            val_batch_size = self.config.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(new_dataset)
            new_dataloader = StatefulDataLoader(
                dataset=new_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data["dataloader_num_workers"],
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=self._val_collate_fn,
            )
        except Exception as exc:
            print(
                "WARNING: failed to build hot validation dataset; keeping current set: "
                f"files={resolved_files}, error={exc}"
            )
            return

        old_dataset = self.val_dataset
        old_dataloader = self.val_dataloader
        self.val_dataset = new_dataset
        self.val_dataloader = new_dataloader
        self._hot_val_signature = signature
        del old_dataset, old_dataloader
        gc.collect()
        print(
            "Hot-switched validation set: "
            f"files={len(resolved_files)}, samples={len(new_dataset)}, batches={len(new_dataloader)}, "
            f"max_samples={max_samples}, control={control_path}"
        )

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)

        visual_folder = os.path.join(dump_path, f"{self.global_steps}")
        os.makedirs(visual_folder, exist_ok=True)

        output_paths = []
        images_pil = outputs.cpu().float().permute(0, 2, 3, 1).numpy()
        images_pil = (images_pil * 255).round().clip(0, 255).astype("uint8")
        for i, image in enumerate(images_pil):
            image_path = os.path.join(visual_folder, f"{i}.jpg")
            Image.fromarray(image).save(image_path)
            output_paths.append(image_path)

        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": output_paths,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = batch.batch["responses"]
            scores = batch.batch["sample_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        if "wandb" in self.config.trainer.logger:
            import wandb

            outputs = [wandb.Image(image.float(), file_type="jpg") for image in outputs]
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self):
        self._maybe_reload_validation_dataloader()
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        replay_validation_metrics: dict[str, list[float]] = defaultdict(list)
        validation_sampler_contract: dict[str, float] = {}
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        media_retention_limit = max(
            int(self.config.trainer.log_val_generations),
            int(self.config.trainer.get("validation_max_media_samples", 0)),
            1 if val_data_dir else 0,
        )

        # Keep validation metrics for the complete validation set, but retain
        # decoded RGB only for the small number of samples that will actually
        # be logged. A 33x3x160x320 float32 response is about 19.3 MiB; retaining
        # 3000 prompts x 8 rollouts previously grew the driver to 439 GiB.
        media_inputs = []
        media_outputs = []
        media_gts = []
        media_scores = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            test_batch.non_tensor_batch["_rollout_seed_global_idx"] = np.arange(len(test_batch), dtype=np.int64)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "recompute_log_prob": False,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            # Verify what the rollout server actually used, rather than merely
            # echoing val_kwargs.  These fields are persisted by the strict WAM
            # replay contract after the request has crossed every Ray/vLLM
            # boundary.
            val_algo = self.config.actor_rollout_ref.rollout.val_kwargs.algo
            val_pipeline = self.config.actor_rollout_ref.rollout.val_kwargs.pipeline
            expected_visual_noise = float(val_algo.noise_level)
            configured_action_noise = val_algo.get("action_noise_level", None)
            expected_action_noise = (
                expected_visual_noise if configured_action_noise is None else float(configured_action_noise)
            )
            expected_sampler_fields = {
                "visual_noise_level": ("noise_level", expected_visual_noise),
                "action_noise_level": ("action_noise_level", expected_action_noise),
                "num_inference_steps": (
                    "num_inference_steps",
                    float(val_pipeline.num_inference_steps),
                ),
                "true_cfg_scale": (
                    "true_cfg_scale",
                    float(val_pipeline.true_cfg_scale),
                ),
            }
            for metric_name, (batch_key, expected_value) in expected_sampler_fields.items():
                actual_value = _validated_rollout_scalar(
                    test_output_gen_batch.batch,
                    batch_key,
                    expected=expected_value,
                    batch_size=len(test_output_gen_batch),
                    context="validation",
                )
                previous = validation_sampler_contract.setdefault(metric_name, actual_value)
                if not math.isclose(previous, actual_value, rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(
                        f"Validation sampler field {metric_name!r} changed between batches: "
                        f"{previous} vs {actual_value}."
                    )

            print("validation generation end")

            # Store generated outputs
            output_images = test_output_gen_batch.batch["responses"]

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            validation_config = self.config.algorithm.get("rollout_log_prob_validation", None)
            validate_replay = bool(
                validation_config
                and validation_config.get("enabled", False)
                and validation_config.get("validate", False)
            )
            if validate_replay:
                if not hasattr(self, "_compute_old_log_prob"):
                    raise TypeError("Held-out rollout/replay validation is only supported by policy-gradient trainers.")
                if "rollout_log_probs" not in test_batch.batch:
                    raise KeyError("Held-out rollout/replay validation requires validation rollout log-probabilities.")
                self.checkpoint_manager.sleep_replicas()
                try:
                    # The CausalWan joint DiT is numerically batch-composition
                    # dependent in BF16: a rollout produced at B=1 cannot be
                    # replayed at actor B=8 and treated as an implementation
                    # parity check.  In particular, a small action-encoder BMM
                    # drift is amplified through 30 transformer blocks and by
                    # the lower action SDE variance.  Keep reward evaluation on
                    # the complete validation batch, but probe one real sample
                    # per DP rank, spread uniformly over each held-out batch,
                    # with the same local batch size as val.n=1.
                    # This is a contract check, not a second validation metric.
                    replay_dp_size = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
                    validation_rollout_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
                    replay_source = test_batch
                    if validation_rollout_n == 1 and len(replay_source) > replay_dp_size:
                        replay_indices = np.linspace(
                            0,
                            len(replay_source) - 1,
                            num=replay_dp_size,
                            dtype=np.int64,
                        )
                        replay_source = replay_source[replay_indices]
                    replay_batch, replay_pad_size = pad_dataproto_to_divisor(replay_source, replay_dp_size)
                    local_replay_size = len(replay_batch) // replay_dp_size
                    configured_micro = int(self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu)
                    if validation_rollout_n == 1:
                        replay_micro = 1
                    else:
                        replay_micro = max(
                            divisor
                            for divisor in range(1, min(configured_micro, local_replay_size) + 1)
                            if local_replay_size % divisor == 0
                        )
                    old_log_prob_padded, _ = self._compute_old_log_prob(
                        replay_batch,
                        micro_batch_size_per_gpu=replay_micro,
                    )
                    old_log_prob = unpad_dataproto(old_log_prob_padded, pad_size=replay_pad_size)
                    replay_source = replay_source.union(old_log_prob)
                    batch_replay_metrics = validate_rollout_replay_log_probs(
                        replay_source,
                        validation_config,
                        expected_global_step=self.global_steps,
                        visual_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                            "visual_log_prob_weight", 1.0
                        ),
                        action_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                            "action_log_prob_weight", 1.0
                        ),
                        clip_ratio=self.config.actor_rollout_ref.actor.diffusion_loss.clip_ratio,
                    )
                    if self.use_validation_reference_policy:
                        reference_kl_config = self.config.algorithm.get("validation_reference_kl")
                        reference_log_prob_padded = self._compute_ref_log_prob(
                            replay_batch,
                            micro_batch_size_per_gpu=int(reference_kl_config.get("micro_batch_size_per_gpu", 1)),
                        )
                        reference_log_prob = unpad_dataproto(
                            reference_log_prob_padded,
                            pad_size=replay_pad_size,
                        )
                        replay_source = replay_source.union(reference_log_prob)
                        batch_replay_metrics.update(
                            compute_validation_reference_kl_metrics(
                                replay_source,
                                visual_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                    "visual_log_prob_weight", 1.0
                                ),
                                action_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                    "action_log_prob_weight", 1.0
                                ),
                            )
                        )
                finally:
                    self.checkpoint_manager.update_weights(self.global_steps)
                for key, value in batch_replay_metrics.items():
                    replay_validation_metrics[key].append(float(value))

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            remaining_media = media_retention_limit - len(media_scores)
            if remaining_media > 0:
                keep = min(remaining_media, len(scores))
                media_inputs.extend(input_texts[:keep])
                # Clone after slicing: validation responses already live on CPU,
                # so `.cpu()` alone would keep a view backed by the complete
                # multi-GiB batch storage.
                media_outputs.append(_clone_validation_media_prefix(output_images, keep))
                media_gts.extend(ground_truths[:keep])
                media_scores.extend(scores[:keep])

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            # Drop the multi-GiB decoded response batch before requesting the
            # next validation batch. Periodic collection also releases cycles
            # held by DataProto/TensorDict wrappers.
            del (
                output_images,
                test_output_gen_batch,
                test_output_gen_batch_padded,
                test_gen_batch,
                test_gen_batch_padded,
                test_batch,
            )
            gc.collect()

        retained_outputs = torch.cat(media_outputs, dim=0) if media_outputs else None
        if retained_outputs is not None:
            self._maybe_log_val_generations(
                inputs=media_inputs,
                outputs=retained_outputs,
                scores=media_scores,
            )

        # dump generations
        if val_data_dir and retained_outputs is not None:
            self._dump_generations(
                inputs=media_inputs,
                outputs=retained_outputs,
                gts=media_gts,
                scores=media_scores,
                reward_extra_infos_dict={},
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        event_metric_dict = _event_validation_metrics(reward_extra_infos_dict)
        generic_reward_extra_infos = {
            key: value for key, value in reward_extra_infos_dict.items() if not key.startswith("event_val_")
        }
        metric_dict = self._val_metrics_update(
            data_sources,
            sample_uids,
            generic_reward_extra_infos,
            sample_turns,
        )
        metric_dict.update(event_metric_dict)
        metric_dict.update({f"val-sampler/{key}": value for key, value in validation_sampler_contract.items()})
        metric_dict["val-sampler/dance_sde"] = float(
            str(self.config.actor_rollout_ref.rollout.val_kwargs.algo.sde_type) == "dance_sde"
        )
        metric_dict["val-sampler/init_same_noise"] = float(
            bool(self.config.actor_rollout_ref.rollout.val_kwargs.algo.init_same_noise)
        )
        metric_dict["val-sampler/rollout_n"] = float(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        for key, values in replay_validation_metrics.items():
            reducer = np.sum if key == "reference_kl/sample_count" else np.mean
            metric_dict[f"val-aux/{key}"] = float(reducer(values))
        return metric_dict

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        actor_rollout_resource_pool = self._init_colocated_workers()
        if self.config.algorithm.sample_source == "offline":
            return
        self._init_online_rollout_stack(actor_rollout_resource_pool)

    def _init_colocated_workers(self):
        """Create Ray pools and colocated actor/ref worker groups (online and offline)."""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout (offline uses Role.Actor only; online uses hybrid actor_rollout roles)
        if Role.Actor in self.role_worker_mapping:
            actor_role = Role.Actor
        elif Role.ActorRolloutRef in self.role_worker_mapping:
            actor_role = Role.ActorRolloutRef
        else:
            actor_role = Role.ActorRollout
        if self.hybrid_engine or actor_role == Role.Actor:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        # Forward profiling steps and (when nsys is selected) per-worker Nsight options to the
        # Ray worker group so that workers can be launched under nsys with the right capture range.
        if OmegaConf.select(self.config, "global_profiler.steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                worker_nsight_options = OmegaConf.select(
                    self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                )
                assert worker_nsight_options is not None, (
                    "global_profiler.global_tool_config.nsys.worker_nsight_options must be set "
                    "when using nsys with global_profiler.steps"
                )
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(worker_nsight_options)
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        return actor_rollout_resource_pool

    def _init_online_rollout_stack(self, actor_rollout_resource_pool):
        """Initialize rollout, reward, and checkpoint engines (online sampling only)."""
        # create reward loop manager
        from verl_omni.reward_loop import OmniRewardLoopManager

        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = OmniRewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

            from verl_omni.agent_loop import DiffusionAgentLoopWorker

            AgentLoopManager.agent_loop_workers_class = ray.remote(DiffusionAgentLoopWorker)

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        self.enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = (
            self.reward_loop_manager.reward_loop_workers if self.enable_agent_reward_loop else None
        )

        self.llm_server_manager = StaggeredLLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self, *, register_actor_retention: bool = True):
        """Save a full checkpoint, optionally excluding it from actor rotation.

        Periodic validation guards are transactional recovery points and are
        deleted after the next successful training step.  They must not enter
        the actor checkpoint manager's retention queue, otherwise frequent
        validation saves rotate out durable ``save_freq`` checkpoints.
        """
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        if not register_actor_retention:
            max_actor_ckpt_to_keep = None
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _checkpoint_folder_for_step(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{int(step)}")

    def _assert_full_checkpoint_complete(self, checkpoint_folder: str) -> None:
        """Fail closed unless actor, optimizer and dataloader state all landed."""

        actor_folder = os.path.join(checkpoint_folder, "actor")
        expected_ranks = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
        model_shards = glob.glob(os.path.join(actor_folder, "model_world_size_*_rank_*.pt"))
        optimizer_shards = glob.glob(os.path.join(actor_folder, "optim_world_size_*_rank_*.pt"))
        extra_state_shards = glob.glob(os.path.join(actor_folder, "extra_state_world_size_*_rank_*.pt"))
        data_state = os.path.join(checkpoint_folder, "data.pt")
        missing = []
        for name, paths in (
            ("model", model_shards),
            ("optimizer", optimizer_shards),
            ("extra_state", extra_state_shards),
        ):
            if len(paths) != expected_ranks or any(os.path.getsize(path) <= 0 for path in paths):
                missing.append(f"{name}={len(paths)}/{expected_ranks}")
        if not os.path.isfile(data_state) or os.path.getsize(data_state) <= 0:
            missing.append("data_state=missing")
        if missing:
            raise RuntimeError(
                "Refusing validation because the pre-validation checkpoint is incomplete: "
                f"folder={checkpoint_folder}, {', '.join(missing)}"
            )

    def _prepare_checkpoint_before_validation(self, *, checkpoint_saved_this_step: bool) -> None:
        """Persist a recoverable transaction boundary before periodic validation."""

        if self.global_steps <= 0 or not self.config.trainer.get("validation_checkpoint_guard_enabled", False):
            return
        checkpoint_folder = self._checkpoint_folder_for_step(self.global_steps)
        if not checkpoint_saved_this_step:
            print(
                "Saving temporary full checkpoint before validation: "
                f"step={self.global_steps}, folder={checkpoint_folder}"
            )
            self._save_checkpoint(register_actor_retention=False)
        self._assert_full_checkpoint_complete(checkpoint_folder)

        if not checkpoint_saved_this_step:
            marker = os.path.join(checkpoint_folder, ".validation_guard_complete.json")
            temporary_marker = f"{marker}.tmp.{os.getpid()}"
            with open(temporary_marker, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "global_step": int(self.global_steps),
                        "checkpoint_folder": checkpoint_folder,
                        "contains_optimizer": True,
                        "validation_started": False,
                    },
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_marker, marker)
            self._pending_validation_guard_path = checkpoint_folder
            print(f"Pre-validation checkpoint committed: {marker}")
        else:
            print(
                "Using scheduled full checkpoint as validation guard: "
                f"step={self.global_steps}, folder={checkpoint_folder}"
            )

    def _discover_pending_validation_guard(self) -> None:
        self._pending_validation_guard_path = None
        if self.global_steps <= 0:
            return
        candidate = self._checkpoint_folder_for_step(self.global_steps)
        marker = os.path.join(candidate, ".validation_guard_complete.json")
        if os.path.isfile(marker):
            self._pending_validation_guard_path = candidate
            print(f"Recovered pending validation guard checkpoint: {candidate}")

    def _cleanup_pending_validation_guard(self) -> None:
        """Delete only a marked temporary guard after one full train step succeeds."""

        checkpoint_folder = getattr(self, "_pending_validation_guard_path", None)
        if not checkpoint_folder:
            return
        root = os.path.realpath(self.config.trainer.default_local_dir)
        target = os.path.realpath(checkpoint_folder)
        marker = os.path.join(target, ".validation_guard_complete.json")
        if os.path.dirname(target) != root or not os.path.basename(target).startswith("global_step_"):
            raise RuntimeError(f"Refusing unsafe validation-guard cleanup target: {target}")
        if not os.path.isfile(marker):
            raise RuntimeError(f"Refusing to delete unmarked checkpoint as validation guard: {target}")

        shutil.rmtree(target)
        self._pending_validation_guard_path = None

        remaining_steps = []
        for path in glob.glob(os.path.join(root, "global_step_*")):
            if not os.path.isdir(path):
                continue
            try:
                step = int(os.path.basename(path).split("global_step_", 1)[1])
            except (IndexError, ValueError):
                continue
            remaining_steps.append(step)
        latest_path = os.path.join(root, "latest_checkpointed_iteration.txt")
        if remaining_steps:
            temporary_latest = f"{latest_path}.tmp.{os.getpid()}"
            with open(temporary_latest, "w", encoding="utf-8") as handle:
                handle.write(str(max(remaining_steps)))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_latest, latest_path)
        elif os.path.exists(latest_path):
            os.unlink(latest_path)
        print(f"Deleted temporary validation guard after a complete subsequent training step: {target}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        # load dataloader,
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        load_dataloader_state = bool(self.config.trainer.get("resume_load_dataloader_state", True))
        if load_dataloader_state and os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        elif not load_dataloader_state:
            print(
                "Skipping checkpoint dataloader state by configuration; "
                "the resumed actor/optimizer will start at the beginning of the new dataset."
            )
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # update actor
        architecture = self.config.actor_rollout_ref.model.architecture
        if architecture in {"WNM2D", "WNM3D"}:
            batch_td, total_bytes, kept_bytes = _prepare_wnm_actor_tensordict(
                batch,
                expected_text_len=int(rollout_config.pipeline.max_sequence_length),
                architecture=architecture,
            )
            if not getattr(self, "_wnm_actor_transport_logged", False):
                sys_logger.warning(
                    "WNM actor transport preserves fixed padded prompt semantics and prunes "
                    "reward/rollout-only tensors: total=%.3f GiB kept=%.3f GiB dropped=%.3f GiB",
                    total_bytes / 2**30,
                    kept_bytes / 2**30,
                    (total_bytes - kept_bytes) / 2**30,
                )
                self._wnm_actor_transport_logged = True
        else:
            batch_td = batch.to_tensordict()
            # Generic diffusion models may use ragged context to reduce work.
            batch_td = embeds_padding_2_no_padding(batch_td)
        # ``ppo_mini_batch_size`` counts prompt groups. The actor consumes all
        # P * N current-policy trajectories sampled for those groups.
        ppo_mini_batch_size = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        )
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            height=self.config.actor_rollout_ref.model.pipeline.height,
            width=self.config.actor_rollout_ref.model.pipeline.width,
            vae_scale_factor=self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if (actor_mfu := actor_output.pop("actor/mfu", None)) is not None:
            actor_output["perf/mfu/actor"] = actor_mfu
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy and not self.ref_in_actor:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy and not self.ref_in_actor:
                self.ref_policy_wg.stop_profile()

    @abstractmethod
    def fit(self):
        """Run the trainer-type-specific training loop."""
        pass


class PolicyGradientRayTrainer(BaseRayDiffusionTrainer):
    """Policy-gradient diffusion trainer for FlowGRPO, MixGRPO, DanceGRPO, GRPO-Guard, etc."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.algorithm.get("sample_source", "online") == "online":
            self._validate_online_log_prob_config()

    def _validate_online_log_prob_config(self) -> None:
        validation_config = self.config.algorithm.get("rollout_log_prob_validation", None)
        reference_kl_config = self.config.algorithm.get("validation_reference_kl", None)
        if (
            reference_kl_config
            and reference_kl_config.get("enabled", False)
            and not (
                validation_config
                and validation_config.get("enabled", False)
                and validation_config.get("validate", False)
            )
        ):
            raise ValueError(
                "algorithm.validation_reference_kl.enabled=true requires "
                "rollout_log_prob_validation.enabled=true and validate=true so "
                "current/reference policies replay the same held-out trajectories."
            )
        if validation_config is None or not validation_config.get("enabled", False):
            return
        if not self.config.actor_rollout_ref.rollout.calculate_log_probs:
            raise ValueError(
                "algorithm.rollout_log_prob_validation.enabled=true requires "
                "actor_rollout_ref.rollout.calculate_log_probs=true."
            )
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config and rollout_corr_config.get("bypass_mode", False):
            if not validation_config.get("validate", False):
                raise ValueError(
                    "Rollout/replay validation with rollout_correction.bypass_mode=true requires "
                    "rollout_log_prob_validation.validate=true so held-out validation still performs "
                    "a real actor replay."
                )
            sys_logger.warning(
                "Rollout log-prob bypass is enabled: skipping the tautological per-step rollout/replay "
                "comparison; held-out validation will still recompute actor log-probabilities."
            )
        if validation_config.get("validate", False):
            val_algo = self.config.actor_rollout_ref.rollout.val_kwargs.algo
            val_noise_level = float(val_algo.noise_level)
            action_noise_level = val_algo.get("action_noise_level", None)
            action_noise_level = val_noise_level if action_noise_level is None else float(action_noise_level)
            if val_noise_level <= 0 or action_noise_level <= 0:
                raise ValueError(
                    "Held-out rollout/replay likelihood diagnostics require positive validation visual/action "
                    "noise levels."
                )

    def _compute_ref_log_prob(
        self,
        batch: DataProto,
        *,
        micro_batch_size_per_gpu: Optional[int] = None,
    ) -> DataProto:
        batch_td = batch.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        metadata = {
            "compute_loss": False,
            "height": self.config.actor_rollout_ref.model.pipeline.height,
            "width": self.config.actor_rollout_ref.model.pipeline.width,
            "vae_scale_factor": self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        if micro_batch_size_per_gpu is not None:
            if int(micro_batch_size_per_gpu) <= 0:
                raise ValueError("micro_batch_size_per_gpu must be positive")
            metadata["micro_batch_size_per_gpu"] = int(micro_batch_size_per_gpu)
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        else:
            output = self.ref_policy_wg.infer_ref_batch(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        prev_sample_mean = tu.get(output, "prev_sample_mean")
        ref_log_prob = tu.get_tensordict(
            {"ref_log_prob": log_probs.float(), "ref_prev_sample_mean": prev_sample_mean.float()}
        )
        action_log_probs = output.get("action_log_probs", None)
        if action_log_probs is not None:
            ref_log_prob["ref_action_log_prob"] = action_log_probs.float()
        action_prev_sample_mean = output.get("action_prev_sample_mean", None)
        if action_prev_sample_mean is not None:
            ref_log_prob["ref_action_prev_sample_mean"] = action_prev_sample_mean.float()
        return DataProto.from_tensordict(ref_log_prob)

    def _compute_old_log_prob(
        self,
        batch: DataProto,
        *,
        micro_batch_size_per_gpu: Optional[int] = None,
    ) -> tuple[DataProto, Optional[float]]:
        batch_td = batch.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        metadata = {
            "compute_loss": False,
            "height": self.config.actor_rollout_ref.model.pipeline.height,
            "width": self.config.actor_rollout_ref.model.pipeline.width,
            "vae_scale_factor": self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        }
        if micro_batch_size_per_gpu is not None:
            if int(micro_batch_size_per_gpu) <= 0:
                raise ValueError("micro_batch_size_per_gpu must be positive")
            metadata["micro_batch_size_per_gpu"] = int(micro_batch_size_per_gpu)
        tu.assign_non_tensor(batch_td, **metadata)
        output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        log_probs = tu.get(output, "log_probs")
        old_log_prob_dict = {"old_log_probs": log_probs.float()}
        action_log_probs = output.get("action_log_probs", None)
        if action_log_probs is not None:
            old_log_prob_dict["old_action_log_probs"] = action_log_probs.float()
        prev_sample_mean = tu.get(output, "prev_sample_mean")
        if prev_sample_mean is not None:
            old_log_prob_dict["old_prev_sample_mean"] = prev_sample_mean.float()
        action_prev_sample_mean = output.get("action_prev_sample_mean", None)
        if action_prev_sample_mean is not None:
            old_log_prob_dict["old_action_prev_sample_mean"] = action_prev_sample_mean.float()
        old_log_prob = tu.get_tensordict(old_log_prob_dict)
        old_log_prob_mfu = tu.get(output, "metrics").get("mfu")
        return DataProto.from_tensordict(old_log_prob), old_log_prob_mfu

    def fit(self):
        """
        The training loop of FlowGRPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._discover_pending_validation_guard()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # Profiler step state machine. Mirrors verl/trainer/ppo/ray_trainer.py.
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_seed_cfg = self.config.actor_rollout_ref.rollout.get("seed")
                if rollout_seed_cfg is not None:
                    gen_batch.meta_info["rollout_seed"] = int(rollout_seed_cfg) + self.global_steps - 1
                rollout_sampling_overrides = {}
                scheduled_noise = _scheduled_rollout_noise_levels(self.config, global_step=self.global_steps)
                if scheduled_noise is not None:
                    rollout_sampling_overrides.update(
                        noise_level=scheduled_noise["noise_level"],
                        action_noise_level=scheduled_noise["action_noise_level"],
                    )
                    metrics["rollout_noise/visual_level"] = scheduled_noise["noise_level"]
                    metrics["rollout_noise/action_level"] = scheduled_noise["action_noise_level"]
                    metrics["rollout_noise/schedule_progress"] = scheduled_noise["progress"]
                if rollout_sampling_overrides:
                    gen_batch.meta_info["rollout_sampling_overrides"] = rollout_sampling_overrides
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                if rollout_sampling_overrides:
                    # DataProto.meta_info is driver-local metadata and is not a
                    # reliable transport for per-request sampling controls once
                    # the repeated batch is split across Ray agent workers.
                    # Carry an explicit object row instead; the worker consumes
                    # and removes it before dataset kwargs/reward packaging.
                    per_request_overrides = np.empty(len(gen_batch_output), dtype=object)
                    per_request_overrides[:] = [dict(rollout_sampling_overrides) for _ in range(len(gen_batch_output))]
                    gen_batch_output.non_tensor_batch["_rollout_sampling_overrides"] = per_request_overrides
                gen_batch_output.non_tensor_batch["_rollout_seed_global_idx"] = np.arange(
                    len(gen_batch_output), dtype=np.int64
                )
                layer_credit_enabled, branches_per_stratum = _layer_conditioned_credit_config(self.config)
                full_replay_enabled, expected_replay_transitions = _full_transition_actor_replay_config(self.config)
                if layer_credit_enabled and full_replay_enabled:
                    raise ValueError(
                        "Full-transition actor replay is incompatible with layer-conditioned "
                        "counterfactual compaction; disable algorithm.layer_conditioned_credit."
                    )
                selected_credit_transitions = None
                if layer_credit_enabled:
                    selected_credit_transitions = _assign_layer_conditioned_credit(
                        gen_batch_output,
                        rollout_n=int(self.config.actor_rollout_ref.rollout.n),
                        branches_per_stratum=branches_per_stratum,
                        global_step=self.global_steps,
                        seed=int(self.config.data.seed),
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Generate P * N current-policy trajectories in vLLM-Omni.
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                            # streaming reward scores inside the gen window; colocate in the reward phase
                            if self.enable_agent_reward_loop:
                                self.reward_loop_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()
                            if self.enable_agent_reward_loop:
                                self.reward_loop_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # Confirm the server-side sampler contract on every train
                    # step.  In particular, this catches a lost per-request
                    # action-noise override instead of allowing actor replay to
                    # train against a different behavior policy silently.
                    rollout_algo = self.config.actor_rollout_ref.rollout.algo
                    requested_visual_noise = (
                        float(scheduled_noise["noise_level"])
                        if scheduled_noise is not None
                        else float(rollout_algo.noise_level)
                    )
                    configured_action_noise = rollout_algo.get("action_noise_level", None)
                    requested_action_noise = (
                        float(scheduled_noise["action_noise_level"])
                        if scheduled_noise is not None
                        else (
                            requested_visual_noise
                            if configured_action_noise is None
                            else float(configured_action_noise)
                        )
                    )
                    metrics["rollout_noise/visual_level"] = _validated_rollout_scalar(
                        gen_batch_output.batch,
                        "noise_level",
                        expected=requested_visual_noise,
                        batch_size=len(gen_batch_output),
                        context="training",
                    )
                    metrics["rollout_noise/action_level"] = _validated_rollout_scalar(
                        gen_batch_output.batch,
                        "action_noise_level",
                        expected=requested_action_noise,
                        batch_size=len(gen_batch_output),
                        context="training",
                    )
                    metrics["rollout_noise/server_contract_match"] = 1.0

                    # Repeat static parquet inputs to align them with P * N rollouts.
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if "rm_scores" not in batch.batch.keys():
                            if not self.use_rm:
                                raise RuntimeError(
                                    "The streaming reward loop did not attach rm_scores to the rollout batch."
                                )
                            if curr_step_profile:
                                self.reward_loop_manager.start_profile()
                            batch_reward = self._compute_reward_colocate(batch)
                            if curr_step_profile:
                                self.reward_loop_manager.stop_profile()
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Bypass mode: skip old_log_prob recompute (2 policies).
                    # Decoupled mode: recompute old_log_probs as proximal anchor (3 policies).
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = bool(
                        rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    )
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        apply_bypass_mode_to_diffusion_batch(batch)
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            if old_log_prob_mfu is not None:
                                metrics.update({"perf/mfu/actor_infer": old_log_prob_mfu})
                            batch = batch.union(old_log_prob)

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
                    if "actions" in batch.batch or "all_action_latents" in batch.batch:
                        if "old_action_log_probs" not in batch.batch:
                            raise KeyError(
                                "WAM actor update requires old_action_log_probs alongside old visual log-probs."
                            )
                        visual_shape = tuple(batch.batch["old_log_probs"].shape)
                        action_shape = tuple(batch.batch["old_action_log_probs"].shape)
                        valid_action_shape = action_shape == visual_shape
                        if action_chunk_credit_enabled():
                            expected_chunks = len(action_chunk_weights())
                            valid_action_shape = (
                                len(action_shape) == len(visual_shape) + 1
                                and action_shape[:-1] == visual_shape
                                and action_shape[-1] == expected_chunks
                            )
                        if not valid_action_shape:
                            raise ValueError(
                                "WAM action old log-probs must match visual [B,T], or "
                                "use [B,T,C] under chunk credit; "
                                f"visual={visual_shape}, action={action_shape}."
                            )
                    validation_config = self.config.algorithm.get("rollout_log_prob_validation", None)
                    if (
                        validation_config
                        and validation_config.get("enabled", False)
                        and not bypass_recomputing_logprobs
                    ):
                        with marked_timer("rollout_replay_validation", timing_raw, color="cyan"):
                            metrics.update(
                                validate_rollout_replay_log_probs(
                                    batch,
                                    validation_config,
                                    # The rollout samples the policy produced by
                                    # the previous completed optimizer step. The
                                    # current step is synchronized only after
                                    # `_update_actor` below.
                                    expected_global_step=self.global_steps - 1,
                                    visual_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                        "visual_log_prob_weight", 1.0
                                    ),
                                    action_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                        "action_log_prob_weight", 1.0
                                    ),
                                    clip_ratio=self.config.actor_rollout_ref.actor.diffusion_loss.clip_ratio,
                                )
                            )

                    # Decoupled-mode rollout correction (old vs rollout).
                    # In bypass mode old == rollout, so correction runs per-step in ``diffusion_loss``.
                    if not bypass_recomputing_logprobs and rollout_correction_enabled(rollout_corr_config):
                        with marked_timer("rollout_corr", timing_raw, color="cyan"):
                            batch, rollout_corr_metrics = apply_rollout_correction_to_diffusion_batch(
                                batch,
                                rollout_corr_config,
                                visual_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                    "visual_log_prob_weight", 1.0
                                ),
                                action_log_prob_weight=self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                    "action_log_prob_weight", 1.0
                                ),
                            )
                            metrics.update(rollout_corr_metrics)

                    if self.use_training_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["sample_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        num_timesteps = batch.batch["old_log_probs"].shape[1]
                        batch.batch["sample_level_rewards"] = batch.batch["sample_level_scores"].expand(
                            -1, num_timesteps
                        )

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            global_std=self.config.algorithm.global_std,
                            config=self.config.algorithm,
                        )
                        if "actions" in batch.batch or "all_action_latents" in batch.batch:
                            missing = [
                                name
                                for name in ("visual_reward", "action_reward")
                                if name not in reward_extra_infos_dict
                            ]
                            if missing:
                                raise KeyError(
                                    "WAM scheme-two training requires reward outputs visual_reward and "
                                    f"action_reward; missing {missing}."
                                )
                            reward_device = reward_tensor.device
                            reward_dtype = reward_tensor.dtype
                            visual_rewards = torch.as_tensor(
                                reward_extra_infos_dict["visual_reward"], device=reward_device, dtype=reward_dtype
                            ).reshape(-1, 1)
                            action_report_rewards = torch.as_tensor(
                                reward_extra_infos_dict["action_reward"], device=reward_device, dtype=reward_dtype
                            ).reshape(-1, 1)
                            terminal_safety_flags = reward_extra_infos_dict.get("action_terminal_safety_enabled", None)
                            action_terminal_safety_rewards = None
                            if terminal_safety_flags is not None:
                                terminal_safety_flags = np.asarray(terminal_safety_flags, dtype=np.float32).reshape(-1)
                                terminal_safety_enabled_mask = terminal_safety_flags > 0.5
                                if np.any(terminal_safety_enabled_mask) and not np.all(terminal_safety_enabled_mask):
                                    raise ValueError(
                                        "A rollout batch cannot mix enabled and disabled terminal-safety samples."
                                    )
                                if np.all(terminal_safety_enabled_mask):
                                    terminal_reward_key = "action_terminal_safety_reward"
                                    if terminal_reward_key not in reward_extra_infos_dict:
                                        raise KeyError(
                                            f"Enabled terminal safety requires reward field {terminal_reward_key}."
                                        )
                                    action_terminal_safety_rewards = torch.as_tensor(
                                        reward_extra_infos_dict[terminal_reward_key],
                                        device=reward_device,
                                        dtype=reward_dtype,
                                    ).reshape(-1, 1)
                            if action_chunk_credit_enabled():
                                chunk_count = len(action_chunk_weights())
                                stop_credit_flags = reward_extra_infos_dict.get("action_stop_credit_enabled", None)
                                stop_credit_enabled = False
                                if stop_credit_flags is not None:
                                    stop_credit_flags = np.asarray(stop_credit_flags, dtype=np.float32).reshape(-1)
                                    enabled_mask = stop_credit_flags > 0.5
                                    if np.any(enabled_mask) and not np.all(enabled_mask):
                                        raise ValueError(
                                            "A rollout batch cannot mix enabled and disabled stop-credit samples."
                                        )
                                    stop_credit_enabled = bool(np.all(enabled_mask))
                                stop_loss_weight = float(
                                    self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                        "action_stop_loss_weight", 0.0
                                    )
                                )
                                if stop_loss_weight > 0.0 and not stop_credit_enabled:
                                    raise ValueError(
                                        "Positive action_stop_loss_weight requires reward "
                                        "field action_stop_credit_enabled=1 for every sample. "
                                        "Check reward metric compaction before actor replay."
                                    )
                                collision_credit_flags = reward_extra_infos_dict.get(
                                    "action_collision_credit_enabled", None
                                )
                                collision_credit_enabled = False
                                if collision_credit_flags is not None:
                                    collision_credit_flags = np.asarray(
                                        collision_credit_flags, dtype=np.float32
                                    ).reshape(-1)
                                    collision_enabled_mask = collision_credit_flags > 0.5
                                    if np.any(collision_enabled_mask) and not np.all(collision_enabled_mask):
                                        raise ValueError(
                                            "A rollout batch cannot mix enabled and disabled collision-credit samples."
                                        )
                                    collision_credit_enabled = bool(np.all(collision_enabled_mask))
                                collision_loss_weight = float(
                                    self.config.actor_rollout_ref.actor.diffusion_loss.get(
                                        "action_collision_loss_weight", 0.0
                                    )
                                )
                                if collision_loss_weight > 0.0 and not collision_credit_enabled:
                                    raise ValueError(
                                        "Positive action_collision_loss_weight requires "
                                        "reward field action_collision_credit_enabled=1 "
                                        "for every sample. Check reward metric compaction "
                                        "before actor replay."
                                    )
                                separate_action_streams_enabled = bool(stop_credit_enabled or collision_credit_enabled)
                                chunk_keys = [
                                    (
                                        f"action_chunk_{index}_nav_reward"
                                        if separate_action_streams_enabled
                                        else f"action_chunk_{index}_reward"
                                    )
                                    for index in range(chunk_count)
                                ]
                                missing_chunk_keys = [key for key in chunk_keys if key not in reward_extra_infos_dict]
                                if missing_chunk_keys:
                                    raise KeyError(
                                        "Chunk action credit requires one reward field per "
                                        f"temporal chunk; missing {missing_chunk_keys}."
                                    )
                                action_rewards = torch.stack(
                                    [
                                        torch.as_tensor(
                                            reward_extra_infos_dict[key],
                                            device=reward_device,
                                            dtype=reward_dtype,
                                        ).reshape(-1)
                                        for key in chunk_keys
                                    ],
                                    dim=1,
                                )
                                chunk_mask_keys = [
                                    (
                                        f"action_chunk_{index}_nav_active"
                                        if separate_action_streams_enabled
                                        else f"action_chunk_{index}_active"
                                    )
                                    for index in range(chunk_count)
                                ]
                                present_chunk_mask_keys = [
                                    key for key in chunk_mask_keys if key in reward_extra_infos_dict
                                ]
                                if present_chunk_mask_keys and len(present_chunk_mask_keys) != len(chunk_mask_keys):
                                    missing_chunk_mask_keys = [
                                        key for key in chunk_mask_keys if key not in reward_extra_infos_dict
                                    ]
                                    raise KeyError(
                                        "Chunk action masks must be emitted for every "
                                        f"temporal chunk; missing {missing_chunk_mask_keys}."
                                    )
                                action_masks = (
                                    torch.stack(
                                        [
                                            torch.as_tensor(
                                                reward_extra_infos_dict[key],
                                                device=reward_device,
                                                dtype=reward_dtype,
                                            ).reshape(-1)
                                            for key in chunk_mask_keys
                                        ],
                                        dim=1,
                                    )
                                    if present_chunk_mask_keys
                                    else None
                                )
                                if stop_credit_enabled:
                                    stop_reward_keys = [
                                        f"action_chunk_{index}_stop_reward" for index in range(chunk_count)
                                    ]
                                    stop_mask_keys = [
                                        f"action_chunk_{index}_stop_active" for index in range(chunk_count)
                                    ]
                                    missing_stop_keys = [
                                        key
                                        for key in (*stop_reward_keys, *stop_mask_keys)
                                        if key not in reward_extra_infos_dict
                                    ]
                                    if missing_stop_keys:
                                        raise KeyError(
                                            "Stop credit requires reward and mask fields "
                                            f"for every chunk; missing {missing_stop_keys}."
                                        )
                                    action_stop_rewards = torch.stack(
                                        [
                                            torch.as_tensor(
                                                reward_extra_infos_dict[key],
                                                device=reward_device,
                                                dtype=reward_dtype,
                                            ).reshape(-1)
                                            for key in stop_reward_keys
                                        ],
                                        dim=1,
                                    )
                                    action_stop_masks = torch.stack(
                                        [
                                            torch.as_tensor(
                                                reward_extra_infos_dict[key],
                                                device=reward_device,
                                                dtype=reward_dtype,
                                            ).reshape(-1)
                                            for key in stop_mask_keys
                                        ],
                                        dim=1,
                                    )
                                else:
                                    action_stop_rewards = None
                                    action_stop_masks = None
                                if collision_credit_enabled:
                                    collision_reward_keys = [
                                        f"action_chunk_{index}_collision_reward" for index in range(chunk_count)
                                    ]
                                    collision_mask_keys = [
                                        f"action_chunk_{index}_collision_active" for index in range(chunk_count)
                                    ]
                                    missing_collision_keys = [
                                        key
                                        for key in (
                                            *collision_reward_keys,
                                            *collision_mask_keys,
                                        )
                                        if key not in reward_extra_infos_dict
                                    ]
                                    if missing_collision_keys:
                                        raise KeyError(
                                            "Collision credit requires reward and mask "
                                            "fields for every chunk; missing "
                                            f"{missing_collision_keys}."
                                        )
                                    action_collision_rewards = torch.stack(
                                        [
                                            torch.as_tensor(
                                                reward_extra_infos_dict[key],
                                                device=reward_device,
                                                dtype=reward_dtype,
                                            ).reshape(-1)
                                            for key in collision_reward_keys
                                        ],
                                        dim=1,
                                    )
                                    action_collision_masks = torch.stack(
                                        [
                                            torch.as_tensor(
                                                reward_extra_infos_dict[key],
                                                device=reward_device,
                                                dtype=reward_dtype,
                                            ).reshape(-1)
                                            for key in collision_mask_keys
                                        ],
                                        dim=1,
                                    )
                                else:
                                    action_collision_rewards = None
                                    action_collision_masks = None
                            else:
                                action_rewards = action_report_rewards
                                action_masks = None
                                action_stop_rewards = None
                                action_stop_masks = None
                                action_collision_rewards = None
                                action_collision_masks = None
                            layer_group_index = _layer_conditioned_group_index(batch) if layer_credit_enabled else None
                            batch = compute_separate_wam_advantages(
                                batch,
                                visual_rewards=visual_rewards,
                                action_rewards=action_rewards,
                                action_masks=action_masks,
                                action_stop_rewards=action_stop_rewards,
                                action_stop_masks=action_stop_masks,
                                action_collision_rewards=action_collision_rewards,
                                action_collision_masks=action_collision_masks,
                                action_terminal_safety_rewards=(action_terminal_safety_rewards),
                                adv_estimator=self.config.algorithm.adv_estimator,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                # Layer-conditioned credit uses a separate conditional baseline and
                                # standard deviation for every (prompt, layer).
                                global_std=(False if layer_credit_enabled else self.config.algorithm.global_std),
                                config=self.config.algorithm,
                                index_override=layer_group_index,
                            )
                            if action_terminal_safety_rewards is not None:
                                terminal_advantages = batch.batch["action_terminal_safety_advantages"]
                                metrics.update(
                                    {
                                        "critic/terminal_safety/reward_mean": float(
                                            action_terminal_safety_rewards.mean().item()
                                        ),
                                        "critic/terminal_safety/reward_std": float(
                                            action_terminal_safety_rewards.std(unbiased=False).item()
                                        ),
                                        "critic/terminal_safety/advantage_mean": float(
                                            terminal_advantages.mean().item()
                                        ),
                                        "critic/terminal_safety/advantage_abs_mean": float(
                                            terminal_advantages.abs().mean().item()
                                        ),
                                        "critic/terminal_safety/nonzero_advantage_fraction": float(
                                            (terminal_advantages.abs() > 1e-8).float().mean().item()
                                        ),
                                    }
                                )
                            if layer_credit_enabled:
                                metrics.update(
                                    _layer_credit_metrics(
                                        visual_rewards=visual_rewards,
                                        action_rewards=action_report_rewards,
                                        strata=np.asarray(
                                            batch.non_tensor_batch["layer_credit_stratum"],
                                            dtype=np.int64,
                                        ),
                                        transitions_by_stratum=selected_credit_transitions,
                                    )
                                )
                            if action_chunk_credit_enabled():
                                metrics.update(
                                    _action_chunk_credit_metrics(
                                        action_rewards=action_rewards,
                                        action_advantages=batch.batch["action_advantages"],
                                        action_masks=action_masks,
                                    )
                                )
                                if action_stop_rewards is not None:
                                    metrics.update(
                                        _action_stop_credit_metrics(
                                            stop_rewards=action_stop_rewards,
                                            stop_advantages=batch.batch["action_stop_advantages"],
                                            stop_masks=action_stop_masks,
                                        )
                                    )
                                if action_collision_rewards is not None:
                                    metrics.update(
                                        _action_collision_credit_metrics(
                                            collision_rewards=(action_collision_rewards),
                                            collision_advantages=batch.batch["action_collision_advantages"],
                                            collision_masks=(action_collision_masks),
                                        )
                                    )

                        if layer_credit_enabled:
                            batch = _compact_wnm_layer_credit_batch(batch)
                        if full_replay_enabled:
                            old_log_probs = batch.batch.get("old_log_probs", None)
                            if not isinstance(old_log_probs, torch.Tensor) or old_log_probs.ndim < 2:
                                raise ValueError(
                                    "Full-transition actor replay requires a rank-2+ old_log_probs tensor."
                                )
                            actual_replay_transitions = int(old_log_probs.shape[1])
                            if actual_replay_transitions != expected_replay_transitions:
                                raise ValueError(
                                    "Full-transition actor replay contract failed: expected "
                                    f"{expected_replay_transitions} transitions, got "
                                    f"{actual_replay_transitions}."
                                )
                            metrics["actor/full_replay_transitions"] = float(actual_replay_transitions)

                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    checkpoint_saved_this_step = False
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                        checkpoint_saved_this_step = True

                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights(self.global_steps)
                    if getattr(self, "_pending_validation_guard_path", None):
                        with marked_timer("cleanup_validation_guard", timing_raw, color="green"):
                            self._cleanup_pending_validation_guard()

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with marked_timer("pre_validation_checkpoint", timing_raw, color="green"):
                        self._prepare_checkpoint_before_validation(
                            checkpoint_saved_this_step=checkpoint_saved_this_step
                        )
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics_diffusion(batch=batch))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                num_images = batch.batch["advantages"].shape[0]
                metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_images))
                metrics.update(compute_throughput_metrics_diffusion(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics.update(compute_reward_extra_metrics_diffusion(reward_extra_infos_dict))
                if "uid" in batch.non_tensor_batch:
                    collision_group_ids = (
                        _layer_conditioned_group_index(batch)
                        if "layer_credit_stratum" in batch.non_tensor_batch
                        else None
                    )
                    metrics.update(
                        compute_wam_exploration_metrics(
                            reward_extra_infos_dict,
                            batch.non_tensor_batch["uid"],
                            collision_group_ids=collision_group_ids,
                        )
                    )
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


class DirectPreferenceRayTrainer(BaseRayDiffusionTrainer):
    """Direct-preference diffusion trainer for DPO, DiffusionNFT, AWM, etc."""

    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.is_offline = config.algorithm.get("sample_source", "online") == "offline"
        loss_mode = config.actor_rollout_ref.actor.diffusion_loss.loss_mode
        # DPO needs trainer-side ref noise preds; DiffusionNFT computes ref in the actor engine.
        self.use_reference_policy = need_reference_policy(self.config) or (loss_mode == "dpo")
        self._has_old_adapter = "old" in tuple(
            config.actor_rollout_ref.model.get("policy_state_adapters", ("default",))
        )
        if self._has_old_adapter:
            self._validate_old_adapter_config()
        self._loss_fn = get_diffusion_loss_fn(loss_mode)

    def _validate_old_adapter_config(self):
        rollout_cfg = self.config.actor_rollout_ref.rollout
        actor_loss_cfg = self.config.actor_rollout_ref.actor.diffusion_loss
        if rollout_cfg.rollout_adapter != "old":
            raise ValueError("Old-adapter algorithms require actor_rollout_ref.rollout.rollout_adapter=old.")
        if actor_loss_cfg.loss_mode != "diffusion_nft":
            raise ValueError(
                "Old-adapter algorithms require actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft."
            )

    def init_workers(self):
        """Initialize actor-only workers for offline, or full stack for online preference training."""
        actor_rollout_resource_pool = self._init_colocated_workers()
        if self.is_offline:
            self.reward_loop_manager = None
            self.llm_server_manager = None
            self.enable_agent_reward_loop = False
            self.checkpoint_manager = NoOpCheckpointManager()
            return
        self._init_online_rollout_stack(actor_rollout_resource_pool)

    def _validate(self):
        if self.is_offline and not hasattr(self, "async_rollout_manager"):
            print("Skipping validation generation because offline rollout is disabled.")
            return {"val/offline/skipped": 1.0}
        return super()._validate()

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch_td = batch.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        paired = self.config.algorithm.get("paired_preference", False)
        ppo_mini_batch_size = (
            ppo_mini_batch_size * 2 if paired else ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        )  # direct preference has a pair per prompt
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        if paired and shuffle:
            sys_logger.warning(
                "Shuffle is not supported for direct preference during actor update."
                "This is to prevent the chosen/rejected pairs from being split across different micro batches."
                "Setting shuffle to False."
            )
            shuffle = False

        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            height=self.config.actor_rollout_ref.model.pipeline.height,
            width=self.config.actor_rollout_ref.model.pipeline.width,
            vae_scale_factor=self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if (actor_mfu := actor_output.pop("actor/mfu", None)) is not None:
            actor_output["perf/mfu/actor"] = actor_mfu
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def _compute_ref_noise_pred(self, batch: DataProto) -> Optional[DataProto]:
        """Reference transformer output and shared flow tensors."""
        batch_td = batch.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        metadata = {
            "compute_loss": False,
            "height": self.config.actor_rollout_ref.model.pipeline.height,
            "width": self.config.actor_rollout_ref.model.pipeline.width,
            "vae_scale_factor": self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        else:
            output = self.ref_policy_wg.infer_ref_batch(batch_td)
        if output is None:
            return None

        noise_pred = tu.get(output, "noise_pred")
        if noise_pred.ndim >= 2 and noise_pred.shape[1] == 1:
            noise_pred = noise_pred[:, 0]
        noise = tu.get(output, "noise")
        if noise.ndim >= 2 and noise.shape[1] == 1:
            noise = noise[:, 0]
        timesteps = tu.get(output, "timesteps")
        if timesteps.ndim >= 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        ref_output = {
            "ref_noise_pred": noise_pred.float(),
            "noise": noise.float(),
            "timesteps": timesteps.float(),
        }
        return DataProto.from_tensordict(tu.get_tensordict(ref_output))

    def _prepare_actor_batch(self, batch: DataProto, reward_tensor: torch.Tensor) -> DataProto:
        """Delegate algorithm-specific rollout-to-actor batch preparation."""
        reward_tensor = reward_tensor.squeeze(-1).float() if reward_tensor.ndim > 1 else reward_tensor.float()
        return self._loss_fn.prepare_actor_batch(batch, reward_tensor, self.config)

    def _update_old_policy(self) -> tuple[bool, float, Literal["none", "copy", "ema"]]:
        algo_cfg = self.config.algorithm
        if self.global_steps % algo_cfg.old_policy_update_interval != 0:
            return False, 0.0, "none"

        decay = algo_cfg.old_policy_decay
        if decay is None:
            decay = old_policy_decay(self.global_steps, algo_cfg.old_policy_decay_schedule)

        if decay == 0:
            self.actor_rollout_wg.copy_adapter(source="default", target="old")
            return True, float(decay), "copy"
        else:
            self.actor_rollout_wg.ema_update_adapter(source="default", target="old", decay=decay)
            return True, float(decay), "ema"

    def fit(self):
        """
        Training loop for direct-preference algorithms (DPO, DiffusionNFT, etc.).
        Offline algorithms read pre-computed rewards from the dataset.
        Online algorithms generate rollouts and compute rewards live.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        if self._has_old_adapter:
            self.actor_rollout_wg.copy_adapter(source="default", target="old")
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # Profiler step state machine. Mirrors verl/trainer/ppo/ray_trainer.py.
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    reward_extra_infos_dict: dict[str, list] = {}

                    if self.is_offline:
                        reward_tensor = batch.batch["sample_level_scores"]

                        with marked_timer("adv", timing_raw, color="brown"):
                            batch.batch["sample_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                        batch.batch["sample_level_rewards"] = batch.batch["sample_level_scores"]
                        if self.use_reference_policy:
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_infer_res = self._compute_ref_noise_pred(batch)
                                if ref_infer_res is not None:
                                    batch = batch.union(ref_infer_res)

                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                    else:
                        gen_batch = self._get_gen_batch(batch)
                        gen_batch.meta_info["global_steps"] = self.global_steps
                        rollout_seed_cfg = self.config.actor_rollout_ref.rollout.get("seed")
                        if rollout_seed_cfg is not None:
                            gen_batch.meta_info["rollout_seed"] = int(rollout_seed_cfg) + self.global_steps - 1

                        gen_batch_output = gen_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n,
                            interleave=True,
                        )
                        gen_batch_output.non_tensor_batch["_rollout_seed_global_idx"] = np.arange(
                            len(gen_batch_output), dtype=np.int64
                        )

                        with marked_timer("gen", timing_raw, color="red"):
                            if curr_step_profile:
                                self.llm_server_manager.start_profile()
                                # streaming reward scores inside the gen window; colocate in the reward phase
                                if self.enable_agent_reward_loop:
                                    self.reward_loop_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.llm_server_manager.stop_profile()
                                if self.enable_agent_reward_loop:
                                    self.reward_loop_manager.stop_profile()
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if curr_step_profile:
                                    self.reward_loop_manager.start_profile()
                                batch_reward = self._compute_reward_colocate(batch)
                                if curr_step_profile:
                                    self.reward_loop_manager.stop_profile()
                                batch = batch.union(batch_reward)
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                        with marked_timer("prepare_actor_batch", timing_raw, color="brown"):
                            batch.batch["sample_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )
                            batch = self._prepare_actor_batch(batch, reward_tensor)

                        batch.batch["sample_level_rewards"] = batch.batch["sample_level_scores"]
                        if self.use_reference_policy:
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_infer_res = self._compute_ref_noise_pred(batch)
                                if ref_infer_res is not None:
                                    batch = batch.union(ref_infer_res)

                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                            if self._has_old_adapter:
                                metrics.update(compute_old_policy_metrics(self._update_old_policy()))

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # update weights from trainer to rollout
                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights(self.global_steps)

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir and not self.is_offline:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics_diffusion(batch=batch))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                num_images = (
                    batch.batch["advantages"].shape[0]
                    if "advantages" in batch.batch
                    else batch.batch["sample_level_scores"].shape[0]
                )
                metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_images))
                metrics.update(compute_throughput_metrics_diffusion(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                if "advantages" in batch.batch:
                    gradient_norm = metrics.get("actor/grad_norm", None)
                    metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
