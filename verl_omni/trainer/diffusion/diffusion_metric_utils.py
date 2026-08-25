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
Metrics for diffusion (image generation) training.
"""

from typing import Any, Literal

import numpy as np
import torch
from verl import DataProto


def compute_data_metrics_diffusion(batch: DataProto) -> dict[str, Any]:
    """
    Computes various metrics from a diffusion training batch.

    For diffusion (image generation) models, rewards and advantages are
    indexed over denoising timesteps rather than output tokens.

    Args:
        batch: A DataProto object containing diffusion batch data. GRPO-style
            batches include sample_level_rewards [B, T], advantages [B, T], and
            returns [B, T]. DPO-style batches may only include sample_level_rewards [B].

    Returns:
        A dictionary of metrics including:
            - critic/rewards/mean, max, min: Per-image reward statistics
            - critic/rewards/zero_std_ratio: Fraction of prompt groups whose reward std is zero
            - critic/rewards/std_mean: Mean per-prompt reward standard deviation
            - critic/rewards/group_size: Average number of images sampled per unique prompt
            - critic/advantages/mean, max, min: Element-wise advantage statistics over B*T, when available
            - critic/returns/mean, max, min: Element-wise return statistics over B*T, when available
    """
    sample_level_rewards = batch.batch["sample_level_rewards"]
    if sample_level_rewards.ndim > 1:
        sequence_reward = sample_level_rewards.mean(dim=1)  # [B]
    else:
        sequence_reward = sample_level_rewards  # [B]

    reward_mean = torch.mean(sequence_reward).detach().item()
    reward_max = torch.max(sequence_reward).detach().item()
    reward_min = torch.min(sequence_reward).detach().item()

    metrics = {
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
    }

    if "advantages" in batch.batch:
        # Flatten [B, T] tensors for aggregate statistics across timesteps.
        advantages = batch.batch["advantages"].flatten()  # [B*T]
        metrics.update(
            {
                "critic/advantages/mean": torch.mean(advantages).detach().item(),
                "critic/advantages/max": torch.max(advantages).detach().item(),
                "critic/advantages/min": torch.min(advantages).detach().item(),
                "critic/advantages/std": torch.std(advantages).detach().item(),
                "critic/advantages/abs_mean": torch.mean(torch.abs(advantages)).detach().item(),
            }
        )

    if "action_advantages" in batch.batch:
        action_advantages = batch.batch["action_advantages"].flatten()
        metrics.update(
            {
                "critic/action_advantages/mean": torch.mean(action_advantages).detach().item(),
                "critic/action_advantages/max": torch.max(action_advantages).detach().item(),
                "critic/action_advantages/min": torch.min(action_advantages).detach().item(),
                "critic/action_advantages/std": torch.std(action_advantages).detach().item(),
                "critic/action_advantages/abs_mean": torch.mean(torch.abs(action_advantages)).detach().item(),
            }
        )

    if "action_stop_advantages" in batch.batch:
        action_stop_advantages = batch.batch["action_stop_advantages"].flatten()
        metrics.update(
            {
                "critic/action_stop_advantages/mean": torch.mean(action_stop_advantages).detach().item(),
                "critic/action_stop_advantages/max": torch.max(action_stop_advantages).detach().item(),
                "critic/action_stop_advantages/min": torch.min(action_stop_advantages).detach().item(),
                "critic/action_stop_advantages/std": torch.std(action_stop_advantages).detach().item(),
                "critic/action_stop_advantages/abs_mean": torch.mean(torch.abs(action_stop_advantages)).detach().item(),
            }
        )

    if "returns" in batch.batch:
        returns = batch.batch["returns"].flatten()  # [B*T]
        metrics.update(
            {
                "critic/returns/mean": torch.mean(returns).detach().item(),
                "critic/returns/max": torch.max(returns).detach().item(),
                "critic/returns/min": torch.min(returns).detach().item(),
            }
        )

    if "uid" in batch.non_tensor_batch:
        rewards_np = sequence_reward.cpu().float().numpy()
        uid_array = np.array(batch.non_tensor_batch["uid"])
        unique_uids = np.unique(uid_array)

        per_prompt_stds = np.array([np.std(rewards_np[uid_array == uid]) for uid in unique_uids])

        metrics["critic/rewards/zero_std_ratio"] = float(np.mean(per_prompt_stds == 0))
        metrics["critic/rewards/std_mean"] = float(np.mean(per_prompt_stds))
        metrics["critic/rewards/group_size"] = float(len(rewards_np) / len(unique_uids))

    return metrics


def compute_old_policy_metrics(
    update_result: tuple[bool, float, Literal["none", "copy", "ema"]],
) -> dict[str, Any]:
    """Build metrics for old-policy adapter refreshes."""
    update_applied, decay, update_type = update_result
    return {
        "old_policy/update_applied": float(update_applied),
        "old_policy/copy_update": float(update_type == "copy"),
        "old_policy/ema_update": float(update_type == "ema"),
        "old_policy/decay": float(decay),
    }


def compute_timing_metrics_diffusion(timing_raw: dict[str, float], num_images: int) -> dict[str, Any]:
    """
    Computes timing metrics for diffusion training.

    Args:
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
        num_images: Total number of images processed in the batch, used to compute per-image timing.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_image_ms/{name}: Per-image timing in milliseconds for core compute stages
              (gen, ref, old_log_prob, adv, update_actor). Non-compute stages such as
              save_checkpoint, update_weights, and testing are excluded.
    """
    num_images_of_section = {name: num_images for name in ["gen", "ref", "old_log_prob", "adv", "update_actor"]}

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_image_ms/{name}": timing_raw[name] * 1000 / num_images_of_section[name]
            for name in set(num_images_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughput_metrics_diffusion(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for diffusion (image/video generation) training.

    Unlike language model training where throughput is measured in tokens/sec,
    diffusion training generates images, so throughput is reported as images
    per second.

    Args:
        batch: A DataProto object containing diffusion batch data.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_images: Number of images processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Images generated per second per GPU
    """
    if "advantages" in batch.batch:
        batch_size = batch.batch["advantages"].shape[0]
    else:
        batch_size = batch.batch["sample_level_rewards"].shape[0]
    time = timing_raw["step"]
    return {
        "perf/total_num_images": batch_size,
        "perf/time_per_step": time,
        "perf/throughput": batch_size / (time * n_gpus),
    }


def compute_reward_extra_metrics_diffusion(reward_extra_infos_dict: dict) -> dict[str, Any]:
    """Computes per-sub-reward mean metrics for multi-reward tracking."""
    metrics = {}
    if not reward_extra_infos_dict:
        return metrics
    for key, values in reward_extra_infos_dict.items():
        if key.startswith("_exploration_"):
            continue
        if isinstance(values, np.ndarray):
            if not np.issubdtype(values.dtype, np.number):
                continue
            metrics[f"critic/{key}/mean"] = float(values.mean())
        elif isinstance(values, list) and len(values) > 0:
            if not isinstance(values[0], int | float):
                continue
            metrics[f"critic/{key}/mean"] = float(np.mean(values))
    return metrics


def _group_indices(uids: np.ndarray) -> list[np.ndarray]:
    """Return stable per-prompt sample indices without assuming contiguous groups."""
    uids = np.asarray(uids).reshape(-1)
    return [np.flatnonzero(uids == uid) for uid in np.unique(uids)]


def _reward_group_metrics(
    values: Any,
    groups: list[np.ndarray],
    *,
    prefix: str,
) -> dict[str, float]:
    rewards = np.asarray(values, dtype=np.float64).reshape(-1)
    stds = []
    best_minus_mean = []
    ranges = []
    unique_ratios = []
    for indices in groups:
        group = rewards[indices]
        group = group[np.isfinite(group)]
        if group.size == 0:
            continue
        stds.append(float(np.std(group)))
        best_minus_mean.append(float(np.max(group) - np.mean(group)))
        ranges.append(float(np.ptp(group)))
        unique_ratios.append(float(len(np.unique(group)) / len(group)))
    if not stds:
        return {}
    stds_array = np.asarray(stds)
    return {
        f"{prefix}/group_std_mean": float(np.mean(stds_array)),
        f"{prefix}/zero_std_ratio": float(np.mean(stds_array <= 1e-12)),
        f"{prefix}/near_zero_std_ratio": float(np.mean(stds_array <= 1e-6)),
        f"{prefix}/best_of_n_minus_mean": float(np.mean(best_minus_mean)),
        f"{prefix}/group_range_mean": float(np.mean(ranges)),
        f"{prefix}/unique_reward_ratio": float(np.mean(unique_ratios)),
    }


def _binary_pair_disagreement(values: np.ndarray) -> float:
    """Fraction of unordered sample pairs with different binary outcomes."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] > 0.5
    count = len(values)
    if count < 2:
        return 0.0
    positives = int(values.sum())
    return float(positives * (count - positives) / (count * (count - 1) / 2))


def _binary_group_mix_ratio(
    values: Any,
    groups: list[np.ndarray],
) -> float:
    """Fraction of non-empty groups containing both binary outcomes."""

    outcomes = np.asarray(values, dtype=np.float64).reshape(-1)
    mixed: list[float] = []
    for indices in groups:
        group = outcomes[indices]
        group = group[np.isfinite(group)] > 0.5
        if group.size == 0:
            continue
        mixed.append(float(np.any(group) and np.any(~group)))
    return float(np.mean(mixed) if mixed else 0.0)


def compute_wam_exploration_metrics(
    reward_extra_infos_dict: dict,
    uids: Any,
    *,
    collision_group_ids: Any | None = None,
) -> dict[str, float]:
    """Measure useful within-prompt WAM exploration for N-sample GRPO groups.

    Reward dispersion answers whether samples can be ranked.  Physical
    trajectory and compact visual-signature dispersion independently answer
    whether the fixed diffusion/SDE noise still creates distinct behavior.
    """
    if not reward_extra_infos_dict:
        return {}
    uid_array = np.asarray(uids).reshape(-1)
    groups = _group_indices(uid_array)
    if not groups:
        return {}

    metrics: dict[str, float] = {}
    collision_key = next(
        (key for key in ("action_any_collision", "action_collision") if key in reward_extra_infos_dict),
        None,
    )
    if collision_key is not None:
        collision_groups = (
            _group_indices(np.asarray(collision_group_ids).reshape(-1)) if collision_group_ids is not None else groups
        )
        metrics["exploration/action/collision_mix_ratio"] = _binary_group_mix_ratio(
            reward_extra_infos_dict[collision_key],
            collision_groups,
        )
    for key, prefix in (
        ("action_reward", "exploration/action_reward"),
        ("visual_reward", "exploration/visual_reward"),
    ):
        if key in reward_extra_infos_dict:
            metrics.update(
                _reward_group_metrics(
                    reward_extra_infos_dict[key],
                    groups,
                    prefix=prefix,
                )
            )

    trajectory_key = "_exploration_action_world_xy"
    if trajectory_key in reward_extra_infos_dict:
        trajectories = np.asarray(
            reward_extra_infos_dict[trajectory_key],
            dtype=np.float64,
        )
        pairwise_ades = []
        endpoint_stds = []
        valid_fractions = []
        usable_groups = 0
        for indices in groups:
            group = trajectories[indices]
            valid = np.isfinite(group).all(axis=tuple(range(1, group.ndim)))
            valid_group = group[valid]
            valid_fractions.append(float(np.mean(valid)))
            if len(valid_group) < 2:
                continue
            usable_groups += 1
            pair_values = []
            for left in range(len(valid_group)):
                for right in range(left + 1, len(valid_group)):
                    pair_values.append(
                        float(
                            np.linalg.norm(
                                valid_group[left] - valid_group[right],
                                axis=-1,
                            ).mean()
                        )
                    )
            pairwise_ades.append(float(np.mean(pair_values)))
            endpoints = valid_group[:, -1]
            endpoint_center = endpoints.mean(axis=0, keepdims=True)
            endpoint_stds.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                np.square(endpoints - endpoint_center),
                                axis=-1,
                            )
                        )
                    )
                )
            )
        metrics.update(
            {
                "exploration/action/group_pairwise_ade_m": float(np.mean(pairwise_ades) if pairwise_ades else 0.0),
                "exploration/action/group_endpoint_std_m": float(np.mean(endpoint_stds) if endpoint_stds else 0.0),
                "exploration/action/group_valid_fraction": float(np.mean(valid_fractions)),
                "exploration/action/groups_with_pairwise_fraction": float(usable_groups / len(groups)),
            }
        )

    if "action_pred_path_length" in reward_extra_infos_dict:
        path_lengths = np.asarray(
            reward_extra_infos_dict["action_pred_path_length"],
            dtype=np.float64,
        ).reshape(-1)
        group_stds = [
            float(np.std(path_lengths[indices][np.isfinite(path_lengths[indices])]))
            for indices in groups
            if np.isfinite(path_lengths[indices]).any()
        ]
        metrics["exploration/action/group_path_length_std_m"] = float(np.mean(group_stds) if group_stds else 0.0)

    for key, metric_name in (
        ("action_collision", "exploration/action/group_collision_disagreement"),
        ("stop_failed", "exploration/action/group_stop_failed_disagreement"),
        ("stop_left_goal", "exploration/action/group_stop_left_goal_disagreement"),
    ):
        if key not in reward_extra_infos_dict:
            continue
        values = np.asarray(reward_extra_infos_dict[key], dtype=np.float64).reshape(-1)
        metrics[metric_name] = float(np.mean([_binary_pair_disagreement(values[indices]) for indices in groups]))

    visual_key = "_exploration_visual_luma"
    if visual_key in reward_extra_infos_dict:
        signatures = np.asarray(
            reward_extra_infos_dict[visual_key],
            dtype=np.float64,
        )
        pairwise_rms = []
        valid_fractions = []
        usable_groups = 0
        for indices in groups:
            group = signatures[indices]
            valid = np.isfinite(group).all(axis=tuple(range(1, group.ndim)))
            valid_group = group[valid]
            valid_fractions.append(float(np.mean(valid)))
            if len(valid_group) < 2:
                continue
            usable_groups += 1
            pair_values = []
            for left in range(len(valid_group)):
                for right in range(left + 1, len(valid_group)):
                    pair_values.append(float(np.sqrt(np.mean(np.square(valid_group[left] - valid_group[right])))))
            pairwise_rms.append(float(np.mean(pair_values)))
        metrics.update(
            {
                "exploration/visual/group_pairwise_luma_rms": float(np.mean(pairwise_rms) if pairwise_rms else 0.0),
                "exploration/visual/group_valid_fraction": float(np.mean(valid_fractions)),
                "exploration/visual/groups_with_pairwise_fraction": float(usable_groups / len(groups)),
            }
        )

    return metrics
