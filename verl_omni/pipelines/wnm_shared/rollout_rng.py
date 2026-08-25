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

"""Per-request RNG and strict Dance-SDE rollout for WNM."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping, Sequence
from types import MethodType
from typing import Any

import numpy as np
import torch

from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.wnm_shared.rollout_common import (
    _DEFAULT_CFG_SCALE,
    _DEFAULT_INFERENCE_STEPS,
    _DIT_PREDICTION_SOURCE_STEPS,
    _DIT_STEP_MASK,
    _LAYER_CREDIT_FIELDS,
    _deployed_action_policy_mask,
    _env_flag,
    build_shifted_schedule,
    derive_rollout_subseed,
)
from verl_omni.utils.action_chunk_credit import (
    action_chunk_credit_enabled,
    action_chunk_size,
)

logger = logging.getLogger(__name__)


def _batched_seed_tuple(value: Any) -> tuple[int, ...] | None:
    """Return validated per-request seeds when ``value`` contains a batch."""

    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        values = list(value)
    else:
        return None
    if len(values) <= 1:
        return None

    seeds = []
    for index, seed in enumerate(values):
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise TypeError(f"WNM batched rollout seed[{index}] must be an integer, got {type(seed).__name__}.")
        seed = int(seed)
        if not 0 <= seed < 2**63:
            raise ValueError(f"WNM batched rollout seed[{index}] is outside [0, 2**63): {seed}.")
        seeds.append(seed)
    return tuple(seeds)


def _randn_per_request(
    shape: tuple[int, ...],
    *,
    generators: Sequence[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Draw each batch item from its own RNG stream."""

    if not shape or shape[0] != len(generators):
        raise ValueError(
            f"Per-request RNG count must match tensor batch dimension: shape={shape}, generators={len(generators)}."
        )
    return torch.cat(
        [
            torch.randn((1, *shape[1:]), generator=generator, device=device, dtype=torch.float32)
            for generator in generators
        ],
        dim=0,
    )


def _dance_transition_per_request(
    *,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    noise_level: float,
    generators: Sequence[torch.Generator],
    log_prob_mask: torch.Tensor | None = None,
    log_prob_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one Dance-SDE transition without sharing RNG state across requests."""

    from verl_omni.pipelines.wnm_shared.wam_dance_sde import (
        sample_dance_sde_previous_step,
    )

    if sample.shape[0] != len(generators) or model_output.shape[0] != len(generators):
        raise ValueError("Dance-SDE per-request generator count does not match the model batch.")
    if log_prob_mask is not None:
        if not isinstance(log_prob_mask, torch.Tensor):
            raise TypeError(
                f"Dance-SDE per-request log_prob_mask must be a torch.Tensor, got {type(log_prob_mask).__name__}."
            )
        if log_prob_mask.shape[0] not in (1, sample.shape[0]):
            raise ValueError(
                "Dance-SDE per-request log_prob_mask batch dimension must be 1 "
                f"or {sample.shape[0]}, got {tuple(log_prob_mask.shape)}."
            )

    def request_mask(index: int) -> torch.Tensor | None:
        if log_prob_mask is None:
            return None
        mask_index = 0 if log_prob_mask.shape[0] == 1 else index
        return log_prob_mask[mask_index : mask_index + 1]

    transitions = [
        sample_dance_sde_previous_step(
            sample=sample[index : index + 1],
            model_output=model_output[index : index + 1],
            sigma=sigma,
            sigma_prev=sigma_prev,
            noise_level=noise_level,
            generator=generator,
            log_prob_mask=request_mask(index),
        )
        for index, generator in enumerate(generators)
    ]
    if log_prob_chunk_size is None:
        log_probs = torch.cat([transition.log_prob for transition in transitions], dim=0)
    else:
        chunk_log_probs = []
        for transition in transitions:
            elementwise_log_prob = -((transition.prev_sample.detach() - transition.prev_sample_mean) ** 2) / (
                2 * (transition.std_dev_t**2)
            )
            elementwise_log_prob = (
                elementwise_log_prob
                - torch.log(transition.std_dev_t)
                - torch.log(
                    torch.sqrt(
                        2
                        * torch.as_tensor(
                            math.pi,
                            dtype=elementwise_log_prob.dtype,
                            device=elementwise_log_prob.device,
                        )
                    )
                )
            )
            chunk_log_probs.append(
                FlowMatchSDEDiscreteScheduler._reduce_log_prob(
                    elementwise_log_prob,
                    request_mask(len(chunk_log_probs)),
                    chunk_size=log_prob_chunk_size,
                )
            )
        log_probs = torch.cat(chunk_log_probs, dim=0)
    return (
        torch.cat([transition.prev_sample for transition in transitions], dim=0),
        log_probs,
    )


def _strict_wam_dance_rollout_per_request_rng(
    action_head: Any,
    *,
    data: Any,
    rollout_seed: tuple[int, ...],
    target_latents: torch.Tensor,
    past_clean_latents: torch.Tensor | None,
    prompt_embs: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    clip_feas: torch.Tensor,
    ys: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor | None,
    seq_len: int,
    full_action_horizon: int,
    num_blocks: int,
    past_obs_tokens: torch.Tensor | None = None,
    include_clean_x: bool = True,
) -> Any:
    """Batch deterministic model work with grouped init and per-request SDE RNG."""

    from transformers.feature_extraction_utils import BatchFeature

    batch_size, _, num_target_latents, _, _ = target_latents.shape
    if (past_clean_latents is None) == (past_obs_tokens is None):
        raise ValueError("WNM rollout requires exactly one history condition: past_clean_latents or past_obs_tokens.")
    history_condition_key = "past_clean_x" if past_clean_latents is not None else "past_obs_tokens"
    history_condition = past_clean_latents if past_clean_latents is not None else past_obs_tokens
    assert history_condition is not None
    stage_timing = _env_flag("WAM_ROLLOUT_STAGE_TIMING", False)

    def new_timing_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    init_timing = new_timing_pair() if stage_timing else None
    negative_text_timing = new_timing_pair() if stage_timing else None
    dit_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    sde_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    if len(rollout_seed) != batch_size:
        raise ValueError(
            f"WNM rollout seed count must match batch size: seeds={len(rollout_seed)}, batch={batch_size}."
        )
    device = target_latents.device
    model_dtype = torch.bfloat16
    if action_head.num_inference_steps != _DEFAULT_INFERENCE_STEPS:
        raise ValueError(
            "WNM RL rollout must match the deployed 16-transition schedule: "
            f"expected={_DEFAULT_INFERENCE_STEPS}, got={action_head.num_inference_steps}."
        )
    if not bool(getattr(action_head, "enable_dit_cache", False)):
        raise ValueError("WNM RL rollout requires ENABLE_DIT_CACHE=true for deployed-sampler parity.")
    if bool(getattr(action_head, "dynamic_cache_schedule", False)):
        raise ValueError("WNM RL rollout requires the fixed DiT mask, not DYNAMIC_CACHE_SCHEDULE.")
    action_head_mask = tuple(bool(value) for value in getattr(action_head, "dit_step_mask", ()))
    if action_head_mask != _DIT_STEP_MASK:
        raise ValueError(
            "WNM RL rollout DiT mask differs from the deployed 16-step mask: "
            f"expected={_DIT_STEP_MASK}, got={action_head_mask}."
        )
    if not bool(getattr(action_head, "enable_cfg", False)):
        raise ValueError("WNM RL rollout requires ENABLE_CFG=true for deployed-sampler parity.")
    cfg_scale = float(getattr(action_head, "cfg_scale", float("nan")))
    if not math.isclose(cfg_scale, _DEFAULT_CFG_SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"WNM RL rollout requires CFG_SCALE={_DEFAULT_CFG_SCALE:g}, got {cfg_scale!r}.")
    for field_name, value in (
        ("wam_noise_level", action_head.wam_noise_level),
        ("wam_action_noise_level", action_head.wam_action_noise_level),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} must be finite and positive, got {value}")

    video_seeds = tuple(derive_rollout_subseed(seed, 0) for seed in rollout_seed)
    action_seeds = tuple(derive_rollout_subseed(seed, 1) for seed in rollout_seed)
    video_generators = tuple(torch.Generator(device=device).manual_seed(seed) for seed in video_seeds)
    action_generators = tuple(torch.Generator(device=device).manual_seed(seed) for seed in action_seeds)

    init_same_noise_raw = data.get("init_same_noise", False)
    init_same_noise_values = torch.as_tensor(init_same_noise_raw).reshape(-1)
    if init_same_noise_values.numel() not in (1, batch_size):
        raise ValueError(
            "init_same_noise must be scalar or have one value per request: "
            f"values={init_same_noise_values.numel()} batch={batch_size}."
        )
    if init_same_noise_values.numel() == 1:
        init_same_noise_values = init_same_noise_values.expand(batch_size)
    if not torch.all(init_same_noise_values == init_same_noise_values[0]):
        raise ValueError("A WNM request batch cannot mix init_same_noise settings.")
    init_same_noise = bool(init_same_noise_values[0].item())

    if init_same_noise:
        initial_seed_raw = data.get("initial_noise_seed", None)
        if initial_seed_raw is None:
            raise KeyError("init_same_noise=true requires an explicit initial_noise_seed.")
        initial_seed_values = torch.as_tensor(initial_seed_raw).reshape(-1)
        if initial_seed_values.numel() not in (1, batch_size):
            raise ValueError(
                "initial_noise_seed must be scalar or have one value per request: "
                f"values={initial_seed_values.numel()} batch={batch_size}."
            )
        if initial_seed_values.numel() == 1:
            initial_seed_values = initial_seed_values.expand(batch_size)
        initial_seeds = tuple(int(value) for value in initial_seed_values.tolist())
        if any(seed < 0 or seed >= 2**63 for seed in initial_seeds):
            raise ValueError(f"initial_noise_seed values must lie in [0, 2**63): {initial_seeds}.")
        init_video_generators = tuple(
            torch.Generator(device=device).manual_seed(derive_rollout_subseed(seed, 0)) for seed in initial_seeds
        )
        init_action_generators = tuple(
            torch.Generator(device=device).manual_seed(derive_rollout_subseed(seed, 1)) for seed in initial_seeds
        )
    else:
        initial_seeds = rollout_seed
        # Preserve the legacy exact RNG sequence: initialization consumes the
        # beginning of each request's transition generator.
        init_video_generators = video_generators
        init_action_generators = action_generators

    present_credit_fields = [name for name in _LAYER_CREDIT_FIELDS if name in data]
    layer_credit_enabled = bool(present_credit_fields)
    if layer_credit_enabled and len(present_credit_fields) != len(_LAYER_CREDIT_FIELDS):
        missing = sorted(set(_LAYER_CREDIT_FIELDS) - set(present_credit_fields))
        raise KeyError(f"Incomplete layer-conditioned rollout metadata; missing {missing}.")
    if layer_credit_enabled:
        credit_values = {}
        for name in _LAYER_CREDIT_FIELDS:
            value = torch.as_tensor(data[name], device="cpu").reshape(-1)
            if value.numel() not in (1, batch_size):
                raise ValueError(
                    f"{name} must be scalar or have one value per request; values={value.numel()} batch={batch_size}."
                )
            if value.numel() == 1:
                value = value.expand(batch_size)
            credit_values[name] = tuple(int(item) for item in value.tolist())
        credit_strata = credit_values["layer_credit_stratum"]
        credit_branches = credit_values["layer_credit_branch"]
        credit_transitions = credit_values["layer_credit_transition"]
        if any(stratum < 0 or stratum >= 4 for stratum in credit_strata):
            raise ValueError(f"layer_credit_stratum must lie in [0,4), got {credit_strata}.")
        if any(branch < 0 for branch in credit_branches):
            raise ValueError(f"layer_credit_branch must be non-negative, got {credit_branches}.")
        if any(
            transition < 0 or transition >= action_head.num_inference_steps or not _DIT_STEP_MASK[transition]
            for transition in credit_transitions
        ):
            raise ValueError(
                "layer_credit_transition must select a transition that executes DiT under the "
                f"deployed cache mask, got {credit_transitions}."
            )
        if not bool(getattr(action_head, "_verl_layer_credit_logged", False)):
            action_head._verl_layer_credit_logged = True
            logger.warning(
                "Enabled layer-conditioned counterfactual RNG: strata=%s branches=%s transitions=%s; "
                "all non-selected transition noise is common within a prompt.",
                credit_strata,
                credit_branches,
                credit_transitions,
            )
    else:
        credit_strata = ()
        credit_branches = ()
        credit_transitions = ()

    # Equal initial seeds intentionally produce equal starting latents. The
    # video/action transition generators above remain derived from the unique
    # rollout_seed tuple and diverge at the first Dance-SDE transition.
    if init_timing is not None:
        init_timing[0].record()
    video_state = _randn_per_request(tuple(target_latents.shape), generators=init_video_generators, device=device)
    action_state = _randn_per_request(
        (batch_size, full_action_horizon, action_head.model.action_dim),
        generators=init_action_generators,
        device=device,
    )
    # The checkpoint pads InteriorGS actions to 32 dimensions, while both SFT
    # supervision and GN0 deployment consume only [dx, dy, dyaw].  Keep the
    # stochastic state full-sized for model compatibility, but exclude padded
    # coordinates from both behavior and replay likelihood reductions.
    action_policy_mask = _deployed_action_policy_mask(action_state)
    sigmas, timesteps = build_shifted_schedule(
        num_inference_steps=action_head.num_inference_steps,
        shift=5.0,
        num_train_timesteps=action_head.scheduler.num_train_timesteps,
        device=device,
    )
    if init_timing is not None:
        init_timing[1].record()

    video_states = [video_state.detach()]
    action_states = [action_state.detach()]
    video_log_probs = []
    action_log_probs = []
    rollout_embodiment_id = (
        torch.zeros(batch_size, device=device, dtype=torch.long) if embodiment_id is not None else None
    )

    prompt_embs = prompt_embs.detach().to(device=device, dtype=model_dtype)
    prompt_embeds_mask = prompt_embeds_mask.detach().to(device=device)
    missing_negative = [name for name in ("text_negative", "text_attention_mask_negative") if name not in data]
    if missing_negative:
        raise KeyError(f"WNM CFG rollout is missing negative text fields: {missing_negative}.")
    if negative_text_timing is not None:
        negative_text_timing[0].record()
    negative_prompt_embs = (
        action_head.encode_prompt(data["text_negative"], data["text_attention_mask_negative"])
        .detach()
        .to(device=device, dtype=model_dtype)
    )
    if negative_text_timing is not None:
        negative_text_timing[1].record()
    negative_prompt_embeds_mask = torch.as_tensor(data["text_attention_mask_negative"]).detach().to(device=device)
    if negative_prompt_embs.shape != prompt_embs.shape:
        raise ValueError(
            "WNM positive/negative prompt embeddings must have identical shapes: "
            f"positive={tuple(prompt_embs.shape)}, negative={tuple(negative_prompt_embs.shape)}."
        )
    if tuple(negative_prompt_embeds_mask.shape) != tuple(negative_prompt_embs.shape[:2]):
        raise ValueError(
            "WNM negative prompt mask does not match its embeddings: "
            f"mask={tuple(negative_prompt_embeds_mask.shape)}, "
            f"embeddings={tuple(negative_prompt_embs.shape)}."
        )
    clip_feas = clip_feas.detach().to(device=device, dtype=model_dtype)
    ys = ys.detach().to(device=device, dtype=model_dtype)
    state_features = state_features.detach().to(device=device, dtype=model_dtype)
    target_latents = target_latents.detach().to(device=device, dtype=model_dtype)
    history_condition = history_condition.detach().to(device=device, dtype=model_dtype)
    num_dit_forwards = 0
    num_dit_prediction_steps = 0
    video_noise_pred = None
    action_noise_pred = None
    for index, timestep_value in enumerate(timesteps):
        video_timestep = timestep_value.expand(batch_size, num_target_latents)
        action_timestep = timestep_value.expand(batch_size, full_action_horizon)
        if _DIT_STEP_MASK[index]:
            dit_timing = new_timing_pair() if stage_timing else None
            if dit_timing is not None:
                dit_timing[0].record()
            joint_inputs = {
                "x": video_state.to(dtype=model_dtype),
                "timestep": video_timestep,
                "timestep_action": action_timestep,
                "clip_feature": clip_feas,
                "y": ys,
                "seq_len": seq_len,
                "state": state_features,
                "action": action_state.to(dtype=model_dtype),
                history_condition_key: history_condition,
            }
            if rollout_embodiment_id is not None:
                joint_inputs["embodiment_id"] = rollout_embodiment_id
            if include_clean_x:
                joint_inputs["clean_x"] = target_latents
            conditional_video, conditional_action = action_head.model(
                context=prompt_embs,
                **joint_inputs,
            )
            unconditional_video, _ = action_head.model(
                context=negative_prompt_embs,
                **joint_inputs,
            )
            video_noise_pred = unconditional_video + cfg_scale * (conditional_video - unconditional_video)
            # Deployment applies CFG only to the visual branch.
            action_noise_pred = conditional_action
            num_dit_prediction_steps += 1
            num_dit_forwards += 2
            if dit_timing is not None:
                dit_timing[1].record()
                dit_timings.append(dit_timing)
        elif video_noise_pred is None or action_noise_pred is None:
            raise RuntimeError(f"WNM DiT cache has no prediction available at transition {index}.")
        sde_timing = new_timing_pair() if stage_timing else None
        if sde_timing is not None:
            sde_timing[0].record()
        if layer_credit_enabled:
            # Every request of one prompt starts from the same initial seed.
            # Use common per-step noise everywhere except the request's
            # selected transition; only that transition receives a branch
            # stream. Keeping the tail streams common makes the terminal
            # reward a counterfactual credit signal for this one noise layer.
            video_step_generators = []
            action_step_generators = []
            for request_index, initial_seed in enumerate(initial_seeds):
                video_step_seed = derive_rollout_subseed(initial_seed, 10_000 + 2 * index)
                action_step_seed = derive_rollout_subseed(initial_seed, 10_001 + 2 * index)
                if index == credit_transitions[request_index]:
                    branch_stream = 1 + 1_000 * credit_strata[request_index] + credit_branches[request_index]
                    video_step_seed = derive_rollout_subseed(video_step_seed, branch_stream)
                    action_step_seed = derive_rollout_subseed(action_step_seed, branch_stream)
                video_step_generators.append(torch.Generator(device=device).manual_seed(video_step_seed))
                action_step_generators.append(torch.Generator(device=device).manual_seed(action_step_seed))
            active_video_generators = tuple(video_step_generators)
            active_action_generators = tuple(action_step_generators)
        else:
            active_video_generators = video_generators
            active_action_generators = action_generators

        video_state, visual_log_prob = _dance_transition_per_request(
            sample=video_state,
            model_output=video_noise_pred.float(),
            sigma=sigmas[index],
            sigma_prev=sigmas[index + 1],
            noise_level=action_head.wam_noise_level,
            generators=active_video_generators,
        )
        action_state, action_log_prob = _dance_transition_per_request(
            sample=action_state,
            model_output=action_noise_pred.float(),
            sigma=sigmas[index],
            sigma_prev=sigmas[index + 1],
            noise_level=action_head.wam_action_noise_level,
            generators=active_action_generators,
            log_prob_mask=action_policy_mask,
            log_prob_chunk_size=(action_chunk_size() if action_chunk_credit_enabled() else None),
        )
        video_states.append(video_state.detach())
        action_states.append(action_state.detach())
        video_log_probs.append(visual_log_prob.detach())
        action_log_probs.append(action_log_prob.detach())
        if sde_timing is not None:
            sde_timing[1].record()
            sde_timings.append(sde_timing)

    expected_prediction_steps = sum(_DIT_STEP_MASK)
    expected_dit_forwards = 2 * expected_prediction_steps
    if num_dit_prediction_steps != expected_prediction_steps or num_dit_forwards != expected_dit_forwards:
        raise RuntimeError(
            "Strict WAM batched rollout did not execute the deployed DiT/CFG schedule: "
            f"prediction_steps={num_dit_prediction_steps}/{expected_prediction_steps}, "
            f"forwards={num_dit_forwards}/{expected_dit_forwards}."
        )
    if stage_timing:
        torch.cuda.synchronize(device)

        def elapsed(pair: tuple[torch.cuda.Event, torch.cuda.Event] | None) -> float:
            return 0.0 if pair is None else pair[0].elapsed_time(pair[1]) / 1000.0

        dit_seconds = sum(elapsed(pair) for pair in dit_timings)
        sde_seconds = sum(elapsed(pair) for pair in sde_timings)
        logger.warning(
            "WNM rollout GPU-stage timing: batch=%d init=%.4fs negative_t5=%.4fs "
            "DiT_cfg_16_forwards=%.4fs SDE_32_transitions=%.4fs accounted=%.4fs",
            batch_size,
            elapsed(init_timing),
            elapsed(negative_text_timing),
            dit_seconds,
            sde_seconds,
            elapsed(init_timing) + elapsed(negative_text_timing) + dit_seconds + sde_seconds,
        )

    prefix_frames = data.get("target_prefix_frames", None)
    if prefix_frames is None:
        block_idx = 0
    else:
        flattened_prefix = torch.as_tensor(prefix_frames).reshape(-1)
        if flattened_prefix.numel() not in (1, batch_size):
            raise ValueError("target_prefix_frames must be scalar or have one value per batched request.")
        if flattened_prefix.numel() > 1 and not torch.all(flattened_prefix == flattened_prefix[0]):
            raise ValueError("A WNM request batch must use the same target_prefix_frames value.")
        prefix_value = int(flattened_prefix[0].item())
        block_idx = max(0, min(num_blocks - 1, (prefix_value - 1) // action_head.model.num_action_per_block))
    action_start = block_idx * action_head.model.num_action_per_block
    action_end = action_start + action_head.model.num_action_per_block
    normalized_actions = action_state[:, action_start:action_end].detach()

    scalar_shape = (batch_size,)
    output_data = {
        "action_pred": normalized_actions,
        "actions": action_state.detach(),
        "normalized_actions": normalized_actions,
        "video_pred": video_state,
        "all_latents": torch.stack(video_states, dim=1),
        "all_timesteps": timesteps.detach().unsqueeze(0).expand(batch_size, -1).clone(),
        "all_log_probs": torch.stack(video_log_probs, dim=1),
        "all_action_latents": torch.stack(action_states, dim=1),
        "all_action_timesteps": timesteps.detach().unsqueeze(0).expand(batch_size, -1).clone(),
        "action_log_probs": torch.stack(action_log_probs, dim=1),
        "action_policy_mask": action_policy_mask.detach(),
        "prompt_embeds": prompt_embs,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embs,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        "clip_feature": clip_feas,
        "y": ys,
        "state": state_features,
        history_condition_key: history_condition,
        "full_action_latent": action_state.detach(),
        "wam_rollout": torch.ones(scalar_shape, dtype=torch.bool, device=device),
        "rollout_seed": torch.tensor(rollout_seed, dtype=torch.long, device=device),
        "initial_noise_seed": torch.tensor(initial_seeds, dtype=torch.long, device=device),
        "init_same_noise": torch.full(scalar_shape, init_same_noise, dtype=torch.bool, device=device),
        "video_rollout_seed": torch.tensor(video_seeds, dtype=torch.long, device=device),
        "action_rollout_seed": torch.tensor(action_seeds, dtype=torch.long, device=device),
        "action_start": torch.full(scalar_shape, action_start, dtype=torch.long, device=device),
        "action_end": torch.full(scalar_shape, action_end, dtype=torch.long, device=device),
        "num_dit_forwards": torch.full(scalar_shape, num_dit_forwards, dtype=torch.long, device=device),
        "num_dit_prediction_steps": torch.full(scalar_shape, num_dit_prediction_steps, dtype=torch.long, device=device),
        "dit_prediction_source_steps": torch.tensor(_DIT_PREDICTION_SOURCE_STEPS, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone(),
        "true_cfg_scale": torch.full(scalar_shape, cfg_scale, dtype=torch.float32, device=device),
        "num_inference_steps": torch.full(
            scalar_shape, action_head.num_inference_steps, dtype=torch.long, device=device
        ),
        "sigma_shift": torch.full(scalar_shape, 5.0, dtype=torch.float32, device=device),
        "noise_level": torch.full(scalar_shape, action_head.wam_noise_level, dtype=torch.float32, device=device),
        "action_noise_level": torch.full(
            scalar_shape, action_head.wam_action_noise_level, dtype=torch.float32, device=device
        ),
        "scheduler_sigmas": sigmas.detach().unsqueeze(0).expand(batch_size, -1).clone(),
        "sde_type": "dance_sde",
    }
    if rollout_embodiment_id is not None:
        output_data["embodiment_id"] = rollout_embodiment_id
    if include_clean_x:
        output_data["clean_x"] = target_latents
    return BatchFeature(data=output_data)


def _install_per_request_rng_rollout(action_head: Any) -> None:
    """Install an RL-worker-local vector-seed shim without editing WNM/GN0."""

    if getattr(action_head, "_verl_per_request_rng_installed", False):
        return
    action_module = importlib.import_module(action_head.__class__.__module__)
    original_resolver = action_module.resolve_rollout_seed

    def resolve_rollout_seed_batch_aware(
        request: Mapping[str, Any],
        default_seed: int,
        *,
        require_explicit_rollout_seed: bool = False,
    ) -> int | tuple[int, ...]:
        seeds = _batched_seed_tuple(request.get("rollout_seed"))
        if seeds is not None:
            return seeds
        return original_resolver(
            request,
            default_seed,
            require_explicit_rollout_seed=require_explicit_rollout_seed,
        )

    def strict_rollout_batch_aware(bound_head: Any, *, rollout_seed: Any, **kwargs: Any) -> Any:
        seed_tuple = rollout_seed if isinstance(rollout_seed, tuple) else (int(rollout_seed),)
        return _strict_wam_dance_rollout_per_request_rng(bound_head, rollout_seed=seed_tuple, **kwargs)

    # This module-global symbol is imported by the action-head method. The
    # mutation is process-local to this vLLM rollout worker and never touches
    # the WNM checkout or GN0 inference process.
    action_module.resolve_rollout_seed = resolve_rollout_seed_batch_aware
    action_head._strict_wam_dance_rollout = MethodType(strict_rollout_batch_aware, action_head)
    action_head._verl_per_request_rng_installed = True
