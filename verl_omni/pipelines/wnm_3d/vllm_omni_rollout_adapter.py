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

"""vLLM-Omni rollout adapter for WNM-3D / 3D WAM.

The adapter mirrors GN0 deployment semantics through the checkpoint transform:
66 RGB frames split into 33 history frames (including the current frame) and
33 target frames.  Frozen VGGT plus frozen TGE produce 450 Wan-width geometry
tokens.  Dance-SDE then rolls out the joint video/action DiT using the same
16-transition, fixed 8-step DiT mask and CFG=5 contract as WNM2D.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.wnm_2d.vllm_omni_rollout_adapter import (
    WNM2DPipelineWithLogProb,
)
from verl_omni.pipelines.wnm_3d.diffusers_training_adapter import (
    ARCHITECTURE,
    _configure_fa2_runtime,
    _validate_fa2_runtime,
)
from verl_omni.pipelines.wnm_shared.rollout_rng import (
    _batched_seed_tuple,
    _strict_wam_dance_rollout_per_request_rng,
)

logger = logging.getLogger(__name__)


_VGGT_PROFILE_FIELDS = {
    "prepare": "vggt_timing_prepare_images_metric",
    "aggregator": "vggt_timing_aggregator_metric",
    "tge": "vggt_timing_adapter_metric",
    "total": "vggt_timing_total_metric",
}


def _explicit_rollout_seed_tuple(value: Any) -> tuple[int, ...] | None:
    """Validate either one seed or a scheduler-batched seed container."""

    batched = _batched_seed_tuple(value)
    if batched is not None:
        return batched
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        values = list(value)
    elif value is None:
        return None
    else:
        values = [value]
    if len(values) != 1:
        return None
    seed = values[0]
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"{ARCHITECTURE} rollout seed must be an integer, got {type(seed).__name__}.")
    seed = int(seed)
    if not 0 <= seed < 2**63:
        raise ValueError(f"{ARCHITECTURE} rollout seed is outside [0, 2**63): {seed}.")
    return (seed,)


@contextmanager
def _sobol_fp32_construction() -> Iterator[None]:
    """Keep TGE's fixed Sobol offsets valid under vLLM's BF16 default dtype.

    vLLM constructs custom diffusion pipelines inside a BF16 default-dtype
    context. PyTorch's Sobol kernel does not implement BF16, while the TGE
    initializer calls ``draw`` without a dtype. Restrict the compatibility
    shim to policy construction and preserve every explicitly requested dtype.
    """

    sobol_type = torch.quasirandom.SobolEngine
    original_draw = sobol_type.draw

    def draw_fp32_if_unspecified(
        engine: torch.quasirandom.SobolEngine,
        n: int = 1,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return original_draw(
            engine,
            n=n,
            out=out,
            dtype=torch.float32 if dtype is None else dtype,
        )

    sobol_type.draw = draw_fp32_if_unspecified
    try:
        yield
    finally:
        sobol_type.draw = original_draw


def _take_vggt_profile_snapshot(action_head: Any) -> dict[str, float] | None:
    """Read synchronized VGGT/TGE timings produced by the checkpoint action head."""

    if not bool(getattr(action_head, "profile_module_times", False)):
        return None
    raw = getattr(action_head, "_last_vggt_geometry_metrics", None)
    if not isinstance(raw, dict):
        return None
    result: dict[str, float] = {}
    for label, key in _VGGT_PROFILE_FIELDS.items():
        value = raw.get(key)
        if torch.is_tensor(value) and value.numel() == 1:
            result[label] = float(value.detach().float().item())
    return result or None


def _log_vggt_profile_snapshots(snapshots: list[dict[str, float]]) -> None:
    if not snapshots:
        return
    totals = {label: sum(snapshot.get(label, 0.0) for snapshot in snapshots) for label in _VGGT_PROFILE_FIELDS}
    logger.warning(
        "WNM-3D rollout conditioning timing: calls=%d prepare=%.4fs aggregator=%.4fs tge=%.4fs total=%.4fs",
        len(snapshots),
        totals["prepare"],
        totals["aggregator"],
        totals["tge"],
        totals["total"],
    )


def _target_video_tensor(action_head: Any, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 5 or images.shape[-1] != 3:
        raise ValueError(f"{ARCHITECTURE} target images must have shape [B,T,H,W,3], got {tuple(images.shape)}.")
    videos = images.permute(0, 4, 1, 2, 3)
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        batch, channels, frames, height, width = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        videos = action_head.normalize_video(videos)
        videos = videos.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    videos = videos.to(device=action_head._device, dtype=torch.bfloat16)

    target_h = getattr(action_head.config, "target_video_height", None)
    target_w = getattr(action_head.config, "target_video_width", None)
    if target_h is not None and target_w is not None:
        batch, channels, frames, height, width = videos.shape
        if (height, width) != (int(target_h), int(target_w)):
            videos = torch.nn.functional.interpolate(
                videos.reshape(batch * frames, channels, height, width),
                size=(int(target_h), int(target_w)),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, channels, frames, int(target_h), int(target_w))
    return videos


def _grouped_vggt_past_obs_tokens(
    action_head: Any,
    past_images: torch.Tensor,
    *,
    target_frames: int,
    target_grid_size: tuple[int, int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run frozen VGGT/TGE at deployment-equivalent B=1 per unique prompt."""

    batch_size = int(past_images.shape[0])
    representatives = getattr(action_head, "_verl_conditioning_representative_indices", None)
    group_index = getattr(action_head, "_verl_conditioning_group_index", None)
    if batch_size <= 1 or not representatives or group_index is None or len(group_index) != batch_size:
        tokens = action_head._build_vggt_past_obs_tokens(
            past_images,
            target_frames=target_frames,
            target_grid_size=target_grid_size,
            device=device,
            dtype=dtype,
        )
        if tokens is None:
            raise RuntimeError(f"{ARCHITECTURE} produced no past_obs_tokens.")
        snapshot = _take_vggt_profile_snapshot(action_head)
        if snapshot is not None:
            _log_vggt_profile_snapshots([snapshot])
        return tokens

    representatives = tuple(int(index) for index in representatives)
    group_index = tuple(int(index) for index in group_index)
    singleton_tokens = []
    profile_snapshots: list[dict[str, float]] = []
    for index in representatives:
        tokens = action_head._build_vggt_past_obs_tokens(
            past_images[index : index + 1],
            target_frames=target_frames,
            target_grid_size=target_grid_size,
            device=device,
            dtype=dtype,
        )
        if tokens is None:
            raise RuntimeError(f"{ARCHITECTURE} produced no past_obs_tokens.")
        singleton_tokens.append(tokens)
        snapshot = _take_vggt_profile_snapshot(action_head)
        if snapshot is not None:
            profile_snapshots.append(snapshot)
    _log_vggt_profile_snapshots(profile_snapshots)
    unique_tokens = torch.cat(singleton_tokens, dim=0)
    indices = torch.as_tensor(group_index, dtype=torch.long, device=unique_tokens.device)
    return unique_tokens.index_select(0, indices)


def _wnm_3d_lazy_joint_video_action(
    action_head: Any,
    backbone_output: Any,
    action_input: Any,
    latent_video: torch.Tensor | None = None,
) -> Any:
    del backbone_output, latent_video
    data = action_input
    past_images = data.get("past_images", None)
    if past_images is None:
        raise KeyError(f"{ARCHITECTURE} requires transformed past_images.")
    rollout_seed = _explicit_rollout_seed_tuple(data.get("rollout_seed", None))
    if rollout_seed is None:
        rollout_seed = _explicit_rollout_seed_tuple(getattr(action_head, "_verl_active_rollout_seed", None))
    if rollout_seed is None:
        raise KeyError(f"{ARCHITECTURE} requires one explicit rollout_seed per request.")
    if len(rollout_seed) != int(past_images.shape[0]):
        raise ValueError(
            f"{ARCHITECTURE} rollout seed count must match the transformed batch: "
            f"seeds={len(rollout_seed)}, batch={int(past_images.shape[0])}."
        )

    videos = _target_video_tensor(action_head, data["images"])
    state_features = action_input.state.to(device=action_head._device, dtype=torch.bfloat16)
    prompt_embs = action_head.encode_prompt(data["text"], data["text_attention_mask"]).to(action_head._device)

    target_latents = action_head.encode_video(
        videos,
        action_head.tiled,
        (action_head.tile_size_height, action_head.tile_size_width),
        (action_head.tile_stride_height, action_head.tile_stride_width),
    ).to(action_head._device)
    if target_latents.shape[2] <= 1:
        raise ValueError(
            f"{ARCHITECTURE} target clip must produce at least two latent frames, got {tuple(target_latents.shape)}."
        )

    past_obs_tokens = _grouped_vggt_past_obs_tokens(
        action_head,
        past_images,
        target_frames=int(target_latents.shape[2]),
        target_grid_size=(
            int(target_latents.shape[3]) // 2,
            int(target_latents.shape[4]) // 2,
        ),
        device=target_latents.device,
        dtype=target_latents.dtype,
    )

    _, _, raw_frames, height, width = videos.shape
    first_image = videos[:, :, :1].transpose(1, 2)
    clip_feature, image_condition, _ = action_head.encode_image(first_image, raw_frames, height, width)
    clip_feature = clip_feature.to(action_head._device)
    image_condition = image_condition.to(action_head._device)

    latent_frames = int(target_latents.shape[2])
    num_blocks = (latent_frames - 1) // int(action_head.num_frame_per_block)
    if num_blocks <= 0:
        raise ValueError(f"{ARCHITECTURE} has invalid target latent count {latent_frames}.")
    full_action_horizon = num_blocks * int(action_head.model.num_action_per_block)
    full_state_horizon = num_blocks * int(action_head.model.num_state_per_block)
    if state_features.shape[1] != full_state_horizon:
        if state_features.shape[1] <= 0:
            state_features = torch.zeros(
                state_features.shape[0],
                full_state_horizon,
                int(action_head.model.max_state_dim),
                device=action_head._device,
                dtype=torch.bfloat16,
            )
        elif state_features.shape[1] < full_state_horizon:
            state_features = torch.cat(
                (
                    state_features,
                    state_features[:, -1:].expand(-1, full_state_horizon - state_features.shape[1], -1),
                ),
                dim=1,
            )
        else:
            state_features = state_features[:, :full_state_horizon]

    tokens_per_frame = (int(target_latents.shape[3]) // 2) * (int(target_latents.shape[4]) // 2)
    seq_len = latent_frames * tokens_per_frame
    return _strict_wam_dance_rollout_per_request_rng(
        action_head,
        data=data,
        rollout_seed=rollout_seed,
        target_latents=target_latents,
        past_clean_latents=None,
        past_obs_tokens=past_obs_tokens,
        prompt_embs=prompt_embs,
        prompt_embeds_mask=data["text_attention_mask"],
        clip_feas=clip_feature,
        ys=image_condition,
        state_features=state_features,
        embodiment_id=None,
        seq_len=seq_len,
        full_action_horizon=full_action_horizon,
        num_blocks=num_blocks,
        include_clean_x=False,
    )


def _install_wnm_3d_rollout(action_head: Any) -> None:
    if getattr(action_head, "_verl_wnm_3d_rollout_installed", False):
        return
    action_head.lazy_joint_video_action = MethodType(_wnm_3d_lazy_joint_video_action, action_head)
    action_head._verl_wnm_3d_rollout_installed = True


def _install_vggt_model_pred_return(policy: Any) -> None:
    """Add the 2D ``return_model_pred`` protocol without editing VGGT/GN0.

    The VGGT SimPolicy accepts arbitrary keyword arguments but always returns
    ``(batch, video_pred)``. RL additionally needs the replay tensors produced
    by the trained VLN. Capture that exact mapping at the VLN boundary and
    append it only when the caller explicitly requests the 2D three-tuple.
    """

    if getattr(policy, "_verl_vggt_model_pred_return_installed", False):
        return
    trained_model = policy.trained_model
    action_head = trained_model.action_head
    original_model_forward = trained_model.lazy_joint_video_action_causal
    original_policy_forward = policy.lazy_joint_forward_causal

    def capture_model_pred(
        bound_model: Any,
        inputs: Any,
        latent_video: torch.Tensor | None = None,
    ) -> Any:
        del bound_model
        model_pred = original_model_forward(inputs, latent_video=latent_video)
        action_head._verl_vggt_last_model_pred = model_pred
        return model_pred

    def policy_forward_with_model_pred(
        bound_policy: Any,
        *args: Any,
        return_model_pred: bool = False,
        **kwargs: Any,
    ) -> Any:
        del bound_policy
        action_head._verl_vggt_last_model_pred = None
        try:
            result = original_policy_forward(*args, **kwargs)
            model_pred = action_head._verl_vggt_last_model_pred
            if not return_model_pred:
                return result
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    f"{ARCHITECTURE} SimPolicy must return (batch, video_pred), got {type(result).__name__}."
                )
            if model_pred is None:
                raise RuntimeError(f"{ARCHITECTURE} failed to capture VLN replay outputs.")
            return result[0], result[1], model_pred
        finally:
            action_head._verl_vggt_last_model_pred = None

    trained_model.lazy_joint_video_action_causal = MethodType(capture_model_pred, trained_model)
    policy.lazy_joint_forward_causal = MethodType(policy_forward_with_model_pred, policy)
    policy._verl_vggt_model_pred_return_installed = True


@VllmOmniPipelineBase.register(ARCHITECTURE, algorithm="dance_grpo")
class WNM3DPipelineWithLogProb(WNM2DPipelineWithLogProb):
    """Serve replayable WNM-3D Dance-SDE trajectories."""

    architecture = ARCHITECTURE

    # VllmOmniPipelineBase.register intentionally does not inherit this
    # capability bit: subclasses must opt in from their own __dict__.  VGGT
    # shares the WAM2D request collation and per-sample RNG implementation, so
    # preserve the parent's batching contract explicitly.
    supports_request_batch = True

    replay_condition_key = "past_obs_tokens"
    replay_condition_rank = 3
    replay_condition_tail_shape = (450, 3072)
    requires_clean_x = False
    uses_embodiment_condition = False

    @staticmethod
    def _configure_attention_runtime() -> None:
        _configure_fa2_runtime(component=f"{ARCHITECTURE} rollout")

    @staticmethod
    def _validate_attention_runtime(module: torch.nn.Module) -> None:
        _validate_fa2_runtime(module, component=f"{ARCHITECTURE} rollout")

    @staticmethod
    def _validate_loaded_action_head(action_head: Any) -> None:
        if not bool(getattr(action_head, "use_vggt_geometry_adapter", False)):
            raise ValueError(f"{ARCHITECTURE} requires a checkpoint with use_vggt_geometry_adapter=true.")
        aggregator = getattr(action_head, "vggt_aggregator", None)
        adapter = getattr(action_head, "vggt_geometry_adapter", None)
        if aggregator is None or adapter is None:
            raise RuntimeError(f"{ARCHITECTURE} checkpoint did not materialize VGGT and TGE modules.")
        aggregator.requires_grad_(False).eval()
        adapter.requires_grad_(False).eval()

    @staticmethod
    def _install_action_head_rollout(action_head: Any) -> None:
        _install_wnm_3d_rollout(action_head)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        with _sobol_fp32_construction():
            super().__init__(od_config=od_config, prefix=prefix)
        _install_vggt_model_pred_return(self._policy)
        # The RL sampler attaches these runtime-only values to WANPolicyHead.
        # The GammaNav inference head intentionally stays free of RL-specific
        # state. Seed the two fields before the inherited
        # request parser evaluates its fallback expression; every request then
        # overwrites them with the values carried in SamplingParams.extra_args.
        self._action_head.wam_noise_level = float(os.environ.get("WAM_NOISE_LEVEL", "0.7"))
        self._action_head.wam_action_noise_level = float(
            os.environ.get(
                "WAM_ACTION_NOISE_LEVEL",
                os.environ.get("WAM_NOISE_LEVEL", "0.7"),
            )
        )
        self._action_head._verl_vggt_tge_rl_trainable = False
        logger.warning(
            "%s RL boundary: train joint DiT; freeze rollout-side VGGT/TGE; "
            "replay past_obs_tokens with shape [B,450,3072]",
            ARCHITECTURE,
        )


__all__ = ["WNM3DPipelineWithLogProb"]
