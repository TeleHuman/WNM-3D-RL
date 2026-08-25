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

"""Training adapter for the WNM-3D 3D world-action model.

The full VLN checkpoint owns frozen T5/CLIP/VAE/VGGT components and the TGE
geometry adapter.  The first RL integration intentionally trains the complete
5B joint ``CausalWanModel`` while replaying detached ``past_obs_tokens`` from
the rollout worker.  Keeping this boundary explicit avoids serializing four
33x32x32x2048 VGGT tap tensors for every PPO transition.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import torch
from accelerate import init_empty_weights

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.wnm_2d.diffusers_training_adapter import (
    _LARGE_MODEL_DIM,
    WNM2D,
    _extract_action_head_config,
    _extract_dit_config,
    _install_gradient_checkpointing_compat,
    _load_joint_dit_state,
    _rebuild_nonbuffer_freqs,
)
from verl_omni.pipelines.wnm_shared.action_gradient_gain import (
    install_action_backbone_gradient_gain,
)
from verl_omni.pipelines.wnm_shared.batch1_equivalent import (
    install_batch1_equivalent_action_encoder,
)
from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)

ARCHITECTURE = "WNM3D"
_REPLAY_CONDITION_KEYS = (
    "clip_feature",
    "y",
    "state",
    "past_obs_tokens",
)


def _validate_vggt_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    action_head = _extract_action_head_config(checkpoint_dir)
    if action_head.get("use_vggt_geometry_adapter") is not True:
        raise ValueError(f"{ARCHITECTURE} requires action_head_cfg.config.use_vggt_geometry_adapter=true.")
    for name, expected in (
        ("vggt_image_resolution", 512),
        ("vggt_patch_size", 16),
        ("vggt_adapter_dim", 512),
    ):
        value = int(action_head.get(name, expected))
        if value != expected:
            raise ValueError(f"{ARCHITECTURE} currently supports {name}={expected}, got {value}.")
    return action_head


def _configure_fa2_runtime(*, component: str) -> None:
    """Match the GN0 WNM-3D deployment attention backend."""

    os.environ["ATTENTION_BACKEND"] = "FA2"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["WAM_ENFORCE_SDPA"] = "false"
    logger.warning("%s configured for the GN0-compatible FA2 attention path", component)


def _validate_fa2_runtime(module: torch.nn.Module, *, component: str) -> None:
    from gammanav.vln.model.wnm_3d.modules.wan2_1_attention import AttentionModule

    backends = {submodule.backend for submodule in module.modules() if isinstance(submodule, AttentionModule)}
    if backends != {"FA2"}:
        raise RuntimeError(f"{component} expected every AttentionModule to use FA2, got {sorted(backends)!r}.")


@DiffusionModelBase.register(ARCHITECTURE, algorithm="dance_grpo")
class WNM3D(WNM2D):
    """DanceGRPO actor for WNM-3D's joint 3D world/action DiT."""

    replay_condition_keys = _REPLAY_CONDITION_KEYS
    past_condition_key = "past_obs_tokens"
    use_clean_x_condition = False
    uses_embodiment_condition = False

    @classmethod
    def build_module(
        cls,
        model_config: DiffusionModelConfig,
        torch_dtype: torch.dtype,
    ) -> torch.nn.Module:
        try:
            _configure_fa2_runtime(component=f"{ARCHITECTURE} actor")
            from gammanav.vln.model.wnm_3d.modules.wan2_1_submodule import rope_params
            from gammanav.vln.model.wnm_3d.modules.wan_video_dit_action_casual_chunk import (
                CausalWanModel,
            )
        except ImportError as exc:
            raise ImportError(
                f"{ARCHITECTURE} requires a WNM-3D checkout on PYTHONPATH; "
                "dependencies and checkpoints are never downloaded automatically."
            ) from exc

        checkpoint_dir = Path(model_config.local_path).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"WNM-3D checkpoint directory not found: {checkpoint_dir}")
        _validate_vggt_checkpoint(checkpoint_dir)
        model_kwargs = _extract_dit_config(checkpoint_dir)
        model_dim = int(model_kwargs.get("dim", 0))
        if model_dim >= _LARGE_MODEL_DIM and torch_dtype == torch.float32:
            raise ValueError(
                f"Refusing to materialize the large {ARCHITECTURE} dim={model_dim} joint DiT "
                "in float32; set actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16."
            )

        with init_empty_weights(include_buffers=False):
            module = CausalWanModel(**model_kwargs)
        _validate_fa2_runtime(module, component=f"{ARCHITECTURE} actor")
        _load_joint_dit_state(module, checkpoint_dir, torch_dtype)
        _rebuild_nonbuffer_freqs(module, rope_params)
        _install_gradient_checkpointing_compat(module)
        install_batch1_equivalent_action_encoder(
            module,
            component=f"{ARCHITECTURE} actor",
        )
        install_action_backbone_gradient_gain(module)
        module._no_split_modules = ["CausalWanAttentionBlock"]
        module._verl_conditioning_contract = "past_obs_tokens[B,450,3072]"
        module._verl_vggt_tge_rl_trainable = False
        logger.info(
            "Loaded %s joint DiT from %s; VGGT/TGE remains rollout-side frozen",
            ARCHITECTURE,
            checkpoint_dir,
        )
        return module

    @classmethod
    def _prepare_past_condition(
        cls,
        frozen_condition: Callable[..., torch.Tensor],
        *,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        inner_module: torch.nn.Module,
    ) -> torch.Tensor:
        del channels
        tokens = frozen_condition(cls.past_condition_key)
        patch_size = tuple(getattr(inner_module, "patch_size", (1, 2, 2)))
        expected_frames = 9
        expected_tokens = (expected_frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])
        expected_dim = int(getattr(inner_module, "dim", 0))
        expected_shape = (batch_size, expected_tokens, expected_dim)
        if tuple(tokens.shape) != expected_shape:
            raise ValueError(
                f"{ARCHITECTURE} past_obs_tokens must have shape {expected_shape}, got {tuple(tokens.shape)}."
            )
        return tokens


__all__ = ["ARCHITECTURE", "WNM3D"]
