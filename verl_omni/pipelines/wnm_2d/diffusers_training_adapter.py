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

"""Training-side adapter for the WNM-2D joint video/action DiT.

The Stage-2 WNM checkpoint is a full VLN checkpoint rather than a
Diffusers pipeline.  This adapter deliberately loads only the joint
``CausalWanModel`` stored below ``action_head.model``.  Frozen text, image and
VAE encoders remain rollout-side components and their precomputed condition
tensors are replayed by the actor.

All GammaNav imports live inside :meth:`WNM2D.build_module`, so merely
importing verl-omni does not make WNM an installation dependency.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from accelerate import init_empty_weights
from diffusers import ModelMixin
from safetensors import safe_open
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase, WorldActionDiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.wan22_dance_grpo.common import sd3_time_shift
from verl_omni.pipelines.wnm_shared.action_gradient_gain import (
    install_action_backbone_gradient_gain,
)
from verl_omni.pipelines.wnm_shared.batch1_equivalent import (
    install_batch1_equivalent_action_encoder,
)
from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)

ARCHITECTURE = "WNM2D"
_SIGMA_SHIFT = 5.0
_LARGE_MODEL_DIM = 1024
_CHECKPOINT_PREFIXES = (
    "action_head.model.base_model.model.",
    "action_head.model.",
)
_HYDRA_CONFIG_KEYS = {"_target_", "_convert_", "_recursive_", "_partial_"}
_CONDITION_KEYS = ("clip_feature", "y", "state", "embodiment_id", "clean_x", "past_clean_x")
_LAYER_CREDIT_REPLAY_TRANSITION_KEY = "layer_credit_replay_transition"
_DEFAULT_INFERENCE_STEPS = 16
_DEFAULT_CFG_SCALE = 5.0
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


def _dit_prediction_source_steps(step_mask: tuple[bool, ...]) -> tuple[int, ...]:
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


def _configure_explicit_sdpa_runtime(*, component: str) -> None:
    """Pin every WNM attention path inside this RL worker to PyTorch SDPA.

    This is intentionally process-local: GN0 and standalone WNM inference
    run in different processes and retain their own backend configuration.
    """
    os.environ["ATTENTION_BACKEND"] = "torch"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
    os.environ["WAM_ENFORCE_SDPA"] = "true"

    # CausalWanModel uses AttentionModule for joint self-attention, and
    # wan2_1_submodule keeps a direct reference to the legacy helper for text/
    # image cross-attention. Disable optional FA dispatch in this worker and
    # refresh that direct reference so both paths are unambiguously SDPA.
    from gammanav.vln.model.wnm_3d.modules import attention as attention_module
    from gammanav.vln.model.wnm_3d.modules import wan2_1_attention, wan2_1_submodule

    attention_module.FLASH_ATTN_2_AVAILABLE = False
    attention_module.FLASH_ATTN_3_AVAILABLE = False
    wan2_1_attention.FLASH_ATTN_2_AVAILABLE = False
    wan2_1_attention.FLASH_ATTN_3_AVAILABLE = False
    wan2_1_submodule.flash_attention = attention_module.flash_attention
    logger.warning("%s configured all WNM attention paths for PyTorch SDPA", component)


def _validate_explicit_sdpa(module: torch.nn.Module, *, component: str) -> None:
    """Fail closed when this run requires the SFT-compatible SDPA path."""
    requested = os.getenv("ATTENTION_BACKEND", "").strip().lower()
    if requested != "torch":
        raise RuntimeError(
            f"{component} requires ATTENTION_BACKEND=torch when WAM_ENFORCE_SDPA is enabled; "
            f"got {requested or '<unset>'!r}."
        )
    from gammanav.vln.model.wnm_3d.modules.wan2_1_attention import AttentionModule

    backends = {submodule.backend for submodule in module.modules() if isinstance(submodule, AttentionModule)}
    if backends != {"torch"}:
        raise RuntimeError(f"{component} contains non-SDPA AttentionModule backends: {sorted(backends)!r}.")
    logger.warning(
        "%s attention contract: all %d AttentionModule instances use PyTorch SDPA",
        component,
        sum(isinstance(submodule, AttentionModule) for submodule in module.modules()),
    )


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"WNM checkpoint config is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"WNM checkpoint config must be a JSON object: {path}")
    return value


def _extract_action_head_config(checkpoint_dir: Path) -> dict[str, Any]:
    """Extract the resolved WNM action-head config from a VLN checkpoint."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"WNM checkpoint config not found: {config_path}")

    root = _load_json_mapping(config_path)
    action_head = root.get("action_head_cfg")
    if not isinstance(action_head, dict):
        raise KeyError("WNM config.json must contain an object at `action_head_cfg`.")
    if isinstance(action_head.get("config"), dict):
        action_head = action_head["config"]
    return action_head


def _extract_dit_config(checkpoint_dir: Path) -> dict[str, Any]:
    """Extract resolved ``CausalWanModel`` kwargs from a Stage-2 VLN config."""
    action_head = _extract_action_head_config(checkpoint_dir)

    diffusion_model = action_head.get("diffusion_model_cfg")
    if not isinstance(diffusion_model, dict):
        raise KeyError("WNM config.json must contain an object at `action_head_cfg.config.diffusion_model_cfg`.")

    target = diffusion_model.get("_target_")
    if target is not None and not str(target).endswith(".CausalWanModel"):
        raise ValueError(f"Expected a CausalWanModel in WNM config, got _target_={target!r}.")

    model_kwargs = {key: value for key, value in diffusion_model.items() if key not in _HYDRA_CONFIG_KEYS}
    if not model_kwargs:
        raise ValueError("WNM diffusion_model_cfg did not contain any constructor arguments.")
    return model_kwargs


def _extract_num_inference_steps(checkpoint_dir: Path) -> int:
    action_head = _extract_action_head_config(checkpoint_dir)
    value = action_head.get("num_inference_timesteps")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"WNM checkpoint action_head_cfg.config.num_inference_timesteps must be a positive integer, got {value!r}."
        )
    return value


def _normalize_checkpoint_key(key: str) -> str | None:
    """Map a full VLN state key to the standalone joint DiT state key."""
    for prefix in _CHECKPOINT_PREFIXES:
        if key.startswith(prefix):
            normalized = key[len(prefix) :]
            return normalized.replace(".base_layer.", ".")
    return None


def _safe_resolve_shard(checkpoint_dir: Path, shard_name: str) -> Path:
    shard_path = (checkpoint_dir / shard_name).resolve()
    checkpoint_root = checkpoint_dir.resolve()
    if shard_path.parent != checkpoint_root:
        raise ValueError(f"WNM checkpoint index references a file outside its directory: {shard_name!r}.")
    if not shard_path.is_file():
        raise FileNotFoundError(f"WNM checkpoint shard not found: {shard_path}")
    return shard_path


def _checkpoint_weight_map(checkpoint_dir: Path) -> dict[str, Path]:
    """Return raw checkpoint key -> local safetensors file without Hub fallback."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"

    if index_path.is_file():
        index = _load_json_mapping(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"WNM checkpoint index has no non-empty `weight_map`: {index_path}")
        result = {}
        for key, shard_name in weight_map.items():
            if not isinstance(key, str) or not isinstance(shard_name, str):
                raise TypeError(f"WNM checkpoint index contains a non-string weight entry: {index_path}")
            result[key] = _safe_resolve_shard(checkpoint_dir, shard_name)
        return result

    if single_path.is_file():
        with safe_open(single_path, framework="pt", device="cpu") as tensors:
            return {key: single_path for key in tensors.keys()}

    raise FileNotFoundError(
        f"No local WNM safetensors checkpoint found in {checkpoint_dir}. Expected "
        "`model.safetensors` or `model.safetensors.index.json`; network fallback is disabled."
    )


def _load_joint_dit_state(
    module: torch.nn.Module,
    checkpoint_dir: Path,
    torch_dtype: torch.dtype,
) -> None:
    """Assign matching Stage-2 tensors directly into a meta-initialized module."""
    expected_state = module.state_dict()
    expected_keys = set(expected_state)
    raw_weight_map = _checkpoint_weight_map(checkpoint_dir)

    normalized_to_raw: dict[str, tuple[str, Path]] = {}
    unexpected_joint_keys = []
    for raw_key, shard_path in raw_weight_map.items():
        normalized = _normalize_checkpoint_key(raw_key)
        if normalized is None:
            continue
        if normalized not in expected_keys:
            unexpected_joint_keys.append(raw_key)
            continue
        if normalized in normalized_to_raw:
            previous = normalized_to_raw[normalized][0]
            raise ValueError(f"WNM checkpoint keys {previous!r} and {raw_key!r} both map to {normalized!r}.")
        normalized_to_raw[normalized] = (raw_key, shard_path)

    if unexpected_joint_keys:
        preview = sorted(unexpected_joint_keys)[:20]
        raise KeyError(
            "WNM checkpoint contains unsupported joint-DiT keys after prefix normalization: "
            f"{preview} (total={len(unexpected_joint_keys)})."
        )

    loaded_keys = set(normalized_to_raw)
    missing_keys = sorted(expected_keys - loaded_keys)
    if missing_keys:
        raise KeyError(
            f"WNM joint-DiT checkpoint is missing model state keys: {missing_keys[:20]} (total={len(missing_keys)})."
        )
    if not loaded_keys:
        raise KeyError("WNM checkpoint contains no `action_head.model.*` tensors for the standalone joint DiT.")

    keys_by_shard: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for normalized, (raw_key, shard_path) in normalized_to_raw.items():
        keys_by_shard[shard_path].append((raw_key, normalized))

    assigned_keys: set[str] = set()
    for shard_path in sorted(keys_by_shard, key=os.fspath):
        state_for_shard = {}
        with safe_open(shard_path, framework="pt", device="cpu") as tensors:
            available_keys = set(tensors.keys())
            for raw_key, normalized in keys_by_shard[shard_path]:
                if raw_key not in available_keys:
                    raise KeyError(f"WNM checkpoint index key {raw_key!r} is absent from {shard_path}.")
                tensor = tensors.get_tensor(raw_key)
                expected = expected_state[normalized]
                if tuple(tensor.shape) != tuple(expected.shape):
                    raise ValueError(
                        f"WNM tensor shape mismatch for {raw_key!r} -> {normalized!r}: "
                        f"checkpoint={tuple(tensor.shape)}, model={tuple(expected.shape)}."
                    )
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=torch_dtype)
                state_for_shard[normalized] = tensor

        incompatible = module.load_state_dict(state_for_shard, strict=False, assign=True)
        if incompatible.unexpected_keys:
            raise KeyError(f"Unexpected keys while assigning WNM shard {shard_path}: {incompatible.unexpected_keys}")
        assigned_keys.update(state_for_shard)

    if assigned_keys != expected_keys:
        missing_after_assign = sorted(expected_keys - assigned_keys)
        raise KeyError(
            "WNM joint-DiT state assignment was incomplete: "
            f"{missing_after_assign[:20]} (total={len(missing_after_assign)})."
        )

    meta_tensors = [name for name, tensor in module.state_dict().items() if tensor.is_meta]
    if meta_tensors:
        raise RuntimeError(
            "WNM joint DiT still has meta tensors after checkpoint loading: "
            f"{meta_tensors[:20]} (total={len(meta_tensors)})."
        )


def _rebuild_nonbuffer_freqs(module: torch.nn.Module, rope_params: Callable[[int, int], torch.Tensor]) -> None:
    """Recreate CausalWanModel RoPE tensors that are intentionally not buffers."""
    dim = int(module.dim)
    num_heads = int(module.num_heads)
    if dim % num_heads != 0:
        raise ValueError(f"WNM dim={dim} is not divisible by num_heads={num_heads}.")
    head_dim = dim // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"WNM attention head dimension must be even, got {head_dim}.")

    module.freqs_action = rope_params(1024 * 10, head_dim)
    module.freqs_state = rope_params(1024, head_dim)
    module.freqs = [
        rope_params(1024, head_dim - 4 * (head_dim // 6)),
        rope_params(1024, 2 * (head_dim // 6)),
        rope_params(1024, 2 * (head_dim // 6)),
    ]
    nonbuffer_freqs = [module.freqs_action, module.freqs_state, *module.freqs]
    if any(freq.is_meta for freq in nonbuffer_freqs):
        raise RuntimeError("WNM non-buffer RoPE frequencies were rebuilt on the meta device.")


def _install_gradient_checkpointing_compat(module: torch.nn.Module) -> None:
    """Use the current Diffusers checkpointing hook on legacy WNM models.

    WNM's ``CausalWanModel`` implements the pre-Diffusers-0.35 hook
    ``_set_gradient_checkpointing(module, value=False)``.  Current Diffusers
    calls that hook with ``enable`` and ``gradient_checkpointing_func`` keyword
    arguments, so enabling activation checkpointing otherwise fails before the
    first FSDP forward.  The base implementation sets the same
    ``gradient_checkpointing`` flag consumed by CausalWanModel's forward and is
    therefore the compatible implementation for this adapter.
    """
    module._set_gradient_checkpointing = MethodType(ModelMixin._set_gradient_checkpointing, module)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    current = module
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "_fsdp_wrapped_module", None)
        if wrapped is None:
            break
        current = wrapped
    return current


def _require_tensor(value: Any, name: str, *, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"WNM condition `{name}` must be a tensor, got {type(value).__name__}.")
    if value.ndim == 0 or value.shape[0] != batch_size:
        raise ValueError(f"WNM condition `{name}` must have batch dimension {batch_size}, got {tuple(value.shape)}.")
    return value


def _validate_compact_replay_transitions(micro_batch: TensorDict, *, batch_size: int) -> torch.Tensor:
    """Validate original source-step indices attached to a layer-conditioned T=1 replay."""

    if _LAYER_CREDIT_REPLAY_TRANSITION_KEY not in micro_batch:
        raise ValueError(
            "WNM T=1 replay is allowed only for layer-conditioned credit and requires "
            f"`{_LAYER_CREDIT_REPLAY_TRANSITION_KEY}`."
        )
    transitions = _require_tensor(
        micro_batch[_LAYER_CREDIT_REPLAY_TRANSITION_KEY],
        _LAYER_CREDIT_REPLAY_TRANSITION_KEY,
        batch_size=batch_size,
    )
    if transitions.ndim == 2 and transitions.shape[1] == 1:
        transitions = transitions[:, 0]
    if transitions.ndim != 1:
        raise ValueError(
            f"{_LAYER_CREDIT_REPLAY_TRANSITION_KEY} must have shape ({batch_size},) or "
            f"({batch_size}, 1), got {tuple(transitions.shape)}."
        )
    transitions = transitions.to(dtype=torch.long)
    allowed = torch.as_tensor(
        [index for index, executes_dit in enumerate(_DIT_STEP_MASK) if executes_dit],
        dtype=torch.long,
        device=transitions.device,
    )
    valid = torch.isin(transitions, allowed)
    if not bool(valid.all()):
        invalid = torch.unique(transitions[~valid]).detach().cpu().tolist()
        raise ValueError(
            "WNM compact replay may select only deployed DiT source transitions; "
            f"invalid={invalid}, allowed={allowed.detach().cpu().tolist()}."
        )
    return transitions


def _compute_dtype(module: torch.nn.Module) -> torch.dtype:
    """Use the live joint-DiT parameters as the replay dtype reference.

    The device deliberately comes from the replay trajectory instead. FSDP
    CPU-offload policies may keep parameter shards on CPU until the forward
    pre-hook while the micro-batch is already on the accelerator.
    """
    parameter = next((value for value in module.parameters() if value.is_floating_point()), None)
    if parameter is None:
        raise ValueError("WNM2D joint DiT has no floating-point parameter to determine compute dtype.")
    if parameter.is_meta:
        raise RuntimeError("WNM2D joint DiT parameters are still on meta during input preparation.")
    return parameter.dtype


@DiffusionModelBase.register(ARCHITECTURE, algorithm="dance_grpo")
class WNM2D(WorldActionDiffusionModelBase):
    """DanceGRPO adapter for WNM-2D's joint video/action CausalWanModel."""

    # Subclasses may keep the joint DiT/replay machinery while replacing the
    # history-conditioning representation.  WNM-2D uses VAE latents;
    # WNM-3D uses geometry tokens produced by its frozen rollout-side
    # VGGT/TGE condition encoder.
    replay_condition_keys = _CONDITION_KEYS
    past_condition_key = "past_clean_x"
    use_clean_x_condition = True
    uses_embodiment_condition = True

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
        del batch_size, inner_module
        past_clean_x = frozen_condition(cls.past_condition_key)
        if past_clean_x.ndim != 5 or (
            past_clean_x.shape[1] != channels or tuple(past_clean_x.shape[3:]) != (height, width)
        ):
            raise ValueError(
                "WNM past_clean_x must have shape (batch, channels, past_frames, height, width) with "
                f"matching channel/spatial dimensions, got {tuple(past_clean_x.shape)}."
            )
        return past_clean_x

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype) -> torch.nn.Module:
        # GammaNav is an optional external project. Keep imports inside this hook so
        # registry discovery and CPU-only verl-omni imports remain dependency-free.
        try:
            _configure_explicit_sdpa_runtime(component="WNM2D actor")
            from gammanav.vln.model.wnm_3d.modules.wan2_1_submodule import rope_params
            from gammanav.vln.model.wnm_3d.modules.wan_video_dit_action_casual_chunk import CausalWanModel
        except ImportError as exc:
            raise ImportError(
                "WNM2D requires the local WNM-2D repository on PYTHONPATH; "
                "automatic dependency or model downloads are disabled."
            ) from exc

        checkpoint_dir = Path(model_config.local_path).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"WNM checkpoint directory not found: {checkpoint_dir}")
        model_kwargs = _extract_dit_config(checkpoint_dir)
        model_dim = int(model_kwargs.get("dim", 0))
        if model_dim >= _LARGE_MODEL_DIM and torch_dtype == torch.float32:
            raise ValueError(
                f"Refusing to materialize the large WNM2D dim={model_dim} joint DiT in float32. "
                "Set `actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16`; otherwise each rank can "
                "materialize more than 20 GiB of model weights before FSDP sharding."
            )

        # Parameters are created on meta; non-buffer tensors stay materialized.
        # Checkpoint tensors are then assigned shard-by-shard without constructing
        # a second full CPU state dict.
        with init_empty_weights(include_buffers=False):
            module = CausalWanModel(**model_kwargs)

        _validate_explicit_sdpa(module, component="WNM2D actor")

        _load_joint_dit_state(module, checkpoint_dir, torch_dtype)
        _rebuild_nonbuffer_freqs(module, rope_params)
        _install_gradient_checkpointing_compat(module)
        install_batch1_equivalent_action_encoder(
            module,
            component="WNM2D actor",
        )
        install_action_backbone_gradient_gain(module)

        # Keep the concrete WNM block name explicit on the standalone
        # actor module so the 5B model is wrapped block-by-block under FSDP.
        module._no_split_modules = ["CausalWanAttentionBlock"]

        logger.info("Loaded WNM2D joint DiT from local checkpoint %s", checkpoint_dir)
        return module

    @classmethod
    def configure_trainable_params(cls, module: torch.nn.Module, model_config: DiffusionModelConfig) -> None:
        """Full-finetune the standalone joint DiT, matching Stage-1/Stage-2 scope."""
        del model_config
        trainable = 0
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable += parameter.numel()
        if trainable == 0:
            raise ValueError("WNM2D joint DiT contains no trainable parameters.")

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchSDEDiscreteScheduler:
        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        visual_sde_type = getattr(model_config.algo, "sde_type", None)
        action_sde_type = getattr(model_config.algo, "action_sde_type", None) or visual_sde_type
        if visual_sde_type != "dance_sde" or action_sde_type != "dance_sde":
            raise ValueError(
                "WNM2D canonical rollout replay requires "
                "algo.sde_type=algo.action_sde_type='dance_sde'; "
                f"got visual={visual_sde_type!r}, action={action_sde_type!r}."
            )

        visual_noise_level = float(getattr(model_config.algo, "noise_level", 0.0))
        action_noise_level = getattr(model_config.algo, "action_noise_level", None)
        action_noise_level = visual_noise_level if action_noise_level is None else float(action_noise_level)
        for name, value in (
            ("algo.noise_level", visual_noise_level),
            ("algo.action_noise_level", action_noise_level),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"WNM2D {name} must be finite and positive, got {value}.")

        num_steps = int(model_config.pipeline.num_inference_steps)
        if num_steps != _DEFAULT_INFERENCE_STEPS:
            raise ValueError(
                "WNM2D deployed-sampler parity requires 16 denoising steps; "
                f"got {num_steps}. The checkpoint's legacy inference-step field is intentionally "
                "overridden by this recorded runtime contract."
            )
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
        sigmas = sd3_time_shift(_SIGMA_SHIFT, sigmas)[:-1].cpu().numpy()
        scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas)

    @classmethod
    def replay_prediction_source_steps(
        cls,
        scheduler_inputs: TensorDict | dict[str, torch.Tensor],
    ) -> tuple[int, ...]:
        """Validate and expose the deployed 8/16 DiT-cache replay contract."""

        timesteps = scheduler_inputs.get("all_timesteps")
        if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 2:
            raise ValueError("WNM replay requires all_timesteps with shape (batch, policy_steps).")
        batch_size, transition_count = timesteps.shape
        if transition_count == 1:
            _validate_compact_replay_transitions(scheduler_inputs, batch_size=batch_size)
            return (0,)
        if transition_count != _DEFAULT_INFERENCE_STEPS:
            raise ValueError(
                "WNM actor replay must use the deployed 16-transition schedule or validated compact T=1 "
                f"credit replay, got T={transition_count}."
            )

        recorded = scheduler_inputs.get("dit_prediction_source_steps")
        if not isinstance(recorded, torch.Tensor):
            raise KeyError("WNM 8/16 replay requires recorded dit_prediction_source_steps.")
        expected = (
            torch.tensor(
                _DIT_PREDICTION_SOURCE_STEPS,
                dtype=recorded.dtype,
                device=recorded.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        if tuple(recorded.shape) != tuple(expected.shape) or not torch.equal(recorded, expected):
            raise ValueError(
                "WNM rollout/actor DiT-cache mismatch: expected source map "
                f"{list(_DIT_PREDICTION_SOURCE_STEPS)}, got shape={tuple(recorded.shape)}."
            )

        scalar_contract = {
            "num_dit_prediction_steps": sum(_DIT_STEP_MASK),
            "num_dit_forwards": 2 * sum(_DIT_STEP_MASK),
            "num_inference_steps": _DEFAULT_INFERENCE_STEPS,
        }
        for key, expected_value in scalar_contract.items():
            value = scheduler_inputs.get(key)
            if not isinstance(value, torch.Tensor) or value.numel() != batch_size:
                raise KeyError(f"WNM 8/16 replay requires one recorded {key} value per sample.")
            if not torch.all(value.reshape(-1) == expected_value):
                raise ValueError(f"WNM rollout/actor contract mismatch for {key}: expected {expected_value}.")
        cfg = scheduler_inputs.get("true_cfg_scale")
        if not isinstance(cfg, torch.Tensor) or cfg.numel() != batch_size:
            raise KeyError("WNM 8/16 replay requires one recorded true_cfg_scale value per sample.")
        if not torch.allclose(
            cfg.detach().float().reshape(-1),
            torch.full((batch_size,), _DEFAULT_CFG_SCALE, device=cfg.device),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"WNM rollout/actor CFG mismatch; expected CFG={_DEFAULT_CFG_SCALE:g}.")
        return _DIT_PREDICTION_SOURCE_STEPS

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        true_cfg_scale = float(getattr(model_config.pipeline, "true_cfg_scale", 1.0))
        if not math.isclose(true_cfg_scale, _DEFAULT_CFG_SCALE, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"WNM2D deployed-sampler parity requires true_cfg_scale={_DEFAULT_CFG_SCALE:g}, got {true_cfg_scale!r}."
            )
        if latents.ndim != 6:
            raise ValueError(
                "WNM visual trajectory must have shape (batch, policy_steps+1, channels, frames, height, width), "
                f"got {tuple(latents.shape)}."
            )
        # The latent trajectory contains one more endpoint than timesteps.
        expected_timesteps = (latents.shape[0], latents.shape[1] - 1)
        if timesteps.ndim != 2 or tuple(timesteps.shape) != expected_timesteps:
            raise ValueError(
                f"WNM visual timesteps must have shape {expected_timesteps}, got {tuple(timesteps.shape)}."
            )
        if not 0 <= step < timesteps.shape[1]:
            raise IndexError(f"WNM denoising step {step} is outside [0, {timesteps.shape[1]}).")
        replay_transitions = int(timesteps.shape[1])
        if replay_transitions == _DEFAULT_INFERENCE_STEPS:
            prediction_step = _DIT_PREDICTION_SOURCE_STEPS[step]
        elif replay_transitions == 1:
            if step != 0:
                raise IndexError(f"WNM compact replay exposes only local step 0, got {step}.")
            _validate_compact_replay_transitions(micro_batch, batch_size=latents.shape[0])
            # The trainer gathered each row's selected source state and exact
            # scheduler timestep into local slot zero.
            prediction_step = 0
        else:
            raise ValueError(
                "WNM replay requires either the deployed 16-transition trajectory or a "
                "validated layer-conditioned T=1 source-transition replay, "
                f"got {replay_transitions} transitions."
            )
        hidden_states = latents[:, prediction_step].detach()
        batch_size, channels, frames, height, width = hidden_states.shape
        inner_module = _unwrap_module(module)
        expected_channels = getattr(inner_module, "in_dim", None)
        if expected_channels is not None and channels != int(expected_channels):
            raise ValueError(
                f"WNM latent channels do not match CausalWanModel.in_dim: {channels} != {expected_channels}."
            )

        patch_size = tuple(getattr(inner_module, "patch_size", (1, 2, 2)))
        if len(patch_size) != 3 or any(size <= 0 for size in patch_size):
            raise ValueError(f"WNM CausalWanModel has invalid patch_size={patch_size!r}.")
        if frames % patch_size[0] or height % patch_size[1] or width % patch_size[2]:
            raise ValueError(f"WNM latent shape {(frames, height, width)} is not divisible by patch_size={patch_size}.")
        seq_len = (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])

        # Rollout trajectories are commonly serialized as float32 even though
        # the 5B actor runs in bf16.  Replay every floating input in the live
        # joint DiT's dtype while retaining the accelerator device selected by
        # the engine for the replay trajectory.
        device = hidden_states.device
        floating_dtype = _compute_dtype(module)
        hidden_states = hidden_states.to(device=device, dtype=floating_dtype)

        def prepare_context(
            embeds: torch.Tensor | None,
            embeds_mask: torch.Tensor | None,
            *,
            embeds_name: str,
            mask_name: str,
        ) -> torch.Tensor:
            if embeds is None:
                raise KeyError(f"WNM CFG replay requires `{embeds_name}` from rollout encoding.")
            context_value = (
                _require_tensor(embeds, embeds_name, batch_size=batch_size)
                .detach()
                .to(device=device, dtype=floating_dtype)
            )
            if context_value.ndim != 3:
                raise ValueError(
                    f"WNM {embeds_name} must have shape (batch, tokens, dim), got {tuple(context_value.shape)}."
                )
            mask_value = None
            if embeds_mask is not None:
                mask_value = _require_tensor(embeds_mask, mask_name, batch_size=batch_size)
                if tuple(mask_value.shape) != tuple(context_value.shape[:2]):
                    raise ValueError(
                        f"WNM {mask_name} shape {tuple(mask_value.shape)} does not match "
                        f"embeddings {tuple(context_value.shape[:2])}."
                    )
            expected_text_len = getattr(inner_module, "text_len", None)
            if expected_text_len is not None:
                expected_text_len = int(expected_text_len)
                if context_value.shape[1] > expected_text_len:
                    raise ValueError(
                        f"WNM {embeds_name} length exceeds CausalWanModel.text_len: "
                        f"{context_value.shape[1]} > {expected_text_len}."
                    )
                if context_value.shape[1] < expected_text_len:
                    padding = expected_text_len - context_value.shape[1]
                    context_value = torch.nn.functional.pad(context_value, (0, 0, 0, padding), value=0)
                    if mask_value is not None:
                        mask_value = torch.nn.functional.pad(mask_value, (0, padding), value=0)
            if mask_value is not None:
                context_value = context_value * mask_value.detach().to(
                    device=context_value.device, dtype=context_value.dtype
                ).unsqueeze(-1)
            return context_value

        context = prepare_context(
            prompt_embeds,
            prompt_embeds_mask,
            embeds_name="prompt_embeds",
            mask_name="prompt_embeds_mask",
        )
        negative_context = prepare_context(
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            embeds_name="negative_prompt_embeds",
            mask_name="negative_prompt_embeds_mask",
        )

        missing_conditions = [key for key in cls.replay_condition_keys if key not in micro_batch]
        if missing_conditions:
            raise KeyError(f"WNM replay is missing condition tensors: {missing_conditions}.")

        def frozen_condition(name: str, *, dtype: torch.dtype | None = floating_dtype) -> torch.Tensor:
            value = _require_tensor(micro_batch[name], name, batch_size=batch_size).detach().to(device=device)
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype=dtype)
            return value

        clean_x = None
        if cls.use_clean_x_condition:
            clean_x = frozen_condition("clean_x")
            if tuple(clean_x.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    f"WNM clean_x must match the current visual latent shape {tuple(hidden_states.shape)}, "
                    f"got {tuple(clean_x.shape)}."
                )
        past_condition = cls._prepare_past_condition(
            frozen_condition,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            inner_module=inner_module,
        )

        # Preserve the exact floating schedule value used by rollout. Stage-1/2
        # trained both modalities with flow-matching timesteps; truncating only
        # the visual branch makes an otherwise identical trace unreplayable.
        video_timestep = timesteps[:, prediction_step].detach().to(device=device, dtype=torch.float32)
        video_timestep = video_timestep[:, None].expand(batch_size, frames)
        clip_feature = frozen_condition("clip_feature")
        if clip_feature.ndim != 3:
            raise ValueError(f"WNM clip_feature must have shape (batch, tokens, dim), got {tuple(clip_feature.shape)}.")
        image_condition = frozen_condition("y")
        if image_condition.ndim != 5:
            raise ValueError(
                "WNM y image condition must have shape (batch, channels, frames, height, width), "
                f"got {tuple(image_condition.shape)}."
            )
        state = frozen_condition("state")
        if state.ndim != 3:
            raise ValueError(f"WNM state must have shape (batch, blocks, state_dim), got {tuple(state.shape)}.")
        num_frame_per_block = int(getattr(inner_module, "num_frame_per_block", 1))
        if (frames - 1) % num_frame_per_block != 0:
            raise ValueError(
                f"WNM target frames after the conditioning frame must divide into full blocks: "
                f"frames={frames}, num_frame_per_block={num_frame_per_block}."
            )
        expected_blocks = (frames - 1) // num_frame_per_block
        if state.shape[1] != expected_blocks:
            raise ValueError(
                f"WNM state block count must be {expected_blocks} for {frames} latent frames and "
                f"num_frame_per_block={num_frame_per_block}, got {state.shape[1]}."
            )
        expected_state_dim = getattr(inner_module, "max_state_dim", None)
        if expected_state_dim is not None and state.shape[2] != int(expected_state_dim):
            raise ValueError(
                f"WNM state dim does not match CausalWanModel.max_state_dim: {state.shape[2]} != {expected_state_dim}."
            )
        embodiment_id = None
        if cls.uses_embodiment_condition:
            provided_embodiment_id = frozen_condition("embodiment_id", dtype=None)
            if provided_embodiment_id.ndim not in (1, 2) or (
                provided_embodiment_id.ndim == 2 and provided_embodiment_id.shape[1] != 1
            ):
                raise ValueError(
                    "WNM embodiment_id must have shape (batch,) or (batch, 1), "
                    f"got {tuple(provided_embodiment_id.shape)}."
                )
            # The WNM-2D checkpoint contains only category 0. Keep that
            # checkpoint contract explicit at the actor boundary.
            embodiment_id = torch.zeros(batch_size, device=device, dtype=torch.long)

        # The generic WAM base validates transition counts. Add WNM's
        # architecture-specific action horizon/dimension checks when the
        # replay trajectory is already present in this micro-batch.
        action_trajectory = micro_batch.get("all_action_latents", None)
        if action_trajectory is not None:
            action_trajectory = _require_tensor(action_trajectory, "all_action_latents", batch_size=batch_size)
            if action_trajectory.ndim != 4:
                raise ValueError(
                    "WNM action trajectory must have shape "
                    "(batch, policy_steps+1, action_horizon, action_dim), "
                    f"got {tuple(action_trajectory.shape)}."
                )
            expected_action_horizon = expected_blocks * int(inner_module.num_action_per_block)
            expected_action_dim = int(inner_module.action_dim)
            expected_action_shape = (
                batch_size,
                latents.shape[1],
                expected_action_horizon,
                expected_action_dim,
            )
            if tuple(action_trajectory.shape) != expected_action_shape:
                raise ValueError(
                    f"WNM action trajectory must have shape {expected_action_shape}, "
                    f"got {tuple(action_trajectory.shape)}."
                )

        model_inputs = {
            "x": hidden_states,
            "timestep": video_timestep,
            "context": context,
            "seq_len": seq_len,
            "clip_feature": clip_feature,
            "y": image_condition,
            "state": state,
            cls.past_condition_key: past_condition,
        }
        if embodiment_id is not None:
            model_inputs["embodiment_id"] = embodiment_id
        if clean_x is not None:
            model_inputs["clean_x"] = clean_x
        negative_model_inputs = dict(model_inputs)
        negative_model_inputs["context"] = negative_context
        return model_inputs, negative_model_inputs

    @classmethod
    def inject_action_inputs(
        cls,
        model_inputs: dict[str, torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> dict[str, torch.Tensor]:
        """Replay the cached action prediction from its original source state."""

        action_timesteps = _require_tensor(
            micro_batch.get("all_action_timesteps"),
            "all_action_timesteps",
            batch_size=model_inputs["x"].shape[0],
        )
        if action_timesteps.ndim != 2:
            raise ValueError(
                f"WNM action timesteps must have shape (batch, policy_steps), got {tuple(action_timesteps.shape)}."
            )
        if action_timesteps.shape[1] == 1:
            if step != 0:
                raise IndexError(f"WNM compact action replay exposes only local step 0, got {step}.")
            _validate_compact_replay_transitions(
                micro_batch,
                batch_size=model_inputs["x"].shape[0],
            )
            replay_step = 0
        elif not 0 <= step < len(_DIT_PREDICTION_SOURCE_STEPS):
            raise IndexError(f"WNM action replay step {step} is outside [0, {len(_DIT_PREDICTION_SOURCE_STEPS)}).")
        else:
            replay_step = _DIT_PREDICTION_SOURCE_STEPS[step]
        model_inputs = super().inject_action_inputs(
            model_inputs,
            micro_batch,
            replay_step,
        )
        return model_inputs

    @classmethod
    def forward_world_action(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply deployed CFG to video while keeping the conditional action head."""

        if negative_model_inputs is None:
            raise ValueError("WNM deployed CFG replay requires negative_model_inputs.")
        conditional_video, conditional_action = super().forward_world_action(
            module,
            model_config,
            model_inputs,
            None,
        )
        negative_inputs = dict(negative_model_inputs)
        # The generic WAM dispatcher injects action state only into the
        # positive inputs. Both CFG branches must see the same action state.
        negative_inputs["action"] = model_inputs["action"]
        negative_inputs["timestep_action"] = model_inputs["timestep_action"]
        unconditional_video, _ = super().forward_world_action(
            module,
            model_config,
            negative_inputs,
            None,
        )
        cfg_scale = float(model_config.pipeline.true_cfg_scale)
        guided_video = unconditional_video + cfg_scale * (conditional_video - unconditional_video)
        return guided_video, conditional_action


__all__ = ["WNM2D"]
