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

"""vLLM-Omni rollout adapter for WNM-2D's joint world/action model.

WNM is not an upstream Diffusers or vLLM-Omni pipeline. vLLM-Omni's
custom-pipeline hook is used as the serving shell, while the local WNM
checkpoint owns input transforms, frozen encoders, VAE, and the joint
``CausalWanModel``. Only the joint DiT is synchronized from the FSDP actor;
all frozen rollout components stay resident in the rollout worker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.wnm_shared.batch1_equivalent import (
    install_batch1_equivalent_action_encoder,
)
from verl_omni.pipelines.wnm_shared.rollout_acceleration import (
    _configure_explicit_sdpa_runtime,
    _install_rollout_dit_acceleration,
    _validate_explicit_sdpa,
)
from verl_omni.pipelines.wnm_shared.rollout_batching import (
    _clone_transform_value,
    _install_repeated_conditioning_encoder_dedup,
    _install_rollout_transform_dedup,
)
from verl_omni.pipelines.wnm_shared.rollout_common import (
    _DEFAULT_CFG_SCALE,
    _DEFAULT_INFERENCE_STEPS,
    _DEPLOYED_ACTION_DIM,
    _DIT_PREDICTION_SOURCE_STEPS,
    _DIT_STEP_MASK,
    _LAYER_CREDIT_FIELDS,
    _PER_REQUEST_TRANSFORM_FIELDS,
    _deployed_action_policy_mask,
    _env_flag,
    build_shifted_schedule,
    derive_rollout_subseed,
)
from verl_omni.pipelines.wnm_shared.rollout_rng import (
    _install_per_request_rng_rollout,
    _randn_per_request,
)
from verl_omni.utils.action_chunk_credit import (
    action_chunk_credit_enabled,
    action_chunk_size,
)

logger = logging.getLogger(__name__)

# Preserve the public helper surface used by probes and downstream tests after
# splitting the implementation into focused modules.
__all__ = [
    "_DEFAULT_INFERENCE_STEPS",
    "_deployed_action_policy_mask",
    "_randn_per_request",
    "build_shifted_schedule",
    "derive_rollout_subseed",
]

_ARCHITECTURE = "WNM2D"
_ACTOR_WEIGHT_PREFIX = "transformer."
_VLN_WEIGHT_PREFIXES = (
    "action_head.model.base_model.model.",
    "action_head.model.",
)


def _normalize_joint_weight_name(name: str) -> str | None:
    """Map actor/full-VLN names to the standalone joint-DiT key space."""

    if name.startswith(_ACTOR_WEIGHT_PREFIX):
        return name[len(_ACTOR_WEIGHT_PREFIX) :]
    for prefix in _VLN_WEIGHT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :].replace(".base_layer.", ".")
    return None


def _as_numpy_video(value: Any) -> np.ndarray:
    """Convert one video clip to contiguous THWC uint8 without file lookup."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        pass
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        frames = []
        for frame in value:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            elif not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            frames.append(frame)
        value = np.stack(frames, axis=0)
    else:
        value = np.asarray(value)

    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"WNM rollout video must be rank-4, got shape {value.shape}.")
    if value.size == 0:
        raise ValueError("WNM rollout video must not be empty.")

    channel_sizes = (1, 3, 4)
    # Prefer an explicit leading/second channel axis when the leading time
    # dimension is not itself channel-sized. This avoids treating a narrow
    # TCHW frame width (for example W=4 in unit tests) as an alpha channel.
    if value.shape[0] not in channel_sizes and value.shape[1] in channel_sizes:
        video = np.moveaxis(value, 1, -1)
    elif value.shape[0] in channel_sizes and value.shape[1] not in channel_sizes:
        video = np.moveaxis(value, 0, -1)
    elif value.shape[-1] in channel_sizes:
        video = value
    elif value.shape[1] in channel_sizes:
        video = np.moveaxis(value, 1, -1)
    elif value.shape[0] in channel_sizes:
        video = np.moveaxis(value, 0, -1)
    else:
        raise ValueError(
            f"WNM rollout video must use THWC, TCHW, or CTHW layout with RGB channels; got shape {value.shape}."
        )

    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    else:
        # Strip an alpha channel after resolving the layout.
        video = video[..., :3]

    if video.dtype != np.uint8:
        video = video.astype(np.float32, copy=False)
        if not np.isfinite(video).all():
            raise ValueError("WNM rollout video contains non-finite values.")
        minimum = float(video.min())
        maximum = float(video.max())
        if minimum >= -1.0 and maximum <= 1.0:
            if minimum < 0.0:
                video = (video + 1.0) * 127.5
            else:
                video = video * 255.0
        elif minimum < 0.0 or maximum > 255.0:
            raise ValueError(
                f"WNM rollout video values must be uint8, [0,1], [-1,1], or [0,255]; got range [{minimum}, {maximum}]."
            )
        video = np.rint(video).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(video)


def _merge_context_video(value: Any, *, expected_clip_frames: int) -> np.ndarray:
    """Return the Stage-1 layout ``[strict-past clip | target clip]``."""

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | np.ndarray | torch.Tensor):
        if len(value) == 2:
            first = _as_numpy_video(value[0])
            second = _as_numpy_video(value[1])
            if first.shape[0] == expected_clip_frames and second.shape[0] == expected_clip_frames:
                value = np.concatenate((first, second), axis=0)

    video = _as_numpy_video(value)
    expected_total = 2 * expected_clip_frames
    if video.shape[0] != expected_total:
        raise ValueError(
            "WNM parquet input must contain the same strict-past/target layout used by Stage-1: "
            f"expected {expected_total} frames ({expected_clip_frames}+{expected_clip_frames}), "
            f"got {video.shape[0]}."
        )
    return video


def _require_rollout_inputs(extra_args: Mapping[str, Any]) -> Mapping[str, Any]:
    value = extra_args.get("rollout_extra_args")
    if not isinstance(value, Mapping):
        raise KeyError(
            "WNM rollout requires parquet extra_info['rollout_extra_args']; "
            "it must contain at least `instruction` and `state`."
        )
    missing = [name for name in ("instruction", "state") if name not in value]
    if missing:
        raise KeyError(f"WNM rollout_extra_args is missing {missing}.")
    return value


def _is_dummy_request(req: OmniDiffusionRequest) -> bool:
    """Recognize both vLLM-Omni dummy request forms used during warm-up."""

    if getattr(req, "request_id", None) == DUMMY_DIFFUSION_REQUEST_ID:
        return True
    prompt = getattr(req, "prompt", None)
    return isinstance(prompt, Mapping) and prompt.get("prompt") == "dummy run"


def _floating_module_reference(module: torch.nn.Module) -> torch.Tensor | None:
    """Find the live device/dtype reference for a frozen runtime component."""

    for values in (module.parameters(), module.buffers()):
        for value in values:
            if torch.is_floating_point(value):
                return value
    return None


def _checkpoint_video_transform_manifest(policy: Any) -> tuple[dict[str, Any], ...]:
    """Return the resolved checkpoint video preprocessing as loggable data."""

    train_cfg = getattr(policy, "train_cfg", None)
    transforms_by_embodiment = getattr(train_cfg, "transforms", None)
    embodiment = getattr(getattr(policy, "embodiment_tag", None), "value", None)
    if transforms_by_embodiment is None or embodiment is None:
        return ()
    transform_config = transforms_by_embodiment[embodiment]
    manifest = []
    recorded_fields = (
        "scale",
        "height",
        "width",
        "interpolation",
        "antialias",
        "brightness",
        "contrast",
        "saturation",
        "hue",
    )
    for transform in transform_config.transforms:
        target = str(transform.get("_target_", type(transform).__name__))
        if ".Video" not in target:
            continue
        entry: dict[str, Any] = {"transform": target.rsplit(".", 1)[-1]}
        for name in recorded_fields:
            value = transform.get(name, None)
            if value is not None:
                entry[name] = value
        if entry["transform"] == "VideoCrop":
            # WNM3DInferencePolicy switches the composed transform to eval mode, and
            # VideoCrop implements eval as a deterministic center crop.
            entry["eval_mode"] = "center"
        manifest.append(entry)
    return tuple(manifest)


def _restore_checkpoint_video_metadata(
    policy: Any,
    *,
    checkpoint_dir: Path,
    embodiment_value: str,
    video_key: str,
) -> tuple[tuple[int, int] | None, tuple[str, ...]]:
    """Mirror GN0 eval's raw-metadata restore before checkpoint transforms."""

    # CPU contract tests use a deliberately minimal fake policy/checkpoint.
    if not hasattr(policy, "train_cfg"):
        return None, ()

    metadata_path = checkpoint_dir / "experiment_cfg" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"WNM checkpoint preprocessing requires {metadata_path} so parquet/source dimensions are not guessed."
        )
    from gammanav.vln.data.schema import DatasetMetadata

    with metadata_path.open("r", encoding="utf-8") as stream:
        all_metadata = json.load(stream)
    metadata = DatasetMetadata.model_validate(all_metadata[embodiment_value])

    restored = []
    for transform in getattr(policy.eval_transform, "transforms", []):
        if video_key not in getattr(transform, "apply_to", []):
            continue
        if not hasattr(transform, "original_resolutions"):
            continue
        if transform.__class__.__name__ == "VideoCrop":
            transform.height = None
            transform.width = None
        transform.set_metadata(metadata)
        transform.eval()
        restored.append(transform.__class__.__name__)

    metadata_key = video_key.removeprefix("video.")
    source_resolution = tuple(metadata.modalities.video[metadata_key].resolution)
    return source_resolution, tuple(restored)


def _conditioning_value_fingerprint(value: Any) -> tuple[Any, ...]:
    """Build a compact, deterministic identity for one raw conditioning value."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest = hashlib.blake2b(memoryview(contiguous).cast("B"), digest_size=16).digest()
        return ("array", tuple(contiguous.shape), contiguous.dtype.str, digest)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bool | int | float | np.generic):
        return ("scalar", type(value).__name__, value.item() if isinstance(value, np.generic) else value)
    raise TypeError(f"WNM conditioning grouping does not support raw value type {type(value).__name__}.")


def _request_conditioning_group_plan(
    requests: Sequence[OmniDiffusionRequest],
    observations: Sequence[Mapping[str, Any]],
    *,
    video_key: str,
) -> dict[str, Any] | None:
    """Group scheduler requests by deterministic prompt conditioning.

    Production parquet requests carry the exact MP4 path, which avoids hashing
    tens of megabytes of repeated RGB data.  Small CPU tests and generic
    callers fall back to a content fingerprint.
    """

    if len(requests) != len(observations):
        raise ValueError(f"WNM request/observation count mismatch: {len(requests)} vs {len(observations)}.")
    if len(requests) <= 1:
        return None

    representatives: list[int] = []
    group_index: list[int] = []
    identity_to_group: dict[tuple[Any, ...], int] = {}
    for request_index, (request, observation) in enumerate(zip(requests, observations, strict=True)):
        extra_args = request.sampling_params.extra_args or {}
        rollout_inputs = _require_rollout_inputs(extra_args)
        exact_video_path = rollout_inputs.get("exact_context_video_path")
        fields: list[tuple[str, tuple[Any, ...]]] = []
        for key in sorted(observation):
            if key in _PER_REQUEST_TRANSFORM_FIELDS:
                continue
            if key == video_key and exact_video_path is not None:
                fingerprint = ("exact_video_path", os.path.abspath(os.fspath(exact_video_path)))
            else:
                fingerprint = _conditioning_value_fingerprint(observation[key])
            fields.append((key, fingerprint))
        identity = tuple(fields)
        group = identity_to_group.get(identity)
        if group is None:
            group = len(representatives)
            identity_to_group[identity] = group
            representatives.append(request_index)
        group_index.append(group)

    if len(representatives) == len(requests):
        return None
    return {
        "batch_size": len(requests),
        "representative_indices": tuple(representatives),
        "group_index": tuple(group_index),
    }


def _collate_raw_observations(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate homogeneous observations without copying repeated conditioning."""

    if not observations:
        raise ValueError("WNM request batch must not be empty.")
    expected_keys = tuple(observations[0])
    if any(tuple(observation) != expected_keys for observation in observations[1:]):
        raise ValueError("WNM request batch contains different observation fields.")
    repeated_conditioning = len(observations) > 1 and _env_flag("WAM_ROLLOUT_DEDUP_TRANSFORM", False)
    for key in expected_keys:
        if not repeated_conditioning or key in _PER_REQUEST_TRANSFORM_FIELDS:
            continue
        first = observations[0][key]
        for observation in observations[1:]:
            value = observation[key]
            if isinstance(first, np.ndarray) and isinstance(value, np.ndarray):
                equal = value.shape == first.shape and value.dtype == first.dtype and np.array_equal(value, first)
            else:
                equal = type(value) is type(first) and value == first
            if not equal:
                repeated_conditioning = False
                break
        if not repeated_conditioning:
            break
    collated: dict[str, Any] = {}
    for key in expected_keys:
        values = [observation[key] for observation in observations]
        if all(isinstance(value, np.ndarray) for value in values):
            try:
                first = values[0]
                if repeated_conditioning and key not in _PER_REQUEST_TRANSFORM_FIELDS:
                    # A read-only stride-zero view is safe here: the dedup
                    # transform copies item zero before checkpoint transforms.
                    collated[key] = np.broadcast_to(first, (len(values), *first.shape))
                else:
                    collated[key] = np.stack(values, axis=0)
            except ValueError as exc:
                raise ValueError(f"WNM request batch field {key!r} has incompatible shapes.") from exc
        elif all(isinstance(value, str) for value in values):
            collated[key] = np.asarray(values)
        elif all(isinstance(value, bool | np.bool_) for value in values):
            collated[key] = np.asarray(values, dtype=np.bool_)
        elif all(isinstance(value, int | np.integer) and not isinstance(value, bool) for value in values):
            collated[key] = np.asarray(values, dtype=np.int64)
        elif all(isinstance(value, float | np.floating) for value in values):
            collated[key] = np.asarray(values, dtype=np.float32)
        else:
            raise TypeError(
                f"WNM request batch field {key!r} has unsupported value types: "
                f"{sorted({type(value).__name__ for value in values})}."
            )
    return collated


@VllmOmniPipelineBase.register(_ARCHITECTURE, algorithm="dance_grpo")
class WNM2DPipelineWithLogProb(torch.nn.Module):
    """Serve joint video/action Dance-SDE rollouts through vLLM-Omni."""

    architecture = _ARCHITECTURE

    supports_request_batch = True
    replay_condition_key = "past_clean_x"
    replay_condition_rank = 5
    replay_condition_tail_shape: tuple[int, ...] | None = None
    requires_clean_x = True
    uses_embodiment_condition = True

    @staticmethod
    def _configure_attention_runtime() -> None:
        _configure_explicit_sdpa_runtime(component="WNM2D rollout")

    @staticmethod
    def _validate_attention_runtime(module: torch.nn.Module) -> None:
        _validate_explicit_sdpa(module, component="WNM2D rollout")

    @staticmethod
    def _validate_loaded_action_head(action_head: Any) -> None:
        del action_head

    @staticmethod
    def _install_action_head_rollout(action_head: Any) -> None:
        _install_per_request_rng_rollout(action_head)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.od_config = od_config

        self._configure_attention_runtime()

        checkpoint_dir = Path(od_config.model).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"WNM rollout requires a local full-VLN checkpoint directory: {checkpoint_dir}. "
                "Hub fallback is intentionally unsupported."
            )

        # This worker is a pure rollout worker.  The flag selects WNM's
        # stateless joint Dance-SDE path instead of its deployment UniPC path.
        os.environ["WNM_ROLLOUT"] = "true"

        try:
            from gammanav.vln.data.schema import EmbodimentTag
            from gammanav.vln.model.wnm_3d.inference_policy import WNM3DInferencePolicy
        except ImportError as exc:
            raise ImportError(
                "WNM rollout requires the local GammaNav checkout on PYTHONPATH; "
                "dependencies and checkpoints are never downloaded automatically."
            ) from exc

        device = get_local_device()
        tokenizer_override = os.environ.get("WNM_ROLLOUT_TOKENIZER_PATH") or None
        policy = WNM3DInferencePolicy(
            embodiment_tag=EmbodimentTag.INTERIORGS,
            model_path=os.fspath(checkpoint_dir),
            device=device,
            tokenizer_path_override=tokenizer_override,
        )
        policy.trained_model.eval()
        action_head = policy.trained_model.action_head
        self._validate_loaded_action_head(action_head)
        install_batch1_equivalent_action_encoder(
            action_head.model,
            component=f"{self.architecture} rollout",
        )
        self._install_action_head_rollout(action_head)
        _install_rollout_transform_dedup(policy)
        _install_repeated_conditioning_encoder_dedup(action_head)

        # Bypass nn.Module registration for the policy/action-head containers;
        # register each runtime component once under the names expected by
        # vLLM weight management.  Actor updates address only ``transformer``.
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_action_head", action_head)
        self.transformer = action_head.model
        self._validate_attention_runtime(self.transformer)
        _install_rollout_dit_acceleration(self.transformer)
        self.text_encoder = action_head.text_encoder
        self.image_encoder = action_head.image_encoder
        self.vae = action_head.vae
        self.device = device

        self._video_key = policy.modality_configs.video.modality_keys[0]
        self._state_key = policy.modality_configs.state.modality_keys[0]
        self._language_key = policy.modality_configs.language.modality_keys[0]
        source_resolution, restored_video_transforms = _restore_checkpoint_video_metadata(
            policy,
            checkpoint_dir=checkpoint_dir,
            embodiment_value=str(getattr(EmbodimentTag.INTERIORGS, "value", EmbodimentTag.INTERIORGS)),
            video_key=self._video_key,
        )
        self._video_transform_manifest = _checkpoint_video_transform_manifest(policy)
        self._logged_input_video_shapes: set[tuple[int, ...]] = set()
        self._bulk_cpu_output = _env_flag("WAM_ROLLOUT_BULK_CPU_OUTPUT", False)
        logger.warning(
            "WNM rollout bulk CPU output transport: enabled=%s",
            self._bulk_cpu_output,
        )
        logger.warning(
            "WNM checkpoint preprocessing: checkpoint=%s, parquet/source_resolution_wh=%s, "
            "clip_layout=%d+%d frames, restored_eval_transforms=%s, transforms=%s",
            checkpoint_dir,
            source_resolution,
            int(action_head.num_frames),
            int(action_head.num_frames),
            restored_video_transforms,
            self._video_transform_manifest,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]] | Mapping[str, torch.Tensor],
    ) -> set[str]:
        """Load actor ``transformer.*`` updates into the joint CausalWanModel.

        The full VLN and frozen encoders are already loaded locally by
        ``WNM3DInferencePolicy``. Initial checkpoint entries outside the joint DiT
        are acknowledged without a second copy; synchronized actor entries are
        checked and copied strictly.
        """

        if isinstance(weights, Mapping):
            weights = weights.items()

        destinations = dict(self.transformer.named_parameters())
        destinations.update(dict(self.transformer.named_buffers()))
        # ``WNM3DInferencePolicy`` loaded the complete local VLN before this hook is
        # called.  vLLM compares the returned names with this adapter's
        # registered names (``transformer.*``, ``text_encoder.*``, ...), not
        # with checkpoint names such as ``action_head.model.*``.  Report the
        # already initialized runtime modules using that canonical namespace;
        # otherwise vLLM falsely treats the entire self-loaded pipeline as
        # uninitialized.  Incoming actor weights are still validated and
        # copied strictly below.
        loaded: set[str] = {name for name, _ in self.named_parameters()}
        loaded.update(name for name, _ in self.named_buffers())
        for source_name, source_tensor in weights:
            normalized = _normalize_joint_weight_name(source_name)
            if normalized is None:
                continue
            destination = destinations.get(normalized)
            if destination is None:
                raise KeyError(
                    f"WNM rollout joint DiT has no parameter/buffer for synchronized weight {source_name!r} "
                    f"(normalized={normalized!r})."
                )
            if tuple(destination.shape) != tuple(source_tensor.shape):
                raise ValueError(
                    f"WNM synchronized weight shape mismatch for {source_name!r}: "
                    f"rollout={tuple(destination.shape)}, actor={tuple(source_tensor.shape)}."
                )
            with torch.no_grad():
                destination.copy_(source_tensor.to(device=destination.device, dtype=destination.dtype))
        return loaded

    @staticmethod
    def _sampling_value(sampling: Any, name: str, fallback: Any = None) -> Any:
        value = getattr(sampling, name, None)
        return fallback if value is None else value

    def _build_raw_observation(self, req: OmniDiffusionRequest) -> dict[str, Any]:
        # The scheduler batches requests, while each OmniDiffusionRequest owns
        # exactly one prompt (singular).
        custom_prompt = req.prompt
        if not isinstance(custom_prompt, Mapping):
            raise ValueError("WNM2D requires one mapping-valued custom prompt per vLLM request.")
        extra_args = req.sampling_params.extra_args or {}
        rollout_inputs = _require_rollout_inputs(extra_args)

        multi_modal_data = custom_prompt.get("multi_modal_data")
        if multi_modal_data is None:
            prompt_extra_args = custom_prompt.get("extra_args")
            if isinstance(prompt_extra_args, Mapping):
                multi_modal_data = prompt_extra_args.get("multi_modal_data")
        if not isinstance(multi_modal_data, Mapping):
            raise KeyError("WNM rollout requires one merged context video in multi_modal_data.")
        video_payload = multi_modal_data.get("video")
        if video_payload is None:
            video_payload = multi_modal_data.get("image")
        if video_payload is None:
            raise KeyError("WNM rollout multi_modal_data must contain `video` (or image-frame fallback).")

        # Vision helpers commonly wrap a single video in a one-element list.
        if isinstance(video_payload, Sequence) and not isinstance(
            video_payload, str | bytes | bytearray | np.ndarray | torch.Tensor
        ):
            if len(video_payload) == 1:
                video_payload = video_payload[0]

        merged_video = _merge_context_video(
            video_payload,
            expected_clip_frames=int(self._action_head.num_frames),
        )
        input_shape = tuple(int(size) for size in merged_video.shape)
        if input_shape not in self._logged_input_video_shapes:
            self._logged_input_video_shapes.add(input_shape)
            crop_scale = next(
                (float(step["scale"]) for step in self._video_transform_manifest if step["transform"] == "VideoCrop"),
                None,
            )
            crop_hw = None
            if crop_scale is not None:
                crop_hw = (
                    int(merged_video.shape[1] * crop_scale),
                    int(merged_video.shape[2] * crop_scale),
                )
            logger.warning(
                "WNM parquet video input: path=%s, decoded_thwc=%s, eval_center_crop_hw=%s, checkpoint_transforms=%s",
                rollout_inputs.get("exact_context_video_path"),
                input_shape,
                crop_hw,
                self._video_transform_manifest,
            )
        state = np.asarray(rollout_inputs["state"], dtype=np.float32)
        if state.ndim != 2:
            raise ValueError(f"WNM state must have shape (state_horizon, state_dim), got {state.shape}.")
        if state.size == 0 or not np.isfinite(state).all():
            raise ValueError("WNM state must be non-empty and contain only finite values.")

        instruction = str(rollout_inputs["instruction"])
        if not instruction.strip():
            raise ValueError("WNM instruction must be non-empty.")

        seed = getattr(req.sampling_params, "seed", None)
        if seed is None:
            raise ValueError(
                "WNM rollout requires an explicit per-request seed so repeated rollout.n samples "
                "cannot silently share one stochastic trajectory."
            )
        if isinstance(seed, bool):
            raise TypeError("WNM rollout seed must be an integer, not bool.")
        seed = int(seed)
        if not 0 <= seed < 2**63:
            raise ValueError(f"WNM rollout seed must be in [0, 2**63), got {seed}.")
        init_same_noise = bool(extra_args.get("init_same_noise", False))
        initial_noise_seed = extra_args.get("initial_noise_seed", None)
        if init_same_noise and initial_noise_seed is None:
            raise KeyError("WNM init_same_noise=true requires initial_noise_seed.")
        if initial_noise_seed is None:
            initial_noise_seed = seed
        if isinstance(initial_noise_seed, bool):
            raise TypeError("WNM initial_noise_seed must be an integer, not bool.")
        initial_noise_seed = int(initial_noise_seed)
        if not 0 <= initial_noise_seed < 2**63:
            raise ValueError(f"WNM initial_noise_seed must be in [0, 2**63), got {initial_noise_seed}.")
        target_prefix_frames = int(rollout_inputs.get("target_prefix_frames", 1))
        observation = {
            self._video_key: merged_video,
            self._state_key: state,
            self._language_key: instruction,
            "target_prefix_frames": target_prefix_frames,
            "rollout_seed": seed,
            "initial_noise_seed": initial_noise_seed,
            "init_same_noise": int(init_same_noise),
        }
        present_credit_fields = [name for name in _LAYER_CREDIT_FIELDS if name in extra_args]
        if present_credit_fields:
            if len(present_credit_fields) != len(_LAYER_CREDIT_FIELDS):
                missing = sorted(set(_LAYER_CREDIT_FIELDS) - set(present_credit_fields))
                raise KeyError(f"Incomplete layer-conditioned sampling metadata; missing {missing}.")
            for name in _LAYER_CREDIT_FIELDS:
                observation[name] = int(extra_args[name])
        return observation

    def _configure_sampling(self, req: OmniDiffusionRequest) -> bool:
        sampling = req.sampling_params
        requested_steps = int(
            self._sampling_value(sampling, "num_inference_steps", self._action_head.num_inference_steps)
        )
        if requested_steps != int(self._action_head.num_inference_steps):
            raise ValueError(
                "WNM rollout num_inference_steps must match the Stage-1/Stage-2 checkpoint config: "
                f"checkpoint={self._action_head.num_inference_steps}, request={requested_steps}."
            )

        extra_args = sampling.extra_args or {}
        self._action_head.wam_noise_level = float(extra_args.get("noise_level", self._action_head.wam_noise_level))
        action_noise_level = extra_args.get("action_noise_level", None)
        self._action_head.wam_action_noise_level = (
            self._action_head.wam_noise_level if action_noise_level is None else float(action_noise_level)
        )
        visual_sde_type = extra_args.get("sde_type", "dance_sde")
        action_sde_type = extra_args.get("action_sde_type", visual_sde_type)
        if visual_sde_type != "dance_sde" or action_sde_type != "dance_sde":
            raise ValueError(
                "WNM2D supports only dance_sde for joint rollout/replay parity; "
                f"got visual={visual_sde_type!r}, action={action_sde_type!r}."
            )
        return bool(extra_args.get("logprobs", True))

    def _decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        vae_reference = _floating_module_reference(self.vae)
        if vae_reference is not None:
            video_latents = video_latents.to(device=vae_reference.device, dtype=vae_reference.dtype)
        with torch.inference_mode():
            decoded = self.vae.decode(
                video_latents,
                tiled=self._action_head.tiled,
                tile_size=(self._action_head.tile_size_height, self._action_head.tile_size_width),
                tile_stride=(self._action_head.tile_stride_height, self._action_head.tile_stride_width),
            )
        # Reward-facing contract: B,T,C,H,W float32 in [0,1].  Replay continues
        # to use the latent trajectory from custom_output.
        if decoded.ndim != 5 or decoded.shape[1] != 3:
            raise ValueError(f"WNM VAE decode must return BCTHW RGB video, got shape {tuple(decoded.shape)}.")
        return ((decoded.float() + 1.0) * 0.5).clamp_(0.0, 1.0).permute(0, 2, 1, 3, 4).contiguous()

    @classmethod
    def _validate_rollout_contract(cls, model_pred: Mapping[str, Any], *, logprobs: bool) -> tuple[str, ...]:
        """Fail before transport when a rollout cannot be replayed by the actor."""

        required = (
            "actions",
            "all_latents",
            "all_timesteps",
            "all_action_latents",
            "all_action_timesteps",
            # Persist the exact sampler contract into actor replay. These are
            # not diagnostics: dropping any of them would allow rollout and
            # training to disagree silently about 8/16 caching, CFG, or SDE.
            "dit_prediction_source_steps",
            "num_dit_prediction_steps",
            "num_dit_forwards",
            "num_inference_steps",
            "true_cfg_scale",
            "noise_level",
            "action_noise_level",
            "action_policy_mask",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "clip_feature",
            "y",
            "state",
            cls.replay_condition_key,
        )
        if cls.uses_embodiment_condition:
            required += ("embodiment_id",)
        if cls.requires_clean_x:
            required += ("clean_x",)
        if logprobs:
            required += ("all_log_probs", "action_log_probs")
        missing = [name for name in required if model_pred.get(name) is None]
        if missing:
            raise KeyError(f"WNM strict WAM rollout did not produce required fields: {missing}.")
        non_tensors = [name for name in required if not isinstance(model_pred[name], torch.Tensor)]
        if non_tensors:
            raise TypeError(f"WNM rollout fields must be tensors: {non_tensors}.")

        visual_latents = model_pred["all_latents"]
        visual_timesteps = model_pred["all_timesteps"]
        action_latents = model_pred["all_action_latents"]
        action_timesteps = model_pred["all_action_timesteps"]
        actions = model_pred["actions"]
        action_policy_mask = model_pred["action_policy_mask"]
        if visual_latents.ndim != 6:
            raise ValueError(
                "WNM all_latents must have shape (batch, steps+1, channels, frames, height, width), "
                f"got {tuple(visual_latents.shape)}."
            )
        if visual_timesteps.ndim != 2:
            raise ValueError(f"WNM all_timesteps must have shape (batch, steps), got {tuple(visual_timesteps.shape)}.")
        if action_latents.ndim != 4 or actions.ndim != 3:
            raise ValueError(
                "WNM actions/all_action_latents must have shapes (batch, horizon, dim) and "
                "(batch, steps+1, horizon, dim); "
                f"got actions={tuple(actions.shape)}, trajectory={tuple(action_latents.shape)}."
            )
        if tuple(action_policy_mask.shape) != tuple(actions.shape):
            raise ValueError(
                "WNM action_policy_mask must match one full action state: "
                f"mask={tuple(action_policy_mask.shape)}, actions={tuple(actions.shape)}."
            )
        expected_action_policy_mask = torch.zeros_like(action_policy_mask, dtype=torch.bool)
        if action_policy_mask.shape[-1] < _DEPLOYED_ACTION_DIM:
            raise ValueError(
                "WNM action policy state does not contain all deployed "
                f"coordinates: action_dim={action_policy_mask.shape[-1]}."
            )
        expected_action_policy_mask[..., :_DEPLOYED_ACTION_DIM] = True
        if not torch.equal(action_policy_mask.to(torch.bool), expected_action_policy_mask):
            raise ValueError(
                "WNM action_policy_mask must select exactly the deployed "
                "[dx, dy, dyaw] coordinates and exclude padded dimensions."
            )
        if action_timesteps.ndim != 2:
            raise ValueError(
                f"WNM all_action_timesteps must have shape (batch, steps), got {tuple(action_timesteps.shape)}."
            )

        batch_size, policy_steps = visual_timesteps.shape
        if policy_steps != len(_DIT_PREDICTION_SOURCE_STEPS):
            raise ValueError(f"WNM deployed rollout must retain all 16 scheduler transitions; got {policy_steps}.")
        expected_source_steps = (
            torch.tensor(
                _DIT_PREDICTION_SOURCE_STEPS,
                dtype=model_pred["dit_prediction_source_steps"].dtype,
                device=model_pred["dit_prediction_source_steps"].device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        recorded_source_steps = model_pred["dit_prediction_source_steps"]
        if tuple(recorded_source_steps.shape) != tuple(expected_source_steps.shape) or not torch.equal(
            recorded_source_steps, expected_source_steps
        ):
            raise ValueError(
                f"WNM rollout produced a non-deployed DiT cache map; expected {list(_DIT_PREDICTION_SOURCE_STEPS)}."
            )
        scalar_contract = {
            "num_dit_prediction_steps": sum(_DIT_STEP_MASK),
            "num_dit_forwards": 2 * sum(_DIT_STEP_MASK),
            "num_inference_steps": len(_DIT_STEP_MASK),
        }
        for name, expected_value in scalar_contract.items():
            value = model_pred[name]
            if value.numel() != batch_size or not torch.all(value.reshape(-1) == expected_value):
                raise ValueError(f"WNM rollout contract mismatch for {name}: expected {expected_value}.")
        cfg = model_pred["true_cfg_scale"].detach().float().reshape(-1)
        if cfg.numel() != batch_size or not torch.allclose(
            cfg,
            torch.full_like(cfg, _DEFAULT_CFG_SCALE),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"WNM rollout must use deployed CFG={_DEFAULT_CFG_SCALE:g}.")
        for name in ("noise_level", "action_noise_level"):
            value = model_pred[name].detach().float().reshape(-1)
            if value.numel() != batch_size or not torch.all(torch.isfinite(value) & (value > 0)):
                raise ValueError(f"WNM rollout {name} must be finite, positive, and per-sample.")
        expected_visual_prefix = (batch_size, policy_steps + 1)
        expected_action_prefix = (batch_size, policy_steps + 1)
        if tuple(visual_latents.shape[:2]) != expected_visual_prefix:
            raise ValueError(
                "WNM visual trajectory must contain one more state than transitions; "
                f"expected prefix={expected_visual_prefix}, got {tuple(visual_latents.shape)}."
            )
        if tuple(action_timesteps.shape) != (batch_size, policy_steps):
            raise ValueError(
                "WNM visual/action timesteps must share batch and transition counts; "
                f"visual={tuple(visual_timesteps.shape)}, action={tuple(action_timesteps.shape)}."
            )
        if tuple(action_latents.shape[:2]) != expected_action_prefix:
            raise ValueError(
                "WNM action trajectory must contain one more state than transitions; "
                f"expected prefix={expected_action_prefix}, got {tuple(action_latents.shape)}."
            )
        if tuple(actions.shape) != (batch_size, *tuple(action_latents.shape[2:])):
            raise ValueError(
                "WNM actions must expose the full final policy action state; "
                f"actions={tuple(actions.shape)}, trajectory={tuple(action_latents.shape)}."
            )

        prompt_embeds = model_pred["prompt_embeds"]
        prompt_mask = model_pred["prompt_embeds_mask"]
        if prompt_embeds.ndim != 3 or tuple(prompt_mask.shape) != tuple(prompt_embeds.shape[:2]):
            raise ValueError(
                "WNM prompt_embeds/prompt_embeds_mask must have shapes (batch, tokens, dim) and "
                "(batch, tokens); "
                f"got embeds={tuple(prompt_embeds.shape)}, mask={tuple(prompt_mask.shape)}."
            )
        negative_prompt_embeds = model_pred["negative_prompt_embeds"]
        negative_prompt_mask = model_pred["negative_prompt_embeds_mask"]
        if (
            negative_prompt_embeds.ndim != 3
            or tuple(negative_prompt_embeds.shape) != tuple(prompt_embeds.shape)
            or tuple(negative_prompt_mask.shape) != tuple(negative_prompt_embeds.shape[:2])
        ):
            raise ValueError(
                "WNM negative prompt embeddings must match positive embeddings and use a "
                "(batch, tokens) mask; "
                f"positive={tuple(prompt_embeds.shape)}, negative={tuple(negative_prompt_embeds.shape)}, "
                f"negative_mask={tuple(negative_prompt_mask.shape)}."
            )
        expected_condition_ranks = {
            "clip_feature": 3,
            "y": 5,
            "state": 3,
            cls.replay_condition_key: cls.replay_condition_rank,
        }
        if cls.requires_clean_x:
            expected_condition_ranks["clean_x"] = 5
        for name, expected_rank in expected_condition_ranks.items():
            value = model_pred[name]
            if value.ndim != expected_rank:
                raise ValueError(
                    f"WNM replay condition {name!r} must be rank {expected_rank}, got {tuple(value.shape)}."
                )
        if cls.uses_embodiment_condition:
            embodiment_id = model_pred["embodiment_id"]
            if embodiment_id.ndim not in (1, 2) or (embodiment_id.ndim == 2 and embodiment_id.shape[1] != 1):
                raise ValueError(
                    f"WNM embodiment_id must have shape (batch,) or (batch, 1), got {tuple(embodiment_id.shape)}."
                )
        visual_state_shape = tuple(visual_latents.shape[:1]) + tuple(visual_latents.shape[2:])
        if cls.requires_clean_x and tuple(model_pred["clean_x"].shape) != visual_state_shape:
            raise ValueError(
                "WNM clean_x must match one visual trajectory state; "
                f"expected {visual_state_shape}, got {tuple(model_pred['clean_x'].shape)}."
            )
        if cls.replay_condition_tail_shape is not None:
            expected_replay_shape = (
                batch_size,
                *cls.replay_condition_tail_shape,
            )
            if tuple(model_pred[cls.replay_condition_key].shape) != expected_replay_shape:
                raise ValueError(
                    f"WNM replay condition {cls.replay_condition_key!r} must have shape "
                    f"{expected_replay_shape}, got {tuple(model_pred[cls.replay_condition_key].shape)}."
                )

        for name in required:
            value = model_pred[name]
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(
                    f"WNM rollout field {name!r} must retain batch dimension {batch_size}, got {tuple(value.shape)}."
                )
        if logprobs:
            visual_log_probs = model_pred["all_log_probs"]
            if tuple(visual_log_probs.shape) != (batch_size, policy_steps):
                raise ValueError(
                    "WNM all_log_probs must have shape "
                    f"{(batch_size, policy_steps)}, got {tuple(visual_log_probs.shape)}."
                )
            action_log_probs = model_pred["action_log_probs"]
            if action_chunk_credit_enabled():
                horizon = int(action_latents.shape[2])
                chunk_size = action_chunk_size()
                if horizon % chunk_size:
                    raise ValueError(
                        "WNM action horizon must be divisible by the configured "
                        f"chunk size: horizon={horizon}, chunk_size={chunk_size}."
                    )
                expected_action_shape = (
                    batch_size,
                    policy_steps,
                    horizon // chunk_size,
                )
            else:
                expected_action_shape = (batch_size, policy_steps)
            if tuple(action_log_probs.shape) != expected_action_shape:
                raise ValueError(
                    "WNM action_log_probs have the wrong temporal-credit shape: "
                    f"expected={expected_action_shape}, got={tuple(action_log_probs.shape)}."
                )
        return required

    def _forward_request_list(self, requests: Sequence[OmniDiffusionRequest]) -> list[DiffusionOutput]:
        """Run one homogeneous scheduler batch and split it back by request."""

        if not requests:
            raise ValueError("WNM request batch must not be empty.")
        dummy_flags = [_is_dummy_request(request) for request in requests]
        if any(dummy_flags):
            if not all(dummy_flags):
                raise ValueError("WNM cannot mix dummy and real requests in one scheduler batch.")
            return [DiffusionOutput(output=None, custom_output={}) for _ in requests]

        sampling_signatures = []
        raw_observations = []
        for request in requests:
            logprobs = self._configure_sampling(request)
            sampling_signatures.append(
                (
                    logprobs,
                    float(self._action_head.wam_noise_level),
                    float(self._action_head.wam_action_noise_level),
                )
            )
            raw_observations.append(self._build_raw_observation(request))
        if any(signature != sampling_signatures[0] for signature in sampling_signatures[1:]):
            raise ValueError(
                "WNM request batching requires identical logprob/noise settings; only per-request seeds may differ."
            )
        logprobs = sampling_signatures[0][0]
        conditioning_group_plan = _request_conditioning_group_plan(
            requests,
            raw_observations,
            video_key=self._video_key,
        )
        raw_observation = (
            raw_observations[0] if len(raw_observations) == 1 else _collate_raw_observations(raw_observations)
        )

        try:
            from tianshou.data import Batch
        except ImportError as exc:
            raise ImportError("WNM2D rollout requires WNM's tianshou runtime.") from exc

        self._action_head._verl_requested_conditioning_group_plan = conditioning_group_plan
        # Some checkpoint VLN schemas (notably WNM-3D) intentionally
        # filter RL-only fields while splitting normalized input into backbone
        # and action-head dictionaries. Preserve the exact request-batch seed
        # beside the action head for the duration of this synchronous worker
        # call. The VGGT rollout adapter consumes it only when the schema did
        # not pass ``rollout_seed`` through directly.
        self._action_head._verl_active_rollout_seed = _clone_transform_value(raw_observation["rollout_seed"])
        try:
            _, video_latents, model_pred = self._policy.lazy_joint_forward_causal(
                Batch(obs=raw_observation),
                return_model_pred=True,
            )
        finally:
            self._action_head._verl_active_rollout_seed = None
            self._action_head._verl_requested_conditioning_group_plan = None
            self._action_head._verl_repeated_conditioning_batch_size = None
            self._action_head._verl_conditioning_representative_indices = None
            self._action_head._verl_conditioning_group_index = None
        if not isinstance(model_pred, Mapping):
            raise TypeError(
                f"WNM strict WAM rollout must return a mapping of replay fields, got {type(model_pred).__name__}."
            )

        required = self._validate_rollout_contract(model_pred, logprobs=logprobs)
        if not isinstance(video_latents, torch.Tensor):
            raise TypeError(
                "WNM strict WAM rollout must return final video latents as a tensor, "
                f"got {type(video_latents).__name__}."
            )
        expected_video_shape = tuple(model_pred["all_latents"].shape[:1]) + tuple(model_pred["all_latents"].shape[2:])
        if tuple(video_latents.shape) != expected_video_shape:
            raise ValueError(
                "WNM final video latent must match one visual trajectory state; "
                f"expected {expected_video_shape}, got {tuple(video_latents.shape)}."
            )
        if video_latents.shape[0] != len(requests):
            raise ValueError(
                "WNM model output batch does not match the vLLM request batch: "
                f"model={video_latents.shape[0]}, requests={len(requests)}."
            )
        decoded_video = self._decode_video(video_latents)
        if self._bulk_cpu_output:
            decoded_video_cpu = decoded_video.detach().cpu()
            model_pred_cpu = {name: model_pred[name].detach().cpu() for name in required}
            # Each request may be serialized independently after this method
            # returns. Clone CPU slices so one response cannot retain or ship
            # the complete batch backing storage.
            return [
                DiffusionOutput(
                    output=decoded_video_cpu[index : index + 1].clone(),
                    custom_output={name: model_pred_cpu[name][index : index + 1].clone() for name in required},
                    to_cpu=False,
                )
                for index in range(len(requests))
            ]
        return [
            DiffusionOutput(
                output=decoded_video[index : index + 1],
                custom_output={name: model_pred[name][index : index + 1].detach() for name in required},
                to_cpu=True,
            )
            for index in range(len(requests))
        ]

    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        **_: Any,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Generate singleton or request-batched replayable WAM trajectories."""

        if isinstance(req, DiffusionRequestBatch):
            return self._forward_request_list(req.requests)
        return self._forward_request_list([req])[0]


__all__ = ["WNM2DPipelineWithLogProb"]
