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

"""Transform and conditioning deduplication for WNM rollout."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping, Sequence
from types import MethodType
from typing import Any

import numpy as np
import torch

from verl_omni.pipelines.wnm_shared.rollout_acceleration import (
    _assert_compiled_output_parity,
    _clone_tensor_tree,
)
from verl_omni.pipelines.wnm_shared.rollout_common import (
    _LAYER_CREDIT_FIELDS,
    _PER_REQUEST_TRANSFORM_FIELDS,
    _env_flag,
)

logger = logging.getLogger(__name__)


def _clone_transform_value(value: Any) -> Any:
    """Copy transform inputs because checkpoint transforms mutate their mappings."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_transform_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_transform_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_transform_value(item) for item in value]
    return value


def _transform_input_batch_size(observation: Mapping[str, Any]) -> int | None:
    """Return the common leading batch size used by raw rollout observations."""

    sizes = {
        int(value.shape[0])
        for value in observation.values()
        if isinstance(value, torch.Tensor | np.ndarray) and value.ndim > 0
    }
    if len(sizes) != 1:
        return None
    batch_size = sizes.pop()
    return batch_size if batch_size > 1 else None


def _batch_values_are_repeated(value: Any, batch_size: int) -> bool:
    """Check that a batch-aligned value repeats item zero exactly."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            return False
        return bool(torch.equal(value, value[0:1].expand_as(value)))
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != batch_size:
            return False
        return bool(np.array_equal(value, np.broadcast_to(value[0:1], value.shape)))
    if isinstance(value, Mapping):
        return all(_batch_values_are_repeated(item, batch_size) for item in value.values())
    return False


def _deduplicated_transform_input(
    observation: Mapping[str, Any],
    group_plan: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any], tuple[int, ...]] | None:
    """Build one transform row per unique prompt-conditioning group.

    A service batch may contain one or more rollout groups.  Within each group
    video, state, text and shared initial noise are identical; only per-request
    RNG/credit fields differ.  DreamTransform.apply_batch otherwise loops over
    all requests in Python.  Transform one representative per group and map
    the normalized result back to the original request order.
    """

    batch_size = _transform_input_batch_size(observation)
    if batch_size is None:
        return None
    if group_plan is None:
        representative_indices = (0,)
        group_index = tuple(0 for _ in range(batch_size))
    else:
        plan_batch_size = int(group_plan.get("batch_size", -1))
        representative_indices = tuple(int(index) for index in group_plan.get("representative_indices", ()))
        group_index = tuple(int(index) for index in group_plan.get("group_index", ()))
        if plan_batch_size != batch_size:
            raise ValueError(
                f"WNM conditioning group plan batch mismatch: plan={plan_batch_size}, observation={batch_size}."
            )
        if not representative_indices or len(group_index) != batch_size:
            raise ValueError(
                "WNM conditioning group plan requires non-empty representatives and "
                f"one group index per request; representatives={representative_indices}, "
                f"group_index_len={len(group_index)}, batch={batch_size}."
            )
        group_count = len(representative_indices)
        if any(index < 0 or index >= batch_size for index in representative_indices):
            raise ValueError(f"WNM conditioning representative indices are out of range: {representative_indices}.")
        if any(index < 0 or index >= group_count for index in group_index):
            raise ValueError(f"WNM conditioning group indices are out of range: {group_index}.")

    # There is nothing to deduplicate if every request owns a unique
    # conditioning row.
    if len(representative_indices) >= batch_size:
        return None

    seed_values: dict[str, Any] = {}
    representatives: dict[str, Any] = {}
    for key, value in observation.items():
        if not isinstance(value, torch.Tensor | np.ndarray) or value.ndim == 0:
            return None
        if value.shape[0] != batch_size:
            return None
        if key in _PER_REQUEST_TRANSFORM_FIELDS:
            seed_values[key] = _clone_transform_value(value)
        elif group_plan is None and not _batch_values_are_repeated(value, batch_size):
            return None
        if isinstance(value, torch.Tensor):
            indices = torch.as_tensor(representative_indices, dtype=torch.long, device=value.device)
            representatives[key] = value.index_select(0, indices).clone()
        else:
            representatives[key] = np.take(value, representative_indices, axis=0).copy()
    if not seed_values:
        return None
    return batch_size, representatives, seed_values, group_index


def _expand_grouped_transform_output(value: Any, group_index: tuple[int, ...]) -> Any:
    """Map unique normalized conditioning rows back to request order."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.reshape(1).expand(len(group_index)).clone()
        indices = torch.as_tensor(group_index, dtype=torch.long, device=value.device)
        if value.shape[0] <= int(indices.max().item()):
            raise RuntimeError(
                f"WNM grouped transform returned too few rows: shape={tuple(value.shape)}, group_index={group_index}."
            )
        return value.index_select(0, indices).clone()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return np.repeat(value.reshape(1), len(group_index), axis=0)
        if value.shape[0] <= max(group_index):
            raise RuntimeError(
                f"WNM grouped transform returned too few rows: shape={value.shape}, group_index={group_index}."
            )
        return np.take(value, group_index, axis=0).copy()
    if isinstance(value, Mapping):
        return {key: _expand_grouped_transform_output(item, group_index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_expand_grouped_transform_output(item, group_index) for item in value)
    if isinstance(value, list):
        return [_expand_grouped_transform_output(item, group_index) for item in value]
    raise TypeError(f"WNM grouped transform returned an unsupported leaf type: {type(value).__name__}.")


def _restore_transform_seed_fields(normalized: dict[str, Any], seed_values: Mapping[str, Any]) -> None:
    """Restore exact per-request rollout controls after checkpoint transforms.

    Checkpoint transforms are allowed to know nothing about RL-only controls.
    The 2D transform currently preserves ``rollout_seed`` while the VGGT
    transform drops it together with layer-credit metadata.
    These controls are consumed by the patched rollout immediately
    after preprocessing and must bypass the checkpoint transform unchanged.
    """

    for key, raw_values in seed_values.items():
        current = normalized.get(key)
        if isinstance(current, torch.Tensor):
            normalized[key] = torch.as_tensor(raw_values, dtype=current.dtype, device=current.device).reshape(-1)
        elif isinstance(current, np.ndarray):
            normalized[key] = np.asarray(raw_values, dtype=current.dtype).reshape(-1)
        elif current is None and key in (
            "rollout_seed",
            "seed",
            *_LAYER_CREDIT_FIELDS,
        ):
            # Keep the normalized observation tensor-only when the checkpoint
            # transform converted the rollout seed (the normal production
            # path).  Otherwise preserve the raw container family.
            tensor_reference = next(
                (
                    normalized[name]
                    for name in ("rollout_seed", "seed")
                    if isinstance(normalized.get(name), torch.Tensor)
                ),
                None,
            )
            if tensor_reference is None:
                tensor_reference = next(
                    (value for value in normalized.values() if isinstance(value, torch.Tensor)),
                    None,
                )
            if tensor_reference is not None:
                restored = torch.as_tensor(
                    raw_values,
                    dtype=torch.long,
                    device=tensor_reference.device,
                ).reshape(-1)
            elif isinstance(raw_values, torch.Tensor):
                restored = raw_values.to(dtype=torch.long).reshape(-1).clone()
            else:
                restored = np.asarray(raw_values, dtype=np.int64).reshape(-1)
            expected_count = int(
                raw_values.numel() if isinstance(raw_values, torch.Tensor) else np.asarray(raw_values).size
            )
            if int(restored.numel() if isinstance(restored, torch.Tensor) else restored.size) != expected_count:
                raise RuntimeError(
                    f"WNM failed to restore every per-request control {key!r}: "
                    f"expected={expected_count}, restored={restored.shape}."
                )
            normalized[key] = restored
        elif current is None:
            raise RuntimeError(f"WNM transform dropped required RNG field {key!r}.")
        else:
            raise TypeError(f"WNM normalized RNG field {key!r} has unsupported type {type(current).__name__}.")


def _assert_transform_output_parity(reference: Any, candidate: Any, *, path: str = "normalized_obs") -> None:
    """Require bitwise-identical checkpoint preprocessing on the first batch."""

    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        if tuple(reference) != tuple(candidate):
            raise RuntimeError(
                f"WNM transform dedup changed fields at {path}: "
                f"reference={tuple(reference)}, candidate={tuple(candidate)}."
            )
        for key in reference:
            _assert_transform_output_parity(reference[key], candidate[key], path=f"{path}.{key}")
        return
    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise RuntimeError(
                f"WNM transform dedup changed metadata at {path}: "
                f"reference={tuple(reference.shape)}/{reference.dtype}, "
                f"candidate={tuple(candidate.shape)}/{candidate.dtype}."
            )
        if not torch.equal(reference, candidate):
            difference = (reference.float() - candidate.float()).abs()
            maximum = float(difference.max().item()) if difference.numel() else 0.0
            raise RuntimeError(f"WNM transform dedup parity failed at {path}: max_abs={maximum:.8g}.")
        return
    if isinstance(reference, np.ndarray) and isinstance(candidate, np.ndarray):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise RuntimeError(
                f"WNM transform dedup changed metadata at {path}: "
                f"reference={reference.shape}/{reference.dtype}, candidate={candidate.shape}/{candidate.dtype}."
            )
        if not np.array_equal(reference, candidate):
            raise RuntimeError(f"WNM transform dedup parity failed at {path}.")
        return
    if isinstance(reference, tuple | list) and isinstance(candidate, type(reference)):
        if len(reference) != len(candidate):
            raise RuntimeError(f"WNM transform dedup changed sequence length at {path}.")
        for index, (reference_item, candidate_item) in enumerate(zip(reference, candidate, strict=True)):
            _assert_transform_output_parity(reference_item, candidate_item, path=f"{path}[{index}]")
        return
    if type(reference) is not type(candidate) or reference != candidate:
        raise RuntimeError(
            f"WNM transform dedup parity failed at {path}: reference={reference!r}, candidate={candidate!r}."
        )


def _install_rollout_transform_dedup(policy: Any) -> None:
    """Transform one copy of repeated prompt conditioning and expand it to N."""

    if getattr(policy, "_verl_rollout_transform_dedup_installed", False):
        return
    if not _env_flag("WAM_ROLLOUT_DEDUP_TRANSFORM", False):
        logger.warning("WNM repeated-conditioning transform dedup is disabled")
        return

    original_apply = policy.apply
    verify_pending = _env_flag("WAM_ROLLOUT_DEDUP_TRANSFORM_VERIFY", True)
    lock = threading.RLock()

    def apply_deduplicated(bound_policy: Any, batch: Any, **kwargs: Any) -> Any:
        nonlocal verify_pending
        action_head = bound_policy.trained_model.action_head
        action_head._verl_repeated_conditioning_batch_size = None
        action_head._verl_conditioning_representative_indices = None
        action_head._verl_conditioning_group_index = None
        requested_group_plan = getattr(action_head, "_verl_requested_conditioning_group_plan", None)
        plan = _deduplicated_transform_input(batch.obs, requested_group_plan)
        if plan is None:
            control_values = {
                key: _clone_transform_value(batch.obs[key]) for key in _PER_REQUEST_TRANSFORM_FIELDS if key in batch.obs
            }
            result = original_apply(batch, **kwargs)
            normalized = result.normalized_obs
            if not isinstance(normalized, Mapping):
                if hasattr(normalized, "__getstate__"):
                    normalized = normalized.__getstate__()
                else:
                    raise TypeError(
                        "WNM transform control passthrough requires normalized_obs to be mapping-like, got "
                        f"{type(normalized).__name__}."
                    )
            _restore_transform_seed_fields(normalized, control_values)
            result.normalized_obs = normalized
            return result
        batch_size, representative_observation, seed_values, group_index = plan
        representative_indices = (
            tuple(int(index) for index in requested_group_plan["representative_indices"])
            if requested_group_plan is not None
            else (0,)
        )

        representative_batch = type(batch)(obs=representative_observation)
        start = time.perf_counter()
        representative_result = original_apply(representative_batch, **kwargs)
        representative_time = time.perf_counter() - start
        normalized_representatives = representative_result.normalized_obs
        if not isinstance(normalized_representatives, Mapping):
            if hasattr(normalized_representatives, "__getstate__"):
                normalized_representatives = normalized_representatives.__getstate__()
            else:
                raise TypeError(
                    "WNM transform dedup requires normalized_obs to be mapping-like, got "
                    f"{type(normalized_representatives).__name__}."
                )
        normalized = _expand_grouped_transform_output(normalized_representatives, group_index)
        _restore_transform_seed_fields(normalized, seed_values)

        with lock:
            if verify_pending:
                reference_batch = type(batch)(obs=_clone_transform_value(batch.obs))
                reference_start = time.perf_counter()
                reference_result = original_apply(reference_batch, **kwargs)
                reference_time = time.perf_counter() - reference_start
                reference = reference_result.normalized_obs
                if not isinstance(reference, Mapping) and hasattr(reference, "__getstate__"):
                    reference = reference.__getstate__()
                _restore_transform_seed_fields(reference, seed_values)
                _assert_transform_output_parity(reference, normalized)
                verify_pending = False
                logger.warning(
                    "WNM grouped-conditioning transform dedup verified exactly: "
                    "batch=%d unique=%d original=%.3fs representatives=%.3fs transform_speedup=%.2fx",
                    batch_size,
                    len(representative_indices),
                    reference_time,
                    representative_time,
                    reference_time / max(representative_time, 1e-9),
                )

        batch.normalized_obs = normalized
        action_head._verl_repeated_conditioning_batch_size = batch_size
        action_head._verl_conditioning_representative_indices = representative_indices
        action_head._verl_conditioning_group_index = group_index
        return batch

    policy.apply = MethodType(apply_deduplicated, policy)
    policy._verl_rollout_transform_dedup_installed = True
    logger.warning(
        "Enabled WNM grouped-conditioning transform dedup: verify=%s seed_fields=%s",
        verify_pending,
        tuple(_PER_REQUEST_TRANSFORM_FIELDS),
    )


def _select_conditioning_call_value(
    value: Any,
    batch_size: int,
    representative_indices: tuple[int, ...],
) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        indices = torch.as_tensor(representative_indices, dtype=torch.long, device=value.device)
        return value.index_select(0, indices)
    if isinstance(value, tuple):
        return tuple(_select_conditioning_call_value(item, batch_size, representative_indices) for item in value)
    if isinstance(value, list):
        return [_select_conditioning_call_value(item, batch_size, representative_indices) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _select_conditioning_call_value(item, batch_size, representative_indices)
            for key, item in value.items()
        }
    return value


def _expand_grouped_conditioning_output(value: Any, group_index: tuple[int, ...]) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        indices = torch.as_tensor(group_index, dtype=torch.long, device=value.device)
        if value.shape[0] <= int(indices.max().item()):
            raise RuntimeError(
                "WNM grouped conditioning encoder returned too few rows: "
                f"shape={tuple(value.shape)}, group_index={group_index}."
            )
        return value.index_select(0, indices).clone()
    if isinstance(value, tuple):
        return tuple(_expand_grouped_conditioning_output(item, group_index) for item in value)
    if isinstance(value, list):
        return [_expand_grouped_conditioning_output(item, group_index) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand_grouped_conditioning_output(item, group_index) for key, item in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"WNM grouped conditioning encoder returned an unsupported leaf type: {type(value).__name__}.")


def _concatenate_singleton_conditioning_outputs(values: Sequence[Any]) -> Any:
    """Join per-prompt B=1 encoder outputs without changing their semantics."""

    if not values:
        raise ValueError("WNM conditioning encoder produced no singleton outputs.")
    first = values[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("WNM singleton encoder returned inconsistent tensor output types.")
        if first.ndim == 0:
            if not all(torch.equal(first, value) for value in values[1:]):
                raise ValueError("WNM singleton encoder returned inconsistent scalar tensors.")
            return first.clone()
        if any(value.ndim == 0 or value.shape[0] != 1 for value in values):
            raise ValueError(
                "WNM singleton encoder outputs must retain a leading batch dimension of one: "
                f"{[tuple(value.shape) for value in values]}."
            )
        if any(tuple(value.shape[1:]) != tuple(first.shape[1:]) for value in values[1:]):
            raise ValueError(
                "WNM singleton encoder outputs have incompatible non-batch shapes: "
                f"{[tuple(value.shape) for value in values]}."
            )
        return torch.cat(tuple(values), dim=0)
    if isinstance(first, tuple):
        if any(not isinstance(value, tuple) or len(value) != len(first) for value in values):
            raise TypeError("WNM singleton encoder returned inconsistent tuple outputs.")
        return tuple(
            _concatenate_singleton_conditioning_outputs([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        if any(not isinstance(value, list) or len(value) != len(first) for value in values):
            raise TypeError("WNM singleton encoder returned inconsistent list outputs.")
        return [
            _concatenate_singleton_conditioning_outputs([value[index] for value in values])
            for index in range(len(first))
        ]
    if isinstance(first, Mapping):
        expected_keys = tuple(first)
        if any(not isinstance(value, Mapping) or tuple(value) != expected_keys for value in values):
            raise TypeError("WNM singleton encoder returned inconsistent mapping outputs.")
        return {
            key: _concatenate_singleton_conditioning_outputs([value[key] for value in values]) for key in expected_keys
        }
    if first is None or isinstance(first, bool | int | float | str):
        if any(type(value) is not type(first) or value != first for value in values[1:]):
            raise ValueError("WNM singleton encoder returned inconsistent scalar outputs.")
        return first
    raise TypeError(f"WNM singleton conditioning encoder returned an unsupported leaf type: {type(first).__name__}.")


def _assert_conditioning_group_input_identity(
    value: Any,
    *,
    batch_size: int,
    representative_indices: tuple[int, ...],
    group_index: tuple[int, ...],
    path: str,
) -> None:
    """Verify every deduplicated encoder input row equals its B=1 source."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            return
        for row, group in enumerate(group_index):
            representative = representative_indices[group]
            if not torch.equal(value[row], value[representative]):
                raise AssertionError(
                    "WNM conditioning group input mismatch: "
                    f"path={path}, row={row}, representative={representative}, "
                    f"shape={tuple(value.shape)}."
                )
        return
    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            _assert_conditioning_group_input_identity(
                item,
                batch_size=batch_size,
                representative_indices=representative_indices,
                group_index=group_index,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_conditioning_group_input_identity(
                item,
                batch_size=batch_size,
                representative_indices=representative_indices,
                group_index=group_index,
                path=f"{path}.{key}",
            )


def _conditioning_call_signature(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    def signature(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
        if isinstance(value, tuple):
            return ("tuple", tuple(signature(item) for item in value))
        if isinstance(value, list):
            return ("list", tuple(signature(item) for item in value))
        if isinstance(value, Mapping):
            return ("mapping", tuple((key, signature(item)) for key, item in value.items()))
        return (type(value).__name__, repr(value))

    return (
        tuple(signature(item) for item in args),
        tuple((key, signature(item)) for key, item in kwargs.items()),
    )


def _install_repeated_conditioning_encoder_dedup(action_head: Any) -> None:
    """Encode repeated rollout conditioning once while preserving per-seed DiT state."""

    if getattr(action_head, "_verl_conditioning_encoder_dedup_installed", False):
        return
    if not _env_flag("WAM_ROLLOUT_DEDUP_ENCODERS", False):
        logger.warning("WNM repeated-conditioning encoder dedup is disabled")
        return
    if not _env_flag("WAM_ROLLOUT_DEDUP_TRANSFORM", False):
        raise RuntimeError("WAM_ROLLOUT_DEDUP_ENCODERS requires WAM_ROLLOUT_DEDUP_TRANSFORM=true.")

    verify = _env_flag("WAM_ROLLOUT_DEDUP_ENCODERS_VERIFY", True)
    atol = float(os.getenv("WAM_ROLLOUT_DEDUP_ENCODERS_ATOL", "0.002"))
    rtol = float(os.getenv("WAM_ROLLOUT_DEDUP_ENCODERS_RTOL", "0.002"))
    encoder_targets = tuple(
        (label, owner, method_name)
        for label, owner, method_name in (
            ("encode_prompt", action_head, "encode_prompt"),
            ("encode_image", action_head, "encode_image"),
            ("vae_encode", action_head.vae, "encode"),
        )
        if callable(getattr(owner, method_name, None))
    )
    verified_signatures: dict[str, set[tuple[Any, ...]]] = {label: set() for label, _, _ in encoder_targets}
    lock = threading.RLock()

    for label, owner, method_name in encoder_targets:
        original = getattr(owner, method_name)

        def encode_once_then_expand(
            *args: Any,
            _label: str = label,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            batch_size = getattr(action_head, "_verl_repeated_conditioning_batch_size", None)
            runtime_enabled = getattr(action_head, "_verl_conditioning_encoder_dedup_runtime_enabled", True)
            if not runtime_enabled or not isinstance(batch_size, int) or batch_size <= 1:
                return _original(*args, **kwargs)
            representative_indices = getattr(
                action_head,
                "_verl_conditioning_representative_indices",
                None,
            )
            group_index = getattr(action_head, "_verl_conditioning_group_index", None)
            if representative_indices is None or group_index is None:
                # Backward-compatible homogeneous plan for callers that only
                # provide the original batch-size marker.
                representative_indices = (0,)
                group_index = tuple(0 for _ in range(batch_size))
            else:
                representative_indices = tuple(int(index) for index in representative_indices)
                group_index = tuple(int(index) for index in group_index)
            if not representative_indices or len(group_index) != batch_size:
                raise ValueError(
                    "WNM conditioning encoder received an invalid grouping plan: "
                    f"representatives={representative_indices}, group_index_len={len(group_index)}, "
                    f"batch={batch_size}."
                )
            primary = args[0] if args else kwargs.get("videos")
            if not isinstance(primary, torch.Tensor) or primary.ndim == 0 or primary.shape[0] != batch_size:
                return _original(*args, **kwargs)
            signature = (
                _conditioning_call_signature(args, kwargs),
                representative_indices,
                group_index,
            )
            # WNM compiles T5/CLIP/VAE with dynamic=False and
            # reduce-overhead.  Its generated BF16 kernels are deterministic
            # for a fixed shape but not numerically invariant across B=1/2/16.
            # GN0 deployment invokes these frozen encoders at B=1, so run each
            # unique conditioning independently at B=1 and batch only the DiT.
            representative_started = time.perf_counter()
            singleton_outputs = []
            for representative_index in representative_indices:
                singleton_args = _select_conditioning_call_value(
                    args,
                    batch_size,
                    (representative_index,),
                )
                singleton_kwargs = _select_conditioning_call_value(
                    kwargs,
                    batch_size,
                    (representative_index,),
                )
                singleton_outputs.append(_clone_tensor_tree(_original(*singleton_args, **singleton_kwargs)))
            representative_output = _concatenate_singleton_conditioning_outputs(singleton_outputs)
            representative_time = time.perf_counter() - representative_started
            expanded_output = _expand_grouped_conditioning_output(representative_output, group_index)

            if verify and signature not in verified_signatures[_label]:
                with lock:
                    if signature not in verified_signatures[_label]:
                        _assert_conditioning_group_input_identity(
                            args,
                            batch_size=batch_size,
                            representative_indices=representative_indices,
                            group_index=group_index,
                            path="args",
                        )
                        _assert_conditioning_group_input_identity(
                            kwargs,
                            batch_size=batch_size,
                            representative_indices=representative_indices,
                            group_index=group_index,
                            path="kwargs",
                        )
                        reference_started = time.perf_counter()
                        reference_singletons = []
                        for representative_index in representative_indices:
                            singleton_args = _select_conditioning_call_value(
                                args,
                                batch_size,
                                (representative_index,),
                            )
                            singleton_kwargs = _select_conditioning_call_value(
                                kwargs,
                                batch_size,
                                (representative_index,),
                            )
                            reference_singletons.append(
                                _clone_tensor_tree(_original(*singleton_args, **singleton_kwargs))
                            )
                        reference_output = _concatenate_singleton_conditioning_outputs(reference_singletons)
                        reference_time = time.perf_counter() - reference_started
                        max_abs, max_rel = _assert_compiled_output_parity(
                            reference_output,
                            representative_output,
                            atol=atol,
                            rtol=rtol,
                            comparison=f"{_label} repeated B=1 inference",
                        )
                        verified_signatures[_label].add(signature)
                        logger.warning(
                            "WNM conditioning encoder B=1 inference parity passed: "
                            "method=%s DiT_batch=%d unique=%d "
                            "repeat=%.3fs representatives=%.3fs "
                            "max_abs=%.8g max_rel=%.8g",
                            _label,
                            batch_size,
                            len(representative_indices),
                            reference_time,
                            representative_time,
                            max_abs,
                            max_rel,
                        )
            return expanded_output

        setattr(owner, method_name, encode_once_then_expand)

    action_head._verl_repeated_conditioning_batch_size = None
    action_head._verl_conditioning_representative_indices = None
    action_head._verl_conditioning_group_index = None
    action_head._verl_requested_conditioning_group_plan = None
    action_head._verl_conditioning_encoder_dedup_runtime_enabled = True
    action_head._verl_conditioning_encoder_dedup_installed = True
    logger.warning(
        "Enabled WNM repeated-conditioning encoder dedup: methods=%s verify=%s atol=%g rtol=%g",
        tuple(verified_signatures),
        verify,
        atol,
        rtol,
    )
