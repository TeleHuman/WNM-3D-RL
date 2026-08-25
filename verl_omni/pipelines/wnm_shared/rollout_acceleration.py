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

"""Optional compile and CUDA Graph acceleration for WNM rollout."""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Mapping
from types import FunctionType, MethodType
from typing import Any

import torch

from verl_omni.pipelines.wnm_shared.rollout_common import _env_flag

logger = logging.getLogger(__name__)


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        result = []
        for key in value:
            result.extend(_tensor_leaves(value[key]))
        return result
    if isinstance(value, tuple | list):
        result = []
        for item in value:
            result.extend(_tensor_leaves(item))
        return result
    if value is None:
        return []
    raise TypeError(f"WNM compiled DiT returned unsupported output type {type(value).__name__}.")


def _clone_tensor_tree(value: Any) -> Any:
    """Clone tensor inputs so parity probes cannot alias or mutate one another."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    return value


def _input_tensor_leaves(value: Any) -> list[torch.Tensor]:
    """Collect tensors while allowing graph-specialized scalar constants."""

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        result = []
        for item in value.values():
            result.extend(_input_tensor_leaves(item))
        return result
    if isinstance(value, tuple | list):
        result = []
        for item in value:
            result.extend(_input_tensor_leaves(item))
        return result
    if value is None or isinstance(value, bool | int | float | str):
        return []
    raise TypeError(f"WNM CUDA Graph input has unsupported value type {type(value).__name__}.")


def _assert_compiled_output_parity(
    eager_output: Any,
    compiled_output: Any,
    *,
    atol: float,
    rtol: float,
    comparison: str = "compiled/eager",
) -> tuple[float, float]:
    """Fail closed if the first compiled DiT call changes rollout semantics."""

    eager_tensors = _tensor_leaves(eager_output)
    compiled_tensors = _tensor_leaves(compiled_output)
    if len(eager_tensors) != len(compiled_tensors):
        raise RuntimeError(
            "WNM compiled/eager DiT output structures differ: "
            f"eager={len(eager_tensors)} tensors, compiled={len(compiled_tensors)} tensors."
        )

    max_abs = 0.0
    max_rel = 0.0
    failures: list[str] = []
    for index, (eager, compiled) in enumerate(zip(eager_tensors, compiled_tensors, strict=True)):
        if eager.shape != compiled.shape or eager.dtype != compiled.dtype:
            raise RuntimeError(
                f"WNM compiled DiT output[{index}] metadata differs: "
                f"eager={tuple(eager.shape)}/{eager.dtype}, compiled={tuple(compiled.shape)}/{compiled.dtype}."
            )
        if eager.numel():
            difference = (compiled.float() - eager.float()).abs()
            leaf_max_abs = float(difference.max().item())
            denominator = eager.float().abs().clamp_min(1e-12)
            leaf_max_rel = float((difference / denominator).max().item())
            rmse = float(difference.square().mean().sqrt().item())
            reference_rms = float(eager.float().square().mean().sqrt().item())
            normalized_rmse = rmse / max(reference_rms, 1e-12)
            close = torch.isclose(compiled, eager, atol=atol, rtol=rtol, equal_nan=True)
            mismatch_fraction = float((~close).float().mean().item())
            max_abs = max(max_abs, leaf_max_abs)
            max_rel = max(max_rel, leaf_max_rel)
            if mismatch_fraction:
                failures.append(
                    f"output[{index}] shape={tuple(eager.shape)} dtype={eager.dtype} "
                    f"mismatch={mismatch_fraction:.3%} max_abs={leaf_max_abs:.7g} "
                    f"max_rel={leaf_max_rel:.7g} rmse={rmse:.7g} nrmse={normalized_rmse:.7g}"
                )
    if failures:
        raise AssertionError(
            f"WNM {comparison} DiT parity failed at atol={atol:g}, rtol={rtol:g}: " + "; ".join(failures)
        )
    return max_abs, max_rel


def _tensor_tree_signature(value: Any) -> Any:
    """Return a hashable CUDA-graph signature without retaining live tensors."""

    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            value.device.type,
            value.device.index,
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
            bool(value.requires_grad),
        )
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _tensor_tree_signature(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return ("tuple", tuple(_tensor_tree_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_tensor_tree_signature(item) for item in value))
    if value is None or isinstance(value, bool | int | float | str):
        return (type(value).__name__, value)
    raise TypeError(f"WNM CUDA Graph input has unsupported value type {type(value).__name__}.")


def _copy_tensor_tree_(destination: Any, source: Any) -> None:
    """Copy runtime tensors into a structurally identical static input tree."""

    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise TypeError("WNM CUDA Graph input structure changed after capture.")
        destination.copy_(source)
        return
    if isinstance(destination, Mapping):
        if not isinstance(source, Mapping) or tuple(destination) != tuple(source):
            raise TypeError("WNM CUDA Graph mapping structure changed after capture.")
        for key in destination:
            _copy_tensor_tree_(destination[key], source[key])
        return
    if isinstance(destination, tuple | list):
        if not isinstance(source, type(destination)) or len(destination) != len(source):
            raise TypeError("WNM CUDA Graph sequence structure changed after capture.")
        for destination_item, source_item in zip(destination, source, strict=True):
            _copy_tensor_tree_(destination_item, source_item)
        return
    if destination != source:
        raise ValueError(
            f"WNM CUDA Graph non-tensor input changed after capture: captured={destination!r}, runtime={source!r}."
        )


def _rollout_compile_options(*, prefix: str, default_mode: str) -> tuple[str, bool, bool, dict[str, Any]]:
    mode = os.getenv(f"{prefix}_MODE", default_mode).strip()
    fullgraph = _env_flag(f"{prefix}_FULLGRAPH", False)
    dynamic = _env_flag(f"{prefix}_DYNAMIC", True)
    emulate_precision_casts = _env_flag(f"{prefix}_EMULATE_PRECISION_CASTS", True)
    mode_options = torch._inductor.list_mode_options()
    if mode not in mode_options:
        raise ValueError(f"Unsupported {prefix} mode {mode!r}; choices={tuple(mode_options)}.")
    compile_options = dict(mode_options[mode])
    compile_options["emulate_precision_casts"] = emulate_precision_casts
    return mode, fullgraph, dynamic, compile_options


def _install_rollout_regional_compile(module: torch.nn.Module) -> None:
    """Compile repeated Causal-WAN blocks and verify the accumulated output.

    Compiling the complete ``_forward_train`` graph changed the BF16 action
    output materially.  vLLM-Omni's diffusion path instead compiles repeated
    transformer regions.  Keep that granularity here, but add a stricter
    end-to-end first-call parity gate because small per-block errors can
    accumulate through every DiT layer.
    """

    legacy_enabled = _env_flag("WAM_ROLLOUT_TORCH_COMPILE", False)
    if not _env_flag("WAM_ROLLOUT_REGIONAL_COMPILE", legacy_enabled):
        logger.warning("WNM rollout regional DiT torch.compile is disabled")
        return

    target_name = "_forward_train" if callable(getattr(module, "_forward_train", None)) else "forward"
    original_model_forward = getattr(module, target_name)
    repeated_blocks = tuple(
        name.strip()
        for name in os.getenv("WAM_ROLLOUT_REPEATED_BLOCKS", "CausalWanAttentionBlock").split(",")
        if name.strip()
    )
    selected = [
        (name, submodule, submodule.forward)
        for name, submodule in module.named_modules()
        if submodule.__class__.__name__ in repeated_blocks
    ]
    if not selected:
        raise RuntimeError(f"WNM regional compile found no repeated blocks matching {repeated_blocks!r}.")

    mode, fullgraph, dynamic, compile_options = _rollout_compile_options(
        prefix="WAM_ROLLOUT_REGIONAL_COMPILE",
        default_mode="default",
    )
    verify = _env_flag("WAM_ROLLOUT_REGIONAL_COMPILE_VERIFY", True)
    atol = float(os.getenv("WAM_ROLLOUT_REGIONAL_COMPILE_ATOL", "0.002"))
    rtol = float(os.getenv("WAM_ROLLOUT_REGIONAL_COMPILE_RTOL", "0.002"))
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError(f"WNM regional compile tolerances must be finite and non-negative: {atol=}, {rtol=}.")

    compiled_forwards = []
    for name, submodule, original_block_forward in selected:
        compiled_block_forward = torch.compile(
            original_block_forward,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=compile_options,
        )
        submodule.forward = compiled_block_forward
        compiled_forwards.append((name, submodule, original_block_forward, compiled_block_forward))

    verify_lock = threading.Lock()
    verified = False

    def regional_forward_with_first_call_parity(*args, **kwargs):
        nonlocal verified
        if not verify or verified:
            return original_model_forward(*args, **kwargs)
        with verify_lock:
            if verified:
                return original_model_forward(*args, **kwargs)

            # Run the eager reference with the exact original block methods,
            # then restore the compiled methods before executing the candidate.
            for _, submodule, original_block_forward, _ in compiled_forwards:
                submodule.forward = original_block_forward
            try:
                eager_warmup = original_model_forward(*_clone_tensor_tree(args), **_clone_tensor_tree(kwargs))
                eager_output = original_model_forward(*_clone_tensor_tree(args), **_clone_tensor_tree(kwargs))
            finally:
                for _, submodule, _, compiled_block_forward in compiled_forwards:
                    submodule.forward = compiled_block_forward

            eager_max_abs, eager_max_rel = _assert_compiled_output_parity(
                eager_warmup,
                eager_output,
                atol=0.0,
                rtol=0.0,
                comparison="regional eager/eager",
            )
            compiled_output = original_model_forward(*_clone_tensor_tree(args), **_clone_tensor_tree(kwargs))
            max_abs, max_rel = _assert_compiled_output_parity(
                eager_output,
                compiled_output,
                atol=atol,
                rtol=rtol,
                comparison="regional compiled/eager",
            )
            verified = True
            logger.warning(
                "WNM rollout regional compiled/eager parity passed: blocks=%d classes=%s "
                "mode=%s fullgraph=%s dynamic=%s eager_max_abs=%g eager_max_rel=%g "
                "atol=%g rtol=%g max_abs=%g max_rel=%g",
                len(compiled_forwards),
                repeated_blocks,
                mode,
                fullgraph,
                dynamic,
                eager_max_abs,
                eager_max_rel,
                atol,
                rtol,
                max_abs,
                max_rel,
            )
            return compiled_output

    if verify:
        setattr(module, target_name, regional_forward_with_first_call_parity)
    module._verl_rollout_regional_compile_enabled = True
    module._verl_rollout_regional_compile_target = target_name
    module._verl_rollout_regional_compile_blocks = tuple(name for name, *_ in compiled_forwards)
    logger.warning(
        "Enabled WNM rollout regional DiT torch.compile: target=%s blocks=%d classes=%s "
        "mode=%s fullgraph=%s dynamic=%s verify=%s",
        target_name,
        len(compiled_forwards),
        repeated_blocks,
        mode,
        fullgraph,
        dynamic,
        verify,
    )


def _install_rollout_cuda_graph(module: torch.nn.Module) -> None:
    """Capture fixed-shape deterministic DiT calls without capturing rollout RNG."""

    if not _env_flag("WAM_ROLLOUT_CUDA_GRAPH", False):
        logger.warning("WNM rollout DiT CUDA Graph is disabled")
        return
    if not torch.cuda.is_available():
        logger.warning("WNM rollout DiT CUDA Graph requested without CUDA; using eager/compiled calls")
        return

    target_name = "_forward_train" if callable(getattr(module, "_forward_train", None)) else "forward"
    graph_forward = getattr(module, target_name)
    verify = _env_flag("WAM_ROLLOUT_CUDA_GRAPH_VERIFY", True)
    atol = float(os.getenv("WAM_ROLLOUT_CUDA_GRAPH_ATOL", "0"))
    rtol = float(os.getenv("WAM_ROLLOUT_CUDA_GRAPH_RTOL", "0"))
    warmup_iters = int(os.getenv("WAM_ROLLOUT_CUDA_GRAPH_WARMUP_ITERS", "2"))
    max_entries = int(os.getenv("WAM_ROLLOUT_CUDA_GRAPH_MAX_ENTRIES", "4"))
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError(f"WNM CUDA Graph tolerances must be finite and non-negative: {atol=}, {rtol=}.")
    if warmup_iters < 1 or max_entries < 1:
        raise ValueError(f"WNM CUDA Graph requires positive warmup/max entries: {warmup_iters=}, {max_entries=}.")

    entries: dict[Any, tuple[Any, Any, torch.cuda.CUDAGraph, Any]] = {}
    unsupported_signatures: set[Any] = set()
    graph_lock = threading.Lock()

    def cuda_graph_forward(*args, **kwargs):
        tensor_inputs = _input_tensor_leaves((args, kwargs))
        if torch.is_grad_enabled() or not tensor_inputs or any(not tensor.is_cuda for tensor in tensor_inputs):
            return graph_forward(*args, **kwargs)
        devices = {tensor.device for tensor in tensor_inputs}
        if len(devices) != 1:
            raise ValueError(f"WNM CUDA Graph inputs span multiple CUDA devices: {devices}.")
        signature = _tensor_tree_signature((args, kwargs))

        with graph_lock:
            if signature in unsupported_signatures:
                return graph_forward(*args, **kwargs)
            entry = entries.get(signature)
            if entry is None:
                if len(entries) >= max_entries:
                    unsupported_signatures.add(signature)
                    logger.warning(
                        "WNM rollout CUDA Graph cache reached %d signatures; using eager/compiled path for %s",
                        max_entries,
                        signature,
                    )
                    return graph_forward(*args, **kwargs)

                # Let a nested regional-compile parity guard finish before CUDA
                # capture. This reference is also the graph-equivalence target.
                reference_output = graph_forward(*_clone_tensor_tree(args), **_clone_tensor_tree(kwargs))
                static_args = _clone_tensor_tree(args)
                static_kwargs = _clone_tensor_tree(kwargs)
                device = next(iter(devices))
                capture_stream = torch.cuda.Stream(device=device)
                current_stream = torch.cuda.current_stream(device)
                capture_stream.wait_stream(current_stream)
                try:
                    with torch.cuda.stream(capture_stream):
                        for _ in range(warmup_iters):
                            graph_forward(*static_args, **static_kwargs)
                    capture_stream.synchronize()
                    current_stream.wait_stream(capture_stream)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        static_output = graph_forward(*static_args, **static_kwargs)
                    graph.replay()
                    torch.cuda.synchronize(device)
                except Exception:
                    unsupported_signatures.add(signature)
                    logger.exception(
                        "WNM rollout CUDA Graph capture failed for signature %s; using eager/compiled path",
                        signature,
                    )
                    return reference_output

                graph_output = _clone_tensor_tree(static_output)
                if verify:
                    max_abs, max_rel = _assert_compiled_output_parity(
                        reference_output,
                        graph_output,
                        atol=atol,
                        rtol=rtol,
                        comparison="CUDA Graph/eager-or-regional",
                    )
                else:
                    max_abs = max_rel = float("nan")
                entry = (static_args, static_kwargs, graph, static_output)
                entries[signature] = entry
                logger.warning(
                    "WNM rollout CUDA Graph captured and verified: target=%s entries=%d "
                    "warmup=%d atol=%g rtol=%g max_abs=%g max_rel=%g signature=%s",
                    target_name,
                    len(entries),
                    warmup_iters,
                    atol,
                    rtol,
                    max_abs,
                    max_rel,
                    signature,
                )
                return graph_output

            static_args, static_kwargs, graph, static_output = entry
            _copy_tensor_tree_(static_args, args)
            _copy_tensor_tree_(static_kwargs, kwargs)
            graph.replay()
            # Graph outputs alias persistent capture buffers. Clone them before
            # the next denoising step replays this entry and overwrites them.
            return _clone_tensor_tree(static_output)

    setattr(module, target_name, cuda_graph_forward)
    module._verl_rollout_cuda_graph_enabled = True
    module._verl_rollout_cuda_graph_target = target_name
    module._verl_rollout_cuda_graph_entries = entries
    logger.warning(
        "Enabled WNM rollout DiT CUDA Graph: target=%s warmup=%d max_entries=%d verify=%s",
        target_name,
        warmup_iters,
        max_entries,
        verify,
    )


class _RolloutTorchFactoryProxy:
    """Redirect one capture-unsafe WNM CPU constant to the rollout GPU."""

    def __init__(self, *, device: torch.device):
        self._device = device

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    def tensor(self, data: Any, *args, **kwargs) -> torch.Tensor:
        # CausalWanModel._forward_train replaces every supplied embodiment ID
        # with an all-zero InteriorGS ID via ``torch.tensor([0]).repeat(...).to``.
        # The pageable CPU -> CUDA copy is illegal during graph capture. Keep
        # identical values/dtype while allocating this exact constant directly
        # on the rollout device. Other factory calls (notably CPU grid_size)
        # retain their original behavior.
        if not args and not kwargs and isinstance(data, list) and len(data) == 1 and data[0] == 0:
            # ``torch.tensor([0], device="cuda")`` still stages the Python
            # list through pageable host memory. A device-native zeros kernel
            # is capture-safe and produces the same default int64 value.
            return torch.zeros((1,), dtype=torch.int64, device=self._device)
        return torch.tensor(data, *args, **kwargs)


def _install_capture_safe_rollout_factories(module: torch.nn.Module) -> None:
    """Clone only this instance's forward globals; never mutate WNM/GN0."""

    if not _env_flag("WAM_ROLLOUT_CUDA_GRAPH", False):
        return
    target_name = "_forward_train" if callable(getattr(module, "_forward_train", None)) else "forward"
    original_forward = getattr(module, target_name)
    function = getattr(original_forward, "__func__", None)
    if function is None:
        raise TypeError(f"WNM CUDA Graph target {target_name!r} is not a bound Python method.")
    reference = next(module.parameters(), None)
    if reference is None or reference.device.type != "cuda":
        return

    isolated_globals = dict(function.__globals__)
    isolated_globals["torch"] = _RolloutTorchFactoryProxy(device=reference.device)
    capture_safe_function = FunctionType(
        function.__code__,
        isolated_globals,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    capture_safe_function.__kwdefaults__ = function.__kwdefaults__
    capture_safe_function.__annotations__ = function.__annotations__
    capture_safe_function.__qualname__ = function.__qualname__
    capture_safe_function.__doc__ = function.__doc__
    setattr(module, target_name, MethodType(capture_safe_function, module))
    module._verl_rollout_capture_safe_factories = True
    logger.warning(
        "WNM rollout installed process-local CUDA Graph-safe constant factories: target=%s device=%s",
        target_name,
        reference.device,
    )


def _install_rollout_dit_acceleration(module: torch.nn.Module) -> None:
    """Install process-local upstream-style compile and launch-overhead reduction."""

    # These attributes are the contract used by vLLM-Omni's own
    # ``regionally_compile`` helper.  Set them only on this rollout model
    # instance; the WNM checkout and GN0 inference remain untouched.
    module._repeated_blocks = ["CausalWanAttentionBlock"]
    module._layerwise_offload_blocks_attrs = ["blocks"]
    _install_capture_safe_rollout_factories(module)
    _install_rollout_regional_compile(module)
    _install_rollout_cuda_graph(module)


def _configure_explicit_sdpa_runtime(*, component: str) -> None:
    """Pin every WNM attention path inside this RL worker to PyTorch SDPA.

    This changes only the rollout worker process; it does not modify GN0 or
    standalone WNM inference configuration.
    """
    os.environ["ATTENTION_BACKEND"] = "torch"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
    os.environ["WAM_ENFORCE_SDPA"] = "true"

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
