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
import asyncio
import functools
import hashlib
import logging
import os
import threading
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_ROLLOUT_GROUP_GATES: dict[str, dict[str, Any]] = {}


def _shared_initial_noise_seed(uid: Any, global_steps: Any = None) -> int:
    """Derive one stable 63-bit initialization seed per prompt group/step."""

    if hasattr(uid, "item"):
        uid = uid.item()
    if uid is None:
        raise KeyError("init_same_noise requires a per-prompt uid.")
    step = "" if global_steps is None else str(int(global_steps))
    payload = f"{uid}\0{step}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


async def _wait_for_rollout_group(request_id: str, expected_size: int, timeout_s: float) -> None:
    """Release one prompt's rollout requests to the server as a single burst."""

    if expected_size <= 1:
        return
    state = _ROLLOUT_GROUP_GATES.get(request_id)
    if state is None:
        state = {"event": asyncio.Event(), "arrived": 0, "departed": 0}
        _ROLLOUT_GROUP_GATES[request_id] = state
    state["arrived"] += 1
    if state["arrived"] > expected_size:
        raise RuntimeError(f"WAM rollout group {request_id!r} exceeded expected size {expected_size}.")
    if state["arrived"] == expected_size:
        state["event"].set()
    try:
        await asyncio.wait_for(state["event"].wait(), timeout=timeout_s)
    except TimeoutError as exc:
        # Fail closed: silently releasing a partial group recreates the 1+7
        # scheduling pattern and defeats transform/VAE conditioning dedup.
        state["event"].set()
        raise RuntimeError(
            f"Timed out waiting for WAM rollout group {request_id!r}: "
            f"arrived={state['arrived']} expected={expected_size}."
        ) from exc
    finally:
        state["departed"] += 1
        if state["event"].is_set() and state["departed"] == state["arrived"]:
            _ROLLOUT_GROUP_GATES.pop(request_id, None)


@register("diffusion_single_turn_agent")
class DiffusionSingleTurnAgentLoop(AgentLoopBase):
    """Agent loop for diffusion model serving."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> DiffusionAgentLoopOutput:
        """Run one diffusion generation turn and package agent-loop output.

        Args:
            sampling_params: Generation parameters forwarded to the server manager.
            **kwargs: Per-sample fields from the dataset, including ``raw_prompt``
                and optional ``raw_negative_prompt``.

        Returns:
            DiffusionAgentLoopOutput: Prompt ids, generated diffusion output,
            optional logprobs, runtime metrics, and extra fields.
        """
        raw_prompt = kwargs["raw_prompt"]
        raw_negative_prompt = kwargs.get("raw_negative_prompt")

        # Conditional diffusion models may need numeric per-row inputs in
        # addition to prompt media (for example WNM state features).
        # Keep the parquet schema standard by carrying those values under
        # ``extra_info.rollout_extra_args`` and forwarding them through the
        # existing vLLM sampling ``extra_args`` channel.
        rollout_extra_args = None
        extra_info = kwargs.get("extra_info")
        if isinstance(extra_info, Mapping) and "rollout_extra_args" in extra_info:
            rollout_extra_args = extra_info["rollout_extra_args"]
            if not isinstance(rollout_extra_args, Mapping):
                raise TypeError("extra_info['rollout_extra_args'] must be a mapping.")
            if "rollout_extra_args" in sampling_params:
                raise KeyError("sampling_params already contains reserved key 'rollout_extra_args'.")
            sampling_params = dict(sampling_params)
            sampling_params["rollout_extra_args"] = dict(rollout_extra_args)

        # 1. extract images and videos from messages. WNM's Stage-1
        # contract requires every source frame in the exact 33+33 layout;
        # generic vision helpers may resample MP4s according to a default FPS.
        exact_video_path = (
            rollout_extra_args.get("exact_context_video_path") if rollout_extra_args is not None else None
        )
        if exact_video_path is not None:
            expected_frames = int(rollout_extra_args.get("exact_context_video_frames", 66))
            exact_video = await asyncio.to_thread(
                _decode_exact_rgb_video,
                os.fspath(exact_video_path),
                expected_frames,
            )
            images = None
            videos = [exact_video]
        else:
            multi_modal_data = await self.process_vision_info(raw_prompt)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        if rollout_extra_args is not None and "instruction" in rollout_extra_args:
            # WNM re-applies its checkpoint transform/tokenizer inside
            # the rollout pipeline.  These IDs only satisfy the generic verl
            # prompt/uid contract, so avoid imposing an unrelated chat
            # template on a plain UMT5 instruction.
            encoded = self.tokenizer(
                str(rollout_extra_args["instruction"]),
                add_special_tokens=True,
                truncation=True,
                max_length=self.rollout_config.prompt_length,
            )
            prompt_ids = encoded["input_ids"]
        else:
            prompt_ids = await self.apply_chat_template(raw_prompt, images=images, videos=videos)

        if raw_negative_prompt is not None:
            negative_prompt_ids = await self.apply_chat_template(raw_negative_prompt, images=images, videos=videos)
        else:
            negative_prompt_ids = None

        if bool(sampling_params.get("init_same_noise", False)):
            # Do not replace the request seed: it continues to own an
            # independent transition RNG stream. This extra seed controls only
            # the initial video/action latents and is shared by the N rollouts
            # of one uid at the current training step.
            sampling_params = dict(sampling_params)
            sampling_params["initial_noise_seed"] = _shared_initial_noise_seed(
                kwargs.get("uid"),
                sampling_params.get("global_steps"),
            )

        layer_credit_fields = (
            "layer_credit_stratum",
            "layer_credit_branch",
            "layer_credit_transition",
        )
        present_layer_credit_fields = [name for name in layer_credit_fields if name in kwargs]
        if present_layer_credit_fields:
            if len(present_layer_credit_fields) != len(layer_credit_fields):
                missing = sorted(set(layer_credit_fields) - set(present_layer_credit_fields))
                raise KeyError(f"Incomplete layer-conditioned rollout metadata; missing {missing}.")
            sampling_params = dict(sampling_params)
            for name in layer_credit_fields:
                value = kwargs[name]
                if hasattr(value, "item"):
                    value = value.item()
                sampling_params[name] = int(value)

        # 3. generate sequences
        request_id = uuid4().hex
        # This is an agent-loop-only admission hint, not a model sampling
        # parameter.  Remove it before forwarding the request to vLLM-Omni.
        sampling_params = dict(sampling_params)
        request_group_size = int(
            sampling_params.pop(
                "_wam_rollout_group_size",
                os.getenv("WAM_ROLLOUT_GROUP_SIZE", "1"),
            )
        )
        if request_group_size <= 0:
            raise ValueError(f"WAM rollout group size must be positive, got {request_group_size}.")
        if os.getenv("WAM_ROLLOUT_GROUP_STICKY", "false").strip().lower() == "true":
            # The trainer repeats each prompt N times with the same uid. Route
            # those requests to one replica so vLLM can form a homogeneous
            # request batch and the WAM pipeline can reuse deterministic
            # transform/VAE conditioning. Different uids are still assigned
            # by the global least-loaded balancer.
            uid = kwargs.get("uid")
            if uid is None:
                raise KeyError("WAM_ROLLOUT_GROUP_STICKY requires a per-prompt uid.")
            if hasattr(uid, "item"):
                uid = uid.item()
            request_id = f"wam-rollout-group:{uid}"
            expected_group_size = request_group_size
            group_timeout_s = float(os.getenv("WAM_ROLLOUT_GROUP_TIMEOUT_S", "30"))
            await _wait_for_rollout_group(request_id, expected_group_size, group_timeout_s)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                negative_prompt_ids=negative_prompt_ids,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        extra_fields = dict(output.extra_fields)
        actions = getattr(output, "actions", None)
        if actions is None:
            actions = extra_fields.pop("actions", None)

        output = DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=output.diffusion_output,
            response_logprobs=output.log_probs,
            actions=actions,
            action_log_probs=output.action_log_probs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output


# One global lock used to make every distinct cache miss serial.  A rollout
# worker receives several prompt groups at once, so that turned eight unrelated
# MP4 decodes into an avoidable input-side queue.  Lock striping still collapses
# concurrent misses for the same cache key while allowing unrelated videos to
# decode in parallel.  The fixed-size pool cannot grow with the dataset.
_EXACT_VIDEO_CACHE_LOCKS = tuple(threading.Lock() for _ in range(64))


def _decode_exact_rgb_video(path: str, expected_frames: int):
    # rollout.n launches identical prompt tasks concurrently. Serialize only
    # matching cache keys so one prompt group does not decode the same 66-frame
    # MP4 N times; distinct prompt videos may decode concurrently.
    cache_key = (os.path.abspath(path), int(expected_frames))
    lock = _EXACT_VIDEO_CACHE_LOCKS[hash(cache_key) % len(_EXACT_VIDEO_CACHE_LOCKS)]
    with lock:
        return _decode_exact_rgb_video_cached(path, expected_frames)


@functools.lru_cache(maxsize=64)
def _decode_exact_rgb_video_cached(path: str, expected_frames: int):
    """Decode every frame without the sampling performed by generic VLM helpers."""
    import cv2
    import numpy as np

    if expected_frames <= 0:
        raise ValueError("exact_context_video_frames must be positive")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open exact context video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != expected_frames:
        raise ValueError(
            f"exact context video must contain {expected_frames} frames, decoded {len(frames)} from {path}"
        )
    return np.stack(frames).astype(np.uint8, copy=False)
