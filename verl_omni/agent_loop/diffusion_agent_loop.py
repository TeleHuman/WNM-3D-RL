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
import os
import random
from collections.abc import Mapping
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from verl.base_config import BaseConfig
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    DictConfigWrap,
    _agent_loop_registry,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.utils.profiler import simple_timer
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.utils import maybe_per_rollout_seeds
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig

_WAM_PLAIN_TEXT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{% if not loop.last %}
{% endif %}{% endfor %}"""


def _config_to_sampling_dict(config: Optional[BaseConfig]) -> dict:
    if config is None:
        return {}
    return {k: v for k, v in config.items() if not k.startswith("_")}


class DiffusionAgentLoopOutput(BaseModel):
    """Agent loop output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    """Prompt token ids."""
    response_diffusion_output: Any
    """Response diffusion output (torch.Tensor): image tensor (CHW) / video tensor (TCHW)."""
    response_logprobs: Optional[Any] = None
    """Log probabilities for image/video response transitions."""
    actions: Optional[Any] = None
    """Optional sampled actions produced by a world-action model."""
    action_log_probs: Optional[Any] = None
    """Optional log probabilities for sampled WAM action transitions."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalDiffusionAgentLoopOutput(DiffusionAgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_diffusion_output: torch.Tensor
    """Response diffusion output: image (NCHW format) / video (NTCHW format)."""
    response_logprobs: Optional[torch.Tensor] = None
    """Log probabilities over denoising timesteps."""
    actions: Optional[torch.Tensor] = None
    """Batched sampled WAM actions."""
    action_log_probs: Optional[torch.Tensor] = None
    """Batched WAM action transition log probabilities."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DiffusionAgentLoopWorker:
    """Diffusion Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server replicas, produced by
            ``LLMServerManager.get_client()`` in the trainer.
        teacher_client (dict[str, LLMServerClient]): Not used by diffusion training; accepted to
            keep the constructor signature compatible with verl's ``AgentLoopManager.create()``,
            which positionally forwards a teacher client argument to each worker.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model
        self.rollout_config: DiffusionRolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: DiffusionModelConfig = omega_conf_to_dataclass(model_config)

        if not hasattr(self, "server_manager"):
            self.server_manager = llm_client

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_loop_worker_handles = reward_loop_worker_handles

        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        # UMT5 is a plain text encoder and intentionally ships without a chat
        # template.  AgentLoopBase nevertheless tokenizes its bookkeeping
        # system prompt through apply_chat_template during construction.  WAM
        # tokenizes the actual navigation instruction directly in
        # DiffusionSingleTurnAgentLoop.run, so use a lossless content-only
        # template solely to satisfy the generic agent-loop contract.
        if self.model_config.architecture in {"WNM2D", "WNM3D"} and not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = _WAM_PLAIN_TEXT_CHAT_TEMPLATE

        self.max_prompt_embed_length = self.rollout_config.pipeline.max_sequence_length

        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch with the following fields.

            - ``prompts``: ``[bsz, prompt_length]`` prompt token ids from dataset.
            - ``responses``: diffusion output, typically ``[bsz, C, H, W]`` (image)
              or ``[bsz, T, C, H, W]`` (video).
            - ``rm_scores`` (optional): ``[bsz, 1]`` reward model scores.
            - ``meta_info``:

              - ``metrics``: ``List[dict]``, per-sample agent loop metrics.
              - ``reward_extra_keys`` (optional): ``List[str]``, keys for reward
                extra info for logging/validation.
        """
        config = self.rollout_config

        # Sampling schedules originate on the trainer driver. meta_info is not
        # preserved reliably when a repeated DataProto is split across Ray
        # agent workers, so the trainer also sends one explicit override dict
        # per request. Consume it here so it cannot leak into dataset kwargs,
        # reward RPCs, or the returned non-tensor batch.
        per_request_sampling_overrides = batch.non_tensor_batch.pop("_rollout_sampling_overrides", None)
        if per_request_sampling_overrides is not None:
            per_request_sampling_overrides = np.asarray(per_request_sampling_overrides, dtype=object).reshape(-1)
            if len(per_request_sampling_overrides) != len(batch):
                raise ValueError(
                    "Per-request rollout override count must match the batch: "
                    f"{len(per_request_sampling_overrides)} vs {len(batch)}."
                )

        sampling_params = {
            **_config_to_sampling_dict(config.pipeline),
            **_config_to_sampling_dict(config.algo),
            "logprobs": config.calculate_log_probs,
        }

        is_validate = batch.meta_info.get("validate", False)
        per_rollout_seeds: Optional[list[int]] = None

        # The sticky WAM admission barrier must follow the number of samples
        # actually repeated for this request.  Training commonly uses n=8,
        # while low-memory validation intentionally uses val_kwargs.n=1.  A
        # process-wide WAM_ROLLOUT_GROUP_SIZE alone would make validation wait
        # forever for seven requests that do not exist.
        sampling_params["_wam_rollout_group_size"] = int(config.val_kwargs.n if is_validate else config.n)

        if is_validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            validation_config = self.config.algorithm.get("rollout_log_prob_validation", None)
            validation_log_probs = bool(
                validation_config
                and validation_config.get("enabled", False)
                and validation_config.get("validate", False)
            )
            sampling_params["logprobs"] = bool(config.calculate_log_probs and validation_log_probs)
            global_indices = batch.non_tensor_batch.get("_rollout_seed_global_idx")
            per_rollout_seeds = maybe_per_rollout_seeds(
                {"rollout_seed": config.val_kwargs.seed}, len(batch), global_indices
            )
        else:
            sampling_params["global_steps"] = batch.meta_info["global_steps"]
            sampling_overrides = batch.meta_info.get("rollout_sampling_overrides", None)
            if sampling_overrides is not None:
                if not isinstance(sampling_overrides, dict):
                    raise TypeError("rollout_sampling_overrides must be a dictionary")
                unknown = set(sampling_overrides) - {
                    "noise_level",
                    "action_noise_level",
                }
                if unknown:
                    raise KeyError(f"unsupported rollout_sampling_overrides keys: {sorted(unknown)}")
                sampling_params.update({name: float(value) for name, value in sampling_overrides.items()})
            global_indices = batch.non_tensor_batch.get("_rollout_seed_global_idx")
            per_rollout_seeds = maybe_per_rollout_seeds(batch.meta_info, len(batch), global_indices)

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            task_sampling_params = sampling_params.copy()
            if per_request_sampling_overrides is not None:
                request_overrides = per_request_sampling_overrides[i]
                if not isinstance(request_overrides, Mapping):
                    raise TypeError(
                        f"Each per-request rollout override must be a mapping, got {type(request_overrides).__name__}."
                    )
                task_sampling_params.update(request_overrides)
            if per_rollout_seeds is not None:
                task_sampling_params["seed"] = per_rollout_seeds[i]
            tasks.append(asyncio.create_task(self._run_agent_loop(task_sampling_params, **kwargs)))
        outputs = await asyncio.gather(*tasks)

        drop_training_responses = (
            self.model_config.architecture in {"WNM2D", "WNM3D"}
            and not is_validate
            and not self.config.trainer.get("rollout_data_dir", None)
            and os.getenv("WAM_DROP_TRAIN_RESPONSES_AFTER_REWARD", "false").strip().lower() == "true"
        )
        output = self._postprocess(
            outputs,
            input_non_tensor_batch=batch.non_tensor_batch,
            drop_responses=drop_training_responses,
        )

        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalDiffusionAgentLoopOutput:
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )
        output: DiffusionAgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalDiffusionAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output_extra_fields = dict(output.extra_fields)
        actions = output.actions
        if actions is None:
            # Compatibility with rollout adapters that still return actions in
            # custom_output/extra_fields. Remove the compatibility copy before
            # collecting generic tensor fields so it cannot overwrite the
            # first-class batch entry later.
            actions = output_extra_fields.pop("actions", None)

        # Pad extra tensor outputs from vllm-omni (e.g. prompt embeddings).
        extra_fields = {}
        for k, v in output_extra_fields.items():
            if isinstance(v, torch.Tensor):
                if k in ["prompt_embeds", "negative_prompt_embeds"]:
                    pad_tuple = (0, 0, 0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                elif k in ["prompt_embeds_mask", "negative_prompt_embeds_mask"]:
                    pad_tuple = (0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                extra_fields[k] = v.unsqueeze(0)
            else:
                extra_fields[k] = v

        extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        response_diffusion_output = output.response_diffusion_output.unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            response_logprobs = output.response_logprobs
            if not isinstance(response_logprobs, torch.Tensor):
                response_logprobs = torch.as_tensor(response_logprobs)
            response_logprobs = response_logprobs.detach().unsqueeze(0)

        if actions is not None:
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions)
            actions = actions.detach().unsqueeze(0)

        action_log_probs = output.action_log_probs
        if action_log_probs is not None:
            if not isinstance(action_log_probs, torch.Tensor):
                action_log_probs = torch.as_tensor(action_log_probs)
            action_log_probs = action_log_probs.detach().unsqueeze(0)

        prompt_ids = prompt_output["input_ids"]
        extra_fields["attention_mask"] = prompt_output["attention_mask"]

        await self._compute_score(
            output,
            prompts=prompt_ids,
            responses=response_diffusion_output,
            actions=actions,
            kwargs=kwargs,
        )

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]

        return _InternalDiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=response_diffusion_output,
            response_logprobs=response_logprobs,
            actions=actions,
            action_log_probs=action_log_probs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    async def _compute_score(self, output, prompts, responses, actions, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        if output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                tensor_batch = {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, C, H, W] or [1, T, C, H, W]
                }
                if actions is not None:
                    tensor_batch["actions"] = actions
                batch = TensorDict(tensor_batch, batch_size=1)
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                }
                if self.model_config.architecture in {"WNM2D", "WNM3D"} and actions is not None:
                    # WAM reward receives actions as a first-class tensor and
                    # ground truth through kwargs. Do not serialize prompt
                    # embeddings, latent trajectories, and frozen conditions
                    # into every one of the 512 reward RPCs.
                    non_tensor_batch["tool_extra_fields"] = np.array([{}], dtype=object)
                else:
                    non_tensor_batch["tool_extra_fields"] = np.array([output.extra_fields], dtype=object)

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                rollout_index = kwargs.get("_rollout_seed_global_idx")
                if rollout_index is None:
                    selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                else:
                    # Stable modulo assignment avoids the queue imbalance and
                    # long-tail latency caused by 512 independent random picks.
                    worker_index = int(np.asarray(rollout_index).reshape(-1)[0]) % len(self.reward_loop_worker_handles)
                    selected_reward_loop_worker_handle = self.reward_loop_worker_handles[worker_index]
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            output.metrics.compute_score = timing["compute_score"]

    def _postprocess(
        self,
        inputs: list[_InternalDiffusionAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        *,
        drop_responses: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        scores = [input.reward_score for input in inputs]
        if drop_responses and not all(score is not None for score in scores):
            raise RuntimeError("Cannot drop WNM training responses before every streaming reward score is available.")
        response_diffusion_output = None
        if not drop_responses:
            response_diffusion_output = torch.cat([input.response_diffusion_output for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        actions_present = [input.actions is not None for input in inputs]
        if any(actions_present):
            if not all(actions_present):
                raise ValueError("WAM rollout batch mixes samples with and without actions.")
            optional_outputs["actions"] = torch.cat([input.actions for input in inputs], dim=0)

        action_log_probs_present = [input.action_log_probs is not None for input in inputs]
        if any(action_log_probs_present):
            if not all(action_log_probs_present):
                raise ValueError("WAM rollout batch mixes samples with and without action log-probabilities.")
            if not all(actions_present):
                raise ValueError("WAM action log-probabilities require actions for every rollout sample.")
            optional_outputs["rollout_action_log_probs"] = torch.cat(
                [input.action_log_probs for input in inputs], dim=0
            )

        # Handle extra fields that are tensors
        extra_keys = [k for k, v in inputs[0].extra_fields.items() if isinstance(v, torch.Tensor)]
        for key in extra_keys:
            optional_outputs[key] = torch.cat([input.extra_fields[key] for input in inputs], dim=0)
            for input in inputs:
                del input.extra_fields[key]

        tensor_outputs = {
            "prompts": prompt_ids,  # [bsz, prompt_length]
            **optional_outputs,
        }
        if response_diffusion_output is not None:
            tensor_outputs["responses"] = response_diffusion_output  # image or video
        batch = TensorDict(tensor_outputs, batch_size=len(inputs))

        if all(score is not None for score in scores):
            rm_scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        duplicate_reward_keys = set(reward_extra_keys).intersection(batch.keys())
        if duplicate_reward_keys:
            raise ValueError(
                "Reward diagnostics must not shadow rollout tensors; rename the "
                f"reward fields {sorted(duplicate_reward_keys)}."
            )
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )
