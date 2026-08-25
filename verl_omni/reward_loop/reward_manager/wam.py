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
"""Reward manager for world-action models (WAMs)."""

import inspect
from collections.abc import Mapping
from typing import Any

import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score as _upstream_default_compute_score

from .visual import VisualRewardManager


def _copy_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Return an independent dictionary for an optional non-tensor field."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")


def _filter_score_kwargs(compute_score, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass only keyword arguments accepted by ``compute_score``.

    WAM rewards use ``responses`` and ``actions``. ``solution_image`` remains
    available as a compatibility alias for existing visual reward functions.
    """
    signature = inspect.signature(compute_score)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class WAMRewardManager(VisualRewardManager):
    """Reward manager for policies that jointly produce responses and actions.

    The preferred tensor contract is ``batch["responses"]`` plus
    ``batch["actions"]``. During streaming reward computation, older rollout
    adapters may still place actions in ``tool_extra_fields``; that location is
    supported as a compatibility fallback.

    Reward functions should use the following signature (sync or async)::

        def compute_score(responses, actions, ground_truth, extra_info, **kwargs):
            return {"score": 1.0, "action_reward": 0.5}

    ``responses`` and ``actions`` are detached before invocation. Reward
    gradients therefore never flow into the sampled action. Policy gradients
    are obtained separately by recomputing current-policy log probabilities for
    the recorded actions.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        if compute_score is None or compute_score is _upstream_default_compute_score:
            raise ValueError(
                "WAMRewardManager requires reward.custom_reward_function.path/name; "
                "the default visual reward function does not define a WAM action reward."
            )
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)

    @staticmethod
    def _extract_actions(data_item: DataProto, extra_info: dict[str, Any], tool_extra_fields: dict[str, Any]):
        actions = data_item.batch.get("actions", None)
        if actions is None:
            actions = tool_extra_fields.get("actions")
        if actions is None:
            actions = extra_info.get("actions")
        if actions is None:
            raise KeyError(
                'WAMRewardManager requires sampled actions in batch["actions"] or tool_extra_fields["actions"].'
            )
        return actions.detach() if isinstance(actions, torch.Tensor) else actions

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        responses = data_item.batch["responses"]
        if isinstance(responses, torch.Tensor):
            responses = responses.detach()

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = _copy_mapping(
            data_item.non_tensor_batch.get("extra_info", None),
            field_name="extra_info",
        )
        tool_extra_fields = _copy_mapping(
            data_item.non_tensor_batch.get("tool_extra_fields", None),
            field_name="tool_extra_fields",
        )
        extra_info.update(tool_extra_fields)

        actions = self._extract_actions(data_item, extra_info, tool_extra_fields)

        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
                "model_name": self.config.reward.reward_model.model_path,
            }
            if self.reward_router_address is not None
            else {}
        )
        score_kwargs = _filter_score_kwargs(
            self.compute_score,
            {
                "data_source": data_source,
                "responses": responses,
                "actions": actions,
                # Compatibility alias for reward functions shared with visual recipes.
                "solution_image": responses,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                **extra_reward_kwargs,
            },
        )

        if self.is_async_reward_score:
            result = await self.compute_score(**score_kwargs)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**score_kwargs))

        reward_extra_info = {}
        if isinstance(result, dict):
            required_rewards = ("score", "visual_reward", "action_reward")
            missing_rewards = [name for name in required_rewards if name not in result]
            if missing_rewards:
                raise KeyError(
                    "WAM reward result dictionaries must explicitly contain "
                    f"score, visual_reward, and action_reward; missing {missing_rewards}."
                )
            score = float(result["score"])
            reward_extra_info.update({key: value for key, value in result.items() if key != "score"})
        else:
            raise TypeError(
                "WAM rewards must return a dictionary with score, visual_reward, and action_reward; "
                f"got {type(result).__name__}."
            )

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
