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
"""CPU tests for the WAM reward-manager data contract."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto

from verl_omni.reward_loop.reward_manager import WAMRewardManager
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


def _make_config():
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.reward_model.enable = False
    return config


def _make_data(*, actions_in_batch: bool = True) -> DataProto:
    actions = torch.randn(1, 8, 3, requires_grad=True)
    tensors = {"responses": torch.randn(1, 2, 3, 16, 16)}
    non_tensors = {
        "data_source": ["wam_navigation"],
        "reward_model": [{"ground_truth": {"success": True}}],
        "extra_info": [{"task_id": "task-1"}],
    }
    if actions_in_batch:
        tensors["actions"] = actions
    else:
        non_tensors["tool_extra_fields"] = [{"actions": actions[0], "initial_state": [0.0, 0.0, 0.0]}]
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_wam_reward_reads_native_actions_and_detaches_inputs():
    observed = {}

    def compute_score(responses, actions, ground_truth, extra_info):
        observed["responses_requires_grad"] = responses.requires_grad
        observed["actions_requires_grad"] = actions.requires_grad
        observed["actions_shape"] = tuple(actions.shape)
        observed["ground_truth"] = ground_truth
        observed["task_id"] = extra_info["task_id"]
        return {
            "score": 1.25,
            "visual_reward": 1.0,
            "task_reward": 1.0,
            "action_reward": 0.25,
        }

    manager = WAMRewardManager(_make_config(), MagicMock(), compute_score)
    result = manager.loop.run_until_complete(manager.run_single(_make_data()))

    assert result["reward_score"] == pytest.approx(1.25)
    assert result["reward_extra_info"] == {
        "visual_reward": 1.0,
        "task_reward": 1.0,
        "action_reward": 0.25,
    }
    assert observed == {
        "responses_requires_grad": False,
        "actions_requires_grad": False,
        "actions_shape": (8, 3),
        "ground_truth": {"success": True},
        "task_id": "task-1",
    }


def test_wam_reward_reads_streaming_tool_extra_fields_fallback():
    def compute_score(responses, actions, ground_truth, extra_info):
        assert tuple(actions.shape) == (8, 3)
        assert extra_info["initial_state"] == [0.0, 0.0, 0.0]
        return {"score": 0.75, "visual_reward": 0.5, "action_reward": 0.25}

    manager = WAMRewardManager(_make_config(), MagicMock(), compute_score)
    result = manager.loop.run_until_complete(manager.run_single(_make_data(actions_in_batch=False)))

    assert result["reward_score"] == pytest.approx(0.75)
    assert result["reward_extra_info"]["action_reward"] == pytest.approx(0.25)


def test_wam_reward_supports_async_functions():
    async def compute_score(responses, actions, **kwargs):
        return {"score": 0.5, "visual_reward": 0.3, "action_reward": 0.2, "temporal_reward": 0.2}

    manager = WAMRewardManager(_make_config(), MagicMock(), compute_score)
    result = manager.loop.run_until_complete(manager.run_single(_make_data()))

    assert result["reward_score"] == pytest.approx(0.5)
    assert result["reward_extra_info"]["temporal_reward"] == pytest.approx(0.2)


def test_wam_reward_keeps_solution_image_compatibility_alias():
    def compute_score(data_source, solution_image, ground_truth, extra_info):
        assert data_source == "wam_navigation"
        assert tuple(solution_image.shape) == (2, 3, 16, 16)
        return {"score": 0.4, "visual_reward": 0.3, "action_reward": 0.1}

    manager = WAMRewardManager(_make_config(), MagicMock(), compute_score)
    result = manager.loop.run_until_complete(manager.run_single(_make_data()))

    assert result["reward_score"] == pytest.approx(0.4)


def test_wam_reward_requires_actions():
    data = DataProto.from_dict(
        tensors={"responses": torch.randn(1, 3, 16, 16)},
        non_tensors={
            "data_source": ["wam_navigation"],
            "reward_model": [{"ground_truth": None}],
            "extra_info": [{}],
        },
    )
    manager = WAMRewardManager(
        _make_config(),
        MagicMock(),
        lambda **kwargs: {"score": 0.0, "visual_reward": 0.0, "action_reward": 0.0},
    )

    with pytest.raises(KeyError, match="requires sampled actions"):
        manager.loop.run_until_complete(manager.run_single(data))


def test_diffusion_output_exposes_actions_as_a_first_class_field():
    actions = torch.randn(8, 3)
    action_log_probs = torch.randn(4)
    output = DiffusionOutput(
        diffusion_output=torch.randn(2, 3, 16, 16),
        actions=actions,
        log_probs=torch.randn(4),
        action_log_probs=action_log_probs,
    )

    assert output.actions is actions
    assert output.action_log_probs is action_log_probs


def test_vllm_server_keeps_visual_and_action_log_probs_separate():
    server = object.__new__(vLLMOmniHttpServer)
    server.global_steps = 3
    final_result = MagicMock()
    final_result.images = [torch.randn(2, 3, 16, 16)]
    final_result.custom_output = {
        "actions": torch.randn(1, 8, 3),
        "action_log_probs": torch.randn(1, 4),
        "all_log_probs": torch.randn(1, 4),
        "all_latents": torch.randn(1, 5, 2, 3, 16, 16),
        "all_timesteps": torch.arange(4).unsqueeze(0),
        "all_action_latents": torch.randn(1, 5, 8, 3),
        "all_action_timesteps": torch.arange(4).unsqueeze(0),
    }
    final_result.request_output = None

    output = server._process_output(final_result, params=MagicMock(), sampling_params={"logprobs": True})

    assert tuple(output.actions.shape) == (8, 3)
    assert torch.equal(output.log_probs, final_result.custom_output["all_log_probs"][0])
    assert torch.equal(output.action_log_probs, final_result.custom_output["action_log_probs"][0])
    assert "actions" not in output.extra_fields
    assert "action_log_probs" not in output.extra_fields
    assert tuple(output.extra_fields["all_action_latents"].shape) == (5, 8, 3)
    assert tuple(output.extra_fields["all_action_timesteps"].shape) == (4,)


def test_vllm_server_rejects_actions_without_action_log_probs():
    server = object.__new__(vLLMOmniHttpServer)
    server.global_steps = 0
    final_result = MagicMock()
    final_result.images = [torch.randn(2, 3, 16, 16)]
    final_result.custom_output = {
        "actions": torch.randn(1, 8, 3),
        "all_log_probs": torch.randn(1, 4),
        "all_latents": torch.randn(1, 5, 2, 3, 16, 16),
        "all_timesteps": torch.arange(4).unsqueeze(0),
        "all_action_latents": torch.randn(1, 5, 8, 3),
        "all_action_timesteps": torch.arange(4).unsqueeze(0),
    }
    final_result.request_output = None

    with pytest.raises(KeyError, match="action_log_probs"):
        server._process_output(final_result, params=MagicMock(), sampling_params={"logprobs": True})


def test_vllm_server_rejects_wam_actions_without_visual_log_probs():
    server = object.__new__(vLLMOmniHttpServer)
    server.global_steps = 0
    final_result = MagicMock()
    final_result.images = [torch.randn(2, 3, 16, 16)]
    final_result.custom_output = {
        "actions": torch.randn(1, 8, 3),
        "action_log_probs": torch.randn(1, 4),
        "all_latents": torch.randn(1, 5, 2, 3, 16, 16),
        "all_timesteps": torch.arange(4).unsqueeze(0),
        "all_action_latents": torch.randn(1, 5, 8, 3),
        "all_action_timesteps": torch.arange(4).unsqueeze(0),
    }
    final_result.request_output = None

    with pytest.raises(KeyError, match="all_log_probs"):
        server._process_output(final_result, params=MagicMock(), sampling_params={"logprobs": True})


def test_vllm_server_rejects_misaligned_wam_trajectory_lengths():
    server = object.__new__(vLLMOmniHttpServer)
    server.global_steps = 0
    final_result = MagicMock()
    final_result.images = [torch.randn(2, 3, 16, 16)]
    final_result.custom_output = {
        "actions": torch.randn(1, 8, 3),
        "all_log_probs": torch.randn(1, 4),
        "action_log_probs": torch.randn(1, 3),
        "all_latents": torch.randn(1, 5, 2, 3, 16, 16),
        "all_timesteps": torch.arange(4).unsqueeze(0),
        "all_action_latents": torch.randn(1, 4, 8, 3),
        "all_action_timesteps": torch.arange(3).unsqueeze(0),
    }
    final_result.request_output = None

    with pytest.raises(ValueError, match="trajectory lengths"):
        server._process_output(final_result, params=MagicMock(), sampling_params={"logprobs": True})
