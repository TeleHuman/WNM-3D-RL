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
"""CPU tests for the one-forward world/action replay contract."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.model_base import WorldActionDiffusionModelBase
from verl_omni.pipelines.utils import normalize_reverse_step_output


class _JointModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0

    def forward(self, hidden_states, action, **kwargs):
        self.forward_calls += 1
        return hidden_states * self.scale, action * self.scale


class _DensityScheduler:
    def __init__(self):
        self.calls = 0
        self.log_prob_masks = []

    def sample_previous_step(self, *, sample, model_output, prev_sample, **kwargs):
        self.calls += 1
        self.log_prob_masks.append(kwargs.get("log_prob_mask"))
        mean = sample + model_output
        std = torch.ones_like(mean)
        log_prob = -((prev_sample.detach() - mean) ** 2).flatten(1).mean(dim=1)
        sqrt_dt = torch.ones(sample.shape[0], device=sample.device, dtype=sample.dtype)
        return prev_sample, log_prob, mean, std, sqrt_dt


def _model_config():
    return SimpleNamespace(
        algo=SimpleNamespace(
            noise_level=1.0,
            sde_type="sde",
            action_noise_level=None,
            action_sde_type=None,
        )
    )


def test_joint_replay_uses_one_model_forward_for_two_log_probs():
    batch_size = 2
    module = _JointModule()
    scheduler = _DensityScheduler()
    visual_trajectory = torch.randn(batch_size, 3, 2, 2, requires_grad=True)
    action_trajectory = torch.randn(batch_size, 3, 4, 3, requires_grad=True)
    scheduler_inputs = {
        "all_latents": visual_trajectory,
        "all_timesteps": torch.tensor([[2, 1], [2, 1]]),
        "all_action_latents": action_trajectory,
        "all_action_timesteps": torch.tensor([[2, 1], [2, 1]]),
    }

    output = WorldActionDiffusionModelBase.forward_and_sample_previous_step(
        module=module,
        scheduler=scheduler,
        model_config=_model_config(),
        model_inputs={
            "hidden_states": visual_trajectory[:, 0].detach(),
            "action": action_trajectory[:, 0].detach(),
        },
        negative_model_inputs=None,
        scheduler_inputs=scheduler_inputs,
        step=0,
    )

    assert module.forward_calls == 1
    assert scheduler.calls == 2
    assert scheduler.log_prob_masks == [None, None]
    assert output["log_probs"].requires_grad
    assert output["action_log_probs"].requires_grad

    joint_loss = -(output["log_probs"] + output["action_log_probs"]).mean()
    joint_loss.backward()
    assert module.scale.grad is not None
    assert torch.isfinite(module.scale.grad)
    assert visual_trajectory.grad is None
    assert action_trajectory.grad is None


def test_action_inputs_are_replayed_detached():
    action_trajectory = torch.randn(2, 3, 4, 3, requires_grad=True)
    action_timesteps = torch.tensor([[2, 1], [2, 1]], dtype=torch.int64)
    model_inputs = WorldActionDiffusionModelBase.inject_action_inputs(
        model_inputs={
            "x": torch.randn(2, 2, 2, dtype=torch.bfloat16),
            "hidden_states": torch.randn(2, 2, 2, dtype=torch.float32),
        },
        micro_batch={
            "all_action_latents": action_trajectory,
            "all_action_timesteps": action_timesteps,
        },
        step=0,
    )

    assert not model_inputs["action"].requires_grad
    assert model_inputs["action"].dtype == torch.bfloat16
    assert model_inputs["action"].device == model_inputs["x"].device
    torch.testing.assert_close(model_inputs["action"].float(), action_trajectory[:, 0].detach(), rtol=0.01, atol=0.01)
    assert tuple(model_inputs["timestep_action"].shape) == (2, 4)
    assert model_inputs["timestep_action"].dtype == action_timesteps.dtype
    assert model_inputs["timestep_action"].device == model_inputs["x"].device


def test_joint_replay_selects_transition_action_policy_mask():
    batch_size = 2
    module = _JointModule()
    scheduler = _DensityScheduler()
    visual_trajectory = torch.randn(batch_size, 3, 2, 2)
    action_trajectory = torch.randn(batch_size, 3, 4, 3)
    action_policy_mask = torch.zeros(batch_size, 2, 4, 3, dtype=torch.bool)
    action_policy_mask[:, 0, :2, :1] = True
    action_policy_mask[:, 1, 2:, :2] = True

    WorldActionDiffusionModelBase.forward_and_sample_previous_step(
        module=module,
        scheduler=scheduler,
        model_config=_model_config(),
        model_inputs={
            "hidden_states": visual_trajectory[:, 1].detach(),
            "action": action_trajectory[:, 1].detach(),
        },
        negative_model_inputs=None,
        scheduler_inputs={
            "all_latents": visual_trajectory,
            "all_timesteps": torch.tensor([[2, 1], [2, 1]]),
            "all_action_latents": action_trajectory,
            "all_action_timesteps": torch.tensor([[2, 1], [2, 1]]),
            "action_policy_mask": action_policy_mask,
        },
        step=1,
    )

    assert scheduler.log_prob_masks[0] is None
    assert torch.equal(scheduler.log_prob_masks[1], action_policy_mask[:, 1])


@pytest.mark.parametrize(
    ("visual_steps", "action_steps", "visual_states", "action_states", "match"),
    [
        (2, 1, 3, 2, "identical batch and transition counts"),
        (2, 2, 4, 3, "exactly one more state"),
        (2, 2, 3, 4, "exactly one more state"),
    ],
)
def test_joint_replay_rejects_misaligned_schedules_and_trajectories(
    visual_steps, action_steps, visual_states, action_states, match
):
    visual_trajectory = torch.randn(2, visual_states, 2, 2)
    action_trajectory = torch.randn(2, action_states, 4, 3)

    with pytest.raises(ValueError, match=match):
        WorldActionDiffusionModelBase.forward_and_sample_previous_step(
            module=_JointModule(),
            scheduler=_DensityScheduler(),
            model_config=_model_config(),
            model_inputs={
                "hidden_states": visual_trajectory[:, 0].detach(),
                "action": action_trajectory[:, 0].detach(),
            },
            negative_model_inputs=None,
            scheduler_inputs={
                "all_latents": visual_trajectory,
                "all_timesteps": torch.ones(2, visual_steps),
                "all_action_latents": action_trajectory,
                "all_action_timesteps": torch.ones(2, action_steps),
            },
            step=0,
        )


def test_reverse_step_output_keeps_both_density_groups():
    visual = torch.randn(2)
    action = torch.randn(2)
    output = normalize_reverse_step_output(
        {
            "log_probs": visual,
            "prev_sample_mean": torch.randn(2, 3),
            "std_dev_t": torch.ones(2, 1),
            "sqrt_dt": torch.ones(2),
            "action_log_probs": action,
            "action_prev_sample_mean": torch.randn(2, 4, 3),
            "action_std_dev_t": torch.ones(2, 1, 1),
            "action_sqrt_dt": torch.ones(2),
        }
    )

    assert output["log_probs"] is visual
    assert output["action_log_probs"] is action
