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

import pytest
import torch

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


def test_log_prob_mask_reduces_only_selected_event_dimensions():
    elementwise_log_prob = torch.tensor(
        [
            [[1.0, 100.0], [3.0, 100.0]],
            [[5.0, 100.0], [7.0, 100.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor([[True, False], [True, False]])

    reduced = FlowMatchSDEDiscreteScheduler._reduce_log_prob(elementwise_log_prob, mask)

    torch.testing.assert_close(reduced, torch.tensor([2.0, 6.0]))
    reduced.sum().backward()
    torch.testing.assert_close(
        elementwise_log_prob.grad,
        torch.tensor(
            [
                [[0.5, 0.0], [0.5, 0.0]],
                [[0.5, 0.0], [0.5, 0.0]],
            ]
        ),
    )


def test_log_prob_mask_rejects_empty_samples():
    log_prob = torch.randn(2, 4, 3)
    mask = torch.ones_like(log_prob, dtype=torch.bool)
    mask[1] = False

    with pytest.raises(ValueError, match="at least one event element"):
        FlowMatchSDEDiscreteScheduler._reduce_log_prob(log_prob, mask)


def test_scheduler_step_accepts_log_prob_mask():
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=10)
    scheduler.set_timesteps(2)
    sample = torch.zeros(2, 2, 2, dtype=torch.float32)
    model_output = torch.zeros_like(sample)
    mask = torch.tensor([[True, False], [False, True]])

    _, log_prob, _, _ = scheduler.step(
        model_output=model_output,
        timestep=scheduler.timesteps[0],
        sample=sample,
        generator=torch.Generator().manual_seed(0),
        log_prob_mask=mask,
        return_dict=False,
    )

    assert log_prob.shape == (2,)
    assert torch.isfinite(log_prob).all()
