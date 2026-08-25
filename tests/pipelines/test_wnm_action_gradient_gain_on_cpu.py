# Copyright 2026 WNM-3D-RL contributors
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

"""CPU tests for backward-only WNM action gradient gain."""

import pytest
import torch

from verl_omni.pipelines.wnm_shared.action_gradient_gain import (
    install_action_backbone_gradient_gain,
)


class _TinyJointDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(3, 3, bias=False)
        self.action_decoder = torch.nn.Linear(3, 2, bias=False)

    def forward(self, value):
        hidden = self.shared(value)
        return hidden, self.action_decoder(hidden)


def _run(gain: float):
    torch.manual_seed(0)
    module = _TinyJointDiT()
    install_action_backbone_gradient_gain(module, gain=gain)
    value = torch.randn(4, 3)
    visual, action = module(value)
    action.square().sum().backward()
    return (
        visual.detach(),
        action.detach(),
        module.shared.weight.grad.detach(),
        module.action_decoder.weight.grad.detach(),
    )


def test_action_gradient_gain_changes_only_shared_backward():
    baseline = _run(1.0)
    gained = _run(4.0)

    torch.testing.assert_close(gained[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(gained[1], baseline[1], rtol=0, atol=0)
    torch.testing.assert_close(gained[2], baseline[2] * 4.0)
    torch.testing.assert_close(gained[3], baseline[3])


def test_action_gradient_gain_is_idempotent_and_fail_closed():
    module = _TinyJointDiT()
    assert install_action_backbone_gradient_gain(module, gain=2.0) == 2.0
    assert install_action_backbone_gradient_gain(module, gain=2.0) == 2.0
    with pytest.raises(RuntimeError, match="already installed"):
        install_action_backbone_gradient_gain(module, gain=3.0)
    with pytest.raises(ValueError, match="finite and positive"):
        install_action_backbone_gradient_gain(_TinyJointDiT(), gain=0.0)
