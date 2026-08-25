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

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl_omni.pipelines.schedulers.flow_match_sde import (
    FlowMatchSDEDiscreteScheduler,
)
from verl_omni.trainer.diffusion import diffusion_algos


def test_action_log_prob_reduction_preserves_four_temporal_chunks():
    elementwise = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
    reduced = FlowMatchSDEDiscreteScheduler._reduce_log_prob(
        elementwise,
        None,
        chunk_size=8,
    )

    assert reduced.shape == (2, 4)
    expected = elementwise.reshape(2, 4, 8, 3).mean(dim=(2, 3))
    torch.testing.assert_close(reduced, expected)


def test_flow_grpo_separate_visual_action_advantages_backpropagate_both_outputs():
    config = SimpleNamespace(diffusion_loss=SimpleNamespace(adv_clip_max=5.0, clip_ratio=10.0))
    visual_log_prob = torch.zeros(4, requires_grad=True)
    action_log_prob = torch.zeros(4, requires_grad=True)
    old_visual_log_prob = torch.full((4,), -0.1)
    old_action_log_prob = torch.full((4,), -0.2)
    advantages = torch.ones(4)

    loss, metrics = diffusion_algos.FlowGRPOLoss.compute_loss(
        old_log_prob=old_visual_log_prob,
        log_prob=visual_log_prob,
        old_action_log_prob=old_action_log_prob,
        action_log_prob=action_log_prob,
        advantages=advantages,
        action_advantages=torch.full((4,), 2.0),
        config=config,
    )
    loss.backward()

    expected_loss = -torch.exp(torch.tensor(0.1)) - 2.0 * torch.exp(torch.tensor(0.2))
    torch.testing.assert_close(loss.detach(), expected_loss)
    assert visual_log_prob.grad is not None
    assert action_log_prob.grad is not None
    torch.testing.assert_close(action_log_prob.grad, visual_log_prob.grad * (2.0 * torch.exp(torch.tensor(0.1))))
    assert "actor/visual_log_ratio_mean" in metrics
    assert "actor/action_log_ratio_mean" in metrics


def test_flow_grpo_separate_log_prob_weights_scale_losses_and_gradients():
    config = SimpleNamespace(
        diffusion_loss=SimpleNamespace(
            adv_clip_max=5.0,
            clip_ratio=10.0,
            visual_log_prob_weight=0.25,
            action_log_prob_weight=2.0,
        )
    )
    visual_log_prob = torch.zeros(4, requires_grad=True)
    action_log_prob = torch.zeros(4, requires_grad=True)
    old_visual_log_prob = torch.full((4,), -0.1)
    old_action_log_prob = torch.full((4,), -0.2)

    loss, metrics = diffusion_algos.FlowGRPOLoss.compute_loss(
        old_log_prob=old_visual_log_prob,
        log_prob=visual_log_prob,
        old_action_log_prob=old_action_log_prob,
        action_log_prob=action_log_prob,
        advantages=torch.ones(4),
        action_advantages=torch.ones(4),
        config=config,
    )
    loss.backward()

    expected_loss = -0.25 * torch.exp(torch.tensor(0.1)) - 2.0 * torch.exp(torch.tensor(0.2))
    torch.testing.assert_close(loss.detach(), expected_loss)
    expected_gradient_ratio = 8.0 * torch.exp(torch.tensor(0.1))
    torch.testing.assert_close(action_log_prob.grad, visual_log_prob.grad * expected_gradient_ratio)
    assert metrics["actor/visual_log_prob_weight"] == pytest.approx(0.25)
    assert metrics["actor/action_log_prob_weight"] == pytest.approx(2.0)


def test_flow_grpo_temporal_action_chunks_keep_independent_gradients(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    config = SimpleNamespace(
        diffusion_loss=SimpleNamespace(
            adv_clip_max=5.0,
            clip_ratio=10.0,
            visual_log_prob_weight=0.0,
            action_log_prob_weight=1.0,
        )
    )
    action_log_prob = torch.zeros(2, 4, requires_grad=True)
    old_action_log_prob = torch.zeros_like(action_log_prob)
    action_advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])

    loss, metrics = diffusion_algos.FlowGRPOLoss.compute_loss(
        old_log_prob=torch.zeros(2),
        log_prob=torch.zeros(2, requires_grad=True),
        old_action_log_prob=old_action_log_prob,
        action_log_prob=action_log_prob,
        advantages=torch.zeros(2),
        action_advantages=action_advantages,
        config=config,
    )
    loss.backward()

    normalized = torch.tensor([1.0, 0.5, 0.25, 0.125]) / 1.875
    torch.testing.assert_close(
        loss.detach(),
        -torch.sum(normalized * torch.tensor([1.0, 2.0, 3.0, 4.0])),
    )
    # The batch mean contributes 1/2; each chunk then receives only its own
    # configured decay and advantage rather than a full-horizon scalar.
    expected_grad = -0.5 * normalized * torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(action_log_prob.grad[0], expected_grad)
    torch.testing.assert_close(action_log_prob.grad[1], expected_grad)
    for index in range(4):
        assert f"actor/action_chunk_{index}_loss" in metrics


def test_flow_grpo_action_active_weights_normalize_over_active_rows(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,1,1,1")
    config = SimpleNamespace(
        diffusion_loss=SimpleNamespace(
            adv_clip_max=10.0,
            clip_ratio=10.0,
            visual_log_prob_weight=0.0,
            action_log_prob_weight=1.0,
        )
    )
    action_log_prob = torch.zeros(4, 4, requires_grad=True)
    action_advantages = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 7.0, 0.0, 0.0],
        ]
    )
    # Chunk 0 has two active rows and is scaled by B/active=2. Chunk 1 has all
    # rows active and keeps unit weight. Empty chunks remain exactly zero.
    active_weights = torch.tensor(
        [
            [2.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    loss, metrics = diffusion_algos.FlowGRPOLoss.compute_loss(
        old_log_prob=torch.zeros(4),
        log_prob=torch.zeros(4, requires_grad=True),
        old_action_log_prob=torch.zeros_like(action_log_prob),
        action_log_prob=action_log_prob,
        advantages=torch.zeros(4),
        action_advantages=action_advantages,
        action_loss_weights=active_weights,
        config=config,
    )
    loss.backward()

    # Equal chunk weighting: -mean_active([1,3])/4 - mean([1,3,5,7])/4.
    torch.testing.assert_close(loss.detach(), torch.tensor(-1.5))
    torch.testing.assert_close(
        action_log_prob.grad,
        -action_advantages * active_weights / 16.0,
    )
    assert metrics["actor/action_active_normalization_enabled"] == 1.0
    assert metrics["actor/action_chunk_0_active_fraction"] == pytest.approx(0.5)
    assert metrics["actor/action_chunk_0_active_normalization_gain"] == pytest.approx(2.0)


def test_flow_grpo_stop_credit_uses_uniform_chunk_weights(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    config = SimpleNamespace(
        diffusion_loss=SimpleNamespace(
            adv_clip_max=5.0,
            clip_ratio=10.0,
            visual_log_prob_weight=0.0,
            action_log_prob_weight=1.0,
            action_stop_loss_weight=0.5,
        )
    )
    action_log_prob = torch.zeros(2, 4, requires_grad=True)
    loss, metrics = diffusion_algos.FlowGRPOLoss.compute_loss(
        old_log_prob=torch.zeros(2),
        log_prob=torch.zeros(2, requires_grad=True),
        old_action_log_prob=torch.zeros_like(action_log_prob),
        action_log_prob=action_log_prob,
        advantages=torch.zeros(2),
        action_advantages=torch.zeros(2, 4),
        action_stop_advantages=torch.ones(2, 4),
        config=config,
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(-0.5))
    # stop weight 0.5, then uniform means over B=2 and C=4.
    torch.testing.assert_close(
        action_log_prob.grad,
        torch.full((2, 4), -0.5 / 8.0),
    )
    assert metrics["actor/action_stop_loss_weight"] == pytest.approx(0.5)
    assert metrics["actor/action_stop_pg_loss"] == pytest.approx(-1.0)
    for index in range(4):
        assert metrics[f"actor/action_chunk_{index}_stop_loss"] == pytest.approx(-1.0)


def test_flow_grpo_collision_credit_uses_independent_uniform_weight(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    config = SimpleNamespace(
        diffusion_loss=SimpleNamespace(
            adv_clip_max=5.0,
            clip_ratio=10.0,
            visual_log_prob_weight=0.0,
            action_log_prob_weight=1.0,
            action_collision_loss_weight=0.5,
            action_collision_use_chunk_weights=False,
        )
    )
    action_log_prob = torch.zeros(2, 4, requires_grad=True)
    loss, metrics = diffusion_algos.FlowGRPOLoss.compute_loss(
        old_log_prob=torch.zeros(2),
        log_prob=torch.zeros(2, requires_grad=True),
        old_action_log_prob=torch.zeros_like(action_log_prob),
        action_log_prob=action_log_prob,
        advantages=torch.zeros(2),
        action_advantages=torch.zeros(2, 4),
        action_collision_advantages=torch.ones(2, 4),
        config=config,
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(-0.5))
    torch.testing.assert_close(
        action_log_prob.grad,
        torch.full((2, 4), -0.5 / 8.0),
    )
    assert metrics["actor/action_collision_loss_weight"] == pytest.approx(0.5)
    assert metrics["actor/action_collision_pg_loss"] == pytest.approx(-1.0)
    for index in range(4):
        assert metrics[f"actor/action_chunk_{index}_collision_loss"] == pytest.approx(-1.0)


def test_joint_log_prob_normalizes_temporal_action_chunk_weights(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    current, old = diffusion_algos.combine_visual_action_log_probs(
        log_prob=torch.tensor([2.0]),
        old_log_prob=torch.tensor([1.0]),
        action_log_prob=torch.tensor([[1.0, 2.0, 4.0, 8.0]]),
        old_action_log_prob=torch.zeros(1, 4),
        visual_log_prob_weight=0.25,
        action_log_prob_weight=2.0,
    )
    normalized = torch.tensor([1.0, 0.5, 0.25, 0.125]) / 1.875
    torch.testing.assert_close(
        current,
        torch.tensor([0.5 + 2.0 * torch.sum(normalized * torch.tensor([1.0, 2.0, 4.0, 8.0]))]),
    )
    torch.testing.assert_close(old, torch.tensor([0.25]))


def test_visual_only_log_prob_rejects_zero_visual_weight():
    with pytest.raises(ValueError, match="visual_log_prob_weight must be positive"):
        diffusion_algos.combine_visual_action_log_probs(
            log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            visual_log_prob_weight=0.0,
            action_log_prob_weight=1.0,
        )


def test_wam_kl_adds_visual_and_action_divergence():
    visual_mean = torch.zeros(2, 3, 4)
    action_mean = torch.zeros(2, 8, 3)

    loss, metrics = diffusion_algos.KLLoss.compute_loss(
        prev_sample_mean=visual_mean,
        ref_prev_sample_mean=torch.ones_like(visual_mean),
        std_dev_t=torch.ones(2, 1, 1),
        action_prev_sample_mean=action_mean,
        ref_action_prev_sample_mean=torch.ones_like(action_mean),
        action_std_dev_t=torch.ones(2, 1, 1),
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert metrics["actor/visual_kl_loss"] == pytest.approx(0.5)
    assert metrics["actor/action_kl_loss"] == pytest.approx(0.5)
    assert metrics["actor/kl_loss"] == pytest.approx(1.0)


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    batch_size = 8
    steps = 10
    sample_level_rewards = torch.randn((batch_size, steps), dtype=torch.float32)
    uid = np.array([f"uid-{idx}" for idx in range(batch_size)], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=sample_level_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)


def test_dance_grpo_loss_registered_and_callable():
    """``dance_grpo`` loss function is registered and can be invoked."""
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 8
    rollout_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    advantages = torch.randn((batch_size,), dtype=torch.float32)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=dance_grpo",
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    dance_grpo_loss = diffusion_algos.get_diffusion_loss_fn("dance_grpo")
    pg_loss, pg_metrics = dance_grpo_loss.compute_loss(
        old_log_prob=rollout_log_probs,
        log_prob=current_log_probs,
        advantages=advantages,
        config=actor_config,
    )

    assert pg_loss.shape == ()
    assert isinstance(pg_loss.item(), float)
    for key in ("actor/ppo_kl", "actor/pg_clipfrac", "actor/pg_clipfrac_higher", "actor/pg_clipfrac_lower"):
        assert key in pg_metrics


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_grouped_uids(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    """Exercises the len > 1 branch: multiple samples sharing the same prompt UID."""
    steps = 5
    # 4 samples: uid-0 × 2, uid-1 × 2  →  2 groups of size 2
    group_rewards = torch.tensor(
        [[1.0] * steps, [3.0] * steps, [0.0] * steps, [2.0] * steps],
        dtype=torch.float32,
    )
    uid = np.array(["uid-0", "uid-0", "uid-1", "uid-1"], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=group_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (4, steps)

    if not norm_adv_by_std_in_grpo:
        # Without std scaling: advantage = reward - group_mean
        # group uid-0 mean = (1+3)/2 = 2.0  →  advantages: -1, +1
        # group uid-1 mean = (0+2)/2 = 1.0  →  advantages: -1, +1
        torch.testing.assert_close(advantages[0], torch.full((steps,), -1.0))
        torch.testing.assert_close(advantages[1], torch.full((steps,), 1.0))
        torch.testing.assert_close(advantages[2], torch.full((steps,), -1.0))
        torch.testing.assert_close(advantages[3], torch.full((steps,), 1.0))
    else:
        # With std scaling: mean should be 0 for each group
        torch.testing.assert_close(advantages[0:2].mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(advantages[2:4].mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)
