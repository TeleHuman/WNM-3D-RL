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
"""Narrow CPU tests for diffusion trainer control-flow helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl_omni.trainer.diffusion import ray_diffusion_trainer


def test_dance_grpo_compute_advantage_forwards_normalization_config(monkeypatch):
    captured = {}

    def fake_estimator(**kwargs):
        captured.update(kwargs)
        rewards = kwargs["sample_level_rewards"]
        return torch.zeros_like(rewards), torch.ones_like(rewards)

    monkeypatch.setattr(ray_diffusion_trainer, "get_diffusion_adv_estimator_fn", lambda _: fake_estimator)
    data = SimpleNamespace(
        batch={"sample_level_rewards": torch.ones(4, 2)},
        non_tensor_batch={"uid": np.array(["a", "a", "b", "b"], dtype=object)},
    )

    result = ray_diffusion_trainer.compute_advantage(
        data,
        adv_estimator="dance_grpo",
        norm_adv_by_std_in_grpo=False,
        global_std=False,
    )

    assert captured["norm_adv_by_std_in_grpo"] is False
    assert captured["global_std"] is False
    assert result.batch["advantages"].shape == (4, 2)


def test_wnm_actor_transport_preserves_dense_prompt_and_prunes_rgb():
    batch_size = 2
    text_len = 4
    tensors = {
        "all_latents": torch.zeros(batch_size, 5, 1),
        "all_timesteps": torch.zeros(batch_size, 4),
        "all_action_latents": torch.zeros(batch_size, 5, 1, 1),
        "all_action_timesteps": torch.zeros(batch_size, 4),
        "dit_prediction_source_steps": torch.tensor([[0, 1, 1, 3]] * batch_size),
        "num_dit_prediction_steps": torch.full((batch_size,), 3, dtype=torch.long),
        "num_dit_forwards": torch.full((batch_size,), 6, dtype=torch.long),
        "num_inference_steps": torch.full((batch_size,), 4, dtype=torch.long),
        "true_cfg_scale": torch.full((batch_size,), 5.0),
        "noise_level": torch.full((batch_size,), 0.7),
        "action_noise_level": torch.full((batch_size,), 0.2),
        "prompt_embeds": torch.arange(batch_size * text_len * 3, dtype=torch.float32).reshape(batch_size, text_len, 3),
        "prompt_embeds_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        "clip_feature": torch.zeros(batch_size, 1, 1),
        "y": torch.zeros(batch_size, 1, 1, 1, 1),
        "state": torch.zeros(batch_size, 1, 1),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "clean_x": torch.zeros(batch_size, 1, 1, 1, 1),
        "past_clean_x": torch.zeros(batch_size, 1, 1, 1, 1),
        "old_log_probs": torch.zeros(batch_size, 4),
        "advantages": torch.zeros(batch_size, 4),
        "old_action_log_probs": torch.zeros(batch_size, 4),
        "action_advantages": torch.zeros(batch_size, 4),
        "action_stop_advantages": torch.ones(batch_size, 4),
        "action_collision_advantages": -torch.ones(batch_size, 4),
        "responses": torch.zeros(batch_size, 3, 3, 8, 8),
        "rm_scores": torch.zeros(batch_size, 1),
    }
    original_prompt = tensors["prompt_embeds"]
    data = SimpleNamespace(batch=TensorDict(tensors, batch_size=[batch_size]))

    selected, total_bytes, kept_bytes = ray_diffusion_trainer._prepare_wnm_actor_tensordict(
        data,
        expected_text_len=text_len,
    )

    assert not selected["prompt_embeds"].is_nested
    assert selected["prompt_embeds"].data_ptr() == original_prompt.data_ptr()
    assert torch.equal(selected["prompt_embeds_mask"], tensors["prompt_embeds_mask"])
    assert "action_stop_advantages" in selected
    assert "action_collision_advantages" in selected
    assert torch.equal(selected["action_noise_level"], tensors["action_noise_level"])
    assert torch.equal(
        selected["dit_prediction_source_steps"],
        tensors["dit_prediction_source_steps"],
    )
    assert "responses" not in selected
    assert "rm_scores" not in selected
    assert kept_bytes < total_bytes


def test_validation_media_slice_must_own_only_retained_storage():
    full_batch = torch.zeros(16, 4, 3, 8, 8)
    retained = ray_diffusion_trainer._clone_validation_media_prefix(full_batch, 1)

    assert retained.shape[0] == 1
    assert retained.untyped_storage().nbytes() == retained.numel() * retained.element_size()
    assert retained.untyped_storage().nbytes() < full_batch.untyped_storage().nbytes()


def test_v7_n8_assigns_two_branches_in_four_conditional_groups():
    data = SimpleNamespace(
        non_tensor_batch={},
        __len__=lambda self: 32,
    )

    # SimpleNamespace does not dispatch special methods from instances.
    class BatchStub:
        def __init__(self):
            self.non_tensor_batch = {}

        def __len__(self):
            return 16

    data = BatchStub()
    selected = ray_diffusion_trainer._assign_layer_conditioned_credit(
        data,
        rollout_n=8,
        branches_per_stratum=2,
        global_step=7,
        seed=42,
    )

    assert selected.shape == (4,)
    for step, candidates in zip(
        selected,
        ray_diffusion_trainer._WNM_LAYER_CREDIT_STRATA,
        strict=True,
    ):
        assert int(step) in candidates
    assert np.array_equal(data.non_tensor_batch["layer_credit_stratum"][:8], np.repeat(np.arange(4), 2))
    assert np.array_equal(data.non_tensor_batch["layer_credit_branch"][:8], np.tile(np.arange(2), 4))
    assert np.array_equal(
        data.non_tensor_batch["layer_credit_transition"][:8],
        selected[np.repeat(np.arange(4), 2)],
    )
    assert np.array_equal(
        data.non_tensor_batch["layer_credit_transition"][:8],
        data.non_tensor_batch["layer_credit_transition"][8:],
    )


def test_v7_conditional_group_index_separates_noise_strata():
    data = SimpleNamespace(
        non_tensor_batch={
            "uid": np.asarray(["prompt"] * 8, dtype=object),
            "layer_credit_stratum": np.repeat(np.arange(4), 2),
        }
    )
    index = ray_diffusion_trainer._layer_conditioned_group_index(data)

    assert len(np.unique(index)) == 4
    for stratum in range(4):
        assert len(np.unique(index[2 * stratum : 2 * stratum + 2])) == 1


def test_v7_compacts_each_rollout_to_its_credited_transition():
    batch_size = 4
    transitions = np.asarray([0, 6, 10, 15], dtype=np.int64)
    num_steps = 16
    transition_values = torch.arange(batch_size * num_steps).reshape(batch_size, num_steps)
    state_values = torch.arange(batch_size * (num_steps + 1)).reshape(batch_size, num_steps + 1, 1)
    source_map = torch.tensor([[0, 1, 2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 13, 14, 15]] * batch_size)
    tensors = {
        "all_latents": state_values.clone(),
        "all_action_latents": state_values.clone().unsqueeze(-1),
        "all_timesteps": transition_values.clone(),
        "all_action_timesteps": transition_values.clone(),
        "old_log_probs": transition_values.float(),
        "old_action_log_probs": transition_values.float(),
        "advantages": transition_values.float(),
        "action_advantages": transition_values.float(),
        "action_stop_advantages": -transition_values.float(),
        "returns": transition_values.float(),
        "dit_prediction_source_steps": source_map,
    }
    data = SimpleNamespace(
        batch=TensorDict(tensors, batch_size=[batch_size]),
        non_tensor_batch={"layer_credit_transition": transitions},
        __len__=lambda self: batch_size,
    )

    class DataStub:
        def __init__(self):
            self.batch = data.batch
            self.non_tensor_batch = data.non_tensor_batch

        def __len__(self):
            return batch_size

    compacted = ray_diffusion_trainer._compact_wnm_layer_credit_batch(DataStub())

    expected = transition_values[torch.arange(batch_size), torch.as_tensor(transitions)]
    assert compacted.batch["old_log_probs"].shape == (batch_size, 1)
    assert torch.equal(compacted.batch["old_log_probs"][:, 0], expected.float())
    assert torch.equal(
        compacted.batch["layer_credit_replay_transition"],
        torch.as_tensor(transitions),
    )
    assert compacted.batch["all_latents"].shape == (batch_size, 2, 1)
    assert compacted.batch["action_stop_advantages"].shape == (batch_size, 1)
    assert torch.equal(
        compacted.batch["action_stop_advantages"][:, 0],
        -expected.float(),
    )
    for row, transition in enumerate(transitions):
        assert torch.equal(
            compacted.batch["all_latents"][row, :, 0],
            state_values[row, transition : transition + 2, 0],
        )


def test_v7_separate_advantages_use_conditional_index(monkeypatch):
    captured_indices = []

    def fake_estimator(**kwargs):
        captured_indices.append(np.asarray(kwargs["index"], dtype=object))
        rewards = kwargs["sample_level_rewards"]
        return rewards - rewards.mean(), rewards - rewards.mean()

    monkeypatch.setattr(ray_diffusion_trainer, "get_diffusion_adv_estimator_fn", lambda _: fake_estimator)
    batch_size = 8
    data = SimpleNamespace(
        batch={
            "old_log_probs": torch.zeros(batch_size, 16),
            "sample_level_rewards": torch.zeros(batch_size, 16),
        },
        non_tensor_batch={
            "uid": np.asarray(["prompt"] * batch_size, dtype=object),
            "layer_credit_stratum": np.repeat(np.arange(4), 2),
        },
        __len__=lambda self: batch_size,
    )

    class DataStub:
        def __init__(self):
            self.batch = data.batch
            self.non_tensor_batch = data.non_tensor_batch

        def __len__(self):
            return batch_size

    stub = DataStub()
    conditional_index = ray_diffusion_trainer._layer_conditioned_group_index(stub)
    ray_diffusion_trainer.compute_separate_wam_advantages(
        stub,
        visual_rewards=torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1),
        action_rewards=torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1),
        adv_estimator="dance_grpo",
        global_std=False,
        index_override=conditional_index,
    )

    assert len(captured_indices) == 2
    assert all(np.array_equal(index, conditional_index) for index in captured_indices)
    assert len(np.unique(conditional_index)) == 4
    assert stub.batch["advantages"].shape == (batch_size, 1)
    assert stub.batch["action_advantages"].shape == (batch_size, 1)


def test_chunk_action_advantages_are_normalized_independently(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    captured_rewards = []

    def fake_estimator(**kwargs):
        rewards = kwargs["sample_level_rewards"]
        captured_rewards.append(rewards.clone())
        centered = rewards - rewards.mean(dim=0, keepdim=True)
        return centered, centered

    monkeypatch.setattr(
        ray_diffusion_trainer,
        "get_diffusion_adv_estimator_fn",
        lambda _: fake_estimator,
    )
    batch_size = 4

    class DataStub:
        def __init__(self):
            self.batch = {
                "old_log_probs": torch.zeros(batch_size, 16),
                "sample_level_rewards": torch.zeros(batch_size, 16),
            }
            self.non_tensor_batch = {"uid": np.asarray(["prompt"] * batch_size, dtype=object)}

        def __len__(self):
            return batch_size

    stub = DataStub()
    action_rewards = torch.tensor(
        [
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 11.0, 21.0, 31.0],
            [2.0, 12.0, 22.0, 32.0],
            [3.0, 13.0, 23.0, 33.0],
        ]
    )
    ray_diffusion_trainer.compute_separate_wam_advantages(
        stub,
        visual_rewards=torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1),
        action_rewards=action_rewards,
        adv_estimator="dance_grpo",
        global_std=False,
        index_override=np.asarray(["prompt"] * batch_size, dtype=object),
    )

    assert len(captured_rewards) == 5
    assert stub.batch["action_advantages"].shape == (batch_size, 1, 4)
    expected = torch.tensor([-1.5, -0.5, 0.5, 1.5])
    for chunk_index in range(4):
        torch.testing.assert_close(
            stub.batch["action_advantages"][:, 0, chunk_index],
            expected,
        )


def test_chunk_action_masks_exclude_post_collision_branches(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    captured_rewards = []

    def fake_estimator(**kwargs):
        rewards = kwargs["sample_level_rewards"]
        captured_rewards.append(rewards.clone())
        centered = rewards - rewards.mean(dim=0, keepdim=True)
        return centered, centered

    monkeypatch.setattr(
        ray_diffusion_trainer,
        "get_diffusion_adv_estimator_fn",
        lambda _: fake_estimator,
    )
    batch_size = 4

    class DataStub:
        def __init__(self):
            self.batch = {
                "old_log_probs": torch.zeros(batch_size, 1),
                "sample_level_rewards": torch.zeros(batch_size, 1),
            }
            self.non_tensor_batch = {"uid": np.asarray(["prompt"] * batch_size, dtype=object)}

        def __len__(self):
            return batch_size

    stub = DataStub()
    action_rewards = torch.tensor(
        [
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 11.0, 21.0, 31.0],
            [2.0, 12.0, 22.0, 32.0],
            [3.0, 13.0, 23.0, 33.0],
        ]
    )
    action_masks = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    ray_diffusion_trainer.compute_separate_wam_advantages(
        stub,
        visual_rewards=torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1),
        action_rewards=action_rewards,
        action_masks=action_masks,
        adv_estimator="dance_grpo",
        global_std=False,
        index_override=np.asarray(["prompt"] * batch_size, dtype=object),
    )

    # Visual, chunk 0, and chunk 1 call the estimator. Chunks with <=1 active
    # branch have exactly zero policy credit without invoking std normalization.
    assert len(captured_rewards) == 3
    expected = stub.batch["action_advantages"][:, 0]
    torch.testing.assert_close(
        expected[:, 0],
        torch.tensor([-1.5, -0.5, 0.5, 1.5]),
    )
    torch.testing.assert_close(
        expected[:, 1],
        torch.tensor([-0.5, 0.5, 0.0, 0.0]),
    )
    torch.testing.assert_close(expected[:, 2], torch.zeros(4))
    torch.testing.assert_close(expected[:, 3], torch.zeros(4))
    torch.testing.assert_close(
        stub.batch["action_loss_weights"][:, 0],
        torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_stop_credit_is_normalized_independently_from_navigation(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")

    def fake_estimator(**kwargs):
        rewards = kwargs["sample_level_rewards"]
        centered = rewards - rewards.mean(dim=0, keepdim=True)
        return centered, centered

    monkeypatch.setattr(
        ray_diffusion_trainer,
        "get_diffusion_adv_estimator_fn",
        lambda _: fake_estimator,
    )
    batch_size = 4

    class DataStub:
        def __init__(self):
            self.batch = {
                "old_log_probs": torch.zeros(batch_size, 1),
                "sample_level_rewards": torch.zeros(batch_size, 1),
            }
            self.non_tensor_batch = {"uid": np.asarray(["prompt"] * batch_size, dtype=object)}

        def __len__(self):
            return batch_size

    stub = DataStub()
    navigation_rewards = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    stop_rewards = -navigation_rewards
    masks = torch.ones_like(navigation_rewards)
    ray_diffusion_trainer.compute_separate_wam_advantages(
        stub,
        visual_rewards=torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1),
        action_rewards=navigation_rewards,
        action_masks=masks,
        action_stop_rewards=stop_rewards,
        action_stop_masks=masks,
        adv_estimator="dance_grpo",
        global_std=False,
        index_override=np.asarray(["prompt"] * batch_size, dtype=object),
    )

    assert stub.batch["action_advantages"].shape == (4, 1, 4)
    assert stub.batch["action_stop_advantages"].shape == (4, 1, 4)
    torch.testing.assert_close(
        stub.batch["action_stop_advantages"],
        -stub.batch["action_advantages"],
    )
    torch.testing.assert_close(
        stub.batch["action_loss_weights"],
        torch.ones(4, 1, 4),
    )
    torch.testing.assert_close(
        stub.batch["action_stop_loss_weights"],
        torch.ones(4, 1, 4),
    )


def test_collision_credit_is_normalized_independently_from_navigation(monkeypatch):
    monkeypatch.setenv("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")

    def fake_estimator(**kwargs):
        rewards = kwargs["sample_level_rewards"]
        centered = rewards - rewards.mean(dim=0, keepdim=True)
        return centered, centered

    monkeypatch.setattr(
        ray_diffusion_trainer,
        "get_diffusion_adv_estimator_fn",
        lambda _: fake_estimator,
    )

    class DataStub:
        def __init__(self):
            self.batch = {
                "old_log_probs": torch.zeros(4, 1),
                "sample_level_rewards": torch.zeros(4, 1),
            }
            self.non_tensor_batch = {"uid": np.asarray(["prompt"] * 4, dtype=object)}

        def __len__(self):
            return 4

    stub = DataStub()
    navigation = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    collision = -navigation
    masks = torch.ones_like(navigation)
    ray_diffusion_trainer.compute_separate_wam_advantages(
        stub,
        visual_rewards=torch.arange(4, dtype=torch.float32).reshape(-1, 1),
        action_rewards=navigation,
        action_masks=masks,
        action_collision_rewards=collision,
        action_collision_masks=masks,
        adv_estimator="dance_grpo",
        global_std=False,
        index_override=np.asarray(["prompt"] * 4, dtype=object),
    )

    assert stub.batch["action_collision_advantages"].shape == (4, 1, 4)
    torch.testing.assert_close(
        stub.batch["action_collision_advantages"],
        -stub.batch["action_advantages"],
    )


def test_event_validation_metrics_report_stop_and_collision_strata():
    extras = {
        "event_val_active": [1.0] * 4,
        "event_val_collision_precursor": [1, 0, 0, 0],
        "event_val_premature_stop_risk": [0, 1, 0, 0],
        "event_val_premature_stop_near": [0, 1, 0, 0],
        "event_val_premature_stop_far": [0, 0, 0, 0],
        "event_val_required_stop": [0, 0, 1, 0],
        "event_val_required_stop_core": [0, 0, 1, 0],
        "event_val_required_stop_mid": [0, 0, 0, 0],
        "event_val_required_stop_boundary": [0, 0, 0, 0],
        "event_val_near_goal_continue": [0, 0, 0, 1],
        "event_val_expected_stop": [0, 0, 1, 0],
        "event_val_distance_to_goal_m": [4.0, 2.0, 0.5, 1.7],
        # Row 0 contains a raw STOP candidate after its collision.  The compact
        # GN0 deployment diagnostics correctly exclude it from executed STOPs.
        "deployment_stop_emitted": [0, 1, 1, 0],
        "deployment_stop_success": [0, 0, 1, 0],
        "action_any_collision": [1, 0, 0, 0],
        "action_chunk_0_collision": [1, 0, 0, 0],
        "action_collision_soft_risk": [0.5, 0, 0, 0],
        "action_reward": [0.1, -0.2, 0.3, 0.4],
        "visual_reward": [0.2, 0.2, 0.2, 0.2],
    }
    metrics = ray_diffusion_trainer._event_validation_metrics(extras)

    assert metrics["val-event/collision_precursor/collision_rate"] == 1.0
    assert metrics["val-event/collision_precursor/stop_rate"] == 0.0
    assert metrics["val-event/required_stop/stop_rate"] == 1.0
    assert metrics["val-event/premature_stop_risk/stop_rate"] == 1.0
    assert metrics["val-event/near_goal_continue/continue_rate"] == 1.0
    assert metrics["val-event/stop/recall"] == 1.0
    assert metrics["val-event/stop/specificity"] == pytest.approx(2.0 / 3.0)


def test_event_validation_metrics_fail_closed_when_deployment_stop_is_missing():
    extras = {
        "event_val_active": [1.0],
        "event_val_collision_precursor": [0],
        "event_val_premature_stop_risk": [0],
        "event_val_premature_stop_near": [0],
        "event_val_premature_stop_far": [0],
        "event_val_required_stop": [1],
        "event_val_required_stop_core": [1],
        "event_val_required_stop_mid": [0],
        "event_val_required_stop_boundary": [0],
        "event_val_near_goal_continue": [0],
    }
    with pytest.raises(KeyError, match="deployment_stop_emitted"):
        ray_diffusion_trainer._event_validation_metrics(extras)


def test_validated_rollout_scalar_checks_recorded_server_value():
    batch = {"action_noise_level": torch.full((4,), 0.1)}
    assert ray_diffusion_trainer._validated_rollout_scalar(
        batch,
        "action_noise_level",
        expected=0.1,
        batch_size=4,
        context="validation",
    ) == pytest.approx(0.1)

    with pytest.raises(ValueError, match="disagrees with the requested value"):
        ray_diffusion_trainer._validated_rollout_scalar(
            batch,
            "action_noise_level",
            expected=0.2,
            batch_size=4,
            context="validation",
        )
