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

"""CPU contract tests for the WNM-3D WAM registration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines import wnm_2d, wnm_3d, wnm_shared
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.pipelines.wnm_3d import (
    WNM3D,
    WNM3DPipelineWithLogProb,
)
from verl_omni.pipelines.wnm_3d.diffusers_training_adapter import (
    ARCHITECTURE,
    _validate_vggt_checkpoint,
)
from verl_omni.pipelines.wnm_3d.vllm_omni_rollout_adapter import (
    _grouped_vggt_past_obs_tokens,
)
from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    _prepare_wnm_actor_tensordict,
)


def _write_config(tmp_path, *, enabled: bool = True):
    config = {
        "action_head_cfg": {
            "config": {
                "use_vggt_geometry_adapter": enabled,
                "vggt_image_resolution": 512,
                "vggt_patch_size": 16,
                "vggt_adapter_dim": 512,
            }
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_registered_under_separate_architecture():
    assert DiffusionModelBase.get_class_by_name(ARCHITECTURE, "dance_grpo") is WNM3D
    assert VllmOmniPipelineBase.get_class(ARCHITECTURE, "dance_grpo") is WNM3DPipelineWithLogProb


def test_model_packages_do_not_cross_export_architectures():
    assert "WNM2D" in wnm_2d.__all__
    assert "WNM3D" not in wnm_2d.__all__
    assert "WNM3D" in wnm_3d.__all__
    assert "WNM2D" not in wnm_3d.__all__
    assert wnm_shared.__all__ == []


def test_vggt_contract_does_not_reuse_2d_clean_video_prefix():
    assert WNM3D.past_condition_key == "past_obs_tokens"
    assert WNM3D.use_clean_x_condition is False
    assert "clean_x" not in WNM3D.replay_condition_keys
    assert "embodiment_id" not in WNM3D.replay_condition_keys
    assert WNM3D.uses_embodiment_condition is False
    assert WNM3DPipelineWithLogProb.replay_condition_key == "past_obs_tokens"
    assert WNM3DPipelineWithLogProb.requires_clean_x is False
    assert WNM3DPipelineWithLogProb.uses_embodiment_condition is False


def test_checkpoint_contract_requires_vggt(tmp_path):
    _write_config(tmp_path, enabled=True)
    assert _validate_vggt_checkpoint(tmp_path)["use_vggt_geometry_adapter"] is True

    _write_config(tmp_path, enabled=False)
    with pytest.raises(ValueError, match="use_vggt_geometry_adapter=true"):
        _validate_vggt_checkpoint(tmp_path)


def test_actor_replays_450_wan_width_geometry_tokens():
    tokens = torch.zeros(2, 450, 3072)
    module = SimpleNamespace(patch_size=(1, 2, 2), dim=3072)
    result = WNM3D._prepare_past_condition(
        lambda name: tokens,
        batch_size=2,
        channels=48,
        height=10,
        width=20,
        inner_module=module,
    )
    assert result is tokens

    with pytest.raises(ValueError, match="past_obs_tokens"):
        WNM3D._prepare_past_condition(
            lambda name: tokens[:, :-1],
            batch_size=2,
            channels=48,
            height=10,
            width=20,
            inner_module=module,
        )


def test_vggt_condition_encoder_runs_b1_per_unique_prompt():
    calls = []

    class FakeHead:
        _verl_conditioning_representative_indices = (0, 2)
        _verl_conditioning_group_index = (0, 0, 1)

        def _build_vggt_past_obs_tokens(self, images, **kwargs):
            calls.append(tuple(images.shape))
            return images[:, :1, :1, :1, :1].reshape(images.shape[0], 1, 1).float()

    images = torch.tensor([1, 1, 7], dtype=torch.uint8).reshape(3, 1, 1, 1, 1)
    result = _grouped_vggt_past_obs_tokens(
        FakeHead(),
        images,
        target_frames=9,
        target_grid_size=(5, 10),
        device="cpu",
        dtype=torch.float32,
    )
    assert calls == [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)]
    assert result.reshape(-1).tolist() == [1.0, 1.0, 7.0]


def test_vggt_actor_transport_keeps_geometry_tokens_without_clean_prefix():
    batch_size = 2
    text_len = 4
    tensors = {
        "all_latents": torch.zeros(batch_size, 5, 1),
        "all_timesteps": torch.zeros(batch_size, 4),
        "all_action_latents": torch.zeros(batch_size, 5, 1, 1),
        "all_action_timesteps": torch.zeros(batch_size, 4),
        "dit_prediction_source_steps": torch.tensor([[0, 1, 1, 3]] * batch_size),
        "num_dit_prediction_steps": torch.full((batch_size,), 3),
        "num_dit_forwards": torch.full((batch_size,), 6),
        "num_inference_steps": torch.full((batch_size,), 4),
        "true_cfg_scale": torch.full((batch_size,), 5.0),
        "noise_level": torch.full((batch_size,), 0.7),
        "action_noise_level": torch.full((batch_size,), 0.2),
        "prompt_embeds": torch.zeros(batch_size, text_len, 3),
        "prompt_embeds_mask": torch.ones(batch_size, text_len, dtype=torch.bool),
        "clip_feature": torch.zeros(batch_size, 1, 1),
        "y": torch.zeros(batch_size, 1, 1, 1, 1),
        "state": torch.zeros(batch_size, 1, 1),
        "past_obs_tokens": torch.zeros(batch_size, 2, 3),
        "old_log_probs": torch.zeros(batch_size, 4),
        "advantages": torch.zeros(batch_size, 4),
        "old_action_log_probs": torch.zeros(batch_size, 4),
        "action_advantages": torch.zeros(batch_size, 4),
    }
    batch = SimpleNamespace(batch=TensorDict(tensors, batch_size=[batch_size]))
    selected, _, _ = _prepare_wnm_actor_tensordict(
        batch,
        expected_text_len=text_len,
        architecture=ARCHITECTURE,
    )
    assert "past_obs_tokens" in selected
    assert "embodiment_id" not in selected
    assert "clean_x" not in selected
    assert "past_clean_x" not in selected
