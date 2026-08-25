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

"""CPU tests for the WNM2D training adapter."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.wnm_2d import WNM2D
from verl_omni.pipelines.wnm_2d import diffusers_training_adapter as wnm_adapter


class _TinyCausalWanModel(torch.nn.Module):
    """Small stand-in whose parameters exercise meta checkpoint assignment."""

    def __init__(
        self,
        *,
        dim: int = 12,
        num_heads: int = 2,
        in_dim: int = 2,
        text_len: int = 4,
        max_state_dim: int = 5,
        num_frame_per_block: int = 1,
        num_action_per_block: int = 1,
        action_dim: int = 3,
        patch_size=(1, 2, 2),
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.dim = dim
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.text_len = text_len
        self.max_state_dim = max_state_dim
        self.num_frame_per_block = num_frame_per_block
        self.num_action_per_block = num_action_per_block
        self.action_dim = action_dim
        self.patch_size = tuple(patch_size)
        self.proj = torch.nn.Linear(3, 4)
        self.action_proj = torch.nn.Linear(2, 2, bias=False)
        self.register_buffer("scale", torch.ones(1))
        self.freqs_action = torch.empty(0)
        self.freqs_state = torch.empty(0)
        self.freqs = []
        self.forward_calls = 0
        self.gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        del module
        self.gradient_checkpointing = value

    def forward(self, **kwargs):
        self.forward_calls += 1
        visual_scale = self.proj.weight.mean() + self.proj.bias.mean()
        action_scale = self.action_proj.weight.mean()
        return kwargs["x"] * visual_scale, kwargs["action"] * action_scale


def _fake_rope_params(length: int, dim: int) -> torch.Tensor:
    return torch.zeros(length, max(dim // 2, 1), dtype=torch.complex64)


@pytest.fixture
def fake_gammanav(monkeypatch):
    package_names = [
        "gammanav",
        "gammanav.vln",
        "gammanav.vln.model",
        "gammanav.vln.model.wnm_3d",
        "gammanav.vln.model.wnm_3d.modules",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    causal_module_name = "gammanav.vln.model.wnm_3d.modules.wan_video_dit_action_casual_chunk"
    causal_module = types.ModuleType(causal_module_name)
    causal_module.CausalWanModel = _TinyCausalWanModel
    monkeypatch.setitem(sys.modules, causal_module_name, causal_module)

    submodule_name = "gammanav.vln.model.wnm_3d.modules.wan2_1_submodule"
    submodule = types.ModuleType(submodule_name)
    submodule.rope_params = _fake_rope_params
    monkeypatch.setitem(sys.modules, submodule_name, submodule)
    # Attention backend wiring is covered by runtime integration tests. These
    # loader tests deliberately install only the tiny joint-DiT stand-in.
    monkeypatch.setattr(
        wnm_adapter,
        "_configure_explicit_sdpa_runtime",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        wnm_adapter,
        "_validate_explicit_sdpa",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        wnm_adapter,
        "install_action_backbone_gradient_gain",
        lambda module: 1.0,
    )


def _write_config(checkpoint_dir, *, dim: int = 12) -> None:
    config = {
        "action_head_cfg": {
            "config": {
                "diffusion_model_cfg": {
                    "_target_": ("gammanav.vln.model.wnm_3d.modules.wan_video_dit_action_casual_chunk.CausalWanModel"),
                    "_convert_": "object",
                    "dim": dim,
                    "num_heads": 2,
                    "in_dim": 2,
                    "text_len": 4,
                    "max_state_dim": 5,
                    "num_frame_per_block": 1,
                    "num_action_per_block": 1,
                    "action_dim": 3,
                    "patch_size": [1, 2, 2],
                }
            }
        }
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _reference_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {key: value.detach().clone() for key, value in _TinyCausalWanModel().state_dict().items()}


def _model_config(
    checkpoint_dir,
    *,
    steps: int = 16,
    true_cfg_scale: float = 5.0,
    sde_type: str = "dance_sde",
    action_sde_type: str | None = None,
):
    return SimpleNamespace(
        local_path=str(checkpoint_dir),
        architecture="WNM2D",
        algorithm="dance_grpo",
        external_lib=None,
        pipeline=SimpleNamespace(num_inference_steps=steps, true_cfg_scale=true_cfg_scale),
        algo=SimpleNamespace(
            noise_level=0.7,
            sde_type=sde_type,
            action_noise_level=None,
            action_sde_type=action_sde_type,
        ),
    )


def _raw_key(name: str, *, peft_prefix: bool = False, base_layer: bool = False) -> str:
    prefix = "action_head.model.base_model.model." if peft_prefix else "action_head.model."
    if base_layer and "." in name:
        owner, leaf = name.rsplit(".", 1)
        name = f"{owner}.base_layer.{leaf}"
    return prefix + name


def _write_single_checkpoint(checkpoint_dir, state, *, omit=(), wrong_shape=None) -> None:
    tensors = {}
    for name, tensor in state.items():
        if name in omit:
            continue
        value = tensor
        if name == wrong_shape:
            value = torch.zeros(*(list(tensor.shape) + [1]), dtype=tensor.dtype)
        tensors[_raw_key(name)] = value
    tensors["action_head.text_encoder.unused"] = torch.ones(1)
    save_file(tensors, checkpoint_dir / "model.safetensors")


def test_registration_resolves_dance_grpo_adapter():
    config = SimpleNamespace(
        architecture="WNM2D",
        algorithm="dance_grpo",
        external_lib=None,
    )
    assert DiffusionModelBase.get_class(config) is WNM2D


def test_local_single_file_loader_assigns_meta_state_and_rebuilds_freqs(tmp_path, fake_gammanav):
    _write_config(tmp_path)
    reference = _reference_state()
    _write_single_checkpoint(tmp_path, reference)

    module = WNM2D.build_module(_model_config(tmp_path), torch.float32)

    assert module._no_split_modules == ["CausalWanAttentionBlock"]
    assert all(not tensor.is_meta for tensor in module.state_dict().values())
    for name, expected in reference.items():
        torch.testing.assert_close(module.state_dict()[name], expected)
    assert module.freqs_action.shape[0] == 10240
    assert module.freqs_state.shape[0] == 1024
    assert len(module.freqs) == 3
    assert not any(freq.is_meta for freq in [module.freqs_action, module.freqs_state, *module.freqs])

    checkpoint_func = lambda function, *args: function(*args)
    module._set_gradient_checkpointing(enable=False)
    assert module.gradient_checkpointing is False
    module._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=checkpoint_func)
    assert module.gradient_checkpointing is True
    assert module._gradient_checkpointing_func is checkpoint_func


def test_local_sharded_loader_supports_peft_base_layer_prefix(tmp_path, fake_gammanav):
    _write_config(tmp_path)
    reference = _reference_state()
    items = list(reference.items())
    midpoint = len(items) // 2
    shards = [items[:midpoint], items[midpoint:]]
    weight_map = {}
    for shard_index, shard_items in enumerate(shards, start=1):
        filename = f"model-0000{shard_index}-of-00002.safetensors"
        shard_state = {}
        for name, tensor in shard_items:
            raw = _raw_key(name, peft_prefix=True, base_layer="." in name)
            shard_state[raw] = tensor
            weight_map[raw] = filename
        save_file(shard_state, tmp_path / filename)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    module = WNM2D.build_module(_model_config(tmp_path), torch.float32)

    for name, expected in reference.items():
        torch.testing.assert_close(module.state_dict()[name], expected)


@pytest.mark.parametrize(
    ("omit", "wrong_shape", "error"),
    [
        (("proj.bias",), None, KeyError),
        ((), "proj.weight", ValueError),
    ],
)
def test_local_loader_fails_closed_on_missing_or_wrong_shape(
    tmp_path,
    fake_gammanav,
    omit,
    wrong_shape,
    error,
):
    _write_config(tmp_path)
    _write_single_checkpoint(tmp_path, _reference_state(), omit=omit, wrong_shape=wrong_shape)

    with pytest.raises(error):
        WNM2D.build_module(_model_config(tmp_path), torch.float32)


def test_large_joint_dit_rejects_float32_before_materialization(tmp_path, fake_gammanav):
    _write_config(tmp_path, dim=1024)

    with pytest.raises(ValueError, match="fsdp_config.model_dtype=bfloat16"):
        WNM2D.build_module(_model_config(tmp_path), torch.float32)


def test_scheduler_requires_16_steps_and_uses_shift_five():
    scheduler = FlowMatchSDEDiscreteScheduler()
    WNM2D.set_timesteps(scheduler, _model_config(".", steps=16), "cpu")

    expected_first_two = torch.tensor([1.0, (5.0 * 15 / 16) / (1 + 4 * 15 / 16)])
    torch.testing.assert_close(scheduler.sigmas[:2].cpu(), expected_first_two)
    assert scheduler.timesteps.shape[0] == 16

    with pytest.raises(ValueError, match="requires 16 denoising steps"):
        WNM2D.set_timesteps(scheduler, _model_config(".", steps=8), "cpu")

    with pytest.raises(ValueError, match="requires.*dance_sde"):
        WNM2D.set_timesteps(scheduler, _model_config(".", sde_type="sde"), "cpu")

    with pytest.raises(ValueError, match="requires.*dance_sde"):
        WNM2D.set_timesteps(
            scheduler,
            _model_config(".", action_sde_type="cps"),
            "cpu",
        )


def test_fixed_dit_cache_source_steps_match_deployed_mask():
    assert wnm_adapter._DIT_PREDICTION_SOURCE_STEPS == (0, 1, 2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 13, 14, 15)


def test_prepare_model_inputs_builds_cfg_past_clean_joint_call_without_grad():
    module = _TinyCausalWanModel().to(dtype=torch.bfloat16)
    batch_size = 2
    policy_steps = 16
    frames = 3
    latents = torch.randn(batch_size, policy_steps + 1, 2, frames, 4, 6, requires_grad=True)
    timesteps = (torch.arange(policy_steps, 0, -1, dtype=torch.float32) + 0.375).repeat(batch_size, 1)
    prompt_embeds = torch.randn(batch_size, 4, 6, requires_grad=True)
    prompt_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    negative_prompt_embeds = torch.randn(batch_size, 4, 6, requires_grad=True)
    negative_prompt_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
    conditions = {
        "clip_feature": torch.randn(batch_size, 3, 4, requires_grad=True),
        "y": torch.randn(batch_size, 2, 1, 4, 6, requires_grad=True),
        "state": torch.randn(batch_size, frames - 1, 5, requires_grad=True),
        "embodiment_id": torch.tensor([17, 17]),
        "clean_x": torch.randn(batch_size, 2, frames, 4, 6, requires_grad=True),
        "past_clean_x": torch.randn(batch_size, 2, 2, 4, 6, requires_grad=True),
    }

    model_inputs, negative_inputs = WNM2D.prepare_model_inputs(
        module=module,
        model_config=_model_config("."),
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_mask=negative_prompt_mask,
        micro_batch=conditions,
        step=0,
    )

    assert negative_inputs is not None
    assert model_inputs["seq_len"] == frames * 2 * 3
    assert model_inputs["timestep"].shape == (batch_size, frames)
    assert model_inputs["timestep"].dtype == torch.float32
    torch.testing.assert_close(
        model_inputs["timestep"],
        timesteps[:, :1].expand(batch_size, frames),
    )
    assert model_inputs["x"].dtype == torch.bfloat16
    assert model_inputs["x"].device == next(module.parameters()).device
    assert not model_inputs["x"].requires_grad
    assert all(
        not model_inputs[key].requires_grad
        for key in ("context", "clip_feature", "y", "state", "clean_x", "past_clean_x")
    )
    assert torch.count_nonzero(model_inputs["context"][0, 2:]) == 0
    assert torch.count_nonzero(negative_inputs["context"][0, 1:]) == 0
    assert all(
        model_inputs[key].dtype == torch.bfloat16
        for key in ("context", "clip_feature", "y", "state", "clean_x", "past_clean_x")
    )
    assert torch.equal(model_inputs["embodiment_id"], torch.zeros(batch_size, dtype=torch.long))

    cached_inputs, _ = WNM2D.prepare_model_inputs(
        module=module,
        model_config=_model_config("."),
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_mask=negative_prompt_mask,
        micro_batch=conditions,
        step=3,
    )
    torch.testing.assert_close(cached_inputs["x"], latents[:, 2].detach().to(torch.bfloat16))
    torch.testing.assert_close(
        cached_inputs["timestep"],
        timesteps[:, 2:3].expand(batch_size, frames),
    )


def test_action_cache_replay_uses_same_source_step_as_video():
    visual_inputs = {"x": torch.zeros(1, 2, 3, 4, 6)}
    action_trajectory = torch.arange(17 * 2 * 3, dtype=torch.float32).reshape(1, 17, 2, 3)
    action_timesteps = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    injected = WNM2D.inject_action_inputs(
        visual_inputs,
        {
            "all_action_latents": action_trajectory,
            "all_action_timesteps": action_timesteps,
        },
        step=5,
    )
    torch.testing.assert_close(injected["action"], action_trajectory[:, 2])
    torch.testing.assert_close(
        injected["timestep_action"],
        action_timesteps[:, 2:3].expand(1, 2),
    )


def test_compacted_v7_replay_uses_local_source_state_and_exact_timestep():
    module = _TinyCausalWanModel().to(dtype=torch.bfloat16)
    batch_size = 2
    frames = 3
    selected_transitions = torch.tensor([6, 15])
    latents = torch.randn(batch_size, 2, 2, frames, 4, 6)
    timesteps = torch.tensor([[812.5], [125.0]], dtype=torch.float32)
    action_trajectory = torch.randn(batch_size, 2, 2, 3)
    conditions = {
        "all_action_latents": action_trajectory,
        "all_action_timesteps": timesteps.clone(),
        "layer_credit_replay_transition": selected_transitions,
        "clip_feature": torch.randn(batch_size, 3, 4),
        "y": torch.randn(batch_size, 2, 1, 4, 6),
        "state": torch.randn(batch_size, frames - 1, 5),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "clean_x": torch.randn(batch_size, 2, frames, 4, 6),
        "past_clean_x": torch.randn(batch_size, 2, 2, 4, 6),
    }

    model_inputs, negative_inputs = WNM2D.prepare_model_inputs(
        module=module,
        model_config=_model_config("."),
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=torch.randn(batch_size, 4, 6),
        prompt_embeds_mask=torch.ones(batch_size, 4),
        negative_prompt_embeds=torch.randn(batch_size, 4, 6),
        negative_prompt_embeds_mask=torch.ones(batch_size, 4),
        micro_batch=conditions,
        step=0,
    )

    assert negative_inputs is not None
    torch.testing.assert_close(model_inputs["x"], latents[:, 0].to(torch.bfloat16))
    torch.testing.assert_close(model_inputs["timestep"], timesteps.expand(batch_size, frames))
    injected = WNM2D.inject_action_inputs(model_inputs, conditions, step=0)
    torch.testing.assert_close(injected["action"], action_trajectory[:, 0].to(torch.bfloat16))
    torch.testing.assert_close(
        injected["timestep_action"],
        timesteps.expand(batch_size, action_trajectory.shape[2]),
    )

    missing_metadata = dict(conditions)
    del missing_metadata["layer_credit_replay_transition"]
    with pytest.raises(ValueError, match="layer_credit_replay_transition"):
        WNM2D.prepare_model_inputs(
            module=module,
            model_config=_model_config("."),
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=torch.randn(batch_size, 4, 6),
            prompt_embeds_mask=torch.ones(batch_size, 4),
            negative_prompt_embeds=torch.randn(batch_size, 4, 6),
            negative_prompt_embeds_mask=torch.ones(batch_size, 4),
            micro_batch=missing_metadata,
            step=0,
        )

    invalid_metadata = dict(conditions)
    invalid_metadata["layer_credit_replay_transition"] = torch.tensor([3, 15])
    with pytest.raises(ValueError, match="only deployed DiT source transitions"):
        WNM2D.prepare_model_inputs(
            module=module,
            model_config=_model_config("."),
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=torch.randn(batch_size, 4, 6),
            prompt_embeds_mask=torch.ones(batch_size, 4),
            negative_prompt_embeds=torch.randn(batch_size, 4, 6),
            negative_prompt_embeds_mask=torch.ones(batch_size, 4),
            micro_batch=invalid_metadata,
            step=0,
        )


def test_registry_build_replay_computes_both_log_probs_in_one_joint_forward(
    tmp_path,
    fake_gammanav,
    monkeypatch,
):
    _write_config(tmp_path)
    _write_single_checkpoint(tmp_path, _reference_state())
    model_config = _model_config(tmp_path)
    monkeypatch.setattr(wnm_adapter, "get_device_name", lambda: "cpu")

    adapter = DiffusionModelBase.get_class(model_config)
    module = adapter.build_module(model_config, torch.float32)
    adapter.configure_trainable_params(module, model_config)
    scheduler = adapter.build_scheduler(model_config)

    batch_size = 2
    policy_steps = 16
    frames = 3
    visual_trajectory = torch.randn(batch_size, policy_steps + 1, 2, frames, 4, 6)
    action_trajectory = torch.randn(batch_size, policy_steps + 1, 2, 3)
    scheduler_timesteps = scheduler.timesteps.unsqueeze(0).expand(batch_size, -1).clone()
    micro_batch = {
        "all_action_latents": action_trajectory,
        "all_action_timesteps": scheduler_timesteps,
        "clip_feature": torch.randn(batch_size, 3, 4),
        "y": torch.randn(batch_size, 2, 1, 4, 6),
        "state": torch.randn(batch_size, frames - 1, 5),
        "embodiment_id": torch.full((batch_size,), 17),
        "clean_x": torch.randn(batch_size, 2, frames, 4, 6),
        "past_clean_x": torch.randn(batch_size, 2, 2, 4, 6),
    }
    model_inputs, negative_inputs = adapter.prepare_model_inputs(
        module=module,
        model_config=model_config,
        latents=visual_trajectory,
        timesteps=scheduler_timesteps,
        prompt_embeds=torch.randn(batch_size, 4, 6),
        prompt_embeds_mask=torch.ones(batch_size, 4),
        negative_prompt_embeds=torch.randn(batch_size, 4, 6),
        negative_prompt_embeds_mask=torch.ones(batch_size, 4),
        micro_batch=micro_batch,
        step=0,
    )
    model_inputs = adapter.inject_action_inputs(model_inputs, micro_batch, step=0)
    scheduler_inputs = {
        "all_latents": visual_trajectory,
        "all_timesteps": scheduler_timesteps,
        "all_action_latents": action_trajectory,
        "all_action_timesteps": scheduler_timesteps,
    }

    output = adapter.forward_and_sample_previous_step(
        module=module,
        scheduler=scheduler,
        model_config=model_config,
        model_inputs=model_inputs,
        negative_model_inputs=negative_inputs,
        scheduler_inputs=scheduler_inputs,
        step=0,
    )

    assert module.forward_calls == 2
    assert output["log_probs"].requires_grad
    assert output["action_log_probs"].requires_grad
    assert torch.isfinite(output["log_probs"]).all()
    assert torch.isfinite(output["action_log_probs"]).all()
    joint_loss = -(output["log_probs"] + output["action_log_probs"]).mean()
    joint_loss.backward()
    trainable_grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert trainable_grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in trainable_grads)

    # Exercise the exact layer-conditioned actor contract: each row may credit a different
    # real DiT source transition, but replay exposes only its local t/t+1 pair.
    rows = torch.arange(batch_size)
    selected_transitions = torch.tensor([6, 15])
    compact_visual = torch.stack(
        (
            visual_trajectory[rows, selected_transitions],
            visual_trajectory[rows, selected_transitions + 1],
        ),
        dim=1,
    )
    compact_action = torch.stack(
        (
            action_trajectory[rows, selected_transitions],
            action_trajectory[rows, selected_transitions + 1],
        ),
        dim=1,
    )
    compact_timesteps = scheduler_timesteps[rows, selected_transitions].unsqueeze(1)
    compact_micro_batch = dict(micro_batch)
    compact_micro_batch.update(
        {
            "all_action_latents": compact_action,
            "all_action_timesteps": compact_timesteps,
            "layer_credit_replay_transition": selected_transitions,
        }
    )
    compact_inputs, compact_negative_inputs = adapter.prepare_model_inputs(
        module=module,
        model_config=model_config,
        latents=compact_visual,
        timesteps=compact_timesteps,
        prompt_embeds=torch.randn(batch_size, 4, 6),
        prompt_embeds_mask=torch.ones(batch_size, 4),
        negative_prompt_embeds=torch.randn(batch_size, 4, 6),
        negative_prompt_embeds_mask=torch.ones(batch_size, 4),
        micro_batch=compact_micro_batch,
        step=0,
    )
    compact_inputs = adapter.inject_action_inputs(compact_inputs, compact_micro_batch, step=0)
    compact_output = adapter.forward_and_sample_previous_step(
        module=module,
        scheduler=scheduler,
        model_config=model_config,
        model_inputs=compact_inputs,
        negative_model_inputs=compact_negative_inputs,
        scheduler_inputs={
            "all_latents": compact_visual,
            "all_timesteps": compact_timesteps,
            "all_action_latents": compact_action,
            "all_action_timesteps": compact_timesteps,
        },
        step=0,
    )
    assert compact_output["log_probs"].shape == (batch_size,)
    assert compact_output["action_log_probs"].shape == (batch_size,)
    assert torch.isfinite(compact_output["log_probs"]).all()
    assert torch.isfinite(compact_output["action_log_probs"]).all()


def test_prepare_model_inputs_rejects_missing_condition_and_cfg():
    module = _TinyCausalWanModel()
    latents = torch.randn(1, 17, 2, 3, 4, 6)
    timesteps = torch.ones(1, 16)
    conditions = {
        "clip_feature": torch.randn(1, 3, 4),
        "y": torch.randn(1, 2, 1, 4, 6),
        "state": torch.randn(1, 2, 5),
        "embodiment_id": torch.tensor([17]),
        "clean_x": torch.randn(1, 2, 3, 4, 6),
        # past_clean_x intentionally absent
    }
    kwargs = dict(
        module=module,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=torch.randn(1, 4, 6),
        prompt_embeds_mask=torch.ones(1, 4),
        negative_prompt_embeds=torch.randn(1, 4, 6),
        negative_prompt_embeds_mask=torch.ones(1, 4),
        micro_batch=conditions,
        step=0,
    )

    with pytest.raises(KeyError, match="past_clean_x"):
        WNM2D.prepare_model_inputs(model_config=_model_config("."), **kwargs)

    with pytest.raises(ValueError, match="requires true_cfg_scale=5"):
        WNM2D.prepare_model_inputs(model_config=_model_config(".", true_cfg_scale=1.0), **kwargs)


def test_prepare_model_inputs_rejects_wrong_action_horizon():
    module = _TinyCausalWanModel()
    latents = torch.randn(1, 17, 2, 3, 4, 6)
    timesteps = torch.ones(1, 16)
    conditions = {
        "all_action_latents": torch.randn(1, 17, 3, 3),
        "all_action_timesteps": timesteps,
        "clip_feature": torch.randn(1, 3, 4),
        "y": torch.randn(1, 2, 1, 4, 6),
        "state": torch.randn(1, 2, 5),
        "embodiment_id": torch.tensor([0]),
        "clean_x": torch.randn(1, 2, 3, 4, 6),
        "past_clean_x": torch.randn(1, 2, 2, 4, 6),
    }

    with pytest.raises(ValueError, match="action trajectory must have shape"):
        WNM2D.prepare_model_inputs(
            module=module,
            model_config=_model_config("."),
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=torch.randn(1, 4, 6),
            prompt_embeds_mask=torch.ones(1, 4),
            negative_prompt_embeds=torch.randn(1, 4, 6),
            negative_prompt_embeds_mask=torch.ones(1, 4),
            micro_batch=conditions,
            step=0,
        )
