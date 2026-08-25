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

"""Dependency-free contract tests for the WNM vLLM-Omni adapter.

The real rollout stack is intentionally not imported here: these tests verify
the custom-pipeline boundary with small CPU stubs so they remain runnable on a
development machine without vLLM-Omni or the WNM runtime installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

_DUMMY_REQUEST_ID = "__wnm_dummy_request__"
_MISSING_MODULE = object()


class _DiffusionOutput:
    def __init__(self, *, output, custom_output, to_cpu=False, **kwargs):
        self.output = output
        self.custom_output = custom_output
        self.to_cpu = to_cpu
        for name, value in kwargs.items():
            setattr(self, name, value)


class _DiffusionRequestBatch:
    def __init__(self, requests):
        self.requests = requests


class _Registry:
    entries: dict[tuple[str, str], type] = {}

    @classmethod
    def register(cls, architecture: str, algorithm: str):
        def decorator(pipeline_cls):
            cls.entries[(architecture, algorithm)] = pipeline_cls
            return pipeline_cls

        return decorator


class _FakeFlowMatchSDEDiscreteScheduler:
    @staticmethod
    def _reduce_log_prob(log_prob, log_prob_mask=None, *, chunk_size=None):
        del log_prob_mask
        if chunk_size is None:
            return log_prob.flatten(1).mean(dim=1)
        batch, horizon = log_prob.shape[:2]
        return log_prob.reshape(batch, horizon // chunk_size, -1).mean(dim=-1)


class _Batch:
    def __init__(self, *, obs):
        self.obs = obs


class _FakeJointDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        self.register_buffer("scale", torch.ones(1))


class _FakeFrozenModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The test deliberately differs from the fp32 rollout latent dtype.
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)
        self.last_decode_dtype = None

    def decode(self, hidden_states, *, tiled, tile_size, tile_stride):
        del tiled, tile_size, tile_stride
        self.last_decode_dtype = hidden_states.dtype
        if hidden_states.dtype != self.weight.dtype:
            raise TypeError("VAE input dtype was not aligned with its live parameters")
        batch, _, frames, height, width = hidden_states.shape
        values = torch.linspace(
            -1.0,
            1.0,
            batch * 3 * frames * height * width,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return values.reshape(batch, 3, frames, height, width)


class _FakeActionHead:
    def __init__(self):
        self.model = _FakeJointDiT()
        self.text_encoder = _FakeFrozenModule()
        self.image_encoder = _FakeFrozenModule()
        self.vae = _FakeVAE()
        self.num_frames = 3
        self.num_inference_steps = 16
        self.wam_noise_level = 0.7
        self.wam_action_noise_level = 0.7
        self.tiled = True
        self.tile_size_height = 4
        self.tile_size_width = 5
        self.tile_stride_height = 2
        self.tile_stride_width = 3

    def _strict_wam_dance_rollout(self, *, rollout_seed, **kwargs):
        del rollout_seed, kwargs
        raise AssertionError("the fake policy does not invoke the real action head")


def resolve_rollout_seed(request, default_seed, *, require_explicit_rollout_seed=False):
    del require_explicit_rollout_seed
    return int(request.get("rollout_seed", default_seed))


class _FakeTrainedModel:
    def __init__(self):
        self.action_head = _FakeActionHead()
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class _FakeWNM3DInferencePolicy:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.trained_model = _FakeTrainedModel()
        self.modality_configs = SimpleNamespace(
            video=SimpleNamespace(modality_keys=["video.observation"]),
            state=SimpleNamespace(modality_keys=["state.observation"]),
            language=SimpleNamespace(modality_keys=["language.instruction"]),
        )
        self.observations = []
        self.malformed = None
        type(self).instances.append(self)

    def lazy_joint_forward_causal(self, batch, *, return_model_pred):
        if not return_model_pred:
            raise AssertionError("rollout adapter must request replay fields")
        observation = batch.obs
        self.observations.append(observation)
        seed_values = np.asarray(observation["rollout_seed"]).reshape(-1).tolist()
        generators = [torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seed_values]
        batch_size = len(generators)
        steps = self.trained_model.action_head.num_inference_steps
        draw = lambda shape: torch.cat(
            [torch.randn((1, *shape), generator=generator) for generator in generators], dim=0
        )
        visual_trajectory = draw((steps + 1, 2, 2, 2, 2))
        action_trajectory = draw((steps + 1, 6, 3))
        timesteps = torch.arange(steps, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).clone()
        model_pred = {
            "actions": action_trajectory[:, -1].clone(),
            "all_latents": visual_trajectory,
            "all_timesteps": timesteps,
            "all_log_probs": draw((steps,)),
            "all_action_latents": action_trajectory,
            "all_action_timesteps": timesteps.clone(),
            "dit_prediction_source_steps": torch.tensor(
                [[0, 1, 2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 13, 14, 15]] * batch_size,
                dtype=torch.long,
            ),
            "num_dit_prediction_steps": torch.full((batch_size,), 8, dtype=torch.long),
            "num_dit_forwards": torch.full((batch_size,), 16, dtype=torch.long),
            "num_inference_steps": torch.full((batch_size,), 16, dtype=torch.long),
            "true_cfg_scale": torch.full((batch_size,), 5.0),
            "noise_level": torch.full((batch_size,), 0.7),
            "action_noise_level": torch.full((batch_size,), 0.7),
            "action_log_probs": draw((steps,)),
            "prompt_embeds": draw((4, 5)),
            "prompt_embeds_mask": torch.ones(batch_size, 4, dtype=torch.bool),
            "negative_prompt_embeds": draw((4, 5)),
            "negative_prompt_embeds_mask": torch.ones(batch_size, 4, dtype=torch.bool),
            "clip_feature": draw((2, 4)),
            "y": draw((2, 2, 2, 2)),
            "state": draw((1, 5)),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "clean_x": draw((2, 2, 2, 2)),
            "past_clean_x": draw((2, 1, 2, 2)),
            # InteriorGS exposes three deployed action coordinates in this
            # compact fake model, so every coordinate is policy-active.
            "action_policy_mask": torch.ones(batch_size, 6, 3, dtype=torch.bool),
        }
        if self.malformed == "short_action_trajectory":
            model_pred["all_action_latents"] = action_trajectory[:, :-1]
        elif self.malformed == "missing_condition":
            model_pred.pop("clean_x")
        return SimpleNamespace(), visual_trajectory[:, -1], model_pred


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _stub_modules() -> dict[str, types.ModuleType]:
    modules = {
        "verl_omni": _package("verl_omni"),
        "verl_omni.pipelines": _package("verl_omni.pipelines"),
        "verl_omni.pipelines.wnm_shared": _package("verl_omni.pipelines.wnm_shared"),
        "verl_omni.pipelines.wnm_shared.batch1_equivalent": types.ModuleType(
            "verl_omni.pipelines.wnm_shared.batch1_equivalent"
        ),
        "verl_omni.pipelines.schedulers": _package("verl_omni.pipelines.schedulers"),
        "verl_omni.pipelines.schedulers.flow_match_sde": types.ModuleType(
            "verl_omni.pipelines.schedulers.flow_match_sde"
        ),
        "verl_omni.utils": _package("verl_omni.utils"),
        "verl_omni.utils.action_chunk_credit": types.ModuleType("verl_omni.utils.action_chunk_credit"),
        "vllm_omni": _package("vllm_omni"),
        "vllm_omni.diffusion": _package("vllm_omni.diffusion"),
        "vllm_omni.diffusion.distributed": _package("vllm_omni.diffusion.distributed"),
        "vllm_omni.diffusion.data": types.ModuleType("vllm_omni.diffusion.data"),
        "vllm_omni.diffusion.distributed.utils": types.ModuleType("vllm_omni.diffusion.distributed.utils"),
        "vllm_omni.diffusion.request": types.ModuleType("vllm_omni.diffusion.request"),
        "vllm_omni.diffusion.worker": _package("vllm_omni.diffusion.worker"),
        "vllm_omni.diffusion.worker.request_batch": types.ModuleType("vllm_omni.diffusion.worker.request_batch"),
        "verl_omni.pipelines.model_base": types.ModuleType("verl_omni.pipelines.model_base"),
        "gammanav": _package("gammanav"),
        "gammanav.vln": _package("gammanav.vln"),
        "gammanav.vln.data": _package("gammanav.vln.data"),
        "gammanav.vln.model": _package("gammanav.vln.model"),
        "gammanav.vln.model.wnm_3d": _package("gammanav.vln.model.wnm_3d"),
        "gammanav.vln.data.schema": types.ModuleType("gammanav.vln.data.schema"),
        "gammanav.vln.model.wnm_3d.inference_policy": types.ModuleType("gammanav.vln.model.wnm_3d.inference_policy"),
        "tianshou": _package("tianshou"),
        "tianshou.data": types.ModuleType("tianshou.data"),
    }
    modules["vllm_omni.diffusion.data"].DiffusionOutput = _DiffusionOutput
    modules["vllm_omni.diffusion.data"].OmniDiffusionConfig = SimpleNamespace
    modules["vllm_omni.diffusion.distributed.utils"].get_local_device = lambda: torch.device("cpu")
    modules["vllm_omni.diffusion.request"].DUMMY_DIFFUSION_REQUEST_ID = _DUMMY_REQUEST_ID
    modules["vllm_omni.diffusion.request"].OmniDiffusionRequest = SimpleNamespace
    modules["vllm_omni.diffusion.worker.request_batch"].DiffusionRequestBatch = _DiffusionRequestBatch
    modules["verl_omni.pipelines.model_base"].VllmOmniPipelineBase = _Registry
    modules["verl_omni.pipelines.wnm_shared.batch1_equivalent"].install_batch1_equivalent_action_encoder = (
        lambda *args, **kwargs: None
    )
    modules[
        "verl_omni.pipelines.schedulers.flow_match_sde"
    ].FlowMatchSDEDiscreteScheduler = _FakeFlowMatchSDEDiscreteScheduler
    modules["verl_omni.utils.action_chunk_credit"].action_chunk_credit_enabled = lambda: False
    modules["verl_omni.utils.action_chunk_credit"].action_chunk_size = lambda: 8
    modules["gammanav.vln.data.schema"].EmbodimentTag = SimpleNamespace(INTERIORGS="interiorgs")
    modules["gammanav.vln.model.wnm_3d.inference_policy"].WNM3DInferencePolicy = _FakeWNM3DInferencePolicy
    modules["tianshou.data"].Batch = _Batch
    return modules


_ROLLOUT_HELPER_MODULES = (
    "rollout_common",
    "rollout_acceleration",
    "rollout_batching",
    "wam_dance_sde",
    "rollout_rng",
)


def _load_adapter_module():
    pipelines_root = Path(__file__).resolve().parents[2] / "verl_omni" / "pipelines"
    helper_root = pipelines_root / "wnm_shared"
    for helper in _ROLLOUT_HELPER_MODULES:
        qualified_name = f"verl_omni.pipelines.wnm_shared.{helper}"
        helper_spec = importlib.util.spec_from_file_location(qualified_name, helper_root / f"{helper}.py")
        if helper_spec is None or helper_spec.loader is None:
            raise RuntimeError(f"Unable to load rollout helper {helper}")
        helper_module = importlib.util.module_from_spec(helper_spec)
        sys.modules[qualified_name] = helper_module
        helper_spec.loader.exec_module(helper_module)

    module_path = pipelines_root / "wnm_2d" / "vllm_omni_rollout_adapter.py"
    module_name = "_wnm_2d_vllm_adapter_cpu_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class WNM2DVllmAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _Registry.entries.clear()
        _FakeWNM3DInferencePolicy.instances.clear()
        cls.stub_modules = _stub_modules()
        cls.previous_stub_modules = {name: sys.modules.get(name, _MISSING_MODULE) for name in cls.stub_modules}
        sys.modules.update(cls.stub_modules)
        cls.adapter = _load_adapter_module()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("_wnm_2d_vllm_adapter_cpu_test", None)
        for helper in _ROLLOUT_HELPER_MODULES:
            sys.modules.pop(f"verl_omni.pipelines.wnm_shared.{helper}", None)
        for name, previous in cls.previous_stub_modules.items():
            if previous is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        super().tearDownClass()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        config = SimpleNamespace(model=self.temp_dir.name)
        self.configure_sdpa_patcher = patch.object(self.adapter, "_configure_explicit_sdpa_runtime")
        self.validate_sdpa_patcher = patch.object(self.adapter, "_validate_explicit_sdpa")
        self.configure_sdpa_patcher.start()
        self.validate_sdpa_patcher.start()
        self.pipeline = self.adapter.WNM2DPipelineWithLogProb(od_config=config)

    def tearDown(self):
        self.validate_sdpa_patcher.stop()
        self.configure_sdpa_patcher.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _request(
        *,
        seed=11,
        logprobs=True,
        steps=16,
        request_id="real-request",
        layer_credit=None,
    ):
        first_clip = torch.zeros(3, 3, 2, 2, dtype=torch.float32)
        second_clip = torch.ones(3, 3, 2, 2, dtype=torch.float32)
        extra_args = {
            "rollout_extra_args": {
                "instruction": "move to the doorway",
                "state": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                "target_prefix_frames": 1,
            },
            "noise_level": 0.8,
            "action_noise_level": 0.6,
            "sde_type": "dance_sde",
            "action_sde_type": "dance_sde",
            "logprobs": logprobs,
        }
        if layer_credit is not None:
            stratum, branch, transition = layer_credit
            extra_args.update(
                {
                    "layer_credit_stratum": stratum,
                    "layer_credit_branch": branch,
                    "layer_credit_transition": transition,
                }
            )
        sampling = SimpleNamespace(
            extra_args=extra_args,
            num_inference_steps=steps,
            seed=seed,
        )
        prompt = {"multi_modal_data": {"video": [first_clip, second_clip]}}
        return SimpleNamespace(request_id=request_id, prompt=prompt, sampling_params=sampling)

    def test_registers_custom_pipeline_and_keeps_local_checkpoint_only(self):
        self.assertIs(
            _Registry.entries[("WNM2D", "dance_grpo")],
            self.adapter.WNM2DPipelineWithLogProb,
        )
        self.assertTrue(self.pipeline.supports_request_batch)
        self.assertEqual(os.environ["WNM_ROLLOUT"], "true")
        self.assertTrue(self.pipeline._policy.trained_model.eval_called)
        self.assertEqual(
            set(self.pipeline._policy.init_kwargs),
            {"embodiment_tag", "model_path", "device", "tokenizer_path_override"},
        )

        with self.assertRaises(FileNotFoundError):
            self.adapter.WNM2DPipelineWithLogProb(
                od_config=SimpleNamespace(model=str(Path(self.temp_dir.name) / "not-local"))
            )

    def test_deployed_action_policy_mask_excludes_padding_dimensions(self):
        action_state = torch.zeros(2, 32, 32)
        mask = self.adapter._deployed_action_policy_mask(action_state)

        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(tuple(mask.shape), tuple(action_state.shape))
        self.assertTrue(mask[..., :3].all())
        self.assertFalse(mask[..., 3:].any())

    def test_dummy_requests_skip_model_and_input_validation(self):
        dummy = SimpleNamespace(request_id=_DUMMY_REQUEST_ID, prompt=None, sampling_params=None)
        output = self.pipeline(dummy)
        self.assertIsNone(output.output)
        self.assertEqual(output.custom_output, {})
        self.assertEqual(self.pipeline._policy.observations, [])

        prompt_dummy = SimpleNamespace(
            request_id="warm-up",
            prompt={"prompt": "dummy run"},
            sampling_params=None,
        )
        self.assertIsNone(self.pipeline(prompt_dummy).output)
        self.assertEqual(self.pipeline._policy.observations, [])

    def test_forward_decodes_video_and_returns_complete_joint_replay_contract(self):
        output = self.pipeline(self._request(seed=17))

        self.assertEqual(tuple(output.output.shape), (1, 2, 3, 2, 2))
        self.assertEqual(output.output.dtype, torch.float32)
        self.assertGreaterEqual(float(output.output.min()), 0.0)
        self.assertLessEqual(float(output.output.max()), 1.0)
        self.assertTrue(output.to_cpu)
        self.assertEqual(self.pipeline.vae.last_decode_dtype, torch.float16)

        expected = {
            "actions",
            "all_latents",
            "all_timesteps",
            "all_log_probs",
            "all_action_latents",
            "all_action_timesteps",
            "action_log_probs",
            "dit_prediction_source_steps",
            "num_dit_prediction_steps",
            "num_dit_forwards",
            "num_inference_steps",
            "true_cfg_scale",
            "noise_level",
            "action_noise_level",
            "action_policy_mask",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "clip_feature",
            "y",
            "state",
            "embodiment_id",
            "clean_x",
            "past_clean_x",
        }
        self.assertEqual(set(output.custom_output), expected)
        self.assertTrue(output.custom_output["action_policy_mask"].all())
        torch.testing.assert_close(
            output.custom_output["actions"],
            output.custom_output["all_action_latents"][:, -1],
        )
        self.assertFalse(any(value.requires_grad for value in output.custom_output.values()))

        observation = self.pipeline._policy.observations[-1]
        video = observation["video.observation"]
        self.assertEqual(video.shape, (6, 2, 2, 3))
        self.assertEqual(video.dtype, np.uint8)
        self.assertTrue(np.all(video[:3] == 0))
        self.assertTrue(np.all(video[3:] == 255))
        self.assertEqual(observation["rollout_seed"], 17)
        self.assertEqual(observation["language.instruction"], "move to the doorway")
        self.assertEqual(self.pipeline._action_head.wam_noise_level, 0.8)
        self.assertEqual(self.pipeline._action_head.wam_action_noise_level, 0.6)

    def test_repeated_prompt_requests_keep_independent_rollout_seeds(self):
        first = self.pipeline(self._request(seed=101))
        second = self.pipeline(self._request(seed=202))

        self.assertEqual([obs["rollout_seed"] for obs in self.pipeline._policy.observations], [101, 202])
        self.assertFalse(torch.equal(first.custom_output["all_latents"], second.custom_output["all_latents"]))
        self.assertFalse(
            torch.equal(first.custom_output["all_action_latents"], second.custom_output["all_action_latents"])
        )

    def test_request_batch_keeps_per_request_rng_and_splits_outputs(self):
        first_request = self._request(seed=101, request_id="first")
        second_request = self._request(seed=202, request_id="second")
        singleton_first = self.pipeline(first_request)
        singleton_second = self.pipeline(second_request)

        outputs = self.pipeline(_DiffusionRequestBatch([first_request, second_request]))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(self.pipeline._policy.observations[-1]["rollout_seed"]), (101, 202))
        torch.testing.assert_close(
            outputs[0].custom_output["all_latents"],
            singleton_first.custom_output["all_latents"],
        )
        torch.testing.assert_close(
            outputs[1].custom_output["all_latents"],
            singleton_second.custom_output["all_latents"],
        )

    def test_request_batch_preserves_v7_layer_credit_controls(self):
        first_request = self._request(seed=101, request_id="first", layer_credit=(0, 0, 1))
        second_request = self._request(seed=202, request_id="second", layer_credit=(0, 1, 1))

        with patch.dict(os.environ, {"WAM_ROLLOUT_DEDUP_TRANSFORM": "true"}):
            self.pipeline(_DiffusionRequestBatch([first_request, second_request]))

        observation = self.pipeline._policy.observations[-1]
        self.assertEqual(tuple(observation["layer_credit_stratum"]), (0, 0))
        self.assertEqual(tuple(observation["layer_credit_branch"]), (0, 1))
        self.assertEqual(tuple(observation["layer_credit_transition"]), (1, 1))

    def test_per_request_randn_matches_singleton_streams(self):
        seeds = (123, 456)
        batched = self.adapter._randn_per_request(
            (2, 3, 4),
            generators=tuple(torch.Generator().manual_seed(seed) for seed in seeds),
            device=torch.device("cpu"),
        )
        expected = torch.cat(
            [torch.randn((1, 3, 4), generator=torch.Generator().manual_seed(seed)) for seed in seeds],
            dim=0,
        )
        torch.testing.assert_close(batched, expected, rtol=0, atol=0)

    def test_repeated_conditioning_transform_runs_once_and_restores_seed_vector(self):
        class FakeTransformPolicy:
            def __init__(self):
                self.batch_sizes = []
                self.trained_model = SimpleNamespace(action_head=SimpleNamespace())

            def apply(self, batch):
                observation = batch.obs
                batch_size = int(observation["video"].shape[0])
                self.batch_sizes.append(batch_size)
                batch.normalized_obs = {
                    "images": torch.from_numpy(observation["video"].copy()),
                    "state": torch.from_numpy(observation["state"].copy()),
                    "text": torch.from_numpy(observation["text"].copy()),
                    "rollout_seed": torch.from_numpy(observation["rollout_seed"].copy()),
                }
                return batch

        policy = FakeTransformPolicy()
        environment = {
            "WAM_ROLLOUT_DEDUP_TRANSFORM": "true",
            "WAM_ROLLOUT_DEDUP_TRANSFORM_VERIFY": "true",
        }
        with patch.dict(os.environ, environment):
            self.adapter._install_rollout_transform_dedup(policy)

        observation = {
            "video": np.repeat(np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3), 3, axis=0),
            "state": np.repeat(np.asarray([[1.0, 2.0]], dtype=np.float32), 3, axis=0),
            "text": np.repeat(np.asarray([[7, 8]], dtype=np.int64), 3, axis=0),
            "rollout_seed": np.asarray([101, 202, 303], dtype=np.int64),
            # Checkpoint transforms do not know these RL-only fields and the
            # fake policy intentionally drops them from normalized_obs.
            "layer_credit_stratum": np.asarray([0, 0, 0], dtype=np.int64),
            "layer_credit_branch": np.asarray([0, 1, 2], dtype=np.int64),
            "layer_credit_transition": np.asarray([1, 1, 1], dtype=np.int64),
        }
        result = policy.apply(_Batch(obs=observation))

        # The first optimized call performs singleton + original-batch parity.
        self.assertEqual(policy.batch_sizes, [1, 3])
        self.assertEqual(tuple(result.normalized_obs["images"].shape), (3, 2, 2, 3))
        torch.testing.assert_close(
            result.normalized_obs["rollout_seed"],
            torch.tensor([101, 202, 303]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.normalized_obs["layer_credit_stratum"],
            torch.tensor([0, 0, 0]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.normalized_obs["layer_credit_branch"],
            torch.tensor([0, 1, 2]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.normalized_obs["layer_credit_transition"],
            torch.tensor([1, 1, 1]),
            rtol=0,
            atol=0,
        )

        policy.apply(_Batch(obs=observation))
        self.assertEqual(policy.batch_sizes, [1, 3, 1])

        different = {key: value.copy() for key, value in observation.items()}
        different["state"][1, 0] = 99.0
        policy.apply(_Batch(obs=different))
        self.assertEqual(policy.batch_sizes, [1, 3, 1, 3])

    def test_two_prompt_groups_transform_once_per_unique_condition(self):
        class FakeTransformPolicy:
            def __init__(self):
                self.batch_sizes = []
                self.trained_model = SimpleNamespace(action_head=SimpleNamespace())

            def apply(self, batch):
                observation = batch.obs
                self.batch_sizes.append(int(observation["video"].shape[0]))
                batch.normalized_obs = {
                    "images": torch.from_numpy(observation["video"].copy()),
                    "state": torch.from_numpy(observation["state"].copy()),
                    "rollout_seed": torch.from_numpy(observation["rollout_seed"].copy()),
                }
                return batch

        policy = FakeTransformPolicy()
        environment = {
            "WAM_ROLLOUT_DEDUP_TRANSFORM": "true",
            "WAM_ROLLOUT_DEDUP_TRANSFORM_VERIFY": "true",
        }
        with patch.dict(os.environ, environment):
            self.adapter._install_rollout_transform_dedup(policy)

        policy.trained_model.action_head._verl_requested_conditioning_group_plan = {
            "batch_size": 4,
            "representative_indices": (0, 2),
            "group_index": (0, 0, 1, 1),
        }
        observation = {
            "video": np.asarray([[[1]], [[1]], [[9]], [[9]]], dtype=np.uint8),
            "state": np.asarray([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32),
            "rollout_seed": np.asarray([11, 12, 21, 22], dtype=np.int64),
        }
        result = policy.apply(_Batch(obs=observation))

        self.assertEqual(policy.batch_sizes, [2, 4])
        self.assertEqual(tuple(result.normalized_obs["images"].reshape(-1)), (1, 1, 9, 9))
        torch.testing.assert_close(
            result.normalized_obs["rollout_seed"],
            torch.tensor([11, 12, 21, 22]),
            rtol=0,
            atol=0,
        )

        policy.apply(_Batch(obs=observation))
        self.assertEqual(policy.batch_sizes, [2, 4, 2])

    def test_repeated_conditioning_encoders_run_singleton_and_expand(self):
        class FakeActionHead:
            def __init__(self):
                self.calls = []
                self.vae = SimpleNamespace(encode=self.vae_encode)

            def vae_encode(self, video, tiled=True):
                del tiled
                self.calls.append(video.shape[0])
                return video.float() * 2

        action_head = FakeActionHead()
        environment = {
            "WAM_ROLLOUT_DEDUP_TRANSFORM": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS_VERIFY": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS_ATOL": "0",
            "WAM_ROLLOUT_DEDUP_ENCODERS_RTOL": "0",
        }
        with patch.dict(os.environ, environment):
            self.adapter._install_repeated_conditioning_encoder_dedup(action_head)

        action_head._verl_repeated_conditioning_batch_size = 3
        video = torch.ones(3, 2, 2, 2)
        encoded_video = action_head.vae.encode(video, tiled=True)
        self.assertEqual(action_head.calls, [1, 1])
        self.assertEqual(tuple(encoded_video.shape), tuple(video.shape))

        encoded_video = action_head.vae.encode(video, tiled=True)
        self.assertEqual(action_head.calls, [1, 1, 1])
        torch.testing.assert_close(encoded_video[0], encoded_video[2], rtol=0, atol=0)

        action_head._verl_repeated_conditioning_batch_size = None
        action_head.vae.encode(video, tiled=True)
        self.assertEqual(action_head.calls, [1, 1, 1, 3])

    def test_two_prompt_groups_conditioning_encoders_run_two_rows(self):
        class FakeActionHead:
            def __init__(self):
                self.calls = []
                self.vae = SimpleNamespace(encode=self.vae_encode)

            def vae_encode(self, video, tiled=True):
                del tiled
                self.calls.append(video.shape[0])
                # Simulate WNM's shape-specialized compiled encoder:
                # eager math is sample-independent, but the compiled result
                # changes with its call batch.  The adapter must preserve B=1
                # deployment semantics rather than calling this at B=2.
                return video.float() * 2 + video.shape[0]

        action_head = FakeActionHead()
        environment = {
            "WAM_ROLLOUT_DEDUP_TRANSFORM": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS_VERIFY": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS_ATOL": "0",
            "WAM_ROLLOUT_DEDUP_ENCODERS_RTOL": "0",
        }
        with patch.dict(os.environ, environment):
            self.adapter._install_repeated_conditioning_encoder_dedup(action_head)

        action_head._verl_repeated_conditioning_batch_size = 4
        action_head._verl_conditioning_representative_indices = (0, 2)
        action_head._verl_conditioning_group_index = (0, 0, 1, 1)
        video = torch.tensor([1.0, 1.0, 3.0, 3.0]).reshape(4, 1, 1, 1)
        encoded_video = action_head.vae.encode(video, tiled=True)

        self.assertEqual(action_head.calls, [1, 1, 1, 1])
        torch.testing.assert_close(
            encoded_video.reshape(-1),
            torch.tensor([3.0, 3.0, 7.0, 7.0]),
            rtol=0,
            atol=0,
        )

    def test_all_frozen_conditioning_encoders_use_group_representatives(self):
        class FakeActionHead:
            def __init__(self):
                self.calls = {"prompt": [], "image": [], "vae": []}
                self.vae = SimpleNamespace(encode=self.vae_encode)

            def encode_prompt(self, token_ids, attention_mask):
                self.calls["prompt"].append(token_ids.shape[0])
                return token_ids.float().unsqueeze(-1) * attention_mask.unsqueeze(-1)

            def encode_image(self, image, num_frames, height, width):
                del num_frames, height, width
                self.calls["image"].append(image.shape[0])
                return image.float(), image.float() + 1, image.float() + 2

            def vae_encode(self, video, tiled=True):
                del tiled
                self.calls["vae"].append(video.shape[0])
                return video.float() * 2

        action_head = FakeActionHead()
        environment = {
            "WAM_ROLLOUT_DEDUP_TRANSFORM": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS_VERIFY": "true",
            "WAM_ROLLOUT_DEDUP_ENCODERS_ATOL": "0",
            "WAM_ROLLOUT_DEDUP_ENCODERS_RTOL": "0",
        }
        with patch.dict(os.environ, environment):
            self.adapter._install_repeated_conditioning_encoder_dedup(action_head)

        action_head._verl_repeated_conditioning_batch_size = 4
        action_head._verl_conditioning_representative_indices = (0, 2)
        action_head._verl_conditioning_group_index = (0, 0, 1, 1)
        tokens = torch.tensor([[1, 2], [1, 2], [7, 8], [7, 8]])
        mask = torch.ones_like(tokens)
        image = torch.tensor([1.0, 1.0, 3.0, 3.0]).reshape(4, 1, 1, 1)

        prompt = action_head.encode_prompt(tokens, mask)
        image_outputs = action_head.encode_image(image, 3, 2, 2)
        vae = action_head.vae.encode(image)

        self.assertEqual(
            action_head.calls,
            {
                "prompt": [1, 1, 1, 1],
                "image": [1, 1, 1, 1],
                "vae": [1, 1, 1, 1],
            },
        )
        self.assertEqual(tuple(prompt.shape), (4, 2, 1))
        self.assertEqual(tuple(image_outputs[0].shape), tuple(image.shape))
        torch.testing.assert_close(vae.reshape(-1), torch.tensor([2.0, 2.0, 6.0, 6.0]))

    def test_request_conditioning_plan_groups_two_rollout_prompts(self):
        requests = []
        observations = []
        for index, (path, initial_seed) in enumerate(
            (("/tmp/a.mp4", 100), ("/tmp/a.mp4", 100), ("/tmp/b.mp4", 200), ("/tmp/b.mp4", 200))
        ):
            rollout_inputs = {
                "instruction": "move",
                "state": [[float(initial_seed)]],
                "exact_context_video_path": path,
            }
            requests.append(
                SimpleNamespace(
                    sampling_params=SimpleNamespace(
                        extra_args={"rollout_extra_args": rollout_inputs},
                    )
                )
            )
            observations.append(
                {
                    "video": np.full((2, 2, 2, 3), index // 2, dtype=np.uint8),
                    "state": np.asarray([[float(initial_seed)]], dtype=np.float32),
                    "instruction": "move",
                    "initial_noise_seed": initial_seed,
                    "rollout_seed": 10 + index,
                }
            )

        plan = self.adapter._request_conditioning_group_plan(
            requests,
            observations,
            video_key="video",
        )

        self.assertEqual(
            plan,
            {
                "batch_size": 4,
                "representative_indices": (0, 2),
                "group_index": (0, 0, 1, 1),
            },
        )

    def test_bulk_cpu_output_moves_batch_before_splitting(self):
        self.pipeline._bulk_cpu_output = True
        outputs = self.pipeline(
            _DiffusionRequestBatch(
                [self._request(seed=101, request_id="first"), self._request(seed=202, request_id="second")]
            )
        )

        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(not output.to_cpu for output in outputs))
        self.assertTrue(all(output.output.device.type == "cpu" for output in outputs))
        self.assertTrue(
            all(value.device.type == "cpu" for output in outputs for value in output.custom_output.values())
        )
        self.assertNotEqual(outputs[0].output.data_ptr(), outputs[1].output.data_ptr())
        self.assertNotEqual(
            outputs[0].custom_output["all_latents"].data_ptr(),
            outputs[1].custom_output["all_latents"].data_ptr(),
        )

    def test_rollout_regional_compile_targets_repeated_blocks_and_checks_parity(self):
        class CausalWanAttentionBlock(torch.nn.Module):
            def forward(self, value):
                return value + 1

        class TinyDiT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([CausalWanAttentionBlock(), CausalWanAttentionBlock()])

            def _forward_train(self, value):
                for block in self.blocks:
                    value = block(value)
                return value * 2, value + 1

        module = TinyDiT()
        compile_calls = []

        def fake_compile(function, **kwargs):
            compile_calls.append(kwargs)
            return lambda *args, **inner_kwargs: function(*args, **inner_kwargs)

        environment = {
            "WAM_ROLLOUT_TORCH_COMPILE": "false",
            "WAM_ROLLOUT_REGIONAL_COMPILE": "true",
            "WAM_ROLLOUT_REGIONAL_COMPILE_MODE": "default",
            "WAM_ROLLOUT_REGIONAL_COMPILE_FULLGRAPH": "false",
            "WAM_ROLLOUT_REGIONAL_COMPILE_DYNAMIC": "true",
            "WAM_ROLLOUT_REGIONAL_COMPILE_VERIFY": "true",
            "WAM_ROLLOUT_CUDA_GRAPH": "false",
        }
        with patch.dict(os.environ, environment), patch.object(torch, "compile", side_effect=fake_compile):
            self.adapter._install_rollout_dit_acceleration(module)
            output = module._forward_train(torch.tensor([2.0]))

        self.assertEqual(
            compile_calls,
            [
                {
                    "fullgraph": False,
                    "dynamic": True,
                    "options": {"emulate_precision_casts": True},
                },
                {
                    "fullgraph": False,
                    "dynamic": True,
                    "options": {"emulate_precision_casts": True},
                },
            ],
        )
        self.assertEqual(module._repeated_blocks, ["CausalWanAttentionBlock"])
        self.assertEqual(module._layerwise_offload_blocks_attrs, ["blocks"])
        self.assertEqual(module._verl_rollout_regional_compile_target, "_forward_train")
        self.assertEqual(len(module._verl_rollout_regional_compile_blocks), 2)
        torch.testing.assert_close(output[0], torch.tensor([8.0]))
        torch.testing.assert_close(output[1], torch.tensor([5.0]))

    def test_logprob_transport_can_be_disabled_without_dropping_trajectories(self):
        output = self.pipeline(self._request(logprobs=False))
        self.assertNotIn("all_log_probs", output.custom_output)
        self.assertNotIn("action_log_probs", output.custom_output)
        self.assertIn("all_latents", output.custom_output)
        self.assertIn("all_action_latents", output.custom_output)

    def test_sampling_and_replay_contract_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            self.pipeline(self._request(steps=5))

        missing_seed = self._request()
        missing_seed.sampling_params.seed = None
        with self.assertRaisesRegex(ValueError, "per-request seed"):
            self.pipeline(missing_seed)

        bad_sde = self._request()
        bad_sde.sampling_params.extra_args["action_sde_type"] = "sde"
        with self.assertRaisesRegex(ValueError, "only dance_sde"):
            self.pipeline(bad_sde)

        self.pipeline._policy.malformed = "short_action_trajectory"
        with self.assertRaisesRegex(ValueError, "one more state"):
            self.pipeline(self._request())

        self.pipeline._policy.malformed = "missing_condition"
        with self.assertRaises(KeyError):
            self.pipeline(self._request())

    def test_weight_sync_accepts_mapping_and_checks_names_and_shapes(self):
        expected_initialized = {name for name, _ in self.pipeline.named_parameters()}
        expected_initialized.update(name for name, _ in self.pipeline.named_buffers())

        new_weight = torch.full_like(self.pipeline.transformer.proj.weight, 3.0)
        loaded = self.pipeline.load_weights({"transformer.proj.weight": new_weight})
        self.assertEqual(loaded, expected_initialized)
        torch.testing.assert_close(self.pipeline.transformer.proj.weight, new_weight)

        new_scale = torch.full_like(self.pipeline.transformer.scale, 4.0)
        loaded = self.pipeline.load_weights([("action_head.model.base_model.model.scale", new_scale)])
        self.assertEqual(loaded, expected_initialized)
        torch.testing.assert_close(self.pipeline.transformer.scale, new_scale)

        with self.assertRaises(KeyError):
            self.pipeline.load_weights([("transformer.unknown", torch.ones(1))])
        with self.assertRaises(ValueError):
            self.pipeline.load_weights([("transformer.proj.weight", torch.ones(1))])

    def test_video_conversion_normalizes_layout_range_and_grayscale(self):
        tchw = torch.ones(6, 3, 2, 4, dtype=torch.float32)
        converted = self.adapter._as_numpy_video(tchw)
        self.assertEqual(converted.shape, (6, 2, 4, 3))
        self.assertEqual(converted.dtype, np.uint8)
        self.assertTrue(np.all(converted == 255))

        grayscale = np.zeros((6, 2, 4, 1), dtype=np.uint8)
        converted = self.adapter._as_numpy_video(grayscale)
        self.assertEqual(converted.shape, (6, 2, 4, 3))

        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.adapter._as_numpy_video(np.full((6, 2, 4, 3), np.nan, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
