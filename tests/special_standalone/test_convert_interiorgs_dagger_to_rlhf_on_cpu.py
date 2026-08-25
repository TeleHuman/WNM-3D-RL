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

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _action_normalization(*, scale: float = 4.0) -> dict:
    return {
        "schema": "wnm_3d_action_normalization_v1",
        "embodiment": "interiorgs",
        "action_key": "action.nav_delta",
        "channel_order": ["dx_m", "dy_m", "dyaw_rad"],
        "model_action_dim": 32,
        "action_horizon_per_chunk": 8,
        "decoded_action_dims": 3,
        "normalization_mode": "q99",
        "normalized_reference_range": [-1.0, 1.0],
        "q01": [-0.6, -0.8, -1.1],
        "q99": [1.5, 0.9, 1.3],
        "nav_action_scale": scale,
        "physical_q01_after_scale": [-0.15, -0.2, -0.275],
        "physical_q99_after_scale": [0.375, 0.225, 0.325],
        "decode_formula": "test decode formula",
        "encode_formula": "test encode formula",
        "decode_clamps_normalized_action": False,
    }


def _module():
    path = Path(__file__).resolve().parents[2] / "tools" / "data" / "convert_interiorgs_dagger_to_rlhf.py"
    spec = importlib.util.spec_from_file_location("_convert_interiorgs_dagger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_keeps_same_source_episode_together():
    module = _module()
    first = module.deterministic_split("0122_840008_134", 0.2, 42)
    assert module.deterministic_split("0122_840008_134", 0.2, 42) == first


def test_smoothing_preserves_shape_and_start():
    module = _module()
    points = np.zeros((33, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0, 4.8, 33)
    points[12:, 1] = 1.0
    smoothed = module.smooth_nav_xy(points)
    assert smoothed.shape == points.shape
    np.testing.assert_allclose(smoothed[0], points[0])


def test_query_suffix_uses_stable_source_episode_id():
    module = _module()

    assert module.canonical_source_episode_id({"trajectory": "0023_840016_209_q000037"}) == "0023_840016_209"


def test_checkpoint_model_record_contains_portable_content_hashes(tmp_path):
    module = _module()
    checkpoint = tmp_path / "checkpoint"
    metadata_path = checkpoint / "experiment_cfg" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"interiorgs": {}}\n', encoding="utf-8")
    normalization_path = checkpoint / "action_normalization.json"
    normalization_path.write_text(json.dumps(_action_normalization()), encoding="utf-8")
    config = {
        "action_dim": 32,
        "action_horizon": 32,
        "torch_dtype": "bfloat16",
        "action_head_cfg": {
            "config": {
                "target_video_height": 160,
                "target_video_width": 320,
                "num_frames": 33,
                "num_inference_timesteps": 4,
                "train_architecture": "CausalWanModel",
                "diffusion_model_cfg": {"diffusion_model_pretrained_path": "dit"},
                "text_encoder_cfg": {"text_encoder_pretrained_path": "text"},
                "image_encoder_cfg": {"image_encoder_pretrained_path": "image"},
                "vae_cfg": {"vae_pretrained_path": "vae"},
            }
        },
    }
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    record = module.load_checkpoint_model_record(checkpoint)

    assert record["action_horizon"] == 32
    assert record["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert record["metadata_sha256"] == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    assert record["action_normalization_sha256"] == hashlib.sha256(normalization_path.read_bytes()).hexdigest()


def test_action_normalization_is_checkpoint_authoritative(tmp_path):
    module = _module()
    checkpoint = tmp_path / "checkpoint"
    metadata_path = checkpoint / "experiment_cfg" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    normalization = _action_normalization()
    metadata_path.write_text(
        json.dumps(
            {
                "interiorgs": {
                    "statistics": {"action": {"nav_delta": {"q01": normalization["q01"], "q99": normalization["q99"]}}}
                }
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "action_normalization.json").write_text(json.dumps(normalization), encoding="utf-8")

    loaded = module.load_checkpoint_action_normalization(checkpoint, nav_action_scale_override=4.0)

    assert loaded["nav_action_scale"] == 4.0
    assert loaded["q01"] == normalization["q01"]
    with pytest.raises(ValueError, match="does not match the checkpoint contract"):
        module.load_checkpoint_action_normalization(checkpoint, nav_action_scale_override=1.0)
