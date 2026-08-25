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
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPO_ROOT / "recipes" / "wnm_3d" / "stage3" / "verify_data_manifest.py"


def _write_contract(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "experiment_cfg").mkdir(parents=True)
    config = checkpoint / "config.json"
    metadata = checkpoint / "experiment_cfg" / "metadata.json"
    normalization = checkpoint / "action_normalization.json"
    config.write_text('{"model_type":"wnm_3d"}\n', encoding="utf-8")
    metadata.write_text('{"statistics":{}}\n', encoding="utf-8")
    normalization_payload = {
        "schema": "wnm_3d_action_normalization_v1",
        "q01": [-0.6, -0.8, -1.1],
        "q99": [1.5, 0.9, 1.3],
        "nav_action_scale": 4.0,
    }
    normalization.write_text(json.dumps(normalization_payload), encoding="utf-8")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "wnm-interiorgs-rlhf",
                "counts": {"train": 8, "val": 0, "skipped": 0},
                "train_files": ["train.parquet"],
                "val_files": [],
                "action_decode": {
                    "checkpoint": str(checkpoint.resolve()),
                    "q01": normalization_payload["q01"],
                    "q99": normalization_payload["q99"],
                    "nav_action_scale": normalization_payload["nav_action_scale"],
                },
                "action_normalization": {
                    **normalization_payload,
                    "sha256": hashlib.sha256(normalization.read_bytes()).hexdigest(),
                },
                "checkpoint_model": {
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    "action_normalization_sha256": hashlib.sha256(normalization.read_bytes()).hexdigest(),
                    "target_video_height": 160,
                    "target_video_width": 320,
                    "num_frames": 33,
                    "num_inference_timesteps": 4,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, checkpoint


def _run(manifest: Path, checkpoint: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--expected-train",
            "8",
            "--expected-val",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_valid_manifest_emits_exact_shell_contract(tmp_path):
    manifest, checkpoint = _write_contract(tmp_path)

    result = _run(manifest, checkpoint)

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        '["train.parquet"]',
        "[]",
        '{"skipped": 0, "train": 8, "val": 0}',
    ]


def test_checkpoint_path_is_informational_when_content_matches(tmp_path):
    manifest, checkpoint = _write_contract(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["action_decode"]["checkpoint"] = "/different/machine/checkpoints/stage2"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(manifest, checkpoint)

    assert result.returncode == 0


def test_checkpoint_mismatch_fails_without_partial_output(tmp_path):
    manifest, checkpoint = _write_contract(tmp_path)
    (checkpoint / "config.json").write_text('{"model_type":"changed"}\n', encoding="utf-8")

    result = _run(manifest, checkpoint)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "dataset/checkpoint content mismatch: config.json" in result.stderr
