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
"""CPU tests for the canonical WAM actor-to-full-VLN exporter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "checkpoint" / "export_wnm_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("wnm_3d_exporter", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


def _make_checkpoints(
    tmp_path: Path,
    *,
    actor_shape=(2, 2),
) -> tuple[Path, Path]:
    base = tmp_path / "stage2"
    actor = tmp_path / "actor"
    base.mkdir()
    actor.mkdir()
    (base / "config.json").write_text('{"model": "full-vln"}\n', encoding="utf-8")
    (base / "processor.json").write_text('{"preserve": true}\n', encoding="utf-8")
    (base / "action_normalization.json").write_text(
        '{"schema":"wnm_3d_action_normalization_v1","nav_action_scale":4.0}\n',
        encoding="utf-8",
    )
    save_file(
        {
            "action_head.model.block.weight": torch.zeros(2, 2),
            "action_head.model.block.bias": torch.zeros(2),
            "frozen_encoder.weight": torch.full((1,), 9.0),
        },
        base / "model.safetensors",
    )
    actor_state = {
        "transformer.block.weight": torch.full(actor_shape, 3.0),
        "transformer.block.bias": torch.full((2,), 4.0),
    }
    save_file(actor_state, actor / "model.safetensors")
    return base, actor


def test_export_replaces_joint_dit_and_preserves_full_vln(tmp_path):
    base, actor = _make_checkpoints(tmp_path)
    output = tmp_path / "exported"

    manifest = exporter.export_checkpoint(
        base_vln=base,
        actor_checkpoint=actor,
        output_dir=output,
    )

    assert manifest["replaced_tensor_count"] == 2
    assert (output / "config.json").read_text(encoding="utf-8") == '{"model": "full-vln"}\n'
    assert (output / "processor.json").read_text(encoding="utf-8") == '{"preserve": true}\n'
    assert (output / "action_normalization.json").read_bytes() == (base / "action_normalization.json").read_bytes()
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        torch.testing.assert_close(handle.get_tensor("action_head.model.block.weight"), torch.full((2, 2), 3.0))
        torch.testing.assert_close(handle.get_tensor("action_head.model.block.bias"), torch.full((2,), 4.0))
        torch.testing.assert_close(handle.get_tensor("frozen_encoder.weight"), torch.full((1,), 9.0))
    export_manifest = json.loads((output / "wam_export_manifest.json").read_text(encoding="utf-8"))
    assert export_manifest["format"] == "wnm_full_vln_export_v1"


def test_export_dry_run_validates_without_writing(tmp_path):
    base, actor = _make_checkpoints(tmp_path)
    output = tmp_path / "not-created"

    manifest = exporter.export_checkpoint(
        base_vln=base,
        actor_checkpoint=actor,
        output_dir=output,
        dry_run=True,
    )

    assert manifest["dry_run"] is True
    assert not output.exists()


def test_export_rejects_actor_shape_mismatch(tmp_path):
    base, actor = _make_checkpoints(tmp_path, actor_shape=(3, 2))

    with pytest.raises(ValueError, match="Tensor shape mismatch"):
        exporter.export_checkpoint(
            base_vln=base,
            actor_checkpoint=actor,
            output_dir=tmp_path / "exported",
            dry_run=True,
        )


def test_export_refuses_existing_output(tmp_path):
    base, actor = _make_checkpoints(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        exporter.export_checkpoint(base_vln=base, actor_checkpoint=actor, output_dir=output)
