#!/usr/bin/env python3
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

"""Validate the Stage-3 dataset/checkpoint contract for the shell launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(
    manifest_path: Path,
    checkpoint: Path,
    *,
    expected_train: int,
    expected_val: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Return validated train files, validation files, and row counts."""

    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "wnm-interiorgs-rlhf":
        raise ValueError(f"unsupported dataset manifest schema: {manifest.get('schema')!r}")
    counts = manifest["counts"]
    if counts.get("skipped") != 0:
        raise ValueError(f"refusing dataset with skipped rows: {counts}")
    if counts.get("train") != expected_train or counts.get("val") != expected_val:
        raise ValueError(f"refusing incomplete dataset: {counts}")

    model = manifest["checkpoint_model"]
    checkpoint = checkpoint.resolve()
    for relative, key in (
        (Path("config.json"), "config_sha256"),
        (Path("experiment_cfg/metadata.json"), "metadata_sha256"),
        (Path("action_normalization.json"), "action_normalization_sha256"),
    ):
        digest = _sha256(checkpoint / relative)
        if model.get(key) != digest:
            raise ValueError(
                f"dataset/checkpoint content mismatch: {relative}; expected={digest}, manifest={model.get(key)}"
            )

    expected_model = {
        "target_video_height": 160,
        "target_video_width": 320,
        "num_frames": 33,
        "num_inference_timesteps": 4,
    }
    mismatched = {
        key: {"expected": expected, "actual": model.get(key)}
        for key, expected in expected_model.items()
        if model.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"unexpected checkpoint transform record: {mismatched}")

    normalization_path = checkpoint / "action_normalization.json"
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    recorded_normalization = manifest.get("action_normalization")
    if not isinstance(recorded_normalization, dict):
        raise TypeError("manifest action_normalization must be an object")
    if recorded_normalization.get("sha256") != _sha256(normalization_path):
        raise ValueError("dataset action_normalization digest mismatch")
    recorded_without_digest = {key: value for key, value in recorded_normalization.items() if key != "sha256"}
    if recorded_without_digest != normalization:
        raise ValueError("dataset/checkpoint action_normalization content mismatch")
    expected_decode = {
        "q01": normalization.get("q01"),
        "q99": normalization.get("q99"),
        "nav_action_scale": normalization.get("nav_action_scale"),
    }
    recorded_decode = manifest.get("action_decode")
    if not isinstance(recorded_decode, dict):
        raise TypeError("manifest action_decode must be an object")
    if {key: recorded_decode.get(key) for key in expected_decode} != expected_decode:
        raise ValueError("dataset action_decode does not match the checkpoint normalization contract")

    train_files = manifest["train_files"]
    val_files = manifest["val_files"]
    if not isinstance(train_files, list) or not all(isinstance(path, str) for path in train_files):
        raise TypeError("manifest train_files must be a list of paths")
    if not isinstance(val_files, list) or not all(isinstance(path, str) for path in val_files):
        raise TypeError("manifest val_files must be a list of paths")
    return train_files, val_files, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-val", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train_files, val_files, counts = verify_manifest(
            args.manifest,
            args.checkpoint,
            expected_train=args.expected_train,
            expected_val=args.expected_val,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(train_files, separators=(",", ":")))
    print(json.dumps(val_files, separators=(",", ":")))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
