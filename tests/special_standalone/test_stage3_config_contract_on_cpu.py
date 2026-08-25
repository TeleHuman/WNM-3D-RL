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

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPO_ROOT / "recipes" / "wnm_3d" / "stage3" / "verify_config.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("stage3_verify_config", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sticky_rollout_partition_accepts_complete_worker_groups():
    verifier = _load_verifier()

    verifier.verify_sticky_rollout_partition(prompt_batch=64, rollout_workers=8, enabled=True)
    verifier.verify_sticky_rollout_partition(prompt_batch=1, rollout_workers=8, enabled=False)


def test_sticky_rollout_partition_rejects_split_prompt_groups():
    verifier = _load_verifier()

    with pytest.raises(RuntimeError, match="prompt batch to be divisible"):
        verifier.verify_sticky_rollout_partition(prompt_batch=1, rollout_workers=8, enabled=True)
