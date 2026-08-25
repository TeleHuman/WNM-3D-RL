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

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _module():
    path = Path(__file__).resolve().parents[2] / "tools" / "data" / "sample_stage3_train.py"
    spec = importlib.util.spec_from_file_location("_sample_stage3_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(source_id: str, value: int) -> dict:
    return {
        "reward_model": json.dumps({"ground_truth": {"source_episode_id": source_id}}),
        "value": value,
    }


def test_dedup_keeps_first_source_row_and_take_preserves_selection_order(tmp_path):
    module = _module()
    first = tmp_path / "train-00000.parquet"
    second = tmp_path / "train-00001.parquet"
    pq.write_table(pa.Table.from_pylist([_row("a", 10), _row("b", 20)]), first)
    pq.write_table(pa.Table.from_pylist([_row("a", 30), _row("c", 40)]), second)

    unique_indices, unique_ids, scanned_rows = module.deduplicated_source_rows([first, second])

    assert scanned_rows == 4
    assert unique_ids == ["a", "b", "c"]
    np.testing.assert_array_equal(unique_indices, np.array([0, 1, 3]))
    selected = module.take_source_rows([first, second], np.array([3, 0]))
    assert selected.column("value").to_pylist() == [40, 10]
