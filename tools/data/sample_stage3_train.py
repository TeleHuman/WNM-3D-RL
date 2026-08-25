#!/usr/bin/env python3
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
"""Build a deterministic, source-deduplicated Stage-3 training set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"manifest must contain a JSON object: {path}")
    return payload


def parquet_row_count(paths: list[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _source_episode_id(reward_model: Any) -> str:
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    if not isinstance(reward_model, dict):
        raise TypeError(f"reward_model must be an object, got {type(reward_model).__name__}")
    ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    if not isinstance(ground_truth, dict):
        raise TypeError("reward_model.ground_truth must be an object")
    source_id = str(ground_truth.get("source_episode_id") or "").strip()
    if not source_id:
        raise ValueError("reward_model.ground_truth.source_episode_id is missing")
    return source_id


def deduplicated_source_rows(paths: list[Path]) -> tuple[np.ndarray, list[str], int]:
    """Return each source episode's first row in stable source-file order."""

    first_rows: dict[str, int] = {}
    total_rows = 0
    for path in paths:
        values = pq.read_table(path, columns=["reward_model"], use_threads=False).column(0)
        for local_index, reward_model in enumerate(values.to_pylist()):
            source_id = _source_episode_id(reward_model)
            first_rows.setdefault(source_id, total_rows + local_index)
        total_rows += len(values)
    indices = np.fromiter(first_rows.values(), dtype=np.int64, count=len(first_rows))
    return indices, list(first_rows), total_rows


def take_source_rows(paths: list[Path], chosen: np.ndarray) -> pa.Table:
    sort_order = np.argsort(chosen)
    sorted_chosen = chosen[sort_order]
    selected_tables: list[pa.Table] = []
    offset = 0
    cursor = 0
    for path in paths:
        row_count = pq.ParquetFile(path).metadata.num_rows
        end = offset + row_count
        next_cursor = int(np.searchsorted(sorted_chosen, end, side="left"))
        if next_cursor > cursor:
            local_indices = sorted_chosen[cursor:next_cursor] - offset
            selected_tables.append(pq.read_table(path).take(pa.array(local_indices, type=pa.int64())))
        cursor = next_cursor
        offset = end
        if cursor == len(chosen):
            break
    if cursor != len(chosen):
        raise RuntimeError(f"selected {cursor} of {len(chosen)} requested rows")
    table_in_source_order = pa.concat_tables(selected_tables)
    return table_in_source_order.take(pa.array(np.argsort(sort_order), type=pa.int64()))


def write_shards(
    table: pa.Table,
    *,
    temp_dir: Path,
    final_dir: Path,
    shard_size: int,
) -> list[str]:
    paths: list[str] = []
    for shard_index, offset in enumerate(range(0, table.num_rows, shard_size)):
        name = f"train-{shard_index:05d}.parquet"
        shard = table.slice(offset, min(shard_size, table.num_rows - offset))
        pq.write_table(shard, temp_dir / name, compression="zstd")
        paths.append(str((final_dir / name).resolve()))
    return paths


def int64_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def lines_sha256(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.train_size <= 0:
        raise ValueError("train-size must be positive")
    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive")

    source_manifest_path = args.source_manifest.expanduser().resolve()
    source = read_manifest(source_manifest_path)
    source_paths = [Path(path).expanduser().resolve() for path in source["train_files"]]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest references missing files: {missing[:10]}")
    source_rows = parquet_row_count(source_paths)
    recorded_rows = int(source.get("counts", {}).get("train", -1))
    if recorded_rows != source_rows:
        raise ValueError(f"source train count does not match parquet metadata: {recorded_rows} != {source_rows}")

    unique_indices, unique_ids, scanned_rows = deduplicated_source_rows(source_paths)
    if scanned_rows != source_rows:
        raise RuntimeError(f"dedup scan changed row count: {scanned_rows} != {source_rows}")
    if args.train_size > len(unique_indices):
        raise ValueError(
            f"requested {args.train_size} rows but only {len(unique_indices)} unique source episodes exist"
        )
    order = np.random.default_rng(args.seed).permutation(len(unique_indices))
    selected_offsets = order[: args.train_size]
    selected_indices = unique_indices[selected_offsets]
    selected_ids = [unique_ids[int(offset)] for offset in selected_offsets]
    table = take_source_rows(source_paths, selected_indices)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temp_dir.exists():
        raise FileExistsError(temp_dir)
    temp_dir.mkdir()
    try:
        train_files = write_shards(
            table,
            temp_dir=temp_dir,
            final_dir=output_dir,
            shard_size=args.shard_size,
        )
        manifest = dict(source)
        manifest["counts"] = {"train": args.train_size, "val": 0, "skipped": 0}
        manifest["train_files"] = train_files
        manifest["val_files"] = []
        manifest["sampling"] = {
            "source_manifest": str(source_manifest_path),
            "source_train_rows": source_rows,
            "method": "first_row_per_source_episode_then_seeded_permutation",
            "dedup_key": "reward_model.ground_truth.source_episode_id",
            "unique_source_episodes": len(unique_indices),
            "duplicates_removed": source_rows - len(unique_indices),
            "seed": args.seed,
            "selected_source_indices_sha256": int64_sha256(selected_indices),
            "selected_source_episode_ids_sha256": lines_sha256(selected_ids),
            "selection_order": "deterministic_random",
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        written = parquet_row_count([temp_dir / Path(path).name for path in train_files])
        if written != args.train_size:
            raise RuntimeError(f"written train row mismatch: {written} != {args.train_size}")
        temp_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train": args.train_size,
                "train_shards": len(train_files),
                "unique_source_episodes": len(unique_indices),
                "duplicates_removed": source_rows - len(unique_indices),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
