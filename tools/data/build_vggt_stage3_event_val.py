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

"""Build a deterministic, event-balanced VGGT validation set.

The validation rows are source-disjoint and balanced across four policy
events: collision precursors, premature-stop risks, states that require STOP,
and near-goal states that must continue.  Collision precursors are accepted
only when a moving query is followed by an almost stationary realized pose and
the current camera centre lies on the OCC safety boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from convert_interiorgs_dagger_to_rlhf import (
    ShardWriter,
    build_row,
    load_checkpoint_action_normalization,
    load_checkpoint_model_record,
)
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exclude-parquet-dir", type=Path, required=True)
    parser.add_argument("--occupancy-root", type=Path, required=True)
    parser.add_argument("--per-event", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--stop-radius-m", type=float, default=1.5)
    parser.add_argument(
        "--nav-action-scale",
        type=float,
        default=None,
        help="Optional assertion for the checkpoint scale; it cannot override the checkpoint contract",
    )
    parser.add_argument("--attempted-motion-min-m", type=float, default=0.08)
    parser.add_argument("--realized-stall-max-m", type=float, default=0.02)
    parser.add_argument("--prior-motion-min-m", type=float, default=0.05)
    parser.add_argument("--collision-clearance-min-px", type=float, default=3.5)
    parser.add_argument("--collision-clearance-max-px", type=float, default=6.0)
    parser.add_argument("--premature-motion-max-m", type=float, default=0.02)
    return parser.parse_args()


def digest(*parts: object) -> bytes:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).digest()


def sha256_lines(values) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def source_paths(root: Path, record: dict) -> tuple[Path, Path]:
    episode = int(record["episode_index"])
    chunk = episode // 1000
    parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    video = root / "videos" / f"chunk-{chunk:03d}" / "observation.images.rgb" / f"episode_{episode:06d}.mp4"
    return parquet, video


def complete(root: Path, record: dict) -> bool:
    return all(path.is_file() for path in source_paths(root, record))


def read_metadata(root: Path) -> tuple[dict[str, list[dict]], set[str], dict]:
    path = root / "meta" / "episodes.jsonl"
    snapshot_size = path.stat().st_size
    payload = path.open("rb").read(snapshot_size)
    if not payload.endswith(b"\n"):
        payload = payload.rsplit(b"\n", 1)[0] + b"\n"
    grouped: dict[str, list[dict]] = defaultdict(list)
    newest_by_worker: dict[str, tuple[int, str]] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        record["_metadata_line"] = line_number
        source = str(record.get("source_episode_id") or record.get("trajectory"))
        grouped[source].append(record)
        newest_by_worker[str(record.get("worker_id"))] = (line_number, source)
    for records in grouped.values():
        records.sort(
            key=lambda row: (
                int(row.get("query_step", -1)),
                int(row.get("episode_index", -1)),
            )
        )
    return (
        grouped,
        {source for _, source in newest_by_worker.values()},
        {
            "path": str(path.resolve()),
            "snapshot_bytes": snapshot_size,
            "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
            "rows": sum(map(len, grouped.values())),
            "source_trajectories": len(grouped),
        },
    )


def read_excluded_sources(directory: Path) -> tuple[set[str], dict]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no exclusion parquet files in {directory}")
    excluded: set[str] = set()
    rows = 0
    for path in files:
        table = pq.read_table(path, columns=["reward_model"], use_threads=False)
        rows += table.num_rows
        for value in table.column(0).to_pylist():
            source = str(value["ground_truth"]["source_episode_id"])
            excluded.add(source)
    return excluded, {
        "directory": str(directory.resolve()),
        "files": len(files),
        "rows": rows,
        "distinct_sources": len(excluded),
        "sources_sha256": sha256_lines(sorted(excluded)),
    }


def current_pose(root: Path, record: dict) -> np.ndarray:
    parquet, _ = source_paths(root, record)
    values = (
        pq.read_table(
            parquet,
            columns=["observation.camera_extrinsic"],
            use_threads=False,
        )
        .column(0)
        .to_pylist()
    )
    extrinsics = np.asarray(values, dtype=np.float32)
    if extrinsics.shape != (66, 4, 4):
        raise ValueError(f"unexpected extrinsic shape {extrinsics.shape}: {parquet}")
    # VGGT collection uses history_includes_current=true; frame 32 is the
    # current observation, frame 33 is its duplicated future anchor.
    return extrinsics[32, :2, 3].copy()


def world_to_pixel(point: np.ndarray, metadata: dict, width: int, height: int):
    lower = metadata.get("lower", metadata.get("min"))
    upper = metadata.get("upper", metadata.get("max"))
    sx = (float(upper[0]) - float(lower[0])) / width
    sy = (float(upper[1]) - float(lower[1])) / height
    return (
        (-float(point[0]) - float(lower[0])) / sx - 0.5,
        (float(upper[1]) + float(point[1])) / sy - 0.5,
    )


def point_clearances(items: list[dict], occupancy_root: Path, *, radius_px: int = 12) -> None:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_scene[item["source_episode_id"].rsplit("_", 1)[0]].append(item)
    for scene, scene_items in by_scene.items():
        root = occupancy_root / scene
        metadata = json.loads((root / "occupancy.json").read_text())
        free = np.asarray(Image.open(root / "occupancy.png").convert("L")) >= 200
        height, width = free.shape
        for item in scene_items:
            px, py = world_to_pixel(item["current_world_xy"], metadata, width, height)
            x, y = int(round(px)), int(round(py))
            if not (0 <= x < width and 0 <= y < height):
                clearance = 0.0
            else:
                y0, y1 = max(0, y - radius_px), min(height, y + radius_px + 1)
                x0, x1 = max(0, x - radius_px), min(width, x + radius_px + 1)
                ys, xs = np.nonzero(~free[y0:y1, x0:x1])
                if len(xs) == 0:
                    clearance = float(radius_px + 1)
                else:
                    clearance = float(np.sqrt(((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2).min()))
            item["current_clearance_px"] = clearance


def collision_candidates(
    grouped: dict[str, list[dict]],
    *,
    input_root: Path,
    occupancy_root: Path,
    excluded: set[str],
    active: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    metadata_candidates = []
    for source, records in grouped.items():
        if source in excluded or source in active:
            continue
        possibilities = []
        for index in range(1, len(records) - 1):
            previous, current, following = records[index - 1 : index + 2]
            if int(following["query_step"]) != int(current["query_step"]) + 1:
                continue
            previous_distance = float(previous["distance_to_goal"])
            current_distance = float(current["distance_to_goal"])
            following_distance = float(following["distance_to_goal"])
            attempted = float(current.get("policy_motion_m", 0.0))
            if (
                current_distance > args.stop_radius_m
                and attempted >= args.attempted_motion_min_m
                and abs(following_distance - current_distance) <= 0.01
                and abs(current_distance - previous_distance) > 0.02
                and complete(input_root, previous)
                and complete(input_root, current)
                and complete(input_root, following)
            ):
                possibilities.append((int(current["query_step"]), previous, current, following))
        if possibilities:
            _, previous, current, following = min(possibilities, key=lambda value: value[0])
            metadata_candidates.append(
                {
                    "event": "collision_precursor",
                    "source_episode_id": source,
                    "record": current,
                    "previous_record": previous,
                    "following_record": following,
                    "expected_stop": False,
                }
            )

    def load(item: dict) -> dict | None:
        try:
            previous = current_pose(input_root, item["previous_record"])
            current = current_pose(input_root, item["record"])
            following = current_pose(input_root, item["following_record"])
        except (FileNotFoundError, ValueError, OSError):
            return None
        item["current_world_xy"] = current
        item["prior_realized_motion_m"] = float(np.linalg.norm(current - previous))
        item["following_realized_motion_m"] = float(np.linalg.norm(following - current))
        return item

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pose_items = [item for item in pool.map(load, metadata_candidates) if item]
    pose_items = [
        item
        for item in pose_items
        if item["prior_realized_motion_m"] >= args.prior_motion_min_m
        and item["following_realized_motion_m"] <= args.realized_stall_max_m
    ]
    point_clearances(pose_items, occupancy_root)
    accepted = [
        item
        for item in pose_items
        if args.collision_clearance_min_px <= item["current_clearance_px"] <= args.collision_clearance_max_px
    ]
    return accepted, {
        "metadata_candidates": len(metadata_candidates),
        "pose_stall_candidates": len(pose_items),
        "occ_boundary_candidates": len(accepted),
    }


def scene_balanced_select(items: list[dict], count: int, *, seed: int, label: str, used: set[str]) -> list[dict]:
    eligible = [item for item in items if item["source_episode_id"] not in used]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for item in eligible:
        by_scene[str(item["record"].get("scene", ""))].append(item)
    for scene, values in by_scene.items():
        values.sort(key=lambda item: digest(seed, label, scene, item["source_episode_id"]))
    scenes = sorted(by_scene, key=lambda scene: digest(seed, label, "scene", scene))
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for scene in scenes:
            if depth < len(by_scene[scene]):
                selected.append(by_scene[scene][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise RuntimeError(f"need {count} {label} samples, only {len(eligible)} eligible")
        depth += 1
    used.update(item["source_episode_id"] for item in selected)
    return selected


def simple_candidates(
    grouped: dict[str, list[dict]],
    *,
    input_root: Path,
    excluded: set[str],
    active: set[str],
    collision_sources: set[str],
    args: argparse.Namespace,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for source, records in grouped.items():
        if source in excluded or source in active:
            continue
        terminal = records[-1]
        terminal_distance = float(terminal["distance_to_goal"])
        terminal_motion = float(terminal.get("policy_motion_m", math.inf))
        if complete(input_root, terminal) and terminal_distance <= args.stop_radius_m:
            if terminal_distance <= 0.75:
                event = "required_stop_core"
            elif terminal_distance <= 1.25:
                event = "required_stop_mid"
            else:
                event = "required_stop_boundary"
            buckets[event].append(
                {
                    "event": event,
                    "source_episode_id": source,
                    "record": terminal,
                    "expected_stop": True,
                }
            )

            continue_candidates = [
                record
                for record in records[:-1]
                if args.stop_radius_m < float(record["distance_to_goal"]) <= 3.0
                and float(record.get("policy_motion_m", 0.0)) >= args.attempted_motion_min_m
                and complete(input_root, record)
            ]
            if continue_candidates:
                record = min(
                    continue_candidates,
                    key=lambda value: (
                        float(value["distance_to_goal"]),
                        -int(value["query_step"]),
                    ),
                )
                buckets["near_goal_continue"].append(
                    {
                        "event": "near_goal_continue",
                        "source_episode_id": source,
                        "record": record,
                        "expected_stop": False,
                    }
                )
        if (
            source not in collision_sources
            and terminal_distance > args.stop_radius_m
            and terminal_motion <= args.premature_motion_max_m
            and int(terminal.get("query_step", 300)) < 299
            and complete(input_root, terminal)
        ):
            event = "premature_stop_near" if terminal_distance <= 3.0 else "premature_stop_far"
            buckets[event].append(
                {
                    "event": event,
                    "source_episode_id": source,
                    "record": terminal,
                    "expected_stop": False,
                }
            )
    return buckets


def annotate_row(row: dict, item: dict, args: argparse.Namespace) -> dict:
    record = item["record"]
    event = item["event"]
    annotation = {
        "event_validation_label": event,
        "event_validation_expected_stop": bool(item["expected_stop"]),
        "event_validation_collision_precursor": event == "collision_precursor",
        "event_validation_distance_to_goal_m": float(record["distance_to_goal"]),
        "event_validation_policy_motion_m": float(record.get("policy_motion_m", -1.0)),
        "event_validation_query_step": int(record.get("query_step", -1)),
        "event_validation_stop_radius_m": float(args.stop_radius_m),
    }
    for key in (
        "prior_realized_motion_m",
        "following_realized_motion_m",
        "current_clearance_px",
    ):
        if key in item:
            annotation[f"event_validation_{key}"] = float(item[key])
    if "following_record" in item:
        annotation["event_validation_following_episode_index"] = int(item["following_record"]["episode_index"])
    row["reward_model"]["ground_truth"].update(annotation)
    row["extra_info"]["event_validation"] = annotation
    return row


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {args.output_dir}")
    if args.per_event <= 0 or args.workers <= 0:
        raise ValueError("per-event and workers must be positive")
    grouped, active, source_snapshot = read_metadata(args.input_root)
    excluded, exclusion_audit = read_excluded_sources(args.exclude_parquet_dir)
    collision, collision_audit = collision_candidates(
        grouped,
        input_root=args.input_root,
        occupancy_root=args.occupancy_root,
        excluded=excluded,
        active=active,
        args=args,
    )
    collision_sources = {item["source_episode_id"] for item in collision}
    buckets = simple_candidates(
        grouped,
        input_root=args.input_root,
        excluded=excluded,
        active=active,
        collision_sources=collision_sources,
        args=args,
    )

    used: set[str] = set()
    selected = scene_balanced_select(collision, args.per_event, seed=args.seed, label="collision_precursor", used=used)
    selected += scene_balanced_select(
        buckets["premature_stop_near"],
        args.per_event // 2,
        seed=args.seed,
        label="premature_stop_near",
        used=used,
    )
    selected += scene_balanced_select(
        buckets["premature_stop_far"],
        args.per_event - args.per_event // 2,
        seed=args.seed,
        label="premature_stop_far",
        used=used,
    )
    selected += scene_balanced_select(
        buckets["required_stop_core"],
        args.per_event // 4,
        seed=args.seed,
        label="required_stop_core",
        used=used,
    )
    selected += scene_balanced_select(
        buckets["required_stop_mid"],
        args.per_event // 4,
        seed=args.seed,
        label="required_stop_mid",
        used=used,
    )
    selected += scene_balanced_select(
        buckets["required_stop_boundary"],
        args.per_event // 2,
        seed=args.seed,
        label="required_stop_boundary",
        used=used,
    )
    selected += scene_balanced_select(
        buckets["near_goal_continue"],
        args.per_event,
        seed=args.seed,
        label="near_goal_continue",
        used=used,
    )
    expected_rows = 4 * args.per_event
    if len(selected) != expected_rows or len(used) != expected_rows:
        raise AssertionError("selection is not row/source-disjoint as requested")
    selected.sort(key=lambda item: digest(args.seed, "output", item["source_episode_id"]))

    action_normalization = load_checkpoint_action_normalization(
        args.checkpoint,
        nav_action_scale_override=args.nav_action_scale,
    )
    q01 = action_normalization["q01"]
    q99 = action_normalization["q99"]
    args.nav_action_scale = action_normalization["nav_action_scale"]
    checkpoint_model = load_checkpoint_model_record(args.checkpoint)

    def convert(item: dict) -> dict:
        row = build_row(
            item["record"],
            input_root=args.input_root,
            occupancy_root=args.occupancy_root,
            q01=q01,
            q99=q99,
            nav_action_scale=args.nav_action_scale,
        )
        return annotate_row(row, item, args)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(convert, selected))

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.staging-",
            dir=args.output_dir.parent,
        )
    )
    published = False
    try:
        writer = ShardWriter(staging, "val", expected_rows)
        for row in rows:
            writer.append(row)
        writer.close()
        if len(writer.paths) != 1:
            raise AssertionError(f"expected one val shard, got {len(writer.paths)}")
        parquet = staging / "val-00000.parquet"
        table = pq.read_table(parquet, columns=["reward_model"], use_threads=False)
        if table.num_rows != expected_rows:
            raise AssertionError(f"output has {table.num_rows} rows")
        labels = []
        output_sources = []
        for value in table.column(0).to_pylist():
            ground_truth = value["ground_truth"]
            labels.append(ground_truth["event_validation_label"])
            output_sources.append(ground_truth["source_episode_id"])
        if len(set(output_sources)) != expected_rows:
            raise AssertionError("output source trajectories are not unique")
        overlap = set(output_sources) & excluded
        if overlap:
            raise AssertionError(f"output overlaps training sources: {sorted(overlap)[:5]}")

        with (staging / "selected_samples.jsonl").open("w") as stream:
            for order, item in enumerate(selected):
                record = item["record"]
                audit = {
                    "selection_order": order,
                    "event": item["event"],
                    "expected_stop": item["expected_stop"],
                    "source_episode_id": item["source_episode_id"],
                    "episode_index": int(record["episode_index"]),
                    "scene": str(record["scene"]),
                    "query_step": int(record["query_step"]),
                    "distance_to_goal_m": float(record["distance_to_goal"]),
                    "policy_motion_m": float(record.get("policy_motion_m", -1.0)),
                    "metadata_line": int(record["_metadata_line"]),
                }
                for key in (
                    "prior_realized_motion_m",
                    "following_realized_motion_m",
                    "current_clearance_px",
                ):
                    if key in item:
                        audit[key] = item[key]
                stream.write(json.dumps(audit, sort_keys=True) + "\n")

        high_level = Counter()
        for label in labels:
            if label.startswith("premature_stop"):
                high_level["premature_stop_risk"] += 1
            elif label.startswith("required_stop"):
                high_level["required_stop"] += 1
            else:
                high_level[label] += 1
        manifest = {
            "schema": "wnm-3d-stage3-event-val",
            "input_root": str(args.input_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "val_files": [str((args.output_dir / "val-00000.parquet").resolve())],
            "counts": {
                "val": expected_rows,
                "high_level_events": dict(sorted(high_level.items())),
                "labels": dict(sorted(Counter(labels).items())),
                "distinct_sources": len(set(output_sources)),
                "distinct_scenes": len({item["record"]["scene"] for item in selected}),
            },
            "selection": {
                "seed": args.seed,
                "one_row_per_source_trajectory": True,
                "scene_balance": "deterministic scene round-robin per label",
                "newest_active_source_per_worker_excluded": sorted(active),
                "collision_definition": (
                    "previous realized XY motion >= prior_motion_min; current policy "
                    "motion >= attempted_motion_min; following realized XY motion <= "
                    "realized_stall_max; current OCC clearance within configured band"
                ),
                "premature_stop_risk_definition": (
                    "terminal query outside 1.5m with low policy XY motion, before query 299, "
                    "and source has no qualified collision precursor; this is a risk label "
                    "because collection metadata does not retain raw dyaw"
                ),
                "required_stop_definition": "terminal successful query at distance <= 1.5m",
                "near_goal_continue_definition": (
                    "closest complete moving query in (1.5m,3m] from a successful trajectory"
                ),
                "thresholds": {
                    key: getattr(args, key)
                    for key in (
                        "stop_radius_m",
                        "attempted_motion_min_m",
                        "prior_motion_min_m",
                        "realized_stall_max_m",
                        "collision_clearance_min_px",
                        "collision_clearance_max_px",
                        "premature_motion_max_m",
                    )
                },
                "selected_sources_sha256": sha256_lines(output_sources),
                "audit_file": str((args.output_dir / "selected_samples.jsonl").resolve()),
            },
            "source_snapshot": source_snapshot,
            "training_exclusion": exclusion_audit,
            "selected_training_overlap": 0,
            "candidate_counts": {
                **collision_audit,
                **{key: len(value) for key, value in sorted(buckets.items())},
            },
            "action_decode": {
                "checkpoint": str(args.checkpoint.resolve()),
                "q01": q01,
                "q99": q99,
                "nav_action_scale": args.nav_action_scale,
            },
            "action_normalization": action_normalization,
            "checkpoint_model": checkpoint_model,
            "media": "absolute source MP4 references; no video bytes copied",
            "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staging, args.output_dir)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    print(json.dumps(manifest["counts"], sort_keys=True))
    print(args.output_dir / "val-00000.parquet")


if __name__ == "__main__":
    main()
