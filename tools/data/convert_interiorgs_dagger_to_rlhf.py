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

"""Convert GN0's 66-frame InteriorGS DAgger episodes to RLHF parquet shards.

The source videos are referenced, not copied.  Run this where the absolute
source paths are also visible to rollout and reward workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

PAST_FRAMES = 33
FUTURE_FRAMES = 33
EPISODE_FRAMES = PAST_FRAMES + FUTURE_FRAMES
CAM_TO_NAV = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
ACTION_NORMALIZATION_FILENAME = "action_normalization.json"
ACTION_NORMALIZATION_SCHEMA = "wnm_3d_action_normalization_v1"
ACTION_CHANNEL_ORDER = ["dx_m", "dy_m", "dyaw_rad"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=f"WNM checkpoint containing the canonical {ACTION_NORMALIZATION_FILENAME}",
    )
    parser.add_argument(
        "--occupancy-root",
        type=Path,
        required=True,
        help="InteriorGS scene root containing occupancy maps",
    )
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--nav-action-scale",
        type=float,
        default=None,
        help="Optional assertion for the checkpoint scale; it cannot override the checkpoint contract",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded thread pool for reading the source's many small parquet files",
    )
    parser.add_argument(
        "--worker-prefetch",
        type=int,
        default=4,
        help="Number of in-flight source rows per worker",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_normalization = load_checkpoint_action_normalization(
        args.checkpoint,
        nav_action_scale_override=args.nav_action_scale,
    )
    args.q01 = action_normalization["q01"]
    args.q99 = action_normalization["q99"]
    args.nav_action_scale = action_normalization["nav_action_scale"]
    checkpoint_model = load_checkpoint_model_record(args.checkpoint)
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("*.parquet"))
    if existing and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already contains parquet shards; pass --overwrite to replace them")
    if args.overwrite:
        for path in existing:
            path.unlink()

    writers = {
        "train": ShardWriter(args.output_dir, "train", args.shard_size),
        "val": ShardWriter(args.output_dir, "val", args.shard_size),
    }
    counts = {"train": 0, "val": 0, "skipped": 0}
    episodes_path = args.input_root / "meta" / "episodes.jsonl"

    def consume(
        source_line: int,
        episode: dict,
        row_or_future: dict | Future,
    ) -> None:
        try:
            row = row_or_future.result() if isinstance(row_or_future, Future) else row_or_future
        except (FileNotFoundError, KeyError, ValueError) as exc:
            counts["skipped"] += 1
            print(f"skip metadata line {source_line}: {exc}")
            return
        split_key = str(episode.get("source_episode_id") or episode["trajectory"])
        split = deterministic_split(split_key, args.val_fraction, args.split_seed)
        writers[split].append(row)
        counts[split] += 1
        total = counts["train"] + counts["val"]
        if total % 10000 == 0:
            print(f"converted={total} train={counts['train']} val={counts['val']} skipped={counts['skipped']}")

    def submit(pool: ThreadPoolExecutor, episode: dict) -> Future:
        return pool.submit(
            build_row,
            episode,
            input_root=args.input_root,
            occupancy_root=args.occupancy_root,
            q01=args.q01,
            q99=args.q99,
            nav_action_scale=args.nav_action_scale,
        )

    with episodes_path.open("r", encoding="utf-8") as stream:
        records = ((source_line, json.loads(line)) for source_line, line in enumerate(stream, start=1) if line.strip())
        if args.workers == 1:
            for source_line, episode in records:
                if args.limit is not None and counts["train"] + counts["val"] >= args.limit:
                    break
                try:
                    row = build_row(
                        episode,
                        input_root=args.input_root,
                        occupancy_root=args.occupancy_root,
                        q01=args.q01,
                        q99=args.q99,
                        nav_action_scale=args.nav_action_scale,
                    )
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    counts["skipped"] += 1
                    print(f"skip metadata line {source_line}: {exc}")
                    continue
                consume(source_line, episode, row)
        else:
            max_pending = args.workers * args.worker_prefetch
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending: list[tuple[int, dict, Future]] = []
                for source_line, episode in records:
                    pending.append((source_line, episode, submit(pool, episode)))
                    if len(pending) < max_pending:
                        continue
                    for pending_line, pending_episode, future in pending:
                        if args.limit is not None and counts["train"] + counts["val"] >= args.limit:
                            break
                        consume(pending_line, pending_episode, future)
                    pending.clear()
                    if args.limit is not None and counts["train"] + counts["val"] >= args.limit:
                        break
                for pending_line, pending_episode, future in pending:
                    if args.limit is not None and counts["train"] + counts["val"] >= args.limit:
                        break
                    consume(pending_line, pending_episode, future)

    for writer in writers.values():
        writer.close()
    manifest = {
        "schema": "wnm-interiorgs-rlhf",
        "input_root": str(args.input_root.resolve()),
        "occupancy_root": str(args.occupancy_root.resolve()),
        "counts": counts,
        "train_files": [str(path) for path in writers["train"].paths],
        "val_files": [str(path) for path in writers["val"].paths],
        "split": {
            "unit": "source_episode_id",
            "val_fraction": args.val_fraction,
            "seed": args.split_seed,
        },
        "action_decode": {
            "checkpoint": str(args.checkpoint.resolve()),
            "q01": list(args.q01),
            "q99": list(args.q99),
            "nav_action_scale": args.nav_action_scale,
        },
        "action_normalization": action_normalization,
        "checkpoint_model": checkpoint_model,
        "conversion": {
            "workers": args.workers,
            "worker_prefetch": args.worker_prefetch,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))


class ShardWriter:
    def __init__(self, output_dir: Path, split: str, shard_size: int):
        self.output_dir = output_dir
        self.split = split
        self.shard_size = shard_size
        self.buffer: list[dict] = []
        self.paths: list[Path] = []

    def append(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.output_dir / f"{self.split}-{len(self.paths):05d}.parquet"
        table = pa.Table.from_pylist(self.buffer)
        pq.write_table(table, path, compression="zstd")
        self.paths.append(path.resolve())
        self.buffer.clear()

    def close(self) -> None:
        self.flush()


def build_row(
    episode: dict,
    *,
    input_root: Path,
    occupancy_root: Path,
    q01: list[float],
    q99: list[float],
    nav_action_scale: float,
) -> dict:
    episode_index = int(episode["episode_index"])
    if int(episode.get("length", EPISODE_FRAMES)) != EPISODE_FRAMES:
        raise ValueError(f"episode {episode_index} is not {EPISODE_FRAMES} frames")
    chunk = episode_index // 1000
    source_parquet = input_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    video_path = (
        input_root / "videos" / f"chunk-{chunk:03d}" / "observation.images.rgb" / f"episode_{episode_index:06d}.mp4"
    )
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    extrinsics = read_extrinsics(source_parquet)
    if extrinsics.shape != (EPISODE_FRAMES, 4, 4):
        raise ValueError(f"episode {episode_index} extrinsics have shape {extrinsics.shape}")

    instruction = str((episode.get("tasks") or [""])[0]).strip()
    if not instruction:
        raise ValueError(f"episode {episode_index} has no instruction")
    goal = episode.get("goal_position_world")
    if not isinstance(goal, list) or len(goal) < 2:
        raise ValueError(f"episode {episode_index} has no world goal")

    source_episode_id = canonical_source_episode_id(episode)
    # DAgger stage-2 trajectories may append a query suffix:
    #   trajectory        = 0023_840016_209_q000037
    #   source_episode_id = 0023_840016_209
    # InteriorGS occupancy assets are stored under 0023_840016.  Resolve the
    # scene from source_episode_id so query-suffixed and legacy records share
    # the same conversion semantics.
    if "_" not in source_episode_id:
        raise ValueError(f"episode {episode_index} has invalid source_episode_id: {source_episode_id!r}")
    scene_name = source_episode_id.rsplit("_", 1)[0]
    scene_dir = resolve_occupancy_scene(str(occupancy_root), scene_name)

    target_extrinsics = extrinsics[PAST_FRAMES:]
    raw_world_xy = target_extrinsics[:, :2, 3]
    smoothed_world_xy = smooth_world_trajectory(target_extrinsics)
    gt_local_smoothed = smooth_nav_xy(_world_to_start_local(target_extrinsics))
    # Store only the two components represented exactly here. Reward consumes
    # the full smoothed world trajectory, not this audit-only field.
    gt_nav_deltas_scaled_xy = np.diff(gt_local_smoothed[:, :2], axis=0) * nav_action_scale

    ground_truth = {
        "episode_index": episode_index,
        "source_episode_id": source_episode_id,
        "scene": str(episode.get("scene", "")),
        "scene_dir": str(scene_dir.resolve()),
        "video_path": str(video_path.resolve()),
        "future_start": PAST_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "start_extrinsic": target_extrinsics[0].tolist(),
        "gt_world_xy_raw": raw_world_xy.tolist(),
        "gt_world_xy": smoothed_world_xy.tolist(),
        "goal_world_xy": [float(goal[0]), float(goal[1])],
        "q01": [float(value) for value in q01],
        "q99": [float(value) for value in q99],
        "nav_action_scale": float(nav_action_scale),
        "stop_radius_m": float(episode.get("stop_radius_m", 1.5)),
    }
    return {
        "data_source": "wnm_interiorgs_navigation",
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "video", "video": str(video_path.resolve())},
                ],
            }
        ],
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {
            "rollout_extra_args": {
                "instruction": instruction,
                "state": np.zeros((4, 3), dtype=np.float32).tolist(),
                "target_prefix_frames": 1,
                "exact_context_video_path": str(video_path.resolve()),
                "exact_context_video_frames": EPISODE_FRAMES,
            },
            "gt_nav_deltas_scaled_xy": gt_nav_deltas_scaled_xy.tolist(),
            "task_id": f"interiorgs-{episode_index:06d}",
        },
    }


def read_extrinsics(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(
        path,
        columns=["observation.camera_extrinsic"],
        use_threads=False,
    )
    return np.asarray(table.column(0).to_pylist(), dtype=np.float32).reshape(-1, 4, 4)


def canonical_source_episode_id(episode: dict) -> str:
    """Return the stable source ID, stripping a DAgger query suffix if needed."""

    raw = str(episode.get("source_episode_id") or episode.get("trajectory") or "")
    return re.sub(r"_q\d+$", "", raw)


@cache
def resolve_occupancy_scene(occupancy_root: str, scene_name: str) -> Path:
    scene_dir = Path(occupancy_root) / scene_name
    if not (scene_dir / "occupancy.png").is_file() or not (scene_dir / "occupancy.json").is_file():
        raise FileNotFoundError(f"occupancy assets not found for {scene_name}: {scene_dir}")
    return scene_dir


def load_checkpoint_action_quantiles(checkpoint: Path) -> tuple[list[float], list[float]]:
    """Return q01/q99 from the canonical checkpoint normalization contract."""

    normalization = load_checkpoint_action_normalization(checkpoint)
    return normalization["q01"], normalization["q99"]


def load_checkpoint_action_normalization(
    checkpoint: Path,
    *,
    nav_action_scale_override: float | None = None,
) -> dict:
    normalization_path = checkpoint / ACTION_NORMALIZATION_FILENAME
    metadata_path = checkpoint / "experiment_cfg" / "metadata.json"
    if not normalization_path.is_file():
        raise FileNotFoundError(normalization_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        q01 = [float(value) for value in normalization["q01"]]
        q99 = [float(value) for value in normalization["q99"]]
        nav_action_scale = float(normalization["nav_action_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid WNM action normalization in {normalization_path}") from exc
    expected_fields = {
        "schema": ACTION_NORMALIZATION_SCHEMA,
        "embodiment": "interiorgs",
        "action_key": "action.nav_delta",
        "channel_order": ACTION_CHANNEL_ORDER,
        "model_action_dim": 32,
        "action_horizon_per_chunk": 8,
        "decoded_action_dims": 3,
        "normalization_mode": "q99",
        "normalized_reference_range": [-1.0, 1.0],
        "decode_clamps_normalized_action": False,
    }
    mismatched = {
        key: {"expected": expected, "actual": normalization.get(key)}
        for key, expected in expected_fields.items()
        if normalization.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"unexpected WNM action normalization contract in {normalization_path}: {mismatched}")
    if (
        len(q01) != 3
        or len(q99) != 3
        or not all(math.isfinite(value) for value in q01 + q99)
        or np.any(np.asarray(q99) <= np.asarray(q01))
    ):
        raise ValueError(f"invalid q01/q99 in {normalization_path}: q01={q01}, q99={q99}")
    if not math.isfinite(nav_action_scale) or nav_action_scale <= 0:
        raise ValueError(f"invalid nav_action_scale in {normalization_path}: {nav_action_scale}")
    if nav_action_scale_override is not None and not math.isclose(
        nav_action_scale_override,
        nav_action_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "--nav-action-scale does not match the checkpoint contract: "
            f"override={nav_action_scale_override}, checkpoint={nav_action_scale}"
        )
    try:
        statistics = metadata["interiorgs"]["statistics"]["action"]["nav_delta"]
        metadata_q01 = [float(value) for value in statistics["q01"]]
        metadata_q99 = [float(value) for value in statistics["q99"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid InteriorGS action statistics in {metadata_path}") from exc
    if metadata_q01 != q01 or metadata_q99 != q99:
        raise ValueError(f"checkpoint normalization and metadata disagree: {normalization_path} vs {metadata_path}")
    return {
        **normalization,
        "q01": q01,
        "q99": q99,
        "nav_action_scale": nav_action_scale,
        "sha256": hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
    }


def load_checkpoint_model_record(checkpoint: Path) -> dict:
    config_path = checkpoint / "config.json"
    metadata_path = checkpoint / "experiment_cfg" / "metadata.json"
    normalization_path = checkpoint / ACTION_NORMALIZATION_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not normalization_path.is_file():
        raise FileNotFoundError(normalization_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        action = config["action_head_cfg"]["config"]
        diffusion = action["diffusion_model_cfg"]
        text_encoder = action["text_encoder_cfg"]
        image_encoder = action["image_encoder_cfg"]
        vae = action["vae_cfg"]
        record = {
            "checkpoint": str(checkpoint.resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "action_normalization_sha256": hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
            "target_video_height": int(action["target_video_height"]),
            "target_video_width": int(action["target_video_width"]),
            "num_frames": int(action["num_frames"]),
            "num_inference_timesteps": int(action["num_inference_timesteps"]),
            "action_dim": int(config["action_dim"]),
            "action_horizon": int(config["action_horizon"]),
            "train_architecture": str(action["train_architecture"]),
            "torch_dtype": str(config["torch_dtype"]),
            "diffusion_model_pretrained_path": str(diffusion["diffusion_model_pretrained_path"]),
            "text_encoder_pretrained_path": str(text_encoder["text_encoder_pretrained_path"]),
            "image_encoder_pretrained_path": str(image_encoder["image_encoder_pretrained_path"]),
            "vae_pretrained_path": str(vae["vae_pretrained_path"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid WNM model config in {config_path}") from exc
    return record


def smooth_world_trajectory(extrinsics: np.ndarray, window: int = 7, passes: int = 2) -> np.ndarray:
    local = _world_to_start_local(extrinsics)
    local = smooth_nav_xy(local, window=window, passes=passes)
    base = extrinsics[0]
    rotation_world_agent = base[:3, :3] @ CAM_TO_NAV.T
    world = (rotation_world_agent @ local.T).T + base[:3, 3]
    return world[:, :2].astype(np.float32)


def _world_to_start_local(extrinsics: np.ndarray) -> np.ndarray:
    base = extrinsics[0]
    camera_points = (base[:3, :3].T @ (extrinsics[:, :3, 3] - base[:3, 3]).T).T
    return (CAM_TO_NAV @ camera_points.T).T.astype(np.float32)


def smooth_nav_xy(points: np.ndarray, window: int = 7, passes: int = 2) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 2 or window <= 1 or passes <= 0:
        return points.copy()
    if window == 7:
        kernel = np.array([1, 3, 6, 7, 6, 3, 1], dtype=np.float32)
    elif window == 5:
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    elif window == 3:
        kernel = np.array([1, 2, 1], dtype=np.float32)
    else:
        kernel = np.ones(window, dtype=np.float32)
    kernel /= kernel.sum()
    output = points.copy()
    for _ in range(passes):
        deltas = output[1:, :2] - output[:-1, :2]
        left = window // 2
        right = window - 1 - left
        smoothed = np.stack(
            [
                np.convolve(
                    np.pad(deltas[:, axis], (left, right), mode="edge"),
                    kernel,
                    mode="valid",
                )
                for axis in range(2)
            ],
            axis=-1,
        )
        next_output = output.copy()
        next_output[0, :2] = output[0, :2]
        next_output[1:, :2] = output[0, :2] + np.cumsum(smoothed, axis=0)
        output = next_output
    return output


def deterministic_split(key: str, val_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{key}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if unit < val_fraction else "train"


def _validate_args(args: argparse.Namespace) -> None:
    if not (args.input_root / "meta" / "episodes.jsonl").is_file():
        raise FileNotFoundError(args.input_root / "meta" / "episodes.jsonl")
    if not args.occupancy_root.is_dir():
        raise FileNotFoundError(args.occupancy_root)
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0,1)")
    if args.nav_action_scale <= 0:
        raise ValueError("--nav-action-scale must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.worker_prefetch <= 0:
        raise ValueError("--worker-prefetch must be positive")
    if np.any(np.asarray(args.q99) <= np.asarray(args.q01)):
        raise ValueError("every q99 value must be greater than q01")


if __name__ == "__main__":
    main()
