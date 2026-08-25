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
"""Merge a trained WNM actor into a complete inference VLN checkpoint.

The frozen VLN components and configuration are copied verbatim from the
base WNM checkpoint. Only tensors below ``action_head.model`` are
replaced. DeepSpeed optimizer, RNG, scheduler, and trainer state are excluded
by default because they are not part of the loadable WNM model format.
No Hub lookup or dependency installation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor

_VLN_PREFIXES = (
    "action_head.model.base_model.model.",
    "action_head.model.",
)
_ACTOR_PREFIXES = (
    "transformer.",
    "_fsdp_wrapped_module.",
    "module.",
)
_TRAINING_STATE_PATTERNS = (
    "global_step*",
    "rng_state_*.pth",
    "latest",
    "scheduler.pt",
    "trainer_state.json",
    "wandb_config.json",
    "zero_to_fp32.py",
)
_ACTION_NORMALIZATION_FILENAME = "action_normalization.json"
_ACTION_NORMALIZATION_SCHEMA = "wnm_3d_action_normalization_v1"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _safe_weight_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError(f"Checkpoint index references a file outside {root}: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _safetensor_weight_map(path: Path) -> dict[str, Path]:
    """Map every safetensors key to its local shard."""

    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Only safetensors actor files are supported, got: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {key: path for key in handle.keys()}

    index_path = path / "model.safetensors.index.json"
    single_path = path / "model.safetensors"
    if index_path.is_file():
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Checkpoint index has no non-empty weight_map: {index_path}")
        return {str(key): _safe_weight_path(path, str(shard)) for key, shard in weight_map.items()}
    if single_path.is_file():
        with safe_open(single_path, framework="pt", device="cpu") as handle:
            return {key: single_path for key in handle.keys()}
    raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json under {path}")


def _normalize_vln_key(key: str) -> str | None:
    for prefix in _VLN_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :].replace(".base_layer.", ".")
    return None


def _normalize_actor_key(key: str) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in _ACTOR_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
                break
    return normalized.replace(".base_layer.", ".")


def _looks_like_fsdp_actor(path: Path) -> bool:
    return path.is_dir() and any(path.glob("model_world_size_*_rank_*.pt"))


def _merge_fsdp_actor(actor_checkpoint: Path, merged_dir: Path) -> Path:
    """Merge VERL FSDP1 DTensor shards without instantiating a HF AutoConfig.

    WNM's standalone joint DiT uses the custom ``model_type=ti2v``.
    VERL's generic model merger constructs ``AutoConfig`` before reading the
    shards, so it rejects this otherwise valid checkpoint. The saved model
    files already contain DTensor placement/global-shape metadata; reconstruct
    tensors directly from that metadata instead.
    """

    fsdp_config_path = actor_checkpoint / "fsdp_config.json"
    fsdp_config = _read_json(fsdp_config_path)
    world_size = int(fsdp_config.get("world_size", 0))
    if world_size <= 0:
        raise ValueError(f"Invalid FSDP world_size in {fsdp_config_path}: {world_size}")

    shard_paths = [actor_checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FSDP model shards: {missing}")

    rank_states = []
    for rank, shard_path in enumerate(shard_paths):
        print(f"Loading WNM FSDP actor shard {rank + 1}/{world_size}: {shard_path}", flush=True)
        state = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict in {shard_path}, got {type(state)!r}")
        rank_states.append(state)

    reference_keys = set(rank_states[0])
    for rank, state in enumerate(rank_states[1:], start=1):
        if set(state) != reference_keys:
            missing_keys = sorted(reference_keys - set(state))
            extra_keys = sorted(set(state) - reference_keys)
            raise KeyError(f"FSDP rank {rank} key mismatch: missing={missing_keys[:20]}, extra={extra_keys[:20]}")

    merged_state: dict[str, torch.Tensor] = {}
    for key_index, key in enumerate(sorted(reference_keys), start=1):
        values = [state.pop(key) for state in rank_states]
        first = values[0]
        if isinstance(first, DTensor):
            if not all(isinstance(value, DTensor) for value in values):
                raise TypeError(f"Mixed DTensor/plain shards for {key!r}")
            placements = tuple(first.placements)
            if len(placements) != 1:
                raise NotImplementedError(f"Only one-dimensional FSDP placements are supported: {key} {placements}")
            if any(tuple(value.placements) != placements for value in values[1:]):
                raise ValueError(f"Inconsistent DTensor placements across ranks for {key!r}")
            global_shape = tuple(first.shape)
            local_tensors = [value._local_tensor for value in values]
            placement = placements[0]
            if placement.is_replicate():
                merged = local_tensors[0]
            elif placement.is_shard():
                merged = torch.cat(local_tensors, dim=placement.dim)
                merged = merged[tuple(slice(0, size) for size in global_shape)]
            else:
                raise NotImplementedError(f"Unsupported DTensor placement for {key!r}: {placement}")
            if tuple(merged.shape) != global_shape:
                raise ValueError(
                    f"Merged shape mismatch for {key!r}: expected={global_shape}, actual={tuple(merged.shape)}"
                )
        else:
            if any(isinstance(value, DTensor) for value in values[1:]):
                raise TypeError(f"Mixed DTensor/plain shards for {key!r}")
            merged = torch.cat(values, dim=0)
        merged_state[key] = merged.contiguous()
        if key_index % 100 == 0 or key_index == len(reference_keys):
            print(f"Merged {key_index}/{len(reference_keys)} actor tensors", flush=True)

    merged_dir.mkdir(parents=True)
    merged_path = merged_dir / "model.safetensors"
    save_file(merged_state, merged_path, metadata={"format": "pt"})
    return merged_dir


@contextmanager
def _resolved_actor_weights(actor_checkpoint: Path) -> Iterator[Path]:
    """Yield actor weights, merging WNM FSDP shards when needed."""

    actor_checkpoint = actor_checkpoint.expanduser().resolve()
    candidates = (actor_checkpoint, actor_checkpoint / "huggingface", actor_checkpoint / "merged")
    for candidate in candidates:
        try:
            _safetensor_weight_map(candidate)
        except (FileNotFoundError, ValueError):
            continue
        yield candidate
        return

    if not _looks_like_fsdp_actor(actor_checkpoint):
        raise FileNotFoundError(f"No merged safetensors or standard verl FSDP shards found under {actor_checkpoint}.")

    with tempfile.TemporaryDirectory(prefix="wnm_3d_actor_merge_") as temp_dir:
        merged_dir = Path(temp_dir) / "merged"
        try:
            _merge_fsdp_actor(actor_checkpoint, merged_dir)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, NotImplementedError) as exc:
            raise RuntimeError(f"Failed to merge the WNM FSDP actor at {actor_checkpoint}") from exc
        _safetensor_weight_map(merged_dir)
        yield merged_dir


def _tensor_shape(shard: Path, key: str) -> tuple[int, ...]:
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return tuple(handle.get_slice(key).get_shape())


def _build_replacement_map(
    base_weight_map: dict[str, Path], actor_weight_map: dict[str, Path]
) -> dict[str, tuple[str, Path]]:
    """Return full-VLN key -> (actor key, actor shard), with strict coverage."""

    vln_normalized: dict[str, str] = {}
    for full_key in base_weight_map:
        normalized = _normalize_vln_key(full_key)
        if normalized is None:
            continue
        if normalized in vln_normalized:
            raise ValueError(
                f"Full VLN keys {vln_normalized[normalized]!r} and {full_key!r} both map to {normalized!r}."
            )
        vln_normalized[normalized] = full_key
    if not vln_normalized:
        raise KeyError("Full VLN checkpoint has no action_head.model tensors.")

    actor_normalized: dict[str, tuple[str, Path]] = {}
    for actor_key, shard in actor_weight_map.items():
        normalized = _normalize_actor_key(actor_key)
        if normalized in actor_normalized:
            raise ValueError(
                f"Actor keys {actor_normalized[normalized][0]!r} and {actor_key!r} both map to {normalized!r}."
            )
        actor_normalized[normalized] = (actor_key, shard)

    missing = sorted(set(vln_normalized) - set(actor_normalized))
    unexpected = sorted(set(actor_normalized) - set(vln_normalized))
    if missing or unexpected:
        raise KeyError(
            "Standalone actor and full VLN joint-DiT keys do not match: "
            f"missing_actor={missing[:20]} (total={len(missing)}), "
            f"unexpected_actor={unexpected[:20]} (total={len(unexpected)})."
        )

    replacements = {full_key: actor_normalized[normalized] for normalized, full_key in vln_normalized.items()}
    for full_key, (actor_key, actor_shard) in replacements.items():
        base_shape = _tensor_shape(base_weight_map[full_key], full_key)
        actor_shape = _tensor_shape(actor_shard, actor_key)
        if base_shape != actor_shape:
            raise ValueError(
                f"Tensor shape mismatch for {full_key!r} <- {actor_key!r}: base={base_shape}, actor={actor_shape}."
            )
    return replacements


def _rewrite_shards(
    output_dir: Path,
    base_dir: Path,
    base_weight_map: dict[str, Path],
    replacements: dict[str, tuple[str, Path]],
) -> None:
    replacements_by_shard: dict[Path, list[tuple[str, str, Path]]] = defaultdict(list)
    for full_key, (actor_key, actor_shard) in replacements.items():
        replacements_by_shard[base_weight_map[full_key]].append((full_key, actor_key, actor_shard))

    for source_shard, shard_replacements in replacements_by_shard.items():
        destination_shard = output_dir / source_shard.relative_to(base_dir)
        with safe_open(destination_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        for full_key, actor_key, actor_shard in shard_replacements:
            with safe_open(actor_shard, framework="pt", device="cpu") as handle:
                replacement = handle.get_tensor(actor_key)
            expected = state[full_key]
            if replacement.dtype != expected.dtype:
                replacement = replacement.to(dtype=expected.dtype)
            state[full_key] = replacement.contiguous()
        temporary = destination_shard.with_suffix(destination_shard.suffix + ".tmp")
        save_file(state, temporary, metadata=metadata)
        temporary.replace(destination_shard)


def _validate_export_values(exported_weight_map: dict[str, Path], replacements: dict[str, tuple[str, Path]]) -> None:
    for full_key, (actor_key, actor_shard) in replacements.items():
        with safe_open(exported_weight_map[full_key], framework="pt", device="cpu") as handle:
            exported = handle.get_tensor(full_key)
        with safe_open(actor_shard, framework="pt", device="cpu") as handle:
            expected = handle.get_tensor(actor_key).to(dtype=exported.dtype)
        if not torch.equal(exported, expected):
            raise RuntimeError(f"Exported tensor does not match the trained actor: {full_key!r} <- {actor_key!r}.")


def export_checkpoint(
    *,
    base_vln: Path,
    actor_checkpoint: Path,
    output_dir: Path,
    include_training_state: bool = False,
    dry_run: bool = False,
) -> dict:
    base_vln = base_vln.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not base_vln.is_dir():
        raise FileNotFoundError(f"Base VLN checkpoint is not a directory: {base_vln}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing export target: {output_dir}")

    action_normalization_path = base_vln / _ACTION_NORMALIZATION_FILENAME
    action_normalization = _read_json(action_normalization_path)
    if action_normalization.get("schema") != _ACTION_NORMALIZATION_SCHEMA:
        raise ValueError(
            f"Unsupported WNM action normalization schema in {action_normalization_path}: "
            f"{action_normalization.get('schema')!r}"
        )
    action_normalization_bytes = action_normalization_path.read_bytes()
    action_normalization_sha256 = hashlib.sha256(action_normalization_bytes).hexdigest()

    base_weight_map = _safetensor_weight_map(base_vln)
    with _resolved_actor_weights(actor_checkpoint) as actor_weights:
        actor_weight_map = _safetensor_weight_map(actor_weights)
        replacements = _build_replacement_map(base_weight_map, actor_weight_map)
        manifest = {
            "format": "wnm_full_vln_export_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "base_vln": str(base_vln),
            "actor_checkpoint": str(actor_checkpoint.expanduser().resolve()),
            "resolved_actor_weights": str(actor_weights),
            "replaced_tensor_count": len(replacements),
            "preserved_tensor_count": len(base_weight_map) - len(replacements),
            "action_normalization_sha256": action_normalization_sha256,
            "include_training_state": include_training_state,
            "dry_run": dry_run,
        }
        if dry_run:
            return manifest

        ignore = None if include_training_state else shutil.ignore_patterns(*_TRAINING_STATE_PATTERNS)
        shutil.copytree(
            base_vln,
            output_dir,
            copy_function=shutil.copy2,
            ignore=ignore,
        )
        try:
            _rewrite_shards(output_dir, base_vln, base_weight_map, replacements)
            exported_action_normalization = output_dir / _ACTION_NORMALIZATION_FILENAME
            if exported_action_normalization.read_bytes() != action_normalization_bytes:
                raise RuntimeError("Export changed the checkpoint action-normalization contract.")
            (output_dir / "wam_export_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            exported_map = _safetensor_weight_map(output_dir)
            exported_replacements = _build_replacement_map(exported_map, actor_weight_map)
            if set(exported_replacements) != set(replacements):
                raise RuntimeError("Exported VLN replacement-key validation failed.")
            _validate_export_values(exported_map, exported_replacements)
        except Exception:
            shutil.rmtree(output_dir)
            raise
        return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vln", type=Path, required=True, help="Complete local WNM VLN checkpoint")
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        required=True,
        help="verl actor directory, merged actor directory, or model.safetensors",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New full VLN checkpoint directory")
    parser.add_argument(
        "--include-training-state",
        action="store_true",
        help="Also copy DeepSpeed optimizer/RNG/trainer state from the base checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate keys/shapes without copying files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = export_checkpoint(
        base_vln=args.base_vln,
        actor_checkpoint=args.actor_checkpoint,
        output_dir=args.output_dir,
        include_training_state=args.include_training_state,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
