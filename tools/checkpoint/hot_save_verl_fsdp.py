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

"""Queue one native FSDP actor snapshot into a live VERL-Omni training job."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

WORKER_API_CANDIDATES = (
    (
        "actor_rollout",
        "WorkerDict.actor_rollout_update_actor",
        "actor_rollout_save_checkpoint",
    ),
    (
        "actor_rollout_ref",
        "WorkerDict.actor_rollout_ref_update_actor",
        "actor_rollout_ref_save_checkpoint",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or create a one-off native FSDP actor/optimizer snapshot "
            "without terminating a live VERL-Omni trainer."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Exact trainer.default_local_dir")
    parser.add_argument(
        "--snapshot-root",
        default=None,
        help=("Existing directory that receives manual_snapshot_step_N. Defaults to RUN_DIR."),
    )
    parser.add_argument("--execute", action="store_true", help="Actually queue the snapshot")
    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help=(
            "Save exactly this actor update. The script waits for all ranks to "
            "run update_actor while rollout servers still report TARGET_STEP-1 "
            "and fails if that point is missed."
        ),
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--namespace", default=None)
    parser.add_argument(
        "--server-name-template",
        default="vllm_omni_server_{rank}_0",
        help="Python format string containing {rank}",
    )
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument("--save-timeout", type=float, default=900.0)
    parser.add_argument("--resume-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--snapshot-prefix", default="manual_snapshot_step")
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=64.0,
        help="Minimum free space required on the snapshot filesystem",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def find_live_trainer_pids(run_dir: Path) -> list[int]:
    needle = f"trainer.default_local_dir={run_dir}"
    matches: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        fail("/proc is unavailable; cannot bind the run directory to a live trainer")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "verl_omni.trainer.main_diffusion" in command and needle in command:
            matches.append(int(entry.name))
    return sorted(matches)


def choose_namespace(actor_rows, expected_names: set[str], requested: str | None) -> str:
    candidates: dict[str, set[str]] = {}
    for actor in actor_rows:
        if actor.state != "ALIVE" or actor.class_name != "vLLMOmniHttpServer":
            continue
        candidates.setdefault(actor.ray_namespace, set()).add(actor.name)
    if requested is not None:
        if not expected_names.issubset(candidates.get(requested, set())):
            fail(
                f"namespace {requested!r} does not contain all expected rollout "
                f"servers: missing={sorted(expected_names - candidates.get(requested, set()))}"
            )
        return requested
    complete = [namespace for namespace, names in candidates.items() if expected_names.issubset(names)]
    if len(complete) != 1:
        fail(
            "expected exactly one Ray namespace containing the rollout server "
            f"set, found {complete}; pass --namespace only after resolving the intended job"
        )
    return complete[0]


def inspect_server(server):
    return server.replica_rank, server.global_steps, server.workers[0]


def read_server_step(server):
    return server.global_steps


def choose_worker_api(workers) -> tuple[str, str, str]:
    supported = []
    for role_prefix, update_actor_task, save_method in WORKER_API_CANDIDATES:
        if all(save_method in worker._ray_method_signatures for worker in workers):
            supported.append((role_prefix, update_actor_task, save_method))
    if len(supported) != 1:
        details = {
            role_prefix: [
                worker._ray_actor_id.hex() for worker in workers if save_method not in worker._ray_method_signatures
            ]
            for role_prefix, _, save_method in WORKER_API_CANDIDATES
        }
        fail(
            "expected exactly one consistently registered actor worker API, "
            f"found={[item[0] for item in supported]}, missing_by_api={details}"
        )
    return supported[0]


def running_update_actor_ids(list_tasks, update_actor_task: str) -> set[str]:
    tasks = list_tasks(
        filters=[
            ("func_or_class_name", "=", update_actor_task),
            ("state", "=", "RUNNING"),
        ],
        limit=1000,
        detail=True,
        raise_on_missing_output=False,
    )
    return {task.actor_id for task in tasks}


def count_shards(actor_dir: Path, pattern: str) -> int:
    return sum(1 for path in actor_dir.rglob(pattern) if path.is_file())


def total_file_bytes(root: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for path in root.rglob("*"):
        if path.is_file():
            count += 1
            size += path.stat().st_size
    return count, size


def process_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def main() -> int:
    args = parse_args()
    if args.world_size <= 0:
        fail("--world-size must be positive")
    if args.target_step is not None and args.target_step <= 0:
        fail("--target-step must be positive")
    if min(args.wait_timeout, args.save_timeout, args.resume_timeout) <= 0:
        fail("timeouts must be positive")
    if args.poll_interval <= 0:
        fail("--poll-interval must be positive")
    if args.min_free_gib < 0:
        fail("--min-free-gib must be non-negative")
    if "{rank}" not in args.server_name_template:
        fail("--server-name-template must contain {rank}")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        fail(f"run directory does not exist: {run_dir}")
    snapshot_root = Path(args.snapshot_root).expanduser().resolve() if args.snapshot_root is not None else run_dir
    if not snapshot_root.is_dir():
        fail(f"snapshot root does not exist: {snapshot_root}")
    free_bytes = shutil.disk_usage(snapshot_root).free
    required_bytes = int(args.min_free_gib * 1024**3)
    if free_bytes < required_bytes:
        fail(f"insufficient free space under {snapshot_root}: free={free_bytes}, required={required_bytes}")
    trainer_pids = find_live_trainer_pids(run_dir)
    if len(trainer_pids) != 1:
        fail(f"expected exactly one live trainer for {run_dir}, found PIDs {trainer_pids}")
    trainer_pid = trainer_pids[0]

    try:
        import ray
        from ray.util.state import list_actors, list_tasks
    except ImportError as exc:
        fail(f"Ray is required in the remote training environment: {exc}")

    ray.init(address=args.ray_address)
    expected_names = {args.server_name_template.format(rank=rank) for rank in range(args.world_size)}
    actor_rows = list_actors(limit=10000)
    namespace = choose_namespace(actor_rows, expected_names, args.namespace)
    servers = [
        ray.get_actor(
            args.server_name_template.format(rank=rank),
            namespace=namespace,
        )
        for rank in range(args.world_size)
    ]
    infos = ray.get(
        [server.__ray_call__.options(concurrency_group="_ray_system").remote(inspect_server) for server in servers]
    )
    infos.sort(key=lambda item: item[0])
    ranks = [int(item[0]) for item in infos]
    if ranks != list(range(args.world_size)):
        fail(f"rollout replica ranks disagree with world size: {ranks}")
    server_steps = [int(item[1]) for item in infos]
    workers = [item[2] for item in infos]
    worker_ids = {worker._ray_actor_id.hex() for worker in workers}
    if len(worker_ids) != args.world_size:
        fail(f"expected {args.world_size} unique FSDP workers, found {len(worker_ids)}")
    worker_role_prefix, update_actor_task, save_method = choose_worker_api(workers)

    active_ids = running_update_actor_ids(list_tasks, update_actor_task) & worker_ids
    preflight = {
        "mode": "execute" if args.execute else "preflight",
        "run_dir": str(run_dir),
        "snapshot_root": str(snapshot_root),
        "snapshot_free_bytes": free_bytes,
        "snapshot_min_free_bytes": required_bytes,
        "trainer_pid": trainer_pid,
        "ray_namespace": namespace,
        "world_size": args.world_size,
        "worker_ids": sorted(worker_ids),
        "worker_role_prefix": worker_role_prefix,
        "update_actor_task": update_actor_task,
        "save_method": save_method,
        "server_steps": server_steps,
        "update_actor_active_ranks": len(active_ids),
        "target_step": args.target_step,
    }
    print("PREFLIGHT " + json.dumps(preflight, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    deadline = time.monotonic() + args.wait_timeout
    last_report = 0.0
    synchronized_steps = server_steps
    while True:
        now = time.monotonic()
        if now >= deadline:
            fail(
                "safe-phase timeout: "
                f"server_steps={synchronized_steps}, "
                f"update_actor={len(active_ids)}/{args.world_size}, "
                f"target_step={args.target_step}"
            )
        if not process_alive(trainer_pid):
            fail(f"trainer PID {trainer_pid} exited while waiting for the save point")
        synchronized_steps = [
            int(step)
            for step in ray.get(
                [
                    server.__ray_call__.options(concurrency_group="_ray_system").remote(read_server_step)
                    for server in servers
                ]
            )
        ]
        minimum_server_step = min(synchronized_steps)
        maximum_server_step = max(synchronized_steps)
        if args.target_step is not None and maximum_server_step > args.target_step - 1:
            fail(
                f"target step {args.target_step} was missed: rollout servers already report steps {synchronized_steps}"
            )
        steps_synchronized = minimum_server_step == maximum_server_step
        server_step = minimum_server_step
        active_ids = running_update_actor_ids(list_tasks, update_actor_task) & worker_ids
        step_ready = args.target_step is None or server_step == args.target_step - 1
        if steps_synchronized and step_ready and len(active_ids) == args.world_size:
            break
        if now - last_report >= 10.0:
            print(
                "WAITING_FOR_SAFE_PHASE "
                f"server_steps={synchronized_steps} "
                f"target_step={args.target_step} "
                f"active_ranks={len(active_ids)}/{args.world_size}",
                flush=True,
            )
            last_report = now
        time.sleep(args.poll_interval)

    snapshot_step = synchronized_steps[0] + 1
    if args.target_step is not None and snapshot_step != args.target_step:
        fail(f"internal target-step mismatch: requested={args.target_step}, resolved={snapshot_step}")
    snapshot_dir = snapshot_root / f"{args.snapshot_prefix}_{snapshot_step}"
    actor_dir = snapshot_dir / "actor"
    if snapshot_dir.exists():
        fail(f"snapshot destination already exists: {snapshot_dir}")

    print(
        f"QUEUE_NATIVE_FSDP_SAVE snapshot_step={snapshot_step} path={actor_dir}",
        flush=True,
    )
    save_refs = [
        getattr(worker, save_method).remote(
            str(actor_dir),
            None,
            snapshot_step,
            None,
        )
        for worker in workers
    ]
    ray.get(save_refs, timeout=args.save_timeout)

    model_shards = count_shards(actor_dir, "model_world_size_*_rank_*.pt")
    optimizer_shards = count_shards(actor_dir, "optim_world_size_*_rank_*.pt")
    extra_state_shards = count_shards(actor_dir, "extra_state_world_size_*_rank_*.pt")
    if model_shards != args.world_size or optimizer_shards != args.world_size:
        fail(
            "snapshot shard verification failed: "
            f"model={model_shards}/{args.world_size}, "
            f"optimizer={optimizer_shards}/{args.world_size}, "
            f"extra_state={extra_state_shards}/{args.world_size}"
        )
    file_count, byte_count = total_file_bytes(actor_dir)
    if byte_count <= 0:
        fail(f"snapshot contains no data: {actor_dir}")

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "kind": "manual_live_actor_snapshot",
        "snapshot_step": snapshot_step,
        "requested_target_step": args.target_step,
        "rollout_server_step_before_actor_update": synchronized_steps[0],
        "actor_world_size": args.world_size,
        "actor_path": str(actor_dir),
        "snapshot_root": str(snapshot_root),
        "contains_dataloader_state": False,
        "full_trainer_resume": False,
        "model_shards": model_shards,
        "optimizer_shards": optimizer_shards,
        "extra_state_shards": extra_state_shards,
        "file_count": file_count,
        "byte_count": byte_count,
        "trainer_pid": trainer_pid,
        "ray_namespace": namespace,
        "note": ("Queued on every FSDP rank during update_actor and saved by the native checkpoint manager."),
    }
    manifest_path = snapshot_dir / "manual_snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resume_deadline = time.monotonic() + args.resume_timeout
    resumed_steps: list[int] = []
    while time.monotonic() < resume_deadline:
        if not process_alive(trainer_pid):
            fail(f"trainer PID {trainer_pid} exited after snapshot")
        resumed_steps = [
            int(step)
            for step in ray.get(
                [
                    server.__ray_call__.options(concurrency_group="_ray_system").remote(read_server_step)
                    for server in servers
                ]
            )
        ]
        if min(resumed_steps) >= snapshot_step:
            break
        time.sleep(args.poll_interval)
    else:
        fail(f"snapshot is complete but rollout servers did not reach step {snapshot_step}: {resumed_steps}")

    result = {
        **manifest,
        "manifest_path": str(manifest_path),
        "rollout_server_steps_after_save": resumed_steps,
        "trainer_alive": True,
    }
    print("SAVE_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
