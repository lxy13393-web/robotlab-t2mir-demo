"""Collect one immutable formal T2MIR task into a staging directory.

The live checkpoint-bank registry grows while PPO specialists are trained.  A
formal rollout must never depend on that mutable file after collection starts,
so this wrapper freezes exactly one completed task entry and invokes the formal
collector against that task-local registry.  Re-running the same command is a
restart: the frozen task identity is checked and valid accepted shards are
reused by the underlying collector.
"""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

import torch

from pipeline.dataset.collect_official_mixed_formal import (
    accepted_shard_is_valid,
    checkpoint_path,
    rollout_checkpoints,
)
from pipeline.dataset.dataset_common import read_json, sha256_file, write_json_atomic
from pipeline.protocols.dynamics_profile import load_manifest


def task_bank_identity(task: dict) -> dict:
    """Return the immutable part of a registered specialist bank."""
    training = task.get("training", {})
    checkpoints = sorted(task.get("checkpoints", []), key=lambda row: int(row["iteration"]))
    return {
        "task_id": int(task["task_id"]),
        "tag": task.get("tag"),
        "split": task.get("split"),
        "profile": task.get("profile"),
        "training": {
            key: training.get(key)
            for key in ("method", "run", "seed", "iterations", "save_interval")
        },
        "checkpoints": [
            {
                "iteration": int(row["iteration"]),
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row.get("checkpoint_sha256"),
            }
            for row in checkpoints
        ],
    }


def frozen_registry(live_registry: dict, task: dict) -> dict:
    frozen = {key: value for key, value in live_registry.items() if key != "tasks"}
    frozen["snapshot_scope"] = "single_completed_task"
    frozen["tasks"] = [task]
    return frozen


def verify_registered_task(repo: Path, task: dict, expected_non_initial: int) -> None:
    if task.get("training", {}).get("method") != "fresh_task_specific_ppo":
        raise ValueError(f"task {task.get('task_id')} is not a fresh task-specific PPO")
    checkpoints = rollout_checkpoints(task, expected_non_initial)
    all_checkpoints = sorted(task.get("checkpoints", []), key=lambda row: int(row["iteration"]))
    if len(all_checkpoints) != expected_non_initial + 1 or int(all_checkpoints[0]["iteration"]) != 0:
        raise ValueError(
            f"task {task.get('task_id')}: expected model_0 plus {expected_non_initial} rollout checkpoints"
        )
    for row in checkpoints:
        path = checkpoint_path(repo, task, row)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if row.get("checkpoint_sha256") != digest:
            raise ValueError(f"registered checkpoint hash mismatch: {path}")


def resolve_repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def verify_complete_stage(
    *,
    repo: Path,
    task_dir: Path,
    frozen_path: Path,
    task_id: int,
    checkpoints: int,
    windows: int,
    horizon: int,
    collection_parameters: dict,
) -> bool:
    """Validate and recognize an immutable completed stage.

    Returns ``False`` only when no completed stage exists.  A manifest claiming
    completion but failing any check raises instead of being silently repaired.
    """
    manifest_path = task_dir / "collection_manifest.json"
    if not manifest_path.exists():
        return False
    collection = read_json(manifest_path)
    if collection.get("status") != "complete":
        return False
    expected = {
        "source_task_ids": [task_id],
        "checkpoints_per_task": checkpoints,
        "windows_per_checkpoint": windows,
        "window_steps": horizon,
        "collection_parameters": collection_parameters,
    }
    mismatches = [key for key, value in expected.items() if collection.get(key) != value]
    if mismatches:
        raise ValueError(
            f"task {task_id} completed stage has incompatible fields {mismatches}; use a new staging root"
        )
    snapshot_path = task_dir / "registry_snapshot.json"
    if (
        not snapshot_path.is_file()
        or sha256_file(snapshot_path) != collection.get("registry_snapshot_sha256")
        or sha256_file(snapshot_path) != sha256_file(frozen_path)
    ):
        raise ValueError(f"task {task_id} completed stage registry provenance is invalid")
    records = collection.get("shards", [])
    if len(records) != checkpoints:
        raise ValueError(f"task {task_id} completed stage has an invalid shard count")
    code_artifacts = collection.get("code_artifacts")
    seen = set()
    for record in records:
        iteration = int(record["checkpoint_iteration"])
        if (
            int(record["source_task_id"]) != task_id
            or record.get("status") not in {"complete", "reused"}
            or iteration in seen
        ):
            raise ValueError(f"task {task_id} completed stage has an invalid shard record")
        seen.add(iteration)
        checkpoint = resolve_repo_path(repo, record["checkpoint"])
        accepted = resolve_repo_path(repo, record["accepted_shard"])
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"task {task_id} checkpoint changed after collection: {checkpoint}")
        if sha256_file(accepted) != record.get("accepted_sha256") or not accepted_shard_is_valid(
            accepted,
            source_task_id=task_id,
            iteration=iteration,
            windows=windows,
            horizon=horizon,
            checkpoint_sha256=record["checkpoint_sha256"],
            registry_snapshot_sha256=collection["registry_snapshot_sha256"],
            collection_parameters=collection_parameters,
            code_artifacts=code_artifacts,
        ):
            raise ValueError(f"task {task_id} accepted shard is invalid: {accepted}")
        shard = torch.load(accepted, map_location="cpu")
        metadata = shard["metadata"]
        for paths_key, hashes_key in (("raw_files", "raw_sha256"), ("result_files", "result_sha256")):
            paths = metadata.get(paths_key, [])
            hashes = metadata.get(hashes_key, [])
            if len(paths) != len(hashes) or not paths:
                raise ValueError(f"task {task_id} malformed source artifact provenance")
            for value, expected_hash in zip(paths, hashes):
                if sha256_file(resolve_repo_path(repo, value)) != expected_hash:
                    raise ValueError(f"task {task_id} source artifact changed after collection")
    return True


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    namespace = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument("--registry", type=Path, default=namespace / "checkpoint_bank/pilot_registry.json")
    parser.add_argument("--staging-root", type=Path, default=namespace / "formal_staging_v1")
    parser.add_argument("--checkpoints-per-task", type=int, default=25)
    parser.add_argument("--windows-per-checkpoint", type=int, default=24)
    parser.add_argument("--window-steps", type=int, default=64)
    parser.add_argument("--attempt-batch", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260927)
    parser.add_argument("--goal-min-distance", type=float, default=1.0)
    parser.add_argument("--goal-max-distance", type=float, default=4.0)
    parser.add_argument("--goal-timeout", type=float, default=40.0)
    parser.add_argument("--goal-hold-time", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest, profiles = load_manifest(args.manifest)
    profiles_by_id = {profile.task_id: profile for profile in profiles}
    if args.task_id not in profiles_by_id:
        parser.error(f"unknown task ID {args.task_id}")
    if profiles_by_id[args.task_id].split != "train":
        parser.error(f"task {args.task_id} is held out and cannot be collected")

    live = read_json(args.registry)
    if live.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("checkpoint registry manifest hash does not match the dynamics manifest")
    matches = [row for row in live.get("tasks", []) if int(row["task_id"]) == args.task_id]
    if len(matches) != 1:
        raise ValueError(f"task {args.task_id} is not uniquely registered as complete")
    task = matches[0]
    verify_registered_task(repo, task, args.checkpoints_per_task)

    staging_root = args.staging_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    task_dir = staging_root / f"task{args.task_id:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    global_lock_path = staging_root / ".collector.lock"
    lock_path = task_dir / ".collection.lock"
    with global_lock_path.open("w") as global_lock_stream, lock_path.open("w") as lock_stream:
        try:
            fcntl.flock(global_lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another formal collector is active under {staging_root}: {global_lock_path}"
            ) from error
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"task {args.task_id} already has an active collector: {lock_path}") from error

        frozen_path = task_dir / "registry_source.json"
        candidate = frozen_registry(live, task)
        if frozen_path.exists():
            existing = read_json(frozen_path)
            existing_tasks = existing.get("tasks", [])
            if len(existing_tasks) != 1 or task_bank_identity(existing_tasks[0]) != task_bank_identity(task):
                raise ValueError(
                    f"task {args.task_id} checkpoint bank differs from its frozen staging registry; "
                    "use a new staging root rather than overwriting provenance"
                )
        else:
            write_json_atomic(frozen_path, candidate)

        collection_parameters = {
            "base_seed": args.base_seed,
            "attempt_batch": args.attempt_batch,
            "max_attempts": args.max_attempts,
            "goal_min_distance": args.goal_min_distance,
            "goal_max_distance": args.goal_max_distance,
            "goal_timeout": args.goal_timeout,
            "goal_hold_time": args.goal_hold_time,
        }
        if verify_complete_stage(
            repo=repo,
            task_dir=task_dir,
            frozen_path=frozen_path,
            task_id=args.task_id,
            checkpoints=args.checkpoints_per_task,
            windows=args.windows_per_checkpoint,
            horizon=args.window_steps,
            collection_parameters=collection_parameters,
        ):
            print(
                f"[FORMAL-STAGING] ALREADY_COMPLETE task={args.task_id} directory={task_dir}",
                flush=True,
            )
            return

        collector = repo / "pipeline/dataset/collect_official_mixed_formal.py"
        command = [
            sys.executable,
            "-u",
            str(collector),
            "--manifest",
            str(args.manifest.resolve()),
            "--registry",
            str(frozen_path),
            "--output-dir",
            str(task_dir),
            "--source-task-ids",
            str(args.task_id),
            "--checkpoints-per-task",
            str(args.checkpoints_per_task),
            "--windows-per-checkpoint",
            str(args.windows_per_checkpoint),
            "--window-steps",
            str(args.window_steps),
            "--attempt-batch",
            str(args.attempt_batch),
            "--max-attempts",
            str(args.max_attempts),
            "--base-seed",
            str(args.base_seed),
            "--goal-min-distance",
            str(args.goal_min_distance),
            "--goal-max-distance",
            str(args.goal_max_distance),
            "--goal-timeout",
            str(args.goal_timeout),
            "--goal-hold-time",
            str(args.goal_hold_time),
        ]
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)
        print(
            f"[FORMAL-STAGING] task={args.task_id} registry_sha256={sha256_file(frozen_path)} "
            f"directory={task_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
