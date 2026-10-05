"""Merge immutable per-task formal staging collections into one dataset manifest.

No trajectory tensor is rewritten.  The merged manifest points at the already
hashed accepted shards and records both the task-local registry snapshot and
task-local collection-manifest hash.  This makes the final merge cheap,
restartable, and auditable after PPO training and collection were overlapped.
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
import json
from collections import Counter
from pathlib import Path

import torch

from pipeline.dataset.collect_official_mixed_formal import accepted_shard_is_valid
from pipeline.dataset.dataset_common import parse_csv_ints, read_json, sha256_file, utc_now, write_json_atomic
from pipeline.dataset.collect_official_mixed_staging import task_bank_identity
from pipeline.protocols.dynamics_profile import load_manifest


SHARED_KEYS = (
    "format_version",
    "dataset",
    "manifest_sha256",
    "held_out_task_ids",
    "checkpoints_per_task",
    "windows_per_checkpoint",
    "window_steps",
    "context_unit",
    "next_state_semantics",
    "policy_action_mode",
    "collection_parameters",
    "code_artifacts",
)


def resolve_repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def validate_stage(repo: Path, task_id: int, task_dir: Path) -> tuple[dict, dict, Path, Path]:
    collection_path = task_dir / "collection_manifest.json"
    snapshot_path = task_dir / "registry_snapshot.json"
    if not collection_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError(f"task {task_id}: incomplete staging metadata in {task_dir}")
    collection = read_json(collection_path)
    snapshot = read_json(snapshot_path)
    if collection.get("status") != "complete":
        raise ValueError(f"task {task_id}: staging collection status={collection.get('status')!r}")
    if list(map(int, collection.get("source_task_ids", []))) != [task_id]:
        raise ValueError(f"task {task_id}: staging manifest is not scoped to exactly this task")
    if sha256_file(snapshot_path) != collection.get("registry_snapshot_sha256"):
        raise ValueError(f"task {task_id}: staging registry snapshot hash mismatch")
    task_rows = [row for row in snapshot.get("tasks", []) if int(row["task_id"]) == task_id]
    if len(task_rows) != 1:
        raise ValueError(f"task {task_id}: task entry missing or duplicated in staging registry")

    checkpoints = int(collection["checkpoints_per_task"])
    windows = int(collection["windows_per_checkpoint"])
    horizon = int(collection["window_steps"])
    records = collection.get("shards", [])
    if len(records) != checkpoints:
        raise ValueError(f"task {task_id}: expected {checkpoints} shards, found {len(records)}")
    for record in records:
        if int(record["source_task_id"]) != task_id:
            raise ValueError(f"task {task_id}: foreign shard record in staging manifest")
        if record.get("status") not in {"complete", "reused"}:
            raise ValueError(f"task {task_id}: incomplete shard record")
        accepted = resolve_repo_path(repo, record["accepted_shard"])
        if sha256_file(accepted) != record.get("accepted_sha256"):
            raise ValueError(f"task {task_id}: accepted shard hash mismatch: {accepted}")
        if not accepted_shard_is_valid(
            accepted,
            source_task_id=task_id,
            iteration=int(record["checkpoint_iteration"]),
            windows=windows,
            horizon=horizon,
            checkpoint_sha256=record["checkpoint_sha256"],
        ):
            raise ValueError(f"task {task_id}: invalid accepted shard: {accepted}")
        shard = torch.load(accepted, map_location="cpu")
        if shard["metadata"].get("registry_snapshot_sha256") != collection["registry_snapshot_sha256"]:
            raise ValueError(f"task {task_id}: shard does not reference its staging registry snapshot")
        metadata = shard["metadata"]
        if collection.get("collection_parameters") is not None and metadata.get(
            "collection_parameters"
        ) != collection.get("collection_parameters"):
            raise ValueError(f"task {task_id}: shard collection parameters differ from manifest")
        if collection.get("code_artifacts") is not None and metadata.get("code_artifacts") != collection.get(
            "code_artifacts"
        ):
            raise ValueError(f"task {task_id}: shard code artifacts differ from manifest")
        for paths_key, hashes_key in (("raw_files", "raw_sha256"), ("result_files", "result_sha256")):
            paths = metadata.get(paths_key, [])
            hashes = metadata.get(hashes_key, [])
            if len(paths) != len(hashes) or not paths:
                raise ValueError(f"task {task_id}: malformed {paths_key}/{hashes_key} provenance")
            for value, expected_hash in zip(paths, hashes):
                path = resolve_repo_path(repo, value)
                if sha256_file(path) != expected_hash:
                    raise ValueError(f"task {task_id}: source artifact hash mismatch: {path}")

    frozen_identity = task_bank_identity(task_rows[0])
    frozen_checkpoints = {
        int(row["iteration"]): row for row in frozen_identity["checkpoints"] if int(row["iteration"]) > 0
    }
    ranks = sorted(int(row["checkpoint_rank"]) for row in records)
    if ranks != list(range(checkpoints)):
        raise ValueError(f"task {task_id}: checkpoint ranks are incomplete or duplicated")
    seen_iterations = set()
    for record in records:
        iteration = int(record["checkpoint_iteration"])
        if iteration in seen_iterations or iteration not in frozen_checkpoints:
            raise ValueError(f"task {task_id}: checkpoint iteration mismatch or duplicate: {iteration}")
        seen_iterations.add(iteration)
        frozen_checkpoint = frozen_checkpoints[iteration]
        checkpoint = resolve_repo_path(repo, record["checkpoint"])
        if (
            checkpoint.name != frozen_checkpoint["checkpoint"]
            or record["checkpoint_sha256"] != frozen_checkpoint["checkpoint_sha256"]
            or sha256_file(checkpoint) != record["checkpoint_sha256"]
        ):
            raise ValueError(f"task {task_id}: checkpoint bank provenance mismatch: {checkpoint}")
    return collection, task_rows[0], collection_path, snapshot_path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    namespace = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument("--staging-root", type=Path, default=namespace / "formal_staging_v1")
    parser.add_argument("--output-dir", type=Path, default=namespace / "formal_v1")
    parser.add_argument(
        "--source-task-ids",
        type=parse_csv_ints,
        default=None,
        help="Default: all 42 training tasks. Subsets require --allow-subset.",
    )
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest, profiles = load_manifest(args.manifest)
    profiles_by_id = {profile.task_id: profile for profile in profiles}
    train_ids = [profile.task_id for profile in profiles if profile.split == "train"]
    held_out_ids = [profile.task_id for profile in profiles if profile.split != "train"]
    source_ids = sorted(args.source_task_ids or train_ids)
    if sorted(source_ids) != sorted(train_ids) and not args.allow_subset:
        parser.error("a partial merge requires --allow-subset; the final dataset must contain all train tasks")
    unknown = sorted(set(source_ids).difference(train_ids))
    if unknown:
        parser.error(f"unknown or held-out source task IDs: {unknown}")

    output_dir = args.output_dir.resolve()
    collection_path = output_dir / "collection_manifest.json"
    snapshot_path = output_dir / "registry_snapshot.json"
    if (output_dir / "dpt").exists():
        raise FileExistsError(
            f"refusing to rebuild merged metadata after DPT artifacts exist: {output_dir}; "
            "use a new output directory"
        )
    if (collection_path.exists() or snapshot_path.exists()) and not args.force:
        raise FileExistsError(
            f"merged output already exists: {output_dir}; use a new directory or --force to rebuild metadata"
        )

    stages = []
    frozen_tasks = []
    merged_records = []
    shared = None
    aggregate_modes: Counter = Counter()
    for task_id in source_ids:
        task_dir = (args.staging_root / f"task{task_id:02d}").resolve()
        collection, frozen_task, stage_manifest, stage_snapshot = validate_stage(
            repo, task_id, task_dir
        )
        values = {key: collection.get(key) for key in SHARED_KEYS}
        if shared is None:
            shared = values
        elif values != shared:
            differing = [key for key in SHARED_KEYS if values[key] != shared[key]]
            raise ValueError(f"task {task_id}: incompatible staging collection fields {differing}")
        if collection["manifest_sha256"] != manifest["sha256"]:
            raise ValueError(f"task {task_id}: dynamics manifest hash mismatch")
        if frozen_task.get("profile") != profiles_by_id[task_id].__dict__:
            raise ValueError(f"task {task_id}: frozen PPO profile differs from dynamics manifest")
        for record in collection["shards"]:
            if record.get("profile") != profiles_by_id[task_id].__dict__:
                raise ValueError(f"task {task_id}: collection profile differs from dynamics manifest")

        source_snapshot_hash = sha256_file(stage_snapshot)
        source_manifest_hash = sha256_file(stage_manifest)
        for record in collection["shards"]:
            copied = dict(record)
            copied["accepted_shard"] = str(resolve_repo_path(repo, record["accepted_shard"]))
            copied["source_collection_manifest"] = str(stage_manifest)
            copied["source_collection_manifest_sha256"] = source_manifest_hash
            copied["source_registry_snapshot"] = str(stage_snapshot)
            copied["source_registry_snapshot_sha256"] = source_snapshot_hash
            merged_records.append(copied)
            aggregate_modes.update(
                {int(key): int(value) for key, value in record["window_start_mode_counts"].items()}
            )
        stages.append(
            {
                "source_task_id": task_id,
                "directory": str(task_dir),
                "collection_manifest": str(stage_manifest),
                "collection_manifest_sha256": source_manifest_hash,
                "registry_snapshot": str(stage_snapshot),
                "registry_snapshot_sha256": source_snapshot_hash,
            }
        )
        frozen_tasks.append(frozen_task)

    identities = [task_bank_identity(task) for task in frozen_tasks]
    if len({row["task_id"] for row in identities}) != len(identities):
        raise ValueError("duplicate task checkpoint banks in staging inputs")
    base_snapshot = read_json(Path(stages[0]["registry_snapshot"]))
    merged_snapshot = {key: value for key, value in base_snapshot.items() if key not in {"tasks", "snapshot_scope"}}
    merged_snapshot["snapshot_scope"] = "merged_completed_tasks"
    merged_snapshot["tasks"] = sorted(frozen_tasks, key=lambda row: int(row["task_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(snapshot_path, merged_snapshot)

    assert shared is not None
    merged = {
        **shared,
        "status": "complete",
        "created_at": utc_now(),
        "completed_at": utc_now(),
        "manifest": str(args.manifest.resolve()),
        "registry_source": "merged immutable per-task staging snapshots",
        "registry_snapshot": str(snapshot_path),
        "registry_snapshot_sha256": sha256_file(snapshot_path),
        "source_task_ids": source_ids,
        "held_out_task_ids": held_out_ids,
        "collection_mode": "merged_per_task_staging",
        "full_training_set": source_ids == sorted(train_ids),
        "scope": "full" if source_ids == sorted(train_ids) else "subset_canary",
        "staging_collections": stages,
        "shards": sorted(
            merged_records,
            key=lambda row: (int(row["source_task_id"]), int(row["checkpoint_iteration"])),
        ),
        "window_start_mode_counts": {
            str(mode): aggregate_modes.get(mode, 0) for mode in range(4)
        },
    }
    expected = len(source_ids) * int(merged["checkpoints_per_task"])
    if len(merged["shards"]) != expected:
        raise ValueError(f"merged shard count={len(merged['shards'])}, expected={expected}")
    write_json_atomic(collection_path, merged)
    print(
        f"[FORMAL-MERGE] PASS tasks={len(source_ids)} shards={len(merged_records)} "
        f"manifest={collection_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
