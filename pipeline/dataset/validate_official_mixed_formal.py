"""Strict semantic and loader validation for formal RobotLab T2MIR Mixed data."""

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
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pipeline.dataset.collect_official_mixed_formal import accepted_shard_is_valid
from pipeline.dataset.dataset_common import ACTION_DIM, STATE_DIM, parse_csv_ints, sha256_file
from pipeline.dataset.prepare_official_mixed_pilot import load_actor


PROMPT_KEYS = ("states", "actions", "next_states", "rewards", "dones", "masks")
QUERY_KEYS = ("states", "actions")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def exact_keys(payload: dict, expected: tuple[str, ...], path: Path) -> None:
    if tuple(payload.keys()) != expected:
        raise ValueError(f"{path}: keys={tuple(payload.keys())}, expected={expected}")


def validate_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected numpy.ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"{name}: dtype={value.dtype}, expected float32")
    if value.shape != shape:
        raise ValueError(f"{name}: shape={value.shape}, expected={shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: contains NaN or Inf")


def task_map(rows: list[dict], key: str = "source_task_id") -> dict[int, dict]:
    result = {int(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key} in provenance")
    return result


def resolve_repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def reproduce_query_actions(states: np.ndarray, provenance: dict) -> np.ndarray:
    checkpoint = Path(provenance["teacher_checkpoint"])
    if sha256_file(checkpoint) != provenance["teacher_checkpoint_sha256"]:
        raise ValueError(f"teacher checkpoint hash mismatch: {checkpoint}")
    actor, std, _ = load_actor(checkpoint)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(provenance["query_seed"]))
    batch_size = int(provenance["query_batch_size"])
    parts = []
    with torch.inference_mode():
        for start in range(0, states.shape[0], batch_size):
            batch = torch.from_numpy(states[start : start + batch_size])
            mean = actor(batch)
            noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
            parts.append((mean + noise * std).numpy())
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_dir = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1/formal_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=default_dir)
    parser.add_argument("--required-controller-modes", type=parse_csv_ints, default=[0, 1, 2, 3])
    args = parser.parse_args()

    collection_path = args.formal_dir / "collection_manifest.json"
    snapshot_path = args.formal_dir / "registry_snapshot.json"
    dpt_dir = args.formal_dir / "dpt"
    collection = json.loads(collection_path.read_text())
    prompt_provenance = json.loads((dpt_dir / "prompt_provenance.json").read_text())
    teacher_registry = json.loads((dpt_dir / "teacher_registry.json").read_text())
    query_provenance = json.loads((dpt_dir / "query_provenance.json").read_text())

    if collection.get("status") != "complete" or teacher_registry.get("status") != "complete":
        raise ValueError("collection and teacher evaluation must be complete")
    if collection.get("context_unit") != "continuous_fixed_length_window":
        raise ValueError("unexpected context unit")
    if collection.get("next_state_semantics") != "next_policy_decision_observation":
        raise ValueError("unexpected next-state semantics")
    if sha256_file(snapshot_path) != collection["registry_snapshot_sha256"]:
        raise ValueError("registry snapshot hash mismatch")
    snapshot = json.loads(snapshot_path.read_text())
    snapshot_task_ids = sorted(int(row["task_id"]) for row in snapshot.get("tasks", []))
    if prompt_provenance["collection_manifest_sha256"] != sha256_file(collection_path):
        raise ValueError("prompt provenance does not match collection manifest")
    if query_provenance["teacher_registry_sha256"] != sha256_file(dpt_dir / "teacher_registry.json"):
        raise ValueError("query provenance does not match teacher registry")

    source_ids = sorted(map(int, collection["source_task_ids"]))
    held_out_ids = sorted(map(int, collection["held_out_task_ids"]))
    if collection.get("collection_mode") == "merged_per_task_staging" and snapshot_task_ids != source_ids:
        raise ValueError(
            f"registry snapshot tasks {snapshot_task_ids} do not match source tasks {source_ids}"
        )
    if not set(source_ids).issubset(snapshot_task_ids):
        raise ValueError("registry snapshot does not contain every collected source task")
    if set(snapshot_task_ids).intersection(held_out_ids):
        raise ValueError("held-out tasks must not appear in the training registry snapshot")
    snapshot_by_task = {int(row["task_id"]): row for row in snapshot["tasks"]}
    if set(source_ids).intersection(held_out_ids):
        raise ValueError("training and held-out task IDs overlap")
    prompt_by_task = task_map(prompt_provenance["tasks"])
    query_by_task = task_map(query_provenance["tasks"])
    teacher_by_task = task_map(teacher_registry["tasks"])
    if (
        sorted(prompt_by_task) != source_ids
        or sorted(query_by_task) != source_ids
        or sorted(teacher_by_task) != source_ids
    ):
        raise ValueError("prompt/query/teacher provenance does not exactly cover the source tasks")
    for task_id in held_out_ids:
        for prefix in ("dataset_task_", "query_dataset_task_"):
            if (dpt_dir / f"{prefix}{task_id}.pkl").exists():
                raise ValueError(f"held-out task file must not exist: {prefix}{task_id}.pkl")

    checkpoints = int(collection["checkpoints_per_task"])
    windows = int(collection["windows_per_checkpoint"])
    horizon = int(collection["window_steps"])
    rows_per_checkpoint = windows * horizon
    expected_rows = checkpoints * rows_per_checkpoint
    collection_by_task: dict[int, list[dict]] = {}
    for row in collection["shards"]:
        collection_by_task.setdefault(int(row["source_task_id"]), []).append(row)

    prompt_task_arrays = {key: [] for key in PROMPT_KEYS}
    query_task_arrays = {key: [] for key in QUERY_KEYS}
    task_reports = []
    for task_id in source_ids:
        prompt_path = dpt_dir / f"dataset_task_{task_id}.pkl"
        query_path = dpt_dir / f"query_dataset_task_{task_id}.pkl"
        if sha256_file(prompt_path) != prompt_by_task[task_id]["prompt_sha256"]:
            raise ValueError(f"task {task_id}: prompt hash mismatch")
        if sha256_file(query_path) != query_by_task[task_id]["query_sha256"]:
            raise ValueError(f"task {task_id}: query hash mismatch")
        prompt, query = load_pickle(prompt_path), load_pickle(query_path)
        exact_keys(prompt, PROMPT_KEYS, prompt_path)
        exact_keys(query, QUERY_KEYS, query_path)
        validate_array(f"task {task_id} prompt.states", prompt["states"], (expected_rows, STATE_DIM))
        validate_array(f"task {task_id} prompt.actions", prompt["actions"], (expected_rows, ACTION_DIM))
        validate_array(f"task {task_id} prompt.next_states", prompt["next_states"], (expected_rows, STATE_DIM))
        for key in ("rewards", "dones", "masks"):
            validate_array(f"task {task_id} prompt.{key}", prompt[key], (expected_rows,))
        validate_array(f"task {task_id} query.states", query["states"], (expected_rows, STATE_DIM))
        validate_array(f"task {task_id} query.actions", query["actions"], (expected_rows, ACTION_DIM))
        if not np.array_equal(prompt["states"], query["states"]):
            raise ValueError(f"task {task_id}: query states are not exact same-state relabels")
        if np.any(prompt["dones"] != 0.0) or np.any(prompt["masks"] != 1.0):
            raise ValueError(f"task {task_id}: retained windows contain a termination")
        for start in range(0, expected_rows, horizon):
            stop = start + horizon
            if not np.array_equal(prompt["next_states"][start : stop - 1], prompt["states"][start + 1 : stop]):
                raise ValueError(f"task {task_id}: decision-state continuity failed at row {start}")

        records = sorted(collection_by_task.get(task_id, []), key=lambda row: int(row["checkpoint_iteration"]))
        if len(records) != checkpoints:
            raise ValueError(f"task {task_id}: expected {checkpoints} checkpoint records")
        iterations = [int(row["checkpoint_iteration"]) for row in records]
        if len(iterations) != len(set(iterations)):
            raise ValueError(f"task {task_id}: duplicate checkpoint iteration records")
        registered = {
            int(row["iteration"]): row
            for row in snapshot_by_task[task_id].get("checkpoints", [])
            if int(row["iteration"]) > 0
        }
        task_modes: Counter = Counter()
        offset = 0
        for record in records:
            shard_path = Path(record["accepted_shard"])
            if sha256_file(shard_path) != record["accepted_sha256"]:
                raise ValueError(f"task {task_id}: accepted shard hash mismatch")
            if not accepted_shard_is_valid(
                shard_path,
                source_task_id=task_id,
                iteration=int(record["checkpoint_iteration"]),
                windows=windows,
                horizon=horizon,
                checkpoint_sha256=record["checkpoint_sha256"],
            ):
                raise ValueError(f"task {task_id}: invalid accepted shard {shard_path}")
            shard = torch.load(shard_path, map_location="cpu")
            iteration = int(record["checkpoint_iteration"])
            registered_checkpoint = registered.get(iteration)
            if registered_checkpoint is None or (
                registered_checkpoint.get("checkpoint") != Path(record["checkpoint"]).name
                or registered_checkpoint.get("checkpoint_sha256") != record["checkpoint_sha256"]
                or sha256_file(Path(record["checkpoint"])) != record["checkpoint_sha256"]
            ):
                raise ValueError(f"task {task_id}: checkpoint differs from merged registry at {iteration}")
            source_snapshot_path = record.get("source_registry_snapshot")
            source_snapshot_hash = record.get(
                "source_registry_snapshot_sha256", collection["registry_snapshot_sha256"]
            )
            if source_snapshot_path is not None:
                if sha256_file(Path(source_snapshot_path)) != source_snapshot_hash:
                    raise ValueError(f"task {task_id}: source staging registry hash mismatch")
            if shard["metadata"].get("registry_snapshot_sha256") != source_snapshot_hash:
                raise ValueError(
                    f"task {task_id}: shard registry provenance differs from its source staging snapshot"
                )
            source_manifest_path = record.get("source_collection_manifest")
            if source_manifest_path is not None:
                if sha256_file(Path(source_manifest_path)) != record.get(
                    "source_collection_manifest_sha256"
                ):
                    raise ValueError(f"task {task_id}: source staging manifest hash mismatch")
            metadata = shard["metadata"]
            for paths_key, hashes_key in (("raw_files", "raw_sha256"), ("result_files", "result_sha256")):
                paths = metadata.get(paths_key, [])
                hashes = metadata.get(hashes_key, [])
                if len(paths) != len(hashes) or not paths:
                    raise ValueError(f"task {task_id}: malformed {paths_key}/{hashes_key} provenance")
                for value, expected_hash in zip(paths, hashes):
                    if sha256_file(resolve_repo_path(repo, value)) != expected_hash:
                        raise ValueError(f"task {task_id}: source rollout artifact hash mismatch")
            stop = offset + rows_per_checkpoint
            comparisons = {
                "states": shard["states"].numpy(),
                "actions": shard["policy_actions"].numpy(),
                "next_states": shard["next_states"].numpy(),
                "rewards": shard["rewards"].numpy(),
                "dones": shard["dones"].numpy(),
                "masks": shard["masks"].numpy(),
            }
            for key, expected in comparisons.items():
                if not np.array_equal(prompt[key][offset:stop], expected):
                    raise ValueError(f"task {task_id}: prompt {key} differs from shard at row {offset}")
            task_modes.update({int(key): int(value) for key, value in shard["metadata"]["window_start_mode_counts"].items()})
            offset = stop
        missing_modes = sorted(set(args.required_controller_modes).difference(task_modes))
        if missing_modes:
            raise ValueError(f"task {task_id}: no accepted windows start in controller modes {missing_modes}")

        reproduced = reproduce_query_actions(prompt["states"], query_by_task[task_id])
        if not np.array_equal(reproduced, query["actions"]):
            max_error = float(np.max(np.abs(reproduced - query["actions"])))
            raise ValueError(f"task {task_id}: query action reproduction failed; max error={max_error}")
        selected_teacher = teacher_by_task[task_id]["selected_teacher"]
        if query_by_task[task_id]["teacher_checkpoint_sha256"] != selected_teacher["checkpoint_sha256"]:
            raise ValueError(f"task {task_id}: query teacher does not match teacher registry")

        for key in PROMPT_KEYS:
            value = prompt[key].reshape(expected_rows, 1) if prompt[key].ndim == 1 else prompt[key]
            prompt_task_arrays[key].append(value)
        for key in QUERY_KEYS:
            query_task_arrays[key].append(query[key])
        task_reports.append(
            {
                "source_task_id": task_id,
                "rows": expected_rows,
                "checkpoint_iterations": [int(row["checkpoint_iteration"]) for row in records],
                "window_start_mode_counts": {str(mode): task_modes.get(mode, 0) for mode in range(4)},
                "teacher_iteration": int(selected_teacher["iteration"]),
                "teacher_success_rate": float(selected_teacher["success_rate"]),
            }
        )

    dpt_root = repo / "methods/t2mir"
    sys.path.insert(0, str(dpt_root))
    from algorithms.datasets import DPT_Dataset  # pylint: disable=import-outside-toplevel
    from algorithms.tools import data_loader, query_data_loader  # pylint: disable=import-outside-toplevel

    prompt_axis = {key: np.stack(values, axis=0) for key, values in prompt_task_arrays.items()}
    query_axis = {key: np.stack(values, axis=0) for key, values in query_task_arrays.items()}
    dataset = DPT_Dataset(
        {key: value.copy() for key, value in prompt_axis.items()},
        {key: value.copy() for key, value in query_axis.items()},
        {"max_episode_steps": horizon, "prompt_episode_horizon": 1},
        state_norm=True,
    )
    prompt_batch, query_batch = dataset.sample_batch_contrastive(6)
    expected_prompt_shapes = [(6, horizon, STATE_DIM), (6, horizon, ACTION_DIM), (6, horizon, 1)]
    expected_query_shapes = [(6, 1, STATE_DIM), (6, 1, ACTION_DIM)]
    if [array.shape for array in prompt_batch] != expected_prompt_shapes:
        raise ValueError(f"unexpected DPT prompt shapes: {[array.shape for array in prompt_batch]}")
    if [array.shape for array in query_batch] != expected_query_shapes:
        raise ValueError(f"unexpected DPT query shapes: {[array.shape for array in query_batch]}")

    full_train_ids = sorted(set(range(48)).difference(held_out_ids))
    official_loader_status = "SUBSET_SKIPPED"
    if source_ids == full_train_ids:
        common = dict(
            dir_path=str(dpt_dir),
            train_tasks=len(full_train_ids),
            total_tasks=48,
            eval_task_ids=held_out_ids,
            load_eval_tasks=False,
        )
        loaded_prompt = data_loader(**common)[0]
        loaded_query = query_data_loader(**common)[0]
        if list(map(int, loaded_prompt[1])) != [expected_rows] * len(full_train_ids):
            raise ValueError("official loader returned incorrect prompt lengths")
        if list(map(int, loaded_query[1])) != [expected_rows] * len(full_train_ids):
            raise ValueError("official loader returned incorrect query lengths")
        official_loader_status = "PASS"

    report = {
        "format_version": 2,
        "validation": "PASS",
        "dataset": collection["dataset"],
        "source_task_ids": source_ids,
        "held_out_task_ids": held_out_ids,
        "tasks": len(source_ids),
        "checkpoints_per_task": checkpoints,
        "windows_per_checkpoint": windows,
        "window_steps": horizon,
        "rows_per_task": expected_rows,
        "total_prompt_transitions": expected_rows * len(source_ids),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "context_unit": collection["context_unit"],
        "next_state_semantics": collection["next_state_semantics"],
        "query_reproduction": "BITWISE_PASS",
        "held_out_isolation": "PASS",
        "official_loader": official_loader_status,
        "dpt_dataset_smoke": {
            "status": "PASS",
            "prompt_shapes": [list(array.shape) for array in prompt_batch],
            "query_shapes": [list(array.shape) for array in query_batch],
        },
        "task_reports": task_reports,
    }
    report_path = dpt_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[FORMAL-VALIDATION] PASS: {report_path}", flush=True)


if __name__ == "__main__":
    main()
