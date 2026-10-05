"""Strictly validate the three-task official-Mixed pilot and its DPT loader path."""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


PROMPT_KEYS = ("states", "actions", "next_states", "rewards", "dones", "masks")
QUERY_KEYS = ("states", "actions")
TASK_IDS = (0, 1, 2)
STATE_DIM = 123
ACTION_DIM = 37
EPISODE_STEPS = 64
CHECKPOINTS_PER_TASK = 3
EPISODES_PER_CHECKPOINT = 4
EXPECTED_ROWS = EPISODE_STEPS * CHECKPOINTS_PER_TASK * EPISODES_PER_CHECKPOINT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_pilot = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1/pilot3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=default_pilot)
    args = parser.parse_args()
    dataset_dir = args.pilot_dir / "dpt"

    collection_path = args.pilot_dir / "collection_manifest.json"
    prompt_provenance_path = dataset_dir / "prompt_provenance.json"
    teacher_registry_path = dataset_dir / "teacher_registry.json"
    query_provenance_path = dataset_dir / "query_provenance.json"
    collection = json.loads(collection_path.read_text())
    prompt_provenance = json.loads(prompt_provenance_path.read_text())
    teacher_registry = json.loads(teacher_registry_path.read_text())
    query_provenance = json.loads(query_provenance_path.read_text())

    if collection.get("status") != "complete" or teacher_registry.get("status") != "complete":
        raise ValueError("collection and teacher evaluation must both be complete")
    if collection.get("pilot_to_source_task") != {"0": 0, "1": 27, "2": 46}:
        raise ValueError("unexpected pilot/source task mapping")
    if prompt_provenance["collection_manifest_sha256"] != sha256_file(collection_path):
        raise ValueError("prompt provenance does not match the collection manifest")
    if query_provenance["teacher_registry_sha256"] != sha256_file(teacher_registry_path):
        raise ValueError("query provenance does not match the teacher registry")

    prompt_by_task = {int(row["pilot_task_id"]): row for row in prompt_provenance["tasks"]}
    query_by_task = {int(row["pilot_task_id"]): row for row in query_provenance["tasks"]}
    teacher_by_task = {int(row["pilot_task_id"]): row for row in teacher_registry["tasks"]}
    if set(prompt_by_task) != set(TASK_IDS) or set(query_by_task) != set(TASK_IDS):
        raise ValueError("prompt/query provenance must cover exactly pilot tasks 0, 1, 2")

    task_reports = []
    for task_id in TASK_IDS:
        prompt_path = dataset_dir / f"dataset_task_{task_id}.pkl"
        query_path = dataset_dir / f"query_dataset_task_{task_id}.pkl"
        if sha256_file(prompt_path) != prompt_by_task[task_id]["prompt_sha256"]:
            raise ValueError(f"task {task_id}: prompt hash mismatch")
        if sha256_file(query_path) != query_by_task[task_id]["query_sha256"]:
            raise ValueError(f"task {task_id}: query hash mismatch")
        prompt, query = load_pickle(prompt_path), load_pickle(query_path)
        exact_keys(prompt, PROMPT_KEYS, prompt_path)
        exact_keys(query, QUERY_KEYS, query_path)
        validate_array(f"task {task_id} prompt.states", prompt["states"], (EXPECTED_ROWS, STATE_DIM))
        validate_array(f"task {task_id} prompt.actions", prompt["actions"], (EXPECTED_ROWS, ACTION_DIM))
        validate_array(f"task {task_id} prompt.next_states", prompt["next_states"], (EXPECTED_ROWS, STATE_DIM))
        for key in ("rewards", "dones", "masks"):
            validate_array(f"task {task_id} prompt.{key}", prompt[key], (EXPECTED_ROWS,))
        validate_array(f"task {task_id} query.states", query["states"], (EXPECTED_ROWS, STATE_DIM))
        validate_array(f"task {task_id} query.actions", query["actions"], (EXPECTED_ROWS, ACTION_DIM))
        if not np.array_equal(prompt["states"], query["states"]):
            raise ValueError(f"task {task_id}: query states are not exact same-state relabels")
        if np.any(prompt["dones"] != 0.0) or np.any(prompt["masks"] != 1.0):
            raise ValueError(f"task {task_id}: retained windows contain a termination")
        checkpoint_rows = prompt_by_task[task_id]["checkpoints"]
        if len(checkpoint_rows) != CHECKPOINTS_PER_TASK:
            raise ValueError(f"task {task_id}: expected three checkpoint contributions")
        expected_offset = 0
        for checkpoint in checkpoint_rows:
            if checkpoint["row_start"] != expected_offset:
                raise ValueError(f"task {task_id}: checkpoint rows are not contiguous")
            contribution = checkpoint["row_stop"] - checkpoint["row_start"]
            if contribution != EPISODE_STEPS * EPISODES_PER_CHECKPOINT:
                raise ValueError(f"task {task_id}: checkpoint contribution is not balanced")
            if checkpoint["episodes"] != EPISODES_PER_CHECKPOINT or checkpoint["episode_steps"] != EPISODE_STEPS:
                raise ValueError(f"task {task_id}: invalid checkpoint episode geometry")
            expected_offset = checkpoint["row_stop"]
        if expected_offset != EXPECTED_ROWS:
            raise ValueError(f"task {task_id}: provenance does not cover the full prompt")
        teacher = teacher_by_task[task_id]["selected_teacher"]
        if query_by_task[task_id]["teacher_checkpoint_sha256"] != teacher["checkpoint_sha256"]:
            raise ValueError(f"task {task_id}: query teacher mismatch")
        task_reports.append(
            {
                "pilot_task_id": task_id,
                "source_task_id": int(prompt_by_task[task_id]["source_task_id"]),
                "prompt_rows": EXPECTED_ROWS,
                "query_rows": EXPECTED_ROWS,
                "teacher_iteration": int(teacher["iteration"]),
                "teacher_success_rate": float(teacher["success_rate"]),
                "teacher_mean_episode_return": float(teacher["mean_episode_return"]),
                "checkpoint_iterations": [int(row["iteration"]) for row in checkpoint_rows],
            }
        )

    # Exercise the unmodified official data loaders and DPT_Dataset class.
    dpt_root = repo / "methods/t2mir"
    sys.path.insert(0, str(dpt_root))
    from algorithms.datasets import DPT_Dataset  # pylint: disable=import-outside-toplevel
    from algorithms.tools import data_loader, query_data_loader  # pylint: disable=import-outside-toplevel

    train_data, lengths, indices = data_loader(
        str(dataset_dir), train_tasks=3, total_tasks=3, eval_task_ids=[], load_eval_tasks=False
    )[0]
    train_query, query_lengths, query_indices = query_data_loader(
        str(dataset_dir), train_tasks=3, total_tasks=3, eval_task_ids=[], load_eval_tasks=False
    )[0]
    if list(map(int, lengths)) != [EXPECTED_ROWS] * 3 or list(map(int, query_lengths)) != [EXPECTED_ROWS] * 3:
        raise ValueError("official loaders returned incorrect per-task lengths")
    task_axis_data = {key: np.array(np.split(value, indices[1:], axis=0)) for key, value in train_data.items()}
    task_axis_query = {
        key: np.array(np.split(value, query_indices[1:], axis=0)) for key, value in train_query.items()
    }
    config = {"max_episode_steps": EPISODE_STEPS, "prompt_episode_horizon": 1}
    dataset = DPT_Dataset(task_axis_data, task_axis_query, config, state_norm=True)
    prompt_batch, query_batch = dataset.sample_batch_contrastive(6)
    expected_prompt_shapes = [(6, 64, 123), (6, 64, 37), (6, 64, 1)]
    expected_query_shapes = [(6, 1, 123), (6, 1, 37)]
    if [array.shape for array in prompt_batch] != expected_prompt_shapes:
        raise ValueError(f"unexpected official prompt batch shapes: {[array.shape for array in prompt_batch]}")
    if [array.shape for array in query_batch] != expected_query_shapes:
        raise ValueError(f"unexpected official query batch shapes: {[array.shape for array in query_batch]}")
    if not all(np.isfinite(array).all() for array in prompt_batch + query_batch):
        raise ValueError("official loader produced NaN or Inf")

    report = {
        "format_version": 1,
        "validation": "PASS",
        "dataset": "RobotLab-G1-MultiDynamics-official-mixed-v1-pilot3",
        "tasks": 3,
        "prompt_rows_per_task": EXPECTED_ROWS,
        "query_rows_per_task": EXPECTED_ROWS,
        "total_prompt_transitions": EXPECTED_ROWS * 3,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "episode_steps": EPISODE_STEPS,
        "checkpoint_contributions_per_task": CHECKPOINTS_PER_TASK,
        "episodes_per_checkpoint": EPISODES_PER_CHECKPOINT,
        "policy_action_mode": "stochastic",
        "query_label_mode": "stochastic",
        "teacher_selection": teacher_registry["selection_rule"],
        "official_loader_smoke": {
            "status": "PASS",
            "prompt_shapes": [list(array.shape) for array in prompt_batch],
            "query_shapes": [list(array.shape) for array in query_batch],
        },
        "task_reports": task_reports,
    }
    report_path = dataset_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[VALIDATION] PASS: {report_path}", flush=True)


if __name__ == "__main__":
    main()
