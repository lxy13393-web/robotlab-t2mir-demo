"""Create deterministic actor-mean query labels from frozen ``formal_v1``.

The frozen prompt trajectories are never rewritten.  The derived directory
contains unchanged links (or copies) of the prompt files and new query files
whose actions are the selected specialist actor means on the exact same
states.  This removes PPO exploration noise for deterministic control
distillation while retaining complete provenance back to the frozen dataset.
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
import os
import pickle
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from pipeline.dataset.dataset_common import sha256_file, utc_now, write_json_atomic
from pipeline.dataset.prepare_official_mixed_pilot import load_actor, write_pickle_atomic


PARENT_DATASET_CONTRACT_FINGERPRINT = (
    "e269786e0c7e64370de73ac4baf534d9339038c230981be337bc0e64eb9d1001"
)
PARENT_DATASET_FILE_FINGERPRINT = (
    "bd09407b769167a34d032af8bd693c9890a93492c8c451be0c40f7570988310a"
)


def materialize_prompt(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if sha256_file(target) == sha256_file(source):
            return
        raise ValueError(f"existing derived prompt differs from frozen source: {target}")
    if mode == "symlink":
        target.symlink_to(os.path.relpath(source, start=target.parent))
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:  # pragma: no cover - argparse owns the public validation.
        raise ValueError(f"unsupported prompt materialization mode: {mode}")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_source = (
        repo
        / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1"
        / "formal_v1/dpt"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_source.parent.parent / "formal_v1_actor_mean" / "dpt",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--prompt-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How unchanged prompt files are exposed in the derived directory.",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir:
        parser.error("--output-dir must differ from the frozen --source-dir")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    validation_path = source_dir / "validation_report.json"
    teacher_registry_path = source_dir / "teacher_registry.json"
    prompt_provenance_path = source_dir / "prompt_provenance.json"
    query_provenance_path = source_dir / "query_provenance.json"
    for path in (
        validation_path,
        teacher_registry_path,
        prompt_provenance_path,
        query_provenance_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    import json

    validation = json.loads(validation_path.read_text())
    if validation.get("validation") != "PASS":
        raise ValueError("frozen parent dataset has not passed strict validation")
    teacher_registry = json.loads(teacher_registry_path.read_text())
    if teacher_registry.get("status") != "complete":
        raise ValueError("teacher registry is incomplete")
    source_task_ids = [int(value) for value in validation["source_task_ids"]]
    teacher_by_task = {
        int(row["source_task_id"]): row for row in teacher_registry["tasks"]
    }
    if sorted(teacher_by_task) != sorted(source_task_ids):
        raise ValueError("teacher registry task IDs do not match frozen validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    provenance = {
        "format_version": 1,
        "status": "building",
        "method": "same-state-deterministic-selected-specialist-actor-mean",
        "parent_dataset": str(source_dir),
        "parent_dataset_contract_fingerprint": PARENT_DATASET_CONTRACT_FINGERPRINT,
        "parent_dataset_file_fingerprint": PARENT_DATASET_FILE_FINGERPRINT,
        "parent_validation_sha256": sha256_file(validation_path),
        "parent_teacher_registry_sha256": sha256_file(teacher_registry_path),
        "parent_prompt_provenance_sha256": sha256_file(prompt_provenance_path),
        "parent_query_provenance_sha256": sha256_file(query_provenance_path),
        "prompt_materialization": args.prompt_mode,
        "device": str(device),
        "batch_size": args.batch_size,
        "source_task_ids": source_task_ids,
        "tasks": [],
    }
    provenance_path = output_dir / "actor_mean_provenance.json"
    write_json_atomic(provenance_path, provenance)

    for task_id in source_task_ids:
        prompt_source = source_dir / f"dataset_task_{task_id}.pkl"
        stochastic_query_path = source_dir / f"query_dataset_task_{task_id}.pkl"
        prompt_target = output_dir / prompt_source.name
        query_target = output_dir / stochastic_query_path.name
        materialize_prompt(prompt_source, prompt_target, args.prompt_mode)
        with prompt_source.open("rb") as stream:
            prompt = pickle.load(stream)
        with stochastic_query_path.open("rb") as stream:
            stochastic_query = pickle.load(stream)
        states = np.asarray(prompt["states"], dtype=np.float32)
        stochastic_states = np.asarray(stochastic_query["states"], dtype=np.float32)
        stochastic_actions = np.asarray(stochastic_query["actions"], dtype=np.float32)
        if not np.array_equal(states, stochastic_states):
            raise ValueError(f"task {task_id}: frozen prompt/query states are not aligned")

        teacher = teacher_by_task[task_id]["selected_teacher"]
        checkpoint = Path(teacher["checkpoint"])
        if sha256_file(checkpoint) != teacher["checkpoint_sha256"]:
            raise ValueError(f"task {task_id}: selected teacher checkpoint changed")
        actor, std, config = load_actor(checkpoint)
        actor.to(device)
        action_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, states.shape[0], args.batch_size):
                state_batch = torch.from_numpy(
                    states[start : start + args.batch_size]
                ).to(device)
                action_parts.append(actor(state_batch).cpu().numpy())
        actions = np.concatenate(action_parts, axis=0).astype(np.float32, copy=False)
        if actions.shape != stochastic_actions.shape or not np.isfinite(actions).all():
            raise ValueError(f"task {task_id}: invalid actor-mean action array")
        query = OrderedDict(states=states.copy(), actions=actions)
        write_pickle_atomic(query_target, query)

        stochastic_delta = stochastic_actions - actions
        task_report = {
            "source_task_id": task_id,
            "rows": int(states.shape[0]),
            "teacher_iteration": int(teacher["iteration"]),
            "teacher_checkpoint": str(checkpoint),
            "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
            "empirical_normalization": bool(config.get("empirical_normalization")),
            "policy_action_mode": "deterministic_actor_mean",
            "learned_std_min": float(std.min()),
            "learned_std_mean": float(std.mean()),
            "learned_std_max": float(std.max()),
            "expected_noise_mse": float(torch.square(std).mean()),
            "stochastic_to_mean_mse": float(np.mean(np.square(stochastic_delta))),
            "stochastic_to_mean_mae": float(np.mean(np.abs(stochastic_delta))),
            "prompt_file": str(prompt_target),
            "prompt_sha256": sha256_file(prompt_target),
            "query_file": str(query_target),
            "query_sha256": sha256_file(query_target),
        }
        provenance["tasks"].append(task_report)
        write_json_atomic(provenance_path, provenance)
        print(
            f"[ACTOR-MEAN] task={task_id:02d} rows={states.shape[0]} "
            f"noise_mse={task_report['stochastic_to_mean_mse']:.6f}",
            flush=True,
        )
        del actor

    provenance["status"] = "complete"
    provenance["completed_at"] = utc_now()
    provenance["aggregate"] = {
        "tasks": len(provenance["tasks"]),
        "rows": sum(row["rows"] for row in provenance["tasks"]),
        "mean_expected_noise_mse": float(
            np.mean([row["expected_noise_mse"] for row in provenance["tasks"]])
        ),
        "mean_empirical_stochastic_to_mean_mse": float(
            np.mean([row["stochastic_to_mean_mse"] for row in provenance["tasks"]])
        ),
    }
    write_json_atomic(provenance_path, provenance)
    derived_task_ids = sorted(
        int(path.stem.removeprefix("query_dataset_task_"))
        for path in output_dir.glob("query_dataset_task_*.pkl")
    )
    derived_prompt_ids = sorted(
        int(path.stem.removeprefix("dataset_task_"))
        for path in output_dir.glob("dataset_task_*.pkl")
        if not path.name.startswith("query_")
    )
    if derived_task_ids != sorted(source_task_ids) or derived_prompt_ids != sorted(source_task_ids):
        raise ValueError(
            "derived directory has stale/missing task files; use a clean output directory"
        )
    validation_report = {
        "validation": "PASS",
        "derived_query_mode": "deterministic_actor_mean",
        "parent_validation_sha256": provenance["parent_validation_sha256"],
        "parent_dataset_contract_fingerprint": PARENT_DATASET_CONTRACT_FINGERPRINT,
        "parent_dataset_file_fingerprint": PARENT_DATASET_FILE_FINGERPRINT,
        "source_task_ids": source_task_ids,
        "tasks": len(provenance["tasks"]),
        "rows": provenance["aggregate"]["rows"],
        "state_alignment": "BITWISE_PASS",
        "teacher_checkpoint_hashes": "PASS",
        "finite_float32_actions": "PASS",
        "actor_mean_provenance_sha256": sha256_file(provenance_path),
    }
    write_json_atomic(output_dir / "derived_validation_report.json", validation_report)
    print(f"[ACTOR-MEAN] PASS: {output_dir}", flush=True)
    print(f"[ACTOR-MEAN] provenance: {provenance_path}", flush=True)


if __name__ == "__main__":
    main()
