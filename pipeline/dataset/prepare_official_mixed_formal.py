"""Build prompt/query pickles from formal RobotLab T2MIR Mixed shards.

The script preserves source task IDs (0..47 with held-out gaps), shortlists
teacher candidates using metrics already produced during collection, evaluates
only the shortlist on independent common seeds, and stochastically relabels
the prompt state pool with the selected specialist.
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
import pickle
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from pipeline.dataset.collect_official_mixed_formal import accepted_shard_is_valid
from pipeline.dataset.dataset_common import sha256_file, utc_now, write_json_atomic
from pipeline.protocols.dynamics_profile import DynamicsProfile
from pipeline.dataset.prepare_official_mixed_pilot import (
    TASK_NAME,
    csv_ints,
    load_actor,
    read_goal_csv,
    write_pickle_atomic,
)


PROMPT_KEYS = ("states", "actions", "next_states", "rewards", "dones", "masks")


def load_collection(path: Path) -> dict:
    collection = json.loads(path.read_text())
    if collection.get("status") != "complete":
        raise ValueError(f"formal collection is not complete: {collection.get('status')!r}")
    if collection.get("policy_action_mode") != "stochastic":
        raise ValueError("formal prompt collection was not stochastic")
    if collection.get("context_unit") != "continuous_fixed_length_window":
        raise ValueError("unexpected formal context unit")
    expected_shards = len(collection["source_task_ids"]) * int(collection["checkpoints_per_task"])
    if len(collection.get("shards", [])) != expected_shards:
        raise ValueError(
            f"formal collection has {len(collection.get('shards', []))} shards; expected {expected_shards}"
        )
    return collection


def profile_from_record(record: dict) -> DynamicsProfile:
    row = record["profile"]
    profile = DynamicsProfile(
        task_id=int(row["task_id"]),
        action_lag=float(row["action_lag"]),
        motor_strength=float(row["motor_strength"]),
        payload_kg=float(row["payload_kg"]),
        friction=float(row["friction"]),
        split=str(row["split"]),
    )
    profile.validate()
    return profile


def ranking_key(metrics: dict, iteration: int) -> tuple:
    success_time = metrics.get("mean_success_time_s")
    return (
        float(metrics["success_rate"]),
        -float(metrics["fall_rate"]),
        -float(success_time) if success_time is not None else float("-inf"),
        -float(metrics["mean_final_position_error_m"]),
        -float(metrics["mean_final_yaw_error_rad"]),
        float(metrics["mean_return_per_second"]),
        int(iteration),
    )


def group_records(collection: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in collection["shards"]:
        grouped.setdefault(int(record["source_task_id"]), []).append(record)
    expected = sorted(map(int, collection["source_task_ids"]))
    if sorted(grouped) != expected:
        raise ValueError(f"formal source task IDs mismatch: {sorted(grouped)} != {expected}")
    checkpoints = int(collection["checkpoints_per_task"])
    for task_id, records in grouped.items():
        if len(records) != checkpoints:
            raise ValueError(f"task {task_id}: expected {checkpoints} checkpoint shards")
        records.sort(key=lambda row: int(row["checkpoint_iteration"]))
    return grouped


def build_prompts(collection_path: Path, collection: dict, output_dir: Path) -> dict:
    windows = int(collection["windows_per_checkpoint"])
    horizon = int(collection["window_steps"])
    provenance = {
        "format_version": 2,
        "method": "equal-continuous-phase-stratified-windows-per-checkpoint",
        "collection_manifest": str(collection_path),
        "collection_manifest_sha256": sha256_file(collection_path),
        "registry_snapshot_sha256": collection["registry_snapshot_sha256"],
        "manifest_sha256": collection["manifest_sha256"],
        "window_steps": horizon,
        "windows_per_checkpoint": windows,
        "checkpoints_per_task": int(collection["checkpoints_per_task"]),
        "tasks": [],
    }
    for task_id, records in sorted(group_records(collection).items()):
        parts = {key: [] for key in PROMPT_KEYS}
        task_row = {"source_task_id": task_id, "checkpoints": []}
        offset = 0
        for record in records:
            shard_path = Path(record["accepted_shard"])
            if sha256_file(shard_path) != record["accepted_sha256"]:
                raise ValueError(f"accepted shard hash mismatch: {shard_path}")
            if not accepted_shard_is_valid(
                shard_path,
                source_task_id=task_id,
                iteration=int(record["checkpoint_iteration"]),
                windows=windows,
                horizon=horizon,
                checkpoint_sha256=record["checkpoint_sha256"],
            ):
                raise ValueError(f"accepted shard failed strict validation: {shard_path}")
            shard = torch.load(shard_path, map_location="cpu")
            arrays = {
                "states": shard["states"].numpy().astype(np.float32, copy=False),
                "actions": shard["policy_actions"].numpy().astype(np.float32, copy=False),
                "next_states": shard["next_states"].numpy().astype(np.float32, copy=False),
                "rewards": shard["rewards"].numpy().astype(np.float32, copy=False).reshape(-1),
                "dones": shard["dones"].numpy().astype(np.float32, copy=False).reshape(-1),
                "masks": shard["masks"].numpy().astype(np.float32, copy=False).reshape(-1),
            }
            for key in PROMPT_KEYS:
                parts[key].append(arrays[key])
            count = int(arrays["states"].shape[0])
            task_row["checkpoints"].append(
                {
                    "iteration": int(record["checkpoint_iteration"]),
                    "checkpoint": record["checkpoint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "accepted_shard": str(shard_path),
                    "accepted_sha256": record["accepted_sha256"],
                    "source_collection_manifest": record.get("source_collection_manifest"),
                    "source_collection_manifest_sha256": record.get(
                        "source_collection_manifest_sha256"
                    ),
                    "source_registry_snapshot": record.get("source_registry_snapshot"),
                    "source_registry_snapshot_sha256": record.get(
                        "source_registry_snapshot_sha256",
                        collection["registry_snapshot_sha256"],
                    ),
                    "row_start": offset,
                    "row_stop": offset + count,
                    "windows": windows,
                    "window_steps": horizon,
                    "window_start_mode_counts": shard["metadata"]["window_start_mode_counts"],
                    "policy_action_seeds": shard["metadata"]["policy_action_seeds"],
                }
            )
            offset += count
        prompt = OrderedDict((key, np.concatenate(parts[key], axis=0)) for key in PROMPT_KEYS)
        prompt_path = output_dir / f"dataset_task_{task_id}.pkl"
        write_pickle_atomic(prompt_path, prompt)
        task_row.update(
            rows=int(prompt["states"].shape[0]),
            prompt_file=str(prompt_path),
            prompt_sha256=sha256_file(prompt_path),
        )
        provenance["tasks"].append(task_row)
        print(f"[FORMAL-PROMPT] task={task_id} rows={task_row['rows']} file={prompt_path}", flush=True)
    provenance["created_at"] = utc_now()
    write_json_atomic(output_dir / "prompt_provenance.json", provenance)
    return provenance


def evaluate_teachers(
    collection: dict,
    output_dir: Path,
    repo: Path,
    *,
    shortlist_size: int,
    seeds: list[int],
    goal_batch: int,
    force: bool,
) -> dict:
    registry_path = output_dir / "teacher_registry.json"
    teacher_registry = {
        "format_version": 2,
        "status": "evaluating",
        "method": "collection-metric-shortlist-then-independent-common-seed-evaluation",
        "selection_rule": (
            "success_rate_then_lower_fall_rate_then_lower_mean_success_time_"
            "then_lower_final_errors_then_mean_return_per_second"
        ),
        "shortlist_size": shortlist_size,
        "evaluation_seeds": seeds,
        "goal_batch": goal_batch,
        "manifest_sha256": collection["manifest_sha256"],
        "tasks": [],
    }
    write_json_atomic(registry_path, teacher_registry)
    play_py = repo / "pipeline/expert/play.py"
    for task_id, records in sorted(group_records(collection).items()):
        ranked = sorted(
            records,
            key=lambda row: ranking_key(row["collection_metrics"], int(row["checkpoint_iteration"])),
            reverse=True,
        )
        # Collection metrics use only the rollout batch that produced prompt
        # windows, so their ranking is deliberately a cheap screening signal.
        # Always retain the latest checkpoint as a safety candidate, then fill
        # the remaining slots by the success-first collection ranking.  This
        # prevents a noisy 12/16/32-goal screening batch from excluding the
        # converged policy before independent common-seed evaluation.
        latest = max(records, key=lambda row: int(row["checkpoint_iteration"]))
        shortlisted = [latest]
        for record in ranked:
            if len(shortlisted) >= shortlist_size:
                break
            if record is latest:
                continue
            shortlisted.append(record)
        profile = profile_from_record(records[0])
        task_row = {
            "source_task_id": task_id,
            "profile": profile.__dict__,
            "collection_shortlist_iterations": [int(row["checkpoint_iteration"]) for row in shortlisted],
            "candidates": [],
        }
        teacher_registry["tasks"].append(task_row)
        for record in shortlisted:
            checkpoint = Path(record["checkpoint"])
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError(f"checkpoint changed after collection: {checkpoint}")
            iteration = int(record["checkpoint_iteration"])
            candidate_dir = output_dir / "teacher_evaluation" / f"task{task_id:02d}" / f"iteration_{iteration:06d}"
            runs = []
            for seed in seeds:
                result_path = candidate_dir / f"seed{seed}_n{goal_batch}.csv"
                command = [
                    sys.executable,
                    "-u",
                    str(play_py),
                    "--task",
                    TASK_NAME,
                    "--headless",
                    "--load_run",
                    checkpoint.parent.name,
                    "--checkpoint",
                    checkpoint.name,
                    "--goal_batch",
                    str(goal_batch),
                    "--goal_min_distance",
                    "1.0",
                    "--goal_max_distance",
                    "4.0",
                    "--goal_timeout",
                    "40",
                    "--goal_hold_time",
                    "2",
                    "--goal_seed",
                    str(seed),
                    "--skip_policy_export",
                    "--goal_results",
                    str(result_path),
                    *profile.cli_args(),
                ]
                candidate_dir.mkdir(parents=True, exist_ok=True)
                print(" ".join(command), flush=True)
                if force or not result_path.exists():
                    subprocess.run(command, cwd=repo, check=True)
                runs.append({"seed": seed, "result_file": str(result_path), **read_goal_csv(result_path, profile, goal_batch)})
            episodes = sum(run["episodes"] for run in runs)
            successes = sum(run["successes"] for run in runs)
            falls = sum(run["falls"] for run in runs)
            candidate = {
                "iteration": iteration,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "episodes": episodes,
                "successes": successes,
                "falls": falls,
                "timeouts": episodes - successes - falls,
                "success_rate": successes / episodes,
                "fall_rate": falls / episodes,
                "mean_episode_return": sum(run["mean_episode_return"] * run["episodes"] for run in runs) / episodes,
                "mean_return_per_second": sum(run["mean_return_per_second"] * run["episodes"] for run in runs) / episodes,
                "mean_episode_duration_s": sum(run["mean_episode_duration_s"] * run["episodes"] for run in runs) / episodes,
                "mean_final_position_error_m": sum(run["mean_final_position_error_m"] * run["episodes"] for run in runs) / episodes,
                "mean_final_yaw_error_rad": sum(run["mean_final_yaw_error_rad"] * run["episodes"] for run in runs) / episodes,
                "runs": runs,
            }
            candidate["mean_success_time_s"] = (
                sum(run["mean_success_time_s"] * run["successes"] for run in runs if run["successes"])
                / successes
                if successes
                else None
            )
            task_row["candidates"].append(candidate)
            write_json_atomic(registry_path, teacher_registry)
        selected = max(
            task_row["candidates"],
            key=lambda row: ranking_key(row, int(row["iteration"])),
        )
        task_row["selected_teacher"] = {
            key: selected[key]
            for key in (
                "iteration",
                "checkpoint",
                "checkpoint_sha256",
                "success_rate",
                "fall_rate",
                "mean_episode_return",
                "mean_return_per_second",
                "mean_episode_duration_s",
                "mean_success_time_s",
                "mean_final_position_error_m",
                "mean_final_yaw_error_rad",
            )
        }
        print(
            f"[FORMAL-TEACHER] task={task_id} iteration={selected['iteration']} "
            f"success={selected['success_rate']:.3f} fall={selected['fall_rate']:.3f}",
            flush=True,
        )
        write_json_atomic(registry_path, teacher_registry)
    teacher_registry["status"] = "complete"
    teacher_registry["completed_at"] = utc_now()
    write_json_atomic(registry_path, teacher_registry)
    return teacher_registry


def build_queries(output_dir: Path, teacher_registry: dict, query_seed: int, batch_size: int) -> dict:
    if teacher_registry.get("status") != "complete":
        raise ValueError("teacher evaluation is not complete")
    provenance = {
        "format_version": 2,
        "method": "same-state-stochastic-best-specialist-relabeling",
        "teacher_registry_sha256": sha256_file(output_dir / "teacher_registry.json"),
        "selection_rule": teacher_registry["selection_rule"],
        "base_query_seed": query_seed,
        "query_batch_size": batch_size,
        "tasks": [],
    }
    for task_row in teacher_registry["tasks"]:
        task_id = int(task_row["source_task_id"])
        prompt_path = output_dir / f"dataset_task_{task_id}.pkl"
        with prompt_path.open("rb") as stream:
            prompt = pickle.load(stream)
        states = np.asarray(prompt["states"], dtype=np.float32)
        teacher = task_row["selected_teacher"]
        checkpoint = Path(teacher["checkpoint"])
        if sha256_file(checkpoint) != teacher["checkpoint_sha256"]:
            raise ValueError(f"selected teacher checkpoint changed: {checkpoint}")
        actor, std, config = load_actor(checkpoint)
        generator = torch.Generator(device="cpu")
        task_seed = query_seed + task_id
        generator.manual_seed(task_seed)
        action_parts = []
        with torch.inference_mode():
            for start in range(0, states.shape[0], batch_size):
                state_batch = torch.from_numpy(states[start : start + batch_size])
                mean = actor(state_batch)
                noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
                action_parts.append((mean + noise * std).numpy())
        actions = np.concatenate(action_parts, axis=0).astype(np.float32, copy=False)
        query = OrderedDict(states=states.copy(), actions=actions)
        query_path = output_dir / f"query_dataset_task_{task_id}.pkl"
        write_pickle_atomic(query_path, query)
        provenance["tasks"].append(
            {
                "source_task_id": task_id,
                "teacher_iteration": int(teacher["iteration"]),
                "teacher_checkpoint": str(checkpoint),
                "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
                "policy_action_mode": "stochastic",
                "query_seed": task_seed,
                "query_batch_size": batch_size,
                "learned_std_min": float(std.min()),
                "learned_std_mean": float(std.mean()),
                "learned_std_max": float(std.max()),
                "empirical_normalization": bool(config.get("empirical_normalization")),
                "rows": int(states.shape[0]),
                "query_file": str(query_path),
                "query_sha256": sha256_file(query_path),
            }
        )
        print(f"[FORMAL-QUERY] task={task_id} rows={states.shape[0]} file={query_path}", flush=True)
    provenance["created_at"] = utc_now()
    write_json_atomic(output_dir / "query_provenance.json", provenance)
    return provenance


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_dir = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1/formal_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=default_dir)
    parser.add_argument("--phase", choices=("prompt", "evaluate", "query", "validate", "all"), default="all")
    parser.add_argument("--teacher-shortlist", type=int, default=3)
    parser.add_argument("--eval-seeds", type=csv_ints, default=[42, 43, 44])
    parser.add_argument("--goal-batch", type=int, default=16)
    parser.add_argument("--query-seed", type=int, default=20260928)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--force-evaluation", action="store_true")
    args = parser.parse_args()
    if min(args.teacher_shortlist, args.goal_batch, args.query_batch_size) <= 0 or args.query_seed < 0:
        parser.error("shortlist/batch sizes must be positive and query seed non-negative")

    collection_path = args.formal_dir / "collection_manifest.json"
    collection = load_collection(collection_path)
    output_dir = args.formal_dir / "dpt"
    if args.phase in ("prompt", "all"):
        build_prompts(collection_path, collection, output_dir)
    if args.phase in ("evaluate", "all"):
        evaluate_teachers(
            collection,
            output_dir,
            repo,
            shortlist_size=args.teacher_shortlist,
            seeds=args.eval_seeds,
            goal_batch=args.goal_batch,
            force=args.force_evaluation,
        )
    if args.phase in ("query", "all"):
        teacher_path = output_dir / "teacher_registry.json"
        if not teacher_path.exists():
            raise FileNotFoundError("teacher_registry.json is missing; run --phase evaluate first")
        build_queries(output_dir, json.loads(teacher_path.read_text()), args.query_seed, args.query_batch_size)
    if args.phase in ("validate", "all"):
        validator = repo / "pipeline/dataset/validate_official_mixed_formal.py"
        subprocess.run(
            [sys.executable, "-u", str(validator), "--formal-dir", str(args.formal_dir)],
            cwd=repo,
            check=True,
        )


if __name__ == "__main__":
    main()
