"""Build, label, and validate the three-task official-Mixed pilot dataset.

The restartable ``all`` phase performs four operations:

1. flatten the already accepted checkpoint-balanced 64-step windows;
2. evaluate the three candidate checkpoints per task on independent common seeds;
3. relabel the prompt state pool with stochastic actions from the selected teacher;
4. run the strict validator and official DPT loader smoke test.

Teacher selection is a documented RobotLab adaptation: closed-loop goal success
rate is primary, followed by safety, completion time, terminal error, and only
then time-normalized return.  The raw-return winner is retained as a diagnostic;
unlike the original fixed-horizon evaluation, RobotLab's early-success episodes
make raw returns across attempts with different durations non-comparable.
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
import csv
import hashlib
import json
import pickle
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from pipeline.dataset.collect_official_mixed_pilot import accepted_shard_is_valid
from pipeline.protocols.dynamics_profile import DynamicsProfile


TASK_NAME = "RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0"
PROMPT_KEYS = ("states", "actions", "next_states", "rewards", "dones", "masks")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def write_pickle_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream)
    temporary.replace(path)


def load_collection_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"collection is not complete: status={payload.get('status')!r}")
    if payload.get("policy_action_mode") != "stochastic":
        raise ValueError("prompt collection was not stochastic")
    if len(payload.get("shards", [])) != 9:
        raise ValueError("pilot collection must contain exactly nine accepted shards")
    return payload


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


def build_prompts(collection: dict, output_dir: Path) -> dict:
    provenance = {
        "format_version": 1,
        "method": "equal-complete-windows-per-selected-checkpoint",
        "collection_manifest_sha256": sha256_file(Path(collection["collection_manifest_path"])),
        "manifest_sha256": collection["manifest_sha256"],
        "episode_steps": int(collection["episode_steps"]),
        "episodes_per_checkpoint": int(collection["episodes_per_checkpoint"]),
        "tasks": [],
    }
    grouped: dict[int, list[dict]] = {}
    for record in collection["shards"]:
        grouped.setdefault(int(record["pilot_task_id"]), []).append(record)
    if sorted(grouped) != [0, 1, 2]:
        raise ValueError(f"pilot task IDs must be [0, 1, 2], got {sorted(grouped)}")

    for pilot_task_id, records in sorted(grouped.items()):
        records.sort(key=lambda row: int(row["checkpoint_iteration"]))
        if len(records) != 3:
            raise ValueError(f"pilot task {pilot_task_id}: expected three checkpoint shards")
        parts = {key: [] for key in PROMPT_KEYS}
        task_provenance = {
            "pilot_task_id": pilot_task_id,
            "source_task_id": int(records[0]["source_task_id"]),
            "checkpoints": [],
        }
        offset = 0
        for record in records:
            shard_path = Path(record["accepted_shard"])
            if sha256_file(shard_path) != record["accepted_sha256"]:
                raise ValueError(f"accepted shard hash mismatch: {shard_path}")
            if not accepted_shard_is_valid(
                shard_path,
                source_task_id=int(record["source_task_id"]),
                pilot_task_id=pilot_task_id,
                iteration=int(record["checkpoint_iteration"]),
                episodes=int(collection["episodes_per_checkpoint"]),
                horizon=int(collection["episode_steps"]),
                checkpoint_sha256=str(record["checkpoint_sha256"]),
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
            task_provenance["checkpoints"].append(
                {
                    "iteration": int(record["checkpoint_iteration"]),
                    "checkpoint": record["checkpoint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "accepted_shard": str(shard_path),
                    "accepted_sha256": record["accepted_sha256"],
                    "row_start": offset,
                    "row_stop": offset + count,
                    "episodes": int(shard["metadata"]["episodes"]),
                    "episode_steps": int(shard["metadata"]["episode_steps"]),
                    "policy_action_seeds": shard["metadata"]["policy_action_seeds"],
                    "source_attempt_ids": sorted(set(map(int, shard["source_attempt_ids"].tolist()))),
                    "source_env_ids": sorted(set(map(int, shard["source_env_ids"].tolist()))),
                }
            )
            offset += count
        dataset = OrderedDict((key, np.concatenate(parts[key], axis=0)) for key in PROMPT_KEYS)
        prompt_path = output_dir / f"dataset_task_{pilot_task_id}.pkl"
        write_pickle_atomic(prompt_path, dataset)
        task_provenance["rows"] = int(dataset["states"].shape[0])
        task_provenance["prompt_file"] = str(prompt_path)
        task_provenance["prompt_sha256"] = sha256_file(prompt_path)
        provenance["tasks"].append(task_provenance)
        print(f"[PROMPT] task={pilot_task_id} rows={task_provenance['rows']} file={prompt_path}", flush=True)
    provenance["created_at"] = utc_now()
    write_json_atomic(output_dir / "prompt_provenance.json", provenance)
    return provenance


def read_goal_csv(path: Path, profile: DynamicsProfile, expected_rows: int) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(rows)}")
    expected = {
        "action_lag": profile.action_lag,
        "motor_strength": profile.motor_strength,
        "payload_kg": profile.payload_kg,
        "friction": profile.friction,
    }
    for row in rows:
        for key, value in expected.items():
            if abs(float(row[key]) - value) > 1.0e-8:
                raise ValueError(f"{path}: {key} does not match the source task")
    successes = sum(row["result"] == "success" for row in rows)
    falls = sum(row["result"] == "fall" for row in rows)
    returns = [float(row["episode_return"]) for row in rows]
    durations = [float(row["time"]) for row in rows]
    return_rates = [value / max(duration, 1.0e-8) for value, duration in zip(returns, durations)]
    success_times = [float(row["time"]) for row in rows if row["result"] == "success"]
    position_errors = [float(row["position_error"]) for row in rows]
    yaw_errors = [float(row["yaw_error"]) for row in rows]
    return {
        "episodes": len(rows),
        "successes": successes,
        "falls": falls,
        "timeouts": len(rows) - successes - falls,
        "mean_episode_return": float(np.mean(returns)),
        # Raw undiscounted return is duration-biased because successful goal
        # attempts stop early while failed attempts can run until timeout.
        # Preserve it for provenance, but use this rate for comparisons among
        # candidates with identical task-success statistics.
        "mean_return_per_second": float(np.mean(return_rates)),
        "mean_episode_duration_s": float(np.mean(durations)),
        "mean_success_time_s": float(np.mean(success_times)) if success_times else None,
        "mean_final_position_error_m": float(np.mean(position_errors)),
        "mean_final_yaw_error_rad": float(np.mean(yaw_errors)),
    }


def evaluate_teachers(
    collection: dict,
    output_dir: Path,
    repo: Path,
    seeds: list[int],
    goal_batch: int,
    force: bool,
) -> dict:
    registry_path = output_dir / "teacher_registry.json"
    registry = {
        "format_version": 1,
        "status": "evaluating",
        "method": "independent-common-seed-deterministic-closed-loop-evaluation",
        "selection_rule": (
            "success_rate_then_lower_fall_rate_then_lower_mean_success_time_"
            "then_lower_final_errors_then_mean_return_per_second"
        ),
        "raw_return_only_winner_recorded": True,
        "manifest_sha256": collection["manifest_sha256"],
        "evaluation_seeds": seeds,
        "goal_batch": goal_batch,
        "tasks": [],
    }
    write_json_atomic(registry_path, registry)
    grouped: dict[int, list[dict]] = {}
    for record in collection["shards"]:
        grouped.setdefault(int(record["pilot_task_id"]), []).append(record)
    play_py = repo / "pipeline/expert/play.py"

    for pilot_task_id, records in sorted(grouped.items()):
        source_task_id = int(records[0]["source_task_id"])
        profile = profile_from_record(records[0])
        task_row = {
            "pilot_task_id": pilot_task_id,
            "source_task_id": source_task_id,
            "profile": profile.__dict__,
            "candidates": [],
        }
        registry["tasks"].append(task_row)
        for record in sorted(records, key=lambda row: int(row["checkpoint_iteration"])):
            checkpoint_path = Path(record["checkpoint"])
            if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                raise ValueError(f"checkpoint changed after collection: {checkpoint_path}")
            candidate_dir = (
                output_dir
                / "teacher_evaluation"
                / f"task{pilot_task_id:02d}_source{source_task_id:02d}"
                / f"iteration_{int(record['checkpoint_iteration']):06d}"
            )
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
                    checkpoint_path.parent.name,
                    "--checkpoint",
                    checkpoint_path.name,
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
                print(" ".join(command), flush=True)
                candidate_dir.mkdir(parents=True, exist_ok=True)
                if force or not result_path.exists():
                    try:
                        subprocess.run(command, cwd=repo, check=True)
                    except subprocess.CalledProcessError as error:
                        registry["status"] = "failed"
                        registry["failure"] = f"pipeline.expert.play.py exited with status {error.returncode}"
                        write_json_atomic(registry_path, registry)
                        raise
                metrics = read_goal_csv(result_path, profile, goal_batch)
                runs.append({"seed": seed, "result_file": str(result_path), **metrics})
            episodes = sum(row["episodes"] for row in runs)
            successes = sum(row["successes"] for row in runs)
            falls = sum(row["falls"] for row in runs)
            candidate = {
                "iteration": int(record["checkpoint_iteration"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "episodes": episodes,
                "successes": successes,
                "falls": falls,
                "timeouts": episodes - successes - falls,
                "success_rate": successes / episodes,
                "fall_rate": falls / episodes,
                "mean_episode_return": sum(row["mean_episode_return"] * row["episodes"] for row in runs) / episodes,
                "mean_return_per_second": sum(
                    row["mean_return_per_second"] * row["episodes"] for row in runs
                ) / episodes,
                "mean_episode_duration_s": sum(
                    row["mean_episode_duration_s"] * row["episodes"] for row in runs
                ) / episodes,
                "mean_final_position_error_m": sum(
                    row["mean_final_position_error_m"] * row["episodes"] for row in runs
                ) / episodes,
                "mean_final_yaw_error_rad": sum(
                    row["mean_final_yaw_error_rad"] * row["episodes"] for row in runs
                ) / episodes,
                "runs": runs,
            }
            success_count = sum(row["successes"] for row in runs)
            candidate["mean_success_time_s"] = (
                sum(row["mean_success_time_s"] * row["successes"] for row in runs if row["successes"])
                / success_count
                if success_count
                else None
            )
            task_row["candidates"].append(candidate)
            write_json_atomic(registry_path, registry)

        selected = max(
            task_row["candidates"],
            key=lambda row: (
                row["success_rate"],
                -row["fall_rate"],
                -row["mean_success_time_s"] if row["mean_success_time_s"] is not None else float("-inf"),
                -row["mean_final_position_error_m"],
                -row["mean_final_yaw_error_rad"],
                row["mean_return_per_second"],
                row["iteration"],
            ),
        )
        return_winner = max(task_row["candidates"], key=lambda row: (row["mean_episode_return"], row["iteration"]))
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
        task_row["diagnostic_raw_return_winner_iteration"] = return_winner["iteration"]
        print(
            f"[TEACHER] task={pilot_task_id} source={source_task_id} "
            f"iteration={selected['iteration']} success={selected['success_rate']:.3f} "
            f"return={selected['mean_episode_return']:.3f} "
            f"return_per_second={selected['mean_return_per_second']:.3f}",
            flush=True,
        )
        write_json_atomic(registry_path, registry)
    registry["status"] = "complete"
    registry["completed_at"] = utc_now()
    write_json_atomic(registry_path, registry)
    return registry


def activation(name: str) -> nn.Module:
    choices = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return choices[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unsupported actor activation {name!r}") from error


def load_actor(checkpoint_path: Path) -> tuple[nn.Sequential, torch.Tensor, dict]:
    agent_config_path = checkpoint_path.parent / "params/agent.yaml"
    config = yaml.safe_load(agent_config_path.read_text())
    if config.get("empirical_normalization"):
        raise ValueError("pilot query relabeling does not yet support empirical observation normalization")
    policy = config["policy"]
    dimensions = [123, *map(int, policy["actor_hidden_dims"]), 37]
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < len(dimensions) - 2:
            layers.append(activation(str(policy["activation"])))
    actor = nn.Sequential(*layers)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    actor_state = OrderedDict(
        (key.removeprefix("actor."), value) for key, value in state.items() if key.startswith("actor.")
    )
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    std = state["std"].detach().cpu().to(torch.float32)
    if tuple(std.shape) != (37,) or not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError(f"invalid learned PPO action std in {checkpoint_path}")
    return actor, std, config


def build_queries(output_dir: Path, teacher_registry: dict, query_seed: int, batch_size: int) -> dict:
    if teacher_registry.get("status") != "complete":
        raise ValueError("teacher evaluation is not complete")
    provenance = {
        "format_version": 1,
        "method": "same-state-stochastic-best-specialist-relabeling",
        "teacher_registry_sha256": sha256_file(output_dir / "teacher_registry.json"),
        "selection_rule": teacher_registry["selection_rule"],
        "base_query_seed": query_seed,
        "tasks": [],
    }
    for task_row in teacher_registry["tasks"]:
        task_id = int(task_row["pilot_task_id"])
        prompt_path = output_dir / f"dataset_task_{task_id}.pkl"
        with prompt_path.open("rb") as stream:
            prompt = pickle.load(stream)
        states = np.asarray(prompt["states"], dtype=np.float32)
        teacher = task_row["selected_teacher"]
        checkpoint_path = Path(teacher["checkpoint"])
        if sha256_file(checkpoint_path) != teacher["checkpoint_sha256"]:
            raise ValueError(f"selected teacher checkpoint changed: {checkpoint_path}")
        actor, std, config = load_actor(checkpoint_path)
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
        task_provenance = {
            "pilot_task_id": task_id,
            "source_task_id": int(task_row["source_task_id"]),
            "teacher_iteration": int(teacher["iteration"]),
            "teacher_checkpoint": str(checkpoint_path),
            "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
            "policy_action_mode": "stochastic",
            "query_seed": task_seed,
            "learned_std_min": float(std.min()),
            "learned_std_mean": float(std.mean()),
            "learned_std_max": float(std.max()),
            "empirical_normalization": bool(config.get("empirical_normalization")),
            "rows": int(states.shape[0]),
            "query_file": str(query_path),
            "query_sha256": sha256_file(query_path),
        }
        provenance["tasks"].append(task_provenance)
        print(
            f"[QUERY] task={task_id} teacher={teacher['iteration']} rows={states.shape[0]} file={query_path}",
            flush=True,
        )
    provenance["created_at"] = utc_now()
    write_json_atomic(output_dir / "query_provenance.json", provenance)
    return provenance


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_pilot = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1/pilot3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=default_pilot)
    parser.add_argument("--phase", choices=("prompt", "evaluate", "query", "validate", "all"), default="all")
    parser.add_argument("--eval-seeds", type=csv_ints, default=[42, 43, 44])
    parser.add_argument("--goal-batch", type=int, default=16)
    parser.add_argument("--query-seed", type=int, default=20260925)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--force-evaluation", action="store_true")
    args = parser.parse_args()
    if args.goal_batch <= 0 or args.query_batch_size <= 0 or args.query_seed < 0:
        parser.error("batch sizes must be positive and query seed non-negative")

    collection_path = args.pilot_dir / "collection_manifest.json"
    collection = load_collection_manifest(collection_path)
    collection["collection_manifest_path"] = str(collection_path)
    output_dir = args.pilot_dir / "dpt"

    if args.phase in ("prompt", "all"):
        build_prompts(collection, output_dir)
    if args.phase in ("evaluate", "all"):
        evaluate_teachers(
            collection,
            output_dir,
            repo,
            args.eval_seeds,
            args.goal_batch,
            args.force_evaluation,
        )
    if args.phase in ("query", "all"):
        teacher_path = output_dir / "teacher_registry.json"
        if not teacher_path.exists():
            raise FileNotFoundError("teacher_registry.json is missing; run --phase evaluate first")
        build_queries(output_dir, json.loads(teacher_path.read_text()), args.query_seed, args.query_batch_size)
    if args.phase in ("validate", "all"):
        validator = repo / "pipeline/dataset/validate_official_mixed_pilot.py"
        subprocess.run(
            [sys.executable, "-u", str(validator), "--pilot-dir", str(args.pilot_dir)],
            cwd=repo,
            check=True,
        )


if __name__ == "__main__":
    main()
