"""Collect restartable checkpoint-balanced RobotLab T2MIR Mixed data.

The formal collector keeps the official T2MIR principles (task-specific
checkpoint sequence, stochastic behavior actions, and equal contribution per
checkpoint) while adapting an episode to a fixed 64-step RobotLab context
window.  Unlike the Pilot collector, windows may start at any valid point in a
goal trajectory.  Selection is stratified by the high-level controller mode so
later ALIGN/HOLD states are not systematically excluded.

Each accepted transition uses the observation that is actually supplied to
the policy at the next decision as ``next_states``.  This matters because the
external GoalController may update the velocity command between two physics
steps.  A valid window therefore requires H+1 consecutive decision states.
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
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pipeline.dataset.dataset_common import (
    ACTION_DIM,
    STATE_DIM,
    TASK_NAME,
    WINDOW_FIELDS,
    git_revision,
    parse_csv_ints,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from pipeline.dataset.collect_official_mixed_pilot import validate_raw_recording
from pipeline.protocols.dynamics_profile import DynamicsProfile, load_manifest


FORMAL_PROVENANCE_FIELDS = (
    "source_attempt_ids",
    "source_env_ids",
    "source_episode_steps",
    "source_next_episode_steps",
    "window_steps",
    "window_start_modes",
)
CONTROLLER_MODES = (0, 1, 2, 3)


def checkpoint_path(repo: Path, registry_task: dict, checkpoint: dict) -> Path:
    run = registry_task.get("training", {}).get("run")
    if not run:
        raise ValueError(f"task {registry_task.get('task_id')} has no registered training run")
    return (repo / "logs/rsl_rl/unitree_g1_flat" / run / checkpoint["checkpoint"]).resolve()


def rollout_checkpoints(registry_task: dict, expected: int) -> list[dict]:
    """Return the non-initial checkpoints in increasing iteration order."""
    checkpoints = sorted(registry_task.get("checkpoints", []), key=lambda row: int(row["iteration"]))
    checkpoints = [row for row in checkpoints if int(row["iteration"]) > 0]
    if len(checkpoints) != expected:
        raise ValueError(
            f"task {registry_task.get('task_id')}: expected {expected} non-initial checkpoints, "
            f"found {len(checkpoints)}"
        )
    iterations = [int(row["iteration"]) for row in checkpoints]
    if len(iterations) != len(set(iterations)):
        raise ValueError(f"task {registry_task.get('task_id')}: duplicate checkpoint iteration")
    return checkpoints


def valid_start_positions(data: dict, horizon: int) -> dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]]:
    """Enumerate H-step windows with H+1 consecutive decision observations.

    The returned mapping is keyed by the controller mode at the first step.
    Values contain ``(env_id, transition_indices, next_decision_indices)``.
    """
    candidates: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {
        mode: [] for mode in CONTROLLER_MODES
    }
    for env_tensor in data["env_ids"].unique(sorted=True):
        env_id = int(env_tensor)
        indices = torch.nonzero(data["env_ids"] == env_tensor, as_tuple=False).squeeze(-1)
        indices = indices[torch.argsort(data["episode_steps"][indices], stable=True)]
        if indices.numel() < horizon + 1:
            continue
        steps = data["episode_steps"][indices].to(torch.long)
        for start in range(0, int(indices.numel()) - horizon):
            transition_indices = indices[start : start + horizon]
            next_indices = indices[start + 1 : start + horizon + 1]
            transition_steps = steps[start : start + horizon]
            next_steps = steps[start + 1 : start + horizon + 1]
            if not torch.equal(next_steps, transition_steps + 1):
                continue
            if bool(data["dones"][transition_indices].any()):
                continue
            if not bool(data["masks"][transition_indices].all()):
                continue
            if bool(data["trajectory_ends"][transition_indices].any()):
                continue
            mode = int(data["controller_modes"][transition_indices[0]])
            if mode not in candidates:
                continue
            candidates[mode].append((env_id, transition_indices, next_indices))
    return candidates


def select_phase_balanced_windows(
    data: dict,
    *,
    horizon: int,
    attempt_id: int,
    limit: int,
    seed: int,
    existing_mode_counts: Counter,
) -> list[dict[str, torch.Tensor]]:
    """Select at most one deterministic, phase-balanced window per environment."""
    by_mode = valid_start_positions(data, horizon)
    by_env: dict[int, dict[int, list[tuple[torch.Tensor, torch.Tensor]]]] = {}
    for mode, rows in by_mode.items():
        for env_id, indices, next_indices in rows:
            by_env.setdefault(env_id, {}).setdefault(mode, []).append((indices, next_indices))

    rng = random.Random(seed)
    env_ids = sorted(by_env)
    rng.shuffle(env_ids)
    selected: list[dict[str, torch.Tensor]] = []
    local_counts = Counter(existing_mode_counts)
    for env_id in env_ids:
        if len(selected) >= limit:
            break
        available_modes = sorted(by_env[env_id])
        if not available_modes:
            continue
        minimum = min(local_counts[mode] for mode in available_modes)
        least_used = [mode for mode in available_modes if local_counts[mode] == minimum]
        mode = rng.choice(least_used)
        indices, next_indices = rng.choice(by_env[env_id][mode])
        window = {key: data[key][indices].clone() for key in WINDOW_FIELDS}
        # Replace the post-physics/pre-command observation by the next decision
        # observation.  Within a window this guarantees next_state[t] == state[t+1].
        window["next_states"] = data["states"][next_indices].clone()
        source_steps = data["episode_steps"][indices].to(torch.long).clone()
        next_steps = data["episode_steps"][next_indices].to(torch.long).clone()
        window["source_attempt_ids"] = torch.full((horizon,), attempt_id, dtype=torch.long)
        window["source_env_ids"] = torch.full((horizon,), env_id, dtype=torch.long)
        window["source_episode_steps"] = source_steps
        window["source_next_episode_steps"] = next_steps
        window["window_steps"] = torch.arange(horizon, dtype=torch.long)
        window["window_start_modes"] = torch.full((horizon,), mode, dtype=torch.long)
        selected.append(window)
        local_counts[mode] += 1
    return selected


def summarize_result_files(paths: list[Path]) -> dict:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise ValueError("no rollout result rows")
    successes = [row for row in rows if row["result"] == "success"]
    falls = [row for row in rows if row["result"] == "fall"]
    returns = np.asarray([float(row["episode_return"]) for row in rows], dtype=np.float64)
    durations = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "successes": len(successes),
        "falls": len(falls),
        "timeouts": len(rows) - len(successes) - len(falls),
        "success_rate": len(successes) / len(rows),
        "fall_rate": len(falls) / len(rows),
        "mean_episode_return": float(returns.mean()),
        "mean_return_per_second": float(np.mean(returns / np.maximum(durations, 1.0e-8))),
        "mean_episode_duration_s": float(durations.mean()),
        "mean_success_time_s": (
            float(np.mean([float(row["time"]) for row in successes])) if successes else None
        ),
        "mean_final_position_error_m": float(
            np.mean([float(row["position_error"]) for row in rows])
        ),
        "mean_final_yaw_error_rad": float(np.mean([float(row["yaw_error"]) for row in rows])),
    }


def consolidate_windows(
    *,
    windows: list[dict[str, torch.Tensor]],
    count: int,
    horizon: int,
    source_task_id: int,
    checkpoint_iteration: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    raw_paths: list[Path],
    result_paths: list[Path],
    action_seeds: list[int],
    output_path: Path,
    profile: DynamicsProfile,
    manifest_sha256: str,
    registry_snapshot_sha256: str,
    robotlab_revision: dict,
    t2mir_revision: dict,
    collection_parameters: dict,
    code_artifacts: dict,
) -> None:
    selected = windows[:count]
    payload = {
        key: torch.cat([window[key] for window in selected], dim=0)
        for key in WINDOW_FIELDS + FORMAL_PROVENANCE_FIELDS
    }
    rows = count * horizon
    payload["window_ids"] = torch.arange(count, dtype=torch.long).repeat_interleave(horizon)
    payload["source_task_ids"] = torch.full((rows,), source_task_id, dtype=torch.long)
    payload["checkpoint_iterations"] = torch.full((rows,), checkpoint_iteration, dtype=torch.long)
    payload["trajectory_ends"] = torch.zeros((rows,), dtype=torch.bool)
    payload["trajectory_ends"][horizon - 1 :: horizon] = True
    mode_counts = Counter(int(window["window_start_modes"][0]) for window in selected)
    payload["metadata"] = {
        "format_version": 2,
        "dataset": "RobotLab-G1-MultiDynamics-official-mixed-v1-formal",
        "context_unit": "continuous_fixed_length_window",
        "next_state_semantics": "next_policy_decision_observation",
        "window_selection": "phase_balanced_one_window_per_environment_per_attempt",
        "policy_role": "prompt",
        "policy_action_mode": "stochastic",
        "source_task_id": source_task_id,
        "profile": profile.__dict__,
        "checkpoint_iteration": checkpoint_iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "registry_snapshot_sha256": registry_snapshot_sha256,
        "windows": count,
        "window_steps": horizon,
        "transitions": rows,
        "window_start_mode_counts": {str(mode): mode_counts.get(mode, 0) for mode in CONTROLLER_MODES},
        "raw_files": [str(path) for path in raw_paths],
        "raw_sha256": [sha256_file(path) for path in raw_paths],
        "result_files": [str(path) for path in result_paths],
        "result_sha256": [sha256_file(path) for path in result_paths],
        "policy_action_seeds": action_seeds,
        "collection_parameters": collection_parameters,
        "code_artifacts": code_artifacts,
        "robotlab_git": robotlab_revision,
        "t2mir_git": t2mir_revision,
        "created_at": utc_now(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)


def accepted_shard_is_valid(
    path: Path,
    *,
    source_task_id: int,
    iteration: int,
    windows: int,
    horizon: int,
    checkpoint_sha256: str,
    registry_snapshot_sha256: str | None = None,
    collection_parameters: dict | None = None,
    code_artifacts: dict | None = None,
) -> bool:
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu")
    metadata = data.get("metadata", {})
    rows = windows * horizon
    required = set(WINDOW_FIELDS + FORMAL_PROVENANCE_FIELDS).union(
        {"window_ids", "source_task_ids", "checkpoint_iterations", "trajectory_ends"}
    )
    if required.difference(data):
        return False
    if not (
        metadata.get("policy_action_mode") == "stochastic"
        and metadata.get("context_unit") == "continuous_fixed_length_window"
        and metadata.get("next_state_semantics") == "next_policy_decision_observation"
        and int(metadata.get("source_task_id", -1)) == source_task_id
        and int(metadata.get("checkpoint_iteration", -1)) == iteration
        and metadata.get("checkpoint_sha256") == checkpoint_sha256
        and int(metadata.get("windows", -1)) == windows
        and int(metadata.get("window_steps", -1)) == horizon
    ):
        return False
    if registry_snapshot_sha256 is not None and metadata.get("registry_snapshot_sha256") != registry_snapshot_sha256:
        return False
    if collection_parameters is not None and metadata.get("collection_parameters") != collection_parameters:
        return False
    if code_artifacts is not None and metadata.get("code_artifacts") != code_artifacts:
        return False
    for key in required:
        value = data[key]
        if not isinstance(value, torch.Tensor) or int(value.shape[0]) != rows:
            return False
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            return False
    if tuple(data["states"].shape) != (rows, STATE_DIM):
        return False
    if tuple(data["policy_actions"].shape) != (rows, ACTION_DIM):
        return False
    if not bool((data["source_task_ids"] == source_task_id).all()):
        return False
    if not bool((data["task_ids"] == source_task_id).all()):
        return False
    if not bool((data["checkpoint_iterations"] == iteration).all()):
        return False
    # The G1 observation contract stores the active velocity command at 9:12.
    if not torch.allclose(data["states"][:, 9:12], data["commands"], atol=1.0e-6, rtol=0.0):
        return False
    if bool(data["dones"].any()) or not bool(data["masks"].all()):
        return False
    expected_ends = torch.zeros(rows, dtype=torch.bool)
    expected_ends[horizon - 1 :: horizon] = True
    if not torch.equal(data["trajectory_ends"].to(torch.bool), expected_ends):
        return False
    for window_id in range(windows):
        start, stop = window_id * horizon, (window_id + 1) * horizon
        if not torch.equal(data["window_steps"][start:stop], torch.arange(horizon)):
            return False
        if not torch.equal(
            data["source_next_episode_steps"][start:stop],
            data["source_episode_steps"][start:stop] + 1,
        ):
            return False
        if not torch.equal(data["next_states"][start : stop - 1], data["states"][start + 1 : stop]):
            return False
        start_mode = data["controller_modes"][start]
        if not bool((data["window_start_modes"][start:stop] == start_mode).all()):
            return False
        lag = data["action_lags"][start:stop]
        if not bool(torch.allclose(lag, lag[:1].expand_as(lag), atol=1.0e-7, rtol=0.0)):
            return False
        expected_executed = (1.0 - lag[1:, None]) * data["policy_actions"][start + 1 : stop]
        expected_executed += lag[1:, None] * data["executed_actions"][start : stop - 1]
        if not torch.allclose(
            data["executed_actions"][start + 1 : stop], expected_executed, atol=1.0e-6, rtol=0.0
        ):
            return False
    return True


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    namespace = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument("--registry", type=Path, default=namespace / "checkpoint_bank/pilot_registry.json")
    parser.add_argument("--output-dir", type=Path, default=namespace / "formal_v1")
    parser.add_argument(
        "--source-task-ids",
        type=parse_csv_ints,
        default=None,
        help="Training source IDs. Default: every train profile in the manifest.",
    )
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

    positive = (
        args.checkpoints_per_task,
        args.windows_per_checkpoint,
        args.window_steps,
        args.attempt_batch,
        args.max_attempts,
    )
    if min(positive) <= 0:
        parser.error("checkpoint, window, horizon, batch and attempt counts must be positive")
    if args.goal_min_distance >= args.goal_max_distance:
        parser.error("--goal-min-distance must be less than --goal-max-distance")

    manifest_payload, profiles = load_manifest(args.manifest)
    profiles_by_id = {profile.task_id: profile for profile in profiles}
    train_ids = [profile.task_id for profile in profiles if profile.split == "train"]
    source_task_ids = args.source_task_ids or train_ids
    unknown = sorted(set(source_task_ids).difference(profiles_by_id))
    held_out = [task_id for task_id in source_task_ids if profiles_by_id[task_id].split != "train"]
    if unknown:
        parser.error(f"unknown source task IDs: {unknown}")
    if held_out:
        parser.error(f"held-out task IDs cannot be collected: {held_out}")

    source_registry = read_json(args.registry)
    if source_registry.get("manifest_sha256") != manifest_payload["sha256"]:
        raise ValueError("checkpoint registry and dynamics manifest SHA256 do not match")
    source_registry_by_id = {int(row["task_id"]): row for row in source_registry.get("tasks", [])}
    source_missing = sorted(set(source_task_ids).difference(source_registry_by_id))
    if source_missing:
        raise ValueError(f"registry is missing completed source tasks {source_missing}")
    # Validate the complete immutable bank before creating any output metadata.
    # This prevents an accidental early invocation from freezing a poisoned
    # registry snapshot while its PPO task is still training.
    for source_task_id in source_task_ids:
        registry_task = source_registry_by_id[source_task_id]
        if registry_task.get("training", {}).get("method") != "fresh_task_specific_ppo":
            raise ValueError(f"source task {source_task_id} is not a fresh task-specific PPO")
        for checkpoint_row in rollout_checkpoints(registry_task, args.checkpoints_per_task):
            checkpoint = checkpoint_path(repo, registry_task, checkpoint_row)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            registered_hash = checkpoint_row.get("checkpoint_sha256")
            if registered_hash is None or sha256_file(checkpoint) != registered_hash:
                raise ValueError(f"registered checkpoint hash mismatch: {checkpoint}")

    snapshot_path = args.output_dir / "registry_snapshot.json"
    if snapshot_path.exists() and not args.force:
        registry = read_json(snapshot_path)
    else:
        registry = source_registry
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(snapshot_path, registry)
    if registry.get("manifest_sha256") != manifest_payload["sha256"]:
        raise ValueError("checkpoint registry and dynamics manifest SHA256 do not match")
    registry_sha256 = sha256_file(snapshot_path)
    registry_by_id = {int(row["task_id"]): row for row in registry.get("tasks", [])}
    missing = sorted(set(source_task_ids).difference(registry_by_id))
    if missing:
        raise ValueError(f"registry snapshot is missing source tasks {missing}")

    robotlab_revision = git_revision(repo)
    t2mir_revision = git_revision(repo / "methods/t2mir")
    collection_parameters = {
        "base_seed": args.base_seed,
        "attempt_batch": args.attempt_batch,
        "max_attempts": args.max_attempts,
        "goal_min_distance": args.goal_min_distance,
        "goal_max_distance": args.goal_max_distance,
        "goal_timeout": args.goal_timeout,
        "goal_hold_time": args.goal_hold_time,
    }
    artifact_paths = {
        "collector": Path(__file__).resolve(),
        "play": repo / "pipeline/expert/play.py",
        "goal_controller": repo / "pipeline/protocols/goal_controller.py",
        "dynamics_profile": repo / "pipeline/protocols/dynamics_profile.py",
    }
    code_artifacts = {name: sha256_file(path) for name, path in artifact_paths.items()}
    manifest_path = args.output_dir / "collection_manifest.json"
    plan = {
        "format_version": 2,
        "dataset": "RobotLab-G1-MultiDynamics-official-mixed-v1-formal",
        "status": "planned" if args.dry_run else "collecting",
        "created_at": utc_now(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_payload["sha256"],
        "registry_source": str(args.registry.resolve()),
        "registry_snapshot": str(snapshot_path.resolve()),
        "registry_snapshot_sha256": registry_sha256,
        "source_task_ids": source_task_ids,
        "held_out_task_ids": [profile.task_id for profile in profiles if profile.split != "train"],
        "checkpoints_per_task": args.checkpoints_per_task,
        "windows_per_checkpoint": args.windows_per_checkpoint,
        "window_steps": args.window_steps,
        "context_unit": "continuous_fixed_length_window",
        "next_state_semantics": "next_policy_decision_observation",
        "policy_action_mode": "stochastic",
        "collection_parameters": collection_parameters,
        "code_artifacts": code_artifacts,
        "robotlab_git": robotlab_revision,
        "t2mir_git": t2mir_revision,
        "shards": [],
    }
    if manifest_path.exists() and not args.force:
        previous = read_json(manifest_path)
        immutable_keys = (
            "dataset",
            "manifest_sha256",
            "registry_snapshot_sha256",
            "source_task_ids",
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
        mismatches = [key for key in immutable_keys if previous.get(key) != plan.get(key)]
        if mismatches:
            raise ValueError(
                f"existing formal output has incompatible immutable fields {mismatches}; "
                "use a new output directory"
            )
        plan["created_at"] = previous.get("created_at", plan["created_at"])
    write_json_atomic(manifest_path, plan)

    play_py = repo / "pipeline/expert/play.py"
    for source_task_id in source_task_ids:
        profile = profiles_by_id[source_task_id]
        registry_task = registry_by_id[source_task_id]
        if registry_task.get("training", {}).get("method") != "fresh_task_specific_ppo":
            raise ValueError(f"source task {source_task_id} is not a fresh task-specific PPO")
        run_name = registry_task["training"]["run"]
        checkpoints = rollout_checkpoints(registry_task, args.checkpoints_per_task)
        for checkpoint_rank, checkpoint_row in enumerate(checkpoints):
            iteration = int(checkpoint_row["iteration"])
            checkpoint = checkpoint_path(repo, registry_task, checkpoint_row)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_hash = sha256_file(checkpoint)
            registered_hash = checkpoint_row.get("checkpoint_sha256")
            if registered_hash and registered_hash != checkpoint_hash:
                raise ValueError(f"registered checkpoint hash mismatch: {checkpoint}")
            shard_dir = args.output_dir / "raw" / f"task{source_task_id:02d}" / f"iteration_{iteration:06d}"
            accepted_path = shard_dir / "accepted.pt"
            record = {
                "source_task_id": source_task_id,
                "profile": profile.__dict__,
                "checkpoint_rank": checkpoint_rank,
                "checkpoint_iteration": iteration,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "accepted_shard": str(accepted_path),
            }
            plan["shards"].append(record)
            write_json_atomic(manifest_path, plan)

            if accepted_path.exists() and not args.force:
                if not accepted_shard_is_valid(
                    accepted_path,
                    source_task_id=source_task_id,
                    iteration=iteration,
                    windows=args.windows_per_checkpoint,
                    horizon=args.window_steps,
                    checkpoint_sha256=checkpoint_hash,
                    registry_snapshot_sha256=registry_sha256,
                    collection_parameters=collection_parameters,
                    code_artifacts=code_artifacts,
                ):
                    raise ValueError(f"existing accepted shard is incompatible or corrupt: {accepted_path}")
                accepted = torch.load(accepted_path, map_location="cpu")
                record.update(
                    status="reused",
                    accepted_sha256=sha256_file(accepted_path),
                    window_start_mode_counts=accepted["metadata"]["window_start_mode_counts"],
                    collection_metrics=summarize_result_files(
                        [Path(path) for path in accepted["metadata"]["result_files"]]
                    ),
                )
                continue

            collected: list[dict[str, torch.Tensor]] = []
            mode_counts: Counter = Counter()
            raw_paths: list[Path] = []
            result_paths: list[Path] = []
            action_seeds: list[int] = []
            for attempt in range(args.max_attempts):
                action_seed = args.base_seed + source_task_id * 100000 + checkpoint_rank * 1000 + attempt
                attempt_path = shard_dir / f"attempt_{attempt:03d}.pt"
                result_path = shard_dir / f"attempt_{attempt:03d}.csv"
                command = [
                    sys.executable,
                    "-u",
                    str(play_py),
                    "--task",
                    TASK_NAME,
                    "--headless",
                    "--load_run",
                    run_name,
                    "--checkpoint",
                    checkpoint.name,
                    "--goal_batch",
                    str(args.attempt_batch),
                    "--goal_min_distance",
                    str(args.goal_min_distance),
                    "--goal_max_distance",
                    str(args.goal_max_distance),
                    "--goal_timeout",
                    str(args.goal_timeout),
                    "--goal_hold_time",
                    str(args.goal_hold_time),
                    "--goal_seed",
                    str(action_seed),
                    "--stochastic_policy",
                    "--policy_action_seed",
                    str(action_seed),
                    "--skip_policy_export",
                    "--record_trajectories",
                    "--trajectory_task_id",
                    str(source_task_id),
                    "--trajectory_policy_role",
                    "prompt",
                    "--trajectory_output",
                    str(attempt_path),
                    "--goal_results",
                    str(result_path),
                    *profile.cli_args(),
                ]
                print(" ".join(command), flush=True)
                if args.dry_run:
                    continue
                shard_dir.mkdir(parents=True, exist_ok=True)
                if args.force or not attempt_path.exists() or not result_path.exists():
                    subprocess.run(command, cwd=repo, check=True)
                data = validate_raw_recording(
                    attempt_path,
                    checkpoint=checkpoint,
                    pilot_task_id=source_task_id,
                    profile=profile,
                    action_seed=action_seed,
                    attempt_batch=args.attempt_batch,
                )
                remaining = args.windows_per_checkpoint - len(collected)
                chosen = select_phase_balanced_windows(
                    data,
                    horizon=args.window_steps,
                    attempt_id=attempt,
                    limit=remaining,
                    seed=action_seed,
                    existing_mode_counts=mode_counts,
                )
                collected.extend(chosen)
                mode_counts.update(int(window["window_start_modes"][0]) for window in chosen)
                raw_paths.append(attempt_path)
                result_paths.append(result_path)
                action_seeds.append(action_seed)
                record.update(
                    status="collecting",
                    attempts_examined=attempt + 1,
                    valid_windows=len(collected),
                    window_start_mode_counts={str(mode): mode_counts.get(mode, 0) for mode in CONTROLLER_MODES},
                )
                write_json_atomic(manifest_path, plan)
                print(
                    f"[FORMAL] task={source_task_id} iteration={iteration} "
                    f"windows={len(collected)}/{args.windows_per_checkpoint} modes={dict(mode_counts)}",
                    flush=True,
                )
                if len(collected) >= args.windows_per_checkpoint:
                    break

            if args.dry_run:
                record["status"] = "planned"
                continue
            if len(collected) < args.windows_per_checkpoint:
                record.update(
                    status="failed",
                    failure=(
                        f"only {len(collected)}/{args.windows_per_checkpoint} valid "
                        f"{args.window_steps}-step windows"
                    ),
                )
                plan["status"] = "failed"
                write_json_atomic(manifest_path, plan)
                raise RuntimeError(record["failure"])
            consolidate_windows(
                windows=collected,
                count=args.windows_per_checkpoint,
                horizon=args.window_steps,
                source_task_id=source_task_id,
                checkpoint_iteration=iteration,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_hash,
                raw_paths=raw_paths,
                result_paths=result_paths,
                action_seeds=action_seeds,
                output_path=accepted_path,
                profile=profile,
                manifest_sha256=manifest_payload["sha256"],
                registry_snapshot_sha256=registry_sha256,
                robotlab_revision=robotlab_revision,
                t2mir_revision=t2mir_revision,
                collection_parameters=collection_parameters,
                code_artifacts=code_artifacts,
            )
            record.update(
                status="complete",
                accepted_sha256=sha256_file(accepted_path),
                raw_attempts=len(raw_paths),
                window_start_mode_counts={str(mode): mode_counts.get(mode, 0) for mode in CONTROLLER_MODES},
                collection_metrics=summarize_result_files(result_paths),
            )
            write_json_atomic(manifest_path, plan)

    if not args.dry_run:
        if not all(row.get("status") in {"complete", "reused"} for row in plan["shards"]):
            raise RuntimeError("collection ended with incomplete shards")
        aggregate_modes = Counter()
        for row in plan["shards"]:
            aggregate_modes.update({int(key): int(value) for key, value in row["window_start_mode_counts"].items()})
        plan["window_start_mode_counts"] = {
            str(mode): aggregate_modes.get(mode, 0) for mode in CONTROLLER_MODES
        }
        plan["status"] = "complete"
        plan["completed_at"] = utc_now()
    write_json_atomic(manifest_path, plan)
    print(f"[FORMAL] collection manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
