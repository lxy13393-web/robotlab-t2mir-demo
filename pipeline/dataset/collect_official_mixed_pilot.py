"""Collect a small, restartable official-Mixed contract-test dataset.

This script deliberately collects only a three-task pilot.  It reads the
checkpoint bank registry, selects early/middle/late training checkpoints by
iteration quantile, invokes ``play.py`` with stochastic PPO actions, and keeps
an equal number of strict 64-step windows from every selected checkpoint.

It does not create the final T2MIR pickle files.  The accepted shards retain
episode/checkpoint provenance so a later builder and validator can prove that
flattening does not cross a reset, termination, or checkpoint boundary.
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
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch

from pipeline.protocols.dynamics_profile import DynamicsProfile, load_manifest
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


DEFAULT_SOURCE_TASKS = (0, 27, 46)
RAW_REQUIRED_FIELDS = WINDOW_FIELDS + ("trajectory_ends",)
WINDOW_PROVENANCE_FIELDS = ("source_attempt_ids", "source_env_ids")


def parse_csv_floats(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(not 0.0 <= item <= 1.0 for item in result):
        raise argparse.ArgumentTypeError("selection fractions must be in [0, 1]")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("selection fractions may not contain duplicates")
    return result


def select_by_fractions(checkpoints: list[dict], fractions: Iterable[float]) -> list[dict]:
    ordered = sorted(checkpoints, key=lambda row: int(row["iteration"]))
    if not ordered:
        raise ValueError("checkpoint bank is empty")
    selected = []
    used = set()
    for fraction in fractions:
        index = round(fraction * (len(ordered) - 1))
        row = ordered[index]
        iteration = int(row["iteration"])
        if iteration in used:
            raise ValueError(
                f"selection fractions map to duplicate checkpoint iteration {iteration}; "
                "choose more widely separated fractions"
            )
        used.add(iteration)
        selected.append(row)
    return selected


def validate_raw_recording(
    path: Path,
    *,
    checkpoint: Path,
    pilot_task_id: int,
    profile: DynamicsProfile,
    action_seed: int,
    attempt_batch: int,
) -> dict:
    data = torch.load(path, map_location="cpu")
    missing = set(RAW_REQUIRED_FIELDS).difference(data)
    if missing:
        raise ValueError(f"{path}: missing trajectory fields {sorted(missing)}")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing trajectory metadata")
    if metadata.get("policy_role") != "prompt":
        raise ValueError(f"{path}: expected prompt policy role")
    if metadata.get("policy_action_mode") != "stochastic":
        raise ValueError(f"{path}: expected stochastic policy actions")
    if int(metadata.get("policy_action_seed", -1)) != action_seed:
        raise ValueError(f"{path}: policy action seed mismatch")
    if int(metadata.get("goal_seed", -1)) != action_seed:
        raise ValueError(f"{path}: goal seed mismatch")
    if int(metadata.get("num_envs", -1)) != attempt_batch:
        raise ValueError(f"{path}: environment batch size mismatch")
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError(f"{path}: checkpoint provenance mismatch")
    if int(metadata.get("task_id", -1)) != pilot_task_id:
        raise ValueError(f"{path}: pilot task ID mismatch")
    expected_dynamics = {
        "action_lag": profile.action_lag,
        "motor_strength": profile.motor_strength,
        "payload_kg": profile.payload_kg,
        "friction": profile.friction,
    }
    for key, expected_value in expected_dynamics.items():
        actual_value = metadata.get(key)
        if actual_value is None or abs(float(actual_value) - expected_value) > 1.0e-8:
            raise ValueError(f"{path}: {key} provenance mismatch")
    if metadata.get("deterministic_dynamics") is not True:
        raise ValueError(f"{path}: deterministic dynamics were not enabled")
    count = int(data["states"].shape[0])
    if data["states"].ndim != 2 or int(data["states"].shape[1]) != STATE_DIM:
        raise ValueError(f"{path}: expected state dimension {STATE_DIM}, got {tuple(data['states'].shape)}")
    if data["policy_actions"].ndim != 2 or int(data["policy_actions"].shape[1]) != ACTION_DIM:
        raise ValueError(
            f"{path}: expected action dimension {ACTION_DIM}, got {tuple(data['policy_actions'].shape)}"
        )
    for key in WINDOW_FIELDS:
        value = data[key]
        if not isinstance(value, torch.Tensor) or value.shape[0] != count:
            raise ValueError(f"{path}: invalid {key} tensor")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{path}: {key} contains NaN or Inf")
    if not bool((data["task_ids"] == pilot_task_id).all()):
        raise ValueError(f"{path}: transition task IDs do not match the pilot task")
    return data


def valid_windows(data: dict, horizon: int, attempt_id: int) -> list[dict[str, torch.Tensor]]:
    """Return at most one strict prefix window from each recorded environment."""
    windows: list[dict[str, torch.Tensor]] = []
    for env_id in data["env_ids"].unique(sorted=True):
        indices = torch.nonzero(data["env_ids"] == env_id, as_tuple=False).squeeze(-1)
        order = torch.argsort(data["episode_steps"][indices], stable=True)
        indices = indices[order]
        if indices.numel() < horizon:
            continue
        indices = indices[:horizon]
        steps = data["episode_steps"][indices].to(torch.long)
        expected = torch.arange(horizon, dtype=torch.long)
        if not torch.equal(steps.cpu(), expected):
            continue
        # Isaac's vector wrapper may expose the reset observation after a done.
        # Retain only fully non-terminal windows so every next_state belongs to
        # the same physical episode as its state/action/reward transition.
        if bool(data["dones"][indices].any()):
            continue
        if bool(data["trajectory_ends"][indices[:-1]].any()):
            continue
        if not bool(data["masks"][indices].all()):
            continue
        window = {key: data[key][indices].clone() for key in WINDOW_FIELDS}
        window["source_attempt_ids"] = torch.full((horizon,), attempt_id, dtype=torch.long)
        window["source_env_ids"] = data["env_ids"][indices].clone().to(torch.long)
        windows.append(window)
    return windows


def consolidate_windows(
    *,
    windows: list[dict[str, torch.Tensor]],
    episodes: int,
    horizon: int,
    pilot_task_id: int,
    source_task_id: int,
    checkpoint_iteration: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    raw_paths: list[Path],
    raw_sha256: list[str],
    action_seeds: list[int],
    output_path: Path,
    profile: DynamicsProfile,
    manifest_sha256: str,
    robotlab_revision: dict,
    t2mir_revision: dict,
) -> None:
    selected = windows[:episodes]
    payload = {
        key: torch.cat([window[key] for window in selected], dim=0)
        for key in WINDOW_FIELDS + WINDOW_PROVENANCE_FIELDS
    }
    episode_ids = torch.arange(episodes, dtype=torch.long).repeat_interleave(horizon)
    payload["episode_ids"] = episode_ids
    payload["source_task_ids"] = torch.full((episodes * horizon,), source_task_id, dtype=torch.long)
    payload["pilot_task_ids"] = torch.full((episodes * horizon,), pilot_task_id, dtype=torch.long)
    payload["checkpoint_iterations"] = torch.full(
        (episodes * horizon,), checkpoint_iteration, dtype=torch.long
    )
    payload["trajectory_ends"] = torch.zeros((episodes * horizon,), dtype=torch.bool)
    payload["trajectory_ends"][horizon - 1 :: horizon] = True
    payload["metadata"] = {
        "format_version": 1,
        "dataset": "RobotLab-G1-MultiDynamics-official-mixed-v1-pilot3",
        "policy_role": "prompt",
        "policy_action_mode": "stochastic",
        "pilot_task_id": pilot_task_id,
        "source_task_id": source_task_id,
        "profile": profile.__dict__,
        "checkpoint_iteration": checkpoint_iteration,
        "checkpoint": str(checkpoint_path.relative_to(checkpoint_path.parents[3])),
        "checkpoint_absolute": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "episodes": episodes,
        "episode_steps": horizon,
        "transitions": episodes * horizon,
        "raw_files": [str(path) for path in raw_paths],
        "raw_sha256": raw_sha256,
        "policy_action_seeds": action_seeds,
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
    pilot_task_id: int,
    iteration: int,
    episodes: int,
    horizon: int,
    checkpoint_sha256: str,
) -> bool:
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu")
    metadata = data.get("metadata", {})
    expected_count = episodes * horizon
    basic_valid = (
        metadata.get("policy_action_mode") == "stochastic"
        and int(metadata.get("source_task_id", -1)) == source_task_id
        and int(metadata.get("pilot_task_id", -1)) == pilot_task_id
        and int(metadata.get("checkpoint_iteration", -1)) == iteration
        and metadata.get("checkpoint_sha256") == checkpoint_sha256
        and int(metadata.get("episodes", -1)) == episodes
        and int(metadata.get("episode_steps", -1)) == horizon
        and data.get("states") is not None
        and int(data["states"].shape[0]) == expected_count
        and int(data.get("trajectory_ends", torch.empty(0)).sum()) == episodes
    )
    if not basic_valid:
        return False
    required = set(WINDOW_FIELDS + WINDOW_PROVENANCE_FIELDS).union(
        {"episode_ids", "source_task_ids", "pilot_task_ids", "checkpoint_iterations", "trajectory_ends"}
    )
    if required.difference(data):
        return False
    for key in required:
        value = data[key]
        if not isinstance(value, torch.Tensor) or int(value.shape[0]) != expected_count:
            return False
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            return False
    if tuple(data["states"].shape) != (expected_count, STATE_DIM):
        return False
    if tuple(data["policy_actions"].shape) != (expected_count, ACTION_DIM):
        return False
    expected_ends = torch.zeros(expected_count, dtype=torch.bool)
    expected_ends[horizon - 1 :: horizon] = True
    if not torch.equal(data["trajectory_ends"].to(torch.bool), expected_ends):
        return False
    if bool(data["dones"].any()) or not bool(data["masks"].all()):
        return False
    for episode_id in range(episodes):
        start, stop = episode_id * horizon, (episode_id + 1) * horizon
        if not torch.equal(data["episode_steps"][start:stop].to(torch.long), torch.arange(horizon)):
            return False
        if not bool((data["episode_ids"][start:stop] == episode_id).all()):
            return False
    return True


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    namespace = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument(
        "--registry", type=Path, default=namespace / "checkpoint_bank/pilot_registry.json"
    )
    parser.add_argument("--output-dir", type=Path, default=namespace / "pilot3")
    parser.add_argument("--source-task-ids", type=parse_csv_ints, default=list(DEFAULT_SOURCE_TASKS))
    parser.add_argument("--selection-fractions", type=parse_csv_floats, default=[0.25, 0.50, 1.0])
    parser.add_argument("--episodes-per-checkpoint", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--attempt-batch", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260924)
    parser.add_argument("--goal-min-distance", type=float, default=1.0)
    parser.add_argument("--goal-max-distance", type=float, default=4.0)
    parser.add_argument("--goal-timeout", type=float, default=40.0)
    parser.add_argument("--goal-hold-time", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="Replace accepted shards; raw attempts are retained")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help=(
            "Record insufficient-window checkpoints and continue instead of failing fast. "
            "Use only for checkpoint viability screening, never for formal dataset acceptance."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned play.py commands only")
    args = parser.parse_args()

    if min(args.episodes_per_checkpoint, args.episode_steps, args.attempt_batch, args.max_attempts) <= 0:
        parser.error("episode, horizon, batch, and attempt counts must be positive")
    if args.goal_min_distance >= args.goal_max_distance:
        parser.error("--goal-min-distance must be less than --goal-max-distance")
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")

    manifest_payload, profiles = load_manifest(args.manifest)
    profiles_by_id = {profile.task_id: profile for profile in profiles}
    missing_profiles = sorted(set(args.source_task_ids).difference(profiles_by_id))
    if missing_profiles:
        parser.error(f"unknown source task IDs: {missing_profiles}")
    held_out = [task_id for task_id in args.source_task_ids if profiles_by_id[task_id].split != "train"]
    if held_out:
        parser.error(f"pilot source tasks must be training tasks, got held-out IDs {held_out}")

    registry = read_json(args.registry)
    if registry.get("manifest_sha256") != manifest_payload["sha256"]:
        raise ValueError("checkpoint registry and dynamics manifest SHA256 do not match")
    registry_by_id = {int(row["task_id"]): row for row in registry.get("tasks", [])}
    missing_registry = sorted(set(args.source_task_ids).difference(registry_by_id))
    if missing_registry:
        raise ValueError(f"checkpoint registry is missing source tasks {missing_registry}")

    task_map = {pilot_id: source_id for pilot_id, source_id in enumerate(args.source_task_ids)}
    robotlab_revision = git_revision(repo)
    t2mir_revision = git_revision(repo / "methods/t2mir")
    plan = {
        "format_version": 1,
        "dataset": "RobotLab-G1-MultiDynamics-official-mixed-v1-pilot3",
        "status": "planned" if args.dry_run else "collecting",
        "mode": "probe" if args.probe_only else "collection",
        "created_at": utc_now(),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_payload["sha256"],
        "registry": str(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "pilot_to_source_task": {str(key): value for key, value in task_map.items()},
        "selection_fractions": args.selection_fractions,
        "episodes_per_checkpoint": args.episodes_per_checkpoint,
        "episode_steps": args.episode_steps,
        "policy_action_mode": "stochastic",
        "robotlab_git": robotlab_revision,
        "t2mir_git": t2mir_revision,
        "shards": [],
    }
    write_json_atomic(args.output_dir / "collection_manifest.json", plan)

    play_py = repo / "pipeline/expert/play.py"
    log_root = repo / "logs/rsl_rl/unitree_g1_flat"
    for pilot_task_id, source_task_id in task_map.items():
        profile = profiles_by_id[source_task_id]
        entry = registry_by_id[source_task_id]
        run_name = entry.get("training", {}).get("run")
        if not run_name:
            raise ValueError(f"source task {source_task_id} has no registered training run")
        if entry.get("training", {}).get("method") != "fresh_task_specific_ppo":
            raise ValueError(f"source task {source_task_id} was not trained as a fresh task-specific PPO")
        selected_checkpoints = select_by_fractions(entry.get("checkpoints", []), args.selection_fractions)
        for checkpoint_rank, checkpoint_row in enumerate(selected_checkpoints):
            iteration = int(checkpoint_row["iteration"])
            checkpoint_name = str(checkpoint_row["checkpoint"])
            checkpoint_path = (log_root / run_name / checkpoint_name).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            checkpoint_hash = sha256_file(checkpoint_path)
            shard_dir = (
                args.output_dir
                / "raw"
                / f"task{pilot_task_id:02d}_source{source_task_id:02d}"
                / f"iteration_{iteration:06d}"
            )
            accepted_path = shard_dir / "accepted.pt"
            shard_record = {
                "pilot_task_id": pilot_task_id,
                "source_task_id": source_task_id,
                "profile": profile.__dict__,
                "selection_fraction": args.selection_fractions[checkpoint_rank],
                "checkpoint_iteration": iteration,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "accepted_shard": str(accepted_path),
            }
            plan["shards"].append(shard_record)
            write_json_atomic(args.output_dir / "collection_manifest.json", plan)

            if accepted_path.exists() and not args.force:
                if not accepted_shard_is_valid(
                    accepted_path,
                    source_task_id=source_task_id,
                    pilot_task_id=pilot_task_id,
                    iteration=iteration,
                    episodes=args.episodes_per_checkpoint,
                    horizon=args.episode_steps,
                    checkpoint_sha256=checkpoint_hash,
                ):
                    raise ValueError(f"existing accepted shard is incompatible or corrupt: {accepted_path}")
                shard_record["status"] = "reused"
                shard_record["accepted_sha256"] = sha256_file(accepted_path)
                print(f"[PILOT] reuse {accepted_path}", flush=True)
                continue

            raw_paths: list[Path] = []
            raw_hashes: list[str] = []
            action_seeds: list[int] = []
            windows: list[dict[str, torch.Tensor]] = []
            for attempt in range(args.max_attempts):
                action_seed = args.base_seed + pilot_task_id * 100000 + checkpoint_rank * 1000 + attempt
                attempt_path = shard_dir / f"attempt_{attempt:03d}.pt"
                results_path = shard_dir / f"attempt_{attempt:03d}.csv"
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
                    checkpoint_name,
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
                    str(pilot_task_id),
                    "--trajectory_policy_role",
                    "prompt",
                    "--trajectory_output",
                    str(attempt_path),
                    "--goal_results",
                    str(results_path),
                    *profile.cli_args(),
                ]
                print(" ".join(command), flush=True)
                if args.dry_run:
                    continue
                shard_dir.mkdir(parents=True, exist_ok=True)
                if not attempt_path.exists():
                    try:
                        subprocess.run(command, cwd=repo, check=True)
                    except subprocess.CalledProcessError as error:
                        shard_record["status"] = "failed"
                        shard_record["failure"] = f"pipeline.expert.play.py exited with status {error.returncode}"
                        plan["status"] = "failed"
                        write_json_atomic(args.output_dir / "collection_manifest.json", plan)
                        raise
                data = validate_raw_recording(
                    attempt_path,
                    checkpoint=checkpoint_path,
                    pilot_task_id=pilot_task_id,
                    profile=profile,
                    action_seed=action_seed,
                    attempt_batch=args.attempt_batch,
                )
                raw_paths.append(attempt_path)
                raw_hashes.append(sha256_file(attempt_path))
                action_seeds.append(action_seed)
                windows.extend(valid_windows(data, args.episode_steps, attempt))
                shard_record["status"] = "collecting"
                shard_record["attempts_examined"] = attempt + 1
                shard_record["valid_windows"] = len(windows)
                write_json_atomic(args.output_dir / "collection_manifest.json", plan)
                print(
                    f"[PILOT] source_task={source_task_id} iteration={iteration} "
                    f"valid_windows={len(windows)}/{args.episodes_per_checkpoint}",
                    flush=True,
                )
                if len(windows) >= args.episodes_per_checkpoint:
                    break

            if args.dry_run:
                shard_record["status"] = "planned"
                continue
            if len(windows) < args.episodes_per_checkpoint:
                shard_record["status"] = "failed"
                shard_record["failure"] = (
                    f"only {len(windows)}/{args.episodes_per_checkpoint} valid "
                    f"{args.episode_steps}-step windows"
                )
                shard_record["raw_attempts"] = len(raw_paths)
                if args.probe_only:
                    print(
                        f"[PROBE] source_task={source_task_id} iteration={iteration} "
                        f"FAILED: {shard_record['failure']}",
                        flush=True,
                    )
                    write_json_atomic(args.output_dir / "collection_manifest.json", plan)
                    continue
                plan["status"] = "failed"
                write_json_atomic(args.output_dir / "collection_manifest.json", plan)
                raise RuntimeError(
                    f"source task {source_task_id}, checkpoint {iteration}: obtained only "
                    f"{len(windows)}/{args.episodes_per_checkpoint} valid {args.episode_steps}-step "
                    f"windows after {args.max_attempts} attempts; raw attempts were retained in {shard_dir}"
                )
            consolidate_windows(
                windows=windows,
                episodes=args.episodes_per_checkpoint,
                horizon=args.episode_steps,
                pilot_task_id=pilot_task_id,
                source_task_id=source_task_id,
                checkpoint_iteration=iteration,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                raw_paths=raw_paths,
                raw_sha256=raw_hashes,
                action_seeds=action_seeds,
                output_path=accepted_path,
                profile=profile,
                manifest_sha256=manifest_payload["sha256"],
                robotlab_revision=robotlab_revision,
                t2mir_revision=t2mir_revision,
            )
            shard_record["status"] = "complete"
            shard_record["accepted_sha256"] = sha256_file(accepted_path)
            shard_record["raw_attempts"] = len(raw_paths)
            print(f"[PILOT] wrote {accepted_path}", flush=True)
            write_json_atomic(args.output_dir / "collection_manifest.json", plan)

    if not args.dry_run:
        failed = sum(row.get("status") == "failed" for row in plan["shards"])
        complete = sum(row.get("status") in {"complete", "reused"} for row in plan["shards"])
        plan["probe_summary"] = {
            "checkpoints": len(plan["shards"]),
            "viable": complete,
            "non_viable": failed,
        }
        plan["status"] = "probe_complete" if args.probe_only else "complete"
        plan["completed_at"] = utc_now()
    write_json_atomic(args.output_dir / "collection_manifest.json", plan)
    print(f"[PILOT] collection manifest: {args.output_dir / 'collection_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
