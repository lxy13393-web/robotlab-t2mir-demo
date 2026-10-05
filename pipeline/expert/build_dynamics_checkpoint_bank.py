"""Build and evaluate official-style task-specific PPO checkpoint banks.

Each training task starts from a fresh PPO initialization under one fixed
dynamics profile. Periodic checkpoints are retained as behavior policies; the
script does not collapse them into one mature expert. Evaluation is a separate,
restartable phase so simulator failures do not invalidate completed training.
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
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline.protocols.dynamics_profile import DynamicsProfile, load_manifest


TASK_NAME = "RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0"


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("task/seed lists may not contain duplicates")
    return values


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as error:
        raise ValueError(f"invalid RSL-RL checkpoint name: {path.name}") from error


def checkpoints_in(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("model_*.pt"), key=checkpoint_iteration)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: dict) -> dict:
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def select_profiles(
    parser: argparse.ArgumentParser,
    manifest_path: Path,
    task_ids: list[int],
    allow_eval_tasks: bool,
) -> tuple[dict, list[DynamicsProfile]]:
    manifest, all_profiles = load_manifest(manifest_path)
    by_id = {profile.task_id: profile for profile in all_profiles}
    missing = sorted(set(task_ids).difference(by_id))
    if missing:
        parser.error(f"unknown task IDs: {missing}")
    profiles = [by_id[task_id] for task_id in task_ids]
    held_out = [profile.task_id for profile in profiles if profile.split == "eval"]
    if held_out and not allow_eval_tasks:
        parser.error(
            f"refusing to train held-out evaluation tasks {held_out}; "
            "pass --allow-eval-tasks only for a non-primary diagnostic"
        )
    return manifest, profiles


def matching_runs(log_root: Path, run_name: str) -> list[Path]:
    return sorted(path for path in log_root.glob(f"*_{run_name}") if path.is_dir())


def validate_run_dynamics(run_dir: Path, profile: DynamicsProfile) -> None:
    """Refuse to register a run whose saved dynamics differ from its task profile."""
    path = run_dir / "params/dynamics.yaml"
    if not path.exists():
        raise RuntimeError(f"missing saved dynamics provenance: {path}")
    saved = yaml.safe_load(path.read_text())
    expected = {
        "action_lag": profile.action_lag,
        "motor_strength": profile.motor_strength,
        "payload_kg": profile.payload_kg,
        "friction": profile.friction,
        "deterministic_dynamics": True,
    }
    for key, value in expected.items():
        actual = saved.get(key)
        if isinstance(value, float):
            matches = actual is not None and abs(float(actual) - value) <= 1.0e-8
        else:
            matches = actual == value
        if not matches:
            raise RuntimeError(
                f"{run_dir}: saved dynamics {key}={actual!r} does not match task profile {value!r}"
            )


def checkpoint_records(found: list[Path], previous: list[dict]) -> list[dict]:
    """Hash checkpoints while preserving completed evaluation metadata."""
    previous_by_iteration = {int(row["iteration"]): row for row in previous}
    records = []
    for path in found:
        iteration = checkpoint_iteration(path)
        old = previous_by_iteration.get(iteration, {})
        digest = sha256_file(path)
        old_digest = old.get("checkpoint_sha256")
        if old_digest is not None and old_digest != digest:
            raise RuntimeError(f"checkpoint hash changed after registration: {path}")
        records.append(
            {
                "iteration": iteration,
                "checkpoint": path.name,
                "checkpoint_sha256": digest,
                "evaluation": old.get("evaluation"),
                "quality": old.get("quality"),
            }
        )
    return records


def register_completed_run(
    *,
    run_dir: Path,
    profile: DynamicsProfile,
    iterations: int,
    save_interval: int,
    seed: int,
    entry: dict,
    registry: dict,
    registry_path: Path,
) -> None:
    validate_run_dynamics(run_dir, profile)
    found = checkpoints_in(run_dir)
    if not found or checkpoint_iteration(found[-1]) < iterations - 1:
        last = checkpoint_iteration(found[-1]) if found else None
        raise RuntimeError(
            f"training run is incomplete: {run_dir} (last checkpoint={last}, expected >= {iterations - 1}); "
            "inspect it before retrying"
        )
    entry.update({"tag": profile.tag, "split": profile.split, "profile": profile.__dict__})
    entry["training"] = {
        "method": "fresh_task_specific_ppo",
        "run": run_dir.name,
        "seed": seed,
        "iterations": iterations,
        "save_interval": save_interval,
        "completed_at": entry.get("training", {}).get("completed_at", utc_now()),
    }
    entry["checkpoints"] = checkpoint_records(found, entry.get("checkpoints", []))
    write_json(registry_path, registry)
    print(f"[CHECKPOINT-BANK] registered {len(found)} hashed checkpoints from {run_dir.name}", flush=True)


def registry_entry(registry: dict, task_id: int) -> dict:
    for row in registry["tasks"]:
        if int(row["task_id"]) == task_id:
            return row
    row = {"task_id": task_id, "training": {}, "checkpoints": []}
    registry["tasks"].append(row)
    registry["tasks"].sort(key=lambda item: int(item["task_id"]))
    return row


def train_profile(
    *,
    repo: Path,
    profile: DynamicsProfile,
    iterations: int,
    save_interval: int,
    num_envs: int | None,
    seed: int,
    registry: dict,
    registry_path: Path,
    dry_run: bool,
) -> None:
    log_root = repo / "logs/rsl_rl/unitree_g1_flat"
    train_py = repo / "pipeline/expert/train.py"
    run_name = f"official_mixed_v1_task{profile.task_id:02d}_seed{seed}"
    entry = registry_entry(registry, profile.task_id)
    prior_run = entry.get("training", {}).get("run")
    if prior_run:
        run_dir = log_root / prior_run
        found = checkpoints_in(run_dir) if run_dir.exists() else []
        if found and checkpoint_iteration(found[-1]) >= iterations - 1:
            register_completed_run(
                run_dir=run_dir,
                profile=profile,
                iterations=iterations,
                save_interval=save_interval,
                seed=seed,
                entry=entry,
                registry=registry,
                registry_path=registry_path,
            )
            print(f"[CHECKPOINT-BANK] reuse completed training: {run_dir}", flush=True)
            return

    existing = matching_runs(log_root, run_name)
    if existing:
        if len(existing) != 1:
            raise RuntimeError(
                f"found multiple unregistered runs for task {profile.task_id}: {existing}; "
                "inspect them before rerunning to avoid ambiguous provenance"
            )
        # The process may have completed but been interrupted just before the
        # atomic registry write. Adopt only a complete, dynamics-matched run.
        register_completed_run(
            run_dir=existing[0],
            profile=profile,
            iterations=iterations,
            save_interval=save_interval,
            seed=seed,
            entry=entry,
            registry=registry,
            registry_path=registry_path,
        )
        return

    command = [
        sys.executable,
        "-u",
        str(train_py),
        "--task",
        TASK_NAME,
        "--headless",
        "--run_name",
        run_name,
        "--max_iterations",
        str(iterations),
        "--save_interval",
        str(save_interval),
        "--seed",
        str(seed),
        *profile.cli_args(),
    ]
    if num_envs is not None:
        command.extend(["--num_envs", str(num_envs)])
    print(" ".join(command), flush=True)
    if dry_run:
        entry.update({"tag": profile.tag, "split": profile.split, "profile": profile.__dict__})
        entry["training"] = {
            "method": "fresh_task_specific_ppo",
            "run": f"<timestamp>_{run_name}",
            "seed": seed,
            "iterations": iterations,
            "save_interval": save_interval,
        }
        saved_iterations = list(range(0, iterations, save_interval))
        if iterations - 1 not in saved_iterations:
            saved_iterations.append(iterations - 1)
        entry["checkpoints"] = [
            {
                "iteration": iteration,
                "checkpoint": f"model_{iteration}.pt",
                "evaluation": None,
                "quality": None,
            }
            for iteration in saved_iterations
        ]
        return

    subprocess.run(command, cwd=repo, check=True)
    candidates = matching_runs(log_root, run_name)
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one run for {run_name}, found {candidates}")
    run_dir = candidates[0]
    register_completed_run(
        run_dir=run_dir,
        profile=profile,
        iterations=iterations,
        save_interval=save_interval,
        seed=seed,
        entry=entry,
        registry=registry,
        registry_path=registry_path,
    )


def classify_quality(success_rate: float) -> str:
    if success_rate < 0.35:
        return "low"
    if success_rate < 0.85:
        return "medium"
    return "high"


def evaluate_profile(
    *,
    repo: Path,
    profile: DynamicsProfile,
    manifest_path: Path,
    seeds: list[int],
    goal_batch: int,
    registry: dict,
    registry_path: Path,
    dry_run: bool,
    force: bool,
) -> None:
    entry = registry_entry(registry, profile.task_id)
    training = entry.get("training", {})
    if not training.get("run") or not entry.get("checkpoints"):
        raise RuntimeError(f"task {profile.task_id} has no registered checkpoint bank; run --phase train first")
    sweep_py = repo / "pipeline/evaluation/run_dynamics_sweep.py"
    for checkpoint in entry["checkpoints"]:
        checkpoint_path = repo / "logs/rsl_rl/unitree_g1_flat" / training["run"] / checkpoint["checkpoint"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"registered checkpoint is missing: {checkpoint_path}")
        actual_hash = sha256_file(checkpoint_path)
        registered_hash = checkpoint.get("checkpoint_sha256")
        if registered_hash is None:
            checkpoint["checkpoint_sha256"] = actual_hash
            write_json(registry_path, registry)
        elif registered_hash != actual_hash:
            raise RuntimeError(f"registered checkpoint hash mismatch: {checkpoint_path}")
        if checkpoint.get("evaluation") is not None and not force:
            print(
                f"[CHECKPOINT-BANK] reuse evaluation task={profile.task_id:02d} "
                f"iteration={checkpoint['iteration']}",
                flush=True,
            )
            continue
        output_dir = (
            registry_path.parent
            / "evaluations"
            / f"task{profile.task_id:02d}"
            / f"iteration_{int(checkpoint['iteration']):06d}"
        )
        command = [
            sys.executable,
            "-u",
            str(sweep_py),
            "--manifest",
            str(manifest_path),
            "--task-ids",
            str(profile.task_id),
            "--seeds",
            ",".join(map(str, seeds)),
            "--goal-batch",
            str(goal_batch),
            "--load-run",
            training["run"],
            "--checkpoint",
            checkpoint["checkpoint"],
            "--output-dir",
            str(output_dir),
        ]
        if force:
            command.append("--force")
        print(" ".join(command), flush=True)
        if dry_run:
            continue
        subprocess.run(command, cwd=repo, check=True)
        report = json.loads((output_dir / "summary.json").read_text())
        metrics = report["overall"]
        checkpoint["evaluation"] = {
            "seeds": seeds,
            "goal_batch": goal_batch,
            "episodes": metrics["episodes"],
            "success_rate": metrics["success_rate"],
            "fall_rate": metrics["fall_rate"],
            "mean_episode_return": metrics.get("mean_episode_return"),
            "mean_return_per_second": metrics.get("mean_return_per_second"),
            "mean_success_time_s": metrics.get("mean_success_time_s"),
            "summary": str((output_dir / "summary.json").relative_to(repo)),
            "evaluated_at": utc_now(),
        }
        checkpoint["quality"] = classify_quality(metrics["success_rate"])
        write_json(registry_path, registry)

    if not dry_run:
        qualities = {row["quality"] for row in entry["checkpoints"] if row.get("quality")}
        entry["quality_coverage"] = {
            "observed": sorted(qualities),
            "required": ["low", "medium", "high"],
            "complete": qualities == {"low", "medium", "high"},
        }
        write_json(registry_path, registry)
        print(f"[CHECKPOINT-BANK] task={profile.task_id:02d} coverage={sorted(qualities)}", flush=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_root = (
        repo
        / "data/t2mir/RobotLab-G1-MultiDynamics"
        / "official_mixed_v1/checkpoint_bank"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument("--task-ids", type=comma_ints, default=comma_ints("0,27,46"))
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--save-interval", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--eval-seeds", type=comma_ints, default=comma_ints("42,43"))
    parser.add_argument("--goal-batch", type=int, default=16)
    parser.add_argument("--registry", type=Path, default=default_root / "pilot_registry.json")
    parser.add_argument("--allow-eval-tasks", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.save_interval <= 0 or args.goal_batch <= 0:
        parser.error("iterations, save interval and goal batch must be positive")
    manifest, profiles = select_profiles(parser, args.manifest, args.task_ids, args.allow_eval_tasks)
    registry = load_json(
        args.registry,
        {
            "format_version": 1,
            "dataset": "RobotLab-G1-MultiDynamics-official-mixed-v1",
            "manifest": str(args.manifest.relative_to(repo)),
            "manifest_sha256": manifest["sha256"],
            "purpose": "periodic task-specific PPO checkpoints for equal-rollout Mixed prompts",
            "quality_thresholds": {
                "metric": "closed_loop_goal_success_rate",
                "low": "success_rate < 0.35",
                "medium": "0.35 <= success_rate < 0.85",
                "high": "success_rate >= 0.85",
            },
            "tasks": [],
        },
    )
    if registry.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("checkpoint registry manifest hash does not match the selected manifest")

    if args.phase in ("train", "all"):
        for profile in profiles:
            train_profile(
                repo=repo,
                profile=profile,
                iterations=args.iterations,
                save_interval=args.save_interval,
                num_envs=args.num_envs,
                seed=args.base_seed + profile.task_id,
                registry=registry,
                registry_path=args.registry,
                dry_run=args.dry_run,
            )
    if args.phase in ("evaluate", "all"):
        for profile in profiles:
            evaluate_profile(
                repo=repo,
                profile=profile,
                manifest_path=args.manifest,
                seeds=args.eval_seeds,
                goal_batch=args.goal_batch,
                registry=registry,
                registry_path=args.registry,
                dry_run=args.dry_run,
                force=args.force_evaluation,
            )
    if args.dry_run:
        print(f"[CHECKPOINT-BANK] dry-run PASS: phase={args.phase} tasks={args.task_ids}")
    else:
        print(f"[CHECKPOINT-BANK] registry: {args.registry}")


if __name__ == "__main__":
    main()
