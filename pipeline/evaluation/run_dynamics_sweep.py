"""Evaluate one checkpoint across a manifest of fixed G1 dynamics profiles."""

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from pipeline.protocols.dynamics_profile import load_manifest


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def read_result(path: Path) -> dict:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"empty goal result: {path}")
    successes = [row for row in rows if row["result"] == "success"]
    falls = [row for row in rows if row["result"] == "fall"]
    episode_returns = [float(row["episode_return"]) for row in rows if row.get("episode_return") not in (None, "")]
    durations = [float(row["time"]) for row in rows]
    return_rates = [
        float(row["episode_return"]) / max(float(row["time"]), 1.0e-8)
        for row in rows
        if row.get("episode_return") not in (None, "")
    ]
    return {
        "episodes": len(rows), "successes": len(successes), "falls": len(falls),
        "timeouts": len(rows) - len(successes) - len(falls),
        "success_rate": len(successes) / len(rows), "fall_rate": len(falls) / len(rows),
        "mean_position_error_m": sum(float(row["position_error"]) for row in rows) / len(rows),
        "mean_yaw_error_rad": sum(float(row["yaw_error"]) for row in rows) / len(rows),
        "mean_episode_return": sum(episode_returns) / len(episode_returns) if episode_returns else None,
        "mean_return_per_second": sum(return_rates) / len(return_rates) if return_rates else None,
        "mean_episode_duration_s": sum(durations) / len(durations),
        "mean_success_time_s": (
            sum(float(row["time"]) for row in successes) / len(successes) if successes else None
        ),
    }


def aggregate(records: list[dict]) -> dict:
    episodes = sum(row["episodes"] for row in records)
    weighted_returns = [
        (row["mean_episode_return"], row["episodes"])
        for row in records
        if row.get("mean_episode_return") is not None
    ]
    weighted_return_rates = [
        (row["mean_return_per_second"], row["episodes"])
        for row in records
        if row.get("mean_return_per_second") is not None
    ]
    success_time_rows = [
        (row["mean_success_time_s"], row["successes"])
        for row in records
        if row.get("mean_success_time_s") is not None and row["successes"]
    ]
    return {
        "profiles": len({row["task_id"] for row in records}),
        "runs": len(records), "episodes": episodes,
        "successes": sum(row["successes"] for row in records),
        "falls": sum(row["falls"] for row in records),
        "timeouts": sum(row["timeouts"] for row in records),
        "success_rate": sum(row["successes"] for row in records) / episodes,
        "fall_rate": sum(row["falls"] for row in records) / episodes,
        "mean_episode_return": (
            sum(value * count for value, count in weighted_returns) / sum(count for _, count in weighted_returns)
            if weighted_returns else None
        ),
        "mean_return_per_second": (
            sum(value * count for value, count in weighted_return_rates)
            / sum(count for _, count in weighted_return_rates)
            if weighted_return_rates else None
        ),
        "mean_success_time_s": (
            sum(value * count for value, count in success_time_rows) / sum(count for _, count in success_time_rows)
            if success_time_rows else None
        ),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument("--task-ids", type=comma_ints, default=None)
    parser.add_argument("--split", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--seeds", type=comma_ints, default=comma_ints("42,43,44"))
    parser.add_argument("--load-run", default="2026-08-30_23-35-43")
    parser.add_argument("--checkpoint", default="model_1499.pt")
    parser.add_argument("--goal-batch", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=repo / "outputs/dynamics48/base_ppo")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload, profiles = load_manifest(args.manifest)
    if args.split != "all": profiles = [p for p in profiles if p.split == args.split]
    if args.task_ids is not None:
        wanted = set(args.task_ids); profiles = [p for p in profiles if p.task_id in wanted]
        missing = wanted.difference(p.task_id for p in profiles)
        if missing: parser.error(f"task IDs not selected/found: {sorted(missing)}")
    if args.max_profiles is not None: profiles = profiles[:args.max_profiles]
    if not profiles or not args.seeds or args.goal_batch <= 0:
        parser.error("profile selection, seeds and goal-batch must be non-empty/positive")

    play = repo / "pipeline/expert/play.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for profile in profiles:
        for seed in args.seeds:
            result = args.output_dir / f"{profile.tag}_seed{seed}_n{args.goal_batch}.csv"
            command = [sys.executable, "-u", str(play),
                       "--task", "RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0",
                       "--load_run", args.load_run, "--checkpoint", args.checkpoint,
                       "--goal_batch", str(args.goal_batch), "--goal_min_distance", "1.0",
                       "--goal_max_distance", "4.0", "--goal_timeout", "40",
                       "--goal_hold_time", "2", "--goal_seed", str(seed),
                       *profile.cli_args(), "--goal_results", str(result), "--headless"]
            print(f"[DYNAMICS-SWEEP] {profile.tag} seed={seed}")
            print(" ".join(command), flush=True)
            if args.dry_run: continue
            if not result.exists() or args.force:
                subprocess.run(command, cwd=repo, check=True)
            records.append({"task_id": profile.task_id, "split": profile.split,
                            "profile": profile.__dict__, "seed": seed,
                            "result_file": str(result), **read_result(result)})
    if args.dry_run:
        print(f"[DYNAMICS-SWEEP] dry-run PASS: {len(profiles) * len(args.seeds)} jobs")
        return

    per_profile = []
    for profile in profiles:
        rows = [row for row in records if row["task_id"] == profile.task_id]
        per_profile.append({"task_id": profile.task_id, "tag": profile.tag,
                            "split": profile.split, **profile.__dict__, **aggregate(rows)})
    dimensions = {}
    for field in ("action_lag", "motor_strength", "payload_kg", "friction"):
        grouped = defaultdict(list)
        for row in records: grouped[str(row["profile"][field])].append(row)
        dimensions[field] = {value: aggregate(rows) for value, rows in grouped.items()}
    report = {"format_version": 1, "manifest": str(args.manifest),
              "manifest_sha256": payload["sha256"],
              "policy": {"load_run": args.load_run, "checkpoint": args.checkpoint},
              "overall": aggregate(records), "profiles": per_profile,
              "dimensions": dimensions, "runs": records}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[DYNAMICS-SWEEP] overall={report['overall']}")
    print(f"[DYNAMICS-SWEEP] summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
