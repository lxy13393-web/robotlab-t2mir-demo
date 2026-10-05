"""Validate and summarize a paired Isaac Sim/MuJoCo goal evaluation."""

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
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _close(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("sample count must be positive")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return centre - margin, centre + margin


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _extract_isaac(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    total = int(summary["total_episodes"])
    return {
        "simulator": "isaac_sim",
        "total": total,
        "successes": round(float(summary["success_rate"]) * total),
        "falls": round(float(summary["fall_rate"]) * total),
        "success_rate": float(summary["success_rate"]),
        "fall_rate": float(summary["fall_rate"]),
    }


def _extract_mujoco(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report["summary"]["overall"]
    total = int(summary["episodes"])
    return {
        "simulator": "mujoco",
        "total": total,
        "successes": round(float(summary["success_rate"]) * total),
        "falls": round(float(summary["fall_rate"]) * total),
        "success_rate": float(summary["success_rate"]),
        "fall_rate": float(summary["fall_rate"]),
    }


def _read_outcomes(path: Path) -> dict[tuple[int, int], str]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    outcomes = {
        (int(row["episode_index"]), int(row["replica_id"])): row["result"]
        for row in rows
    }
    if len(outcomes) != len(rows):
        raise ValueError(f"duplicate episode/replica keys in {path}")
    return outcomes


def _mcnemar_exact_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(left_only, right_only) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-report", type=Path, required=True)
    parser.add_argument("--mujoco-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    isaac = json.loads(args.isaac_report.expanduser().resolve().read_text())
    mujoco = json.loads(args.mujoco_report.expanduser().resolve().read_text())
    _expect(
        isaac.get("scenario_sha256") == mujoco.get("scenario_sha256"),
        "scenario hashes differ; this is not a paired comparison",
    )
    _expect(
        isaac.get("paired_goal_manifest_sha256")
        == mujoco.get("paired_goal_manifest_sha256"),
        "paired-goal manifest hashes differ",
    )
    _expect(
        isaac["checkpoint"]["checkpoint_sha256"]
        == mujoco["model_artifact"]["checkpoint_sha256"],
        "policy checkpoint hashes differ",
    )
    isaac_goal = isaac["goal_contract"]
    mujoco_goal = mujoco["goal_contract"]
    comparable_fields = {
        "min_distance": (isaac_goal["min_distance"], mujoco_goal["min_distance"]),
        "max_distance": (isaac_goal["max_distance"], mujoco_goal["max_distance"]),
        "timeout": (isaac_goal["timeout"], mujoco_goal["timeout_seconds"]),
        "hold_time": (isaac_goal["hold_time"], mujoco_goal["hold_seconds"]),
        "turn_forward_speed": (
            isaac_goal["turn_forward_speed"],
            mujoco_goal["turn_forward_speed"],
        ),
        "align_forward_speed": (
            isaac_goal["align_forward_speed"],
            mujoco_goal["align_forward_speed"],
        ),
    }
    for name, (left, right) in comparable_fields.items():
        _expect(_close(left, right), f"goal contract mismatch for {name}: {left} != {right}")
    for name in ("hold_command_mode", "reset_context_each_goal"):
        _expect(
            isaac_goal[name] == mujoco_goal[name],
            f"goal contract mismatch for {name}",
        )
    isaac_metrics = _extract_isaac(isaac)
    mujoco_metrics = _extract_mujoco(mujoco)
    _expect(isaac_metrics["total"] == mujoco_metrics["total"], "sample counts differ")
    for metrics in (isaac_metrics, mujoco_metrics):
        low, high = _wilson(metrics["successes"], metrics["total"])
        metrics["success_rate_ci95"] = [low, high]
    intervals_overlap = not (
        isaac_metrics["success_rate_ci95"][1] < mujoco_metrics["success_rate_ci95"][0]
        or mujoco_metrics["success_rate_ci95"][1] < isaac_metrics["success_rate_ci95"][0]
    )
    isaac_outcomes = _read_outcomes(Path(isaac["result_csv"]))
    mujoco_outcomes = _read_outcomes(Path(mujoco["episodes_csv"]))
    _expect(isaac_outcomes.keys() == mujoco_outcomes.keys(), "paired CSV goal keys differ")
    both_success = sum(
        isaac_outcomes[key] == "success" and mujoco_outcomes[key] == "success"
        for key in isaac_outcomes
    )
    isaac_only = sum(
        isaac_outcomes[key] == "success" and mujoco_outcomes[key] != "success"
        for key in isaac_outcomes
    )
    mujoco_only = sum(
        isaac_outcomes[key] != "success" and mujoco_outcomes[key] == "success"
        for key in isaac_outcomes
    )
    neither_success = len(isaac_outcomes) - both_success - isaac_only - mujoco_only
    mcnemar_p = _mcnemar_exact_two_sided(isaac_only, mujoco_only)
    if mcnemar_p < 0.05:
        interpretation = (
            "paired outcomes show a statistically detectable simulator difference "
            "at the 0.05 level"
        )
    elif intervals_overlap:
        interpretation = (
            "no statistically detectable paired simulator difference at this sample size"
        )
    else:
        interpretation = (
            "marginal confidence intervals are separated, although the paired exact test "
            "does not cross 0.05"
        )
    result = {
        "format_version": 1,
        "protocol_validation": "PASS",
        "scenario_sha256": isaac["scenario_sha256"],
        "checkpoint_sha256": isaac["checkpoint"]["checkpoint_sha256"],
        "sample_count_per_simulator": isaac_metrics["total"],
        "isaac_sim": isaac_metrics,
        "mujoco": mujoco_metrics,
        "mujoco_minus_isaac_success_rate": (
            mujoco_metrics["success_rate"] - isaac_metrics["success_rate"]
        ),
        "success_rate_ci95_overlap": intervals_overlap,
        "paired_goal_outcomes": {
            "both_success": both_success,
            "isaac_only_success": isaac_only,
            "mujoco_only_success": mujoco_only,
            "neither_success": neither_success,
            "agreement_rate": (both_success + neither_success) / len(isaac_outcomes),
            "mcnemar_exact_two_sided_p": mcnemar_p,
        },
        "interpretation": interpretation,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[CROSS-SIM] PASS output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
