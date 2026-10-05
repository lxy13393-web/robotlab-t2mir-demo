"""Aggregate already validated paired cross-simulator comparison reports."""

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
import math
from pathlib import Path


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [centre - margin, centre + margin]


def mcnemar_exact(left_only: int, right_only: int) -> float:
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
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.expanduser().resolve().read_text()) for path in args.comparisons]
    if any(report.get("protocol_validation") != "PASS" for report in reports):
        raise ValueError("all input comparisons must pass protocol validation")
    checkpoint_hashes = {report["checkpoint_sha256"] for report in reports}
    if len(checkpoint_hashes) != 1:
        raise ValueError("comparison reports use different checkpoints")
    scenario_hashes = [report["scenario_sha256"] for report in reports]
    if len(scenario_hashes) != len(set(scenario_hashes)):
        raise ValueError("comparison reports repeat a scenario set")

    total = sum(int(report["sample_count_per_simulator"]) for report in reports)
    isaac_successes = sum(int(report["isaac_sim"]["successes"]) for report in reports)
    mujoco_successes = sum(int(report["mujoco"]["successes"]) for report in reports)
    isaac_falls = sum(int(report["isaac_sim"]["falls"]) for report in reports)
    mujoco_falls = sum(int(report["mujoco"]["falls"]) for report in reports)
    paired_keys = (
        "both_success",
        "isaac_only_success",
        "mujoco_only_success",
        "neither_success",
    )
    paired = {
        key: sum(int(report["paired_goal_outcomes"][key]) for report in reports)
        for key in paired_keys
    }
    paired["agreement_rate"] = (paired["both_success"] + paired["neither_success"]) / total
    paired["mcnemar_exact_two_sided_p"] = mcnemar_exact(
        paired["isaac_only_success"], paired["mujoco_only_success"]
    )
    result = {
        "format_version": 1,
        "protocol_validation": "PASS",
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "independent_scenario_sets": len(reports),
        "scenario_sha256s": scenario_hashes,
        "sample_count_per_simulator": total,
        "isaac_sim": {
            "successes": isaac_successes,
            "success_rate": isaac_successes / total,
            "success_rate_ci95": wilson(isaac_successes, total),
            "falls": isaac_falls,
            "fall_rate": isaac_falls / total,
        },
        "mujoco": {
            "successes": mujoco_successes,
            "success_rate": mujoco_successes / total,
            "success_rate_ci95": wilson(mujoco_successes, total),
            "falls": mujoco_falls,
            "fall_rate": mujoco_falls / total,
        },
        "mujoco_minus_isaac_success_rate": (mujoco_successes - isaac_successes) / total,
        "paired_goal_outcomes": paired,
        "per_scenario_set": reports,
        "interpretation": (
            "no statistically detectable paired simulator difference across the pooled sets"
            if paired["mcnemar_exact_two_sided_p"] >= 0.05
            else "pooled paired outcomes show a statistically detectable simulator difference"
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[CROSS-SIM-AGGREGATE] PASS output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
