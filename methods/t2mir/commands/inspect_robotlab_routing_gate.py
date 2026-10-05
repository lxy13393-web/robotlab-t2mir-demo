"""Fail-fast routing-collapse gate for RobotLab T2MIR validation reports."""

from __future__ import annotations

# Make the method root importable when this command is run by file path.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _method_root = _Path(__file__).resolve().parents[1]
    if str(_method_root) not in _sys.path:
        _sys.path.insert(0, str(_method_root))

import argparse
import collections
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-effective-experts", type=float, default=3.0)
    parser.add_argument("--min-active-experts", type=int, default=4)
    parser.add_argument("--min-active-fraction", type=float, default=0.01)
    parser.add_argument("--max-expert-fraction", type=float, default=0.65)
    parser.add_argument("--max-dominant-pair-share", type=float, default=0.80)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text())
    balance = payload.get("balance_signature", {}).get("task", {})
    if balance.get("enabled") is not True or balance.get("mode") != "switch":
        raise SystemExit(
            "FAIL: report is not from a Task-balanced run "
            f"(task balance signature={balance!r})"
        )
    routes = payload.get("routes_by_task")
    if not isinstance(routes, dict) or not routes:
        raise SystemExit(f"FAIL: no routes_by_task in {args.report}")

    total_counts = None
    dominant_pairs: collections.Counter[tuple[int, int]] = collections.Counter()
    for task_routes in routes.values():
        candidates = [
            route for route in task_routes.values() if route.get("gate_kind") == "task"
        ]
        if len(candidates) != 1:
            raise SystemExit("FAIL: expected exactly one tracked Task-MoE gate")
        counts = list(map(int, candidates[0]["counts"]))
        if total_counts is None:
            total_counts = [0] * len(counts)
        total_counts = [left + right for left, right in zip(total_counts, counts)]
        pair = tuple(sorted(range(len(counts)), key=counts.__getitem__, reverse=True)[:2])
        dominant_pairs[pair] += 1

    assert total_counts is not None
    total = sum(total_counts)
    fractions = [count / max(total, 1) for count in total_counts]
    entropy = -sum(value * math.log(value) for value in fractions if value > 0)
    effective = math.exp(entropy)
    active = sum(value >= args.min_active_fraction for value in fractions)
    maximum = max(fractions)
    pair, pair_count = dominant_pairs.most_common(1)[0]
    pair_share = pair_count / len(routes)

    checks = {
        "effective_experts": effective >= args.min_effective_experts,
        "active_experts": active >= args.min_active_experts,
        "maximum_expert_fraction": maximum <= args.max_expert_fraction,
        "dominant_pair_share": pair_share <= args.max_dominant_pair_share,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "status": status,
        "step": payload.get("step"),
        "validation_mse": payload.get("validation_mse"),
        "counts": total_counts,
        "fractions": fractions,
        "effective_experts": effective,
        "active_experts_at_or_above_fraction": active,
        "dominant_pair": pair,
        "dominant_pair_share": pair_share,
        "checks": checks,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
