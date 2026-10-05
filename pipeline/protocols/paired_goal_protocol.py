"""Shared deterministic goal lists for paired Isaac Sim/MuJoCo evaluation.

The simulator evaluators intentionally consume the same JSON artifact instead
of independently seeding different random-number libraries.  This makes the
relative goal pose an audited part of the cross-simulator experiment contract.
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
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT_VERSION = 1
PROTOCOL = "robotlab-paired-goals-v1"


@dataclass(frozen=True)
class PairedGoalScenario:
    profile_id: int
    episode_index: int
    replica_id: int
    relative_x: float
    relative_y: float
    relative_yaw: float


def scenario_sha256(scenarios: Sequence[PairedGoalScenario]) -> str:
    canonical = json.dumps(
        [asdict(row) for row in scenarios], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def generate_scenarios(
    profile_ids: Sequence[int],
    *,
    episodes: int,
    replicas: int,
    seed: int,
    min_distance: float,
    max_distance: float,
) -> list[PairedGoalScenario]:
    if episodes <= 0 or replicas <= 0:
        raise ValueError("episodes and replicas must be positive")
    if not 0.0 < min_distance < max_distance:
        raise ValueError("goal distances must satisfy 0 < min < max")
    rows: list[PairedGoalScenario] = []
    for profile_id in profile_ids:
        generator = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(profile_id)])
        )
        shape = (episodes, replicas)
        distances = generator.uniform(min_distance, max_distance, size=shape)
        bearings = generator.uniform(-math.pi, math.pi, size=shape)
        yaws = generator.uniform(-math.pi, math.pi, size=shape)
        for episode_index in range(episodes):
            for replica_id in range(replicas):
                distance = float(distances[episode_index, replica_id])
                bearing = float(bearings[episode_index, replica_id])
                rows.append(
                    PairedGoalScenario(
                        profile_id=int(profile_id),
                        episode_index=episode_index,
                        replica_id=replica_id,
                        relative_x=distance * math.cos(bearing),
                        relative_y=distance * math.sin(bearing),
                        relative_yaw=float(yaws[episode_index, replica_id]),
                    )
                )
    return rows


def build_manifest(
    scenarios: Sequence[PairedGoalScenario],
    *,
    episodes: int,
    replicas: int,
    seed: int,
    min_distance: float,
    max_distance: float,
) -> dict[str, Any]:
    profile_ids = sorted({row.profile_id for row in scenarios})
    return {
        "format_version": FORMAT_VERSION,
        "protocol": PROTOCOL,
        "seed": int(seed),
        "profile_ids": profile_ids,
        "episodes": int(episodes),
        "replicas": int(replicas),
        "goals_per_profile": int(episodes * replicas),
        "min_distance": float(min_distance),
        "max_distance": float(max_distance),
        "scenario_sha256": scenario_sha256(scenarios),
        "scenarios": [asdict(row) for row in scenarios],
    }


def validate_manifest(value: Mapping[str, Any]) -> tuple[dict[str, Any], list[PairedGoalScenario]]:
    manifest = dict(value)
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported paired-goal format version")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("not a RobotLab paired-goal manifest")
    episodes = int(manifest["episodes"])
    replicas = int(manifest["replicas"])
    if episodes <= 0 or replicas <= 0:
        raise ValueError("episodes and replicas must be positive")
    profile_ids = [int(value) for value in manifest["profile_ids"]]
    if len(profile_ids) != len(set(profile_ids)) or not profile_ids:
        raise ValueError("profile_ids must be unique and non-empty")
    scenarios = [PairedGoalScenario(**row) for row in manifest["scenarios"]]
    expected_keys = {
        (profile_id, episode_index, replica_id)
        for profile_id in profile_ids
        for episode_index in range(episodes)
        for replica_id in range(replicas)
    }
    actual_keys = {
        (row.profile_id, row.episode_index, row.replica_id) for row in scenarios
    }
    if len(actual_keys) != len(scenarios) or actual_keys != expected_keys:
        raise ValueError("paired goals do not cover each profile/episode/replica exactly once")
    actual_hash = scenario_sha256(scenarios)
    if actual_hash != manifest.get("scenario_sha256"):
        raise ValueError("paired-goal scenario hash mismatch")
    if int(manifest.get("goals_per_profile", -1)) != episodes * replicas:
        raise ValueError("goals_per_profile does not match episodes * replicas")
    return manifest, scenarios


def load_manifest(path: Path) -> tuple[dict[str, Any], list[PairedGoalScenario]]:
    return validate_manifest(json.loads(path.expanduser().resolve().read_text()))


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-ids", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-min-distance", type=float, default=1.0)
    parser.add_argument("--goal-max-distance", type=float, default=2.0)
    args = parser.parse_args()
    scenarios = generate_scenarios(
        args.profile_ids,
        episodes=args.episodes,
        replicas=args.replicas,
        seed=args.seed,
        min_distance=args.goal_min_distance,
        max_distance=args.goal_max_distance,
    )
    manifest = build_manifest(
        scenarios,
        episodes=args.episodes,
        replicas=args.replicas,
        seed=args.seed,
        min_distance=args.goal_min_distance,
        max_distance=args.goal_max_distance,
    )
    output = write_manifest(args.output, manifest)
    print(
        f"[PAIRED-GOALS] PASS profiles={len(args.profile_ids)} "
        f"goals={len(scenarios)} sha256={manifest['scenario_sha256']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
