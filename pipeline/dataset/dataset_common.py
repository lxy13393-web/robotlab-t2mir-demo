"""Shared contracts and filesystem helpers for official-mixed datasets.

This module deliberately contains no pilot- or formal-dataset orchestration.
Both workflows import the same dimensions, trajectory fields and provenance
helpers from here so the formal pipeline does not depend on a pilot CLI.
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
from datetime import datetime, timezone
from pathlib import Path


TASK_NAME = "RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0"
STATE_DIM = 123
ACTION_DIM = 37
WINDOW_FIELDS = (
    "states",
    "next_states",
    "commands",
    "policy_actions",
    "executed_actions",
    "rewards",
    "dones",
    "time_outs",
    "env_ids",
    "episode_steps",
    "goal_relative",
    "controller_modes",
    "tracking_errors",
    "action_lags",
    "motor_strengths",
    "payload_kg",
    "frictions",
    "masks",
    "task_ids",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated list of unique integers"
        )
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> dict:
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
