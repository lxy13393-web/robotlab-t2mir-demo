"""Guarded serial queue for the formal T2MIR Isaac Sim online evaluation.

The queue is intentionally a CPU-only orchestrator.  It does not import Isaac
Sim or PyTorch, and it will launch no simulator unless ``--execute`` is given.
Before a job is scheduled it requires an immutable, completed A/D training
contract whose recorded ``best.pt`` SHA-256 still matches the checkpoint.

Each evaluator invocation owns a directory whose path encodes the complete
comparison identity::

    variant_A/train_seed_42/checkpoint_sha256_<64 hex>/
      eval_seed_42/profile_05/

Interrupted or failed invocations can be resumed by running the same command.
Only a structurally complete JSON/CSV result pair is skipped; partial or
invalid output is rerun with the evaluator's explicit ``--force`` flag.  The
same non-blocking GPU lock as the A--D trainer prevents evaluation from racing
training on CUDA device 0.
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
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from pipeline.protocols.ppo_online_protocol import EPISODE_BOUNDARY_MECHANISM


FORMAT_VERSION = 1
ALLOWED_VARIANTS = ("A", "D")
REQUIRED_HELD_OUT = (5, 14, 23, 32, 41, 47)
DEFAULT_EPISODES = 8
DEFAULT_NUM_ENVS = 32
DEFAULT_TASK = "RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_GOAL_MIN_DISTANCE = 1.0
DEFAULT_GOAL_MAX_DISTANCE = 4.0
DEFAULT_GOAL_TIMEOUT = 40.0
DEFAULT_GOAL_HOLD_TIME = 2.0
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class QueueGateError(RuntimeError):
    """Raised when a reproducibility, completeness, or serialization gate fails."""


@dataclass(frozen=True)
class TrainingCheckpoint:
    variant: str
    train_seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    run_contract_path: Path
    run_contract_sha256: str
    routing_signature: dict[str, Any]


@dataclass(frozen=True)
class EvaluationJob:
    variant: str
    train_seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    run_contract_path: Path
    run_contract_sha256: str
    routing_signature: dict[str, Any]
    eval_seed: int
    profile_id: int
    episodes: int
    num_envs: int
    output_dir: Path
    evaluator_path: Path
    dynamics_manifest_path: Path
    dynamics_manifest_file_sha256: str
    dynamics_manifest_sha256: str
    python_executable: Path
    task: str = DEFAULT_TASK
    device: str = DEFAULT_DEVICE

    @property
    def job_id(self) -> str:
        return (
            f"variant_{self.variant}__train_seed_{self.train_seed}__"
            f"checkpoint_{self.checkpoint_sha256}__eval_seed_{self.eval_seed}__"
            f"profile_{self.profile_id:02d}"
        )

    @property
    def stem(self) -> str:
        return f"variant{self.variant}_task{self.profile_id:02d}_seed{self.eval_seed}"

    @property
    def report_path(self) -> Path:
        return self.output_dir / f"{self.stem}.json"

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"{self.stem}.csv"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "evaluation.log"

    @property
    def state_path(self) -> Path:
        return self.output_dir / "evaluation_job_state.json"

    def identity(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "variant": self.variant,
            "train_seed": self.train_seed,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_run_contract": str(self.run_contract_path),
            "training_run_contract_sha256": self.run_contract_sha256,
            "routing_signature": self.routing_signature,
            "eval_seed": self.eval_seed,
            "profile_id": self.profile_id,
            "episodes": self.episodes,
            "num_envs": self.num_envs,
            "task": self.task,
            "device": self.device,
            "dynamics_manifest": str(self.dynamics_manifest_path),
            "dynamics_manifest_file_sha256": self.dynamics_manifest_file_sha256,
            "dynamics_manifest_sha256": self.dynamics_manifest_sha256,
        }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_sha256(x: float, y: float, yaw: float) -> str:
    return hashlib.sha256(struct.pack("<fff", x, y, yaw)).hexdigest()


def load_json(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    if not path.is_file():
        raise QueueGateError(f"required {label} file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueGateError(f"cannot read {label} file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueGateError(f"{label} must be a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def resolve_repo_path(repo: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo / path).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as exc:
        raise QueueGateError(f"{label} must stay inside the repository: {resolved}") from exc
    return resolved


def parse_csv_strings(value: str, *, label: str, upper: bool = False) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if upper:
        parsed = [part.upper() for part in parsed]
    if not parsed:
        raise argparse.ArgumentTypeError(f"{label} must contain at least one value")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{label} must not contain duplicates")
    return parsed


def parse_variants(value: str) -> list[str]:
    parsed = parse_csv_strings(value, label="--variants", upper=True)
    unknown = sorted(set(parsed).difference(ALLOWED_VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--variants only supports formal A/D checkpoints; got {unknown}"
        )
    return parsed


def parse_int_list(value: str, *, label: str) -> list[int]:
    raw = parse_csv_strings(value, label=label)
    try:
        parsed = [int(part) for part in raw]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{label} values must be non-negative")
    return parsed


def parse_train_seeds(value: str) -> list[int]:
    return parse_int_list(value, label="--train-seeds")


def parse_eval_seeds(value: str) -> list[int]:
    return parse_int_list(value, label="--eval-seeds")


def parse_profiles(value: str) -> list[int]:
    parsed = parse_int_list(value, label="--profiles")
    unknown = sorted(set(parsed).difference(REQUIRED_HELD_OUT))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--profiles must be selected from {list(REQUIRED_HELD_OUT)}; got {unknown}"
        )
    return parsed


def validate_training_manifest(
    repo: Path, manifest_path: Path
) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path, label="training manifest")
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise QueueGateError(
            f"unsupported training manifest format_version={manifest.get('format_version')!r}"
        )
    experiment_name = str(manifest.get("experiment_name", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_name):
        raise QueueGateError(f"unsafe experiment_name: {experiment_name!r}")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) for seed in seeds):
        raise QueueGateError("training manifest seeds must be a non-empty integer list")
    try:
        normalized_seeds = [int(seed) for seed in seeds]
    except (TypeError, ValueError) as exc:
        raise QueueGateError("training manifest seeds must be integers") from exc
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise QueueGateError("training manifest seeds must be unique")
    manifest["seeds"] = normalized_seeds

    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise QueueGateError("training manifest variants must be a list")
    rows = {
        str(row.get("id")): row for row in variants if isinstance(row, dict)
    }
    for variant in ALLOWED_VARIANTS:
        if variant not in rows:
            raise QueueGateError(f"training manifest is missing variant {variant}")
        if not isinstance(rows[variant].get("routing_signature"), dict):
            raise QueueGateError(f"variant {variant} has no routing_signature")

    runs_root = resolve_repo_path(repo, manifest.get("runs_root", ""), label="runs_root")
    gpu_lock = resolve_repo_path(repo, manifest.get("gpu_lock", ""), label="gpu_lock")
    return manifest, runs_root, gpu_lock


def _canonical_profile_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 48:
        raise QueueGateError("dynamics manifest must contain exactly 48 task rows")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise QueueGateError("dynamics manifest task rows must be JSON objects")
        try:
            task_id = int(row["task_id"])
            action_lag = float(row["action_lag"])
            motor_strength = float(row["motor_strength"])
            payload_kg = float(row["payload_kg"])
            friction = float(row["friction"])
            split = str(row["split"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QueueGateError(f"invalid dynamics manifest task row: {row!r}") from exc
        if not (0.0 <= action_lag < 1.0):
            raise QueueGateError(f"profile {task_id} has invalid action_lag")
        if motor_strength <= 0.0 or payload_kg < 0.0 or friction <= 0.0:
            raise QueueGateError(f"profile {task_id} has invalid dynamics values")
        if split not in {"train", "eval"}:
            raise QueueGateError(f"profile {task_id} has invalid split={split!r}")
        # Keep this byte-for-byte compatible with
        # ``dynamics_profile.manifest_payload``.  The human-readable tag is
        # part of the canonical task row and therefore part of the semantic
        # manifest digest.
        tag = str(row.get("tag", ""))
        if not tag:
            raise QueueGateError(f"profile {task_id} has no canonical tag")
        normalized.append(
            {
                "task_id": task_id,
                "action_lag": action_lag,
                "motor_strength": motor_strength,
                "payload_kg": payload_kg,
                "friction": friction,
                "split": split,
                "tag": tag,
            }
        )
    ids = [row["task_id"] for row in normalized]
    if ids != list(range(48)):
        raise QueueGateError("dynamics manifest task IDs must be ordered contiguously 0..47")
    return normalized


def validate_dynamics_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    manifest = load_json(path, label="dynamics manifest")
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise QueueGateError("unsupported dynamics manifest format_version")
    rows = _canonical_profile_rows(manifest.get("tasks"))
    eval_tasks = [row["task_id"] for row in rows if row["split"] == "eval"]
    if eval_tasks != list(REQUIRED_HELD_OUT):
        raise QueueGateError(
            f"dynamics manifest eval split={eval_tasks}, expected={list(REQUIRED_HELD_OUT)}"
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    expected_sha = hashlib.sha256(canonical.encode()).hexdigest()
    if manifest.get("sha256") != expected_sha:
        raise QueueGateError(
            f"dynamics manifest semantic SHA-256 mismatch: "
            f"{manifest.get('sha256')!r} != {expected_sha}"
        )
    if int(manifest.get("task_count", -1)) != 48:
        raise QueueGateError("dynamics manifest task_count must be 48")
    if list(map(int, manifest.get("eval_tasks", []))) != list(REQUIRED_HELD_OUT):
        raise QueueGateError("dynamics manifest eval_tasks field is not the frozen six-profile split")
    return manifest


def validate_completed_training_run(
    *,
    repo: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    runs_root: Path,
    variant: str,
    train_seed: int,
) -> TrainingCheckpoint:
    if variant not in ALLOWED_VARIANTS:
        raise QueueGateError(f"only formal variants A/D are supported, got {variant!r}")
    if train_seed not in manifest["seeds"]:
        raise QueueGateError(
            f"train seed {train_seed} is absent from the training manifest: {manifest['seeds']}"
        )
    variant_rows = {
        str(row["id"]): row
        for row in manifest["variants"]
        if isinstance(row, dict) and "id" in row
    }
    expected_routing = variant_rows[variant]["routing_signature"]
    run_dir = (
        runs_root / manifest["experiment_name"] / variant / f"seed_{train_seed}"
    ).resolve()
    try:
        run_dir.relative_to(runs_root.resolve())
    except ValueError as exc:
        raise QueueGateError(f"training run escaped runs_root: {run_dir}") from exc
    contract_path = run_dir / "run_contract.json"
    contract = load_json(contract_path, label="training run contract")
    if contract.get("status") != "complete":
        raise QueueGateError(
            f"training run {variant}/seed_{train_seed} is not complete: "
            f"status={contract.get('status')!r}"
        )
    identity = contract.get("identity")
    if not isinstance(identity, dict):
        raise QueueGateError(f"training run {variant}/seed_{train_seed} has no identity")
    if identity.get("variant") != variant or int(identity.get("seed", -1)) != train_seed:
        raise QueueGateError(f"training run identity does not match {variant}/seed_{train_seed}")
    if identity.get("experiment_name") != manifest["experiment_name"]:
        raise QueueGateError("training run experiment identity does not match the manifest")
    if identity.get("routing_signature") != expected_routing:
        raise QueueGateError("training run routing signature does not match the requested variant")
    manifest_digest = sha256_file(manifest_path)
    if identity.get("manifest_sha256") != manifest_digest:
        raise QueueGateError("training run was produced from a different training manifest")
    dataset = identity.get("dataset")
    if not isinstance(dataset, dict) or sorted(
        map(int, dataset.get("held_out_task_ids", []))
    ) != list(REQUIRED_HELD_OUT):
        raise QueueGateError("training run does not record the frozen held-out profile split")

    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise QueueGateError("completed training run has no artifacts mapping")
    recorded_sha = str(artifacts.get("best.pt", "")).lower()
    if SHA256_PATTERN.fullmatch(recorded_sha) is None:
        raise QueueGateError(
            "completed training run artifacts/best.pt must be a full 64-character SHA-256"
        )
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise QueueGateError(f"completed training checkpoint is missing or empty: {checkpoint_path}")
    if checkpoint_path.is_symlink() or checkpoint_path.resolve().parent != run_dir:
        raise QueueGateError(f"best.pt must be a regular artifact inside its run: {checkpoint_path}")
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != recorded_sha:
        raise QueueGateError(
            f"completed training checkpoint SHA-256 mismatch: {actual_sha} != {recorded_sha}"
        )
    return TrainingCheckpoint(
        variant=variant,
        train_seed=train_seed,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_sha256=actual_sha,
        run_contract_path=contract_path.resolve(),
        run_contract_sha256=sha256_file(contract_path),
        routing_signature=expected_routing,
    )


def build_jobs(
    *,
    checkpoints: Sequence[TrainingCheckpoint],
    eval_seeds: Sequence[int],
    profiles: Sequence[int],
    episodes: int,
    num_envs: int,
    output_root: Path,
    evaluator_path: Path,
    dynamics_manifest_path: Path,
    dynamics_manifest: dict[str, Any],
    python_executable: Path,
    task: str = DEFAULT_TASK,
    device: str = DEFAULT_DEVICE,
) -> list[EvaluationJob]:
    if episodes < 2:
        raise QueueGateError("episodes must be at least 2 to measure online adaptation")
    if num_envs <= 0:
        raise QueueGateError("num_envs must be positive")
    if not eval_seeds or len(set(eval_seeds)) != len(eval_seeds):
        raise QueueGateError("eval seeds must be non-empty and unique")
    if not profiles or len(set(profiles)) != len(profiles):
        raise QueueGateError("profiles must be non-empty and unique")
    if sorted(set(profiles).difference(REQUIRED_HELD_OUT)):
        raise QueueGateError("jobs may use only the frozen six held-out profiles")
    if not evaluator_path.is_file():
        raise QueueGateError(f"online evaluator is missing: {evaluator_path}")
    if not python_executable.is_file():
        raise QueueGateError(f"Python executable is missing: {python_executable}")

    manifest_file_sha = sha256_file(dynamics_manifest_path)
    jobs: list[EvaluationJob] = []
    # Profile-major ordering keeps paired A/D observations adjacent while every
    # process remains strictly serial.
    checkpoint_by_identity = {
        (checkpoint.train_seed, checkpoint.variant): checkpoint
        for checkpoint in checkpoints
    }
    train_seeds = list(dict.fromkeys(checkpoint.train_seed for checkpoint in checkpoints))
    variants = list(dict.fromkeys(checkpoint.variant for checkpoint in checkpoints))
    for train_seed in train_seeds:
        for eval_seed in eval_seeds:
            for profile_id in profiles:
                for variant in variants:
                    checkpoint = checkpoint_by_identity.get((train_seed, variant))
                    if checkpoint is None:
                        continue
                    output_dir = (
                        output_root
                        / f"variant_{variant}"
                        / f"train_seed_{train_seed}"
                        / f"checkpoint_sha256_{checkpoint.checkpoint_sha256}"
                        / f"eval_seed_{eval_seed}"
                        / f"profile_{profile_id:02d}"
                    ).resolve()
                    jobs.append(
                        EvaluationJob(
                            variant=variant,
                            train_seed=train_seed,
                            checkpoint_path=checkpoint.checkpoint_path,
                            checkpoint_sha256=checkpoint.checkpoint_sha256,
                            run_contract_path=checkpoint.run_contract_path,
                            run_contract_sha256=checkpoint.run_contract_sha256,
                            routing_signature=checkpoint.routing_signature,
                            eval_seed=int(eval_seed),
                            profile_id=int(profile_id),
                            episodes=episodes,
                            num_envs=num_envs,
                            output_dir=output_dir,
                            evaluator_path=evaluator_path.resolve(),
                            dynamics_manifest_path=dynamics_manifest_path.resolve(),
                            dynamics_manifest_file_sha256=manifest_file_sha,
                            dynamics_manifest_sha256=str(dynamics_manifest["sha256"]),
                            python_executable=python_executable.resolve(),
                            task=task,
                            device=device,
                        )
                    )
    identities = [job.job_id for job in jobs]
    outputs = [job.output_dir for job in jobs]
    if len(identities) != len(set(identities)) or len(outputs) != len(set(outputs)):
        raise QueueGateError("evaluation matrix produced duplicate identities or output paths")
    expected_count = len(checkpoints) * len(eval_seeds) * len(profiles)
    if len(jobs) != expected_count:
        raise QueueGateError(
            f"evaluation matrix is incomplete: {len(jobs)} jobs != {expected_count}"
        )
    return jobs


def evaluator_command(job: EvaluationJob, *, force: bool) -> list[str]:
    command = [
        str(job.python_executable),
        "-u",
        str(job.evaluator_path),
        "--task",
        job.task,
        "--offline-checkpoint",
        str(job.checkpoint_path),
        "--checkpoint-sha256",
        job.checkpoint_sha256,
        "--variant",
        job.variant,
        "--manifest",
        str(job.dynamics_manifest_path),
        "--profile-id",
        str(job.profile_id),
        "--episodes",
        str(job.episodes),
        "--num_envs",
        str(job.num_envs),
        "--seed",
        str(job.eval_seed),
        "--goal-min-distance",
        str(DEFAULT_GOAL_MIN_DISTANCE),
        "--goal-max-distance",
        str(DEFAULT_GOAL_MAX_DISTANCE),
        "--goal-timeout",
        str(DEFAULT_GOAL_TIMEOUT),
        "--goal-hold-time",
        str(DEFAULT_GOAL_HOLD_TIME),
        "--output-dir",
        str(job.output_dir),
        "--device",
        job.device,
        "--headless",
    ]
    if force:
        command.append("--force")
    return command


def _require_int(value: Any, expected: int, label: str) -> None:
    try:
        actual = int(value)
    except (TypeError, ValueError) as exc:
        raise QueueGateError(f"{label} is not an integer: {value!r}") from exc
    if actual != expected:
        raise QueueGateError(f"{label}={actual}, expected={expected}")


def validate_complete_result(job: EvaluationJob) -> dict[str, str]:
    report = load_json(job.report_path, label="online evaluation report")
    if int(report.get("format_version", -1)) != FORMAT_VERSION:
        raise QueueGateError("online report format_version mismatch")
    if report.get("evaluation") != "source_faithful_stationary_online_dpt":
        raise QueueGateError("online report has the wrong evaluation protocol")
    if report.get("variant") != job.variant:
        raise QueueGateError("online report variant does not match the queue identity")
    _require_int(report.get("seed"), job.eval_seed, "online report seed")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise QueueGateError("online report has no checkpoint provenance")
    if checkpoint.get("checkpoint_sha256") != job.checkpoint_sha256:
        raise QueueGateError("online report checkpoint SHA-256 does not match the queue identity")
    try:
        reported_checkpoint = Path(str(checkpoint.get("checkpoint"))).resolve()
    except (TypeError, ValueError) as exc:
        raise QueueGateError("online report checkpoint path is invalid") from exc
    if reported_checkpoint != job.checkpoint_path.resolve():
        raise QueueGateError("online report checkpoint path does not match the queue identity")
    if checkpoint.get("routing_signature") != job.routing_signature:
        raise QueueGateError("online report routing signature does not match training")

    profile = report.get("profile")
    if not isinstance(profile, dict):
        raise QueueGateError("online report has no dynamics profile")
    _require_int(profile.get("task_id"), job.profile_id, "online report profile task_id")
    if profile.get("split") != "eval":
        raise QueueGateError("online report profile is not held out")
    try:
        reported_manifest = Path(str(report.get("manifest"))).resolve()
    except (TypeError, ValueError) as exc:
        raise QueueGateError("online report dynamics manifest path is invalid") from exc
    if reported_manifest != job.dynamics_manifest_path.resolve():
        raise QueueGateError("online report used a different dynamics manifest")
    if report.get("manifest_sha256") != job.dynamics_manifest_sha256:
        raise QueueGateError("online report dynamics manifest semantic hash mismatch")

    goal_contract = report.get("goal_contract")
    if not isinstance(goal_contract, dict):
        raise QueueGateError("online report has no goal contract")
    _require_int(goal_contract.get("episodes"), job.episodes, "goal episodes")
    _require_int(goal_contract.get("replicas"), job.num_envs, "goal replicas")
    expected_goal_values = {
        "min_distance": DEFAULT_GOAL_MIN_DISTANCE,
        "max_distance": DEFAULT_GOAL_MAX_DISTANCE,
        "timeout": DEFAULT_GOAL_TIMEOUT,
        "hold_time": DEFAULT_GOAL_HOLD_TIME,
    }
    for key, expected in expected_goal_values.items():
        try:
            actual = float(goal_contract.get(key))
        except (TypeError, ValueError) as exc:
            raise QueueGateError(f"goal contract {key} is invalid") from exc
        if actual != expected:
            raise QueueGateError(f"goal contract {key}={actual}, expected={expected}")

    scenario_sha = str(report.get("scenario_sha256", "")).lower()
    if SHA256_PATTERN.fullmatch(scenario_sha) is None:
        raise QueueGateError("online report scenario_sha256 is missing or incomplete")

    if not job.csv_path.is_file() or job.csv_path.stat().st_size <= 0:
        raise QueueGateError(f"online result CSV is missing or empty: {job.csv_path}")
    reported_csv = Path(str(report.get("result_csv"))).resolve()
    if reported_csv != job.csv_path.resolve():
        raise QueueGateError("online report result_csv path does not match its job directory")
    csv_sha = sha256_file(job.csv_path)
    if report.get("result_csv_sha256") != csv_sha:
        raise QueueGateError("online result CSV SHA-256 does not match the report")

    try:
        with job.csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise QueueGateError(f"cannot read online result CSV: {exc}") from exc
    expected_total = job.episodes * job.num_envs
    if len(rows) != expected_total:
        raise QueueGateError(
            f"online result has {len(rows)} rows, expected exactly {expected_total}"
        )
    required_columns = {
        "episode_index",
        "replica_id",
        "profile_id",
        "goal_seed",
        "relative_x",
        "relative_y",
        "relative_yaw",
        "result",
        "episode_return",
        "position_error",
        "yaw_error",
        "policy_kind",
        "checkpoint_sha256",
        "scenario_sha256",
        "initial_x",
        "initial_y",
        "initial_yaw",
        "initial_pose_sha256",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise QueueGateError(
            f"online result CSV is missing columns: {sorted(required_columns - set(rows[0] if rows else []))}"
        )
    seen: set[tuple[int, int]] = set()
    allowed_results = {"success", "fall", "timeout"}
    for row_index, row in enumerate(rows, 1):
        try:
            episode_index = int(row["episode_index"])
            replica_id = int(row["replica_id"])
            profile_id = int(row["profile_id"])
            goal_seed = int(row["goal_seed"])
            float(row["relative_x"])
            float(row["relative_y"])
            float(row["relative_yaw"])
            float(row["episode_return"])
            float(row["position_error"])
            float(row["yaw_error"])
            initial_x = float(row["initial_x"])
            initial_y = float(row["initial_y"])
            initial_yaw = float(row["initial_yaw"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QueueGateError(f"invalid numeric field in CSV row {row_index}") from exc
        if not 0 <= episode_index < job.episodes or not 0 <= replica_id < job.num_envs:
            raise QueueGateError(f"CSV row {row_index} has an out-of-range episode/replica")
        if profile_id != job.profile_id or goal_seed != job.eval_seed:
            raise QueueGateError(f"CSV row {row_index} does not match the job profile/seed")
        if row.get("result") not in allowed_results:
            raise QueueGateError(f"CSV row {row_index} has invalid result={row.get('result')!r}")
        if row.get("policy_kind") != "t2mir":
            raise QueueGateError(f"CSV row {row_index} is not a T2MIR policy result")
        if row.get("checkpoint_sha256") != job.checkpoint_sha256:
            raise QueueGateError(f"CSV row {row_index} checkpoint SHA-256 mismatch")
        if row.get("scenario_sha256") != scenario_sha:
            raise QueueGateError(f"CSV row {row_index} scenario SHA-256 mismatch")
        expected_pose_sha = pose_sha256(initial_x, initial_y, initial_yaw)
        if row.get("initial_pose_sha256") != expected_pose_sha:
            raise QueueGateError(f"CSV row {row_index} initial-pose SHA-256 mismatch")
        key = (episode_index, replica_id)
        if key in seen:
            raise QueueGateError(f"online result contains duplicate episode/replica {key}")
        seen.add(key)
    expected_keys = {
        (episode_index, replica_id)
        for episode_index in range(job.episodes)
        for replica_id in range(job.num_envs)
    }
    if seen != expected_keys:
        raise QueueGateError("online result does not contain the complete episode/replica grid")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise QueueGateError("online report has no summary")
    _require_int(summary.get("episodes_per_replica"), job.episodes, "summary episodes_per_replica")
    _require_int(summary.get("replicas"), job.num_envs, "summary replicas")
    _require_int(summary.get("total_episodes"), expected_total, "summary total_episodes")
    per_episode = summary.get("per_episode")
    if not isinstance(per_episode, list) or len(per_episode) != job.episodes:
        raise QueueGateError("online summary per_episode is incomplete")
    for episode_index, row in enumerate(per_episode):
        if not isinstance(row, dict):
            raise QueueGateError("online summary per_episode rows must be objects")
        _require_int(row.get("episode_index"), episode_index, "summary episode_index")
        _require_int(row.get("episodes"), job.num_envs, "summary episode count")

    reset_protocol = report.get("reset_protocol")
    if not isinstance(reset_protocol, dict):
        raise QueueGateError("online report has no reset protocol")
    required_reset_flags = {
        "full_vector_reset_before_each_episode": True,
        "episode_boundary_mechanism": EPISODE_BOUNDARY_MECHANISM,
        "explicit_global_reset_calls_after_wrapper_construction": 0,
        "boundary_transition_excluded_from_metrics_and_prompt": True,
        "action_lag_state_crosses_reset": False,
        "initial_pose_recorded_per_row": True,
        "deterministic_reset": True,
    }
    for name, expected in required_reset_flags.items():
        if reset_protocol.get(name) != expected:
            raise QueueGateError(
                f"online reset protocol {name}={reset_protocol.get(name)!r}, expected={expected!r}"
            )
    return {
        job.report_path.name: sha256_file(job.report_path),
        job.csv_path.name: csv_sha,
    }


def classify_job(job: EvaluationJob) -> tuple[str, dict[str, str] | None, str | None]:
    """Return ``skip`` only for a fully valid result; otherwise schedule a run."""

    if not job.report_path.exists() and not job.csv_path.exists():
        return "new", None, None
    try:
        artifacts = validate_complete_result(job)
    except QueueGateError as exc:
        return "rerun_incomplete", None, str(exc)
    return "skip", artifacts, None


@contextlib.contextmanager
def exclusive_gpu_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueueGateError(
                f"training or another evaluator owns the shared GPU lock: {path}"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {"owner": "t2mir_online_evaluation", "pid": os.getpid(), "acquired_at": utc_now()}
            )
            + "\n"
        )
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_with_log(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{utc_now()}] command={json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def _job_state(
    job: EvaluationJob,
    *,
    status: str,
    command: list[str],
    artifacts: dict[str, str] | None = None,
    error: str | None = None,
    return_code: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "status": status,
        "identity": job.identity(),
        "command": command,
        "output_dir": str(job.output_dir),
        "report_path": str(job.report_path),
        "csv_path": str(job.csv_path),
        "log_path": str(job.log_path),
        "artifacts": artifacts,
        "error": error,
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": utc_now(),
    }


def _queue_job_row(
    job: EvaluationJob,
    *,
    action: str,
    status: str,
    command: list[str],
    artifacts: dict[str, str] | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "action": action,
        "identity": job.identity(),
        "output_dir": str(job.output_dir),
        "report_path": str(job.report_path),
        "csv_path": str(job.csv_path),
        "log_path": str(job.log_path),
        "command": command,
        "artifacts": artifacts,
        "reason": reason,
    }


def run_queue(
    *,
    repo: Path,
    queue_state_path: Path,
    jobs: Sequence[EvaluationJob],
) -> None:
    if not jobs:
        raise QueueGateError("evaluation queue contains no jobs")
    classifications = {job.job_id: classify_job(job) for job in jobs}
    commands = {
        job.job_id: evaluator_command(
            job, force=classifications[job.job_id][0] == "rerun_incomplete"
        )
        for job in jobs
    }
    created_at = utc_now()
    first_job = jobs[0]
    variants = list(dict.fromkeys(job.variant for job in jobs))
    train_seeds = list(dict.fromkeys(job.train_seed for job in jobs))
    eval_seeds = list(dict.fromkeys(job.eval_seed for job in jobs))
    profile_ids = list(dict.fromkeys(job.profile_id for job in jobs))
    if len({job.episodes for job in jobs}) != 1 or len({job.num_envs for job in jobs}) != 1:
        raise QueueGateError("all evaluation jobs must share episodes and num_envs")
    queue_state: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "running",
        "created_at": created_at,
        "started_at": created_at,
        "updated_at": created_at,
        "finished_at": None,
        "serial": True,
        "stop_on_failure": True,
        "selection": {
            "variants": variants,
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "profile_ids": profile_ids,
        },
        "contract": {
            "episodes": first_job.episodes,
            "num_envs": first_job.num_envs,
            "task": first_job.task,
            "device": first_job.device,
            "goal_min_distance": DEFAULT_GOAL_MIN_DISTANCE,
            "goal_max_distance": DEFAULT_GOAL_MAX_DISTANCE,
            "goal_timeout": DEFAULT_GOAL_TIMEOUT,
            "goal_hold_time": DEFAULT_GOAL_HOLD_TIME,
            "dynamics_manifest": str(first_job.dynamics_manifest_path),
            "dynamics_manifest_file_sha256": first_job.dynamics_manifest_file_sha256,
            "dynamics_manifest_sha256": first_job.dynamics_manifest_sha256,
            "evaluator": str(first_job.evaluator_path),
            "evaluator_sha256": sha256_file(first_job.evaluator_path),
        },
        "selected_job_count": len(jobs),
        "jobs": {
            job.job_id: _queue_job_row(
                job,
                action=classifications[job.job_id][0],
                status=(
                    "complete_skipped"
                    if classifications[job.job_id][0] == "skip"
                    else "pending"
                ),
                command=commands[job.job_id],
                artifacts=classifications[job.job_id][1],
                reason=classifications[job.job_id][2],
            )
            for job in jobs
        },
    }
    atomic_write_json(queue_state_path, queue_state)
    active_job: EvaluationJob | None = None
    try:
        for index, job in enumerate(jobs, 1):
            action, existing_artifacts, reason = classifications[job.job_id]
            if action == "skip":
                print(
                    f"[T2MIR-EVAL-QUEUE] SKIP_COMPLETE {index}/{len(jobs)} "
                    f"job={job.job_id}",
                    flush=True,
                )
                continue

            active_job = job
            command = commands[job.job_id]
            started_at = utc_now()
            job.output_dir.mkdir(parents=True, exist_ok=True)
            job_state = _job_state(
                job,
                status="running",
                command=command,
                error=reason,
                started_at=started_at,
            )
            atomic_write_json(job.state_path, job_state)
            queue_row = queue_state["jobs"][job.job_id]
            queue_row.update(status="running", started_at=started_at, updated_at=utc_now())
            queue_state["updated_at"] = utc_now()
            atomic_write_json(queue_state_path, queue_state)
            print(
                f"[T2MIR-EVAL-QUEUE] START {index}/{len(jobs)} action={action} "
                f"job={job.job_id}",
                flush=True,
            )
            return_code = run_with_log(command, cwd=repo, log_path=job.log_path)
            if return_code != 0:
                finished_at = utc_now()
                error = f"evaluator exited with return code {return_code}"
                job_state.update(
                    status="failed",
                    return_code=return_code,
                    error=error,
                    finished_at=finished_at,
                    updated_at=finished_at,
                )
                atomic_write_json(job.state_path, job_state)
                queue_row.update(
                    status="failed",
                    return_code=return_code,
                    error=error,
                    finished_at=finished_at,
                    updated_at=finished_at,
                )
                queue_state.update(status="failed", finished_at=finished_at, updated_at=finished_at)
                atomic_write_json(queue_state_path, queue_state)
                raise QueueGateError(
                    f"job {job.job_id} failed with exit code {return_code}; serial queue stopped"
                )
            try:
                artifacts = validate_complete_result(job)
            except QueueGateError as exc:
                finished_at = utc_now()
                error = f"evaluator returned zero but result validation failed: {exc}"
                job_state.update(
                    status="failed",
                    return_code=0,
                    error=error,
                    finished_at=finished_at,
                    updated_at=finished_at,
                )
                atomic_write_json(job.state_path, job_state)
                queue_row.update(
                    status="failed",
                    return_code=0,
                    error=error,
                    finished_at=finished_at,
                    updated_at=finished_at,
                )
                queue_state.update(status="failed", finished_at=finished_at, updated_at=finished_at)
                atomic_write_json(queue_state_path, queue_state)
                raise QueueGateError(error) from exc
            finished_at = utc_now()
            job_state.update(
                status="complete",
                return_code=0,
                error=None,
                artifacts=artifacts,
                finished_at=finished_at,
                updated_at=finished_at,
            )
            atomic_write_json(job.state_path, job_state)
            queue_row.update(
                status="complete",
                return_code=0,
                error=None,
                artifacts=artifacts,
                finished_at=finished_at,
                updated_at=finished_at,
            )
            queue_state["updated_at"] = finished_at
            atomic_write_json(queue_state_path, queue_state)
            print(f"[T2MIR-EVAL-QUEUE] COMPLETE job={job.job_id}", flush=True)
            active_job = None
    except KeyboardInterrupt:
        interrupted_at = utc_now()
        queue_state.update(status="interrupted", updated_at=interrupted_at)
        if active_job is not None:
            row = queue_state["jobs"][active_job.job_id]
            row.update(status="interrupted", updated_at=interrupted_at)
            if active_job.state_path.exists():
                state = load_json(active_job.state_path, label="evaluation job state")
                state.update(status="interrupted", updated_at=interrupted_at)
                atomic_write_json(active_job.state_path, state)
        atomic_write_json(queue_state_path, queue_state)
        print("[T2MIR-EVAL-QUEUE] INTERRUPTED; rerun the same command to resume", flush=True)
        raise

    finished_at = utc_now()
    queue_state.update(status="complete", finished_at=finished_at, updated_at=finished_at)
    atomic_write_json(queue_state_path, queue_state)
    print(f"[T2MIR-EVAL-QUEUE] ALL_COMPLETE jobs={len(jobs)}", flush=True)


def prepare_jobs(
    *,
    repo: Path,
    training_manifest_path: Path,
    dynamics_manifest_path: Path,
    evaluator_path: Path,
    output_root: Path | None,
    python_executable: Path,
    variants: Sequence[str],
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    profiles: Sequence[int],
    episodes: int,
    num_envs: int,
    task: str,
    device: str,
) -> tuple[list[EvaluationJob], Path, Path, dict[str, Any]]:
    manifest, runs_root, gpu_lock = validate_training_manifest(repo, training_manifest_path)
    absent_seeds = sorted(set(train_seeds).difference(manifest["seeds"]))
    if absent_seeds:
        raise QueueGateError(
            f"selected train seeds are absent from the manifest: {absent_seeds}"
        )
    dynamics_manifest = validate_dynamics_manifest(dynamics_manifest_path)
    checkpoints = [
        validate_completed_training_run(
            repo=repo,
            manifest_path=training_manifest_path,
            manifest=manifest,
            runs_root=runs_root,
            variant=variant,
            train_seed=int(train_seed),
        )
        for train_seed in train_seeds
        for variant in variants
    ]
    if output_root is None:
        output_root = (
            repo
            / "outputs"
            / "t2mir_online_stationary"
            / manifest["experiment_name"]
        )
    else:
        output_root = output_root.resolve()
    jobs = build_jobs(
        checkpoints=checkpoints,
        eval_seeds=eval_seeds,
        profiles=profiles,
        episodes=episodes,
        num_envs=num_envs,
        output_root=output_root,
        evaluator_path=evaluator_path,
        dynamics_manifest_path=dynamics_manifest_path,
        dynamics_manifest=dynamics_manifest,
        python_executable=python_executable,
        task=task,
        device=device,
    )
    return jobs, output_root, gpu_lock, manifest


def print_plan(jobs: Sequence[EvaluationJob]) -> None:
    for index, job in enumerate(jobs, 1):
        action, _, reason = classify_job(job)
        command = evaluator_command(job, force=action == "rerun_incomplete")
        print(
            f"[T2MIR-EVAL-PLAN] {index:03d}/{len(jobs):03d} action={action} "
            f"job={job.job_id} output={job.output_dir}",
            flush=True,
        )
        if reason is not None:
            print(f"[T2MIR-EVAL-PLAN] incomplete_reason={reason}", flush=True)
        print(f"[T2MIR-EVAL-COMMAND] {json.dumps(command)}", flush=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_training_manifest = (
        repo
        / "methods/t2mir/configs/robotlab_g1_abcd_training_manifest.json"
    )
    default_dynamics_manifest = repo / "configs/g1_dynamics_48.json"
    default_evaluator = Path(__file__).with_name("pipeline.evaluation.evaluate_t2mir_online.py")
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate completed training contracts/results and print the plan; launch nothing",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Acquire the shared GPU lock and execute the serial evaluation queue",
    )
    parser.add_argument("--training-manifest", type=Path, default=default_training_manifest)
    parser.add_argument("--dynamics-manifest", type=Path, default=default_dynamics_manifest)
    parser.add_argument("--evaluator", type=Path, default=default_evaluator)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--variants", type=parse_variants, default=list(ALLOWED_VARIANTS))
    parser.add_argument("--train-seeds", type=parse_train_seeds, default=[42])
    parser.add_argument("--eval-seeds", type=parse_eval_seeds, default=[42])
    parser.add_argument(
        "--profiles", type=parse_profiles, default=list(REQUIRED_HELD_OUT)
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    if args.execute:
        # Acquire before reading completion contracts so a trainer cannot mutate
        # ``best.pt`` between the gate and simulator launch.
        manifest, _, gpu_lock = validate_training_manifest(
            repo, args.training_manifest.resolve()
        )
        del manifest
        with exclusive_gpu_lock(gpu_lock):
            jobs, output_root, _, training_manifest = prepare_jobs(
                repo=repo,
                training_manifest_path=args.training_manifest.resolve(),
                dynamics_manifest_path=args.dynamics_manifest.resolve(),
                evaluator_path=args.evaluator.resolve(),
                output_root=args.output_root,
                python_executable=args.python_executable.resolve(),
                variants=args.variants,
                train_seeds=args.train_seeds,
                eval_seeds=args.eval_seeds,
                profiles=args.profiles,
                episodes=args.episodes,
                num_envs=args.num_envs,
                task=args.task,
                device=args.device,
            )
            print(
                f"[T2MIR-EVAL-GATE] PASS experiment={training_manifest['experiment_name']} "
                f"jobs={len(jobs)}",
                flush=True,
            )
            print_plan(jobs)
            run_queue(
                repo=repo,
                queue_state_path=output_root / "evaluation_queue_state.json",
                jobs=jobs,
            )
        return

    jobs, _, _, training_manifest = prepare_jobs(
        repo=repo,
        training_manifest_path=args.training_manifest.resolve(),
        dynamics_manifest_path=args.dynamics_manifest.resolve(),
        evaluator_path=args.evaluator.resolve(),
        output_root=args.output_root,
        python_executable=args.python_executable.resolve(),
        variants=args.variants,
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        profiles=args.profiles,
        episodes=args.episodes,
        num_envs=args.num_envs,
        task=args.task,
        device=args.device,
    )
    print(
        f"[T2MIR-EVAL-GATE] PASS experiment={training_manifest['experiment_name']} "
        f"jobs={len(jobs)}",
        flush=True,
    )
    print_plan(jobs)
    print(
        "[T2MIR-EVAL-GATE] DRY_RUN_COMPLETE; no lock acquired, files created, or simulator launched",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except QueueGateError as exc:
        print(f"[T2MIR-EVAL-GATE] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
