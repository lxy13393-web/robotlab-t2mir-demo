"""Validate and aggregate the formal T2MIR stationary online evaluation.

The online evaluator intentionally writes one JSON report and one CSV per
``(variant, train seed, held-out profile, evaluation seed)``.  This module is
the CPU-only, fail-closed consumer of those artifacts.  It validates the
queue's complete declared Cartesian matrix before computing any metric.

The primary metric is success after context is available (episode 1 onward).
The A/D uncertainty estimate is a paired cluster bootstrap.  Its resampling
unit is an entire per-replica episode chain, so no episode from a chain can be
sampled independently of the other episodes in that chain.
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
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipeline.protocols.ppo_online_protocol import EPISODE_BOUNDARY_MECHANISM


REPO = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_DIR = (
    REPO / "outputs/t2mir_online_stationary/official_mixed_v1_abcd_formal"
)
QUEUE_STATE_NAMES = ("evaluation_queue_state.json", "queue_manifest.json")
VARIANT_MODES = {
    "A": ("topk", "topk"),
    "B": ("topp", "topk"),
    "C": ("topk", "topp"),
    "D": ("topp", "topp"),
}
COMPLETE_JOB_STATUSES = {"complete", "complete_skipped", "completed", "skipped"}
RESULT_FILE_RE = re.compile(r"variant[A-D]_task\d+_seed-?\d+\.json$")


class ValidationError(ValueError):
    """Raised when formal result provenance or matrix completeness is invalid."""


@dataclass(frozen=True, order=True)
class JobKey:
    variant: str
    train_seed: int
    profile_id: int
    eval_seed: int

    def label(self) -> str:
        return (
            f"variant={self.variant},train_seed={self.train_seed},"
            f"profile={self.profile_id},eval_seed={self.eval_seed}"
        )


@dataclass(frozen=True)
class ExpectedJob:
    job_id: str
    key: JobKey
    episodes: int
    replicas: int
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    manifest_path: str | None
    manifest_sha256: str | None
    routing_signature: Mapping[str, Any] | None
    report_path: Path
    csv_path: Path
    status: str


@dataclass(frozen=True)
class ValidatedJob:
    expected: ExpectedJob
    report: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_sha256(x: float, y: float, yaw: float) -> str:
    return hashlib.sha256(struct.pack("<fff", x, y, yaw)).hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"expected a JSON object: {path}")
    return value


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _required(mapping: Mapping[str, Any], names: Sequence[str], context: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    raise ValidationError(f"{context} is missing one of {list(names)}")


def _optional(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _as_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{context} must be an integer, got boolean")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValidationError(f"{context} must be an integer, got {value!r}")
    return converted


def _as_float(value: Any, context: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} must be numeric, got {value!r}") from exc
    if not math.isfinite(converted):
        raise ValidationError(f"{context} must be finite, got {value!r}")
    return converted


def _resolve_artifact_path(raw: Any, queue_dir: Path, context: str) -> Path:
    if raw is None or str(raw).strip() == "":
        raise ValidationError(f"{context} path is empty")
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Queue launchers normally serialize absolute paths.  Supporting paths
    # relative to the queue root keeps copied result bundles self-contained.
    return (queue_dir / path).resolve()


def locate_queue_state(queue_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser()
        if not candidate.is_absolute():
            candidate = queue_dir / candidate
        if not candidate.is_file():
            raise ValidationError(f"queue manifest/state does not exist: {candidate}")
        return candidate.resolve()
    matches = [queue_dir / name for name in QUEUE_STATE_NAMES if (queue_dir / name).is_file()]
    if len(matches) != 1:
        raise ValidationError(
            f"expected exactly one queue state in {queue_dir} named one of "
            f"{QUEUE_STATE_NAMES}; found {[str(path) for path in matches]}"
        )
    return matches[0].resolve()


def _selection_list(
    selection: Mapping[str, Any], names: Sequence[str], context: str
) -> tuple[Any, ...]:
    value = _required(selection, names, context)
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{context} must be a non-empty list")
    if len(value) != len({canonical_json(item) for item in value}):
        raise ValidationError(f"{context} contains duplicate values: {value}")
    return tuple(value)


def _job_items(state: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    jobs = _required(state, ("jobs",), "queue state")
    if isinstance(jobs, dict):
        items = list(jobs.items())
    elif isinstance(jobs, list):
        items = []
        for index, value in enumerate(jobs):
            if not isinstance(value, dict):
                raise ValidationError(f"queue jobs[{index}] must be an object")
            job_id = str(value.get("job_id", f"job_{index:04d}"))
            items.append((job_id, value))
    else:
        raise ValidationError("queue state jobs must be a mapping or list")
    if not items:
        raise ValidationError("queue state contains no jobs")
    if len(items) != len({job_id for job_id, _ in items}):
        raise ValidationError("queue state contains duplicate job IDs")
    for job_id, value in items:
        if not isinstance(value, dict):
            raise ValidationError(f"queue job {job_id!r} must be an object")
    return items


def _identity_for_job(job: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = job.get("identity", job)
    if not isinstance(identity, dict):
        raise ValidationError("queue job identity must be an object")
    return identity


def parse_queue_state(
    queue_dir: Path, state_path: Path
) -> tuple[Mapping[str, Any], tuple[ExpectedJob, ...]]:
    state = load_json(state_path)
    if _as_int(state.get("format_version", 1), "queue format_version") != 1:
        raise ValidationError(f"unsupported queue format_version in {state_path}")
    status = str(state.get("status", "")).lower()
    if status not in {"complete", "completed"}:
        raise ValidationError(f"queue is not complete: status={status!r}")

    selection = state.get("selection")
    if not isinstance(selection, dict):
        raise ValidationError("queue state must contain a selection object")
    variants = tuple(
        str(value).upper()
        for value in _selection_list(selection, ("variants",), "selection.variants")
    )
    train_seeds = tuple(
        _as_int(value, "selection.train_seeds")
        for value in _selection_list(
            selection, ("train_seeds", "training_seeds"), "selection.train_seeds"
        )
    )
    eval_seeds = tuple(
        _as_int(value, "selection.eval_seeds")
        for value in _selection_list(
            selection, ("eval_seeds", "evaluation_seeds"), "selection.eval_seeds"
        )
    )
    profile_ids = tuple(
        _as_int(value, "selection.profile_ids")
        for value in _selection_list(
            selection,
            ("profile_ids", "profiles", "eval_profile_ids", "held_out_profile_ids"),
            "selection.profile_ids",
        )
    )
    unknown_variants = sorted(set(variants) - set(VARIANT_MODES))
    if unknown_variants:
        raise ValidationError(f"unsupported variants in queue selection: {unknown_variants}")
    if not {"A", "D"}.issubset(variants):
        raise ValidationError("formal A/D aggregation requires both variants A and D")

    contract = state.get("contract", {})
    if not isinstance(contract, dict):
        raise ValidationError("queue contract must be an object")
    default_episodes = _optional(contract, ("episodes",))
    default_replicas = _optional(contract, ("num_envs", "replicas"))

    parsed: list[ExpectedJob] = []
    seen: dict[JobKey, str] = {}
    for job_id, job in _job_items(state):
        identity = _identity_for_job(job)
        key = JobKey(
            variant=str(_required(identity, ("variant",), f"job {job_id}")).upper(),
            train_seed=_as_int(
                _required(identity, ("train_seed", "training_seed"), f"job {job_id}"),
                f"job {job_id} train_seed",
            ),
            profile_id=_as_int(
                _required(identity, ("profile_id", "task_id"), f"job {job_id}"),
                f"job {job_id} profile_id",
            ),
            eval_seed=_as_int(
                _required(identity, ("eval_seed", "evaluation_seed", "seed"), f"job {job_id}"),
                f"job {job_id} eval_seed",
            ),
        )
        if key in seen:
            raise ValidationError(
                f"duplicate queue matrix entry {key.label()}: jobs {seen[key]!r} and {job_id!r}"
            )
        seen[key] = job_id
        episodes = _as_int(
            _optional(identity, ("episodes",), default_episodes), f"job {job_id} episodes"
        )
        replicas = _as_int(
            _optional(identity, ("num_envs", "replicas"), default_replicas),
            f"job {job_id} replicas",
        )
        if episodes < 2 or replicas <= 0:
            raise ValidationError(
                f"job {job_id} invalid contract: episodes={episodes}, replicas={replicas}"
            )
        output_dir_raw = _optional(job, ("output_dir",))
        output_dir = (
            _resolve_artifact_path(output_dir_raw, queue_dir, f"job {job_id} output_dir")
            if output_dir_raw is not None
            else queue_dir
        )
        stem = f"variant{key.variant}_task{key.profile_id:02d}_seed{key.eval_seed}"
        report_raw = _optional(job, ("report_path", "report"))
        csv_raw = _optional(job, ("csv_path", "result_csv", "csv"))
        report_path = (
            _resolve_artifact_path(report_raw, queue_dir, f"job {job_id} report")
            if report_raw is not None
            else (output_dir / f"{stem}.json").resolve()
        )
        csv_path = (
            _resolve_artifact_path(csv_raw, queue_dir, f"job {job_id} CSV")
            if csv_raw is not None
            else (output_dir / f"{stem}.csv").resolve()
        )
        parsed.append(
            ExpectedJob(
                job_id=job_id,
                key=key,
                episodes=episodes,
                replicas=replicas,
                checkpoint_path=(
                    str(_optional(identity, ("checkpoint_path", "offline_checkpoint", "checkpoint")))
                    if _optional(identity, ("checkpoint_path", "offline_checkpoint", "checkpoint"))
                    is not None
                    else None
                ),
                checkpoint_sha256=_optional(identity, ("checkpoint_sha256",)),
                manifest_path=(
                    str(_optional(identity, ("manifest_path", "manifest", "dynamics_manifest")))
                    if _optional(identity, ("manifest_path", "manifest", "dynamics_manifest")) is not None
                    else None
                ),
                manifest_sha256=_optional(
                    identity, ("manifest_sha256", "dynamics_manifest_sha256")
                ),
                routing_signature=_optional(identity, ("routing_signature",)),
                report_path=report_path,
                csv_path=csv_path,
                status=str(job.get("status", "")).lower(),
            )
        )

    expected_keys = {
        JobKey(variant, train_seed, profile_id, eval_seed)
        for variant in variants
        for train_seed in train_seeds
        for profile_id in profile_ids
        for eval_seed in eval_seeds
    }
    actual_keys = set(seen)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + "; ".join(key.label() for key in missing))
        if extra:
            details.append("extra=" + "; ".join(key.label() for key in extra))
        raise ValidationError("queue job matrix does not equal selection Cartesian product: " + " | ".join(details))
    return state, tuple(sorted(parsed, key=lambda job: job.key))


def _normalized_path(value: str, queue_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = queue_dir / path
    return str(path.resolve())


def _expect_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ValidationError(f"{context} mismatch: actual={actual!r}, expected={expected!r}")


def _expect_close(actual: Any, expected: Any, context: str, tolerance: float = 1e-12) -> None:
    actual_float = _as_float(actual, context)
    expected_float = _as_float(expected, context)
    if not math.isclose(actual_float, expected_float, rel_tol=tolerance, abs_tol=tolerance):
        raise ValidationError(
            f"{context} mismatch: actual={actual_float!r}, expected={expected_float!r}"
        )


def read_result_csv(path: Path, expected: ExpectedJob) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise ValidationError(f"missing result CSV for {expected.key.label()}: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValidationError(f"result CSV has no header: {path}")
            required = {
                "episode_index",
                "replica_id",
                "profile_id",
                "goal_seed",
                "relative_x",
                "relative_y",
                "relative_yaw",
                "result",
                "policy_kind",
                "checkpoint_sha256",
                "scenario_sha256",
                "initial_x",
                "initial_y",
                "initial_yaw",
                "initial_pose_sha256",
            }
            missing_columns = sorted(required - set(reader.fieldnames))
            if missing_columns:
                raise ValidationError(f"result CSV {path} missing columns {missing_columns}")
            rows = list(reader)
    except OSError as exc:
        raise ValidationError(f"cannot read result CSV {path}: {exc}") from exc

    expected_count = expected.episodes * expected.replicas
    if len(rows) != expected_count:
        raise ValidationError(
            f"result CSV row count mismatch for {expected.key.label()}: "
            f"{len(rows)} != {expected.episodes}*{expected.replicas}={expected_count}"
        )
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row_number, raw in enumerate(rows, start=2):
        context = f"{path}:{row_number}"
        episode = _as_int(raw["episode_index"], f"{context} episode_index")
        replica = _as_int(raw["replica_id"], f"{context} replica_id")
        coordinate = (episode, replica)
        if coordinate in seen:
            raise ValidationError(f"duplicate episode/replica row {coordinate} in {path}")
        seen.add(coordinate)
        if not 0 <= episode < expected.episodes:
            raise ValidationError(f"{context} episode_index out of range: {episode}")
        if not 0 <= replica < expected.replicas:
            raise ValidationError(f"{context} replica_id out of range: {replica}")
        profile_id = _as_int(raw["profile_id"], f"{context} profile_id")
        goal_seed = _as_int(raw["goal_seed"], f"{context} goal_seed")
        _expect_equal(profile_id, expected.key.profile_id, f"{context} profile_id")
        _expect_equal(goal_seed, expected.key.eval_seed, f"{context} goal_seed")
        outcome = str(raw["result"]).lower()
        if outcome not in {"success", "fall", "timeout"}:
            raise ValidationError(f"{context} invalid result {outcome!r}")
        normalized = dict(raw)
        initial_x = _as_float(raw["initial_x"], f"{context} initial_x")
        initial_y = _as_float(raw["initial_y"], f"{context} initial_y")
        initial_yaw = _as_float(raw["initial_yaw"], f"{context} initial_yaw")
        expected_pose_sha = pose_sha256(initial_x, initial_y, initial_yaw)
        _expect_equal(
            raw["initial_pose_sha256"], expected_pose_sha, f"{context} initial pose SHA"
        )
        normalized.update(
            {
                "episode_index": episode,
                "replica_id": replica,
                "profile_id": profile_id,
                "goal_seed": goal_seed,
                "relative_x": _as_float(raw["relative_x"], f"{context} relative_x"),
                "relative_y": _as_float(raw["relative_y"], f"{context} relative_y"),
                "relative_yaw": _as_float(raw["relative_yaw"], f"{context} relative_yaw"),
                "result": outcome,
                "initial_x": initial_x,
                "initial_y": initial_y,
                "initial_yaw": initial_yaw,
            }
        )
        result.append(normalized)
    expected_coordinates = {
        (episode, replica)
        for episode in range(expected.episodes)
        for replica in range(expected.replicas)
    }
    if seen != expected_coordinates:
        missing = sorted(expected_coordinates - seen)
        raise ValidationError(f"result CSV {path} has an incomplete grid; missing={missing}")
    return tuple(sorted(result, key=lambda row: (row["episode_index"], row["replica_id"])))


def _validate_report_summary(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], expected: ExpectedJob
) -> None:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValidationError(f"report {expected.report_path} has no summary object")
    _expect_equal(
        _as_int(summary.get("episodes_per_replica"), "summary episodes_per_replica"),
        expected.episodes,
        "summary episodes_per_replica",
    )
    _expect_equal(
        _as_int(summary.get("replicas"), "summary replicas"),
        expected.replicas,
        "summary replicas",
    )
    _expect_equal(
        _as_int(summary.get("total_episodes"), "summary total_episodes"),
        len(rows),
        "summary total_episodes",
    )
    successes = sum(row["result"] == "success" for row in rows)
    falls = sum(row["result"] == "fall" for row in rows)
    _expect_close(summary.get("success_rate"), successes / len(rows), "summary success_rate")
    _expect_close(summary.get("fall_rate"), falls / len(rows), "summary fall_rate")
    per_episode = summary.get("per_episode")
    if not isinstance(per_episode, list) or len(per_episode) != expected.episodes:
        raise ValidationError(
            f"report summary per_episode must contain {expected.episodes} entries"
        )
    by_index: dict[int, Mapping[str, Any]] = {}
    for item in per_episode:
        if not isinstance(item, dict):
            raise ValidationError("report summary per_episode entries must be objects")
        index = _as_int(item.get("episode_index"), "summary per_episode episode_index")
        if index in by_index:
            raise ValidationError(f"report summary duplicates episode {index}")
        by_index[index] = item
    for episode in range(expected.episodes):
        if episode not in by_index:
            raise ValidationError(f"report summary is missing episode {episode}")
        selected = [row for row in rows if row["episode_index"] == episode]
        item = by_index[episode]
        _expect_equal(_as_int(item.get("episodes"), "per_episode episodes"), len(selected), "per_episode episodes")
        expected_successes = sum(row["result"] == "success" for row in selected)
        expected_falls = sum(row["result"] == "fall" for row in selected)
        _expect_equal(_as_int(item.get("successes"), "per_episode successes"), expected_successes, "per_episode successes")
        _expect_equal(_as_int(item.get("falls"), "per_episode falls"), expected_falls, "per_episode falls")
        _expect_close(item.get("success_rate"), expected_successes / len(selected), "per_episode success_rate")
        _expect_close(item.get("fall_rate"), expected_falls / len(selected), "per_episode fall_rate")


def validate_job(expected: ExpectedJob, queue_dir: Path) -> ValidatedJob:
    if expected.status not in COMPLETE_JOB_STATUSES:
        raise ValidationError(
            f"queue job {expected.job_id} is not complete: status={expected.status!r}"
        )
    if not expected.report_path.is_file():
        raise ValidationError(
            f"missing result report for {expected.key.label()}: {expected.report_path}"
        )
    report = load_json(expected.report_path)
    _expect_equal(report.get("evaluation"), "source_faithful_stationary_online_dpt", "evaluation type")
    _expect_equal(str(report.get("variant", "")).upper(), expected.key.variant, "report variant")
    _expect_equal(_as_int(report.get("seed"), "report seed"), expected.key.eval_seed, "report seed")
    profile = report.get("profile")
    if not isinstance(profile, dict):
        raise ValidationError(f"report {expected.report_path} has no profile object")
    _expect_equal(
        _as_int(profile.get("task_id"), "report profile task_id"),
        expected.key.profile_id,
        "report profile task_id",
    )
    _expect_equal(profile.get("split"), "eval", "report profile split")
    goal_contract = report.get("goal_contract")
    if not isinstance(goal_contract, dict):
        raise ValidationError(f"report {expected.report_path} has no goal_contract")
    _expect_equal(_as_int(goal_contract.get("episodes"), "goal episodes"), expected.episodes, "goal episodes")
    _expect_equal(_as_int(goal_contract.get("replicas"), "goal replicas"), expected.replicas, "goal replicas")
    for field in ("min_distance", "max_distance", "timeout", "hold_time"):
        _as_float(goal_contract.get(field), f"goal_contract.{field}")
    scenario = str(report.get("scenario_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", scenario):
        raise ValidationError(f"invalid scenario_sha256 in {expected.report_path}: {scenario!r}")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValidationError(f"report {expected.report_path} has no checkpoint object")
    checkpoint_sha = str(checkpoint.get("checkpoint_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
        raise ValidationError(f"invalid checkpoint SHA in {expected.report_path}")
    if expected.checkpoint_sha256 is not None:
        _expect_equal(checkpoint_sha, str(expected.checkpoint_sha256).lower(), "checkpoint SHA")
    if expected.checkpoint_path is not None:
        actual_path = checkpoint.get("checkpoint")
        if not isinstance(actual_path, str):
            raise ValidationError("checkpoint provenance is missing checkpoint path")
        _expect_equal(
            _normalized_path(actual_path, queue_dir),
            _normalized_path(expected.checkpoint_path, queue_dir),
            "checkpoint path",
        )
    routing = checkpoint.get("routing_signature")
    if not isinstance(routing, dict):
        raise ValidationError(f"report {expected.report_path} has no routing signature")
    if expected.routing_signature is not None:
        _expect_equal(routing, expected.routing_signature, "routing signature")
    try:
        modes = (str(routing["token"]["mode"]), str(routing["task"]["mode"]))
    except (KeyError, TypeError) as exc:
        raise ValidationError(f"malformed routing signature in {expected.report_path}") from exc
    _expect_equal(modes, VARIANT_MODES[expected.key.variant], "variant routing modes")

    report_manifest_sha = str(report.get("manifest_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", report_manifest_sha):
        raise ValidationError(f"invalid manifest_sha256 in {expected.report_path}")
    if expected.manifest_sha256 is not None:
        _expect_equal(
            report_manifest_sha,
            str(expected.manifest_sha256),
            "dynamics manifest SHA",
        )
    if expected.manifest_path is not None:
        actual_manifest = report.get("manifest")
        if not isinstance(actual_manifest, str):
            raise ValidationError("report is missing dynamics manifest path")
        _expect_equal(
            _normalized_path(actual_manifest, queue_dir),
            _normalized_path(expected.manifest_path, queue_dir),
            "dynamics manifest path",
        )
    code_sha = report.get("code_sha256")
    if not isinstance(code_sha, dict) or not code_sha:
        raise ValidationError(f"report {expected.report_path} has no code_sha256 mapping")
    for name, digest in code_sha.items():
        if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValidationError(f"malformed code_sha256 entry in {expected.report_path}")

    report_csv = report.get("result_csv")
    if not isinstance(report_csv, str):
        raise ValidationError(f"report {expected.report_path} has no result_csv path")
    _expect_equal(
        _normalized_path(report_csv, queue_dir), str(expected.csv_path), "report result_csv path"
    )
    rows = read_result_csv(expected.csv_path, expected)
    for row in rows:
        coordinate = (row["episode_index"], row["replica_id"])
        _expect_equal(row.get("policy_kind"), "t2mir", f"row {coordinate} policy kind")
        _expect_equal(
            row.get("checkpoint_sha256"), checkpoint_sha, f"row {coordinate} checkpoint SHA"
        )
        _expect_equal(
            row.get("scenario_sha256"), scenario, f"row {coordinate} scenario SHA"
        )
    reset_protocol = report.get("reset_protocol")
    if not isinstance(reset_protocol, dict):
        raise ValidationError(f"report {expected.report_path} has no reset_protocol")
    for name, expected_value in {
        "full_vector_reset_before_each_episode": True,
        "episode_boundary_mechanism": EPISODE_BOUNDARY_MECHANISM,
        "explicit_global_reset_calls_after_wrapper_construction": 0,
        "boundary_transition_excluded_from_metrics_and_prompt": True,
        "action_lag_state_crosses_reset": False,
        "initial_pose_recorded_per_row": True,
        "deterministic_reset": True,
    }.items():
        _expect_equal(
            reset_protocol.get(name), expected_value, f"reset_protocol.{name}"
        )
    actual_csv_sha = sha256_file(expected.csv_path)
    _expect_equal(
        str(report.get("result_csv_sha256", "")).lower(), actual_csv_sha, "result CSV SHA"
    )
    _validate_report_summary(report, rows, expected)
    return ValidatedJob(expected=expected, report=report, rows=rows)


def _common_value(jobs: Sequence[ValidatedJob], getter, label: str) -> Any:
    values: dict[str, tuple[Any, list[str]]] = {}
    for job in jobs:
        value = getter(job)
        fingerprint = canonical_json(value)
        values.setdefault(fingerprint, (value, []))[1].append(job.expected.key.label())
    if len(values) != 1:
        groups = [f"{keys}: {value!r}" for value, keys in values.values()]
        raise ValidationError(f"{label} is inconsistent across reports: " + " | ".join(groups))
    return next(iter(values.values()))[0]


def _routing_without_modes(signature: Mapping[str, Any]) -> Mapping[str, Any]:
    value = json.loads(json.dumps(signature))
    for branch in ("token", "task"):
        if isinstance(value.get(branch), dict):
            value[branch].pop("mode", None)
    return value


def validate_cross_job_contracts(jobs: Sequence[ValidatedJob]) -> dict[str, Any]:
    if not jobs:
        raise ValidationError("no validated result jobs")
    goal_contract = _common_value(jobs, lambda job: job.report["goal_contract"], "goal contract")
    reset_protocol = _common_value(
        jobs, lambda job: job.report["reset_protocol"], "reset protocol"
    )
    manifest_sha = _common_value(jobs, lambda job: job.report["manifest_sha256"], "manifest SHA")
    code_sha = _common_value(jobs, lambda job: job.report["code_sha256"], "evaluation code SHA")

    checkpoint_groups: dict[tuple[str, int], list[ValidatedJob]] = defaultdict(list)
    routing_groups: dict[str, list[ValidatedJob]] = defaultdict(list)
    for job in jobs:
        checkpoint_groups[(job.expected.key.variant, job.expected.key.train_seed)].append(job)
        routing_groups[job.expected.key.variant].append(job)
    checkpoint_summary: dict[str, Any] = {}
    for (variant, train_seed), selected in sorted(checkpoint_groups.items()):
        checkpoint_sha = _common_value(
            selected,
            lambda job: job.report["checkpoint"]["checkpoint_sha256"],
            f"checkpoint SHA for variant={variant},train_seed={train_seed}",
        )
        checkpoint_path = _common_value(
            selected,
            lambda job: job.report["checkpoint"]["checkpoint"],
            f"checkpoint path for variant={variant},train_seed={train_seed}",
        )
        checkpoint_summary[f"{variant}/seed_{train_seed}"] = {
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": selected[0].report["checkpoint"].get("checkpoint_step"),
        }
    routing_summary: dict[str, Any] = {}
    for variant, selected in sorted(routing_groups.items()):
        routing_summary[variant] = _common_value(
            selected,
            lambda job: job.report["checkpoint"]["routing_signature"],
            f"routing signature for variant={variant}",
        )
    routing_controls = {
        canonical_json(_routing_without_modes(signature)) for signature in routing_summary.values()
    }
    if len(routing_controls) != 1:
        raise ValidationError(
            "variants differ in routing-signature fields other than the declared routing modes"
        )
    return {
        "goal_contract": goal_contract,
        "reset_protocol": reset_protocol,
        "manifest_sha256": manifest_sha,
        "code_sha256": code_sha,
        "checkpoints": checkpoint_summary,
        "routing_signatures": routing_summary,
    }


def _job_map(jobs: Sequence[ValidatedJob]) -> dict[JobKey, ValidatedJob]:
    result: dict[JobKey, ValidatedJob] = {}
    for job in jobs:
        if job.expected.key in result:
            raise ValidationError(f"duplicate validated job {job.expected.key.label()}")
        result[job.expected.key] = job
    return result


def validate_paired_scenarios(jobs: Sequence[ValidatedJob]) -> None:
    by_key = _job_map(jobs)
    base_keys = {
        (key.train_seed, key.profile_id, key.eval_seed)
        for key in by_key
        if key.variant in {"A", "D"}
    }
    for train_seed, profile_id, eval_seed in sorted(base_keys):
        key_a = JobKey("A", train_seed, profile_id, eval_seed)
        key_d = JobKey("D", train_seed, profile_id, eval_seed)
        if key_a not in by_key or key_d not in by_key:
            raise ValidationError(
                "A/D pairing is incomplete for "
                f"train_seed={train_seed},profile={profile_id},eval_seed={eval_seed}"
            )
        job_a, job_d = by_key[key_a], by_key[key_d]
        scenario_a = job_a.report["scenario_sha256"]
        scenario_d = job_d.report["scenario_sha256"]
        if scenario_a != scenario_d:
            raise ValidationError(
                "A/D scenario SHA mismatch for "
                f"train_seed={train_seed},profile={profile_id},eval_seed={eval_seed}: "
                f"A={scenario_a}, D={scenario_d}"
            )
        rows_a = {
            (row["episode_index"], row["replica_id"]): row for row in job_a.rows
        }
        rows_d = {
            (row["episode_index"], row["replica_id"]): row for row in job_d.rows
        }
        _expect_equal(set(rows_a), set(rows_d), "A/D episode/replica grid")
        for goal_key in sorted(rows_a):
            goal_a = tuple(rows_a[goal_key][name] for name in ("relative_x", "relative_y", "relative_yaw"))
            goal_d = tuple(rows_d[goal_key][name] for name in ("relative_x", "relative_y", "relative_yaw"))
            if goal_a != goal_d:
                raise ValidationError(
                    "A/D goal mismatch for "
                    f"train_seed={train_seed},profile={profile_id},eval_seed={eval_seed},"
                    f"episode={goal_key[0]},replica={goal_key[1]}: A={goal_a}, D={goal_d}"
                )
            pose_a = tuple(
                rows_a[goal_key][name]
                for name in ("initial_x", "initial_y", "initial_yaw", "initial_pose_sha256")
            )
            pose_d = tuple(
                rows_d[goal_key][name]
                for name in ("initial_x", "initial_y", "initial_yaw", "initial_pose_sha256")
            )
            if pose_a != pose_d:
                raise ValidationError(
                    "A/D initial-pose mismatch for "
                    f"train_seed={train_seed},profile={profile_id},eval_seed={eval_seed},"
                    f"episode={goal_key[0]},replica={goal_key[1]}: A={pose_a}, D={pose_d}"
                )


def discover_result_reports(queue_dir: Path) -> set[Path]:
    result: set[Path] = set()
    for path in queue_dir.rglob("*.json"):
        if RESULT_FILE_RE.search(path.name):
            result.add(path.resolve())
    return result


def validate_no_unexpected_reports(queue_dir: Path, expected: Sequence[ExpectedJob]) -> None:
    declared = {job.report_path for job in expected}
    discovered = discover_result_reports(queue_dir)
    unexpected = sorted(discovered - declared)
    if unexpected:
        raise ValidationError(
            "queue directory contains result reports not declared by its matrix: "
            + ", ".join(str(path) for path in unexpected)
        )


def rate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValidationError("cannot summarize zero result rows")
    successes = sum(row["result"] == "success" for row in rows)
    falls = sum(row["result"] == "fall" for row in rows)
    timeouts = count - successes - falls
    return {
        "trials": count,
        "successes": successes,
        "falls": falls,
        "timeouts": timeouts,
        "success_rate": successes / count,
        "fall_rate": falls / count,
        "timeout_rate": timeouts / count,
    }


def _rows_with_identity(jobs: Sequence[ValidatedJob], variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.expected.key.variant != variant:
            continue
        for raw in job.rows:
            row = dict(raw)
            row.update(
                {
                    "variant": variant,
                    "train_seed": job.expected.key.train_seed,
                    "eval_seed": job.expected.key.eval_seed,
                }
            )
            rows.append(row)
    return rows


def variant_metrics(jobs: Sequence[ValidatedJob], variant: str, episodes: int) -> dict[str, Any]:
    rows = _rows_with_identity(jobs, variant)
    episode_zero = [row for row in rows if row["episode_index"] == 0]
    post_context = [row for row in rows if row["episode_index"] >= 1]
    per_episode = []
    for episode in range(episodes):
        summary = rate_summary([row for row in rows if row["episode_index"] == episode])
        per_episode.append({"episode_index": episode, **summary})
    per_profile = []
    for profile_id in sorted({row["profile_id"] for row in rows}):
        selected = [
            row
            for row in post_context
            if row["profile_id"] == profile_id
        ]
        per_profile.append({"profile_id": profile_id, **rate_summary(selected)})
    worst = min(per_profile, key=lambda row: (row["success_rate"], row["profile_id"]))
    return {
        "all_episodes": rate_summary(rows),
        "episode_0_no_context": rate_summary(episode_zero),
        "post_context_primary": rate_summary(post_context),
        "per_episode": per_episode,
        "per_profile_post_context": per_profile,
        "worst_profile_post_context": dict(worst),
        "adaptation_success_delta_last_minus_episode_0": (
            per_episode[-1]["success_rate"] - per_episode[0]["success_rate"]
        ),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValidationError("cannot compute a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(
    values_a: Sequence[float],
    values_d: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Return deterministic paired cluster-bootstrap statistics for D minus A."""

    if len(values_a) != len(values_d) or not values_a:
        raise ValidationError(
            f"paired bootstrap requires equal non-empty vectors: {len(values_a)} vs {len(values_d)}"
        )
    if samples <= 0:
        raise ValidationError("bootstrap samples must be positive")
    deltas = [float(value_d) - float(value_a) for value_a, value_d in zip(values_a, values_d)]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    mean_draw = sum(draws) / len(draws)
    standard_error = math.sqrt(
        sum((value - mean_draw) ** 2 for value in draws) / max(len(draws) - 1, 1)
    )
    return {
        "clusters": len(deltas),
        "samples": samples,
        "seed": seed,
        "observed_d_minus_a": observed,
        "observed_a_minus_d": -observed,
        "ci95_d_minus_a": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "bootstrap_standard_error": standard_error,
    }


ClusterKey = tuple[int, int, int, int]  # train seed, profile, eval seed, replica


def _variant_chains(
    jobs: Sequence[ValidatedJob], variant: str, episodes: int
) -> dict[ClusterKey, tuple[Mapping[str, Any], ...]]:
    grouped: dict[ClusterKey, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _rows_with_identity(jobs, variant):
        key = (
            int(row["train_seed"]),
            int(row["profile_id"]),
            int(row["eval_seed"]),
            int(row["replica_id"]),
        )
        grouped[key].append(row)
    result: dict[ClusterKey, tuple[Mapping[str, Any], ...]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["episode_index"])
        indices = [row["episode_index"] for row in rows]
        if indices != list(range(episodes)):
            raise ValidationError(f"incomplete episode chain for variant={variant}, cluster={key}: {indices}")
        result[key] = tuple(rows)
    return result


def _chain_metric(chain: Sequence[Mapping[str, Any]], metric: str, episodes: Iterable[int]) -> float:
    selected_indices = set(episodes)
    selected = [row for row in chain if row["episode_index"] in selected_indices]
    if not selected:
        raise ValidationError("cluster metric selected no episodes")
    if metric == "success":
        return sum(row["result"] == "success" for row in selected) / len(selected)
    if metric == "fall":
        return sum(row["result"] == "fall" for row in selected) / len(selected)
    raise ValueError(f"unsupported chain metric: {metric}")


def paired_comparison(
    jobs: Sequence[ValidatedJob],
    *,
    episodes: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chains_a = _variant_chains(jobs, "A", episodes)
    chains_d = _variant_chains(jobs, "D", episodes)
    if set(chains_a) != set(chains_d):
        missing_a = sorted(set(chains_d) - set(chains_a))
        missing_d = sorted(set(chains_a) - set(chains_d))
        raise ValidationError(
            f"A/D cluster sets differ: missing_A={missing_a}, missing_D={missing_d}"
        )
    keys = sorted(chains_a)

    def compare(metric: str, episode_indices: Iterable[int], seed_offset: int) -> dict[str, Any]:
        selected = tuple(episode_indices)
        values_a = [_chain_metric(chains_a[key], metric, selected) for key in keys]
        values_d = [_chain_metric(chains_d[key], metric, selected) for key in keys]
        result = paired_cluster_bootstrap(
            values_a,
            values_d,
            samples=bootstrap_samples,
            seed=bootstrap_seed + seed_offset,
        )
        result.update(
            {
                "a_rate": sum(values_a) / len(values_a),
                "d_rate": sum(values_d) / len(values_d),
                "episode_indices": list(selected),
            }
        )
        return result

    post_context = compare("success", range(1, episodes), 0)
    episode_zero = compare("success", (0,), 1)
    post_context_fall = compare("fall", range(1, episodes), 2)
    per_episode = []
    for episode in range(episodes):
        per_episode.append(
            {
                "episode_index": episode,
                "success": compare("success", (episode,), 100 + episode),
                "fall": compare("fall", (episode,), 1000 + episode),
            }
        )
    cluster_rows: list[dict[str, Any]] = []
    for key in keys:
        train_seed, profile_id, eval_seed, replica_id = key
        a_post = _chain_metric(chains_a[key], "success", range(1, episodes))
        d_post = _chain_metric(chains_d[key], "success", range(1, episodes))
        a_fall = _chain_metric(chains_a[key], "fall", range(1, episodes))
        d_fall = _chain_metric(chains_d[key], "fall", range(1, episodes))
        cluster_rows.append(
            {
                "train_seed": train_seed,
                "profile_id": profile_id,
                "eval_seed": eval_seed,
                "replica_id": replica_id,
                "a_episode0_success": _chain_metric(chains_a[key], "success", (0,)),
                "d_episode0_success": _chain_metric(chains_d[key], "success", (0,)),
                "a_post_context_success": a_post,
                "d_post_context_success": d_post,
                "d_minus_a_post_context_success": d_post - a_post,
                "a_post_context_fall": a_fall,
                "d_post_context_fall": d_fall,
                "d_minus_a_post_context_fall": d_fall - a_fall,
            }
        )
    comparison = {
        "orientation": "All reported paired deltas are D minus A; positive success delta favors D, negative fall delta favors D.",
        "cluster_definition": ["profile_id", "eval_seed", "replica_id"],
        "pairing_stratum": ["train_seed"],
        "complete_episode_chain_required": True,
        "cluster_count": len(keys),
        "post_context_success_primary": post_context,
        "episode_0_success": episode_zero,
        "post_context_fall": post_context_fall,
        "per_episode": per_episode,
    }
    return comparison, cluster_rows


def _metric_csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, metrics in summary["variants"].items():
        for scope, value in (
            ("episode_0_no_context", metrics["episode_0_no_context"]),
            ("post_context_primary", metrics["post_context_primary"]),
            ("all_episodes", metrics["all_episodes"]),
        ):
            rows.append(
                {
                    "record_type": "variant_scope",
                    "variant": variant,
                    "scope": scope,
                    "episode_index": "",
                    "profile_id": "",
                    **value,
                    "d_minus_a": "",
                    "ci95_low": "",
                    "ci95_high": "",
                }
            )
        for value in metrics["per_episode"]:
            rows.append(
                {
                    "record_type": "variant_episode",
                    "variant": variant,
                    "scope": "per_episode",
                    "episode_index": value["episode_index"],
                    "profile_id": "",
                    **{key: item for key, item in value.items() if key != "episode_index"},
                    "d_minus_a": "",
                    "ci95_low": "",
                    "ci95_high": "",
                }
            )
        for value in metrics["per_profile_post_context"]:
            rows.append(
                {
                    "record_type": "variant_profile",
                    "variant": variant,
                    "scope": "post_context",
                    "episode_index": "",
                    "profile_id": value["profile_id"],
                    **{key: item for key, item in value.items() if key != "profile_id"},
                    "d_minus_a": "",
                    "ci95_low": "",
                    "ci95_high": "",
                }
            )
    comparison = summary["comparison_a_d"]
    for scope, value in (
        ("post_context_success_primary", comparison["post_context_success_primary"]),
        ("episode_0_success", comparison["episode_0_success"]),
        ("post_context_fall", comparison["post_context_fall"]),
    ):
        rows.append(
            {
                "record_type": "paired_comparison",
                "variant": "D-A",
                "scope": scope,
                "episode_index": "",
                "profile_id": "",
                "trials": "",
                "successes": "",
                "falls": "",
                "timeouts": "",
                "success_rate": "",
                "fall_rate": "",
                "timeout_rate": "",
                "d_minus_a": value["observed_d_minus_a"],
                "ci95_low": value["ci95_d_minus_a"][0],
                "ci95_high": value["ci95_d_minus_a"][1],
            }
        )
    return rows


METRIC_CSV_FIELDS = [
    "record_type",
    "variant",
    "scope",
    "episode_index",
    "profile_id",
    "trials",
    "successes",
    "falls",
    "timeouts",
    "success_rate",
    "fall_rate",
    "timeout_rate",
    "d_minus_a",
    "ci95_low",
    "ci95_high",
]


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate_queue(
    queue_dir: Path,
    *,
    queue_state_path: Path | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_928,
    check_unexpected_reports: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue_dir = queue_dir.expanduser().resolve()
    state_path = locate_queue_state(queue_dir, queue_state_path)
    state, expected = parse_queue_state(queue_dir, state_path)
    if check_unexpected_reports:
        validate_no_unexpected_reports(queue_dir, expected)
    validated = tuple(validate_job(job, queue_dir) for job in expected)
    common = validate_cross_job_contracts(validated)
    validate_paired_scenarios(validated)

    episodes = next(iter({job.expected.episodes for job in validated}), None)
    replicas = next(iter({job.expected.replicas for job in validated}), None)
    if len({job.expected.episodes for job in validated}) != 1:
        raise ValidationError("episode count is inconsistent across queue jobs")
    if len({job.expected.replicas for job in validated}) != 1:
        raise ValidationError("replica count is inconsistent across queue jobs")
    assert episodes is not None and replicas is not None
    variants = sorted({job.expected.key.variant for job in validated})
    metrics = {variant: variant_metrics(validated, variant, episodes) for variant in variants}
    comparison, cluster_rows = paired_comparison(
        validated,
        episodes=episodes,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    selection = state["selection"]
    summary = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "status": "PASS",
            "queue_dir": str(queue_dir),
            "queue_state": str(state_path),
            "queue_state_sha256": sha256_file(state_path),
            "expected_jobs": len(expected),
            "validated_jobs": len(validated),
            "validated_csv_rows": sum(len(job.rows) for job in validated),
            "matrix_is_exact_cartesian_product": True,
            "a_d_scenario_and_goal_pairing": "PASS",
        },
        "selection": selection,
        "protocol": {
            "episodes": episodes,
            "replicas": replicas,
            **common,
        },
        "primary_metric": {
            "name": "post_context_success_rate",
            "episode_indices": list(range(1, episodes)),
            "description": "Success rate from episode 1 onward, after one or more self-generated context episodes are available.",
        },
        "variants": metrics,
        "comparison_a_d": comparison,
    }
    return summary, cluster_rows


def write_outputs(
    summary: Mapping[str, Any],
    cluster_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    json_path = output_dir / "aggregate_summary.json"
    metrics_path = output_dir / "aggregate_metrics.csv"
    clusters_path = output_dir / "paired_clusters.csv"
    atomic_write_text(json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv_atomic(metrics_path, METRIC_CSV_FIELDS, _metric_csv_rows(summary))
    cluster_fields = [
        "train_seed",
        "profile_id",
        "eval_seed",
        "replica_id",
        "a_episode0_success",
        "d_episode0_success",
        "a_post_context_success",
        "d_post_context_success",
        "d_minus_a_post_context_success",
        "a_post_context_fall",
        "d_post_context_fall",
        "d_minus_a_post_context_fall",
    ]
    write_csv_atomic(clusters_path, cluster_fields, cluster_rows)
    return {"json": json_path, "metrics_csv": metrics_path, "clusters_csv": clusters_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument(
        "--queue-state",
        "--queue-manifest",
        dest="queue_state",
        type=Path,
        default=None,
        help="Queue state/manifest (default: discover it under --queue-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <queue-dir>/aggregate",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_928)
    parser.add_argument(
        "--allow-unexpected-reports",
        action="store_true",
        help="Do not reject evaluator reports that are not declared in the queue matrix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.queue_dir / "aggregate"
    try:
        summary, cluster_rows = aggregate_queue(
            args.queue_dir,
            queue_state_path=args.queue_state,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            check_unexpected_reports=not args.allow_unexpected_reports,
        )
        paths = write_outputs(summary, cluster_rows, output_dir)
    except ValidationError as exc:
        print(f"[T2MIR-ONLINE-AGGREGATE] FAIL: {exc}", file=sys.stderr)
        return 2
    primary = summary["comparison_a_d"]["post_context_success_primary"]
    print(
        "[T2MIR-ONLINE-AGGREGATE] PASS "
        f"jobs={summary['validation']['validated_jobs']} "
        f"clusters={primary['clusters']} "
        f"D-A={primary['observed_d_minus_a']:.6f} "
        f"CI95={primary['ci95_d_minus_a']}",
        flush=True,
    )
    for name, path in paths.items():
        print(f"[T2MIR-ONLINE-AGGREGATE] {name}={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
