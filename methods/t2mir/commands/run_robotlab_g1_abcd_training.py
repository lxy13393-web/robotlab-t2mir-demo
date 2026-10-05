"""Guarded, manifest-driven serial launcher for the RobotLab G1 A--D ablation.

This launcher deliberately does not build or repair a dataset.  It refuses to
start a GPU job until the formal 42-task dataset has passed the complete
RobotLab validator and the staging queue records ``ALL_COMPLETE``.  A single
process/file lock then serializes every variant/seed job.
"""

from __future__ import annotations

# Make the method root importable when this command is run by file path.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _method_root = _Path(__file__).resolve().parents[1]
    if str(_method_root) not in _sys.path:
        _sys.path.insert(0, str(_method_root))

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


FORMAT_VERSION = 1
EXPECTED_VARIANT_MODES = {
    "A": ("topk", "topk"),
    "B": ("topp", "topk"),
    "C": ("topk", "topp"),
    "D": ("topp", "topp"),
}
REQUIRED_HELD_OUT = [5, 14, 23, 32, 41, 47]
REQUIRED_TRAIN_TASKS = sorted(set(range(48)).difference(REQUIRED_HELD_OUT))
ROUTING_KEYS = {
    "routing_mode",
    "top_p_threshold",
    "top_p_max_selects",
    "token_routing_mode",
    "token_top_p_threshold",
    "token_top_p_max_selects",
    "task_routing_mode",
    "task_top_p_threshold",
    "task_top_p_max_selects",
}


class GateError(RuntimeError):
    """Raised when a reproducibility or safety gate is not satisfied."""


@dataclass(frozen=True)
class Job:
    variant: str
    seed: int
    config_path: Path
    output_dir: Path
    routing_signature: dict[str, Any]

    @property
    def job_id(self) -> str:
        return f"{self.variant}-seed-{self.seed}"


def parse_variant_filter(value: str) -> list[str]:
    variants = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("--variants must contain at least one variant ID")
    if len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError("--variants must not contain duplicates")
    unknown = sorted(set(variants).difference(EXPECTED_VARIANT_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {unknown}")
    return variants


def parse_seed_filter(value: str) -> list[int]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    try:
        seeds = [int(item) for item in raw]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


def select_jobs(
    jobs: list[Job], variants: list[str] | None = None, seeds: list[int] | None = None
) -> list[Job]:
    """Select a reproducible subset after the complete A--D contract is validated."""
    available_variants = {job.variant for job in jobs}
    available_seeds = {job.seed for job in jobs}
    selected_variants = available_variants if variants is None else set(variants)
    selected_seeds = available_seeds if seeds is None else set(seeds)
    missing_variants = sorted(selected_variants.difference(available_variants))
    missing_seeds = sorted(selected_seeds.difference(available_seeds))
    if missing_variants:
        raise GateError(f"selected variants are absent from the manifest: {missing_variants}")
    if missing_seeds:
        raise GateError(f"selected seeds are absent from the manifest: {missing_seeds}")
    selected = [
        job for job in jobs if job.variant in selected_variants and job.seed in selected_seeds
    ]
    if not selected:
        raise GateError("job filter selected no training jobs")
    return selected


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"required JSON file is missing: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"expected a JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"required YAML file is missing: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise GateError(f"expected a YAML mapping: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_repo_path(repo: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo / path).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as exc:
        raise GateError(f"{label} must stay inside the repository: {resolved}") from exc
    return resolved


def routing_signature(config: dict[str, Any]) -> dict[str, Any]:
    try:
        moe = config["moe_config"]
    except KeyError as exc:
        raise GateError("training config has no moe_config") from exc

    def branch(prefix: str, experts_key: str, selects_key: str) -> dict[str, Any]:
        mode = str(moe.get(f"{prefix}_routing_mode", moe.get("routing_mode", "topk"))).lower()
        if mode not in {"topk", "topp"}:
            raise GateError(f"unsupported {prefix} routing mode: {mode}")
        num_experts = int(moe[experts_key])
        fixed_top_k = int(moe[selects_key])
        threshold = float(moe.get(f"{prefix}_top_p_threshold", moe.get("top_p_threshold", 0.4)))
        raw_max = moe.get(f"{prefix}_top_p_max_selects", moe.get("top_p_max_selects", num_experts))
        max_selects = num_experts if raw_max is None else int(raw_max)
        if not 1 <= fixed_top_k <= num_experts:
            raise GateError(f"{prefix} fixed_top_k={fixed_top_k} is invalid")
        if not 0.0 < threshold <= 1.0:
            raise GateError(f"{prefix} top-p threshold={threshold} is invalid")
        if not 1 <= max_selects <= num_experts:
            raise GateError(f"{prefix} top-p max_selects={max_selects} is invalid")
        return {
            "mode": mode,
            "num_experts": num_experts,
            "fixed_top_k": fixed_top_k,
            "top_p_threshold": threshold,
            "top_p_max_selects": max_selects,
        }

    return {
        "schema_version": 1,
        "token": branch("token", "num_experts", "num_selects"),
        "task": branch("task", "num_experts_contrastive", "num_selects_contrastive"),
        "task_hard_router": bool(moe.get("task_hard_router", False)),
    }


def config_without_routing(config: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(config))
    moe = clone.get("moe_config", {})
    for key in ROUTING_KEYS:
        moe.pop(key, None)
    return clone


def validate_training_configs(
    repo: Path, manifest: dict[str, Any]
) -> list[tuple[str, Path, dict[str, Any]]]:
    raw_variants = manifest.get("variants")
    if not isinstance(raw_variants, list):
        raise GateError("manifest variants must be a list")
    ids = [str(row.get("id")) for row in raw_variants if isinstance(row, dict)]
    if ids != list(EXPECTED_VARIANT_MODES):
        raise GateError(f"variants must be ordered exactly A,B,C,D; got {ids}")

    validated = []
    common_config = None
    seen_paths: set[Path] = set()
    for row in raw_variants:
        variant = str(row["id"])
        path = resolve_repo_path(repo, row["config"], label=f"variant {variant} config")
        if path in seen_paths:
            raise GateError(f"A--D variants must use four distinct config files: {path}")
        seen_paths.add(path)
        config = load_yaml(path)
        signature = routing_signature(config)
        expected_signature = row.get("routing_signature")
        if signature != expected_signature:
            raise GateError(
                f"variant {variant} routing signature mismatch:\n"
                f"config={json.dumps(signature, sort_keys=True)}\n"
                f"manifest={json.dumps(expected_signature, sort_keys=True)}"
            )
        expected_modes = EXPECTED_VARIANT_MODES[variant]
        actual_modes = (signature["token"]["mode"], signature["task"]["mode"])
        if actual_modes != expected_modes:
            raise GateError(f"variant {variant} modes={actual_modes}, expected={expected_modes}")
        if signature["task_hard_router"]:
            raise GateError(f"variant {variant}: task_hard_router must remain false")
        if signature["token"]["fixed_top_k"] != 2 or signature["task"]["fixed_top_k"] != 2:
            raise GateError(f"variant {variant}: both fixed reference widths must remain Top-2")

        if int(config.get("num_tasks", -1)) != 48:
            raise GateError(f"variant {variant}: num_tasks must be 48")
        if int(config.get("training_tasks", -1)) != 42:
            raise GateError(f"variant {variant}: training_tasks must be 42")
        if sorted(map(int, config.get("eval_tasks", []))) != REQUIRED_HELD_OUT:
            raise GateError(f"variant {variant}: eval_tasks do not match the frozen split")
        if int(config.get("max_episode_steps", -1)) != 64 or int(config.get("prompt_horizon", -1)) != 64:
            raise GateError(f"variant {variant}: the formal prompt must be one 64-step window")
        if config.get("supervision_mode") != "all_tokens":
            raise GateError(f"variant {variant}: supervision_mode must remain all_tokens")

        fairness = manifest.get("fairness_contract", {})
        config_expectations = {
            "total_steps": fairness.get("total_steps"),
            "eval_every": fairness.get("eval_every"),
            "checkpoint_every": fairness.get("checkpoint_every"),
            "early_stopping_patience": fairness.get("early_stopping_patience"),
            "early_stopping_min_delta": fairness.get("early_stopping_min_delta"),
        }
        if manifest.get("checkpoint_selection") != "best_validation":
            raise GateError("checkpoint_selection must be best_validation")
        if fairness.get("selection_metric") != "validation_mse":
            raise GateError("fairness_contract selection_metric must be validation_mse")
        for key, expected in config_expectations.items():
            if expected is None or config.get(key) != expected:
                raise GateError(
                    f"variant {variant}: {key}={config.get(key)!r}, expected={expected!r}"
                )

        normalized = config_without_routing(config)
        if common_config is None:
            common_config = normalized
        elif normalized != common_config:
            raise GateError(
                f"variant {variant} changes fields outside the declared routing controls"
            )
        validated.append((variant, path, signature))
    return validated


def _exact_ints(value: Any, expected: list[int], label: str) -> None:
    try:
        actual = sorted(map(int, value))
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label} is not an integer list") from exc
    if actual != expected:
        raise GateError(f"{label}={actual}, expected={expected}")


def validate_dataset_gate(
    repo: Path, dataset_dir: Path, queue_state_path: Path
) -> dict[str, Any]:
    if dataset_dir.name != "dpt":
        raise GateError(f"dataset_dir must point directly to the final dpt directory: {dataset_dir}")
    formal_dir = dataset_dir.parent
    if formal_dir.name != "formal_v1":
        raise GateError(f"only the frozen formal_v1 dataset may train A--D: {formal_dir}")

    collection_path = formal_dir / "collection_manifest.json"
    validation_path = dataset_dir / "validation_report.json"
    snapshot_path = formal_dir / "registry_snapshot.json"
    prompt_provenance_path = dataset_dir / "prompt_provenance.json"
    teacher_registry_path = dataset_dir / "teacher_registry.json"
    query_provenance_path = dataset_dir / "query_provenance.json"

    queue = load_json(queue_state_path)
    collection = load_json(collection_path)
    validation = load_json(validation_path)
    prompt_provenance = load_json(prompt_provenance_path)
    teacher_registry = load_json(teacher_registry_path)
    query_provenance = load_json(query_provenance_path)

    if queue.get("status") != "complete":
        raise GateError(f"staging queue is not ALL_COMPLETE: status={queue.get('status')!r}")
    contract = queue.get("contract", {})
    _exact_ints(contract.get("target_task_ids"), REQUIRED_TRAIN_TASKS, "queue target_task_ids")
    _exact_ints(queue.get("completed_task_ids"), REQUIRED_TRAIN_TASKS, "queue completed_task_ids")
    if queue.get("pending_task_ids") not in ([], None):
        raise GateError(f"queue still contains pending tasks: {queue.get('pending_task_ids')}")
    if queue.get("active_task_id") is not None:
        raise GateError(f"queue still has active task {queue.get('active_task_id')}")
    queue_tasks = queue.get("tasks", {})
    for task_id in REQUIRED_TRAIN_TASKS:
        row = queue_tasks.get(str(task_id), {})
        if row.get("status") != "complete" or int(row.get("return_code", -1)) != 0:
            raise GateError(f"queue task {task_id} is not a successful complete task")

    if collection.get("status") != "complete":
        raise GateError("final collection manifest status is not complete")
    _exact_ints(collection.get("source_task_ids"), REQUIRED_TRAIN_TASKS, "collection source_task_ids")
    _exact_ints(collection.get("held_out_task_ids"), REQUIRED_HELD_OUT, "collection held_out_task_ids")
    expected_collection = {
        "scope": "full",
        "full_training_set": True,
        "collection_mode": "merged_per_task_staging",
        "context_unit": "continuous_fixed_length_window",
        "next_state_semantics": "next_policy_decision_observation",
        "checkpoints_per_task": 25,
        "windows_per_checkpoint": 24,
        "window_steps": 64,
    }
    for key, expected in expected_collection.items():
        if collection.get(key) != expected:
            raise GateError(f"collection {key}={collection.get(key)!r}, expected={expected!r}")

    required_validation = {
        "validation": "PASS",
        "official_loader": "PASS",
        "query_reproduction": "BITWISE_PASS",
        "held_out_isolation": "PASS",
        "tasks": 42,
        "checkpoints_per_task": 25,
        "windows_per_checkpoint": 24,
        "window_steps": 64,
        "rows_per_task": 38_400,
        "total_prompt_transitions": 1_612_800,
        "state_dim": 123,
        "action_dim": 37,
    }
    for key, expected in required_validation.items():
        if validation.get(key) != expected:
            raise GateError(f"validation {key}={validation.get(key)!r}, expected={expected!r}")
    _exact_ints(validation.get("source_task_ids"), REQUIRED_TRAIN_TASKS, "validation source_task_ids")
    _exact_ints(validation.get("held_out_task_ids"), REQUIRED_HELD_OUT, "validation held_out_task_ids")
    if validation.get("dpt_dataset_smoke", {}).get("status") != "PASS":
        raise GateError("DPT_Dataset smoke test did not pass")
    if teacher_registry.get("status") != "complete":
        raise GateError("teacher registry is not complete")
    if query_provenance.get("method") != "same-state-stochastic-best-specialist-relabeling":
        raise GateError("query provenance is not the formal stochastic teacher relabeling method")

    if sha256_file(snapshot_path) != collection.get("registry_snapshot_sha256"):
        raise GateError("final registry snapshot hash does not match collection manifest")
    collection_sha = sha256_file(collection_path)
    teacher_sha = sha256_file(teacher_registry_path)
    if prompt_provenance.get("collection_manifest_sha256") != collection_sha:
        raise GateError("prompt provenance does not match final collection manifest")
    if query_provenance.get("teacher_registry_sha256") != teacher_sha:
        raise GateError("query provenance does not match final teacher registry")

    for provenance, label in (
        (prompt_provenance, "prompt provenance"),
        (teacher_registry, "teacher registry"),
        (query_provenance, "query provenance"),
    ):
        rows = provenance.get("tasks", [])
        ids = [int(row["source_task_id"]) for row in rows]
        _exact_ints(ids, REQUIRED_TRAIN_TASKS, f"{label} task IDs")
    for row in query_provenance.get("tasks", []):
        if row.get("policy_action_mode") != "stochastic":
            raise GateError(
                f"query task {row.get('source_task_id')} is not stochastic teacher relabeling"
            )

    prompt_rows = {
        int(row["source_task_id"]): row for row in prompt_provenance.get("tasks", [])
    }
    query_rows = {
        int(row["source_task_id"]): row for row in query_provenance.get("tasks", [])
    }
    dataset_files = {}
    for task_id in REQUIRED_TRAIN_TASKS:
        expected_files = (
            (
                dataset_dir / f"dataset_task_{task_id}.pkl",
                prompt_rows[task_id].get("prompt_sha256"),
                "prompt",
            ),
            (
                dataset_dir / f"query_dataset_task_{task_id}.pkl",
                query_rows[task_id].get("query_sha256"),
                "query",
            ),
        )
        for path, expected_sha, label in expected_files:
            if not path.is_file() or path.stat().st_size == 0:
                raise GateError(f"missing or empty formal dataset file: {path}")
            actual_sha = sha256_file(path)
            if not expected_sha or actual_sha != expected_sha:
                raise GateError(
                    f"task {task_id} {label} pickle SHA-256 mismatch: "
                    f"{actual_sha} != {expected_sha}"
                )
            dataset_files[path.name] = actual_sha
    for task_id in REQUIRED_HELD_OUT:
        for prefix in ("dataset_task_", "query_dataset_task_"):
            if (dataset_dir / f"{prefix}{task_id}.pkl").exists():
                raise GateError(f"held-out task leaked into training dataset: {prefix}{task_id}.pkl")

    components = {
        "queue_state_sha256": sha256_file(queue_state_path),
        "collection_manifest_sha256": collection_sha,
        "registry_snapshot_sha256": sha256_file(snapshot_path),
        "prompt_provenance_sha256": sha256_file(prompt_provenance_path),
        "teacher_registry_sha256": teacher_sha,
        "query_provenance_sha256": sha256_file(query_provenance_path),
        "validation_report_sha256": sha256_file(validation_path),
        "dataset_files_fingerprint_sha256": canonical_hash(dataset_files),
    }
    return {
        "dataset_dir": str(dataset_dir),
        "formal_dir": str(formal_dir),
        "source_task_ids": REQUIRED_TRAIN_TASKS,
        "held_out_task_ids": REQUIRED_HELD_OUT,
        "components": components,
        "fingerprint_sha256": canonical_hash(components),
    }


def validate_manifest(repo: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = load_json(manifest_path)
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise GateError(f"unsupported manifest format_version={manifest.get('format_version')}")
    name = str(manifest.get("experiment_name", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise GateError(f"unsafe experiment_name: {name!r}")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(x, bool) for x in seeds):
        raise GateError("seeds must be a non-empty integer list")
    try:
        int_seeds = [int(x) for x in seeds]
    except (TypeError, ValueError) as exc:
        raise GateError("seeds must be integers") from exc
    if len(set(int_seeds)) != len(int_seeds):
        raise GateError("seeds must be unique")
    manifest["seeds"] = int_seeds

    paths = {
        key: resolve_repo_path(repo, manifest[key], label=key)
        for key in ("dataset_dir", "queue_state", "trainer", "validator", "runs_root", "gpu_lock")
    }
    if paths["trainer"].resolve() == Path(__file__).resolve():
        raise GateError("trainer path points back to the launcher")
    for key in ("trainer", "validator"):
        if not paths[key].is_file():
            raise GateError(f"{key} is not a file: {paths[key]}")
    validate_training_configs(repo, manifest)
    for raw in manifest.get("code_files", []):
        path = resolve_repo_path(repo, raw, label="code_files entry")
        if not path.is_file():
            raise GateError(f"code provenance file is missing: {path}")
    return manifest, paths


def build_jobs(
    repo: Path,
    manifest: dict[str, Any],
    runs_root: Path,
) -> list[Job]:
    configs = validate_training_configs(repo, manifest)
    experiment_root = runs_root / manifest["experiment_name"]
    jobs = []
    # Seed-major order ensures the first common seed completes A--D before any
    # supplementary seed, matching the approved experiment plan.
    for seed in manifest["seeds"]:
        for variant, config_path, signature in configs:
            jobs.append(
                Job(
                    variant=variant,
                    seed=int(seed),
                    config_path=config_path,
                    output_dir=experiment_root / variant / f"seed_{int(seed)}",
                    routing_signature=signature,
                )
            )
    outputs = [job.output_dir for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise GateError("job matrix produced duplicate output directories")
    return jobs


def code_provenance(repo: Path, manifest: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    files = [paths["trainer"], paths["validator"]]
    files.extend(resolve_repo_path(repo, value, label="code_files entry") for value in manifest.get("code_files", []))
    unique = sorted(set(files))
    hashes = {str(path.relative_to(repo)): sha256_file(path) for path in unique}

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "files": hashes,
        "fingerprint_sha256": canonical_hash(hashes),
        "git_head": git("rev-parse", "HEAD"),
        "git_tracked_diff_sha256": canonical_hash(git("diff", "--binary", "HEAD") or ""),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": _package_version("torch"),
        "numpy_version": _package_version("numpy"),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def base_command(
    manifest: dict[str, Any], paths: dict[str, Path], job: Job
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(paths["trainer"]),
        "--config",
        str(job.config_path),
        "--dataset-dir",
        str(paths["dataset_dir"]),
        "--output-dir",
        str(job.output_dir),
        "--seed",
        str(job.seed),
    ]
    overrides = manifest.get("training_overrides", {})
    option_names = {
        "steps": "--steps",
        "eval_every": "--eval-every",
        "validation_samples_per_task": "--validation-samples-per-task",
        "device": "--device",
    }
    unknown = sorted(set(overrides).difference(option_names))
    if unknown:
        raise GateError(f"unsupported training_overrides: {unknown}")
    for key, option in option_names.items():
        if key in overrides and overrides[key] is not None:
            command.extend([option, str(overrides[key])])
    return command


def contract_identity(
    manifest_path: Path,
    manifest: dict[str, Any],
    dataset: dict[str, Any],
    code: dict[str, Any],
    job: Job,
    command: list[str],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "experiment_name": manifest["experiment_name"],
        "variant": job.variant,
        "seed": job.seed,
        "routing_signature": job.routing_signature,
        "config_path": str(job.config_path),
        "config_sha256": sha256_file(job.config_path),
        "dataset": dataset,
        "code": code,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "base_command": command,
    }
    value["identity_sha256"] = canonical_hash(value)
    return value


def latest_resume_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (output_dir / "checkpoints").glob("policy_*.pt"):
        match = re.fullmatch(r"policy_(\d+)\.pt", path.name)
        if match and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def validate_completed_run(job: Job) -> dict[str, str]:
    required = [
        job.output_dir / "best.pt",
        job.output_dir / "metrics.csv",
        job.output_dir / "latest_validation.json",
        job.output_dir / "resolved_config.yaml",
        job.output_dir / "routing_signature.json",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise GateError(f"completed job {job.job_id} is missing artifact: {path}")
    saved_signature = load_json(job.output_dir / "routing_signature.json")
    if saved_signature != job.routing_signature:
        raise GateError(f"completed job {job.job_id} has the wrong routing signature")
    latest = load_json(job.output_dir / "latest_validation.json")
    if latest.get("routing_signature") != job.routing_signature:
        raise GateError(f"completed job {job.job_id} validation used the wrong routing signature")
    with (job.output_dir / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise GateError(f"completed job {job.job_id} has no validation metric rows")
    return {path.name: sha256_file(path) for path in required}


def inspect_existing_job(
    job: Job, identity: dict[str, Any], *, resume_incomplete: bool
) -> tuple[str, Path | None]:
    if not job.output_dir.exists():
        return "new", None
    contract_path = job.output_dir / "run_contract.json"
    if not contract_path.exists():
        raise GateError(
            f"refusing to reuse non-empty/unmanaged output directory: {job.output_dir}"
        )
    contract = load_json(contract_path)
    if contract.get("identity", {}).get("identity_sha256") != identity["identity_sha256"]:
        raise GateError(f"existing output contract differs from requested job: {job.output_dir}")
    status = contract.get("status")
    if status == "complete":
        current_artifacts = validate_completed_run(job)
        recorded_artifacts = contract.get("artifacts")
        if not isinstance(recorded_artifacts, dict):
            raise GateError(
                f"completed job {job.job_id} has no recorded artifact hashes"
            )
        if recorded_artifacts != current_artifacts:
            raise GateError(
                f"completed job {job.job_id} artifacts changed after completion"
            )
        return "skip", None
    if not resume_incomplete:
        raise GateError(f"job {job.job_id} is {status!r}; resume_incomplete is disabled")
    checkpoint = latest_resume_checkpoint(job.output_dir)
    if checkpoint is None:
        raise GateError(
            f"job {job.job_id} is incomplete but has no numbered checkpoint; "
            "the launcher will not overwrite it"
        )
    try:
        checkpoint.resolve().relative_to(job.output_dir.resolve())
    except ValueError as exc:
        raise GateError(f"refusing cross-job resume checkpoint: {checkpoint}") from exc
    return "resume", checkpoint


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateError(f"another A--D launcher owns the GPU lock: {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "acquired_at": utc_now()}) + "\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_with_log(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
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


def run_validator(repo: Path, validator: Path, formal_dir: Path) -> None:
    command = [sys.executable, "-u", str(validator), "--formal-dir", str(formal_dir)]
    print(f"[A-D-GATE] revalidating formal dataset: {json.dumps(command)}", flush=True)
    subprocess.run(command, cwd=repo, check=True)


def print_plan(jobs: list[Job], commands: dict[str, list[str]], actions: dict[str, str]) -> None:
    for index, job in enumerate(jobs, 1):
        print(
            f"[A-D-PLAN] {index:02d}/{len(jobs):02d} job={job.job_id} "
            f"action={actions[job.job_id]} output={job.output_dir}",
            flush=True,
        )
        print(f"[A-D-COMMAND] {json.dumps(commands[job.job_id])}", flush=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    default_manifest = Path(__file__).resolve().parents[1] / "configs/robotlab_g1_abcd_training_manifest.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Read-only gate/config/contract check and command preview; launch nothing",
    )
    action.add_argument(
        "--check-manifest-only",
        action="store_true",
        help="Validate the launcher manifest and A--D config fairness without requiring formal_v1",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="After every gate passes, acquire the GPU lock and run the serial queue",
    )
    parser.add_argument(
        "--skip-revalidation",
        action="store_true",
        help="Trust the existing PASS report (real runs revalidate by default)",
    )
    parser.add_argument(
        "--variants",
        type=parse_variant_filter,
        default=None,
        help="Optional comma-separated execution subset, e.g. A,D. Full A--D fairness is still validated.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_filter,
        default=None,
        help="Optional comma-separated subset of manifest seeds, e.g. 42 or 42,43.",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest, paths = validate_manifest(repo, manifest_path)

    if args.check_manifest_only:
        configs = validate_training_configs(repo, manifest)
        for variant, path, signature in configs:
            print(
                f"[A-D-MANIFEST] variant={variant} config={path} "
                f"routing={json.dumps(signature, sort_keys=True)}",
                flush=True,
            )
        print(
            "[A-D-MANIFEST] PASS; dataset and training gates were intentionally not evaluated",
            flush=True,
        )
        return

    if args.execute and not args.skip_revalidation:
        run_validator(repo, paths["validator"], paths["dataset_dir"].parent)
    dataset = validate_dataset_gate(repo, paths["dataset_dir"], paths["queue_state"])
    code = code_provenance(repo, manifest, paths)
    jobs = select_jobs(
        build_jobs(repo, manifest, paths["runs_root"]),
        variants=args.variants,
        seeds=args.seeds,
    )
    commands: dict[str, list[str]] = {}
    identities: dict[str, dict[str, Any]] = {}
    actions: dict[str, str] = {}
    resumes: dict[str, Path | None] = {}
    resume_incomplete = bool(manifest.get("resume_incomplete", True))
    for job in jobs:
        command = base_command(manifest, paths, job)
        identity = contract_identity(manifest_path, manifest, dataset, code, job, command)
        action, checkpoint = inspect_existing_job(
            job, identity, resume_incomplete=resume_incomplete
        )
        command = [
            *command,
            "--run-contract-sha256",
            identity["identity_sha256"],
        ]
        if checkpoint is not None:
            command = [*command, "--resume", str(checkpoint)]
        commands[job.job_id] = command
        identities[job.job_id] = identity
        actions[job.job_id] = action
        resumes[job.job_id] = checkpoint

    print(
        f"[A-D-GATE] PASS dataset={dataset['fingerprint_sha256']} "
        f"jobs={len(jobs)} variants={sorted({job.variant for job in jobs})} "
        f"seeds={sorted({job.seed for job in jobs})}",
        flush=True,
    )
    print_plan(jobs, commands, actions)
    if args.dry_run:
        print("[A-D-GATE] DRY_RUN_COMPLETE; no directories created and no training launched", flush=True)
        return

    if not args.execute:
        raise GateError("internal action dispatch error: --execute was not selected")

    queue_path = paths["runs_root"] / manifest["experiment_name"] / "training_queue_state.json"
    with exclusive_lock(paths["gpu_lock"]):
        queue_state = {
            "format_version": 1,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_fingerprint_sha256": dataset["fingerprint_sha256"],
            "selected_variants": sorted({job.variant for job in jobs}),
            "selected_seeds": sorted({job.seed for job in jobs}),
            "jobs": {job.job_id: {"status": actions[job.job_id]} for job in jobs},
        }
        atomic_write_json(queue_path, queue_state)
        try:
            for job in jobs:
                action = actions[job.job_id]
                if action == "skip":
                    queue_state["jobs"][job.job_id]["status"] = "complete_skipped"
                    queue_state["jobs"][job.job_id]["updated_at"] = utc_now()
                    atomic_write_json(queue_path, queue_state)
                    print(f"[A-D-QUEUE] SKIP_COMPLETE job={job.job_id}", flush=True)
                    continue

                job.output_dir.mkdir(parents=True, exist_ok=True)
                contract_path = job.output_dir / "run_contract.json"
                contract = {
                    "format_version": 1,
                    "status": "running",
                    "identity": identities[job.job_id],
                    "command": commands[job.job_id],
                    "resume_checkpoint": str(resumes[job.job_id]) if resumes[job.job_id] else None,
                    "started_at": utc_now(),
                    "updated_at": utc_now(),
                }
                atomic_write_json(contract_path, contract)
                queue_state["jobs"][job.job_id] = {
                    "status": "running",
                    "output_dir": str(job.output_dir),
                    "started_at": contract["started_at"],
                }
                queue_state["updated_at"] = utc_now()
                atomic_write_json(queue_path, queue_state)
                print(f"[A-D-QUEUE] START job={job.job_id} action={action}", flush=True)
                return_code = run_with_log(
                    commands[job.job_id], cwd=repo, log_path=job.output_dir / "training.log"
                )
                if return_code != 0:
                    contract.update(
                        status="failed", return_code=return_code, finished_at=utc_now(), updated_at=utc_now()
                    )
                    atomic_write_json(contract_path, contract)
                    queue_state["status"] = "failed"
                    queue_state["jobs"][job.job_id].update(
                        status="failed", return_code=return_code, finished_at=utc_now()
                    )
                    queue_state["updated_at"] = utc_now()
                    atomic_write_json(queue_path, queue_state)
                    raise GateError(
                        f"job {job.job_id} failed with exit code {return_code}; serial queue stopped"
                    )
                artifacts = validate_completed_run(job)
                contract.update(
                    status="complete",
                    return_code=0,
                    artifacts=artifacts,
                    finished_at=utc_now(),
                    updated_at=utc_now(),
                )
                atomic_write_json(contract_path, contract)
                queue_state["jobs"][job.job_id].update(
                    status="complete", return_code=0, finished_at=utc_now(), artifacts=artifacts
                )
                queue_state["updated_at"] = utc_now()
                atomic_write_json(queue_path, queue_state)
                print(f"[A-D-QUEUE] COMPLETE job={job.job_id}", flush=True)
        except KeyboardInterrupt:
            queue_state["status"] = "interrupted"
            queue_state["updated_at"] = utc_now()
            atomic_write_json(queue_path, queue_state)
            print("[A-D-QUEUE] INTERRUPTED; rerun the same command to resume", flush=True)
            raise

        queue_state["status"] = "complete"
        queue_state["updated_at"] = utc_now()
        atomic_write_json(queue_path, queue_state)
        print(f"[A-D-QUEUE] ALL_COMPLETE jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (GateError, subprocess.CalledProcessError) as exc:
        print(f"[A-D-GATE] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
