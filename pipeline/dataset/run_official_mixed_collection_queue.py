"""Serially collect every completed PPO bank into formal task staging.

The checkpoint-bank registry grows as specialist PPO jobs finish.  This runner
polls that registry, invokes ``collect_official_mixed_staging.py`` for complete
training-task banks, and waits when training is ahead of collection.  It never
starts, stops, or inspects a live PPO run and it never schedules held-out tasks.

The queue is deliberately operational only: the staging wrapper remains the
authority for checkpoint hashes, frozen provenance, restartability, and shard
validation.  A non-zero child exit still stops the queue unless its output
matches a narrow set of transient remote-asset/network failures.  Those
failures are retried with bounded exponential backoff using the exact same
command, seeds and staging directory; data-integrity failures are never
retried or hidden.
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
from collections import deque
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from pipeline.dataset.dataset_common import parse_csv_ints, read_json, utc_now, write_json_atomic
from pipeline.protocols.dynamics_profile import load_manifest


_TRANSIENT_NETWORK_MARKERS: tuple[tuple[str, str], ...] = (
    ("libcurl error (35)", "ssl_connect_error"),
    ("ssl connect error", "ssl_connect_error"),
    ("could not resolve host", "dns_resolution_error"),
    ("temporary failure in name resolution", "dns_resolution_error"),
    ("unable to connect to server", "remote_server_unreachable"),
    ("connection timed out", "connection_timeout"),
    ("operation timed out", "connection_timeout"),
    ("connection reset by peer", "connection_reset"),
)


def transient_network_failure_reason(output_tail: str) -> str | None:
    """Classify only failures that are safe to repeat with identical inputs.

    Isaac/OmniClient often collapses an underlying network failure into a
    generic ``USD file not found`` message in the parent process.  A missing
    *remote* HTTP(S) USD is therefore retryable, while a missing local file is
    intentionally not.  The bounded retry limit still exposes a genuinely
    removed or renamed remote asset instead of looping forever.
    """

    lowered = output_tail.lower()
    for marker, reason in _TRANSIENT_NETWORK_MARKERS:
        if marker in lowered:
            return reason
    if "usd file not found at path:" in lowered and (
        "http://" in lowered or "https://" in lowered
    ):
        return "remote_usd_unavailable"
    return None


def network_retry_delay(base_seconds: float, retry_number: int) -> float:
    """Return bounded exponential backoff for a one-based retry number."""

    if base_seconds <= 0:
        raise ValueError("network retry base delay must be positive")
    if retry_number <= 0:
        raise ValueError("retry_number must be positive")
    return float(base_seconds) * float(2 ** (retry_number - 1))


def registered_bank_is_complete(task: dict, expected_non_initial: int) -> bool:
    """Cheap readiness check; the staging wrapper performs the strict hash check."""
    if task.get("training", {}).get("method") != "fresh_task_specific_ppo":
        return False
    checkpoints = task.get("checkpoints", [])
    if len(checkpoints) != expected_non_initial + 1:
        return False
    try:
        iterations = sorted(int(row["iteration"]) for row in checkpoints)
    except (KeyError, TypeError, ValueError):
        return False
    if len(iterations) != len(set(iterations)) or iterations[0] != 0:
        return False
    if len([iteration for iteration in iterations if iteration > 0]) != expected_non_initial:
        return False
    return all(row.get("checkpoint") and row.get("checkpoint_sha256") for row in checkpoints)


def stage_claims_complete(staging_root: Path, task_id: int) -> bool:
    path = staging_root / f"task{task_id:02d}" / "collection_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("status") == "complete" and list(
        map(int, manifest.get("source_task_ids", []))
    ) == [task_id]


def queue_contract(args: argparse.Namespace, manifest_sha256: str, targets: list[int]) -> dict:
    return {
        "manifest_sha256": manifest_sha256,
        "target_task_ids": targets,
        "registry": str(args.registry.resolve()),
        "staging_root": str(args.staging_root.resolve()),
        "checkpoints_per_task": args.checkpoints_per_task,
        "windows_per_checkpoint": args.windows_per_checkpoint,
        "window_steps": args.window_steps,
        "attempt_batch": args.attempt_batch,
        "max_attempts": args.max_attempts,
        "base_seed": args.base_seed,
        "goal_min_distance": args.goal_min_distance,
        "goal_max_distance": args.goal_max_distance,
        "goal_timeout": args.goal_timeout,
        "goal_hold_time": args.goal_hold_time,
        "dry_run": args.dry_run,
    }


def new_state(contract: dict) -> dict:
    now = utc_now()
    return {
        "format_version": 1,
        "queue": "RobotLab-G1-MultiDynamics/official_mixed_v1",
        "created_at": now,
        "updated_at": now,
        "status": "starting",
        "contract": contract,
        "registry_ready_task_ids": [],
        "completed_task_ids": [],
        "pending_task_ids": list(contract["target_task_ids"]),
        "active_task_id": None,
        "tasks": {},
    }


def load_or_create_state(path: Path, contract: dict) -> dict:
    if not path.exists():
        return new_state(contract)
    state = read_json(path)
    if state.get("contract") != contract:
        raise ValueError(
            f"queue contract differs from existing state {path}; use the original arguments "
            "or an explicit different --state-file"
        )
    state.setdefault("tasks", {})
    return state


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    write_json_atomic(path, state)


def staging_command(repo: Path, args: argparse.Namespace, task_id: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(repo / "pipeline/dataset/collect_official_mixed_staging.py"),
        "--task-id",
        str(task_id),
        "--manifest",
        str(args.manifest.resolve()),
        "--registry",
        str(args.registry.resolve()),
        "--staging-root",
        str(args.staging_root.resolve()),
        "--checkpoints-per-task",
        str(args.checkpoints_per_task),
        "--windows-per-checkpoint",
        str(args.windows_per_checkpoint),
        "--window-steps",
        str(args.window_steps),
        "--attempt-batch",
        str(args.attempt_batch),
        "--max-attempts",
        str(args.max_attempts),
        "--base-seed",
        str(args.base_seed),
        "--goal-min-distance",
        str(args.goal_min_distance),
        "--goal-max-distance",
        str(args.goal_max_distance),
        "--goal-timeout",
        str(args.goal_timeout),
        "--goal-hold-time",
        str(args.goal_hold_time),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_and_tee(command: list[str], *, repo: Path, log_path: Path) -> tuple[int, str]:
    """Run one child and return its code plus a bounded diagnostic tail."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_tail: deque[str] = deque(maxlen=600)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] command: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                output_tail.append(line)
            return process.wait(), "".join(output_tail)
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    namespace = repo / "data/t2mir/RobotLab-G1-MultiDynamics/official_mixed_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument("--registry", type=Path, default=namespace / "checkpoint_bank/pilot_registry.json")
    parser.add_argument("--staging-root", type=Path, default=namespace / "formal_staging_v1")
    parser.add_argument(
        "--task-ids",
        type=parse_csv_ints,
        default=None,
        help="Default: all 42 training tasks from the dynamics manifest.",
    )
    parser.add_argument("--checkpoints-per-task", type=int, default=25)
    parser.add_argument("--windows-per-checkpoint", type=int, default=24)
    parser.add_argument("--window-steps", type=int, default=64)
    parser.add_argument("--attempt-batch", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260927)
    parser.add_argument("--goal-min-distance", type=float, default=1.0)
    parser.add_argument("--goal-max-distance", type=float, default=4.0)
    parser.add_argument("--goal-timeout", type=float, default=40.0)
    parser.add_argument("--goal-hold-time", type=float, default=2.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=900.0)
    parser.add_argument(
        "--network-retries",
        type=int,
        default=3,
        help=(
            "Retries after the initial attempt for recognised transient SSL/DNS/remote-USD "
            "failures. Operational only; it does not alter the dataset contract."
        ),
    )
    parser.add_argument(
        "--network-retry-backoff-seconds",
        type=float,
        default=60.0,
        help="Base delay for exponential transient-network retry backoff.",
    )
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Attempt each task that is ready now once, then exit instead of waiting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0:
        parser.error("poll and heartbeat intervals must be positive")
    if args.network_retries < 0:
        parser.error("--network-retries must be non-negative")
    if args.network_retry_backoff_seconds <= 0:
        parser.error("--network-retry-backoff-seconds must be positive")
    if args.dry_run and not args.once:
        parser.error("--dry-run requires --once so planned shards are not scheduled repeatedly")

    manifest, profiles = load_manifest(args.manifest)
    train_ids = sorted(profile.task_id for profile in profiles if profile.split == "train")
    targets = sorted(args.task_ids or train_ids)
    invalid = sorted(set(targets).difference(train_ids))
    if invalid:
        parser.error(f"unknown or held-out task IDs cannot enter the collection queue: {invalid}")

    staging_root = args.staging_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    state_path = (args.state_file or staging_root / "queue_state.json").resolve()
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    contract = queue_contract(args, manifest["sha256"], targets)

    with lock_path.open("w") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another collection queue owns {lock_path}") from error

        state = load_or_create_state(state_path, contract)
        validated_this_run: set[int] = set()
        attempted_this_run: set[int] = set()
        last_wait_signature = None
        last_heartbeat = 0.0
        try:
            while True:
                live = read_json(args.registry)
                if live.get("manifest_sha256") != manifest["sha256"]:
                    raise ValueError("checkpoint registry manifest hash does not match the dynamics manifest")
                rows = {int(row["task_id"]): row for row in live.get("tasks", [])}
                ready = sorted(
                    task_id
                    for task_id in targets
                    if task_id in rows
                    and registered_bank_is_complete(rows[task_id], args.checkpoints_per_task)
                )
                claimed_complete = sorted(
                    task_id for task_id in targets if stage_claims_complete(staging_root, task_id)
                )
                pending = sorted(set(targets).difference(claimed_complete))
                state["registry_ready_task_ids"] = ready
                state["completed_task_ids"] = claimed_complete
                state["pending_task_ids"] = pending

                # Every completed stage is sent through the wrapper once per queue
                # process, so a restart revalidates immutable hashes before trusting it.
                candidates = [
                    task_id
                    for task_id in ready
                    if task_id not in validated_this_run and task_id not in attempted_this_run
                ]
                if candidates:
                    task_id = candidates[0]
                    task_key = str(task_id)
                    task_state = state["tasks"].setdefault(task_key, {"attempts": 0})
                    task_state["attempts"] = int(task_state.get("attempts", 0)) + 1
                    task_state["status"] = "running"
                    task_state["started_at"] = utc_now()
                    task_state.pop("error", None)
                    state["status"] = "running"
                    state["active_task_id"] = task_id
                    save_state(state_path, state)

                    attempt = task_state["attempts"]
                    log_path = staging_root / "queue_logs" / f"task{task_id:02d}_attempt{attempt:03d}.log"
                    print(
                        f"[FORMAL-QUEUE] START task={task_id} attempt={attempt} log={log_path}",
                        flush=True,
                    )
                    command = staging_command(repo, args, task_id)
                    network_retries_this_attempt = 0
                    while True:
                        return_code, output_tail = run_and_tee(
                            command, repo=repo, log_path=log_path
                        )
                        if return_code == 0:
                            break
                        reason = transient_network_failure_reason(output_tail)
                        if (
                            reason is None
                            or network_retries_this_attempt >= args.network_retries
                        ):
                            break
                        network_retries_this_attempt += 1
                        delay = network_retry_delay(
                            args.network_retry_backoff_seconds,
                            network_retries_this_attempt,
                        )
                        task_state["network_retries"] = int(
                            task_state.get("network_retries", 0)
                        ) + 1
                        task_state["last_network_retry"] = {
                            "at": utc_now(),
                            "reason": reason,
                            "retry_number": network_retries_this_attempt,
                            "delay_seconds": delay,
                        }
                        save_state(state_path, state)
                        retry_message = (
                            f"[FORMAL-QUEUE] TRANSIENT_NETWORK_RETRY task={task_id} "
                            f"retry={network_retries_this_attempt}/{args.network_retries} "
                            f"reason={reason} delay={delay:.1f}s"
                        )
                        print(retry_message, flush=True)
                        with log_path.open("a") as retry_log:
                            retry_log.write(f"\n[{utc_now()}] {retry_message}\n")
                        time.sleep(delay)
                    attempted_this_run.add(task_id)
                    state["active_task_id"] = None
                    task_state["finished_at"] = utc_now()
                    task_state["return_code"] = return_code
                    task_state["network_retries_this_attempt"] = network_retries_this_attempt
                    if return_code != 0:
                        task_state["status"] = "failed"
                        final_reason = transient_network_failure_reason(output_tail)
                        task_state["error"] = (
                            f"staging collector exited with code {return_code}"
                            + (f" after transient failure {final_reason}" if final_reason else "")
                        )
                        state["status"] = "failed"
                        save_state(state_path, state)
                        raise RuntimeError(
                            f"task {task_id} collection failed with exit code {return_code}; "
                            f"inspect {log_path} and restart the same queue command after fixing it"
                        )
                    if args.dry_run:
                        task_state["status"] = "dry_run_pass"
                        print(f"[FORMAL-QUEUE] DRY_RUN_PASS task={task_id}", flush=True)
                    elif stage_claims_complete(staging_root, task_id):
                        task_state["status"] = "complete"
                        validated_this_run.add(task_id)
                        print(f"[FORMAL-QUEUE] COMPLETE task={task_id}", flush=True)
                    else:
                        task_state["status"] = "failed"
                        task_state["error"] = "collector returned zero without a complete stage manifest"
                        state["status"] = "failed"
                        save_state(state_path, state)
                        raise RuntimeError(task_state["error"])
                    save_state(state_path, state)
                    continue

                if args.once:
                    state["status"] = "dry_run_complete" if args.dry_run else "once_complete"
                    save_state(state_path, state)
                    print(
                        f"[FORMAL-QUEUE] ONCE_COMPLETE attempted={sorted(attempted_this_run)} "
                        f"ready={ready}",
                        flush=True,
                    )
                    return

                # Refresh after validation/collection; all targets are only done when
                # every stage claims completion and was validated during this process.
                if not pending and set(targets).issubset(validated_this_run):
                    state["status"] = "complete"
                    state["active_task_id"] = None
                    save_state(state_path, state)
                    print(f"[FORMAL-QUEUE] ALL_COMPLETE tasks={len(targets)}", flush=True)
                    return

                waiting_for_banks = sorted(set(targets).difference(ready))
                waiting_for_collection = sorted(set(ready).difference(claimed_complete))
                signature = (tuple(ready), tuple(claimed_complete), tuple(waiting_for_banks))
                now = time.monotonic()
                if signature != last_wait_signature or now - last_heartbeat >= args.heartbeat_seconds:
                    print(
                        f"[FORMAL-QUEUE] WAITING complete={len(claimed_complete)}/{len(targets)} "
                        f"registry_ready={len(ready)}/{len(targets)} "
                        f"collectable={waiting_for_collection} waiting_for_ppo={waiting_for_banks}",
                        flush=True,
                    )
                    last_wait_signature = signature
                    last_heartbeat = now
                state["status"] = "waiting_for_ppo"
                state["active_task_id"] = None
                save_state(state_path, state)
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            state["active_task_id"] = None
            save_state(state_path, state)
            print("[FORMAL-QUEUE] INTERRUPTED; restart the same command to resume", flush=True)
            raise


if __name__ == "__main__":
    main()
