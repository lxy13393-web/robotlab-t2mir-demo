"""Synthetic CPU-only tests for the formal online result aggregator."""

from __future__ import annotations

import csv
import json
import hashlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipeline.evaluation.aggregate_t2mir_online_results import (  # noqa: E402
    ValidationError,
    aggregate_queue,
    paired_cluster_bootstrap,
    sha256_file,
    write_outputs,
)


EPISODES = 3
REPLICAS = 2
PROFILES = (5, 14)


def routing(variant: str) -> dict:
    return {
        "schema_version": 1,
        "token": {
            "mode": "topk" if variant == "A" else "topp",
            "num_experts": 6,
            "fixed_top_k": 2,
            "top_p_threshold": 0.4,
            "top_p_max_selects": 6,
        },
        "task": {
            "mode": "topk" if variant == "A" else "topp",
            "num_experts": 8,
            "fixed_top_k": 2,
            "top_p_threshold": 0.4,
            "top_p_max_selects": 8,
        },
        "task_hard_router": False,
    }


class SyntheticQueue:
    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / "evaluation_queue_state.json"
        self.state = {
            "format_version": 1,
            "status": "complete",
            "selection": {
                "variants": ["A", "D"],
                "train_seeds": [42],
                "eval_seeds": [7],
                "profile_ids": list(PROFILES),
            },
            "contract": {"episodes": EPISODES, "num_envs": REPLICAS},
            "jobs": {},
        }
        for variant in ("A", "D"):
            for profile in PROFILES:
                self.add_job(variant, profile)
        self.flush_state()

    def job_id(self, variant: str, profile: int) -> str:
        return f"variant_{variant}_train_seed_42_eval_seed_7_profile_{profile:02d}"

    def add_job(self, variant: str, profile: int) -> None:
        output_dir = (
            self.root
            / f"variant_{variant}"
            / "train_seed_42"
            / f"checkpoint_{variant.lower() * 64}"
            / "eval_seed_7"
            / f"profile_{profile:02d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"variant{variant}_task{profile:02d}_seed7"
        report_path = output_dir / f"{stem}.json"
        csv_path = output_dir / f"{stem}.csv"
        checkpoint_path = self.root / "checkpoints" / variant / "best.pt"
        identity = {
            "variant": variant,
            "train_seed": 42,
            "profile_id": profile,
            "eval_seed": 7,
            "episodes": EPISODES,
            "num_envs": REPLICAS,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": variant.lower() * 64,
            "manifest_path": str(self.root / "g1_dynamics_48.json"),
            "manifest_sha256": "9" * 64,
            "routing_signature": routing(variant),
        }
        job = {
            "status": "complete",
            "identity": identity,
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "csv_path": str(csv_path),
        }
        self.state["jobs"][self.job_id(variant, profile)] = job
        self.write_result(
            job, scenario_sha=hashlib.sha256(f"profile-{profile}".encode()).hexdigest()
        )

    def rows(self, variant: str, profile: int) -> list[dict]:
        # D is deliberately a little better after context; all goal fields are
        # generated independently of the variant and therefore pair exactly.
        outcomes = {
            "A": ["timeout", "success", "fall", "timeout", "success", "timeout"],
            "D": ["timeout", "success", "success", "success", "success", "timeout"],
        }[variant]
        rows = []
        for episode in range(EPISODES):
            for replica in range(REPLICAS):
                index = episode * REPLICAS + replica
                rows.append(
                    {
                        "episode_index": episode,
                        "replica_id": replica,
                        "profile_id": profile,
                        "goal_seed": 7,
                        "relative_x": profile + episode + replica / 10,
                        "relative_y": -profile - episode - replica / 10,
                        "relative_yaw": episode / 10 + replica / 100,
                        "prompt_selected_length": 0 if episode == 0 else 64,
                        "result": outcomes[index],
                        "policy_kind": "t2mir",
                        "checkpoint_sha256": variant.lower() * 64,
                        "scenario_sha256": hashlib.sha256(
                            f"profile-{profile}".encode()
                        ).hexdigest(),
                        "initial_x": float(replica * 4),
                        "initial_y": 0.0,
                        "initial_yaw": 0.0,
                        "initial_pose_sha256": hashlib.sha256(
                            struct.pack("<fff", float(replica * 4), 0.0, 0.0)
                        ).hexdigest(),
                    }
                )
        return rows

    @staticmethod
    def summary(rows: list[dict]) -> dict:
        per_episode = []
        for episode in range(EPISODES):
            selected = [row for row in rows if row["episode_index"] == episode]
            successes = sum(row["result"] == "success" for row in selected)
            falls = sum(row["result"] == "fall" for row in selected)
            per_episode.append(
                {
                    "episode_index": episode,
                    "episodes": len(selected),
                    "successes": successes,
                    "falls": falls,
                    "timeouts": len(selected) - successes - falls,
                    "success_rate": successes / len(selected),
                    "fall_rate": falls / len(selected),
                }
            )
        successes = sum(row["result"] == "success" for row in rows)
        falls = sum(row["result"] == "fall" for row in rows)
        return {
            "episodes_per_replica": EPISODES,
            "replicas": REPLICAS,
            "total_episodes": len(rows),
            "success_rate": successes / len(rows),
            "fall_rate": falls / len(rows),
            "per_episode": per_episode,
        }

    def write_result(self, job: dict, scenario_sha: str) -> None:
        identity = job["identity"]
        rows = self.rows(identity["variant"], identity["profile_id"])
        csv_path = Path(job["csv_path"])
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "format_version": 1,
            "evaluation": "source_faithful_stationary_online_dpt",
            "variant": identity["variant"],
            "checkpoint": {
                "checkpoint": identity["checkpoint_path"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "checkpoint_step": 100,
                "state_dim": 123,
                "action_dim": 37,
                "prompt_horizon": 64,
                "routing_signature": identity["routing_signature"],
            },
            "manifest": identity["manifest_path"],
            "manifest_sha256": identity["manifest_sha256"],
            "profile": {"task_id": identity["profile_id"], "split": "eval"},
            "seed": identity["eval_seed"],
            "scenario_sha256": scenario_sha,
            "goal_contract": {
                "episodes": EPISODES,
                "replicas": REPLICAS,
                "min_distance": 1.0,
                "max_distance": 4.0,
                "timeout": 40.0,
                "hold_time": 2.0,
            },
            "reset_protocol": {
                "full_vector_reset_before_each_episode": True,
                "episode_boundary_mechanism": (
                    "wrapper_initial_reset_then_forced_timeout_auto_reset"
                ),
                "explicit_global_reset_calls_after_wrapper_construction": 0,
                "boundary_transition_excluded_from_metrics_and_prompt": True,
                "action_lag_state_crosses_reset": False,
                "initial_pose_recorded_per_row": True,
                "deterministic_reset": True,
            },
            "result_csv": str(csv_path),
            "result_csv_sha256": sha256_file(csv_path),
            "summary": self.summary(rows),
            "code_sha256": {
                "pipeline/evaluation/evaluate_t2mir_online.py": "1" * 64,
                "pipeline/protocols/t2mir_online_context.py": "2" * 64,
                "pipeline/protocols/goal_controller.py": "3" * 64,
                "pipeline/protocols/dynamics_profile.py": "4" * 64,
            },
        }
        Path(job["report_path"]).write_text(json.dumps(report), encoding="utf-8")

    def flush_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state), encoding="utf-8")


class AggregateOnlineResultsTest(unittest.TestCase):
    def test_valid_queue_aggregates_primary_metric_and_writes_json_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = SyntheticQueue(Path(directory))
            summary, clusters = aggregate_queue(
                queue.root, bootstrap_samples=200, bootstrap_seed=123
            )
            self.assertEqual(summary["validation"]["status"], "PASS")
            self.assertEqual(summary["validation"]["validated_jobs"], 4)
            self.assertEqual(summary["primary_metric"]["episode_indices"], [1, 2])
            self.assertGreater(
                summary["comparison_a_d"]["post_context_success_primary"][
                    "observed_d_minus_a"
                ],
                0.0,
            )
            self.assertEqual(len(clusters), len(PROFILES) * REPLICAS)
            paths = write_outputs(summary, clusters, queue.root / "aggregate")
            self.assertTrue(all(path.is_file() for path in paths.values()))
            with paths["metrics_csv"].open(newline="", encoding="utf-8") as stream:
                self.assertGreater(len(list(csv.DictReader(stream))), 0)

    def test_missing_matrix_job_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = SyntheticQueue(Path(directory))
            queue.state["jobs"].pop(queue.job_id("D", 14))
            queue.flush_state()
            with self.assertRaisesRegex(ValidationError, "matrix.*missing"):
                aggregate_queue(queue.root, bootstrap_samples=20)

    def test_missing_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = SyntheticQueue(Path(directory))
            job = queue.state["jobs"][queue.job_id("A", 5)]
            Path(job["report_path"]).unlink()
            with self.assertRaisesRegex(ValidationError, "missing result report"):
                aggregate_queue(queue.root, bootstrap_samples=20)

    def test_a_d_scenario_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = SyntheticQueue(Path(directory))
            job = queue.state["jobs"][queue.job_id("D", 5)]
            report_path = Path(job["report_path"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["scenario_sha256"] = "f" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "scenario SHA mismatch"):
                aggregate_queue(queue.root, bootstrap_samples=20)

    def test_a_d_initial_pose_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = SyntheticQueue(Path(directory))
            job = queue.state["jobs"][queue.job_id("D", 5)]
            csv_path = Path(job["csv_path"])
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                fieldnames = list(rows[0])
            rows[0]["initial_x"] = "1.0"
            rows[0]["initial_pose_sha256"] = hashlib.sha256(
                struct.pack("<fff", 1.0, 0.0, 0.0)
            ).hexdigest()
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            report_path = Path(job["report_path"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["result_csv_sha256"] = sha256_file(csv_path)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "initial-pose mismatch"):
                aggregate_queue(queue.root, bootstrap_samples=20)

    def test_cluster_bootstrap_is_exactly_reproducible(self) -> None:
        first = paired_cluster_bootstrap(
            [0.0, 0.5, 1.0, 0.25],
            [0.5, 0.5, 1.0, 1.0],
            samples=501,
            seed=20260928,
        )
        second = paired_cluster_bootstrap(
            [0.0, 0.5, 1.0, 0.25],
            [0.5, 0.5, 1.0, 1.0],
            samples=501,
            seed=20260928,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["observed_d_minus_a"], 0.3125)


if __name__ == "__main__":
    unittest.main()
