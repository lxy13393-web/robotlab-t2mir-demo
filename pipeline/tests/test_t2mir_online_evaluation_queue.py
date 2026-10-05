"""CPU-only gates for the formal A/D online-evaluation queue."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BASE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipeline.evaluation.aggregate_t2mir_online_results import parse_queue_state  # noqa: E402
from pipeline.evaluation.run_t2mir_online_evaluation_queue import (  # noqa: E402
    EvaluationJob,
    QueueGateError,
    TrainingCheckpoint,
    build_jobs,
    run_queue,
    sha256_file,
    validate_dynamics_manifest,
)


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


class QueueContractTest(unittest.TestCase):
    def test_canonical_dynamics_manifest_passes(self) -> None:
        payload = validate_dynamics_manifest(REPO / "configs/g1_dynamics_48.json")
        self.assertEqual(payload["eval_tasks"], [5, 14, 23, 32, 41, 47])
        self.assertEqual(payload["task_count"], 48)

    def test_matrix_paths_encode_full_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluator.py"
            evaluator.write_text("# test\n", encoding="utf-8")
            python = root / "python"
            python.write_text("test\n", encoding="utf-8")
            manifest_path = REPO / "configs/g1_dynamics_48.json"
            manifest = validate_dynamics_manifest(manifest_path)
            checkpoints = []
            for variant, byte in (("A", b"a"), ("D", b"d")):
                checkpoint = root / variant / "best.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(byte)
                contract = root / variant / "run_contract.json"
                contract.write_text("{}", encoding="utf-8")
                checkpoints.append(
                    TrainingCheckpoint(
                        variant=variant,
                        train_seed=42,
                        checkpoint_path=checkpoint,
                        checkpoint_sha256=sha256_file(checkpoint),
                        run_contract_path=contract,
                        run_contract_sha256=sha256_file(contract),
                        routing_signature=routing(variant),
                    )
                )
            jobs = build_jobs(
                checkpoints=checkpoints,
                eval_seeds=[7, 9],
                profiles=[5, 47],
                episodes=8,
                num_envs=32,
                output_root=root / "results",
                evaluator_path=evaluator,
                dynamics_manifest_path=manifest_path,
                dynamics_manifest=manifest,
                python_executable=python,
            )
            self.assertEqual(len(jobs), 8)
            self.assertEqual(len({job.job_id for job in jobs}), 8)
            for job in jobs:
                rendered = str(job.output_dir)
                self.assertIn(f"variant_{job.variant}", rendered)
                self.assertIn("train_seed_42", rendered)
                self.assertIn(job.checkpoint_sha256, rendered)
                self.assertIn(f"eval_seed_{job.eval_seed}", rendered)
                self.assertIn(f"profile_{job.profile_id:02d}", rendered)

    def test_queue_state_is_directly_accepted_by_aggregator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluate.py"
            evaluator.write_text("# test\n", encoding="utf-8")
            python = root / "python"
            python.write_text("test\n", encoding="utf-8")
            manifest_path = REPO / "configs/g1_dynamics_48.json"
            manifest = validate_dynamics_manifest(manifest_path)
            checkpoints = []
            for variant in ("A", "D"):
                run = root / variant
                run.mkdir()
                checkpoint = run / "best.pt"
                checkpoint.write_bytes(variant.encode())
                contract = run / "run_contract.json"
                contract.write_text("{}", encoding="utf-8")
                checkpoints.append(
                    TrainingCheckpoint(
                        variant=variant,
                        train_seed=42,
                        checkpoint_path=checkpoint,
                        checkpoint_sha256=sha256_file(checkpoint),
                        run_contract_path=contract,
                        run_contract_sha256=sha256_file(contract),
                        routing_signature=routing(variant),
                    )
                )
            jobs = build_jobs(
                checkpoints=checkpoints,
                eval_seeds=[7],
                profiles=[5],
                episodes=3,
                num_envs=2,
                output_root=root / "results",
                evaluator_path=evaluator,
                dynamics_manifest_path=manifest_path,
                dynamics_manifest=manifest,
                python_executable=python,
            )
            state_path = root / "results/evaluation_queue_state.json"
            with mock.patch(
                "pipeline.evaluation.run_t2mir_online_evaluation_queue.classify_job",
                return_value=("skip", {"synthetic": "0" * 64}, None),
            ):
                run_queue(repo=REPO, queue_state_path=state_path, jobs=jobs)
            state, parsed = parse_queue_state(root / "results", state_path)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["selection"]["variants"], ["A", "D"])
            self.assertEqual(state["selection"]["profile_ids"], [5])
            self.assertEqual(state["contract"]["episodes"], 3)
            self.assertEqual(len(parsed), 2)

    def test_empty_queue_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QueueGateError, "no jobs"):
                run_queue(
                    repo=REPO,
                    queue_state_path=Path(directory) / "state.json",
                    jobs=[],
                )


if __name__ == "__main__":
    unittest.main()
