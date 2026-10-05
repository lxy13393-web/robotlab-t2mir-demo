"""CPU-only safety tests for the formal A--D serial launcher."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


DPT_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
if str(DPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DPT_ROOT))

from commands.run_robotlab_g1_abcd_training import (  # noqa: E402
    GateError,
    REQUIRED_HELD_OUT,
    REQUIRED_TRAIN_TASKS,
    build_jobs,
    select_jobs,
    validate_dataset_gate,
    validate_manifest,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class DatasetFixture:
    def __init__(self, root: Path):
        self.formal = root / "formal_v1"
        self.dpt = self.formal / "dpt"
        self.dpt.mkdir(parents=True)
        self.queue = root / "queue_state.json"
        self.prompt_rows = []
        self.query_rows = []
        self.teacher_rows = []
        for task_id in REQUIRED_TRAIN_TASKS:
            prompt = self.dpt / f"dataset_task_{task_id}.pkl"
            query = self.dpt / f"query_dataset_task_{task_id}.pkl"
            prompt.write_bytes(f"prompt-{task_id}".encode())
            query.write_bytes(f"query-{task_id}".encode())
            self.prompt_rows.append(
                {"source_task_id": task_id, "prompt_sha256": digest(prompt)}
            )
            self.query_rows.append(
                {
                    "source_task_id": task_id,
                    "query_sha256": digest(query),
                    "policy_action_mode": "stochastic",
                }
            )
            self.teacher_rows.append({"source_task_id": task_id})

        snapshot = self.formal / "registry_snapshot.json"
        snapshot.write_text("{}")
        collection = {
            "status": "complete",
            "source_task_ids": REQUIRED_TRAIN_TASKS,
            "held_out_task_ids": REQUIRED_HELD_OUT,
            "scope": "full",
            "full_training_set": True,
            "collection_mode": "merged_per_task_staging",
            "context_unit": "continuous_fixed_length_window",
            "next_state_semantics": "next_policy_decision_observation",
            "checkpoints_per_task": 25,
            "windows_per_checkpoint": 24,
            "window_steps": 64,
            "registry_snapshot_sha256": digest(snapshot),
        }
        collection_path = self.formal / "collection_manifest.json"
        write_json(collection_path, collection)
        write_json(
            self.dpt / "prompt_provenance.json",
            {
                "collection_manifest_sha256": digest(collection_path),
                "tasks": self.prompt_rows,
            },
        )
        teacher_path = self.dpt / "teacher_registry.json"
        write_json(teacher_path, {"status": "complete", "tasks": self.teacher_rows})
        write_json(
            self.dpt / "query_provenance.json",
            {
                "method": "same-state-stochastic-best-specialist-relabeling",
                "teacher_registry_sha256": digest(teacher_path),
                "tasks": self.query_rows,
            },
        )
        write_json(
            self.dpt / "validation_report.json",
            {
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
                "source_task_ids": REQUIRED_TRAIN_TASKS,
                "held_out_task_ids": REQUIRED_HELD_OUT,
                "dpt_dataset_smoke": {"status": "PASS"},
            },
        )
        write_json(
            self.queue,
            {
                "status": "complete",
                "contract": {"target_task_ids": REQUIRED_TRAIN_TASKS},
                "completed_task_ids": REQUIRED_TRAIN_TASKS,
                "pending_task_ids": [],
                "active_task_id": None,
                "tasks": {
                    str(task_id): {"status": "complete", "return_code": 0}
                    for task_id in REQUIRED_TRAIN_TASKS
                },
            },
        )


class AbcdGateTest(unittest.TestCase):
    def test_real_manifest_defines_twelve_fair_serial_jobs(self):
        path = DPT_ROOT / "configs/robotlab_g1_abcd_training_manifest.json"
        manifest, paths = validate_manifest(REPO, path)
        jobs = build_jobs(REPO, manifest, paths["runs_root"])
        self.assertEqual(len(jobs), 12)
        self.assertEqual([job.job_id for job in jobs[:4]], [
            "A-seed-42", "B-seed-42", "C-seed-42", "D-seed-42"
        ])
        self.assertEqual(len({job.output_dir for job in jobs}), 12)

    def test_job_subset_keeps_manifest_order_and_rejects_unknown_seed(self):
        path = DPT_ROOT / "configs/robotlab_g1_abcd_training_manifest.json"
        manifest, paths = validate_manifest(REPO, path)
        jobs = build_jobs(REPO, manifest, paths["runs_root"])
        selected = select_jobs(jobs, variants=["A", "D"], seeds=[42])
        self.assertEqual([job.job_id for job in selected], ["A-seed-42", "D-seed-42"])
        with self.assertRaisesRegex(GateError, "absent from the manifest"):
            select_jobs(jobs, variants=["A", "D"], seeds=[999])

    def test_complete_dataset_contract_passes_and_is_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            report = validate_dataset_gate(REPO, fixture.dpt, fixture.queue)
            self.assertEqual(report["source_task_ids"], REQUIRED_TRAIN_TASKS)
            self.assertEqual(len(report["fingerprint_sha256"]), 64)
            self.assertIn("dataset_files_fingerprint_sha256", report["components"])

    def test_mutated_pickle_is_rejected_even_with_stale_pass_report(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            (fixture.dpt / "dataset_task_0.pkl").write_bytes(b"tampered")
            with self.assertRaisesRegex(GateError, "pickle SHA-256 mismatch"):
                validate_dataset_gate(REPO, fixture.dpt, fixture.queue)

    def test_incomplete_queue_and_subset_loader_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            queue = json.loads(fixture.queue.read_text())
            queue["status"] = "running"
            write_json(fixture.queue, queue)
            with self.assertRaisesRegex(GateError, "not ALL_COMPLETE"):
                validate_dataset_gate(REPO, fixture.dpt, fixture.queue)

            queue["status"] = "complete"
            write_json(fixture.queue, queue)
            report_path = fixture.dpt / "validation_report.json"
            report = json.loads(report_path.read_text())
            report["official_loader"] = "SUBSET_SKIPPED"
            write_json(report_path, report)
            with self.assertRaisesRegex(GateError, "official_loader"):
                validate_dataset_gate(REPO, fixture.dpt, fixture.queue)


if __name__ == "__main__":
    unittest.main()
