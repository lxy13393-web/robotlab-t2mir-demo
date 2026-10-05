from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from deployment.robotlab_g1.contract import DEFAULT_CONTRACT
from deployment.robotlab_g1.model_registry import (
    canonical_sha256,
    load_artifact,
    promote_artifact,
    sha256_file,
    validate_active_pointer,
    validate_artifact,
    write_json_atomic,
)


def _artifact(checkpoint: Path, name: str) -> dict:
    core = {
        "format_version": 1,
        "name": name,
        "backend": "ppo",
        "variant": "PPO",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "deployment_contract_sha256": DEFAULT_CONTRACT.sha256,
        "observation_dim": DEFAULT_CONTRACT.state_dim,
        "action_dim": DEFAULT_CONTRACT.action_dim,
        "routing_signature": None,
        "checkpoint_contract": {
            "fixture": True,
            "observation_dim": DEFAULT_CONTRACT.state_dim,
            "action_dim": DEFAULT_CONTRACT.action_dim,
        },
        "metadata": {},
        "created_at": "2026-09-28T00:00:00+00:00",
    }
    return core | {"artifact_sha256": canonical_sha256(core)}


class ModelRegistryTest(unittest.TestCase):
    def test_artifact_detects_checkpoint_and_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "policy.onnx"
            checkpoint.write_bytes(b"model-v1")
            artifact = _artifact(checkpoint, "baseline")
            self.assertEqual(validate_artifact(artifact)["name"], "baseline")

            tampered = dict(artifact)
            tampered["name"] = "other"
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                validate_artifact(tampered)

            checkpoint.write_bytes(b"model-v2")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
                validate_artifact(artifact)

    def test_atomic_promotion_history_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_checkpoint = root / "first.pt"
            second_checkpoint = root / "second.pt"
            first_checkpoint.write_bytes(b"first")
            second_checkpoint.write_bytes(b"second")
            first_manifest = write_json_atomic(root / "first.json", _artifact(first_checkpoint, "first"))
            second_manifest = write_json_atomic(root / "second.json", _artifact(second_checkpoint, "second"))
            active = root / "active.json"

            promote_artifact(first_manifest, active)
            first_pointer = validate_active_pointer(json.loads(active.read_text()))
            first_active_sha = first_pointer["active_pointer_sha256"]
            self.assertEqual(load_artifact(active)["name"], "first")

            promote_artifact(
                second_manifest,
                active,
                expected_active_sha256=first_active_sha,
            )
            self.assertEqual(load_artifact(active)["name"], "second")
            history = sorted((root / "history").glob("active_*.json"))
            self.assertEqual(len(history), 1)
            self.assertEqual(load_artifact(history[0])["name"], "first")

            current = validate_active_pointer(json.loads(active.read_text()))
            promote_artifact(
                history[0],
                active,
                expected_active_sha256=current["active_pointer_sha256"],
            )
            self.assertEqual(load_artifact(active)["name"], "first")

    def test_compare_and_swap_rejects_stale_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"model")
            manifest = write_json_atomic(root / "model.json", _artifact(checkpoint, "model"))
            active = root / "active.json"
            promote_artifact(manifest, active)
            with self.assertRaisesRegex(ValueError, "changed since approval"):
                promote_artifact(
                    manifest,
                    active,
                    expected_active_sha256="0" * 64,
                )

    def test_pointer_checksum_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"model")
            manifest = write_json_atomic(root / "model.json", _artifact(checkpoint, "model"))
            active = root / "active.json"
            promote_artifact(manifest, active)
            pointer = json.loads(active.read_text())
            pointer["activation"]["activated_at"] = "tampered"
            with self.assertRaisesRegex(ValueError, "active pointer SHA-256 mismatch"):
                validate_active_pointer(pointer)


if __name__ == "__main__":
    unittest.main()
