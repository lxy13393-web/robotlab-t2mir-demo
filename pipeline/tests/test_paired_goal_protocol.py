import json
import unittest

from pipeline.protocols.paired_goal_protocol import (
    build_manifest,
    generate_scenarios,
    load_manifest,
    validate_manifest,
)


class PairedGoalProtocolTest(unittest.TestCase):
    def test_paired_goal_manifest_round_trip(self):
        import tempfile
        from pathlib import Path

        rows = generate_scenarios(
            [47], episodes=4, replicas=8, seed=42, min_distance=1.0, max_distance=2.0
        )
        manifest = build_manifest(
            rows,
            episodes=4,
            replicas=8,
            seed=42,
            min_distance=1.0,
            max_distance=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goals.json"
            path.write_text(json.dumps(manifest))
            loaded, loaded_rows = load_manifest(path)
        self.assertEqual(loaded["goals_per_profile"], 32)
        self.assertEqual(loaded_rows, rows)
        self.assertEqual(len({(row.episode_index, row.replica_id) for row in rows}), 32)

    def test_paired_goal_manifest_rejects_modified_coordinate(self):
        rows = generate_scenarios(
            [47], episodes=2, replicas=2, seed=42, min_distance=1.0, max_distance=2.0
        )
        manifest = build_manifest(
            rows,
            episodes=2,
            replicas=2,
            seed=42,
            min_distance=1.0,
            max_distance=2.0,
        )
        manifest["scenarios"][0]["relative_x"] += 0.01
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
