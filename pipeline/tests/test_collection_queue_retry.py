"""Tests for the formal collection queue's narrow network retry classifier."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.dataset.run_official_mixed_collection_queue import (  # noqa: E402
    network_retry_delay,
    run_and_tee,
    transient_network_failure_reason,
)


class TransientNetworkClassifierTest(unittest.TestCase):
    def test_remote_usd_missing_is_retryable(self) -> None:
        output = (
            "FileNotFoundError: USD file not found at path: "
            "'https://example.invalid/Assets/G1/g1_minimal.usd'"
        )
        self.assertEqual(
            transient_network_failure_reason(output), "remote_usd_unavailable"
        )

    def test_ssl_dns_and_timeout_are_retryable(self) -> None:
        self.assertEqual(
            transient_network_failure_reason("libcurl error (35): SSL connect error"),
            "ssl_connect_error",
        )
        self.assertEqual(
            transient_network_failure_reason("curl: Could not resolve host: assets.test"),
            "dns_resolution_error",
        )
        self.assertEqual(
            transient_network_failure_reason("Connection timed out while reading asset"),
            "connection_timeout",
        )

    def test_local_missing_file_and_integrity_failures_are_not_retryable(self) -> None:
        self.assertIsNone(
            transient_network_failure_reason(
                "FileNotFoundError: /tmp/iteration_001140/accepted.pt"
            )
        )
        self.assertIsNone(
            transient_network_failure_reason("ValueError: checkpoint SHA-256 mismatch")
        )

    def test_exponential_backoff(self) -> None:
        self.assertEqual([network_retry_delay(60.0, i) for i in (1, 2, 3)], [60, 120, 240])
        with self.assertRaises(ValueError):
            network_retry_delay(0.0, 1)
        with self.assertRaises(ValueError):
            network_retry_delay(60.0, 0)

    def test_run_and_tee_returns_only_current_child_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "child.log"
            return_code, output_tail = run_and_tee(
                [sys.executable, "-c", "print('SSL connect error'); raise SystemExit(7)"],
                repo=BASE_DIR,
                log_path=log_path,
            )
            self.assertEqual(return_code, 7)
            self.assertIn("SSL connect error", output_tail)
            self.assertIn("SSL connect error", log_path.read_text())


if __name__ == "__main__":
    unittest.main()
