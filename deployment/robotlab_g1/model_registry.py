"""Immutable policy artifacts and atomic active-model promotion.

The deployment/evaluation code never treats a mutable ``best.pt`` path as the
identity of a formal model.  A model artifact pins the checkpoint digest, the
123/37 deployment contract and (for T2MIR) the routing signature extracted
from the checkpoint itself.  Promotion atomically replaces a small JSON
pointer; checkpoint bytes are never copied or rewritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .contract import DEFAULT_CONTRACT
from .policy_backends import PPOOnnxBackend, T2MIRTorchBackend


FORMAT_VERSION = 1
VARIANT_ROUTING_MODES = {
    "A": ("topk", "topk"),
    "B": ("topp", "topk"),
    "C": ("topk", "topp"),
    "D": ("topp", "topp"),
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return output


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"unsafe model name: {value!r}")
    return value


def _load_t2mir_contract(checkpoint: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("registering T2MIR requires torch") from exc
    from pipeline.protocols.t2mir_online_context import validate_checkpoint_contract

    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("T2MIR checkpoint must contain a mapping")
    return validate_checkpoint_contract(
        checkpoint,
        payload,
        expected_supervision_modes=("all_tokens", "final_token"),
    )


def create_artifact(
    *,
    name: str,
    backend: str,
    checkpoint: str | Path,
    variant: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a checkpoint and return its immutable deployment record."""

    name = _safe_name(name)
    backend = backend.lower()
    if backend not in {"ppo", "t2mir"}:
        raise ValueError("backend must be 'ppo' or 't2mir'")
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {path}")

    checkpoint_contract: dict[str, Any] = {}
    routing_signature = None
    if backend == "t2mir":
        checkpoint_contract = _load_t2mir_contract(path)
        if (
            int(checkpoint_contract["state_dim"]) != DEFAULT_CONTRACT.state_dim
            or int(checkpoint_contract["action_dim"]) != DEFAULT_CONTRACT.action_dim
        ):
            raise ValueError(
                "T2MIR checkpoint dimensions do not match the deployment contract: "
                f"{checkpoint_contract['state_dim']}->{checkpoint_contract['action_dim']} "
                f"!= {DEFAULT_CONTRACT.state_dim}->{DEFAULT_CONTRACT.action_dim}"
            )
        routing_signature = checkpoint_contract["routing_signature"]
        if variant is not None:
            variant = variant.upper()
            if variant not in VARIANT_ROUTING_MODES:
                raise ValueError(f"unknown T2MIR variant: {variant!r}")
            actual = (
                routing_signature["token"]["mode"],
                routing_signature["task"]["mode"],
            )
            if actual != VARIANT_ROUTING_MODES[variant]:
                raise ValueError(
                    f"checkpoint routing modes {actual} do not match variant "
                    f"{variant}={VARIANT_ROUTING_MODES[variant]}"
                )
    elif variant not in {None, "PPO", "ppo"}:
        raise ValueError("PPO artifacts cannot claim a T2MIR A-D variant")
    else:
        # Registration is a one-time operation, so pay the small CPU cost to
        # prove that the ONNX graph really exposes the frozen 123->37 contract.
        backend_probe = PPOOnnxBackend(path)
        checkpoint_contract = dict(backend_probe.provenance)

    core: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "backend": backend,
        "variant": variant.upper() if variant else ("PPO" if backend == "ppo" else None),
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "deployment_contract_sha256": DEFAULT_CONTRACT.sha256,
        "observation_dim": DEFAULT_CONTRACT.state_dim,
        "action_dim": DEFAULT_CONTRACT.action_dim,
        "routing_signature": routing_signature,
        "checkpoint_contract": checkpoint_contract,
        "metadata": dict(metadata or {}),
        "created_at": _timestamp(),
    }
    core["artifact_sha256"] = canonical_sha256(core)
    return core


def validate_artifact(value: Mapping[str, Any], *, verify_checkpoint: bool = True) -> dict[str, Any]:
    artifact = dict(value)
    expected_artifact_sha = artifact.pop("artifact_sha256", None)
    if int(artifact.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported artifact format: {artifact.get('format_version')}")
    _safe_name(str(artifact.get("name", "")))
    if artifact.get("backend") not in {"ppo", "t2mir"}:
        raise ValueError("artifact backend must be ppo or t2mir")
    checkpoint_contract = artifact.get("checkpoint_contract")
    if not isinstance(checkpoint_contract, Mapping):
        raise ValueError("artifact checkpoint_contract must be a mapping")
    if artifact.get("deployment_contract_sha256") != DEFAULT_CONTRACT.sha256:
        raise ValueError("artifact deployment contract does not match this checkout")
    if int(artifact.get("observation_dim", -1)) != DEFAULT_CONTRACT.state_dim:
        raise ValueError("artifact observation dimension mismatch")
    if int(artifact.get("action_dim", -1)) != DEFAULT_CONTRACT.action_dim:
        raise ValueError("artifact action dimension mismatch")
    if artifact["backend"] == "ppo":
        if (
            int(checkpoint_contract.get("observation_dim", -1))
            != DEFAULT_CONTRACT.state_dim
            or int(checkpoint_contract.get("action_dim", -1))
            != DEFAULT_CONTRACT.action_dim
        ):
            raise ValueError("registered PPO graph contract is not 123->37")
        if artifact.get("variant") != "PPO" or artifact.get("routing_signature") is not None:
            raise ValueError("PPO artifact has invalid variant/routing metadata")
    else:
        if (
            int(checkpoint_contract.get("state_dim", -1)) != DEFAULT_CONTRACT.state_dim
            or int(checkpoint_contract.get("action_dim", -1)) != DEFAULT_CONTRACT.action_dim
        ):
            raise ValueError("registered T2MIR checkpoint contract is not 123->37")
        routing = artifact.get("routing_signature")
        if not isinstance(routing, Mapping) or checkpoint_contract.get("routing_signature") != routing:
            raise ValueError("T2MIR artifact routing signature is missing or inconsistent")
        variant = artifact.get("variant")
        if variant is not None:
            if variant not in VARIANT_ROUTING_MODES:
                raise ValueError(f"invalid T2MIR artifact variant: {variant!r}")
            actual_modes = (routing["token"]["mode"], routing["task"]["mode"])
            if actual_modes != VARIANT_ROUTING_MODES[variant]:
                raise ValueError("T2MIR artifact variant disagrees with routing signature")
    actual_artifact_sha = canonical_sha256(artifact)
    if expected_artifact_sha != actual_artifact_sha:
        raise ValueError(
            f"artifact SHA-256 mismatch: {expected_artifact_sha} != {actual_artifact_sha}"
        )
    artifact["artifact_sha256"] = expected_artifact_sha
    if verify_checkpoint:
        checkpoint = Path(str(artifact["checkpoint"])).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"artifact checkpoint is missing: {checkpoint}")
        actual = sha256_file(checkpoint)
        if actual != artifact.get("checkpoint_sha256"):
            raise ValueError(
                f"checkpoint SHA-256 mismatch: {actual} != {artifact.get('checkpoint_sha256')}"
            )
        if checkpoint.stat().st_size != int(artifact.get("checkpoint_size_bytes", -1)):
            raise ValueError("checkpoint size differs from the registered artifact")
    return artifact


def validate_active_pointer(
    value: Mapping[str, Any], *, verify_checkpoint: bool = True
) -> dict[str, Any]:
    """Validate the checksum and embedded artifact of an active pointer."""

    pointer = dict(value)
    expected = pointer.pop("active_pointer_sha256", None)
    if int(pointer.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported active pointer format: {pointer.get('format_version')}")
    activation = pointer.get("activation")
    if not isinstance(activation, Mapping) or not activation.get("activated_at"):
        raise ValueError("active pointer has no valid activation record")
    artifact = pointer.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("active pointer has no embedded artifact")
    actual = canonical_sha256(pointer)
    if expected != actual:
        raise ValueError(f"active pointer SHA-256 mismatch: {expected} != {actual}")
    pointer["artifact"] = validate_artifact(
        artifact, verify_checkpoint=verify_checkpoint
    )
    pointer["active_pointer_sha256"] = expected
    return pointer


def load_artifact(path: str | Path, *, verify_checkpoint: bool = True) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if "artifact" in payload:  # active pointer
        pointer = validate_active_pointer(
            payload, verify_checkpoint=verify_checkpoint
        )
        payload = pointer["artifact"]
    if not isinstance(payload, dict):
        raise ValueError("model artifact must decode to a JSON object")
    return validate_artifact(payload, verify_checkpoint=verify_checkpoint)


def promote_artifact(
    artifact_path: str | Path,
    active_path: str | Path,
    *,
    expected_active_sha256: str | None = None,
    allow_incompatible_active: bool = False,
) -> Path:
    """Atomically promote one verified artifact and archive the old pointer."""

    artifact_source = Path(artifact_path).expanduser().resolve()
    artifact = load_artifact(artifact_source)
    active = Path(active_path).expanduser().resolve()
    if active.exists():
        old = json.loads(active.read_text(encoding="utf-8"))
        try:
            old = validate_active_pointer(old)
        except ValueError as exc:
            if not allow_incompatible_active or "deployment contract" not in str(exc):
                raise
            # Contract migrations intentionally invalidate the embedded
            # artifact against the new checkout. Preserve and checksum-check
            # the old pointer, but do not pretend it is loadable now.
            raw = dict(old)
            expected_pointer = raw.pop("active_pointer_sha256", None)
            if expected_pointer != canonical_sha256(raw):
                raise ValueError("incompatible active pointer checksum mismatch") from exc
            old["active_pointer_sha256"] = expected_pointer
        if (
            expected_active_sha256 is not None
            and old["active_pointer_sha256"] != expected_active_sha256
        ):
            raise ValueError(
                "active pointer changed since approval: "
                f"{old['active_pointer_sha256']} != {expected_active_sha256}"
            )
        history = active.parent / "history"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        write_json_atomic(history / f"{active.stem}_{stamp}.json", old)
    elif expected_active_sha256 is not None:
        raise FileNotFoundError("cannot compare expected active SHA: active pointer does not exist")
    pointer = {
        "format_version": FORMAT_VERSION,
        "activation": {
            "activated_at": _timestamp(),
            "source_manifest": str(artifact_source),
            "source_manifest_sha256": sha256_file(artifact_source),
        },
        "artifact": artifact,
    }
    pointer["active_pointer_sha256"] = canonical_sha256(pointer)
    return write_json_atomic(active, pointer)


def build_backend_from_artifact(
    artifact_or_path: Mapping[str, Any] | str | Path,
    *,
    device: str = "cpu",
    prompt_window: str = "last",
    prompt_update_mode: str = "episode",
    prompt_update_interval: int = 64,
):
    reference: dict[str, Any] | None = None
    if isinstance(artifact_or_path, Mapping):
        artifact = validate_artifact(artifact_or_path)
    else:
        reference_path = Path(artifact_or_path).expanduser().resolve()
        artifact = load_artifact(reference_path)
        reference = {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        }
    checkpoint = artifact["checkpoint"]
    digest = artifact["checkpoint_sha256"]
    if artifact["backend"] == "ppo":
        backend = PPOOnnxBackend(checkpoint, expected_sha256=digest)
    else:
        backend = T2MIRTorchBackend(
            checkpoint,
            device=device,
            window=prompt_window,
            update_mode=prompt_update_mode,
            update_interval=prompt_update_interval,
            expected_sha256=digest,
            expected_routing=artifact.get("routing_signature"),
        )
    backend.provenance["model_artifact"] = {
        "name": artifact["name"],
        "variant": artifact["variant"],
        "artifact_sha256": artifact["artifact_sha256"],
        "reference": reference,
    }
    return backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="create an immutable model artifact")
    register.add_argument("--name", required=True)
    register.add_argument("--backend", choices=("ppo", "t2mir"), required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--variant", choices=("A", "B", "C", "D", "PPO"))
    register.add_argument("--output", type=Path, required=True)
    promote = commands.add_parser("promote", help="atomically replace the active pointer")
    promote.add_argument("--artifact", type=Path, required=True)
    promote.add_argument("--active", type=Path, required=True)
    promote.add_argument("--expected-active-sha256", default=None)
    promote.add_argument("--allow-incompatible-active", action="store_true")
    rollback = commands.add_parser(
        "rollback", help="promote an archived history pointer back to active"
    )
    rollback.add_argument("--history", type=Path, required=True)
    rollback.add_argument("--active", type=Path, required=True)
    rollback.add_argument("--expected-active-sha256", default=None)
    rollback.add_argument("--allow-incompatible-active", action="store_true")
    inspect = commands.add_parser("inspect", help="verify and print an artifact/pointer")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--no-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        artifact = create_artifact(
            name=args.name,
            backend=args.backend,
            checkpoint=args.checkpoint,
            variant=args.variant,
        )
        output = write_json_atomic(args.output, artifact)
        print(json.dumps({"status": "REGISTERED", "output": str(output), "artifact": artifact}, indent=2))
        return 0
    if args.command in {"promote", "rollback"}:
        source = args.artifact if args.command == "promote" else args.history
        output = promote_artifact(
            source,
            args.active,
            expected_active_sha256=args.expected_active_sha256,
            allow_incompatible_active=args.allow_incompatible_active,
        )
        status = "PROMOTED" if args.command == "promote" else "ROLLED_BACK"
        print(json.dumps({"status": status, "active": str(output)}, indent=2))
        return 0
    payload = json.loads(args.path.expanduser().read_text(encoding="utf-8"))
    if "artifact" in payload:
        pointer = validate_active_pointer(
            payload, verify_checkpoint=not args.no_checkpoint
        )
        result = {"status": "PASS", "active_pointer": pointer}
    else:
        artifact = validate_artifact(
            payload, verify_checkpoint=not args.no_checkpoint
        )
        result = {"status": "PASS", "artifact": artifact}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
