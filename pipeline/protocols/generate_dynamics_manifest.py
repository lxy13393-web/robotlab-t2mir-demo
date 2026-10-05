"""Generate and validate the canonical 48-task G1 dynamics manifest."""

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import argparse
from pathlib import Path

from pipeline.protocols.dynamics_profile import load_manifest, write_manifest


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=repo / "configs/g1_dynamics_48.json")
    parser.add_argument("--check", action="store_true", help="Validate an existing manifest without overwriting it")
    args = parser.parse_args()
    if not args.check:
        write_manifest(args.output)
    payload, profiles = load_manifest(args.output)
    print(
        f"manifest: {args.output}\n"
        f"tasks: {len(profiles)} train={payload['training_tasks']} eval={len(payload['eval_tasks'])}\n"
        f"sha256: {payload['sha256']}\nvalidation: PASS"
    )


if __name__ == "__main__":
    main()
