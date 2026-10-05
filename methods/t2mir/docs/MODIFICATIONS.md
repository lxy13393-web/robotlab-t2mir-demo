# Local modifications

This project preserves the T2MIR-DPT architecture while extending it for the
RobotLab Unitree G1 multi-dynamics benchmark.

## Modified capabilities

- Added 123-state/37-action RobotLab G1 training support.
- Added the 42-train/6-held-out dynamics dataset interface.
- Added configurable fixed Top-k and dynamic Top-p routing to both MoE layers.
- Added the dual Top-p **DT2MIR** configuration.
- Added optional load-balancing objectives and routing diagnostics.
- Added actor-mean query supervision support.
- Added random training batches and independent validation queries.
- Added checkpointing, resume state, early stopping and RNG restoration.
- Added formal run contracts and routing-signature checks.
- Added CPU tests for routing behavior and training gates.

## New project-facing files

- `commands/train_robotlab_g1_formal.py`
- `commands/run_robotlab_g1_abcd_training.py`
- `commands/inspect_robotlab_routing_gate.py`
- RobotLab-specific files under `configs/`
- RobotLab-specific tests under `tests/`
- `ROBOTLAB_DYNAMIC_ROUTING.md`

## Terminology

- **T2MIR** denotes the matched fixed Top-2 token/task routing baseline.
- **DT2MIR** denotes dynamic Top-p routing in both token and task MoE layers.

Internal A/B/C/D identifiers may remain in experiment manifests for backward
compatibility, but user-facing reports use the names T2MIR and DT2MIR.
