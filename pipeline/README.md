# Experiment pipeline

This package contains the end-to-end RobotLab workflow, grouped by responsibility:

- `expert/`: PPO expert training, playback and checkpoint-bank construction.
- `dataset/`: restartable collection, merge, preparation, relabeling and validation.
- `protocols/`: shared dynamics, goal, paired-scenario and causal-context contracts.
- `evaluation/`: Isaac Sim online evaluation, sweeps and report aggregation.
- `tests/`: CPU-only contract tests for the shared pipeline.

Prefer package entry points such as:

```bash
python -m pipeline.protocols.generate_dynamics_manifest --check
python -m pipeline.dataset.validate_official_mixed_formal --help
python -m pipeline.evaluation.evaluate_t2mir_online --help
```
