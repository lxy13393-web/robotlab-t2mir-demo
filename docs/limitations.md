# Limitations and claim boundary

## Supported by the current evidence

- DT2MIR improves success and online adaptation over the matched fixed-routing
  T2MIR in the frozen held-out Isaac Sim protocol.
- Dynamic routing reduces the average number of logically active expert
  parameters in the reported validation audit.
- One frozen DT2MIR policy completes a strict 128-goal paired transfer audit in
  Isaac Sim and MuJoCo with useful aggregate performance.
- The ROS 2 to policy to MuJoCo command and feedback path is implemented.

## Not claimed

- No real Unitree G1 hardware experiment has been performed.
- The current Top-p implementation does not provide measured sparse-kernel or
  wall-clock inference acceleration.
- The paired MuJoCo study covers one held-out dynamics profile, one trained
  checkpoint and two goal sets; it is not a full multi-profile transfer study.
- Isaac Sim and MuJoCo rewards are not term-by-term identical. Success and
  falls, rather than cross-simulator return, are the primary transfer metrics.
- The results do not establish a universal ranking for every dynamics profile,
  seed or controller setting.

## Reproducibility boundary

Datasets, checkpoints and generated robot assets are excluded from Git because
of size and redistribution constraints. The repository retains code, frozen
small result summaries, goal manifests, hashes and protocol documentation.

The upstream T2MIR snapshot has no explicit license in the version used here.
This repository must remain private unless the upstream license is clarified
or the copied source is replaced by an upstream reference and a legally
distributable patch set. Robot assets require their own license review.
