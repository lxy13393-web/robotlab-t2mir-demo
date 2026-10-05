# T2MIR and DT2MIR implementation

This directory contains the project's upstream-derived T2MIR baseline and the
DT2MIR dynamic-routing extension for Unitree G1 control.

## What is implemented here

- a RobotLab G1 interface with 123-dimensional observations and 37 actions;
- official-style prompt/query dataset loading;
- fixed Top-2 routing for the T2MIR baseline;
- independently configurable Top-p routing in the token and task MoE layers;
- the dual-dynamic DT2MIR variant;
- optional routing load-balancing losses;
- actor-mean supervision, validation, checkpointing and early stopping;
- routing statistics and formal training gates.

The primary training entry point is:

```bash
python -m commands.train_robotlab_g1_formal --help
```

The upstream project and the local changes are recorded separately in
[`docs/UPSTREAM.md`](docs/UPSTREAM.md) and
[`docs/MODIFICATIONS.md`](docs/MODIFICATIONS.md).

This directory must not be described as an independently authored clean-room
implementation. It contains modified upstream source whose license was not
explicitly declared in the snapshot used by this project.
