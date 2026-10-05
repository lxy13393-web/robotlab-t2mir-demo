# Method

## Problem

The controller must drive a Unitree G1 toward successive goal poses while its
hidden dynamics vary. The policy observes a 123-dimensional proprioceptive and
command vector and produces 37 joint actions. It is not given action lag,
motor strength, payload or friction as privileged inputs.

The benchmark contains 48 fixed dynamics profiles formed from:

- four action-lag values;
- three motor-strength values;
- two torso payloads;
- two friction values.

Forty-two profiles are used for offline training and six combinations are held
out. The held-out parameters remain hidden during closed-loop evaluation.

## T2MIR baseline

The baseline follows the DPT branch of T2MIR. A query state is evaluated
against a prompt of previous `(state, action, reward)` transitions. The model
uses a token-level mixture of experts and a task-level mixture of experts;
both routers select a fixed Top-2 set.

Online evaluation is causal. The first episode starts with an empty prompt.
Later episodes use only policy-generated history from completed interaction;
task IDs and dynamics parameters are never supplied to the model.

## DT2MIR

DT2MIR replaces fixed Top-2 selection with independently configurable Top-p
routing at both MoE layers. The active set is the smallest probability prefix
that crosses the configured threshold, subject to the layer limit. The
training objective and model width are otherwise matched to T2MIR.

The formal comparison uses:

- **T2MIR:** token Top-2, task Top-2;
- **DT2MIR:** token Top-p, task Top-p;
- the same actor-mean query labels, optimizer family and online controller;
- a matched load-balancing coefficient where specified by the frozen run
  contract.

DT2MIR activates 183.8K expert parameters per routing decision on average,
versus 197.9K for T2MIR, a 7.1% reduction. This is a logical activation metric.
The current implementation uses padded rectangular tensors and therefore does
not claim measured wall-clock sparse-compute acceleration.

## Control and deployment

A high-level goal controller maps relative position and yaw errors to velocity
commands. The learned policy remains the low-level adaptive controller. The
same observation/action contract is used by Isaac Sim, MuJoCo and the ROS 2
bridge. MuJoCo uses native implicit PD at a 1 ms integration step after
open-loop actuator calibration against PhysX.
