# Evaluation

Success rate and fall rate are the primary closed-loop metrics. Offline action
MSE is used only for training diagnostics because a small one-step error does
not guarantee stable long-horizon control.

## Held-out dynamics experiment

The formal Isaac Sim experiment evaluates a held-out dynamics profile with 16
rounds of 32 parallel environments, totaling 512 episodes per policy. All
policies share the same goals, reset distribution, controller, success rule
and episode limits. Context policies begin without privileged task information
and update causal history between rounds.

| Policy | Success | Fall | Last minus first-round success |
|---|---:|---:|---:|
| Base PPO | 40.63% | 59.38% | +9.38 pp |
| Query-only MLP | 19.92% | 77.73% | +12.50 pp |
| T2MIR | 74.41% | 9.96% | +25.00 pp |
| DT2MIR | **79.30%** | 13.09% | **+40.63 pp** |

These values belong to this frozen held-out protocol and must not be mixed
with the paired transfer experiment.

## Routing efficiency

Routing statistics are aggregated over all 42 validation tasks.

| Policy | Mean token experts | Mean task experts | Activated expert parameters |
|---|---:|---:|---:|
| T2MIR | 2.00 | 2.00 | 197.9K |
| DT2MIR | **1.79** | **1.92** | **183.8K** |

The parameter count represents experts contributing to one routing decision,
not stored parameters or measured inference latency.

## Paired Isaac Sim to MuJoCo transfer

The transfer audit freezes one DT2MIR checkpoint and evaluates identical goal
files and controller settings in both simulators. Two independently generated
64-goal sets provide 128 paired trials.

| Simulator | Success | Fall |
|---|---:|---:|
| Isaac Sim | 114/128 = **89.06%** | 4/128 = 3.13% |
| MuJoCo | 105/128 = **82.03%** | 9/128 = 7.03% |

The pooled success gap is -7.03 percentage points. The pooled McNemar exact
test is `p=0.163`; one individual goal set has a significant gap. The supported
claim is useful aggregate sim-to-sim transfer with target-set sensitivity, not
lossless equivalence between simulators.

Machine-readable reports and goal manifests are under `artifacts/results/`
and `artifacts/manifests/`.
