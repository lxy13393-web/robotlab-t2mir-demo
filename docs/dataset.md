# Dataset

## Split and scale

The formal dataset contains 42 training dynamics profiles and excludes six
held-out profiles. Each training profile contributes:

- 25 PPO checkpoints sampled across training;
- 24 valid continuous windows per checkpoint;
- 64 transitions per window;
- 38,400 prompt transitions per profile.

The complete prompt set therefore contains 1,612,800 transitions with
123-dimensional states and 37-dimensional actions.

## Official-style construction

The pipeline mirrors the T2MIR-DPT separation between prompt and query data:

1. Train a specialist PPO for each training dynamics profile and retain a
   checkpoint bank spanning weak to mature behavior.
2. Roll out every checkpoint under its own hidden dynamics.
3. Extract continuous 64-step windows without reset, fall or timeout
   boundaries.
4. Preserve `(state, action, reward, next_state)` prompt transitions.
5. Relabel the same query states with the selected task specialist.
6. Build task-indexed prompt and query pickle files consumed by the unmodified
   T2MIR dataset loader.

The task index is used for storage and training-time sampling only. It is not
part of the model input.

## Actor-mean derivative

The frozen collected dataset retains stochastic PPO actions. The formal
control models use a separately versioned derivative in which query actions
are replaced by deterministic actor means. Prompt trajectories, states,
rewards and task splits are unchanged. Its provenance points back to the
frozen formal dataset; it is not presented as a replacement for the source
collection.

## Validation contract

The strict validator requires:

- exactly 42 training tasks and no held-out task files;
- 25 checkpoint shards and 600 windows per task;
- finite `float32` tensors with exact state/action dimensions;
- no terminal boundary inside a window;
- `next_state[t] == state[t+1]` under the recorded observation semantics;
- prompt and query state alignment;
- reproducible query labels from the recorded checkpoint and seed;
- successful loading through the official T2MIR loader.

The full dataset and checkpoints are not committed to Git. Reproduction relies
on the collection code, immutable provenance records and SHA-256 fingerprints.
