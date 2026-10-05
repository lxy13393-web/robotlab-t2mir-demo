# Reproduction

## Prerequisites

The validated stack is Ubuntu 22.04, Python 3.10.20, PyTorch 2.4.0+cu121,
NumPy 1.26.4, Isaac Sim 4.2, Isaac Lab 1.2, MuJoCo 3.1.5 and ROS 2 Humble.
Install RobotLab and Isaac Lab following their upstream instructions, then
install the project packages:

```bash
python -m pip install -e ./exts/robot_lab
python -m pip install -r methods/t2mir/requirements.txt
python -m pip install -r deployment/requirements.txt
```

Both requirement files pin MuJoCo 3.1.5 so installing the method dependencies
does not silently replace the validated deployment runtime.

The commands below are run from the repository root unless a `cd` is shown.
Replace angle-bracket paths with local datasets, checkpoints or model
manifests. These large artifacts are intentionally not stored in Git.

## Validate the dynamics definition

```bash
python -m pipeline.protocols.generate_dynamics_manifest --check
```

## Build and validate the dataset

The full pipeline is restartable. Inspect each command before launching Isaac
Sim because specialist training and collection are GPU-intensive.

```bash
python -m pipeline.expert.build_dynamics_checkpoint_bank --help
python -m pipeline.dataset.run_official_mixed_collection_queue --help
python -m pipeline.dataset.merge_official_mixed_staging --help
python -m pipeline.dataset.prepare_official_mixed_formal --help
python -m pipeline.dataset.validate_official_mixed_formal --help
python -m pipeline.dataset.derive_official_mixed_actor_mean --help
```

The final validator must report `FORMAL-VALIDATION PASS` before training.

## Train T2MIR or DT2MIR

```bash
cd methods/t2mir

# Fixed Top-2 T2MIR
python -m commands.train_robotlab_g1_formal \
  --config configs/args_robotlab_g1_official_mixed_v1_mid_balanced.yaml \
  --dataset-dir <ACTOR_MEAN_DATASET>/dpt \
  --output-dir <RUNS>/t2mir \
  --steps 100000 --seed 42 --device cuda:0

# Dual Top-p DT2MIR
python -m commands.train_robotlab_g1_formal \
  --config configs/args_robotlab_g1_official_mixed_v1_dual_topp_mid_balanced.yaml \
  --dataset-dir <ACTOR_MEAN_DATASET>/dpt \
  --output-dir <RUNS>/dt2mir \
  --steps 100000 --seed 42 --device cuda:0
```

Do not compare runs unless their dataset fingerprint, label source, seed
policy, optimizer settings and evaluation protocol match.

## Isaac Sim closed-loop evaluation

The evaluator requires a registered model manifest or an explicit checkpoint,
variant and SHA-256. It uses causal online context and never reads held-out
dataset files.

```bash
python -m pipeline.evaluation.evaluate_t2mir_online \
  --model-manifest <MODEL_MANIFEST.json> \
  --profile-id 47 \
  --episodes 16 \
  --num_envs 32 \
  --seed 42 \
  --output-dir <OUTPUT>/isaac \
  --headless
```

Use `python -m pipeline.evaluation.evaluate_t2mir_online --help` to verify
arguments against the checked-out version before launching a long run.

## MuJoCo preflight and evaluation

```bash
python -m deployment.robotlab_g1.preflight \
  --robot-mjcf <G1_ROBOT.xml> \
  --scene-mjcf <G1_SCENE.xml> \
  --strict

python -m deployment.robotlab_g1.evaluate_mujoco \
  --mjcf <G1_SCENE.xml> \
  --robot-mjcf <G1_ROBOT.xml> \
  --model-manifest <MODEL_MANIFEST.json> \
  --profile-ids 47 \
  --episodes 64 \
  --seed 42 \
  --allow-proxy-reward \
  --output-dir <OUTPUT>/mujoco
```

## ROS 2 bridge

```bash
python -m deployment.robotlab_g1.run_ros2_mujoco --help
```

The bridge accepts high-level commands and publishes diagnostics while feeding
MuJoCo observations and rewards back into the same context-policy interface.

## Regenerate figures

```bash
python tools/generate_public_figures.py \
  --metrics artifacts/results/final_metrics.json \
  --output-dir artifacts/figures
```

## Lightweight checks

```bash
python -m compileall -q deployment pipeline methods/t2mir
python -m unittest discover -s pipeline/tests -p 'test_*.py'
python -m unittest discover -s deployment/robotlab_g1/tests -p 'test_*.py'
(cd methods/t2mir && python -m unittest discover -s tests -p 'test_*.py')
```

The method tests run from `methods/t2mir/` because the upstream-derived code
uses that directory as its Python import root.
