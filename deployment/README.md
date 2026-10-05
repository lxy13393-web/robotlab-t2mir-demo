# Unitree G1 MuJoCo and ROS 2 deployment

The Python package is located in `robotlab_g1/`. It is the minimal deployment
boundary for the trained RobotLab G1
policies. It keeps the 123-dimensional observation, 37-dimensional action and
joint-order contracts identical across policy backends.

## Components

- `contract.py`, `observation.py`: frozen policy dimensions and observation layout.
- `control.py`, `mujoco_runner.py`: action lag, implicit PD and MuJoCo execution.
- `context.py`, `policy_backends.py`: causal T2MIR/DT2MIR context and inference.
- `model_registry.py`: immutable checkpoint identity and routing metadata.
- `evaluate_mujoco.py`: goal-level closed-loop evaluation.
- `ros2_node.py`, `mujoco_ros2.py`: ROS-independent core and MuJoCo feedback bridge.
- `run_mujoco.py`, `run_ros2_mujoco.py`: command-line entry points.
- `prepare_mjcf.py`, `assets.py`, `preflight.py`: asset conversion and validation.

## Preflight

```bash
python -m deployment.robotlab_g1.preflight \
  --robot-mjcf <G1_ROBOT.xml> \
  --scene-mjcf <G1_SCENE.xml> \
  --strict
```

## MuJoCo evaluation

```bash
python -m deployment.robotlab_g1.evaluate_mujoco \
  --mjcf <G1_SCENE.xml> \
  --robot-mjcf <G1_ROBOT.xml> \
  --model-manifest <MODEL_MANIFEST.json> \
  --profile-ids 47 \
  --episodes 64 \
  --allow-proxy-reward \
  --output-dir <OUTPUT_DIR>
```

## ROS 2 bridge

```bash
python -m deployment.robotlab_g1.run_ros2_mujoco --help
```

The ROS 2 path accepts velocity/goal commands, executes the registered policy
in MuJoCo and returns reward, termination and timing diagnostics to the causal
context buffer.

## Terminal video recording

The MuJoCo runner supports EGL off-screen H.264 recording without a desktop
viewer. Add these flags to a normal `run_mujoco` command:

```bash
--video \
--video-output outputs/deployment/videos/dt2mir_mujoco.mp4 \
--video-fps 30
```

Recording observes the existing policy-rate state and does not change the
controller, model actions or dynamics.

## Verification

```bash
python -m unittest discover -s deployment/robotlab_g1/tests -p 'test_*.py'
```

Generated MJCF files and checkpoints belong under ignored local asset
directories and are not source-controlled.
