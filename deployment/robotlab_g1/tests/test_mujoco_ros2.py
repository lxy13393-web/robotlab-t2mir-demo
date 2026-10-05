from __future__ import annotations

import unittest

import numpy as np

from deployment.robotlab_g1.control import DynamicsParameters
from deployment.robotlab_g1.mujoco_ros2 import MujocoRos2Plant, make_mujoco_step_handler
from deployment.robotlab_g1.mujoco_runner import MujocoEpisodeConfig
from deployment.robotlab_g1.ros2_node import PolicyLoopConfig, Ros2PolicyCore
from deployment.robotlab_g1.tests.test_mujoco_runner import _FakeData, _FakeModel, _FakeMujoco


class _SwitchingPolicy:
    observation_dim = 123
    action_dim = 37
    batch_size = 1
    provenance = {"backend": "switching-test"}

    def __init__(self):
        self.calls = 0
        self.transitions = []

    def start_episode(self, episode_index=None):
        pass

    def predict(self, observation):
        self.calls += 1
        return np.full(37, 1.0 if self.calls == 1 else 0.0, dtype=np.float32)

    def record_transition(self, states, policy_actions, rewards, active=None, controller_modes=None):
        self.transitions.append((np.asarray(states).copy(), np.asarray(policy_actions).copy(), float(rewards)))

    def finish_episode(self):
        pass


class MujocoRos2BridgeTest(unittest.TestCase):
    def test_lagged_action_enters_next_observation_but_raw_action_enters_context(self):
        model = _FakeModel()
        data = _FakeData(model)
        policy = _SwitchingPolicy()
        core = Ros2PolicyCore(policy, config=PolicyLoopConfig(require_fresh_reward=True))
        plant = MujocoRos2Plant(
            _FakeMujoco,
            model,
            data,
            dynamics=DynamicsParameters(action_lag=0.4),
        )
        handle = make_mujoco_step_handler(core, plant)

        first = core.step(plant.state(), now_s=0.0)
        handle(first)
        second = core.step(plant.state(), now_s=0.02)
        handle(second)
        third = core.step(plant.state(), now_s=0.04)

        previous = core.contract.observation_slices["previous_action"]
        np.testing.assert_array_equal(second.observation[previous], np.ones(37))
        np.testing.assert_allclose(third.observation[previous], np.full(37, 0.4), atol=1e-7)
        np.testing.assert_array_equal(policy.transitions[0][1], np.ones(37))
        np.testing.assert_array_equal(policy.transitions[1][1], np.zeros(37))

    def test_time_limit_closes_prompt_episode_and_resets_plant(self):
        model = _FakeModel()
        data = _FakeData(model)
        policy = _SwitchingPolicy()
        starts = []
        finishes = []
        policy.start_episode = lambda episode_index=None: starts.append(episode_index)
        policy.finish_episode = lambda: finishes.append(True)
        core = Ros2PolicyCore(policy, config=PolicyLoopConfig(require_fresh_reward=True))
        plant = MujocoRos2Plant(
            _FakeMujoco,
            model,
            data,
            episode_config=MujocoEpisodeConfig(max_policy_steps=2),
        )
        handle = make_mujoco_step_handler(core, plant)

        first = core.step(plant.state(), now_s=0.0)
        result1 = handle(first)
        second = core.step(plant.state(), now_s=0.02)
        result2 = handle(second)
        third = core.step(plant.state(), now_s=0.04)

        self.assertFalse(result1.truncated)
        self.assertTrue(result2.truncated)
        self.assertEqual(result2.termination_reason, "time_limit")
        self.assertEqual(starts, [0, 1])
        self.assertEqual(len(finishes), 1)
        self.assertEqual(third.episode_index, 1)
        self.assertEqual(third.episode_step, 0)
        self.assertEqual(plant.policy_steps, 0)


if __name__ == "__main__":
    unittest.main()
