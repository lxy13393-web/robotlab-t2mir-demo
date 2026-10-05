import copy
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from algorithms.moe import (
    LinearGLUMoELayer,
    LinearGLUMoELayerContrastive,
    TopKBalancedNoisyGate,
)
from algorithms.policy import DPTTransformerMOE
from commands.train_robotlab_g1_formal import (
    RouteCollector,
    assert_resume_balance_compatible,
    assert_resume_contract_compatible,
    assert_resume_routing_compatible,
    capture_rng_state,
    restore_rng_state,
    balance_signature,
    routing_signature,
)


class GateHarness(nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def forward(self, inputs):
        return self.gate(inputs)


class MoERoutingTest(unittest.TestCase):
    def make_gate(self, mode="topk", threshold=0.6, max_selects=4):
        gate = TopKBalancedNoisyGate(
            input_size=4,
            num_experts=4,
            num_selects=2,
            gate_network="linear",
            add_noise=False,
            use_balance=False,
            routing_mode=mode,
            top_p_threshold=threshold,
            top_p_max_selects=max_selects,
        )
        with torch.no_grad():
            gate.gate_network.weight.copy_(torch.eye(4))
        gate.eval()
        return gate

    def test_topk_matches_legacy_formula(self):
        gate = self.make_gate("topk")
        inputs = torch.tensor(
            [[2.0, -1.0, 0.5, 1.0], [-0.5, 3.0, 1.5, 0.25]]
        )
        output = gate(inputs)
        logits = gate.gate_network(inputs)
        expected_logits, expected_indices = logits.topk(3, dim=1)
        expected_scores = F.softmax(expected_logits[:, :2].float(), dim=1)
        self.assertTrue(torch.equal(output["topK_indices"], expected_indices[:, :2]))
        self.assertTrue(torch.equal(output["topK_scores"], expected_scores))

    def test_topp_dynamic_cardinality_normalization_and_backward(self):
        gate = self.make_gate("topp")
        inputs = torch.tensor(
            [[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            requires_grad=True,
        )
        output = gate(inputs)
        active_widths = (output["topK_scores"] > 0).sum(dim=-1)
        self.assertEqual(active_widths.tolist(), [1, 3])
        self.assertTrue(
            torch.allclose(output["topK_scores"].sum(dim=-1), torch.ones(2))
        )
        position_weights = torch.arange(1, 5, dtype=output["topK_scores"].dtype)
        loss = (output["topK_scores"] * position_weights).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_switch_balance_reaches_unselected_topk_experts(self):
        gate = TopKBalancedNoisyGate(
            input_size=4,
            num_experts=4,
            num_selects=2,
            gate_network="linear",
            add_noise=False,
            use_balance=True,
            balance_loss_weight=1.0,
            balance_loss_mode="switch",
            routing_mode="topk",
        )
        with torch.no_grad():
            gate.gate_network.weight.copy_(torch.tensor([
                [2.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-2.0, -2.0, -2.0, -2.0],
            ]))
        output = gate(torch.ones(8, 4))
        selected = torch.unique(output["topK_indices"])
        self.assertEqual(selected.tolist(), [0, 1])
        output["balance_loss"].backward()
        row_norms = gate.gate_network.weight.grad.norm(dim=1)
        self.assertTrue(torch.all(row_norms[2:] > 0))

    def test_switch_balance_supports_variable_width_topp(self):
        gate = TopKBalancedNoisyGate(
            input_size=4,
            num_experts=4,
            num_selects=2,
            gate_network="linear",
            add_noise=False,
            use_balance=True,
            balance_loss_weight=0.01,
            balance_loss_mode="switch",
            routing_mode="topp",
            top_p_threshold=0.6,
            top_p_max_selects=4,
        )
        inputs = torch.eye(4, requires_grad=True)
        output = gate(inputs)
        self.assertTrue(torch.isfinite(output["balance_loss"]))
        output["balance_loss"].backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_token_and_task_modes_are_independent(self):
        for token_mode in ("topk", "topp"):
            for task_mode in ("topk", "topp"):
                with self.subTest(token=token_mode, task=task_mode):
                    common = {
                        "gate_network": "linear",
                        "gate_add_noise": False,
                        "gate_add_noise_contrastive": False,
                        "gate_use_balance": False,
                        "gate_use_balance_contrastive": False,
                        "token_routing_mode": token_mode,
                        "token_top_p_threshold": 0.6,
                        "token_top_p_max_selects": 4,
                        "task_routing_mode": task_mode,
                        "task_top_p_threshold": 0.6,
                        "task_top_p_max_selects": 4,
                        "task_hard_router": False,
                        "tau": 0.05,
                    }
                    token_layer = LinearGLUMoELayer(
                        4, 8, 3, "gelu", 4, 2, dropout=0.0, **common
                    )
                    task_layer = LinearGLUMoELayerContrastive(
                        4, 8, 3, "gelu", 4, 2, dropout=0.0, **common
                    )
                    self.assertEqual(token_layer.gate.routing_mode, token_mode)
                    self.assertEqual(task_layer.gate.routing_mode, task_mode)
                    self.assertEqual(task_layer.gate_target.routing_mode, task_mode)
                    inputs = torch.randn(2, 5, 4, requires_grad=True)
                    token_output = token_layer(inputs).hidden_states
                    task_output = task_layer(inputs).hidden_states
                    self.assertEqual(token_output.shape, (2, 5, 3))
                    self.assertEqual(task_output.shape, (2, 5, 3))
                    (token_output.sum() + task_output.sum()).backward()
                    self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_task_hard_router_uses_dynamic_width(self):
        layer = LinearGLUMoELayerContrastive(
            4,
            8,
            3,
            "gelu",
            4,
            2,
            dropout=0.0,
            gate_network="linear",
            gate_add_noise_contrastive=False,
            gate_use_balance_contrastive=False,
            task_routing_mode="topp",
            task_top_p_threshold=0.6,
            task_top_p_max_selects=4,
            task_hard_router=True,
            tau=0.05,
        )
        layer.eval()
        output = layer(torch.zeros(2, 5, 4))
        self.assertEqual(output.hidden_states.shape, (2, 5, 3))
        # Uniform four-expert probabilities need three experts to cross p=.6.
        gate_output = layer.gate(torch.zeros(2, 4))
        self.assertEqual((gate_output["topK_scores"] > 0).sum(-1).tolist(), [3, 3])

    def test_route_collector_ignores_topp_padding(self):
        model = GateHarness(self.make_gate("topp"))
        collector = RouteCollector(model)
        # Training keeps hooks disabled to avoid a GPU synchronization on every
        # forward; validation explicitly enables route accounting.
        model(torch.zeros(2, 4))
        self.assertEqual(collector.snapshot()["gate"]["routing_decisions"], 0)
        collector.enable()
        model(torch.zeros(2, 4))
        snapshot = collector.snapshot()["gate"]
        self.assertEqual(snapshot["routing_mode"], "topp")
        self.assertEqual(snapshot["active_width_histogram"], {"3": 2})
        self.assertEqual(snapshot["mean_active_experts"], 3.0)
        self.assertEqual(sum(snapshot["counts"]), 6)

    def test_full_model_state_dict_is_compatible_across_modes(self):
        base_config = {
            "hidden_dim": 8,
            "n_layer": 1,
            "n_head": 2,
            "ff_pdrop": 0.0,
            "ff_moe_pdrop": 0.0,
            "emb_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "activation_function": "gelu_new",
            "prompt_horizon": 2,
            "ff_dim": [16],
            "moe_config": {
                "moe_layers_contrastive_and_balance": [0],
                "task_hard_router": False,
                "use_top_k_indices": False,
                "add_softmax": False,
                "detach_gate_input": False,
                "num_experts": 4,
                "num_selects": 2,
                "token_routing_mode": "topk",
                "token_top_p_threshold": 0.6,
                "token_top_p_max_selects": 4,
                "expert_dim": 16,
                "gate_balance_loss_weight": 0.01,
                "gate_add_noise": False,
                "gate_noise_epsilon": 0.01,
                "tau": 0.05,
                "contrastive_loss_weight": 0.01,
                "num_experts_contrastive": 4,
                "num_selects_contrastive": 2,
                "task_routing_mode": "topk",
                "task_top_p_threshold": 0.6,
                "task_top_p_max_selects": 4,
                "expert_dim_contrastive": 16,
            },
        }
        dual_config = copy.deepcopy(base_config)
        dual_config["moe_config"]["token_routing_mode"] = "topp"
        dual_config["moe_config"]["task_routing_mode"] = "topp"
        fixed = DPTTransformerMOE(
            3, 2, base_config, action_tanh=False, discrete_environment=False
        )
        dynamic = DPTTransformerMOE(
            3, 2, dual_config, action_tanh=False, discrete_environment=False
        )
        fixed_state = fixed.state_dict()
        dynamic_state = dynamic.state_dict()
        self.assertEqual(list(fixed_state), list(dynamic_state))
        self.assertEqual(
            {key: value.shape for key, value in fixed_state.items()},
            {key: value.shape for key, value in dynamic_state.items()},
        )
        dynamic.load_state_dict(fixed_state, strict=True)
        fixed.load_state_dict(dynamic_state, strict=True)

    def test_routing_signature_covers_both_switches(self):
        config = {
            "moe_config": {
                "num_experts": 6,
                "num_selects": 2,
                "token_routing_mode": "topp",
                "token_top_p_threshold": 0.4,
                "token_top_p_max_selects": 6,
                "num_experts_contrastive": 8,
                "num_selects_contrastive": 2,
                "task_routing_mode": "topk",
                "task_top_p_threshold": 0.5,
                "task_top_p_max_selects": 8,
                "task_hard_router": False,
            }
        }
        signature = routing_signature(config)
        self.assertEqual(signature["token"]["mode"], "topp")
        self.assertEqual(signature["task"]["mode"], "topk")

    def test_official_configs_cover_complete_ablation_matrix(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        expected = {
            "args_robotlab_g1_official_mixed_v1.yaml": ("topk", "topk"),
            "args_robotlab_g1_official_mixed_v1_token_topp.yaml": ("topp", "topk"),
            "args_robotlab_g1_official_mixed_v1_task_topp.yaml": ("topk", "topp"),
            "args_robotlab_g1_official_mixed_v1_dual_topp.yaml": ("topp", "topp"),
        }
        loaded = {}
        for filename, modes in expected.items():
            with self.subTest(config=filename):
                config = yaml.safe_load((config_dir / filename).read_text())
                loaded[filename] = config
                signature = routing_signature(config)
                self.assertEqual(
                    (signature["token"]["mode"], signature["task"]["mode"]),
                    modes,
                )
                self.assertFalse(signature["task_hard_router"])

        # Routing modes are the only experimental variable across A/B/C/D.
        normalized = []
        for filename in expected:
            config = copy.deepcopy(loaded[filename])
            config["moe_config"]["token_routing_mode"] = "topk"
            config["moe_config"]["task_routing_mode"] = "topk"
            normalized.append(config)
        for config in normalized[1:]:
            self.assertEqual(config, normalized[0])

    def test_resume_rejects_router_semantic_mismatch(self):
        fixed = {
            "moe_config": {
                "num_experts": 6,
                "num_selects": 2,
                "token_routing_mode": "topk",
                "num_experts_contrastive": 8,
                "num_selects_contrastive": 2,
                "task_routing_mode": "topk",
            }
        }
        dynamic = copy.deepcopy(fixed)
        dynamic["moe_config"]["task_routing_mode"] = "topp"
        checkpoint = {"config": fixed}
        assert_resume_routing_compatible(checkpoint, fixed)
        with self.assertRaises(ValueError):
            assert_resume_routing_compatible(checkpoint, dynamic)

    def test_balance_signature_rejects_cross_objective_resume(self):
        legacy = {
            "moe_config": {
                "gate_balance_loss_weight": 0.01,
            }
        }
        balanced = copy.deepcopy(legacy)
        balanced["moe_config"].update({
            "gate_use_balance_contrastive": True,
            "gate_balance_loss_weight_contrastive": 0.01,
            "task_balance_loss_mode": "switch",
        })
        checkpoint = {"config": legacy}
        self.assertFalse(balance_signature(legacy)["task"]["enabled"])
        assert_resume_balance_compatible(checkpoint, legacy)
        with self.assertRaises(ValueError):
            assert_resume_balance_compatible(checkpoint, balanced)

    def test_resume_contract_and_rng_state_are_exact(self):
        assert_resume_contract_compatible({"run_contract_sha256": "abc"}, "abc")
        with self.assertRaises(ValueError):
            assert_resume_contract_compatible({"run_contract_sha256": "abc"}, "other")
        # A legacy/manual run remains usable when no formal launcher contract
        # was requested, but the A--D launcher always supplies one.
        assert_resume_contract_compatible({}, None)

        torch.manual_seed(123)
        state = capture_rng_state()
        first = torch.rand(5)
        restore_rng_state(state)
        second = torch.rand(5)
        self.assertTrue(torch.equal(first, second))

    def test_invalid_router_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_gate("unsupported")
        with self.assertRaises(ValueError):
            self.make_gate("topp", threshold=0.0)
        with self.assertRaises(ValueError):
            self.make_gate("topp", max_selects=5)


if __name__ == "__main__":
    unittest.main()
