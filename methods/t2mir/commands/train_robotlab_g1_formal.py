"""Formal offline T2MIR-DPT training for RobotLab G1 datasets."""

# Make the method root importable when this command is run by file path.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _method_root = _Path(__file__).resolve().parents[1]
    if str(_method_root) not in _sys.path:
        _sys.path.insert(0, str(_method_root))

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from algorithms.datasets import DPT_Dataset
from algorithms.moe import TopKBalancedNoisyGate
from algorithms.policy import DPTTransformerMOE
from algorithms.tools import data_loader, query_data_loader


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def routing_signature(config):
    """Return the routing semantics that must travel with every checkpoint."""
    moe = config["moe_config"]

    def branch(prefix, experts_key, selects_key):
        mode = str(moe.get(f"{prefix}_routing_mode", moe.get("routing_mode", "topk"))).lower()
        if mode not in {"topk", "topp"}:
            raise ValueError(f"unsupported {prefix}_routing_mode: {mode}")
        num_experts = int(moe[experts_key])
        fixed_top_k = int(moe[selects_key])
        threshold = float(
            moe.get(f"{prefix}_top_p_threshold", moe.get("top_p_threshold", 0.4))
        )
        raw_max_selects = moe.get(
            f"{prefix}_top_p_max_selects", moe.get("top_p_max_selects", num_experts)
        )
        max_selects = num_experts if raw_max_selects is None else int(raw_max_selects)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"{prefix}_top_p_threshold must be in (0, 1], got {threshold}")
        if not 1 <= fixed_top_k <= num_experts:
            raise ValueError(
                f"{prefix} fixed Top-K must be in [1, {num_experts}], got {fixed_top_k}"
            )
        if not 1 <= max_selects <= num_experts:
            raise ValueError(
                f"{prefix}_top_p_max_selects must be in [1, {num_experts}], got {max_selects}"
            )
        return {
            "mode": mode,
            "num_experts": num_experts,
            "fixed_top_k": fixed_top_k,
            "top_p_threshold": threshold,
            "top_p_max_selects": max_selects,
        }

    return {
        "schema_version": 1,
        "token": branch("token", "num_experts", "num_selects"),
        "task": branch(
            "task", "num_experts_contrastive", "num_selects_contrastive"
        ),
        "task_hard_router": bool(moe.get("task_hard_router", False)),
    }


def balance_signature(config):
    """Return auxiliary-routing semantics that define the training objective."""
    moe = config["moe_config"]
    return {
        "schema_version": 1,
        "token": {
            "enabled": bool(moe.get("gate_use_balance", True)),
            "mode": str(moe.get("token_balance_loss_mode", "legacy")).lower(),
            "weight": float(moe.get("gate_balance_loss_weight", 1e-2)),
        },
        "task": {
            "enabled": bool(moe.get("gate_use_balance_contrastive", False)),
            "mode": str(moe.get("task_balance_loss_mode", "legacy")).lower(),
            "weight": float(moe.get("gate_balance_loss_weight_contrastive", 1e-2)),
        },
    }


def assert_resume_routing_compatible(checkpoint, config):
    expected = routing_signature(config)
    saved = checkpoint.get("routing_signature")
    if saved is None and "config" in checkpoint:
        saved = routing_signature(checkpoint["config"])
    if saved is not None and saved != expected:
        raise ValueError(
            "checkpoint routing configuration does not match the requested run:\n"
            f"checkpoint={json.dumps(saved, sort_keys=True)}\n"
            f"requested={json.dumps(expected, sort_keys=True)}"
        )


def assert_resume_balance_compatible(checkpoint, config):
    expected = balance_signature(config)
    saved = checkpoint.get("balance_signature")
    if saved is None and "config" in checkpoint:
        # Legacy checkpoints did not persist a separate signature, but their
        # resolved config is sufficient to reconstruct the exact defaults.
        saved = balance_signature(checkpoint["config"])
    if saved is not None and saved != expected:
        raise ValueError(
            "checkpoint balance objective does not match the requested run:\n"
            f"checkpoint={json.dumps(saved, sort_keys=True)}\n"
            f"requested={json.dumps(expected, sort_keys=True)}"
        )


def assert_resume_contract_compatible(checkpoint, expected_sha256):
    """Reject cross-run or legacy resume when the launcher supplies a contract."""

    if expected_sha256 is None:
        return
    saved = checkpoint.get("run_contract_sha256")
    if saved != expected_sha256:
        raise ValueError(
            "resume checkpoint belongs to a different or legacy run contract: "
            f"checkpoint={saved!r}, requested={expected_sha256!r}"
        )


def task_axis(values, indices):
    return {key: np.stack(np.split(value, indices[1:], axis=0)) for key, value in values.items()}


def split_queries(query, fraction, seed):
    rng = np.random.default_rng(seed)
    train = {key: [] for key in query}
    validation = {key: [] for key in query}
    count = query["states"].shape[1]
    validation_count = max(1, round(count * fraction))
    for task_id in range(query["states"].shape[0]):
        order = rng.permutation(count)
        validation_ids, train_ids = order[:validation_count], order[validation_count:]
        for key in query:
            validation[key].append(query[key][task_id, validation_ids])
            train[key].append(query[key][task_id, train_ids])
    return ({key: np.stack(value) for key, value in train.items()},
            {key: np.stack(value) for key, value in validation.items()})


class RouteCollector:
    def __init__(self, model, *, enabled=False):
        self.enabled = enabled
        self.gates = {}
        self.counts = {}
        self.width_histograms = {}
        self.decision_counts = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, TopKBalancedNoisyGate) and getattr(
                module, "track_activation", True
            ):
                self.gates[name] = module
                self.counts[name] = torch.zeros(module.num_experts, dtype=torch.long)
                self.width_histograms[name] = torch.zeros(
                    module.num_experts + 1, dtype=torch.long
                )
                self.decision_counts[name] = 0
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name, module):
        def hook(_module, _inputs, output):
            if not self.enabled:
                return
            indices = output["topK_indices"].detach().cpu()
            scores = output["topK_scores"].detach().cpu()
            if module.routing_mode == "topp":
                active = scores > 0
            else:
                # Fixed Top-K positions remain selected even when a caller opts
                # out of the softmax and a valid gate score is non-positive.
                active = torch.ones_like(scores, dtype=torch.bool)
            active_indices = indices[active]
            self.counts[name] += torch.bincount(
                active_indices, minlength=self.counts[name].numel()
            )
            widths = active.sum(dim=-1).to(torch.long)
            self.width_histograms[name] += torch.bincount(
                widths, minlength=self.width_histograms[name].numel()
            )
            self.decision_counts[name] += int(widths.numel())
        return hook

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def reset(self):
        for value in self.counts.values():
            value.zero_()
        for value in self.width_histograms.values():
            value.zero_()
        for name in self.decision_counts:
            self.decision_counts[name] = 0

    def snapshot(self):
        result = {}
        for name, counts in self.counts.items():
            total = counts.sum().item()
            width_histogram = self.width_histograms[name]
            decisions = self.decision_counts[name]
            weighted_width = sum(
                width * int(count) for width, count in enumerate(width_histogram.tolist())
            )
            gate = self.gates[name]
            result[name] = {
                "gate_kind": gate.gate_kind,
                "routing_mode": gate.routing_mode,
                "fixed_top_k": gate.num_selects if gate.routing_mode == "topk" else None,
                "top_p_threshold": (
                    gate.top_p_threshold if gate.routing_mode == "topp" else None
                ),
                "top_p_max_selects": (
                    gate.top_p_max_selects if gate.routing_mode == "topp" else None
                ),
                "counts": counts.tolist(),
                "fractions": (counts.float() / max(total, 1)).tolist(),
                "routing_decisions": decisions,
                "mean_active_experts": weighted_width / max(decisions, 1),
                "active_width_histogram": {
                    str(width): int(count)
                    for width, count in enumerate(width_histogram.tolist())
                    if count
                },
            }
        return result


def capture_rng_state():
    """Capture every RNG consumed by formal training for equivalent resume."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint has no complete RNG state")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"resume checkpoint RNG state is missing {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Resume checkpoints are loaded with ``map_location=device`` so their CPU
    # RNG ByteTensor is moved to CUDA together with model/optimizer tensors.
    # ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all`` both expect
    # CPU ByteTensors; normalize explicitly instead of depending on the load
    # location.  This preserves exact RNG bytes and is backward-compatible
    # with checkpoints already saved on CPU.
    def cpu_byte_tensor(value):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()

    torch.set_rng_state(cpu_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available():
        cuda_state = state["torch_cuda"]
        if cuda_state is None:
            raise ValueError("CUDA resume requested from a checkpoint without CUDA RNG state")
        torch.cuda.set_rng_state_all([cpu_byte_tensor(value) for value in cuda_state])


def save_checkpoint(
    path,
    policy,
    optimizer,
    scheduler,
    config,
    mean,
    std,
    step,
    best_loss,
    stale_evaluations,
    run_contract_sha256,
    initialization_provenance=None,
    dataset_provenance=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "config": config,
                "routing_signature": routing_signature(config),
                "balance_signature": balance_signature(config),
                "state_mean": torch.from_numpy(mean), "state_std": torch.from_numpy(std),
                "step": step, "best_validation_mse": best_loss,
                "stale_evaluations": stale_evaluations,
                "rng_state": capture_rng_state(),
                "run_contract_sha256": run_contract_sha256,
                "initialization_provenance": initialization_provenance,
                "dataset_provenance": dataset_provenance,
                "state_dim": mean.shape[0], "action_dim": policy.action_dim}, path)


@torch.no_grad()
def validate(policy, dataset, validation, config, device, action_mean, route_collector, seed):
    policy.eval()
    route_collector.enable()
    route_by_task = {}
    task_mse, baseline_mse, behavior_mse = [], [], []
    rng = np.random.default_rng(seed)
    horizon = config["prompt_episode_horizon"]
    sample_limit = config["validation_samples_per_task"]
    batch_size = config.get("validation_batch_size", config["batch_size"])
    for task_id in range(dataset.num_tasks):
        episode_ids = np.sort(rng.choice(dataset.num_episodes, horizon, replace=False))
        prompt_states = dataset.datasets["states"][task_id, episode_ids].reshape(1, -1, dataset.datasets["states"].shape[-1])
        prompt_actions = dataset.datasets["actions"][task_id, episode_ids].reshape(1, -1, dataset.datasets["actions"].shape[-1])
        prompt_rewards = dataset.datasets["rewards"][task_id, episode_ids].reshape(1, -1, 1)
        query_count = min(sample_limit, validation["states"].shape[1])
        query_ids = rng.choice(validation["states"].shape[1], query_count, replace=False)
        squared_error, baseline_squared_error, behavior_squared_error, values = 0.0, 0.0, 0, 0
        route_collector.reset()
        for start in range(0, query_count, batch_size):
            ids = query_ids[start:start + batch_size]
            query_states = torch.from_numpy(validation["states"][task_id, ids]).float().to(device).unsqueeze(1)
            targets = torch.from_numpy(validation["actions"][task_id, ids]).float().to(device)
            behavior = torch.from_numpy(validation["behavior_actions"][task_id, ids]).float().to(device)
            batch = len(ids)
            states = torch.from_numpy(prompt_states).float().to(device).repeat(batch, 1, 1)
            actions = torch.from_numpy(prompt_actions).float().to(device).repeat(batch, 1, 1)
            rewards = torch.from_numpy(prompt_rewards).float().to(device).repeat(batch, 1, 1)
            predictions = policy(states, actions, rewards, query_states, eval=True)
            squared_error += F.mse_loss(predictions, targets, reduction="sum").item()
            baseline_squared_error += F.mse_loss(action_mean.expand_as(targets), targets, reduction="sum").item()
            behavior_squared_error += F.mse_loss(behavior, targets, reduction="sum").item()
            values += targets.numel()
        task_mse.append(squared_error / values)
        baseline_mse.append(baseline_squared_error / values)
        behavior_mse.append(behavior_squared_error / values)
        route_by_task[str(task_id)] = route_collector.snapshot()
    route_collector.disable()
    policy.train()
    return (float(np.mean(task_mse)), task_mse, float(np.mean(baseline_mse)), baseline_mse,
            float(np.mean(behavior_mse)), behavior_mse, route_by_task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/args_robotlab_g1_lag_formal.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("../datasets/RobotLab-G1-Lag/formal_aligned"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/RobotLab-G1-Lag/formal_fixed_topk_aligned"))
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume policy, optimizer, scheduler and step from a checkpoint")
    parser.add_argument(
        "--finetune-from", type=Path, default=None,
        help=("Initialize policy weights only, then reset optimizer, scheduler, step, "
              "validation best and RNG for a new dataset/training objective"),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config learning rate for a fresh run or --finetune-from")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override config warmup steps (0 is useful for low-LR fine-tuning)")
    parser.add_argument(
        "--supervision-mode", choices=("all_tokens", "final_token"), default=None,
        help="Override the config supervision objective.",
    )
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--validation-samples-per-task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--run-contract-sha256",
        default=None,
        help="Immutable launcher job identity embedded in every checkpoint and checked on resume",
    )
    args = parser.parse_args()
    if args.resume is not None and args.finetune_from is not None:
        parser.error("--resume and --finetune-from are mutually exclusive")
    config = yaml.safe_load(args.config.read_text())
    if args.lr is not None:
        if args.lr <= 0:
            parser.error("--lr must be positive")
        config["lr"] = args.lr
    if args.supervision_mode is not None:
        config["supervision_mode"] = args.supervision_mode
    if args.warmup_steps is not None:
        if args.warmup_steps < 0:
            parser.error("--warmup-steps must be non-negative")
        config["warmup_steps"] = args.warmup_steps
    active_routing = routing_signature(config)
    active_balance = balance_signature(config)
    if args.eval_every is not None:
        config["eval_every"] = args.eval_every
    if args.validation_samples_per_task is not None:
        config["validation_samples_per_task"] = args.validation_samples_per_task
    total_steps = args.steps or config["total_steps"]
    if config["prompt_horizon"] != config["max_episode_steps"] * config["prompt_episode_horizon"]:
        raise ValueError("prompt_horizon does not match the dataset context length")
    if config["batch_size"] % 2:
        raise ValueError("T2MIR contrastive sampling requires an even batch_size")
    supervision_mode = config.get("supervision_mode", "all_tokens")
    if supervision_mode not in {"all_tokens", "final_token"}:
        raise ValueError(f"unsupported supervision_mode: {supervision_mode}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset_provenance = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "label_mode": "unspecified",
    }
    actor_mean_provenance_path = args.dataset_dir / "actor_mean_provenance.json"
    derived_validation_path = args.dataset_dir / "derived_validation_report.json"
    if actor_mean_provenance_path.exists() or derived_validation_path.exists():
        if not actor_mean_provenance_path.is_file() or not derived_validation_path.is_file():
            raise ValueError("derived dataset requires both provenance and validation reports")
        actor_mean_provenance = json.loads(actor_mean_provenance_path.read_text())
        derived_validation = json.loads(derived_validation_path.read_text())
        if actor_mean_provenance.get("status") != "complete":
            raise ValueError("actor-mean provenance is incomplete")
        if derived_validation.get("validation") != "PASS":
            raise ValueError("actor-mean derived validation did not pass")
        provenance_sha256 = sha256_file(actor_mean_provenance_path)
        if derived_validation.get("actor_mean_provenance_sha256") != provenance_sha256:
            raise ValueError("actor-mean provenance changed after derived validation")
        expected_task_ids = sorted(
            set(range(int(config["num_tasks"]))) - set(map(int, config["eval_tasks"]))
        )
        if sorted(map(int, actor_mean_provenance.get("source_task_ids", []))) != expected_task_ids:
            raise ValueError("actor-mean task IDs do not match the configured training split")
        dataset_provenance.update(
            label_mode="deterministic_actor_mean",
            actor_mean_provenance_sha256=provenance_sha256,
            derived_validation_sha256=sha256_file(derived_validation_path),
            parent_dataset_contract_fingerprint=actor_mean_provenance.get(
                "parent_dataset_contract_fingerprint"
            ),
            parent_dataset_file_fingerprint=actor_mean_provenance.get(
                "parent_dataset_file_fingerprint"
            ),
        )
    common = dict(dir_path=str(args.dataset_dir), train_tasks=config["training_tasks"],
                  total_tasks=config["num_tasks"], eval_task_ids=config["eval_tasks"],
                  load_eval_tasks=False)
    data, _, indices = data_loader(**common)[0]
    query, _, query_indices = query_data_loader(**common)[0]
    data = task_axis(data, indices)
    query = task_axis(query, query_indices)
    # Aligned query rows have the same ordering as dataset rows. Retain the
    # original base-PPO action as a strong, same-state comparison baseline.
    query["behavior_actions"] = data["actions"].copy()
    train_query, validation = split_queries(query, config["validation_fraction"], args.seed)
    dataset = DPT_Dataset(data, train_query, config, state_norm=True)
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.get("enforce_unique_contrastive_tasks", False):
        pairs_per_micro_batch = config["batch_size"] // 2
        if pairs_per_micro_batch > dataset.num_tasks:
            raise ValueError(
                f"contrastive micro-batch needs {pairs_per_micro_batch} unique tasks, "
                f"but the dataset only has {dataset.num_tasks}; reduce batch_size"
            )
    mean, std = dataset.get_norm_params()
    validation["states"] = (validation["states"] - mean) / std
    action_mean = torch.from_numpy(train_query["actions"].reshape(-1, train_query["actions"].shape[-1]).mean(0)).float().to(device)

    policy = DPTTransformerMOE(mean.shape[0], train_query["actions"].shape[-1], config,
                               action_tanh=False, discrete_environment=False).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    print(f"trainable parameters: {trainable_parameters:,}", flush=True)
    optimizer = AdamW(policy.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    warmup_steps = int(config.get("warmup_steps", 0))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / max(warmup_steps, 1), 1.0)
        if warmup_steps > 0 else 1.0,
    )
    start_step = 0
    resumed_best_loss = float("inf")
    resumed_stale_evaluations = 0
    resumed_rng_state = None
    initialization_provenance = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        assert_resume_routing_compatible(checkpoint, config)
        assert_resume_balance_compatible(checkpoint, config)
        assert_resume_contract_compatible(checkpoint, args.run_contract_sha256)
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        resumed_best_loss = float(checkpoint.get("best_validation_mse", float("inf")))
        resumed_stale_evaluations = int(checkpoint.get("stale_evaluations", 0))
        resumed_rng_state = checkpoint.get("rng_state")
        initialization_provenance = checkpoint.get("initialization_provenance")
        if start_step >= total_steps:
            raise ValueError(f"checkpoint step {start_step} already reached requested total_steps {total_steps}")
        print(f"resuming from {args.resume} at step={start_step}; target step={total_steps}", flush=True)
    elif args.finetune_from is not None:
        checkpoint = torch.load(args.finetune_from, map_location=device)
        assert_resume_routing_compatible(checkpoint, config)
        assert_resume_balance_compatible(checkpoint, config)
        if int(checkpoint.get("state_dim", -1)) != int(mean.shape[0]):
            raise ValueError("finetune checkpoint state dimension does not match dataset")
        if int(checkpoint.get("action_dim", -1)) != int(policy.action_dim):
            raise ValueError("finetune checkpoint action dimension does not match dataset")
        checkpoint_mean = checkpoint.get("state_mean")
        checkpoint_std = checkpoint.get("state_std")
        if checkpoint_mean is None or checkpoint_std is None:
            raise ValueError("finetune checkpoint is missing state normalization")
        if not np.array_equal(checkpoint_mean.cpu().numpy(), mean):
            raise ValueError("finetune checkpoint state mean differs from derived dataset")
        if not np.array_equal(checkpoint_std.cpu().numpy(), std):
            raise ValueError("finetune checkpoint state std differs from derived dataset")
        policy.load_state_dict(checkpoint["policy"])
        initialization_provenance = {
            "mode": "policy_only_finetune",
            "checkpoint": str(args.finetune_from.resolve()),
            "checkpoint_sha256": sha256_file(args.finetune_from),
            "source_step": int(checkpoint.get("step", -1)),
            "source_best_validation_mse": float(
                checkpoint.get("best_validation_mse", float("nan"))
            ),
            "optimizer_scheduler_rng_reset": True,
        }
        print(
            f"finetune initialization from {args.finetune_from}; "
            f"new_steps={total_steps} lr={config['lr']} optimizer=reset",
            flush=True,
        )
    routes = RouteCollector(policy, enabled=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finetune_from is not None and (args.output_dir / "metrics.csv").exists():
        raise FileExistsError(
            f"finetune output already contains metrics.csv: {args.output_dir}"
        )
    (args.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (args.output_dir / "routing_signature.json").write_text(
        json.dumps(active_routing, indent=2)
    )
    (args.output_dir / "balance_signature.json").write_text(
        json.dumps(active_balance, indent=2)
    )
    (args.output_dir / "dataset_provenance.json").write_text(
        json.dumps(dataset_provenance, indent=2) + "\n"
    )
    print(f"routing: {json.dumps(active_routing, sort_keys=True)}", flush=True)
    print(f"balance: {json.dumps(active_balance, sort_keys=True)}", flush=True)
    metrics_path = args.output_dir / "metrics.csv"
    append_metrics = args.resume is not None and metrics_path.exists()
    metrics_file = metrics_path.open("a" if append_metrics else "w", newline="")
    metric_fieldnames = [
        "step", "train_prediction_mse", "train_balance_loss",
        "train_contrastive_loss", "total_loss", "validation_mse",
        "mean_action_baseline_mse", "base_ppo_action_mse",
    ]
    if append_metrics:
        # Preserve the six-column layout of checkpoints created before the
        # auxiliary-loss diagnostics were added; never append malformed rows.
        with metrics_path.open(newline="") as existing_metrics:
            existing_fieldnames = csv.DictReader(existing_metrics).fieldnames
        if existing_fieldnames:
            metric_fieldnames = existing_fieldnames
    writer = csv.DictWriter(
        metrics_file,
        fieldnames=metric_fieldnames,
    )
    if not append_metrics:
        writer.writeheader()
    best_loss, stale_evaluations = resumed_best_loss, resumed_stale_evaluations
    last_train_mse = last_balance_loss = last_contrastive_loss = last_total_loss = float("nan")
    if resumed_rng_state is not None:
        restore_rng_state(resumed_rng_state)

    for step in range(start_step + 1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_prediction_loss = 0.0
        accumulated_balance_loss = 0.0
        accumulated_contrastive_loss = 0.0
        accumulated_total_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            prompt, target = dataset.sample_batch_contrastive(config["batch_size"])
            ps, pa, pr = [torch.from_numpy(x).float().to(device) for x in prompt]
            qs, qa = [torch.from_numpy(x).float().to(device) for x in target]
            predictions, balance_loss, contrastive_loss = policy(ps, pa, pr, qs)
            if supervision_mode == "final_token":
                # This is the same output consumed by policy(..., eval=True).  It has
                # causal access to the query token and the complete prompt history.
                prediction_loss = F.mse_loss(predictions[:, -1], qa[:, 0])
            else:
                # Original T2MIR objective: every state-position predicts the same
                # query action, including positions with only a partial prompt.
                prediction_loss = F.mse_loss(predictions, qa.repeat(1, predictions.shape[1], 1))
            loss = prediction_loss + balance_loss + contrastive_loss
            (loss / gradient_accumulation_steps).backward()
            accumulated_prediction_loss += prediction_loss.item()
            accumulated_balance_loss += balance_loss.item()
            accumulated_contrastive_loss += contrastive_loss.item()
            accumulated_total_loss += loss.item()
        clip_grad_norm_(policy.parameters(), config["max_grad_norm"]); optimizer.step(); scheduler.step()
        policy.update_target_network()
        last_train_mse = accumulated_prediction_loss / gradient_accumulation_steps
        last_balance_loss = accumulated_balance_loss / gradient_accumulation_steps
        last_contrastive_loss = accumulated_contrastive_loss / gradient_accumulation_steps
        last_total_loss = accumulated_total_loss / gradient_accumulation_steps

        checkpoint_due = step % config["checkpoint_every"] == 0
        evaluation_due = step == 1 or step % config["eval_every"] == 0
        if checkpoint_due and not evaluation_due:
            save_checkpoint(args.output_dir / "checkpoints" / f"policy_{step}.pt", policy, optimizer, scheduler,
                            config, mean, std, step, best_loss, stale_evaluations,
                            args.run_contract_sha256, initialization_provenance,
                            dataset_provenance)
        if evaluation_due:
            val_mse, task_mse, baseline, baseline_tasks, base_mse, base_tasks, route_stats = validate(
                policy, dataset, validation, config, device, action_mean, routes, args.seed + 10_000)
            metric_row = {
                "step": step,
                "train_prediction_mse": last_train_mse,
                "train_balance_loss": last_balance_loss,
                "train_contrastive_loss": last_contrastive_loss,
                "total_loss": last_total_loss,
                "validation_mse": val_mse,
                "mean_action_baseline_mse": baseline,
                "base_ppo_action_mse": base_mse,
            }
            writer.writerow({key: metric_row[key] for key in metric_fieldnames})
            metrics_file.flush()
            report = {"step": step, "supervision_mode": supervision_mode,
                      "validation_mse": val_mse, "validation_mse_by_task": task_mse,
                      "mean_action_baseline_mse": baseline, "baseline_mse_by_task": baseline_tasks,
                      "base_ppo_action_mse": base_mse, "base_ppo_action_mse_by_task": base_tasks,
                      "routing_signature": active_routing,
                      "balance_signature": active_balance,
                      "routes_by_task": route_stats}
            (args.output_dir / "latest_validation.json").write_text(json.dumps(report, indent=2))
            print(f"step={step:05d} train_mse={last_train_mse:.6f} "
                  f"balance={last_balance_loss:.6f} contrastive={last_contrastive_loss:.6f} "
                  f"val_mse={val_mse:.6f} "
                  f"mean_baseline={baseline:.6f} base_ppo={base_mse:.6f}", flush=True)
            should_stop = False
            if val_mse < best_loss - config["early_stopping_min_delta"]:
                best_loss, stale_evaluations = val_mse, 0
                save_checkpoint(
                    args.output_dir / "best.pt", policy, optimizer, scheduler, config,
                    mean, std, step, best_loss, stale_evaluations,
                    args.run_contract_sha256, initialization_provenance,
                    dataset_provenance,
                )
            else:
                stale_evaluations += 1
                if stale_evaluations >= config["early_stopping_patience"]:
                    print(f"early stopping at step={step}; best_validation_mse={best_loss:.6f}")
                    should_stop = True
            # Periodic checkpoints used for resume must include the validation
            # decision made at this same step.  Rewriting the just-created file
            # is cheap relative to validation and makes interrupted continuation
            # equivalent to the uninterrupted RNG/early-stop trajectory.
            if checkpoint_due:
                save_checkpoint(
                    args.output_dir / "checkpoints" / f"policy_{step}.pt",
                    policy, optimizer, scheduler, config, mean, std, step, best_loss,
                    stale_evaluations, args.run_contract_sha256,
                    initialization_provenance,
                    dataset_provenance,
                )
            if should_stop:
                break
    metrics_file.close()
    print(f"best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
