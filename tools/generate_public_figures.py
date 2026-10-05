"""Generate the three publication-ready figures used by the local release.

The script reads only frozen JSON result artifacts and writes deterministic PNG
files.  It deliberately avoids loading checkpoints, datasets, Isaac Sim, or
MuJoCo so the figures can be regenerated on a CPU-only machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "artifacts/results/final_metrics.json").is_file():
    # Running from the self-contained public release.
    DEFAULT_METRICS = ROOT / "artifacts/results/final_metrics.json"
    DEFAULT_OUTPUT = ROOT / "artifacts/figures"
else:
    # Running from the research workspace that builds the release.
    DEFAULT_METRICS = (
        ROOT / "public_release/robotlab-t2mir-demo/artifacts/results/final_metrics.json"
    )
    DEFAULT_OUTPUT = ROOT / "release_assets/artifacts/figures"

BG = "#F6F8FC"
INK = "#172033"
MUTED = "#61708A"
GRID = "#DCE3EE"
NAVY = "#173B67"
BLUE = "#2878D0"
CYAN = "#29A3B1"
ORANGE = "#F08A4B"
GREEN = "#26A269"
RED = "#D95C5C"
PURPLE = "#7656C8"
LIGHT_BLUE = "#DCEBFA"
LIGHT_ORANGE = "#FCE7D9"
LIGHT_GREEN = "#DDF3E8"
LIGHT_PURPLE = "#EAE4F8"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    edgecolor: str,
    title_color: str = INK,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.025 * width,
        y + 0.68 * height,
        title,
        fontsize=11,
        fontweight="bold",
        color=title_color,
        va="center",
    )
    ax.text(
        x + 0.025 * width,
        y + 0.36 * height,
        body,
        fontsize=8.5,
        color=MUTED,
        va="center",
        linespacing=1.3,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
        )
    )


def architecture_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Dynamic-Routing In-Context Control for Unitree G1",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.02,
        0.89,
        "TRAINING & DATASET",
        fontsize=9,
        fontweight="bold",
        color=BLUE,
        alpha=0.95,
    )
    ax.text(
        0.02,
        0.41,
        "CLOSED-LOOP DEPLOYMENT",
        fontsize=9,
        fontweight="bold",
        color=PURPLE,
        alpha=0.95,
    )
    ax.plot([0.02, 0.98], [0.455, 0.455], color=GRID, lw=1.1)

    # Training and dataset row.
    rounded_box(
        ax,
        0.025,
        0.62,
        0.19,
        0.18,
        "Dynamics grid",
        "lag × motor strength\npayload × friction\n42 train / 6 held-out",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    rounded_box(
        ax,
        0.275,
        0.62,
        0.19,
        0.18,
        "PPO checkpoint bank",
        "Training-process policies\nnon-optimal + expert rollouts\nreward / done preserved",
        facecolor="#E7EEF7",
        edgecolor=NAVY,
    )
    rounded_box(
        ax,
        0.525,
        0.62,
        0.19,
        0.18,
        "Official-style dataset",
        "64-step prompt windows\nindependent query targets\nvalidated provenance",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
    )
    rounded_box(
        ax,
        0.775,
        0.62,
        0.19,
        0.18,
        "T2MIR policy training",
        "Behavior cloning objective\nload-balanced routing\nfrozen best checkpoint",
        facecolor=LIGHT_PURPLE,
        edgecolor=PURPLE,
    )
    for x0, x1 in ((0.215, 0.275), (0.465, 0.525), (0.715, 0.775)):
        arrow(ax, (x0 + 0.006, 0.71), (x1 - 0.006, 0.71))

    # Deployment row.
    rounded_box(
        ax,
        0.025,
        0.145,
        0.19,
        0.18,
        "Goal controller",
        "Random target position + yaw\ncontinuous dynamics drift\nshared evaluation contract",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    rounded_box(
        ax,
        0.275,
        0.145,
        0.19,
        0.18,
        "Online context",
        "(state, action, reward, done)\ncausal block-64 history\nreset-safe streaming",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
    )
    rounded_box(
        ax,
        0.525,
        0.145,
        0.19,
        0.18,
        "Dual dynamic routing",
        "Token MoE: Top-p\nTask MoE: Top-p\n123 observations → 37 actions",
        facecolor=LIGHT_PURPLE,
        edgecolor=PURPLE,
    )
    rounded_box(
        ax,
        0.775,
        0.145,
        0.19,
        0.18,
        "Deployment targets",
        "Isaac Sim closed loop\nMuJoCo paired sim-to-sim\nROS 2 command + diagnostics",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    for x0, x1 in ((0.215, 0.275), (0.465, 0.525), (0.715, 0.775)):
        arrow(ax, (x0 + 0.006, 0.235), (x1 - 0.006, 0.235), color=PURPLE)

    # Policy initialization and closed-loop feedback.
    arrow(
        ax,
        (0.87, 0.61),
        (0.62, 0.335),
        color=PURPLE,
        connectionstyle="arc3,rad=0.18",
    )
    ax.text(
        0.785,
        0.455,
        "frozen policy",
        fontsize=8.5,
        color=PURPLE,
        ha="center",
        bbox={"boxstyle": "round,pad=0.2", "fc": BG, "ec": "none"},
    )
    arrow(
        ax,
        (0.87, 0.135),
        (0.37, 0.135),
        color=CYAN,
        connectionstyle="arc3,rad=-0.18",
        linestyle="--",
    )
    ax.text(
        0.62,
        0.082,
        "closed-loop feedback updates context — model weights stay frozen",
        fontsize=8.5,
        color=MUTED,
        ha="center",
        bbox={"boxstyle": "round,pad=0.2", "fc": BG, "ec": "none"},
    )

    ax.text(
        0.02,
        0.006,
        "DT2MIR: dual Top-p routing • causal online adaptation • identical frozen policy across simulators",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, output / "architecture.png")


def load_metrics(metrics_path: Path) -> dict:
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text())
    release_metrics = (
        ROOT / "public_release/robotlab-t2mir-demo/artifacts/results/final_metrics.json"
    )
    if release_metrics.is_file():
        return json.loads(release_metrics.read_text())
    raise FileNotFoundError(
        f"metrics not found at {metrics_path} or {release_metrics}; build the release first"
    )


def routing_efficiency_data(metrics: dict) -> dict:
    if "routing_efficiency" in metrics:
        return metrics["routing_efficiency"]

    # Bootstrap the first release build from the frozen research reports.
    run_root = (
        ROOT
        / "methods/t2mir/runs/RobotLab-G1-MultiDynamics/"
        "balanced_control"
    )
    reports = {
        "t2mir": run_root / "A_midLB003_seed42/actor_mean/latest_validation.json",
        "dt2mir": run_root / "D_midLB003_seed42/actor_mean/latest_validation.json",
    }
    if not all(path.is_file() for path in reports.values()):
        raise KeyError("routing_efficiency is missing from the metrics artifact")
    expert_parameters = 49_472
    models = {}
    for name, path in reports.items():
        report = json.loads(path.read_text())
        aggregate = {
            "token": {"decisions": 0, "active_sum": 0.0},
            "task": {"decisions": 0, "active_sum": 0.0},
        }
        for gates in report["routes_by_task"].values():
            for stats in gates.values():
                kind = stats["gate_kind"]
                decisions = int(stats["routing_decisions"])
                aggregate[kind]["decisions"] += decisions
                aggregate[kind]["active_sum"] += (
                    decisions * float(stats["mean_active_experts"])
                )
        token = aggregate["token"]["active_sum"] / aggregate["token"]["decisions"]
        task = aggregate["task"]["active_sum"] / aggregate["task"]["decisions"]
        models[name] = {
            "token_mean_active_experts": token,
            "task_mean_active_experts": task,
            "activated_expert_parameters": (token + task) * expert_parameters,
        }
    return {
        "expert_parameters_per_expert": expert_parameters,
        "models": models,
        "dt2mir_reduction_fraction": 1.0
        - models["dt2mir"]["activated_expert_parameters"]
        / models["t2mir"]["activated_expert_parameters"],
    }


def closed_loop_figure(metrics: dict, output: Path) -> None:
    gate = metrics.get(
        "closed_loop_task47_seed42_16x32",
        metrics.get("gate_a_task47_seed42_16x32"),
    )
    if gate is None:
        raise KeyError("closed-loop task47 metrics are missing")
    keys = [
        "base_ppo",
        "query_only_mlp_matched",
        "t2mir_a_midlb003",
        "t2mir_d_midlb003",
    ]
    labels = ["Base PPO", "Query-only MLP", "T2MIR", "DT2MIR"]
    success = np.array([gate[key]["success_rate"] for key in keys]) * 100
    falls = np.array([gate[key]["fall_rate"] for key in keys]) * 100
    adaptation = (
        np.array(
            [gate[key]["adaptation_success_delta_last_minus_first"] for key in keys]
        )
        * 100
    )

    fig = plt.figure(figsize=(12, 6.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.23)
    ax = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    fig.suptitle(
        "Closed-Loop Performance under Held-Out Dynamics",
        x=0.07,
        y=0.97,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.80, bottom=0.12)

    y = np.arange(len(labels))
    h = 0.28
    bars_success = ax.barh(y - h / 2, success, h, color=GREEN, label="Success")
    bars_fall = ax.barh(y + h / 2, falls, h, color=RED, alpha=0.86, label="Fall")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rate (%)")
    ax.set_title("Task outcome", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    for bars in (bars_success, bars_fall):
        for bar in bars:
            value = bar.get_width()
            ax.text(
                min(value + 1.2, 95),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="left" if value < 92 else "right",
                fontsize=9,
                fontweight="bold",
                color=INK,
            )
    ax.axvline(success[0], color=NAVY, lw=1.1, ls="--", alpha=0.75)
    ax.text(
        success[0] + 1,
        -0.46,
        "PPO reference",
        fontsize=8,
        color=NAVY,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "fc": BG, "ec": "none"},
    )

    colors = ["#8D99A8", "#AAB2BD", BLUE, PURPLE]
    bars = ax2.bar(y, adaptation, color=colors, width=0.62)
    ax2.set_ylim(0, 50)
    ax2.set_xticks(y, ["PPO", "MLP", "T2MIR", "DT2MIR"])
    ax2.set_ylabel("Last-round minus first-round success (pp)")
    ax2.set_title("Online adaptation gain", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax2.yaxis.grid(True, color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, adaptation):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"+{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    save(fig, output / "closed_loop_results.png")


def routing_efficiency_figure(metrics: dict, output: Path) -> None:
    routing = routing_efficiency_data(metrics)
    t2mir = routing["models"]["t2mir"]
    dt2mir = routing["models"]["dt2mir"]

    fig = plt.figure(figsize=(11.5, 6.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    fig.suptitle(
        "DT2MIR Activates Fewer Expert Parameters",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.13)

    layers = ["Token MoE", "Task MoE"]
    t2mir_active = np.array(
        [t2mir["token_mean_active_experts"], t2mir["task_mean_active_experts"]]
    )
    dt2mir_active = np.array(
        [dt2mir["token_mean_active_experts"], dt2mir["task_mean_active_experts"]]
    )
    x = np.arange(len(layers))
    width = 0.34
    b1 = ax.bar(x - width / 2, t2mir_active, width, color=BLUE, label="T2MIR")
    b2 = ax.bar(x + width / 2, dt2mir_active, width, color=PURPLE, label="DT2MIR")
    ax.set_xticks(x, layers)
    ax.set_ylim(0, 2.35)
    ax.set_ylabel("Mean active experts per decision")
    ax.set_title("Expert activation", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.045,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    parameter_values = np.array(
        [
            t2mir["activated_expert_parameters"],
            dt2mir["activated_expert_parameters"],
        ]
    ) / 1000.0
    bars = ax2.bar([0, 1], parameter_values, color=[BLUE, PURPLE], width=0.58)
    ax2.set_xticks([0, 1], ["T2MIR", "DT2MIR"])
    ax2.set_ylim(0, 220)
    ax2.set_ylabel("Activated expert parameters (K)")
    ax2.set_title(
        "Logical parameter activation",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax2.yaxis.grid(True, color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, parameter_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value:.1f}K",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    reduction = routing["dt2mir_reduction_fraction"] * 100
    ax2.text(
        0.5,
        0.52,
        f"−{reduction:.1f}%",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=PURPLE,
    )
    save(fig, output / "routing_efficiency.png")


def sim2sim_figure(metrics: dict, output: Path) -> None:
    cross = metrics["cross_sim_paired"]
    sets = cross["per_scenario_set"]
    labels = ["Seed 42\n64 goals", "Seed 43\n64 goals", "Pooled\n128 goals"]
    isaac = np.array(
        [sets[0]["isaac_sim"]["success_rate"], sets[1]["isaac_sim"]["success_rate"], cross["isaac_sim"]["success_rate"]]
    ) * 100
    mujoco = np.array(
        [sets[0]["mujoco"]["success_rate"], sets[1]["mujoco"]["success_rate"], cross["mujoco"]["success_rate"]]
    ) * 100
    isaac_ci = np.array(cross["isaac_sim"]["success_rate_ci95"]) * 100
    mujoco_ci = np.array(cross["mujoco"]["success_rate_ci95"]) * 100

    fig = plt.figure(figsize=(12, 6.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 0.85], wspace=0.25)
    ax = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    fig.suptitle(
        "Paired Sim-to-Sim Transfer — Same Goals, Same Frozen Policy",
        x=0.07,
        y=0.97,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.12)

    x = np.arange(3)
    width = 0.34
    b1 = ax.bar(x - width / 2, isaac, width, color=BLUE, label="Isaac Sim")
    b2 = ax.bar(x + width / 2, mujoco, width, color=ORANGE, label="MuJoCo")
    ax.errorbar(
        x[2] - width / 2,
        isaac[2],
        yerr=[[isaac[2] - isaac_ci[0]], [isaac_ci[1] - isaac[2]]],
        fmt="none",
        ecolor=INK,
        elinewidth=1.3,
        capsize=4,
    )
    ax.errorbar(
        x[2] + width / 2,
        mujoco[2],
        yerr=[[mujoco[2] - mujoco_ci[0]], [mujoco_ci[1] - mujoco[2]]],
        fmt="none",
        ecolor=INK,
        elinewidth=1.3,
        capsize=4,
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Goal success rate (%)")
    ax.set_title("Per-set and pooled transfer", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    fall_values = np.array(
        [cross["isaac_sim"]["fall_rate"], cross["mujoco"]["fall_rate"]]
    ) * 100
    fall_bars = ax2.bar([0, 1], fall_values, color=[BLUE, ORANGE], width=0.58)
    ax2.set_xticks([0, 1], ["Isaac Sim", "MuJoCo"])
    ax2.set_ylim(0, 12)
    ax2.set_ylabel("Pooled fall rate (%)")
    ax2.set_title("Safety outcome", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax2.yaxis.grid(True, color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    for bar, value in zip(fall_bars, fall_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    save(fig, output / "sim2sim_results.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    metrics = load_metrics(args.metrics.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    architecture_figure(output)
    closed_loop_figure(metrics, output)
    routing_efficiency_figure(metrics, output)
    sim2sim_figure(metrics, output)
    print(f"[FIGURES] PASS output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
