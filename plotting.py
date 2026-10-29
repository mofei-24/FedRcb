from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .runner import configuration_fingerprint, load_config


def plot(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    config = load_config(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "fedrcb_summary_v1" or summary.get("configuration_fingerprint") != configuration_fingerprint(config):
        raise ValueError("summary differs from the current configuration; rerun aggregate")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False})
    figure, axes = plt.subplots(1, 2, figsize=(8, 2.7), constrained_layout=True)
    for condition in summary["conditions"]:
        trajectory = condition["trajectory"]
        rounds = np.asarray([0, *trajectory["round"]], dtype=float)
        label = "FedRCB" if config.dataset == "jester" else f"h = {condition['heterogeneity']:g}"
        regret = np.asarray([0, *trajectory["group_cumulative_regret_mean"]])
        deviation = np.asarray([0, *trajectory["group_cumulative_regret_std"]])
        line, = axes[0].plot(rounds, regret, label=label)
        axes[0].fill_between(rounds, np.maximum(0, regret - deviation), regret + deviation, alpha=0.15, color=line.get_color())
        bits = np.asarray([0, *trajectory["communication_bits_mean"]]) / 1000
        axes[1].plot(rounds, bits, color=line.get_color(), label=label)
    for axis in axes:
        axis.set_xlabel("Round")
        axis.set_xlim(left=-config.horizon * 0.02, right=config.horizon * 1.02)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper left")
    axes[0].set_ylabel("Cumulative regret per client")
    axes[1].set_ylabel("Payload (kbit/client)")
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / "fedrcb.png", destination / "fedrcb.pdf"]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths
