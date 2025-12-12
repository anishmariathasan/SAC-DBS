"""
Evaluate a trained SAC DBS controller and visualize its behavior.

Runs a single episode of the Kuramoto tremor environment with deterministic
policy actions, capturing neural synchrony (R), stimulation voltage, and
medication level to illustrate closed-loop adaptation as medication wears off.
"""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from train_agent import build_env


def run_episode(model: SAC, steps: int = 500) -> Tuple[List[float], List[float], List[float]]:
    """Execute one deterministic episode and collect trajectories."""
    env = DummyVecEnv([build_env])
    obs = env.reset()

    r_hist: List[float] = []
    v_hist: List[float] = []
    med_hist: List[float] = []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)

        info = infos[0]
        r_hist.append(float(info["r"]))
        v_hist.append(float(info["voltage"]))
        med_hist.append(float(info["medication_level"]))

        if done[0]:
            break

    env.close()
    return r_hist, v_hist, med_hist


def plot_results(r_hist: List[float], v_hist: List[float], med_hist: List[float]) -> None:
    """Create a dual-axis plot of tremor severity, voltage, and medication level."""
    steps = np.arange(len(r_hist))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_r = "tab:red"
    ax1.set_xlabel("Time steps")
    ax1.set_ylabel("Tremor Severity (R)", color=color_r)
    ax1.plot(steps, r_hist, color=color_r, label="Order Parameter R")
    ax1.tick_params(axis="y", labelcolor=color_r)
    ax1.axhline(0.3, color="tab:gray", linestyle=":", linewidth=1, label="Healthy threshold")

    ax2 = ax1.twinx()
    color_v = "tab:blue"
    ax2.set_ylabel("Stimulation Voltage (V)", color=color_v)
    ax2.plot(steps, v_hist, color=color_v, label="Voltage")
    ax2.tick_params(axis="y", labelcolor=color_v)

    ax2.plot(steps, med_hist, color="tab:gray", linestyle="--", linewidth=1.5, label="Medication Level")

    fig.suptitle("Adaptive DBS Control: SAC Agent Response to Medication Wearing-Off")

    # Merge legends from both axes.
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig("results_plot.png", dpi=300)
    print("Saved evaluation plot to results_plot.png")
    plt.close(fig)


def main() -> None:
    """Load the trained model, run evaluation, and plot the results."""
    model = SAC.load("sac_dbs_model.zip")
    r_hist, v_hist, med_hist = run_episode(model, steps=500)
    plot_results(r_hist, v_hist, med_hist)


if __name__ == "__main__":
    main()
