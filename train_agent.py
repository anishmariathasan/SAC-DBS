"""
Train a Soft Actor-Critic (SAC) agent to deliver closed-loop DBS.

The agent observes neural synchrony (order parameter R), mean phase, and
medication level derived from a Kuramoto tremor model, and outputs a
continuous stimulation voltage to suppress pathological oscillations.

Training logs are saved via Monitor files and a training curve is rendered
to `training_curve.png` for quick visual inspection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, make_vec_env

from neuromodulation_env import KuramotoParams, KuramotoTremorEnv
from patient_loader import PatientLoader


def build_env() -> gym.Env:
    """
    Construct the tremor environment using patient-inspired parameters.

    Returns
    -------
    gym.Env
        A configured KuramotoTremorEnv instance ready for vectorization.
    """
    loader = PatientLoader()
    patient_data: Dict[str, float | np.ndarray] = loader.load_patient_data()

    natural_freq = float(patient_data["dominant_frequency"])
    noise_var = float(patient_data["noise_variance"])

    params = KuramotoParams(
        natural_freq_hz=natural_freq,
        noise_scale=0.1 * np.sqrt(noise_var),
    )
    env = KuramotoTremorEnv(params=params, episode_length=500)
    return env


def main() -> None:
    """Train the SAC agent, save the policy, and plot training curves."""
    log_dir = Path("logs") / "sac"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use VecMonitor to capture per-episode rewards for plotting.
    vec_env = make_vec_env(build_env, n_envs=1, monitor_dir=str(log_dir))
    vec_env.seed(42)

    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(log_dir / "tb"))
    model.learn(total_timesteps=15_000, progress_bar=True)
    model.save("sac_dbs_model.zip")

    plot_training_curve(log_dir, Path("training_curve.png"))

    vec_env.close()


def plot_training_curve(log_dir: Path, output_path: Path) -> None:
    """Plot rolling episode rewards from Monitor logs."""
    try:
        results = load_results(str(log_dir))
    except Exception:
        return

    if results.empty:
        return

    timesteps = results["l"].cumsum()
    rewards = results["r"]
    rolling = rewards.rolling(window=20, min_periods=1).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, rewards, alpha=0.3, label="Episode reward")
    plt.plot(timesteps, rolling, color="tab:blue", label="Rolling mean (20)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title("SAC Training Curve (DBS)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
