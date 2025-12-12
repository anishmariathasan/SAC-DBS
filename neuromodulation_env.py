"""
Custom Gymnasium environment modeling tremor and closed-loop DBS control.

The environment implements a Kuramoto network of coupled neural oscillators
whose synchrony (order parameter R) reflects tremor severity in Parkinson's
disease. An agent modulates deep brain stimulation voltage to desynchronize
pathological oscillations while accounting for a time-varying medication level
that gradually wears off, increasing the effective coupling strength.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class KuramotoParams:
    """
    Parameter bundle for the Kuramoto-based tremor simulator.

    Attributes
    ----------
    n_oscillators : int
        Number of neural oscillators representing a local circuit.
    base_coupling : float
        Baseline coupling strength when medication is fully effective.
    worsening_rate : float
        Incremental coupling added as medication wears off (0 -> 1).
    dt : float
        Integration time step (seconds).
    stim_gain : float
        Gain translating applied voltage into a desynchronizing torque.
    noise_scale : float
        Standard deviation of the phase noise term per step.
    natural_freq_hz : float
        Central natural frequency (Hz) of oscillators (aligned to tremor band).
    """

    n_oscillators: int = 100
    base_coupling: float = 0.5
    worsening_rate: float = 2.5
    dt: float = 0.01
    stim_gain: float = 0.8
    noise_scale: float = 0.05
    natural_freq_hz: float = 5.0


class KuramotoTremorEnv(gym.Env):
    """
    Gymnasium environment for closed-loop DBS using a Kuramoto tremor model.

    Observation space
    -----------------
    Box([0, -pi, 0], [1, pi, 1]) -> (R, psi, medication_level)
        R    : Order parameter magnitude (synchrony / tremor severity).
        psi  : Mean phase of the neural population.
        medication_level: Linear decay from 1 -> 0 over the episode.

    Action space
    ------------
    Box([0.0], [5.0]) representing stimulation voltage (V).

    Dynamics
    --------
    dtheta/dt = omega + K(t) * R * sin(psi - theta) - stim_gain * V * sin(theta - psi) + noise
    where K(t) increases as medication wanes, driving synchrony higher if unopposed.

    Reward
    ------
    reward = -(R**2) - lambda * (V**2) + bonus (if R < 0.3)
    The squared penalty discourages residual tremor and excessive power use.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        params: Optional[KuramotoParams] = None,
        episode_length: int = 500,
        voltage_penalty: float = 0.05,
        bonus_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.params = params or KuramotoParams()
        self.episode_length = episode_length
        self.voltage_penalty = voltage_penalty
        self.bonus_threshold = bonus_threshold

        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([5.0], dtype=np.float32))
        obs_low = np.array([0.0, -np.pi, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, np.pi, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._rng = np.random.default_rng()
        self._t = 0
        self._thetas = np.zeros(self.params.n_oscillators, dtype=np.float64)
        self._natural_freqs = self._init_frequencies()

    def _init_frequencies(self) -> np.ndarray:
        """Initialize natural frequencies near the tremor band (Hz -> rad/s)."""
        center = self.params.natural_freq_hz
        spread = 0.3 * center
        freqs_hz = self._rng.normal(loc=center, scale=spread, size=self.params.n_oscillators)
        return 2 * np.pi * freqs_hz

    def _order_parameter(self) -> Tuple[float, float]:
        """Compute Kuramoto order parameter magnitude R and mean phase psi."""
        complex_order = np.mean(np.exp(1j * self._thetas))
        r = np.abs(complex_order)
        psi = np.angle(complex_order)
        return float(r), float(psi)

    def _medication_level(self) -> float:
        """Linearly decaying medication level over the episode."""
        progress = self._t / max(1, self.episode_length)
        return float(max(0.0, 1.0 - progress))

    def _effective_coupling(self, medication_level: float) -> float:
        """Medication loss increases coupling, worsening tremor."""
        return self.params.base_coupling + self.params.worsening_rate * (1.0 - medication_level)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._thetas = self._rng.uniform(low=-np.pi, high=np.pi, size=self.params.n_oscillators)
        self._natural_freqs = self._init_frequencies()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        r, psi = self._order_parameter()
        med = self._medication_level()
        return np.array([r, psi, med], dtype=np.float32)

    def step(self, action):
        voltage = float(np.clip(action, self.action_space.low, self.action_space.high))

        r, psi = self._order_parameter()
        med = self._medication_level()
        coupling = self._effective_coupling(med)

        # Phase update via Euler-Maruyama integration of the Kuramoto model.
        coupling_term = coupling * r * np.sin(psi - self._thetas)
        stimulation_term = -self.params.stim_gain * voltage * np.sin(self._thetas - psi)
        noise = self.params.noise_scale * self._rng.normal(size=self.params.n_oscillators)

        dtheta = (self._natural_freqs + coupling_term + stimulation_term + noise) * self.params.dt
        self._thetas = (self._thetas + dtheta + np.pi) % (2 * np.pi) - np.pi

        self._t += 1
        new_r, new_psi = self._order_parameter()
        obs = np.array([new_r, new_psi, self._medication_level()], dtype=np.float32)

        reward = -(new_r ** 2) - self.voltage_penalty * (voltage ** 2)
        if new_r < self.bonus_threshold:
            reward += 1.0

        terminated = False
        truncated = self._t >= self.episode_length
        info = {"r": new_r, "psi": new_psi, "medication_level": obs[2], "voltage": voltage}
        return obs, reward, terminated, truncated, info

    def render(self):
        r, psi = self._order_parameter()
        print(f"t={self._t}, R={r:.3f}, psi={psi:.2f}, med={self._medication_level():.2f}")

    def close(self):
        return None


__all__ = ["KuramotoParams", "KuramotoTremorEnv"]
