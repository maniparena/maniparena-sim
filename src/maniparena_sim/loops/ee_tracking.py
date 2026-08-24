"""Shared EE command vs simulated-follow plotting."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def obs_follow_xyz(obs: Any, key: str) -> np.ndarray | None:
    """Read env-local EE xyz from a policy observation group."""
    import torch

    if not isinstance(obs, dict):
        return None
    policy_obs = obs.get('policy', obs)
    if not isinstance(policy_obs, dict):
        return None
    value = policy_obs.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 1:
        value = value[0]
    if value.size < 3:
        return None
    return value[:3].astype(np.float32)


def cmd_ee_to_xyz(row: np.ndarray) -> np.ndarray:
    """14D [L xyz+rpy+g, R xyz+rpy+g] -> 6D [L xyz, R xyz]."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    return np.concatenate([row[0:3], row[7:10]]).astype(np.float32)


def save_ee_tracking_png(
    path: str,
    cmd,
    act,
    hz: float = 30.0,
    title: str = 'Policy EE command vs simulated EE (env-local xyz)',
) -> str:
    """Overlay commanded vs simulated EE xyz for both arms on one PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmd = np.asarray(cmd, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    t = np.arange(cmd.shape[0], dtype=np.float32) / float(hz)
    labels = ('x (m)', 'y (m)', 'z (m)')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for col, lab in enumerate(labels):
        axes[0, col].plot(t, cmd[:, col], color='C0', lw=1.4, label='cmd')
        axes[0, col].plot(t, act[:, col], color='C1', ls='--', lw=1.2, label='sim')
        axes[0, col].set_title(f'left {lab}')
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(t, cmd[:, 3 + col], color='C0', lw=1.4, label='cmd')
        axes[1, col].plot(t, act[:, 3 + col], color='C1', ls='--', lw=1.2, label='sim')
        axes[1, col].set_title(f'right {lab}')
        axes[1, col].set_xlabel('time (s)')
        axes[1, col].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('left EE')
    axes[1, 0].set_ylabel('right EE')
    axes[0, 2].legend(loc='upper right', fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
