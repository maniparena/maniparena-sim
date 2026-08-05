"""Replay loop functions for state / joint / ee modes."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
)

from maniparena_sim.terms.replay.reader import ReplayData
from maniparena_sim.utils.math_utils import euler_xyz_to_quat_xyzw


def warmup_rtx_cameras(env: Any, extra_renders: int = 2):
    """Render extra frames to initialise RTX sensors."""
    sim = getattr(env, 'sim', None)
    if sim is None or not sim.has_rtx_sensors():
        return
    for _ in range(max(extra_renders, 0)):
        sim.render()


def restore_initial_state(env: Any, initial_state):
    """Restore recorded initial state and warm up sensors."""
    env.reset_to(
        initial_state,
        torch.tensor([0], device=env.device),
        is_relative=True,
    )
    warmup_rtx_cameras(env)


def get_ee_poses(env: Any):
    """Return (left_pos, left_quat, right_pos, right_quat)."""
    left = env.scene['left_ee_frame']
    right = env.scene['right_ee_frame']
    lp = left.data.target_pos_w[0, 0, :].clone()
    lq = left.data.target_quat_w[0, 0, :].clone()
    rp = right.data.target_pos_w[0, 0, :].clone()
    rq = right.data.target_quat_w[0, 0, :].clone()
    return lp, lq, rp, rq


def compute_tracking_delta(
    cur_pos, cur_quat,
    tgt_pos, tgt_quat,
    pos_gain: float = 1.0,
    rot_gain: float = 1.0,
):
    """Compute 6D (pos+rot) tracking delta."""
    d_pos = (tgt_pos - cur_pos) * pos_gain
    quat_err = quat_mul(
        tgt_quat.unsqueeze(0),
        quat_conjugate(cur_quat.unsqueeze(0)),
    )
    d_rot = axis_angle_from_quat(quat_err)[0] * rot_gain
    return torch.cat([d_pos, d_rot])


def run_state_replay(
    env: Any,
    replay_data: ReplayData,
    simulation_app,
) -> int:
    """Exact state replay with env.step for recorder capture."""
    env_id = torch.tensor([0], device=env.device)
    step_count = 0
    started = False

    tm = getattr(env, 'termination_manager', None)
    original_compute = None
    if tm is not None:
        original_compute = tm.compute

        def _noop():
            return torch.zeros(
                env.num_envs, dtype=torch.bool,
                device=env.device,
            )

        tm.compute = _noop

    try:
        with contextlib.suppress(KeyboardInterrupt):
            with torch.no_grad():
                while (
                    simulation_app.is_running()
                    and not simulation_app.is_exiting()
                ):
                    ed = replay_data.episode_data
                    action = ed.get_next_action()
                    if action is None:
                        break
                    state = ed.get_next_state()
                    if state is not None:
                        env.scene.reset_to(
                            state, env_ids=env_id,
                            is_relative=True,
                        )
                    if not started:
                        at = (
                            action
                            if isinstance(action, torch.Tensor)
                            else torch.as_tensor(action)
                        )
                        if torch.max(torch.abs(at)).item() < 1e-6:
                            continue
                        started = True
                    env.scene.write_data_to_sim()
                    ab = torch.zeros(
                        env.action_space.shape,
                        device=env.device,
                    )
                    at = (
                        action.to(
                            device=env.device,
                            dtype=torch.float32,
                        )
                        if isinstance(action, torch.Tensor)
                        else torch.as_tensor(
                            action,
                            device=env.device,
                            dtype=torch.float32,
                        )
                    )
                    if at.ndim == 1:
                        n = min(at.shape[0], ab.shape[-1])
                        ab[0, :n] = at[:n]
                    elif at.ndim == 2 and at.shape[0] > 0:
                        n = min(at.shape[-1], ab.shape[-1])
                        ab[0, :n] = at[0, :n]
                    env.step(ab)
                    step_count += 1
    finally:
        if original_compute is not None:
            tm.compute = original_compute
    return step_count


def run_joint_replay(
    env: Any,
    joint_sequence,
    simulation_app,
) -> int:
    """Replay 14D joint targets as absolute joint commands."""
    actions = torch.tensor(
        joint_sequence, device=env.device,
    )
    step_count = 0
    with contextlib.suppress(KeyboardInterrupt):
        with torch.no_grad():
            for i in range(actions.shape[0]):
                if (
                    not simulation_app.is_running()
                    or simulation_app.is_exiting()
                ):
                    break
                ab = torch.zeros(
                    env.action_space.shape,
                    device=env.device,
                )
                ab[0] = actions[i]
                env.step(ab)
                step_count += 1
    return step_count


def run_ee_replay(
    env: Any,
    ee_sequence,
    simulation_app,
    pos_gain: float = 1.0,
    rot_gain: float = 1.0,
) -> int:
    """Replay 14D EE actions via closed-loop tracking."""
    import numpy as np

    seq = torch.tensor(
        ee_sequence, dtype=torch.float32,
        device=env.device,
    )
    (
        l_init_pos, l_init_quat,
        r_init_pos, r_init_quat,
    ) = get_ee_poses(env)

    l_rel_q = torch.tensor(
        euler_xyz_to_quat_xyzw(
            np.asarray(ee_sequence[:, 3:6]),
        ),
        dtype=torch.float32, device=env.device,
    )
    r_rel_q = torch.tensor(
        euler_xyz_to_quat_xyzw(
            np.asarray(ee_sequence[:, 10:13]),
        ),
        dtype=torch.float32, device=env.device,
    )

    step_count = 0
    with contextlib.suppress(KeyboardInterrupt):
        with torch.no_grad():
            for i in range(seq.shape[0]):
                if (
                    not simulation_app.is_running()
                    or simulation_app.is_exiting()
                ):
                    break

                tgt_lp = l_init_pos + seq[i, 0:3]
                tgt_rp = r_init_pos + seq[i, 7:10]
                tgt_lq = quat_mul(
                    l_rel_q[i:i + 1],
                    l_init_quat.unsqueeze(0),
                )[0]
                tgt_rq = quat_mul(
                    r_rel_q[i:i + 1],
                    r_init_quat.unsqueeze(0),
                )[0]

                clp, clq, crp, crq = get_ee_poses(env)
                ld = compute_tracking_delta(
                    clp, clq, tgt_lp, tgt_lq,
                    pos_gain, rot_gain,
                )
                rd = compute_tracking_delta(
                    crp, crq, tgt_rp, tgt_rq,
                    pos_gain, rot_gain,
                )

                ab = torch.zeros(
                    env.action_space.shape,
                    device=env.device,
                )
                ab[0, 0:6] = ld
                ab[0, 6] = seq[i, 6]
                ab[0, 7:13] = rd
                ab[0, 13] = seq[i, 13]
                env.step(ab)
                step_count += 1
    return step_count
