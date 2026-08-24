"""Replay loop functions for state / joint / ee modes."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
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


def _recorded_robot_joint_dim(initial_state) -> int | None:
    try:
        robot_state = initial_state['articulation']['robot']
        return int(robot_state['joint_position'].shape[-1])
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def restore_initial_state(env: Any, initial_state):
    """Restore recorded initial state and warm up sensors."""
    recorded_dof = _recorded_robot_joint_dim(initial_state)
    env_dof = int(env.scene['robot'].num_joints)
    if recorded_dof is not None and recorded_dof != env_dof:
        print(
            f'[replay] skip reset_to: recorded robot has {recorded_dof} '
            f'joints, env has {env_dof}. Using the scene reset instead.'
        )
        warmup_rtx_cameras(env)
        return
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

    action_dim = int(env.action_space.shape[-1])
    robot = env.scene['robot']
    n_frames = int(seq.shape[0])
    print(f'[replay] ee frames={n_frames} action_dim={action_dim}', flush=True)
    step_count = 0
    with contextlib.suppress(KeyboardInterrupt):
        with torch.no_grad():
            for i in range(n_frames):
                if (
                    not simulation_app.is_running()
                    or simulation_app.is_exiting()
                ):
                    print(
                        f'[replay] stop at frame {i}/{n_frames} '
                        '(app exiting)',
                        flush=True,
                    )
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

                ab = torch.zeros(
                    env.action_space.shape,
                    device=env.device,
                )
                if action_dim >= 16:
                    root_pos = robot.data.root_pos_w[0]
                    root_quat = robot.data.root_quat_w[0]
                    l_pos_b, l_quat_b = subtract_frame_transforms(
                        root_pos.unsqueeze(0), root_quat.unsqueeze(0),
                        tgt_lp.unsqueeze(0), tgt_lq.unsqueeze(0),
                    )
                    r_pos_b, r_quat_b = subtract_frame_transforms(
                        root_pos.unsqueeze(0), root_quat.unsqueeze(0),
                        tgt_rp.unsqueeze(0), tgt_rq.unsqueeze(0),
                    )
                    ab[0, 0:3] = l_pos_b[0]
                    ab[0, 3:7] = l_quat_b[0]
                    ab[0, 7] = seq[i, 6]
                    ab[0, 8:11] = r_pos_b[0]
                    ab[0, 11:15] = r_quat_b[0]
                    ab[0, 15] = seq[i, 13]
                else:
                    clp, clq, crp, crq = get_ee_poses(env)
                    ld = compute_tracking_delta(
                        clp, clq, tgt_lp, tgt_lq,
                        pos_gain, rot_gain,
                    )
                    rd = compute_tracking_delta(
                        crp, crq, tgt_rp, tgt_rq,
                        pos_gain, rot_gain,
                    )
                    ab[0, 0:6] = ld
                    ab[0, 6] = seq[i, 6]
                    ab[0, 7:13] = rd
                    ab[0, 13] = seq[i, 13]
                env.step(ab)
                step_count += 1
                if step_count == 1 or step_count % 50 == 0:
                    print(
                        f'[replay] {step_count}/{n_frames}',
                        flush=True,
                    )
    return step_count


def run_model_ee_replay(
    env: Any,
    cmd_ee,
    simulation_app,
    plot_path: str | None = None,
    hz: float = 30.0,
) -> int:
    """Replay saved model EE commands with the same 16D absolute IK as eval."""
    import numpy as np

    from maniparena_sim.embodiment.robots.bimanual import (
        _ARM_EFFORT,
        _ARM_KD,
        _ARM_KP,
    )
    from maniparena_sim.loops.ee_tracking import (
        cmd_ee_to_xyz,
        obs_follow_xyz,
        save_ee_tracking_png,
    )
    from maniparena_sim.policy.robot_policy import follow_pair_to_ik16

    seq = np.asarray(cmd_ee, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] < 14:
        raise ValueError(
            f'model EE cmd must be (T, 14), got {seq.shape}'
        )
    n_frames = int(seq.shape[0])
    robot = env.scene['robot']
    origin = env.scene.env_origins[0].to(
        device=env.device, dtype=torch.float32,
    )
    cmds: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    print(
        f'[replay] model EE frames={n_frames} (16D IK) '
        f'arm Kp={_ARM_KP:g} Kd={_ARM_KD:g} effort={_ARM_EFFORT:g}',
        flush=True,
    )
    step_count = 0
    with contextlib.suppress(KeyboardInterrupt):
        with torch.no_grad():
            for i in range(n_frames):
                if (
                    not simulation_app.is_running()
                    or simulation_app.is_exiting()
                ):
                    print(
                        f'[replay] stop at frame {i}/{n_frames} '
                        '(app exiting)',
                        flush=True,
                    )
                    break
                root_pos = robot.data.root_pos_w[0].to(
                    dtype=torch.float32,
                )
                root_quat = robot.data.root_quat_w[0].to(
                    dtype=torch.float32,
                )
                ik = follow_pair_to_ik16(
                    seq[i, 0:7], seq[i, 7:14],
                    origin, root_pos, root_quat, env.device,
                )
                ab = torch.zeros(
                    env.action_space.shape,
                    device=env.device,
                )
                ab[0] = ik
                obs, _, _, _, _ = env.step(ab)
                left_xyz = obs_follow_xyz(obs, 'follow1_pos')
                right_xyz = obs_follow_xyz(obs, 'follow2_pos')
                if left_xyz is not None and right_xyz is not None:
                    cmds.append(cmd_ee_to_xyz(seq[i]))
                    acts.append(np.concatenate([left_xyz, right_xyz]))
                step_count += 1
                if step_count == 1 or step_count % 50 == 0:
                    print(
                        f'[replay] {step_count}/{n_frames}',
                        flush=True,
                    )
    if cmds:
        cmd_xyz = np.asarray(cmds, dtype=np.float32)
        act_xyz = np.asarray(acts, dtype=np.float32)
        err = act_xyz - cmd_xyz
        rmse_mm = np.sqrt((err * err).mean(axis=0)) * 1000.0
        max_mm = np.abs(err).max(axis=0) * 1000.0
        print(
            '[track] RMSE mm Lxyz '
            f'{rmse_mm[0]:.1f} {rmse_mm[1]:.1f} {rmse_mm[2]:.1f}  '
            f'Rxyz {rmse_mm[3]:.1f} {rmse_mm[4]:.1f} {rmse_mm[5]:.1f}',
            flush=True,
        )
        print(
            '[track] max mm  Lxyz '
            f'{max_mm[0]:.1f} {max_mm[1]:.1f} {max_mm[2]:.1f}  '
            f'Rxyz {max_mm[3]:.1f} {max_mm[4]:.1f} {max_mm[5]:.1f}',
            flush=True,
        )
    if plot_path and cmds:
        save_ee_tracking_png(
            plot_path, cmds, acts, hz=hz,
            title='Replay model EE cmd vs simulated EE (16D IK)',
        )
        print(f'[plot] EE cmd vs sim -> {plot_path}', flush=True)
    return step_count
