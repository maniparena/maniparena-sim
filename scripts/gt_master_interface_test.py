#!/usr/bin/env python3
"""GT master EE action interface test (no VLA / Wall-X model).

Replay LeRobot 14D master actions through the same AbsIK + gripper interface
as closed-loop eval. Target composition matches Ex001WallxPolicy:

    target_pos  = home_pos + relative_pos
    target_quat = quat(relative_euler) * home_quat
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from isaaclab.app import AppLauncher

OBJECT_NAME_MAP = {
    'apple_a': 'apple',
    'banana': 'banana',
    'bread': 'bread',
    'pear': 'pear',
    'basket_b': 'platform_pink',
}
TARGET_OBJECT_NAMES = ('bread', 'apple', 'banana', 'pear')
LEFT_EE_BODY = 'left_arm_gripper_base_link'
RIGHT_EE_BODY = 'right_arm_gripper_base_link'
LEFT_GRIPPER_JOINT = 'left_arm_gripper'
RIGHT_GRIPPER_JOINT = 'right_arm_gripper'
LIFT_JOINT = 'lift_joint'
HEAD_YAW_JOINT = 'head_yaw_joint'
HEAD_PITCH_JOINT = 'head_pitch_joint'
WHEEL_JOINTS = ('left_wheel_joint', 'right_wheel_joint')
MASTER_CSV_COLS = [
    'L_pos_x', 'L_pos_y', 'L_pos_z', 'L_roll', 'L_pitch', 'L_yaw', 'L_gripper',
    'R_pos_x', 'R_pos_y', 'R_pos_z', 'R_roll', 'R_pitch', 'R_yaw', 'R_gripper',
]


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='GT master action interface test for fruits_to_basket.',
    )
    parser.add_argument('--video', action='store_true', default=True)
    parser.add_argument('--no-video', dest='video', action='store_false')
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        '--robot', default='bimanual', choices=['bimanual', 'ex001'],
    )
    parser.add_argument('--task', default='fruits_to_basket')
    parser.add_argument('--config', default='configs/eval/fruits_to_basket.yaml')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--dataset-hz', type=float, default=20.0)
    parser.add_argument('--max-lag-frames', type=int, default=20)
    parser.add_argument('--success-hold-frames', type=int, default=10)
    parser.add_argument('--settle-frames', type=int, default=40)
    parser.add_argument(
        '--lerobot-root',
        default=os.path.expanduser(
            '~/maniparena_output/datasets/fruits_to_basket_0520_ee14',
        ),
    )
    parser.add_argument(
        '--hdf5-root',
        default=os.path.expanduser(
            '~/maniparena_output/datasets/ex001_6r_fruits_to_basket_env',
        ),
    )
    parser.add_argument(
        '--output-root',
        default=os.path.expanduser('~/maniparena_output/gt_interface_test'),
    )
    return parser.parse_args()


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


def resolve_output_dir(args) -> Path:
    root = Path(os.path.expanduser(str(args.output_root)))
    if args.robot == 'bimanual':
        return root / f'episode_{int(args.episode):06d}'
    return root / str(args.robot) / f'episode_{int(args.episode):06d}'


def resolve_hdf5_path(hdf5_root: str, episode: int) -> Path:
    root = Path(os.path.expanduser(hdf5_root))
    name = (
        'ex001_6r_fruits_to_basket_env_master_slave_episode'
        f'{int(episode)}.hdf5'
    )
    path = root / name
    if path.is_file():
        return path
    matches = sorted(root.glob(f'*episode{int(episode)}.hdf5'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'HDF5 episode {episode} not found under {root}')


def load_lerobot_actions(lerobot_root: str, episode: int) -> np.ndarray:
    import pyarrow.parquet as pq

    root = Path(os.path.expanduser(lerobot_root))
    parquet_path = (
        root / 'data' / 'chunk-000' / f'episode_{int(episode):06d}.parquet'
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path, columns=['action'])
    action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f'Expected (N, 14) action, got {action.shape}')
    return action


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    return np.stack(
        [quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=-1,
    ).astype(np.float32)


def load_hdf5_initial_state(hdf5_path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(hdf5_path, 'r') as handle:
        demo = handle['data/demo_0']
        init = demo['initial_state']
        robot = {
            'root_pose_wxyz': np.asarray(
                init['articulation/robot/root_pose'][()], dtype=np.float32,
            ).reshape(1, 7),
            'root_velocity': np.asarray(
                init['articulation/robot/root_velocity'][()], dtype=np.float32,
            ).reshape(1, 6),
            'joint_position': np.asarray(
                init['articulation/robot/joint_position'][()], dtype=np.float32,
            ).reshape(-1),
            'joint_velocity': np.asarray(
                init['articulation/robot/joint_velocity'][()], dtype=np.float32,
            ).reshape(-1),
        }
        objects = {}
        rigid = init['rigid_object']
        for src_name in rigid.keys():
            objects[src_name] = {
                'root_pose_wxyz': np.asarray(
                    rigid[f'{src_name}/root_pose'][()], dtype=np.float32,
                ).reshape(1, 7),
                'root_velocity': np.asarray(
                    rigid[f'{src_name}/root_velocity'][()], dtype=np.float32,
                ).reshape(1, 6),
            }
    return {'robot': robot, 'objects': objects, 'path': str(hdf5_path)}


def _to_torch_f32(value: Any, device=None):
    import torch
    from isaaclab.utils.array import convert_to_torch

    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = None
        try:
            import warp as wp
            tensor = wp.to_torch(value)
        except Exception:
            tensor = None
        if not isinstance(tensor, torch.Tensor):
            try:
                tensor = convert_to_torch(value)
            except Exception:
                raw = value.numpy() if hasattr(value, 'numpy') else value
                tensor = torch.as_tensor(np.asarray(raw))
    tensor = tensor.to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _has_joint(robot, name: str) -> bool:
    try:
        ids, _ = robot.find_joints(name)
        return len(ids) > 0
    except Exception:
        return False


def _read_joint(env, name: str) -> float | None:
    robot = env.scene['robot']
    if not _has_joint(robot, name):
        return None
    ids, _ = robot.find_joints(name)
    joint_pos = _to_torch_f32(robot.data.joint_pos)
    return float(joint_pos[0, ids[0]].item())


def _get_ee_root_local(env, body_name: str):
    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene['robot']
    body_ids, _ = robot.find_bodies(body_name)
    body_idx = int(body_ids[0])
    root_pos = _to_torch_f32(robot.data.root_pos_w)
    root_quat = _to_torch_f32(robot.data.root_quat_w)
    body_pos = _to_torch_f32(robot.data.body_pos_w)[:, body_idx, :]
    body_quat = _to_torch_f32(robot.data.body_quat_w)[:, body_idx, :]
    if root_quat.ndim == 1:
        root_quat = root_quat.reshape(-1, 4)
    if body_quat.ndim == 1:
        body_quat = body_quat.reshape(root_pos.shape[0], 4)
    elif body_quat.ndim == 2 and body_quat.shape[-1] != 4:
        body_quat = body_quat.reshape(root_pos.shape[0], 4)
    return subtract_frame_transforms(root_pos, root_quat, body_pos, body_quat)


def capture_home_refs(env) -> dict[str, Any]:
    l_pos, l_quat = _get_ee_root_local(env, LEFT_EE_BODY)
    r_pos, r_quat = _get_ee_root_local(env, RIGHT_EE_BODY)
    lift = _read_joint(env, LIFT_JOINT)
    return {
        'left_pos': l_pos[0].clone(),
        'left_quat': l_quat[0].clone(),
        'right_pos': r_pos[0].clone(),
        'right_quat': r_quat[0].clone(),
        'lift': lift,
        'head_yaw': _read_joint(env, HEAD_YAW_JOINT),
        'head_pitch': _read_joint(env, HEAD_PITCH_JOINT),
    }


def read_follow_14d(env, home: dict[str, Any]) -> np.ndarray:
    import torch
    from isaaclab.utils.math import (
        euler_xyz_from_quat,
        quat_conjugate,
        quat_mul,
    )

    l_pos, l_quat = _get_ee_root_local(env, LEFT_EE_BODY)
    r_pos, r_quat = _get_ee_root_local(env, RIGHT_EE_BODY)
    l_rel = l_pos[0] - home['left_pos']
    r_rel = r_pos[0] - home['right_pos']
    lift_now = _read_joint(env, LIFT_JOINT)
    if home['lift'] is not None and lift_now is not None:
        dz = float(lift_now) - float(home['lift'])
        l_rel = l_rel.clone()
        r_rel = r_rel.clone()
        l_rel[2] -= dz
        r_rel[2] -= dz
    l_qrel = quat_mul(
        l_quat[0:1], quat_conjugate(home['left_quat'].unsqueeze(0)),
    )
    r_qrel = quat_mul(
        r_quat[0:1], quat_conjugate(home['right_quat'].unsqueeze(0)),
    )
    l_euler = torch.stack(euler_xyz_from_quat(l_qrel), dim=-1)[0]
    r_euler = torch.stack(euler_xyz_from_quat(r_qrel), dim=-1)[0]
    lg = _read_joint(env, LEFT_GRIPPER_JOINT) or 0.0
    rg = _read_joint(env, RIGHT_GRIPPER_JOINT) or 0.0
    follow = torch.cat([
        l_rel, l_euler, torch.tensor([lg], device=l_rel.device),
        r_rel, r_euler, torch.tensor([rg], device=r_rel.device),
    ])
    return follow.detach().cpu().numpy().astype(np.float32)


def master_to_abs_ik(
    master_14d: np.ndarray,
    env,
    home: dict[str, Any],
    action_dim: int,
):
    import torch
    from isaaclab.utils.math import quat_mul

    from maniparena_sim.utils.math_utils import euler_xyz_to_quat_xyzw

    device = env.device
    row = np.asarray(master_14d, dtype=np.float32)
    l_rel_q = torch.as_tensor(
        euler_xyz_to_quat_xyzw(np.asarray([row[3:6]], dtype=np.float32))[0],
        dtype=torch.float32, device=device,
    )
    r_rel_q = torch.as_tensor(
        euler_xyz_to_quat_xyzw(np.asarray([row[10:13]], dtype=np.float32))[0],
        dtype=torch.float32, device=device,
    )
    l_pos = home['left_pos'] + torch.as_tensor(
        row[0:3], dtype=torch.float32, device=device,
    )
    r_pos = home['right_pos'] + torch.as_tensor(
        row[7:10], dtype=torch.float32, device=device,
    )
    l_quat = quat_mul(
        l_rel_q.unsqueeze(0), home['left_quat'].unsqueeze(0),
    )[0]
    r_quat = quat_mul(
        r_rel_q.unsqueeze(0), home['right_quat'].unsqueeze(0),
    )[0]
    action = torch.zeros(1, action_dim, dtype=torch.float32, device=device)
    action[0, 0:3] = l_pos
    action[0, 3:7] = l_quat
    action[0, 7] = float(row[6])
    action[0, 8:11] = r_pos
    action[0, 11:15] = r_quat
    action[0, 15] = float(row[13])
    if action_dim >= 21:
        action[0, 16] = 0.0
        action[0, 17] = 0.0
        action[0, 18] = float(home['lift'] or 0.0)
        action[0, 19] = float(home['head_yaw'] or 0.0)
        action[0, 20] = float(home['head_pitch'] or 0.0)
    elif action_dim > 16:
        if home['lift'] is not None and action_dim >= 19:
            action[0, 18] = float(home['lift'])
    return action


def _write_pose_vel(asset, env, pose_wxyz: np.ndarray, vel: np.ndarray) -> dict:
    import torch

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    pose = torch.zeros(1, 7, dtype=torch.float32, device=env.device)
    local = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
    pose[0, :3] = torch.as_tensor(local[:3], device=env.device)
    pose[0, 3:7] = torch.as_tensor(_wxyz_to_xyzw(local[3:7])[0], device=env.device)
    pose[:, :3] += env.scene.env_origins[env_ids]
    vel_t = torch.as_tensor(
        np.asarray(vel, dtype=np.float32).reshape(1, 6),
        device=env.device, dtype=torch.float32,
    )
    info = {'pose_written': False, 'velocity_written': False, 'error': None}
    try:
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        info['pose_written'] = True
    except Exception as exc:
        info['error'] = f'pose: {exc}'
        return info
    try:
        asset.write_root_velocity_to_sim(vel_t, env_ids=env_ids)
        info['velocity_written'] = True
    except Exception as exc:
        info['error'] = f'velocity: {exc}'
    return info


def restore_initial_state(env, initial: dict[str, Any]) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        'hdf5_path': initial['path'],
        'object_map': dict(OBJECT_NAME_MAP),
        'robot_root_pose_restored': False,
        'robot_root_velocity_restored': False,
        'robot_joint_restore': False,
        'robot_joint_mismatch': None,
        'objects': {},
        'warnings': [],
    }
    robot = env.scene['robot']
    sim_names = list(robot.data.joint_names)
    hdf5_q = initial['robot']['joint_position']
    hdf5_dq = initial['robot']['joint_velocity']
    report['sim_joint_names'] = sim_names
    report['sim_joint_count'] = len(sim_names)
    report['hdf5_joint_count'] = int(hdf5_q.shape[0])
    report['hdf5_joint_names'] = None

    robot_info = _write_pose_vel(
        robot, env,
        initial['robot']['root_pose_wxyz'],
        initial['robot']['root_velocity'],
    )
    report['robot_root_pose_restored'] = bool(robot_info['pose_written'])
    report['robot_root_velocity_restored'] = bool(robot_info['velocity_written'])
    if robot_info['error']:
        report['warnings'].append(f'robot root: {robot_info["error"]}')

    count_ok = int(hdf5_q.shape[0]) == len(sim_names)
    # HDF5 has no joint names. Sim order is interleaved (L1,R1,L2,R2,...);
    # EX001-6R recordings are grouped left-then-right. Do not write the whole
    # array unless names and order are known identical.
    if count_ok:
        print('[restore] joint COUNT matches 18, but order is not verified.')
        print(f'  HDF5 joint_position: {hdf5_q.tolist()}')
        print(f'  sim joint_names: {sim_names}')
        print('[restore] skip whole-array joint write; keep current home joints')
        report['robot_joint_restore'] = False
        report['robot_joint_mismatch'] = {
            'hdf5_joint_count': int(hdf5_q.shape[0]),
            'sim_joint_count': len(sim_names),
            'sim_joint_names': sim_names,
            'hdf5_joint_names': None,
            'count_matches': True,
            'reason': (
                'HDF5 has no joint names; sim order is interleaved L/R and '
                'cannot be verified. Kept home joints. HDF5 init is all zeros.'
            ),
        }
    else:
        report['robot_joint_restore'] = False
        report['robot_joint_mismatch'] = {
            'hdf5_joint_count': int(hdf5_q.shape[0]),
            'sim_joint_count': len(sim_names),
            'sim_joint_names': sim_names,
            'reason': 'count mismatch; skipped whole-array joint write',
        }
        print('[restore] joint count mismatch; keep current home joints')
        print(f'  HDF5 joint_position: {hdf5_q.shape}')
        print(f'  sim joint_names ({len(sim_names)}): {sim_names}')

    def _scene_has(name: str) -> bool:
        try:
            env.scene[name]
            return True
        except Exception:
            return False

    for src_name, dst_name in OBJECT_NAME_MAP.items():
        obj_info = {
            'source': src_name, 'restored': False, 'shape_ok': False,
            'error': None,
        }
        if src_name not in initial['objects']:
            obj_info['error'] = 'missing in HDF5'
            report['objects'][dst_name] = obj_info
            print(f'[restore] missing HDF5 rigid_object/{src_name}')
            continue
        if not _scene_has(dst_name):
            obj_info['error'] = f'missing in sim scene ({dst_name})'
            report['objects'][dst_name] = obj_info
            print(f'[restore] missing sim asset {dst_name} for {src_name}')
            continue
        blob = initial['objects'][src_name]
        pose = blob['root_pose_wxyz']
        vel = blob['root_velocity']
        obj_info['shape_ok'] = pose.shape[-1] == 7 and vel.shape[-1] == 6
        written = _write_pose_vel(env.scene[dst_name], env, pose, vel)
        obj_info['restored'] = bool(
            written['pose_written'] and written['velocity_written'],
        )
        obj_info['error'] = written['error']
        report['objects'][dst_name] = obj_info
        print(
            f'[restore] {src_name} -> {dst_name} '
            f'pose={written["pose_written"]} vel={written["velocity_written"]}',
        )

    env.scene.write_data_to_sim()
    env.sim.forward()
    try:
        from maniparena_sim.loops.replay_loop import warmup_rtx_cameras
        warmup_rtx_cameras(env)
    except Exception as exc:
        print(f'[restore] camera warmup skipped: {exc}')
    return report


def disable_auto_termination(env):
    tm = getattr(env, 'termination_manager', None)
    if tm is None:
        raise RuntimeError('env has no termination_manager')
    success_cfg = tm.get_term_cfg('success')
    original = {
        'compute': tm.compute,
        'success_fn': success_cfg.func,
        'success_params': dict(success_cfg.params),
    }

    def _noop():
        import torch
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    tm.compute = _noop
    return original


def eval_objects_in_basket(env, success_params: dict) -> dict[str, bool]:
    from isaaclab.managers import SceneEntityCfg

    from maniparena_sim.terms.terminations import all_objects_in_basket

    flags = {}
    basket_cfg = success_params['basket_cfg']
    extra = {
        k: success_params[k]
        for k in ('x_threshold', 'y_threshold', 'z_threshold', 'velocity_threshold')
        if k in success_params
    }
    for name in TARGET_OBJECT_NAMES:
        ok = all_objects_in_basket(
            env,
            object_cfgs=[SceneEntityCfg(name)],
            basket_cfg=basket_cfg,
            **extra,
        )
        flags[name] = bool(ok[0].item())
    return flags


def read_object_positions(env) -> dict[str, Any]:
    out = {}
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    names = list(TARGET_OBJECT_NAMES) + ['platform_pink']
    for name in names:
        try:
            env.scene[name]
        except Exception:
            continue
        obj = env.scene[name]
        pos_w = _to_torch_f32(obj.data.root_pos_w)[0].detach().cpu().numpy()
        quat = _to_torch_f32(obj.data.root_quat_w)[0].detach().cpu().numpy()
        vel = _to_torch_f32(obj.data.root_lin_vel_w)[0].detach().cpu().numpy()
        out[name] = {
            'root_pos_w': pos_w.tolist(),
            'root_pos_env': (pos_w - origin).tolist(),
            'root_quat_xyzw': quat.tolist(),
            'root_lin_vel_w': vel.tolist(),
        }
    return out


def geodesic_error(master_euler: np.ndarray, follow_euler: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    r_m = Rotation.from_euler('xyz', master_euler)
    r_f = Rotation.from_euler('xyz', follow_euler)
    return (r_m.inv() * r_f).magnitude().astype(np.float64)


def best_lag(master: np.ndarray, follow: np.ndarray, max_lag: int) -> dict[str, Any]:
    def pos_mae(m, f):
        left = np.mean(np.abs(m[:, 0:3] - f[:, 0:3]), axis=1)
        right = np.mean(np.abs(m[:, 7:10] - f[:, 7:10]), axis=1)
        return 0.5 * (left + right)

    def left_pos_mae(m, f):
        return np.mean(np.abs(m[:, 0:3] - f[:, 0:3]), axis=1)

    def right_pos_mae(m, f):
        return np.mean(np.abs(m[:, 7:10] - f[:, 7:10]), axis=1)

    def rot_err(m, f):
        return 0.5 * (
            geodesic_error(m[:, 3:6], f[:, 3:6])
            + geodesic_error(m[:, 10:13], f[:, 10:13])
        )

    def left_grip(m, f):
        return np.abs(m[:, 6] - f[:, 6])

    def right_grip(m, f):
        return np.abs(m[:, 13] - f[:, 13])

    def search(fn):
        best = None
        for lag in range(0, int(max_lag) + 1):
            n = master.shape[0]
            if lag >= n:
                break
            m = master[:-lag] if lag else master
            f = follow[lag:] if lag else follow
            value = float(np.mean(fn(m, f)))
            item = {
                'lag_frames': lag,
                'lag_ms': lag / 20.0 * 1000.0,
                'mae': value,
            }
            if best is None or value < best['mae']:
                best = item
        return best

    return {
        'position_both': search(pos_mae),
        'position_left': search(left_pos_mae),
        'position_right': search(right_pos_mae),
        'rotation': search(rot_err),
        'gripper_left': search(left_grip),
        'gripper_right': search(right_grip),
    }


def save_tracking_plot(
    path: Path,
    master: np.ndarray,
    follow: np.ndarray,
    hz: float,
    lags: dict[str, Any],
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(master.shape[0], dtype=np.float32) / float(hz)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    labels = ['x (m)', 'y (m)', 'z (m)']
    for i, lab in enumerate(labels):
        axes[0].plot(t, master[:, i], label=f'master L {lab}', lw=1.2)
        axes[0].plot(t, follow[:, i], '--', label=f'follow L {lab}', lw=1.0)
        axes[1].plot(t, master[:, 7 + i], label=f'master R {lab}', lw=1.2)
        axes[1].plot(t, follow[:, 7 + i], '--', label=f'follow R {lab}', lw=1.0)
    axes[0].set_ylabel('left EE pos')
    axes[1].set_ylabel('right EE pos')
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(t, master[:, 6], label='master L grip')
    axes[2].plot(t, follow[:, 6], '--', label='follow L grip')
    axes[2].plot(t, master[:, 13], label='master R grip')
    axes[2].plot(t, follow[:, 13], '--', label='follow R grip')
    axes[2].set_ylabel('gripper')
    axes[2].legend(ncol=2, fontsize=8)

    pos_err = 0.5 * (
        np.mean(np.abs(master[:, 0:3] - follow[:, 0:3]), axis=1)
        + np.mean(np.abs(master[:, 7:10] - follow[:, 7:10]), axis=1)
    )
    rot_err = 0.5 * (
        geodesic_error(master[:, 3:6], follow[:, 3:6])
        + geodesic_error(master[:, 10:13], follow[:, 10:13])
    )
    axes[3].plot(t, pos_err, label='pos MAE (m)')
    axes[3].plot(t, rot_err, label='rot geodesic (rad)')
    axes[3].set_ylabel('tracking error')
    axes[3].set_xlabel('time (s)')
    axes[3].legend(fontsize=8)

    pos_lag = lags['position_both']
    fig.suptitle(
        'GT master vs sim follow (init-EE relative, 20 Hz)\n'
        f"best pos lag={pos_lag['lag_frames']} frames "
        f"({pos_lag['lag_ms']:.0f} ms), MAE={pos_lag['mae']:.4f} m",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_csv(path: Path, master: np.ndarray, follow: np.ndarray, hz: float):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['frame', 'time_s']
        header += [f'master_{c}' for c in MASTER_CSV_COLS]
        header += [f'follow_{c}' for c in MASTER_CSV_COLS]
        writer.writerow(header)
        for i in range(master.shape[0]):
            row = [i, i / float(hz)]
            row.extend(float(x) for x in master[i])
            row.extend(float(x) for x in follow[i])
            writer.writerow(row)


def save_video(path: Path, frames: list[np.ndarray], hz: float) -> str | None:
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(path), frames, fps=float(hz))
        return str(path)
    except Exception as exc:
        print(f'[video] imageio failed: {exc}')
    try:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*'mp4v'), float(hz), (w, h),
        )
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        return str(path)
    except Exception as exc:
        print(f'[video] cv2 failed: {exc}')
        return None


def capture_frame(gym_env) -> np.ndarray | None:
    try:
        frame = gym_env.render()
    except Exception:
        return None
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def describe_action_space(env) -> dict[str, Any]:
    info: dict[str, Any] = {}
    space = getattr(env, 'single_action_space', None) or getattr(env, 'action_space', None)
    info['action_space'] = str(space)
    info['action_shape'] = list(getattr(space, 'shape', ()))
    am = getattr(env, 'action_manager', None)
    if am is not None:
        info['total_action_dim'] = int(getattr(am, 'total_action_dim', 0) or 0)
        terms = {}
        for name in getattr(am, 'active_terms', []) or []:
            term = am.get_term(name)
            terms[name] = {
                'action_dim': int(getattr(term, 'action_dim', 0) or 0),
                'class': type(term).__name__,
            }
        info['action_terms'] = terms
    robot = env.scene['robot']
    info['joint_names'] = list(robot.data.joint_names)
    info['body_names_ee'] = [LEFT_EE_BODY, RIGHT_EE_BODY]
    return info


def main() -> int:
    args = parse_args()
    payload = load_yaml(args.config)
    payload = dict(payload)
    payload['enable_recording'] = False
    payload['num_envs'] = 1
    # Wrist cameras are not required for GT replay; keep AppLauncher cameras
    # for offscreen viewer frames.
    payload['enable_cameras'] = False
    args.enable_cameras = bool(args.video)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    from maniparena_sim.environment.builder import (
        EX001_SUPPORTED_TASKS,
        build_eval_gym_env,
        build_ex001_eval_gym_env,
    )
    from maniparena_sim.environment.registry import bootstrap_arena_registry

    bootstrap_arena_registry()
    render_mode = 'rgb_array' if args.video else None
    if args.robot == 'ex001':
        if args.task not in EX001_SUPPORTED_TASKS:
            raise SystemExit(f'ex001 unsupported task: {args.task}')
        ctx = build_ex001_eval_gym_env(
            args.task, payload,
            headless=bool(getattr(args, 'headless', False)),
            device=getattr(args, 'device', 'cuda:0'),
            render_mode=render_mode,
        )
    else:
        ctx = build_eval_gym_env(
            args.task, payload,
            headless=bool(getattr(args, 'headless', False)),
            device=getattr(args, 'device', 'cuda:0'),
            render_mode=render_mode,
        )

    gym_env = ctx.gym_env
    env = getattr(gym_env, 'unwrapped', gym_env)
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = resolve_hdf5_path(args.hdf5_root, args.episode)
    master = load_lerobot_actions(args.lerobot_root, args.episode)
    initial = load_hdf5_initial_state(hdf5_path)

    gym_env.reset()
    restore_report = restore_initial_state(env, initial)
    home = capture_home_refs(env)
    term_orig = disable_auto_termination(env)
    action_info = describe_action_space(env)
    action_dim = int(action_info.get('total_action_dim') or 16)
    sim_step_dt = float(getattr(env, 'step_dt', 1.0 / 60.0))
    repeat = max(1, int(round((1.0 / float(args.dataset_hz)) / sim_step_dt)))

    print('=' * 60)
    print(f'  GT master interface test')
    print(f'  robot:     {args.robot}')
    print(f'  episode:   {args.episode}')
    print(f'  frames:    {master.shape[0]}')
    print(f'  dataset_hz:{args.dataset_hz}')
    print(f'  step_dt:   {sim_step_dt}')
    print(f'  repeat:    {repeat}')
    print(f'  action:    {action_info.get("action_shape")} dim={action_dim}')
    print(f'  joints:    {action_info.get("joint_names")}')
    print(f'  joint_restore: {restore_report["robot_joint_restore"]}')
    print(f'  output:    {out_dir}')
    print('=' * 60)

    follows = []
    frames = []
    success_streak = 0
    success_first = None
    success_confirmed = False
    n_replay = int(master.shape[0])
    with torch.inference_mode():
        for i in range(n_replay + int(args.settle_frames)):
            if not simulation_app.is_running():
                break
            src = master[min(i, n_replay - 1)]
            action = master_to_abs_ik(src, env, home, action_dim)
            for _ in range(repeat):
                gym_env.step(action)
            follow = read_follow_14d(env, home)
            follows.append(follow)
            if args.video:
                frame = capture_frame(gym_env)
                if frame is not None:
                    frames.append(frame)
            raw_success = bool(
                term_orig['success_fn'](env, **term_orig['success_params'])[0].item()
            )
            if raw_success:
                success_streak += 1
            else:
                success_streak = 0
            if (
                success_streak >= int(args.success_hold_frames)
                and not success_confirmed
            ):
                success_confirmed = True
                success_first = i - int(args.success_hold_frames) + 1
            if i % 50 == 0 or i == n_replay - 1:
                flags = eval_objects_in_basket(env, term_orig['success_params'])
                print(
                    f'[gt] frame {i}/{n_replay + args.settle_frames} '
                    f'success_raw={raw_success} streak={success_streak} '
                    f'in_basket={flags}',
                )

    follow_arr = np.stack(follows, axis=0).astype(np.float32)
    master_full = np.concatenate(
        [
            master,
            np.repeat(master[-1:], max(0, follow_arr.shape[0] - master.shape[0]), axis=0),
        ],
        axis=0,
    )[:follow_arr.shape[0]]
    master_cmp = master
    follow_cmp = follow_arr[: master.shape[0]]
    lags = best_lag(master_cmp, follow_cmp, args.max_lag_frames)
    objects_end = eval_objects_in_basket(env, term_orig['success_params'])
    positions = read_object_positions(env)
    all_in = all(objects_end.values())
    success = bool(success_confirmed)
    failure_reason = None
    if not success:
        missing = [k for k, v in objects_end.items() if not v]
        if missing:
            failure_reason = f'objects_not_in_basket:{",".join(missing)}'
        else:
            failure_reason = 'success_not_held_for_10_frames'

    summary = {
        'episode': int(args.episode),
        'robot': str(args.robot),
        'dataset_hz': float(args.dataset_hz),
        'sim_step_dt': sim_step_dt,
        'repeat_per_gt_frame': repeat,
        'initial_state_restore': restore_report,
        'best_position_lag_frames': lags['position_both']['lag_frames'],
        'best_position_lag_ms': lags['position_both']['lag_ms'],
        'best_left_position_lag_frames': lags['position_left']['lag_frames'],
        'best_right_position_lag_frames': lags['position_right']['lag_frames'],
        'best_rotation_lag_frames': lags['rotation']['lag_frames'],
        'best_rotation_lag_ms': lags['rotation']['lag_ms'],
        'best_left_gripper_lag_frames': lags['gripper_left']['lag_frames'],
        'best_right_gripper_lag_frames': lags['gripper_right']['lag_frames'],
        'best_left_gripper_lag_ms': lags['gripper_left']['lag_ms'],
        'best_right_gripper_lag_ms': lags['gripper_right']['lag_ms'],
        'best_position_mae_m': lags['position_both']['mae'],
        'lags': lags,
        'success': success,
        'success_first_frame': success_first,
        'objects_in_basket': objects_end,
        'failure_reason': failure_reason,
        'n_replay_frames': n_replay,
        'n_recorded_frames': int(follow_arr.shape[0]),
        'action_space': action_info,
        'home_ee': {
            'left_pos': home['left_pos'].detach().cpu().numpy().tolist(),
            'right_pos': home['right_pos'].detach().cpu().numpy().tolist(),
            'lift': home['lift'],
        },
        'official_eval_robot_note': (
            'fruits_to_basket.yaml closed-loop eval uses --robot bimanual; '
            'this run is tagged by --robot.'
        ),
        'all_objects_in_basket_end': all_in,
        'dataset_baseline_lag_note': (
            'emma-server dataset self-lag: EE 5-6 frames (250-300 ms), '
            'gripper 2-4 frames (100-200 ms).'
        ),
    }

    save_csv(out_dir / 'master_follow.csv', master_full, follow_arr, args.dataset_hz)
    save_tracking_plot(
        out_dir / 'tracking_plot.png',
        master_cmp, follow_cmp, args.dataset_hz, lags,
    )
    (out_dir / 'final_object_positions.json').write_text(
        json.dumps(positions, indent=2, default=_json_default),
        encoding='utf-8',
    )
    video_path = None
    if args.video:
        video_path = save_video(out_dir / 'replay.mp4', frames, args.dataset_hz)
    summary['video_path'] = video_path
    summary['n_video_frames'] = len(frames)
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding='utf-8',
    )
    print(json.dumps(
        {k: summary[k] for k in (
            'episode', 'robot', 'success', 'success_first_frame',
            'best_position_lag_frames', 'best_position_lag_ms',
            'best_left_gripper_lag_frames', 'best_right_gripper_lag_frames',
            'objects_in_basket', 'failure_reason', 'repeat_per_gt_frame',
        )},
        indent=2,
    ))
    print(f'[gt] wrote {out_dir}')

    term_orig['compute']  # keep reference; env is closing
    env.termination_manager.compute = term_orig['compute']
    gym_env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
