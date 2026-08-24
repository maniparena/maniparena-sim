"""EX001 Wall-X whole-body closed-loop policy (21D).

Wire action layout (``ActionsCfgWallxWholebody``):
``[L_pose7, Lg, R_pose7, Rg, L_wheel, R_wheel, lift, head_yaw, head_pitch]``.

Wall-X EX001 pose contract (``wire_pose_reference: init_ee_pose``):
- Request ``follow1/2_pos`` is root-local, relative to home EE, with lift
  ΔZ removed from the Z channel.
- Response uses the same relative wire; client composes
  ``target_b = home_b + wire`` and ``q_b = q_rel * q_home``.
- Quaternions use Isaac Lab / DiffIK **XYZW**.
- No env-origin / world-frame path and no client ``ee_pose_normalize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)
from scipy.spatial.transform import Rotation

from maniparena_sim.embodiment.robots.ex001 import (
    EX001_WHEEL_RADIUS_M,
    EX001_WHEEL_TRACK_WIDTH_M,
    twist_to_wheel_vel,
)
from maniparena_sim.policy.robot_policy import RobotClosedloopPolicy, RobotPolicyConfig
from maniparena_sim.utils.math_utils import euler_xyz_to_quat_xyzw


def _to_torch_f32(value: Any, device: torch.device | str | None = None) -> torch.Tensor:
    """Coerce Lab/Fabric tensors (incl. Warp ProxyArray / quatf) to float32 torch."""
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = None
        # Fabric often exposes Warp ProxyArray; prefer wp.to_torch before Lab convert.
        try:
            import warp as wp

            tensor = wp.to_torch(value)
        except Exception:
            tensor = None
        if not isinstance(tensor, torch.Tensor):
            try:
                tensor = convert_to_torch(value)
            except Exception:
                raw = value.numpy() if hasattr(value, "numpy") else value
                tensor = torch.as_tensor(np.asarray(raw))
    tensor = tensor.to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


@dataclass
class Ex001WallxPolicyConfig(RobotPolicyConfig):
    wheel_radius: float = EX001_WHEEL_RADIUS_M
    wheel_track_width: float = EX001_WHEEL_TRACK_WIDTH_M
    lift_joint_name: str = "lift_joint"
    head_yaw_joint_name: str = "head_yaw_joint"
    head_pitch_joint_name: str = "head_pitch_joint"
    # Match AbsIK ``body_name``; avoid FrameTransformer Warp ProxyArray.
    left_ee_body: str = "left_arm_gripper_base_link"
    right_ee_body: str = "right_arm_gripper_base_link"
    left_gripper_joint: str = "left_arm_gripper"
    right_gripper_joint: str = "right_arm_gripper"
    # follow* is already init-relative on the Wall-X wire; do not double-normalize.
    ee_pose_normalize: bool = False


class Ex001WallxPolicy(RobotClosedloopPolicy):
    """Closed-loop Wall-X policy driving EX001 ``ActionsCfgWallxWholebody``."""

    ACTION_DIM = 21
    _LEFT_REF_KEY = "_left_ee_ref"
    _RIGHT_REF_KEY = "_right_ee_ref"

    def __init__(self, config: Ex001WallxPolicyConfig):
        super().__init__(config)
        self.cfg: Ex001WallxPolicyConfig = config

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        # Force re-capture of home EE / lift refs on the next query.
        self._home_refs_stale = True

    def _read_joint(self, env: Any, name: str) -> float:
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(name)
        joint_pos = _to_torch_f32(robot.data.joint_pos)
        return float(joint_pos[0, ids[0]].item())

    def _read_joint_tensor(self, env: Any, name: str) -> torch.Tensor:
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(name)
        joint_pos = _to_torch_f32(robot.data.joint_pos)
        return joint_pos[:, ids[0]].reshape(-1, 1)

    def _body_twist(self, env: Any) -> np.ndarray:
        """Approximate ``velocity_decomposed_odom`` from root twist in body frame."""
        robot = env.scene["robot"]
        lin_w = _to_torch_f32(robot.data.root_lin_vel_w)[0].detach().cpu().numpy()
        ang_w = _to_torch_f32(robot.data.root_ang_vel_w)[0].detach().cpu().numpy()
        # Arena Lab body/root quats are XYZW (scipy convention).
        quat_xyzw = _to_torch_f32(robot.data.root_quat_w)[0].detach().cpu().numpy()
        rot = Rotation.from_quat(quat_xyzw)
        lin_b = rot.inv().apply(lin_w)
        ang_b = rot.inv().apply(ang_w)
        return np.asarray([lin_b[0], lin_b[1], ang_b[2]], dtype=np.float32)

    def _get_ee_root_local(
        self, env: Any, body_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EE pose in robot root frame from articulation body state (torch)."""
        robot = env.scene["robot"]
        body_ids, _ = robot.find_bodies(body_name)
        body_idx = int(body_ids[0])
        root_pos = _to_torch_f32(robot.data.root_pos_w)
        root_quat = _to_torch_f32(robot.data.root_quat_w)
        body_pos = _to_torch_f32(robot.data.body_pos_w)[:, body_idx, :]
        body_quat = _to_torch_f32(robot.data.body_quat_w)[:, body_idx, :]
        # Packed warp quatf may arrive as (N,) → reshape to (N, 4).
        if root_quat.ndim == 1:
            root_quat = root_quat.reshape(-1, 4)
        if body_quat.ndim == 1:
            body_quat = body_quat.reshape(root_pos.shape[0], 4)
        elif body_quat.ndim == 2 and body_quat.shape[-1] != 4:
            body_quat = body_quat.reshape(root_pos.shape[0], 4)
        return subtract_frame_transforms(root_pos, root_quat, body_pos, body_quat)

    def _ensure_home_refs(self, env: Any) -> None:
        """Cache root-local home EE (+ lift) refs for init-relative wire poses."""
        stale = getattr(self, "_home_refs_stale", True)
        need_left = not hasattr(env, self._LEFT_REF_KEY)
        need_right = not hasattr(env, self._RIGHT_REF_KEY)
        lift_key = "_lift_ref_" + self.cfg.lift_joint_name
        need_lift = not hasattr(env, lift_key)
        if not (stale or need_left or need_right or need_lift):
            return

        l_pos, l_quat = self._get_ee_root_local(env, self.cfg.left_ee_body)
        r_pos, r_quat = self._get_ee_root_local(env, self.cfg.right_ee_body)
        if stale or need_left:
            setattr(env, self._LEFT_REF_KEY, (l_pos.clone(), l_quat.clone()))
        if stale or need_right:
            setattr(env, self._RIGHT_REF_KEY, (r_pos.clone(), r_quat.clone()))
        if stale or need_lift:
            setattr(env, lift_key, self._read_joint_tensor(env, self.cfg.lift_joint_name).clone())
        self._home_refs_stale = False

    def _read_home_ee(
        self, env: Any, ref_key: str, device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ref = getattr(env, ref_key)
        return (
            ref[0][0].to(device=device, dtype=torch.float32),
            ref[1][0].to(device=device, dtype=torch.float32),
        )

    def _compute_follow_arm(
        self,
        env: Any,
        *,
        body_name: str,
        gripper_joint: str,
        ref_key: str,
    ) -> np.ndarray:
        """7D follow wire: init-relative root-local + lift Z compensation."""
        pos, quat = self._get_ee_root_local(env, body_name)
        ref_pos, ref_quat = getattr(env, ref_key)
        relative_pos = pos - ref_pos
        lift_now = self._read_joint_tensor(env, self.cfg.lift_joint_name)
        lift_init = getattr(env, "_lift_ref_" + self.cfg.lift_joint_name)
        relative_pos = relative_pos.clone()
        relative_pos[:, 2:3] -= lift_now - lift_init
        relative_quat = quat_mul(quat, quat_conjugate(ref_quat))
        roll, pitch, yaw = euler_xyz_from_quat(relative_quat)
        relative_euler = torch.stack([roll, pitch, yaw], dim=-1)
        grip = self._read_joint_tensor(env, gripper_joint)
        follow = torch.cat([relative_pos, relative_euler, grip], dim=-1)[0]
        return follow.detach().cpu().numpy().astype(np.float32)

    def _compute_follow_states(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_home_refs(env)
        follow1 = self._compute_follow_arm(
            env,
            body_name=self.cfg.left_ee_body,
            gripper_joint=self.cfg.left_gripper_joint,
            ref_key=self._LEFT_REF_KEY,
        )
        follow2 = self._compute_follow_arm(
            env,
            body_name=self.cfg.right_ee_body,
            gripper_joint=self.cfg.right_gripper_joint,
            ref_key=self._RIGHT_REF_KEY,
        )
        return follow1, follow2

    def build_model_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # Parent fills cameras; follow is overwritten in ``_augment_model_input``.
        return super().build_model_input(observation)

    def _augment_model_input(
        self, model_input: Dict[str, Any], env: Any,
    ) -> Dict[str, Any]:
        follow1, follow2 = self._compute_follow_states(env)
        state = dict(model_input.get("state") or {})
        state["follow1_pos"] = follow1
        state["follow2_pos"] = follow2
        state["velocity_decomposed_odom"] = self._body_twist(env)
        state["lift"] = np.asarray(
            [self._read_joint(env, self.cfg.lift_joint_name)], dtype=np.float32
        )
        state["head_pos"] = np.asarray(
            [
                self._read_joint(env, self.cfg.head_yaw_joint_name),
                self._read_joint(env, self.cfg.head_pitch_joint_name),
            ],
            dtype=np.float32,
        )
        out = dict(model_input)
        out["state"] = state
        return out

    def _odom_to_wheels(self, odom: np.ndarray) -> np.ndarray:
        odom = np.asarray(odom, dtype=np.float32)
        if odom.ndim == 1:
            odom = odom.reshape(1, -1)
        left = []
        right = []
        for row in odom:
            vx = float(row[0]) if row.shape[0] > 0 else 0.0
            vyaw = float(row[2]) if row.shape[0] > 2 else 0.0
            lw, rw = twist_to_wheel_vel(
                vx,
                vyaw,
                wheel_radius=float(self.cfg.wheel_radius),
                wheel_track_width=float(self.cfg.wheel_track_width),
            )
            left.append(lw)
            right.append(rw)
        return np.stack([left, right], axis=-1).astype(np.float32)

    def _response_to_action_chunk(
        self,
        response: Dict[str, Any],
        env: Any,
        device,
    ) -> Optional[torch.Tensor]:
        f1 = response.get("follow1_pos")
        f2 = response.get("follow2_pos")
        if f1 is None or f2 is None:
            return None

        f1 = np.asarray(f1, dtype=np.float32)
        f2 = np.asarray(f2, dtype=np.float32)
        if f1.ndim == 1:
            f1 = f1.reshape(1, -1)
        if f2.ndim == 1:
            f2 = f2.reshape(1, -1)
        if f1.shape[-1] < 7 or f2.shape[-1] < 7:
            print(
                "[Ex001WallxPolicy] Invalid response shape: "
                f"follow1={f1.shape}, follow2={f2.shape}",
            )
            return None

        horizon = min(f1.shape[0], f2.shape[0], self.cfg.action_horizon)
        if horizon <= 0:
            return None

        # Wire is already init-relative; no client denorm layer.
        f1 = f1[:horizon, :7]
        f2 = f2[:horizon, :7]

        odom = response.get("velocity_decomposed_odom")
        if odom is None:
            wheels = np.zeros((horizon, 2), dtype=np.float32)
        else:
            wheels = self._odom_to_wheels(odom)[:horizon]
            if wheels.shape[0] < horizon:
                pad = np.repeat(wheels[-1:], horizon - wheels.shape[0], axis=0)
                wheels = np.concatenate([wheels, pad], axis=0)

        lift = response.get("lift")
        if lift is None:
            lift_arr = np.full(
                (horizon, 1),
                self._read_joint(env, self.cfg.lift_joint_name),
                dtype=np.float32,
            )
        else:
            lift_arr = np.asarray(lift, dtype=np.float32).reshape(-1, 1)[:horizon]
            if lift_arr.shape[0] < horizon:
                pad = np.repeat(lift_arr[-1:], horizon - lift_arr.shape[0], axis=0)
                lift_arr = np.concatenate([lift_arr, pad], axis=0)

        head = response.get("head_pos")
        if head is None:
            head_arr = np.tile(
                np.asarray(
                    [
                        self._read_joint(env, self.cfg.head_yaw_joint_name),
                        self._read_joint(env, self.cfg.head_pitch_joint_name),
                    ],
                    dtype=np.float32,
                ),
                (horizon, 1),
            )
        else:
            head_arr = np.asarray(head, dtype=np.float32)
            if head_arr.ndim == 1:
                head_arr = head_arr.reshape(1, -1)
            head_arr = head_arr[:horizon, :2]
            if head_arr.shape[0] < horizon:
                pad = np.repeat(head_arr[-1:], horizon - head_arr.shape[0], axis=0)
                head_arr = np.concatenate([head_arr, pad], axis=0)

        self._ensure_home_refs(env)
        l_init_pos, l_init_quat = self._read_home_ee(env, self._LEFT_REF_KEY, device)
        r_init_pos, r_init_quat = self._read_home_ee(env, self._RIGHT_REF_KEY, device)

        chunk = torch.zeros(
            self.cfg.action_horizon, self.ACTION_DIM,
            dtype=torch.float32, device=device,
        )

        for i in range(horizon):
            tgt_lp = torch.as_tensor(f1[i, 0:3], dtype=torch.float32, device=device)
            tgt_rp = torch.as_tensor(f2[i, 0:3], dtype=torch.float32, device=device)
            # DiffIK + Lab math use XYZW.
            tgt_lq = torch.as_tensor(
                euler_xyz_to_quat_xyzw(np.asarray([f1[i, 3:6]], dtype=np.float32))[0],
                dtype=torch.float32, device=device,
            )
            tgt_rq = torch.as_tensor(
                euler_xyz_to_quat_xyzw(np.asarray([f2[i, 3:6]], dtype=np.float32))[0],
                dtype=torch.float32, device=device,
            )
            # init_ee_pose: q_target = q_rel * q_home.
            l_pos_b = l_init_pos + tgt_lp
            r_pos_b = r_init_pos + tgt_rp
            l_quat_b = quat_mul(tgt_lq.unsqueeze(0), l_init_quat.unsqueeze(0))[0]
            r_quat_b = quat_mul(tgt_rq.unsqueeze(0), r_init_quat.unsqueeze(0))[0]

            chunk[i, 0:3] = l_pos_b
            chunk[i, 3:7] = l_quat_b
            chunk[i, 7] = float(f1[i, 6])
            chunk[i, 8:11] = r_pos_b
            chunk[i, 11:15] = r_quat_b
            chunk[i, 15] = float(f2[i, 6])
            chunk[i, 16] = float(wheels[i, 0])
            chunk[i, 17] = float(wheels[i, 1])
            chunk[i, 18] = float(lift_arr[i, 0])
            chunk[i, 19] = float(head_arr[i, 0])
            chunk[i, 20] = float(head_arr[i, 1])

        if horizon < self.cfg.action_horizon:
            chunk[horizon:] = chunk[horizon - 1]

        return chunk

    def _query_action_chunk(
        self, env: Any, observation: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        self._client.ensure_connected()
        model_input = self._augment_model_input(
            self.build_model_input(observation), env,
        )
        device = env.device if hasattr(env, "device") else "cpu"
        self._query_count += 1
        try:
            response = self._client.predict(model_input)
        except Exception as e:
            print(f"[Ex001WallxPolicy] Inference failed: {e}")
            return None
        if "error" in response:
            print(f"[Ex001WallxPolicy] Server error: {response['error']}")
            return None
        return self._response_to_action_chunk(response, env, device)
