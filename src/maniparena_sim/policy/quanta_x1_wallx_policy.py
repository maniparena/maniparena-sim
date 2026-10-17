"""QUANTA_X1 Wall-X whole-body closed-loop policy (21D).

Wire action layout (``ActionsCfgWallxWholebody``):
``[L_pose7, Lg, R_pose7, Rg, L_wheel, R_wheel, lift, head_yaw, head_pitch]``.

Wall-X QUANTA_X1 pose contract (``wire_pose_reference: init_ee_pose``):
- Request ``follow1/2_pos`` is in the ``lift_link`` (arm base) frame, relative to
  home EE. Arms and lift are independent: no lift term enters the arm wire.
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

from maniparena_sim.embodiment.robots.quanta_x1 import (
    QUANTA_X1_WHEEL_RADIUS_M,
    QUANTA_X1_WHEEL_TRACK_WIDTH_M,
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
class QuantaX1WallxPolicyConfig(RobotPolicyConfig):
    wheel_radius: float = QUANTA_X1_WHEEL_RADIUS_M
    wheel_track_width: float = QUANTA_X1_WHEEL_TRACK_WIDTH_M
    lift_joint_name: str = "lift_joint"
    head_yaw_joint_name: str = "head_yaw_joint"
    head_pitch_joint_name: str = "head_pitch_joint"
    # Match AbsIK ``body_name``; avoid FrameTransformer Warp ProxyArray.
    left_ee_body: str = "left_arm_gripper_base_link"
    right_ee_body: str = "right_arm_gripper_base_link"
    left_gripper_joint: str = "left_arm_gripper"
    right_gripper_joint: str = "right_arm_gripper"
    # Arms are posed in this body's frame, independent of lift.
    arm_base_body: str = "lift_link"
    # follow* is already init-relative on the Wall-X wire; do not double-normalize.
    ee_pose_normalize: bool = False
    # "absolute": response ``lift`` is the joint target (m).
    # "init_relative": response ``lift`` is a delta from the lift at reset.
    lift_wire_mode: str = "init_relative"
    # "dataset": remap the server fields to the training-data tail order [twist, lift, head];
    # "named": use the server field names as is (legacy).
    wire_layout: str = "dataset"
    # The checkpoint copies the measured base twist into its twist output, so from exact rest it
    # never starts driving. Training episodes start moving ~1 s in with ~0.06 m/s: report this
    # forward speed in the first query of an episode so the model emits its whole start-up ramp.
    base_kick_vx: float = 0.0
    # Experiment switch (default off = unchanged behaviour). Base commands are velocities, so holding
    # every chunk row ``interpolation_multiplier`` times longer multiplies the base displacement per
    # chunk by the same factor while arm position targets only slow down. When enabled, wheel speeds
    # are divided by ``interpolation_multiplier`` so each chunk moves the base by the demonstrated amount.
    base_velocity_inv_interp: bool = False
    # Extra constant factor on the wheel speeds (experiment; 1.0 = off).
    base_velocity_scale: float = 1.0
    # Absolute lift command (m) held for the whole episode; the model's lift output is ignored.
    # Demos keep the lift command constant per task (sort 0.455, fruits/mobile 0.379), and the
    # checkpoints drift to the sort lift (~0.455) once the lift state passes ~0.41. eval.py fills this
    # per task from the demo value; None = follow the model.
    lift_hold: Optional[float] = None


class QuantaX1WallxPolicy(RobotClosedloopPolicy):
    """Closed-loop Wall-X policy driving QUANTA_X1 ``ActionsCfgWallxWholebody``."""

    ACTION_DIM = 21
    _LEFT_REF_KEY = "_left_ee_ref"
    _RIGHT_REF_KEY = "_right_ee_ref"

    def __init__(self, config: QuantaX1WallxPolicyConfig):
        super().__init__(config)
        self.cfg: QuantaX1WallxPolicyConfig = config

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

    def _get_ee_arm_base_local(
        self, env: Any, body_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EE pose in the arm-base (``lift_link``) frame, so it is independent of lift."""
        robot = env.scene["robot"]
        body_ids, _ = robot.find_bodies(body_name)
        base_ids, _ = robot.find_bodies(self.cfg.arm_base_body)
        body_idx, base_idx = int(body_ids[0]), int(base_ids[0])
        body_pos_all = _to_torch_f32(robot.data.body_pos_w)
        body_quat_all = _to_torch_f32(robot.data.body_quat_w)
        n = body_pos_all.shape[0]
        # Packed warp quatf may arrive as (N, B*4) or (N,) → reshape to (N, B, 4).
        body_quat_all = body_quat_all.reshape(n, -1, 4)
        return subtract_frame_transforms(
            body_pos_all[:, base_idx, :], body_quat_all[:, base_idx, :],
            body_pos_all[:, body_idx, :], body_quat_all[:, body_idx, :],
        )

    def _ensure_home_refs(self, env: Any) -> None:
        """Cache root-local home EE (+ lift) refs for init-relative wire poses."""
        stale = getattr(self, "_home_refs_stale", True)
        need_left = not hasattr(env, self._LEFT_REF_KEY)
        need_right = not hasattr(env, self._RIGHT_REF_KEY)
        lift_key = "_lift_ref_" + self.cfg.lift_joint_name
        need_lift = not hasattr(env, lift_key)
        if not (stale or need_left or need_right or need_lift):
            return

        l_pos, l_quat = self._get_ee_arm_base_local(env, self.cfg.left_ee_body)
        r_pos, r_quat = self._get_ee_arm_base_local(env, self.cfg.right_ee_body)
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
        """7D follow wire: init-relative pose in the arm-base (lift_link) frame."""
        pos, quat = self._get_ee_arm_base_local(env, body_name)
        ref_pos, ref_quat = getattr(env, ref_key)
        relative_pos = pos - ref_pos
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
        twist = self._body_twist(env)
        if self.cfg.base_kick_vx and self._query_count == 0:
            twist = twist.copy()
            twist[0] = max(float(twist[0]), float(self.cfg.base_kick_vx))
        lift = np.asarray([self._read_joint(env, self.cfg.lift_joint_name)], dtype=np.float32)
        head = np.asarray(
            [
                self._read_joint(env, self.cfg.head_yaw_joint_name),
                self._read_joint(env, self.cfg.head_pitch_joint_name),
            ],
            dtype=np.float32,
        )
        if self.cfg.wire_layout == "dataset":
            # The checkpoint's 26-D tail is [twist(3), lift, head(2)] (training data order) but the
            # server fills it from dof_config names [head(2), height(1), velocity(3)]. Send the
            # values under the names whose slots hold them in the training data.
            state["head_pos"] = twist[:2]
            state["lift"] = twist[2:3]
            state["velocity_decomposed"] = np.concatenate([lift, head]).astype(np.float32)
        else:
            state["velocity_decomposed_odom"] = twist
            state["lift"] = lift
            state["head_pos"] = head
        out = dict(model_input)
        out["state"] = state
        return out

    def _base_velocity_factor(self) -> float:
        """Experiment-only wheel-speed factor (1.0 unless a ``base_velocity_*`` switch is set)."""
        factor = float(self.cfg.base_velocity_scale)
        if self.cfg.base_velocity_inv_interp:
            factor /= max(1.0, float(self.cfg.interpolation_multiplier))
        return factor

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

    @staticmethod
    def _remap_dataset_layout(response: Dict[str, Any]) -> Dict[str, Any]:
        """Server fields -> robot semantics for the training-data tail order [twist, lift, head].

        ``head_pos`` carries (vx, vy), ``lift`` carries wz, and ``velocity_decomposed_odom``
        carries (lift, head_yaw, head_pitch).
        """
        head, lift, odom = (response.get(k) for k in ("head_pos", "lift", "velocity_decomposed_odom"))
        if head is None or lift is None or odom is None:
            return response
        head = np.asarray(head, dtype=np.float32).reshape(-1, 2)
        wz = np.asarray(lift, dtype=np.float32).reshape(-1, 1)
        odom = np.asarray(odom, dtype=np.float32).reshape(-1, 3)
        n = min(len(head), len(wz), len(odom))
        out = dict(response)
        out["velocity_decomposed_odom"] = np.concatenate([head[:n], wz[:n]], axis=1)
        out["lift"] = odom[:n, 0:1]
        out["head_pos"] = odom[:n, 1:3]
        return out

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
                "[QuantaX1WallxPolicy] Invalid response shape: "
                f"follow1={f1.shape}, follow2={f2.shape}",
            )
            return None

        horizon = min(f1.shape[0], f2.shape[0], self.cfg.action_horizon)
        if horizon <= 0:
            return None

        if self.cfg.wire_layout == "dataset":
            response = self._remap_dataset_layout(response)

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
            wheels = wheels * self._base_velocity_factor()

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

        if self.cfg.lift_hold is not None:
            lift_arr = np.full_like(lift_arr, float(self.cfg.lift_hold))
            lift = lift_arr  # absolute command; skip the init-relative offset below
        self._ensure_home_refs(env)
        if (
            lift is not None
            and self.cfg.wire_layout != "dataset"  # dataset layout already carries absolute lift
            and self.cfg.lift_wire_mode == "init_relative"
        ):
            lift_init = getattr(env, "_lift_ref_" + self.cfg.lift_joint_name)
            lift_arr = lift_arr + float(lift_init[0, 0].item())
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

        # Receding horizon: execute only the first ``action_chunk_length`` rows, then re-query.
        chunk = chunk[: max(1, min(int(self.cfg.action_chunk_length), self.cfg.action_horizon))]

        if self.cfg.interpolation_multiplier > 1:
            interpolated = self._interpolate_rows(
                chunk.detach().cpu().numpy(),
                self.cfg.interpolation_multiplier,
            )
            return torch.as_tensor(interpolated, dtype=torch.float32, device=device)
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
            print(f"[QuantaX1WallxPolicy] Inference failed: {e}")
            return None
        if "error" in response:
            print(f"[QuantaX1WallxPolicy] Server error: {response['error']}")
            return None
        return self._response_to_action_chunk(response, env, device)
