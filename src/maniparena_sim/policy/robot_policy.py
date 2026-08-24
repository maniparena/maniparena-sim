"""Self-contained Robot closed-loop EE policy for maniparena_sim.

Communicates with a remote policy server
via WebSocket (PolicyServerClient) and converts EE predictions
to absolute IK actions for the Bimanual embodiment.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from isaaclab.utils.math import subtract_frame_transforms
from scipy.spatial.transform import Rotation

from maniparena_sim.policy.server_client import PolicyServerClient
from maniparena_sim.utils.math_utils import euler_xyz_to_quat_xyzw


def follow_pair_to_ik16(
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    origin: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    device,
) -> torch.Tensor:
    """Convert env-local 7D+7D follow poses to a 16D absolute IK action."""
    left_ee = np.asarray(left_ee, dtype=np.float32).reshape(-1)
    right_ee = np.asarray(right_ee, dtype=np.float32).reshape(-1)
    tgt_lp = (
        torch.as_tensor(left_ee[0:3], dtype=torch.float32, device=device)
        + origin
    )
    tgt_rp = (
        torch.as_tensor(right_ee[0:3], dtype=torch.float32, device=device)
        + origin
    )
    tgt_lq = torch.as_tensor(
        euler_xyz_to_quat_xyzw(np.asarray([left_ee[3:6]], dtype=np.float32))[0],
        dtype=torch.float32, device=device,
    )
    tgt_rq = torch.as_tensor(
        euler_xyz_to_quat_xyzw(np.asarray([right_ee[3:6]], dtype=np.float32))[0],
        dtype=torch.float32, device=device,
    )
    l_pos_b, l_quat_b = subtract_frame_transforms(
        root_pos.unsqueeze(0), root_quat.unsqueeze(0),
        tgt_lp.unsqueeze(0), tgt_lq.unsqueeze(0),
    )
    r_pos_b, r_quat_b = subtract_frame_transforms(
        root_pos.unsqueeze(0), root_quat.unsqueeze(0),
        tgt_rp.unsqueeze(0), tgt_rq.unsqueeze(0),
    )
    action = torch.zeros(16, dtype=torch.float32, device=device)
    action[0:3] = l_pos_b[0]
    action[3:7] = l_quat_b[0]
    action[7] = float(left_ee[6])
    action[8:11] = r_pos_b[0]
    action[11:15] = r_quat_b[0]
    action[15] = float(right_ee[6])
    return action


@dataclass
class RobotPolicyConfig:
    model_address: str = 'localhost'
    model_port: int = 8000
    instruction: str = 'pick up the object'

    action_horizon: int = 32
    action_chunk_length: int = 32
    ee_pose_normalize: bool = True

    camera_left: str = 'left_wrist_cam'
    camera_right: str = 'right_wrist_cam'
    camera_front: str = 'head_cam'
    target_image_size: tuple = (480, 640, 3)

    obs_left_follow_key: str = 'follow1_pos'
    obs_right_follow_key: str = 'follow2_pos'
    obs_left_eef_pos_key: str = 'eef_delta_pos'
    obs_left_eef_quat_key: str = 'eef_delta_quat'
    obs_right_eef_pos_key: str = 'right_eef_delta_pos'
    obs_right_eef_quat_key: str = 'right_eef_delta_quat'
    obs_joint_pos_key: str = 'joint_pos'
    obs_left_gripper_index: int = 6
    obs_right_gripper_index: int = 13
    obs_quat_convention: str = 'xyzw'

    pos_gain: float = 1.0
    rot_gain: float = 1.0
    # Desktop H-jaw is already ~0–5.72 in data/sim. Keep 1.0; do not use EX001 G-jaw 1.89.
    gripper_scale: float = 1.0
    # None: use the model gripper as-is. Set a value to snap below it to 0.
    gripper_open_threshold: Optional[float] = None
    gripper_close_threshold: Optional[float] = None


class RobotClosedloopPolicy:
    """Closed-loop EE policy with built-in action chunk buffering."""

    ACTION_DIM = 16

    def __init__(self, config: RobotPolicyConfig):
        self.cfg = config
        self._client = PolicyServerClient(
            config.model_address, config.model_port,
        )
        self._left_init_pos: Optional[torch.Tensor] = None
        self._left_init_quat: Optional[torch.Tensor] = None
        self._right_init_pos: Optional[torch.Tensor] = None
        self._right_init_quat: Optional[torch.Tensor] = None
        self._left_init_follow: Optional[np.ndarray] = None
        self._right_init_follow: Optional[np.ndarray] = None
        self._left_default_follow: Optional[np.ndarray] = None
        self._right_default_follow: Optional[np.ndarray] = None
        self._stale_after_reset = False
        self._query_count = 0

        self._chunk: Optional[torch.Tensor] = None
        self._chunk_idx = -1
        self._needs_chunk = True
        self._cmd_ee_chunk: Optional[np.ndarray] = None
        self.last_cmd_ee: Optional[np.ndarray] = None
        self.last_cmd_xyz: Optional[np.ndarray] = None

    # -- PolicyLike interface --

    def get_actions(
        self, env: Any, observation: Dict[str, Any],
    ) -> torch.Tensor:
        device = (
            env.device if hasattr(env, 'device') else 'cpu'
        )
        num_envs = (
            env.num_envs if hasattr(env, 'num_envs') else 1
        )
        actions = torch.zeros(
            num_envs, self.ACTION_DIM,
            dtype=torch.float32, device=device,
        )

        if self._needs_chunk:
            chunk = self._query_action_chunk(
                env, observation,
            )
            if chunk is not None:
                self._chunk = chunk
                self._chunk_idx = 0
                self._needs_chunk = False

        if self._chunk is not None:
            actions[0] = self._chunk[self._chunk_idx]
            if (
                self._cmd_ee_chunk is not None
                and self._chunk_idx < self._cmd_ee_chunk.shape[0]
            ):
                row = self._cmd_ee_chunk[self._chunk_idx]
                self.last_cmd_ee = row
                self.last_cmd_xyz = np.concatenate(
                    [row[0:3], row[7:10]],
                ).astype(np.float32)
            self._chunk_idx += 1
            if self._chunk_idx >= self._chunk.shape[0]:
                self._chunk_idx = -1
                self._needs_chunk = True

        return actions

    def reset(self, env_ids: Any = None) -> None:
        self._left_init_pos = None
        self._left_init_quat = None
        self._right_init_pos = None
        self._right_init_quat = None
        self._left_init_follow = None
        self._right_init_follow = None
        if env_ids is None:
            self._left_default_follow = None
            self._right_default_follow = None
            self._stale_after_reset = False
        elif (
            self._left_default_follow is not None
            and self._right_default_follow is not None
        ):
            self._stale_after_reset = True
        self._query_count = 0
        if self._chunk is not None:
            self._chunk.zero_()
        self._chunk_idx = -1
        self._needs_chunk = True
        self._cmd_ee_chunk = None
        self.last_cmd_ee = None
        self.last_cmd_xyz = None

    # -- observation helpers --

    @staticmethod
    def _compress_image_b64(image: np.ndarray) -> str:
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not ok:
            return ''
        return base64.b64encode(encoded).decode('utf-8')

    @staticmethod
    def _to_numpy(x: Any) -> Optional[np.ndarray]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim > 1:
            x = x[0]
        return x

    def _get_gripper_from_joint(
        self, policy_obs: Dict[str, Any], side: str,
    ) -> float:
        jp = self._to_numpy(
            policy_obs.get(self.cfg.obs_joint_pos_key),
        )
        if jp is None:
            return 0.0
        idx = (
            self.cfg.obs_left_gripper_index
            if side == 'left'
            else self.cfg.obs_right_gripper_index
        )
        if idx >= len(jp):
            return 0.0
        return float(jp[idx])

    def _apply_gripper_command(self, raw: float) -> float:
        """Scale the model gripper. Optionally snap below open_threshold to 0."""
        value = float(raw) * float(self.cfg.gripper_scale)
        threshold = self.cfg.gripper_open_threshold
        if threshold is not None and value < float(threshold):
            return 0.0
        return value

    def _get_follow_state(
        self, policy_obs: Dict[str, Any], side: str,
    ) -> Optional[np.ndarray]:
        key = (
            self.cfg.obs_left_follow_key
            if side == 'left'
            else self.cfg.obs_right_follow_key
        )
        value = self._to_numpy(policy_obs.get(key))
        if value is not None and value.shape[0] >= 7:
            return value[:7].astype(np.float32)
        return None

    @staticmethod
    def _normalize_one_ee_pose(
        pose: np.ndarray, init_pose: np.ndarray,
    ) -> np.ndarray:
        out = pose.astype(np.float32, copy=True)
        out[0:3] = pose[0:3] - init_pose[0:3]
        rel_rot = (
            Rotation.from_euler('xyz', pose[3:6])
            * Rotation.from_euler('xyz', init_pose[3:6]).inv()
        )
        out[3:6] = rel_rot.as_euler('xyz').astype(np.float32)
        return out

    @staticmethod
    def _denormalize_one_ee_pose(
        pose: np.ndarray, init_pose: np.ndarray,
    ) -> np.ndarray:
        out = pose.astype(np.float32, copy=True)
        out[0:3] = pose[0:3] + init_pose[0:3]
        world_rot = (
            Rotation.from_euler('xyz', pose[3:6])
            * Rotation.from_euler('xyz', init_pose[3:6])
        )
        out[3:6] = world_rot.as_euler('xyz').astype(np.float32)
        return out

    def _normalize_request_follow_states(
        self, left: np.ndarray, right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.cfg.ee_pose_normalize:
            return left, right
        if (
            self._stale_after_reset
            and self._left_default_follow is not None
            and self._right_default_follow is not None
        ):
            left = self._left_default_follow.copy()
            right = self._right_default_follow.copy()
            self._left_init_follow = left.copy()
            self._right_init_follow = right.copy()
            self._stale_after_reset = False
        if self._left_init_follow is None:
            self._left_init_follow = left.copy()
        if self._right_init_follow is None:
            self._right_init_follow = right.copy()
        if self._left_default_follow is None:
            self._left_default_follow = self._left_init_follow.copy()
        if self._right_default_follow is None:
            self._right_default_follow = self._right_init_follow.copy()
        return (
            self._normalize_one_ee_pose(left, self._left_init_follow),
            self._normalize_one_ee_pose(right, self._right_init_follow),
        )

    def _get_camera_b64(
        self, observation: Dict[str, Any],
        cam_name: Optional[str],
    ) -> Optional[str]:
        if cam_name is None:
            return None
        camera_obs = observation.get('camera_obs', {})
        policy_obs = observation.get('policy', {})
        rgb = camera_obs.get(cam_name)
        if rgb is None:
            rgb = policy_obs.get(cam_name)
        if rgb is None:
            return None
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if len(rgb.shape) == 4:
            rgb = rgb[0]
        return self._compress_image_b64(rgb)

    def build_model_input(
        self, observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        po = observation.get('policy', {})

        follow1 = self._get_follow_state(po, 'left')
        follow2 = self._get_follow_state(po, 'right')

        if follow1 is None or follow2 is None:
            follow1, follow2 = self._build_legacy_delta_follow_state(po)

        follow1, follow2 = self._normalize_request_follow_states(
            follow1, follow2,
        )

        cam_left = self._get_camera_b64(
            observation, self.cfg.camera_left,
        )
        cam_right = self._get_camera_b64(
            observation, self.cfg.camera_right,
        )
        cam_front = self._get_camera_b64(
            observation, self.cfg.camera_front,
        )

        views = {}
        if cam_left is not None:
            views['camera_left'] = cam_left
        if cam_front is not None:
            views['camera_front'] = cam_front
        if cam_right is not None:
            views['camera_right'] = cam_right

        return {
            'state': {
                'follow1_pos': follow1,
                'follow2_pos': follow2,
            },
            'views': views,
            'instruction': self.cfg.instruction,
        }

    def _build_legacy_delta_follow_state(
        self, policy_obs: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        lp = self._to_numpy(
            policy_obs.get(self.cfg.obs_left_eef_pos_key),
        )
        lq = self._to_numpy(
            policy_obs.get(self.cfg.obs_left_eef_quat_key),
        )
        rp = self._to_numpy(
            policy_obs.get(self.cfg.obs_right_eef_pos_key),
        )
        rq = self._to_numpy(
            policy_obs.get(self.cfg.obs_right_eef_quat_key),
        )

        if lp is None:
            lp = np.zeros(3, dtype=np.float32)
        if rp is None:
            rp = np.zeros(3, dtype=np.float32)

        if lq is None:
            le = np.zeros(3, dtype=np.float32)
        else:
            if self.cfg.obs_quat_convention == 'wxyz':
                lq = lq[[1, 2, 3, 0]]
            le = Rotation.from_quat(lq).as_euler(
                'xyz',
            ).astype(np.float32)

        if rq is None:
            re = np.zeros(3, dtype=np.float32)
        else:
            if self.cfg.obs_quat_convention == 'wxyz':
                rq = rq[[1, 2, 3, 0]]
            re = Rotation.from_quat(rq).as_euler(
                'xyz',
            ).astype(np.float32)

        lg = self._get_gripper_from_joint(policy_obs, 'left')
        rg = self._get_gripper_from_joint(policy_obs, 'right')
        return (
            np.concatenate(
                [lp.astype(np.float32), le, [lg]],
            ).astype(np.float32),
            np.concatenate(
                [rp.astype(np.float32), re, [rg]],
            ).astype(np.float32),
        )

    # -- EE coordinate transform --

    def _denormalize_response(
        self, rows: np.ndarray, side: str,
    ) -> np.ndarray:
        if not self.cfg.ee_pose_normalize:
            return rows
        init_pose = (
            self._left_init_follow
            if side == 'left'
            else self._right_init_follow
        )
        if init_pose is None:
            return rows
        return np.stack(
            [
                self._denormalize_one_ee_pose(row[:7], init_pose)
                for row in rows
            ],
        ).astype(np.float32)

    @staticmethod
    def _get_env_origin(env: Any, device) -> torch.Tensor:
        origins = getattr(env.scene, 'env_origins', None)
        if origins is None:
            return torch.zeros(3, dtype=torch.float32, device=device)
        return origins[0].to(device=device, dtype=torch.float32)

    @staticmethod
    def _get_robot_root_pose(env: Any):
        robot = env.scene['robot']
        return robot.data.root_pos_w[0], robot.data.root_quat_w[0]

    def _response_to_action_chunk(
        self,
        response: Dict[str, Any],
        env: Any,
        device,
    ) -> Optional[torch.Tensor]:
        f1 = response.get('follow1_pos')
        f2 = response.get('follow2_pos')
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
                '[RobotPolicy] Invalid response shape: '
                f'follow1={f1.shape}, follow2={f2.shape}',
            )
            return None

        horizon = min(
            f1.shape[0], f2.shape[0],
            self.cfg.action_horizon,
        )
        if horizon <= 0:
            return None

        f1 = self._denormalize_response(f1[:horizon, :7], 'left')
        f2 = self._denormalize_response(f2[:horizon, :7], 'right')
        cmd_ee = np.concatenate([f1[:, :7], f2[:, :7]], axis=-1)
        cmd_ee[:, 6] = [
            self._apply_gripper_command(v) for v in cmd_ee[:, 6]
        ]
        cmd_ee[:, 13] = [
            self._apply_gripper_command(v) for v in cmd_ee[:, 13]
        ]
        if horizon < self.cfg.action_horizon:
            cmd_ee = np.concatenate(
                [
                    cmd_ee,
                    np.repeat(
                        cmd_ee[-1:],
                        self.cfg.action_horizon - horizon,
                        axis=0,
                    ),
                ],
                axis=0,
            )
        self._cmd_ee_chunk = cmd_ee.astype(np.float32)

        chunk = torch.zeros(
            self.cfg.action_horizon, self.ACTION_DIM,
            dtype=torch.float32, device=device,
        )

        origin = self._get_env_origin(env, device)
        root_pos, root_quat = self._get_robot_root_pose(env)
        root_pos = root_pos.to(device=device, dtype=torch.float32)
        root_quat = root_quat.to(device=device, dtype=torch.float32)

        for i in range(self.cfg.action_horizon):
            chunk[i] = follow_pair_to_ik16(
                self._cmd_ee_chunk[i, 0:7],
                self._cmd_ee_chunk[i, 7:14],
                origin, root_pos, root_quat, device,
            )

        return chunk

    def _query_action_chunk(
        self, env: Any, observation: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        self._client.ensure_connected()
        model_input = self.build_model_input(observation)
        device = (
            env.device if hasattr(env, 'device') else 'cpu'
        )
        self._query_count += 1

        try:
            response = self._client.predict(model_input)
        except Exception as e:
            print(f'[RobotPolicy] Inference failed: {e}')
            return None

        if 'error' in response:
            print(
                f'[RobotPolicy] Server error: '
                f'{response["error"]}'
            )
            return None

        return self._response_to_action_chunk(
            response, env, device,
        )

    def cleanup(self) -> None:
        self._client.close()

    def __del__(self):
        self.cleanup()
