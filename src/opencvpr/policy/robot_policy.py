"""Self-contained Robot closed-loop EE policy for opencvpr.

Communicates with a remote policy server
via WebSocket (PolicyServerClient) and converts EE predictions
to delta-pose actions for the Desktop embodiment.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
)
from scipy.spatial.transform import Rotation

from opencvpr.policy.server_client import PolicyServerClient
from opencvpr.utils.math_utils import euler_xyz_to_quat_wxyz


@dataclass
class RobotPolicyConfig:
    model_address: str = 'localhost'
    model_port: int = 8000
    instruction: str = 'pick up the object'

    action_horizon: int = 16
    action_chunk_length: int = 4

    camera_left: str = 'left_wrist_cam'
    camera_right: str = 'right_wrist_cam'
    camera_front: str = 'head_cam'
    target_image_size: tuple = (480, 640, 3)

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
    gripper_open_threshold: float = 3.3
    gripper_close_threshold: float = 1.0


class RobotClosedloopPolicy:
    """Closed-loop EE policy with built-in action chunk buffering."""

    ACTION_DIM = 14

    def __init__(self, config: RobotPolicyConfig):
        self.cfg = config
        self._client = PolicyServerClient(
            config.model_address, config.model_port,
        )
        self._left_init_pos: Optional[torch.Tensor] = None
        self._left_init_quat: Optional[torch.Tensor] = None
        self._right_init_pos: Optional[torch.Tensor] = None
        self._right_init_quat: Optional[torch.Tensor] = None
        self._query_count = 0

        self._chunk: Optional[torch.Tensor] = None
        self._chunk_idx = -1
        self._needs_chunk = True

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
            self._chunk_idx += 1
            if self._chunk_idx >= self.cfg.action_chunk_length:
                self._chunk_idx = -1
                self._needs_chunk = True

        return actions

    def reset(self, env_ids: Any = None) -> None:
        self._left_init_pos = None
        self._left_init_quat = None
        self._right_init_pos = None
        self._right_init_quat = None
        self._query_count = 0
        if self._chunk is not None:
            self._chunk.zero_()
        self._chunk_idx = -1
        self._needs_chunk = True

    # -- observation helpers --

    @staticmethod
    def _compress_image_b64(image: np.ndarray) -> str:
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode('.jpg', image)
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
        th, tw = self.cfg.target_image_size[:2]
        if rgb.shape[0] != th or rgb.shape[1] != tw:
            rgb = cv2.resize(
                rgb, (tw, th),
                interpolation=cv2.INTER_AREA,
            )
        return self._compress_image_b64(rgb)

    def build_model_input(
        self, observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        po = observation.get('policy', {})

        lp = self._to_numpy(
            po.get(self.cfg.obs_left_eef_pos_key),
        )
        lq = self._to_numpy(
            po.get(self.cfg.obs_left_eef_quat_key),
        )
        rp = self._to_numpy(
            po.get(self.cfg.obs_right_eef_pos_key),
        )
        rq = self._to_numpy(
            po.get(self.cfg.obs_right_eef_quat_key),
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

        lg = self._get_gripper_from_joint(po, 'left')
        rg = self._get_gripper_from_joint(po, 'right')

        follow1 = np.concatenate(
            [lp.astype(np.float32), le, [lg]],
        ).astype(np.float32)
        follow2 = np.concatenate(
            [rp.astype(np.float32), re, [rg]],
        ).astype(np.float32)

        cam_left = self._get_camera_b64(
            observation, self.cfg.camera_left,
        )
        cam_right = self._get_camera_b64(
            observation, self.cfg.camera_right,
        )
        cam_front = self._get_camera_b64(
            observation, self.cfg.camera_front,
        )
        if cam_front is None:
            h, w = self.cfg.target_image_size[:2]
            cam_front = self._compress_image_b64(
                np.zeros((h, w, 3), dtype=np.uint8),
            )

        return {
            'state': {
                'follow1_pos': follow1,
                'follow2_pos': follow2,
            },
            'views': {
                'camera_left': cam_left,
                'camera_front': cam_front,
                'camera_right': cam_right,
            },
            'instruction': self.cfg.instruction,
        }

    # -- EE coordinate transform --

    @staticmethod
    def _get_current_ee_poses(env: Any):
        left = env.scene['left_ee_frame']
        right = env.scene['right_ee_frame']
        lp = left.data.target_pos_w[0, 0, :].detach().clone()
        lq = left.data.target_quat_w[0, 0, :].detach().clone()
        rp = right.data.target_pos_w[0, 0, :].detach().clone()
        rq = right.data.target_quat_w[0, 0, :].detach().clone()
        return lp, lq, rp, rq

    def _ensure_initial_ee_pose(self, env: Any):
        if self._left_init_pos is not None:
            return
        (
            self._left_init_pos,
            self._left_init_quat,
            self._right_init_pos,
            self._right_init_quat,
        ) = self._get_current_ee_poses(env)

    def _gripper_command(self, val: float) -> float:
        if val >= self.cfg.gripper_open_threshold:
            return 1.0
        if val <= self.cfg.gripper_close_threshold:
            return -1.0
        return 0.0

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

        horizon = min(
            f1.shape[0], f2.shape[0],
            self.cfg.action_horizon,
        )
        if horizon <= 0:
            return None

        self._ensure_initial_ee_pose(env)
        clp, clq, crp, crq = self._get_current_ee_poses(env)

        chunk = torch.zeros(
            self.cfg.action_horizon, self.ACTION_DIM,
            dtype=torch.float32, device=device,
        )

        prev_lp, prev_rp = clp.clone(), crp.clone()
        prev_lq, prev_rq = clq.clone(), crq.clone()

        for i in range(horizon):
            tgt_lp = self._left_init_pos + torch.as_tensor(
                f1[i, 0:3], dtype=torch.float32, device=device,
            )
            tgt_rp = self._right_init_pos + torch.as_tensor(
                f2[i, 0:3], dtype=torch.float32, device=device,
            )
            l_rq = torch.as_tensor(
                euler_xyz_to_quat_wxyz(
                    np.asarray([f1[i, 3:6]], dtype=np.float32),
                )[0],
                dtype=torch.float32, device=device,
            )
            r_rq = torch.as_tensor(
                euler_xyz_to_quat_wxyz(
                    np.asarray([f2[i, 3:6]], dtype=np.float32),
                )[0],
                dtype=torch.float32, device=device,
            )
            tgt_lq = quat_mul(
                l_rq.unsqueeze(0),
                self._left_init_quat.unsqueeze(0),
            )[0]
            tgt_rq = quat_mul(
                r_rq.unsqueeze(0),
                self._right_init_quat.unsqueeze(0),
            )[0]

            ref_lp = clp if i == 0 else prev_lp
            ref_rp = crp if i == 0 else prev_rp
            ref_lq = clq if i == 0 else prev_lq
            ref_rq = crq if i == 0 else prev_rq

            pg = float(self.cfg.pos_gain)
            rg = float(self.cfg.rot_gain)
            l_dp = (tgt_lp - ref_lp) * pg
            r_dp = (tgt_rp - ref_rp) * pg
            l_dr = axis_angle_from_quat(
                quat_mul(
                    tgt_lq.unsqueeze(0),
                    quat_conjugate(ref_lq.unsqueeze(0)),
                ),
            )[0] * rg
            r_dr = axis_angle_from_quat(
                quat_mul(
                    tgt_rq.unsqueeze(0),
                    quat_conjugate(ref_rq.unsqueeze(0)),
                ),
            )[0] * rg

            chunk[i, 0:3] = l_dp
            chunk[i, 3:6] = l_dr
            chunk[i, 6] = self._gripper_command(
                float(f1[i, 6]),
            )
            chunk[i, 7:10] = r_dp
            chunk[i, 10:13] = r_dr
            chunk[i, 13] = self._gripper_command(
                float(f2[i, 6]),
            )

            prev_lp, prev_rp = tgt_lp, tgt_rp
            prev_lq, prev_rq = tgt_lq, tgt_rq

        if horizon < self.cfg.action_horizon:
            chunk[horizon:] = chunk[horizon - 1]

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
