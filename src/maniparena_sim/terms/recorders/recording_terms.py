"""Recorder terms used during collection."""

from __future__ import annotations

import torch
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass


class PreStepCameraObservationsRecorder(RecorderTerm):
    """Record `camera_obs` as uint8 images."""

    def record_pre_step(self):
        obs_buf = getattr(self._env, "obs_buf", None)
        if obs_buf is None:
            return None, None
        camera_data = obs_buf.get("camera_obs")
        if camera_data is None:
            return None, None
        if isinstance(camera_data, dict):
            return "camera_obs", {
                name: tensor.clamp(0, 255).to(torch.uint8) if isinstance(tensor, torch.Tensor) and tensor.ndim >= 3 else tensor
                for name, tensor in camera_data.items()
            }
        if isinstance(camera_data, torch.Tensor):
            return "camera_obs", camera_data.clamp(0, 255).to(torch.uint8)
        return None, None


class PreStepPolicyObservationsRecorder(RecorderTerm):
    """Record policy observations as structured `obs/*` keys."""

    def record_pre_step(self):
        obs_manager = getattr(self._env, "observation_manager", None)
        if obs_manager is not None:
            policy_obs = obs_manager.compute_group("policy", update_history=False)
            if isinstance(policy_obs, dict):
                return "obs", policy_obs
            if isinstance(policy_obs, torch.Tensor):
                return "obs_flat", policy_obs
        obs_buf = getattr(self._env, "obs_buf", None)
        if isinstance(obs_buf, dict):
            policy_obs = obs_buf.get("policy")
            if isinstance(policy_obs, dict):
                return "obs", policy_obs
            if isinstance(policy_obs, torch.Tensor):
                return "obs_flat", policy_obs
        return None, None


@configclass
class PreStepCameraObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepCameraObservationsRecorder


@configclass
class PreStepPolicyObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepPolicyObservationsRecorder
