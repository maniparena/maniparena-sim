"""Runtime environment wrapper for collection and validation."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from maniparena_sim.environment.render_settings import apply_carb_settings, patch_env_cfg_render
from maniparena_sim.terms.recorders.recording_terms import PreStepCameraObservationsRecorderCfg


class AssembledEnvironment:
    """Wrap a scene, embodiment and task into a lazily built gym env."""

    def __init__(self, name: str, scene: Any, embodiment: Any, task: Any, teleop_device: Any = None, orchestrator: Any = None, num_envs: int = 1):
        self.name = name
        self.scene = scene
        self.embodiment = embodiment
        self.task = task
        self.teleop_device = teleop_device
        self.orchestrator = orchestrator
        self.num_envs = num_envs
        self._arena_env = None
        self._env_builder = None
        self._gym_env = None
        self.is_built = False
        self.is_running = False

    @property
    def default_num_episodes(self) -> int | None:
        task_cfg = getattr(self.task, "config", None)
        return getattr(task_cfg, "num_episodes", None) if task_cfg is not None else None

    @property
    def dataset_fps(self) -> float | None:
        sim_fps = getattr(self.scene, "sim_fps", None)
        render_decremental = getattr(self.scene, "render_decremental", None)
        if sim_fps and render_decremental:
            return float(sim_fps) / float(render_decremental)
        return None

    def get_prompt(self) -> str:
        return self.task.get_prompt()

    def build_arena_environment(self):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

        self._arena_env = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=self.embodiment,
            scene=self.scene,
            task=self.task,
            teleop_device=self.teleop_device,
            orchestrator=self.orchestrator,
        )
        return self._arena_env

    def build_gym_environment(
        self,
        args: argparse.Namespace = None,
        output_dir: str = None,
        output_file_name: str = None,
        enable_recording: bool = False,
        dataset_file_handler_class_type: type | None = None,
        mode: str = "collect",
    ):
        if self._arena_env is None:
            self.build_arena_environment()
        if self._arena_env is None:
            return None
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        import gymnasium as gym

        args = args or self._create_default_args()
        self._env_builder = ArenaEnvBuilder(self._arena_env, args)
        env_name, env_cfg = self._env_builder.build_registered()

        render_cfg_dict = getattr(self.scene, "render_cfg_dict", None) or {}
        patch_env_cfg_render(env_cfg, render_cfg_dict)
        task_seed = self._resolve_task_seed()
        if task_seed is not None and hasattr(env_cfg, "seed"):
            env_cfg.seed = task_seed
        if mode == "collect":
            self._apply_collection_overrides(
                env_cfg,
                output_dir=output_dir,
                output_file_name=output_file_name,
                enable_recording=enable_recording,
                dataset_file_handler_class_type=dataset_file_handler_class_type,
            )
        elif mode == "evaluate":
            self._apply_evaluation_overrides(env_cfg, output_dir, output_file_name)
        self._gym_env = gym.make(env_name, cfg=env_cfg).unwrapped
        self._run_post_spawn_hooks()

        apply_carb_settings(getattr(self.scene, "render_carb_dict", None) or {})
        self.is_built = True
        return self._gym_env

    def _apply_collection_overrides(self, env_cfg, output_dir: str | None = None, output_file_name: str | None = None, enable_recording: bool = False, dataset_file_handler_class_type: type | None = None):
        if enable_recording and output_dir and output_file_name:
            self._configure_recorder(env_cfg, output_dir, output_file_name, dataset_file_handler_class_type=dataset_file_handler_class_type)
        if hasattr(env_cfg, "terminations"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.concatenate_terms = False

    def _apply_evaluation_overrides(self, env_cfg, output_dir: str | None = None, output_file_name: str | None = None):
        if output_dir and hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
            env_cfg.recorders.dataset_export_dir_path = output_dir
            if output_file_name:
                env_cfg.recorders.dataset_filename = output_file_name
            env_cfg.recorders.export_in_record_pre_reset = True
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.concatenate_terms = False

    def _configure_recorder(self, env_cfg, output_dir: str, output_file_name: str, dataset_file_handler_class_type: type | None = None):
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers import DatasetExportMode

        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        if dataset_file_handler_class_type is not None:
            env_cfg.recorders.dataset_file_handler_class_type = dataset_file_handler_class_type
        if getattr(self.embodiment, "enable_cameras", False):
            env_cfg.recorders.record_camera_obs = PreStepCameraObservationsRecorderCfg()

    def _run_post_spawn_hooks(self) -> None:
        hooks = getattr(self.scene, "_post_spawn_hooks", None)
        if not hooks:
            return
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        for hook in hooks:
            hook(stage, self.num_envs)

    def _create_default_args(self) -> argparse.Namespace:
        args = argparse.Namespace()
        args.device = "cuda:0"
        args.num_envs = self.num_envs
        args.seed = self._resolve_task_seed()
        args.disable_fabric = False
        args.mimic = False
        args.xr = False
        args.enable_cameras = getattr(self.embodiment, "enable_cameras", False)
        return args

    def _resolve_task_seed(self) -> int | None:
        task_cfg = getattr(self.task, "config", None)
        return getattr(task_cfg, "seed", None) if task_cfg is not None else None

    def get_env(self):
        if self._gym_env is None:
            self.build_gym_environment()
        return self._gym_env

    def settle_and_export_to_usd(self, output_path: str, settle_steps: int = 120) -> bool:
        env = self.get_env()
        if env is None:
            return False
        if hasattr(env, "reset"):
            env.reset()
        if settle_steps > 0:
            action = self._make_zero_action(env)
            for _ in range(settle_steps):
                env.step(action)
        return self.export_to_usd(output_path)

    def export_to_usd(self, output_path: str) -> bool:
        if not self.is_built or self._gym_env is None:
            return False
        from isaacsim.core.utils import stage as stage_utils

        return bool(stage_utils.save_stage(output_path, save_and_reload_in_place=False))

    def _make_zero_action(self, env):
        action_space = getattr(env, "action_space", None)
        if action_space is None or not hasattr(action_space, "shape"):
            raise RuntimeError("Environment action_space missing")
        return np.zeros(action_space.shape, dtype=getattr(action_space, "dtype", np.float32))

    def cleanup(self):
        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:
                pass
            self._gym_env = None
        self._env_builder = None
        self._arena_env = None
        self.is_built = False
        self.is_running = False
