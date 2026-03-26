"""Explicit builders for the supported Bimanual environments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maniparena_sim.embodiment.robots.bimanual import BimanualEmbodiment
from maniparena_sim.environment.assembled_environment import AssembledEnvironment
from maniparena_sim.environment.render_settings import (
    apply_carb_settings,
    patch_env_cfg_render,
)
from maniparena_sim.environment.scene_builder import build_scene
from maniparena_sim.task.builders.buttons_contact_builder import ButtonsContactBuilder
from maniparena_sim.task.builders.fruits_to_basket_builder import FruitsToBasketBuilder
from maniparena_sim.task.builders.sort_blocks_builder import SortBlocksBuilder
from maniparena_sim.task.buttons_contact import ButtonsContactTaskCFG
from maniparena_sim.task.fruits_to_basket import FruitsToBasketTaskCFG
from maniparena_sim.task.sort_blocks import SortBlocksTaskCFG


@dataclass
class EnvironmentBuildRequest:
    task_name: str
    enable_cameras: bool = True
    num_envs: int = 1
    name: str | None = None


@dataclass
class CollectEnv:
    """All artifacts needed to run a collection loop."""

    gym_env: Any
    task: Any
    success_term: Any
    env_name: str
    output_dir: str
    embodiment: Any = None
    exported_paths: list[str] = field(default_factory=list)
    is_direct_lerobot: bool = False
    prefix: str = ''


SUPPORTED_TASKS = {
    "sort_blocks": (SortBlocksTaskCFG, SortBlocksBuilder),
    "fruits_to_basket": (FruitsToBasketTaskCFG, FruitsToBasketBuilder),
    "buttons_contact": (ButtonsContactTaskCFG, ButtonsContactBuilder),
}


def build_task_runtime(task_name: str, scene):
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task_name}'. Supported: {list(SUPPORTED_TASKS)}")
    cfg_cls, builder_cls = SUPPORTED_TASKS[task_name]
    task_cfg = cfg_cls(task_type=task_name)
    task = builder_cls().finalize(task_cfg, scene=scene)
    task.config = task_cfg
    return task


def build_environment(request: EnvironmentBuildRequest) -> AssembledEnvironment:
    scene = build_scene(request.task_name)
    embodiment = BimanualEmbodiment(enable_cameras=request.enable_cameras)
    if not request.enable_cameras:
        embodiment.camera_config = None
    task = build_task_runtime(request.task_name, scene)
    env_name = request.name or f"bimanual_{request.task_name}_env"
    return AssembledEnvironment(
        name=env_name,
        scene=scene,
        embodiment=embodiment,
        task=task,
        num_envs=request.num_envs,
    )


def build_collect_gym_env(
    task_name: str,
    payload: dict,
    control_mode: str = 'keyboard',
    headless: bool = False,
    device: str = 'cuda:0',
) -> CollectEnv:
    """Build a gym env fully configured for teleop data collection."""
    import gymnasium as gym
    import torch
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.helpers import (  # noqa: E501
        create_bimanual_lerobot_handler_type,
        resolve_bimanual_lerobot_dataset_layout,
    )
    from maniparena_sim.terms.recorders.recording_terms import PreStepCameraObservationsRecorderCfg
    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))

    scene = build_scene(task_name)
    embodiment = BimanualEmbodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    task = build_task_runtime(task_name, scene)

    env_name = f'bimanual_{task_name}_collect'
    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene,
        task=task, teleop_device=None,
    )

    cli_args = get_isaaclab_arena_cli_parser().parse_args([])
    cli_args.headless = headless
    cli_args.device = device
    cli_args.enable_cameras = enable_cameras
    reg_name, env_cfg = ArenaEnvBuilder(arena_env, cli_args).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, 'render_cfg_dict', None) or {},
    )

    if control_mode == 'master_slave':
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

        ac = env_cfg.actions
        ac.arm_action = JointPositionActionCfg(
            asset_name='robot',
            joint_names=['left_arm_joint[1-6]'],
            scale=1.0,
            use_default_offset=False,
        )
        ac.right_arm_action = JointPositionActionCfg(
            asset_name='robot',
            joint_names=['right_arm_joint[1-6]'],
            scale=1.0,
            use_default_offset=False,
        )
        ac.gripper_action = JointPositionActionCfg(
            asset_name='robot',
            joint_names=['left_arm_gripper'],
            scale=1.0,
            use_default_offset=False,
        )
        ac.right_gripper_action = JointPositionActionCfg(
            asset_name='robot',
            joint_names=['right_arm_gripper'],
            scale=1.0,
            use_default_offset=False,
        )

    # -- resolve output paths --
    working_path = str(payload.get('working_path', '~/maniparena_output/recordings/'))
    fmt = str(payload.get('format', 'bimanual_lerobot'))
    device_tag = control_mode
    handler_type = None
    exported_paths: list[str] = []
    is_direct_lerobot = False

    if fmt == 'bimanual_lerobot':
        prompt = task.get_prompt()
        layout = resolve_bimanual_lerobot_dataset_layout(
            save_path=working_path, env_name=env_name, device_name=device_tag,
        )
        handler_type = create_bimanual_lerobot_handler_type(
            save_path=working_path, env_name=env_name,
            task_name=prompt or env_name,
            fps=float(payload.get('step_hz', 30)),
            device_name=device_tag,
        )
        exported_paths = [layout['joint_root'], layout['ee_root']]
        output_dir = str(Path(layout['base_root']).expanduser())
        is_direct_lerobot = True
    else:
        output_dir = str(Path(os.path.expanduser(working_path)) / env_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prefix = f'{env_name}_{device_tag}'
    bootstrap_name = f'{prefix}_bootstrap'

    # -- configure recorder on env_cfg --
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = bootstrap_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    if handler_type is not None:
        env_cfg.recorders.dataset_file_handler_class_type = handler_type
    if enable_cameras:
        env_cfg.recorders.record_camera_obs = PreStepCameraObservationsRecorderCfg()

    if hasattr(env_cfg, 'terminations'):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.concatenate_terms = False

    gym_env = gym.make(reg_name, cfg=env_cfg).unwrapped
    apply_carb_settings(
        getattr(scene, 'render_carb_dict', None) or {},
    )

    if not enable_cameras:
        n_disabled = deactivate_robot_camera_prims(logger=print)
        if n_disabled:
            print(f'deactivated {n_disabled} USD camera prim(s)')

    # -- post-build recorder config --
    rm = gym_env.recorder_manager
    rm.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    rm.cfg.export_in_record_pre_reset = False
    if not is_direct_lerobot and rm._dataset_file_handler is not None:
        rm._dataset_file_handler.close()
        rm._dataset_file_handler = None
    boot_f = os.path.join(output_dir, f'{bootstrap_name}.hdf5')
    if not is_direct_lerobot and os.path.exists(boot_f):
        os.remove(boot_f)

    # -- disable auto-success termination (manual control via keyboard) --
    success_term = getattr(task.termination_cfg, 'success', None)

    def _disabled_success(env, **kwargs):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    tm = gym_env.termination_manager
    if 'success' in tm.active_terms:
        cfg_term = tm.get_term_cfg('success')
        cfg_term.func = _disabled_success
        cfg_term.params = {}
        tm.set_term_cfg('success', cfg_term)

    return CollectEnv(
        gym_env=gym_env, task=task, success_term=success_term,
        env_name=env_name, output_dir=output_dir,
        embodiment=embodiment,
        exported_paths=exported_paths,
        is_direct_lerobot=is_direct_lerobot, prefix=prefix,
    )


@dataclass
class ReplayEnv:
    """Artifacts needed to run a replay loop."""

    gym_env: Any
    task: Any
    env_name: str
    output_dir: str
    exported_paths: list[str] = field(default_factory=list)
    export_lerobot: bool = False


def build_replay_gym_env(
    task_name: str,
    payload: dict,
    replay_mode: str = 'state',
    headless: bool = False,
    device: str = 'cuda:0',
) -> ReplayEnv:
    """Build a gym env configured for replay with LeRobot export."""
    import gymnasium as gym
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.helpers import (
        create_bimanual_lerobot_handler_type,
        resolve_bimanual_lerobot_dataset_layout,
    )
    from maniparena_sim.terms.recorders.recording_terms import PreStepCameraObservationsRecorderCfg

    enable_cameras = bool(payload.get('enable_cameras', True))

    scene = build_scene(task_name)
    embodiment = BimanualEmbodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None

    if replay_mode == 'joint':
        if not hasattr(embodiment, 'JointActionsCfg'):
            raise RuntimeError(
                'Embodiment has no JointActionsCfg for joint replay.'
            )
        embodiment.action_config = embodiment.JointActionsCfg()
    elif replay_mode == 'ee':
        if not hasattr(embodiment, 'ActionsCfg'):
            raise RuntimeError(
                'Embodiment has no ActionsCfg for EE replay.'
            )
        embodiment.action_config = embodiment.ActionsCfg()

    if replay_mode in ('joint', 'ee'):
        ac = embodiment.action_config
        ac.gripper_action = JointPositionActionCfg(
            asset_name='robot',
            joint_names=['left_arm_gripper'],
            scale=1.0, use_default_offset=False,
        )
        ac.right_gripper_action = JointPositionActionCfg(
            asset_name='robot',
            joint_names=['right_arm_gripper'],
            scale=1.0, use_default_offset=False,
        )

    task = build_task_runtime(task_name, scene)
    env_name = f'bimanual_{task_name}_replay'

    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene,
        task=task, teleop_device=None,
    )
    cli_args = get_isaaclab_arena_cli_parser().parse_args([])
    cli_args.headless = headless
    cli_args.device = device
    cli_args.enable_cameras = enable_cameras
    reg_name, env_cfg = ArenaEnvBuilder(
        arena_env, cli_args,
    ).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, 'render_cfg_dict', None) or {},
    )

    export_lerobot = bool(payload.get('export_lerobot', False))

    working_path = str(
        payload.get(
            'working_path',
            '~/maniparena_output/recordings/',
        )
    )
    exported_paths: list[str] = []
    output_dir = str(
        Path(os.path.expanduser(working_path)) / env_name
    )

    if export_lerobot:
        prompt = task.get_prompt()
        layout = resolve_bimanual_lerobot_dataset_layout(
            save_path=working_path,
            env_name=env_name,
        )
        handler_type = create_bimanual_lerobot_handler_type(
            save_path=working_path,
            env_name=env_name,
            task_name=prompt or env_name,
            fps=float(payload.get('step_hz', 30)),
        )
        exported_paths = [
            layout['joint_root'], layout['ee_root'],
        ]
        output_dir = str(
            Path(layout['base_root']).expanduser()
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if export_lerobot:
        bootstrap_name = f'{env_name}_bootstrap'
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = bootstrap_name
        env_cfg.recorders.dataset_export_mode = (
            DatasetExportMode.EXPORT_ALL
        )
        env_cfg.recorders.dataset_file_handler_class_type = (
            handler_type
        )
        if enable_cameras:
            env_cfg.recorders.record_camera_obs = (
                PreStepCameraObservationsRecorderCfg()
            )

    if hasattr(env_cfg, 'terminations'):
        env_cfg.terminations.time_out = None
    if (
        hasattr(env_cfg, 'observations')
        and hasattr(env_cfg.observations, 'policy')
    ):
        env_cfg.observations.policy.concatenate_terms = False

    gym_env = gym.make(reg_name, cfg=env_cfg).unwrapped
    apply_carb_settings(
        getattr(scene, 'render_carb_dict', None) or {},
    )

    if export_lerobot:
        rm = gym_env.recorder_manager
        rm.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        rm.cfg.export_in_record_pre_reset = False

    return ReplayEnv(
        gym_env=gym_env, task=task,
        env_name=env_name, output_dir=output_dir,
        exported_paths=exported_paths,
        export_lerobot=export_lerobot,
    )


@dataclass
class EvalEnv:
    """Artifacts needed to run an eval loop."""

    gym_env: Any
    task: Any
    env_name: str
    output_dir: str


def build_eval_gym_env(
    task_name: str,
    payload: dict,
    headless: bool = False,
    device: str = 'cuda:0',
) -> EvalEnv:
    """Build a gym env configured for policy evaluation."""
    import gymnasium as gym
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))

    scene = build_scene(task_name)
    embodiment = BimanualEmbodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    task = build_task_runtime(task_name, scene)

    env_name = f'bimanual_{task_name}_eval'
    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene,
        task=task, teleop_device=None,
    )
    cli_args = get_isaaclab_arena_cli_parser().parse_args([])
    cli_args.headless = headless
    cli_args.device = device
    cli_args.enable_cameras = enable_cameras
    reg_name, env_cfg = ArenaEnvBuilder(
        arena_env, cli_args,
    ).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, 'render_cfg_dict', None) or {},
    )

    if (
        hasattr(env_cfg, 'observations')
        and hasattr(env_cfg.observations, 'policy')
    ):
        env_cfg.observations.policy.concatenate_terms = False

    gym_env = gym.make(reg_name, cfg=env_cfg).unwrapped
    apply_carb_settings(
        getattr(scene, 'render_carb_dict', None) or {},
    )

    if not enable_cameras:
        n = deactivate_robot_camera_prims(logger=print)
        if n:
            print(f'deactivated {n} USD camera prim(s)')

    working_path = str(
        payload.get(
            'working_path',
            '~/maniparena_output/eval/',
        )
    )
    output_dir = str(
        Path(os.path.expanduser(working_path)) / env_name
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return EvalEnv(
        gym_env=gym_env, task=task,
        env_name=env_name, output_dir=output_dir,
    )
