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
    ensure_kit_viewport_color_render,
    patch_env_cfg_render,
)
from maniparena_sim.environment.scene_builder import build_scene
from maniparena_sim.task.builders.buttons_contact_builder import ButtonsContactBuilder
from maniparena_sim.task.builders.fruits_to_basket_builder import FruitsToBasketBuilder
from maniparena_sim.task.builders.dummy_task_builder import DummyTaskBuilder
from maniparena_sim.task.builders.put_bottle_on_woodshelf_builder import PutBottleOnWoodshelfBuilder
from maniparena_sim.task.builders.sort_blocks_builder import SortBlocksBuilder
from maniparena_sim.task.buttons_contact import ButtonsContactTaskCFG
from maniparena_sim.task.fruits_to_basket import FruitsToBasketTaskCFG
from maniparena_sim.task.dummy_task import DummyTaskCFG
from maniparena_sim.task.put_bottle_on_woodshelf import PutBottleOnWoodshelfTaskCFG
from maniparena_sim.task.sort_blocks import SortBlocksTaskCFG


def _arena_builder_cfg(
    *,
    device: str = "cuda:0",
    num_envs: int = 1,
    language_instruction: str | None = None,
):
    """Build typed ArenaEnvBuilderCfg for the installed IsaacLab-Arena API."""
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    return ArenaEnvBuilderCfg(
        num_envs=num_envs,
        device=device,
        language_instruction=language_instruction,
    )


def _apply_viewer_cfg_override(task: Any, payload: dict) -> None:
    """Override task viewer eye/lookat from eval YAML ``viewer_cfg``."""
    viewer = payload.get('viewer_cfg')
    if not viewer:
        return
    eye = viewer.get('eye')
    lookat = viewer.get('lookat')
    if eye is None or lookat is None:
        return
    from isaaclab.envs.common import ViewerCfg

    task.viewer_cfg = ViewerCfg(
        eye=tuple(float(x) for x in eye),
        lookat=tuple(float(x) for x in lookat),
        origin_type=str(viewer.get('origin_type', 'world')),
    )


def _apply_video_recorder_cfg(env_cfg: Any, payload: dict) -> None:
    """Tune Isaac Lab perspective recorder from eval YAML ``video_recorder``."""
    video_cfg = payload.get('video_recorder') or {}
    recorder = getattr(env_cfg, 'video_recorder', None)
    if recorder is None:
        return
    backend = video_cfg.get('backend_source')
    if backend:
        recorder.backend_source = backend
    if 'window_width' in video_cfg:
        recorder.window_width = int(video_cfg['window_width'])
    if 'window_height' in video_cfg:
        recorder.window_height = int(video_cfg['window_height'])


def _make_unwrapped_gym_env(
    reg_name: str,
    env_cfg: Any,
    env_kwargs: dict,
    *,
    render_mode: str | None = None,
):
    """``gym.make`` then re-apply Arena viewer eye/lookat for Lab 6 KitVisualizer.

    ``ViewportCameraController`` sets the camera before ``KitVisualizer.initialize()``,
    which ignores early ``set_camera_view`` calls. Arena's ``ArenaEnvBuilder.make``
    re-applies after init; our builders call ``gym.make`` directly and must do the same
    or the viewport stays at a bad/default pose (often reads as a black screen).

    Also clear Sim6 ``disableColorRender`` after sensors spawn: depth-only cameras
    (EX001 chassis) otherwise black the Kit viewport under ``--viz kit``.
    """
    import gymnasium as gym
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    env = gym.make(
        reg_name, cfg=env_cfg, render_mode=render_mode, **env_kwargs,
    )
    reapply_viewer_cfg(env)
    ensure_kit_viewport_color_render()
    return env.unwrapped


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
    # None → default head main + wrist previews; False → disabled.
    stream_viewports: dict | bool | None = None


SUPPORTED_TASKS = {
    "sort_blocks": (SortBlocksTaskCFG, SortBlocksBuilder),
    "fruits_to_basket": (FruitsToBasketTaskCFG, FruitsToBasketBuilder),
    "buttons_contact": (ButtonsContactTaskCFG, ButtonsContactBuilder),
    "dummy_task": (DummyTaskCFG, DummyTaskBuilder),
    "put_bottle_on_woodshelf": (PutBottleOnWoodshelfTaskCFG, PutBottleOnWoodshelfBuilder),
}


def _payload_episode_length_s(payload: dict | None) -> float | None:
    """Read eval/collect YAML ``episode_length_s`` (seconds)."""
    if not payload:
        return None
    raw = payload.get('episode_length_s')
    if raw is None:
        return None
    value = float(raw)
    if value <= 0.0:
        raise ValueError(f'episode_length_s must be > 0, got {value}')
    return value


def build_task_runtime(
    task_name: str,
    scene,
    *,
    episode_length_s: float | None = None,
):
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task_name}'. Supported: {list(SUPPORTED_TASKS)}")
    cfg_cls, builder_cls = SUPPORTED_TASKS[task_name]
    task_cfg = cfg_cls(task_type=task_name)
    if episode_length_s is not None:
        task_cfg.episode_length_s = episode_length_s
    task = builder_cls().finalize(task_cfg, scene=scene)
    if episode_length_s is not None:
        task.episode_length_s = float(episode_length_s)
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
    import torch
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.helpers import (  # noqa: E501
        create_bimanual_lerobot_handler_type,
        resolve_bimanual_lerobot_dataset_layout,
    )
    from maniparena_sim.terms.recorders.recording_terms import PreStepCameraObservationsRecorderCfg
    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))

    scene = build_scene(task_name, robot='bimanual')
    embodiment = BimanualEmbodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    task = build_task_runtime(task_name, scene)

    env_name = f'bimanual_{task_name}_collect'
    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene,
        task=task, teleop_device=None,
    )

    reg_name, env_cfg, env_kwargs = ArenaEnvBuilder(
        arena_env,
        _arena_builder_cfg(
            device=device,
            num_envs=int(payload.get('num_envs', 1)),
        ),
    ).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, 'render_cfg_dict', None) or {},
        sim_fps=getattr(scene, 'sim_fps', 120),
        render_interval=getattr(scene, 'render_decremental', 2),
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
            fps=float(payload.get('step_hz', 20)),
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

    gym_env = _make_unwrapped_gym_env(reg_name, env_cfg, env_kwargs)
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
        stream_viewports=payload.get('stream_viewports'),
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
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
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
    elif replay_mode in ('ee', 'model_ee'):
        if not hasattr(embodiment, 'ActionsCfg'):
            raise RuntimeError(
                'Embodiment has no ActionsCfg for EE replay.'
            )
        embodiment.action_config = embodiment.ActionsCfg()

    if replay_mode in ('joint', 'ee', 'model_ee'):
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
    reg_name, env_cfg, env_kwargs = ArenaEnvBuilder(
        arena_env,
        _arena_builder_cfg(
            device=device,
            num_envs=int(payload.get('num_envs', 1)),
        ),
    ).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, 'render_cfg_dict', None) or {},
        sim_fps=getattr(scene, 'sim_fps', 120),
        render_interval=getattr(scene, 'render_decremental', 2),
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
            fps=float(payload.get('step_hz', 20)),
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

    gym_env = _make_unwrapped_gym_env(reg_name, env_cfg, env_kwargs)
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
    recording: Any = None


def build_eval_gym_env(
    task_name: str,
    payload: dict,
    headless: bool = False,
    device: str = 'cuda:0',
    *,
    render_mode: str | None = None,
) -> EvalEnv:
    """Build a gym env configured for policy evaluation."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.loops.dataset_export import recording_enabled
    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))
    policy_cfg = payload.get('policy_config') or {}
    instruction = policy_cfg.get('instruction')

    scene = build_scene(task_name)
    embodiment = BimanualEmbodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    task = build_task_runtime(
        task_name, scene,
        episode_length_s=_payload_episode_length_s(payload),
    )
    _apply_viewer_cfg_override(task, payload)

    env_name = f'bimanual_{task_name}_eval'
    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene,
        task=task, teleop_device=None,
    )
    reg_name, env_cfg, env_kwargs = ArenaEnvBuilder(
        arena_env,
        _arena_builder_cfg(
            device=device,
            num_envs=int(payload.get('num_envs', 1)),
            language_instruction=(
                str(instruction) if instruction else None
            ),
        ),
    ).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, 'render_cfg_dict', None) or {},
        sim_fps=getattr(scene, 'sim_fps', 120),
        render_interval=getattr(scene, 'render_decremental', 2),
    )
    _apply_video_recorder_cfg(env_cfg, payload)
    if hasattr(task, 'get_episode_length_s'):
        env_cfg.episode_length_s = float(task.get_episode_length_s())

    if (
        hasattr(env_cfg, 'observations')
        and hasattr(env_cfg.observations, 'policy')
    ):
        env_cfg.observations.policy.concatenate_terms = False

    recording = None
    if recording_enabled(payload):
        from maniparena_sim.loops.dataset_export import (
            configure_env_recorder,
            dataset_fps_from_scene,
            finalize_bimanual_recorder,
            resolve_recording_layout,
        )

        working_path = str(
            payload.get('working_path', '~/maniparena_output/eval/'),
        )
        fmt = str(payload.get('format', 'bimanual_lerobot'))
        recording = resolve_recording_layout(
            working_path=working_path,
            env_name=env_name,
            task_prompt=str(instruction or task.get_prompt()),
            device_tag='eval',
            fmt=fmt,
            dataset_fps=dataset_fps_from_scene(payload, scene),
        )
        boot_f = os.path.join(
            recording.output_dir, f'{recording.bootstrap_name}.hdf5',
        )
        if os.path.exists(boot_f):
            os.remove(boot_f)
        configure_env_recorder(
            env_cfg, recording, enable_cameras=enable_cameras,
            export_on_reset=True,
        )

    gym_env = _make_unwrapped_gym_env(
        reg_name, env_cfg, env_kwargs, render_mode=render_mode,
    )
    apply_carb_settings(
        getattr(scene, 'render_carb_dict', None) or {},
    )

    if recording is not None:
        # Must export in record_pre_reset: step()-side auto-reset clears the
        # episode buffer in recorder_manager.reset() before eval can flush.
        finalize_bimanual_recorder(
            gym_env, recording, export_on_reset=True,
        )

    if not enable_cameras:
        n = deactivate_robot_camera_prims(logger=print)
        if n:
            print(f'deactivated {n} USD camera prim(s)')

    output_dir = (
        recording.output_dir if recording is not None
        else str(
            Path(os.path.expanduser(
                payload.get('working_path', '~/maniparena_output/eval/'),
            )) / env_name
        )
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return EvalEnv(
        gym_env=gym_env, task=task,
        env_name=env_name, output_dir=output_dir,
        recording=recording,
    )


EX001_SUPPORTED_TASKS = (
    "sort_blocks",
    "fruits_to_basket",
    "buttons_contact",
    "dummy_task",
    "put_bottle_on_woodshelf",
)


def _apply_ex001_task_spawn(embodiment, task_name: str) -> None:
    """Override EX001 root spawn for task-specific layouts.

    Must set ``embodiment.initial_pose`` (not only ``init_state``): Arena builds
    a reset-mode event from ``initial_pose``. Without it, keyboard ``R`` /
    ``gym_env.reset()`` restores joints but leaves the mobile base where it was.
    """
    if task_name != "put_bottle_on_woodshelf":
        return
    from isaaclab_arena.utils.pose import Pose

    from maniparena_sim.environment.scene_builder import (
        PUT_BOTTLE_EX001_INIT_POS,
        PUT_BOTTLE_EX001_INIT_QUAT_XYZW,
    )

    pose = Pose(
        position_xyz=PUT_BOTTLE_EX001_INIT_POS,
        rotation_xyzw=PUT_BOTTLE_EX001_INIT_QUAT_XYZW,
    )
    # Construction spawn only; episode reset samples robot/bottle in the task.
    embodiment.set_initial_pose(pose, create_reset_event=False)
    embodiment.scene_config.robot.init_state.pos = PUT_BOTTLE_EX001_INIT_POS
    embodiment.scene_config.robot.init_state.rot = PUT_BOTTLE_EX001_INIT_QUAT_XYZW


def build_ex001_collect_gym_env(
    task_name: str,
    payload: dict,
    control_mode: str = 'keyboard',
    headless: bool = False,
    device: str = 'cuda:0',
) -> CollectEnv:
    """Build a gym env configured for ex001 whole-body teleop collection."""
    if task_name not in EX001_SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported ex001 task '{task_name}'. "
            f"Supported: {list(EX001_SUPPORTED_TASKS)}"
        )
    import torch
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.embodiment.robots.ex001 import EX001Embodiment
    from maniparena_sim.terms.recorders.recording_terms import PreStepCameraObservationsRecorderCfg
    from maniparena_sim.terms.recorders.streaming.hdf5_handler import PlainHDF5DatasetFileHandler
    from maniparena_sim.terms.recorders.streaming.install import (
        install_async_writer, install_staging, install_sync_export_sidecar_finalize,
    )
    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))

    scene = build_scene(task_name, robot='ex001')
    embodiment = EX001Embodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    # Keyboard integrates deltas into an absolute DiffIK hold pose.
    if control_mode == 'keyboard':
        embodiment.action_config = embodiment.ActionsCfgAbsIK()
    else:
        embodiment.action_config = embodiment.ActionsCfgAbsIK()
    _apply_ex001_task_spawn(embodiment, task_name)
    task = build_task_runtime(task_name, scene)

    env_name = f'ex001_{task_name}_collect'
    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene, task=task, teleop_device=None,
    )
    reg_name, env_cfg, env_kwargs = ArenaEnvBuilder(
        arena_env,
        _arena_builder_cfg(
            device=device,
            num_envs=int(payload.get('num_envs', 1)),
        ),
    ).build_registered()
    patch_env_cfg_render(
        env_cfg, getattr(scene, 'render_cfg_dict', None) or {},
        sim_fps=getattr(scene, 'sim_fps', 120), render_interval=getattr(scene, 'render_decremental', 2),
    )

    working_path = str(payload.get('working_path', '~/maniparena_output/recordings/'))
    output_dir = str(Path(os.path.expanduser(working_path)) / env_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prefix = f'{env_name}_{control_mode}'
    bootstrap_name = f'{prefix}_bootstrap'
    boot_f = os.path.join(output_dir, f'{bootstrap_name}.hdf5')
    # RecorderManager opens dataset_filename during gym.make(); drop any stale
    # bootstrap left by a previous crashed run before we recreate it.
    if os.path.exists(boot_f):
        os.remove(boot_f)

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = bootstrap_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.recorders.dataset_file_handler_class_type = PlainHDF5DatasetFileHandler
    if enable_cameras:
        env_cfg.recorders.record_camera_obs = PreStepCameraObservationsRecorderCfg()

    if hasattr(env_cfg, 'terminations'):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.concatenate_terms = False

    gym_env = _make_unwrapped_gym_env(reg_name, env_cfg, env_kwargs)
    apply_carb_settings(getattr(scene, 'render_carb_dict', None) or {})
    if not enable_cameras:
        n_disabled = deactivate_robot_camera_prims(logger=print)
        if n_disabled:
            print(f'deactivated {n_disabled} USD camera prim(s)')

    # Streaming recorder: dump obs GPU->CPU per step, stream camera frames to MP4.
    rm = gym_env.recorder_manager
    rm.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    rm.cfg.export_in_record_pre_reset = False
    video_fps = float(payload.get('step_hz', 20))
    install_staging(rm, pinned_memory=True, chunk_size=int(payload.get('stager_chunk_size', 32)), video_fps=video_fps)
    # async_writer defaults OFF: the generic collection loop closes the
    # per-episode handler synchronously right after export_episodes(), which
    # would race a background write. Staging + sidecar MP4 run either way.
    if bool(payload.get('async_writer', False)):
        install_async_writer(rm, max_queue=int(payload.get('async_max_queue', 8)),
                             pinned_memory=True, chunk_size=int(payload.get('stager_chunk_size', 32)))
    else:
        install_sync_export_sidecar_finalize(rm)

    # Discard the throwaway bootstrap HDF5 opened at env init (see bootstrap_name).
    if os.path.exists(boot_f):
        os.remove(boot_f)

    # Disable auto-success (episodes ended manually).
    def _disabled_success(env, **kwargs):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    tm = gym_env.termination_manager
    if 'success' in tm.active_terms:
        cfg_term = tm.get_term_cfg('success')
        cfg_term.func = _disabled_success
        cfg_term.params = {}
        tm.set_term_cfg('success', cfg_term)

    return CollectEnv(
        gym_env=gym_env, task=task, success_term=None,
        env_name=env_name, output_dir=output_dir, embodiment=embodiment,
        exported_paths=[output_dir], is_direct_lerobot=False, prefix=prefix,
        stream_viewports=payload.get('stream_viewports'),
    )


def build_ex001_nav_gym_env(
    payload: dict,
    headless: bool = False,
    device: str = 'cuda:0',
):
    """Build an ex001 gym env for ROS2 navigation (no recorder, dummy_task scene).

    Returns ``(gym_env, embodiment)``. Navigation does not record data, so this
    intentionally skips the RecorderManager streaming stack used by collection.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.embodiment.robots.ex001 import EX001Embodiment

    enable_cameras = True
    scene = build_scene('dummy_task', robot='ex001')
    embodiment = EX001Embodiment(enable_cameras=enable_cameras)
    # Nav spawns on the floor: drop the tabletop z-offset so the base sits at z=0.
    _p = embodiment.scene_config.robot.init_state.pos
    embodiment.scene_config.robot.init_state.pos = (_p[0], _p[1], 0.0)
    # Joint-position hold over non-wheel joints; chassis driven directly to sim.
    embodiment.action_config = embodiment.ActionsCfgNav()
    task = build_task_runtime('dummy_task', scene)

    env_name = 'ex001_nav'
    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene, task=task, teleop_device=None,
    )
    reg_name, env_cfg, env_kwargs = ArenaEnvBuilder(
        arena_env, _arena_builder_cfg(device=device),
    ).build_registered()
    patch_env_cfg_render(
        env_cfg, getattr(scene, 'render_cfg_dict', None) or {},
        sim_fps=getattr(scene, 'sim_fps', 120), render_interval=getattr(scene, 'render_decremental', 2),
    )
    # No recorder for navigation.
    env_cfg.recorders = None
    if hasattr(env_cfg, 'terminations'):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.concatenate_terms = False

    gym_env = _make_unwrapped_gym_env(reg_name, env_cfg, env_kwargs)
    apply_carb_settings(getattr(scene, 'render_carb_dict', None) or {})
    return gym_env, embodiment


def build_ex001_eval_gym_env(
    task_name: str,
    payload: dict,
    headless: bool = False,
    device: str = 'cuda:0',
    *,
    render_mode: str | None = None,
) -> EvalEnv:
    """Build an EX001 gym env for Wall-X whole-body (21D) policy evaluation."""
    if task_name not in EX001_SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported ex001 task '{task_name}'. "
            f"Supported: {list(EX001_SUPPORTED_TASKS)}"
        )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.loops.dataset_export import recording_enabled
    from maniparena_sim.embodiment.robots.ex001 import EX001Embodiment
    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))
    policy_cfg = payload.get('policy_config') or {}
    instruction = policy_cfg.get('instruction')

    scene = build_scene(task_name, robot='ex001')
    embodiment = EX001Embodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    embodiment.action_config = embodiment.ActionsCfgWallxWholebody()
    _apply_ex001_task_spawn(embodiment, task_name)
    task = build_task_runtime(
        task_name, scene,
        episode_length_s=_payload_episode_length_s(payload),
    )
    _apply_viewer_cfg_override(task, payload)

    env_name = f'ex001_{task_name}_eval'
    arena_env = IsaacLabArenaEnvironment(
        name=env_name, embodiment=embodiment, scene=scene,
        task=task, teleop_device=None,
    )
    reg_name, env_cfg, env_kwargs = ArenaEnvBuilder(
        arena_env,
        _arena_builder_cfg(
            device=device,
            num_envs=int(payload.get('num_envs', 1)),
            language_instruction=(
                str(instruction) if instruction else None
            ),
        ),
    ).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, 'render_cfg_dict', None) or {},
        sim_fps=getattr(scene, 'sim_fps', 120),
        render_interval=getattr(scene, 'render_decremental', 2),
    )
    _apply_video_recorder_cfg(env_cfg, payload)
    if hasattr(task, 'get_episode_length_s'):
        env_cfg.episode_length_s = float(task.get_episode_length_s())
    if hasattr(env_cfg, 'terminations'):
        env_cfg.terminations.time_out = None
    if (
        hasattr(env_cfg, 'observations')
        and hasattr(env_cfg.observations, 'policy')
    ):
        env_cfg.observations.policy.concatenate_terms = False

    recording = None
    if recording_enabled(payload):
        from maniparena_sim.loops.dataset_export import (
            RecordingLayout,
            configure_env_recorder,
            finalize_ex001_recorder,
        )
        from maniparena_sim.terms.recorders.streaming.hdf5_handler import PlainHDF5DatasetFileHandler

        working_path = str(
            payload.get('working_path', '~/maniparena_output/eval/'),
        )
        output_dir = str(Path(os.path.expanduser(working_path)) / env_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        prefix = f'{env_name}_eval'
        bootstrap_name = f'{prefix}_bootstrap'
        boot_f = os.path.join(output_dir, f'{bootstrap_name}.hdf5')
        if os.path.exists(boot_f):
            os.remove(boot_f)
        recording = RecordingLayout(
            env_name=env_name,
            output_dir=output_dir,
            prefix=prefix,
            bootstrap_name=bootstrap_name,
            exported_paths=[output_dir],
            handler_type=PlainHDF5DatasetFileHandler,
            fmt='hdf5',
        )
        configure_env_recorder(
            env_cfg,
            recording,
            enable_cameras=enable_cameras,
            handler_type=PlainHDF5DatasetFileHandler,
            export_on_reset=True,
        )

    gym_env = _make_unwrapped_gym_env(
        reg_name, env_cfg, env_kwargs, render_mode=render_mode,
    )
    apply_carb_settings(
        getattr(scene, 'render_carb_dict', None) or {},
    )

    if recording is not None:
        # Same as bimanual: flush in record_pre_reset before buffer clear.
        finalize_ex001_recorder(
            gym_env, recording, payload, export_on_reset=True,
        )

    if not enable_cameras:
        n = deactivate_robot_camera_prims(logger=print)
        if n:
            print(f'deactivated {n} USD camera prim(s)')

    output_dir = (
        recording.output_dir if recording is not None
        else str(
            Path(os.path.expanduser(
                payload.get('working_path', '~/maniparena_output/eval/'),
            )) / env_name
        )
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return EvalEnv(
        gym_env=gym_env, task=task,
        env_name=env_name, output_dir=output_dir,
        recording=recording,
    )
