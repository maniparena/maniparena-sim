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
from maniparena_sim.task.builders.move_bottle_to_bin_builder import MoveBottleToBinBuilder
from maniparena_sim.task.builders.move_fruit_to_basket_builder import MoveFruitToBasketBuilder
from maniparena_sim.task.builders.put_bottle_on_woodshelf_builder import PutBottleOnWoodshelfBuilder
from maniparena_sim.task.builders.quanta_tabletop_builders import (
    QuantaFruitsToBasketBuilder,
    QuantaSortBlocksBuilder,
)
from maniparena_sim.task.builders.sort_blocks_builder import SortBlocksBuilder
from maniparena_sim.task.buttons_contact import ButtonsContactTaskCFG
from maniparena_sim.task.fruits_to_basket import FruitsToBasketTaskCFG
from maniparena_sim.task.dummy_task import DummyTaskCFG
from maniparena_sim.task.mobile_layouts import MOBILE_HEAD_PITCH_INIT, MOBILE_LIFT_INIT
from maniparena_sim.task.mobile_pick_place import MoveBottleToBinTaskCFG, MoveFruitToBasketTaskCFG
from maniparena_sim.task.put_bottle_on_woodshelf import PutBottleOnWoodshelfTaskCFG
from maniparena_sim.task.quanta_tabletop_layouts import TABLETOP_ROBOT
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
    """Override task viewer eye/lookat from YAML ``viewer_cfg``."""
    viewer = payload.get('viewer_cfg')
    if not viewer:
        return
    eye = viewer.get('eye')
    lookat = viewer.get('lookat')
    if eye is None or lookat is None:
        return
    from isaaclab.envs.common import ViewerCfg

    kwargs: dict[str, Any] = {
        'eye': tuple(float(x) for x in eye),
        'lookat': tuple(float(x) for x in lookat),
        'origin_type': str(viewer.get('origin_type', 'world')),
    }
    asset_name = viewer.get('asset_name')
    if asset_name:
        kwargs['asset_name'] = str(asset_name)
    body_name = viewer.get('body_name')
    if body_name:
        kwargs['body_name'] = str(body_name)
    task.viewer_cfg = ViewerCfg(**kwargs)


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
    (QUANTA_X1 chassis) otherwise black the Kit viewport under ``--viz kit``.
    """
    import gymnasium as gym

    env = gym.make(
        reg_name, cfg=env_cfg, render_mode=render_mode, **env_kwargs,
    )
    _reapply_viewer_after_make(env)
    ensure_kit_viewport_color_render()
    return env.unwrapped


def _reapply_viewer_after_make(env: Any) -> None:
    """Re-apply viewer after KitVisualizer init; resolve asset-relative origins."""
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    unwrapped = getattr(env, 'unwrapped', env)
    vcc = getattr(unwrapped, 'viewport_camera_controller', None)
    cfg = getattr(vcc, 'cfg', None) if vcc is not None else None
    origin = getattr(cfg, 'origin_type', None)
    asset_name = getattr(cfg, 'asset_name', None)
    if origin == 'asset_root' and asset_name and vcc is not None:
        try:
            vcc.update_view_to_asset_root(asset_name)
            return
        except Exception:
            pass
    reapply_viewer_cfg(env)


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


# quanta_x1 task table: the mobile tasks and the quanta variants of the shared
# tabletop ids. The bimanual desktop-arm tasks above are intentionally left untouched.
QUANTA_X1_TASKS = {
    "sort_blocks": (SortBlocksTaskCFG, QuantaSortBlocksBuilder),
    "fruits_to_basket": (FruitsToBasketTaskCFG, QuantaFruitsToBasketBuilder),
    "move_bottle_to_bin": (MoveBottleToBinTaskCFG, MoveBottleToBinBuilder),
    "move_fruit_to_basket": (MoveFruitToBasketTaskCFG, MoveFruitToBasketBuilder),
}


def build_task_runtime(task_name: str, scene, robot: str = "bimanual"):
    if robot == "quanta_x1" and task_name in QUANTA_X1_TASKS:
        cfg_cls, builder_cls = QUANTA_X1_TASKS[task_name]
    elif task_name in SUPPORTED_TASKS:
        cfg_cls, builder_cls = SUPPORTED_TASKS[task_name]
    else:
        raise ValueError(f"Unsupported task '{task_name}'. Supported: {list(SUPPORTED_TASKS)}")
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
    task = build_task_runtime(task_name, scene)
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


QUANTA_X1_SUPPORTED_TASKS = (
    "sort_blocks",
    "fruits_to_basket",
    "move_bottle_to_bin",
    "move_fruit_to_basket",
    "put_bottle_on_woodshelf",
    "dummy_task",
)

# Floor-standing (mobile) tasks: construction spawn only, episode resets sample the
# real base pose in the task event.
_QUANTA_X1_MOBILE_TASKS = ("move_bottle_to_bin", "move_fruit_to_basket", "put_bottle_on_woodshelf")
_QUANTA_X1_MOBILE_INIT_POS = (-0.3, 0.0, 0.0)
_QUANTA_X1_MOBILE_JOINT_POS = {
    "^(?!lift_joint$|head_pitch_joint$).*": 0.0,
    "lift_joint": MOBILE_LIFT_INIT,
    "head_pitch_joint": MOBILE_HEAD_PITCH_INIT,
}


_QUANTA_X1_TABLETOP_TASKS = ("sort_blocks", "fruits_to_basket")

# Wall-X eval timing only (collect / SDK keep the scene defaults). Physics matches the
# data-collection env (manaenv, 240 Hz); the env step is 1/30 s so every 30 fps chunk row
# is played for exactly one env step, with no interpolation. Cameras render each step.
_QUANTA_X1_EVAL_SIM_FPS = 240
_QUANTA_X1_EVAL_DECIMATION = 8
_QUANTA_X1_EVAL_RENDER_INTERVAL = 8


def _apply_quanta_x1_task_spawn(embodiment, task_name: str) -> None:
    """Override QUANTA_X1 root spawn for task-specific layouts.

    Must set ``embodiment.initial_pose`` (not only ``init_state``): Arena builds
    a reset-mode event from ``initial_pose``. Without it, keyboard ``R`` /
    ``gym_env.reset()`` restores joints but leaves the mobile base where it was.
    """
    if task_name in _QUANTA_X1_TABLETOP_TASKS:
        # Fixed tabletop spawn: ManipArena-style root pose, lift / head from the task layout.
        spec = TABLETOP_ROBOT[task_name]
        init_state = embodiment.scene_config.robot.init_state
        init_state.pos = spec["pos"]
        init_state.joint_pos = {
            "^(?!lift_joint$|head_pitch_joint$).*": 0.0,
            "lift_joint": spec["lift"],
            "head_pitch_joint": spec["head_pitch"],
        }
        return
    if task_name not in _QUANTA_X1_MOBILE_TASKS:
        return
    from isaaclab_arena.utils.pose import Pose

    from maniparena_sim.environment.scene_builder import (
        PUT_BOTTLE_QUANTA_X1_INIT_POS,
        PUT_BOTTLE_QUANTA_X1_INIT_QUAT_XYZW,
    )

    if task_name == "put_bottle_on_woodshelf":
        position, rotation = PUT_BOTTLE_QUANTA_X1_INIT_POS, PUT_BOTTLE_QUANTA_X1_INIT_QUAT_XYZW
    else:
        position, rotation = _QUANTA_X1_MOBILE_INIT_POS, (0.0, 0.0, 0.0, 1.0)
    # Construction spawn only; episode reset samples the base pose in the task.
    embodiment.set_initial_pose(Pose(position_xyz=position, rotation_xyzw=rotation), create_reset_event=False)
    init_state = embodiment.scene_config.robot.init_state
    init_state.pos = position
    init_state.rot = rotation
    init_state.joint_pos = dict(_QUANTA_X1_MOBILE_JOINT_POS)


# Demo data: lift command 0.379 with measured lift 0.365 on every frame (the joint sags under load),
# while the eval robot spawns at the command (0.379) and has not sagged yet at the first queries.
# The checkpoints read the lift state as a context cue: an unsagged 0.38+ state makes them command
# the sort_blocks lift (0.455) and the arms end up ~7 cm too high. Spawn at the demo state instead.
_QUANTA_X1_EVAL_LIFT_STATE_SAG = 0.014
_QUANTA_X1_EVAL_LIFT_SAG_TASKS = ("fruits_to_basket", "move_bottle_to_bin", "move_fruit_to_basket", "put_bottle_on_woodshelf")


def quanta_x1_demo_lift_command(task_name: str) -> float:
    """Lift command (m) held for the whole episode in the task's demonstrations (not a policy output)."""
    if task_name in _QUANTA_X1_TABLETOP_TASKS:
        return float(TABLETOP_ROBOT[task_name]["lift"])
    return float(MOBILE_LIFT_INIT)


def _apply_eval_lift_sag_init(embodiment, task_name: str) -> None:
    if task_name not in _QUANTA_X1_EVAL_LIFT_SAG_TASKS:
        return
    joint_pos = dict(embodiment.scene_config.robot.init_state.joint_pos)
    joint_pos["lift_joint"] = float(joint_pos["lift_joint"]) - _QUANTA_X1_EVAL_LIFT_STATE_SAG
    embodiment.scene_config.robot.init_state.joint_pos = joint_pos


def build_quanta_x1_collect_gym_env(
    task_name: str,
    payload: dict,
    control_mode: str = 'keyboard',
    headless: bool = False,
    device: str = 'cuda:0',
) -> CollectEnv:
    """Build a gym env configured for quanta_x1 whole-body teleop collection."""
    if task_name not in QUANTA_X1_SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported quanta_x1 task '{task_name}'. "
            f"Supported: {list(QUANTA_X1_SUPPORTED_TASKS)}"
        )
    import torch
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.embodiment.robots.quanta_x1 import QuantaX1Embodiment
    from maniparena_sim.terms.recorders.recording_terms import PreStepCameraObservationsRecorderCfg
    from maniparena_sim.terms.recorders.streaming.hdf5_handler import PlainHDF5DatasetFileHandler
    from maniparena_sim.terms.recorders.streaming.install import (
        install_async_writer, install_staging, install_sync_export_sidecar_finalize,
    )
    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))

    scene = build_scene(task_name, robot='quanta_x1', scene_options=payload.get('scene_options'))
    embodiment = QuantaX1Embodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    # Keyboard integrates deltas into an absolute DiffIK hold pose.
    if control_mode == 'keyboard':
        embodiment.action_config = embodiment.ActionsCfgAbsIK()
    else:
        embodiment.action_config = embodiment.ActionsCfgAbsIK()
    _apply_quanta_x1_task_spawn(embodiment, task_name)
    task = build_task_runtime(task_name, scene, robot='quanta_x1')

    env_name = f'quanta_x1_{task_name}_collect'
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
    # QUANTA_X1 teleop writes the raw HDF5 episode and a native LeRobot
    # parquet dataset together.  ``hdf5`` is an explicit opt-out.
    recording_format = str(payload.get('format', 'hdf5+lerobot')).strip().lower()
    dual_recording = recording_format in {
        'hdf5+lerobot', 'hdf5_lerobot', 'both', 'bimanual_lerobot', 'lerobot',
    }
    lerobot_root = str(Path(output_dir) / 'lerobot')
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
    if dual_recording:
        from maniparena_sim.terms.recorders.dataset_handlers.quanta_x1_dual import (
            create_quanta_x1_dual_handler_type,
        )
        env_cfg.recorders.dataset_file_handler_class_type = create_quanta_x1_dual_handler_type(
            output_root=lerobot_root,
            task_name=task.get_prompt() or env_name,
            fps=float(payload.get('step_hz', 20)),
        )
    else:
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
        exported_paths=[output_dir] + ([lerobot_root] if dual_recording else []),
        is_direct_lerobot=False, prefix=prefix,
        stream_viewports=payload.get('stream_viewports'),
    )


def build_quanta_x1_sdk_ros2_gym_env(
    payload: dict,
    headless: bool = False,
    device: str = 'cuda:0',
):
    """Build a quanta_x1 gym env for SDK ROS2 (no recorder, dummy_task scene).

    Returns ``(gym_env, embodiment)``. SDK ROS2 does not record data, so this
    intentionally skips the RecorderManager streaming stack used by collection.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.embodiment.robots.quanta_x1 import QuantaX1Embodiment

    from maniparena_sim.environment.sdk_builder import apply_actuator_overrides

    enable_cameras = True
    scene = build_scene('dummy_task', robot='quanta_x1')
    embodiment = QuantaX1Embodiment(enable_cameras=enable_cameras)
    embodiment.enable_head_depth()
    robot_cfg = payload.get('robot') or {}
    if robot_cfg.get('usd_path'):
        from maniparena_sim.environment.sdk_builder import resolve_sdk_usd_path

        embodiment.scene_config.robot.spawn.usd_path = resolve_sdk_usd_path('quanta_x1', robot_cfg)
    apply_actuator_overrides(embodiment, robot_cfg)
    # SDK ROS2 spawns on the floor: drop the tabletop z-offset so the base sits at z=0.
    _p = embodiment.scene_config.robot.init_state.pos
    embodiment.scene_config.robot.init_state.pos = (_p[0], _p[1], 0.0)
    from maniparena_sim.ros.quanta_x1_sdk_topics import normalize_arm_control

    arm_control = normalize_arm_control((payload.get("ros") or {}).get("arm_control", "ee"))
    if arm_control == "ee":
        embodiment.action_config = embodiment.ActionsCfgSdkEe()
    else:
        embodiment.action_config = embodiment.ActionsCfgSdkRos2()
    task = build_task_runtime('dummy_task', scene)
    _apply_viewer_cfg_override(task, payload)

    env_name = 'quanta_x1_sdk_ros2'
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
    # No recorder for SDK ROS2.
    env_cfg.recorders = None
    if hasattr(env_cfg, 'terminations'):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.concatenate_terms = False

    gym_env = _make_unwrapped_gym_env(reg_name, env_cfg, env_kwargs)
    apply_carb_settings(getattr(scene, 'render_carb_dict', None) or {})
    return gym_env, embodiment


def build_quanta_x1_eval_gym_env(
    task_name: str,
    payload: dict,
    headless: bool = False,
    device: str = 'cuda:0',
    *,
    render_mode: str | None = None,
) -> EvalEnv:
    """Build an QUANTA_X1 gym env for Wall-X whole-body (21D) policy evaluation."""
    if task_name not in QUANTA_X1_SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported quanta_x1 task '{task_name}'. "
            f"Supported: {list(QUANTA_X1_SUPPORTED_TASKS)}"
        )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.loops.dataset_export import recording_enabled
    from maniparena_sim.embodiment.robots.quanta_x1 import QuantaX1Embodiment
    from maniparena_sim.utils.camera_utils import deactivate_robot_camera_prims

    enable_cameras = bool(payload.get('enable_cameras', True))
    policy_cfg = payload.get('policy_config') or {}
    instruction = policy_cfg.get('instruction')

    scene = build_scene(task_name, robot='quanta_x1', scene_options=payload.get('scene_options'))
    embodiment = QuantaX1Embodiment(enable_cameras=enable_cameras)
    if not enable_cameras:
        embodiment.camera_config = None
    embodiment.disable_chassis_depth()
    embodiment.use_dataset_head_view()
    embodiment.action_config = embodiment.ActionsCfgWallxWholebody()
    _apply_quanta_x1_task_spawn(embodiment, task_name)
    _apply_eval_lift_sag_init(embodiment, task_name)
    task = build_task_runtime(task_name, scene, robot='quanta_x1')
    _apply_viewer_cfg_override(task, payload)

    env_name = f'quanta_x1_{task_name}_eval'
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
        sim_fps=_QUANTA_X1_EVAL_SIM_FPS,
        render_interval=_QUANTA_X1_EVAL_RENDER_INTERVAL,
    )
    env_cfg.decimation = _QUANTA_X1_EVAL_DECIMATION
    _apply_video_recorder_cfg(env_cfg, payload)
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
            finalize_quanta_x1_recorder,
        )
        from maniparena_sim.terms.recorders.streaming.hdf5_handler import PlainHDF5DatasetFileHandler

        working_path = str(
            payload.get('working_path', '~/maniparena_output/eval/'),
        )
        output_dir = str(Path(os.path.expanduser(working_path)) / env_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        recording_format = str(payload.get('format', 'hdf5+lerobot')).strip().lower()
        dual_recording = recording_format in {
            'hdf5+lerobot', 'hdf5_lerobot', 'both', 'bimanual_lerobot', 'lerobot',
        }
        lerobot_root = str(Path(output_dir) / 'lerobot')
        handler_type = PlainHDF5DatasetFileHandler
        if dual_recording:
            from maniparena_sim.terms.recorders.dataset_handlers.quanta_x1_dual import (
                create_quanta_x1_dual_handler_type,
            )
            handler_type = create_quanta_x1_dual_handler_type(
                output_root=lerobot_root,
                task_name=str(instruction or task.get_prompt()) or env_name,
                fps=float(payload.get('step_hz', 20)),
            )
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
            exported_paths=[output_dir] + ([lerobot_root] if dual_recording else []),
            handler_type=handler_type,
            fmt='hdf5+lerobot' if dual_recording else 'hdf5',
        )
        configure_env_recorder(
            env_cfg,
            recording,
            enable_cameras=enable_cameras,
            handler_type=handler_type,
            export_on_reset=True,
        )

    gym_env = _make_unwrapped_gym_env(
        reg_name, env_cfg, env_kwargs, render_mode=render_mode,
    )
    apply_carb_settings(
        getattr(scene, 'render_carb_dict', None) or {},
    )
    if enable_cameras:
        from maniparena_sim.embodiment.sensors.lens_models import enable_usd_lens_models

        print(f'[lens] enabled OpenCV lens model on {enable_usd_lens_models()} camera prim(s)')

    if recording is not None:
        # Same as bimanual: flush in record_pre_reset before buffer clear.
        finalize_quanta_x1_recorder(
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
