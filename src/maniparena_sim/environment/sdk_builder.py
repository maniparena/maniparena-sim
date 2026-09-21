"""Build SDK-only robot environments without recording or policy runtimes."""

from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import urlparse

from maniparena_sim.assets import ASSETS_DIR

SUPPORTED_SDK_ROBOTS = ("quanta_x1", "quanta_x2", "artixon_arm_6a")


def sdk_robot_type(payload: dict) -> str:
    robot = payload.get("robot") or {}
    value = robot.get("type", "quanta_x1") if isinstance(robot, dict) else robot
    key = str(value).strip().lower()
    if key not in SUPPORTED_SDK_ROBOTS:
        raise ValueError(f"Unsupported SDK robot {value!r}; expected one of {SUPPORTED_SDK_ROBOTS}.")
    return key


def resolve_sdk_usd_path(robot_type: str, robot_cfg: dict) -> str:
    """Resolve local assets or an explicitly supplied remote USD reference."""
    default = {
        "quanta_x1": Path(ASSETS_DIR) / "quanta_x1" / "quanta_x1.usd",
        "quanta_x2": Path(ASSETS_DIR) / "quanta_x2" / "quanta_x2.usd",
        "artixon_arm_6a": Path(ASSETS_DIR) / "bimanual_robot" / "bimanual_robot.usd",
    }[robot_type]
    value = str(robot_cfg.get("usd_path") or default).strip()
    if urlparse(value).scheme in {"http", "https", "omniverse"}:
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(ASSETS_DIR).parent / path
    if not path.is_file():
        raise FileNotFoundError(
            f"{robot_type} USD asset not found: {path}. Set robot.usd_path to a complete "
            "assembled robot USD with its referenced meshes and textures. "
            "For the bundled assets, run git lfs pull in the repository."
        )
    return str(path.resolve())


def sdk_simulation_timing(payload: dict) -> tuple[float, int, int]:
    """Return physics dt, action decimation, and physics steps between renders."""
    cfg = payload.get("simulation") or {}
    dt = float(cfg.get("physics_dt", 1.0 / 120.0))
    decimation = cfg.get("decimation", 6)
    render_interval = cfg.get("render_interval", 2)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("simulation.physics_dt must be a positive finite number.")
    for name, value in (
        ("decimation", decimation),
        ("render_interval", render_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"simulation.{name} must be a positive integer.")
    return dt, decimation, render_interval


def _configure_robot(embodiment, robot_cfg: dict) -> None:
    initial = embodiment.scene_config.robot.init_state
    for key, attr, length in (("position_xyz", "pos", 3), ("rotation_xyzw", "rot", 4)):
        if key not in robot_cfg:
            continue
        values = tuple(float(x) for x in robot_cfg[key])
        if len(values) != length or not all(math.isfinite(x) for x in values):
            raise ValueError(f"robot.{key} requires {length} finite numbers.")
        if attr == "rot":
            norm = math.sqrt(sum(x * x for x in values))
            if norm <= 1e-8:
                raise ValueError("robot.rotation_xyzw must be a nonzero quaternion.")
            values = tuple(x / norm for x in values)
        setattr(initial, attr, values)
    for name, path in (robot_cfg.get("camera_paths") or {}).items():
        camera = getattr(embodiment.camera_config, name, None)
        if camera is None:
            raise ValueError(f"Unknown or disabled SDK camera {name!r}.")
        path = str(path).strip()
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("robot.camera_paths values must be paths relative to the Robot prim.")
        camera.prim_path = "{ENV_REGEX_NS}/Robot/" + path


def _configure_bimanual_sdk(embodiment, arm_control: str) -> None:
    """Adapt the existing desktop actions to the SDK's flange and joint contract."""
    if arm_control == "joint":
        actions = embodiment.JointActionsCfg()
        # SDK grippers accept continuous angles; reuse the existing raw terms.
        actions.gripper_action = embodiment.action_config.gripper_action
        actions.right_gripper_action = embodiment.action_config.right_gripper_action
        embodiment.action_config = actions
    for side, term_name in (("left", "arm_action"), ("right", "right_arm_action")):
        arm = getattr(embodiment.action_config, term_name)
        arm.joint_names = [f"{side}_arm_joint{i}" for i in range(1, 7)]
        if arm_control == "joint":
            arm.preserve_order = True
        else:
            arm.body_name = f"{side}_arm_link6"
        frame = getattr(embodiment.scene_config, f"{side}_ee_frame")
        frame.prim_path = f"{{ENV_REGEX_NS}}/Robot/{side}_arm_link6"
        frame.target_frames[0].prim_path = frame.prim_path


def build_sdk_ros2_gym_env(payload: dict, headless: bool = False, device: str = "cpu"):
    """Return ``(gym_env, embodiment)`` for the selected public SDK robot."""
    robot_type = sdk_robot_type(payload)
    if robot_type == "quanta_x1":
        from maniparena_sim.environment.builder import build_quanta_x1_sdk_ros2_gym_env

        return build_quanta_x1_sdk_ros2_gym_env(payload, headless=headless, device=device)
    robot_cfg = payload.get("robot") or {}
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}
    usd_path = resolve_sdk_usd_path(robot_type, robot_cfg)
    dt, decimation, render_interval = sdk_simulation_timing(payload)
    if int(payload.get("num_envs", 1)) != 1:
        raise ValueError("The SDK ROS2 bridge supports exactly one environment.")

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from maniparena_sim.environment.builder import (
        _apply_viewer_cfg_override,
        _arena_builder_cfg,
        _make_unwrapped_gym_env,
    )
    from maniparena_sim.environment.render_settings import apply_carb_settings, patch_env_cfg_render
    from maniparena_sim.environment.scene_builder import build_scene
    from maniparena_sim.ros.quanta_x1_sdk_topics import normalize_arm_control
    from maniparena_sim.task.sdk_task import SdkTask, configure_sdk_session

    enable_cameras = bool(payload.get("enable_cameras", True))
    arm_control = normalize_arm_control((payload.get("ros") or {}).get("arm_control", "ee"))
    if robot_type == "quanta_x2":
        from maniparena_sim.embodiment.robots.quanta_x2 import QuantaX2Embodiment

        embodiment = QuantaX2Embodiment(usd_path=usd_path, enable_cameras=enable_cameras)
        embodiment.prepare_sdk(arm_control)
    else:
        from maniparena_sim.embodiment.robots.bimanual import BimanualEmbodiment

        embodiment = BimanualEmbodiment(enable_cameras=enable_cameras)
        embodiment.scene_config.robot.spawn.usd_path = usd_path
        if not enable_cameras:
            embodiment.camera_config = None
        _configure_bimanual_sdk(embodiment, arm_control)
    _configure_robot(embodiment, robot_cfg)

    scene_name = "fruits_to_basket" if robot_type == "artixon_arm_6a" else "dummy_task"
    scene = build_scene(scene_name)
    task = SdkTask()
    _apply_viewer_cfg_override(task, payload)
    arena_env = IsaacLabArenaEnvironment(
        name=f"{robot_type}_sdk_ros2",
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=None,
        env_cfg_callback=configure_sdk_session,
    )
    reg_name, env_cfg, env_kwargs = ArenaEnvBuilder(
        arena_env,
        _arena_builder_cfg(device=device),
    ).build_registered()
    patch_env_cfg_render(
        env_cfg,
        getattr(scene, "render_cfg_dict", None) or {},
        sim_fps=1.0 / dt,
        render_interval=render_interval,
    )
    env_cfg.decimation = decimation
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = False
    gym_env = _make_unwrapped_gym_env(reg_name, env_cfg, env_kwargs)
    apply_carb_settings(getattr(scene, "render_carb_dict", None) or {})
    return gym_env, embodiment
