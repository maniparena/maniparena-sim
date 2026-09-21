#!/usr/bin/env python3
"""ROS2 SDK runtime for QUANTA_X1, quanta_x2 and ArtiXon Arm-6A.

SDK environments run without recorders. Mobile bases combine keyboard and ROS
velocity commands. Each robot profile defines its joint order, Cartesian
frames, sensors and gripper range. ``ros.arm_control`` selects ``ee`` or ``joint``.

Usage:
    python scripts/sdk_ros2.py --robot artixon_arm_6a --viz kit
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import traceback
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="ManipArena ROS2 SDK.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--config", default=None)
    parser.add_argument("--robot", choices=("quanta_x1", "quanta_x2", "artixon_arm_6a"), default=None)
    # RTX cameras/lidar still render on the NVIDIA GPU.  Keep PhysX tensors on
    # CPU by default because this Sim/Lab stack can otherwise fall back to CPU
    # PhysX while leaving Warp on CUDA, producing ProxyArray type mismatches.
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if args.config is None:
        args.config = str(
            Path(__file__).resolve().parents[1] / "configs" / "sdk_ros2" / f"{args.robot or 'quanta_x1'}_sdk_ros2.yaml"
        )
    return args


def _ensure_isaac_ros2_runtime() -> None:
    """Restart once with Isaac Sim's matching ROS2 shared libraries first."""
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.executable).parent.parent / "lib" / python_version / "site-packages"
    ros2_lib = site_packages / "isaacsim" / "exts" / "isaacsim.ros2.core" / "jazzy" / "lib"
    entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if entries and entries[0] == str(ros2_lib):
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([str(ros2_lib), *filter(None, entries)])
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


class _KeyboardTwist:
    """Subscribe to carb keyboard and expose a (linear_x, angular_z) twist."""

    def __init__(self, linear_velocity: float, angular_velocity: float):
        import carb
        import omni

        self._lin = float(linear_velocity)
        self._ang = float(angular_velocity)
        self._pressed: set[str] = set()
        self.reset_requested = False
        self._carb = carb
        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    @staticmethod
    def _key_name(event) -> str | None:
        # ``event.input`` is a KeyboardInput enum on KEY_PRESS/RELEASE but a raw
        # str on CHAR events; only the enum exposes ``.name``.
        return getattr(event.input, "name", None)

    def _on_event(self, event, *args, **kwargs) -> bool:
        carb = self._carb
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = self._key_name(event)
            if key is None:
                return True
            if key == "R":
                self.reset_requested = True
            else:
                self._pressed.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = self._key_name(event)
            if key is not None:
                self._pressed.discard(key)
        return True

    def twist(self) -> tuple[float, float, float]:
        lin = 0.0
        ang = 0.0
        if "W" in self._pressed:
            lin += self._lin
        if "S" in self._pressed:
            lin -= self._lin
        if "A" in self._pressed or "Q" in self._pressed:
            ang += self._ang
        if "D" in self._pressed or "E" in self._pressed:
            ang -= self._ang
        return lin, 0.0, ang

    def shutdown(self) -> None:
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass


def main(args: argparse.Namespace | None = None) -> int:
    args = args if args is not None else parse_args()
    payload = load_yaml(args.config)
    robot_cfg = payload.setdefault("robot", {})
    profile = args.robot or robot_cfg.get("type", "quanta_x1")
    robot_cfg["type"] = profile

    import torch

    from maniparena_sim.environment.registry import bootstrap_arena_registry
    from maniparena_sim.environment.sdk_builder import build_sdk_ros2_gym_env
    from maniparena_sim.ros.quanta_x1_joint_mapping import build_action_slot_map
    from maniparena_sim.ros.ros2_config import QuantaX1RosConfig
    from maniparena_sim.ros.ros_bridge import RosBridgeExtension, load_ros_bridge_cfg

    bootstrap_arena_registry()
    gym_env, _embodiment = build_sdk_ros2_gym_env(
        payload,
        headless=bool(getattr(args, "headless", False)),
        device=getattr(args, "device", "cuda:0"),
    )
    gym_env.reset()
    robot = gym_env.scene["robot"]

    # Single action vector drives everything: arms/lift/grippers hold absolute
    # joint positions (seeded from defaults so idle keeps the pose), the two
    # wheels take velocity targets. The ROS bridge fills the cmd_vel buffer and
    # writes joint/head commands into this same buffer via the slot map.
    ros_cfg = load_ros_bridge_cfg(args.config)
    actions = torch.zeros(gym_env.num_envs, gym_env.action_manager.total_action_dim, device=gym_env.device)

    slot_map = build_action_slot_map(gym_env.action_manager)

    # QUANTA_X1 retains its original hold setup. The other profiles seed through
    # the bridge, which distinguishes 7-D Cartesian actions from 7-joint arms.
    default_q = robot.data.default_joint_pos[0]
    joint_name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    wheel_names = ("left_wheel_joint", "right_wheel_joint")
    for jn, slot in slot_map.items() if profile == "quanta_x1" else ():
        if jn in wheel_names:
            continue
        gi = joint_name_to_idx.get(jn)
        if gi is not None:
            actions[0, slot] = float(default_q[gi])
    left_wheel_slot = slot_map.get("left_wheel_joint")
    right_wheel_slot = slot_map.get("right_wheel_joint")

    if profile == "quanta_x1":
        ros_ext = RosBridgeExtension(ros_cfg)
    else:
        from maniparena_sim.ros.sdk_bridge import SdkRosBridge

        ros_ext = SdkRosBridge(ros_cfg, profile)
    try:
        ros_ext.setup(gym_env, robot, action_buffer=actions)
        if profile == "quanta_x1":
            ros_ext.seed_ee_hold(robot)
        else:
            ros_ext.reset()
    except Exception:
        ros_ext.shutdown()
        gym_env.close()
        raise

    cc = (
        QuantaX1RosConfig.CHASSIS_CONTROL_CONFIG
        if profile == "quanta_x1"
        else {"wheel_radius": 0.078, "wheel_track_width": 0.48}
    )
    wheel_radius = float(cc["wheel_radius"])
    wheel_track = float(cc["wheel_track_width"])

    def _twist_to_wheels(lin_x: float, ang_z: float) -> tuple[float, float]:
        left = (lin_x - 0.5 * ang_z * wheel_track) / wheel_radius
        right = (lin_x + 0.5 * ang_z * wheel_track) / wheel_radius
        return left, right

    kb_cfg = payload.get("keyboard") or {}
    kb = None
    import omni.appwindow

    if omni.appwindow.get_default_app_window() is not None:
        kb = _KeyboardTwist(
            linear_velocity=float(kb_cfg.get("linear_velocity", 0.5)),
            angular_velocity=float(kb_cfg.get("angular_velocity", 2.0)),
        )
    print(f"[INFO] {profile} SDK ROS2 ready ({ros_cfg.arm_control} arm control).")
    if kb is not None:
        print("[INFO] Keyboard: R reset; W/S and A/D drive mobile bases alongside /chassis/cmd_vel.")

    simulation_app = globals().get("_APP")
    if simulation_app is None:
        raise RuntimeError("AppLauncher app not initialized; run this file as __main__.")
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                # R resets the episode (robot back to its initial pose).
                if kb is not None and kb.reset_requested:
                    kb.reset_requested = False
                    if left_wheel_slot is not None:
                        actions[0, left_wheel_slot] = 0.0
                    if right_wheel_slot is not None:
                        actions[0, right_wheel_slot] = 0.0
                    gym_env.reset()
                    if profile == "quanta_x1":
                        ros_ext.seed_ee_hold(robot)
                    else:
                        ros_ext.reset()
                    print("[INFO] reset: robot returned to initial pose.")
                    continue

                ros_ext.project_ee_pose_commands(robot)
                # Sum keyboard twist + latest /chassis/cmd_vel (with timeout) -> wheels.
                k_lin, _, k_ang = kb.twist() if kb is not None else (0.0, 0.0, 0.0)
                sim_t = float(getattr(ros_ext, "_sim_time_acc", 0.0)) + float(gym_env.step_dt)
                c_lin, _c_y, c_ang = ros_ext.latest_cmd_vel(sim_t)
                lw, rw = _twist_to_wheels(k_lin + c_lin, k_ang + c_ang)
                if left_wheel_slot is not None:
                    actions[0, left_wheel_slot] = lw
                if right_wheel_slot is not None:
                    actions[0, right_wheel_slot] = rw

                gym_env.step(actions)
                ros_ext.update(gym_env.step_dt)
    except KeyboardInterrupt:
        print("[INFO] interrupted")
    finally:
        if kb is not None:
            kb.shutdown()
        ros_ext.shutdown()
        gym_env.close()
    return 0


if __name__ == "__main__":
    _ensure_isaac_ros2_runtime()
    _args = parse_args()
    sys.argv += ["--enable", "isaacsim.ros2.bridge"]
    _app_launcher = AppLauncher(_args)
    _APP = _app_launcher.app
    try:
        main(_args)
    except BaseException:
        # Kit can terminate the process during close(), before Python prints
        # an unhandled exception. Preserve both the diagnostic and exit status.
        traceback.print_exc()
        if "exit_code" in inspect.signature(_APP.close).parameters:
            _APP.close(exit_code=1)
        else:
            import carb.settings

            _APP.config["fast_shutdown"] = False
            carb.settings.get_settings().set_bool("/app/fastShutdown", False)
            _APP.close()
        raise
    else:
        _APP.close()
