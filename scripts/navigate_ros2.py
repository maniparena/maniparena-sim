#!/usr/bin/env python3
"""EX001 ROS2 navigation: 2D-lidar bridge + keyboard/cmd_vel co-controlled base.

The dummy_task open scene (nav_f16 background) is loaded WITHOUT any recorder —
navigation does not record data. The chassis is driven by the SUM of two
sources, so the keyboard and an external ROS2 nav stack can drive it together:

  * keyboard: W/S forward/back, A/D (or Q/E) yaw, R reset, T randomize (no-op).
  * ROS topic: /chassis/cmd_vel (geometry_msgs/Twist).

nav_mode (2d) and cameras (on) are fixed in code; the ``ros`` YAML block only
carries runtime knobs (use_sim_time, control_rate_hz, cmd_vel_timeout_s).

Usage:
    python scripts/navigate_ros2.py --config configs/navigate/ex001_nav.yaml --enable_cameras
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="EX001 ROS2 navigation.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--config", default="configs/navigate/ex001_nav.yaml")
    return parser.parse_args()


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
        self.randomize_requested = False
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
            if key == "T":
                self.randomize_requested = True
            elif key == "R":
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

    import torch

    from maniparena_sim.environment.builder import build_ex001_nav_gym_env
    from maniparena_sim.environment.registry import bootstrap_arena_registry
    from maniparena_sim.ros.ex001_joint_mapping import build_action_slot_map
    from maniparena_sim.ros.ros2_config import EX001RosConfig
    from maniparena_sim.ros.ros_bridge import RosBridgeExtension, load_ros_bridge_cfg

    bootstrap_arena_registry()
    gym_env, _embodiment = build_ex001_nav_gym_env(
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

    # Seed non-wheel slots with default joint positions so an idle action holds.
    default_q = robot.data.default_joint_pos[0]
    joint_name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    wheel_names = ("left_wheel_joint", "right_wheel_joint")
    for jn, slot in slot_map.items():
        if jn in wheel_names:
            continue
        gi = joint_name_to_idx.get(jn)
        if gi is not None:
            actions[0, slot] = float(default_q[gi])
    left_wheel_slot = slot_map.get("left_wheel_joint")
    right_wheel_slot = slot_map.get("right_wheel_joint")

    ros_ext = RosBridgeExtension(ros_cfg)
    try:
        ros_ext.setup(gym_env, robot, action_buffer=actions)
    except Exception:
        ros_ext.shutdown()
        gym_env.close()
        raise

    cc = EX001RosConfig.CHASSIS_CONTROL_CONFIG
    wheel_radius = float(cc["wheel_radius"])
    wheel_track = float(cc["wheel_track_width"])

    def _twist_to_wheels(lin_x: float, ang_z: float) -> tuple[float, float]:
        left = (lin_x - 0.5 * ang_z * wheel_track) / wheel_radius
        right = (lin_x + 0.5 * ang_z * wheel_track) / wheel_radius
        return left, right

    kb_cfg = payload.get("keyboard") or {}
    kb = _KeyboardTwist(
        linear_velocity=float(kb_cfg.get("linear_velocity", 0.5)),
        angular_velocity=float(kb_cfg.get("angular_velocity", 2.0)),
    )
    print("[INFO] nav controls: W/S move, A/D or Q/E yaw, R reset; external /chassis/cmd_vel also drives base.")

    simulation_app = globals().get("_APP")
    if simulation_app is None:
        raise RuntimeError("AppLauncher app not initialized; run this file as __main__.")
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                # R resets the episode (robot back to its initial pose).
                if kb.reset_requested:
                    kb.reset_requested = False
                    if left_wheel_slot is not None:
                        actions[0, left_wheel_slot] = 0.0
                    if right_wheel_slot is not None:
                        actions[0, right_wheel_slot] = 0.0
                    gym_env.reset()
                    print("[INFO] reset: robot returned to initial pose.")
                    continue

                # Sum keyboard twist + latest /chassis/cmd_vel -> wheel slots.
                k_lin, _, k_ang = kb.twist()
                c_lin = c_ang = 0.0
                buf = getattr(ros_ext, "_cmd_vel_buffer", None)
                if buf is not None:
                    c_lin, _c_y, c_ang = buf.as_tuple()
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
    main(_args)
    _APP.close()
