"""EX001 whole-body embodiment (mobile base + dual 6-DOF arms + grippers + lift + head).

Separate robot from the desktop BimanualEmbodiment. Joint/link names:
arms ``*_arm_joint[1-6]``, grippers ``*_arm_gripper`` ([0, 1.89]), EE
links ``*_arm_gripper_base_link``, wheels ``left/right_wheel_joint``, prismatic
``lift_joint`` ([0, 0.78] m), head ``head_yaw_joint`` / ``head_pitch_joint``.

AbsIK action layout (vuer/openxr teleop): ``[L_pos(3), L_quat(4), L_grip(1),
R_pos(3), R_quat(4), R_grip(1), L_wheel(1), R_wheel(1), lift(1)]`` (19D).

Wallx whole-body layout (eval): AbsIK + ``head_yaw(1), head_pitch(1)`` (21D).
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.backend_utils import get_default_renderer_cfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab_arena.utils.pose import Pose
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from maniparena_sim.assets import ASSETS_DIR
from maniparena_sim.embodiment.sensors.update_camera import (
    OpenCVFisheyeCameraCfg,
    OpenCVPinholeCameraCfg,
)
from maniparena_sim.embodiment.teleop_devices.differential_drive_keyboard_controller import (
    DifferentialDriveKeyboardControllerCfg,
)

# Depth RenderProduct + Kit viewport: Lab default clipping is "none" (returns inf),
# which blacks the main viewport on Sim6. Clip to the far plane instead.
_EX001_DEPTH_RENDERER_CFG = IsaacRtxRendererCfg(depth_clipping_behavior="max")

_EX001_GRIPPER_OPEN = 1.89
_EX001_GRIPPER_CLOSE = 0.0

# Prismatic lift joint travel (meters), from the USD joint limits.
_EX001_LIFT_LOWER = 0.0
_EX001_LIFT_UPPER = 0.78

# Sim-effective EX001 wheel geometry. Older nominal 0.078 / 0.48 over-drives
# the simulated base.
EX001_WHEEL_RADIUS_M = 0.084
EX001_WHEEL_TRACK_WIDTH_M = 0.458

EX001_DIFF_DRIVE_KEYBOARD_CFG = DifferentialDriveKeyboardControllerCfg(
    mode_name="ex001_differential",
    linear_velocity=0.5,
    angular_velocity=2.0,
    wheel_joint_names=("left_wheel_joint", "right_wheel_joint"),
    wheel_radius=EX001_WHEEL_RADIUS_M,
    wheel_track_width=EX001_WHEEL_TRACK_WIDTH_M,
)


def twist_to_wheel_vel(
    linear_x: float,
    angular_z: float,
    *,
    wheel_radius: float = EX001_WHEEL_RADIUS_M,
    wheel_track_width: float = EX001_WHEEL_TRACK_WIDTH_M,
) -> tuple[float, float]:
    """Convert planar twist to left/right wheel angular velocities."""
    radius = max(float(wheel_radius), 1e-6)
    track = float(wheel_track_width)
    left = (float(linear_x) - 0.5 * float(angular_z) * track) / radius
    right = (float(linear_x) + 0.5 * float(angular_z) * track) / radius
    return left, right


class ClampedRawGripperAction(joint_actions.JointPositionAction):
    """Absolute gripper joint target with clamp bounds."""

    cfg: "ClampedRawGripperActionCfg"

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=float(self.cfg.clamp_min),
            max=float(self.cfg.clamp_max),
        )


@configclass
class ClampedRawGripperActionCfg(JointPositionActionCfg):
    """Config for raw gripper joint targets."""

    class_type: type[ActionTerm] = ClampedRawGripperAction

    clamp_min: float = _EX001_GRIPPER_CLOSE
    clamp_max: float = _EX001_GRIPPER_OPEN


@register_asset
class EX001Embodiment(EmbodimentBase):
    """EX001 whole-body mobile manipulator."""

    name = "ex001"

    @configclass
    class SceneCfg:
        robot: ArticulationCfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(ASSETS_DIR, "ex001", "ex001.usd"),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                ),
            ),
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.72, -0.26, -0.93),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            actuators={
                "wheels": ImplicitActuatorCfg(
                    joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
                    effort_limit=500.0, velocity_limit=200.0, stiffness=0.0, damping=1000.0,
                ),
                "lift_acts": ImplicitActuatorCfg(
                    joint_names_expr=["lift_joint"],
                    effort_limit=2000.0, velocity_limit=100.0, stiffness=5000.0, damping=1000.0,
                ),
                "left_arm_acts": ImplicitActuatorCfg(joint_names_expr=["left_arm_joint[1-6]"], effort_limit_sim=200.0, stiffness=1500.0, damping=150.0),
                "right_arm_acts": ImplicitActuatorCfg(joint_names_expr=["right_arm_joint[1-6]"], effort_limit_sim=200.0, stiffness=1500.0, damping=150.0),
                "left_gripper_acts": ImplicitActuatorCfg(joint_names_expr=["left_arm_gripper"], effort_limit_sim=200.0, stiffness=200.0, damping=30.0),
                "right_gripper_acts": ImplicitActuatorCfg(joint_names_expr=["right_arm_gripper"], effort_limit_sim=200.0, stiffness=200.0, damping=30.0),
                "head_acts": ImplicitActuatorCfg(
                    joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
                    effort_limit_sim=200.0, stiffness=1500.0, damping=150.0,
                ),
            },
        )
        left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_base_link",
            debug_vis=False,
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_base_link", name="end_effector")],
        )
        right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_base_link",
            debug_vis=False,
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_base_link", name="right_end_effector")],
        )
        left_gripper_contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_.*_link", update_period=0.0, history_length=1, track_air_time=False)
        right_gripper_contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_.*_link", update_period=0.0, history_length=1, track_air_time=False)
        imu: ImuCfg = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/imu_link", update_period=1.0 / 60.0)
        # SCENE_END_MARKER

    @configclass
    class CameraCfg(ArenaCameraCfg):
        left_wrist_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_camera_color_frame/Left_Gripper_Camera",
            update_period=0.0333, height=480, width=640, data_types=["rgb"],
            spawn=OpenCVFisheyeCameraCfg(clipping_range=(0.03, 1.0e5)),
            renderer_cfg=get_default_renderer_cfg(),
        )
        right_wrist_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_camera_color_frame/Right_Gripper_Camera",
            update_period=0.0333, height=480, width=640, data_types=["rgb"],
            spawn=OpenCVFisheyeCameraCfg(clipping_range=(0.03, 1.0e5)),
            renderer_cfg=get_default_renderer_cfg(),
        )
        head_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_head_front_color_optical_frame/Head_Front_Color_Camera",
            update_period=0.0333, height=480, width=640, data_types=["rgb"],
            spawn=OpenCVPinholeCameraCfg(clipping_range=(0.03, 1.0e5)),
            renderer_cfg=get_default_renderer_cfg(),
        )
        chassis_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_chassis_front_depth_optical_frame/Chassis_Front_Depth_Camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["depth"],
            # Clip far-plane holes (Lab default "none" -> inf). Kit viewport black
            # with depth is handled separately via ensure_kit_viewport_color_render.
            depth_clipping_behavior="max",
            spawn=OpenCVPinholeCameraCfg(clipping_range=(0.03, 1.0e5)),
            renderer_cfg=_EX001_DEPTH_RENDERER_CFG,
        )
        # CAMERA_END_MARKER

    @configclass
    class ActionsCfgAbsIK:
        """19D abs-IK + diff-drive base + lift. Term order yields planner diffik slots.

        Layout: ``[L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1),
        L_wheel(1), R_wheel(1), lift(1)]``. Used by vuer/openxr teleop: the arms
        track absolute base-frame pose targets, the base is differential-drive
        wheel velocity, and the prismatic lift joint takes an absolute position
        target (vuer right-joystick).
        """

        arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["left_arm_joint[1-6]"], body_name="left_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(asset_name="robot", joint_names=["left_arm_gripper"], scale=1.0, offset=0.0, use_default_offset=False)
        right_arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["right_arm_joint[1-6]"], body_name="right_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(asset_name="robot", joint_names=["right_arm_gripper"], scale=1.0, offset=0.0, use_default_offset=False)
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        lift_action: ActionTermCfg = JointPositionActionCfg(asset_name="robot", joint_names=["lift_joint"], scale=1.0, use_default_offset=False)

    @configclass
    class ActionsCfgWallxWholebody:
        """21D Wall-X whole-body inference: AbsIK + head.

        Layout: ``[L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1),
        L_wheel(1), R_wheel(1), lift(1), head_yaw(1), head_pitch(1)]``.
        """

        arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["left_arm_joint[1-6]"], body_name="left_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(asset_name="robot", joint_names=["left_arm_gripper"], scale=1.0, offset=0.0, use_default_offset=False)
        right_arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["right_arm_joint[1-6]"], body_name="right_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(asset_name="robot", joint_names=["right_arm_gripper"], scale=1.0, offset=0.0, use_default_offset=False)
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        lift_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot", joint_names=["lift_joint"], scale=1.0, use_default_offset=False,
        )
        head_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_yaw_joint", "head_pitch_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )

    @configclass
    class ActionsCfgNav:
        """Joint-position arms/lift/grippers + diff-drive wheel velocity (ROS2 nav).

        ``joint_pos`` covers every non-wheel joint with ABSOLUTE position targets
        (``use_default_offset=False``); the nav loop seeds the action vector with
        the robot's default joint positions so an idle command holds the pose and
        ``/mock_robot_interface/command`` writes absolute targets directly.
        ``base_action`` drives the two wheels by velocity (summed keyboard +
        ``/chassis/cmd_vel``), so everything flows through one ``env.step``.
        """

        joint_pos: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["^(?!left_wheel_joint$|right_wheel_joint$).*"],
            scale=1.0,
            use_default_offset=False,
        )
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )

    @configclass
    class ActionsCfgRelIK:
        """16D relative-IK + diff-drive base, used by KEYBOARD teleop.

        Layout: ``[L_dpos(3), L_drotvec(3), L_grip(1), R_dpos(3), R_drotvec(3),
        R_grip(1), L_wheel(1), R_wheel(1)]``. Keyboard emits per-arm delta
        pose (axis-angle), so the arms use ``use_relative_mode=True`` (6D/arm).
        """

        arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["left_arm_joint[1-6]"], body_name="left_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"), scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(asset_name="robot", joint_names=["left_arm_gripper"], scale=1.0, offset=0.0, use_default_offset=False)
        right_arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["right_arm_joint[1-6]"], body_name="right_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"), scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(asset_name="robot", joint_names=["right_arm_gripper"], scale=1.0, offset=0.0, use_default_offset=False)
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )

    @configclass
    class StateObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            root_pos = ObsTerm(func=mdp_isaac_lab.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
            root_quat = ObsTerm(func=mdp_isaac_lab.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class CameraObservationsCfg:
        @configclass
        class CameraObsCfg(ObsGroup):
            left_wrist_cam = ObsTerm(func=mdp_isaac_lab.image, params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False})
            right_wrist_cam = ObsTerm(func=mdp_isaac_lab.image, params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False})
            head_cam = ObsTerm(func=mdp_isaac_lab.image, params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False})
            chassis_cam = ObsTerm(
                func=mdp_isaac_lab.image,
                params={
                    "sensor_cfg": SceneEntityCfg("chassis_camera"),
                    "data_type": "depth",
                    "normalize": False,
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        camera_obs: CameraObsCfg = CameraObsCfg()
        # ACTIONS_OBS_END_MARKER

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = self.SceneCfg()
        self.camera_config = self.CameraCfg() if enable_cameras else None
        self.action_config = self.ActionsCfgAbsIK()
        self.observation_config = self.StateObservationsCfg()
        self._camera_observation_config = self.CameraObservationsCfg()
        self.diff_drive_keyboard_controller_cfg = EX001_DIFF_DRIVE_KEYBOARD_CFG
        self.xr = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))

    def get_observation_cfg(self):
        if self.enable_cameras:
            return combine_configclass_instances("ObservationCfg", self.observation_config, self._camera_observation_config)
        return self.observation_config

    def get_xr_cfg(self):
        return self.xr

    def get_vr_gripper_clamp(self) -> dict[str, tuple[float, float]]:
        return {
            "left_arm_gripper": (_EX001_GRIPPER_CLOSE, _EX001_GRIPPER_OPEN),
            "right_arm_gripper": (_EX001_GRIPPER_CLOSE, _EX001_GRIPPER_OPEN),
        }

    def get_vr_gripper_joint_names(self) -> tuple[str, str]:
        return "left_arm_gripper", "right_arm_gripper"

    def get_vr_ee_frame_names(self) -> tuple[str, str]:
        return "left_ee_frame", "right_ee_frame"

    def get_vr_lift_joint_name(self) -> str:
        return "lift_joint"

    def get_vr_lift_limits(self) -> tuple[float, float]:
        return _EX001_LIFT_LOWER, _EX001_LIFT_UPPER
