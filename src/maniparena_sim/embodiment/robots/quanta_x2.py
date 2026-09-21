"""QUANTA_X2 SDK embodiment for the bundled whole-robot USD asset.

The asset contains both seven-joint arms, gripper-base end frames, paired G grippers,
four waist joints, head, and differential-drive wheels.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ImuCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.backend_utils import get_default_renderer_cfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg

from maniparena_sim.embodiment.robots.bimanual import ClampedRawGripperActionCfg
from maniparena_sim.embodiment.sensors.update_camera import OpenCVFisheyeCameraCfg, OpenCVPinholeCameraCfg

ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_roll_joint",
    "elbow_yaw_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
)
LEFT_ARM_JOINTS = tuple(f"left_{name}" for name in ARM_JOINT_SUFFIXES)
RIGHT_ARM_JOINTS = tuple(f"right_{name}" for name in ARM_JOINT_SUFFIXES)
WAIST_JOINTS = ("bow_pitch_joint_01", "bow_pitch_joint_02", "bow_pitch_joint_03", "bow_yaw_joint")


@configclass
class QuantaX2SdkActionsCfg:
    """24-D joint mode, or 24-D Cartesian mode with seven values per end frame."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_ARM_JOINTS),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    right_arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(RIGHT_ARM_JOINTS),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    left_gripper: ActionTermCfg = ClampedRawGripperActionCfg(
        asset_name="robot",
        joint_names=["left_gripper"],
        scale=1.0,
        use_default_offset=False,
        clamp_min=0.0,
        clamp_max=1.89,
    )
    right_gripper: ActionTermCfg = ClampedRawGripperActionCfg(
        asset_name="robot",
        joint_names=["right_gripper"],
        scale=1.0,
        use_default_offset=False,
        clamp_min=0.0,
        clamp_max=1.89,
    )
    waist_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(WAIST_JOINTS),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    head_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["head_pitch_joint", "head_yaw_joint"],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    base_action: ActionTermCfg = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class QuantaX2SceneCfg:
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # Required and checked by the SDK environment builder.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.72, -0.26, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "left_shoulder_pitch_joint": -math.pi / 2,
                "left_shoulder_roll_joint": math.pi / 2,
                "left_shoulder_yaw_joint": math.pi / 2,
                "left_elbow_roll_joint": -math.pi / 2,
                "left_elbow_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "right_shoulder_pitch_joint": -math.pi / 2,
                "right_shoulder_roll_joint": -math.pi / 2,
                "right_shoulder_yaw_joint": math.pi / 2,
                "right_elbow_roll_joint": -math.pi / 2,
                "right_elbow_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
            },
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=list(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS),
                effort_limit_sim=400.0,
                velocity_limit_sim=5.0,
                stiffness=1000.0,
                damping=100.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=["left_gripper", "right_gripper"],
                effort_limit_sim=400.0,
                velocity_limit_sim=2.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=list(WAIST_JOINTS),
                effort_limit_sim=5000.0,
                velocity_limit_sim=5.0,
                stiffness=5000.0,
                damping=1000.0,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_pitch_joint", "head_yaw_joint"],
                effort_limit_sim=50.0,
                velocity_limit_sim=2.0,
                stiffness=200.0,
                damping=10.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                effort_limit_sim=5000.0,
                velocity_limit_sim=5.0,
                stiffness=0.0,
                damping=5000.0,
            ),
        },
    )
    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_gripper_base_link",
                name="end_effector",
            )
        ],
    )
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_gripper_base_link",
                name="right_end_effector",
            )
        ],
    )
    imu: ImuCfg = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/imu_link", update_period=1.0 / 60.0)


@configclass
class QuantaX2CameraCfg(ArenaCameraCfg):
    head_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_head_front_color_optical_frame/Head_Front_Color_Camera",
        update_period=1.0 / 30.0,
        width=1280,
        height=720,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=OpenCVPinholeCameraCfg(clipping_range=(0.01, 1.0e5)),
        renderer_cfg=get_default_renderer_cfg(),
    )
    left_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_camera_color_frame/Left_Gripper_Camera",
        update_period=1.0 / 30.0,
        width=1600,
        height=1200,
        data_types=["rgb"],
        spawn=OpenCVFisheyeCameraCfg(clipping_range=(0.0001, 1.0e5)),
        renderer_cfg=get_default_renderer_cfg(),
    )
    right_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_camera_color_frame/Right_Gripper_Camera",
        update_period=1.0 / 30.0,
        width=1600,
        height=1200,
        data_types=["rgb"],
        spawn=OpenCVFisheyeCameraCfg(clipping_range=(0.0001, 1.0e5)),
        renderer_cfg=get_default_renderer_cfg(),
    )
    chassis_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_chassis_front_depth_optical_frame/Chassis_Front_Depth_Camera",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["distance_to_image_plane"],
        spawn=OpenCVPinholeCameraCfg(clipping_range=(0.03, 1.0e5)),
        renderer_cfg=get_default_renderer_cfg(),
    )


@configclass
class QuantaX2StateObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


class QuantaX2Embodiment(EmbodimentBase):
    name = "QUANTA_X2"

    def __init__(self, *, usd_path: str, enable_cameras: bool = True):
        super().__init__(enable_cameras=enable_cameras)
        self.scene_config = QuantaX2SceneCfg()
        self.scene_config.robot.spawn.usd_path = usd_path
        self.camera_config = QuantaX2CameraCfg() if enable_cameras else None
        self.observation_config = QuantaX2StateObservationsCfg()
        self.prepare_sdk("joint")

    def prepare_sdk(self, arm_control: str) -> None:
        if arm_control not in {"joint", "ee"}:
            raise ValueError(f"Unsupported arm control mode: {arm_control!r}")
        self.action_config = QuantaX2SdkActionsCfg()
        arms = self.scene_config.robot.actuators["arms"]
        arms.stiffness = 16000.0 if arm_control == "ee" else 1000.0
        arms.damping = 350.0 if arm_control == "ee" else 100.0
        if arm_control == "ee":
            for side, joints, term in (
                ("left", LEFT_ARM_JOINTS, "arm_action"),
                ("right", RIGHT_ARM_JOINTS, "right_arm_action"),
            ):
                setattr(
                    self.action_config,
                    term,
                    DifferentialInverseKinematicsActionCfg(
                        asset_name="robot",
                        joint_names=list(joints),
                        body_name=f"{side}_gripper_base_link",
                        controller=DifferentialIKControllerCfg(
                            command_type="pose",
                            use_relative_mode=False,
                            ik_method="dls",
                        ),
                        scale=1.0,
                    ),
                )

    def get_observation_cfg(self):
        return self.observation_config
