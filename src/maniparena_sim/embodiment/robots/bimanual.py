"""Bimanual embodiment config for the open-source runtime."""

from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.backend_utils import get_default_renderer_cfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.assets import ASSETS_DIR
from maniparena_sim.embodiment.sensors.update_camera import OpenCVFisheyeCameraCfg


_BIMANUAL_GRIPPER_OPEN = 5.717175
_BIMANUAL_GRIPPER_CLOSE = 0.0


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

    clamp_min: float = _BIMANUAL_GRIPPER_CLOSE
    clamp_max: float = _BIMANUAL_GRIPPER_OPEN


def _get_ee_world(env, ee_frame_cfg: SceneEntityCfg):
    ee = env.scene[ee_frame_cfg.name]
    pos = ee.data.target_pos_w[:, 0, :] - env.scene.env_origins
    quat = ee.data.target_quat_w[:, 0, :]
    return pos, quat


def _get_or_init_ee_ref(env, key: str, pos: torch.Tensor, quat: torch.Tensor):
    if not hasattr(env, key):
        setattr(env, key, (pos.clone(), quat.clone()))
    ref = getattr(env, key)
    if hasattr(env, "episode_length_buf"):
        mask = env.episode_length_buf == 0
        if mask.any():
            ref[0][mask] = pos[mask]
            ref[1][mask] = quat[mask]
    return ref


def bimanual_left_eef_delta_pos(env, ee_frame_cfg=SceneEntityCfg("left_ee_frame")) -> torch.Tensor:
    pos, quat = _get_ee_world(env, ee_frame_cfg)
    ref_pos, _ = _get_or_init_ee_ref(env, "_left_ee_ref", pos, quat)
    return pos - ref_pos


def bimanual_left_eef_delta_quat(env, ee_frame_cfg=SceneEntityCfg("left_ee_frame")) -> torch.Tensor:
    pos, quat = _get_ee_world(env, ee_frame_cfg)
    _, ref_quat = _get_or_init_ee_ref(env, "_left_ee_ref", pos, quat)
    return math_utils.quat_mul(quat, math_utils.quat_conjugate(ref_quat))


def bimanual_right_eef_delta_pos(env, ee_frame_cfg=SceneEntityCfg("right_ee_frame")) -> torch.Tensor:
    pos, quat = _get_ee_world(env, ee_frame_cfg)
    ref_pos, _ = _get_or_init_ee_ref(env, "_right_ee_ref", pos, quat)
    return pos - ref_pos


def bimanual_right_eef_delta_quat(env, ee_frame_cfg=SceneEntityCfg("right_ee_frame")) -> torch.Tensor:
    pos, quat = _get_ee_world(env, ee_frame_cfg)
    _, ref_quat = _get_or_init_ee_ref(env, "_right_ee_ref", pos, quat)
    return math_utils.quat_mul(quat, math_utils.quat_conjugate(ref_quat))


_GRIPPER_JOINT_IDX: dict[str, int] = {}


def _gripper_joint_idx(env, joint_name: str) -> int:
    if joint_name not in _GRIPPER_JOINT_IDX:
        ids, _ = env.scene["robot"].find_joints(joint_name)
        _GRIPPER_JOINT_IDX[joint_name] = ids[0]
    return _GRIPPER_JOINT_IDX[joint_name]


def wallx_ee_follow_pos(env, ee_frame_cfg=SceneEntityCfg("left_ee_frame"), gripper_joint_name: str = "left_arm_gripper"):
    pos, quat = _get_ee_world(env, ee_frame_cfg)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    idx = _gripper_joint_idx(env, gripper_joint_name)
    gripper = env.scene["robot"].data.joint_pos[:, idx : idx + 1]
    return torch.cat([pos, euler, gripper], dim=-1)


@register_asset
class BimanualEmbodiment(EmbodimentBase):
    """Bimanual dual-arm fixed-base embodiment."""

    name = "Bimanual"

    @configclass
    class SceneCfg:
        robot: ArticulationCfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(ASSETS_DIR, "bimanual_robot", "bimanual_robot.usd"),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                ),
            ),
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.32165, -0.25918, -0.23),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            actuators={
                "left_arm_acts": ImplicitActuatorCfg(joint_names_expr=["left_arm_joint[1-6]"], effort_limit_sim=200.0, stiffness=1500.0, damping=150.0),
                "right_arm_acts": ImplicitActuatorCfg(joint_names_expr=["right_arm_joint[1-6]"], effort_limit_sim=200.0, stiffness=1500.0, damping=150.0),
                "left_gripper_acts": ImplicitActuatorCfg(joint_names_expr=["left_arm_gripper"], effort_limit_sim=200.0, stiffness=200.0, damping=30.0),
                "right_gripper_acts": ImplicitActuatorCfg(joint_names_expr=["right_arm_gripper"], effort_limit_sim=200.0, stiffness=200.0, damping=30.0),
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
        left_gripper_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_.*_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
        )
        right_gripper_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_.*_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
        )

    @configclass
    class CameraCfg(ArenaCameraCfg):
        left_wrist_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_camera_color_frame/Left_Gripper_Camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=OpenCVFisheyeCameraCfg(clipping_range=(0.03, 1.0e5)),
            renderer_cfg=get_default_renderer_cfg(),
        )
        right_wrist_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_camera_color_frame/Right_Gripper_Camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=OpenCVFisheyeCameraCfg(clipping_range=(0.03, 1.0e5)),
            renderer_cfg=get_default_renderer_cfg(),
        )
        head_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Head_Camera",
            update_period=0.0333,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                horizontal_aperture=5.54,
                vertical_aperture=3.09,
                focal_length=2.97,
                clipping_range=(0.03, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.084, 0.0, 0.517),
                # Lab 3 OffsetCfg.rot is xyzw (converted from legacy wxyz).
                rot=(0.21263, -0.21263, -0.67438, 0.67438),
                convention="opengl",
            ),
            renderer_cfg=get_default_renderer_cfg(),
        )

    @configclass
    class ActionsCfg:
        arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
        )
        right_arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
        )

    @configclass
    class JointActionsCfg:
        arm_action: ActionTermCfg = JointPositionActionCfg(asset_name="robot", joint_names=["left_arm_joint[1-6]"], scale=1.0, use_default_offset=False)
        gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_arm_gripper"],
            open_command_expr={"left_arm_gripper": 5.0},
            close_command_expr={"left_arm_gripper": 0.0},
        )
        right_arm_action: ActionTermCfg = JointPositionActionCfg(asset_name="robot", joint_names=["right_arm_joint[1-6]"], scale=1.0, use_default_offset=False)
        right_gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            open_command_expr={"right_arm_gripper": 5.0},
            close_command_expr={"right_arm_gripper": 0.0},
        )

    @configclass
    class StateObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            eef_delta_pos = ObsTerm(func=bimanual_left_eef_delta_pos, params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")})
            eef_delta_quat = ObsTerm(func=bimanual_left_eef_delta_quat, params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")})
            right_eef_delta_pos = ObsTerm(func=bimanual_right_eef_delta_pos, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")})
            right_eef_delta_quat = ObsTerm(func=bimanual_right_eef_delta_quat, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")})
            follow1_pos = ObsTerm(func=wallx_ee_follow_pos, params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame"), "gripper_joint_name": "left_arm_gripper"})
            follow2_pos = ObsTerm(func=wallx_ee_follow_pos, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame"), "gripper_joint_name": "right_arm_gripper"})

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

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        camera_obs: CameraObsCfg = CameraObsCfg()

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = self.SceneCfg()
        self.camera_config = self.CameraCfg()
        self.action_config = self.ActionsCfg()
        self.observation_config = self.StateObservationsCfg()
        self._camera_observation_config = self.CameraObservationsCfg()

    def get_observation_cfg(self):
        if self.enable_cameras:
            return combine_configclass_instances("ObservationCfg", self.observation_config, self._camera_observation_config)
        return self.observation_config

    def get_vr_gripper_clamp(self) -> dict[str, tuple[float, float]]:
        return {
            "left_arm_gripper": (_BIMANUAL_GRIPPER_CLOSE, _BIMANUAL_GRIPPER_OPEN),
            "right_arm_gripper": (_BIMANUAL_GRIPPER_CLOSE, _BIMANUAL_GRIPPER_OPEN),
        }

    def get_vr_gripper_joint_names(self) -> tuple[str, str]:
        return "left_arm_gripper", "right_arm_gripper"

    def get_vr_ee_frame_names(self) -> tuple[str, str]:
        return "left_ee_frame", "right_ee_frame"
