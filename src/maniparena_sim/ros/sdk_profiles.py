"""Import-safe ROS SDK contracts for Quanta X2 and ArtiXon Arm-6A."""

from __future__ import annotations

from dataclasses import dataclass

from maniparena_sim.ros.quanta_x1_sdk_topics import normalize_arm_control


@dataclass(frozen=True)
class SdkProfile:
    name: str
    base_body_name: str
    arm_joint_names: tuple[tuple[str, ...], tuple[str, ...]]
    flange_names: tuple[str, str]
    gripper_joint_names: tuple[str, str]
    arm_joint_commands: tuple[str, str]
    arm_pose_commands: tuple[str, str]
    arm_states: tuple[str, str]
    arm_poses: tuple[str, str]
    gripper_commands: tuple[str, str]
    gripper_states: tuple[str, str]
    gripper_state_names: tuple[str, str]
    mobile: bool = False

    @property
    def arm_dof(self) -> int:
        return len(self.arm_joint_names[0])

    def subscribe_topics(self, arm_control: str = "ee") -> frozenset[str]:
        arm_topics = self.arm_pose_commands if normalize_arm_control(arm_control) == "ee" else self.arm_joint_commands
        topics = set(arm_topics + self.gripper_commands)
        if self.mobile:
            topics.update(
                {
                    "/head_position_controller/commands",
                    "/waist_forward_position_controller/commands",
                    "/chassis/cmd_vel",
                }
            )
        return frozenset(topics)

    @property
    def publish_topics(self) -> frozenset[str]:
        topics = set(self.arm_states + self.arm_poses + self.gripper_states)
        topics.update({"/joint_states", "/tf", "/tf_static", "/clock", *RGB_TOPICS})
        if self.mobile:
            topics.update(
                {
                    "/waist/state/joint",
                    "/head/joint_states",
                    "/odom",
                    "/tracked_pose",
                    "/scan",
                    "/hal/chassis/imu",
                    *DEPTH_TOPICS,
                }
            )
        return frozenset(topics)


RGB_TOPICS = frozenset(
    {
        "/camera_head_front/color/image_raw/compressed",
        "/camera1/usb_cam1/image_raw/image_compressed",
        "/camera3/usb_cam3/image_raw/image_compressed",
    }
)
DEPTH_TOPICS = frozenset(
    {
        "/camera_head_front/depth/image_raw/compressedDepth",
        "/camera_chassis_front/depth/points",
    }
)
WAIST_JOINT_NAMES = (
    "bow_pitch_joint_01",
    "bow_pitch_joint_02",
    "bow_pitch_joint_03",
    "bow_yaw_joint",
)
HEAD_JOINT_NAMES = ("head_pitch_joint", "head_yaw_joint")
WHEEL_JOINT_NAMES = ("left_wheel_joint", "right_wheel_joint")

QUANTA_X2 = SdkProfile(
    name="quanta_x2",
    base_body_name="base_link",
    arm_joint_names=tuple(
        tuple(
            f"{side}_{part}_joint"
            for part in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow_roll",
                "elbow_yaw",
                "wrist_roll",
                "wrist_pitch",
            )
        )
        for side in ("left", "right")
    ),
    flange_names=("left_gripper_base_link", "right_gripper_base_link"),
    gripper_joint_names=("left_gripper", "right_gripper"),
    arm_joint_commands=(
        "/smoother_command_input_left",
        "/smoother_command_input_right",
    ),
    arm_pose_commands=(
        "/left_arm/pose_command/root_frame",
        "/right_arm/pose_command/root_frame",
    ),
    arm_states=("/left_arm/state/joint", "/right_arm/state/joint"),
    arm_poses=(
        "/whole_body_controller/left_wrist_pose",
        "/whole_body_controller/right_wrist_pose",
    ),
    gripper_commands=(
        "/left_g_gripper_controller/commands",
        "/right_g_gripper_controller/commands",
    ),
    gripper_states=(
        "/left_g_gripper_states/joint_states",
        "/right_g_gripper_states/joint_states",
    ),
    gripper_state_names=("left_gripper", "right_gripper"),
    mobile=True,
)
ARTIXON_ARM_6A = SdkProfile(
    name="artixon_arm_6a",
    base_body_name="base",
    arm_joint_names=tuple(tuple(f"{side}_arm_joint{i}" for i in range(1, 7)) for side in ("left", "right")),
    flange_names=("left_arm_link6", "right_arm_link6"),
    gripper_joint_names=("left_arm_gripper", "right_arm_gripper"),
    arm_joint_commands=(
        "/left_arm_joint_controller/commands",
        "/right_arm_joint_controller/commands",
    ),
    arm_pose_commands=(
        "/left_arm_cartesian_controller/pose_cmd",
        "/right_arm_cartesian_controller/pose_cmd",
    ),
    arm_states=("/left_arm/joint_states", "/right_arm/joint_states"),
    arm_poses=("/left_arm/end_pose", "/right_arm/end_pose"),
    gripper_commands=(
        "/left_gripper_controller/commands",
        "/right_gripper_controller/commands",
    ),
    gripper_states=("/left_gripper/joint_states", "/right_gripper/joint_states"),
    gripper_state_names=("left_arm_gripper", "right_arm_gripper"),
)


def get_sdk_profile(profile: str | SdkProfile) -> SdkProfile:
    if isinstance(profile, SdkProfile):
        return profile
    profiles = {p.name: p for p in (QUANTA_X2, ARTIXON_ARM_6A)}
    try:
        return profiles[str(profile).strip().lower()]
    except KeyError:
        raise ValueError(f"Unsupported SDK robot: {profile!r}; choose {', '.join(profiles)}") from None


class SdkJointMapping:
    """Resolve authored names explicitly; articulation order is not SDK order."""

    def __init__(self, robot, profile: str | SdkProfile):
        self.profile = get_sdk_profile(profile)
        self.joint_names = list(robot.data.joint_names)
        self.body_names = list(robot.data.body_names)
        self.arms = [self._indices(self.joint_names, names) for names in self.profile.arm_joint_names]
        self.grippers = self._indices(self.joint_names, self.profile.gripper_joint_names)
        self.flanges = self._indices(self.body_names, self.profile.flange_names)
        self.base = self._indices(self.body_names, (self.profile.base_body_name,))[0]
        self.waist = self._indices(self.joint_names, WAIST_JOINT_NAMES) if self.profile.mobile else []
        self.head = self._indices(self.joint_names, HEAD_JOINT_NAMES) if self.profile.mobile else []

    @staticmethod
    def _indices(available, required) -> list[int]:
        missing = [name for name in required if name not in available]
        if missing:
            raise ValueError(f"SDK articulation is missing required names: {missing}")
        return [available.index(name) for name in required]
