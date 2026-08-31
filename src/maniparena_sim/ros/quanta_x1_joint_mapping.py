"""QUANTA_X1 robot joint index mapping for ROS data publishing."""

LEFT_ARM_JOINT_PATTERN = "left_arm_joint.*"
RIGHT_ARM_JOINT_PATTERN = "right_arm_joint.*"


class QuantaX1JointIndexMapping:
    """Pre-compute joint indices for efficient ROS data publishing.

    Stores joint and body indices for all controlled joints of the QUANTA_X1
    robot so that data acquirers can publish ``JointState`` messages
    efficiently without string lookups every step.
    """

    def __init__(self, robot):
        self.joint_names = robot.data.joint_names
        self.joint_name_to_index = {name: i for i, name in enumerate(self.joint_names)}

        self.left_arm = robot.find_joints(LEFT_ARM_JOINT_PATTERN)[0]
        self.right_arm = robot.find_joints(RIGHT_ARM_JOINT_PATTERN)[0]
        self.left_gripper = robot.find_joints("left_arm_gripper")[0]
        self.right_gripper = robot.find_joints("right_arm_gripper")[0]
        self.head_pitch = robot.find_joints("head_pitch_joint")[0]
        self.head_yaw = robot.find_joints("head_yaw_joint")[0]
        self.lift = robot.find_joints("lift_joint")[0]

        self.all_controlled_indices = (
            self.lift + self.left_arm + self.right_arm + self.left_gripper + self.right_gripper
        )

        self.left_gripper_body = robot.find_bodies("left_arm_gripper_base_link")[0]
        self.right_gripper_body = robot.find_bodies("right_arm_gripper_base_link")[0]

        # Optionally set by main script when base_velocity action term exists
        self.base_velocity_start = None
        self.base_velocity_end = None


def build_action_slot_map(action_manager) -> dict:
    """Map each joint name to its slot index in the env action vector.

    Walks the ActionManager terms in declaration order (the layout
    ``env.step`` expects) and, for each ``JointAction``-style term, assigns its
    per-joint slots. Returns ``{joint_name: slot_index}``.
    """
    slot_map: dict[str, int] = {}
    offset = 0
    for name in action_manager.active_terms:
        term = action_manager.get_term(name)
        joint_names = getattr(term, "_joint_names", None)
        dim = int(term.action_dim)
        if joint_names is not None and len(joint_names) == dim:
            for i, jn in enumerate(joint_names):
                slot_map[jn] = offset + i
        offset += dim
    return slot_map

