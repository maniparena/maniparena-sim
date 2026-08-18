"""TF transform publisher and odom origin state."""

from geometry_msgs.msg import TransformStamped

from maniparena_sim.ros.math_utils import compute_relative_pose
from maniparena_sim.ros.stamp_utils import copy_stamp

from .sim_utils import get_body_pose, get_root_pose

# Bolted to the chassis. Dynamic /tf can lag the RTX /scan stamp; RViz then
# fails with "extrapolation into the future". These frames stay on /tf_static.
CHASSIS_STATIC_FRAMES = frozenset(
    {
        "a_d_laser",
        "a_d_laser_base",
        "a_d_laser_link",
        "camera_chassis_front_depth_optical_frame",
        "camera_chassis_front_depth_frame",
        "camera_chassis_front_color_optical_frame",
        "camera_chassis_front_color_frame",
        "camera_chassis_front_link",
        "base_front_camera_frame_link",
        "imu_link",
    }
)


class OdomOrigin:
    """Shared odom origin state."""

    def __init__(self):
        self.position = None
        self.orientation = None

    def init_from_pose(self, position, orientation):
        self.position = position.copy()
        self.orientation = orientation.copy()

    @property
    def is_initialized(self):
        return self.position is not None


class TfPublisher:
    """TF transform publisher with optional publish-rate decimation."""

    def __init__(self, broadcaster, odom_origin, publish_interval=1):
        self.broadcaster = broadcaster
        self.odom_origin = odom_origin
        self.parent_map = {}
        self.body_name_to_idx = {}
        self.publish_interval = max(1, int(publish_interval))
        self._publish_call_count = 0

    def init_from_robot(self, robot):
        """Build parent-child relationship by parsing USD joint structure."""
        import omni.usd
        from pxr import Usd, UsdPhysics

        body_names = list(robot.data.body_names)
        self.body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
        self.parent_map = {}

        stage = omni.usd.get_context().get_stage()
        robot_prim_path = robot.cfg.prim_path
        if ".*" in robot_prim_path:
            robot_prim_path = robot_prim_path.replace(".*", "0")

        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim.IsValid():
            return

        for prim in Usd.PrimRange(robot_prim):
            joint = UsdPhysics.Joint(prim)
            if not joint:
                continue
            body0_rel = joint.GetBody0Rel()
            body1_rel = joint.GetBody1Rel()
            if not body0_rel or not body1_rel:
                continue
            body0_targets = body0_rel.GetTargets()
            body1_targets = body1_rel.GetTargets()
            if not body0_targets or not body1_targets:
                continue
            parent_prim = stage.GetPrimAtPath(body0_targets[0])
            child_prim = stage.GetPrimAtPath(body1_targets[0])
            if not parent_prim.IsValid() or not child_prim.IsValid():
                continue
            parent_name = parent_prim.GetName()
            child_name = child_prim.GetName()
            if parent_name in self.body_name_to_idx and child_name in self.body_name_to_idx:
                self.parent_map[child_name] = parent_name

    # -- transform helpers -------------------------------------------------

    def _build_transform(self, frame_id, child_frame_id, position, orientation, stamp):
        t = TransformStamped()
        copy_stamp(t.header.stamp, stamp)
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        # Lab / Arena quats are xyzw.
        t.transform.rotation.x = float(orientation[0])
        t.transform.rotation.y = float(orientation[1])
        t.transform.rotation.z = float(orientation[2])
        t.transform.rotation.w = float(orientation[3])
        return t

    # -- individual publish methods ----------------------------------------

    def publish_map_to_odom(self, stamp):
        """Publish ``map -> odom`` transform (odom origin in world frame)."""
        if not self.odom_origin.is_initialized:
            return
        t = self._build_transform("map", "odom", self.odom_origin.position, self.odom_origin.orientation, stamp)
        self.broadcaster.sendTransform(t)

    def publish_odom_to_base(self, obs, stamp):
        root_pos, root_quat = get_root_pose(obs)
        position, orientation = compute_relative_pose(
            root_pos, root_quat, self.odom_origin.position, self.odom_origin.orientation
        )
        t = self._build_transform("odom", "base_link", position, orientation, stamp)
        self.broadcaster.sendTransform(t)

    def publish_body_frames(self, obs, body_names_list, stamp):
        if not self.parent_map:
            self._publish_flat(obs, body_names_list, stamp)
            return

        body_poses = {}
        for idx, body_name in enumerate(body_names_list):
            pos, quat = get_body_pose(obs, idx)
            if pos is not None:
                body_poses[body_name] = (pos, quat)

        transforms = []
        for child_name, parent_name in self.parent_map.items():
            if child_name in CHASSIS_STATIC_FRAMES:
                continue
            if child_name not in body_poses or parent_name not in body_poses:
                continue
            child_pos, child_quat = body_poses[child_name]
            parent_pos, parent_quat = body_poses[parent_name]
            rel_pos, rel_quat = compute_relative_pose(child_pos, child_quat, parent_pos, parent_quat)
            t = self._build_transform(parent_name, child_name, rel_pos, rel_quat, stamp)
            transforms.append(t)

        if transforms:
            self.broadcaster.sendTransform(transforms)

    def _publish_flat(self, obs, body_names_list, stamp):
        root_pos, root_quat = get_root_pose(obs)
        transforms = []
        for idx, body_name in enumerate(body_names_list):
            if body_name == "base_link" or body_name in CHASSIS_STATIC_FRAMES:
                continue
            body_pos, body_quat = get_body_pose(obs, idx)
            if body_pos is None:
                continue
            position, orientation = compute_relative_pose(body_pos, body_quat, root_pos, root_quat)
            t = self._build_transform("base_link", body_name, position, orientation, stamp)
            transforms.append(t)
        if transforms:
            self.broadcaster.sendTransform(transforms)

    # -- top-level entry point ---------------------------------------------

    def publish_static_on_base(self, obs, body_names_list, stamp, broadcaster):
        """Publish chassis-fixed frames as ``base_link -> child`` on /tf_static."""
        root_pos, root_quat = get_root_pose(obs)
        transforms = []
        for idx, body_name in enumerate(body_names_list):
            if body_name not in CHASSIS_STATIC_FRAMES:
                continue
            body_pos, body_quat = get_body_pose(obs, idx)
            if body_pos is None:
                continue
            position, orientation = compute_relative_pose(body_pos, body_quat, root_pos, root_quat)
            transforms.append(self._build_transform("base_link", body_name, position, orientation, stamp))
        if transforms:
            broadcaster.sendTransform(transforms)

    def publish_all(self, obs, body_names_list, stamp):
        """Publish all TF transforms (every *publish_interval*-th call)."""
        self._publish_call_count += 1
        if self._publish_call_count % self.publish_interval != 0:
            return
        self.publish_map_to_odom(stamp)
        self.publish_odom_to_base(obs, stamp)
        self.publish_body_frames(obs, body_names_list, stamp)
