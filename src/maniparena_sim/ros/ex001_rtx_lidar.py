"""EX001 RTX lidar data extraction and ROS ``LaserScan`` construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from maniparena_sim.utils.debug_print import manaprint


def _sane_beam_count(count: float) -> bool:
    return 50.0 <= float(count) <= 20000.0


def laser_scan_writer_params(
    *,
    scan_type: str,
    scan_rate_hz: float,
    near_m: float,
    far_m: float,
    firing_rate_hz: float = 0.0,
    report_rate_hz: float = 0.0,
    azimuth_deg: Sequence[float] | None = None,
    fallback_beams: int = 3140,
) -> dict[str, object]:
    """Build 2D LaserScan geometry from OmniLidar attributes.

    EX001 authors ``reportRateBaseHz`` as points per revolution (3140), not Hz.
    ``patternFiringRateHz`` is only a fallback when that count is unavailable.
    """
    rotation_rate = max(float(scan_rate_hz), 1e-6)
    if str(scan_type).upper() == "SOLID_STATE" and azimuth_deg:
        az_start = float(min(azimuth_deg))
        az_end = float(max(azimuth_deg))
        h_fov = az_end - az_start
        h_res = h_fov / float(len(azimuth_deg))
        if az_end > 180.0:
            az_start -= 180.0
            az_end -= 180.0
    else:
        report_count = float(report_rate_hz or 0.0)
        firing_rate = float(firing_rate_hz or 0.0)
        if _sane_beam_count(report_count):
            beam_count = report_count
        elif firing_rate > 0.0 and _sane_beam_count(firing_rate / rotation_rate):
            beam_count = firing_rate / rotation_rate
        else:
            beam_count = float(fallback_beams)
        h_res = 360.0 / beam_count
        az_start = -180.0
        az_end = 180.0
        h_fov = 360.0
    return {
        "horizontalFov": float(h_fov),
        "horizontalResolution": float(h_res),
        "depthRange": [float(near_m), float(far_m)],
        "rotationRate": float(rotation_rate),
        "azimuthRange": [float(az_start), float(az_end)],
    }


def full_scan_prim_overrides(
    scan_rate_hz: float,
    sim_fps: float = 120.0,
    firing_rate_hz: float = 0.0,
) -> dict[str, object]:
    """USD attrs the official Lidar wrap otherwise skips when missing."""
    del sim_fps, firing_rate_hz
    return {
        "omni:sensor:Core:accumulateOutputs": True,
        "omni:sensor:Core:instantLidar": True,
        "omni:sensor:Core:elementsCoordsType": "CARTESIAN",
        "omni:sensor:tickRate": max(float(scan_rate_hz), 1e-6),
    }


def lidar_publish_step(sim_fps: float = 120.0, scan_rate_hz: float = 10.0) -> int:
    """Frames per published scan at ``scan_rate_hz``."""
    return max(1, int(round(float(sim_fps) / max(float(scan_rate_hz), 1e-6))))


def bin_points_to_laser_ranges(
    points: np.ndarray,
    intensities: np.ndarray | None,
    *,
    azimuth_range_deg: Sequence[float],
    horizontal_resolution_deg: float,
    range_min: float,
    range_max: float,
) -> tuple[list[float], list[float], float, float, float]:
    """Bin sensor-frame XYZ hits into ROS ``LaserScan`` ranges."""
    az_min_deg = float(azimuth_range_deg[0])
    az_max_deg = float(azimuth_range_deg[1])
    h_res_deg = max(float(horizontal_resolution_deg), 1e-9)
    angle_min = math.radians(az_min_deg)
    angle_max = math.radians(az_max_deg)
    angle_increment = math.radians(h_res_deg)
    num_beams = max(1, int(round((az_max_deg - az_min_deg) / h_res_deg)))

    ranges = [math.inf] * num_beams
    out_intensity = [0.0] * num_beams
    if points is None or len(points) == 0:
        return ranges, out_intensity, angle_min, angle_max, angle_increment

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if intensities is None or len(intensities) == 0:
        inten = np.ones(len(pts), dtype=np.float64)
    else:
        inten = np.asarray(intensities, dtype=np.float64).reshape(-1)
        if len(inten) != len(pts):
            inten = np.ones(len(pts), dtype=np.float64)

    xy_range = np.linalg.norm(pts[:, :2], axis=1)
    azimuth = np.arctan2(pts[:, 1], pts[:, 0])
    valid = (xy_range >= float(range_min)) & (xy_range <= float(range_max))
    if not np.any(valid):
        return ranges, out_intensity, angle_min, angle_max, angle_increment

    indices = np.floor((azimuth[valid] - angle_min) / angle_increment).astype(np.int64)
    keep = (indices >= 0) & (indices < num_beams)
    if not np.any(keep):
        return ranges, out_intensity, angle_min, angle_max, angle_increment

    indices = indices[keep]
    distances = xy_range[valid][keep]
    valid_intensities = inten[valid][keep]
    for beam, distance, intensity in zip(indices.tolist(), distances.tolist(), valid_intensities.tolist()):
        if distance < ranges[beam]:
            ranges[beam] = float(distance)
            out_intensity[beam] = float(intensity)
    return ranges, out_intensity, angle_min, angle_max, angle_increment


def resolve_lidar_prim_path(prim_path: str, env_id: int = 0, scene=None) -> str:
    """Replace ``{ENV_REGEX_NS}`` using the spawned scene env prims."""
    if "{ENV_REGEX_NS}" not in prim_path:
        return prim_path
    suffix = prim_path.split("{ENV_REGEX_NS}", 1)[1]
    env_paths = getattr(scene, "env_prim_paths", None) if scene is not None else None
    if env_paths:
        return f"{env_paths[env_id]}{suffix}"
    env_ns = getattr(scene, "env_ns", None) if scene is not None else None
    if not env_ns:
        cfg = getattr(scene, "cfg", None) if scene is not None else None
        env_ns = getattr(cfg, "env_ns", None) if cfg is not None else None
    if env_ns:
        return prim_path.replace("{ENV_REGEX_NS}", f"{env_ns}/env_{env_id}")
    raise ValueError(
        "Cannot resolve lidar prim path: scene has no env_prim_paths or env_ns. "
        f"template={prim_path!r} env_id={env_id}"
    )


class RtxLidarHelper:
    """Isaac Sim 6 RTX lidar read through the GenericModelOutput annotator."""

    def __init__(
        self,
        prim_path: str,
        env_id: int = 0,
        *,
        frame_id: str = "a_d_laser",
        topic_name: str = "scan",
        scan_rate_hz: float = 10.0,
        num_beams: int = 3140,
        use_sim_time: bool = False,
    ):
        del use_sim_time, topic_name
        self._prim_path = prim_path
        self._env_id = env_id
        self._frame_id = frame_id
        self._scan_rate_hz = scan_rate_hz
        self._num_beams = num_beams
        self._lidar = None
        self._render_product = None
        self._annotator = None
        self._geometry: dict[str, object] | None = None
        self._near_m = 0.05
        self._far_m = 25.0
        self._resolved_prim_path: str | None = None

    def initialize(self, scene) -> bool:
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd

        prim_path = resolve_lidar_prim_path(self._prim_path, self._env_id, scene)
        self._resolved_prim_path = prim_path
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        if not extension_manager.set_extension_enabled_immediate("isaacsim.sensors.experimental.rtx", True):
            return False

        from isaacsim.sensors.experimental.rtx import Lidar

        stage = omni.usd.get_context().get_stage()
        lidar_prim = stage.GetPrimAtPath(prim_path)
        if not lidar_prim.IsValid():
            return False

        self._near_m = float(self._read_attr(lidar_prim, "omni:sensor:Core:nearRangeM", 0.05))
        self._far_m = float(self._read_attr(lidar_prim, "omni:sensor:Core:farRangeM", 25.0))
        scan_rate = float(self._read_attr(lidar_prim, "omni:sensor:Core:scanRateBaseHz", self._scan_rate_hz))
        self._scan_rate_hz = scan_rate
        firing_rate_hz = float(self._read_attr(lidar_prim, "omni:sensor:Core:patternFiringRateHz", 0.0) or 0.0)
        report_rate_hz = float(self._read_attr(lidar_prim, "omni:sensor:Core:reportRateBaseHz", 0.0) or 0.0)
        scan_type = str(self._read_attr(lidar_prim, "omni:sensor:Core:scanType", "") or "")
        prim_overrides = full_scan_prim_overrides(scan_rate)
        for name, value in prim_overrides.items():
            self._ensure_attr(lidar_prim, name, value)

        self._lidar = Lidar(
            prim_path,
            accumulate_outputs=True,
            aux_output_level="NONE",
            tick_rate=float(prim_overrides["omni:sensor:tickRate"]),
            reset_xform_op_properties=False,
        )
        self._geometry = laser_scan_writer_params(
            scan_type=scan_type,
            scan_rate_hz=scan_rate,
            near_m=self._near_m,
            far_m=self._far_m,
            firing_rate_hz=firing_rate_hz,
            report_rate_hz=report_rate_hz,
            azimuth_deg=self._read_attr(lidar_prim, "omni:sensor:Core:emitterState:s001:azimuthDeg", None),
            fallback_beams=self._num_beams,
        )
        manaprint(
            "INFO: [EX001 lidar] "
            f"type={lidar_prim.GetPrimTypeInfo().GetTypeName()} "
            f"tickRate={self._read_attr(lidar_prim, 'omni:sensor:tickRate', None)} "
            f"scanRateBaseHz={scan_rate} "
            f"scanType={scan_type} "
            f"patternFiringRateHz={firing_rate_hz} "
            f"reportRateBaseHz={report_rate_hz} "
            f"accumulateOutputs={self._read_attr(lidar_prim, 'omni:sensor:Core:accumulateOutputs', None)} "
            f"instantLidar={self._read_attr(lidar_prim, 'omni:sensor:Core:instantLidar', None)} "
            f"elementsCoordsType={self._read_attr(lidar_prim, 'omni:sensor:Core:elementsCoordsType', None)} "
            f"python_scan={self._geometry}"
        )
        self._render_product = rep.create.render_product(
            camera=prim_path,
            resolution=(1, 1),
            render_vars=["GenericModelOutput"],
        )
        self._annotator = rep.AnnotatorRegistry.get_annotator("GenericModelOutput")
        self._annotator.attach([self._render_product.path])
        return True

    def get_point_cloud_data(self) -> dict | None:
        if self._annotator is None:
            return None
        raw = self._annotator.get_data()
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = raw.get("data", raw)
        try:
            from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data
        except ImportError:
            return None
        gmo = parse_generic_model_output_data(raw)
        num_elements = int(getattr(gmo, "numElements", 0) or 0)
        if num_elements <= 0:
            return None
        points = np.column_stack(
            [
                np.asarray(gmo.x, dtype=np.float32)[:num_elements],
                np.asarray(gmo.y, dtype=np.float32)[:num_elements],
                np.asarray(gmo.z, dtype=np.float32)[:num_elements],
            ]
        )
        return {"points": points, "intensities": np.ones(num_elements, dtype=np.float32)}

    def build_laserscan(self, stamp):
        """Build ``sensor_msgs/LaserScan`` with the bridge's current stamp."""
        if self._geometry is None:
            return None
        data = self.get_point_cloud_data()
        if data is None:
            return None
        from maniparena_sim.ros.message_builder import MessageBuilder

        geometry = self._geometry
        ranges, intensities, angle_min, angle_max, angle_increment = bin_points_to_laser_ranges(
            data["points"],
            data["intensities"],
            azimuth_range_deg=geometry["azimuthRange"],
            horizontal_resolution_deg=float(geometry["horizontalResolution"]),
            range_min=self._near_m,
            range_max=self._far_m,
        )
        template = SimpleNamespace(
            angle_min=angle_min,
            angle_max=angle_max,
            angle_increment=angle_increment,
        )
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return MessageBuilder.laserscan(
            template,
            stamp_ns=stamp_ns,
            frame_id=self._frame_id,
            scan_rate_hz=float(geometry["rotationRate"]),
            range_min=self._near_m,
            range_max=self._far_m,
            ranges=ranges,
            intensities=intensities,
        )

    def shutdown(self) -> None:
        if self._annotator is not None:
            try:
                self._annotator.detach()
            except Exception:
                pass
            self._annotator = None
        if self._render_product is not None:
            self._render_product.destroy()
            self._render_product = None
        self._lidar = None
        self._geometry = None

    @staticmethod
    def _ensure_attr(prim, name: str, value) -> None:
        from pxr import Sdf

        attr = prim.GetAttribute(name)
        if not (attr and attr.IsValid()):
            if isinstance(value, bool):
                type_name = Sdf.ValueTypeNames.Bool
            elif isinstance(value, (int, float)):
                type_name = Sdf.ValueTypeNames.Float
            else:
                type_name = Sdf.ValueTypeNames.Token
            attr = prim.CreateAttribute(name, type_name)
        attr.Set(value)

    @staticmethod
    def _read_attr(prim, name: str, default):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid() and attr.Get() is not None:
            return attr.Get()
        return default


def create_ex001_lidar(
    prim_path: str,
    env_id: int = 0,
    *,
    frame_id: str = "a_d_laser",
    topic_name: str = "scan",
    scan_rate_hz: float = 10.0,
    num_beams: int = 3140,
    use_sim_time: bool = False,
) -> RtxLidarHelper:
    """Create EX001 lidar helper. Path must be passed from config."""
    return RtxLidarHelper(
        prim_path,
        env_id,
        frame_id=frame_id,
        topic_name=topic_name,
        scan_rate_hz=scan_rate_hz,
        num_beams=num_beams,
        use_sim_time=use_sim_time,
    )
