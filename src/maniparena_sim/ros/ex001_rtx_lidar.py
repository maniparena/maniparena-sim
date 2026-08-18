"""EX001 RTX lidar: official ``RtxLidarROS2PublishLaserScan`` → ``/scan``.

No ``/maniparena/scan_raw`` assembler and no second ``rclpy`` spin thread —
those conflict with the main ROS communicator and can segfault.
"""

from __future__ import annotations

from collections.abc import Sequence

from maniparena_sim.utils.debug_print import manaprint


def laser_scan_writer_params(
    *,
    scan_type: str,
    scan_rate_hz: float,
    near_m: float,
    far_m: float,
    firing_rate_hz: float = 0.0,
    report_rate_hz: float = 0.0,
    azimuth_deg: Sequence[float] | None = None,
    fallback_beams: int = 3600,
) -> dict[str, object]:
    """Build official ``RtxLidarROS2PublishLaserScan`` geometry from OmniLidar attrs."""
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
        firing_rate = float(firing_rate_hz or 0.0)
        beam_count = firing_rate / rotation_rate if firing_rate > 0.0 else 0.0
        if beam_count < 50.0 or beam_count > 20000.0:
            firing_rate = float(report_rate_hz or 0.0)
            beam_count = firing_rate / rotation_rate if firing_rate > 0.0 else 0.0
        if beam_count < 50.0 or beam_count > 20000.0:
            firing_rate = rotation_rate * float(fallback_beams)
        h_res = 360.0 * rotation_rate / firing_rate
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
    """IsaacSimulationGate step so ``/scan`` stays near ``scan_rate_hz``."""
    return max(1, int(round(float(sim_fps) / max(float(scan_rate_hz), 1e-6))))


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
    """Isaac Sim 6 RTX lidar that publishes ``/scan`` via the official writer."""

    def __init__(
        self,
        prim_path: str,
        env_id: int = 0,
        *,
        frame_id: str = "a_d_laser",
        topic_name: str = "scan",
        scan_rate_hz: float = 10.0,
        num_beams: int = 3600,
        use_sim_time: bool = False,
    ):
        del use_sim_time
        self._prim_path = prim_path
        self._env_id = env_id
        self._frame_id = frame_id
        self._topic_name = topic_name
        self._scan_rate_hz = scan_rate_hz
        self._num_beams = num_beams
        self._lidar = None
        self._render_product = None
        self._writer = None
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
        writer_params = laser_scan_writer_params(
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
            f"writer={writer_params}"
        )
        self._render_product = rep.create.render_product(
            camera=prim_path,
            resolution=(1, 1),
            render_vars=["GenericModelOutput"],
        )
        self._writer = rep.writers.get("RtxLidarROS2PublishLaserScan")
        self._writer.initialize(
            frameId=self._frame_id,
            nodeNamespace="",
            queueSize=1,
            topicName=self._topic_name,
            context=0,
            qosProfile="",
            **writer_params,
        )
        rp_path = self._render_product.path
        self._writer.attach([rp_path])
        self._bind_writer_simulation_time(rp_path)
        return True

    def _render_product_path(self) -> str | None:
        path = getattr(self._render_product, "path", None)
        if path is None:
            return None
        if isinstance(path, (list, tuple)):
            path = path[0]
        return str(path)

    def stamp_simulation_time(self, sim_time: float) -> None:
        """Stamp official ``/scan`` with the same Kit timeline used by ``/clock``."""
        path = self._render_product_path()
        if path is None:
            return
        seconds = float(sim_time)
        try:
            import omni.syntheticdata

            sd = omni.syntheticdata.SyntheticData.Get()
            sd.set_node_attributes(
                "IsaacReadSimulationTime",
                {"inputs:resetOnStop": False},
                path,
            )
            for template in ("ROS2PublishLaserScan", "RtxLidarROS2PublishLaserScan"):
                try:
                    sd.set_node_attributes(template, {"inputs:timeStamp": seconds}, path)
                    break
                except Exception:
                    continue
        except Exception as exc:
            manaprint(f"WARNING: [EX001 lidar] simulation-time stamp failed: {exc}")

    @staticmethod
    def _bind_writer_simulation_time(render_product_path) -> None:
        import omni.syntheticdata

        path = render_product_path
        if isinstance(path, (list, tuple)):
            path = path[0]
        path = str(path)
        sd = omni.syntheticdata.SyntheticData.Get()
        try:
            sd.disconnect_node_template(
                "rpFabricTime",
                "IsaacReadSimulationTime",
                path,
                {
                    "outputs:fabricFrameTimeNumerator": "inputs:referenceTimeNumerator",
                    "outputs:fabricFrameTimeDenominator": "inputs:referenceTimeDenominator",
                },
            )
        except Exception as exc:
            manaprint(f"WARNING: [EX001 lidar] rpFabricTime disconnect failed: {exc}")
        try:
            sd.set_node_attributes(
                "IsaacReadSimulationTime",
                {
                    "inputs:resetOnStop": False,
                    "inputs:referenceTimeNumerator": 0,
                    "inputs:referenceTimeDenominator": 0,
                },
                path,
            )
            sd.set_node_attributes(
                "PostProcessDispatchIsaacSimulationGate",
                {"inputs:step": lidar_publish_step()},
                path,
            )
        except Exception as exc:
            manaprint(f"WARNING: [EX001 lidar] IsaacReadSimulationTime bind failed: {exc}")

    def shutdown(self) -> None:
        if self._writer is not None:
            self._writer.detach()
            self._writer = None
        if self._render_product is not None:
            self._render_product.destroy()
            self._render_product = None
        self._lidar = None

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
    num_beams: int = 3600,
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
