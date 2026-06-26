from __future__ import annotations

import omni.replicator.core as rep
import omni.usd


class RtxLidarHelper:
    def __init__(self, prim_path: str, env_id: int = 0):
        self._prim_path = prim_path
        self._env_id = env_id
        self.scan_buffer_annotator = None
        self.flat_scan_annotator = None

    def initialize(self, scene) -> bool:
        prim_path = self._prim_path.replace("{ENV_REGEX_NS}", f"{scene.env_ns}/env_{self._env_id}")

        stage = omni.usd.get_context().get_stage()
        lidar_prim = stage.GetPrimAtPath(prim_path)

        if not lidar_prim.IsValid():
            return False

        hydra_texture = rep.create.render_product(
            lidar_prim.GetPath(),
            [1, 1],
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )

        self.scan_buffer_annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBufferForFlatScan")
        self.scan_buffer_annotator.attach([hydra_texture.path])

        self.flat_scan_annotator = rep.AnnotatorRegistry.get_annotator("IsaacComputeRTXLidarFlatScan")
        self.flat_scan_annotator.attach([hydra_texture.path])

        return True

    def get_flat_scan_data(self) -> dict:
        if self.flat_scan_annotator is None:
            return None
        return self.flat_scan_annotator.get_data()


def create_ex001_lidar(prim_path: str, env_id: int = 0) -> RtxLidarHelper:
    """Create EX001 lidar helper. Path must be passed from config."""
    return RtxLidarHelper(prim_path, env_id)
