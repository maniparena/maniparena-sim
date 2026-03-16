"""Master-slave teleop planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from opencvpr.embodiment.teleop_devices.desktop_master import DesktopMasterCfg, DesktopMasterTeleop
from opencvpr.embodiment.teleop_devices.keyboard import _get_expected_action_dim, _map_action_dim
from opencvpr.planners.teleop_base import TeleopPlanner, TeleopSettings


@dataclass
class MasterSlaveTeleopSettings(TeleopSettings):
    remote_ip: str = "10.0.0.100"
    remote_port: int = 5555
    reconnect_interval: float = 2.0
    debug: bool = False
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class MasterSlaveTeleopPlanner(TeleopPlanner):
    Settings = MasterSlaveTeleopSettings
    device_name = "master_slave"

    def __init__(self):
        super().__init__()
        self._master_controller = None
        self._expected_action_dim = None

    def _init_device(self, env: Any) -> None:
        embodiment = env.embodiment
        embodiment.action_config = embodiment.JointActionsCfg()
        cfg = DesktopMasterCfg(
            remote_ip=self.settings.remote_ip,
            remote_port=self.settings.remote_port,
            sim_device=str(env.device),
            reconnect_interval=self.settings.reconnect_interval,
            debug=self.settings.debug,
            joint_signs=self.settings.joint_signs,
            joint_offsets=self.settings.joint_offsets,
        )
        self._master_controller = DesktopMasterTeleop(cfg)
        self._expected_action_dim = _get_expected_action_dim(env)
        self._master_controller.add_callback("R", lambda: self.signal_done(success=False))
        self._master_controller.add_callback("H", lambda: self.signal_done(success=True))

    def _get_device_action(self, obs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raw = self._master_controller.advance()
        action = _map_action_dim(raw, self._expected_action_dim) if self._expected_action_dim is not None else raw
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        if self._master_controller is not None:
            self._master_controller.reset()
