"""Arm control mode (``ee`` / ``joint``) state and its ``std_srvs/SetBool`` service.

``SetBool.data`` is ``True`` for ``ee`` and ``False`` for ``joint``. The server
keeps the active mode (initially ``ros.arm_control``) and every response reports
it in ``message``, so a client that asks for the mode already in force gets it
back unchanged and a client that asks for the other one switches it. This module
is import-safe without rclpy.
"""

from __future__ import annotations

from collections.abc import Callable

from maniparena_sim.ros.quanta_x1_sdk_topics import normalize_arm_control

DEFAULT_MODE_SERVICE = "/set_arm_control_mode"


def mode_from_flag(data: bool) -> str:
    return "ee" if data else "joint"


class ArmControlMode:
    """Server-side holder of the active arm control mode."""

    def __init__(self, initial: str = "ee", on_change: Callable[[str], None] | None = None):
        self._mode = normalize_arm_control(initial)
        self._on_change = on_change

    @property
    def mode(self) -> str:
        return self._mode

    def bind(self, on_change: Callable[[str], None]) -> None:
        self._on_change = on_change

    def request(self, mode: str) -> tuple[bool, str]:
        """Switch to ``mode``; return ``(changed, active_mode)``.

        The previous mode stays active when the switch hook raises.
        """
        mode = normalize_arm_control(mode)
        if mode == self._mode:
            return False, self._mode
        if self._on_change is not None:
            self._on_change(mode)
        self._mode = mode
        return True, self._mode


def install_mode_service(node, state: ArmControlMode, name: str = DEFAULT_MODE_SERVICE):
    """Create the ``SetBool`` service on an rclpy node (callbacks run in ``spin_once``)."""
    from std_srvs.srv import SetBool

    def handle(request, response):
        try:
            changed, active = state.request(mode_from_flag(bool(request.data)))
        except Exception as exc:  # report, keep serving the previous mode
            node.get_logger().error(f"arm control mode switch failed: {exc!r}")
            response.success = False
            response.message = state.mode
            return response
        if changed:
            node.get_logger().info(f"arm control mode -> {active}")
        response.success = True
        response.message = active
        return response

    return node.create_service(SetBool, name, handle)
