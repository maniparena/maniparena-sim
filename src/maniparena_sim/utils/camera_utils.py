"""Camera-related utility helpers."""

from __future__ import annotations

from collections.abc import Callable


def deactivate_robot_camera_prims(
    *,
    robot_path_token: str = '/Robot/',
    logger: Callable[[str], None] | None = None,
) -> int:
    """Deactivate USD Camera prims under the robot path token.

    Returns:
        Number of camera prims successfully deactivated.
    """
    try:
        from isaacsim.core.utils import stage as stage_utils
    except Exception as exc:
        if logger is not None:
            logger(
                f'WARNING: Cannot import stage utils '
                f'to disable cameras: {exc}'
            )
        return 0

    stage = stage_utils.get_current_stage()
    if stage is None:
        return 0

    disabled = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Camera':
            continue
        prim_path = str(prim.GetPath())
        if robot_path_token not in prim_path:
            continue
        try:
            prim.SetActive(False)
            disabled += 1
        except Exception:
            continue
    return disabled
