"""Open head / wrist camera preview windows during teleop collection.

Default Kit ``Viewport`` shows the head camera; left/right wrists open as
extra Omni UI windows that reuse each Camera sensor RenderProduct (manaenv
``stream_viewports`` pattern).
"""

from __future__ import annotations

from typing import Any

from maniparena_sim.utils.viewport_utils import (
    close_viewport_windows,
    open_extra_viewports,
    resolve_sensor_render_product_paths,
)

_CAMERA_PREVIEW_BIND_MAX_ATTEMPTS = 30

# Main Kit Viewport → head; extra windows → left / right wrists (manaenv-like).
_DEFAULT_STREAM_VIEWPORTS: dict[str, Any] = {
    'width': 640,
    'height': 480,
    'position_x': 1280,
    'position_y': 0,
    'main_camera': 'head_camera',
    'cameras': [
        'left_wrist_camera',
        'right_wrist_camera',
    ],
}


def resolve_camera_paths(
    camera_bundle: Any,
    names: list[str] | None = None,
) -> list[tuple[str, str]] | None:
    """Resolve ``(name, prim_path)`` from embodiment ``CameraCfg``."""
    if camera_bundle is None:
        return None
    items = vars(camera_bundle)
    if names:
        selected = [(str(n), items.get(str(n))) for n in names]
    else:
        selected = list(items.items())
    cameras: list[tuple[str, str]] = []
    for name, cam in selected:
        prim_path = getattr(cam, 'prim_path', None)
        if prim_path:
            cameras.append(
                (
                    name,
                    str(prim_path).replace(
                        '{ENV_REGEX_NS}', '/World/envs/env_0',
                    ),
                )
            )
        elif names:
            print(
                f'WARNING: stream_viewports camera {name!r} '
                'not found on embodiment CameraCfg'
            )
    return cameras if cameras else None


def _parse_stream_cfg(
    stream_viewports: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str] | None, dict[str, tuple[int, int]]]:
    sv_cfg = {**_DEFAULT_STREAM_VIEWPORTS, **(stream_viewports or {})}
    names: list[str] | None = None
    size_by_name: dict[str, tuple[int, int]] = {}
    for entry in sv_cfg.get('cameras') or ():
        if names is None:
            names = []
        if isinstance(entry, dict):
            name = str(entry.get('name', '')).strip()
            if not name:
                print(
                    "WARNING: stream_viewports camera entry "
                    "without 'name', skipped"
                )
                continue
            names.append(name)
            if 'width' in entry or 'height' in entry:
                size_by_name[name] = (
                    int(entry.get('width', sv_cfg.get('width', 640))),
                    int(entry.get('height', sv_cfg.get('height', 480))),
                )
        else:
            names.append(str(entry))
    return sv_cfg, names, size_by_name


def _viewport_active() -> bool:
    try:
        import carb

        return bool(
            carb.settings.get_settings().get(
                '/isaaclab/render/active_viewport',
            )
        )
    except Exception:
        return False


def bind_main_viewport_to_camera(
    gym_env: Any,
    camera_path: str,
    *,
    logger: Any | None = print,
) -> bool:
    """Point the default Kit Viewport at a robot camera prim (replace third-person)."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except Exception as exc:
        if logger is not None:
            logger(
                f'WARNING: Cannot bind main Viewport '
                f'(viewport utility unavailable): {exc}'
            )
        return False

    try:
        viewport = get_active_viewport()
        if viewport is None:
            if logger is not None:
                logger('WARNING: No active Kit Viewport to bind head camera.')
            return False
        viewport.set_active_camera(str(camera_path))
    except Exception as exc:
        if logger is not None:
            logger(f'WARNING: Failed to bind main Viewport to {camera_path!r}: {exc}')
        return False

    # Stop Isaac Lab from re-applying third-person eye/lookat each frame.
    env = getattr(gym_env, 'unwrapped', gym_env)
    if hasattr(env, 'viewport_camera_controller'):
        env.viewport_camera_controller = None

    if logger is not None:
        logger(f'INFO: Main Kit Viewport bound to {camera_path!r}.')
    return True


def open_stream_viewports(
    *,
    gym_env: Any,
    camera_bundle: Any,
    stream_viewports: dict[str, Any] | None,
    defer_if_unavailable: bool = False,
) -> list[Any] | None:
    """Bind main Viewport to head; open wrist preview windows.

    Returns:
        Extra wrist window list on success (possibly empty if viewport inactive),
        or ``None`` when sensor render products are not ready yet and
        ``defer_if_unavailable`` is True.
    """
    if not _viewport_active():
        return []

    sv_cfg, names, size_by_name = _parse_stream_cfg(stream_viewports)
    main_name = str(sv_cfg.get('main_camera') or '').strip() or None
    main_paths = (
        resolve_camera_paths(camera_bundle, names=[main_name])
        if main_name
        else None
    )
    if main_name and not main_paths:
        print(
            f'WARNING: stream_viewports main_camera {main_name!r} '
            'not found on embodiment CameraCfg'
        )

    cameras = resolve_camera_paths(camera_bundle, names=names) or []
    if not cameras and not main_paths:
        print(
            'WARNING: No camera prim paths resolved; '
            'skipping stream viewports.'
        )
        return []

    render_product_paths = None
    if cameras:
        render_product_paths = resolve_sensor_render_product_paths(
            gym_env, cameras,
        )
        if render_product_paths is None:
            if defer_if_unavailable:
                return None
            print(
                'WARNING: Camera sensor render products unavailable; '
                'using legacy viewport windows for wrists.'
            )

    preview_logger = None if defer_if_unavailable else print
    windows: list[Any] = []
    if cameras:
        windows = open_extra_viewports(
            count=len(cameras),
            name_prefix='ManipArena Viewport',
            camera_paths=[path for _, path in cameras],
            render_product_paths=render_product_paths,
            sizes=[size_by_name.get(name) for name, _ in cameras],
            width=int(sv_cfg.get('width', 640)),
            height=int(sv_cfg.get('height', 480)),
            position_x=int(sv_cfg.get('position_x', 1280)),
            position_y=int(sv_cfg.get('position_y', 0)),
            canvas_height=(
                int(sv_cfg['canvas_height'])
                if sv_cfg.get('canvas_height') is not None
                else None
            ),
            row_vertical_align=str(sv_cfg.get('row_vertical_align', 'top')),
            vertical_align=str(sv_cfg.get('vertical_align', 'top')),
            allow_legacy_fallback=not defer_if_unavailable,
            logger=preview_logger,
        )
        if len(windows) != len(cameras):
            close_viewport_windows(windows, logger=print)
            if defer_if_unavailable:
                return None
            windows = []

    if main_paths:
        bind_main_viewport_to_camera(
            gym_env, main_paths[0][1], logger=print,
        )

    if cameras and not windows and defer_if_unavailable:
        return None
    return windows


class StreamViewportBinder:
    """Retry binding until camera RenderProducts are initialized."""

    def __init__(
        self,
        *,
        camera_bundle: Any,
        stream_viewports: dict[str, Any] | bool | None,
        enabled: bool = True,
    ):
        self.camera_bundle = camera_bundle
        self.stream_viewports = (
            None if stream_viewports is False else stream_viewports
        )
        self.enabled = (
            bool(enabled)
            and stream_viewports is not False
            and camera_bundle is not None
        )
        self.windows: list[Any] = []
        self._pending = self.enabled
        self._attempts = 0
        self._ready = not self.enabled

    def try_open(self, gym_env: Any) -> bool:
        if not self._pending:
            return self._ready

        self._attempts += 1
        windows = open_stream_viewports(
            gym_env=gym_env,
            camera_bundle=self.camera_bundle,
            stream_viewports=self.stream_viewports,
            defer_if_unavailable=True,
        )
        if windows is not None:
            self.windows = list(windows)
            self._pending = False
            self._ready = True
            print(
                f'INFO: Stream viewports ready '
                f'(main=head, extra={len(self.windows)} wrist window(s)) '
                f'after {self._attempts} binding attempt(s).'
            )
            return True

        if self._attempts < _CAMERA_PREVIEW_BIND_MAX_ATTEMPTS:
            return False

        self._pending = False
        self._ready = False
        print(
            'WARNING: Camera preview render products unavailable after '
            f'{self._attempts} attempts; skipping wrist viewports.'
        )
        return False

    def close(self) -> None:
        close_viewport_windows(self.windows, logger=print)
        self.windows = []
        self._pending = False
        self._ready = False
