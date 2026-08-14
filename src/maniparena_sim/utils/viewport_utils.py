"""Viewport utility helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _sensor_camera_paths(sensor: Any) -> tuple[str, ...]:
    render_data = getattr(sensor, "_render_data", None)
    spec = getattr(render_data, "spec", None)
    initialized_paths = getattr(spec, "camera_prim_paths", ()) or ()
    if initialized_paths:
        return tuple(str(path) for path in initialized_paths)

    configured_path = getattr(getattr(sensor, "cfg", None), "prim_path", None)
    return (str(configured_path),) if configured_path else ()


def _sensor_render_product_path(sensor: Any) -> str | None:
    render_data = getattr(sensor, "_render_data", None)
    paths = getattr(render_data, "render_product_paths", ()) or ()
    if paths:
        return str(paths[0])

    # Isaac Sim 6.0's IsaacRtxRenderData owns one tiled HydraTexture directly
    # instead of exposing the legacy render_product_paths collection.
    render_product = getattr(render_data, "render_product", None)
    path = getattr(render_product, "path", None)
    return str(path) if path else None


def _replicator_viewport_manager() -> Any:
    # Kit 110 aliases the utils package but not its child modules. Import the
    # canonical module used by rep.create to avoid constructing a second manager.
    from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager

    return ViewportManager()


def resolve_sensor_render_product_paths(
    gym_env: Any,
    cameras: list[tuple[str, str]],
) -> list[str] | None:
    """Resolve already-created Camera sensor render products in camera order.

    Returns ``None`` unless every requested camera has an initialized render
    product. Callers can then fall back as one unit instead of mixing sensor
    previews with independently rendered viewport windows.
    """
    env = getattr(gym_env, "unwrapped", gym_env)
    sensors = getattr(getattr(env, "scene", None), "sensors", None)
    if sensors is None:
        return None
    try:
        sensor_items = list(sensors.items())
    except (AttributeError, TypeError):
        return None

    sensor_by_name = {str(name): sensor for name, sensor in sensor_items}
    resolved: list[str] = []
    for name, camera_path in cameras:
        requested_path = str(camera_path)
        sensor = sensor_by_name.get(str(name))
        if sensor is not None and _sensor_camera_paths(sensor) != (requested_path,):
            sensor = None
        if sensor is None:
            sensor = next(
                (candidate for _, candidate in sensor_items if _sensor_camera_paths(candidate) == (requested_path,)),
                None,
            )
        render_product_path = _sensor_render_product_path(sensor) if sensor is not None else None
        if render_product_path is None:
            return None
        resolved.append(render_product_path)
    return resolved


def _find_existing_hydra_texture(render_product_path: str, *, manager: Any = None) -> Any | None:
    """Find the HydraTexture already owned by Replicator without creating one."""
    if manager is None:
        try:
            manager = _replicator_viewport_manager()
        except Exception:
            return None

    multi_texture = getattr(manager, "_hydra_textures", None)
    # Kit 110's public iterator proxies each context list instead of each
    # wrapper. Flatten the manager's read-only groups without attaching a new
    # HydraTexture or render product.
    texture_groups = getattr(multi_texture, "_hydra_textures", {})
    try:
        wrappers = [wrapper for group in texture_groups.values() for wrapper in group]
    except (AttributeError, ReferenceError, TypeError):
        return None
    for wrapper in wrappers:
        try:
            hydra_texture = getattr(wrapper, "hydra_texture", None)
            if hydra_texture is not None and str(hydra_texture.get_render_product_path()) == str(render_product_path):
                return hydra_texture
        except (AttributeError, ReferenceError, RuntimeError):
            continue
    return None


class _StaticViewportApi:
    """Read-only viewport subset used by display-only overlays."""

    def __init__(self, camera_path: str, render_product_path: str):
        self._camera_path = str(camera_path)
        self._render_product_path = str(render_product_path)
        self._texture_resolution: tuple[int, int] | None = None

    def get_active_camera(self) -> str:
        return self._camera_path

    def get_render_product_path(self) -> str:
        return self._render_product_path

    def get_texture_resolution(self) -> tuple[int, int] | None:
        return self._texture_resolution

    def set_texture_resolution(self, resolution: Any) -> None:
        try:
            width, height = int(resolution[0]), int(resolution[1])
        except (IndexError, TypeError, ValueError):
            return
        if width > 0 and height > 0:
            self._texture_resolution = (width, height)


class _SensorRenderProductPreview:
    """Omni UI window that displays an existing HydraTexture GPU resource."""

    def __init__(
        self,
        *,
        name: str,
        width: int,
        height: int,
        position_x: int,
        position_y: int,
        camera_path: str,
        render_product_path: str,
        hydra_texture: Any,
        ui: Any,
        event_dispatcher: Any,
        drawable_event_name: Any,
        logger: Callable[[str], None] | None = None,
    ):
        self.viewport_api = _StaticViewportApi(camera_path, render_product_path)
        self._ui = ui
        self._hydra_texture = hydra_texture
        self._logger = logger
        self._name = str(name)
        self._drawable_error_reported = False
        self._reported_texture_resolution: tuple[int, int] | None = None
        self._destroyed = False
        self._drawable_change_sub = None
        self._overlay_frames: dict[str, Any] = {}
        self.window = None
        self._stack = None
        self._image_provider = None
        self._image = None

        try:
            self.window = ui.Window(
                name,
                width=int(width),
                height=int(height),
                position_x=int(position_x),
                position_y=int(position_y),
            )
            with self.window.frame:
                self._stack = ui.ZStack()
                with self._stack:
                    ui.Rectangle(style={"background_color": 0xFF000000})
                    self._image_provider = ui.ImageProvider()
                    self._image = ui.ImageWithProvider(
                        self._image_provider,
                        alignment=ui.Alignment.CENTER,
                        fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                    )

            self._drawable_change_sub = event_dispatcher.observe_event(
                observer_name=f"maniparena_sim.SensorRenderProductPreview.{name}",
                event_name=drawable_event_name,
                filter=hydra_texture.get_event_key(),
                on_event=self._on_drawable_changed,
            )
        except Exception:
            self.destroy()
            raise

    def _on_drawable_changed(self, event: Any) -> None:
        provider = self._image_provider
        hydra_texture = self._hydra_texture
        if provider is None or hydra_texture is None:
            return
        try:
            result_handle = event["result_handle"]
            aov_info = hydra_texture.get_aov_info(result_handle, aov_name="LdrColor", include_texture=True)
            if not aov_info:
                aov_info = hydra_texture.get_aov_info(result_handle, aov_name="", include_texture=True)
            if not aov_info:
                raise RuntimeError("render product has no displayable color AOV")
            texture = aov_info[0]["texture"]
            resource = texture["rp_resource"]
            viewport_api = getattr(self, "viewport_api", None)
            set_resolution = getattr(viewport_api, "set_texture_resolution", None)
            if callable(set_resolution):
                set_resolution(texture.get("resolution"))
            get_resolution = getattr(viewport_api, "get_texture_resolution", None)
            texture_resolution = get_resolution() if callable(get_resolution) else None
            if (
                self._logger is not None
                and texture_resolution is not None
                and texture_resolution != getattr(self, "_reported_texture_resolution", None)
            ):
                self._logger(
                    f"INFO: Sensor preview {getattr(self, '_name', '<unnamed>')!r} displays "
                    f"{texture_resolution[0]}x{texture_resolution[1]} render-product pixels."
                )
                self._reported_texture_resolution = texture_resolution
            presentation_key = event["presentation_key"] or 0
            provider.set_image_data(resource, presentation_key)
        except Exception as exc:
            if self._logger is not None and not self._drawable_error_reported:
                self._logger(f"WARNING: Sensor render-product preview update failed: {exc}")
            self._drawable_error_reported = True

    def get_frame(self, layer_name: str) -> Any:
        """Return a named frame above the sensor image for viewport overlays."""
        frame = self._overlay_frames.get(layer_name)
        if frame is None:
            with self._stack:
                frame = self._ui.Frame()
            self._overlay_frames[layer_name] = frame
        return frame

    def destroy(self) -> None:
        """Release preview-owned UI state without touching the sensor texture."""
        if self._destroyed:
            return
        self._destroyed = True

        def cleanup(resource: Any, method_name: str, description: str) -> None:
            if resource is None:
                return
            try:
                method = getattr(resource, method_name, None)
                if callable(method):
                    method()
            except Exception as exc:
                if self._logger is not None:
                    try:
                        self._logger(f"WARNING: Failed to {description}: {exc}")
                    except Exception:
                        pass

        subscription, self._drawable_change_sub = self._drawable_change_sub, None
        cleanup(subscription, "reset", "reset sensor preview subscription")

        overlay_frames = list(self._overlay_frames.values())
        self._overlay_frames.clear()
        for frame in overlay_frames:
            cleanup(frame, "clear", "clear sensor preview overlay")

        image, self._image = self._image, None
        cleanup(image, "destroy", "destroy sensor preview image")

        stack, self._stack = self._stack, None
        cleanup(stack, "clear", "clear sensor preview stack")

        window, self.window = self.window, None
        cleanup(window, "destroy", "destroy sensor preview window")

        provider, self._image_provider = self._image_provider, None
        cleanup(provider, "destroy", "destroy sensor preview image provider")

        self._hydra_texture = None


def _open_sensor_render_product_previews(
    *,
    count: int,
    name_prefix: str,
    width: int,
    height: int,
    position_x: int,
    positions_y: list[int],
    camera_paths: list[str],
    render_product_paths: list[str],
    sizes: list[tuple[int, int] | None] | None,
    logger: Callable[[str], None] | None,
) -> list[Any]:
    count = max(int(count), 0)
    if count == 0:
        return []
    if len(camera_paths) < count or len(render_product_paths) < count:
        return []

    try:
        import omni.hydratexture
        import omni.ui as ui
        from carb.eventdispatcher import get_eventdispatcher
    except Exception as exc:
        if logger is not None:
            logger(f"WARNING: Sensor render-product preview APIs unavailable: {exc}")
        return []

    try:
        manager = _replicator_viewport_manager()
    except Exception as exc:
        if logger is not None:
            logger(f"WARNING: Replicator viewport manager unavailable: {exc}")
        return []
    textures = [_find_existing_hydra_texture(render_product_paths[idx], manager=manager) for idx in range(count)]
    if any(texture is None for texture in textures):
        if logger is not None:
            missing = [render_product_paths[idx] for idx, texture in enumerate(textures) if texture is None]
            logger(f"WARNING: Existing sensor HydraTexture not found for {missing}; using legacy viewports.")
        return []

    windows: list[Any] = []
    next_x = int(position_x)
    try:
        for idx, hydra_texture in enumerate(textures):
            window_name = f"{name_prefix} {idx + 1}"
            vp_w, vp_h = int(width), int(height)
            if sizes and idx < len(sizes) and sizes[idx]:
                vp_w, vp_h = int(sizes[idx][0]), int(sizes[idx][1])
            windows.append(
                _SensorRenderProductPreview(
                    name=window_name,
                    width=vp_w,
                    height=vp_h,
                    position_x=next_x,
                    position_y=int(positions_y[idx]),
                    camera_path=camera_paths[idx],
                    render_product_path=render_product_paths[idx],
                    hydra_texture=hydra_texture,
                    ui=ui,
                    event_dispatcher=get_eventdispatcher(),
                    drawable_event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
                    logger=logger,
                )
            )
            next_x += vp_w
            if logger is not None:
                logger(
                    f"INFO: {window_name!r} reuses sensor render product "
                    f"{render_product_paths[idx]!r} for camera {camera_paths[idx]!r}"
                )
    except Exception as exc:
        close_viewport_windows(windows)
        if logger is not None:
            logger(f"WARNING: Failed to open sensor render-product previews: {exc}")
        return []
    return windows


def close_viewport_windows(
    windows: list[Any] | None,
    *,
    logger: Callable[[str], None] | None = None,
) -> None:
    """Destroy preview or legacy viewport windows without owning their cameras."""
    for viewport in reversed(list(windows or [])):
        try:
            destroy = getattr(viewport, "destroy", None)
            if callable(destroy):
                destroy()
                continue
            window = getattr(viewport, "window", None)
            destroy = getattr(window, "destroy", None)
            if callable(destroy):
                destroy()
        except Exception as exc:
            if logger is not None:
                logger(f"WARNING: Failed to close viewport window: {exc}")


def reapply_viewport_camera_view(
    gym_env: Any,
    *,
    logger: Callable[[str], None] | None = None,
) -> bool:
    """Re-point the interactive Kit Perspective viewport at the configured view."""
    env = getattr(gym_env, "unwrapped", gym_env)
    env_cfg = getattr(env, "cfg", None)
    viewer_cfg = getattr(env_cfg, "viewer", None)
    sim_cfg = getattr(env_cfg, "sim", None)
    visualizer_cfg = getattr(sim_cfg, "default_visualizer_cfg", None)

    # Older IsaacLab releases owned the camera through this controller. It only
    # understands ViewerCfg, so use it solely when no new-style visualizer pose
    # is available.
    controller = getattr(env, "viewport_camera_controller", None)
    if visualizer_cfg is None and controller is not None:
        try:
            controller.update_view_location()
            if logger is not None:
                logger("INFO: Re-applied Perspective viewport camera via ViewportCameraController.")
            return True
        except Exception as exc:
            if logger is not None:
                logger(f"WARNING: ViewportCameraController.update_view_location failed: {exc}")

    camera_cfg = visualizer_cfg if visualizer_cfg is not None else viewer_cfg
    if camera_cfg is None:
        return False
    try:
        from isaacsim.core.rendering_manager import ViewportManager
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        ViewportManager.set_camera_view(
            str(getattr(viewer_cfg, "cam_prim_path", "/OmniverseKit_Persp")),
            eye=list(camera_cfg.eye),
            target=list(camera_cfg.lookat),
        )
    except Exception as exc:
        if logger is not None:
            logger(f"WARNING: ViewportManager.set_camera_view failed: {exc}")
        return False
    if logger is not None:
        logger("INFO: Re-applied Perspective viewport camera from visualizer cfg.")
    return True


def open_extra_viewports(
    *,
    count: int = 2,
    name_prefix: str = "ManipArena Viewport",
    width: int = 640,
    height: int = 480,
    position_x: int = 1280,
    position_y: int = 0,
    canvas_height: int | None = None,
    row_vertical_align: str = "top",
    camera_paths: list[str] | None = None,
    render_product_paths: list[str] | None = None,
    sizes: list[tuple[int, int] | None] | None = None,
    vertical_align: str = "top",
    allow_legacy_fallback: bool = True,
    logger: Callable[[str], None] | None = None,
) -> list[Any]:
    """Open extra viewport windows and optionally bind them to camera prims.

    ``sizes`` optionally gives a per-viewport (width, height) aligned with the
    index; ``None`` entries (or no ``sizes``) fall back to ``width``/``height``.
    Windows are laid out in a row, advancing by each window's own width.
    ``row_vertical_align`` optionally aligns the complete row within
    ``canvas_height``; ``position_y`` remains an offset from that aligned row
    origin.  Within the row, ``vertical_align`` aligns each window against the
    tallest item.
    ``allow_legacy_fallback=False`` returns no windows when an existing sensor
    render-product preview cannot be opened, instead of creating independent
    Kit viewport render products.
    """
    viewport_count = max(int(count), 0)
    if viewport_count == 0:
        return []

    resolved_sizes: list[tuple[int, int]] = []
    for idx in range(viewport_count):
        vp_w, vp_h = int(width), int(height)
        if sizes and idx < len(sizes) and sizes[idx]:
            vp_w, vp_h = int(sizes[idx][0]), int(sizes[idx][1])
        resolved_sizes.append((vp_w, vp_h))
    row_height = max((vp_h for _, vp_h in resolved_sizes), default=int(height))
    row_align = str(row_vertical_align).strip().lower()
    if row_align not in {"top", "center", "bottom"}:
        if logger is not None:
            logger(f"WARNING: Unknown row_vertical_align={row_vertical_align!r}; using 'top'.")
        row_align = "top"
    row_position_y = int(position_y)
    if canvas_height is not None:
        row_space = int(canvas_height) - row_height
        if row_align == "center":
            row_position_y += row_space // 2
        elif row_align == "bottom":
            row_position_y += row_space

    align = str(vertical_align).strip().lower()
    if align not in {"top", "center", "bottom"}:
        if logger is not None:
            logger(f"WARNING: Unknown vertical_align={vertical_align!r}; using 'top'.")
        align = "top"
    positions_y = [
        row_position_y
        + (row_height - vp_h if align == "bottom" else (row_height - vp_h) // 2 if align == "center" else 0)
        for _, vp_h in resolved_sizes
    ]

    if render_product_paths is not None and camera_paths is not None:
        previews = _open_sensor_render_product_previews(
            count=viewport_count,
            name_prefix=name_prefix,
            width=width,
            height=height,
            position_x=position_x,
            positions_y=positions_y,
            camera_paths=camera_paths,
            render_product_paths=render_product_paths,
            sizes=sizes,
            logger=logger,
        )
        if len(previews) == viewport_count:
            return previews
        close_viewport_windows(previews, logger=logger)
        if not allow_legacy_fallback:
            if logger is not None:
                logger("WARNING: Sensor render-product previews unavailable; legacy viewport fallback disabled.")
            return []

    try:
        from omni.kit.viewport.utility import create_viewport_window
    except Exception as exc:
        if logger is not None:
            logger(f"WARNING: Cannot import viewport utilities: {exc}")
        return []

    windows = []
    next_x = int(position_x)
    for idx in range(viewport_count):
        window_name = f"{name_prefix} {idx + 1}"
        vp_w, vp_h = resolved_sizes[idx]
        try:
            vp = create_viewport_window(
                name=window_name,
                width=vp_w,
                height=vp_h,
                position_x=next_x,
                position_y=positions_y[idx],
            )
            windows.append(vp)
            next_x += vp_w
        except Exception as exc:
            if logger is not None:
                logger(f"WARNING: Failed to open viewport {window_name!r}: {exc}")
            continue
        if camera_paths and idx < len(camera_paths) and camera_paths[idx]:
            try:
                vp.viewport_api.set_active_camera(camera_paths[idx])
                if logger is not None:
                    logger(f"INFO: Viewport {window_name!r} bound to camera {camera_paths[idx]!r}")
            except Exception as exc:
                if logger is not None:
                    logger(f"WARNING: Failed to bind camera on {window_name!r}: {exc}")
        elif logger is not None:
            logger(f"INFO: Opened extra viewport {window_name!r}")
    return windows
