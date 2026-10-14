"""Mobile pick-and-place objects: trash can plus bottle / fruit / basket pools.

The members are listed in ``assets/mobile_manip/pools.json`` (every ``usd`` path is relative
to ``assets/``). Registered asset names are ``mobile_<name>``: ``mobile_trash_can``,
``mobile_bottle_01``, ``mobile_peach_01``, ``mobile_basket_01``, ... Nothing is registered
when ``pools.json`` is missing; scene assembly then raises with a pointer to it.
"""

from __future__ import annotations

import json
import os

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset

from maniparena_sim.assets import ASSETS_DIR

POOLS_PATH = os.path.join(ASSETS_DIR, "mobile_manip", "pools.json")

_POOL_TAGS = {
    "bottles": ["object", "pickable", "bottle"],
    "fruits": ["object", "pickable", "fruit"],
    "baskets": ["object", "container", "basket"],
}
ASSET_PREFIX = "mobile_"


def load_pools() -> dict:
    if not os.path.isfile(POOLS_PATH):
        return {}
    with open(POOLS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def pool_asset_names(kind: str) -> list[str]:
    """Registered asset names of one pool (``bottles`` / ``fruits`` / ``baskets``), in pool order."""
    return [ASSET_PREFIX + entry["name"] for entry in load_pools().get(kind, [])]


def _scale_tuple(value) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _register_entry(asset_name: str, entry: dict, tags: list[str]) -> None:
    attrs = {
        "name": asset_name,
        "tags": list(tags),
        "usd_path": os.path.join(ASSETS_DIR, entry["usd"]),
        "object_type": ObjectType.RIGID,
        "scale": _scale_tuple(entry.get("scale", 1.0)),
    }
    class_name = "".join(part.capitalize() for part in asset_name.split("_"))
    register_asset(type(class_name, (LibraryObject,), attrs))


def _register_pools() -> None:
    pools = load_pools()
    if not pools:
        return
    if "bin" in pools:
        _register_entry(ASSET_PREFIX + pools["bin"]["name"], pools["bin"], ["object", "container", "place_target", "bin"])
    for kind, tags in _POOL_TAGS.items():
        for entry in pools.get(kind, []):
            _register_entry(ASSET_PREFIX + entry["name"], entry, tags)


_register_pools()
