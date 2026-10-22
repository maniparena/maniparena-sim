"""Extract a serial arm chain from the PhysX joints authored in a robot USD.

The joint frames (``localPos0/localRot0/localPos1/localRot1``) are the exact
data PhysX simulates, so the chain needs no URDF and agrees with the sim body
poses. ``pxr`` is imported lazily so the rest of the package stays importable
without a USD runtime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from maniparena_sim.kinematics.chain import KinematicChain
from maniparena_sim.kinematics.se3 import invert, pose_to_matrix

_AXES = {"X": 0, "Y": 1, "Z": 2}


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str  # "revolute" | "fixed" | "other"
    parent: str
    child: str
    frame0: np.ndarray
    frame1: np.ndarray
    axis: int | None
    lower: float
    upper: float


def _frame(pos, rot) -> np.ndarray:
    imag = rot.GetImaginary()
    return pose_to_matrix(
        (pos[0], pos[1], pos[2]),
        (imag[0], imag[1], imag[2], rot.GetReal()),
    )


@lru_cache(maxsize=8)
def _read_joints(usd_path: str) -> dict[str, _Joint]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"cannot open USD: {usd_path}")
    joints: dict[str, _Joint] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parents = joint.GetBody0Rel().GetTargets()
        children = joint.GetBody1Rel().GetTargets()
        if len(parents) != 1 or len(children) != 1:
            continue
        kind, axis, lower, upper = "other", None, -np.inf, np.inf
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rev = UsdPhysics.RevoluteJoint(prim)
            kind = "revolute"
            axis = _AXES[str(rev.GetAxisAttr().Get() or "X")]
            lo, hi = rev.GetLowerLimitAttr().Get(), rev.GetUpperLimitAttr().Get()
            lower = np.deg2rad(lo) if lo is not None and np.isfinite(lo) else -np.inf
            upper = np.deg2rad(hi) if hi is not None and np.isfinite(hi) else np.inf
        elif prim.IsA(UsdPhysics.FixedJoint):
            kind = "fixed"
        joints[prim.GetName()] = _Joint(
            name=prim.GetName(),
            kind=kind,
            parent=parents[0].name,
            child=children[0].name,
            frame0=_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            frame1=_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
            axis=axis,
            lower=float(lower),
            upper=float(upper),
        )
    return joints


def extract_chain(usd_path: str, joint_names: list[str] | tuple[str, ...], end_body: str) -> KinematicChain:
    """Chain from the parent body of ``joint_names[0]`` to ``end_body``.

    Fixed joints between the actuated ones are folded into constants. The
    revolute joints met on the path must be exactly ``joint_names`` in order.
    """
    joints = _read_joints(str(usd_path))
    missing = [name for name in joint_names if name not in joints]
    if missing:
        raise ValueError(f"{usd_path}: joints not found in USD: {missing}")
    base_body = joints[joint_names[0]].parent
    out_edges: dict[str, list[_Joint]] = {}
    for joint in joints.values():
        if joint.kind in ("revolute", "fixed"):
            out_edges.setdefault(joint.parent, []).append(joint)

    previous: dict[str, tuple[str, _Joint]] = {}
    queue = deque([base_body])
    seen = {base_body}
    while queue and end_body not in seen:
        body = queue.popleft()
        for joint in out_edges.get(body, ()):
            if joint.child not in seen:
                seen.add(joint.child)
                previous[joint.child] = (body, joint)
                queue.append(joint.child)
    if end_body not in seen:
        raise ValueError(f"{usd_path}: no joint path from {base_body!r} to {end_body!r}")

    path: list[_Joint] = []
    body = end_body
    while body != base_body:
        body, joint = previous[body]
        path.append(joint)
    path.reverse()

    actuated = [j.name for j in path if j.kind == "revolute"]
    if actuated != list(joint_names):
        raise ValueError(f"{usd_path}: path {base_body}->{end_body} actuates {actuated}, expected {list(joint_names)}")

    constants: list[np.ndarray] = []
    axes: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    K = np.eye(4)
    for joint in path:
        if joint.kind == "fixed":
            K = K @ joint.frame0 @ invert(joint.frame1)
        else:
            constants.append(K @ joint.frame0)
            axes.append(joint.axis)
            lower.append(joint.lower)
            upper.append(joint.upper)
            K = invert(joint.frame1)
    constants.append(K)
    return KinematicChain(
        joint_names=tuple(joint_names),
        constants=tuple(constants),
        axes=tuple(axes),
        lower=np.array(lower, dtype=np.float64),
        upper=np.array(upper, dtype=np.float64),
        base_body=base_body,
        end_body=end_body,
    )
