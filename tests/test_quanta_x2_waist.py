"""quanta_x2 folding-waist hold: implicit PD plus 0-D gravity overlays."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maniparena_sim.ros.sdk_control import SdkControl
from maniparena_sim.ros.sdk_profiles import HEAD_JOINT_NAMES, WAIST_JOINT_NAMES, WHEEL_JOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANTA_X2_PY = REPO_ROOT / "src" / "maniparena_sim" / "embodiment" / "robots" / "quanta_x2.py"

ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_roll_joint",
    "elbow_yaw_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
)
LEFT_ARM = tuple(f"left_{name}" for name in ARM_JOINT_SUFFIXES)
RIGHT_ARM = tuple(f"right_{name}" for name in ARM_JOINT_SUFFIXES)
GRIPPERS = ("left_gripper", "right_gripper")
BODIES = ("base_link", "left_gripper_base_link", "right_gripper_base_link")


def test_quanta_x2_waist_hold_gains_and_overlays() -> None:
    source = QUANTA_X2_PY.read_text(encoding="utf-8")
    waist = source.split('"waist": ImplicitActuatorCfg(', 1)[1].split('"head": ImplicitActuatorCfg(', 1)[0]
    assert "stiffness=_QUANTA_X2_WAIST_KP" in waist
    assert "damping=_QUANTA_X2_WAIST_KD" in waist
    assert "_QUANTA_X2_WAIST_KP = 32000.0" in source
    assert "_QUANTA_X2_WAIST_KD = 2000.0" in source
    assert "_QUANTA_X2_WAIST_PITCH01_GRAVITY_FF_N = -37.0" in source
    assert "_QUANTA_X2_WAIST_PITCH03_GRAVITY_FF_N = -24.0" in source
    assert "waist_pitch01_gravity_overlay" in source
    assert "waist_pitch03_gravity_overlay" in source
    assert "SafeEffortOverlayActionCfg(" in source
    assert "overlay_joint_names=" not in source


def _term(dim: int, joint_names: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(action_dim=dim, _joint_names=joint_names)


def _robot(joint_names: tuple[str, ...]):
    n = len(joint_names)
    return SimpleNamespace(
        data=SimpleNamespace(
            joint_names=list(joint_names),
            body_names=list(BODIES),
            joint_pos=np.zeros((1, n), dtype=np.float64),
        )
    )


def test_sdk_control_skips_zero_dim_waist_overlays() -> None:
    joint_names = LEFT_ARM + RIGHT_ARM + GRIPPERS + WAIST_JOINT_NAMES + HEAD_JOINT_NAMES + WHEEL_JOINT_NAMES
    terms = {
        "arm_action": _term(7, LEFT_ARM),
        "right_arm_action": _term(7, RIGHT_ARM),
        "left_gripper": _term(1, (GRIPPERS[0],)),
        "right_gripper": _term(1, (GRIPPERS[1],)),
        "waist_action": _term(4, WAIST_JOINT_NAMES),
        "head_action": _term(2, HEAD_JOINT_NAMES),
        "base_action": _term(2, WHEEL_JOINT_NAMES),
        "waist_pitch01_gravity_overlay": _term(0, ("bow_pitch_joint_01",)),
        "waist_pitch03_gravity_overlay": _term(0, ("bow_pitch_joint_03",)),
    }
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            active_terms=list(terms),
            total_action_dim=24,
            get_term=terms.__getitem__,
        )
    )
    actions = np.zeros((1, 24), dtype=np.float64)
    control = SdkControl(env, _robot(joint_names), actions, "quanta_x2", arm_control="joint")
    assert [control.joint_slots[name] for name in WAIST_JOINT_NAMES] == [16, 17, 18, 19]
    assert control.actions.shape == (1, 24)
