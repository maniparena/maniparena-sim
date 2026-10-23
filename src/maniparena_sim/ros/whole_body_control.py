"""Quanta X2 dual-flange QP controller with waist ownership in EE mode.

Targets and measurements are in base_link; public quaternions remain XYZW.
No learned policy is loaded. The packaged geometric model is tied to the
bundled X2 USD. Collision/COM inequalities are proxies, not a motion planner.
"""
from pathlib import Path
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from maniparena_sim.kinematics.whole_body_qp import Model, WholeBodyQP, error
from maniparena_sim.kinematics.whole_body_recovery import StationaryGoalRecovery
from maniparena_sim.kinematics.ik import IkResult
from maniparena_sim.kinematics.se3 import invert
from maniparena_sim.ros.sdk_control import SdkControl, _body_pose_matrix
from maniparena_sim.ros.sdk_profiles import get_sdk_profile, WAIST_JOINT_NAMES

MODEL_PATH = Path(__file__).resolve().parents[1] / 'kinematics/data/quanta_x2.json'
USD_SHA256 = '9882a513f6e71a5712585bc772ee2b97415e69e936b396bf49631ee7a76d42bd'


class WholeBodySdkControl(SdkControl):
    def __init__(self, env, robot, actions, profile, *args, feedback_gain=4., **kwargs):
        if get_sdk_profile(profile).name != 'quanta_x2':
            raise ValueError('whole_body_qp supports only the bundled quanta_x2 robot')
        usd = Path(robot.cfg.spawn.usd_path)
        if hashlib.sha256(usd.read_bytes()).hexdigest() != USD_SHA256:
            raise ValueError('whole_body_qp model does not match this USD; use ik_backend: arm')
        self.model = Model(MODEL_PATH)
        self.control_dt = float(env.step_dt)
        if not np.isfinite(self.control_dt) or not 0 < self.control_dt <= .1:
            raise ValueError('whole_body_qp requires a finite control period in (0, 0.1] seconds')
        self.feedback_gain = float(feedback_gain)
        if not 0 < self.feedback_gain * self.control_dt <= .5:
            raise ValueError('feedback gain times control period must be in (0, 0.5]')
        self.qp = WholeBodyQP(self.model, dt=self.control_dt)
        self.recovery = StationaryGoalRecovery(self.qp)
        self.active = False
        self.ignored_waist_commands = 0
        super().__init__(env, robot, actions, profile, *args, **kwargs)
        self.ids = [list(robot.data.joint_names).index(n) for n in self.model.names]
        self.slots = [self.joint_slots[n] for n in self.model.names]
        self.capture()

    def joint_command(self, msg, names):
        # Keep the SDK subscription set stable; the QP is the sole writer of
        # waist targets in EE mode. Joint mode preserves the original contract.
        if self.mode == 'ee' and tuple(names) == tuple(WAIST_JOINT_NAMES):
            self.ignored_waist_commands += 1
            return
        super().joint_command(msg, names)

    def pose_command(self, msg, side):
        previous = self.ee_commands[side]
        super().pose_command(msg, side)
        if self.ee_commands[side] is not previous and not self.active:
            self.capture()
            self.active = True

    def measured(self):
        base = _body_pose_matrix(self.robot, self.mapping.base)
        hands = np.array([invert(base) @ _body_pose_matrix(self.robot, f) for f in self.mapping.flanges])
        q = np.array([float(self.robot.data.joint_pos[0, i]) for i in self.ids])
        return hands, q, base[:3, :3].T @ np.array([0., 0., -1.])

    def capture(self):
        hands, _, _ = self.measured()
        self.hold = hands.copy()
        command_q = np.array([float(self.actions[0, s]) for s in self.slots])
        commanded = self.model.forward(command_q)['hands']
        self.offset_pos = np.clip(commanded[:, :3, 3] - hands[:, :3, 3], -.04, .04)
        self.offset_rot = np.clip(Rotation.from_matrix(
            commanded[:, :3, :3] @ hands[:, :3, :3].transpose(0, 2, 1)).as_rotvec(), -.05, .05)
        self.previous_target = hands.copy()
        self.previous_hands = hands.copy()
        self.stationary_time = 0.
        self.qp.reset(command_q)
        self.recovery.reset(command_q)

    def reset(self, robot=None):
        super().reset(robot)
        self.active = False
        if hasattr(self, 'ids'):
            self.capture()

    def set_mode(self, mode):
        old = self.mode
        callbacks = super().set_mode(mode)
        if old != self.mode:
            self.active = False
            self.capture()
        return callbacks

    def project_ee_pose_commands(self, robot=None):
        if self.mode != 'ee' or not self.active:
            return
        target = self.hold.copy()
        for side, command in enumerate(self.ee_commands):
            if command is not None:
                target[side, :3, 3] = command[:3]
                target[side, :3, :3] = Rotation.from_quat(command[3:]).as_matrix()
        hands, q, gravity = self.measured()
        e = error(hands, self.previous_target)
        stable = (np.linalg.norm(e[:, :3], axis=1) < .02) & (np.linalg.norm(e[:, 3:], axis=1) < .05)
        # An alternate branch can carry greater gravity sag than the initial
        # pose. Once joints have settled, admit that bounded model error so
        # the feedback gate does not permanently strand a reached IK target.
        measured_step = error(self.previous_hands, hands)
        stationary = (np.max(np.linalg.norm(measured_step[:,:3],axis=1)) < 2e-4 and
                      np.max(np.linalg.norm(measured_step[:,3:],axis=1)) < 1e-3)
        self.stationary_time = self.stationary_time + self.control_dt if stationary else 0.
        if self.stationary_time >= .25:
            stable |= ((np.linalg.norm(e[:, :3], axis=1) < .08) &
                       (np.linalg.norm(e[:, 3:], axis=1) < .15))
        stable &= np.array(self.qp.info.get('pos_mm', [0, 0])) < 2.
        if self.recovery.busy:
            stable[:] = False
        self.offset_pos[stable] += self.feedback_gain * self.control_dt * e[stable, :3]
        self.offset_rot[stable] += self.feedback_gain * self.control_dt * e[stable, 3:]
        self.offset_pos = np.clip(self.offset_pos, -.08, .08)
        self.offset_rot = np.clip(self.offset_rot, -.15, .15)
        corrected = target.copy()
        corrected[:, :3, 3] += self.offset_pos
        corrected[:, :3, :3] = Rotation.from_rotvec(self.offset_rot).as_matrix() @ target[:, :3, :3]
        self.qp.gravity_b = gravity
        out = self.recovery.update(target, corrected, gravity)
        if out is None:
            out = self.qp.solve(corrected, q, iterations=2)
        else:
            residual = error(self.model.forward(out)['hands'], corrected)
            self.qp.info = dict(pos_mm=(1000*np.linalg.norm(residual[:,:3],axis=1)).tolist(),
                                rot_deg=np.rad2deg(np.linalg.norm(residual[:,3:],axis=1)).tolist(),
                                status=[])
        self.qp.info['recovery'] = self.recovery.state
        self.previous_target = target.copy()
        self.previous_hands = hands.copy()
        measured_error = error(hands, target)
        for side in range(2):
            pe = float(np.linalg.norm(measured_error[side, :3]))
            re = float(np.linalg.norm(measured_error[side, 3:]))
            self.ik_results[side] = IkResult(out[4+7*side:11+7*side].copy(), pe, re,
                                           len(self.qp.info['status']), pe < 1e-4 and re < 1e-3)
        for slot, value in zip(self.slots, out):
            self.actions[0, slot] = float(value)
