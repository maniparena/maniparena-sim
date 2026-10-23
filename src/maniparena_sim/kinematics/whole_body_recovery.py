"""Bounded background branch search for a stationary, stalled Cartesian goal.

This checks sampled joint paths against the same geometric proxies as the QP;
it is not a mesh collision planner. Moving Cartesian commands cancel recovery.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

import numpy as np
from scipy.optimize import minimize

from .whole_body_qp import error


class _Cancelled(Exception):
    pass


def find_recovery_path(model, start, home, target, gravity, cancel, timeout=20., history=None):
    """Search independent endpoint branches and validate direct/retreat paths."""
    began = time.monotonic()
    weights = np.tile([1., 1., 1., .3, .3, .3], 2)
    rng = np.random.default_rng(20261022)

    def check():
        if cancel.is_set() or time.monotonic() - began > timeout:
            raise _Cancelled()

    def constraints(q):
        check()
        return model.constraints(model.forward(q), com_margin=.065, gravity_b=gravity)

    try:
        floor = np.minimum(constraints(start)[0], 0.) - 2e-6
    except _Cancelled:
        return None

    def segment(a, b):
        count = max(2, int(np.ceil(np.max(abs(b-a))/.02)) + 1)
        for u in np.linspace(0., 1., count):
            if np.any(constraints(a + u*(b-a))[0] < floor):
                return False
        return True

    def objective(q):
        check()
        fk = model.forward(q)
        e = error(fk['hands'], target).ravel()
        return np.sum((weights*e)**2), -2*fk['jac'].reshape(12,18).T @ (weights**2*e)

    def bent_path(goal):
        """Fit one intermediate posture; add checks where coarse samples miss."""
        middle = .5*(start+goal)
        waypoint = middle.copy()
        posture_weights = np.r_[np.full(4, 4.), np.ones(14)]
        samples = [list(np.linspace(.05, .95, 10)) for _ in range(2)]
        for _ in range(6):
            cached_q, cached_result = None, None

            def path_constraints(q):
                nonlocal cached_q, cached_result
                check()
                if cached_q is not None and np.array_equal(q, cached_q):
                    return cached_result
                values, jacobians = [], []
                for side, (a, b) in enumerate(((start, q), (q, goal))):
                    for u in sorted(set(samples[side])):
                        g, jac = constraints((1-u)*a+u*b)
                        # Taper a 1 mm interior buffer to zero at endpoints.
                        # It prevents narrow box-face crossings between samples.
                        values.append(g-floor-.002*min(u, 1-u))
                        jacobians.append(jac*(u if side == 0 else 1-u))
                cached_q = q.copy()
                cached_result = np.concatenate(values), np.vstack(jacobians)
                return cached_result

            result = minimize(
                lambda q: (np.sum(posture_weights*(q-middle)**2), 2*posture_weights*(q-middle)),
                waypoint, jac=True, method='SLSQP', bounds=list(zip(model.lower, model.upper)),
                constraints={'type':'ineq', 'fun':lambda q:path_constraints(q)[0],
                             'jac':lambda q:path_constraints(q)[1]},
                options={'maxiter':40, 'ftol':1e-10})
            waypoint = result.x
            valid = True
            for side, (a, b) in enumerate(((start, waypoint), (waypoint, goal))):
                count = max(1001, int(np.ceil(np.max(abs(b-a))/.002))+1)
                violations = [(float(np.min(constraints((1-u)*a+u*b)[0]-floor)), u)
                              for u in np.linspace(0., 1., count)]
                bad = [(v, u) for v, u in violations if v < 0.]
                if bad:
                    valid = False
                    samples[side].extend(u for _, u in sorted(bad)[:3])
            if valid:
                return [start, waypoint, goal]
        return None

    try:
        retreat = None
        retreat_checked = False
        blocked = []
        for index in range(24):
            if index >= 12 and len(blocked) >= 6:
                break
            seed = home.copy() if index == 0 else np.clip(
                home + rng.normal(0., .9, 18), model.lower+.001, model.upper-.001)
            result = minimize(objective, seed, jac=True, method='SLSQP',
                              bounds=list(zip(model.lower, model.upper)),
                              constraints={'type':'ineq', 'fun':lambda q:constraints(q)[0],
                                           'jac':lambda q:constraints(q)[1]},
                              options={'maxiter':150, 'ftol':1e-12})
            goal = result.x
            e = error(model.forward(goal)['hands'], target)
            if (np.max(np.linalg.norm(e[:,:3], axis=1)) > 1e-4 or
                    np.max(np.linalg.norm(e[:,3:], axis=1)) > 1e-3 or
                    constraints(goal)[0].min() < -1e-6):
                continue
            if segment(start, goal):
                return [start, goal]
            if not retreat_checked:
                retreat_checked = True
                if segment(start, home):
                    retreat = [start, home]
                elif history is not None:
                    # A straight chord back to home can cut through the torso,
                    # even though the actual approach was collision-free.
                    # Retrace checked commands, shortcutting only valid chords.
                    trail = [start] + list(reversed(history))
                    candidate = [start]
                    cursor = 0
                    while cursor < len(trail)-1:
                        next_index = cursor+1
                        if not segment(trail[cursor], trail[next_index]):
                            candidate = None
                            break
                        for j in range(len(trail)-1, next_index, -1):
                            if segment(trail[cursor], trail[j]):
                                next_index = j
                                break
                        candidate.append(trail[next_index])
                        cursor = next_index
                    if candidate is not None and segment(candidate[-1], home):
                        retreat = candidate + [home]
            if retreat is not None and segment(home, goal):
                return retreat + [goal]
            score = min(constraints((1-u)*start+u*goal)[0].min() for u in np.linspace(0,1,15))
            blocked.append((float(score), goal.copy()))
        for _, goal in sorted(blocked, key=lambda row:row[0], reverse=True)[:3]:
            path = bent_path(goal)
            if path is not None:
                return path
    except _Cancelled:
        pass
    return None


class StationaryGoalRecovery:
    """Run search off the control thread, then follow a rate-limited joint path."""
    def __init__(self, qp):
        self.qp = qp
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ik-recovery')
        self.future = None
        self.cancel = Event()
        self.reset(qp.model.home)

    def reset(self, home, preserve_history=False):
        self.cancel.set()
        if self.future is not None:
            self.future.cancel()
        self.future = None
        self.home = np.asarray(home).copy()
        if not preserve_history:
            self.history = [self.home.copy()]
        self.target = None
        self.static_time = 0.
        self.attempted = False
        self.path = None
        self.segment_time = 0.
        self.state = 'tracking'

    @property
    def busy(self):
        return self.future is not None or self.path is not None

    def update(self, target, corrected, gravity):
        qp = self.qp
        if np.max(abs(qp.last-self.history[-1])) > .01:
            self.history.append(qp.last.copy())
            if len(self.history) > 4096:
                self.history = self.history[::2] + [self.history[-1]]
        # Compare against the target captured when stability counting began,
        # so many small streaming updates cannot masquerade as a static goal.
        changed = self.target is None
        if not changed:
            delta = error(self.target, target)
            changed = (np.max(np.linalg.norm(delta[:,:3],axis=1)) > 1e-4 or
                       np.max(np.linalg.norm(delta[:,3:],axis=1)) > 1e-3)
        if changed:
            home = self.home.copy()
            self.reset(home, preserve_history=True)
            self.target = target.copy()
        else:
            self.static_time += qp.dt

        if self.future is not None:
            if not self.future.done():
                return qp.last.copy()
            try:
                self.path = self.future.result()
            except Exception as exc:
                self.path = None
                self.state = 'search_error:' + type(exc).__name__
            self.future = None
            if self.path is None:
                if not self.state.startswith('search_error:'):
                    self.state = 'no_checked_path'
            else:
                self.state = 'following_path'
                self.segment_time = 0.

        if self.path is not None:
            a, b = self.path[:2]
            duration = max(qp.dt, 1.875*np.max(abs(b-a)/qp.velocity))
            self.segment_time = min(duration, self.segment_time + qp.dt)
            u = self.segment_time/duration
            blend = u**3*(10 - 15*u + 6*u*u)
            candidate = a + blend*(b-a)
            old_g = qp.model.constraints(qp.model.forward(qp.last), com_margin=qp.com_margin,
                                         gravity_b=gravity)[0]
            new_g = qp.model.constraints(qp.model.forward(candidate), com_margin=qp.com_margin,
                                         gravity_b=gravity)[0]
            # Recheck with current measured gravity before every command.
            if np.any(new_g < np.minimum(old_g, 0.) - 2e-6):
                self.path = None
                self.state = 'path_invalidated'
                return qp.last.copy()
            if np.any(abs(candidate-qp.last) > qp.velocity*qp.dt + 1e-8):
                self.path = None
                self.state = 'path_rate_rejected'
                return qp.last.copy()
            qp.last = candidate.copy()
            if self.segment_time >= duration:
                self.path.pop(0)
                self.segment_time = 0.
                if len(self.path) == 1:
                    self.path = None
                    self.state = 'complete'
            return candidate

        if not self.attempted and self.static_time >= 1.:
            e = error(qp.model.forward(qp.last)['hands'], corrected)
            if (np.max(np.linalg.norm(e[:,:3],axis=1)) > .01 or
                    np.max(np.linalg.norm(e[:,3:],axis=1)) > .035):
                self.attempted = True
                self.cancel = Event()
                self.future = self.executor.submit(find_recovery_path, qp.model,
                    qp.last.copy(), self.home.copy(), corrected.copy(), gravity.copy(), self.cancel,
                    history=[q.copy() for q in self.history])
                self.state = 'searching'
                return qp.last.copy()
        return None
