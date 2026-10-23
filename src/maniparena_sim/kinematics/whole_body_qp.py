"""CPU whole-body sequential QP, with explicit frame and joint contracts.

The optimization variable is a joint displacement (rad). Joint position and
per-control-period velocity bounds are hard constraints. COM rectangle and
sampled arm/torso clearances are linearized, then checked after line search.
The latter are geometric proxies, not a full mesh collision checker or a
dynamic balance guarantee. All transforms are in base_link, radians/metres.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation


def cross(a, b):
    # Fixed-size vectors: avoid numpy.cross's axis/broadcast machinery in the
    # many small geometric Jacobian terms (profiled bottleneck).
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])


def rotation(axis, angle):
    t = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    a, b = ((1, 2), (2, 0), (0, 1))[axis]
    t[a, a] = t[b, b] = c
    t[a, b], t[b, a] = -s, s
    return t


class Model:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text()) if not isinstance(path, dict) else path
        m = self.data
        self.names = m['joint_names']
        self.home, self.lower, self.upper = [np.array(m[k], float) for k in ('home', 'lower', 'upper')]
        self.arms = []
        for side in ('left', 'right'):
            a = m['arms'][side]
            self.arms.append(([self.names.index(n) for n in a['joint_names']], np.array(a['constants']), a['axes']))
        self.segments = [(self.names.index(s['joint']) if s['joint'] else -1,
                          s['mass'], np.array(s['com'])) for s in m['mass_segments']]
        self.mass = sum(s[1] for s in self.segments)

    def forward(self, q):
        frames, origins, axes, ancestors = {}, {}, {}, {}
        hands, jac, points, pjac = [], [], [], []
        for ids, C, ax in self.arms:
            T = C[0].copy()
            for i, (j, a) in enumerate(zip(ids, ax)):
                T = T @ rotation(a, q[j])
                frames[j], origins[j], axes[j] = T.copy(), T[:3, 3].copy(), T[:3, a].copy()
                ancestors[j] = ids[:i+1]
                T = T @ C[i+1]
            p = T[:3, 3]
            J = np.zeros((6, 18))
            for j in ids:
                J[:3, j] = cross(axes[j], p - origins[j])
                J[3:, j] = axes[j]
            hands.append(T); jac.append(J)
            ps, js = [], []
            for idx, extra in ((7, None), (9, None), (10, None), (10, np.array([.17, 0, 0]))):
                point = frames[ids[idx]][:3, 3] if idx < 10 else p.copy()
                if extra is not None:
                    point = point + T[:3, :3] @ extra
                jp = np.zeros((3, 18))
                for j in ids[:idx+1]:
                    jp[:, j] = cross(axes[j], point - origins[j])
                ps.append(point); js.append(jp)
            # elbow, forearm midpoint, wrist, flange, fingertip
            ps.insert(1, .5*(ps[0]+ps[1])); js.insert(1, .5*(js[0]+js[1]))
            points.append(ps); pjac.append(js)
        com, cj = np.zeros(3), np.zeros((3, 18))
        for j, mass, c in self.segments:
            if j < 0:
                p = c
            else:
                p = frames[j][:3, :3] @ c + frames[j][:3, 3]
                for k in ancestors[j]:
                    cj[:, k] += mass*cross(axes[k], p-origins[k])
            com += mass*p
        return dict(hands=np.array(hands), jac=np.array(jac), points=np.array(points),
                    pjac=np.array(pjac), com=com/self.mass, com_jac=cj/self.mass,
                    frames=frames, origins=origins, axes=axes)

    def constraints(self, fk, torso_margin=.005, arm_gap=.08, com_margin=.025, gravity_b=None):
        """Return signed clearance g and dg/dq; valid configurations have g >= 0."""
        g, J = [], []
        sp = self.data['support_polygon']
        com = fk['com'].copy(); cj = fk['com_jac'].copy()
        if gravity_b is not None:
            gravity_b = np.asarray(gravity_b)
            slope = gravity_b[:2]/min(float(gravity_b[2]),-1e-3)
            com[:2] -= com[2]*slope
            cj[:2] -= slope[:,None]*cj[2]
        for a, key in enumerate(('x','y')):
            g.extend((com[a]-sp[key][0]-com_margin, sp[key][1]-com[a]-com_margin))
            J.extend((cj[a], -cj[a]))
        for b in self.data['keepout_boxes']:
            frame = b['frame']; lo = np.array(b['min'])-torso_margin; hi = np.array(b['max'])+torso_margin
            T = np.eye(4) if frame < 0 else fk['frames'][frame]
            R, p = T[:3,:3], T[:3,3]
            for point, jp in zip(fk['points'].reshape(-1,3), fk['pjac'].reshape(-1,3,18)):
                local = R.T @ (point-p)
                jl = R.T @ jp
                if frame >= 0:
                    for k in range(frame+1):
                        jl[:,k] -= R.T @ cross(fk['axes'][k], point-fk['origins'][k])
                faces = np.r_[local-hi, lo-local]
                f = int(np.argmax(faces))
                g.append(faces[f]); J.append(jl[f] if f < 3 else -jl[f-3])
        for i in range(1,5):
            for j in range(1,5):
                d = fk['points'][0,i]-fk['points'][1,j]
                r = np.linalg.norm(d)
                g.append(r-arm_gap)
                J.append((d/max(r,1e-9)) @ (fk['pjac'][0,i]-fk['pjac'][1,j]))
        return np.array(g), np.array(J)


def error(hands, target):
    return np.concatenate((target[:,:3,3]-hands[:,:3,3],
                           Rotation.from_matrix(target[:,:3,:3] @ hands[:,:3,:3].transpose(0,2,1)).as_rotvec()), axis=1)


class WholeBodyQP:
    def __init__(self, model, dt=.05, waist_cost=25., constraints=True):
        self.model = model
        self.dt = dt
        self.weights = np.r_[np.full(4,waist_cost),np.ones(14)]
        self.velocity = np.r_[np.full(4,.20),np.full(14,.8)]
        self.gravity_b = None
        self.com_margin = .065
        self.constrained = constraints
        self.last = None
        self.info = {}

    def reset(self, q):
        self.last = np.asarray(q, float).copy()
        self.info = {}

    def solve(self, target, measured_q, iterations=4, rate_limit=True):
        t0 = time.perf_counter()
        m = self.model
        q = np.asarray(measured_q, float).copy()
        if target.shape != (2,4,4) or not np.isfinite(target).all() or q.shape != (18,) or not np.isfinite(q).all():
            raise ValueError('finite dual-hand 4x4 target and 18-joint state required')
        anchor = q.copy() if self.last is None else self.last.copy()
        lo, hi = m.lower.copy(), m.upper.copy()
        if rate_limit:
            lo = np.maximum(lo,anchor-self.velocity*self.dt)
            hi = np.minimum(hi,anchor+self.velocity*self.dt)
        # Warm-start commands so measured gravity sag cannot ratchet redundant joints.
        q = np.clip(anchor,lo,hi)
        task_w = np.tile([1,1,1,.3,.3,.3],2)
        statuses = []
        for _ in range(iterations):
            fk = m.forward(q)
            pose_err = error(fk['hands'], target)
            if np.max(np.linalg.norm(pose_err[:,:3],axis=1)) < 1e-6 and np.max(np.linalg.norm(pose_err[:,3:],axis=1)) < 5e-6:
                break
            e = pose_err.ravel()*task_w
            J = fk['jac'].reshape(12,18)*task_w[:,None]
            # Tiny posture objective preserves precision; waist cost mainly affects damping.
            damp = 2e-6*self.weights
            H = J.T@J + np.diag(damp)
            b = -J.T@e + 1e-8*self.weights*(q-m.home)
            gs, Gj = m.constraints(fk,com_margin=self.com_margin,gravity_b=self.gravity_b)
            if self.constrained:
                # If measured state starts infeasible, never demand an impossible instant recovery.
                A = np.vstack((np.eye(18),Gj))
                low = np.r_[lo-q, -gs + np.minimum(gs,0)*.8]
                high = np.r_[hi-q, np.full(len(gs),np.inf)]
            else:
                A,low,high = np.eye(18),lo-q,hi-q
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(np.triu(H)),q=b,A=sparse.csc_matrix(A),l=low,u=high,
                         verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=1500,polishing=False)
            result = solver.solve(raise_error=False)
            statuses.append(result.info.status)
            if result.x is None or result.info.status_val not in (1,2):
                break
            dq = result.x
            cost = e@e
            accepted = False
            for alpha in (1.,.5,.25,.125,.0625):
                candidate = np.clip(q+alpha*dq,lo,hi)
                next_fk = m.forward(candidate)
                next_e = error(next_fk['hands'],target).ravel()*task_w
                next_g,_ = m.constraints(next_fk,com_margin=self.com_margin,gravity_b=self.gravity_b)
                feasible = (not self.constrained) or np.all(next_g >= np.minimum(gs,0)-1e-6)
                if feasible and next_e@next_e <= cost+1e-12:
                    q = candidate; accepted = True; break
            if not accepted or np.linalg.norm(dq)<1e-7:
                break
        self.last = q.copy()
        fk = m.forward(q); e = error(fk['hands'],target)
        g,_ = m.constraints(fk,com_margin=self.com_margin,gravity_b=self.gravity_b)
        self.info = dict(pos_mm=(1000*np.linalg.norm(e[:,:3],axis=1)).tolist(),
                         rot_deg=np.rad2deg(np.linalg.norm(e[:,3:],axis=1)).tolist(),
                         min_clearance=float(g.min()),status=statuses,
                         elapsed_ms=1000*(time.perf_counter()-t0))
        return q
