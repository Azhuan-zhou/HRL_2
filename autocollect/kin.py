"""SO-101 forward kinematics from the official URDF + this arm's LeRobot calibration.
Joint values are LeRobot's normalized units (RANGE_M100_100; gripper 0..100)."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path.home() / 'SO-ARM100/Simulation/SO101/so101_new_calib.urdf'
CALIB = Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json'
ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
CHAIN = ARM + ['gripper_frame_joint']


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _load():
    joints = {j.get('name'): j for j in ET.parse(URDF).getroot().findall('joint')}
    out = []
    for name in CHAIN:
        o = joints[name].find('origin')
        T = np.eye(4)
        T[:3, :3] = _rpy(*map(float, o.get('rpy').split()))
        T[:3, 3] = list(map(float, o.get('xyz').split()))
        out.append(T)
    return out


ORIGINS = _load()
CAL = json.loads(CALIB.read_text())
SIGN = {m: 1.0 for m in ARM}  # flipped per joint if the URDF axis disagrees with the motor


def to_rad(m, v):
    c = CAL[m]
    ticks = c['range_min'] + (v + 100) / 200 * (c['range_max'] - c['range_min'])
    return SIGN[m] * (ticks - 2048) * 2 * np.pi / 4096


def to_norm(m, rad):
    c = CAL[m]
    ticks = SIGN[m] * rad * 4096 / (2 * np.pi) + 2048
    return (ticks - c['range_min']) / (c['range_max'] - c['range_min']) * 200 - 100


def fk(q):
    """q: dict of normalized joint values -> 4x4 pose of gripper_frame_link (fingertip frame) in base_link."""
    T = np.eye(4)
    for name, O in zip(CHAIN, ORIGINS):
        T = T @ O
        if name in ARM:
            a = to_rad(name, q[name])
            R = np.eye(4); R[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            T = T @ R
    return T


IK_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex']
# Normalized joint limits used by IK. wrist_flex physically stalls near 90 on this arm.
LIMITS = {'shoulder_pan': (-90, 90), 'shoulder_lift': (-100, 95), 'elbow_flex': (-95, 97), 'wrist_flex': (-95, 86)}
LO = np.array([LIMITS[m][0] for m in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex']], float)
HI = np.array([LIMITS[m][1] for m in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex']], float)


def approach_dir(T):
    return T[:3, 2]  # gripper_frame z axis points out of the fingertips


def ik(xyz, pitch_deg, q0, iters=200, tol=1e-4):
    """Joint values putting the fingertip at xyz (m, base frame) with the approach axis pitched
    pitch_deg below horizontal (90 = straight down), in the vertical plane through the base.
    Returns (q, pos_err_m). wrist_roll/gripper are copied from q0."""
    xyz = np.asarray(xyz, float)
    u = xyz[:2] / (np.linalg.norm(xyz[:2]) + 1e-9)
    p = np.radians(pitch_deg)
    a_goal = np.array([np.cos(p) * u[0], np.cos(p) * u[1], -np.sin(p)])
    q = dict(q0)
    x = np.array([q[m] for m in IK_JOINTS], float)

    def resid(x):
        qq = dict(q, **dict(zip(IK_JOINTS, x)))
        T = fk(qq)
        return np.concatenate([(T[:3, 3] - xyz) * 10.0, (approach_dir(T) - a_goal) * 0.3])

    for _ in range(iters):
        r = resid(x)
        if np.linalg.norm(r[:3]) / 10 < tol and np.linalg.norm(r[3:]) < 0.01:
            break
        J = np.zeros((6, 4))
        for i in range(4):
            dx = np.zeros(4); dx[i] = 0.05
            J[:, i] = (resid(x + dx) - r) / 0.05
        step = np.linalg.solve(J.T @ J + 1e-3 * np.eye(4), -J.T @ r)
        x = np.clip(x + np.clip(step, -10, 10), LO, HI)
    q.update(zip(IK_JOINTS, x))
    return q, float(np.linalg.norm(fk(q)[:3, 3] - xyz))


def pose_of(q):
    """(xyz, pitch_deg) of the fingertip for joint values q."""
    T = fk(q)
    return T[:3, 3].copy(), float(np.degrees(np.arcsin(-approach_dir(T)[2])))


def line(q0, xyz, pitch_deg, step_m=0.004):
    """Joint waypoints moving the fingertip on a straight line (pitch interpolated) from q0 to xyz.
    Raises if any waypoint is unreachable (> 4 mm IK error)."""
    p0, a0 = pose_of(q0)
    xyz = np.asarray(xyz, float)
    n = max(2, int(np.linalg.norm(xyz - p0) / step_m) + 1, int(abs(pitch_deg - a0) / 2) + 1)
    path, q = [], dict(q0)
    for i in range(1, n + 1):
        t = i / n
        q, err = ik(p0 + (xyz - p0) * t, a0 + (pitch_deg - a0) * t, q)
        if err > 0.004:
            raise ValueError(f'unreachable at t={t:.2f}: {p0 + (xyz - p0) * t}, pitch {a0 + (pitch_deg - a0) * t:.0f}, err {err*1000:.1f}mm')
        path.append(dict(q))
    return path
