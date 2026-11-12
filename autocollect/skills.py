"""Scripted skills on top of hw.Arm: search, visual servo, grasp, place.
Fingertip pose is parameterized as (theta, r, z): base yaw (rad), horizontal reach (m), height (m), with a fixed pitch."""
import numpy as np

from . import hw, kin

PITCH = 55.0  # approach angle below horizontal, kept fixed for every move (consistent demos)
Z_HOVER = 0.15
Z_FLOOR = 0.045  # never command the fingertip below this (human grasps were at z~0.062 +- 0.012)
R_RANGE = (0.17, 0.44)
# Where the cube sits in the wrist image when it is between the open fingers (from the 45 human grasps).
GRASP_PX = np.array([435.0, 339.0, np.sqrt(21700.0)])
PX_TOL = np.array([12.0, 12.0, 8.0])


def tcyl(q):
    p, _ = kin.pose_of(q)
    return np.array([np.arctan2(p[1], p[0]), np.hypot(p[0], p[1]), p[2]])


def joints_at(c, seed):
    th, r, z = c
    q, err = kin.ik([r * np.cos(th), r * np.sin(th), z], PITCH, seed)
    if err > 0.004:
        raise ValueError(f'unreachable theta={np.degrees(th):.0f} r={r:.3f} z={z:.3f} ({err*1000:.1f} mm)')
    return q


def go(arm, c, rate=None, **kw):
    """Straight line in (theta, r, z) from the current command.
    rate: optional base-yaw speed (deg/s) for slow sweeps; default spacing is 4 mm of fingertip travel."""
    c0 = tcyl(arm.cmd)
    c = np.asarray(c, float)
    n = int(max(abs(c[0] - c0[0]) * c0[1], np.linalg.norm(c[1:] - c0[1:])) / 0.004) + 1
    if rate:
        n = max(n, int(abs(np.degrees(c[0] - c0[0])) / (rate / hw.FPS)) + 1)
    path, seed = [], arm.cmd
    for t in np.linspace(0, 1, max(n, 2))[1:]:
        seed = joints_at(c0 + (c - c0) * t, seed)
        path.append(seed)
    return arm.follow(path, **kw)


def feature(img, mask_fn=hw.red_mask, min_area=400):
    b = hw.blob(mask_fn(img), min_area, compact=mask_fn is hw.red_mask)
    return None if b is None else np.array([b[0], b[1], np.sqrt(b[2])])


# Raster search like the human demos: rows from far/high to near/low, sweeping left -> right then back.
SEARCH_ROWS = [(0.32, 0.17), (0.25, 0.15), (0.20, 0.13)]  # (reach m, height m)
SEARCH_THETA = (np.radians(40), np.radians(-45))           # left edge, right edge (base yaw); keyboard/mouse pad beyond -50
SEARCH_RATE = 12.0                                          # deg/s: slow enough to see the cube while moving


def visible(img, mask_fn=hw.red_mask, min_area=800):
    """Blob big enough and away from the image border (rejects clutter at the edges)."""
    f = feature(img, mask_fn, min_area)
    return f is not None and 40 < f[0] < 600 and f[1] > 30


THETA_MIN = np.radians(-48)  # nothing of ours lives right of this (keyboard + red/black mouse pad there)


def est_theta(obs, mask_fn=hw.red_mask):
    """Rough base yaw of the blob: camera yaw minus its pixel offset (content right of center = smaller yaw)."""
    f = feature(obs['wrist'], mask_fn)
    th = tcyl(hw.state(obs))[0]
    return None if f is None else th - np.radians((f[0] - 320) / PX_PER_DEG * SIZE_REF / max(f[2], 40.0))


def search(arm, mask_fn=hw.red_mask, log=print):
    """Serpentine sweep over SEARCH_ROWS; stops as soon as the blob is visible. Returns True if found."""
    left, right = SEARCH_THETA

    def until(obs):
        if not visible(obs['wrist'], mask_fn):
            return False
        th = est_theta(obs, mask_fn)
        return mask_fn is not hw.red_mask or (th is not None and th > THETA_MIN)
    for i, (r, z) in enumerate(SEARCH_ROWS):
        a, b = (left, right) if i % 2 == 0 else (right, left)
        go(arm, np.array([a, r, z]), timeout=6)          # move to the row start
        if until(arm.obs):
            return True
        if go(arm, np.array([b, r, z]), rate=SEARCH_RATE, timeout=6, until=until) is None:
            log(f'search: found on row {i} (r={r}, z={z}) at theta={np.degrees(tcyl(arm.cmd)[0]):.0f}')
            return True
    log('search: not found')
    return False


def servo(arm, target=GRASP_PX, tol=PX_TOL, mask_fn=hw.red_mask, z_floor=Z_FLOOR, use_z=True,
          max_iter=150, log=print):
    """Image-based visual servo on (theta, r, z) until the blob matches `target` (x, y, sqrt(area)).
    Jacobian is probed with small moves, then Broyden-updated. Returns True on convergence."""
    dims = 3 if use_z else 2
    probe = np.array([np.radians(3), 0.015, -0.015])[:dims]

    def look():
        arm.hold(0.15)
        return feature(arm.image(), mask_fn)

    c = tcyl(arm.cmd)
    f = look()
    if f is None:
        return False
    J = np.zeros((3, dims))
    for i in range(dims):
        d = np.zeros(3); d[i] = probe[i]
        go(arm, c + d, timeout=3)
        f2 = look()
        go(arm, c, timeout=3)
        if f2 is None:
            return False
        J[:, i] = (f2 - f) / probe[i]
    f = look()
    W = np.diag([1.0, 1.0, 2.0 if use_z else 0.0])  # weight: size error matters as much as position
    log(f'J=\n{J.round(1)}')
    for it in range(max_iter):
        e = f - target
        log(f'  it {it:3d} theta={np.degrees(c[0]):6.1f} r={c[1]:.3f} z={c[2]:.3f} f={f.round(0)} err={e.round(0)}')
        if np.all(np.abs(e[:dims]) < tol[:dims]) or (not use_z and np.all(np.abs(e[:2]) < tol[:2])):
            log(f'servo converged in {it} steps, err {e.round(1)}')
            return True
        A = W @ J
        du = -np.linalg.lstsq(A, W @ e, rcond=None)[0] * 0.5
        du = np.clip(du, -np.array([np.radians(4), 0.02, 0.015])[:dims], np.array([np.radians(4), 0.02, 0.015])[:dims])
        c_new = c.copy(); c_new[:dims] += du
        c_new[1] = np.clip(c_new[1], *R_RANGE)
        c_new[2] = max(c_new[2], z_floor)
        try:
            go(arm, c_new, timeout=2, tol=3.0)
        except ValueError as err:
            log('servo: ' + str(err))
            return False
        f_new = look()
        if f_new is None:
            log('servo: lost target')
            return False
        step = c_new[:dims] - c[:dims]
        if np.linalg.norm(step) > 1e-6:  # Broyden rank-1 update
            J += np.outer((f_new - f) - J @ step, step) / (step @ step)
        c, f = c_new, f_new
    log(f'servo: no convergence, err {(f - target).round(1)}')
    return False


ROLL_90 = 53.0  # wrist_roll units for ~90 deg (calibrated range ~338 deg over 200 units)


def blue_at_fingers(img, f):
    """Fraction of tray (blue) pixels where the fingers will close: bands left and right of the cube."""
    blue = hw.blue_mask(img) > 0
    cx, cy, s = f
    y0, y1 = int(max(0, cy - 0.5 * s)), int(min(480, cy + 0.5 * s))
    out = 0.0
    for xa, xb in ((cx - 1.1 * s, cx - 0.5 * s), (cx + 0.5 * s, cx + 1.1 * s)):  # where the fingers land
        xa, xb = int(max(0, xa)), int(min(640, xb))
        if xb - xa > 5 and y1 - y0 > 5:
            out = max(out, float(blue[y0:y1, xa:xb].mean()))
    return out


def clear_fingers(arm, log=print, limit=0.3):
    """Before closing: if the tray is where a finger would land, rotate the wrist 90 deg (fingers then close
    front/back instead of left/right). Returns False if neither orientation is clear."""
    f = feature(arm.image())
    if f is None:
        return False
    b = blue_at_fingers(arm.image(), f)
    if b < limit:
        return True
    log(f'tray at the fingers ({b:.2f}): rotating wrist 90 deg')
    c = tcyl(arm.cmd)
    go(arm, np.array([c[0], c[1], c[2] + 0.03]), timeout=3)
    roll = arm.cmd['wrist_roll']
    arm.move({'wrist_roll': roll + ROLL_90 if roll + ROLL_90 < 95 else roll - ROLL_90}, timeout=3)
    go(arm, c, timeout=3)
    arm.hold(0.2)
    f = feature(arm.image())
    if f is None or np.any(np.abs(f - GRASP_PX) > 2 * PX_TOL):
        log(f'cube no longer between the fingers after rotating ({None if f is None else f.round(0)}): skip')
        return False
    b = blue_at_fingers(arm.image(), f)
    if b >= limit:
        log(f'tray still at the fingers ({b:.2f}): skip this grasp')
        return False
    return True


def grasp(arm, lift=0.06, log=print, check_tray=True):
    """Close on the cube under the fingers, lift, and verify with the camera: a held cube moves with
    the camera (same pixel and size after lifting further); a cube left on the table shrinks/shifts.
    If the tray rises with the gripper too, put everything back down and report failure."""
    f0 = feature(arm.image())
    tray_near = check_tray and f0 is not None and blue_at_fingers(arm.image(), f0) > 0.05  # only then can the tray come up
    arm.move({'gripper': hw.GRIP_CLOSED}, timeout=2, settle=0.6)
    # don't keep driving the gripper far past where the (foam) cube stopped it: the gripper motor has a low
    # overload threshold and trips (stops responding until power-cycled) if it stalls at full effort
    g = hw.state(arm.obs)['gripper']
    arm.move({'gripper': max(g - 4.0, hw.GRIP_CLOSED)}, timeout=1, settle=0.1)
    c = tcyl(arm.cmd)
    go(arm, np.array([c[0], c[1], c[2] + lift]), timeout=4)
    arm.hold(0.3)
    f1, t1 = feature(arm.image()), feature(arm.image(), hw.blue_mask, 2000)
    go(arm, np.array([c[0], c[1], c[2] + lift + 0.04]), timeout=4)
    arm.hold(0.3)
    f2, t2 = feature(arm.image()), feature(arm.image(), hw.blue_mask, 2000)
    tray_up = (tray_near and t1 is not None and t2 is not None and np.linalg.norm(t2[:2] - t1[:2]) < 15
               and abs(t2[2] - t1[2]) < 0.05 * t1[2])
    if tray_up:
        log('grasp: tray came up with the gripper -> putting it back down')
        go(arm, c, timeout=4)
        arm.move({'gripper': hw.GRIP_OPEN}, timeout=2, settle=0.4)
        go(arm, np.array([c[0], c[1], c[2] + lift]), timeout=4)
        return False
    ok = (f1 is not None and f2 is not None and np.linalg.norm(f2[:2] - f1[:2]) < 25
          and abs(f2[2] - f1[2]) < 0.1 * f1[2] and f1[1] > 280)
    log(f'grasp: {"holding" if ok else "missed"} f1={None if f1 is None else f1.round(0)} f2={None if f2 is None else f2.round(0)}')
    return ok


def release(arm, lift=0.05):
    arm.move({'gripper': hw.GRIP_OPEN}, timeout=2, settle=0.4)
    c = tcyl(arm.cmd)
    go(arm, np.array([c[0], c[1], c[2] + lift]), timeout=4)


# Measured on this arm at blob size ~150 px (sqrt area): +1 deg base yaw -> +23 px in x; +1 cm reach -> +35 px in y.
PX_PER_DEG, PX_PER_M, SIZE_REF = 23.0, 3500.0, 150.0


def approach(arm, target=GRASP_PX, tol=PX_TOL, mask_fn=hw.red_mask, z_floor=Z_FLOOR, max_iter=60, log=print):
    """Staged visual servo, like the human demos: center the blob (yaw -> x, reach -> y), then advance
    along the approach axis, re-centering each step, until the blob reaches the grasp size.
    Image gains scale with blob size (closer = larger shift per move). Returns True when at grasp pose."""
    p = np.radians(PITCH)
    for it in range(max_iter):
        arm.hold(0.15)
        img = arm.image()
        b = hw.blob(mask_fn(img), 400)
        if b is None:
            log(f'approach: lost target at it {it}')
            return False
        f = np.array([b[0], b[1], np.sqrt(b[2])])
        e = f - target
        c = tcyl(arm.cmd)
        scale = max(f[2], 40.0) / SIZE_REF
        dth = -0.6 * np.radians(e[0] / (PX_PER_DEG * scale))
        dr = -0.6 * e[1] / (PX_PER_M * scale)
        near_edge = f[0] < 60 or f[0] > 580 or f[1] < 50 or f[1] > 430
        centered = abs(e[0]) < 30 and abs(e[1]) < 30
        adv = 0.0
        if centered and not near_edge and e[2] < -tol[2]:
            adv = float(np.clip(-e[2] / 400.0, 0.004, 0.02))  # bigger step when far
        log(f'  it {it:2d} th={np.degrees(c[0]):6.1f} r={c[1]:.3f} z={c[2]:.3f} f={f.round(0)} err={e.round(0)} adv={adv*1000:.0f}mm')
        if np.all(np.abs(e) < tol):
            log(f'approach: at grasp pose after {it} steps, err {e.round(1)}')
            return True
        dth = float(np.clip(dth, -np.radians(5), np.radians(5)))
        dr = float(np.clip(dr, -0.02, 0.02)) + adv * np.cos(p)
        dz = -adv * np.sin(p)
        new = np.array([c[0] + dth, np.clip(c[1] + dr, *R_RANGE), max(c[2] + dz, z_floor)])
        if new[2] == z_floor and abs(e[2]) >= tol[2] and centered and adv:
            log('approach: reached height floor before grasp size')
        try:
            go(arm, new, timeout=2, tol=4.0)
        except ValueError as err:
            log('approach: ' + str(err))
            return False
    log('approach: no convergence')
    return False


def center(arm, target_xy, mask_fn, scale=1.0, tol=20.0, max_iter=25, log=print):
    """Center a blob at target_xy using only yaw and reach (fixed gains, no probing). Height unchanged."""
    for it in range(max_iter):
        arm.hold(0.15)
        b = hw.blob(mask_fn(arm.image()), 400)
        if b is None:
            log('center: lost target')
            return False
        e = np.array(b[:2]) - target_xy
        if np.all(np.abs(e) < tol):
            log(f'center: done in {it} steps, err {e.round(0)}')
            return True
        c = tcyl(arm.cmd)
        dth = float(np.clip(-0.6 * np.radians(e[0] / (PX_PER_DEG * scale)), -np.radians(5), np.radians(5)))
        dr = float(np.clip(-0.6 * e[1] / (PX_PER_M * scale), -0.02, 0.02))
        try:
            go(arm, np.array([c[0] + dth, np.clip(c[1] + dr, *R_RANGE), c[2]]), timeout=2, tol=3.0)
        except ValueError as err:
            log('center: ' + str(err))
            return False
    log('center: no convergence')
    return False


def approach_cont(arm, target=GRASP_PX, tol=PX_TOL, mask_fn=hw.red_mask, z_floor=Z_FLOOR, timeout=25.0,
                  look_every=4, log=print):
    """Continuous version of `approach`: re-plans every `look_every` ticks (~7 Hz) while moving,
    no stop-and-settle between steps. Same strategy: center (yaw->x, reach->y), advance along the approach axis."""
    import time
    p = np.radians(PITCH)
    goal = tcyl(arm.cmd)
    seed = dict(arm.cmd)
    good, lost, t_end, it, stuck = 0, 0, time.perf_counter() + timeout, 0, 0
    while time.perf_counter() < t_end:
        it += 1
        b = hw.blob(mask_fn(arm.image()), 400, compact=mask_fn is hw.red_mask)
        if b is None:
            lost += 1
            if lost > 8:
                log('approach: lost target')
                return False
        else:
            lost = 0
            q = hw.state(arm.obs)
            lag = abs(arm.cmd['shoulder_pan'] - q['shoulder_pan'])  # lift/elbow lag anyway under gravity at long reach
            stuck = stuck + 1 if lag > 8 else 0
            if stuck >= 8:  # commanded far from actual for a while: blocked, or dragging something (finger on a tray wall)
                log(f'approach: arm blocked (lag {lag:.0f}) -> backing off')
                c = tcyl(hw.state(arm.obs))
                arm.cmd = hw.state(arm.obs)
                go(arm, np.array([c[0], c[1], c[2] + 0.05]), timeout=3)
                return False
            f = np.array([b[0], b[1], np.sqrt(b[2])])
            e = f - target
            good = good + 1 if np.all(np.abs(e) < tol) else 0
            if good >= 3:
                arm.hold(0.2)
                log(f'approach: at grasp pose after {it} looks, err {e.round(1)}')
                return True
            scale = max(f[2], 40.0) / SIZE_REF
            dth = float(np.clip(-0.35 * np.radians(e[0] / (PX_PER_DEG * scale)), -np.radians(2.5), np.radians(2.5)))
            dr = float(np.clip(-0.25 * e[1] / (PX_PER_M * scale), -0.008, 0.008))
            near_edge = f[0] < 60 or f[0] > 580 or f[1] < 50 or f[1] > 430
            adv = 0.0
            if abs(e[0]) < 20 and abs(e[1]) < 20 and not near_edge and e[2] < -tol[2]:
                adv = float(np.clip(-e[2] / 900.0, 0.002, 0.008))
            elif e[2] > tol[2]:
                adv = -0.003  # too close: back off along the axis
            if it % 10 == 0:
                log(f'  look {it}: f={f.round(0)} err={e.round(0)} goal th={np.degrees(goal[0]):.1f} r={goal[1]:.3f} z={goal[2]:.3f}')
            base = tcyl(arm.cmd)  # where the arm is actually being driven now: no wind-up while it lags
            if base[0] + dth < THETA_MIN:
                log('approach: target drifted outside the workspace (keyboard side)')
                return False
            goal = np.array([base[0] + dth, np.clip(base[1] + dr + adv * np.cos(p), *R_RANGE),
                             max(base[2] - adv * np.sin(p), z_floor)])
            try:
                seed = joints_at(goal, seed)
            except ValueError as err:
                log('approach: ' + str(err))
                return False
        for _ in range(look_every):
            arm.tick(dict(arm.cmd, **seed))
    log('approach: timeout')
    return False
