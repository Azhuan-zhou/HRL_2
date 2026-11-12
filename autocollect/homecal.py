"""Pixel <-> table mapping for the fixed Home view (a homography), calibrated by the robot itself.

python -m autocollect.homecal        # drops the cube at known spots, records its pixel at Home, fits + saves

Then: home_xy(px) -> table (x, y) in the arm's base frame; tray_from_home(img) -> (theta, r) of the tray center.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from . import hw

BASE = Path(__file__).resolve().parent.parent
CAL = BASE / 'configs' / 'home_homography.json'


def load():
    return np.array(json.loads(CAL.read_text())['H']) if CAL.exists() else None


def home_xy(H, px):
    p = cv2.perspectiveTransform(np.array([[px]], np.float32), H)[0, 0]
    return np.array(p, float)


def cube_from_home(H, img):
    b = hw.blob(hw.red_mask(img), 400, compact=True)
    if b is None:
        return None
    x, y = home_xy(H, b[:2])
    return np.array([np.arctan2(y, x), np.hypot(x, y)])


def tray_from_home(H, img):
    """Tray center: mean table position of its blue pixels (largest blue blob)."""
    m = hw.blue_mask(img)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    if st[i, 4] < 3000:
        return None
    ys, xs = np.nonzero(lab == i)
    pts = cv2.perspectiveTransform(np.stack([xs, ys], 1).astype(np.float32)[None], H)[0]
    x, y = pts.mean(0)
    return np.array([np.arctan2(y, x), np.hypot(x, y)])


def cube_in_tray_px(img):
    """Cube centroid inside the tray's blue outline (convex hull) in the Home image."""
    b = hw.blob(hw.red_mask(img), 300)
    m = hw.blue_mask(img)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if b is None or n < 2:
        return False
    i = 1 + int(np.argmax(st[1:, 4]))
    ys, xs = np.nonzero(lab == i)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    return cv2.pointPolygonTest(hull, (float(b[0]), float(b[1])), False) >= 0


def calibrate():
    import time
    from . import run, skills, task1
    robot = hw.connect()
    arm = hw.Arm(robot)
    arm.max_step = hw.MAX_STEP * 2
    tray = run.locate_tray(arm, print)
    spots = [(th, r) for th in (-20, 0, 20, 33) for r in (0.25, 0.31, 0.36)]
    px, xy = [], []
    try:
        for th, r in spots:
            if not task1.find_and_hold_cube(arm, print):
                print('could not pick the cube; skipping spot')
                continue
            t = np.radians(th)
            if np.linalg.norm(run.xy([t, r]) - run.xy(tray)) < run.TRAY_HALF + 0.05 or not run.reachable(t, r, run.Z_DROP):
                arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
                continue
            run.to_hover(arm, t, r, run.Z_DROP)
            skills.release(arm, lift=0.06)
            task1.at_home(arm)
            b = hw.blob(hw.red_mask(arm.image()), 400, compact=True)
            # where the cube actually is: re-measure with the arm (approach until it is between the fingers)
            if b is None:
                print(f'spot {th},{r}: not visible from Home')
                continue
            if not skills.approach_cont(arm, log=lambda m: None):
                continue
            p = skills.kin.fk(arm.cmd)[:3, 3] if hasattr(skills, 'kin') else None
            from . import kin
            p = kin.fk(arm.cmd)[:3, 3]
            px.append(b[:2]); xy.append(p[:2])
            print(f'spot {th:>4},{r:.2f}: pixel {np.round(b[:2])} -> table {np.round(p[:2]*1000)} mm  ({len(px)} points)')
            task1.at_home(arm)
        H, inl = cv2.findHomography(np.array(px, np.float32), np.array(xy, np.float32), cv2.RANSAC, 0.015)
        err = [np.linalg.norm(home_xy(H, a) - b) for a, b in zip(px, xy)]
        print(f'homography from {len(px)} points, inliers {int(inl.sum())}, error mm: mean {np.mean(err)*1000:.1f} max {np.max(err)*1000:.1f}')
        CAL.write_text(json.dumps({'H': H.tolist(), 'points': len(px), 'mean_err_mm': float(np.mean(err) * 1000),
                                   'time': time.strftime('%F %T')}, indent=2))
        print('saved', CAL)
    finally:
        task1.at_home(arm)
        robot.disconnect()


if __name__ == '__main__':
    calibrate()
