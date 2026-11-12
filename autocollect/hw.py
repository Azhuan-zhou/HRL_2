"""Robot + camera I/O for scripted collection: connect, smooth moves, color detection."""
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from . import kin

PORT = '/dev/tty.usbmodem5B421342761'
CALIB_DIR = Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower'
FPS = 30
MAX_STEP = 0.8  # normalized units per tick (~24/s): gentle, slower than human demos
JOINTS = kin.ARM + ['gripper']
GRIP_OPEN, GRIP_CLOSED = 66.0, 5.0


def connect(camera=True):
    cams = {'wrist': OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)} if camera else {}
    cfg = SOFollowerRobotConfig(port=PORT, id='my_awesome_follower_arm', calibration_dir=CALIB_DIR,
                                max_relative_target=5.0, cameras=cams,
                                disable_torque_on_disconnect=False)  # stay raised after the script ends
    for attempt in range(3):
        robot = SOFollower(cfg)
        try:
            robot.connect(calibrate=False)
            return robot
        except (ConnectionError, RuntimeError) as e:  # stray "Incorrect status packet" at connect
            if attempt == 2:
                raise
            print('connect retry:', e, flush=True)
            try:
                robot.disconnect()
            except Exception:
                pass
            time.sleep(1)


def state(obs):
    return {m: float(obs[m + '.pos']) for m in JOINTS}


def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | cv2.inRange(hsv, (165, 90, 60), (180, 255, 255))


def blue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, (100, 120, 40), (130, 255, 255))


def blob(mask, min_area=300, compact=False):
    """Largest connected blob -> (cx, cy, area) in pixels, or None.
    compact: only blobs that fill their box and aren't elongated (the cube; rejects e.g. striped mouse pads)."""
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < min_area:
            continue
        if compact and (area / (w * h) < 0.5 or max(w, h) / max(1, min(w, h)) > 2.5):
            continue
        if best is None or area > best[2]:
            best = (float(cen[i][0]), float(cen[i][1]), int(area))
    return best


class Arm:
    """Ticks at FPS. Every tick: observe, send one action, hand (obs, action) to `on_tick` (the recorder)."""

    def __init__(self, robot, on_tick=None):
        self.robot, self.on_tick = robot, on_tick
        self.obs = robot.get_observation()
        self.cmd = state(self.obs)
        self.t = time.perf_counter()
        self.max_step = MAX_STEP  # raised during (unrecorded) resets

    def tick(self, target):
        q = state(self.obs)
        step = {m: float(np.clip(target[m] - self.cmd[m], -self.max_step, self.max_step)) for m in JOINTS}
        self.cmd = {m: self.cmd[m] + step[m] for m in JOINTS}
        action = {m + '.pos': self.cmd[m] for m in JOINTS}
        self.robot.send_action(action)
        if self.on_tick:
            self.on_tick(self.obs, action)
        self.t += 1 / FPS
        time.sleep(max(0.0, self.t - time.perf_counter()))
        if time.perf_counter() - self.t > 0.5:  # fell behind (e.g. slow camera): resync, don't burst
            self.t = time.perf_counter()
        self.obs = self.robot.get_observation()
        return q

    def move(self, target, tol=4.0, timeout=8.0, settle=0.15):  # tol: lift sags 3-4 units under gravity
        """Go to a joint target (dict, may be partial) at bounded speed; returns when reached."""
        target = dict(self.cmd, **target)
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            q = self.tick(target)
            done = all(abs(self.cmd[m] - target[m]) < 1e-6 for m in JOINTS)
            if done and all(abs(q[m] - target[m]) < tol for m in kin.ARM):
                break
        for _ in range(int(settle * FPS)):
            self.tick(target)
        return state(self.obs)

    def follow(self, path, tol=4.0, timeout=10.0, until=None):
        """Stream joint waypoints (e.g. kin.line) one per tick, then converge on the last.
        until(obs) -> True stops early (holding the current command); returns None in that case."""
        for wp in path:
            self.tick(dict(self.cmd, **wp))
            if until and until(self.obs):
                return None
        return self.move(path[-1], tol=tol, timeout=timeout)

    def to(self, xyz, pitch_deg, **kw):
        """Straight-line fingertip move from the current command."""
        return self.follow(kin.line(self.cmd, xyz, pitch_deg), **kw)

    def hold(self, seconds):
        for _ in range(int(seconds * FPS)):
            self.tick(self.cmd)

    def image(self):
        return self.obs['wrist']

    def tip(self):
        return kin.fk(state(self.obs))
