"""Fully automatic cube -> tray data collection. No human in the loop.

python -m autocollect.run --name auto_cube --episodes 200 [--lighting night_lamp] [--no-record]

Each recorded episode: start pose -> raster search -> visual servo -> grasp -> tray -> release.
Only episodes verified by the camera (held after grasp, cube in tray after release) are saved.
Between episodes (not recorded) the robot takes the cube out of the tray and drops it somewhere random.
"""
import argparse
import json
import random
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from . import hw, skills

TASK = 'Pick up the cube and place it into the tray.'
BASE = Path(__file__).resolve().parent.parent
READY = np.array([skills.SEARCH_THETA[0], *skills.SEARCH_ROWS[0]])  # row-0 start: every episode begins here
Z_PLACE = 0.13      # release height over the tray
Z_DROP = 0.085      # release height when scattering the cube on the table
TRAY_PX = np.array([435.0, 300.0, 0.0])  # tray centroid under the fingers (x, y); size unused
Z_TABLE = 0.06      # table-level height in the FK frame (cube center at grasp ~0.06-0.08)
Z_LOOK = 0.20       # height for looking at the whole tray
Z_PUSH = 0.07       # fingertip height when pushing the tray wall
TRAY_HALF = 0.09    # ~half tray width (m); pushes start outside this
# Keep the tray where every place/scatter stays reachable.
TRAY_THETA, TRAY_R = (np.radians(-30), np.radians(25)), (0.33, 0.40)  # estimated center; keep clear of the base
SCATTER_D = (0.16, 0.22)  # cube drop distance from the tray center
SEARCH_THETA_DEG = (-35, 35)  # cube drop range (keyboard/mouse pad on the right past -50 deg)
ROLL0 = -100.0      # default wrist roll


def ahead(z):
    """Horizontal distance from the fingertip to where the approach axis (~camera center) meets the table."""
    return (z - Z_TABLE) / np.tan(np.radians(skills.PITCH))


class Recorder:
    def __init__(self, ds):
        self.ds, self.on, self.frames = ds, False, 0

    def __call__(self, obs, action):
        if not self.on:
            return
        self.frames += 1
        if self.ds is None:
            return
        from lerobot.utils.feature_utils import build_dataset_frame
        frame = {**build_dataset_frame(self.ds.features, obs, 'observation'),
                 **build_dataset_frame(self.ds.features, action, 'action'), 'task': TASK}
        self.ds.add_frame(frame)


def xy(c):
    return np.array([c[1] * np.cos(c[0]), c[1] * np.sin(c[0])])


def cyl(p):
    return np.array([np.arctan2(p[1], p[0]), np.hypot(p[0], p[1])])


def reachable(th, r, z):
    try:
        skills.joints_at([th, r, z], skills.joints_at([th, 0.3, 0.15], hw_seed()))
        return True
    except ValueError:
        return False


def hw_seed():
    return {'shoulder_pan': 0, 'shoulder_lift': -40, 'elbow_flex': 40, 'wrist_flex': 75, 'wrist_roll': -100, 'gripper': 66}


def in_tray(img):
    """Red cube surrounded by blue tray pixels."""
    red = hw.red_mask(img)
    b = hw.blob(red, 400)
    if b is None:
        return False
    x, y = int(b[0]), int(b[1])
    half = int(np.sqrt(b[2]) * 0.9)
    y0, y1, x0, x1 = max(0, y - half), min(480, y + half), max(0, x - half), min(640, x + half)
    ring = np.ones((y1 - y0, x1 - x0), bool)
    ring &= red[y0:y1, x0:x1] == 0
    blue = hw.blue_mask(img)[y0:y1, x0:x1] > 0
    return bool(ring.sum() > 0 and blue[ring].mean() > 0.25)  # np.bool_ breaks json.dumps(entry)


def locate_tray(arm, log, guess=None):
    """Center the tray in view from Z_LOOK and return its center (theta, r, z_table).
    guess: expected center -> look there first; otherwise raster-search for it."""
    if guess is not None:
        to_hover(arm, guess[0], max(guess[1] - ahead(Z_LOOK), skills.R_RANGE[0]), Z_LOOK)
    if guess is None or not skills.visible(arm.image(), hw.blue_mask):
        skills.go(arm, READY, timeout=8)
        if not skills.search(arm, mask_fn=hw.blue_mask, log=log):
            raise RuntimeError('tray not found')
        c = skills.tcyl(arm.cmd)
        to_hover(arm, c[0], c[1], Z_LOOK)
    if not skills.center(arm, TRAY_PX[:2], hw.blue_mask, log=log):
        raise RuntimeError('could not center over tray')
    c = skills.tcyl(arm.cmd)
    center = np.array([c[0], c[1] + 0.03, Z_TABLE])  # rough; place_in_tray re-centers on the tray visually
    log(f'tray center at theta={np.degrees(center[0]):.1f} r={center[1]:.3f} (fingertip was at r={c[1]:.3f} z={c[2]:.3f})')
    return center


def to_hover(arm, theta, r, z=skills.Z_HOVER):
    c = skills.tcyl(arm.cmd)
    zt = max(z, skills.Z_HOVER)  # travel height: high enough to clear things, low enough to reach far
    skills.go(arm, np.array([c[0], c[1], zt]), timeout=5)    # up/down to travel height
    skills.go(arm, np.array([theta, r, zt]), timeout=8)      # across
    skills.go(arm, np.array([theta, r, z]), timeout=5)       # to the requested height


SNAPS = BASE / 'logs' / 'autocollect' / 'snaps'


def snap(arm, tag):
    """Save the current wrist image for later diagnosis."""
    SNAPS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SNAPS / f'{time.strftime("%H%M%S")}_{tag}.jpg'), cv2.cvtColor(arm.image(), cv2.COLOR_RGB2BGR))


def pick(arm, log, retries=2, check_tray=True):
    """Search, servo, grasp; on a miss, re-approach from where it is (no new search) up to `retries` times.
    Returns True if holding, False if all tries failed, None if the cube is nowhere in the search area."""
    if not skills.search(arm, log=log):
        return None
    for k in range(retries + 1):
        if k:
            log(f'retry grasp {k}/{retries}')
            arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
            c = skills.tcyl(arm.cmd)
            skills.go(arm, np.array([c[0], c[1], max(c[2], 0.12)]), timeout=4)  # back up so the cube is in view
            if not skills.visible(arm.image()) and not skills.search(arm, log=log):
                return None
        if not skills.approach_cont(arm, log=log):
            snap(arm, 'approach_fail')
            continue
        c = skills.tcyl(arm.cmd)
        log(f'grasping at theta={np.degrees(c[0]):.1f} r={c[1]:.3f} z={c[2]:.3f}')
        if check_tray and in_tray(arm.image()):
            log('cube is inside the tray: taking it out (not an episode)')
            return 'from_tray' if skills.grasp(arm, log=log, check_tray=False) else False
        if check_tray and not skills.clear_fingers(arm, log=log):  # skipped when picking out of the tray
            snap(arm, 'tray_at_fingers')
            arm.move({'wrist_roll': ROLL0}, timeout=3)
            continue
        if skills.grasp(arm, log=log, check_tray=check_tray):
            return True
        snap(arm, 'missed')
    return False


def place_in_tray(arm, tray, log):
    """Drop over the located tray center, then look at the tray from above to verify."""
    to_hover(arm, tray[0], tray[1], Z_PLACE)
    skills.release(arm)
    to_hover(arm, tray[0], max(tray[1] - 0.03, skills.R_RANGE[0]), Z_LOOK)  # the tray-locating view
    arm.hold(0.3)
    ok = in_tray(arm.image())
    log(f'place: cube {"in" if ok else "NOT in"} tray')
    if not ok:
        snap(arm, 'place_fail')
    return ok


def move_line(arm, p_end, z, step=0.01):
    """Fingertip along a straight line in the table plane (xy) at height z."""
    p0 = xy(skills.tcyl(arm.cmd))
    n = max(1, int(np.linalg.norm(p_end - p0) / step))
    for t in np.linspace(0, 1, n + 1)[1:]:
        skills.go(arm, np.array([*cyl(p0 + (p_end - p0) * t), z]), timeout=3, tol=4.0)


def push_tray(arm, tray, log, away_from=None):
    """Nudge the (empty) tray 3-6 cm, roughly away from the cube if given, then re-locate it."""
    cur = xy(tray)
    a0 = None if away_from is None else np.arctan2(*(cur - away_from)[::-1])
    for _ in range(100):
        a = random.uniform(0, 2 * np.pi) if a0 is None else a0 + np.radians(random.uniform(-60, 60))
        d = random.uniform(0.03, 0.06)
        v = np.array([np.cos(a), np.sin(a)])
        tgt = cyl(cur + d * v)
        start, end = cur - v * (TRAY_HALF + 0.03), cur - v * TRAY_HALF + v * d
        if (TRAY_THETA[0] <= tgt[0] <= TRAY_THETA[1] and TRAY_R[0] <= tgt[1] <= TRAY_R[1]
                and all(reachable(*cyl(p), Z_PUSH) for p in (start, end))):
            break
    else:
        log('push: no valid direction, tray left as is')
        return tray
    log(f'push tray {d*100:.0f} cm toward {np.degrees(a):.0f} deg')
    arm.move({'gripper': hw.GRIP_CLOSED}, timeout=2)
    to_hover(arm, *cyl(start), skills.Z_HOVER)
    skills.go(arm, np.array([*cyl(start), Z_PUSH]), timeout=4)
    move_line(arm, end, Z_PUSH)
    c = skills.tcyl(arm.cmd)
    skills.go(arm, np.array([c[0], c[1], skills.Z_HOVER]), timeout=4)
    arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
    return locate_tray(arm, log, guess=np.array([*tgt, Z_TABLE]))


def scatter(arm, tray, log):
    """Take the cube out of the tray and drop it beside the tray: in front / behind / left / right of it
    (with some jitter), 13-20 cm from the tray center."""
    for attempt in range(3):
        to_hover(arm, tray[0], max(tray[1] - ahead(skills.Z_HOVER), skills.R_RANGE[0]), skills.Z_HOVER)
        if not skills.visible(arm.image()):
            skills.go(arm, READY, timeout=8)
        if pick(arm, log, check_tray=False):
            break
        arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
    else:
        raise RuntimeError('could not take the cube out of the tray')
    return drop_beside(arm, tray, log)


def drop_beside(arm, tray, log):
    """Holding the cube: drop it in front of / behind / left / right of the tray (with jitter)."""
    u = xy([tray[0], 1.0])            # radial direction (away from the robot)
    n = np.array([-u[1], u[0]])       # robot's left
    names = {0: 'behind', 90: 'left of', 180: 'in front of', 270: 'right of'}
    for _ in range(200):
        side = random.choice(list(names))
        a = np.radians(side + random.uniform(-20, 20))
        th, r = cyl(xy(tray) + random.uniform(*SCATTER_D) * (np.cos(a) * u + np.sin(a) * n))
        if (SEARCH_THETA_DEG[0] <= np.degrees(th) <= SEARCH_THETA_DEG[1] and 0.20 <= r <= 0.40
                and reachable(th, r, Z_DROP)):
            break
    else:
        raise RuntimeError('no reachable spot around the tray')
    log(f'scatter: cube {names[side]} the tray -> theta={np.degrees(th):.0f} r={r:.2f}')
    to_hover(arm, th, r, Z_DROP)
    skills.release(arm, lift=0.06)
    return th, r


def reset(arm, tray, log, entry):
    """Unrecorded: cube out of the tray -> beside it, then nudge the empty tray away from the cube, re-locate.
    Failures here are logged and skipped; the next episode's search copes with wherever things ended up."""
    cube = None
    try:
        th, r = scatter(arm, tray, log)
        cube = xy([th, r])
        entry['scatter'] = [round(float(np.degrees(th)), 1), round(float(r), 3)]
    except (ValueError, RuntimeError) as e:
        log(f'reset: scatter failed ({e})')
        arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
    try:
        tray = push_tray(arm, tray, log, away_from=cube)
    except (ValueError, RuntimeError) as e:
        log(f'reset: push skipped ({e})')
        try:
            tray = locate_tray(arm, log, guess=tray)
        except (ValueError, RuntimeError) as e2:
            log(f'reset: tray not re-located ({e2})')
    entry['tray'] = [round(float(np.degrees(tray[0])), 1), round(float(tray[1]), 3)]
    return tray


def make_dataset(name, robot):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features
    root = BASE / 'data' / name
    features = combine_feature_dicts(hw_to_dataset_features(robot.action_features, 'action'),
                                     hw_to_dataset_features(robot.observation_features, 'observation'))
    if root.exists():
        return LeRobotDataset.resume(f'local/{name}', root=root, image_writer_threads=4)
    return LeRobotDataset.create(f'local/{name}', hw.FPS, features=features, root=root, robot_type='so101_follower',
                                 use_videos=True, image_writer_threads=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='auto_cube')
    ap.add_argument('--episodes', type=int, default=50, help='successful episodes to collect')
    ap.add_argument('--lighting', default='unspecified', help='free-form tag stored per episode')
    ap.add_argument('--no-record', action='store_true')
    ap.add_argument('--max-failures', type=int, default=10, help='consecutive failed attempts before stopping')
    a = ap.parse_args()

    logdir = BASE / 'logs' / 'autocollect'
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    textlog = (logdir / f'{a.name}_{stamp}.log').open('a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        textlog.write(line + '\n'); textlog.flush()

    import fcntl
    lock = open('/tmp/autocollect.lock', 'w')  # one process on the arm/camera at a time
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another autocollect run is using the arm')
    robot = hw.connect()
    ds = None if a.no_record else make_dataset(a.name, robot)
    rec = Recorder(ds)
    arm = hw.Arm(robot, on_tick=rec)
    journal = None if ds is None else (Path(ds.root) / 'meta' / 'auto_log.jsonl')
    saved, fails, attempt = 0, 0, 0
    try:
        tray = locate_tray(arm, log)
        arm.hold(0.3)
        if in_tray(arm.image()):  # leftover from a previous run: take it out before recording
            log('cube already in the tray: resetting first')
            arm.max_step = hw.MAX_STEP * 2
            tray = reset(arm, tray, log, {})
        while saved < a.episodes and fails < a.max_failures:
            attempt += 1
            arm.max_step = hw.MAX_STEP * 2  # still unrecorded
            skills.go(arm, READY, timeout=8)
            arm.move({'wrist_roll': ROLL0}, timeout=3)  # undo any 90-deg grasp rotation
            arm.max_step = hw.MAX_STEP
            arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
            arm.hold(0.3)
            rec.on, rec.frames, t0 = True, 0, time.time()
            held = pick(arm, log)
            placed = held is True and place_in_tray(arm, tray, log)
            if held == 'from_tray':  # leftover in the tray: not a demo; drop it beside the tray and start over
                rec.on = False
                if ds is not None:
                    ds.clear_episode_buffer()
                arm.max_step = hw.MAX_STEP * 2
                try:
                    drop_beside(arm, tray, log)
                except (ValueError, RuntimeError) as e:
                    log(f'drop failed ({e})')
                    arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
                continue
            if placed:
                skills.go(arm, READY, timeout=8)  # back to the start pose: part of the recorded episode
                arm.hold(1.5)
                log('back at start pose (episode end)')
            rec.on = False
            arm.max_step = hw.MAX_STEP * 2  # reset moves are not recorded: go faster
            entry = {'attempt': attempt, 'held': bool(held), 'placed': placed, 'frames': rec.frames,
                     'seconds': round(time.time() - t0, 1), 'lighting': a.lighting, 'time': time.strftime('%F %T')}
            if held is None:
                if ds is not None:
                    ds.clear_episode_buffer()
                raise RuntimeError('cube not found anywhere in the search area')
            if placed and rec.frames > 60:
                if ds is not None:
                    entry['episode_index'] = ds.num_episodes
                    ds.save_episode()
                saved += 1
                fails = 0
                log(f'== saved {saved}/{a.episodes} ({rec.frames} frames, {entry["seconds"]} s)')
                tray = reset(arm, tray, log, entry)
                skills.go(arm, READY, timeout=8)  # cycle ends at the start pose
                arm.hold(1.5)
                log('back at start pose (reset done)')
            else:
                if ds is not None:
                    ds.clear_episode_buffer()
                fails += 1
                log(f'-- discarded attempt {attempt} (held={held}, placed={placed}); consecutive fails {fails}')
                if held and not placed:  # cube fell somewhere: open and let the next search find it
                    arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
            if journal:
                with journal.open('a') as f:
                    f.write(json.dumps(entry) + '\n')
        log(f'done: {saved} saved in {attempt} attempts' + ('' if fails < a.max_failures else ' (too many failures)'))
    except Exception as e:
        log('STOPPED: ' + repr(e))
        log(traceback.format_exc())
        try:
            cv2.imwrite(str(logdir / f'{a.name}_{stamp}_stop.jpg'), cv2.cvtColor(arm.image(), cv2.COLOR_RGB2BGR))
        except Exception:
            pass
    finally:
        rec.on = False
        if ds is not None:
            try:
                ds.clear_episode_buffer()
            except Exception:
                pass
            ds.finalize()
        try:
            skills.go(arm, READY, timeout=8)
        finally:
            robot.disconnect()


if __name__ == '__main__':
    main()
