"""Task 1 automatic collection. Every episode starts at the Home pose with cube AND tray in view.

python -m autocollect.task1 --name task1_auto --episodes 50 [--no-record] [--lighting night_lamp]

Recorded episode (Task 1 success criterion): Home -> approach cube -> grasp -> over tray -> release ->
lift slightly. Ends there.
Unrecorded reset: verify cube in tray -> take it out -> drop it at a random spot -> nudge the tray ->
re-locate tray -> Home, and only start the next episode if both cube and tray are visible from Home.
"""
import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np

from . import hw, run, skills

BASE = Path(__file__).resolve().parent.parent
HOME = json.loads((BASE / 'configs' / 'home_pose.json').read_text())
CUBE_THETA, CUBE_R = (-25.0, 35.0), (0.24, 0.36)  # beyond ~0.37 m the shoulder sags under load  # candidates; each drop is verified visible from Home
LIFT_END = 0.03                                    # "move up slightly" after release
# Tray pose seen fully from Home when Home was picked (configs/home_pose_view.jpg); pushes return toward it.
TRAY_ANCHOR = np.array([np.radians(-10.0), 0.38, 0.06])


def push_toward(arm, tray, target, log):
    """Push the tray toward target (theta, r) by up to 6 cm, then re-locate it."""
    cur, tgt = run.xy(tray), run.xy(target)
    d = min(np.linalg.norm(tgt - cur), 0.06)
    if d < 0.02:
        return tray
    v = (tgt - cur) / np.linalg.norm(tgt - cur)
    # push the near wall from outside; if that start is out of reach (tray close to the base),
    # push the far wall from inside the tray instead
    for start, end, how in ((cur - v * (run.TRAY_HALF + 0.03), cur - v * run.TRAY_HALF + v * d, 'outside'),
                            (cur + v * (run.TRAY_HALF - 0.04), cur + v * (run.TRAY_HALF - 0.01 + d), 'inside')):
        if all(run.reachable(*run.cyl(q), run.Z_PUSH) for q in (start, end)):
            break
    else:
        log('push back: not reachable')
        return tray
    log(f'push tray {d*100:.0f} cm toward theta={np.degrees(target[0]):.0f} r={target[1]:.2f} (from {how})')
    arm.move({'gripper': hw.GRIP_CLOSED}, timeout=2)
    run.to_hover(arm, *run.cyl(start), skills.Z_HOVER)
    skills.go(arm, np.array([*run.cyl(start), run.Z_PUSH]), timeout=4)
    run.move_line(arm, end, run.Z_PUSH)
    c = skills.tcyl(arm.cmd)
    skills.go(arm, np.array([c[0], c[1], skills.Z_HOVER]), timeout=4)
    arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
    return run.locate_tray(arm, log, guess=np.array([*run.cyl(end + v * run.TRAY_HALF), run.Z_TABLE]))


def at_home(arm, settle=0.4):
    c = skills.tcyl(arm.cmd)
    if c[2] < skills.Z_HOVER:  # come up before swinging back
        skills.go(arm, np.array([c[0], c[1], skills.Z_HOVER]), timeout=4)
    arm.move(HOME, timeout=8)
    arm.hold(settle)


def cube_seen(img):
    return skills.visible(img)


def tray_seen(img):
    return hw.blob(hw.blue_mask(img), 3000) is not None


def grasp_visible_cube(arm, log, retries=2, check_tray=True):
    """Approach the cube that is in view (no search), grasp, verify. Re-approach on a miss."""
    for k in range(retries + 1):
        if k:
            log(f'retry grasp {k}/{retries}')
            arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
            c = skills.tcyl(arm.cmd)
            skills.go(arm, np.array([c[0], c[1], max(c[2], 0.12)]), timeout=4)
            if not cube_seen(arm.image()):
                return False
        if not skills.approach_cont(arm, log=log):
            run.snap(arm, 'approach_fail')
            continue
        if check_tray and not skills.clear_fingers(arm, log=log):
            arm.move({'wrist_roll': HOME['wrist_roll']}, timeout=3)
            continue
        if skills.grasp(arm, log=log, check_tray=check_tray):
            return True
        run.snap(arm, 'missed')
    return False


def place(arm, tray, log):
    run.to_hover(arm, tray[0], tray[1], run.Z_PLACE)
    arm.move({'gripper': hw.GRIP_OPEN}, timeout=2, settle=0.3)
    c = skills.tcyl(arm.cmd)
    skills.go(arm, np.array([c[0], c[1], c[2] + LIFT_END]), timeout=3)


def cube_in_tray(arm, tray):
    run.to_hover(arm, tray[0], max(tray[1] - 0.03, skills.R_RANGE[0]), run.Z_LOOK)
    arm.hold(0.3)
    ok = run.in_tray(arm.image())
    if not ok:
        run.snap(arm, 'not_in_tray')
    return ok


def find_and_hold_cube(arm, log):
    """Wherever the cube is: look from Home, else raster-search; then grasp it."""
    at_home(arm)
    if not cube_seen(arm.image()) and not skills.search(arm, log=log):
        raise RuntimeError('cube not found')
    return grasp_visible_cube(arm, log, retries=3, check_tray=False)


def drop_random(arm, tray, log):
    """Holding the cube: drop it at a random spot away from the tray. Returns (theta, r)."""
    txy = run.xy(tray)
    for _ in range(300):
        th, r = np.radians(random.uniform(*CUBE_THETA)), random.uniform(*CUBE_R)
        if np.linalg.norm(run.xy([th, r]) - txy) > run.TRAY_HALF + 0.08 and run.reachable(th, r, run.Z_DROP):
            break
    else:
        raise RuntimeError('no drop spot')
    run.to_hover(arm, th, r, run.Z_DROP)
    skills.release(arm, lift=0.06)
    log(f'drop cube at theta={np.degrees(th):.0f} r={r:.2f}')
    return th, r


def reset(arm, tray, log, take_out=True):
    """Leave cube + tray at new poses, both visible from Home. Returns the tray estimate."""
    for attempt in range(5):
        if take_out or attempt:
            if take_out and attempt == 0:  # cube is in the tray: pick it from there
                run.to_hover(arm, tray[0], max(tray[1] - run.ahead(skills.Z_HOVER), skills.R_RANGE[0]), skills.Z_HOVER)
                held = cube_seen(arm.image()) and grasp_visible_cube(arm, log, retries=3, check_tray=False)
                if not held:
                    held = find_and_hold_cube(arm, log)
            else:
                held = find_and_hold_cube(arm, log)
            if not held:
                log('reset: could not pick the cube up')
                continue
            cube = run.xy(drop_random(arm, tray, log))
        else:
            cube = None
        try:  # new tray pose: a random target near the anchor, keeping clear of the cube
            for _ in range(100):
                tgt = np.array([TRAY_ANCHOR[0] + np.radians(random.uniform(-10, 10)),
                                min(TRAY_ANCHOR[1] + random.uniform(-0.05, 0.05), 0.38), run.Z_TABLE])
                if cube is None or np.linalg.norm(run.xy(tgt) - cube) > run.TRAY_HALF + 0.07:
                    break
            tray = push_toward(arm, tray, tgt, log)
        except (ValueError, RuntimeError) as e:
            log(f'reset: push skipped ({e})')
        at_home(arm)
        img = arm.image()
        if cube_seen(img) and tray_seen(img):
            return tray
        log(f'reset: from Home cube {"ok" if cube_seen(img) else "NOT visible"}, '
            f'tray {"ok" if tray_seen(img) else "NOT visible"} -> again')
        if not tray_seen(img):  # bring the tray back where Home can see it
            try:
                tray = push_toward(arm, tray, TRAY_ANCHOR, log)
            except (ValueError, RuntimeError) as e:
                log(f'push back failed ({e})')
        take_out = False
    raise RuntimeError('could not set up a scene visible from Home')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='task1_auto')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--lighting', default='night_lamp')
    ap.add_argument('--no-record', action='store_true')
    ap.add_argument('--max-failures', type=int, default=8)
    a = ap.parse_args()

    import fcntl
    lock = open('/tmp/autocollect.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another autocollect run is using the arm')

    logdir = BASE / 'logs' / 'autocollect'
    logdir.mkdir(parents=True, exist_ok=True)
    textlog = (logdir / f'{a.name}_{time.strftime("%Y%m%d_%H%M%S")}.log').open('a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        textlog.write(line + '\n'); textlog.flush()

    robot = hw.connect()
    ds = None if a.no_record else run.make_dataset(a.name, robot)
    rec = run.Recorder(ds)
    arm = hw.Arm(robot, on_tick=rec)
    journal = None if ds is None else Path(ds.root) / 'meta' / 'auto_log.jsonl'
    saved, fails, attempt = 0, 0, 0
    try:
        arm.max_step = hw.MAX_STEP * 2
        tray = run.locate_tray(arm, log)
        at_home(arm)
        global TRAY_ANCHOR
        if tray_seen(arm.image()):  # the scene as set up is visible from Home: vary the tray around here
            TRAY_ANCHOR = tray.copy()
            log(f'tray anchor = theta {np.degrees(tray[0]):.0f} r {tray[1]:.2f}')
        for _ in range(4):  # make sure Home sees the tray before anything else
            if tray_seen(arm.image()):
                break
            tray = push_toward(arm, tray, TRAY_ANCHOR, log)
            at_home(arm)
        if run.in_tray(arm.image()) or not cube_seen(arm.image()):
            tray = reset(arm, tray, log, take_out=run.in_tray(arm.image()))
        while saved < a.episodes and fails < a.max_failures:
            attempt += 1
            arm.max_step = hw.MAX_STEP * 2
            at_home(arm, settle=0.6)
            arm.max_step = hw.MAX_STEP
            rec.on, rec.frames, t0 = True, 0, time.time()
            held = grasp_visible_cube(arm, log)
            if held:
                place(arm, tray, log)
            rec.on = False
            arm.max_step = hw.MAX_STEP * 2
            secs = round(time.time() - t0, 1)
            placed = bool(held) and cube_in_tray(arm, tray)
            entry = {'attempt': attempt, 'held': bool(held), 'placed': placed, 'frames': rec.frames, 'seconds': secs,
                     'lighting': a.lighting, 'tray': [round(float(np.degrees(tray[0])), 1), round(float(tray[1]), 3)],
                     'time': time.strftime('%F %T')}
            if placed and rec.frames > 60:
                if ds is not None:
                    entry['episode_index'] = ds.num_episodes
                    ds.save_episode()
                saved, fails = saved + 1, 0
                log(f'== saved {saved}/{a.episodes} ({rec.frames} frames, {secs} s)')
            else:
                if ds is not None:
                    ds.clear_episode_buffer()
                fails += 1
                log(f'-- discarded attempt {attempt} (held={held}, placed={placed}); consecutive fails {fails}')
                arm.move({'gripper': hw.GRIP_OPEN}, timeout=2)
            if journal:
                with journal.open('a') as f:
                    f.write(json.dumps(entry) + '\n')
            if saved < a.episodes:
                tray = reset(arm, tray, log, take_out=placed)
        log(f'done: {saved} saved in {attempt} attempts' + ('' if fails < a.max_failures else ' (too many failures)'))
    except Exception as e:
        log('STOPPED: ' + repr(e))
        log(traceback.format_exc())
        try:
            run.snap(arm, 'stopped')
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
            at_home(arm)
        finally:
            robot.disconnect()


if __name__ == '__main__':
    main()
