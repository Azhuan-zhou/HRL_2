"""lerobot-rollout, but every run starts from the training start pose.
Usage: python eval.py --policy.path=$HOME/so101_models/v14_stack_30k [--start=stack|pick|night|home]
       (other lerobot-rollout flags pass through)
"""
import json
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower.so_follower import SOFollower
import lerobot.scripts.lerobot_rollout as roll

from collect import calibration

BASE = Path(__file__).resolve().parent

# Mean first frame of cube_placement_clean [-14,-103,96,90,-107,66];
# lift/wrist backed off their mechanical stops so the motors don't stall pushing against them.
START = {'shoulder_pan': -14, 'shoulder_lift': -101, 'elbow_flex': 95, 'wrist_flex': 87, 'wrist_roll': -107, 'gripper': 66}
# --start=night: mean first frame of cube_night_clean (v6 and later models trained on it).
STARTS = {'placement': START,
          'home': json.loads((BASE / 'configs' / 'home_pose.json').read_text()),  # Task 1 Home pose
          'night': {'shoulder_pan': 1, 'shoulder_lift': -77, 'elbow_flex': 91, 'wrist_flex': 83, 'wrist_roll': -100, 'gripper': 64},
          # Mean first frame of cube_stack_merged (elbow ~49; home_pose.json is 46).
          'stack': {'shoulder_pan': -9.5, 'shoulder_lift': -70.1, 'elbow_flex': 49.2, 'wrist_flex': 76.1, 'wrist_roll': -99.1, 'gripper': 64.9},  # v14/v15
          'pick': {'shoulder_pan': -9.4, 'shoulder_lift': -69.9, 'elbow_flex': 49.8, 'wrist_flex': 75.7, 'wrist_roll': -99.5, 'gripper': 64.9}}  # pick_place_merged, v16/v17
GLIDE_S = 3.0
# --grip_offset=N: while the policy commands the gripper closed (below GRIP_CLOSE_BELOW), command N further
# closed so the jaws squeeze the cube harder. Written after max_relative_target clipping, which otherwise caps the
# squeeze (commanded minus actual jaw position) at 5. Units are the 0-100 gripper range; 0 = fully closed.
# In training data a held cube sits at about 9 with the command at about 4 (a squeeze of about 5).
GRIP_OFFSET = 0.0
GRIP_CLOSE_BELOW = 20.0
# Release: once the command rises GRIP_RELEASE_DELTA above its lowest value in this close, the offset is
# dropped at once (no squeeze while the policy opens) and stays off until the gripper opens past
# GRIP_CLOSE_BELOW. Smaller wobbles while holding keep the squeeze.
GRIP_RELEASE_DELTA = 3.0
_grip = {'min': None, 'releasing': False}

connect = SOFollower.connect
send_action = SOFollower.send_action


def hardware():
    """Ports, calibration and cameras live in config.json. Rollout files only keep the run."""
    c = json.loads((BASE / 'config.json').read_text())
    follower = calibration(c, 'follower')
    return {'type': 'so101_follower', 'port': c['follower_port'], 'id': follower.stem,
            'calibration_dir': str(follower.parent), 'cameras': c['cameras']}


def rollout_config(name):
    cfg = json.loads((BASE / name).read_text())
    robot = hardware()
    robot['max_relative_target'] = cfg.get('robot', {}).get('max_relative_target', 5.0)
    cfg['robot'] = robot
    out = BASE / 'logs' / ('_' + Path(name).name)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    return out


def send_action_grip(self, action):
    sent = send_action(self, action)
    g = action.get('gripper.pos')
    if not GRIP_OFFSET or g is None:
        return sent
    g = float(g)
    if g >= GRIP_CLOSE_BELOW:  # open: reset for the next grasp
        _grip.update(min=None, releasing=False)
        return sent
    _grip['min'] = g if _grip['min'] is None else min(_grip['min'], g)
    if g > _grip['min'] + GRIP_RELEASE_DELTA:
        _grip['releasing'] = True
    if not _grip['releasing']:
        tgt = max(g - GRIP_OFFSET, 0.0)
        self.bus.sync_write('Goal_Position', {'gripper': tgt})
        sent = {**sent, 'gripper.pos': tgt}
    return sent


def glide(bus, goal, seconds=GLIDE_S, hz=30):
    start = bus.sync_read('Present_Position')
    steps = int(seconds * hz)
    for i in range(1, steps + 1):
        bus.sync_write('Goal_Position', {m: start[m] + (goal[m] - start[m]) * i / steps for m in goal})
        time.sleep(1 / hz)
    time.sleep(0.5)


def release(self):
    # Camera connect runs after the serial port opens. A failed camera leaves the port open,
    # and the next connect() then dies with "already connected".
    bus = getattr(self, "bus", None)
    if bus is not None and bus.is_connected:
        try:
            bus.disconnect(False)
        except Exception:
            try:
                bus.port_handler.closePort()
            except Exception:
                pass
    for cam in getattr(self, "cameras", {}).values():
        if getattr(cam, "is_connected", False):
            try:
                cam.disconnect()
            except Exception:
                pass


def connect_and_go_start(self, *args, **kwargs):
    # Connect sometimes hits a stray "Incorrect status packet"; a retry clears it.
    for attempt in range(3):
        try:
            connect(self, *args, **kwargs)
            break
        except (ConnectionError, RuntimeError) as e:
            release(self)
            if "OpenCVCamera" in str(e):
                raise ConnectionError(
                    "腕部相机打不开。系统设置 → 隐私与安全性 → 相机，允许 Terminal（在 Cursor 里跑就允许 Cursor），然后重新运行。"
                ) from e
            if attempt == 2:
                raise
            print(f'连接失败，重试 ({e})', flush=True)
            time.sleep(1)
    print('移动到训练起始姿态……', flush=True)
    glide(self.bus, START)
    wait_for_enter(self)
    # rollout captures the "initial position" right after connect, so it also returns here at the end.


def wait_for_enter(self):
    """Hold the start pose and show the wrist camera until Enter."""
    import select

    import cv2

    print('已到起始姿态。看相机窗口，在终端按 Enter 开始。', flush=True)
    while True:
        self.bus.sync_write('Goal_Position', START)
        for name, cam in self.cameras.items():
            try:
                frame = cam.read_latest()
            except Exception:
                continue
            cv2.imshow(name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(30) & 0xFF in (10, 13):
            break
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    SOFollower.connect = connect_and_go_start
    ours = ('--start=', '--grip_offset=', '--grip_close_below=', '--grip_release_delta=')
    args = [a for a in sys.argv[1:] if not a.startswith(ours)]
    for a in sys.argv[1:]:
        if a.startswith('--start='):
            START = STARTS[a.split('=', 1)[1]]
        elif a.startswith('--grip_offset='):
            GRIP_OFFSET = float(a.split('=', 1)[1])
        elif a.startswith('--grip_close_below='):
            GRIP_CLOSE_BELOW = float(a.split('=', 1)[1])
        elif a.startswith('--grip_release_delta='):
            GRIP_RELEASE_DELTA = float(a.split('=', 1)[1])
    if GRIP_OFFSET:
        SOFollower.send_action = send_action_grip
        print(f'夹爪加紧：指令 < {GRIP_CLOSE_BELOW} 时再往里 {GRIP_OFFSET}', flush=True)
    sys.argv = ['lerobot-rollout', '--config_path=' + str(rollout_config('configs/eval_rollout.json')), *args]
    roll.main()
