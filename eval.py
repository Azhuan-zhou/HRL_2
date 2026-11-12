"""lerobot-rollout, but every run starts from the training start pose.
Usage: python eval.py --policy.path=$HOME/HRL_models/v5_auto_c100x30 [--start=night|home]   (other lerobot-rollout flags pass through)
"""
import sys
import time

from lerobot.robots.so_follower.so_follower import SOFollower
import lerobot.scripts.lerobot_rollout as roll

# Mean first frame of cube_placement_clean [-14,-103,96,90,-107,66];
# lift/wrist backed off their mechanical stops so the motors don't stall pushing against them.
START = {'shoulder_pan': -14, 'shoulder_lift': -101, 'elbow_flex': 95, 'wrist_flex': 87, 'wrist_roll': -107, 'gripper': 66}
# --start=night: mean first frame of cube_night_clean (v6 and later models trained on it).
STARTS = {'placement': START,
          'home': {k: v for k, v in __import__('json').load(open('configs/home_pose.json')).items()},  # Task 1 Home pose
          'night': {'shoulder_pan': 1, 'shoulder_lift': -77, 'elbow_flex': 91, 'wrist_flex': 83, 'wrist_roll': -100, 'gripper': 64}}
GLIDE_S = 3.0

connect = SOFollower.connect


def glide(bus, goal, seconds=GLIDE_S, hz=30):
    start = bus.sync_read('Present_Position')
    steps = int(seconds * hz)
    for i in range(1, steps + 1):
        bus.sync_write('Goal_Position', {m: start[m] + (goal[m] - start[m]) * i / steps for m in goal})
        time.sleep(1 / hz)
    time.sleep(0.5)


def connect_and_go_start(self, *args, **kwargs):
    # Connect sometimes hits a stray "Incorrect status packet"; a retry clears it.
    for attempt in range(3):
        try:
            connect(self, *args, **kwargs)
            break
        except (ConnectionError, RuntimeError) as e:
            if attempt == 2:
                raise
            print(f'连接失败，重试 ({e})', flush=True)
            try:
                self.disconnect()
            except Exception:
                pass
            time.sleep(1)
    print('移动到训练起始姿态……', flush=True)
    glide(self.bus, START)
    # rollout captures the "initial position" right after connect, so it also returns here at the end.


if __name__ == '__main__':
    SOFollower.connect = connect_and_go_start
    args = [a for a in sys.argv[1:] if not a.startswith('--start=')]
    for a in sys.argv[1:]:
        if a.startswith('--start='):
            START = STARTS[a.split('=', 1)[1]]
    sys.argv = ['lerobot-rollout', '--config_path=configs/eval_rollout.json', *args]
    roll.main()
