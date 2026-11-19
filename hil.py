"""Human-in-the-loop (DAgger) collection: the policy runs, you take over with the leader arm before it fails.
Same start pose handling as eval.py (glide to the training start pose, Enter to begin).

Usage: python hil.py --policy.path=<model dir> --start=home --dataset.repo_id=local/hil_v11 --dataset.root=data/hil_v11
Keys (this terminal must stay focused): space = pause / resume the policy
       (on pause the leader is driven to the follower pose)
       tab = start / stop a correction while paused (each correction is one episode)
       ESC = end the session
Other lerobot-rollout flags pass through, e.g. --strategy.num_episodes=30.
"""
import sys

import lerobot.scripts.lerobot_rollout as roll
import lerobot.utils.keyboard_input as keys
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.robots.so_follower.so_follower import SOFollower

import eval as ev

_read_latest = OpenCVCamera.read_latest


def read_latest(self, max_age_ms: int = 2000):
    # A diffusion replan on MPS can stall the camera thread for several seconds.
    # Returning the last frame keeps the session alive; raising aborts collection.
    try:
        return _read_latest(self, max_age_ms=max_age_ms)
    except TimeoutError:
        with self.frame_lock:
            frame = self.latest_frame
        if frame is None:
            raise
        return frame


if __name__ == '__main__':
    # pynput's global hook raises SIGTRAP on this Mac before its
    # untrusted-listener fallback can run. Read space/tab/esc from the terminal.
    keys.pynput_can_capture = lambda: False
    OpenCVCamera.read_latest = read_latest
    SOFollower.connect = ev.connect_and_go_start
    args = []
    for a in sys.argv[1:]:
        if a.startswith('--start='):
            ev.START = ev.STARTS[a.split('=', 1)[1]]
        else:
            args.append(a)
    sys.argv = ['lerobot-rollout', '--config_path=configs/hil_dagger.json', *args]
    roll.main()
