"""Human-in-the-loop (DAgger) collection: the policy runs, you take over with the leader arm before it fails.
Same start pose handling as eval.py (glide to the training start pose, Enter to begin).

Usage: python hil.py --policy.path=<model dir> --start=home --dataset.repo_id=local/hil_v11 --dataset.root=data/hil_v11
Keys:  space = pause / resume the policy (on pause the leader is driven to the follower pose)
       tab   = start / stop a correction while paused (each correction is saved as one episode)
       ESC   = end the session
Other lerobot-rollout flags pass through, e.g. --strategy.num_episodes=30.
"""
import sys

import lerobot.scripts.lerobot_rollout as roll
from lerobot.robots.so_follower.so_follower import SOFollower

import eval as ev

if __name__ == '__main__':
    SOFollower.connect = ev.connect_and_go_start
    args = []
    for a in sys.argv[1:]:
        if a.startswith('--start='):
            ev.START = ev.STARTS[a.split('=', 1)[1]]
        else:
            args.append(a)
    sys.argv = ['lerobot-rollout', '--config_path=configs/hil_dagger.json', *args]
    roll.main()
