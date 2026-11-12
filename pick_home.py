"""Pick the Home pose: teleoperate with the leader, watch the wrist camera, save the pose.

python pick_home.py      (window keys: s = save current pose to configs/home_pose.json, q = quit)
Red box = cube detected, blue box = tray detected (both should be in view from Home).
"""
import json
import time
from pathlib import Path

import cv2

from autocollect import hw
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

BASE = Path(__file__).resolve().parent
cfg = json.loads((BASE / 'config.json').read_text())
leader_cal = Path(cfg['leader_calibration_file'])


def draw(img, mask, color):
    b = hw.blob(mask, 400, compact=color == (0, 0, 255))
    if b is None:
        return False
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    i = max(range(1, n), key=lambda k: st[k, 4])
    x, y, w, h = st[i, :4]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return True


def main():
    robot = hw.connect()
    leader = SOLeader(SOLeaderTeleopConfig(port=cfg['leader_port'], id=leader_cal.stem, calibration_dir=leader_cal.parent))
    leader.connect(calibrate=False)
    out = BASE / 'configs' / 'home_pose.json'
    # glide to the leader's pose first instead of jumping there
    start, goal = hw.state(robot.get_observation()), leader.get_action()
    for i in range(1, 91):
        robot.send_action({f'{m}.pos': start[m] + (goal[f'{m}.pos'] - start[m]) * i / 90 for m in hw.JOINTS})
        time.sleep(1 / 30)
    print('Move the leader. Window: s = save Home pose, q = quit.', flush=True)
    saved_msg, t_msg = '', 0
    try:
        while True:
            robot.send_action(leader.get_action())
            obs = robot.get_observation()
            q = hw.state(obs)
            img = cv2.cvtColor(obs['wrist'], cv2.COLOR_RGB2BGR)
            cube = draw(img, hw.red_mask(obs['wrist']), (0, 0, 255))
            tray = draw(img, hw.blue_mask(obs['wrist']), (255, 0, 0))
            lines = [' '.join(f'{k[:5]}={v:.0f}' for k, v in q.items()),
                     f'cube {"YES" if cube else "no"}   tray {"YES" if tray else "no"}   (s=save, q=quit)']
            if time.time() - t_msg < 3:
                lines.append(saved_msg)
            for i, t in enumerate(lines):
                cv2.putText(img, t, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(img, t, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow('pick Home pose', img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                pose = {m: round(v, 1) for m, v in q.items()}
                out.write_text(json.dumps(pose, indent=2))
                cv2.imwrite(str(BASE / 'configs' / 'home_pose_view.jpg'), img)
                saved_msg, t_msg = f'saved -> {out.name}', time.time()
                print('saved Home pose:', pose, flush=True)
            elif k == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        leader.disconnect()
        robot.disconnect()  # follower keeps torque (holds the pose)


if __name__ == '__main__':
    main()
