"""Pick the Home pose: teleoperate with the leader, watch the wrist camera, save the pose.

python pick_home.py      (window keys: s = save current pose to configs/home_pose.json, q = quit)
Red box = cube detected, blue box = tray detected (both should be in view from Home).
"""
import json
import time
from pathlib import Path

import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

BASE = Path(__file__).resolve().parent
cfg = json.loads((BASE / 'config.json').read_text())
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def mask_blob(mask, min_area, compact):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < min_area:
            continue
        if compact and (area / (w * h) < 0.5 or max(w, h) / max(1, min(w, h)) > 2.5):
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    return best


def draw(img, rgb, color, compact):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if color == (0, 0, 255):
        mask = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | cv2.inRange(hsv, (165, 90, 60), (180, 255, 255))
    else:
        mask = cv2.inRange(hsv, (100, 120, 40), (130, 255, 255))
    box = mask_blob(mask, 400, compact)
    if box is None:
        return False
    x, y, w, h, _ = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return True


def main():
    cal = Path(cfg['follower_calibration_file'])
    wrist = cfg['cameras']['wrist']
    robot = SOFollower(SOFollowerRobotConfig(
        port=cfg['follower_port'], id=cal.stem, calibration_dir=cal.parent, max_relative_target=5.0,
        cameras={'wrist': OpenCVCameraConfig(index_or_path=wrist['index_or_path'], width=wrist['width'],
                                            height=wrist['height'], fps=wrist['fps'])},
        disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)
    leader_cal = Path(cfg['leader_calibration_file'])
    leader = SOLeader(SOLeaderTeleopConfig(port=cfg['leader_port'], id=leader_cal.stem, calibration_dir=leader_cal.parent))
    leader.connect(calibrate=False)
    out = BASE / 'configs' / 'home_pose.json'
    obs = robot.get_observation()
    start = {m: float(obs[m + '.pos']) for m in JOINTS}
    goal = leader.get_action()
    for i in range(1, 91):
        robot.send_action({f'{m}.pos': start[m] + (goal[f'{m}.pos'] - start[m]) * i / 90 for m in JOINTS})
        time.sleep(1 / 30)
    print('Move the leader. Window: s = save Home pose, q = quit.', flush=True)
    saved_msg, t_msg = '', 0
    try:
        while True:
            robot.send_action(leader.get_action())
            obs = robot.get_observation()
            q = {m: float(obs[m + '.pos']) for m in JOINTS}
            img = cv2.cvtColor(obs['wrist'], cv2.COLOR_RGB2BGR)
            cube = draw(img, obs['wrist'], (0, 0, 255), True)
            tray = draw(img, obs['wrist'], (255, 0, 0), False)
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
                out.write_text(json.dumps(pose, indent=2) + '\n')
                cv2.imwrite(str(BASE / 'configs' / 'home_pose_view.jpg'), img)
                saved_msg, t_msg = f'saved -> {out.name}', time.time()
                print('saved Home pose:', pose, flush=True)
            elif k == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        leader.disconnect()
        robot.disconnect()


if __name__ == '__main__':
    main()
