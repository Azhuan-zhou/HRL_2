#!/usr/bin/env python3
"""Local SO-101 collection launcher. Hardware I/O is delegated to LeRobot."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

BASE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def calibration(cfg, role):
    explicit = cfg.get(role + '_calibration_file')
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
        folder = Path(HF_LEROBOT_CALIBRATION) / ('robots' if role == 'follower' else 'teleoperators') / ('so101_' + role)
        identity = cfg.get(role + '_id')
        candidates = [folder / (identity + '.json')] if identity else sorted(folder.glob('*.json'))
        require(len(candidates) == 1, f'{role}: specify {role}_calibration_file in config.json. Candidates: {candidates}')
        path = candidates[0]
    require(path.is_file(), f'Calibration file does not exist: {path}')
    data = json.loads(path.read_text())
    require(isinstance(data, dict) and bool(data), f'Invalid calibration file: {path}')
    return path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preview', 'check', 'teleop', 'record', 'inspect'])
    p.add_argument('--config', type=Path, default=BASE / 'config.json')
    p.add_argument('--name', default='cube_placement')
    p.add_argument('--episodes', type=int, default=5, help='Session target in manual mode (recording may continue); number of new episodes in timed mode')
    p.add_argument('--timed', action='store_true', help='Use timed recording; by default Enter starts and ends each episode manually')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Print the command without connecting hardware')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    require(a.name and all(x.isalnum() or x in '_-' for x in a.name), 'name may contain only letters, digits, underscores, and hyphens')
    root = BASE / 'data' / a.name
    if a.mode == 'inspect':
        info = json.loads((root / 'meta/info.json').read_text())
        print(json.dumps({k: info.get(k) for k in ['codebase_version', 'fps', 'total_episodes', 'total_frames', 'features']}, ensure_ascii=False, indent=2))
        return
    cameras = c['cameras']
    require(bool(cameras), 'Configure at least one camera')
    require(all(x['fps'] == c['fps'] for x in cameras.values()), 'All camera frame rates must match the recording fps')
    indices = [x['index_or_path'] for x in cameras.values()]
    require(len(indices) == len(set(indices)), 'Camera indices must be unique')
    if a.mode == 'preview':
        import cv2
        caps = []
        try:
            for name, camera in cameras.items():
                cap = cv2.VideoCapture(camera['index_or_path'])
                caps.append((name, cap))
                require(cap.isOpened(), f'Cannot open camera {name}: {camera}')
                for prop, key in [(cv2.CAP_PROP_FRAME_WIDTH, 'width'), (cv2.CAP_PROP_FRAME_HEIGHT, 'height'), (cv2.CAP_PROP_FPS, 'fps')]:
                    cap.set(prop, camera[key])
            print('Check that each view matches its camera name. Press q to close.')
            while True:
                for name, cap in caps:
                    ok, frame = cap.read()
                    require(ok, f'Cannot read a frame from {name}')
                    cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            for _, cap in caps:
                cap.release()
            cv2.destroyAllWindows()
        return
    version = importlib.metadata.version('lerobot')
    print('LeRobot:', version, 'Python:', sys.executable)
    paths = {role: calibration(c, role) for role in ['leader', 'follower']}
    for role, path in paths.items():
        print(role, c[role + '_port'], 'Calibration:', path)
        require(Path(c[role + '_port']).exists(), f'The {role} serial port does not exist; check the connection and configuration')
    require(c['leader_port'] != c['follower_port'], 'Leader and follower serial ports must differ')
    if a.mode == 'check':
        require(shutil.which('lerobot-record'), 'lerobot-record is missing from the current environment')
        print('Configuration checks passed (motors and cameras were not connected). Next, run preview and teleop.')
        return
    exe = shutil.which('lerobot-record' if a.mode == 'record' else 'lerobot-teleoperate')
    require(exe, 'Activate the Python environment containing LeRobot')
    cmd = [exe]
    for prefix, role in [('robot', 'follower'), ('teleop', 'leader')]:
        cmd += [f'--{prefix}.type=so101_{role}', f'--{prefix}.port={c[role + "_port"]}', f'--{prefix}.id={paths[role].stem}', f'--{prefix}.calibration_dir={paths[role].parent}']
    cmd += ['--robot.cameras=' + json.dumps(cameras), '--display_data=' + str(c['display_data']).lower()]
    if a.mode == 'record':
        require(a.episodes > 0, 'episodes must be greater than 0')
        require(c['episode_time_s'] > 0 and c['reset_time_s'] >= 0 and c['fps'] > 0, 'Invalid duration or fps')
        require(not root.exists() or a.resume, 'Directory already exists: use --resume to append or choose another --name; data will not be overwritten')
        require(not a.resume or (root / 'meta/info.json').is_file(), '--resume requires an existing valid dataset')
        repo_id = 'local/' + a.name
        if a.resume:
            old = json.loads((root / 'meta/info.json').read_text())
            require(old['fps'] == c['fps'], 'The fps must match the existing dataset when appending')
            for name, camera in cameras.items():
                feature = old['features'].get('observation.images.' + name)
                require(feature is not None, f'The existing dataset is missing camera {name}')
                require(feature['shape'][:2] == [camera['height'], camera['width']], f'The resolution of {name} does not match the existing dataset')
            session = BASE / 'logs' / (a.name + '_config.json')
            if session.exists():
                saved = json.loads(session.read_text())
                require(saved['config'] == c, 'Configuration differs from the initial recording; restore the original config or choose another dataset name')
                repo_id = saved['repo_id']
        cmd += [f'--dataset.repo_id={repo_id}', f'--dataset.root={root}', f'--dataset.num_episodes={a.episodes}', f'--dataset.fps={c["fps"]}', f'--dataset.episode_time_s={c["episode_time_s"]}', f'--dataset.reset_time_s={c["reset_time_s"]}', f'--dataset.single_task={c["task"]}', '--dataset.push_to_hub=false', '--resume=' + str(a.resume).lower()]
        # Newer LeRobot stamps repo IDs; disable stamping when supported.
        help_result = subprocess.run([exe, '--help'], capture_output=True, text=True)
        require(help_result.returncode == 0, 'lerobot-record --help failed; check the installed dependencies:\n' + help_result.stderr[-3000:])
        if '--dataset.no_stamp' in help_result.stdout:
            cmd.append('--dataset.no_stamp=true')
    else:
        cmd.append(f'--fps={c["fps"]}')
    if a.mode == 'record' and not a.timed:
        print('Manual mode: Enter starts/ends recording; s marks success, f failure; q then Enter while waiting quits. The target count is not a limit.')
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    (BASE / 'logs').mkdir(exist_ok=True)
    if a.mode == 'record':
        root.parent.mkdir(exist_ok=True)
        snapshot = {'config': c, 'repo_id': repo_id, 'lerobot_version': version, 'command': cmd}
        (BASE / 'logs' / (a.name + '_config.json')).write_text(json.dumps(snapshot, indent=2))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (BASE / 'logs' / (stamp + '_command.json')).write_text(json.dumps({'command': cmd, 'lerobot_version': version}, indent=2))
    if a.mode == 'record' and not a.timed:
        from manual_record import run
        run(cmd[1:], a.episodes)
        return
    # Inherit the terminal so keyboard controls and signal handling remain native.
    os.execv(exe, cmd)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
