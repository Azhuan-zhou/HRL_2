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
        require(len(candidates) == 1, f'{role}: 请在 config.json 指定 {role}_calibration_file。候选: {candidates}')
        path = candidates[0]
    require(path.is_file(), f'标定文件不存在: {path}')
    data = json.loads(path.read_text())
    require(isinstance(data, dict) and bool(data), f'无效标定文件: {path}')
    return path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preview', 'check', 'teleop', 'record', 'inspect'])
    p.add_argument('--config', type=Path, default=BASE / 'config.json')
    p.add_argument('--name', default='cube_placement')
    p.add_argument('--episodes', type=int, default=5, help='手动模式为本次目标条数（达到后可继续）；定时模式为新增数量')
    p.add_argument('--timed', action='store_true', help='使用原始定时录制；默认 Enter 手动开始/结束')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='只打印命令；不连接硬件')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    require(a.name and all(x.isalnum() or x in '_-' for x in a.name), 'name 只能包含字母、数字、下划线和连字符')
    root = BASE / 'data' / a.name
    if a.mode == 'inspect':
        info = json.loads((root / 'meta/info.json').read_text())
        print(json.dumps({k: info.get(k) for k in ['codebase_version', 'fps', 'total_episodes', 'total_frames', 'features']}, ensure_ascii=False, indent=2))
        return
    cameras = c['cameras']
    require(bool(cameras), '至少配置一台相机')
    require(all(x['fps'] == c['fps'] for x in cameras.values()), '所有相机 fps 应与采集 fps 相同')
    indices = [x['index_or_path'] for x in cameras.values()]
    require(len(indices) == len(set(indices)), '相机编号不能重复')
    if a.mode == 'preview':
        import cv2
        caps = []
        try:
            for name, camera in cameras.items():
                cap = cv2.VideoCapture(camera['index_or_path'])
                caps.append((name, cap))
                require(cap.isOpened(), f'无法打开相机 {name}: {camera}')
                for prop, key in [(cv2.CAP_PROP_FRAME_WIDTH, 'width'), (cv2.CAP_PROP_FRAME_HEIGHT, 'height'), (cv2.CAP_PROP_FPS, 'fps')]:
                    cap.set(prop, camera[key])
            print('检查画面是否对应相机名称。按 q 关闭。')
            while True:
                for name, cap in caps:
                    ok, frame = cap.read()
                    require(ok, f'{name} 无法读取画面')
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
        print(role, c[role + '_port'], '标定:', path)
        require(Path(c[role + '_port']).exists(), f'{role} 串口不存在，请检查连接和配置')
    require(c['leader_port'] != c['follower_port'], 'leader/follower 串口不能相同')
    if a.mode == 'check':
        require(shutil.which('lerobot-record'), '当前环境缺少 lerobot-record')
        print('配置检查通过（未连接电机/相机）；接下来运行 preview 和 teleop。')
        return
    exe = shutil.which('lerobot-record' if a.mode == 'record' else 'lerobot-teleoperate')
    require(exe, '请激活安装 LeRobot 的 Python 环境')
    cmd = [exe]
    for prefix, role in [('robot', 'follower'), ('teleop', 'leader')]:
        cmd += [f'--{prefix}.type=so101_{role}', f'--{prefix}.port={c[role + "_port"]}', f'--{prefix}.id={paths[role].stem}', f'--{prefix}.calibration_dir={paths[role].parent}']
    cmd += ['--robot.cameras=' + json.dumps(cameras), '--display_data=' + str(c['display_data']).lower()]
    if a.mode == 'record':
        require(a.episodes > 0, 'episodes 必须大于 0')
        require(c['episode_time_s'] > 0 and c['reset_time_s'] >= 0 and c['fps'] > 0, '时长或 fps 无效')
        require(not root.exists() or a.resume, '目录已存在：使用 --resume 追加，或换 --name；不会覆盖数据')
        require(not a.resume or (root / 'meta/info.json').is_file(), '--resume 需要已有有效数据集')
        repo_id = 'local/' + a.name
        if a.resume:
            old = json.loads((root / 'meta/info.json').read_text())
            require(old['fps'] == c['fps'], '追加时 fps 必须与已有数据一致')
            for name, camera in cameras.items():
                feature = old['features'].get('observation.images.' + name)
                require(feature is not None, f'已有数据集缺少相机 {name}')
                require(feature['shape'][:2] == [camera['height'], camera['width']], f'{name} 分辨率与已有数据不一致')
            session = BASE / 'logs' / (a.name + '_config.json')
            if session.exists():
                saved = json.loads(session.read_text())
                require(saved['config'] == c, '配置与首次录制不一致；请恢复原 config，或换数据集名称')
                repo_id = saved['repo_id']
        cmd += [f'--dataset.repo_id={repo_id}', f'--dataset.root={root}', f'--dataset.num_episodes={a.episodes}', f'--dataset.fps={c["fps"]}', f'--dataset.episode_time_s={c["episode_time_s"]}', f'--dataset.reset_time_s={c["reset_time_s"]}', f'--dataset.single_task={c["task"]}', '--dataset.push_to_hub=false', '--resume=' + str(a.resume).lower()]
        # Newer LeRobot stamps repo IDs; disable stamping when supported.
        help_result = subprocess.run([exe, '--help'], capture_output=True, text=True)
        require(help_result.returncode == 0, 'lerobot-record --help 失败，请检查安装依赖:\n' + help_result.stderr[-3000:])
        if '--dataset.no_stamp' in help_result.stdout:
            cmd.append('--dataset.no_stamp=true')
    else:
        cmd.append(f'--fps={c["fps"]}')
    if a.mode == 'record' and not a.timed:
        print('手动模式：Enter 开始/结束；s 成功、f 失败；等待时 q + Enter 退出。目标条数不是上限。')
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
