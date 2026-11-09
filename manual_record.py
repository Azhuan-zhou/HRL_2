"""Terminal-controlled recording adapter; hardware and storage stay in LeRobot."""
import inspect
import json
import math
import os
from pathlib import Path
import select
import sys
import termios


class ManualStop(Exception):
    """Leave the official recording function through its cleanup block."""


class TerminalEvents(dict):
    def __init__(self):
        super().__init__(exit_early=False, stop_recording=False, rerecord_episode=False)
        self.phase = 'idle'
        self.pending = b''

    def begin(self, phase):
        self.phase = phase
        self.pending = b''
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        self['exit_early'] = False

    def line(self, block=False):
        while b'\n' not in self.pending:
            if not select.select([sys.stdin], [], [], None if block else 0)[0]:
                return None
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                raise ManualStop()
            self.pending += chunk
        line, self.pending = self.pending.split(b'\n', 1)
        return line.decode(errors='replace').strip().lower()

    def __getitem__(self, key):
        if key == 'exit_early' and self.phase in ('waiting', 'recording'):
            line = self.line()
            if line == 'q':
                if self.phase == 'waiting':
                    raise ManualStop()
                self['stop_recording'] = True
                return True
            if line == '':
                return True
            if line is not None:
                print('请按 Enter；退出请输 q 后按 Enter。', flush=True)
        return super().__getitem__(key)


def append_label(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def has_pending_frames(dataset):
    """Prefer the public writer API; retain compatibility with older datasets."""
    check = getattr(dataset, 'has_pending_frames', None)
    if callable(check):
        return check()
    buffer = getattr(dataset, 'episode_buffer', None)
    if buffer is not None:
        return buffer.get('size', 0) > 0
    raise ValueError('LeRobot 数据集缺少 has_pending_frames 接口，无法确认当前片段。')


class ManualController:
    def __init__(self, loop, target):
        self.loop = loop
        self.target = target
        self.saved = 0
        self.events = TerminalEvents()
        self.installed = False
        self.outcome = None

    def install_save(self, dataset):
        original = dataset.save_episode
        journal = Path(dataset.root) / 'meta' / 'manual_outcomes.jsonl'

        def save(*args, **kwargs):
            index = dataset.num_episodes
            record = {'episode_index': index, 'success': self.outcome == 'success',
                      'outcome': self.outcome}
            # Durable intent precedes data save; only a saved entry confirms completion.
            append_label(journal, {**record, 'status': 'pending'})
            result = original(*args, **kwargs)
            append_label(journal, {**record, 'status': 'saved'})
            self.saved += 1
            print(f'已保存 episode {index}: {self.outcome}；本次 {self.saved}/{self.target}', flush=True)
            if self.saved == self.target:
                print('已达到目标，仍可 Enter 继续录制；等待时 q + Enter 退出。', flush=True)
            return result

        dataset.save_episode = save
        self.installed = True

    def record_loop(self, **kwargs):
        dataset = kwargs.get('dataset')
        if dataset is None:
            return  # Reset is performed in the next unrecorded waiting phase.
        if not self.installed:
            self.install_save(dataset)
        events = self.events
        args = dict(kwargs, events=events, control_time_s=math.inf)
        events.begin('waiting')
        print('\n等待开始：可遥操作复位（不录数据）。Enter 开始；q + Enter 退出。', flush=True)
        self.loop(**dict(args, dataset=None))
        events.begin('recording')
        print('正在录制：Enter 结束本条（无时间上限）。', flush=True)
        self.loop(**args)
        events.begin('label')
        if not has_pending_frames(dataset):
            print('没有采集到帧，本条不保存。', flush=True)
            events['rerecord_episode'] = True
            return
        print('录制已停止。机械臂暂停跟随。输入 s 成功 / f 失败，然后 Enter（两者均保存）：', flush=True)
        while True:
            choice = events.line(block=True)
            if choice in ('s', 'f'):
                self.outcome = 'success' if choice == 's' else 'failure'
                return
            print('请输入 s 或 f，再按 Enter。', flush=True)


def loop_signature(loop):
    """Inspect LeRobot's safety wrapper without removing it during execution."""
    current = inspect.unwrap(loop)
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        signature = inspect.signature(current)
        required = {'robot', 'events', 'dataset', 'control_time_s'}
        if required <= signature.parameters.keys():
            return signature
        # safe_stop_image_writer uses a closure named func, without functools.wraps.
        if not inspect.isfunction(current):
            break
        wrapped = inspect.getclosurevars(current).nonlocals.get('func')
        if not callable(wrapped):
            break
        current = inspect.unwrap(wrapped)
    raise ValueError(f'无法确认 LeRobot record_loop 接口: {inspect.signature(loop)}；未连接硬件。')


def run(cli_args, target):
    if not sys.stdin.isatty():
        raise ValueError('手动录制需要交互式终端，请在 Terminal 或 IDE Terminal 中运行。')
    from lerobot.scripts import lerobot_record as module

    original_record = inspect.unwrap(module.record)
    loop_signature(module.record_loop)
    if original_record is module.record:
        raise ValueError('无法解析 LeRobot record 的配置装饰器；未连接硬件。')
    controller = ManualController(module.record_loop, target)
    old_loop, old_listener = module.record_loop, module.init_keyboard_listener
    old_argv = sys.argv

    def manual_record(cfg):
        # The target is informational: only the operator ends the session.
        cfg.dataset.num_episodes = math.inf
        cfg.play_sounds = False
        return original_record(cfg)

    manual_record.__annotations__ = {'cfg': module.RecordConfig}
    try:
        module.record_loop = controller.record_loop
        module.init_keyboard_listener = lambda: (None, controller.events)
        sys.argv = ['lerobot-record', *cli_args]
        module.register_third_party_plugins()
        module.parser.wrap()(manual_record)()
    except ManualStop:
        print('手动录制已结束。', flush=True)
    finally:
        module.record_loop, module.init_keyboard_listener = old_loop, old_listener
        sys.argv = old_argv
