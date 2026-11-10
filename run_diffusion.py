#!/usr/bin/env python3
"""Run one planned SO-101 trial and record every observation/action/video frame."""

import argparse
import hashlib
import json
import select
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import torch

from diffusion_common import (
    BASE,
    IMAGE,
    JOINTS,
    device_for,
    load_policy,
    read_json,
    require,
    seed_all,
    versions,
    write_json,
)


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with (Path(path) / "model.safetensors").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrialVideo:
    def __init__(self, path, fps, height, width):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "23", "preset": "ultrafast"}

    def add(self, rgb):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def terminal_result():
    if not sys.stdin.isatty() or not select.select([sys.stdin], [], [], 0)[0]:
        return None
    line = sys.stdin.readline().strip().split(maxsplit=2)
    if line == ["s"]:
        return {"success": 1, "partial_score": 1.0, "reason": "full_success"}
    if line and line[0] in ("f", "q"):
        partial = float(line[1]) if len(line) > 1 else 0.0
        require(partial in (0.0, 0.25, 0.5, 0.75), "失败的 partial score 应为 0/0.25/0.5/0.75")
        return {
            "success": 0,
            "partial_score": partial,
            "reason": line[2] if len(line) > 2 else "operator_stop",
        }
    print("s + Enter = 完整成功；f 0.5 grasp_failed + Enter = 失败；q + Enter = 停止", flush=True)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--trial", type=int, required=True, help="plan.json 中 trial 的编号，从 0 开始")
    p.add_argument("--robot-config", type=Path, default=BASE / "config.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dry-run", action="store_true", help="检查计划、checkpoint、相机配置；不连接硬件")
    a = p.parse_args()
    plan = read_json(a.plan)
    require(0 <= a.trial < len(plan["trials"]), "trial 编号不在计划中")
    trial = plan["trials"][a.trial]
    checkpoint = Path(plan["checkpoint"]).expanduser().resolve()
    require(checkpoint_hash(checkpoint) == plan["checkpoint_sha256"], "checkpoint 权重与预先固定的计划不一致")
    metadata = read_json(checkpoint / "training_metadata.json")
    require(not metadata["smoke_test"], "smoke-test checkpoint 仅用于软件验证，不能用于实机评测")
    robot_cfg = read_json(a.robot_config)
    fps = metadata["dataset"]["fps"]
    camera = robot_cfg["cameras"]["wrist"]
    shape = metadata["dataset"]["features"][IMAGE]["shape"]
    require(camera["type"] == "opencv", "当前部署支持采集时的 OpenCV wrist 相机")
    require(
        [camera["height"], camera["width"], 3] == shape and camera["fps"] == fps,
        "部署相机分辨率/fps 必须与采集相同",
    )
    require(robot_cfg["fps"] == fps, "机器人控制 fps 与数据不一致")
    policy, pre, post = load_policy(checkpoint, device_for(a.device), trial["execution_horizon"])
    print(
        json.dumps(
            {
                "trial": trial,
                "fps": fps,
                "checkpoint": str(checkpoint),
                "follower_port": robot_cfg["follower_port"],
            },
            indent=2,
        )
    )
    if a.dry_run:
        return
    evaluation_root = a.plan.expanduser().resolve().parent
    for previous in range(a.trial):
        require(
            (evaluation_root / "trials" / f"{previous:04d}" / "result.json").is_file(),
            f"请按预先随机化的计划顺序执行；尚缺 trial {previous}",
        )
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    calibration = Path(robot_cfg["follower_calibration_file"]).expanduser().resolve()
    require(calibration.is_file(), f"请更新 {a.robot_config} 中 follower_calibration_file: {calibration}")
    require(Path(robot_cfg["follower_port"]).exists(), "follower 串口不存在，请更新机器人配置")
    settings = metadata["settings"]
    robot = SO101Follower(
        SO101FollowerConfig(
            port=robot_cfg["follower_port"],
            id=calibration.stem,
            calibration_dir=calibration.parent,
            cameras={"wrist": OpenCVCameraConfig(**{k: v for k, v in camera.items() if k != "type"})},
            use_degrees=settings["use_degrees"],
            max_relative_target=settings["max_relative_target"],
        )
    )
    output = a.plan.expanduser().resolve().parent / "trials" / f"{a.trial:04d}"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        **trial,
        "trial_index": a.trial,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "versions": versions(),
        "robot_config": robot_cfg,
        "deployment_settings": settings,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fps_requested": fps,
        "success": 0,
        "partial_score": None,
        "reason": "setup_error",
        "frames": 0,
        "duration_s": 0.0,
    }
    write_json(output / "attempt.json", result)
    writer, start = None, None
    try:
        robot.connect()
        # Warm up inference using the real current observation, without sending any command.
        observation = robot.get_observation()

        def tensors(observation):
            return {
                "observation.state": torch.tensor([[observation[k] for k in JOINTS]], dtype=torch.float32),
                IMAGE: torch.from_numpy(np.array(observation["wrist"], copy=True))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255),
            }

        with torch.inference_mode():
            post(policy.select_action(pre(tensors(observation))))
        policy.reset()
        seed_all(trial["seed"])
        writer = TrialVideo(output / "wrist.mp4", fps, shape[0], shape[1])
        print(
            "开始。完整成功（放入托盘、松爪并抬起）时输入 s + Enter；失败输入 f 0.5 原因 + Enter；Ctrl-C 停止。",
            flush=True,
        )
        start = time.perf_counter()
        result["reason"] = "timeout"
        late, latencies = 0, []
        with (output / "frames.jsonl").open("w") as log:
            while time.perf_counter() - start < plan["timeout_s"]:
                outcome = terminal_result()
                if outcome is not None:
                    result.update(outcome)
                    break
                tick = time.perf_counter()
                observation = robot.get_observation()
                writer.add(observation["wrist"])
                record = {
                    "frame_index": result["frames"],
                    "timestamp_s": time.perf_counter() - start,
                    "observation.state": [observation[k] for k in JOINTS],
                }
                try:
                    infer_start = time.perf_counter()
                    with torch.inference_mode():
                        action = post(policy.select_action(pre(tensors(observation)))).squeeze(0).cpu()
                    require(
                        action.shape == (6,) and torch.isfinite(action).all().item(), "policy 输出无效动作"
                    )
                    latency = time.perf_counter() - infer_start
                    latencies.append(latency)
                    target = dict(zip(JOINTS, action.tolist(), strict=True))
                    applied = robot.send_action(target)
                    record.update(action=target, applied_action=applied, inference_s=latency)
                finally:
                    record["cycle_s"] = time.perf_counter() - tick
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    result["frames"] += 1
                if time.perf_counter() - tick > 1 / fps:
                    late += 1
                time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))
        result.update(
            deadline_misses=late, inference_median_s=float(np.median(latencies)) if latencies else None
        )
    except KeyboardInterrupt:
        result["reason"] = "keyboard_interrupt"
    except Exception as error:
        result.update(reason="runtime_error", error=repr(error))
        raise
    finally:
        result["duration_s"] = time.perf_counter() - start if start is not None else 0.0
        result["completion_time_s"] = result["duration_s"] if result["success"] else None
        result["fps_observed"] = result["frames"] / result["duration_s"] if result["duration_s"] else 0
        # Persist the outcome even if video flushing or disconnection subsequently fails.
        write_json(output / "result.json", result)
        try:
            if writer is not None:
                writer.close()
        finally:
            # SOFollower.is_connected requires every camera as well as the bus.
            # Close components independently even when connect() failed partway.
            try:
                robot.bus.disconnect(robot.config.disable_torque_on_disconnect)
            finally:
                for camera_device in robot.cameras.values():
                    if camera_device.is_connected:
                        camera_device.disconnect()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
