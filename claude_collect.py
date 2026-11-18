"""Claude drives the SO-101 follower (no leader arm) and records a LeRobot dataset.

Every step Claude sees the camera image(s) + joint angles and returns the next joint target;
the arm moves there over `--move-s` seconds while frames are recorded at the dataset fps.
After each episode a human types s/f; only successes are saved (same rule as collect.py).

  pip install anthropic            # once, in the LeRobot env; plus ANTHROPIC_API_KEY
  python claude_collect.py --name claude_cube --episodes 5 --dry-run   # Claude thinks, arm does not move
  python claude_collect.py --name claude_cube --episodes 5
"""
import argparse, base64, json, time
from pathlib import Path

import anthropic
import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

from collect import calibration

BASE = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MAX_STEP = np.array([15, 15, 15, 15, 20, 30.0])  # calibration knob: max change per decision (deg; gripper 0-100)


class Decision(BaseModel):
    observation: str  # what Claude sees: cube, tray, gripper
    plan: str  # what this move should achieve
    target: list[float]  # 6 values in JOINTS order
    done: bool  # true when the cube is in the tray (or the attempt cannot be recovered)


def reference(ref_root, n_keyframes=12):
    """Joint limits from human demos and keyframes of one demo, for the prompt and for clamping."""
    stats = json.loads((ref_root / "meta/stats.json").read_text())["action"]
    lo, hi = np.array(stats["min"]) - 5, np.array(stats["max"]) + 5
    df = pd.read_parquet(next((ref_root / "data").rglob("*.parquet")), columns=["episode_index", "action"])
    demo = np.stack(df[df.episode_index == df.episode_index.min()]["action"].to_numpy())
    idx = np.linspace(0, len(demo) - 1, n_keyframes).astype(int)
    keyframes = "\n".join(f"t={i / 30:5.1f}s  " + ", ".join(f"{v:.0f}" for v in demo[i]) for i in idx)
    return lo, hi, keyframes


def system_prompt(task, lo, hi, keyframes):
    limits = "\n".join(f"- {j}: {a:.0f} .. {b:.0f}" for j, a, b in zip(JOINTS, lo, hi))
    return f"""You control a 6-DoF SO-101 robot arm by giving absolute joint targets.
Task: {task}

Joints, in order (degrees; gripper is 0..100, larger = more open). Allowed range:
{limits}
Each move is clamped to at most {", ".join(f"{s:.0f}" for s in MAX_STEP)} per joint, so plan in small steps.

A human demonstration of the same task (joint targets in the same order), for reference:
{keyframes}

Each turn you get the camera image(s) and the current joint angles. Return the next target.
Approach above the cube, open, descend, close the gripper, lift, move over the tray, open, retreat.
Set done=true once the cube is released in the tray, or if the cube fell somewhere unreachable."""


def to_jpeg_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return base64.standard_b64encode(buf.tobytes()).decode()


def ask_claude(client, model, system, obs, cams, history):
    state = [obs[j + ".pos"] for j in JOINTS]
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": to_jpeg_b64(obs[c])}}
               for c in cams]
    content.append({"type": "text", "text": f"Cameras: {', '.join(cams)}.\nCurrent joints: "
                    + ", ".join(f"{v:.1f}" for v in state) + "\nPrevious moves:\n" + ("\n".join(history[-8:]) or "none")})
    r = client.messages.parse(model=model, max_tokens=16000, output_config={"effort": "low"},
                              system=system, messages=[{"role": "user", "content": content}], output_format=Decision)
    if r.stop_reason == "refusal":
        raise RuntimeError(f"Claude refused: {r.stop_details}")
    return r.parsed_output


def safe_target(state, target, lo, hi):
    """Clamp a Claude target: per-step change, then the human-demo joint range."""
    if len(target) != len(JOINTS) or not np.all(np.isfinite(target)):
        return state.copy()
    step = np.clip(np.asarray(target, float) - state, -MAX_STEP, MAX_STEP)
    return np.clip(state + step, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--reference", default="data/cube_placement_v2", help="human demos: joint limits + example")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--max-steps", type=int, default=40, help="Claude decisions per episode")
    ap.add_argument("--move-s", type=float, default=1.0, help="seconds to reach each target")
    ap.add_argument("--dry-run", action="store_true", help="ask Claude and print targets; never move, never save")
    a = ap.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    c = json.loads((BASE / "config.json").read_text())
    lo, hi, keyframes = reference(BASE / a.reference)
    system = system_prompt(c["task"], lo, hi, keyframes)
    cal = calibration(c, "follower")
    robot = SO101Follower(SO101FollowerConfig(
        port=c["follower_port"], id=cal.stem, calibration_dir=cal.parent,
        max_relative_target=5.0,  # calibration knob: hardware-side clamp per 1/fps frame
        cameras={k: OpenCVCameraConfig(**{x: y for x, y in v.items() if x != "type"}) for k, v in c["cameras"].items()}))
    cams, fps = list(c["cameras"]), c["fps"]
    client = anthropic.Anthropic()

    dataset = None
    if not a.dry_run:
        root = BASE / "data" / a.name
        assert not root.exists(), f"{root} exists; pick another --name (never overwrites)"
        features = combine_feature_dicts(hw_to_dataset_features(robot.action_features, "action"),
                                         hw_to_dataset_features(robot.observation_features, "observation"))
        dataset = LeRobotDataset.create(f"local/{a.name}", fps, root=root, robot_type=robot.name, features=features,
                                        image_writer_threads=4 * len(cams))

    robot.connect()
    saved = 0
    try:
        for ep in range(a.episodes):
            input(f"\n[episode {ep + 1}/{a.episodes}] Reset the cube, then press Enter (Ctrl+C aborts)...")
            history = []
            for step in range(a.max_steps):
                obs = robot.get_observation()
                state = np.array([obs[j + ".pos"] for j in JOINTS])
                d = ask_claude(client, a.model, system, obs, cams, history)
                target = safe_target(state, d.target, lo, hi)
                print(f"  step {step:2d} | {d.plan} | target {np.round(target, 1).tolist()}{' | done' if d.done else ''}")
                history.append(f"{step}: {d.plan} -> " + ", ".join(f"{v:.0f}" for v in target))
                if d.done:
                    break
                for k in range(1, int(a.move_s * fps) + 1):  # linear interpolation, recorded at fps
                    t0 = time.perf_counter()
                    act = {j + ".pos": float(v) for j, v in zip(JOINTS, state + (target - state) * k / (a.move_s * fps))}
                    if not a.dry_run:
                        obs = robot.get_observation()
                        sent = robot.send_action(act)
                        dataset.add_frame({**build_dataset_frame(dataset.features, obs, prefix="observation"),
                                           **build_dataset_frame(dataset.features, sent, prefix="action"),
                                           "task": c["task"]})
                    time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))
            if a.dry_run:
                continue
            ok = input("  success? s = save, f = discard: ").strip().lower() == "s"
            if ok and dataset.episode_buffer["size"] > 0:
                dataset.save_episode()
                saved += 1
                print(f"  saved ({saved} so far)")
            else:
                dataset.clear_episode_buffer()
                print("  discarded")
    finally:
        robot.disconnect()
        if dataset is not None:
            dataset.finalize()
    print(f"done: {saved} episodes saved" + ("" if a.dry_run else f" to data/{a.name}"))


if __name__ == "__main__":
    main()
