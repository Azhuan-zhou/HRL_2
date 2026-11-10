"""Data selection and LeRobot DiffusionPolicy construction for cube placement."""

import hashlib
import importlib.metadata
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = Path.home() / "Desktop/cube_placement_v2"
IMAGE = "observation.images.wrist"
JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def read_json(path):
    return json.loads(Path(path).expanduser().read_text())


def write_json(path, value):
    path = Path(path)
    # Readers see either the old complete file or the new complete file.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temp = Path(stream.name)
        try:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def versions():
    return {
        p: importlib.metadata.version(p)
        for p in ["lerobot", "torch", "torchvision", "diffusers", "datasets", "av"]
    }


def device_for(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_data(root, config):
    root = Path(root).expanduser().resolve()
    info = read_json(root / "meta/info.json")
    require(info["codebase_version"] == "v3.0", "A complete LeRobot v3 dataset directory is required")
    features = info["features"]
    for key in ("action", "observation.state"):
        require(
            features[key]["shape"] == [6] and features[key]["names"] == JOINTS,
            f"{key} joint dimensions or order do not match SO-101",
        )
    require(features[IMAGE]["shape"][-1] == 3, "An RGB wrist camera is required")
    require(info["fps"] > 0, "Invalid fps")
    episodes = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda r: r["episode_index"])
    require(
        [r["episode_index"] for r in episodes] == list(range(info["total_episodes"])),
        "Episode metadata is incomplete or duplicated",
    )
    require(sum(r["length"] for r in episodes) == info["total_frames"], "Metadata frame counts do not match")
    for row in episodes:
        data = root / info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        video = root / info["video_path"].format(
            video_key=IMAGE,
            chunk_index=row[f"videos/{IMAGE}/chunk_index"],
            file_index=row[f"videos/{IMAGE}/file_index"],
        )
        require(data.is_file() and video.is_file(), f"episode {row['episode_index']} has missing data or video")
    # A pending label is an intent, never evidence that an episode was saved.
    labels = {}
    journal = root / "meta/manual_outcomes.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "saved":
                require(type(row.get("success")) is bool, "The success label must be a boolean")
                labels[row["episode_index"]] = row["success"]
    all_ids = list(range(info["total_episodes"]))
    if config["success_only"]:
        require(all(i in labels for i in all_ids), "Missing saved success labels; check the outcome journal")
        selected = [i for i in all_ids if labels[i]]
    else:
        selected = all_ids
    require(len(selected) >= 2, "At least two eligible demonstrations are required")
    require(0 < config["val_fraction"] < 1, "val_fraction must be between 0 and 1")
    shuffled = selected.copy()
    random.Random(config["seed"]).shuffle(shuffled)
    n_val = max(1, round(len(selected) * config["val_fraction"]))
    require(n_val < len(selected), "The training split is empty after validation splitting")
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    # Hash every source file, including videos, so a relocated dataset can be verified.
    hashes = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store"):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.relative_to(root))] = digest.hexdigest()
    manifest = {
        "root": str(root),
        "fps": info["fps"],
        "total_episodes": len(episodes),
        "total_frames": info["total_frames"],
        "success_only": config["success_only"],
        "train_episodes": train,
        "val_episodes": val,
        "excluded_episodes": [i for i in all_ids if i not in selected],
        "labels": labels,
        "sha256": hashes,
        "features": features,
        "task": episodes[0]["tasks"][0],
    }
    return manifest, episodes


def make_policy_config(settings, features, device):
    from lerobot.configs import FeatureType
    from lerobot.policies.diffusion import DiffusionConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features

    require(
        1 <= settings["n_action_steps"] <= settings["horizon"] - settings["n_obs_steps"] + 1,
        "Execution horizon must be <= horizon - n_obs_steps + 1",
    )
    ft = dataset_to_policy_features(features)
    return DiffusionConfig(
        input_features={k: v for k, v in ft.items() if k in ("observation.state", IMAGE)},
        output_features={k: v for k, v in ft.items() if v.type is FeatureType.ACTION},
        device=device,
        push_to_hub=False,
        n_obs_steps=settings["n_obs_steps"],
        horizon=settings["horizon"],
        n_action_steps=settings["n_action_steps"],
        down_dims=tuple(settings["down_dims"]),
        resize_shape=tuple(settings["resize_shape"]),
        crop_shape=None,
        crop_ratio=1.0,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        noise_scheduler_type=settings["noise_scheduler_type"],
        num_train_timesteps=settings["num_train_timesteps"],
        num_inference_steps=settings["num_inference_steps"],
        do_mask_loss_for_padding=True,
        drop_n_last_frames=0,
    )


def train_stats(rows, train_ids):
    from lerobot.datasets.compute_stats import aggregate_stats

    stats = []
    for row in rows:
        if row["episode_index"] not in train_ids:
            continue
        episode_stats = {}
        for name, value in row.items():
            if name.startswith("stats/"):
                key, stat = name[6:].rsplit("/", 1)
                if key in ("action", "observation.state", IMAGE):
                    episode_stats.setdefault(key, {})[stat] = np.asarray(value)
        require(bool(episode_stats), "Per-episode statistics are required")
        stats.append(episode_stats)
    return {
        k: {s: torch.as_tensor(v, dtype=torch.float32) for s, v in d.items()}
        for k, d in aggregate_stats(stats).items()
    }


def make_dataset(root, ids, cfg):
    from lerobot.datasets import LeRobotDataset

    # Let LeRobot handle v3 video offsets, temporal windows and episode-boundary padding.
    delta = {
        k: [i / read_json(Path(root) / "meta/info.json")["fps"] for i in cfg.observation_delta_indices]
        for k in cfg.input_features
    }
    delta["action"] = [i / read_json(Path(root) / "meta/info.json")["fps"] for i in cfg.action_delta_indices]
    return LeRobotDataset(
        "local/cube_placement_v2", root=root, episodes=ids, delta_timestamps=delta, video_backend="pyav"
    )


def load_policy(checkpoint, device, execution_horizon=None):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy

    checkpoint = Path(checkpoint).expanduser().resolve()
    cfg = DiffusionConfig.from_pretrained(checkpoint)
    cfg.device = device
    if execution_horizon is not None:
        require(1 <= execution_horizon <= cfg.horizon - cfg.n_obs_steps + 1, "Execution horizon exceeds the prediction range")
        cfg.n_action_steps = execution_horizon
    policy = DiffusionPolicy.from_pretrained(checkpoint, config=cfg, strict=True).eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()
    return policy, pre, post
