import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diffusion_common import BASE, IMAGE, JOINTS, inspect_data, read_json, train_stats, write_json
from evaluate_diffusion import make_plan, wilson90
from run_diffusion import TrialVideo


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    for folder in ("meta/episodes/chunk-000", "data/chunk-000", f"videos/{IMAGE}/chunk-000"):
        (root / folder).mkdir(parents=True)
    features = {
        key: {"shape": [6], "names": JOINTS, "dtype": "float32"} for key in ("action", "observation.state")
    }
    features[IMAGE] = {"shape": [48, 64, 3], "dtype": "video"}
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 4,
        "total_frames": 40,
        "fps": 30,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    write_json(root / "meta/info.json", info)
    rows = [
        {
            "episode_index": i,
            "length": 10,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{IMAGE}/chunk_index": 0,
            f"videos/{IMAGE}/file_index": 0,
            "tasks": ["place cube"],
        }
        for i in range(4)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "data/chunk-000/file-000.parquet").write_bytes(b"not decoded in metadata test")
    (root / f"videos/{IMAGE}/chunk-000/file-000.mp4").write_bytes(b"not decoded in metadata test")
    labels = [{"episode_index": i, "success": i != 1, "status": "saved"} for i in range(4)]
    labels.append({"episode_index": 1, "success": True, "status": "pending"})
    (root / "meta/manual_outcomes.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
    return root


def test_success_selection_ignores_pending_and_splits_whole_episodes(dataset):
    manifest, _ = inspect_data(dataset, read_json(BASE / "diffusion_config.json"))
    assert manifest["excluded_episodes"] == [1]
    assert not set(manifest["train_episodes"]) & set(manifest["val_episodes"])
    assert sorted(manifest["train_episodes"] + manifest["val_episodes"]) == [0, 2, 3]
    repeat, _ = inspect_data(dataset, read_json(BASE / "diffusion_config.json"))
    assert repeat["train_episodes"] == manifest["train_episodes"]


def test_missing_saved_labels_and_files_are_rejected(dataset):
    (dataset / "meta/manual_outcomes.jsonl").write_text("")
    with pytest.raises(ValueError, match="labels"):
        inspect_data(dataset, read_json(BASE / "diffusion_config.json"))
    (dataset / f"videos/{IMAGE}/chunk-000/file-000.mp4").unlink()
    with pytest.raises(ValueError, match="missing data"):
        inspect_data(dataset, read_json(BASE / "diffusion_config.json"))


def test_stats_exclude_validation_and_failed_episode_extremes():
    def stats_row(index, value):
        stats = {
            "min": [value] * 6,
            "max": [value + 1] * 6,
            "mean": [value + 0.5] * 6,
            "std": [0.5] * 6,
            "count": [10],
        }
        return {"episode_index": index, **{f"stats/action/{k}": v for k, v in stats.items()}}

    rows = [stats_row(0, 0), stats_row(1, -10000), stats_row(2, 10000)]
    stats = train_stats(rows, [0])
    assert stats["action"]["min"].tolist() == [0] * 6
    assert stats["action"]["max"].tolist() == [1] * 6


def test_video_writer_preserves_every_frame(tmp_path):
    path = tmp_path / "trial.mp4"
    writer = TrialVideo(path, 30, 48, 64)
    for i in range(5):
        writer.add(np.full((48, 64, 3), i * 30, dtype=np.uint8))
    writer.close()
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 5
    assert frames[0].width == 64 and frames[0].height == 48


def test_plan_precommits_budget_and_matched_randomized_scenes(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "training_metadata.json", {"smoke_test": False})
    write_json(checkpoint / "config.json", {"horizon": 16, "n_obs_steps": 2})
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    output = tmp_path / "evaluation"
    make_plan(checkpoint, output, 25, 123, 60)
    plan = read_json(output / "plan.json")
    h4 = [r["config_id"] for r in plan["trials"] if r["execution_horizon"] == 4]
    h8 = [r["config_id"] for r in plan["trials"] if r["execution_horizon"] == 8]
    assert h4 == h8 and len(set(h4)) == 25
    assert plan["alpha_pair"] == 0.1 and plan["n_max"] == 25
    with pytest.raises(ValueError, match="at least 25"):
        make_plan(checkpoint, tmp_path / "too_small", 24, 123, 60)
    with pytest.raises(FileExistsError):
        make_plan(checkpoint, output, 25, 123, 60)


def test_wilson_interval_handles_all_failures_or_successes():
    lower, upper = wilson90(0, 25)
    assert abs(lower) < 1e-12 and 0 < upper < 0.2
    lower, upper = wilson90(25, 25)
    assert 0.8 < lower < 1 and abs(upper - 1) < 1e-12
