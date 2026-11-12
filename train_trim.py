"""lerobot-train, but ignore the first/last frames of every episode. Dataset files are not changed.
Usage: python train_trim.py --drop_first=5 --drop_last=5 --config_path=configs/v1.json
--drop_first=auto drops each episode's idle start (until the action first moves), keeping a few frames before it.
"""
import argparse, sys

from torch.utils.data import SubsetRandomSampler


def motion_start(episode_index, action, thresh=3.0, margin=5):
    """{episode: frames to drop} so each episode starts `margin` frames before its action first moves > thresh."""
    import numpy as np
    starts = {}
    for e in np.unique(episode_index):
        a = action[episode_index == e]
        moved = np.abs(a - a[0]).max(1) > thresh
        starts[int(e)] = max(int(moved.argmax()) - margin, 0) if moved.any() else 0
    return starts


def trim(ds, first, last):
    """Narrow episode bounds: frames outside are neither sampled nor used as action targets.
    first: int, or 'auto' for a per-episode idle-start cut (see motion_start)."""
    # ponytail: normalization stats still include the dropped frames; negligible for a few frames
    if first == "auto":
        import numpy as np
        h = ds.hf_dataset.select_columns(["episode_index", "action"]).with_format("numpy")[:]
        starts = motion_start(h["episode_index"], np.stack(h["action"]))
        print("drop_first=auto per episode:", starts, flush=True)
        cut = lambda e: starts[int(e["episode_index"])]
    else:
        cut = lambda e: int(first)
    ds.meta.episodes = ds.meta.episodes.map(
        lambda e: {"dataset_from_index": e["dataset_from_index"] + cut(e), "dataset_to_index": e["dataset_to_index"] - last},
        load_from_cache_file=False, keep_in_memory=True)
    return ds


def kept_indices(ds):
    """Indices for ds[i]. LeRobot's EpisodeAwareSampler returns absolute ones, which crash with --dataset.episodes."""
    start, end = ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"]
    eps = range(len(start)) if ds.episodes is None else ds.episodes
    kept = [i for e in eps for i in range(start[e], end[e])]
    assert kept, "no frames left after trimming"
    to_rel = ds._absolute_to_relative_idx  # None when every episode is used
    return [to_rel[i] for i in kept] if to_rel else kept


if __name__ == "__main__":
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_train as t

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--drop_first", required=True, help="int, or auto")
    ap.add_argument("--drop_last", type=int, required=True)
    args, sys.argv[1:] = ap.parse_known_args()

    make_dataset = t.make_dataset

    def make_trimmed(cfg):
        ds = trim(make_dataset(cfg), args.drop_first, args.drop_last)
        t.EpisodeAwareSampler = lambda *a, **k: SubsetRandomSampler(kept_indices(ds))
        return ds

    t.make_dataset = make_trimmed
    ACTConfig.drop_n_last_frames = 0  # makes lerobot-train take the sampler branch patched above
    t.main()
