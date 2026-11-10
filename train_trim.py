"""lerobot-train, but ignore the first/last frames of every episode. Dataset files are not changed.
Usage: python train_trim.py --drop_first=5 --drop_last=5 --config_path=configs/v1.json
"""
import argparse, sys

from torch.utils.data import SubsetRandomSampler


def trim(ds, first, last):
    """Narrow episode bounds: frames outside are neither sampled nor used as action targets."""
    # ponytail: normalization stats still include the dropped frames; negligible for a few frames
    ds.meta.episodes = ds.meta.episodes.map(
        lambda e: {"dataset_from_index": e["dataset_from_index"] + first, "dataset_to_index": e["dataset_to_index"] - last},
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
    ap.add_argument("--drop_first", type=int, required=True)
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
