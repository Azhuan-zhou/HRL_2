#!/usr/bin/env python3
"""Write a clean LeRobot v3 dataset: saved successes only, trim each episode at both ends."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from diffusion_common import BASE, DEFAULT_DATA, IMAGE, read_json, require, write_json

DEFAULT_PREPARED = BASE / "data/cube_placement_v2_diffusion"


def source_hashes(root):
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store"):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def prepare_data(source, output=DEFAULT_PREPARED, first=15, last=15):
    from lerobot.datasets.compute_stats import (
        aggregate_stats,
        auto_downsample_height_width,
        compute_episode_stats,
        get_feature_stats,
        sample_indices,
    )

    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    require(first >= 0 and last >= 0, "Trim frame counts must be nonnegative")
    # A prepared dataset is already trimmed; never apply the operation twice.
    existing = source / "meta/preparation.json"
    if existing.is_file():
        previous = read_json(existing)
        require(
            previous["trim_first_frames"] == first and previous["trim_last_frames"] == last,
            "Trim settings differ from the prepared dataset; prepare again from the original data",
        )
        require(previous["success_only"] is True, "Prepared data must contain only successful demonstrations")
        return source
    require(source != output and source not in output.parents, "Store prepared data outside the original dataset directory")
    hashes = source_hashes(source)
    if output.exists():
        previous = read_json(output / "meta/preparation.json")
        require(
            previous["source_sha256"] == hashes
            and previous["trim_first_frames"] == first
            and previous["trim_last_frames"] == last,
            "The prepared cache does not match the source data or trim settings; choose another output directory",
        )
        require((output / "meta/info.json").is_file(), "The prepared cache is incomplete")
        return output
    info = read_json(source / "meta/info.json")
    require(info["codebase_version"] == "v3.0", "A LeRobot v3 dataset is required")
    require(
        [k for k, v in info["features"].items() if v["dtype"] in ("video", "image")] == [IMAGE],
        "This preprocessor supports the single wrist-camera dataset collected for this task",
    )
    labels = {}
    for line in (source / "meta/manual_outcomes.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry.get("status") == "saved":
            require(type(entry.get("success")) is bool, "The success label must be a boolean")
            labels[entry["episode_index"]] = entry["success"]
    rows = []
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda r: r["episode_index"])
    require([r["episode_index"] for r in rows] == list(range(info["total_episodes"])), "Episode metadata is incomplete")
    require(all(r["episode_index"] in labels for r in rows), "Missing saved success/failure labels; labels cannot be inferred")
    kept = [r for r in rows if labels[r["episode_index"]]]
    require(bool(kept), "No successful demonstrations were found")
    require(all(r["length"] > first + last for r in kept), "A successful episode is too short to retain any frames after trimming")
    temp = output.with_name(output.name + ".preparing")
    require(not temp.exists(), f"An incomplete preparation directory exists; retain it for inspection or choose another output directory: {temp}")
    for folder in ("meta/episodes/chunk-000", "data/chunk-000", f"videos/{IMAGE}/chunk-000"):
        (temp / folder).mkdir(parents=True, exist_ok=True)
    tables, episode_rows, all_stats, mapping, video_groups = [], [], [], [], {}
    data_files = {}
    cursor = 0
    fps = info["fps"]
    try:
        for new_id, row in enumerate(kept):
            old_id, length = row["episode_index"], row["length"] - first - last
            relative = info["data_path"].format(
                chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
            )
            if relative not in data_files:
                data_files[relative] = pq.read_table(source / relative)
            table = data_files[relative]
            table = table.filter(pc.equal(table["episode_index"], old_id)).slice(first, length)
            require(table.num_rows == length, f"Episode {old_id} data frame count does not match its metadata")
            values = {
                "episode_index": np.full(length, new_id, dtype=np.int64),
                "frame_index": np.arange(length, dtype=np.int64),
                "index": np.arange(cursor, cursor + length, dtype=np.int64),
                "timestamp": (np.arange(length) / fps).astype(np.float32),
            }
            for key, value in values.items():
                column = table.schema.get_field_index(key)
                table = table.set_column(
                    column, table.schema.field(column), pa.array(value, type=table.schema.field(column).type)
                )
            arrays = {k: np.asarray(table[k].to_pylist()) for k in table.column_names}
            require(
                all(np.isfinite(arrays[k]).all() for k in ("action", "observation.state")), "Data contains NaN or Inf"
            )
            stats = compute_episode_stats(arrays, info["features"])
            all_stats.append(stats)
            tables.append(table)
            new_row = {
                "episode_index": new_id,
                "tasks": row["tasks"],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
                f"videos/{IMAGE}/chunk_index": 0,
                f"videos/{IMAGE}/file_index": new_id,
                f"videos/{IMAGE}/from_timestamp": 0.0,
                f"videos/{IMAGE}/to_timestamp": length / fps,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            episode_rows.append(new_row)
            mapping.append(
                {
                    "source_episode_index": old_id,
                    "episode_index": new_id,
                    "source_length": row["length"],
                    "kept_source_frames": [first, row["length"] - last],
                    "length": length,
                    "success": True,
                }
            )
            video_path = info["video_path"].format(
                video_key=IMAGE,
                chunk_index=row[f"videos/{IMAGE}/chunk_index"],
                file_index=row[f"videos/{IMAGE}/file_index"],
            )
            start = round(row[f"videos/{IMAGE}/from_timestamp"] * fps) + first
            video_groups.setdefault(video_path, []).append((start, start + length, new_id))
            cursor += length
        # Decode each source container once. Failed episodes and trimmed frames never reach the output video.
        for relative, segments in video_groups.items():
            segments.sort()
            segment_index = 0
            container, stream, sampled = None, None, []
            encoded = 0
            try:
                with av.open(str(source / relative)) as original:
                    for frame in original.decode(video=0):
                        frame_index = round(float(frame.pts * frame.time_base) * fps)
                        begin, end, new_id = segments[segment_index]
                        if frame_index < begin:
                            continue
                        require(frame_index < end, f"Video timestamps skipped a target frame: {relative}, {frame_index}")
                        if container is None:
                            target = temp / f"videos/{IMAGE}/chunk-000/file-{new_id:03d}.mp4"
                            container = av.open(str(target), mode="w")
                            stream = container.add_stream("libx264", rate=fps)
                            stream.width, stream.height, stream.pix_fmt = frame.width, frame.height, "yuv420p"
                            stream.codec_context.gop_size = 2
                            stream.options = {"crf": "18", "preset": "ultrafast", "tune": "zerolatency",
                                              "g": "2", "keyint_min": "2", "sc_threshold": "0", "bf": "0"}
                            sample_set = set(sample_indices(end - begin))
                            encoded, sampled = 0, []
                        rgb = frame.to_ndarray(format="rgb24")
                        if frame_index - begin in sample_set:
                            sampled.append(auto_downsample_height_width(rgb.transpose(2, 0, 1)))
                        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                            container.mux(packet)
                        encoded += 1
                        if frame_index == end - 1:
                            for packet in stream.encode():
                                container.mux(packet)
                            container.close()
                            container = None
                            require(encoded == end - begin, "Video frame count is incomplete")
                            image_stats = get_feature_stats(np.stack(sampled), axis=(0, 2, 3), keepdims=True)
                            all_stats[new_id][IMAGE] = {
                                k: np.array([encoded]) if k == "count" else np.squeeze(v / 255.0, axis=0)
                                for k, v in image_stats.items()
                            }
                            print(
                                f"Prepared successful episode {new_id + 1}/{len(kept)}: source episode {mapping[new_id]['source_episode_index']}, retained {encoded} frames",
                                flush=True,
                            )
                            segment_index += 1
                            if segment_index == len(segments):
                                break
                require(segment_index == len(segments), f"Video {relative} ended early")
            finally:
                if container is not None:
                    container.close()
        for row, stats in zip(episode_rows, all_stats, strict=True):
            row.update(
                {
                    f"stats/{key}/{stat}": value.tolist()
                    for key, feature in stats.items()
                    for stat, value in feature.items()
                }
            )
        pq.write_table(pa.concat_tables(tables), temp / "data/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(episode_rows), temp / "meta/episodes/chunk-000/file-000.parquet")
        shutil.copy2(source / "meta/tasks.parquet", temp / "meta/tasks.parquet")
        clean_info = copy.deepcopy(info)
        clean_info.update(total_episodes=len(kept), total_frames=cursor, splits={"train": f"0:{len(kept)}"})
        video_info = clean_info["features"][IMAGE]["info"]
        video_info.update({"video.codec": "h264", "video.crf": 18, "video.preset": "ultrafast", "video.g": 2})
        # Encoder-specific AV1 options from the source do not describe these new H.264 files.
        for key in ("video.fast_decode", "video.extra_options"):
            video_info.pop(key, None)
        write_json(temp / "meta/info.json", clean_info)
        write_json(
            temp / "meta/stats.json",
            {k: {s: v.tolist() for s, v in d.items()} for k, d in aggregate_stats(all_stats).items()},
        )
        with (temp / "meta/manual_outcomes.jsonl").open("w") as journal:
            for row in mapping:
                journal.write(
                    json.dumps(
                        {
                            "episode_index": row["episode_index"],
                            "success": True,
                            "outcome": "success",
                            "status": "saved",
                            "source_episode_index": row["source_episode_index"],
                        }
                    )
                    + "\n"
                )
        preparation = {
            "source_root": str(source),
            "source_sha256": hashes,
            "success_only": True,
            "trim_first_frames": first,
            "trim_last_frames": last,
            "excluded_failed_episodes": [r["episode_index"] for r in rows if not labels[r["episode_index"]]],
            "total_episodes": len(kept),
            "total_frames": cursor,
            "episode_mapping": mapping,
            "video_encoding": "H.264 CRF 18; only kept frames; image stats sampled from kept frames",
            "video_gop_size": 2,
        }
        write_json(temp / "meta/preparation.json", preparation)
        temp.rename(output)
    except BaseException:
        print(f"Preparation did not finish; original data was not modified. Intermediate files remain at {temp}", flush=True)
        raise
    print(f"Saved: {output}\n{len(kept)} successful episodes, {cursor} frames; failed episodes were excluded.", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--trim-first", type=int, default=15)
    parser.add_argument("--trim-last", type=int, default=15)
    args = parser.parse_args()
    prepare_data(args.data, args.output, args.trim_first, args.trim_last)


if __name__ == "__main__":
    main()
