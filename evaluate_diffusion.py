#!/usr/bin/env python3
"""Pre-register a matched h4/h8 evaluation; summarize results using official STEP."""

import argparse
import csv
import importlib.metadata
import json
import math
import random
from pathlib import Path

import numpy as np

from diffusion_common import read_json, require, write_json
from run_diffusion import checkpoint_hash


def make_plan(checkpoint, output, budget, seed, timeout):
    require(budget >= 25, "作业要求每种配置至少 25 次评测")
    require(timeout > 0, "timeout 必须为正数")
    checkpoint = checkpoint.expanduser().resolve()
    metadata = read_json(checkpoint / "training_metadata.json")
    cfg = read_json(checkpoint / "config.json")
    require(cfg["horizon"] - cfg["n_obs_steps"] + 1 >= 8, "此模型的 horizon 无法执行 8 步")
    require(not metadata["smoke_test"], "请选择正式训练的 checkpoint")
    rng = random.Random(seed)
    configs = list(range(budget))
    rng.shuffle(configs)
    trials = []
    for config in configs:
        policies = [("diffusion_h4", 4), ("diffusion_h8", 8)]
        rng.shuffle(policies)
        for name, horizon in policies:
            trials.append(
                {
                    "config_id": f"scene_{config:03d}",
                    "policy": name,
                    "execution_horizon": horizon,
                    "seed": seed + config,
                }
            )
    plan = {
        "n_max": budget,
        "alpha_pair": 0.10,
        "inference_mode": "ranking",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "seed": seed,
        "timeout_s": timeout,
        "trials": trials,
    }
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    with (output / "schedule.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["trial", "config_id", "policy", "execution_horizon", "scene_notes"]
        )
        writer.writeheader()
        for index, row in enumerate(trials):
            writer.writerow(
                {"trial": index, **{k: v for k, v in row.items() if k != "seed"}, "scene_notes": ""}
            )
    print(output / "plan.json")


def wilson90(successes, n):
    z = 1.6448536269514722
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def summarize(plan_path):
    import matplotlib
    from sequentialized_barnard_tests import Decision, Hypothesis, MirroredStepTest

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = read_json(plan_path)
    root = plan_path.expanduser().resolve().parent
    records = []
    for index, trial in enumerate(plan["trials"]):
        path = root / "trials" / f"{index:04d}" / "result.json"
        require(path.is_file(), f"尚未完成计划：缺少 trial {index}；保留原 n_max，不提前结束")
        row = read_json(path)
        require(all(row[k] == v for k, v in trial.items()), f"trial {index} 与计划不匹配")
        require(row["checkpoint_sha256"] == plan["checkpoint_sha256"], "评测混用了不同权重")
        require(type(row["success"]) is int and row["success"] in (0, 1), "STEP 只接受二元成功标签")
        require(row["frames"] > 0, f"trial {index} 未执行 policy；请检查并保留失败尝试记录")
        row["video"] = str(path.parent / "wrist.mp4")
        records.append(row)
    require(
        [r["started_at"] for r in records] == sorted(r["started_at"] for r in records),
        "实际执行顺序与预先固定的计划不一致，不能重排结果用于 STEP",
    )
    # Plan order is randomized before evaluation; never sort by outcome or reshuffle it.
    names = ["diffusion_h4", "diffusion_h8"]
    grouped = [[r for r in records if r["policy"] == name] for name in names]
    require([r["config_id"] for r in grouped[0]] == [r["config_id"] for r in grouped[1]], "初始场景未对齐")
    require(all(len(rows) == plan["n_max"] for rows in grouped), "每种配置的 trial 数与预算不同")
    sequences = [[row["success"] for row in rows] for rows in grouped]
    write_json(
        root / "step_sequences.json",
        {
            "policies": names,
            "outcomes": sequences,
            "config_ids": [r["config_id"] for r in grouped[0]],
            "n_max": plan["n_max"],
        },
    )
    test = MirroredStepTest(
        n_max=plan["n_max"],
        alternative=Hypothesis.P0LessThanP1,
        alpha=plan["alpha_pair"],
        inference_mode="ranking",
    )
    # Calling on successive original prefixes explicitly preserves the first decision.
    decision, first = Decision.FailToDecide, None
    for n in range(1, plan["n_max"] + 1):
        decision = test.run_on_sequence(sequences[0][:n], sequences[1][:n]).decision
        if decision != Decision.FailToDecide:
            first = n
            break
    winner = (
        names[1]
        if decision == Decision.AcceptAlternative
        else names[0]
        if decision == Decision.AcceptNull
        else None
    )
    labels = ["a", "a"] if winner is None else ["a", "b"]
    summary = []
    for name, rows in zip(names, grouped, strict=True):
        s, n = sum(row["success"] for row in rows), len(rows)
        times = [r["completion_time_s"] for r in rows if r["success"]]
        quantiles = np.quantile(times, [0.25, 0.5, 0.75]).tolist() if times else [None] * 3
        summary.append(
            {
                "policy": name,
                "successes": s,
                "failures": n - s,
                "n": n,
                "success_rate": s / n,
                "wilson90": wilson90(s, n),
                "successful_time_q1_median_q3_s": quantiles,
                "unsuccessful_trials": [
                    {
                        "config_id": r["config_id"],
                        "duration_s": r["duration_s"],
                        "reason": r["reason"],
                        "partial_score": r["partial_score"],
                    }
                    for r in rows
                    if not r["success"]
                ],
            }
        )
    report = {
        "policies": summary,
        "step": {
            "decision": decision.name,
            "winner": winner,
            "first_decision_trials_per_policy": first,
            "n_max": plan["n_max"],
            "alpha_pair": plan["alpha_pair"],
            "inference_mode": "ranking",
            "risk_budget": "default linear",
            "version": importlib.metadata.version("sequentialized_barnard_tests"),
            "source_commit": "3ae1623906f82fe472a352ed4a6b8e4a712f1f3d",
        },
    }
    write_json(root / "report.json", report)
    with (root / "rollout_manifest.csv").open("w", newline="") as stream:
        fields = [
            "trial_index",
            "config_id",
            "policy",
            "success",
            "partial_score",
            "duration_s",
            "completion_time_s",
            "reason",
            "fps_observed",
            "video",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    rng = np.random.default_rng(0)
    draws = [rng.beta(r["successes"] + 1, r["failures"] + 1, 20000) for r in summary]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.violinplot(draws, showmedians=True, showextrema=False)
    ax.set(
        xticks=[1, 2],
        xticklabels=names,
        ylim=(0, 1.08),
        ylabel="Success probability",
        title="Beta(1, 1) posterior; STEP ranking, overall alpha=0.10",
    )
    for i, label in enumerate(labels, 1):
        ax.text(i, 1.02, label, ha="center")
    fig.tight_layout()
    fig.savefig(root / "posterior_violin.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("plan")
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--output", type=Path, default=Path("eval/cube_h4_h8"))
    create.add_argument("--budget", type=int, default=25)
    create.add_argument("--seed", type=int, default=2026)
    create.add_argument("--timeout", type=float, default=60)
    report = sub.add_parser("report")
    report.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "plan":
        make_plan(a.checkpoint, a.output, a.budget, a.seed, a.timeout)
    else:
        summarize(a.plan)


if __name__ == "__main__":
    main()
