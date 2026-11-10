#!/usr/bin/env python3
"""Train LeRobot's DiffusionPolicy from local v3 cube-placement demonstrations."""

import argparse
import json
import math
import shutil
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ColorJitter

from diffusion_common import (
    BASE,
    DEFAULT_DATA,
    IMAGE,
    device_for,
    inspect_data,
    load_policy,
    make_dataset,
    make_policy_config,
    read_json,
    require,
    seed_all,
    train_stats,
    versions,
    write_json,
)
from prepare_diffusion_data import DEFAULT_PREPARED, prepare_data


def rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def validation(policy, pre, loader, seed):
    state = rng_state()
    policy.eval()
    total, n = 0.0, 0
    try:
        torch.manual_seed(seed)
        with torch.inference_mode():
            for batch in loader:
                loss, _ = policy(pre(batch))
                total += loss.item() * batch["action"].shape[0]
                n += batch["action"].shape[0]
        return total / n
    finally:
        restore_rng(state)
        policy.train()


def save_checkpoint(output, step, policy, pre, post, optimizer, scheduler, best, metadata):
    # Write into a temporary directory, then rename; incomplete saves cannot look valid.
    final = output / "checkpoints" / f"{step:06d}"
    temp = final.with_name(final.name + ".tmp")
    temp.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(temp)
    pre.save_pretrained(temp)
    post.save_pretrained(temp)
    write_json(temp / "training_metadata.json", metadata)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best,
            "rng": rng_state(),
        },
        temp / "training_state.pt",
    )
    temp.rename(final)
    write_json(output / "latest.json", {"checkpoint": str(final.resolve())})
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--prepared-data", type=Path, default=DEFAULT_PREPARED)
    p.add_argument("--config", type=Path, default=BASE / "diffusion_config.json")
    p.add_argument("--output", type=Path, default=BASE / "runs/diffusion_cube")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--steps", type=int, help="总学习率计划步数；续训时保持原值")
    p.add_argument("--stop-after", type=int, help="到此全局步数保存并结束，不改变学习率计划")
    p.add_argument("--inspect", action="store_true", help="预处理数据并显示划分，不训练")
    p.add_argument("--smoke-test", action="store_true", help="真实数据训练两步，验证保存及重载")
    p.add_argument("--resume", type=Path, help="恢复本脚本保存的 checkpoint 目录")
    a = p.parse_args()
    settings = read_json(a.resume / "training_metadata.json")["settings"] if a.resume else read_json(a.config)
    for arg, key in [(a.batch_size, "batch_size"), (a.steps, "steps")]:
        if arg is not None:
            settings[key] = arg
    require(settings["success_only"] is True, "按本次要求，训练只允许成功演示")
    if a.smoke_test:
        settings.update(steps=2, batch_size=2, eval_freq=2, save_freq=2, val_batches=1, num_workers=0)
    for key in ("steps", "batch_size", "save_freq", "eval_freq", "val_batches"):
        require(settings[key] > 0, f"{key} 必须为正数")
    prepared = prepare_data(
        a.data, a.prepared_data, settings["trim_first_frames"], settings["trim_last_frames"]
    )
    manifest, rows = inspect_data(prepared, settings)
    manifest["preparation"] = read_json(prepared / "meta/preparation.json")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "total_episodes",
                    "total_frames",
                    "fps",
                    "train_episodes",
                    "val_episodes",
                    "excluded_episodes",
                )
            },
            indent=2,
        )
    )
    if a.inspect:
        return
    require(versions()["lerobot"] == "0.7.0", "请使用 requirements-diffusion.txt 中的 LeRobot 版本")
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion import DiffusionPolicy

    device = device_for(a.device)
    seed_all(settings["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    output = a.output.expanduser().resolve()
    metadata = {
        "settings": settings,
        "dataset": manifest,
        "versions": versions(),
        "device": device,
        "smoke_test": a.smoke_test,
        "action_representation": "absolute joint targets; body=degrees, gripper=0..100"
        if settings["use_degrees"]
        else "absolute joint targets; body=-100..100, gripper=0..100",
    }
    if a.resume:
        old = read_json(a.resume / "training_metadata.json")
        require(old["dataset"]["sha256"] == manifest["sha256"], "恢复训练的数据内容发生变化")
        require(old["settings"] == settings, "恢复时必须使用原训练设置")
        require(output == a.resume.expanduser().resolve().parent.parent, "恢复时 --output 必须为原 run 目录")
        policy, pre, post = load_policy(a.resume, device)
        cfg = policy.config
    else:
        require(not output.exists(), f"输出目录已存在，请换 --output 或使用 --resume: {output}")
        cfg = make_policy_config(settings, manifest["features"], device)
        policy = DiffusionPolicy(cfg).to(device)
        pre, post = make_pre_post_processors(cfg, dataset_stats=train_stats(rows, manifest["train_episodes"]))
        output.mkdir(parents=True)
        write_json(output / "training_metadata.json", metadata)
        shutil.copy2(a.config, output / "source_config.json")
    train = make_dataset(manifest["root"], manifest["train_episodes"], cfg)
    val = make_dataset(manifest["root"], manifest["val_episodes"], cfg)
    sampler = EpisodeAwareSampler(
        dataset_from_indices=[r["dataset_from_index"] for r in rows],
        dataset_to_indices=[r["dataset_to_index"] for r in rows],
        episode_indices_to_use=manifest["train_episodes"],
        shuffle=True,
        seed=settings["seed"],
        absolute_to_relative_idx=train.absolute_to_relative_idx,
    )
    # A private generator prevents DataLoader iterator creation from changing diffusion noise RNG.
    loader = DataLoader(
        train,
        batch_size=settings["batch_size"],
        sampler=sampler,
        num_workers=settings["num_workers"],
        generator=torch.Generator().manual_seed(settings["seed"]),
    )
    count = min(len(val), settings["val_batches"] * settings["batch_size"])
    indices = torch.linspace(0, len(val) - 1, count).long().tolist()
    val_loader = DataLoader(
        Subset(val, indices),
        batch_size=settings["batch_size"],
        num_workers=0,
        generator=torch.Generator().manual_seed(settings["seed"]),
    )
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=settings["lr"], betas=(0.95, 0.999), weight_decay=1e-6
    )

    def lr_scale(step):
        warmup = min(settings["warmup_steps"], max(1, settings["steps"] // 10))
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, settings["steps"] - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    step, best = 0, float("inf")
    if a.resume:
        # Only load training_state.pt generated by this script from your own trusted runs.
        state = torch.load(a.resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best = state["step"], state["best_val"]
        restore_rng(state["rng"])
    stop_at = settings["steps"] if a.stop_after is None else a.stop_after
    require(step < stop_at <= settings["steps"], "停止步数必须大于 checkpoint 步数且不超过总计划")
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("收到停止请求；当前训练步结束后验证并保存 checkpoint。", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    epoch = step // len(loader)
    sampler.set_epoch(epoch)
    skip = step % len(loader)
    jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03)
    policy.train()
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a") as log:
        while step < stop_at:
            for batch_index, batch in enumerate(loader):
                if batch_index < skip:
                    continue
                if settings["color_jitter"]:
                    # One transform shared across time within a batch; preserves geometry.
                    batch[IMAGE] = jitter(batch[IMAGE])
                optimizer.zero_grad(set_to_none=True)
                loss, _ = policy(pre(batch))
                require(torch.isfinite(loss).item(), f"step {step}: loss 非有限数")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), settings["grad_clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                step += 1
                record = {
                    "step": step,
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.monotonic() - started,
                }
                stopping = step >= stop_at or stop_requested
                is_best = False
                if step % settings["eval_freq"] == 0 or stopping:
                    record["val_loss"] = validation(policy, pre, val_loader, settings["seed"] + 1)
                    require(math.isfinite(record["val_loss"]), "验证 loss 非有限数；不覆盖最佳模型")
                    is_best = record["val_loss"] < best
                    best = min(best, record["val_loss"])
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step == 1 or step % 100 == 0 or "val_loss" in record:
                    print(json.dumps(record), flush=True)
                if is_best or step % settings["save_freq"] == 0 or stopping:
                    path = save_checkpoint(
                        output, step, policy, pre, post, optimizer, scheduler, best, metadata
                    )
                    if is_best:
                        write_json(
                            output / "best.json", {"checkpoint": str(path.resolve()), "val_loss": best}
                        )
                if stopping:
                    break
            skip = 0
            if stopping:
                break
            epoch += 1
            sampler.set_epoch(epoch)
    if a.smoke_test:
        reloaded, reload_pre, reload_post = load_policy(path, device)
        sample = val[0]
        obs = {k: sample[k][-1].unsqueeze(0) for k in cfg.input_features}
        with torch.inference_mode():
            action = reload_post(reloaded.select_action(reload_pre(obs)))
        require(action.shape == (1, 6) and torch.isfinite(action).all().item(), "重载后动作检查失败")
        report = {
            "passed": True,
            "train_steps": step,
            "action_shape": list(action.shape),
            "val_loss": best,
            "device": device,
            "versions": versions(),
        }
        write_json(output / "smoke_result.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
