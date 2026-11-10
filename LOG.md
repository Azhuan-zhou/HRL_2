# HRL_2 训练 LOG

## 合并进 HRL_2 / 2026-11-09
- 从 `so101_act/` 合并训练部分：`train_trim.py`、`configs/v1.json`、`configs/v1_eval.json`，`README.md` 加了“训练 ACT”“真机推理”两节。
- 已删 `so101_act/` 下的 `README.md`、`LOG.md`、`train_trim.py`、`configs/`（内容都已并入这里）。此前（同日）已删手写 ACT：`train.py`、`deploy.py`、`test_act.py`、`configs/smoke.json`、`runs/smoke_endoscope/`；手写版只跑过 20 步 smoke，官方 LeRobot 0.5.1 自带同样功能。
- 配置对齐 HRL_2：数据 `data/cube_placement`，串口/标定/单个 wrist 相机抄自 `config.json`。
- `train_trim.py` 修复：`--dataset.episodes` 时 LeRobot 的 `EpisodeAwareSampler` 用绝对帧号会 `IndexError`，改为自己算相对帧号。

## train_trim 验证 / 2026-11-09
- 数据 `jiafeng/endoscope_pick_cube_slices_v1`（不是 SO-101，只验证通路）。
- 全部 9 段各丢 5+7 帧：8329 → 8221；只选第 1、3 段：1100 → 1076，首尾帧号正确，末帧之后全部 pad。
- `configs/v1.json` + `--dataset.episodes='[1,3]'` 训练 2 步跑通。
- 结论：保留。首尾帧数待按真实数据定。

## 数据 cube_placement_v2 / 2026-11-10
- 来源：`~/cube_placement_v2.zip`（Mac 微信传来，原 zip 保留），解压到 `data/cube_placement_v2/`：38 段、35252 帧、30 fps、6 维关节、1 个 wrist 相机 480×640。
- 失败片段 [1,2,28,30]（来自 `manual_outcomes.jsonl`），用 `lerobot-edit-dataset delete_episodes` 生成 `data/cube_placement_v2_clean/`：34 段、32712 帧。
- 校验：原数据 md5 不变；抽查 6 段，在第 10 帧和最后一帧，state/action 与原数据逐值相同，图像差约 0.003（视频重新编码）。

## v1 / 2026-11-10 完成
- 配置：`configs/v1.json` → `cube_placement_v2_clean`，ACT 默认参数，n_action_steps 30，batch 8，10 万步，不丢帧（基线）。
- 命令见 `runs/v1_act/run.sh`；日志 `runs/v1_act/train.log`；配置快照 `runs/v1_act/config.json`。
- 用时 46 分钟（约 36 步/秒），显存 3.2 GB。loss：7.27（第 200 步）→ 0.248（8K）→ 0.096（40K）→ 0.064（100K），还在缓慢下降。
- 检查（训练数据，非留出集）：在第 0、17、33 段各取 10 帧，每帧 reset 后预测，chunk 首个动作的平均 |pred−gt| 分别为 0.92 / 2.15 / 1.84 / 1.63 / 1.29 / 0.34，各关节动作范围 66 / 176 / 149 / 91 / 126 / 49。
- checkpoint：`runs/v1_act/checkpoints/last/pretrained_model`（1.8G 含 040000/080000/100000）。
- 结论：待真机测试（Mac 上用 `configs/v1_eval.json`）。无视频。

## v2_trim20 / 2026-11-10 完成
- 改了什么：首尾各丢 20 帧，用 `train_trim.py`，其他和 v1 相同。依据：干净数据中每段开头静止帧中位数 29（2–64），结尾中位数 21（1–236），以 action 与首/末帧相差不超过 2 判定。20 帧能去掉大部分静止样本，又不切掉太多起步动作。
- 配置 `configs/v2.json`；命令 `runs/v2_trim20/run.sh`；日志 `runs/v2_trim20/train.log`。
- 用时 46 分钟。loss：7.23（第 200 步）→ 0.128（20K）→ 0.092（40K）→ 0.061（100K）。
- 检查（和 v1 相同的 30 帧，训练数据，非留出集），chunk 首个动作的平均 |pred−gt|：v1 0.92 / 2.15 / 1.84 / 1.63 / 1.29 / 0.34，v2 0.71 / 0.88 / 1.27 / 1.36 / 1.18 / 0.33。
- 结论：两个都保留，等真机对比成功率。checkpoint 在 `runs/v2_trim20/checkpoints/last/pretrained_model`。

