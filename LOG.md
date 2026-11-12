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


## 数据 cube_placement_clean / 2026-11-11
- 从 `data/cube_placement/`（52 段）删除 [0,5,30,41,42,49,50]，用 `lerobot-edit-dataset delete_episodes` 生成 `data/cube_placement_clean/`：45 段、39694 帧，全部 success。
  - 0、30：标 failure。
  - 5、41、42、49、50：action 全程不变（机械臂没动，疑似 leader 未读到），却标 success。按每段关节 action 的极差扫出来的；v2 无此问题（最小极差 11.8，为已删的 failure 段 1）。
- `manual_outcomes.jsonl` 按新编号同步复制。
- 校验：原数据未改；新 45 段 action 与原对应段逐值相同；最短 18.3 秒，最小关节极差 118.6。

## 数据 cube_night_clean / 2026-11-11
- 从 `data/cube_night/`（22 段、14468 帧）删除 [1,12]，用 `lerobot-edit-dataset delete_episodes` 生成 `data/cube_night_clean/`：20 段、14057 帧，全部 success。
  - 1：标 failure，action 全程不变（4.9 秒）。
  - 12：action 全程不变（8.8 秒），却标 success。
- `manual_outcomes.jsonl` 按新编号同步复制；新 20 段 action 与原对应段逐值相同；最小关节极差 113.1。
- 已删空目录 `data/auto_cube/`（0 段）。`cube_night` 与 `cube_night_clean` 均已 rsync 到 Dell `~/so101_act/HRL_2/data/`。

## v6_night / 2026-11-11 完成
- 配置 `configs/v6.json` → `cube_night_clean`（20 段），chunk 100 / n_action_steps 30，batch 128，lr 5e-5，2000 步（约 18 轮）。
- 命令：`python train_trim.py --drop_first=auto --drop_last=15 --config_path=configs/v6.json`；开头按段删 42–128 帧静止，结尾删 15 帧。日志 `runs/v6_night.train.log`（Dell）。
- 用时 12 分钟，约 3.2 步/秒，显存 32 GB。loss：3.11（200）→ 0.195（1K）→ 0.101（2K），还在下降。
- 模型已拉到 Mac `~/HRL_models/v6_night`，已跑 `fix_dell_model.py`。cube_night 起始姿态和 cube_placement 不同，真机跑要加 `--start=night`。
- 修复 `autocollect/run.py` 的 `in_tray` 返回 np.bool_，导致写 journal 时 json 报错退出（run3 21:27）。

## 数据 task1_home_clean / 2026-11-12
- `data/task1_home/`（Home 起点，54 段、24050 帧）删除 failure [0,43]；最短 284 帧（约 9.5 秒），没有过短或静止的片段。
- 用 `lerobot-edit-dataset delete_episodes` 生成 `data/task1_home_clean/`：52 段、22451 帧，全部 success。原始和 clean 两份都已 rsync 到 Dell `~/so101_act/HRL_2/data/`。

## v11_home_c50x25 / 2026-11-12 完成
- 配置 `configs/v11.json` → `task1_home_clean`（52 段），chunk 50 / n_action_steps 25，batch 128，lr 5e-5，10000 步（约 57 轮），每 2000 步存一次。
- 命令：`python train_trim.py --drop_first=auto --drop_last=15 --config_path=configs/v11.json`；日志 `runs/v11_home_c50x25.train.log`（Dell）。
- 拉回 `~/HRL_models/v11_home_4k`、`v11_home_6k`、`v11_home_c50x25`（10k），均已过 `fix_dell_model.py`。最终 loss 0.044。
- 初测：6k 表现还可以（Home 起步，max_relative_target=null）。
