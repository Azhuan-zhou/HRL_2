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

## 数据 sim_cube / 2026-11-11
- `python -m sim.collect_sim --name sim_cube --episodes 50 --seed 0`：50 段、22199 帧，123 次尝试（失败全是抓取没抓起来，不存）；用时 7.7 分钟。
- 搜索改成两个固定视角（先左 20°、再右 -15°，指尖 r 0.22 / z 0.13 / 俯仰 45°），看到就停。投影算过：sim 相机比夹爪朝上约 40°，抬高时手腕顶限位，只有低位前视能看到桌面，点头不增加视野；两个视角覆盖方块区域 r 25–36 cm、角度 -30–35°，所以方块随机距离从 22 cm 改为 25 cm。
- 起点仍是 `manual_record.py` 的 HOME，和 cube_night 平均起点不同。

## sim 数据 v2 + v7_sim / 2026-11-12
- 试过 sim 外观随机化（白色机械臂、条纹桌面、百叶窗墙、鱼眼/调色/噪点），采了 `data/sim_dr_0..3`（200 段），用户评价"太假"，已撤回代码，数据保留未用。
- 恢复原始渲染后采 `data/sim_plain_0/1`（seed 20/21，各 75 段，约 15 分钟）；合并为 `data/sim_only`（150 段、71780 帧）和 `data/sim_mix`（+ cube_night_clean，170 段）。合并前把 sim 的 robot_type 改为 `so_follower`（和真机一致，否则 merge 报错）。
- 起点改为 cube_night 平均姿态 ±4 抖动；托盘位置/朝向范围放大。
- Dell 训练 `configs/v7_sim.json`（sim_only，6000 步，batch 128，drop_first=auto/drop_last=15）。v7_sim_mix 开了后按用户要求停掉。
- 新增 `sim/eval_sim.py`：在 sim 随机场景（seed 100 起，没见过的布局）跑模型，统计方块进托盘的成功率，可存视频。

## v7_sim 第 1 轮评测 + sim 抓取修复 / 2026-11-12
- v7_sim（sim_only 150 段，6000 步，loss 0.047）在 sim 新布局（seed 100）20 段：成功 9/20 = 45%。失败基本都是抓取没抓起，之后模型照常去托盘或回起点，不会重试（数据里没有失败重试）。
- 根因 1：夹爪（gripper body）的碰撞是网格凸包，比真实固定指胖很多，下降时先压在方块顶上，夹在顶边把方块挤飞。修复：`scene.xml` 排除 gripper–cube 碰撞，在固定指内侧加薄垫片 `fixed_jaw_pad`（位置从网格量出：x -0.008..-0.012，z -0.099..-0.07）。专家抓取 13/25 → 20/25。
- 根因 2：抓取点设在"夹紧后方块中心"，方块侧面和固定指零间隙，稍偏就压到方块顶。修复：下降时方块沿合拢方向往活动指偏 `GRASP_CLEAR`（单边夹爪的正常用法）。扫描：-8 mm 0/40，4 mm 22/40，12 mm 39/40，16–30 mm ≈40/40，36 mm 27/40；取 22 mm。另加下垂补偿（悬停时按实测抓取点误差修正 2 次）。
- 专家完整流程 20/20（之前约 40%）。
- v7_sim 在新垫片物理下复测 10/20 = 50%。

## v8_sim_g + 推理设置对比 / 2026-11-12
- 数据 `data/sim_g`：sim_g_0/1（seed 30/31，各 150 段，专家约 98%）合并，300 段、152962 帧。`configs/v8.json`，10000 步，loss 3.10 → 0.115（2K）→ 0.060（5K）→ 0.042（10K）。模型 `~/HRL_models/v8_sim_g`。
- sim 评测（seed 100，30 段）。`eval_sim.py` 新增 `--n_action_steps`、`--ensemble`，并打印方块最大高度 cube_zmax：

  | 推理设置 | 成功 | 抬起（zmax > 5 cm） |
  |---|---|---|
  | 默认 n_action_steps 30，25 s | 15/30 = 50% | – |
  | n_action_steps 10 | 2/30 = 7% | 3/30 |
  | temporal ensembling 0.01，25 s | 18/30 = 60% | 24/30 |
  | temporal ensembling 0.01，35 s | 17/30 = 57% | 24/30 |

- 数据翻倍、专家变好，默认设置仍是 50%，所以瓶颈不在数据量。ensembling 明显改善抓取；n_action_steps 10 在动作块衔接处跳变，反而更差。
- 剩下的失败：没夹起（6 段，从不重试），以及抬起后在托盘外沿松手（7 段）。

## 专家加纠错/重抓 + 数据 sim_r / 2026-11-12
- `collect_sim.py` 把抓取拆成 `grasp()`：首次悬停加 `HOVER_NOISE` 1.2 cm 误差，再修正 3 次（教模型看图修正对准）；15% 的段首次故意偏 3 cm（`MISS_P`/`MISS_OFFSET`），没抬起就张开夹爪重新对准再抓，最多 3 次。L1 损失学到的是中位数行为，少数故意抓偏不会带偏正常抓取。
- `data/sim_r_0/1`（seed 40/41），各 157 次存下 150 段。合并为 `data/sim_r`，`configs/v9.json`（同 v8）在 Dell 训练。
- v9_sim_r（10000 步，loss 0.048）sim 评测 30 段：默认 18/30 = 60%（抬起 23/30）；**ensembling 0.01：25/30 = 83%，30/30 都抬起**，5 段失败都是在托盘外沿松手。
- v9 续训到 40000 步（`--resume=true --steps=40000 --save_freq=10000`，日志 `runs/v9_sim_r.resume.log`），20K/40K 各测一次。

## 专家加放置修正 + 数据 sim_p / 2026-11-12
- 放置也用同样办法：先到托盘上方并带 `PLACE_NOISE` 2 cm 偏差，再按手里方块的实际位置修正 3 次后松手。试跑 15/15。
- `data/sim_p_0/1`（seed 50/51）各 150 段 → `data/sim_p`；`configs/v10.json` 从头训 40000 步，save_freq 10000。
- v9 续训 20K（loss 0.033，约 15 轮）：**默认 28/30 = 93%**，ensembling 26/30 = 87%。训练不足是主要瓶颈之一，训够后不再需要 ensembling。模型 `~/HRL_models/v9_sim_r_020000`。
- 按用户要求停止 sim 工作（2026-11-12）。Dell 上 v9 续训停在 30K 之后（有 030000 检查点，未评测），v10_sim_p 停在 10K 之后（有 010000 检查点，未评测）。真机试用 `~/HRL_models/v9_sim_r_020000`（`--start=night`）。

## 数据 task1_home_clean / 2026-11-12
- `data/task1_home/`（Home 起点，54 段、24050 帧）删除 failure [0,43]；最短 284 帧（约 9.5 秒），没有过短或静止的片段。
- 用 `lerobot-edit-dataset delete_episodes` 生成 `data/task1_home_clean/`：52 段、22451 帧，全部 success。原始和 clean 两份都已 rsync 到 Dell `~/so101_act/HRL_2/data/`。

## v11_home_c50x25 / 2026-11-12 完成
- 配置 `configs/v11.json` → `task1_home_clean`（52 段），chunk 50 / n_action_steps 25，batch 128，lr 5e-5，10000 步（约 57 轮），每 2000 步存一次。
- 命令：`python train_trim.py --drop_first=auto --drop_last=15 --config_path=configs/v11.json`；日志 `runs/v11_home_c50x25.train.log`（Dell）。
- 拉回 `~/HRL_models/v11_home_4k`、`v11_home_6k`、`v11_home_c50x25`（10k），均已过 `fix_dell_model.py`。最终 loss 0.044。
- 初测：6k 表现还可以（Home 起步，max_relative_target=null）。

## v12_home_b16 / 2026-11-12 完成
- 数据：`task1_home_b` 录制中，取 16:32 停录时已写完的前 16 段，单独复制成 `data/task1_home_b16`（44858 帧，全部 success，每段 80–130 秒）。
- `configs/v12.json`：chunk 50 / n_action_steps 25，batch 128，lr 5e-5，4000 步；drop_first=auto / drop_last=15。Dell 上和另一个 GPU 任务共用，约 1.8 步/秒，用时约 37 分钟。
- 模型 `~/HRL_models/v12_home_b16_2k`、`v12_home_b16_4k`（均已过 fix_dell_model）。真机：`python eval.py --policy.path=... --start=home --robot.max_relative_target=null --duration=150`。
