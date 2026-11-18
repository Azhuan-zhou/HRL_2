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

## v3_c30x10 / v4_c100x30 / 2026-11-11 完成
- 数据：新的 `data/cube_placement_clean`（11-11 上传，45 段、39694 帧、30 fps、单 wrist 相机，45 段均标成功），不和 v2 数据合并。
- 两版都用 `train_trim.py` 首尾各丢 20 帧，batch 64，12500 步（样本总量 = 原 batch 8 × 10 万步），lr 3e-5（原 1e-5），AMP 开，8 workers，save_freq 2500。两版同时跑，每版约 3.1 步/秒、约 65 分钟。
- 提速记录：先按 batch 8 / 10 万步开跑（每版 20 步/秒 = 160 样本/秒，约 80 分钟），3 分钟后停掉；batch 64 + AMP 后每版约 190 样本/秒，只快约 20%，GPU 已到 100%，瓶颈是 480×640 图像过 ResNet。曾短暂试 2 万步，算下来要 1 小时 45 分，又改成 12500 步。三次启动的 runs 目录都还没生成，无残留。
- v3（`configs/v3.json`）：chunk_size 30 / n_action_steps 10，即预测 1 s、执行 0.33 s（按建议改短 chunk）。
- v4（`configs/v4.json`）：chunk_size 100 / n_action_steps 30，同 v2 设置，作为新数据上的对照。
- 用时：v3 66 分钟，v4 67 分钟（16:18 / 16:19 结束），约 20 epoch。
- loss（第 200 步 → 2K → 5K → 8K → 12.5K 末）：v3 3.78 → 0.233 → 0.102 → 0.076 → 0.059；v4 3.85 → 0.286 → 0.140 → 0.108 → 0.083。两版 chunk 长度不同，loss 不能直接比（v4 要预测 3.3 s，更难）。
- checkpoint：`runs/v3_c30x10/checkpoints/last/pretrained_model`、`runs/v4_c100x30/checkpoints/last/pretrained_model`（每 2500 步一个，每版 2.9G）。
- 结论：两个都保留，等真机对比成功率。无视频。

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

## 数据 cube_stack_merged / 2026-11-16
- 来源：`~/cube_stack.zip`（微信传来，52 段、120439 帧，全部标成功），解压到 `data/cube_stack/`；`data/task1_home_b_clean/` 从 Mac 同步（32 段的 task1_home_b 去掉失败段 18/21/22/28 后剩 28 段，72005 帧；b16 是 task1_home_b 的前 16 段）。
- 查重：cube_stack 与 cube_night / cube_placement / task1_home / b16 的 action 哈希、中间帧均无重合。
- 合并：`lerobot-edit-dataset --operation.type merge` → `data/cube_stack_merged/`，80 段、192444 帧；0–51 段对应 cube_stack，52–79 段对应 b_clean。
- 校验：抽 6 段首/中/末帧，state/action 完全相同，图像差为 0（视频直接拷贝）。

## v13_stack_merged / 2026-11-16 完成
- 数据 `data/cube_stack_merged`（80 段、192444 帧）。配置 `configs/v13.json`：除数据和步数外同 v12（ACT，chunk 50 / 执行 25，batch 128，lr 5e-5，AMP）；15000 步 ≈ 10 epoch，每 3000 步存一次。
- 命令 `runs/v13_stack_merged.run.sh`（`train_trim.py --drop_first=auto --drop_last=20`）；日志 `runs/v13_stack_merged.train.log`。
- 用时 77 分钟（14:48–16:05，约 3.3 步/秒）。loss：3.127（第 200 步）→ 0.189（1.8K）→ 0.150（3K）→ 0.075（12K）→ 0.066（15K）。
- checkpoint：`runs/v13_stack_merged/checkpoints/last/pretrained_model`（每 3000 步一个，共 2.9G）。
- 结论：保留，待真机测试。无离线检查、无视频。

## 数据 pick_place_merged / 2026-11-17
- 来源：`~/pick_place.zip`（微信传来，54 段、39443 帧，manual_outcomes 全部 success），解压到 `data/pick_place/`；我们的 `data/task1_home_clean/`（52 段、22451 帧，Home 起点单次抓放）。
- 查重：pick_place 与 task1_home / b16 / cube_stack / cube_night / cube_placement_v2 / cube_placement_clean 的逐段 action 哈希无重合。两边视频都是 av1 / yuv420p / g=2。
- 合并：`lerobot-edit-dataset --operation.type merge` → `data/pick_place_merged/`，106 段、61894 帧；0–53 段对应 pick_place，54–105 段对应 task1_home_clean。
- 校验：抽 6 段首/中/末帧，state/action 完全相同，图像差为 0。

## v14_stack_30k / 2026-11-17 完成
- 改了什么：从 v13 第 15000 步续训到 30000 步（约 20 epoch），其余同 v13。v13 的 `015000` checkpoint 拷到 `runs/v14_stack_30k/checkpoints/` 后在新目录续训，v13 目录不动。
- 配置 `configs/v14.json`；命令 `runs/v14_stack_30k.run.sh`（`--resume=true --steps=30000`）；日志 `runs/v14_stack_30k.train.log`。
- 续训后第 15200 步 loss 0.065（接上 v13 末尾的 0.066），约 3.3 步/秒，原预计 05:50 结束；04:45 起和 v15 共用 GPU，降到约 1.7 步/秒，实际 06:46 结束，用时 2 小时 13 分。
- loss：0.065（15.2K）→ 0.059（20K）→ 0.053（25K）→ 0.049（30K）。checkpoint：`runs/v14_stack_30k/checkpoints/last/pretrained_model`（每 3000 步一个）。
- 结论：保留，待真机和 v13、v15 对比。无离线检查、无视频。

## v15_stack_diffusion / 2026-11-17 完成
- 改了什么：策略换成 LeRobot 自带的 Diffusion Policy（`policy.type=diffusion`），没有新写代码；`train_trim.py` 原样可用。数据同 v13/v14。
- 配置 `configs/v15.json`：看 2 帧、预测 64 步（2.1 s）、执行 30 步（1 s）；图像缩到 240×320 再随机裁 216×288；DDIM，推理 10 步；batch 128，3 万步（约 20 epoch），lr 1e-4 余弦（默认），AMP，每 5000 步存一次。模型 263M 参数。
- 注意：`train_trim.py` 的采样器不使用 Diffusion 的 `drop_n_last_frames`，每段末尾的样本会用最后一个动作补齐（未改）。
- 命令 `runs/v15_stack_diffusion.run.sh`；日志 `runs/v15_stack_diffusion.train.log`。
- 第 200 步 loss 0.446，约 2.8 步/秒（和 v14 共用 GPU），显存两版合计 51 GB，07:35 结束，用时 2 小时 49 分（v14 结束后升到约 4 步/秒）。
- loss（噪声预测误差，不能和 ACT 比）：0.446（200）→ 0.014（5K）→ 0.010（15K）→ 0.006（30K）。checkpoint：`runs/v15_stack_diffusion/checkpoints/last/pretrained_model`（每 5000 步一个）。
- 结论：保留，待真机和 v14 对比。无离线检查、无视频。

## v16_pick_diffusion / 2026-11-17 完成
- 改了什么：Diffusion Policy 换到单次抓放数据 `data/pick_place_merged`（106 段、61894 帧），其余同 v15；1 万步（约 20 epoch），每 2500 步存一次。
- 配置 `configs/v16.json`；命令 `runs/v16_pick_diffusion.run.sh`；日志 `runs/v16_pick_diffusion.train.log`。
- 用时 43 分钟（11:33–12:16）。loss（噪声预测误差）：0.468（200）→ 0.015（3K）→ 0.012（5K）→ 0.009（8K）→ 0.007（10K）。
- checkpoint：`runs/v16_pick_diffusion/checkpoints/last/pretrained_model`（每 2500 步一个，共 12G）。
- 结论：保留，待真机和 v17 对比。无离线检查、无视频。

## v17_pick_act / 2026-11-17 完成
- 改了什么：ACT 在单次抓放数据 `data/pick_place_merged` 上训 1 万步（约 20 epoch），其余同 v13（chunk 50 / 执行 25，batch 128，lr 5e-5，AMP），每 2500 步存一次。用来和 v16（Diffusion）对比。
- 配置 `configs/v17.json`；命令 `runs/v17_pick_act.run.sh`；日志 `runs/v17_pick_act.train.log`；由 `runs/v17_pick_act.wait.sh` 等 v16 结束后自动启动。
- 用时 53 分钟（12:16–13:09，约 3.3 步/秒）。loss：3.112（200）→ 0.106（3K）→ 0.076（5K）→ 0.059（8K）→ 0.052（10K）。
- checkpoint：`runs/v17_pick_act/checkpoints/last/pretrained_model`（每 2500 步一个，共 2.4G）。
- 结论：保留，待真机和 v16 对比。无离线检查、无视频。

## v18_pickonly_diffusion / v19_pickonly_act / 2026-11-17 完成
- 改了什么：只用对方的 `data/pick_place`（54 段、39443 帧，不合并我们的 task1_home_clean），Diffusion 和 ACT 各训一版；6000 步（约 20 epoch），每 2000 步存一次，其余分别同 v16 / v17。两版同时跑。
- 配置 `configs/v18.json`（Diffusion）、`configs/v19.json`（ACT）；命令 `runs/v18_pickonly_diffusion.run.sh`、`runs/v19_pickonly_act.run.sh`；日志同名 `.train.log`。
- v18 第 200 步 loss 0.473，约 2.6 步/秒；v19 约 1.8 步/秒。v18 14:51 结束（41 分钟），v19 15:00 结束（50 分钟）。
- loss（2K → 4K → 6K）：v18 0.017 → 0.012 → 0.008（噪声预测误差）；v19 0.150 → 0.081 → 0.061。
- checkpoint：`runs/v18_pickonly_diffusion/checkpoints/last/pretrained_model`（8.9G）、`runs/v19_pickonly_act/checkpoints/last/pretrained_model`（1.8G），每 2000 步一个。
- 结论：两个都保留，待真机和 v16 / v17 对比。无离线检查、无视频。

## v20_stackonly_diffusion / v21_stackonly_act / 2026-11-17 完成
- 改了什么：只用对方的 `data/cube_stack`（52 段、120439 帧，不合并我们的 task1_home_b_clean），Diffusion 和 ACT 各训一版；20000 步（约 21 epoch），每 4000 步存一次，其余分别同 v18 / v19（即 v15 / v13 的设置）。两版同时跑。
- 配置 `configs/v20.json`（Diffusion）、`configs/v21.json`（ACT）；命令 `runs/v20_stackonly_diffusion.run.sh`、`runs/v21_stackonly_act.run.sh`；日志同名 `.train.log`。
- 开始时共用 GPU：v20 约 2.8 步/秒，v21 约 1.7 步/秒，显存合计 48 GB。
- v20 21:09 结束（2 小时 7 分），v21 21:44 结束（2 小时 42 分，v20 结束后提速）。
- loss（200 → 4K → 8K → 12K → 16K → 20K）：v20 0.454 → 0.014 → 0.012 → 0.010 → 0.007 → 0.006（噪声预测误差）；v21 3.122 → 0.101 → 0.075 → 0.062 → 0.055 → 0.050。
- checkpoint：`runs/v20_stackonly_diffusion/checkpoints/last/pretrained_model`（15G）、`runs/v21_stackonly_act/checkpoints/last/pretrained_model`（2.9G），每 4000 步一个。最后一步已拉到 Mac `~/so101_models/`。
- 结论：两个都保留，待真机和 v14 / v15（合并数据）对比。无离线检查、无视频。
