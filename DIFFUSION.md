# Cube placement：LeRobot Diffusion Policy

训练、保存和部署直接调用 LeRobot 0.7.0 的 `DiffusionPolicy`，不使用其他机器人任务预训练的权重。ResNet18 图像编码器也从零训练。原来的 `collect.py`、`manual_record.py` 和采集配置继续保留。

## 数据处理规则

原始数据默认在 `~/Desktop/cube_placement_v2`，处理后的数据在本仓库的 `data/cube_placement_v2_diffusion/`。原始文件不改动。

1. 仅接受 `meta/manual_outcomes.jsonl` 中 `status="saved"` 且 `success=true` 的演示。`pending` 不作为有效标签；缺少已保存标签会停止处理。所有 fail 演示完全排除。
2. **每条 episode 的开头 15 帧、结尾 15 帧都删除**。按 0 开始的下标，原长为 `N` 的演示保留 `[15, N-15)`，共 `N-30` 帧。30 Hz 下前后各删除 0.5 秒。
3. 同步裁剪图像、动作、状态等所有逐帧数据；重新编号 episode、frame、全局 index，并把每条演示的 timestamp 从 0 开始。
4. 为每条保留的成功演示重新输出一段 H.264 视频，视频中也不包含被删帧或失败演示。重编码使用 CRF 18、每 2 帧一个关键帧，便于训练时随机读取；会有轻微有损压缩，不裁图、不调整原始分辨率。
5. 重新计算逐条演示的统计量；图像统计量从保留帧中均匀抽样估计。训练集和验证集按完整 episode 划分，比例约 9:1，seed=42。归一化仅使用训练集统计量。

这里的“每条视频”指每条演示 episode：原始 v3 数据将 38 条演示放在了同一个 MP4 中，不能仅裁掉这个 MP4 的头尾。预处理脚本根据 episode 元数据定位每个片段。

本次原始数据有 38 条、35,252 帧；失败 episode 为 **1、2、28、30**，成功演示共 **34 条**。去除失败演示及成功演示前后各 15 帧后，已输出 **34 条、31,692 帧**。原始/新 episode 对应关系记录在 `data/cube_placement_v2_diffusion/meta/preparation.json`。

训练读取新的 episode 边界，观察历史和未来 action chunk 都在裁剪后的片段内取样；边界以片段内的首/尾帧补齐，并使用 padding mask。被删除的头尾不会通过时间窗口重新进入训练。

已经处理过的数据会根据来源文件 SHA256 与裁剪设置复用；不会再删除第二次。原始数据发生变化或调整裁剪设置时，请换一个 `--prepared-data` 目录。发生中断时中间结果保留为 `.preparing`，不会冒充完整数据集。

## 安装

建议使用独立的 Python 3.12 环境，避免影响采集环境。在 `HRL_2` 目录中执行：

```bash
python3.12 -m venv .venv-diffusion
source .venv-diffusion/bin/activate
python -m pip install -r requirements-diffusion.txt
```

代码、数据和日志均使用仓库位置或 `~/Desktop` 定位，不依赖最初的 Codex 工作目录。整个仓库移到桌面后仍可使用；`data/` 一起移动就能复用已经处理好的数据。

## 数据处理与训练

只处理数据，不启动训练：

```bash
python prepare_diffusion_data.py
```

如果想显式指定输入和输出：

```bash
python prepare_diffusion_data.py \
  --data ~/Desktop/cube_placement_v2 \
  --output data/cube_placement_v2_diffusion \
  --trim-first 15 --trim-last 15
```

Mac 上开始正式训练：

```bash
python train_diffusion.py --device mps --batch-size 8
```

训练脚本会自动调用相同的预处理逻辑；如果数据已经处理好，会直接复用。也可以把处理后的目录作为 `--data`，不会再次裁剪：

```bash
python train_diffusion.py \
  --data data/cube_placement_v2_diffusion \
  --device mps --batch-size 8
```

CUDA 机器可用 `--device cuda --batch-size 32`。自动选择设备用 `--device auto`。如需改训练步数，可加 `--steps 30000`；其他设置在 `diffusion_config.json` 中。

输出默认写入 `runs/diffusion_cube/`：

- `training_metadata.json`：数据哈希、裁剪记录、训练/验证 episode、版本、有效设置。
- `metrics.jsonl`：逐步训练 loss、学习率、定期验证 loss。
- `checkpoints/000001/` 等：模型、配置、输入/输出处理器、optimizer/scheduler/RNG 状态。实际编号对应保存时的训练步数。
- `best.json`：验证 loss 最低的 checkpoint 路径；`latest.json`：最近保存的 checkpoint 路径。

恢复自己保存的 checkpoint 时，训练设置（包括 batch size）自动从 checkpoint 读取；提供原 run 目录，例如：

```bash
python train_diffusion.py --device mps \
  --output runs/diffusion_cube \
  --resume runs/diffusion_cube/checkpoints/001000 \
  --stop-after 3000
```

30,000 步是可调整的训练预算，不是作业要求。分阶段训练可以在启动时加 `--stop-after 1000`：到全局第 1,000 步验证并保存后退出，后续从该 checkpoint 接着训练到 3,000 步。`--stop-after` 不修改学习率计划；续训不要修改 `--steps`。1,000 步只是第一次观察点，不保证已学会任务。

新启动的脚本收到 Ctrl+C 或 SIGTERM 后，会在当前更新完成后验证并保存。强制终止（SIGKILL）、断电和运行错误不在此保障范围内。已经运行中的旧进程不会自动加载这些代码修改。

`--resume` 会读取本脚本保存的训练状态文件，请只使用自己的 checkpoint。验证 loss 用于观察训练过程和选模型，不代表实机成功率。

## 模型与动作

| 项目 | 默认值 |
|---|---|
| 输入 | wrist RGB + 6 维关节状态 |
| 图像 | 原始 480×640，模型内缩放至 240×320，保留完整视野 |
| 观察历史 | 2 帧，间隔 1/30 秒 |
| 编码器 | ResNet18，从零训练，GroupNorm |
| 动作网络 | LeRobot conditional 1D U-Net，通道 128/256/512 |
| diffusion | DDIM，训练噪声时间步 100，推理去噪 10 步，epsilon prediction |
| 预测 horizon | 16 个位置：相对当前帧 `[-1, 0, …, 14]` |
| 默认执行长度 | 每次预测后执行 8 步，再规划 |
| 归一化 | state/action 按训练集 min/max 映射至 [-1,1]；RGB 用训练集 mean/std |
| 数据增强 | 仅训练时轻度 brightness/contrast/saturation/hue；不翻转、不裁掉视野 |
| 优化 | Adam，lr=1e-4，warmup + cosine，gradient clipping |

action 是 **6 个绝对目标关节位置**，顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。当前数据按前 5 关节使用角度、夹爪使用 0–100 的 LeRobot 表示处理；部署显式设置 `use_degrees=true`，需与采集保持一致，不是末端位姿或 delta action。

部署默认限制单次目标变化 `max_relative_target=10`，单位为上述关节单位。返回的实际发送目标会另存到 `applied_action`。该限制对两种比较配置保持相同。

## 作业所需的两种配置比较

实验问题：在相同数据、训练权重和去噪步数下，每次执行 **4 步**与 **8 步**如何影响成功率、耗时和失败阶段？训练一次即可；部署时改变 `n_action_steps`，不需要训练第二个模型。

训练完成后，用选定的 checkpoint 预先固定 25 对初始场景和评测顺序：

```bash
python evaluate_diffusion.py plan \
  --checkpoint runs/diffusion_cube/checkpoints/030000 \
  --output eval/cube_h4_h8 --budget 25 --timeout 60
```

如 `best.json` 指向其他步数，应改成那个 checkpoint。计划会固定权重哈希、n_max=25、alpha_pair=0.10，事先随机化场景顺序以及每个场景内的两种 policy 顺序。`schedule.csv` 的 scene_notes 可填写方块/托盘位置或场景照片编号。相同 config_id 必须复现相同初始场景；两种配置总共运行 50 次。

在机械臂所在电脑修改原来的 `config.json`：follower 串口、follower 标定文件和 wrist 相机编号。原文件中的 `/Users/azhuan/...` 标定路径需要换成该电脑的真实路径。保持采集时的相机布局、分辨率、关节单位和标定。

只查看部署配置，不连接机械臂：

```bash
python run_diffusion.py --plan eval/cube_h4_h8/plan.json --trial 0 --dry-run
```

实际执行第 0 次计划 trial：

```bash
python run_diffusion.py --plan eval/cube_h4_h8/plan.json --trial 0 --device mps
```

每次按 schedule 顺序增加 `--trial`，手动复位并复现场景后再运行下一次。每个 trial 目录只能创建一次，避免覆盖失败尝试。输入：

- `s` 后 Enter：完整成功，即方块进入托盘，夹爪松开并向上离开。
- `f 0.5 grasp_failed` 后 Enter：失败，保存部分分数及原因；部分分数只能为 0、0.25、0.5、0.75。
- `q` 后 Enter 或 Ctrl-C：停止，保留本次尝试记录；到达 timeout 也记录为未成功。

每次保存 `attempt.json`、`result.json`、`frames.jsonl` 和 `wrist.mp4`。逐帧日志包括观察时间戳、状态、预测动作、发送动作、推理延迟和控制周期。视频按名义 30 FPS 编码，实际耗时以 JSON 时间戳为准。同步 diffusion 在重规划时可能超时，因此报告实际控制频率和 deadline misses，不假设达到 30 Hz。完成时间由观察者输入成功时刻确定，包含人的反应延迟。

STEP 和 posterior violin 使用作业指定的官方实现；仅分析时需要额外安装：

```bash
python -m pip install -r requirements-eval.txt
python evaluate_diffusion.py report --plan eval/cube_h4_h8/plan.json
```

报告要求所有计划 trial 完成，保留最早 STEP 决策，最终成功率和耗时使用全部 trial。输出 `report.json`、`rollout_manifest.csv`、二元 `step_sequences.json`、Beta(1,1) posterior 的 `posterior_violin.png`，以及根据两种配置的 STEP 决策产生的 CLD 标签。90% Wilson 区间、成功耗时的中位数/IQR、失败原因和部分分数分开记录。匹配场景能提升比较公平性，也可能产生相关性；官方两样本检验不显式建模该相关性，应在报告中说明。

## 当前完成范围

已提供代码并完成数据预处理；MPS 正式训练已按要求在 2,000 步保存并停止（batch size 8，保留原 30,000 步学习率计划）。第 1,000 / 2,000 步验证 loss 分别为 0.06705 / 0.04364。模型在本地 `runs/diffusion_cube/checkpoints/002000`。训练曲线、逐步指标和训练元数据在 `results/diffusion_cube/`；元数据中的绝对路径记录训练时的位置，迁移后请指定本机数据目录。未继续运行测试，也未连接机械臂；实机成功率、课堂迁移表现和完整 50 次评测需要实际收集。代码中保留 `--smoke-test` 作为以后可选的软件检查入口，其两步模型不能用于实机评测。

Task 2 堆叠需要单独收集堆叠演示，再从头训练单任务模型；当前 cube placement 数据不能替代堆叠数据。提交时还需整理作业要求的说明、贡献声明，并披露 Codex 用于代码编写和数据处理。

参考：[Assignment 2](https://hands-on-robot-learning.github.io/assignments/assignment-2/)、[LeRobot DiffusionPolicy 源码](https://github.com/huggingface/lerobot/tree/v0.7.0/src/lerobot/policies/diffusion)、[LeRobot 实机模仿学习](https://huggingface.co/docs/lerobot/il_robots)、[官方 STEP](https://github.com/TRI-ML/sequentialized_barnard_tests)。
