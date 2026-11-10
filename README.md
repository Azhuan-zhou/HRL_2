# SO-101 数据采集（HRL_2）

使用已有 LeRobot calibration，通过 leader 遥操作 follower，保存标准 LeRobot 图像、关节状态、动作和任务信息。此包是官方 CLI 的配置封装，不实现新的电机控制协议。不自动上传数据，不执行电机配置或重新标定。

## 1. 放置文件并激活环境

将 `collect.py`、`manual_record.py`、`config.json`、`README.md` 放在：

```bash
cd /Users/azhuan/Documents/JHU/course/HRL/HRL_2
```

激活你之前成功完成 calibration 的 Python/Conda 环境（使用你自己的环境名称）。优先保留该环境及其 LeRobot 版本，不要为了运行脚本盲目升级。

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('lerobot'))"
command -v python
command -v lerobot-record
lerobot-record --help
```

若尚未安装，在独立的 Python 3.12 环境中按官方安装指南安装；当前官方安装方式示例：

```bash
python -m pip install 'lerobot[core_scripts,feetech]'
brew install ffmpeg
```

这里的安装命令跟随官方当前版本；本包没有锁定你的未提供的已有版本。若当前安装没有 `lerobot-record` 命令，不要直接开始录制，应先完成 LeRobot 安装。保留实际版本：

```bash
python -m pip freeze > environment-lock.txt
```

## 2. 确认标定文件

已预填串口：

| 机械臂 | 串口 |
| --- | --- |
| Leader | `/dev/tty.usbmodem5B421343381` |
| Follower | `/dev/tty.usbmodem5B421342761` |

脚本从当前 LeRobot 的 `HF_LEROBOT_CALIBRATION` 路径寻找对应 SO-101 标定文件；每类只有一个文件时自动选用。需要确认该文件确实对应当前实物。存在多个或文件保存在其他位置时，在 `config.json` 填写绝对路径，例如：

```json
"leader_calibration_file": "/Users/azhuan/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json",
"follower_calibration_file": "/Users/azhuan/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json"
```

这些是路径示例，必须换成实际文件。可用以下命令查看当前环境默认目录：

```bash
python -c "from lerobot.utils.constants import HF_LEROBOT_CALIBRATION; print(HF_LEROBOT_CALIBRATION)"
```

脚本以标定文件的文件名（去除 `.json`）作为 robot/teleop ID，并显式传递标定目录。不会复制或修改标定文件。如果 LeRobot 仍提示重新校准，先退出并核对文件与机械臂，不要随意覆盖标定。

## 3. 确认相机（必须完成）

默认配置仅包含一台占位相机 `wrist: 0`，编号 0 可能是 Mac 内置相机，不能假定是腕部相机。

```bash
lerobot-find-cameras opencv
```

根据检测结果修改 `config.json` 的 `index_or_path`。如果有外部相机，可将整个 cameras 字段改成下面形式（编号仍需实测）：

```json
"cameras": {
  "wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  "front": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
}
```

```bash
python collect.py preview
```

逐个窗口检查：wrist 确实是腕部相机；front 是外部相机；画面清晰且能看到操作区域。按 `q` 关闭预览后再采集，避免占用摄像头。预览不连接机械臂。

macOS：系统设置 → 隐私与安全性 → 相机，允许运行 Python 的 Terminal/iTerm/IDE 访问。录制快捷键不生效时，也检查辅助功能及输入监控权限，重启终端。相机名称、安装位置、分辨率和 fps 在同一数据集内保持一致。

## 4. 检查配置并试遥操作

```bash
python collect.py check
python collect.py teleop --dry-run
python collect.py teleop
```

`check` 仅检查软件、串口文件和标定文件，不证明设备已通信成功。`teleop` 会实际驱动机械臂。开始前让 leader/follower 处于近似一致的姿态，清空碰撞范围；先缓慢移动检查关节映射。结束遥操作用 Ctrl+C。支持好机械臂，断开连接后可能解除力矩。

## 5. 先采集 2 条测试数据

```bash
python collect.py record --name cube_test --episodes 2 --dry-run
python collect.py record --name cube_test --episodes 2
python collect.py inspect --name cube_test
```

默认使用手动录制（30 FPS），需要在交互式终端中运行，并将键盘焦点留在该终端：

1. 等待时可遥操作复位，不记录数据；按 **Enter** 开始一条。
2. 再按 **Enter** 结束本条，没有时长上限（超过 50 秒也仍需 Enter）。
3. 输入 **s + Enter** 标记成功并保存，或 **f + Enter** 标记失败并丢弃。失败片段通过 LeRobot 的清理接口丢弃，不写入数据集或结果标签，不占用 episode 编号，也不计入目标条数。选择结果及保存期间暂停遥操作跟随。
4. 保存完成后回到等待，按 Enter 开始下一条；等待期间 **q + Enter** 退出。

`--episodes 50` 是本次目标提示，不是上限；达到 50 条后仍可继续 Enter 录制。录制期间也可输入 q + Enter，选择当前条的成功/失败后，成功则保存、失败则丢弃，然后退出。正常退出请使用 q，等待编码及保存完成。Ctrl+C / 进程中断不能保证当前未完成片段保存。

手动模式忽略 `episode_time_s` 和 `reset_time_s` 的计时。恢复原始自动计时模式可加 `--timed`，此时条数是硬上限，方向键及 Esc 使用 LeRobot 原生行为。

每条结果额外写入 `data/<name>/meta/manual_outcomes.jsonl`，以 `episode_index` 对应 LeRobot 数据。每次保存先写 `status: "pending"`，保存成功再写 `status: "saved"`；读取时只选择 saved 行。`success` 为布尔值，新录制的 `outcome` 为 success。仅有 pending 表示保存过程被中断，需核对数据后再使用。标签是额外元数据，不会自动成为 LeRobot 训练特征；现在只保存成功片段。已有数据和标签不修改；旧版本已保存的 failure 仍然存在，训练旧数据时需要据此筛选。

手动适配器已核对本机 LeRobot 源码的录制循环及 has_pending_frames 接口，启动前检查基础函数签名；自动测试不代替实机验证。先做少量试录。

`inspect` 显示元数据中的 episode 数、帧数、特征等；它不是完整数据质量检查。请实际播放 `data/cube_test/videos/` 下的视频，并检查数据中有 `observation.state`、`action` 与预期的 `observation.images.*`。不同 LeRobot 版本可能将多条 episode 合并在一个视频/Parquet 文件中。

## 6. 正式录制和追加

测试确认正常后，新建正式数据集：

```bash
python collect.py record --name cube_placement --episodes 50
```

之后追加，设置本次目标为 20 条（手动模式可继续超过目标）：

```bash
python collect.py record --name cube_placement --episodes 20 --resume
python collect.py inspect --name cube_placement
```

同名目录已存在时，脚本会拒绝非 resume 录制，不会自动覆盖或删除。只有完整保存且存在 `meta/info.json` 的数据集才能追加。损坏的未完成会话不能保证能恢复，不要通过删除元数据强行续录。

初始数据建议固定托盘、有限范围改变方块位置，保持动作一致；每次记录完整的抓取放置过程。50 条是实践起点，不是保证成功的数量。先检查少量数据的质量，再扩大规模。

## 7. 输出位置

```text
HRL_2/
  collect.py
  config.json
  README.md
  data/
    cube_test/
    cube_placement/
      meta/
      data/
      videos/
  logs/
    cube_placement_config.json
    <timestamp>_command.json
```

默认 `repo_id=local/<name>`，`push_to_hub=false`，不需要 Hugging Face 登录。对支持自动时间戳的新版本，脚本使用 `no_stamp=true` 保持名称稳定。日志包含实际命令、版本及录制配置；终端输出由 LeRobot 直接显示，不另存文本日志，以保持交互式键盘控制。后续训练时保留整个数据目录。

## 常见问题

- 串口不存在：重新插拔后编号可能改变，更新 config.json。
- 标定文件不唯一：显式设置两个 `*_calibration_file`。
- 相机黑屏/打不开：检查编号、权限、USB 带宽，并关闭其他使用相机的程序。
- 视频编码或依赖报错：在当前环境运行 `lerobot-record --help`，检查 FFmpeg/视频编码依赖；先完成 2 条试录。
- 循环达不到 30 FPS：检查终端耗时提示；减少相机数量或分辨率后用新数据集重新测试，不能只按声明的 fps 假定采样真实达标。
- 追加配置不一致：恢复首次采集配置，或使用新名字建立新数据集；不要直接混合不同相机布局。
- 没有实机测试声明：此包在独立环境检查 Python 语法和命令构造/保护逻辑；未连接你的 Mac、相机或机械臂。必须完成以上试录验证。

## 官方参考

- https://huggingface.co/docs/lerobot/so101
- https://huggingface.co/docs/lerobot/il_robots
- https://huggingface.co/docs/lerobot/installation
