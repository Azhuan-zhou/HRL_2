# SO-101 data collection and evaluation

LeRobot 0.6.2 wrapper for the SO-101 arm. `config.json` is the only place that stores the serial ports, calibration files, and wrist camera. The scripts collect teleoperated demonstrations, train ACT or diffusion policies, and run a checkpoint on the follower. Nothing is uploaded to the Hugging Face Hub.

Working directory:

```bash
cd /Users/jiafeng/HRL_2
conda activate lerobot
```

Use this conda environment. The base conda Python hits an OpenMP error when it imports LeRobot. Check the install with:

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('lerobot'))"
lerobot-record --help
```

## Hardware

| Arm | Port |
| --- | --- |
| Leader | `/dev/tty.usbmodem5B421343381` |
| Follower | `/dev/tty.usbmodem5B421342761` |

Calibration files are the absolute paths in `config.json`. The script uses the filename (without `.json`) as the robot or teleop id and passes that directory through. It does not copy or rewrite calibration files.

The wrist camera is OpenCV index 1, 640×480, 30 fps. Confirm it before recording:

```bash
lerobot-find-cameras opencv
python collect.py preview
```

`preview` opens the camera and does not connect the arms. On macOS, allow the terminal or Cursor under System Settings → Privacy & Security → Camera. Press `q` to close the preview before recording, so the camera is free.

```bash
python collect.py check
python collect.py teleop --dry-run
python collect.py teleop
```

`check` only looks at the software, the serial-port device files, and the calibration files. `teleop` moves the arms. Start with the leader and follower in similar poses and the workspace clear. End teleop with Ctrl+C. Support the arm: disconnecting can release torque.

If a port name changes after replugging, update `config.json`. A resume of an existing dataset also requires the saved camera settings to match the current file.

## Record

Manual recording is the default. Run it in a terminal that has keyboard focus.

```bash
python collect.py record --name task1_home --episodes 2 --dry-run
python collect.py record --name task1_home --episodes 2
python collect.py inspect --name task1_home
```

1. While waiting, teleoperate the reset. That motion is not recorded. Press **Enter** to start an episode.
2. Press **Enter** again to end it. There is no time limit.
3. Type **s** then Enter to save the episode, or **f** then Enter to discard it. A discarded episode is not written and does not take an index.
4. Press **q** then Enter while waiting to stop.

`--episodes` is a target, not a hard stop, so recording can continue past it. Ctrl+C does not reliably save the episode in progress. `--timed` switches back to LeRobot's timed episodes (`episode_time_s` 40, `reset_time_s` 20 in `config.json`).

An existing directory is not overwritten. Append only when `meta/info.json` is present:

```bash
python collect.py record --name task1_home --episodes 20 --resume
```

`inspect` prints episode count, frame count, and features. Also play the files under `data/<name>/videos/`.

Datasets kept for the assignment:

| Directory | Episodes | Task |
| --- | --- | --- |
| `data/task1_home` | 54 | Pick up the cube and place it in the tray |
| `data/cube_stack_80_10_13` | 103 | Stack blocks |

Both are LeRobot dataset v3.0, 30 fps, wrist image plus six joint positions. The action is the same six values: five joint positions in degrees and a gripper command from 0 to 100, larger meaning more open.

## Train

`train_trim.py` calls `lerobot-train` and ignores idle frames at the start of each episode plus a fixed number of frames at the end. The dataset files are not modified. Normalization statistics still include the dropped frames. `output_dir` must not already exist.

```bash
python train_trim.py --drop_first=auto --drop_last=20 --config_path=configs/bench_v30m.json
```

`configs/bench_v30m.json` is a 30-step diffusion smoke test on `data/cube_stack_80_10_13`. The stacking checkpoints in `models/` were trained on that dataset: `v29_stack80_10_13_imnet` is 30,000 steps, and `v29_stack80_10_13_imnet_020000` is the 20,000-step checkpoint. Diffusion, ImageNet ResNet-18, observation window 2, prediction horizon 64, execution horizon 30. `configs/v18.json` through `configs/v21.json` point at datasets that are no longer in `data/`.

A checkpoint saved on the training machine can contain extra tactile fields and `frame_stride`. After `model.safetensors` is fully copied (about 1 GB), strip those keys so LeRobot 0.6.2 can load it:

```bash
python fix_dell_model.py models/<model>
```

Original files are kept as `*.dell.json`. Weights are not changed. `use_tactile` must be false and `frame_stride` must be 1.

## Evaluate

`eval.py` is `lerobot-rollout` with two additions. After connect it glides for 3 seconds to a named start pose, shows the wrist view, and waits for Enter in the terminal. The duration timer starts after Enter. A failed connect is retried.

```bash
python eval.py --policy.path=models/v29_stack80_10_13_imnet \
  --start=stack --duration=240 --policy.num_inference_steps=5 \
  --robot.max_relative_target=null
```

The 20,000-step checkpoint is `models/v29_stack80_10_13_imnet_020000`. Same flags.

- `--start=stack` is the stacking start pose. `--start=home` reads `configs/home_pose.json`.
- `--policy.num_inference_steps=5` shortens the diffusion replan. The saved checkpoint uses 10. The weights do not change.
- `--robot.max_relative_target=null` turns off the 5-degree-per-tick limit in `configs/eval_rollout.json`.
- `--grip_offset=8` commands the gripper further closed while the policy command is below 20, and removes that offset once the command opens by 3. Default is 0, so the learned command is what the arm receives.

Ctrl+C stops early. The arm returns to the start pose at the end of a normal run.

To keep a video of every attempt, add a rollout dataset whose repo id starts with `rollout_`:

```bash
python eval.py --policy.path=models/v29_stack80_10_13_imnet \
  --start=stack --duration=240 --policy.num_inference_steps=5 \
  --robot.max_relative_target=null \
  --dataset.repo_id=local/rollout_stack --dataset.root=data/rollout_stack \
  --dataset.num_episodes=25 --dataset.single_task="Stack the blocks."
```

## Layout

```text
HRL_2/
  collect.py          record and teleop
  eval.py             rollout from a start pose
  train_trim.py       lerobot-train with trimmed episodes
  fix_dell_model.py   strip extra keys from a copied checkpoint
  pick_home.py        save configs/home_pose.json
  manual_record.py    Enter / s / f recording used by collect.py
  config.json         ports, calibration, wrist camera
  configs/            train and rollout settings
  data/               task1_home, cube_stack_80_10_13
  models/             v29 20k and 30k stacking checkpoints
```

`data/`, `models/`, `runs/`, and `logs/` are gitignored. `logs/` is rewritten on each eval launch.

## References

- https://huggingface.co/docs/lerobot/so101
- https://huggingface.co/docs/lerobot/il_robots
- https://huggingface.co/docs/lerobot/installation
