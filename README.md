# SO-101

LeRobot 0.6.2 scripts for the SO-101 arm. Ports, calibration files, and the wrist camera are in `config.json`. Data is not uploaded.

```bash
conda activate lerobot
```

## Record

```bash
python collect.py preview
python collect.py record --name <name> --episodes 2
python collect.py record --name <name> --episodes 20 --resume
python collect.py inspect --name <name>
```

Press Enter to start an episode, Enter again to end it, then `s` to save or `f` to discard. `q` stops while waiting. An existing directory is not overwritten.

| Directory | Episodes | Task |
| --- | --- | --- |
| `data/task1_home` | 54 | Place the cube in the tray |
| `data/cube_stack_80_10_13` | 103 | Stack blocks |

## Train

```bash
python train_trim.py --drop_first=auto --drop_last=20 --config_path=configs/bench_v30m.json
python fix_dell_model.py models/<model>
```

`configs/v18.json` and `configs/v19.json` are diffusion and ACT for placement. `configs/v20.json` and `configs/v21.json` are diffusion and ACT for stacking. `configs/bench_v30m.json` is a short diffusion run on `data/cube_stack_80_10_13`.

Stacking weights trained on that dataset are `models/v29_stack80_10_13_imnet` (30,000 steps) and `models/v29_stack80_10_13_imnet_020000` (20,000 steps).

## Evaluate

```bash
python eval.py --policy.path=models/v29_stack80_10_13_imnet \
  --start=stack --duration=240 --policy.num_inference_steps=5 \
  --robot.max_relative_target=null
```

`--start=home` uses `configs/home_pose.json`. `--grip_offset=8` closes the gripper further while the policy command stays below 20. The default is 0.

`data/`, `models/`, `runs/`, and `logs/` are gitignored.
