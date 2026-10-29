[**English**](README.md) | [中文](README.zh.md)

# Piper LeRobot

LeRobot integration for AgileX Piper arms. It supports single-arm and bimanual
teleoperation, dataset recording, ACT/pi05 training, and real-robot deployment.

## Features

- `piper_follower` and `bi_piper_follower` robot configurations
- Bimanual leader/follower control with optional EMA smoothing
- Asynchronous video encoding during dataset recording
- CAN setup, camera setup, gripper calibration, and safe-disable utilities
- Local, RTC, and asynchronous policy inference

## Installation

Use a dedicated Conda environment and install this checkout once in editable
mode. After this, `lerobot` resolves to `src/lerobot` without setting
`PYTHONPATH` for every command.

```bash
conda create -n piper-lerobot python=3.10 -y
conda activate piper-lerobot

cd /path/to/piper_lerobot
python -m pip install -e .
python -c "import lerobot; print(lerobot.__file__)"
```

Install `pyAgxArm` in the same environment according to the AgileX Piper SDK
instructions.

## Record a bimanual dataset

```bash
# Prepare the four CAN interfaces and stable camera names.
bash utils/activate_all_can.sh
sudo bash utils/setup_camera_symlinks.sh

# Record and upload a dataset.
bash record_bimanual.sh <repo_id> <task> <num_episodes>
```

## Record a single-arm dataset

Prepare the matching follower and leader CAN interfaces, then run the recorder
for the left or right arm:

```bash
bash record_single_arm.sh left <repo_id> <task> <num_episodes>
bash record_single_arm.sh right <repo_id> <task> <num_episodes>
```

The single-arm recorder uses the matching wrist camera and the top camera,
saves locally under `dataset/<repo_id>/`, and does not upload to the Hub. Its
datasets contain one Piper arm (7 action/state values), so they are intended
for a single-arm policy and are not directly compatible with the current
14-value bimanual policy.

Recording keys: `Space` starts an episode, `Right Arrow` finishes it,
`Left Arrow` retries it, and `Esc` ends the session.

## Human-in-the-loop (HIL) data collection

A deployed policy drives the follower while the leader mirrors it; press
`Space` to take over with the leader, `Space` again to hand back. Every episode
is saved with a success/failure label and a per-frame control flag for
Recap-style training. The recorder lives in the
[`deploy_piper`](https://github.com/thusigs-eilab/deploy_piper/tree/hil) repo
(branch `hil`, checked out here as `deploy_piper_hil/`):

```bash
cd deploy_piper_hil
python -m deploy.server --config pi05_long_horizon_plate --checkpoint=<ckpt>   # env lerobot_pi05
python -m deploy.hil_record --config pi05_long_horizon_plate --task="..." \
  --teleop.type=piper_leader --teleop.id=left_leader --teleop.can_name=left_leader \
  --dataset.repo_id=<repo_id> --dataset.root=../dataset/<repo_id>              # env piper_sdk
```

See [HIL quick start](docs/hil.md) and the
[full reference](https://github.com/thusigs-eilab/deploy_piper/blob/hil/docs/hil_collection.md)
(options, keys, saved data, troubleshooting).

## Safety

Do not disable or power off a raised arm directly—it can fall. Use the soft
disable utility:

```bash
python utils/gentle_disable_arm.py
```

Connect high-bandwidth cameras to separate USB buses. Use
`python utils/capture_cameras.py` to verify the configured camera feeds.

## Utilities

- `activate_all_can.sh` — activate all configured Piper CAN interfaces
- `find_all_can_port.sh` — find connected CAN adapters
- `can_activate.sh` — activate one CAN interface
- `can_health.sh` — inspect CAN interface health
- `setup_camera_symlinks.sh` — install stable camera device names using
  `99-camera-symlinks.rules`
- `capture_cameras.py` — capture test frames from the cameras
- `gentle_disable_arm.py` — lower and safely disable Piper arms
- `gripper_zero_calibration.py` — calibrate a gripper's closed position
- `push_dataset.py` — upload a local dataset to Hugging Face Hub

## Documentation

- [Environment and hardware setup](docs/setup.md)
- [Teleoperation](docs/teleop.md)
- [Dataset recording](docs/recording.md)
- [HIL data collection](docs/hil.md) — policy-driven recording with leader takeover ([full reference](https://github.com/thusigs-eilab/deploy_piper/blob/hil/docs/hil_collection.md))
- [Hugging Face upload](docs/huggingface.md)
- [Training](docs/training.md)
- [Inference and deployment](docs/inference.md)
- [Arm utilities](docs/arm_utils.md)
