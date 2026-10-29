# HIL data collection on Piper (built-in leader/follower takeover)

Human-in-the-loop (HIL) data collection: the policy drives the follower arm,
the leader arm mirrors it, and the operator takes over with the leader when the
policy goes wrong, then hands back. Every episode (success or failure) is saved
with an outcome label and a per-frame "who was in control" flag, the data a
π\*0.6 / Recap-style loop needs (value function on outcomes + advantage-
conditioned policy).

Code: `deploy_piper_hil/deploy/hil_record.py` (`deploy_piper` repo, branch `hil`).
Detailed reference: `deploy_piper_hil/docs/hil_collection.md`.

> `record_hil.sh` still runs the older `deploy_piper/deploy/hil_record.py`
> (TEACH-button takeover). The built-in takeover described here lives in
> `deploy_piper_hil/`.

## 1. How the takeover works

The Piper firmware has a built-in leader-follower linkage, switched over CAN
with the linkage config frame (pyAgxArm):

| Call | Code | Arm behaviour |
|---|---|---|
| `robot.set_leader_mode()` | 0xFA | teaching input arm: gravity-compensated zero-force drag, streams its joint (0x155-0x157) and gripper (0x159) **control frames** |
| `robot.set_follower_mode()` | 0xFC | motion output arm: accepts `move_j` / gripper commands, sends normal feedback |

`--takeover=builtin` (default) uses this, so **no TEACH button is needed**:

```
           SPACE                       leader frames arrive              SPACE
POLICY ──────────────► AWAIT (0xFA sent) ───────────────────► HUMAN ──────────────► POLICY
  leader mirrors         follower holds,                     follower tracks        leader 0xFC,
  the follower           not recorded                        leader frames          re-enabled, mirrors;
                         (timeout 2 s → back to POLICY)                              new policy stream
```

- The follower only follows the leader once **fresh** joint *and* gripper
  frames have arrived after 0xFA; cached frames from an earlier takeover are
  never used.
- If the leader is > 0.1 rad (or > 10 mm gripper) from the follower at
  takeover, the follower ramps to it over 1 s (`ramp` frames) instead of
  snapping.
- If no frames arrive within `--leader_switch_timeout_s` (2 s), the leader is
  put back to 0xFC and the policy continues. Then use `--takeover=teach_button`.
- If the episode ends or the script errors while the leader is in 0xFA, it is
  switched back to 0xFC.
- With `builtin`, `--teleop.ema_alpha` is not applied (the follower tracks the
  raw leader control frames).

Fallback `--takeover=teach_button`: SPACE (hold, leader gripper limp) → press
the leader's TEACH button → SPACE (leader drives) → SPACE (hand back).

## 2. Setup (once per session)

```bash
cd /data/wanshan/piper_lerobot
bash utils/activate_all_can.sh            # brings up left/right leader/follower at 1 Mbit/s (sudo)
ip -br link show type can                 # the arms you use must be UP
ls -l /dev/r_wrist /dev/top               # cameras present
```

Policy server (GPU machine, conda env `lerobot_pi05`):

```bash
cd deploy_piper_hil
python -m deploy.server --config pi05_long_horizon_plate \
  --checkpoint=../model_ckpts/pi05_long_horizon_plate/005000
curl --noproxy '*' http://127.0.0.1:8082/info
```

## 3. Record (desktop terminal, conda env `piper_sdk`)

Single right arm:

```bash
cd /data/wanshan/piper_lerobot/deploy_piper_hil
python -m deploy.hil_record --config pi05_long_horizon_plate \
  --task="Open the top drawer, put the strawberry inside, and close it. ..." \
  --teleop.type=piper_leader --teleop.id=right_leader --teleop.can_name=right_leader \
  --dataset.repo_id=samithva/plate_hil --dataset.root=../dataset/samithva/plate_hil \
  --dataset.num_episodes=10 --notes="5k, nominal scene"
```

Bimanual: `--config pi05 --teleop.type=bi_piper_leader --teleop.id=bi_piper_leader`.

`--dataset.root` must not exist yet; add `--dataset.resume=true` to append.

| Option | Default | Meaning |
|---|---|---|
| `--takeover` | `builtin` | `builtin` (0xFA/0xFC from code) or `teach_button` |
| `--leader_switch_timeout_s` | `2.0` | wait for leader frames after 0xFA; bound on re-enable after 0xFC |
| `--intervention_key` | `space` | key toggling takeover |
| `--takeover_ramp_s` / `--takeover_warn_rad` | `1.0` / `0.1` | ramp instead of snap if leader and follower disagree |
| `--mirror_gripper` / `--leader_gripper_force_n` | `true` / `1.0` | leader gripper mirrors the follower at low force |

### Keys

| Key | Effect |
|---|---|
| SPACE (ready prompt) | start the episode |
| SPACE (in episode) | take over with the leader ↔ hand back to the policy |
| → | end episode, then **S** = success / **F** = failure (failures are kept) |
| ← | re-record (discard this episode) |
| ESC | label + save the current episode, end the session |

Reaching `--dataset.episode_time_s` (300 s) saves the episode as `timeout`.

### Expected log on takeover

```
>>> switching the leader to teaching role (0xFA)...
>>> HUMAN INTERVENTION #1: follower follows the leader (leader-follower joint err 0.012 rad, gripper err 2 mm) — drag the leader, space to hand back
<<< policy resumed; leader back to mirroring
```

## 4. What is saved

LeRobotDataset v3 at `--dataset.root`, plus a sidecar:

| Item | Content |
|---|---|
| `observation.images.*`, `observation.state` | cameras, follower state |
| `action` | command actually sent (policy or leader) |
| `intervention` | 1.0 in `human`/`ramp` frames, else 0.0 |
| `control_mode` | 0 policy, 1 ramp, 2 human, 3 hold (switching into 0xFA is not recorded) |
| `meta/hil_episodes.jsonl` | per episode: `success`, `outcome`, `autonomous`, `interventions` (frame ranges), `frames_by_mode`, `paused_s`, takeover error, `checkpoint`, task, notes |

Review videos with interventions highlighted:

```bash
python -m deploy.hil_video ../dataset/samithva/plate_hil --all
```

## 5. Verification status (2026-10-28)

| Check | Status |
|---|---|
| State machine (0xFA → frames → human → 0xFC; timeout fallback; end-of-episode restore) | tested with simulated arms |
| Leader in 0xFA streams control frames readable by `get_leader_joint_angles()` | seen on `left_leader` (~170 Hz) |
| 0xFA → 0xFC live from code, re-enable without power cycle | works on `left_leader` (S-V1.8-8) |
| 0xFC → 0xFA live from code | **not yet verified** |
| Full HIL episode with the policy | **not yet run** |

First real test: record one episode (`--dataset.num_episodes=1` into a throwaway
root), press SPACE, and check that `HUMAN INTERVENTION #1` appears within a
fraction of a second. A `no leader control frames ... after 0xFA` error means
this firmware cannot switch to leader live → use `--takeover=teach_button`.
Keep a hand on the leader the first time.

## 6. Troubleshooting

### `RTNETLINK answers: Broken pipe` from `activate_all_can.sh`

The script brings up all four ports in order and stops at the first failure
(`set -e`), so one bad adapter blocks the rest (arms already `UP` stay usable).
Check the kernel log:

```bash
journalctl -k -n 20 | grep -E "gs_usb|1-5"
# gs_usb 1-5.2:1.0 left_follower: failed to set bittiming: -EPIPE
```

`-EPIPE` / `Error -32` means the USB-CAN adapter (candleLight, `gs_usb`) itself
rejects every request: its firmware is hung. Typical cause: it kept
transmitting with no node acknowledging (arm off or CAN cable loose), visible as
`TX packets 0, errors N` in `ip -s link show <can>`. Re-running the script does
not help; reset the adapter:

1. Check the arm is powered and the CAN cable is seated.
2. Unplug the adapter for ~5 s and replug (most reliable), or re-enumerate it:
   ```bash
   echo 0 | sudo tee /sys/bus/usb/devices/<port>/authorized   # e.g. 1-5.2
   sleep 2
   echo 1 | sudo tee /sys/bus/usb/devices/<port>/authorized
   ```
   (driver unbind/bind does not reset the adapter firmware.)
3. `bash utils/activate_all_can.sh` again. Still `-EPIPE` right after a fresh
   plug → try another adapter.

Find which USB port an interface is on:
`readlink -f /sys/class/net/<can>/device`.

### Leader stays in drag mode / "there is force" after disabling

In 0xFA the firmware runs gravity-compensated drag and does **not** send normal
status/enable feedback, so `disable()` appears to succeed while the arm keeps
its force. To make it limp: switch to 0xFC first (feedback returns), confirm
fresh joint feedback, hold the current pose with MIT, ramp kp → 0, then
`disable(255)`. Only use a hold pose from fresh feedback: holding a default
`0.0` pose pulls the arm toward zero. The arm then stays in follower role;
setting 0xFA again re-enters drag mode.

### Leader limp after hand-back (`teach_button` only)

Press its TEACH button once.


```
conda activate piper_sdk
  cd /data/wanshan/piper_lerobot/deploy_piper_hil
  python -m deploy.hil_record --config pi05_long_horizon_plate \
    --task="Open the top drawer, put the strawberry inside, and close it. Then put the pink plate
  on top of the cabinet and put the pear on the pink plate." \
    --teleop.type=piper_leader --teleop.id=left_leader --teleop.can_name=left_leader \
    --dataset.repo_id=samithva/plate_hil --dataset.root=../dataset/samithva/plate_hil \
    --dataset.resume=true --dataset.num_episodes=1 --notes="5k, right follower, left leader"

```