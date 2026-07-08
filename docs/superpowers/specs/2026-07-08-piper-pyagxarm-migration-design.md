# Migrate piper_lerobot from `piper_sdk` to `pyAgxArm`

**Date:** 2026-07-08
**Status:** Approved design

## Goal

Replace the arm-control SDK across the whole repo: swap AgileX's legacy
`piper_sdk` (`C_PiperInterface_V2`) for the new `pyAgxArm` SDK
(`AgxArmFactory` / `create_agx_arm_config`), and remove `piper_sdk` entirely.

## Guiding principle — faithful behavioral port

Every consumer (recorded datasets, `piper_leader` teleop, `piper_follower`,
`bi_piper_follower`, `deploy/client.py`) keeps working with **zero changes to
its own code and zero change to the numeric units on the wire.** Only the
internals that talk to the arm change. This is a like-for-like SDK swap, not a
redesign of the control stack.

## Target hardware / config

- **Model:** standard AgileX Piper 6-DOF → `ArmModel.PIPER`
- **Firmware:** `S-V1.8-8` or newer → `PiperFW.V188` (affects MIT `t_ff` encoding)
- **Transport (Linux):** `interface="socketcan"`, `channel=<can_name>` (e.g. `can0`), `bitrate=1_000_000`
- **Env:** the `lerobot` conda env (`/home/embodied/miniconda3/envs/lerobot/bin/python`, 3.12.13).
  Already has `python-can 4.6.1` (> 3.3.4 required). `pyAgxArm` to be installed there.

## The load-bearing invariant (correctness anchor)

`PiperMotorsBus.read()` MUST keep returning **raw firmware units**:

- joints: `0.001°` (integer scale of the legacy `joint_state.joint_i`)
- gripper: raw (~µm scale of the legacy `grippers_angle`)

Because downstream code depends on that exact scale:

- `piper_follower.get_observation()` records `read()` values **directly** as
  `observation.state` → the exact numbers are baked into every recorded dataset.
- `piper_leader.get_action()` divides joints by `57324.840764` and gripper by
  `1_000_000` to produce rad / m.

`pyAgxArm` returns **radians** (`get_joint_angles().msg`) and **meters**
(`get_gripper_status().msg.value`). Conversion preserves round-trip identity
because `57324.840764 == 1000 * 180 / π` is exactly the raw-0.001°-per-radian
factor:

- `read()` joint raw  = `angle_rad * 57324.840764`
- `read()` gripper raw = `value_m * 1_000_000`
- `write()` already receives rad / m, so it maps straight to
  `move_j(joints_rad)` + `move_gripper_m(value_m)` — no scaling, slightly higher
  precision than the old integer-rounded `JointCtrl`.

**Any migration that changes the numbers `read()` emits is wrong.**

## Component design — `PiperMotorsBus` (core rewrite)

`src/lerobot/motors/piper/piper.py`. Keeps its public surface identical
(`connect`, `read`, `write`, `move_to_home`, `apply_calibration`,
`apply_calibration_master`, `gentle_disable`, `safe_disconnect`,
`is_connected`, `is_calibrated`, the `motor_*` properties, etc.).

Internally wraps **two** pyAgxArm objects behind the unchanged 7-DOF interface:

- `self.robot` = `AgxArmFactory.create_arm(create_agx_arm_config(
    robot=ArmModel.PIPER, firmeware_version=PiperFW.V188,
    interface="socketcan", channel=config.can_name))`
- `self.gripper` = `self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)`
  — created **before** `robot.connect()` (per SDK docs: init effector before
  connect; can only be called once).

`connect()` calls `self.robot.connect()` (starts the shared read thread that
feeds both arm and gripper feedback).

### Method mapping (all faithful)

| Legacy `piper_sdk` call | New `pyAgxArm` call |
|---|---|
| `EnablePiper()` (loop until all enabled) | `robot.enable(255)` (loop `while not robot.enable(): ...`) |
| `DisableArm(7)` | `robot.disable(255)` |
| `_all_motors_enabled()` via `GetArmLowSpdInfoMsgs().motor_i.foc_status.driver_enable_status` | `robot.get_joint_enable_status(255)` (already aggregates all 6 with `all()`) |
| `GetArmJointMsgs().Hz > 0` (feedback live) | `robot.get_joint_angles().hz > 0` |
| `MotionCtrl_2(0x01,0x01,speed,0x00)` + `JointCtrl(...)` | `robot.set_speed_percent(speed)` + `robot.move_j([rad×6])` |
| `GripperCtrl(range,1000,0x01,0)` | `gripper.move_gripper_m(value_m, force=1.0)` |
| `GetArmJointMsgs().joint_state.joint_i` | `robot.get_joint_angles().msg[i-1]` (rad → raw ×57324.840764) |
| `GetArmGripperMsgs().gripper_state.grippers_angle` | `gripper.get_gripper_status().msg.value` (m → raw ×1e6) |
| `GetArmStatus().arm_status.ctrl_mode` (diag; `0x01`=CAN ctrl) | `robot.get_arm_status().msg.ctrl_mode` (`0x01`=CAN_CTRL) |
| `gentle_disable` MIT ramp: `MotionCtrl_2(0x01,0x04,0,0xAD)` + `JointMitCtrl(j,hold,0,kp,kd,0)` | `robot.move_mit(joint_index=j, p_des=hold, v_des=0, kp=kp, kd=kd, t_ff=0)` |
| `DisablePiper()` loop + `MotionCtrl_1(0x02,0,0)` (restore mode) | `robot.disable(255)` loop; mode auto-restored on next `move_j` (auto-motion-mode is on by default) |

### Preserved as-is (re-sourced onto new getters, logic unchanged)

The hard-won cold-start readiness architecture is kept verbatim in structure:

- `is_ready()` = all drivers enabled **and** joint feedback live
- `_wait_until_ready(timeout, stable_needed)` — stability window that resets on
  any dropout
- `connect(enable, timeout)` — enable-and-verify with the same timeout bounding
- `move_to_home(timeout, tol_rad, speed)` — **polled and verified**: repeat
  `move_j([0]*6)` until joint feedback shows every joint within `tol_rad` or
  timeout, surfacing `ctrl_mode` diagnostics on success/failure
- `apply_calibration` / `apply_calibration_master` / `safe_disconnect` unchanged
- `gentle_disable(kp0=10, kd=0.8, duration, go_home, home_speed=15, settle)` —
  same kp-linear-ramp-to-0 + damped settle profile, on `move_mit`

Gripper `home`/close semantics preserved: `move_to_home` closes the gripper
(`move_gripper_m(0.0)`), matching the legacy `GripperCtrl(0,...)`.

## Other files

### Code migration

- `utils/home_each_arm.py` — enable + `move_j([0]*6)` + `move_gripper_m(0)`
- `utils/gentle_disable_arm.py` — standalone twin of `gentle_disable`, on `move_mit`
- `utils/zero_gripper.py` — `gripper.disable_gripper()` → `gripper.calibrate_gripper()` (zero) → re-enable via `move_gripper_m`; replaces the legacy `GripperCtrl` opcodes `0x00`/`0xAE`/`0x01`
- `utils/test_gripper.py` — `move_gripper_deg()` / `move_gripper_m()`; enable+clear-error via the effector API; read via `get_gripper_status()`
- `utils/watch_leader_grippers.py` — two arms, read gripper via `get_gripper_status().msg.value`
- `tests/motors/test_piper_connect.py` — rewrite the `FakePiper` mock to the
  pyAgxArm surface (`enable`, `get_joint_enable_status`, `get_joint_angles`
  with `.hz`/`.msg`, `move_j`, `get_arm_status` with `.msg.ctrl_mode`). The
  readiness **assertions stay the same** — the regression it guards
  (success only after stable enable + live feedback) is unchanged.

### Docs / wording only (no logic)

- `docs/setup.md` — `pip install piper_sdk` → install `pyAgxArm` + `python-can`
- `deploy/README.md`, `deploy/run_client.sh`, `deploy/run_client_pi05.sh`,
  `deploy/client.py` docstring, `utils/bandwidth_debug.py` docstring — update the
  "base python (piper_sdk + cameras)" wording to `pyAgxArm`

### Not touched (protected by the invariant)

`piper_leader.py`, `piper_follower.py`, `bi_piper_follower`, `deploy/client.py`
logic, `tests/teleoperators/test_piper_leader_ema.py` (unless it references the
legacy mock — verify during implementation). Historical
`docs/superpowers/{plans,specs}/2026-07-04-*` left as records of past work.

## Dependency change

- Remove `piper_sdk`; add `pyAgxArm` (editable install of the vendored `third_party/pyAgxArm`
  checkout) + `python-can>=3.3.4`.
- Install into the `lerobot` env: `pip install -e third_party/pyAgxArm`.
- The deploy client historically ran in "base python"; since the user is
  standardizing on the `lerobot` env, the client runs there too — one env to
  install into. (If base python is still used for the client, install there as
  well; no code difference.)

## Risks — verify on hardware (not unit-testable)

1. Gripper raw unit is exactly µm (the leader's `/1_000_000` strongly implies it;
   confirm with a real `get_gripper_status()` read against a known width).
2. `move_gripper_m` auto-enables the gripper (the new API exposes no separate
   gripper-enable; the legacy `0x01` enable flag has no direct analog).
3. MIT `t_ff` / gentle-disable feel under `PiperFW.V188` matches the current
   tuned descent (kp0=10, kd=0.8).

## Test / verification strategy

- **Unit (lerobot env, no hardware):** rewritten `tests/motors/test_piper_connect.py`
  with the pyAgxArm-shaped `FakePiper`; asserts the same readiness behavior.
- **Import/smoke:** every migrated module imports under the lerobot env with
  `pyAgxArm` installed and `piper_sdk` uninstalled (proves no lingering imports).
- **Hardware (user-run, out of scope for the plan's automated steps):** enable +
  home + teleop round-trip on a real arm to confirm the three risks above and
  that recorded `read()` values match the legacy scale.
