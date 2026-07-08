# Piper `piper_sdk` → `pyAgxArm` Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AgileX `piper_sdk` (`C_PiperInterface_V2`) with the new `pyAgxArm` SDK across the whole repo, preserving every consumer's behavior and on-the-wire numeric units, then remove `piper_sdk`.

**Architecture:** `PiperMotorsBus` (the one abstraction the robot/teleop wrap) is rewritten to drive `pyAgxArm`'s `robot` + gripper `effector` behind an unchanged 7-DOF `read()`/`write()` interface. A raw-units conversion at the `read()` boundary keeps recorded datasets and `piper_leader`/`piper_follower` byte-for-byte compatible. The five hardware util scripts and one mock-based unit test are ported alongside.

**Tech Stack:** Python 3.12 (`lerobot` conda env), `pyAgxArm` (editable, in-repo `./pyAgxArm`), `python-can>=3.3.4`, pytest.

## Global Constraints

- **Env / python:** run everything with `LEROBOT_PY=/home/embodied/miniconda3/envs/lerobot/bin/python`; all commands run from `REPO=/data/wanshan/VLAs/piper_lerobot`.
- **Hardware config (verbatim):** `robot=ArmModel.PIPER`, `firmeware_version=PiperFW.V188`, `interface="socketcan"`, `channel=<can_name>`, `bitrate=1_000_000` (SDK default).
- **Load-bearing invariant:** `PiperMotorsBus.read()` MUST return **raw firmware units** — joints in 0.001° via `angle_rad * 57324.840764`, gripper raw via `value_m * 1_000_000`. `57324.840764 == 1000 * 180 / π`. Changing the numbers `read()` emits is a defect (it silently breaks every recorded dataset's `observation.state` scale and the leader's `/57324.840764` and `/1e6` divisions).
- **`write()` receives radians (joints) + meters (gripper 0–0.08)** and maps to `robot.move_j(joints_rad)` + `gripper.move_gripper_m(value_m)`.
- **Faithful port only.** Do NOT add `is_connected`/`is_calibrated` to the bus (absent today — preserving parity; a separate pre-existing gap, out of scope). Do NOT change `piper_leader.py`, `piper_follower.py`, `bi_piper_follower`, or `deploy/client.py` logic.
- **Streaming control uses `move_j`** (position-velocity, faithful to the old `MOVE_J`+`JointCtrl`), never `move_js`.
- **`gentle_disable` reimplements the MIT kp-ramp** on `robot.move_mit` (kp0=10, kd=0.8), not the SDK e-stop.
- Keep `piper_sdk` installed until the final task so intermediate states import cleanly.

---

### Task 1: Install `pyAgxArm`, update setup docs

**Files:**
- Modify: `docs/setup.md:17-21`

**Interfaces:**
- Produces: `pyAgxArm` importable in the lerobot env (`from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW`), so every later task's module top-level import resolves.

- [ ] **Step 1: Install pyAgxArm (editable) into the lerobot env**

Run:
```bash
REPO=/data/wanshan/VLAs/piper_lerobot
LEROBOT_PY=/home/embodied/miniconda3/envs/lerobot/bin/python
cd "$REPO"
$LEROBOT_PY -m pip install -e ./pyAgxArm
```
Expected: ends with `Successfully installed pyAgxArm-<version>`. `python-can` is already present (4.6.1) and satisfies `>=3.3.4`.

- [ ] **Step 2: Verify the SDK imports and the constants exist**

Run:
```bash
$LEROBOT_PY -c "from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW; print(ArmModel.PIPER, PiperFW.V188)"
```
Expected: prints `piper v188` (no ImportError / AttributeError). If `PiperFW.V188` raises AttributeError, stop and re-check the installed SDK version against the spec's firmware table.

- [ ] **Step 3: Update the dependency doc**

In `docs/setup.md`, replace the `pip install piper_sdk` dependency block:
```
### 安装 piper 依赖

````
pip install python-can
pip install piper_sdk
````
```
with:
```
### 安装 piper 依赖 (pyAgxArm SDK)

````
pip install python-can
pip install -e ./pyAgxArm
````
```

- [ ] **Step 4: Commit**

```bash
cd "$REPO"
git add docs/setup.md
git commit -m "build(piper): install pyAgxArm SDK, update setup deps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Rewrite `PiperMotorsBus` onto pyAgxArm (TDD via the mock)

**Files:**
- Rewrite: `tests/motors/test_piper_connect.py` (new pyAgxArm-shaped mock; same assertions)
- Rewrite: `src/lerobot/motors/piper/piper.py`

**Interfaces:**
- Consumes: `pyAgxArm` (Task 1).
- Produces: `PiperMotorsBus` with unchanged public surface — `__init__(config)`, `connect(enable, timeout=10)->bool`, `read()->dict` (raw units), `write(target_joint: list)` (rad+m), `move_to_home(timeout=10.0, tol_rad=0.05, speed=50)->bool`, `apply_calibration()->bool`, `apply_calibration_master()`, `gentle_disable(kp0=10.0, kd=0.8, duration=2.0, go_home=True, home_speed=15, settle=0.6)`, `safe_disconnect()`, `is_ready()->bool`, `_wait_until_ready(timeout, stable_needed)->bool`, `_all_motors_enabled()->bool`, `set_calibration()`, `revert_calibration()`, `motor_names`/`motor_models`/`motor_indices` properties. Internals expose `self.robot` (arm driver) and `self.gripper` (effector driver) — the mock test sets these directly.

**Intentional removal:** the old `safe_disconnect_master()` method is dropped. It called a `self.write_master(...)` method that never existed (a latent `AttributeError` if ever invoked) and has **zero callers** in the repo (verified: only `apply_calibration_master`, which is kept, is used — by `piper_leader.py:95`). Removing dead+broken code is in scope here; this is not a behavior change.

- [ ] **Step 1: Rewrite the test mock to the pyAgxArm surface**

Replace the entire contents of `tests/motors/test_piper_connect.py` with:

```python
#!/usr/bin/env python
"""Regression tests for PiperMotorsBus.connect() cold-start readiness.

Root cause guarded here: connect() must not report success until the arm is
actually READY and STAYS ready -- all joints enabled AND joint feedback live
(get_joint_angles().hz > 0), held continuously for a settle window. Declaring
success on a single lucky sample causes the intermittent "follower sometimes not
activated -> leader can't control it" / "works on the 2nd run".

Runnable two ways:
    python -m pytest tests/motors/test_piper_connect.py
    python tests/motors/test_piper_connect.py        # no pytest needed
"""

from types import SimpleNamespace

from lerobot.motors.piper.piper import PiperMotorsBus


def _resolve(v):
    return v() if callable(v) else v


class _Msg:
    """Stand-in for pyAgxArm MessageAbstract (.msg / .hz / .timestamp)."""

    def __init__(self, msg, hz=30.0):
        self.msg = msg
        self.hz = hz
        self.timestamp = 0.0


class FakeRobot:
    """Stand-in for the pyAgxArm arm driver driving connect()'s branches.

    `enabled` / `joint_hz` may be constants or zero-arg callables (evaluated per
    read) to simulate enable flicker / feedback coming up late.
    """

    def __init__(self, *, enabled=True, joint_hz=30.0, moves=True):
        self._enabled = enabled
        self._joint_hz = joint_hz
        self._moves = moves
        # Start ~1 rad from home so move_to_home has to actually drive there.
        self._joints_rad = [1.0] * 6

    def _en(self):
        return bool(_resolve(self._enabled))

    def get_joint_enable_status(self, joint_index=255):
        return self._en()

    def get_joints_enable_status_list(self):
        return [self._en()] * 6

    def enable(self, joint_index=255):
        return self._en()

    def disable(self, joint_index=255):
        return True

    def set_speed_percent(self, pct):
        pass

    def get_joint_angles(self):
        return _Msg(list(self._joints_rad), hz=_resolve(self._joint_hz))

    def get_arm_status(self):
        return _Msg(SimpleNamespace(ctrl_mode=0x01, arm_status=0x00))

    def move_j(self, joints):
        if self._moves:
            self._joints_rad = [float(x) for x in joints]

    def move_mit(self, **kwargs):
        pass


class FakeGripper:
    """Stand-in for the pyAgxArm gripper effector driver."""

    def move_gripper_m(self, value=0.0, force=1.0):
        pass

    def get_gripper_status(self):
        return _Msg(SimpleNamespace(value=0.0, force=0.0, mode="width", foc_status=None))

    def disable_gripper(self):
        return True


def _make_bus(robot: FakeRobot, gripper: FakeGripper | None = None) -> PiperMotorsBus:
    # Bypass __init__: the real one constructs the pyAgxArm driver + effector and
    # calls robot.connect(), which needs live CAN hardware. We only exercise
    # connect()/readiness/home logic against the fakes.
    bus = object.__new__(PiperMotorsBus)
    bus.robot = robot
    bus.gripper = gripper if gripper is not None else FakeGripper()
    bus.motors = {f"joint_{i}": (i, "agilex_piper") for i in range(1, 7)}
    return bus


# --- full connect() integration checks -------------------------------------

def test_connect_fails_when_feedback_not_flowing():
    """Enabled but joint stream dead (hz==0) -> report failure so the caller's
    retry re-runs connect() instead of streaming stale zeros."""
    bus = _make_bus(FakeRobot(enabled=True, joint_hz=0.0))
    assert bus.connect(enable=True) is False


def test_connect_succeeds_when_enabled_and_feedback_live():
    bus = _make_bus(FakeRobot(enabled=True, joint_hz=30.0))
    assert bus.connect(enable=True) is True


# --- readiness-gate unit checks (fast, no 10s enable loop) ------------------

def test_is_ready_false_when_disabled():
    bus = _make_bus(FakeRobot(enabled=False, joint_hz=30.0))
    assert bus.is_ready() is False


def test_is_ready_false_when_feedback_dead():
    bus = _make_bus(FakeRobot(enabled=True, joint_hz=0.0))
    assert bus.is_ready() is False


def test_wait_until_ready_true_when_stable():
    bus = _make_bus(FakeRobot(enabled=True, joint_hz=30.0))
    assert bus._wait_until_ready(timeout=2.0, stable_needed=0.3) is True


def test_wait_until_ready_false_on_enable_flicker():
    """Enable that drops out every other read never satisfies the stability
    window -> not ready (this is the 'enable didn't stick' case)."""
    flip = {"n": 0}

    def flicker():
        flip["n"] += 1
        return flip["n"] % 2 == 0  # alternates True/False

    bus = _make_bus(FakeRobot(enabled=flicker, joint_hz=30.0))
    assert bus._wait_until_ready(timeout=0.6, stable_needed=0.3) is False


# --- polled home checks (the "follower must reach home" root cause) ----------

def test_move_to_home_true_when_arm_accepts_motion():
    """Responsive arm: repeated move_j([0]*6) drives joints to home -> True."""
    bus = _make_bus(FakeRobot(moves=True))
    assert bus.move_to_home(timeout=2.0) is True


def test_move_to_home_false_when_arm_ignores_motion():
    """Arm stuck / ignoring motion never reaches home, so calibration reports
    failure instead of starting dead teleop."""
    bus = _make_bus(FakeRobot(moves=False))
    assert bus.move_to_home(timeout=0.4) is False


# --- enable-loop must not hang forever ---------------------------------------

def test_connect_terminates_when_enable_never_succeeds():
    """Arm that never reports enabled (enable() stays False) must NOT hang in the
    inner 'while not robot.enable()' loop -- it must time out and return False so
    the caller can retry / raise."""
    import time as _time

    bus = _make_bus(FakeRobot(enabled=False, joint_hz=0.0))
    start = _time.time()
    result = bus.connect(enable=True, timeout=0.3)
    elapsed = _time.time() - start
    assert result is False
    assert elapsed < 5.0, f"connect() took {elapsed:.1f}s -- inner enable loop is unbounded"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e or 'assertion failed'}")
    raise SystemExit(failures)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd "$REPO"; $LEROBOT_PY -m pytest tests/motors/test_piper_connect.py -q
```
Expected: FAIL — the current `piper.py` still uses `self.piper`/`piper_sdk`, so `connect()`/`is_ready()`/`move_to_home()` reference attributes the new `FakeRobot` doesn't provide (e.g. `AttributeError: 'FakeRobot' object has no attribute 'GetArmLowSpdInfoMsgs'`).

- [ ] **Step 3: Rewrite `src/lerobot/motors/piper/piper.py`**

Replace the entire file with:

```python
import time
from dataclasses import dataclass
from typing import Dict

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

# rad <-> raw 0.001-deg firmware scale. 57324.840764 == 1000 * 180 / pi.
# read() MUST keep emitting these raw units: piper_follower records them directly
# as observation.state and piper_leader divides joints by this factor (and the
# gripper by 1e6). pyAgxArm reports rad / m, so we convert back at the boundary.
JOINT_RAD_TO_RAW = 57324.840764
GRIPPER_M_TO_RAW = 1_000_000.0  # meters -> raw (~micrometers); leader divides by 1e6


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]


class PiperMotorsBus:
    """对 AgileX pyAgxArm SDK 的二次封装 (由旧 piper_sdk 封装迁移而来)。

    对外接口保持不变: read() 返回原始固件单位 (关节 0.001°, 夹爪 ~µm),
    write() 接收弧度 (关节) 与米 (夹爪 0~0.08)。内部驱动改为 pyAgxArm 的
    arm driver (self.robot) 与 gripper effector (self.gripper)。
    """

    def __init__(self, config: PiperMotorsBusConfig):
        cfg = create_agx_arm_config(
            robot=ArmModel.PIPER,
            firmeware_version=PiperFW.V188,
            interface="socketcan",
            channel=config.can_name,
        )
        self.robot = AgxArmFactory.create_arm(cfg)
        # Effector must be created BEFORE connect() and can only be created once.
        self.gripper = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.robot.connect()  # start the shared read thread (arm + gripper feedback)
        self.motors = config.motors
        self._speed_pct = None
        # 录制数据集时改成0
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [6 joints + 1 gripper]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pose_factor = 1000  # 单位 0.001mm
        self.joint_factor = JOINT_RAD_TO_RAW  # rad -> 0.001°

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def _ensure_speed(self, pct: int) -> None:
        """Set joint speed percent only when it changes (avoid flooding config
        frames at teleop rate). move_j runs at whatever percent was last set."""
        if getattr(self, "_speed_pct", None) != pct:
            self.robot.set_speed_percent(pct)
            self._speed_pct = pct

    def connect(self, enable: bool, timeout: float = 10) -> bool:
        '''
            使能机械臂并检测使能状态; 若使能超时则返回 False (由调用方重试)。
            timeout: 使能等待上限(秒)。从冷/失能状态使能较慢, 给足时间避免误判未使能。
        '''
        enable_flag = False
        loop_flag = False
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            # get_joints_enable_status_list() -> [bool]*6, mirrors reading the 6
            # per-motor foc_status.driver_enable_status flags in the old SDK.
            enable_list = self.robot.get_joints_enable_status_list()
            if enable:
                enable_flag = all(enable_list)
                # robot.enable(255) returns True once all joints already report
                # enabled, so it only flips True after the enable actually stuck.
                # Bound this wait by the outer timeout: if the arm never enables
                # (enable frame dropped in the connect burst, arm unpowered/
                # e-stopped, driver fault, or status frames not arriving) an
                # UNBOUNDED loop here hangs forever printing "piper initing" and
                # blocks both the outer timeout and the caller's retry. On
                # timeout, fall through -> outer loop returns False -> caller
                # retries / raises loudly.
                while not self.robot.enable(255):
                    if time.time() - start_time > timeout:
                        print("enable timed out (motors never reported enabled)")
                        break
                    print('piper initing')
                    time.sleep(0.1)
                self.gripper.move_gripper_m(0.0, 1.0)  # close gripper (was GripperCtrl enable)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.robot.disable(255)
                self.gripper.disable_gripper()
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            if (enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        if enable and resp:
            # Readiness gate: do NOT hand control to teleop/record until the arm
            # is actually READY and STAYS ready. Declaring success the instant
            # enable first reads True (then a blind sleep) causes the intermittent
            # "follower sometimes not activated -> leader can't control it". Three
            # cold-start races feed that flake, all covered by requiring the ready
            # condition to hold *continuously* for a short window:
            #   1. enable status can read True for a single sample before the
            #      driver is really holding torque -> require it STABLE, not one-shot.
            #   2. an enable frame issued during the 4-arm connect burst can be
            #      dropped (ENOBUFS), so the enable doesn't stick -> a dropout
            #      resets the window and the loop re-confirms.
            #   3. on the first process after CAN (re)activation joint feedback
            #      isn't flowing yet (get_joint_angles().hz == 0), so a leader
            #      would stream stale zeros / a follower would record stale state
            #      -> require hz > 0.
            if not self._wait_until_ready(timeout=5.0, stable_needed=0.6):
                print("arm not stably ready (enable+feedback) within deadline; failing connect to trigger retry")
                return False
        print(f"Returning response: {resp}")
        return resp

    def _all_motors_enabled(self) -> bool:
        """True iff all 6 joint drivers currently report enabled."""
        return bool(self.robot.get_joint_enable_status(255))

    def is_ready(self) -> bool:
        """Arm is holding torque (all drivers enabled) AND joint feedback is live.

        Cheap, side-effect-free snapshot -- safe to poll as a readiness barrier
        before starting teleop.
        """
        try:
            ja = self.robot.get_joint_angles()
            return self._all_motors_enabled() and ja is not None and ja.hz > 0
        except Exception:
            return False

    def _wait_until_ready(self, timeout: float = 5.0, stable_needed: float = 0.6) -> bool:
        """Block until is_ready() holds continuously for `stable_needed` seconds.

        Any dropout (enable flicker, feedback stall) resets the stability window,
        so a single lucky sample can't declare the arm ready. Returns False if it
        never stabilizes within `timeout`.
        """
        deadline = time.time() + timeout
        stable_since = None
        while time.time() < deadline:
            if self.is_ready():
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_needed:
                    return True
            else:
                stable_since = None
            time.sleep(0.05)
        return False

    def set_calibration(self):
        return

    def revert_calibration(self):
        return

    def move_to_home(self, timeout: float = 10.0, tol_rad: float = 0.05, speed: int = 50) -> bool:
        """Drive all joints to home (0) under position control, REPEATING the
        command until the arm actually reaches home (verified from joint feedback)
        or `timeout`. Returns True iff home was reached.

        Why polled instead of a single write(): right after enable a dropped
        motion frame (4-arm connect burst) can mean the follower never moves to
        home AND ignores the teleop move_j that follow. Repeating the command
        proves the follower is accepting motion before teleop starts.
        """
        def joints_rad() -> list[float]:
            ja = self.robot.get_joint_angles()
            return [round(a, 3) for a in ja.msg] if ja is not None else [0.0] * 6

        def status_str() -> str:
            st = self.robot.get_arm_status()
            if st is None:
                return "ctrl_mode=?? arm_status=??"
            return f"ctrl_mode=0x{st.msg.ctrl_mode:02X} arm_status=0x{st.msg.arm_status:02X}"

        start_joints = joints_rad()
        start = time.time()
        cmds_sent = 0
        self._ensure_speed(speed)
        while time.time() - start < timeout:
            self.robot.move_j([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.gripper.move_gripper_m(0.0, 1.0)
            cmds_sent += 1
            time.sleep(0.05)
            ja = self.robot.get_joint_angles()
            cur = ja.msg if ja is not None else [1.0] * 6
            if max(abs(a) for a in cur) < tol_rad:
                # ctrl_mode must be 0x01 (CAN command control) for teleop move_j
                # to be obeyed; surface it so "home ok but ctrl_mode=0x00" (arm at
                # home yet ignoring commands) is visible.
                print(f"move_to_home ok after {cmds_sent} cmd(s): {status_str()} start={start_joints}")
                return True
        print(
            f"move_to_home timed out after {timeout}s: {status_str()} "
            f"joints(rad) start={start_joints} end={joints_rad()}"
        )
        return False

    def apply_calibration(self) -> bool:
        """移动到初始位置 (follower). Polled + verified: returns True iff home reached."""
        return self.move_to_home()

    def apply_calibration_master(self):
        """master移动到初始位置"""
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint: list):
        """
            Joint control
            - target_joint: 前 6 个为关节角度 (弧度), 第 7 个为夹爪行程 (米, 0~0.08)
        """
        self._ensure_speed(50)
        self.robot.move_j([float(target_joint[i]) for i in range(6)])
        gripper_m = abs(float(target_joint[6]))
        self.gripper.move_gripper_m(gripper_m, 1.0)

    def read(self) -> Dict:
        """
            返回原始固件单位:
            - 关节: 0.001度 (rad * 57324.840764)
            - 夹爪: 原始单位 (~µm, m * 1e6)
        """
        ja = self.robot.get_joint_angles()
        joints_rad = ja.msg if ja is not None else [0.0] * 6

        gs = self.gripper.get_gripper_status()
        gripper_m = gs.msg.value if gs is not None else 0.0

        return {
            "joint_1": joints_rad[0] * self.joint_factor,
            "joint_2": joints_rad[1] * self.joint_factor,
            "joint_3": joints_rad[2] * self.joint_factor,
            "joint_4": joints_rad[3] * self.joint_factor,
            "joint_5": joints_rad[4] * self.joint_factor,
            "joint_6": joints_rad[5] * self.joint_factor,
            "gripper": gripper_m * GRIPPER_M_TO_RAW,
        }

    def safe_disconnect(self):
        """Move to safe disconnect position"""
        self.write(target_joint=self.safe_disable_position)

    def gentle_disable(self, kp0: float = 10.0, kd: float = 0.8, duration: float = 2.0,
                       go_home: bool = True, home_speed: int = 15, settle: float = 0.6):
        """
            软失能: 避免机械臂直接断电自由落体硬砸下去。
            流程:
              1. (可选) 用位置控制缓慢回到 home 姿态;
              2. 用 MIT 力控保持当前关节角, 把位置增益 kp 在 duration 秒内线性降到 0,
                 保留阻尼 kd, 机械臂被阻尼缓慢放下;
              3. kp=0 只留阻尼沉降一小段, 最后真正失能。

            kp0  初始保持增益 (SDK 参考 10)
            kd   阻尼增益 (SDK 参考 0.8, 最大 5), 越大放下越慢越软, 过大会抖动
            duration  kp 从 kp0 降到 0 的时长(秒)
        """
        NUM_JOINTS = 6
        RATE_HZ = 100.0
        HOME_TOL_RAD = 0.05  # ~3 度, 每个关节都在此范围内视为已到 home
        dt = 1.0 / RATE_HZ

        def read_joints_rad():
            ja = self.robot.get_joint_angles()
            return list(ja.msg) if ja is not None else [0.0] * NUM_JOINTS

        # 确保已使能, 才能在放下前接管 MIT 控制 (录制时通常已使能)
        self.robot.enable(255)

        # 阶段 0: 缓慢回 home, 轮询直到到位或超时
        if go_home:
            self._ensure_speed(home_speed)
            start = time.time()
            while time.time() - start < 10.0:
                if max(abs(a) for a in read_joints_rad()) < HOME_TOL_RAD:
                    break
                self.robot.move_j([0.0] * NUM_JOINTS)
                self.gripper.move_gripper_m(0.0, 1.0)
                time.sleep(0.05)

        hold = read_joints_rad()
        steps = max(1, int(duration * RATE_HZ))

        # 阶段 1: kp 线性降到 0, 阻尼 kd 抵抗下落 (move_mit 内部会切到 MIT 模式)
        for s in range(steps + 1):
            kp = kp0 * (1.0 - s / steps)
            for j in range(NUM_JOINTS):
                self.robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=kp, kd=kd, t_ff=0.0)
            time.sleep(dt)

        # 阶段 2: kp=0, 仅保留阻尼, 缓慢沉降
        for _ in range(int(settle * RATE_HZ)):
            for j in range(NUM_JOINTS):
                self.robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=0.0, kd=kd, t_ff=0.0)
            time.sleep(dt)

        # 阶段 3: 真正失能 (下一次 move_j 会自动切回位置/速度控制模式)
        while not self.robot.disable(255):
            time.sleep(0.01)
        time.sleep(0.3)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cd "$REPO"; $LEROBOT_PY -m pytest tests/motors/test_piper_connect.py -q
```
Expected: PASS (10 passed).

- [ ] **Step 5: Verify the module imports cleanly (no piper_sdk)**

Run:
```bash
cd "$REPO"; PYTHONPATH=src $LEROBOT_PY -c "import lerobot.motors.piper.piper as m; print('ok', m.JOINT_RAD_TO_RAW, m.GRIPPER_M_TO_RAW)"
```
Expected: `ok 57324.840764 1000000.0`.

- [ ] **Step 6: Confirm no piper_sdk reference remains in the module**

Run:
```bash
cd "$REPO"; grep -n "piper_sdk\|C_PiperInterface\|GripperCtrl\|JointCtrl\|MotionCtrl\|GetArm" src/lerobot/motors/piper/piper.py || echo "CLEAN"
```
Expected: `CLEAN`.

- [ ] **Step 7: Commit**

```bash
cd "$REPO"
git add src/lerobot/motors/piper/piper.py tests/motors/test_piper_connect.py
git commit -m "feat(piper): port PiperMotorsBus to pyAgxArm SDK

Faithful port: read() still emits raw 0.001deg/um units; write() maps to
move_j (rad) + move_gripper_m (m); readiness gate + polled move_to_home +
MIT-ramp gentle_disable preserved on the new getters.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Port `utils/home_each_arm.py`

**Files:**
- Rewrite: `utils/home_each_arm.py`

**Interfaces:**
- Consumes: `pyAgxArm` (Task 1).
- Produces: standalone script; no importable API relied on by other tasks.

- [ ] **Step 1: Rewrite the file**

Replace the entire contents of `utils/home_each_arm.py` with:

```python
#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Home each Piper arm one at a time to identify which CAN interface is which arm.

For every CAN name given (default: all four), this script enables the arm,
moves it to the home position (all joints = 0), prints which CAN it is,
then waits 5 seconds before moving to the next arm.

Watch the robots: the arm that moves is the one bound to the printed CAN name.

Usage:
    python utils/home_each_arm.py                       # all four, in order
    python utils/home_each_arm.py can_master can_follower
"""
import sys
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

DEFAULT_CANS = ["can_master", "can_follower", "can_master2", "can_follower2"]
GAP_S = 5


def make_robot(can):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V188,
        interface="socketcan",
        channel=can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    return robot, gripper


def enable(robot, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        if robot.get_joint_enable_status(255):
            return True
        robot.enable(255)
        time.sleep(0.2)
    return False


def home(robot, gripper):
    robot.set_speed_percent(50)  # position control, speed 50%
    robot.move_j([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gripper.move_gripper_m(0.0, 1.0)


def main(cans):
    for idx, can in enumerate(cans):
        print("=" * 50)
        print(f">>> Arm {idx + 1}/{len(cans)}  ->  CAN = {can}")
        print("=" * 50)
        try:
            robot, gripper = make_robot(can)
            time.sleep(0.5)
            if not enable(robot):
                print(f"[WARN] {can}: enable timed out (arm powered on / CAN up?)")
            home(robot, gripper)
            print(f"    {can}: home command sent. WATCH which arm moves now.")
        except Exception as e:
            print(f"[ERROR] {can}: {e}")
        print(f"    waiting {GAP_S}s before next arm...\n")
        time.sleep(GAP_S)
    print("Done. Note which physical arm moved for each CAN name above.")


if __name__ == "__main__":
    cans = sys.argv[1:] or DEFAULT_CANS
    main(cans)
```

- [ ] **Step 2: Verify it compiles and has no piper_sdk reference**

Run:
```bash
cd "$REPO"; $LEROBOT_PY -m py_compile utils/home_each_arm.py && grep -n "piper_sdk\|C_PiperInterface" utils/home_each_arm.py || echo "COMPILES + CLEAN"
```
Expected: `COMPILES + CLEAN`.

- [ ] **Step 3: Commit**

```bash
cd "$REPO"
git add utils/home_each_arm.py
git commit -m "refactor(utils): port home_each_arm to pyAgxArm

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Port `utils/gentle_disable_arm.py`

**Files:**
- Rewrite: `utils/gentle_disable_arm.py`

**Interfaces:**
- Consumes: `pyAgxArm` (Task 1).
- Produces: standalone script.

**Note:** the current file is in a half-commented experimental state; this task restores it as a faithful standalone twin of the authoritative `PiperMotorsBus.gentle_disable` (MIT kp-ramp), which is the working reference.

- [ ] **Step 1: Rewrite the file**

Replace the entire contents of `utils/gentle_disable_arm.py` with:

```python
#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Gently disable (软失能) a Piper arm so it does NOT free-fall.

Sequence: (1) optionally move the arm slowly to home (all joints 0) with position
control, then (2) softly release it via MIT control.

Why: disable() cuts all motor torque instantly. At the home pose the arm is still
holding itself against gravity, so it drops hard the moment torque is removed.
This script holds the current joint angles with MIT control and ramps the
position gain kp -> 0 while keeping a damping gain kd > 0. The arm slowly,
dampedly droops (no slam), and is already limp by the time disable() is called.

Usage:
    python utils/gentle_disable_arm.py                       # all 4 arms (dual-arm)
    python utils/gentle_disable_arm.py left_follower
    python utils/gentle_disable_arm.py left_follower right_follower
    python utils/gentle_disable_arm.py left_follower --duration 3.0 --kp 12 --kd 1.2
    python utils/gentle_disable_arm.py left_follower --no-home   # skip move-to-home

Tuning on real hardware:
    --kp   initial holding gain (SDK reference 10). Higher = firmer hold at start.
    --kd   damping gain (SDK reference 0.8, max 5). Higher = slower/softer droop,
           too high = jitter/oscillation.
    --duration  seconds to ramp kp from --kp down to 0.
"""
import argparse
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

NUM_JOINTS = 6
RATE_HZ = 100.0
HOME_TOL_RAD = 0.05  # ~3 deg: consider "at home" when every joint is within this


def make_robot(can):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V188,
        interface="socketcan",
        channel=can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    return robot, gripper


def read_joints_rad(robot):
    ja = robot.get_joint_angles()
    return list(ja.msg) if ja is not None else [0.0] * NUM_JOINTS


def enable(robot, timeout=3.0):
    """Make sure motors are enabled so we can take MIT control before releasing."""
    start = time.time()
    while time.time() - start < timeout:
        if robot.enable(255):
            return True
        time.sleep(0.05)
    return False


def move_to_home(robot, gripper, speed, timeout=10.0):
    """Slowly move all joints to 0 with position control; wait until arrived."""
    robot.set_speed_percent(speed)
    start = time.time()
    while time.time() - start < timeout:
        if max(abs(a) for a in read_joints_rad(robot)) < HOME_TOL_RAD:
            return True
        robot.move_j([0.0] * NUM_JOINTS)
        gripper.move_gripper_m(0.0, 1.0)
        time.sleep(0.05)
    return False


def gentle_disable(can, kp0, kd, duration, go_home, home_speed, settle=0.6):
    robot, gripper = make_robot(can)
    time.sleep(0.2)

    if not enable(robot):
        print(f"[WARN] {can}: could not confirm enable — arm may already be limp.")

    if go_home:
        print(f"{can}: moving to home (speed {home_speed}%)...")
        if move_to_home(robot, gripper, home_speed):
            print(f"{can}: reached home.")
        else:
            print(f"[WARN] {can}: home move timed out — releasing from current pose.")

    hold = read_joints_rad(robot)
    print(f"{can}: holding {[round(a, 3) for a in hold]} rad, ramping kp {kp0}->0 over {duration}s")

    dt = 1.0 / RATE_HZ
    steps = max(1, int(duration * RATE_HZ))

    # Phase 1: ramp kp -> 0 while damping (kd) resists any fall.
    for s in range(steps + 1):
        kp = kp0 * (1.0 - s / steps)  # linear fade to zero
        for j in range(NUM_JOINTS):
            robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=kp, kd=kd, t_ff=0.0)
        time.sleep(dt)

    # Phase 2: kp = 0, keep only damping so it settles softly at the bottom.
    for _ in range(int(settle * RATE_HZ)):
        for j in range(NUM_JOINTS):
            robot.move_mit(joint_index=j + 1, p_des=hold[j], v_des=0.0, kp=0.0, kd=kd, t_ff=0.0)
        time.sleep(dt)

    # Phase 3: fully release (next move_j restores position/speed control mode).
    while not robot.disable(255):
        time.sleep(0.01)
    time.sleep(0.3)
    print(f"{can}: 软失能成功!!!!")


def main():
    ap = argparse.ArgumentParser(description="Gently disable one or more Piper arms.")
    ap.add_argument("cans", nargs="*",
                    default=["left_leader", "left_follower", "right_leader", "right_follower"],
                    help="CAN names (default: all 4 arms). See can_arm_mapping.md")
    ap.add_argument("--kp", type=float, default=10.0, help="initial holding gain (ref 10)")
    ap.add_argument("--kd", type=float, default=0.8, help="damping gain (ref 0.8, max 5)")
    ap.add_argument("--duration", type=float, default=2.0, help="kp ramp-down seconds")
    ap.add_argument("--home-speed", type=int, default=15,
                    help="move-to-home speed percent (default 15, slow)")
    ap.add_argument("--no-home", dest="home", action="store_false",
                    help="skip move-to-home; soft-release from current pose")
    args = ap.parse_args()

    for can in args.cans:
        gentle_disable(can, args.kp, args.kd, args.duration, args.home, args.home_speed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it compiles and has no piper_sdk reference**

Run:
```bash
cd "$REPO"; $LEROBOT_PY -m py_compile utils/gentle_disable_arm.py && grep -n "piper_sdk\|C_PiperInterface" utils/gentle_disable_arm.py || echo "COMPILES + CLEAN"
```
Expected: `COMPILES + CLEAN`.

- [ ] **Step 3: Commit**

```bash
cd "$REPO"
git add utils/gentle_disable_arm.py
git commit -m "refactor(utils): port gentle_disable_arm to pyAgxArm (MIT ramp restored)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Port `utils/zero_gripper.py`

**Files:**
- Rewrite: `utils/zero_gripper.py`

**Interfaces:**
- Consumes: `pyAgxArm` (Task 1).
- Produces: standalone script.

- [ ] **Step 1: Rewrite the file**

Replace the entire contents of `utils/zero_gripper.py` with:

```python
#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Set a Piper gripper's ZERO point at its fully-closed position.

Symptom this fixes: a gripper whose closed position reads a non-zero (often
negative) value. In teleop that offset makes the follower open instead of close.

The zero MUST be set while the gripper is DISABLED. pyAgxArm exposes this as:
    gripper.disable_gripper()        # gripper goes limp
    <move jaws fully closed by hand>
    gripper.calibrate_gripper()      # store current position as zero
Setting zero while enabled is silently ignored by the firmware.

Procedure:
  1. Run this script; it disables the gripper so you can move it by hand.
  2. Push the gripper jaws FULLY CLOSED (to the mechanical stop).
  3. Press Enter; it stores that position as value 0.
  4. It reads back the value -- should now be ~0.

Do this for BOTH arms of a teleop pair (leader and follower).

Usage:
    python utils/zero_gripper.py left_follower
    python utils/zero_gripper.py left_leader
    python utils/zero_gripper.py left_follower --yes   # skip the confirm prompt
"""
import argparse
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


def make_gripper(can):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V188,
        interface="socketcan",
        channel=can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    return robot, gripper


def read_value(gripper):
    gs = gripper.get_gripper_status()
    return gs.msg.value if gs is not None else None  # meters


def main():
    ap = argparse.ArgumentParser(description="Set a Piper gripper zero at its closed position.")
    ap.add_argument("can", help="CAN name, e.g. left_follower / left_leader")
    ap.add_argument("--force", type=float, default=1.0, help="re-enable gripping force in N (default 1.0)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    robot, gripper = make_gripper(args.can)
    time.sleep(0.3)

    print(f"{args.can}: current gripper value = {read_value(gripper)} m")

    # Disable the gripper so it goes limp and the firmware will accept a new zero.
    gripper.disable_gripper()
    time.sleep(1.5)

    if not args.yes:
        input(f"{args.can} gripper is now limp. Push the jaws FULLY CLOSED, then press Enter to set zero... ")

    # calibrate_gripper() stores the current position as value 0.
    ok = gripper.calibrate_gripper()
    print(f"{args.can}: calibrate_gripper() -> {ok}")
    time.sleep(0.5)

    for _ in range(5):
        print(f"  value after zeroing -> {read_value(gripper)} m")
        time.sleep(0.2)

    # Re-enable so the gripper is ready to hold/track again (move enables it).
    gripper.move_gripper_m(0.0, args.force)
    print(f"{args.can}: done. Closed should now read ~0; gripper re-enabled.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it compiles and has no piper_sdk reference**

Run:
```bash
cd "$REPO"; $LEROBOT_PY -m py_compile utils/zero_gripper.py && grep -n "piper_sdk\|C_PiperInterface\|GripperCtrl" utils/zero_gripper.py || echo "COMPILES + CLEAN"
```
Expected: `COMPILES + CLEAN`.

- [ ] **Step 3: Commit**

```bash
cd "$REPO"
git add utils/zero_gripper.py
git commit -m "refactor(utils): port zero_gripper to pyAgxArm calibrate_gripper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Port `utils/test_gripper.py`

**Files:**
- Rewrite: `utils/test_gripper.py`

**Interfaces:**
- Consumes: `pyAgxArm` (Task 1).
- Produces: standalone script.

- [ ] **Step 1: Rewrite the file**

Replace the entire contents of `utils/test_gripper.py` with:

```python
#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Test / drive a Piper gripper on a given CAN interface.

Recovers the gripper from a fault (disable, then move re-enables), commands it
open or close, and prints the gripper feedback so you can see if it actually
moves / how far.

Usage:
    python utils/test_gripper.py left_follower              # open
    python utils/test_gripper.py left_follower --close      # close
    python utils/test_gripper.py left_follower --mm 40      # open to 40 mm
    python utils/test_gripper.py left_follower --force 2.0  # more force (N)

Notes:
- gripper value unit is meters (so 70 mm -> 0.070). Range ~0..0.07 m.
- force unit is N, range 0..3.
"""
import argparse
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


def make_gripper(can):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V188,
        interface="socketcan",
        channel=can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    return robot, gripper


def read_gripper(gripper):
    try:
        gs = gripper.get_gripper_status()
        if gs is None:
            return "no feedback yet"
        return f"value={gs.msg.value} m  force={gs.msg.force} N  mode={gs.msg.mode}  foc_status={gs.msg.foc_status}"
    except Exception as e:
        return f"read err: {e}"


def main():
    ap = argparse.ArgumentParser(description="Open/close a Piper gripper to test it.")
    ap.add_argument("can", nargs="?", default="left_follower", help="CAN name")
    ap.add_argument("--close", action="store_true", help="close instead of open")
    ap.add_argument("--mm", type=float, default=70.0, help="open target in mm (default 70)")
    ap.add_argument("--force", type=float, default=1.0, help="gripping force in N, 0..3 (default 1.0)")
    args = ap.parse_args()

    value_m = 0.0 if args.close else args.mm / 1000.0  # mm -> m

    robot, gripper = make_gripper(args.can)
    time.sleep(0.2)

    # Enable the arm so the gripper responds.
    for _ in range(20):
        if robot.enable(255):
            break
        time.sleep(0.05)

    print(f"{args.can}: gripper before -> {read_gripper(gripper)}")

    # Recover from a fault: disable then move re-enables and clears the error.
    gripper.disable_gripper()
    time.sleep(0.5)
    action = "CLOSE" if args.close else f"OPEN {args.mm}mm"
    print(f"{args.can}: commanding {action} (value={value_m} m, force={args.force} N)")
    gripper.move_gripper_m(value_m, args.force)

    # Watch it move for a couple seconds.
    for _ in range(10):
        time.sleep(0.2)
        print(f"  gripper -> {read_gripper(gripper)}")

    print(f"{args.can}: done. If value did not change, gripper is likely faulty / mis-cabled.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it compiles and has no piper_sdk reference**

Run:
```bash
cd "$REPO"; $LEROBOT_PY -m py_compile utils/test_gripper.py && grep -n "piper_sdk\|C_PiperInterface\|GripperCtrl" utils/test_gripper.py || echo "COMPILES + CLEAN"
```
Expected: `COMPILES + CLEAN`.

- [ ] **Step 3: Commit**

```bash
cd "$REPO"
git add utils/test_gripper.py
git commit -m "refactor(utils): port test_gripper to pyAgxArm effector API

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Port `utils/watch_leader_grippers.py`

**Files:**
- Rewrite: `utils/watch_leader_grippers.py`

**Interfaces:**
- Consumes: `pyAgxArm` (Task 1).
- Produces: standalone script.

- [ ] **Step 1: Rewrite the file**

Replace the entire contents of `utils/watch_leader_grippers.py` with:

```python
#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Read-only monitor of both leader grippers. Moves nothing.

Squeeze each leader gripper fully closed, then fully open, and watch the value.
A healthy leader reads ~0.0 m when closed and ~0.07 m when open. If a leader
never drops near 0 when you close it, its zero point is offset -> the follower is
commanded to stay open (that's the bug, on the LEADER).

The "-> follower cmd" column is exactly what the record pipeline would send:
    action = value (meters, from get_gripper_status().msg.value)
    follower move_gripper_m(value) == same width   (round-trip)

Usage:
    python utils/watch_leader_grippers.py
    python utils/watch_leader_grippers.py --left left_leader --right right_leader
"""
import argparse
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


def make_gripper(can):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V188,
        interface="socketcan",
        channel=can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    return robot, gripper


def read_value(gripper):
    try:
        gs = gripper.get_gripper_status()
        return gs.msg.value if gs is not None else "no fb"
    except Exception as e:
        return f"err:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="left_leader")
    ap.add_argument("--right", default="right_leader")
    args = ap.parse_args()

    _, left = make_gripper(args.left)
    _, right = make_gripper(args.right)
    time.sleep(0.3)

    print("Close then open each leader gripper. Ctrl-C to stop.\n")
    print(f"{'left value(m)':>14} {'-> foll cmd':>12}   |  {'right value(m)':>14} {'-> foll cmd':>12}")
    try:
        while True:
            lv = read_value(left)
            rv = read_value(right)
            lc = lv if isinstance(lv, float) else lv
            rc = rv if isinstance(rv, float) else rv
            print(f"{lv:>14} {lc:>12}   |  {rv:>14} {rc:>12}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it compiles and has no piper_sdk reference**

Run:
```bash
cd "$REPO"; $LEROBOT_PY -m py_compile utils/watch_leader_grippers.py && grep -n "piper_sdk\|C_PiperInterface" utils/watch_leader_grippers.py || echo "COMPILES + CLEAN"
```
Expected: `COMPILES + CLEAN`.

- [ ] **Step 3: Commit**

```bash
cd "$REPO"
git add utils/watch_leader_grippers.py
git commit -m "refactor(utils): port watch_leader_grippers to pyAgxArm

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Update docstrings / wording (no logic)

**Files:**
- Modify: `deploy/README.md:5`
- Modify: `deploy/run_client.sh:2`
- Modify: `deploy/run_client_pi05.sh:2`
- Modify: `deploy/client.py:3`
- Modify: `utils/bandwidth_debug.py:17`

**Interfaces:** none — comment/doc text only.

- [ ] **Step 1: Update each mention of "piper_sdk" in user-facing wording**

Make these exact replacements (text `piper_sdk` → `pyAgxArm`):

- `deploy/README.md` line 5: `(piper_sdk + cameras) and streams action chunks over localhost HTTP with no` → `(pyAgxArm + cameras) and streams action chunks over localhost HTTP with no`
- `deploy/run_client.sh` line 2: `# Run the Piper deployment client (base python: piper_sdk + cameras).` → `# Run the Piper deployment client (pyAgxArm + cameras).`
- `deploy/run_client_pi05.sh` line 2: `# Run the Piper deployment client against a pi05 server (base python: piper_sdk + cameras).` → `# Run the Piper deployment client against a pi05 server (pyAgxArm + cameras).`
- `deploy/client.py` line 3: `Runs in base python (piper_sdk + cameras) with this repo's lerobot on the` → `Runs with pyAgxArm + cameras and this repo's lerobot on the`
- `utils/bandwidth_debug.py` line 17: `No root needed. Run in an env with piper_sdk + opencv (e.g. base python):` → `No root needed. Run in an env with pyAgxArm + opencv (e.g. the lerobot env):`

- [ ] **Step 2: Verify only intended files still mention piper_sdk (none of these five)**

Run:
```bash
cd "$REPO"; grep -rn "piper_sdk" deploy/README.md deploy/run_client.sh deploy/run_client_pi05.sh deploy/client.py utils/bandwidth_debug.py || echo "CLEAN"
```
Expected: `CLEAN`.

- [ ] **Step 3: Commit**

```bash
cd "$REPO"
git add deploy/README.md deploy/run_client.sh deploy/run_client_pi05.sh deploy/client.py utils/bandwidth_debug.py
git commit -m "docs(deploy): reword piper_sdk -> pyAgxArm in client wording

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Remove `piper_sdk`, verify no references remain

**Files:** none (environment + repo-wide verification).

**Interfaces:**
- Consumes: all prior tasks complete.
- Produces: `piper_sdk`-free repo and env; full unit-test suite green.

- [ ] **Step 1: Prove no source/test/util file imports piper_sdk anymore**

Run:
```bash
cd "$REPO"; grep -rn "import piper_sdk\|from piper_sdk\|C_PiperInterface" --include=*.py src deploy utils tests | grep -v pyAgxArm || echo "NO piper_sdk IMPORTS"
```
Expected: `NO piper_sdk IMPORTS`. If anything prints, fix that file before continuing.

- [ ] **Step 2: Uninstall piper_sdk from the lerobot env**

Run:
```bash
$LEROBOT_PY -m pip uninstall -y piper_sdk
```
Expected: `Successfully uninstalled piper_sdk-...` (or "not installed", which is also fine).

- [ ] **Step 3: Smoke-import every migrated module + run the full unit suite**

Run:
```bash
cd "$REPO"
PYTHONPATH=src $LEROBOT_PY -c "import lerobot.motors.piper.piper"
for f in utils/home_each_arm.py utils/gentle_disable_arm.py utils/zero_gripper.py utils/test_gripper.py utils/watch_leader_grippers.py; do $LEROBOT_PY -m py_compile "$f" && echo "ok $f"; done
$LEROBOT_PY -m pytest tests/motors/test_piper_connect.py tests/teleoperators/test_piper_leader_ema.py -q
```
Expected: the import prints nothing (success), each util prints `ok <file>`, and pytest reports all tests passed (no `ModuleNotFoundError: piper_sdk`).

- [ ] **Step 4: Commit the verification milestone (empty commit if nothing to stage)**

```bash
cd "$REPO"
git commit --allow-empty -m "chore(piper): remove piper_sdk; migration to pyAgxArm complete

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Hardware verification (user-run, after the plan)

These cannot be unit-tested — the user runs them on a real arm (see spec §Risks):

1. **Units round-trip:** teleop-record a few frames; confirm `observation.state`
   joint values match the pre-migration scale (raw 0.001°) and the gripper
   column matches the leader's `/1e6` expectation (gripper raw ≈ µm).
2. **Gripper auto-enable:** confirm `move_gripper_m` moves the gripper without a
   separate enable call.
3. **Gentle disable feel:** confirm the arm droops softly (no slam) under
   `PiperFW.V188` MIT with kp0=10, kd=0.8.

If any differ, adjust `GRIPPER_M_TO_RAW` (risk 1) or `firmeware_version` (risk 3)
and re-run the affected task's verification.
```
