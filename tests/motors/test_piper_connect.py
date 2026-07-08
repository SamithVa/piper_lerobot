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
