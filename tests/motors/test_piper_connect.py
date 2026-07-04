#!/usr/bin/env python
"""Regression tests for PiperMotorsBus.connect() cold-start readiness.

Root cause guarded here: connect() must not report success until the arm is
actually READY and STAYS ready -- all 6 drivers enabled AND joint feedback live
(GetArmJointMsgs().Hz > 0), held continuously for a settle window. Declaring
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


class FakePiper:
    """Stand-in for C_PiperInterface_V2 driving connect()'s branches.

    `enabled` / `joint_hz` may be constants or zero-arg callables (evaluated per
    read) to simulate flicker / feedback coming up late.
    """

    def __init__(self, *, enabled=True, joint_hz=30.0, moves=True):
        self._enabled = enabled
        self._joint_hz = joint_hz
        self._moves = moves
        # Start far from home (raw units 0.001 deg); ~57000 raw ~= 1 rad.
        self._joints_raw = [57000] * 6

    def GetArmLowSpdInfoMsgs(self):
        en = _resolve(self._enabled)
        motor = SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=en))
        return SimpleNamespace(**{f"motor_{i}": motor for i in range(1, 7)})

    def GetArmJointMsgs(self):
        js = SimpleNamespace(**{f"joint_{i}": self._joints_raw[i - 1] for i in range(1, 7)})
        return SimpleNamespace(Hz=_resolve(self._joint_hz), joint_state=js)

    def GetArmStatus(self):
        return SimpleNamespace(arm_status=SimpleNamespace(ctrl_mode=0x01, arm_status=0x00))

    def JointCtrl(self, *targets):
        # A responsive arm in position mode tracks the commanded target.
        if self._moves:
            self._joints_raw = list(targets)

    def MotionCtrl_2(self, *args):
        pass

    def EnablePiper(self):
        # Mirrors the real SDK: returns whether all motors ALREADY report enabled.
        return bool(_resolve(self._enabled))

    def GripperCtrl(self, *args):
        pass

    def DisableArm(self, *args):
        pass


def _make_bus(fake: FakePiper) -> PiperMotorsBus:
    # Bypass __init__: the real one constructs C_PiperInterface_V2 + ConnectPort(),
    # which needs live CAN hardware. We only exercise connect()/readiness logic.
    bus = object.__new__(PiperMotorsBus)
    bus.piper = fake
    bus.motors = {f"joint_{i}": (i, "agilex_piper") for i in range(1, 7)}
    return bus


# --- full connect() integration checks -------------------------------------

def test_connect_fails_when_feedback_not_flowing():
    """Enabled but joint stream dead (Hz==0) -> report failure so the caller's
    retry re-runs connect() instead of streaming stale zeros."""
    bus = _make_bus(FakePiper(enabled=True, joint_hz=0.0))
    assert bus.connect(enable=True) is False


def test_connect_succeeds_when_enabled_and_feedback_live():
    bus = _make_bus(FakePiper(enabled=True, joint_hz=30.0))
    assert bus.connect(enable=True) is True


# --- readiness-gate unit checks (fast, no 10s enable loop) ------------------

def test_is_ready_false_when_disabled():
    bus = _make_bus(FakePiper(enabled=False, joint_hz=30.0))
    assert bus.is_ready() is False


def test_is_ready_false_when_feedback_dead():
    bus = _make_bus(FakePiper(enabled=True, joint_hz=0.0))
    assert bus.is_ready() is False


def test_wait_until_ready_true_when_stable():
    bus = _make_bus(FakePiper(enabled=True, joint_hz=30.0))
    assert bus._wait_until_ready(timeout=2.0, stable_needed=0.3) is True


def test_wait_until_ready_false_on_enable_flicker():
    """Enable that drops out every other read never satisfies the stability
    window -> not ready (this is the 'enable didn't stick' case)."""
    flip = {"n": 0}

    def flicker():
        flip["n"] += 1
        return flip["n"] % 2 == 0  # alternates True/False

    bus = _make_bus(FakePiper(enabled=flicker, joint_hz=30.0))
    assert bus._wait_until_ready(timeout=0.6, stable_needed=0.3) is False


# --- polled home checks (the "follower must reach home" root cause) ----------

def test_move_to_home_true_when_arm_accepts_motion():
    """Responsive arm: repeated JointCtrl(0) drives joints to home -> True."""
    bus = _make_bus(FakePiper(moves=True))
    assert bus.move_to_home(timeout=2.0) is True


def test_move_to_home_false_when_arm_ignores_motion():
    """Arm stuck in a non-position mode ignores JointCtrl -> never reaches home,
    so calibration reports failure instead of starting dead teleop."""
    bus = _make_bus(FakePiper(moves=False))
    assert bus.move_to_home(timeout=0.4) is False


# --- enable-loop must not hang forever ---------------------------------------

def test_connect_terminates_when_enable_never_succeeds():
    """Arm that never reports enabled (EnablePiper stays False) must NOT hang in
    the inner 'while not EnablePiper()' loop -- it must time out and return False
    so the caller can retry / raise."""
    import time as _time

    bus = _make_bus(FakePiper(enabled=False, joint_hz=0.0))
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
