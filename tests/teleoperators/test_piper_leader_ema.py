"""Unit tests for the EMA action smoothing in PIPERLeader (no CAN hardware needed)."""

import pytest

from lerobot.teleoperators.piper_leader.config_piper_leader import PIPERLeaderConfig
from lerobot.teleoperators.piper_leader.piper_leader import PIPERLeader


def make_leader(ema_alpha):
    """Build a PIPERLeader without touching the CAN bus / piper_sdk."""
    leader = PIPERLeader.__new__(PIPERLeader)
    leader.config = PIPERLeaderConfig(ema_alpha=ema_alpha)
    leader._ema_state = None
    return leader


def raw_action(joint_val, gripper_val=0.0):
    action = {f"joint_{i}.pos": joint_val for i in range(1, 7)}
    action["gripper.pos"] = gripper_val
    return action


def test_disabled_is_passthrough():
    leader = make_leader(ema_alpha=None)
    action = raw_action(1.0, gripper_val=0.5)
    assert leader._apply_ema(action) == action
    # Second call still passes through unchanged
    action2 = raw_action(-2.0, gripper_val=0.1)
    assert leader._apply_ema(action2) == action2


def test_first_read_initializes_without_transient():
    leader = make_leader(ema_alpha=0.4)
    action = raw_action(1.0)
    assert leader._apply_ema(action) == action


def test_smoothing_blends_joints():
    leader = make_leader(ema_alpha=0.5)
    leader._apply_ema(raw_action(0.0))
    smoothed = leader._apply_ema(raw_action(1.0))
    for i in range(1, 7):
        assert smoothed[f"joint_{i}.pos"] == pytest.approx(0.5)


def test_converges_to_constant_input():
    leader = make_leader(ema_alpha=0.3)
    leader._apply_ema(raw_action(0.0))
    for _ in range(100):
        smoothed = leader._apply_ema(raw_action(1.0))
    assert smoothed["joint_1.pos"] == pytest.approx(1.0, abs=1e-10)


def test_gripper_bypasses_filter():
    leader = make_leader(ema_alpha=0.1)
    leader._apply_ema(raw_action(0.0, gripper_val=0.0))
    smoothed = leader._apply_ema(raw_action(0.0, gripper_val=1.0))
    assert smoothed["gripper.pos"] == 1.0


def test_state_reset_restarts_filter():
    leader = make_leader(ema_alpha=0.5)
    leader._apply_ema(raw_action(0.0))
    leader._ema_state = None  # what connect() does
    smoothed = leader._apply_ema(raw_action(10.0))
    assert smoothed["joint_1.pos"] == 10.0


@pytest.mark.parametrize("bad_alpha", [0.0, -0.5, 1.5])
def test_invalid_alpha_rejected(bad_alpha):
    with pytest.raises(ValueError):
        PIPERLeaderConfig(ema_alpha=bad_alpha).validate_ema_alpha()
