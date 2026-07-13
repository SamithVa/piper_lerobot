import math
from types import SimpleNamespace

from lerobot.motors.piper.piper import JOINT_RAD_TO_RAW, PiperCommandLimiter
from lerobot.robots.piper_follower.piper_follower import PIPERFollower


class FakeRobot:
    def __init__(self):
        self.emergency_stops = 0

    def electronic_emergency_stop(self):
        self.emergency_stops += 1


class FakeBus:
    def __init__(self):
        self.robot = FakeRobot()
        self.joints = [0.0] * 6
        self.write_calls = []
        self.write_js_calls = []

    def read(self):
        return {
            **{f"joint_{index + 1}": value * JOINT_RAD_TO_RAW for index, value in enumerate(self.joints)},
            "gripper": 0.0,
        }

    def write(self, target):
        self.write_calls.append(list(target))

    def write_js(self, target):
        self.write_js_calls.append(list(target))


def make_follower(mode: str = "move_js") -> PIPERFollower:
    follower = object.__new__(PIPERFollower)
    follower.config = SimpleNamespace(
        teleop_control_mode=mode,
        teleop_control_hz=50.0,
        max_velocity_deg_s=45.0,
        max_acceleration_deg_s2=200.0,
        max_tracking_error_deg=12.0,
    )
    follower.bus = FakeBus()
    follower._is_connected = True
    follower._command_limiter = None
    follower._move_js_safety_tripped = False
    follower._last_sent_action = None
    return follower


def action_at(joint_rad: float) -> dict[str, float]:
    return {
        **{f"joint_{index}.pos": joint_rad for index in range(1, 7)},
        "gripper.pos": 0.04,
    }


def test_move_j_remains_unfiltered_default_path():
    follower = make_follower("move_j")
    action = action_at(0.1)

    sent = follower.send_action(action)

    assert sent == action
    assert follower.bus.write_calls == [[0.1] * 6 + [0.04]]
    assert follower.bus.write_js_calls == []


def test_move_js_returns_and_writes_limited_action():
    follower = make_follower()

    sent = follower.send_action(action_at(0.5))

    # From rest: a=200 deg/s² for 20 ms -> v=4 deg/s -> step=0.08 deg.
    expected_first_step = math.radians(0.08)
    assert math.isclose(sent["joint_1.pos"], expected_first_step)
    assert follower.bus.write_calls == []
    assert len(follower.bus.write_js_calls) == 1
    assert math.isclose(follower.bus.write_js_calls[0][0], expected_first_step)
    assert follower.bus.write_js_calls[0][-1] == 0.04


def test_move_js_tracking_watchdog_stops_without_sending_another_target():
    follower = make_follower()
    follower._command_limiter = PiperCommandLimiter.from_position(
        [math.radians(20.0)] * 6,
        max_velocity_deg_s=45.0,
        max_acceleration_deg_s2=200.0,
    )
    follower._last_sent_action = action_at(0.0)

    sent = follower.send_action(action_at(math.radians(20.0)))

    assert sent == follower._last_sent_action
    assert follower._move_js_safety_tripped is True
    assert follower.bus.robot.emergency_stops == 1
    assert follower.bus.write_js_calls == []
