from lerobot.teleoperators.piper_leader.piper_leader import PIPERLeader


class RecordingBus:
    def __init__(self):
        self.connect_calls = []
        self.apply_calibration_master_calls = 0

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        return True

    def apply_calibration_master(self):
        self.apply_calibration_master_calls += 1


def make_leader() -> tuple[PIPERLeader, RecordingBus]:
    leader = PIPERLeader.__new__(PIPERLeader)
    leader.id = "test_leader"
    leader._is_connected = False
    leader._is_calibrated = False
    leader._ema_state = None
    bus = RecordingBus()
    leader.bus = bus
    return leader, bus


def test_connect_selects_passive_gripper_mode():
    leader, bus = make_leader()

    leader.connect(_is_calibrate=False)

    assert bus.connect_calls == [{"enable": True, "command_gripper": False}]


def test_calibrate_is_bookkeeping_only_for_leader():
    leader, bus = make_leader()
    leader._is_connected = True

    leader.calibrate()

    assert leader._is_calibrated is True
    assert bus.apply_calibration_master_calls == 0
