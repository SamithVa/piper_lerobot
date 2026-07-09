import json

import pytest

from deploy import launch


def write_preset(tmp_path, name, body):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(body))
    return path


MINIMAL = {"server": {"python": "python3", "adapter": "dummy", "port": 8090}}


def test_load_preset_by_name(tmp_path):
    write_preset(tmp_path, "mini", MINIMAL)
    assert launch.load_preset("mini", presets_dir=tmp_path)["server"]["adapter"] == "dummy"


def test_load_preset_by_path(tmp_path):
    path = write_preset(tmp_path, "mini", MINIMAL)
    assert launch.load_preset(str(path))["server"]["port"] == 8090


def test_load_preset_unknown_lists_available(tmp_path):
    write_preset(tmp_path, "mini", MINIMAL)
    with pytest.raises(SystemExit, match="mini"):
        launch.load_preset("nope", presets_dir=tmp_path)


def test_load_preset_rejects_missing_server_field(tmp_path):
    write_preset(tmp_path, "bad", {"server": {"python": "python3"}})
    with pytest.raises(SystemExit, match="adapter"):
        launch.load_preset("bad", presets_dir=tmp_path)


def test_decide_spawn_when_nothing_listening():
    assert launch.decide(None, "/ckpt") == "spawn"


def test_decide_reuse_on_matching_checkpoint():
    assert launch.decide({"checkpoint": "/ckpt"}, "/ckpt") == "reuse"
    assert launch.decide({"checkpoint": None}, None) == "reuse"  # dummy adapter


def test_decide_refuse_on_mismatch_or_foreign_server():
    assert launch.decide({"checkpoint": "/other"}, "/ckpt") == "refuse"
    assert launch.decide({}, None) == "refuse"  # no checkpoint key: not ours


def test_resolve_path_repo_relative_and_hub_id():
    assert launch.resolve_path(".") == str(launch.REPO_ROOT)
    assert launch.resolve_path("samithva/pi05_stack_cup_bowl") == "samithva/pi05_stack_cup_bowl"


def test_shipped_presets_are_valid():
    for name in ("pi05", "smolvla", "example"):
        preset = launch.load_preset(name)
        assert "server" in preset
    assert "client" not in launch.load_preset("example")  # server-only sample


import socket
import sys


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def dummy_preset(port):
    return {
        "server": {
            "python": sys.executable,
            "pythonpath": ["."],
            "adapter": "dummy",
            "args": {"fps": "30"},
            "port": port,
        }
    }


def test_build_client_cmd_maps_preset_to_flags():
    preset = {
        "server": {"python": "p", "adapter": "lerobot", "port": 8080},
        "client": {
            "python": "/some/python",
            "pythonpath": ["src", "."],
            "robot_type": "bi_piper_follower",
            "robot_id": "bi_piper",
            "cameras": {"top": {"type": "opencv"}},
            "camera_map": {"top": "camera1"},
            "first_predict_timeout_s": 90,
        },
    }
    cmd, env = launch.build_client_cmd(preset, "stack", ["--fps=15"])
    assert cmd[0] == "/some/python"
    assert cmd[1:3] == ["-m", "deploy.client"]
    assert "--robot.type=bi_piper_follower" in cmd
    assert "--robot.id=bi_piper" in cmd
    assert '--robot.cameras={"top": {"type": "opencv"}}' in cmd
    assert "--server=http://127.0.0.1:8080" in cmd
    assert "--task=stack" in cmd
    assert '--camera_map={"top": "camera1"}' in cmd
    assert "--first_predict_timeout_s=90" in cmd  # non-structural keys forwarded
    assert cmd[-1] == "--fps=15"  # CLI extras win (draccus takes the last value)
    assert str(launch.REPO_ROOT / "src") in env["PYTHONPATH"]


def test_launch_spawns_then_reuses_then_refuses(tmp_path):
    port = free_port()
    path = write_preset(tmp_path, "it", dummy_preset(port))
    proc = launch.main([str(path)])
    try:
        assert proc is not None, "first launch should spawn a server"
        info = launch.server_info(port)
        assert info["name"] == "dummy"
        assert info["checkpoint"] is None
        assert launch.main([str(path)]) is None, "second launch should reuse it"

        other = dummy_preset(port)
        other["server"]["args"]["checkpoint"] = "outputs/does_not_exist"
        other_path = write_preset(tmp_path, "other", other)
        with pytest.raises(SystemExit, match="busy"):
            launch.main([str(other_path)])
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def test_wait_ready_reports_dead_server(tmp_path):
    port = free_port()
    preset = dummy_preset(port)
    preset["server"]["args"] = {"no_such_flag": "boom"}  # DummyAdapter() raises TypeError
    path = write_preset(tmp_path, "dead", preset)
    with pytest.raises(SystemExit, match="server-dead.log"):
        launch.main([str(path)])


def test_wait_ready_timeout_names_pid():
    class FakeProc:
        pid = 12345

        def poll(self):
            return None

    with pytest.raises(SystemExit, match="12345"):
        launch.wait_ready(free_port(), FakeProc(), "fake", timeout_s=1.0)
