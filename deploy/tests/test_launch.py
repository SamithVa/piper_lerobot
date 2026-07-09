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
