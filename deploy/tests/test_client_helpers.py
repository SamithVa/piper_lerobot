import numpy as np
import pytest

from deploy.client import action_to_dict, build_images, build_state, resolve_camera_map


def test_build_state_orders_by_motor_keys():
    obs = {"right_joint_1.pos": 2.0, "left_joint_1.pos": 1.0, "top": "img"}
    state = build_state(obs, ["left_joint_1.pos", "right_joint_1.pos"])
    np.testing.assert_array_equal(state, np.array([1.0, 2.0], dtype=np.float32))
    assert state.dtype == np.float32


def test_build_images_maps_robot_cams_to_policy_keys():
    img_top = np.zeros((480, 640, 3), dtype=np.uint8)
    img_lw = np.ones((640, 480, 3), dtype=np.uint8)
    obs = {"top": img_top, "l_wrist": img_lw, "left_joint_1.pos": 0.0}
    images = build_images(obs, {"top": "camera1", "l_wrist": "camera2"})
    assert set(images) == {"camera1", "camera2"}
    assert images["camera1"] is img_top
    assert images["camera2"] is img_lw


def test_action_to_dict_zips_motor_keys():
    row = np.array([0.5, -0.5], dtype=np.float32)
    action = action_to_dict(row, ["left_joint_1.pos", "right_joint_1.pos"])
    assert action == {"left_joint_1.pos": 0.5, "right_joint_1.pos": -0.5}
    assert all(isinstance(v, float) for v in action.values())


def test_resolve_camera_map_defaults_to_identity():
    assert resolve_camera_map({}, ["camera1"], ["camera1", "extra_cam"]) == {"camera1": "camera1"}


def test_resolve_camera_map_validates_policy_keys_covered():
    with pytest.raises(ValueError, match="camera1"):
        resolve_camera_map({"top": "cameraX"}, ["camera1"], ["top"])


def test_resolve_camera_map_validates_robot_cameras_exist():
    with pytest.raises(ValueError, match="no_such_cam"):
        resolve_camera_map({"no_such_cam": "camera1"}, ["camera1"], ["top"])


def test_resolve_camera_map_passthrough_when_valid():
    cmap = {"top": "camera1", "l_wrist": "camera2"}
    assert resolve_camera_map(cmap, ["camera1", "camera2"], ["top", "l_wrist"]) == cmap
