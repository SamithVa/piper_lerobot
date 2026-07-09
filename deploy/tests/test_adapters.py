import numpy as np
import pytest

from deploy.adapters import make_adapter


def test_registry_unknown_name():
    with pytest.raises(ValueError, match="Unknown adapter"):
        make_adapter("nope")


def test_dummy_info_and_chunk():
    adapter = make_adapter(
        "dummy", state_dim=14, action_dim=14, chunk_size=10, fps=30, image_keys="camera1,camera2"
    )
    info = adapter.info()
    assert info["image_keys"] == ["camera1", "camera2"]
    assert info["state_dim"] == 14
    assert info["action_dim"] == 14
    assert info["chunk_size"] == 10
    assert info["fps"] == 30.0

    chunk = adapter.predict_chunk({}, np.zeros(14, dtype=np.float32), "task")
    assert chunk.shape == (10, 14)
    assert chunk.dtype == np.float32
    # row t is constant t -> lets tests identify which action step executed
    np.testing.assert_array_equal(chunk[3], np.full(14, 3.0, dtype=np.float32))


def test_dummy_string_kwargs():
    # server forwards CLI flags as strings; ctor must coerce
    adapter = make_adapter("dummy", state_dim="7", chunk_size="5", fps="15")
    assert adapter.info()["state_dim"] == 7
    assert adapter.info()["chunk_size"] == 5


def test_dummy_fail_flag_and_reset():
    adapter = make_adapter("dummy", fail="1")
    with pytest.raises(RuntimeError, match="dummy failure"):
        adapter.predict_chunk({}, np.zeros(14), "")
    adapter.reset()
    assert adapter.reset_count == 1


def test_dummy_info_reports_null_checkpoint():
    assert make_adapter("dummy").info()["checkpoint"] is None
