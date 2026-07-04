import numpy as np
import pytest

from deploy import protocol


def test_observation_roundtrip():
    images = {
        "camera1": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "camera2": np.random.randint(0, 256, size=(640, 480, 3), dtype=np.uint8),
    }
    state = np.arange(14, dtype=np.float32)
    payload = protocol.encode_observation(images, state, "stack the cup")
    assert isinstance(payload, bytes)

    images2, state2, task2 = protocol.decode_observation(payload)
    assert task2 == "stack the cup"
    np.testing.assert_array_equal(state2, state)
    assert set(images2) == {"camera1", "camera2"}
    np.testing.assert_array_equal(images2["camera1"], images["camera1"])
    assert images2["camera2"].dtype == np.uint8


def test_observation_accepts_list_state_and_unicode_task():
    payload = protocol.encode_observation({}, [0.0, 1.0], "叠杯子")
    images, state, task = protocol.decode_observation(payload)
    assert images == {}
    assert state.dtype == np.float32
    assert task == "叠杯子"


def test_chunk_roundtrip():
    chunk = np.random.uniform(-1, 1, size=(50, 14)).astype(np.float32)
    out = protocol.decode_chunk(protocol.encode_chunk(chunk))
    np.testing.assert_array_equal(out, chunk)
    assert out.dtype == np.float32


def test_chunk_casts_to_float32():
    out = protocol.decode_chunk(protocol.encode_chunk(np.zeros((5, 3), dtype=np.float64)))
    assert out.dtype == np.float32
