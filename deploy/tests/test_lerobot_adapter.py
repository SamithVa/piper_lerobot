"""Integration test: loads a real checkpoint. Run manually in the lerobot env:

    DEPLOY_IT=1 PYTHONPATH=src:. \
        /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest \
        deploy/tests/test_lerobot_adapter.py -v

Downloads lerobot/smolvla_base (~1GB, cached) and needs a GPU or patience.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEPLOY_IT"),
    reason="integration test: set DEPLOY_IT=1 (downloads checkpoint, needs GPU)",
)

CHECKPOINT = os.environ.get("DEPLOY_IT_CHECKPOINT", "lerobot/smolvla_base")


@pytest.fixture(scope="module")
def adapter():
    from deploy.adapters import make_adapter

    return make_adapter("lerobot", checkpoint=CHECKPOINT, fps="30")


def test_info_is_consistent(adapter):
    info = adapter.info()
    assert info["image_keys"], "policy must have at least one image input"
    assert info["state_dim"] > 0
    assert info["action_dim"] > 0
    assert info["chunk_size"] >= 1
    assert info["fps"] == 30.0


def test_predict_chunk_shape_and_finiteness(adapter):
    info = adapter.info()
    images = {
        key: np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        for key in info["image_keys"]
    }
    state = np.random.uniform(-1, 1, size=(info["state_dim"],)).astype(np.float32)
    chunk = adapter.predict_chunk(images, state, "stack the cup on top of the bowl")
    assert chunk.shape == (info["chunk_size"], info["action_dim"])
    assert np.isfinite(chunk).all()


def test_missing_image_key_raises(adapter):
    info = adapter.info()
    state = np.zeros(info["state_dim"], dtype=np.float32)
    with pytest.raises(ValueError, match="missing images"):
        adapter.predict_chunk({}, state, "task")


def test_reset_does_not_crash(adapter):
    adapter.reset()
