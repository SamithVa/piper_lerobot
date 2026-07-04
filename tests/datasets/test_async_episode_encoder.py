"""Unit tests for background episode video encoding (no CAN hardware, no real ffmpeg encode)."""

import threading
import time
from types import SimpleNamespace

import pytest

from lerobot.datasets.lerobot_dataset import AsyncEpisodeEncoder, LeRobotDataset


# ---------------------------------------------------------------- AsyncEpisodeEncoder


def test_processes_in_fifo_order():
    processed = []
    encoder = AsyncEpisodeEncoder(lambda idx: processed.append(idx))
    for i in range(5):
        encoder.enqueue(i)
    encoder.wait_until_idle()
    assert processed == [0, 1, 2, 3, 4]
    encoder.stop()


def test_wait_until_idle_blocks_until_inflight_done():
    release = threading.Event()
    processed = []

    def slow_encode(idx):
        release.wait(timeout=5)
        processed.append(idx)

    encoder = AsyncEpisodeEncoder(slow_encode)
    encoder.enqueue(0)
    time.sleep(0.05)  # let the worker pick it up
    release.set()
    encoder.wait_until_idle()
    assert processed == [0]
    encoder.stop()


def test_worker_survives_encode_failure():
    processed = []

    def flaky_encode(idx):
        if idx == 1:
            raise RuntimeError("boom")
        processed.append(idx)

    encoder = AsyncEpisodeEncoder(flaky_encode)
    for i in range(3):
        encoder.enqueue(i)
    encoder.wait_until_idle()
    assert processed == [0, 2]
    # Worker is still alive and accepts new work after a failure
    encoder.enqueue(3)
    encoder.wait_until_idle()
    assert processed == [0, 2, 3]
    encoder.stop()


def test_stop_joins_worker_thread():
    encoder = AsyncEpisodeEncoder(lambda idx: None)
    encoder.stop()
    assert not encoder._thread.is_alive()


# ---------------------------------------------------------------- dataset integration


def make_dataset(tmp_path, video_keys=("top", "l_wrist"), encode_fn=None):
    """Bare LeRobotDataset with just the attributes the encode paths need."""
    ds = LeRobotDataset.__new__(LeRobotDataset)
    ds.root = tmp_path
    ds.meta = SimpleNamespace(video_keys=list(video_keys))
    ds._encode_frames_dir = encode_fn or (lambda img_dir, video_path: video_path.write_bytes(b"video"))
    return ds


def make_pngs(ds, episode_index, video_keys):
    for key in video_keys:
        img_dir = ds._get_image_file_dir(episode_index, key)
        img_dir.mkdir(parents=True)
        (img_dir / "frame-000000.png").write_bytes(b"png")


def test_preencode_episode_writes_final_files_and_deletes_pngs(tmp_path):
    ds = make_dataset(tmp_path)
    make_pngs(ds, 3, ds.meta.video_keys)

    ds._preencode_episode(3)

    for key in ds.meta.video_keys:
        dest = ds._preencoded_video_path(key, 3)
        assert dest.is_file() and dest.read_bytes() == b"video"
        assert not ds._get_image_file_dir(3, key).exists()
    # No leftover partial files
    assert not list((tmp_path / "videos_tmp").glob("_part_*"))


def test_preencode_failure_keeps_pngs(tmp_path):
    def failing_encode(img_dir, video_path):
        if "l_wrist" in video_path.name:
            raise RuntimeError("encode failed")
        video_path.write_bytes(b"video")

    ds = make_dataset(tmp_path, encode_fn=failing_encode)
    make_pngs(ds, 0, ds.meta.video_keys)

    with pytest.raises(RuntimeError):
        ds._preencode_episode(0)

    # PNGs are still there for the end-of-session pass to re-encode from
    for key in ds.meta.video_keys:
        assert ds._get_image_file_dir(0, key).is_dir()
    # No final file for the failed camera
    assert not ds._preencoded_video_path("l_wrist", 0).exists()


def test_preencode_skips_already_encoded_camera(tmp_path):
    calls = []

    def counting_encode(img_dir, video_path):
        calls.append(video_path.name)
        video_path.write_bytes(b"video")

    ds = make_dataset(tmp_path, encode_fn=counting_encode)
    make_pngs(ds, 0, ds.meta.video_keys)
    ds._preencoded_video_path("top", 0).parent.mkdir(parents=True)
    ds._preencoded_video_path("top", 0).write_bytes(b"already done")

    ds._preencode_episode(0)

    assert not any("top" in name for name in calls)  # only l_wrist encoded
    assert ds._preencoded_video_path("top", 0).read_bytes() == b"already done"


def test_encode_temporary_uses_preencoded_file(tmp_path):
    def must_not_encode(img_dir, video_path):
        raise AssertionError("should not re-encode when a pre-encoded file exists")

    ds = make_dataset(tmp_path, encode_fn=must_not_encode)
    pre = ds._preencoded_video_path("top", 7)
    pre.parent.mkdir(parents=True)
    pre.write_bytes(b"pre-encoded")

    result = ds._encode_temporary_episode_video("top", 7)

    assert result.read_bytes() == b"pre-encoded"
    assert not pre.exists()  # moved, not copied
    assert result.parent != pre.parent  # lives in its own temp dir (caller rmtree's it)


def test_encode_temporary_falls_back_to_pngs(tmp_path):
    ds = make_dataset(tmp_path)
    make_pngs(ds, 2, ["top"])

    result = ds._encode_temporary_episode_video("top", 2)

    assert result.read_bytes() == b"video"
    assert not ds._get_image_file_dir(2, "top").exists()
