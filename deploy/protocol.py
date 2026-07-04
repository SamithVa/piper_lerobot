"""Wire format shared by the deploy server and client.

numpy-only so both sides work in any conda env. Observations travel as
npz archives (images + state + task), chunks as raw .npy bytes.
"""
from __future__ import annotations

import io

import numpy as np

IMG_PREFIX = "img_"


def encode_observation(images: dict[str, np.ndarray], state, task: str) -> bytes:
    arrays = {IMG_PREFIX + key: np.ascontiguousarray(img) for key, img in images.items()}
    arrays["state"] = np.asarray(state, dtype=np.float32)
    arrays["task"] = np.array(task)  # 0-d unicode array, no pickle involved
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def decode_observation(payload: bytes) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        images = {
            key[len(IMG_PREFIX):]: archive[key]
            for key in archive.files
            if key.startswith(IMG_PREFIX)
        }
        state = archive["state"]
        task = str(archive["task"])
    return images, state, task


def encode_chunk(chunk: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(chunk, dtype=np.float32))
    return buf.getvalue()


def decode_chunk(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)
