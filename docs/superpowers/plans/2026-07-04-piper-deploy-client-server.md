# Piper Deploy Client-Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reusable client-server pair so any trained policy (SmolVLA, ACT, Pi0.5, future) deploys on the Piper arms by implementing one adapter, with the server in the policy's conda env and the client in base python.

**Architecture:** Top-level `deploy/` package. Server = stdlib `http.server` + numpy (imports in any env) driving a `PolicyAdapter`; a generic `lerobot` adapter covers all lerobot checkpoints. Client = lerobot robot classes (bi_piper_follower) + async-overlap chunk executor over localhost HTTP with npz payloads.

**Tech Stack:** Python stdlib (`http.server`, `urllib`, `threading`), numpy, draccus (client config parsing, already a lerobot dep), pytest. Spec: `docs/superpowers/specs/2026-07-04-piper-deploy-client-server-design.md`.

**Environments (critical):**
- Tests + server run: `LEROBOT_PY=/home/embodied/miniconda3/envs/lerobot/bin/python` (has torch/transformers/pytest)
- Client runs: base python `/home/embodied/miniconda3/bin/python` (has piper_sdk, cameras, draccus)
- Repo: `REPO=/data/wanshan/VLAs/piper_lerobot`. All commands below run from `$REPO`.
- Both `deploy` and `lerobot` import via `PYTHONPATH=src:.`
- Test command used throughout: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`

**File map (final state):**
```
deploy/
├── __init__.py              # empty
├── protocol.py              # npz wire format encode/decode (Task 1)
├── chunking.py              # ChunkExecutor: async-overlap queue logic (Task 4)
├── server.py                # stdlib HTTP server (Task 3)
├── client.py                # robot loop, pure helpers + main (Task 6)
├── adapters/
│   ├── __init__.py          # make_adapter registry (Task 2)
│   ├── base.py              # PolicyAdapter ABC (Task 2)
│   ├── dummy.py             # test/reference adapter, no torch (Task 2)
│   └── lerobot.py           # generic lerobot-checkpoint adapter (Task 5)
├── serve_smolvla.sh         # launcher (Task 7)
├── run_client.sh            # launcher (Task 7)
├── README.md                # usage + how to add a policy (Task 7)
└── tests/
    ├── __init__.py
    ├── test_protocol.py
    ├── test_adapters.py
    ├── test_server.py
    ├── test_chunking.py
    ├── test_client_helpers.py
    └── test_lerobot_adapter.py   # integration, gated by DEPLOY_IT=1
```

---

### Task 1: Wire protocol (`deploy/protocol.py`)

**Files:**
- Create: `deploy/__init__.py` (empty), `deploy/tests/__init__.py` (empty)
- Create: `deploy/protocol.py`
- Test: `deploy/tests/test_protocol.py`

- [ ] **Step 1.1: Write the failing test**

```python
# deploy/tests/test_protocol.py
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
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_protocol.py -v`
Expected: FAIL / errors with `ModuleNotFoundError: No module named 'deploy.protocol'` (after creating the two empty `__init__.py` files).

- [ ] **Step 1.3: Write the implementation**

```python
# deploy/protocol.py
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
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_protocol.py -v`
Expected: 4 PASSED

- [ ] **Step 1.5: Commit**

```bash
git add deploy/__init__.py deploy/protocol.py deploy/tests/
git commit -m "feat(deploy): npz wire protocol for observation/chunk transport"
```

---

### Task 2: Adapter interface, dummy adapter, registry

**Files:**
- Create: `deploy/adapters/__init__.py`, `deploy/adapters/base.py`, `deploy/adapters/dummy.py`
- Test: `deploy/tests/test_adapters.py`

- [ ] **Step 2.1: Write the failing test**

```python
# deploy/tests/test_adapters.py
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
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_adapters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.adapters'`

- [ ] **Step 2.3: Write the implementation (three files)**

```python
# deploy/adapters/base.py
"""The one interface a policy must implement to be deployable on the Piper."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PolicyAdapter(ABC):
    """Serve one policy. Implement these methods in whatever conda env the
    policy needs; the server, transport, and robot client never change."""

    @abstractmethod
    def info(self) -> dict:
        """Static metadata:
        {"name": str, "image_keys": list[str], "state_dim": int,
         "action_dim": int, "chunk_size": int, "fps": float}"""

    @abstractmethod
    def predict_chunk(
        self, images: dict[str, np.ndarray], state: np.ndarray, task: str
    ) -> np.ndarray:
        """images: HWC uint8 RGB keyed by the policy's image keys;
        state: (state_dim,). Returns (chunk_size, action_dim) float32."""

    def reset(self) -> None:
        """Clear per-episode state (action queues etc.). Optional."""
```

```python
# deploy/adapters/dummy.py
"""Deterministic stand-in policy: no torch, instant. Used by the deploy test
suite and as the smallest reference for writing new adapters.

Action row t is the constant vector [t, t, ..., t], so tests can tell which
step of a chunk got executed.
"""
from __future__ import annotations

import numpy as np

from .base import PolicyAdapter


class DummyAdapter(PolicyAdapter):
    def __init__(
        self,
        state_dim=14,
        action_dim=14,
        chunk_size=10,
        fps=30.0,
        image_keys="camera1,camera2",
        fail=False,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.fps = float(fps)
        self.image_keys = [key for key in str(image_keys).split(",") if key]
        self.fail = bool(int(fail)) if isinstance(fail, str) else bool(fail)
        self.reset_count = 0

    def info(self) -> dict:
        return {
            "name": "dummy",
            "image_keys": self.image_keys,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "fps": self.fps,
        }

    def predict_chunk(self, images, state, task) -> np.ndarray:
        if self.fail:
            raise RuntimeError("dummy failure requested")
        steps = np.arange(self.chunk_size, dtype=np.float32)[:, None]
        return np.repeat(steps, self.action_dim, axis=1)

    def reset(self) -> None:
        self.reset_count += 1
```

```python
# deploy/adapters/__init__.py
"""Adapter registry. Imports are lazy so listing/creating the dummy adapter
never pulls torch, and the lerobot adapter only imports in the server env."""
from __future__ import annotations

from .base import PolicyAdapter as PolicyAdapter

_ADAPTERS = ("dummy", "lerobot")


def make_adapter(name: str, **kwargs) -> PolicyAdapter:
    if name == "dummy":
        from .dummy import DummyAdapter

        return DummyAdapter(**kwargs)
    if name == "lerobot":
        from .lerobot import LerobotAdapter

        return LerobotAdapter(**kwargs)
    raise ValueError(f"Unknown adapter '{name}'. Available: {', '.join(_ADAPTERS)}")
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_adapters.py -v`
Expected: 4 PASSED

- [ ] **Step 2.5: Commit**

```bash
git add deploy/adapters/ deploy/tests/test_adapters.py
git commit -m "feat(deploy): PolicyAdapter interface, dummy adapter, registry"
```

---

### Task 3: HTTP server (`deploy/server.py`)

**Files:**
- Create: `deploy/server.py`
- Test: `deploy/tests/test_server.py`

- [ ] **Step 3.1: Write the failing test**

```python
# deploy/tests/test_server.py
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from deploy import protocol
from deploy.adapters.dummy import DummyAdapter
from deploy.server import create_server, parse_args


@pytest.fixture
def running_server():
    adapter = DummyAdapter(state_dim=14, action_dim=14, chunk_size=10)
    server = create_server(adapter, host="127.0.0.1", port=0)  # 0 = ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, adapter
    server.shutdown()


def _post(url, body=b"", timeout=5.0):
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def test_info(running_server):
    base, adapter = running_server
    with urllib.request.urlopen(base + "/info", timeout=5.0) as resp:
        info = json.loads(resp.read())
    assert info == adapter.info()


def test_predict_roundtrip(running_server):
    base, _ = running_server
    payload = protocol.encode_observation(
        {"camera1": np.zeros((480, 640, 3), dtype=np.uint8)},
        np.zeros(14, dtype=np.float32),
        "stack",
    )
    chunk = protocol.decode_chunk(_post(base + "/predict", payload))
    assert chunk.shape == (10, 14)


def test_reset(running_server):
    base, adapter = running_server
    _post(base + "/reset")
    assert adapter.reset_count == 1


def test_predict_error_returns_500_with_traceback(running_server):
    base, adapter = running_server
    adapter.fail = True
    payload = protocol.encode_observation({}, np.zeros(14), "")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base + "/predict", payload)
    assert excinfo.value.code == 500
    assert b"dummy failure requested" in excinfo.value.read()


def test_unknown_path_404(running_server):
    base, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(base + "/nope", timeout=5.0)
    assert excinfo.value.code == 404


def test_parse_args_forwards_extra_flags():
    args, kwargs = parse_args(
        ["--adapter=lerobot", "--port=9000", "--checkpoint=/some/path", "--device=cuda"]
    )
    assert args.adapter == "lerobot"
    assert args.port == 9000
    assert kwargs == {"checkpoint": "/some/path", "device": "cuda"}
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.server'`

- [ ] **Step 3.3: Write the implementation**

```python
# deploy/server.py
"""Policy inference server for Piper deployment.

stdlib + numpy only, so it runs inside ANY conda env — launch it from the env
the policy needs and point the client at it.

Usage:
    python -m deploy.server --adapter=lerobot --port=8080 \
        --checkpoint=outputs/train/.../pretrained_model --device=cuda --fps=30

Flags other than --adapter/--host/--port are forwarded to the adapter
constructor as string keyword arguments.

Endpoints:
    GET  /info     -> adapter.info() as JSON
    POST /predict  -> body: protocol.encode_observation(...); reply: encode_chunk(...)
    POST /reset    -> clears adapter episode state
Any adapter exception -> HTTP 500 with the traceback as text; the server stays up.
"""
from __future__ import annotations

import json
import threading
import traceback
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deploy import protocol
from deploy.adapters import make_adapter


def make_handler(adapter):
    lock = threading.Lock()  # serialize GPU inference across connections

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, content_type="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/info":
                self._send(200, json.dumps(adapter.info()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if self.path == "/predict":
                    images, state, task = protocol.decode_observation(body)
                    with lock:
                        chunk = adapter.predict_chunk(images, state, task)
                    self._send(200, protocol.encode_chunk(chunk))
                elif self.path == "/reset":
                    with lock:
                        adapter.reset()
                    self._send(200, b"", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception:
                self._send(500, traceback.format_exc().encode(), "text/plain")

        def log_message(self, fmt, *args):
            pass  # keep per-request noise out of the console

    return Handler


def create_server(adapter, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(adapter))


def parse_args(argv=None):
    parser = ArgumentParser(description="Piper deploy policy server")
    parser.add_argument("--adapter", required=True, help="adapter name, e.g. lerobot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args, extra = parser.parse_known_args(argv)
    kwargs = {}
    for item in extra:
        if not (item.startswith("--") and "=" in item):
            parser.error(f"adapter flags must look like --key=value, got: {item}")
        key, value = item[2:].split("=", 1)
        kwargs[key] = value
    return args, kwargs


def main(argv=None):
    args, kwargs = parse_args(argv)
    adapter = make_adapter(args.adapter, **kwargs)
    print(f"[deploy.server] adapter={args.adapter} info={adapter.info()}")
    server = create_server(adapter, args.host, args.port)
    print(f"[deploy.server] listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[deploy.server] shutting down")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_server.py -v`
Expected: 6 PASSED

- [ ] **Step 3.5: Commit**

```bash
git add deploy/server.py deploy/tests/test_server.py
git commit -m "feat(deploy): stdlib HTTP policy server with adapter forwarding"
```

---

### Task 4: Async-overlap chunk executor (`deploy/chunking.py`)

**Files:**
- Create: `deploy/chunking.py`
- Test: `deploy/tests/test_chunking.py`

- [ ] **Step 4.1: Write the failing test**

```python
# deploy/tests/test_chunking.py
import numpy as np
import pytest

from deploy.chunking import ChunkExecutor


def chunk(n=10, dim=2, start=0.0):
    # row t = [start + t, start + t]
    steps = np.arange(n, dtype=np.float32)[:, None] + start
    return np.repeat(steps, dim, axis=1)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        ChunkExecutor(chunk_threshold=0.0)
    with pytest.raises(ValueError):
        ChunkExecutor(chunk_threshold=1.5)


def test_empty_queue_requests_immediately_and_returns_none():
    ex = ChunkExecutor()
    assert ex.next_action() is None
    assert ex.should_request() is True


def test_no_double_request_while_in_flight():
    ex = ChunkExecutor()
    assert ex.should_request()
    ex.mark_requested()
    assert ex.in_flight
    assert ex.should_request() is False


def test_executes_chunk_in_order():
    ex = ChunkExecutor()
    ex.mark_requested()
    ex.on_chunk(chunk(3))
    assert not ex.in_flight
    np.testing.assert_array_equal(ex.next_action(), [0.0, 0.0])
    np.testing.assert_array_equal(ex.next_action(), [1.0, 1.0])
    np.testing.assert_array_equal(ex.next_action(), [2.0, 2.0])
    assert ex.next_action() is None


def test_should_request_at_threshold():
    ex = ChunkExecutor(chunk_threshold=0.5)
    ex.mark_requested()
    ex.on_chunk(chunk(10))
    for _ in range(4):  # consumed 4/10 < 0.5
        ex.next_action()
    assert ex.should_request() is False
    ex.next_action()  # consumed 5/10 >= 0.5
    assert ex.should_request() is True


def test_on_chunk_skips_ticks_elapsed_since_request():
    ex = ChunkExecutor(chunk_threshold=0.5)
    ex.mark_requested()
    ex.on_chunk(chunk(10))
    for _ in range(5):
        ex.next_action()
    ex.mark_requested()  # obs captured now (tick 5)
    ex.next_action()  # 2 more ticks pass while inference runs
    ex.next_action()
    ex.on_chunk(chunk(10, start=100.0))
    # new chunk row 0 corresponds to the obs tick; 2 ticks elapsed -> start at row 2
    np.testing.assert_array_equal(ex.next_action(), [102.0, 102.0])


def test_on_chunk_all_rows_stale_leaves_queue_dry():
    ex = ChunkExecutor()
    ex.mark_requested()
    ex.on_chunk(chunk(2))
    for _ in range(5):  # burn well past the 2 delivered rows
        ex.next_action()
    ex.mark_requested()
    for _ in range(3):
        ex.next_action()
    ex.on_chunk(chunk(3))  # 3 elapsed >= 3 rows -> nothing usable
    assert ex.next_action() is None
    assert ex.should_request() is True  # must be able to recover


def test_request_failed_clears_in_flight():
    ex = ChunkExecutor()
    ex.mark_requested()
    assert ex.should_request() is False
    ex.on_request_failed()
    assert ex.should_request() is True
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.chunking'`

- [ ] **Step 4.3: Write the implementation**

```python
# deploy/chunking.py
"""Pure queue logic for the async-overlap control loop. No I/O, no threads —
the client calls everything from its single control-loop thread.

Per control tick:
    action = executor.next_action()          # None -> queue dry, hold position
    if executor.should_request():
        # capture obs NOW, then:
        executor.mark_requested()
        # ... send obs to the server on a background thread ...
    # when the reply lands (any later tick):
    executor.on_chunk(chunk)                  # or executor.on_request_failed()
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ChunkExecutor:
    def __init__(self, chunk_threshold: float = 0.5):
        if not 0.0 < chunk_threshold <= 1.0:
            raise ValueError(f"chunk_threshold must be in (0, 1], got {chunk_threshold}")
        self.chunk_threshold = chunk_threshold
        self._queue: deque = deque()
        self._chunk_len = 0  # length of the chunk the current queue came from
        self._tick = 0
        self._request_tick: int | None = None

    @property
    def in_flight(self) -> bool:
        return self._request_tick is not None

    def next_action(self):
        """Advance one control tick; return the next action row or None if dry."""
        self._tick += 1
        if self._queue:
            return self._queue.popleft()
        return None

    def should_request(self) -> bool:
        if self.in_flight:
            return False
        if self._chunk_len == 0 or not self._queue:
            return True
        consumed = 1.0 - len(self._queue) / self._chunk_len
        return consumed >= self.chunk_threshold

    def mark_requested(self) -> None:
        """Call at the tick the observation was captured."""
        self._request_tick = self._tick

    def on_chunk(self, chunk: np.ndarray) -> None:
        """Install a fresh chunk, skipping the rows whose time already passed.

        Row 0 of the chunk is the action for the observation's tick; if k ticks
        elapsed between capture and arrival, rows [0, k) are stale.
        """
        elapsed = 0 if self._request_tick is None else self._tick - self._request_tick
        self._request_tick = None
        rows = list(np.asarray(chunk))
        usable = rows[elapsed:]
        self._queue = deque(usable)
        self._chunk_len = len(usable)

    def on_request_failed(self) -> None:
        self._request_tick = None
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_chunking.py -v`
Expected: 8 PASSED

- [ ] **Step 4.5: Commit**

```bash
git add deploy/chunking.py deploy/tests/test_chunking.py
git commit -m "feat(deploy): async-overlap chunk executor"
```

---

### Task 5: Generic lerobot adapter (`deploy/adapters/lerobot.py`)

**Files:**
- Create: `deploy/adapters/lerobot.py`
- Test: `deploy/tests/test_lerobot_adapter.py` (integration, gated — model download + GPU)

Reference: this generalizes `packaged/pi05_deploy/_core.py` (same load-and-predict
path) from pi05-only to any lerobot policy via `get_policy_class`.

- [ ] **Step 5.1: Write the gated integration test**

```python
# deploy/tests/test_lerobot_adapter.py
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
```

- [ ] **Step 5.2: Verify the gate works (test skips without DEPLOY_IT)**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_lerobot_adapter.py -v`
Expected: 4 SKIPPED (and an import error at this point is fine — the module doesn't exist yet; collection must still show the skip once Step 5.3 lands. If collection fails on the missing module, proceed to 5.3 and re-run.)

- [ ] **Step 5.3: Write the implementation**

```python
# deploy/adapters/lerobot.py
"""Generic adapter for ANY lerobot policy checkpoint (smolvla, act, pi05, ...).

Run the server from a conda env that can import this repo's lerobot
(PYTHONPATH=<repo>/src:<repo>) plus the policy's own deps.

The checkpoint may be a Hub id (samithva/smolvla_bimanual_stack_cup_bowl) or a
local path (outputs/train/<job>/checkpoints/last/pretrained_model).
"""
from __future__ import annotations

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from .base import PolicyAdapter

IMG_PREFIX = "observation.images."


class LerobotAdapter(PolicyAdapter):
    def __init__(self, checkpoint: str = "", device: str = "", fps="30"):
        if not checkpoint:
            raise ValueError("--checkpoint=<hub id or local path> is required")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = checkpoint
        self.device = device
        self.fps = float(fps)

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(checkpoint)
        policy.to(device)
        policy.eval()
        self.policy = policy

        self.preprocess, self.postprocess = make_pre_post_processors(
            policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        pcfg = policy.config
        self._image_keys = [
            key[len(IMG_PREFIX):]
            for key, feat in pcfg.input_features.items()
            if feat.type == FeatureType.VISUAL
        ]
        self._state_dim = pcfg.input_features["observation.state"].shape[0]
        self._action_dim = pcfg.output_features["action"].shape[0]
        self._chunk_size = int(getattr(pcfg, "chunk_size", getattr(pcfg, "n_action_steps", 1)))

    def info(self) -> dict:
        return {
            "name": f"lerobot:{self.policy.config.type}:{self.checkpoint}",
            "image_keys": self._image_keys,
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "chunk_size": self._chunk_size,
            "fps": self.fps,
        }

    def _image_tensor(self, img) -> torch.Tensor:
        t = torch.as_tensor(np.ascontiguousarray(img))
        if t.ndim == 3 and t.shape[0] not in (1, 3):  # HWC -> CHW
            t = t.permute(2, 0, 1)
        t = t.float()
        if float(t.max()) > 1.5:  # uint8 [0,255] -> [0,1]
            t = t / 255.0
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_chunk(self, images, state, task) -> np.ndarray:
        missing = [key for key in self._image_keys if key not in images]
        if missing:
            raise ValueError(f"missing images for keys {missing}; got {sorted(images)}")
        obs = {IMG_PREFIX + key: self._image_tensor(images[key]) for key in self._image_keys}
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(-1))
        obs["observation.state"] = state_t.unsqueeze(0).to(self.device)
        obs["task"] = task
        batch = self.preprocess(obs)
        chunk = self.policy.predict_action_chunk(batch)
        chunk = self.postprocess(chunk)
        return chunk.squeeze(0).detach().float().cpu().numpy()

    def reset(self) -> None:
        self.policy.reset()
```

- [ ] **Step 5.4: Run the integration test for real**

Run: `DEPLOY_IT=1 PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_lerobot_adapter.py -v`
Expected: 4 PASSED (first run downloads `lerobot/smolvla_base`; allow a few minutes)

Also re-run without the gate to confirm it skips:
`PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_lerobot_adapter.py -v` → 4 SKIPPED

- [ ] **Step 5.5: Commit**

```bash
git add deploy/adapters/lerobot.py deploy/tests/test_lerobot_adapter.py
git commit -m "feat(deploy): generic lerobot-checkpoint adapter"
```

---

### Task 6: Robot client (`deploy/client.py`)

**Files:**
- Create: `deploy/client.py`
- Test: `deploy/tests/test_client_helpers.py` (pure helpers only; the loop is exercised on the real robot in Task 8)

- [ ] **Step 6.1: Write the failing test for the pure helpers**

```python
# deploy/tests/test_client_helpers.py
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
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_client_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.client'`

- [ ] **Step 6.3: Write the implementation**

```python
# deploy/client.py
"""Robot-side deployment client for the Piper arms.

Runs in base python (piper_sdk + cameras) with this repo's lerobot on the
path; talks to a deploy.server over HTTP. Async-overlap execution: while a
chunk is being executed, the next observation is sent ~halfway through so the
arms never pause for inference.

Example (bimanual):
    PYTHONPATH=src:. /home/embodied/miniconda3/bin/python -m deploy.client \
        --robot.type=bi_piper_follower \
        --robot.id=bi_piper \
        --robot.cameras='{"top": {"type": "opencv", ...}, ...}' \
        --server=http://127.0.0.1:8080 \
        --task="Stack the cup on top of the bowl." \
        --camera_map='{"top": "camera1", "l_wrist": "camera2", "r_wrist": "camera3"}' \
        --duration_s=60
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import draccus
import numpy as np

from deploy import protocol
from deploy.chunking import ChunkExecutor
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401  (register robot configs)
    RobotConfig,
    bi_piper_follower,
    make_robot_from_config,
    piper_follower,
)
from lerobot.utils.utils import init_logging

# ---------- pure helpers (unit-tested) ----------


def build_state(obs: dict, motor_keys: list[str]) -> np.ndarray:
    """Observation dict -> state vector in the same motor order recording used."""
    return np.array([obs[key] for key in motor_keys], dtype=np.float32)


def build_images(obs: dict, camera_map: dict[str, str]) -> dict[str, np.ndarray]:
    """{robot_cam_name: policy_image_key} -> {policy_image_key: image}."""
    return {policy_key: obs[cam] for cam, policy_key in camera_map.items()}


def action_to_dict(row: np.ndarray, motor_keys: list[str]) -> dict[str, float]:
    return {key: float(value) for key, value in zip(motor_keys, row)}


def resolve_camera_map(
    camera_map: dict[str, str], policy_keys: list[str], robot_cams: list[str]
) -> dict[str, str]:
    """Validate/derive the robot-camera -> policy-key mapping before touching arms."""
    if not camera_map:
        camera_map = {key: key for key in policy_keys if key in robot_cams}
    unknown = [cam for cam in camera_map if cam not in robot_cams]
    if unknown:
        raise ValueError(f"camera_map names robot cameras that don't exist: {unknown} (robot has {robot_cams})")
    uncovered = [key for key in policy_keys if key not in camera_map.values()]
    if uncovered:
        raise ValueError(
            f"camera_map does not provide policy image keys: {uncovered}. "
            f"Pass --camera_map mapping robot cameras {robot_cams} onto them."
        )
    return camera_map


# ---------- HTTP ----------


def http_get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, body: bytes = b"", timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------- main loop ----------


@dataclass
class DeployClientConfig:
    robot: RobotConfig
    server: str = "http://127.0.0.1:8080"
    task: str = ""
    duration_s: float = 60.0
    fps: float = 0.0  # 0 -> use the server's fps
    chunk_threshold: float = 0.5
    predict_timeout_s: float = 15.0
    # {"robot_cam_name": "policy_image_key"}, e.g. {"top": "camera1"}
    camera_map: dict[str, str] = field(default_factory=dict)


def capture_payload(robot, camera_map, motor_keys, task) -> bytes:
    obs = robot.get_observation()
    return protocol.encode_observation(
        build_images(obs, camera_map), build_state(obs, motor_keys), task
    )


@draccus.wrap()
def main(cfg: DeployClientConfig):
    init_logging()

    info = http_get_json(cfg.server + "/info")
    logging.info(f"policy: {info}")
    fps = cfg.fps or float(info["fps"])
    period = 1.0 / fps

    robot = make_robot_from_config(cfg.robot)
    motor_keys = list(robot.action_features)
    camera_map = resolve_camera_map(
        cfg.camera_map, info["image_keys"], list(robot.cameras)
    )
    if info["state_dim"] != len(motor_keys):
        raise SystemExit(
            f"state_dim mismatch: policy expects {info['state_dim']}, robot has {len(motor_keys)} motors"
        )
    if info["action_dim"] != len(motor_keys):
        raise SystemExit(
            f"action_dim mismatch: policy outputs {info['action_dim']}, robot has {len(motor_keys)} motors"
        )

    robot.connect()
    executor = ChunkExecutor(cfg.chunk_threshold)
    pending: dict = {}  # worker thread -> loop: {"chunk": ...} or {"error": ...}

    def post_predict_async(payload: bytes):
        def work():
            try:
                raw = http_post(cfg.server + "/predict", payload, cfg.predict_timeout_s)
                pending["chunk"] = protocol.decode_chunk(raw)
            except Exception as exc:  # noqa: BLE001 — must never kill the worker silently
                pending["error"] = exc

        threading.Thread(target=work, daemon=True).start()

    try:
        http_post(cfg.server + "/reset")

        # Blocking first chunk so the loop starts with actions in hand.
        logging.info("requesting first chunk...")
        payload = capture_payload(robot, camera_map, motor_keys, cfg.task)
        executor.mark_requested()
        executor.on_chunk(
            protocol.decode_chunk(http_post(cfg.server + "/predict", payload, cfg.predict_timeout_s))
        )
        logging.info(f"running at {fps:.0f} fps for {cfg.duration_s:.0f}s — Ctrl-C to stop")

        last_action: dict | None = None
        dry_ticks = 0
        t_end = time.perf_counter() + cfg.duration_s
        next_t = time.perf_counter()
        while time.perf_counter() < t_end:
            if "chunk" in pending:
                executor.on_chunk(pending.pop("chunk"))
            if "error" in pending:
                logging.warning(f"/predict failed: {pending.pop('error')}")
                executor.on_request_failed()

            row = executor.next_action()
            if row is not None:
                last_action = action_to_dict(row, motor_keys)
                robot.send_action(last_action)
                dry_ticks = 0
            elif last_action is not None:
                robot.send_action(last_action)  # hold position while recovering
                dry_ticks += 1
                if dry_ticks % int(fps) == 1:
                    logging.warning("action queue dry — holding position")

            if executor.should_request():
                payload = capture_payload(robot, camera_map, motor_keys, cfg.task)
                executor.mark_requested()
                post_predict_async(payload)

            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        logging.info("interrupted — stopping")
    finally:
        robot.disconnect()
        logging.info("robot disconnected")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_client_helpers.py -v`
Expected: 7 PASSED

- [ ] **Step 6.5: No-robot end-to-end sanity check (dummy adapter, no arms)**

Start a dummy server, then verify the client passes /info validation and fails
only at CAN connect (proving config parsing, camera_map validation, and HTTP work):

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m deploy.server \
  --adapter=dummy --port=8123 --state_dim=14 --action_dim=14 --image_keys=camera1 &
sleep 1
PYTHONPATH=src:. /home/embodied/miniconda3/bin/python -m deploy.client \
  --robot.type=bi_piper_follower --robot.id=deploy_sanity \
  --server=http://127.0.0.1:8123 --task=test \
  --camera_map='{}' --duration_s=1 ; echo "exit=$?"
kill %1
```

Expected: client prints the policy info line, then raises about camera_map
(`camera_map does not provide policy image keys: ['camera1']` — the robot was
configured with no cameras) BEFORE touching the arms; exit code is non-zero.
This confirms fail-fast validation. (With arms + cameras attached, Task 8 does
the real run.)

- [ ] **Step 6.6: Commit**

```bash
git add deploy/client.py deploy/tests/test_client_helpers.py
git commit -m "feat(deploy): async-overlap robot client"
```

---

### Task 7: Launchers + README

**Files:**
- Create: `deploy/serve_smolvla.sh`, `deploy/run_client.sh`, `deploy/README.md`

- [ ] **Step 7.1: Write `deploy/serve_smolvla.sh`**

```bash
#!/bin/bash
# Serve a SmolVLA checkpoint for Piper deployment (lerobot conda env).
# Usage: bash deploy/serve_smolvla.sh [checkpoint] [port]
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
PY=/home/embodied/miniconda3/envs/lerobot/bin/python

CHECKPOINT="${1:-$REPO/outputs/train/smolvla_bimanual_stack_cup_bowl/checkpoints/last/pretrained_model}"
PORT="${2:-8080}"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO/src:$REPO" "$PY" -m deploy.server \
  --adapter=lerobot \
  --port="$PORT" \
  --checkpoint="$CHECKPOINT" \
  --device=cuda \
  --fps=30
```

- [ ] **Step 7.2: Write `deploy/run_client.sh`**

Camera configs copied from `record_bimanual.sh` (identical hardware setup);
camera_map matches the training rename map in `train_smolvla_bimanual.sh`
(top→camera1, l_wrist→camera2, r_wrist→camera3).

```bash
#!/bin/bash
# Run the Piper deployment client (base python: piper_sdk + cameras).
# Usage: bash deploy/run_client.sh ["task"] [duration_s] [server]
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
PY=/home/embodied/miniconda3/bin/python

TASK="${1:-Stack the cup on top of the bowl.}"
DURATION="${2:-60}"
SERVER="${3:-http://127.0.0.1:8080}"

CAMERAS='{
  "l_wrist": {"type": "opencv", "index_or_path": "/dev/l_wrist", "width": 480, "height": 640, "fps": 120, "rotation": -90, "fourcc": "MJPG"},
  "top":     {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 120, "rotation": 0,   "fourcc": "MJPG"},
  "r_wrist": {"type": "opencv", "index_or_path": "/dev/r_wrist", "width": 480, "height": 640, "fps": 120, "rotation": 90,  "fourcc": "MJPG"}
}'

CAMERA_MAP='{"top": "camera1", "l_wrist": "camera2", "r_wrist": "camera3"}'

PYTHONPATH="$REPO/src:$REPO" "$PY" -m deploy.client \
  --robot.type=bi_piper_follower \
  --robot.id=bi_piper \
  --robot.cameras="$CAMERAS" \
  --server="$SERVER" \
  --task="$TASK" \
  --camera_map="$CAMERA_MAP" \
  --duration_s="$DURATION"
```

- [ ] **Step 7.3: Write `deploy/README.md`**

```markdown
# deploy — policy client-server for the Piper arms

Run any trained policy on the Piper robot. The **server** hosts the policy in
whatever conda env it needs; the **client** always runs in base python
(piper_sdk + cameras) and streams action chunks over localhost HTTP with no
pauses between chunks (async overlap).

Design spec: `docs/superpowers/specs/2026-07-04-piper-deploy-client-server-design.md`

## Quick start (SmolVLA, bimanual)

```bash
# terminal 1 — policy server (lerobot env)
bash deploy/serve_smolvla.sh                       # or: serve_smolvla.sh <checkpoint> <port>

# terminal 2 — robot client (base python)
bash deploy/run_client.sh "Stack the cup on top of the bowl." 60
```

`run_client.sh` maps cameras per the training rename map
(top→camera1, l_wrist→camera2, r_wrist→camera3).

## Any lerobot checkpoint

The `lerobot` adapter serves any lerobot policy (smolvla / act / pi05):

```bash
PYTHONPATH=src:. <env-python> -m deploy.server --adapter=lerobot \
    --checkpoint=<hub-id-or-local-path> --device=cuda --fps=30
```

Pick `<env-python>` = the env that can run that policy (pi05 needs the patched
transformers fork env, etc.).

## Adding a NON-lerobot policy (openpi, GR00T, ...)

1. Copy `deploy/adapters/dummy.py` to `deploy/adapters/<name>.py`; implement
   `info()`, `predict_chunk(images, state, task)`, `reset()` (see
   `deploy/adapters/base.py` for the contract).
2. Register it in `deploy/adapters/__init__.py` (`make_adapter`).
3. Launch `deploy.server --adapter=<name> ...` from the policy's env.

The client, protocol, and control loop stay untouched.

## Protocol

- `GET /info` → `{"name", "image_keys", "state_dim", "action_dim", "chunk_size", "fps"}`
- `POST /predict` — body: npz of images (`img_<key>`, HWC uint8 RGB) + `state`
  (float32) + `task` (str); reply: `.npy` bytes, shape `(chunk_size, action_dim)`
- `POST /reset` — clear episode state
- Errors → HTTP 500 with the Python traceback as the body

## Tests

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v
# integration (downloads smolvla_base, needs GPU):
DEPLOY_IT=1 PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python \
    -m pytest deploy/tests/test_lerobot_adapter.py -v
```
```

- [ ] **Step 7.4: Make the launchers executable and sanity-check bash syntax**

Run: `chmod +x deploy/serve_smolvla.sh deploy/run_client.sh && bash -n deploy/serve_smolvla.sh && bash -n deploy/run_client.sh && echo OK`
Expected: `OK`

- [ ] **Step 7.5: Commit**

```bash
git add deploy/serve_smolvla.sh deploy/run_client.sh deploy/README.md
git commit -m "feat(deploy): launchers and README"
```

---

### Task 8: Full verification

- [ ] **Step 8.1: Run the whole deploy test suite**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
Expected: all PASSED except `test_lerobot_adapter.py` (4 SKIPPED without `DEPLOY_IT`)

- [ ] **Step 8.2: Live server end-to-end with the dummy adapter**

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m deploy.server \
  --adapter=dummy --port=8124 &
sleep 1
curl -s http://127.0.0.1:8124/info
kill %1
```

Expected: JSON info blob printed.

- [ ] **Step 8.3: Real-robot run (requires arms powered + CAN up + cameras)**

This step needs the physical robot and a trained checkpoint — coordinate with
the user before moving the arms. With `bash utils/activate_all_can.sh` done and
a finished `train_smolvla_bimanual.sh` checkpoint:

```bash
# terminal 1
bash deploy/serve_smolvla.sh
# terminal 2
bash deploy/run_client.sh "Stack the cup on top of the bowl." 30
```

Expected: arms execute the task; log shows no "queue dry" warnings at steady
state (inference keeps up via overlap). If this cannot be run now (no
checkpoint / robot off), note it and leave this step unchecked.

- [ ] **Step 8.4: Final commit if anything changed**

```bash
git status --short   # commit any leftovers with an appropriate message
```
