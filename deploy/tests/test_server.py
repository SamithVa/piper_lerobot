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


def test_info_error_returns_500(running_server):
    base, adapter = running_server
    adapter.info = lambda: (_ for _ in ()).throw(RuntimeError("info boom"))
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(base + "/info", timeout=5.0)
    assert excinfo.value.code == 500
    assert b"info boom" in excinfo.value.read()


def test_unknown_path_404(running_server):
    base, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(base + "/nope", timeout=5.0)
    assert excinfo.value.code == 404


def test_predict_forwards_meta_to_adapter(running_server):
    base, adapter = running_server
    payload = protocol.encode_observation(
        {k: np.zeros((8, 8, 3), np.uint8) for k in adapter.image_keys},
        np.zeros(adapter.state_dim, np.float32), "t", consumed=7, delay_ticks=4,
    )
    _post(base + "/predict", payload)
    assert adapter.last_meta == {"consumed": 7, "delay_ticks": 4}


class ThreadRecordingAdapter(DummyAdapter):
    """Records the thread predict_chunk runs on, so tests can assert every
    /predict lands on the same dedicated inference thread. Thread *name* is
    the reliable signal here (not just get_ident()): the OS can recycle a
    native thread id after a handler thread exits, so two sequential
    per-connection handler threads could coincidentally share an ident even
    though they are genuinely different threads. Python's anonymous-thread
    name counter, in contrast, never repeats within a process."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predict_thread_ids = []
        self.predict_thread_names = []

    def predict_chunk(self, *args, **kwargs):
        current = threading.current_thread()
        self.predict_thread_ids.append(threading.get_ident())
        self.predict_thread_names.append(current.name)
        return super().predict_chunk(*args, **kwargs)


def test_predict_runs_on_single_dedicated_thread_across_requests():
    adapter = ThreadRecordingAdapter(state_dim=14, action_dim=14, chunk_size=10)
    server = create_server(adapter, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        payload = protocol.encode_observation(
            {k: np.zeros((8, 8, 3), np.uint8) for k in adapter.image_keys},
            np.zeros(adapter.state_dim, np.float32), "t",
        )
        # Two requests over separate connections exercise separate handler
        # threads (ThreadingHTTPServer spins up a new thread per connection).
        for _ in range(2):
            conn = urllib.request.urlopen(
                urllib.request.Request(base + "/predict", data=payload, method="POST"),
                timeout=5.0,
            )
            conn.read()
            conn.close()
    finally:
        server.shutdown()

    assert len(adapter.predict_thread_names) == 2
    # Same dedicated worker thread both times...
    assert adapter.predict_thread_names[0] == adapter.predict_thread_names[1]
    assert adapter.predict_thread_ids[0] == adapter.predict_thread_ids[1]
    # ...and it must not be a per-connection handler thread (those get the
    # default "Thread-N" name; the dedicated worker is named "inference*").
    assert not adapter.predict_thread_names[0].startswith("Thread-")
    assert adapter.predict_thread_names[0] != threading.current_thread().name


def test_parse_args_forwards_extra_flags():
    args, kwargs = parse_args(
        ["--adapter=lerobot", "--port=9000", "--checkpoint=/some/path", "--device=cuda"]
    )
    assert args.adapter == "lerobot"
    assert args.port == 9000
    assert kwargs == {"checkpoint": "/some/path", "device": "cuda"}
