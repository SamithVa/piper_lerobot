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


def test_on_chunk_skips_rows_executed_during_flight():
    ex = ChunkExecutor(chunk_threshold=0.5)
    ex.mark_requested()
    ex.on_chunk(chunk(10))
    for _ in range(5):
        ex.next_action()
    ex.mark_requested()  # obs captured now (tick 5)
    ex.next_action()  # 2 more rows execute while inference runs
    ex.next_action()
    ex.on_chunk(chunk(10, start=100.0))
    # new chunk row 0 corresponds to the obs tick; 2 rows executed -> start at row 2
    np.testing.assert_array_equal(ex.next_action(), [102.0, 102.0])


def test_dry_hold_does_not_consume_chunk_rows():
    # Queue runs dry while a request is in flight: the robot holds position,
    # so NO rows of the arriving chunk are stale.
    ex = ChunkExecutor()
    ex.mark_requested()
    ex.on_chunk(chunk(2))
    ex.next_action()
    ex.next_action()          # queue now empty
    ex.mark_requested()       # obs captured while dry
    for _ in range(3):
        assert ex.next_action() is None  # holding; nothing executed
    assert ex.on_chunk(chunk(3)) == 3    # all rows usable
    np.testing.assert_array_equal(ex.next_action(), [0.0, 0.0])


def test_fully_stale_chunk_under_overlap_recovers():
    # Overlap case: old queue long enough that MORE rows execute during
    # flight than the new chunk holds -> new chunk fully stale, queue dry,
    # executor must allow an immediate re-request.
    ex = ChunkExecutor(chunk_threshold=0.5)
    ex.mark_requested()
    ex.on_chunk(chunk(10))
    for _ in range(5):
        ex.next_action()
    ex.mark_requested()
    for _ in range(4):
        ex.next_action()      # 4 rows executed while in flight
    assert ex.on_chunk(chunk(3)) == 0    # 4 executed >= 3 rows -> nothing usable
    assert ex.next_action() is None
    assert ex.should_request() is True


def test_request_failed_clears_in_flight():
    ex = ChunkExecutor()
    ex.mark_requested()
    assert ex.should_request() is False
    ex.on_request_failed()
    assert ex.should_request() is True


def test_consumed_rows_tracks_full_chunk_coordinates():
    ex = ChunkExecutor()
    ex.mark_requested()
    ex.on_chunk(chunk(10))
    assert ex.consumed_rows == 0
    ex.next_action()
    ex.next_action()
    assert ex.consumed_rows == 2

    # Request mid-chunk, execute 3 rows in flight, then install a new chunk:
    ex.mark_requested()
    ex.next_action(), ex.next_action(), ex.next_action()
    ex.on_chunk(chunk(10, start=100.0))
    assert ex.last_skip == 3
    # New chunk: 3 rows skipped at install, 0 executed since.
    assert ex.consumed_rows == 3
    ex.next_action()
    assert ex.consumed_rows == 4


def test_consumed_rows_before_any_chunk_is_zero():
    ex = ChunkExecutor()
    assert ex.consumed_rows == 0
    assert ex.last_skip == 0


def test_dry_queue_counts_all_rows_consumed():
    ex = ChunkExecutor()
    ex.mark_requested()
    ex.on_chunk(chunk(2))
    ex.next_action(), ex.next_action()
    assert ex.next_action() is None  # dry
    assert ex.consumed_rows == 2
