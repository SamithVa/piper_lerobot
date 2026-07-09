import pytest

torch = pytest.importorskip("torch")

from deploy.adapters.rtc_helpers import build_rtc_kwargs, clamp_delay, slice_leftover


def raw(n=50, dim=4):
    return torch.arange(n, dtype=torch.float32)[:, None].repeat(1, dim)


def test_clamp_delay_quantizes_to_few_values():
    assert clamp_delay(0, 10) == 5     # floor 1, round up to quantum
    assert clamp_delay(3, 10) == 5
    assert clamp_delay(6, 10) == 10
    assert clamp_delay(23, 10) == 10   # capped at horizon
    assert {clamp_delay(d, 10) for d in range(0, 30)} == {5, 10}


def test_slice_leftover_fixed_shape_zero_padded():
    left = slice_leftover(raw(), consumed=10, execution_horizon=10)
    assert left.shape == (50, 4)                       # fixed shape always
    assert torch.equal(left[:40], raw()[10:])          # leftover rows first
    assert torch.all(left[40:] == 0)                   # zero tail


def test_slice_leftover_none_cases():
    assert slice_leftover(None, 0, 10) is None                 # no chunk yet
    assert slice_leftover(raw(), -1, 10) is None               # client has no chunk
    assert slice_leftover(raw(), 45, 10) is None               # < horizon rows left
    assert slice_leftover(raw(), 60, 10) is None               # over-consumed


def test_build_rtc_kwargs():
    assert build_rtc_kwargs(None, -1, 5, 10) == {}
    kw = build_rtc_kwargs(raw(), 10, 6, 10)
    assert kw["prev_chunk_left_over"].shape == (1, 50, 4)
    assert kw["inference_delay"] == 10
