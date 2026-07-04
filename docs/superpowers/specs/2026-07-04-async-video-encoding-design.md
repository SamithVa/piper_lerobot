# Async (background) video encoding during recording

**Date:** 2026-07-04
**Status:** approved

## Problem

With `video_encoding_batch_size=N`, all PNG→AV1 encoding happens in one serial
pass at the end of a recording session (30 episodes × 3 cameras ≈ minutes of
waiting) while 20 CPU cores sit idle during teleop. The operator explicitly
wants to stop waiting at session end; using CPU during teleop is acceptable.

## Approach (chosen: A — background encode worker)

Encode each episode's videos in a background thread *while the operator
teleoperates the next episodes*. The end-of-session pass then finds the
expensive work already done and only does stream-copy concat + metadata merge.

Rejected alternatives:
- **B — parallelize the final batch encode:** still minutes of dead wait; idle
  CPU during the session is wasted.
- **C — stream frames straight into ffmpeg (no PNGs):** architecturally
  cleaner but rewrites `add_frame`/`save_episode`/stats/crash-recovery; too
  risky for a working rig for the same practical outcome.

## Design

### AsyncEpisodeEncoder (new, `lerobot_dataset.py`)

Daemon thread + `queue.Queue` owned by the dataset.

- `enqueue(episode_index)` — called by `save_episode` after the episode's
  metadata row is written.
- `wait_until_idle()` — blocks until the queue is drained and any in-flight
  encode finished (Queue.join semantics).
- `stop()` — sentinel + thread join.
- Worker encodes one episode at a time, cameras **serially** (one ffmpeg at a
  time ≈ 5 of 20 cores, keeping the 30 fps record loop clean).
- Output: deterministic path `<root>/videos_tmp/{video_key}_ep{index:06d}.mp4`,
  written as `.part` then atomically renamed.
- PNGs of the episode are deleted only after **all** its cameras encoded
  successfully. On failure: log, keep PNGs, let the end-of-session pass
  re-encode from them. Worker never dies on an exception.
- Testability: the per-episode encode is a dataset method
  (`_preencode_episode`) injected into the worker as a callable, so unit tests
  can substitute a fake.

### Skip-check in `_encode_temporary_episode_video`

If the pre-encoded `videos_tmp` file exists, move it into a fresh
`tempfile.mkdtemp(dir=root)` dir and return that path (preserves the caller's
"rmtree the parent dir" contract — `videos_tmp/` is shared and must not be
rmtree'd). Otherwise encode from PNGs exactly as today. The whole merge
machinery (`_batch_save_episode_video`, chunked concat, `combine_first`
metadata merge) runs unchanged.

### Wiring

- `save_episode` (batched branch): enqueue after `meta.save_episode`. If the
  in-session batch trigger fires, call `wait_until_idle()` **before**
  `_batch_save_episode_video` (closes the double-encode race).
- `VideoEncodingManager.__exit__`: drain (`wait_until_idle`, with a log line)
  and `stop()` the worker **before** the existing leftover-merge. Runs on
  normal exit, ESC, Ctrl-C, and exceptions alike.
- Config: `async_video_encoding: bool = False` on `DatasetRecordConfig`,
  threaded through `LeRobotDataset.create()`/`__init__` like
  `batch_encoding_size`. `record_bimanual.sh` sets it to true. Only active when
  `batch_encoding_size > 1` and the dataset has video keys.

### Known exposure

After SIGKILL/power loss mid-session, `videos_tmp/` may hold the only encoded
copy of episodes whose PNGs were already deleted and whose metadata rows lack
video columns (today's exposure window is one episode; this widens it).
Recoverable by hand; accepted trade-off. Normal exits are covered by the
context-manager drain.

## Expected outcome

Per-episode encode (~30 s) overlaps the next episodes' teleop. Session end
waits only for the last episode's in-flight encode + stream-copy concat: tens
of seconds instead of minutes.

## Testing

- Unit tests with a fake encode callable: FIFO processing, `wait_until_idle`,
  worker survives a failing episode, `stop()` joins, atomic-rename/skip-check
  behavior, failure keeps PNGs.
- End-to-end: a short real recording on the rig (encode timing and CPU
  contention are only observable there).
