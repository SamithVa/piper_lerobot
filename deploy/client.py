"""Robot-side deployment client for the Piper arms.

Runs with pyAgxArm + cameras and this repo's lerobot on the
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
import concurrent.futures
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


def check_dims(info: dict, motor_keys: list[str]) -> None:
    """Fail fast — before touching the arms — if policy and robot disagree."""
    for key in ("state_dim", "action_dim"):
        if info[key] != len(motor_keys):
            raise SystemExit(
                f"{key} mismatch: policy expects {info[key]}, robot has {len(motor_keys)} motors"
            )


# ---------- HTTP ----------


def http_get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, body: bytes = b"", timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_post_async(url: str, body: bytes, timeout: float) -> concurrent.futures.Future:
    """http_post on a daemon thread, result delivered via a Future.

    Deliberately NOT a ThreadPoolExecutor: its workers are non-daemon and an
    atexit hook joins them all, so Ctrl-C while a /predict is in flight would
    stall process exit for up to the request timeout. A daemon thread is
    simply abandoned, keeping Ctrl-C exit instant.
    """
    future: concurrent.futures.Future = concurrent.futures.Future()

    def work():
        try:
            future.set_result(http_post(url, body, timeout))
        except Exception as exc:  # noqa: BLE001 — delivered via the Future
            future.set_exception(exc)

    threading.Thread(target=work, daemon=True).start()
    return future


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
    # The server's FIRST predict is slow (~15-20s: torch.compile + CUDA-graph
    # capture for pi05); every predict after is <0.5s. Give the initial blocking
    # request its own generous budget so the cold start doesn't trip the tight
    # steady-state predict_timeout_s (which stays short to catch real hangs).
    first_predict_timeout_s: float = 90.0
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
    check_dims(info, motor_keys)

    robot.connect()
    executor = ChunkExecutor(cfg.chunk_threshold)
    inflight: concurrent.futures.Future | None = None  # at most one /predict in flight

    try:
        http_post(cfg.server + "/reset")

        # Blocking first chunk so the loop starts with actions in hand. Uses the
        # generous first_predict_timeout_s because the server pays a one-time
        # cold-start (compile + CUDA-graph capture) on its very first predict.
        logging.info("requesting first chunk (server cold start may take ~15-20s)...")
        payload = capture_payload(robot, camera_map, motor_keys, cfg.task)
        executor.mark_requested()
        executor.on_chunk(
            protocol.decode_chunk(http_post(cfg.server + "/predict", payload, cfg.first_predict_timeout_s))
        )
        logging.info(f"running at {fps:.0f} fps for {cfg.duration_s:.0f}s — Ctrl-C to stop")

        last_action: dict | None = None
        dry_ticks = 0
        t_end = time.perf_counter() + cfg.duration_s
        next_t = time.perf_counter()
        while time.perf_counter() < t_end:
            if inflight is not None and inflight.done():
                try:
                    usable = executor.on_chunk(protocol.decode_chunk(inflight.result()))
                    if usable == 0:
                        logging.warning(
                            "chunk arrived fully stale (inference slower than execution) — re-requesting"
                        )
                except Exception as exc:  # noqa: BLE001 — a failed predict must not kill the loop
                    logging.warning(f"/predict failed: {exc}")
                    executor.on_request_failed()
                inflight = None

            row = executor.next_action()
            if row is not None:
                last_action = action_to_dict(row, motor_keys)
                robot.send_action(last_action)
                dry_ticks = 0
            elif last_action is not None:
                robot.send_action(last_action)  # hold position while recovering
                dry_ticks += 1
                if dry_ticks % max(int(fps), 1) == 1:
                    logging.warning("action queue dry — holding position")

            if executor.should_request():
                payload = capture_payload(robot, camera_map, motor_keys, cfg.task)
                executor.mark_requested()
                inflight = http_post_async(cfg.server + "/predict", payload, cfg.predict_timeout_s)

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
