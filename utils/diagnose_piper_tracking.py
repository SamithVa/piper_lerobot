#!/usr/bin/env python
"""Trace Piper leader commands against follower feedback during teleoperation.

This diagnostic intentionally runs one arm with no cameras or dataset writer. It
helps separate three common causes of jerky motion:

1. irregular control-loop timing;
2. discontinuous/noisy leader commands;
3. follower-side tracking oscillation while the command is nearly stationary.

It does not directly read or modify the firmware's position-loop PID gains.
``move_j`` measures the firmware trajectory planner and motor controller
together. The guarded ``move_js`` option bypasses trajectory smoothing and
adds software velocity/acceleration limits for a one-arm comparison test.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lerobot.robots.piper_follower import PIPERFollower, PIPERFollowerConfig
from lerobot.teleoperators.piper_leader import PIPERLeader, PIPERLeaderConfig


JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
ALL_KEYS = (*JOINT_KEYS, "gripper.pos")
RAW_JOINTS_TO_RAD = 1.0 / 57324.840764
RAW_GRIPPER_TO_M = 1.0 / 1_000_000.0
JOINT_LIMITS_RAD = {
    "joint_1.pos": (-2.617994, 2.617994),
    "joint_2.pos": (0.0, 3.141593),
    "joint_3.pos": (-2.967060, 0.0),
    "joint_4.pos": (-1.745330, 1.745330),
    "joint_5.pos": (-1.221730, 1.221730),
    "joint_6.pos": (-2.094396, 2.094396),
}


@dataclass
class Sample:
    elapsed_s: float
    loop_dt_s: float
    leader: dict[str, float]
    command: dict[str, float]
    state: dict[str, float]


@dataclass
class MotionLimiter:
    """Velocity- and acceleration-limit joint targets before ``move_js``."""

    position: dict[str, float]
    velocity: dict[str, float]
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float

    @classmethod
    def from_state(
        cls,
        state: dict[str, float],
        max_velocity_rad_s: float,
        max_acceleration_rad_s2: float,
    ) -> MotionLimiter:
        return cls(
            position={key: state[key] for key in JOINT_KEYS},
            velocity=dict.fromkeys(JOINT_KEYS, 0.0),
            max_velocity_rad_s=max_velocity_rad_s,
            max_acceleration_rad_s2=max_acceleration_rad_s2,
        )

    def step(self, target: dict[str, float], dt: float) -> dict[str, float]:
        command: dict[str, float] = {}
        for key in JOINT_KEYS:
            lower, upper = JOINT_LIMITS_RAD[key]
            bounded_target = min(upper, max(lower, target[key]))
            error = bounded_target - self.position[key]
            # Braking-distance speed cap makes the generated position target
            # decelerate before it reaches the leader instead of stopping in one
            # control tick at the target.
            braking_speed = math.sqrt(2.0 * self.max_acceleration_rad_s2 * abs(error))
            desired_speed = min(self.max_velocity_rad_s, braking_speed)
            desired_velocity = math.copysign(desired_speed, error) if error else 0.0
            max_velocity_change = self.max_acceleration_rad_s2 * dt
            velocity_change = min(
                max_velocity_change,
                max(-max_velocity_change, desired_velocity - self.velocity[key]),
            )
            velocity = self.velocity[key] + velocity_change
            self.position[key] += velocity * dt
            # Hard protocol bounds remain the final safety barrier. With the
            # braking-speed cap this should only activate at a physical limit,
            # not at ordinary interior targets.
            if self.position[key] < lower:
                self.position[key] = lower
                velocity = 0.0
            elif self.position[key] > upper:
                self.position[key] = upper
                velocity = 0.0
            self.velocity[key] = velocity
            command[key] = self.position[key]

        # The gripper retains its own firmware motion/force limiting.
        command["gripper.pos"] = min(0.08, max(0.0, abs(target["gripper.pos"])))
        return command


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rms(values: list[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(sum(value * value for value in values) / len(values))


def _read_follower_state(follower: PIPERFollower) -> dict[str, float]:
    """Read follower state in the same SI units as the leader command."""
    raw = follower.bus.read()
    return {
        f"{motor}.pos": (
            float(value) * RAW_GRIPPER_TO_M if motor == "gripper" else float(value) * RAW_JOINTS_TO_RAD
        )
        for motor, value in raw.items()
    }


def _send_follower_action(
    follower: PIPERFollower,
    command: dict[str, float],
    control_mode: str,
) -> None:
    if control_mode == "move_j":
        follower.send_action(command)
        return

    follower.bus.robot.move_js([command[key] for key in JOINT_KEYS])
    follower.bus.gripper.move_gripper_m(command["gripper.pos"], 3.0)


def _wait_for_safe_start(
    leader: PIPERLeader,
    follower: PIPERFollower,
    max_gap_rad: float,
) -> None:
    while True:
        command = leader.get_action()
        state = _read_follower_state(follower)
        gaps = {key: abs(command[key] - state[key]) for key in JOINT_KEYS}
        worst_key = max(gaps, key=gaps.get)
        worst_gap = gaps[worst_key]
        print(f"Largest leader/follower gap: {worst_key}={math.degrees(worst_gap):.2f} deg")
        if worst_gap <= max_gap_rad:
            input("Start positions are close. Press ENTER to begin commanding the follower... ")
            return
        input(
            f"Move the leader close to the follower (limit {math.degrees(max_gap_rad):.1f} deg), "
            "then press ENTER to check again... "
        )


def _write_csv(samples: list[Sample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["elapsed_s", "loop_dt_s"]
    for key in ALL_KEYS:
        label = key.removesuffix(".pos")
        fieldnames.extend(
            (f"leader_{label}", f"command_{label}", f"state_{label}", f"error_{label}")
        )

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row: dict[str, float] = {
                "elapsed_s": sample.elapsed_s,
                "loop_dt_s": sample.loop_dt_s,
            }
            for key in ALL_KEYS:
                label = key.removesuffix(".pos")
                row[f"leader_{label}"] = sample.leader[key]
                row[f"command_{label}"] = sample.command[key]
                row[f"state_{label}"] = sample.state[key]
                row[f"error_{label}"] = sample.command[key] - sample.state[key]
            writer.writerow(row)


def _print_summary(samples: list[Sample], requested_hz: float) -> None:
    if len(samples) < 3:
        print("Not enough samples for tracking analysis.")
        return

    loop_dts = [sample.loop_dt_s for sample in samples[1:] if sample.loop_dt_s > 0]
    overruns = [dt for dt in loop_dts if dt > 1.1 / requested_hz]
    print("\nTiming")
    print(f"  samples: {len(samples)}")
    print(f"  median rate: {1.0 / statistics.median(loop_dts):.1f} Hz")
    print(f"  p95 interval: {_percentile(loop_dts, 95) * 1e3:.2f} ms")
    print(f"  max interval: {max(loop_dts) * 1e3:.2f} ms")
    print(f"  >10% deadline overruns: {len(overruns)} ({100.0 * len(overruns) / len(loop_dts):.1f}%)")

    print("\nJoint tracking (angles in degrees)")
    print("  joint       cmd_step_p95   error_rms   error_p95   still_vel_p95   hold_crossings")
    for key in JOINT_KEYS:
        command_steps: list[float] = []
        errors: list[float] = []
        stationary_state_velocities: list[float] = []
        stationary_error_crossings = 0
        previous_stationary_error: float | None = None
        for previous, current in zip(samples, samples[1:], strict=False):
            dt = current.elapsed_s - previous.elapsed_s
            if dt <= 0:
                continue
            command_delta = current.command[key] - previous.command[key]
            state_delta = current.state[key] - previous.state[key]
            command_velocity = command_delta / dt
            command_steps.append(abs(command_delta))
            errors.append(current.command[key] - current.state[key])
            # A nearly stationary target should not produce sustained follower motion.
            if abs(command_velocity) < math.radians(0.5):
                stationary_state_velocities.append(abs(state_delta / dt))
                current_error = current.command[key] - current.state[key]
                if (
                    previous_stationary_error is not None
                    and current_error * previous_stationary_error < 0
                    and max(abs(current_error), abs(previous_stationary_error)) > math.radians(0.05)
                ):
                    stationary_error_crossings += 1
                previous_stationary_error = current_error
            else:
                previous_stationary_error = None

        print(
            f"  {key.removesuffix('.pos'):<11}"
            f"{math.degrees(_percentile(command_steps, 95)):>10.3f}"
            f"{math.degrees(_rms(errors)):>13.3f}"
            f"{math.degrees(_percentile([abs(value) for value in errors], 95)):>12.3f}"
            f"{math.degrees(_percentile(stationary_state_velocities, 95)):>16.3f} deg/s"
            f"{stationary_error_crossings:>15}"
        )

    print("\nInterpretation")
    print("  Large/irregular command steps => leader feedback, filtering, or loop timing is the source.")
    print("  Repeated hold crossings with a still target => underdamped control/mechanics are suspect.")
    print("  Large tracking error without still-target motion => trajectory speed/acceleration limits or lag.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--control-mode", choices=("move_j", "move_js"), default="move_j")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=None,
        help="Control frequency (default: 50 for move_j, 100 for move_js).",
    )
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    parser.add_argument("--max-velocity-deg-s", type=float, default=25.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=100.0)
    parser.add_argument(
        "--max-tracking-error-deg",
        type=float,
        default=12.0,
        help="Abort move_js if a commanded joint differs from feedback by more than this.",
    )
    parser.add_argument(
        "--max-start-gap-deg",
        type=float,
        default=10.0,
        help="Refuse to command if any leader/follower joint differs by more than this.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: diagnostics/piper_tracking_<arm>_<mode>_<timestamp>.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.rate_hz is None:
        args.rate_hz = 100.0 if args.control_mode == "move_js" else 50.0
    if args.rate_hz <= 0 or args.duration_s <= 0:
        raise ValueError("--rate-hz and --duration-s must be positive")
    if not 0 < args.ema_alpha <= 1:
        raise ValueError("--ema-alpha must be in (0, 1]")
    if args.max_start_gap_deg <= 0:
        raise ValueError("--max-start-gap-deg must be positive")
    if args.max_velocity_deg_s <= 0 or args.max_acceleration_deg_s2 <= 0:
        raise ValueError("velocity and acceleration limits must be positive")
    if args.max_tracking_error_deg <= 0:
        raise ValueError("--max-tracking-error-deg must be positive")

    follower_can = f"{args.arm}_follower"
    leader_can = f"{args.arm}_leader"
    follower = PIPERFollower(PIPERFollowerConfig(id=f"diagnostic_{args.arm}", can_name=follower_can, cameras={}))
    leader = PIPERLeader(
        PIPERLeaderConfig(id=f"diagnostic_{args.arm}_leader", can_name=leader_can, ema_alpha=args.ema_alpha)
    )
    output_path = args.output or Path("diagnostics") / (
        f"piper_tracking_{args.arm}_{args.control_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    samples: list[Sample] = []
    follower_connected = False
    leader_connected = False
    print(f"Connecting {args.arm} follower on {follower_can}; it will home before the test.")
    try:
        follower.connect()
        follower_connected = True
        leader.connect()
        leader_connected = True
        _wait_for_safe_start(leader, follower, math.radians(args.max_start_gap_deg))

        if args.control_mode == "move_js":
            print("\nWARNING: move_js is direct follower/pass-through mode with no firmware trajectory smoothing.")
            print(
                f"Software limits: {args.max_velocity_deg_s:g} deg/s, "
                f"{args.max_acceleration_deg_s2:g} deg/s^2, "
                f"abort error {args.max_tracking_error_deg:g} deg."
            )
            confirmation = input("Keep an emergency stop within reach. Type MOVE_JS to continue: ")
            if confirmation != "MOVE_JS":
                print("Confirmation not received; test cancelled.")
                return 2

        period_s = 1.0 / args.rate_hz
        initial_state = _read_follower_state(follower)
        limiter = MotionLimiter.from_state(
            initial_state,
            max_velocity_rad_s=math.radians(args.max_velocity_deg_s),
            max_acceleration_rad_s2=math.radians(args.max_acceleration_deg_s2),
        )
        start = time.perf_counter()
        previous_start = start
        next_tick = start
        print(
            f"Tracing {args.control_mode} at {args.rate_hz:g} Hz for {args.duration_s:g} s; "
            "press Ctrl+C to stop early. Move slowly."
        )
        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start
            if elapsed >= args.duration_s:
                break

            leader_command = leader.get_action()
            state = _read_follower_state(follower)
            command = (
                limiter.step(leader_command, period_s)
                if args.control_mode == "move_js"
                else dict(leader_command)
            )
            if args.control_mode == "move_js":
                errors = {key: abs(command[key] - state[key]) for key in JOINT_KEYS}
                worst_key = max(errors, key=errors.get)
                if errors[worst_key] > math.radians(args.max_tracking_error_deg):
                    print(
                        f"SAFETY ABORT: {worst_key} tracking error "
                        f"{math.degrees(errors[worst_key]):.2f} deg exceeds "
                        f"{args.max_tracking_error_deg:g} deg."
                    )
                    break
            _send_follower_action(follower, command, args.control_mode)
            samples.append(
                Sample(
                    elapsed_s=elapsed,
                    loop_dt_s=loop_start - previous_start,
                    leader=dict(leader_command),
                    command=dict(command),
                    state=state,
                )
            )
            previous_start = loop_start

            next_tick += period_s
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif -remaining > period_s:
                # Do not execute a burst of stale catch-up commands after a long stall.
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nTrace stopped by operator.")
    finally:
        if samples:
            _write_csv(samples, output_path)
            _print_summary(samples, args.rate_hz)
            print(f"\nCSV trace: {output_path.resolve()}")
        try:
            if follower_connected:
                print("Returning follower home and gently disabling it...")
                follower.disconnect()
        finally:
            if leader_connected:
                leader.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
