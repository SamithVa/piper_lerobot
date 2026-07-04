#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Gently disable (软失能) a Piper arm so it does NOT free-fall.

Sequence: (1) slowly move the arm to the home pose (all joints 0) with position
control, then (2) softly release it.

Why: DisablePiper() cuts all motor torque instantly. At the home pose the arm
is still holding itself against gravity, so it drops hard the moment torque is
removed. This script first parks the arm at home, then uses MIT control to hold
the current joint angles and ramps the position gain kp -> 0 while keeping a
damping gain kd > 0. The arm slowly, dampedly droops under gravity (no slam),
and is already limp by the time DisablePiper() is finally called.

Usage:
    python utils/gentle_disable_arm.py                       # all 4 arms (dual-arm)
    python utils/gentle_disable_arm.py left_follower
    python utils/gentle_disable_arm.py left_follower right_follower
    python utils/gentle_disable_arm.py left_follower --duration 3.0 --kp 12 --kd 1.2
    python utils/gentle_disable_arm.py left_follower --no-home   # skip move-to-home

Tuning on real hardware:
    --kp   initial holding gain (SDK reference 10). Higher = firmer hold at start.
    --kd   damping gain (SDK reference 0.8, max 5). Higher = slower/softer droop,
           too high = jitter/oscillation.
    --duration  seconds to ramp kp from --kp down to 0.
"""
import argparse
import math
import time

from piper_sdk import C_PiperInterface_V2

# joint feedback is in 0.001 deg -> rad
RAW_TO_RAD = 0.001 * math.pi / 180.0
NUM_JOINTS = 6
RATE_HZ = 100.0
HOME_TOL_RAD = 0.05  # ~3 deg: consider "at home" when every joint is within this


def read_joints_rad(piper):
    js = piper.GetArmJointMsgs().joint_state
    return [getattr(js, f"joint_{i}") * RAW_TO_RAD for i in range(1, NUM_JOINTS + 1)]


def enable(piper, timeout=3.0):
    """Make sure motors are enabled so we can take MIT control before releasing."""
    start = time.time()
    while time.time() - start < timeout:
        if piper.EnablePiper():
            return True
        time.sleep(0.05)
    return False


def move_to_home(piper, speed, timeout=10.0):
    """Slowly move all joints to 0 with position control; wait until arrived."""
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # position control, speed %
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    start = time.time()
    while time.time() - start < timeout:
        if max(abs(a) for a in read_joints_rad(piper)) < HOME_TOL_RAD:
            return True
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(0.05)
    return False


def gentle_disable(can, kp0, kd, duration, go_home, home_speed, settle=0.6):
    piper = C_PiperInterface_V2(can, judge_flag=True)
    piper.ConnectPort()
    time.sleep(0.2)

    if not enable(piper):
        print(f"[WARN] {can}: could not confirm enable — arm may already be limp.")

    if go_home:
        print(f"{can}: moving to home (speed {home_speed}%)...")
        if move_to_home(piper, home_speed):
            print(f"{can}: reached home.")
        else:
            print(f"[WARN] {can}: home move timed out — releasing from current pose.")

    hold = read_joints_rad(piper)
    print(f"{can}: holding {[round(a, 3) for a in hold]} rad, ramping kp {kp0}->0 over {duration}s")

    dt = 1.0 / RATE_HZ
    steps = max(1, int(duration * RATE_HZ))

    # Phase 1: ramp kp -> 0 while damping (kd) resists any fall.
    for s in range(steps + 1):
        kp = kp0 * (1.0 - s / steps)  # linear fade to zero
        piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)  # enter/keep MIT mode
        for j in range(NUM_JOINTS):
            piper.JointMitCtrl(j + 1, hold[j], 0.0, kp, kd, 0.0)
        time.sleep(dt)

    # Phase 2: kp = 0, keep only damping so it settles softly at the bottom.
    for _ in range(int(settle * RATE_HZ)):
        piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
        for j in range(NUM_JOINTS):
            piper.JointMitCtrl(j + 1, hold[j], 0.0, 0.0, kd, 0.0)
        time.sleep(dt)

    # Phase 3: fully release + restore position/speed control mode.
    while piper.DisablePiper():
        time.sleep(0.01)
    piper.MotionCtrl_1(0x02, 0, 0)
    time.sleep(0.3)
    print(f"{can}: 软失能成功!!!!")


def main():
    ap = argparse.ArgumentParser(description="Gently disable one or more Piper arms.")
    ap.add_argument("cans", nargs="*",
                    default=["left_leader", "left_follower", "right_leader", "right_follower"],
                    help="CAN names (default: all 4 arms). See can_arm_mapping.md")
    ap.add_argument("--kp", type=float, default=10.0, help="initial holding gain (ref 10)")
    ap.add_argument("--kd", type=float, default=0.8, help="damping gain (ref 0.8, max 5)")
    ap.add_argument("--duration", type=float, default=2.0, help="kp ramp-down seconds")
    ap.add_argument("--home-speed", type=int, default=15,
                    help="move-to-home speed percent (default 15, slow)")
    ap.add_argument("--no-home", dest="home", action="store_false",
                    help="skip move-to-home; soft-release from current pose")
    args = ap.parse_args()

    for can in args.cans:
        gentle_disable(can, args.kp, args.kd, args.duration, args.home, args.home_speed)


if __name__ == "__main__":
    main()
