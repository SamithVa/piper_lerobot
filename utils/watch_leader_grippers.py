#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Read-only monitor of both leader grippers. Moves nothing.

Squeeze each leader gripper fully closed, then fully open, and watch the value.
A healthy leader reads ~0.0 m when closed and ~0.07 m when open. If a leader
never drops near 0 when you close it, its zero point is offset -> the follower is
commanded to stay open (that's the bug, on the LEADER).

The "-> follower cmd" column is exactly what the record pipeline would send:
    action = value (meters, from get_gripper_status().msg.value)
    follower move_gripper_m(value) == same width   (round-trip)

Usage:
    python utils/watch_leader_grippers.py
    python utils/watch_leader_grippers.py --left left_leader --right right_leader
"""
import argparse
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


def make_gripper(can):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V188,
        interface="socketcan",
        channel=can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.connect()
    return robot, gripper


def read_value(gripper):
    try:
        gs = gripper.get_gripper_status()
        return gs.msg.value if gs is not None else "no fb"
    except Exception as e:
        return f"err:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="left_leader")
    ap.add_argument("--right", default="right_leader")
    args = ap.parse_args()

    _, left = make_gripper(args.left)
    _, right = make_gripper(args.right)
    time.sleep(0.3)

    print("Close then open each leader gripper. Ctrl-C to stop.\n")
    print(f"{'left value(m)':>14} {'-> foll cmd':>12}   |  {'right value(m)':>14} {'-> foll cmd':>12}")
    try:
        while True:
            lv = read_value(left)
            rv = read_value(right)
            lc = lv if isinstance(lv, float) else lv
            rc = rv if isinstance(rv, float) else rv
            print(f"{lv:>14} {lc:>12}   |  {rv:>14} {rc:>12}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
