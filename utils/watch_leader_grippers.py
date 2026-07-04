#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Read-only monitor of both leader grippers. Moves nothing.

Squeeze each leader gripper fully closed, then fully open, and watch the raw
angle. A healthy leader reads ~0 when closed and ~70000 (0.001mm units, ~70mm)
when open. If a leader never drops near 0 when you close it, its zero point is
offset -> the follower is commanded to stay open (that's the bug, on the LEADER).

The "-> follower cmd" column is exactly what the record pipeline would send:
    action = raw / 1e6   (piper_leader.get_action)
    follower GripperCtrl(round(action * 1e6)) == raw   (round-trip)

Usage:
    python utils/watch_leader_grippers.py
    python utils/watch_leader_grippers.py --left left_leader --right right_leader
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2


def read_angle(piper):
    try:
        return piper.GetArmGripperMsgs().gripper_state.grippers_angle
    except Exception as e:
        return f"err:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="left_leader")
    ap.add_argument("--right", default="right_leader")
    args = ap.parse_args()

    left = C_PiperInterface_V2(args.left, judge_flag=True)
    right = C_PiperInterface_V2(args.right, judge_flag=True)
    left.ConnectPort()
    right.ConnectPort()
    time.sleep(0.3)

    print("Close then open each leader gripper. Ctrl-C to stop.\n")
    print(f"{'left raw':>12} {'-> foll cmd':>12}   |  {'right raw':>12} {'-> foll cmd':>12}")
    try:
        while True:
            lr = read_angle(left)
            rr = read_angle(right)
            lc = lr / 1e6 if isinstance(lr, int) else lr
            rc = rr / 1e6 if isinstance(rr, int) else rr
            print(f"{lr:>12} {lc:>12}   |  {rr:>12} {rc:>12}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
