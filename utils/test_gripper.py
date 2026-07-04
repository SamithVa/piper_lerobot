#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Test / drive a Piper gripper on a given CAN interface.

Sends an enable+clear-error to the gripper (recovers from a fault state), then
commands it open or close, and prints the gripper feedback so you can see if it
actually moves / how far.

Usage:
    python utils/test_gripper.py left_follower              # open
    python utils/test_gripper.py left_follower --close      # close
    python utils/test_gripper.py left_follower --mm 40      # open to 40 mm
    python utils/test_gripper.py left_follower --effort 2000  # more torque

Notes:
- gripper_angle unit is 0.001 mm (so 70 mm -> 70000). Range ~0..70 mm.
- gripper_effort unit is 0.001 N/m, range 0..5000 (0..5 N/m).
- gripper_code 0x03 = enable AND clear error (best for a stuck/faulty gripper).
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2


def read_gripper(piper):
    try:
        g = piper.GetArmGripperMsgs().gripper_state
        return f"angle={g.grippers_angle} (0.001mm)  effort={g.grippers_effort}  foc_status={g.foc_status}"
    except Exception as e:
        return f"read err: {e}"


def main():
    ap = argparse.ArgumentParser(description="Open/close a Piper gripper to test it.")
    ap.add_argument("can", nargs="?", default="left_follower", help="CAN name")
    ap.add_argument("--close", action="store_true", help="close instead of open")
    ap.add_argument("--mm", type=float, default=70.0, help="open target in mm (default 70)")
    ap.add_argument("--effort", type=int, default=1000, help="0.001 N/m, 0..5000 (default 1000)")
    args = ap.parse_args()

    angle = 0 if args.close else int(args.mm * 1000)  # mm -> 0.001mm

    piper = C_PiperInterface_V2(args.can, judge_flag=True)
    piper.ConnectPort()
    time.sleep(0.2)

    # Enable the arm so the gripper responds.
    for _ in range(20):
        if piper.EnablePiper():
            break
        time.sleep(0.05)

    print(f"{args.can}: gripper before -> {read_gripper(piper)}")

    # Enable + clear any error first, then send the target.
    piper.GripperCtrl(0, args.effort, 0x03, 0)  # 0x03 = enable & clear error
    time.sleep(0.5)
    action = "CLOSE" if args.close else f"OPEN {args.mm}mm"
    print(f"{args.can}: commanding {action} (angle={angle}, effort={args.effort})")
    piper.GripperCtrl(angle, args.effort, 0x01, 0)  # 0x01 = enable

    # Watch it move for a couple seconds.
    for _ in range(10):
        time.sleep(0.2)
        print(f"  gripper -> {read_gripper(piper)}")

    print(f"{args.can}: done. If angle did not change, gripper is likely faulty / mis-cabled.")


if __name__ == "__main__":
    main()
