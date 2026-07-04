#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Set a Piper gripper's ZERO point at its fully-closed position.

Symptom this fixes: a gripper whose closed position reads a non-zero (often
negative) angle, e.g. left_follower closed == -26810 while right_follower
closed == 0. In teleop that offset makes the follower open instead of close.

The zero MUST be set while the gripper is DISABLED (gripper_code=0x00). This
matches the official piper_sdk demo (demo/V2/piper_set_gripper_zero.py):
    GripperCtrl(0, 1000, 0x00, 0)      # disable -> gripper goes limp
    sleep 1.5
    GripperCtrl(0, 1000, 0x00, 0xAE)   # disable + store current pos as zero
Setting zero while ENABLED (0x01) is silently ignored by the firmware.

Procedure:
  1. Run this script; it disables the gripper so you can move it by hand.
  2. Push the gripper jaws FULLY CLOSED (to the mechanical stop).
  3. Press Enter; it stores that position as angle 0.
  4. It reads back the angle -- should now be ~0.

Do this for BOTH arms of a teleop pair (leader and follower).

Usage:
    python utils/zero_gripper.py left_follower
    python utils/zero_gripper.py left_leader
    python utils/zero_gripper.py left_follower --yes   # skip the confirm prompt
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2


def read_angle(piper):
    return piper.GetArmGripperMsgs().gripper_state.grippers_angle


def main():
    ap = argparse.ArgumentParser(description="Set a Piper gripper zero at its closed position.")
    ap.add_argument("can", help="CAN name, e.g. left_follower / left_leader")
    ap.add_argument("--effort", type=int, default=1000, help="0.001 N/m (default 1000)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    piper = C_PiperInterface_V2(args.can, judge_flag=True)
    piper.ConnectPort()
    time.sleep(0.3)

    print(f"{args.can}: current gripper angle = {read_angle(piper)} (0.001mm)")

    # Disable the gripper so it goes limp and the firmware will accept a new zero.
    piper.GripperCtrl(0, args.effort, 0x00, 0)
    time.sleep(1.5)

    if not args.yes:
        input(f"{args.can} gripper is now limp. Push the jaws FULLY CLOSED, then press Enter to set zero... ")

    # Disable + set_zero=0xAE stores the current position as angle 0.
    piper.GripperCtrl(0, args.effort, 0x00, 0xAE)
    time.sleep(0.5)

    for _ in range(5):
        print(f"  angle after zeroing -> {read_angle(piper)} (0.001mm)")
        time.sleep(0.2)

    # Re-enable so the gripper is ready to hold/track again.
    piper.GripperCtrl(0, args.effort, 0x01, 0)
    print(f"{args.can}: done. Closed should now read ~0; gripper re-enabled.")


if __name__ == "__main__":
    main()
