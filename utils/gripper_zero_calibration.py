#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Set a Piper gripper's ZERO point at its fully-closed position.

Symptom this fixes: a gripper whose closed position reads a non-zero (often
negative) value. In teleop that offset makes the follower open instead of close.

The zero MUST be set while the gripper is DISABLED. pyAgxArm exposes this as:
    gripper.disable_gripper()        # gripper goes limp
    <move jaws fully closed by hand>
    gripper.calibrate_gripper()      # store current position as zero
Setting zero while enabled is silently ignored by the firmware.

Procedure:
  1. Run this script; it disables the gripper so you can move it by hand.
  2. Push the gripper jaws FULLY CLOSED (to the mechanical stop).
  3. Press Enter; it stores that position as value 0.
  4. It reads back the value -- should now be ~0.

Do this for BOTH arms of a teleop pair (leader and follower).

Usage:
    python utils/zero_gripper.py left_follower
    python utils/zero_gripper.py left_leader
    python utils/zero_gripper.py left_follower --yes   # skip the confirm prompt
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
    gs = gripper.get_gripper_status()
    return gs.msg.value if gs is not None else None  # meters


def main():
    ap = argparse.ArgumentParser(description="Set a Piper gripper zero at its closed position.")
    ap.add_argument("can", help="CAN name, e.g. left_follower / left_leader")
    ap.add_argument("--force", type=float, default=1.0, help="re-enable gripping force in N (default 1.0)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    robot, gripper = make_gripper(args.can)
    time.sleep(0.3)

    print(f"{args.can}: current gripper value = {read_value(gripper)} m")

    # Disable the gripper so it goes limp and the firmware will accept a new zero.
    gripper.disable_gripper()
    time.sleep(1.5)

    if not args.yes:
        input(f"{args.can} gripper is now limp. Push the jaws FULLY CLOSED, then press Enter to set zero... ")

    # calibrate_gripper() stores the current position as value 0.
    ok = gripper.calibrate_gripper()
    print(f"{args.can}: calibrate_gripper() -> {ok}")
    time.sleep(0.5)

    for _ in range(5):
        print(f"  value after zeroing -> {read_value(gripper)} m")
        time.sleep(0.2)

    # Re-enable so the gripper is ready to hold/track again (move enables it).
    gripper.move_gripper_m(0.0, args.force)
    print(f"{args.can}: done. Closed should now read ~0; gripper re-enabled.")


if __name__ == "__main__":
    main()
