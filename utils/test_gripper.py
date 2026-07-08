#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Test / drive a Piper gripper on a given CAN interface.

Recovers the gripper from a fault (disable, then move re-enables), commands it
open or close, and prints the gripper feedback so you can see if it actually
moves / how far.

Usage:
    python utils/test_gripper.py left_follower              # open
    python utils/test_gripper.py left_follower --close      # close
    python utils/test_gripper.py left_follower --mm 40      # open to 40 mm
    python utils/test_gripper.py left_follower --force 2.0  # more force (N)

Notes:
- gripper value unit is meters (so 70 mm -> 0.070). Range ~0..0.07 m.
- force unit is N, range 0..3.
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


def read_gripper(gripper):
    try:
        gs = gripper.get_gripper_status()
        if gs is None:
            return "no feedback yet"
        return f"value={gs.msg.value} m  force={gs.msg.force} N  mode={gs.msg.mode}  foc_status={gs.msg.foc_status}"
    except Exception as e:
        return f"read err: {e}"


def main():
    ap = argparse.ArgumentParser(description="Open/close a Piper gripper to test it.")
    ap.add_argument("can", nargs="?", default="left_follower", help="CAN name")
    ap.add_argument("--close", action="store_true", help="close instead of open")
    ap.add_argument("--mm", type=float, default=70.0, help="open target in mm (default 70)")
    ap.add_argument("--force", type=float, default=1.0, help="gripping force in N, 0..3 (default 1.0)")
    args = ap.parse_args()

    value_m = 0.0 if args.close else args.mm / 1000.0  # mm -> m

    robot, gripper = make_gripper(args.can)
    time.sleep(0.2)

    # Enable the arm so the gripper responds.
    for _ in range(20):
        if robot.enable(255):
            break
        time.sleep(0.05)

    print(f"{args.can}: gripper before -> {read_gripper(gripper)}")

    # Recover from a fault: disable then move re-enables and clears the error.
    gripper.disable_gripper()
    time.sleep(0.5)
    action = "CLOSE" if args.close else f"OPEN {args.mm}mm"
    print(f"{args.can}: commanding {action} (value={value_m} m, force={args.force} N)")
    gripper.move_gripper_m(value_m, args.force)

    # Watch it move for a couple seconds.
    for _ in range(10):
        time.sleep(0.2)
        print(f"  gripper -> {read_gripper(gripper)}")

    print(f"{args.can}: done. If value did not change, gripper is likely faulty / mis-cabled.")


if __name__ == "__main__":
    main()
