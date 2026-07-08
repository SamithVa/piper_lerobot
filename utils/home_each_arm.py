#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Home each Piper arm one at a time to identify which CAN interface is which arm.

For every CAN name given (default: all four), this script enables the arm,
moves it to the home position (all joints = 0), prints which CAN it is,
then waits 5 seconds before moving to the next arm.

Watch the robots: the arm that moves is the one bound to the printed CAN name.

Usage:
    python utils/home_each_arm.py                       # all four, in order
    python utils/home_each_arm.py can_master can_follower
"""
import sys
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

DEFAULT_CANS = ["can_master", "can_follower", "can_master2", "can_follower2"]
GAP_S = 5


def make_robot(can):
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


def enable(robot, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        if robot.get_joint_enable_status(255):
            return True
        robot.enable(255)
        time.sleep(0.2)
    return False


def home(robot, gripper):
    robot.set_speed_percent(50)  # position control, speed 50%
    robot.move_j([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gripper.move_gripper_m(0.0, 1.0)


def main(cans):
    for idx, can in enumerate(cans):
        print("=" * 50)
        print(f">>> Arm {idx + 1}/{len(cans)}  ->  CAN = {can}")
        print("=" * 50)
        try:
            robot, gripper = make_robot(can)
            time.sleep(0.5)
            if not enable(robot):
                print(f"[WARN] {can}: enable timed out (arm powered on / CAN up?)")
            home(robot, gripper)
            print(f"    {can}: home command sent. WATCH which arm moves now.")
        except Exception as e:
            print(f"[ERROR] {can}: {e}")
        print(f"    waiting {GAP_S}s before next arm...\n")
        time.sleep(GAP_S)
    print("Done. Note which physical arm moved for each CAN name above.")


if __name__ == "__main__":
    cans = sys.argv[1:] or DEFAULT_CANS
    main(cans)
