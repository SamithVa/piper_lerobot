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

from piper_sdk import C_PiperInterface_V2

DEFAULT_CANS = ["can_master", "can_follower", "can_master2", "can_follower2"]
JOINT_FACTOR = 57324.840764  # rad -> 0.001 deg
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 joints + gripper
GAP_S = 5


def enable(piper, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        st = piper.GetArmLowSpdInfoMsgs()
        enabled = all(
            getattr(st, f"motor_{i}").foc_status.driver_enable_status
            for i in range(1, 7)
        )
        if enabled:
            return True
        piper.EnablePiper()
        time.sleep(0.2)
    return False


def home(piper):
    j = [round(HOME[i] * JOINT_FACTOR) for i in range(6)]
    grip = round(HOME[6] * 1000 * 1000)
    piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)  # position control, speed 50%
    piper.JointCtrl(*j)
    piper.GripperCtrl(abs(grip), 1000, 0x01, 0)


def main(cans):
    for idx, can in enumerate(cans):
        print("=" * 50)
        print(f">>> Arm {idx + 1}/{len(cans)}  ->  CAN = {can}")
        print("=" * 50)
        try:
            piper = C_PiperInterface_V2(can)
            piper.ConnectPort()
            time.sleep(0.5)
            if not enable(piper):
                print(f"[WARN] {can}: enable timed out (arm powered on / CAN up?)")
            home(piper)
            print(f"    {can}: home command sent. WATCH which arm moves now.")
        except Exception as e:
            print(f"[ERROR] {can}: {e}")
        print(f"    waiting {GAP_S}s before next arm...\n")
        time.sleep(GAP_S)
    print("Done. Note which physical arm moved for each CAN name above.")


if __name__ == "__main__":
    cans = sys.argv[1:] or DEFAULT_CANS
    main(cans)
