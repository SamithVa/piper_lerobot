#!/usr/bin/env python
"""Regression tests for Piper robot shutdown cleanup."""

from lerobot.robots.piper_follower.piper_follower import PIPERFollower


class FakeBus:
    def __init__(self):
        self.calls = []

    def gentle_disable(self):
        self.calls.append("gentle_disable")

    def disconnect(self):
        self.calls.append("disconnect")


def test_piper_follower_disconnect_closes_bus_after_soft_disable():
    robot = object.__new__(PIPERFollower)
    robot.bus = FakeBus()
    robot.cameras = {}
    robot._is_connected = True

    robot.disconnect()

    assert robot.bus.calls == ["gentle_disable", "disconnect"]
    assert robot._is_connected is False
