#!/bin/bash
# Re-bring-up only can_master2 with auto-restart on bus-off, then show state.
sudo ip link set can_master2 down 2>/dev/null || true
sudo ip link set can_master2 type can bitrate 1000000 restart-ms 100
sudo ip link set can_master2 up
sleep 1
ip -details link show can_master2 | grep -oE "state [A-Z]+|can state [A-Z-]+|bitrate [0-9]+"
