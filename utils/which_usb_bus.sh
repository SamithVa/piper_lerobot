#!/bin/bash
# Show which USB bus each camera is on. bus=1 -> independent controller (good for
# splitting YUYV); bus=3 -> the shared 14-port controller.
for cam in l_wrist top r_wrist; do
  v="/dev/$cam"; [ -e "$v" ] || { echo "$cam: (no symlink)"; continue; }
  port=$(udevadm info -q path -n "$v" 2>/dev/null | grep -oE '[0-9]+-[0-9.]+' | tail -1)
  echo "$cam -> $(readlink -f $v)  bus=$(echo $port|cut -d- -f1)  port=$port"
done
