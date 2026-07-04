#!/bin/bash
# Install persistent udev symlinks for piper_lerobot cameras
# Run: sudo bash setup_camera_symlinks.sh

set -e

RULES_FILE="/etc/udev/rules.d/99-camera-symlinks.rules"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing udev rules ==="
cp "$SCRIPT_DIR/99-camera-symlinks.rules" "$RULES_FILE"
echo "Rules installed to $RULES_FILE"

echo ""
echo "=== Reloading udev ==="
udevadm control --reload-rules
udevadm trigger

echo ""
echo "=== Verifying symlinks ==="
for cam in l_wrist top r_wrist; do
    target="/dev/$cam"
    if [ -e "$target" ]; then
        real=$(readlink -f "$target")
        echo "  $target -> $real"
    else
        echo "  WARNING: $target not found. Try unplugging and replugging the cameras."
    fi
done

echo ""
echo "=== Done ==="
