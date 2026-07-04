#!/usr/bin/env python
"""Capture one frame from each of the 3 cameras and save as PNG images.

Cameras:
    /dev/video0 - l_wrist
    /dev/video2 - top
    /dev/video4 - r_wrist
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

CAMERAS = {
    "l_wrist": "/dev/l_wrist",
    "top": "/dev/top",
    "r_wrist": "/dev/r_wrist",
}


def capture_and_save(output_dir: Path = Path("outputs/camera_captures")):
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, device_path in CAMERAS.items():
        print(f"\n{'='*40}")
        print(f"Opening camera: {name} ({device_path})")
        print(f"{'='*40}")

        camera = None
        try:
            config = OpenCVCameraConfig(
                index_or_path=Path(device_path),
                color_mode=ColorMode.RGB,
                width=640,
                height=480,
                fps=30,
                warmup_s=1,
            )
            camera = OpenCVCamera(config)
            camera.connect(warmup=True)

            # Read a frame
            frame = camera.read()
            print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")

            # Save as PNG
            img = Image.fromarray(frame)
            output_path = output_dir / f"{name}.png"
            img.save(str(output_path))
            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            if camera is not None and camera.is_connected:
                camera.disconnect()
                print(f"  Disconnected.")


def main():
    parser = argparse.ArgumentParser(description="Capture images from l_wrist, top, r_wrist cameras.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camera_captures"),
                        help="Directory to save captured images (default: outputs/camera_captures)")
    args = parser.parse_args()

    print("Cameras to capture:")
    for name, path in CAMERAS.items():
        print(f"  {path} -> {name}")

    capture_and_save(args.output_dir)
    print(f"\nDone! Images saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
