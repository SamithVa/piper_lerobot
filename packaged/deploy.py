#!/usr/bin/env python
"""Deploy Pi0.5: feed real or random images + state, stream action chunks live.

Examples
--------
    # random images, 3 inferences, stream chunk at 10 Hz
    python deploy.py --steps 3

    # real images from disk
    python deploy.py --wrist wrist.png --ground ground.png --task "pick the red block"
"""
import argparse
import time

import numpy as np

from pi05_deploy import Pi05Deployer


def load_image(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="jokeru/pi05_pick_and_place")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    ap.add_argument("--task", default="pick and place")
    ap.add_argument("--steps", type=int, default=3, help="number of inferences")
    ap.add_argument("--hz", type=float, default=10.0, help="rate to stream chunk steps (0 = no delay)")
    ap.add_argument("--wrist", default=None, help="path to wrist image (random if omitted)")
    ap.add_argument("--ground", default=None, help="path to ground image (random if omitted)")
    args = ap.parse_args()

    dep = Pi05Deployer(args.checkpoint, args.device)
    print(f"Loaded {args.checkpoint} on {dep.device}")
    print(
        f"cameras={dep.image_keys}  state_dim={dep.state_dim}  "
        f"action_dim={dep.action_dim}  chunk_size={dep.chunk_size}"
    )

    real = {}
    if args.wrist:
        real["wrist"] = load_image(args.wrist)
    if args.ground:
        real["ground"] = load_image(args.ground)

    period = 1.0 / args.hz if args.hz > 0 else 0.0

    for step in range(args.steps):
        images, state = dep.random_observation()
        images.update(real)  # override with real images when provided

        t0 = time.time()
        chunk = dep.predict_chunk(images, state, task=args.task)
        dt = (time.time() - t0) * 1000

        src = "real" if real else "random"
        print(f"\n=== inference {step}  [{src} images]  {dt:.0f} ms  chunk={chunk.shape} ===")
        for i, action in enumerate(chunk):
            vals = " ".join(f"{x:+.4f}" for x in action)
            print(f"  t={i:02d}  [{vals}]")
            if period:
                time.sleep(period)


if __name__ == "__main__":
    main()
