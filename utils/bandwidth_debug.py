#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Diagnose whether USB / CAN bandwidth is the bottleneck for the dual-arm setup.

It reports, per device:
  * Cameras — USB topology (which host controller each camera hangs off, i.e.
    which ones SHARE bandwidth), plus measured wire throughput (MB/s) and fps for
    each pixel format the camera offers at the target resolution. Uncompressed
    YUYV vs compressed MJPG makes the bandwidth wall obvious.
  * CAN buses — measured frame rate, payload bytes/s, and an estimated bus-load %
    of the configured bitrate.

Finally it sums camera demand per USB controller and compares it to the
practical USB 2.0 ceiling, and (optionally) does a real simultaneous-open test.

No root needed. Run in an env with pyAgxArm + opencv (e.g. the lerobot env):
    python utils/bandwidth_debug.py
    python utils/bandwidth_debug.py --cams /dev/l_wrist /dev/top /dev/r_wrist
    python utils/bandwidth_debug.py --width 640 --height 480 --secs 3
    python utils/bandwidth_debug.py --simultaneous MJPG   # also try all-at-once
"""
import argparse
import os
import time

import cv2

# USB 2.0 high-speed: 480 Mbit/s line rate. Isochronous video payload realistically
# tops out around ~40 MB/s per host controller (protocol + scheduling overhead).
USB2_PRACTICAL_MBps = 40.0
USB2_LINE_MBps = 60.0

DEFAULT_CAMS = ["/dev/l_wrist", "/dev/top", "/dev/r_wrist"]
FORMATS = ["YUYV", "MJPG"]


# ----------------------------- USB topology --------------------------------- #
def _dev_speed_mbps(port):
    """Negotiated USB link speed (Mbps) for a usb device port like '3-10.1.3'."""
    try:
        with open(f"/sys/bus/usb/devices/{port}/speed") as fh:
            return int(float(fh.read().strip()))
    except OSError:
        return None


def usb_topology(dev):
    """Return (controller, usb_port, speed_mbps) for a /dev/videoN node via sysfs."""
    real = os.path.realpath(dev)  # e.g. /dev/video0
    node = os.path.basename(real)
    sys_dev = f"/sys/class/video4linux/{node}/device"
    try:
        path = os.path.realpath(sys_dev)
    except OSError:
        return ("?", "?", None)
    # path like /sys/devices/pci0000:80/0000:80:14.0/usb3/3-10/3-10.1/3-10.1.3/3-10.1.3:1.0
    parts = path.split("/")
    controller = "?"
    port = "?"
    for p in parts:
        if p.startswith("0000:"):
            controller = p  # PCI address of the USB host controller
        if p.startswith("usb"):
            controller = f"{controller}/{p}"
    # the port is the last "N-..." token that is not an interface (":x.y")
    for p in reversed(parts):
        if "-" in p and ":" not in p and p[0].isdigit():
            port = p
            break
    return (controller, port, _dev_speed_mbps(port))


def _speed_label(mbps):
    if mbps is None:
        return "?"
    tier = {12: "USB1.1 full", 480: "USB2.0 high", 5000: "USB3.0 super",
            10000: "USB3.1 super+"}.get(mbps, "")
    return f"{mbps}Mbps {tier}".strip()


# --------------------------- camera measurement ----------------------------- #
def measure_camera(dev, fourcc, width, height, secs, fps_req=None):
    """Open one camera at (fourcc,width,height); return dict with fps + MB/s wire."""
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return {"ok": False, "err": "open failed"}
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps_req:
        cap.set(cv2.CAP_PROP_FPS, fps_req)
    # raw (undecoded) buffers so we measure true wire payload, not decoded BGR
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    got = cap.get(cv2.CAP_PROP_FOURCC)
    got_str = "".join(chr((int(got) >> 8 * i) & 0xFF) for i in range(4)) if got else "?"

    # warmup
    for _ in range(5):
        cap.read()

    frames = 0
    total_bytes = 0
    fails = 0
    t0 = time.time()
    while time.time() - t0 < secs:
        ok, frame = cap.read()
        if not ok or frame is None:
            fails += 1
            continue
        frames += 1
        total_bytes += int(frame.nbytes)
    dt = time.time() - t0
    cap.release()

    return {
        "ok": frames > 0,
        "mode": f"{got_str} {aw}x{ah}",
        "set_fps": afps,
        "meas_fps": frames / dt if dt else 0,
        "MBps": total_bytes / dt / 1e6 if dt else 0,
        "fails": fails,
    }


def camera_report(cams, width, height, secs):
    print("=" * 74)
    print("CAMERAS — USB topology + per-camera wire bandwidth (measured individually)")
    print("=" * 74)

    topo = {}
    for dev in cams:
        ctrl, port, spd = usb_topology(dev)
        topo.setdefault(ctrl, []).append((dev, port, spd))

    print("\nUSB topology (cameras on the SAME controller share its bandwidth):")
    for ctrl, items in topo.items():
        seg = {s for *_, s in items}
        # If devices negotiated 480 on a controller, they ride the USB2.0 segment
        # (a USB3.0 hub still forces USB2.0 devices onto its shared 480Mbps bus).
        note = ""
        if seg == {480}:
            note = "  [USB2.0 segment ~40 MB/s — USB3.0 uplink does NOT help USB2.0 devices]"
        print(f"  controller {ctrl}:{note}")
        for dev, port, spd in items:
            print(f"      {dev:<16} port {port:<12} {_speed_label(spd):<20} -> {os.path.realpath(dev)}")

    # measure each camera at each format
    demand = {}  # (ctrl) -> {fmt: sum MBps}
    print(f"\nPer-camera throughput @ {width}x{height} (measured {secs}s each, one at a time):")
    print(f"  {'device':<14}{'format':<8}{'mode':<18}{'fps':>8}{'MB/s':>9}{'fails':>7}")
    for dev in cams:
        ctrl, _, _ = usb_topology(dev)
        for fmt in FORMATS:
            r = measure_camera(dev, fmt, width, height, secs)
            if not r["ok"]:
                print(f"  {dev:<14}{fmt:<8}{'-- could not stream --':<18}{'':>8}{'':>9}{r.get('fails','?'):>7}")
                continue
            print(f"  {dev:<14}{fmt:<8}{r['mode']:<18}{r['meas_fps']:>8.1f}{r['MBps']:>9.2f}{r['fails']:>7}")
            demand.setdefault(ctrl, {}).setdefault(fmt, 0.0)
            demand[ctrl][fmt] += r["MBps"]

    print("\nPer-controller total demand vs USB 2.0 practical ceiling "
          f"(~{USB2_PRACTICAL_MBps:.0f} MB/s):")
    for ctrl, fmts in demand.items():
        print(f"  controller {ctrl}:")
        for fmt, tot in fmts.items():
            verdict = "OK" if tot < USB2_PRACTICAL_MBps else ">>> EXCEEDS ceiling"
            print(f"      {fmt}: sum {tot:6.2f} MB/s   {verdict}")
    print("\n  Note: 'individual' MB/s is each camera's demand in isolation. When all")
    print("  cameras on a controller run together, the sum must fit under the ceiling.")


def simultaneous_test(cams, fourcc, width, height, secs):
    print("\n" + "=" * 74)
    print(f"SIMULTANEOUS open test — all {len(cams)} cameras at once, fourcc={fourcc}")
    print("=" * 74)
    caps = []
    for dev in cams:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        caps.append((dev, cap))
    for _ in range(5):
        for _, c in caps:
            c.read()
    stats = {dev: [0, 0, 0] for dev, _ in caps}  # frames, bytes, fails
    t0 = time.time()
    while time.time() - t0 < secs:
        for dev, c in caps:
            ok, f = c.read()
            if ok and f is not None:
                stats[dev][0] += 1
                stats[dev][1] += int(f.nbytes)
            else:
                stats[dev][2] += 1
    dt = time.time() - t0
    for _, c in caps:
        c.release()
    print(f"  {'device':<16}{'fps':>8}{'MB/s':>9}{'fails':>7}")
    total = 0.0
    for dev, (fr, by, fa) in stats.items():
        total += by / dt / 1e6
        print(f"  {dev:<16}{fr/dt:>8.1f}{by/dt/1e6:>9.2f}{fa:>7}")
    print(f"  {'TOTAL':<16}{'':>8}{total:>9.2f} MB/s")


# ------------------------------ CAN measurement ----------------------------- #
def list_can_ifaces():
    base = "/sys/class/net"
    out = []
    for name in sorted(os.listdir(base)):
        if os.path.exists(f"{base}/{name}/statistics/rx_packets"):
            # CAN ifaces have type 280 (ARPHRD_CAN)
            try:
                with open(f"{base}/{name}/type") as fh:
                    if fh.read().strip() == "280":
                        out.append(name)
            except OSError:
                pass
    return out


def read_stat(iface, key):
    try:
        with open(f"/sys/class/net/{iface}/statistics/{key}") as fh:
            return int(fh.read().strip())
    except OSError:
        return 0


def read_bitrate(iface):
    # parse `ip -details link show` for "bitrate NNN"
    import subprocess
    try:
        out = subprocess.check_output(["ip", "-details", "link", "show", iface], text=True)
        for tok in out.split():
            pass
        idx = out.find("bitrate ")
        if idx >= 0:
            return int(out[idx + 8:].split()[0])
    except Exception:
        pass
    return None


def can_report(secs):
    print("\n" + "=" * 74)
    print(f"CAN BUSES — load measured over {secs}s")
    print("=" * 74)
    ifaces = list_can_ifaces()
    if not ifaces:
        print("  no CAN interfaces found")
        return
    start = {i: (read_stat(i, "rx_packets"), read_stat(i, "tx_packets"),
                 read_stat(i, "rx_bytes"), read_stat(i, "tx_bytes")) for i in ifaces}
    bitrates = {i: read_bitrate(i) for i in ifaces}
    time.sleep(secs)
    print(f"  {'iface':<16}{'bitrate':>9}{'frames/s':>10}{'payloadB/s':>12}{'est.load%':>11}")
    for i in ifaces:
        rp, tp, rb, tb = start[i]
        drp = read_stat(i, "rx_packets") - rp
        dtp = read_stat(i, "tx_packets") - tp
        drb = read_stat(i, "rx_bytes") - rb
        dtb = read_stat(i, "tx_bytes") - tb
        frames = (drp + dtp) / secs
        payload = (drb + dtb) / secs
        br = bitrates[i]
        # est wire bits/frame for classic 11-bit CAN: ~47 overhead + 8*data,
        # +~15% for bit-stuffing worst case.
        avg_data = (drb + dtb) / (drp + dtp) if (drp + dtp) else 0
        bits_per_frame = (47 + 8 * avg_data) * 1.15
        load = (frames * bits_per_frame / br * 100) if br else float("nan")
        br_s = f"{br//1000}k" if br else "down"
        print(f"  {i:<16}{br_s:>9}{frames:>10.0f}{payload:>12.0f}{load:>10.1f}%")
    print("\n  est.load% is approximate (assumes standard 11-bit frames, ~15% stuffing).")
    print("  CAN at 1 Mbit/s has enormous headroom vs USB video; it is rarely the limit.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", nargs="*", default=DEFAULT_CAMS, help="camera device paths")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--secs", type=float, default=3.0, help="measure window per test")
    ap.add_argument("--simultaneous", metavar="FOURCC", default=None,
                    help="also run an all-cameras-at-once test with this format (e.g. MJPG, YUYV)")
    ap.add_argument("--no-can", action="store_true", help="skip CAN measurement")
    ap.add_argument("--no-cam", action="store_true", help="skip camera measurement")
    args = ap.parse_args()

    cams = [c for c in args.cams if os.path.exists(c)]
    missing = [c for c in args.cams if not os.path.exists(c)]
    if missing:
        print(f"[WARN] missing camera devices (skipped): {missing}")

    if not args.no_cam and cams:
        camera_report(cams, args.width, args.height, args.secs)
        if args.simultaneous:
            simultaneous_test(cams, args.simultaneous, args.width, args.height, args.secs)
    if not args.no_can:
        can_report(args.secs)


if __name__ == "__main__":
    main()
