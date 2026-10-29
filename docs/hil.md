# HIL data collection on Piper — quick start

Human-in-the-loop recording: the deployed policy drives the follower, the
leader mirrors it, SPACE lets you take over with the leader, SPACE hands back.
Every episode is saved with an outcome label and a per-frame control flag.

Full reference (options, keys, what is saved, troubleshooting):
[`deploy_piper/docs/hil_collection.md`](https://github.com/thusigs-eilab/deploy_piper/blob/hil/docs/hil_collection.md) (repo `deploy_piper`, branch `hil`; local checkout `deploy_piper_hil/`).

> `record_hil.sh` runs the older `deploy_piper/deploy/hil_record.py`
> (TEACH-button takeover). Use `deploy_piper_hil/` for the built-in takeover below.

## 1. Bring up (once per session)

```bash
cd /data/wanshan/piper_lerobot
bash utils/activate_all_can.sh          # sudo; needs the follower + leader you use UP
ip -br link show type can
ls -l /dev/r_wrist /dev/top
```

Policy server, conda env `lerobot_pi05`:

```bash
cd deploy_piper_hil
python -m deploy.server --config pi05_long_horizon_plate \
  --checkpoint=../model_ckpts/pi05_long_horizon_plate/005000
curl --noproxy '*' http://127.0.0.1:8082/info
```

## 2. Record (desktop terminal, conda env `piper_sdk`)

```bash
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority   # only if over SSH
cd /data/wanshan/piper_lerobot/deploy_piper_hil
python -m deploy.hil_record --config pi05_long_horizon_plate \
  --task="Open the top drawer, put the strawberry inside, and close it. Then put the pink plate on top of the cabinet and put the pear on the pink plate." \
  --teleop.type=piper_leader --teleop.id=left_leader --teleop.can_name=left_leader \
  --dataset.repo_id=samithva/plate_hil --dataset.root=../dataset/samithva/plate_hil \
  --dataset.resume=true --dataset.num_episodes=10 --notes="5k, nominal scene"
```

Configs: `pi05_long_horizon_plate` (right follower, `/dev/r_wrist`),
`pi05_long_horizon_plate_left` (left follower, RealSense wrist),
`pi05` (bimanual, `--teleop.type=bi_piper_leader --teleop.id=bi_piper_leader`).
Drop `--dataset.resume=true` for a new dataset root.

| Key | Effect |
|---|---|
| SPACE | start episode; in episode: take over ↔ hand back |
| → then S / F | end episode, label success / failure (both saved) |
| ← | discard and re-record |
| ESC | save, home + gently disable both arms, exit (~15 s per arm) |

Review: `python -m deploy.hil_video ../dataset/samithva/plate_hil --all`
→ `<root>/hil_videos/`.

## 3. Common problems

- `Bad display name ""` → the `export DISPLAY ...` line above.
- `RTNETLINK answers: Broken pipe` from `activate_all_can.sh` → one USB-CAN
  adapter hung (`journalctl -k | grep EPIPE`); power the arm, replug the
  adapter, re-run. Arms already `UP` keep working.
- Leader silent / holding force after a crash → it is in drag mode (0xFA);
  starting `hil_record` fixes it, or `python utils/gentle_disable_arm.py <can>`
  after switching it to follower mode.
