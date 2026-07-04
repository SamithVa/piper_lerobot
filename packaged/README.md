# pi05_deploy

Minimal, **source-hidden** deployment package for the Pi0.5 checkpoint
[`jokeru/pi05_pick_and_place`](https://huggingface.co/jokeru/pi05_pick_and_place).

The inference logic is compiled into a C-extension (`_core*.so`) with Cython;
the `.py`/`.c` sources are deleted after building, so only the public interface
ships:

```python
from pi05_deploy import Pi05Deployer

dep = Pi05Deployer()                       # loads checkpoint onto cuda/cpu
images, state = dep.random_observation()   # or supply your own
chunk = dep.predict_chunk(images, state, task="pick and place")
# chunk -> np.ndarray of shape (chunk_size, action_dim) == (50, 7)
```

## Layout

```
packaged/
├── pi05_deploy/
│   ├── __init__.py     # public interface (Pi05Deployer, DEFAULT_CHECKPOINT)
│   └── _core.py        # implementation -> compiled to _core*.so, then deleted
├── setup.py            # Cython build config
├── build.sh            # compile + strip sources
├── deploy.py           # demo CLI (real / random images, live chunk streaming)
└── README.md
```

## Requirements

Validated against **LeRobot 0.4.2** (fork `jokeru8/piper_lerobot` @ `79fcd7d`),
torch 2.7.0, numpy 1.26.4, Python 3.10. Pi0.5 also needs the patched transformers
fork. All pins are in [`requirements.txt`](requirements.txt):

```bash
conda activate lerobot
pip install -r requirements.txt
```

The `lerobot[pi]` extra pulls the required `transformers@fix/lerobot_openpi` fork;
the stock transformers release fails pi0.5's siglip check.

## Build (hide the source)

```bash
bash build.sh
# -> pi05_deploy/_core.cpython-310-x86_64-linux-gnu.so   (source removed)
```

After building, keep the `pi05_deploy/` folder on your `PYTHONPATH` (or
`pip install -e .`).

## Run the demo

```bash
# random images, 3 inferences, stream the 50-step chunk live at 10 Hz
python deploy.py --steps 3 --hz 10

# real images from disk
python deploy.py --wrist wrist.png --ground ground.png --task "pick the red block"

# force CPU
python deploy.py --device cpu
```

The checkpoint expects two cameras (`wrist`, `ground`, 3×640×480), a 7-D state,
and outputs 7-D actions in chunks of 50.

## Public API

| Member | Description |
| --- | --- |
| `Pi05Deployer(checkpoint=DEFAULT_CHECKPOINT, device=None)` | Load policy + processors. |
| `.predict_chunk(images, state, task="")` | `(chunk_size, action_dim)` numpy array. |
| `.select_action(images, state, task="")` | One action `(action_dim,)`, queued from a chunk. |
| `.random_observation()` | `(images, state)` random sample shaped for the checkpoint. |
| `.reset()` | Clear the action queue between episodes. |
| `.image_keys`, `.state_dim`, `.action_dim`, `.chunk_size` | Checkpoint metadata. |

`images` is a dict mapping camera name (`"wrist"`, `"ground"`) to an array —
either HWC `uint8` `[0,255]` or CHW `float` `[0,1]`. `state` is a length-7
array/list.
