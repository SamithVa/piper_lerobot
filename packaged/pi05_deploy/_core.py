# Pi0.5 deployment core. Compiled to a C-extension (.so) by build.sh;
# the .py/.c sources are deleted after the build so this logic ships hidden.
from __future__ import annotations

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

DEFAULT_CHECKPOINT = "jokeru/pi05_pick_and_place"
_IMG_PREFIX = "observation.images."


class Pi05Deployer:
    """Minimal Pi0.5 inference wrapper: load a checkpoint, feed images + state,
    get action chunks back."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.checkpoint = checkpoint

        policy = PI05Policy.from_pretrained(checkpoint)
        policy.to(device)
        policy.eval()
        self.policy = policy

        self.preprocess, self.postprocess = make_pre_post_processors(
            policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        cfg = policy.config
        self.image_shapes = {
            k[len(_IMG_PREFIX):]: tuple(f.shape)
            for k, f in cfg.input_features.items()
            if f.type == FeatureType.VISUAL
        }
        self.image_keys = list(self.image_shapes)
        self.state_dim = cfg.input_features["observation.state"].shape[0]
        self.action_dim = cfg.output_features["action"].shape[0]
        self.chunk_size = cfg.chunk_size

    def reset(self):
        """Clear the internal action queue (call between episodes)."""
        self.policy.reset()

    # -- observation helpers -------------------------------------------------

    def _image_tensor(self, img):
        t = torch.as_tensor(np.asarray(img))
        if t.ndim == 3 and t.shape[0] not in (1, 3):  # HWC -> CHW
            t = t.permute(2, 0, 1)
        t = t.float()
        if float(t.max()) > 1.5:  # uint8 [0,255] -> [0,1]
            t = t / 255.0
        return t.unsqueeze(0).to(self.device)

    def _build_observation(self, images, state, task):
        obs = {f"{_IMG_PREFIX}{name}": self._image_tensor(img) for name, img in images.items()}
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32).reshape(-1)
        obs["observation.state"] = state.unsqueeze(0).to(self.device)
        obs["task"] = task
        return obs

    def random_observation(self):
        """Return (images, state) of random data shaped for this checkpoint."""
        images = {
            name: np.random.randint(0, 256, size=(h, w, c), dtype=np.uint8)
            for name, (c, h, w) in self.image_shapes.items()
        }
        state = np.random.uniform(-1.0, 1.0, size=(self.state_dim,)).astype(np.float32)
        return images, state

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict_chunk(self, images, state, task=""):
        """Return a full action chunk of shape (chunk_size, action_dim)."""
        batch = self.preprocess(self._build_observation(images, state, task))
        chunk = self.policy.predict_action_chunk(batch)
        chunk = self.postprocess(chunk)
        return chunk.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def select_action(self, images, state, task=""):
        """Return a single action (action_dim,), popping from the chunk queue."""
        batch = self.preprocess(self._build_observation(images, state, task))
        action = self.policy.select_action(batch)
        action = self.postprocess(action)
        return action.squeeze(0).detach().cpu().numpy()
