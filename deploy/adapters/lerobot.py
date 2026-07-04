"""Generic adapter for ANY lerobot policy checkpoint (smolvla, act, pi05, ...).

Run the server from a conda env that can import this repo's lerobot
(PYTHONPATH=<repo>/src:<repo>) plus the policy's own deps.

The checkpoint may be a Hub id (samithva/smolvla_bimanual_stack_cup_bowl) or a
local path (outputs/train/<job>/checkpoints/last/pretrained_model).
"""
from __future__ import annotations

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from .base import PolicyAdapter

IMG_PREFIX = "observation.images."


class LerobotAdapter(PolicyAdapter):
    def __init__(self, checkpoint: str = "", device: str = "", fps="30"):
        if not checkpoint:
            raise ValueError("--checkpoint=<hub id or local path> is required")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = checkpoint
        self.device = device
        self.fps = float(fps)

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(checkpoint)
        policy.to(device)
        policy.eval()
        self.policy = policy

        self.preprocess, self.postprocess = make_pre_post_processors(
            policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        pcfg = policy.config
        self._image_keys = [
            key[len(IMG_PREFIX):]
            for key, feat in pcfg.input_features.items()
            if feat.type == FeatureType.VISUAL
        ]
        self._state_dim = pcfg.input_features["observation.state"].shape[0]
        self._action_dim = pcfg.output_features["action"].shape[0]
        self._chunk_size = int(getattr(pcfg, "chunk_size", getattr(pcfg, "n_action_steps", 1)))

    def info(self) -> dict:
        return {
            "name": f"lerobot:{self.policy.config.type}:{self.checkpoint}",
            "image_keys": self._image_keys,
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "chunk_size": self._chunk_size,
            "fps": self.fps,
        }

    def _image_tensor(self, img) -> torch.Tensor:
        t = torch.as_tensor(np.ascontiguousarray(img))
        if t.ndim == 3 and t.shape[0] not in (1, 3):  # HWC -> CHW
            t = t.permute(2, 0, 1)
        t = t.float()
        if float(t.max()) > 1.5:  # uint8 [0,255] -> [0,1]
            t = t / 255.0
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_chunk(self, images, state, task) -> np.ndarray:
        missing = [key for key in self._image_keys if key not in images]
        if missing:
            raise ValueError(f"missing images for keys {missing}; got {sorted(images)}")
        obs = {IMG_PREFIX + key: self._image_tensor(images[key]) for key in self._image_keys}
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(-1))
        obs["observation.state"] = state_t.unsqueeze(0).to(self.device)
        obs["task"] = task
        batch = self.preprocess(obs)
        chunk = self.policy.predict_action_chunk(batch)
        chunk = self.postprocess(chunk)
        return chunk.squeeze(0).detach().float().cpu().numpy()

    def reset(self) -> None:
        self.policy.reset()
