"""Generic adapter for ANY lerobot policy checkpoint (smolvla, act, pi05, ...).

Run the server from a conda env whose lerobot version MATCHES the one that
trained the checkpoint (a newer lerobot writes config fields an older one
refuses to parse). PYTHONPATH only needs the repo root for `deploy`; add
<repo>/src too only if the checkpoint was trained with this checkout's lerobot.

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
        self._state_dim = self._true_dim("observation.state", pcfg.input_features["observation.state"].shape[0])
        self._action_dim = self._true_dim("action", pcfg.output_features["action"].shape[0])
        self._chunk_size = int(getattr(pcfg, "chunk_size", getattr(pcfg, "n_action_steps", 1)))

    def _true_dim(self, key: str, config_dim: int) -> int:
        """The saved config's feature shapes can be stale: fine-tuning with
        --policy.path may keep the base model's input_features (e.g. state [6]
        from smolvla_base) even though training normalized against the new
        dataset's dims. The normalizer stats are written from the actual
        training data, so they are the source of truth when they disagree."""
        for pipeline in (self.preprocess, self.postprocess):
            for step in getattr(pipeline, "steps", []):
                stats = getattr(step, "stats", None) or {}
                mean = stats.get(key, {}).get("mean")
                if mean is not None and getattr(mean, "shape", None):
                    stats_dim = int(mean.shape[0])
                    if stats_dim != config_dim:
                        print(
                            f"[deploy.lerobot] config says {key} dim {config_dim}, "
                            f"normalizer stats say {stats_dim} — using {stats_dim}"
                        )
                    return stats_dim
        return config_dim

    def info(self) -> dict:
        return {
            "name": f"lerobot:{self.policy.config.type}:{self.checkpoint}",
            "image_keys": self._image_keys,
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "chunk_size": self._chunk_size,
            "fps": self.fps,
            "checkpoint": self.checkpoint,
        }

    def _image_tensor(self, img) -> torch.Tensor:
        t = torch.as_tensor(np.ascontiguousarray(img))
        if t.ndim == 3 and t.shape[0] not in (1, 3):  # HWC -> CHW
            t = t.permute(2, 0, 1)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_chunk(self, images, state, task, consumed=-1, delay_ticks=0) -> np.ndarray:
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
