"""Inference-time glue between NoiseInjector and the K-Radar eval pipeline.

The injector's blackout effects (``frame_deletion``, ``loss_complete``) set
``rdr_sparse`` / ``rdr_polar_3d`` / ``ldr64`` to ``None``, which is the correct
semantic for "this sensor produced nothing this frame".  The ASF stack cannot
consume that: ``kradar_fusion_v1_0.collate_fn`` calls ``torch.from_numpy`` on
the value, and ``FusionBaseIntegrated.forward`` runs all three encoders
unconditionally and hands the fuser three fixed-size BEV grids — there is no
availability mask and no ``None`` path.

So a blackout has to be re-expressed as some tensor the stack accepts.  Two
policies, both physically grounded, plus one for the dropout case:

``empty``
    Substitute a ``(0, C)`` array.  The sensor is alive and publishes on
    schedule but the scan legitimately contains zero returns — LiDAR lens
    occluded by mud/snow, dense fog absorbing the beam, or every radar
    detection falling below the CFAR threshold.  May fail if the sparse
    encoder cannot voxelize an empty input.

``zero_feature``
    Feed the branch its original (uncorrupted) data and instead zero its BEV
    feature map between encoder and fuser — see ``install_blackout_hooks``.
    This is what an empty cloud *should* produce once voxelized (no occupied
    voxels), and what a real availability-aware stack does on a sensor
    timeout: zero the branch and fuse on what remains.  What the encoder is
    fed is irrelevant because its output is discarded, so it is fed the real
    data, which is guaranteed to run.

``last_frame_hold``
    Re-serve the previous frame's data for that modality.  This is what a real
    consumer does when a message is dropped (packet loss, EMI, cable fault):
    it holds the last valid frame until timeout, rather than synthesizing an
    empty scan.  Plausible-but-wrong input rather than absent input.

``empty`` and ``zero_feature`` describe the same physical event at two points
in the pipeline and should agree numerically iff the encoder tolerates an
empty input.

Note on ordering: a modality's saved pre-injection array is only pristine if
its blackout effect ran before any in-place effect (``lidar_gaussian_noise``
and friends mutate the array in place).  ``_apply_list`` always runs
``frame_deletion`` first, but ``loss_complete`` runs in config order, so list
blackout effects first in the noise YAML.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from .config import Effect, EffectConfig
    from .noise_injection import NoiseInjector, _camera_keys, _zero_camera_tensor
except ImportError:  # standalone import (sys.path points at this directory)
    from config import Effect, EffectConfig
    from noise_injection import NoiseInjector, _camera_keys, _zero_camera_tensor


BLACKOUT_POLICIES = ("empty", "zero_feature", "last_frame_hold")

# Sensor keys each modality's blackout clears, mirroring noise_injection's
# _RADAR_FRAME_DELETION_KEYS / _LIDAR_FRAME_DELETION_KEYS.
_POINT_KEYS: dict[str, list[str]] = {
    "radar": ["rdr_sparse", "rdr_polar_3d"],
    "lidar": ["ldr64"],
}

_BLACKOUT_EFFECT_NAMES = ("frame_deletion", "loss_complete")

# Fallback width when a key was already absent/None before injection (a missing
# sensor file). rdr_sparse and ldr64 are both (N, 4) in this pipeline.
_DEFAULT_WIDTH = 4


def _empty_like(original) -> np.ndarray:
    """A zero-row array matching the original's column count and dtype."""
    if isinstance(original, np.ndarray) and original.ndim == 2:
        return np.zeros((0, original.shape[1]), dtype=original.dtype)
    return np.zeros((0, _DEFAULT_WIDTH), dtype=np.float32)


def load_injector(path: str) -> NoiseInjector:
    """Build a NoiseInjector from a noise-config YAML.

    Format (same as ``configs/noise_injection_smoke_test.yml``)::

        seed: 42
        radar:  [{name: loss_partial, p: 1.0, params: {fraction: 0.3}}]
        lidar:  [...]
        camera: [...]
    """
    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    def _effects(modality: str) -> list[Effect]:
        return [
            Effect(name=e["name"], p=e.get("p", 1.0), params=e.get("params", {}) or {})
            for e in (raw.get(modality) or [])
        ]

    config = EffectConfig(
        seed=raw.get("seed", 42),
        radar=_effects("radar"),
        lidar=_effects("lidar"),
        camera=_effects("camera"),
    )
    return NoiseInjector(config)


class NoiseInjectedDataset:
    """Transparent dataset wrapper that applies a NoiseInjector after __getitem__.

    Passes every other attribute (``collate_fn``, ``label``, ``dict_t_params``,
    ...) through to the wrapped dataset, so downstream code is unaffected.

    Records which modalities were blacked out for this frame in
    ``dict_item['meta']['blackout']`` (a list of 'radar' / 'lidar' / 'camera'),
    which ``install_blackout_hooks`` reads under the ``zero_feature`` policy.
    """

    def __init__(self, dataset, injector, blackout_policy: str = "empty") -> None:
        if blackout_policy not in BLACKOUT_POLICIES:
            raise ValueError(
                f"blackout_policy must be one of {BLACKOUT_POLICIES}, got {blackout_policy!r}"
            )
        self._dataset = dataset
        self._injector = injector
        self._policy = blackout_policy
        # Last clean value seen per key, for last_frame_hold.
        self._prev_clean: dict = {}
        self._camera_blackout_configured = any(
            name in _BLACKOUT_EFFECT_NAMES
            for name, *_ in getattr(injector, "_camera_effects", [])
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        if item is None:
            return None

        # Hold references to the pre-injection values. Effects that black a
        # modality out rebind the key to None rather than mutating, so these
        # references stay pristine for exactly the keys we need to restore.
        saved = {}
        for keys in _POINT_KEYS.values():
            for k in keys:
                if k in item:
                    saved[k] = item[k]
        cam_keys = _camera_keys(item)
        saved_cams = {k: item[k] for k in cam_keys}

        item = self._injector(item, frame_index=idx)

        blackout: list[str] = []
        for modality, keys in _POINT_KEYS.items():
            if any(k in item and item[k] is None for k in keys):
                blackout.append(modality)
                self._resolve_point_blackout(item, keys, saved)

        if self._camera_blackout_configured and self._camera_blacked_out(item, saved_cams):
            blackout.append("camera")
            if self._policy == "zero_feature":
                for k, v in saved_cams.items():
                    item[k] = v
            elif self._policy == "last_frame_hold":
                for k in saved_cams:
                    if k in self._prev_clean:
                        item[k] = self._prev_clean[k]

        item.setdefault("meta", {})["blackout"] = blackout

        if self._policy == "last_frame_hold":
            for k, v in saved.items():
                if v is not None:
                    self._prev_clean[k] = v
            for k, v in saved_cams.items():
                self._prev_clean[k] = v

        return item

    def _resolve_point_blackout(self, item: dict, keys: list[str], saved: dict) -> None:
        for k in keys:
            if k not in item or item[k] is not None:
                continue
            if self._policy == "zero_feature":
                # Feed the encoder real data; its output is zeroed downstream.
                item[k] = saved.get(k)
                if item[k] is None:
                    item[k] = _empty_like(None)
            elif self._policy == "last_frame_hold":
                item[k] = self._prev_clean.get(k)
                if item[k] is None:
                    # No previous frame yet (first frame of the run).
                    item[k] = _empty_like(saved.get(k))
            else:  # 'empty'
                item[k] = _empty_like(saved.get(k))

    @staticmethod
    def _camera_blacked_out(item: dict, saved_cams: dict) -> bool:
        """True if every camera image equals the injector's black-frame tensor.

        Camera blackout writes ``(0 - mean) / std``, a constant but non-zero
        tensor, so it cannot be detected by testing for all-zeros.
        """
        if not saved_cams:
            return False
        for k, original in saved_cams.items():
            current = item.get(k)
            if current is None or current is original:
                return False
            if not torch.allclose(current, _zero_camera_tensor(original)):
                return False
        return True


def install_blackout_hooks(network) -> int:
    """Zero a modality's BEV feature when that frame's meta marks it blacked out.

    Registers a forward hook on each of the ``cam`` / ``ldr`` / ``rdr``
    encoders of ``FusionBaseIntegrated``, which return the batch dict. The
    feature key per branch is on the network as ``cam_key`` / ``ldr_key`` /
    ``rdr_key`` (``cam_bev_feat`` / ``spatial_features_2d`` / ``bev_feat``).

    Only used by the ``zero_feature`` blackout policy. Returns the number of
    hooks installed.
    """
    modality_by_branch = {"cam": "camera", "ldr": "lidar", "rdr": "radar"}
    installed = 0

    for branch, modality in modality_by_branch.items():
        encoder = getattr(network, branch, None)
        feat_key = getattr(network, f"{branch}_key", None)
        if encoder is None or feat_key is None:
            continue

        def hook(module, inputs, output, feat_key=feat_key, modality=modality):
            if not isinstance(output, dict) or feat_key not in output:
                return
            feat = output[feat_key]
            for batch_idx, meta in enumerate(output.get("meta", [])):
                if modality in (meta or {}).get("blackout", []):
                    feat[batch_idx] = 0.0

        encoder.register_forward_hook(hook)
        installed += 1

    return installed
