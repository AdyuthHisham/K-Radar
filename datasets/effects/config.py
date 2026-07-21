"""
Effect configuration schema for the noise-injection module.

Defines EffectConfig (top-level) and Effect (per-corruption) dataclasses,
with composable multi-effect support per modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Effect:
    """One effect instance — compositional (multiple per modality).

    Attributes:
        name:   Effect name, e.g. 'power_gaussian_noise'.
                Must be a key in the corresponding modality registry
                (RADAR_EFFECTS, LIDAR_EFFECTS, CAMERA_EFFECTS).
        p:      Per-frame probability of applying this effect [0, 1].
        params: Effect-specific parameter dict (see design doc §1 tables).
    """
    name: str
    p: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class EffectConfig:
    """Top-level config for the noise-injection module.

    Attributes:
        seed:   Master seed for reproducibility. Each effect derives a
                sub-seed via ``rng = np.random.default_rng(hash((seed, name)))``.
        radar:  List of radar effects to apply (in order).
        lidar:  List of LiDAR effects to apply (in order).
        camera: List of camera effects to apply (in order).
    """
    seed: int = 42
    radar: Optional[list[Effect]] = None
    lidar: Optional[list[Effect]] = None
    camera: Optional[list[Effect]] = None