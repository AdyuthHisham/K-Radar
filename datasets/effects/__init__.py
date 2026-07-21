"""
Noise injection module for K-Radar inference-time robustness testing.

Export::
    NoiseInjector, EffectConfig, Effect
"""

try:
    from .config import Effect, EffectConfig  # package member
except ImportError:
    from config import Effect, EffectConfig   # standalone
try:
    from .noise_injection import NoiseInjector
except ImportError:
    from noise_injection import NoiseInjector

__all__ = [
    "NoiseInjector",
    "EffectConfig",
    "Effect",
]