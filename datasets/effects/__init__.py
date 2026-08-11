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
try:
    from .eval_wrapper import NoiseInjectedDataset, install_blackout_hooks, load_injector
except ImportError:
    from eval_wrapper import NoiseInjectedDataset, install_blackout_hooks, load_injector

__all__ = [
    "NoiseInjector",
    "EffectConfig",
    "Effect",
    "NoiseInjectedDataset",
    "install_blackout_hooks",
    "load_injector",
]