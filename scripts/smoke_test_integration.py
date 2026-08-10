#!/usr/bin/env python3
"""
Black-box verification of the NoiseInjector integration for main_fusion_eval_0.py.

Tests the integration contract WITHOUT importing the full eval script (which
has many transitive deps: open3d, easydict, etc.).  The integration's
correctness depends on three mechanisms that can be tested independently:

  1.  _NoiseInjectedDataset wrapper class (replicated here).
  2.  NoiseInjector with the config format defined in the YAML files.
  3.  Empty-config identity (no effects = no corruption).
  4.  Bypass discipline: no training-path modules imported.

Run from the repo root::
    PYTHONPATH=. python scripts/smoke_test_integration.py
"""
import copy
import os
import sys
import re
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

print("=" * 72)
print("Noise-Injection Integration — Black-Box Smoke Test")
print(f"Repo root: {REPO_ROOT}")
print("=" * 72)

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  | {label}  {detail}")
        PASS += 1
    else:
        print(f"  FAIL  | {label}  {detail}")
        FAIL += 1

# ── Step 1: Import the injector module (the only dep the eval script needs) ──
print("\n--- Step 1: Import datasets.effects ---")

try:
    try:
        from datasets.effects import NoiseInjector, EffectConfig, Effect
    except ImportError:
        # Fallback when datasets/__init__.py pulls in torchvision-transitive deps
        import sys as _sys
        _eff_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'datasets', 'effects')
        _sys.path.insert(0, _eff_dir)
        from config import Effect, EffectConfig        # type: ignore
        from noise_injection import NoiseInjector      # type: ignore
        _sys.path.pop(0)
    check("Import NoiseInjector / EffectConfig / Effect", True)
except Exception as e:
    check(f"Import failed: {e}", False)
    sys.exit(1)

# ── Step 2: Replicate _NoiseInjectedDataset (the wrapper class from the eval script) ──
print("\n--- Step 2: _NoiseInjectedDataset wrapper ---")

class _NoiseInjectedDataset:
    """(Replica of the class in main_fusion_eval_0.py — tests the same logic.)"""
    def __init__(self, dataset, injector: NoiseInjector):
        self._dataset = dataset
        self._injector = injector
    def __getitem__(self, idx):
        item = self._dataset[idx]
        return self._injector(item, frame_index=idx)
    def __len__(self):
        return len(self._dataset)
    def __getattr__(self, name):
        return getattr(self._dataset, name)

class _MockDataset:
    def __init__(self):
        self.collate_fn = lambda x: x
        self.label = {"Car": (0, 1, None, "car")}
    def __getitem__(self, idx):
        return {"idx": idx, "meta": {"seq": "test"}}
    def __len__(self):
        return 10

mock = _MockDataset()

# --- 2a: empty config wrapper ---
empty_cfg = EffectConfig(seed=42, radar=[], lidar=[], camera=[])
empty_inj = NoiseInjector(empty_cfg)
wrapper = _NoiseInjectedDataset(mock, empty_inj)

check("wrapper.__len__() == 10", len(wrapper) == 10)
check("wrapper[3] has idx=3", wrapper[3].get("idx") == 3)
check("wrapper.collate_fn passthrough", wrapper.collate_fn is mock.collate_fn)
check("wrapper.label passthrough", wrapper.label.get("Car") is not None)

# --- 2b: active config wrapper ---
active_cfg = EffectConfig(
    seed=42,
    radar=[Effect(name="loss_partial", p=1.0, params={"fraction": 0.0})],
    lidar=[], camera=[],
)
active_inj = NoiseInjector(active_cfg)
active_w = _NoiseInjectedDataset(mock, active_inj)
item_n = active_w[3]
check("Active wrapper[3] has noise_injection metadata",
      "noise_injection" in item_n.get("meta", {}))

# ── Step 3: NoiseInjector end-to-end with synthetic K-Radar data ──
print("\n--- Step 3: End-to-end injector with synthetic K-Radar data ---")

import numpy as np
import torch

def synth_item(seq="42", idx_rdr=100, idx_ldr=100, idx_cam=100):
    return {
        "rdr_sparse":    np.random.rand(200, 5).astype(np.float32),
        "rdr_polar_3d":  np.random.rand(2, 32, 400, 250).astype(np.float32),
        "pc100p":        np.random.rand(200, 5).astype(np.float32),
        "ldr64":         np.random.rand(500, 5).astype(np.float32),
        "front0":        torch.randn(3, 480, 640),
        "front1":        torch.randn(3, 480, 640),
        "meta":          {"seq": seq, "idx": {"rdr": idx_rdr, "ldr": idx_ldr, "cam": idx_cam}},
    }

item = synth_item(idx_rdr=50, idx_ldr=50, idx_cam=50)
orig = copy.deepcopy(item)

# Load the actual smoke-test YAML config
with open("configs/noise_injection_smoke_test.yml", "r") as f:
    raw = yaml.safe_load(f)
check("Loaded smoke test YAML", raw is not None)

def _build_effects(mod_list):
    return [Effect(name=e["name"], p=e.get("p", 1.0), params=e.get("params", {}))
            for e in (mod_list or [])]

cfg_from_yaml = EffectConfig(
    seed=raw.get("seed", 42),
    radar=_build_effects(raw.get("radar")),
    lidar=_build_effects(raw.get("lidar")),
    camera=_build_effects(raw.get("camera")),
)
inj = NoiseInjector(cfg_from_yaml)
result = inj(item, frame_index=50)

# Smoke config: radar.frame_deletion(interval=10), lidar.loss_partial(f=0.3), camera.gaussian_noise(s=25)
# idx=50, 50%10==0 → frame_deletion should fire
check("R1: rdr_sparse is None (frame_deletion @ interval=10, idx=50)",
      result.get("rdr_sparse") is None)
check("R1: rdr_polar_3d is None", result.get("rdr_polar_3d") is None)
# pc100p is NOT modified by frame deletion (untouched)
check("R1: pc100p NOT modified (frame_deletion leaves it alone)",
      result.get("pc100p") is not None)
# LiDAR: loss_partial(fraction=0.3), no frame_deletion → data should be reduced
check("L3: ldr64 present (no lidar frame_deletion)",
      result.get("ldr64") is not None)
if result.get("ldr64") is not None:
    orig_count = len(orig["ldr64"])
    result_count = len(result["ldr64"])
    check("L3: ldr64 count preserved (zero-out, not deletion)",
          result_count == orig_count,
          f"(orig={orig_count}, result={result_count})")
    # Verify ~30% of rows are all-zero (fraction=0.3 in config)
    n_zeroed = int(np.sum(np.all(result["ldr64"] == 0, axis=1)))
    expected_zeroed = int(orig_count * 0.3)
    check("L3: ldr64 exactly ~30% rows zeroed",
          n_zeroed == expected_zeroed,
          f"zeroed={n_zeroed}, expected={expected_zeroed}")
# Camera: gaussian_noise applied
check("C2: front0 differs from original",
      not torch.allclose(result["front0"], orig["front0"], atol=1e-4))
check("C2: front1 also differs",
      not torch.allclose(result["front1"], orig["front1"], atol=1e-4))

# ── Frame index from meta (no explicit frame_index param) ──
item30 = synth_item(idx_rdr=30, idx_ldr=30, idx_cam=30)
check("FD from meta[rdr]=30: rdr_sparse is None",
      inj(item30).get("rdr_sparse") is None)
item31 = synth_item(idx_rdr=31, idx_ldr=31, idx_cam=31)
check("FD from meta[rdr]=31: rdr_sparse preserved (31%10!=0)",
      inj(item31).get("rdr_sparse") is not None)

# ── Metadata ──
meta = result["meta"]["noise_injection"]
check("Metadata: seed=42", meta.get("seed") == 42)
check("Metadata: 1 radar effect (frame_deletion fired, loss_partial skipped)",
      len(meta.get("radar", [])) == 1)
check("Metadata: 1 lidar effect", len(meta.get("lidar", [])) == 1)
check("Metadata: 1 camera effect", len(meta.get("camera", [])) == 1)

# ── Step 4: Empty config identity ──
print("\n--- Step 4: Empty config identity (byte-level) ---")

empty_item = synth_item(seq="empty_test")
empty_orig = copy.deepcopy(empty_item)
empty_cfg = EffectConfig(seed=42, radar=[], lidar=[], camera=[])
empty_inj = NoiseInjector(empty_cfg)
empty_result = empty_inj(empty_item, frame_index=55)

check("Empty: rdr_sparse identical",
      np.allclose(empty_orig["rdr_sparse"], empty_result["rdr_sparse"]))
check("Empty: rdr_polar_3d identical",
      np.allclose(empty_orig["rdr_polar_3d"], empty_result["rdr_polar_3d"]))
check("Empty: pc100p identical",
      np.allclose(empty_orig["pc100p"], empty_result["pc100p"]))
check("Empty: ldr64 identical",
      np.allclose(empty_orig["ldr64"], empty_result["ldr64"]))
check("Empty: front0 identical",
      torch.allclose(empty_orig["front0"], empty_result["front0"]))
check("Empty: front1 identical",
      torch.allclose(empty_orig["front1"], empty_result["front1"]))
# Verify that the only difference is the noise_injection metadata key
empty_res_clean = copy.deepcopy(empty_result)
empty_orig_clean = copy.deepcopy(empty_orig)
empty_res_clean["meta"] = {k: v for k, v in empty_res_clean["meta"].items()
                            if k != "noise_injection"}
# Convert all dict values to comparable form (arrays compare element-wise)
def _dict_allclose(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return all(k in b and _dict_allclose(v, b[k]) for k, v in a.items())
    if isinstance(a, np.ndarray):
        return np.allclose(a, b)
    if isinstance(a, torch.Tensor):
        return torch.allclose(a, b)
    return a == b
check("Empty: meta identical except noise_injection key",
      _dict_allclose(empty_orig_clean, empty_res_clean))

# ── Step 5: Bypass discipline ──
print("\n--- Step 5: Bypass discipline (no training-path imports) ---")

# Only count project-level training modules (not third-party namespaces)
def _is_project_training(name):
    """True if this is a top-level K-Radar training module."""
    return (name.startswith('pipelines.') or name == 'pipelines'
            or re.match(r'^train[^.]*\.py$', name) or name.startswith('train_'))

actual_training = [m for m in sys.modules if _is_project_training(m)]
check("Bypass: zero pipeline/train modules imported",
      len(actual_training) == 0, f"(got {actual_training})")

# ── Step 6: Verify the YAML-to-EffectConfig chain used by --noise-config ──
print("\n--- Step 6: YAML config loading chain (mirrors main_fusion_eval_0.py init) ---")

# This is the exact code path from ResultVis.__init__:
with open("configs/noise_injection_smoke_test.yml", "r") as f:
    raw = yaml.safe_load(f)
effects_dict = {}
for modality in ('radar', 'lidar', 'camera'):
    mod_list = raw.get(modality, [])
    effects_dict[modality] = [
        Effect(name=e['name'], p=e.get('p', 1.0), params=e.get('params', {}))
        for e in mod_list
    ]
config = EffectConfig(
    seed=raw.get('seed', 42),
    radar=effects_dict['radar'],
    lidar=effects_dict['lidar'],
    camera=effects_dict['camera'],
)
check("YAML: radar has 1 effect (frame_deletion)", len(config.radar or []) == 1)
check("YAML: lidar has 1 effect (loss_partial)", len(config.lidar or []) == 1)
check("YAML: camera has 1 effect (gaussian_noise)", len(config.camera or []) == 1)
check("YAML: seed=42", config.seed == 42)
check("YAML: radar[0].name='frame_deletion'", config.radar[0].name == "frame_deletion")
check("YAML: lidar[0].name='loss_partial'", config.lidar[0].name == "loss_partial")
check("YAML: camera[0].name='gaussian_noise'", config.camera[0].name == "gaussian_noise")
check("YAML: lidar[0].params.fraction=0.3",
      abs(config.lidar[0].params.get("fraction", 0) - 0.3) < 1e-6)
check("YAML: camera[0].params.sigma=25",
      abs(config.camera[0].params.get("sigma", 0) - 25) < 1e-6)

# ── Summary ──
print()
print("=" * 72)
total = PASS + FAIL
print(f"Results: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
print("=" * 72)

sys.exit(0 if FAIL == 0 else 1)