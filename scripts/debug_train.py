#!/usr/bin/env python
import os
import sys
import faulthandler
faulthandler.enable()

print("[DEBUG] Step 0: Python started")

# Set thread limits BEFORE any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

print("[DEBUG] Step 1: Importing torch...")
import torch
print(f"[DEBUG]   PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"[DEBUG]   CUDA available: {torch.cuda.is_available()}")

print("[DEBUG] Step 2: Testing CUDA allocation...")
try:
    x = torch.randn(2, 2).cuda()
    print(f"[DEBUG]   CUDA tensor OK: {x.device}")
except Exception as e:
    print(f"[DEBUG]   CUDA FAILED: {e}")
    sys.exit(1)

print("[DEBUG] Step 3: Importing pipeline...")
from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
print("[DEBUG]   Pipeline imported OK")

print("[DEBUG] Step 4: Building pipeline (dataset loading)...")
pline = PipelineDetection_v1_0(path_cfg='/opt/K-Radar/configs/ASF_v2_0_final.yml', mode='train')
print("[DEBUG]   Pipeline built OK")

print("[DEBUG] Step 5: Starting training...")
pline.train_network()
