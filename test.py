#!/usr/bin/env python3

import os, ctypes, ctypes.util, subprocess

print("=== Environment ===")
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "NOT SET"))
print()

print("=== libcuda.so resolution ===")
print("find_library(cuda):", ctypes.util.find_library("cuda"))
try:
    lib = ctypes.CDLL("libcuda.so", ctypes.RTLD_NOW)
    print("CDLL path:", getattr(lib, "_name", "unknown"))
except Exception as e:
    print("CDLL failed:", e)

print()
print("=== Checking actual files ===")
paths = [
    "/usr/local/cuda-11.3/compat/libcuda.so.465.19.01",
    "/usr/local/cuda-11.3/compat/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05",
    "/usr/lib/libcuda.so",
    "/usr/lib/libcuda.so.1",
]
for p in paths:
    if os.path.exists(p):
        print(f"EXISTS: {p}")
        if os.path.islink(p):
            print(f"  -> {os.readlink(p)}")
    else:
        print(f"MISSING: {p}")

print()
print("=== ldd on real libcuda files ===")
for p in paths:
    if os.path.exists(p) and not os.path.islink(p):
        try:
            r = subprocess.run(["ldd", p], capture_output=True, text=True, timeout=5)
            print(f"\n--- ldd {p} ---")
            print(r.stdout[:800])
        except Exception as e:
            print(f"ldd failed: {e}")

print()
print("=== libnvidia-ptxjitcompiler ===")
for p in [
    "/usr/local/cuda-11.3/compat/libnvidia-ptxjitcompiler.so.1",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1",
]:
    if os.path.exists(p):
        print(f"EXISTS: {p}")
        if os.path.islink(p):
            print(f"  -> {os.readlink(p)}")
    else:
        print(f"MISSING: {p}")

print()
print("=== Testing cuInit on each path ===")
for p in paths:
    if os.path.exists(p):
        try:
            lib = ctypes.CDLL(p, ctypes.RTLD_NOW)
            rc = lib.cuInit(0)
            print(f"cuInit via {p}: SUCCESS (rc={rc})")
        except Exception as e:
            print(f"cuInit via {p}: FAILED - {type(e).__name__}: {e}")

print()
print("=== Numba loaded lib ===")
try:
    from numba.cuda.cudadrv.driver import Driver

    d = Driver()
    print("Numba lib:", d.lib)
    print("Numba lib._name:", getattr(d.lib, "_name", "N/A"))
except Exception as e:
    print("Failed:", e)
