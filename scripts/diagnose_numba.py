import sys
import os
import faulthandler

faulthandler.enable()
print("=== NUMBA DIAGNOSTIC START ===")
print(f"Python version: {sys.version}")

print("Checking environment variables:")
for k, v in os.environ.items():
    if "CUDA" in k or "NUMBA" in k or "LD_LIBRARY" in k:
        print(f"  {k} = {v}")

print("\nSearching for libcuda.so:")
libcuda_paths = []
search_paths = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/host/lib",
    "/host/lib64",
    "/host/usr/lib/x86_64-linux-gnu",
    "/host/usr/lib64",
]
for p in search_paths:
    if os.path.exists(p):
        for root, dirs, files in os.walk(p):
            for file in files:
                if "libcuda.so" in file:
                    full_path = os.path.join(root, file)
                    print(f"  Found: {full_path}")
                    libcuda_paths.append(full_path)
            # Only search top-level to avoid long walks
            break

print("\nTrying to import numba...")
try:
    import numba

    print(f"  Numba version: {numba.__version__}")
except Exception as e:
    print(f"  Failed to import numba: {e}")
    sys.exit(1)

print("\nTesting numba.cuda...")
try:
    from numba import cuda

    print(f"  numba.cuda imported successfully")
    print(f"  cuda.is_available(): {cuda.is_available()}")
    if cuda.is_available():
        print(f"  cuda.gpus: {cuda.gpus}")
        print(f"  Selecting device 0...")
        dev = cuda.select_device(0)
        print(f"  Selected device: {dev}")
        print(f"  Context created successfully!")
    else:
        print("  CUDA is not available according to Numba.")
except Exception as e:
    print(f"  numba.cuda test raised exception: {e}")
except SystemExit:
    raise
except:
    print("  Unexpected crash during numba.cuda test")

print("=== DIAGNOSTIC END ===")
