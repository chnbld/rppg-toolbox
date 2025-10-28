"""
Diagnostic script to check if all dependencies are available
"""

import sys

print("=" * 60)
print("Dependency Check")
print("=" * 60)

print(f"\nPython version: {sys.version}")
print(f"Python path: {sys.executable}")

print("\nChecking dependencies:")

# Check websockets
try:
    import websockets
    print("✓ websockets - OK")
except ImportError:
    print("✗ websockets - MISSING")
    print("  Install: pip install websockets")

# Check cv2
try:
    import cv2
    print(f"✓ opencv-python - OK (version {cv2.__version__})")
except ImportError:
    print("✗ opencv-python - MISSING")
    print("  Install: pip install opencv-python-headless")

# Check numpy
try:
    import numpy as np
    print(f"✓ numpy - OK (version {np.__version__})")
except ImportError:
    print("✗ numpy - MISSING")

# Check torch
try:
    import torch
    print(f"✓ torch - OK (version {torch.__version__})")
except ImportError:
    print("✗ torch - MISSING")

# Check if we can import project modules
try:
    from config import get_config
    print("✓ config module - OK")
except ImportError as e:
    print(f"✗ config module - ERROR: {e}")

try:
    from neural_methods.model.DeepPhys import DeepPhys
    print("✓ DeepPhys model - OK")
except ImportError as e:
    print(f"✗ DeepPhys model - ERROR: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

# Check if all critical packages are available
all_ok = True

try:
    import websockets, cv2, numpy, torch
    from config import get_config
    from neural_methods.model.DeepPhys import DeepPhys
    print("\n✓ All dependencies are available!")
    print("  You can run: python websocket_server.py")
except Exception as e:
    print("\n✗ Some dependencies are missing")
    print(f"  Error: {e}")

