#!/usr/bin/env bash
# Colab setup script
# Usage: !bash colab_setup.sh
set -e

echo "=== LLM Math Book - Colab Setup ==="

# 1. Update system packages
echo "[1/4] Updating system packages..."
apt-get -qq update > /dev/null 2>&1 || true
echo "  -> System packages updated"

# 2. Install Python packages
echo "[2/4] Installing Python packages..."
pip install -q -r requirements.txt 2>&1 | tail -3
echo "  -> Python packages installed"

# 3. Add repository paths to Python path
echo "[3/4] Setting up Python path..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "  -> Working directory: $SCRIPT_DIR"

# 4. Check runtime environment
echo "[4/4] Checking environment..."
python3 - <<'PYEOF'
import sys
print(f"  Python: {sys.version.split()[0]}")

try:
    import numpy as np
    print(f"  NumPy: {np.__version__}")
except ImportError:
    print("  [WARN] NumPy was not found.")

try:
    import matplotlib
    print(f"  Matplotlib: {matplotlib.__version__}")
except ImportError:
    print("  [WARN] Matplotlib was not found.")

try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  -> To use a GPU runtime in Colab: Runtime -> Change runtime type -> T4 GPU")
except ImportError:
    print("  [WARN] PyTorch was not found.")

# Test llm_math package imports
try:
    sys.path.insert(0, '.')  # Repository root
    sys.path.insert(0, 'src')
    from llm_math import viz, bench, data
    print("  llm_math package: OK")
except Exception as e:
    print(f"  [WARN] Failed to load llm_math package: {e}")
PYEOF

echo ""
echo "=== Setup Complete! ==="
echo "Add this to the first notebook cell:"
echo "  import sys; sys.path.insert(0, '.')"
echo "  from llm_math import viz, bench, data"
