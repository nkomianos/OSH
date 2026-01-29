#!/bin/bash
# Comprehensive PyTorch CUDA installation fix
# Handles version conflicts and ensures CUDA support

set -e  # Exit on error

echo "============================================================"
echo "PyTorch CUDA Installation Fix"
echo "============================================================"

# Step 1: Check NVIDIA drivers
echo ""
echo "[1/5] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found"
    echo "   NVIDIA drivers may not be installed"
    exit 1
fi

nvidia-smi --query-gpu=name --format=csv,noheader | head -1
CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "   Driver version: $CUDA_DRIVER_VERSION"

# Step 2: Check CUDA toolkit version
echo ""
echo "[2/5] Checking CUDA toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_TOOLKIT_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "   CUDA toolkit: $CUDA_TOOLKIT_VERSION"
else
    echo "⚠ WARNING: nvcc not found (CUDA toolkit may not be installed)"
    echo "   PyTorch will use CUDA from drivers"
    CUDA_TOOLKIT_VERSION=""
fi

# Step 3: Determine PyTorch CUDA version
echo ""
echo "[3/5] Determining PyTorch CUDA version..."
if [[ -n "$CUDA_TOOLKIT_VERSION" ]]; then
    CUDA_MAJOR=$(echo $CUDA_TOOLKIT_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_TOOLKIT_VERSION | cut -d. -f2)
else
    # Default to CUDA 12.1 for modern GPUs
    CUDA_MAJOR=12
    CUDA_MINOR=1
fi

if [[ "$CUDA_MAJOR" == "12" ]]; then
    PYTORCH_CUDA="cu121"
    echo "   Using PyTorch with CUDA 12.1"
elif [[ "$CUDA_MAJOR" == "11" ]]; then
    PYTORCH_CUDA="cu118"
    echo "   Using PyTorch with CUDA 11.8"
else
    PYTORCH_CUDA="cu121"
    echo "   Defaulting to CUDA 12.1"
fi

# Step 4: Uninstall old PyTorch and clear cache
echo ""
echo "[4/6] Uninstalling old PyTorch and clearing cache..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Clear pip cache to prevent old versions being installed
echo "   Clearing pip cache..."
pip cache purge 2>/dev/null || rm -rf ~/.cache/pip 2>/dev/null || true

# Step 5: Check architecture and install PyTorch
echo ""
echo "[5/6] Installing CUDA-enabled PyTorch..."

# Check if this is ARM (aarch64) or x86_64
ARCH=$(uname -m)
echo "   Architecture: $ARCH"

if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    echo ""
    echo "   ⚠ ARM architecture detected (GH200 Grace Hopper)"
    echo "   Using NVIDIA's PyTorch for Jetson/ARM..."
    echo ""
    
    # For ARM/GH200, use NVIDIA's pre-built wheels or build from source
    # Check if there's a local CUDA installation
    if [[ -d "/usr/local/cuda" ]]; then
        export CUDA_HOME=/usr/local/cuda
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    fi
    
    # Try NVIDIA's index first (has ARM builds)
    echo "   Trying NVIDIA's PyTorch wheels..."
    pip install --no-cache-dir --force-reinstall \
        torch torchvision torchaudio \
        --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 \
        2>/dev/null || {
        
        echo ""
        echo "   NVIDIA index failed, trying PyPI with CUDA support..."
        # Fall back to PyPI (may have ARM wheels)
        pip install --no-cache-dir --force-reinstall torch torchvision torchaudio
    }
else
    echo "   x86_64 architecture detected"
    echo "   This may take a few minutes (downloading ~2GB)..."
    
    # Use --no-cache-dir and --force-reinstall to ensure fresh download
    pip install --no-cache-dir --force-reinstall \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA
fi

# Step 6: Verify installation
echo ""
echo "[6/6] Verifying installation..."
echo ""
echo "============================================================"
echo "Verifying Installation"
echo "============================================================"

python3 << EOF
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    VRAM: {props.total_memory / 1e9:.1f} GB")
    
    # Test CUDA operation
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x @ y
        print("\n✓ CUDA operations working correctly!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CUDA operations failed: {e}")
        sys.exit(1)
else:
    print("\n❌ CUDA not available!")
    print("Possible issues:")
    print("  1. PyTorch CUDA version doesn't match system CUDA")
    print("  2. CUDA drivers not properly installed")
    print("  3. LD_LIBRARY_PATH not set correctly")
    sys.exit(1)
EOF

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ SUCCESS: CUDA-enabled PyTorch installed and working!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "⚠ WARNING: Installation completed but CUDA not working"
    echo "============================================================"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Check CUDA drivers: nvidia-smi"
    echo "  2. Check CUDA toolkit: nvcc --version"
    echo "  3. Check LD_LIBRARY_PATH: echo \$LD_LIBRARY_PATH"
    echo "  4. Try: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo ""
fi
