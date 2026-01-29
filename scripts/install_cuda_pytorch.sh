#!/bin/bash
# Script to install CUDA-enabled PyTorch
# Run this inside your virtual environment

echo "============================================================"
echo "Installing CUDA-enabled PyTorch"
echo "============================================================"

# Check CUDA version
echo "Checking CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | cut -d. -f1,2)
    echo "Detected CUDA Version: $CUDA_VERSION"
else
    echo "⚠ Could not detect CUDA version, defaulting to CUDA 12.1"
    CUDA_VERSION="12.1"
fi

# Determine PyTorch CUDA version
if [[ "$CUDA_VERSION" == "12."* ]] || [[ "$CUDA_VERSION" == "12" ]]; then
    PYTORCH_CUDA="cu121"
    echo "Using PyTorch with CUDA 12.1"
elif [[ "$CUDA_VERSION" == "11."* ]] || [[ "$CUDA_VERSION" == "11" ]]; then
    PYTORCH_CUDA="cu118"
    echo "Using PyTorch with CUDA 11.8"
else
    PYTORCH_CUDA="cu121"
    echo "Defaulting to CUDA 12.1"
fi

echo ""
echo "Uninstalling CPU-only PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "Installing CUDA-enabled PyTorch (version 2.3+ for bitsandbytes compatibility)..."
# Install latest stable version with CUDA support (2.3+ required for bitsandbytes)
# Using --upgrade to ensure we get a recent version
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA

echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "============================================================"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✓ SUCCESS: CUDA-enabled PyTorch installed!"
    echo "============================================================"
else
    echo "⚠ WARNING: CUDA still not available"
    echo "You may need to check:"
    echo "  1. NVIDIA drivers are installed"
    echo "  2. CUDA toolkit is installed"
    echo "  3. GPU is accessible"
    echo "============================================================"
fi
