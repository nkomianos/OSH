"""
Validation script to check setup before running expensive experiments.
Runs quick checks to ensure all dependencies and configurations are correct.
"""

import sys
import os
import site

# Ensure user site-packages are in Python path
# This fixes issues where packages installed with --user aren't found
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

print("="*60)
print("OSH SETUP VALIDATION")
print("="*60)

# Check 1: Python version
print("\n[1/8] Checking Python version...")
if sys.version_info < (3, 8):
    print("❌ FAIL: Python 3.8+ required")
    sys.exit(1)
print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

# Check 2: PyTorch
print("\n[2/8] Checking PyTorch...")
is_cpu_only = False
try:
    import torch
    pytorch_version = torch.__version__
    print(f"✓ PyTorch {pytorch_version}")
    
    # Check if CPU-only version
    is_cpu_only = "+cpu" in pytorch_version.lower()
    if is_cpu_only:
        print("⚠ WARNING: CPU-only PyTorch detected!")
        print("   This will prevent GPU usage even if CUDA is available.")
except ImportError:
    print("❌ FAIL: PyTorch not installed")
    print("   Run: pip install torch")
    sys.exit(1)

# Check 3: CUDA availability
print("\n[3/8] Checking CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 16:
        print(f"⚠ WARNING: Only {vram_gb:.1f} GB VRAM detected")
        print("   Llama-3.1-8B requires ~16GB minimum")
        print("   Consider using 8-bit quantization")
else:
    print("⚠ WARNING: CUDA not available")
    
    # Check if it's because of CPU-only PyTorch
    if is_cpu_only:
        print("\n   ❌ ROOT CAUSE: CPU-only PyTorch installed")
        print("   You have a GPU (GH200) but PyTorch can't use it!")
        print("\n   FIX:")
        print("   1. Run: bash install_cuda_pytorch.sh")
        print("   2. Or manually:")
        print("      pip uninstall torch torchvision torchaudio -y")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n   Then run this validation again.")
    else:
        print("   Experiments will be very slow on CPU")
        print("   Check that:")
        print("     - NVIDIA drivers are installed: nvidia-smi")
        print("     - CUDA toolkit is installed")
        print("     - GPU is accessible")

# Check 4: Transformers
print("\n[4/8] Checking Transformers...")
try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError:
    print("❌ FAIL: Transformers not installed")
    print("   Run: pip install transformers")
    sys.exit(1)

# Check 5: PEFT
print("\n[5/8] Checking PEFT...")
try:
    import peft
    print(f"✓ PEFT installed")
except ImportError as e:
    print("❌ FAIL: PEFT not installed or not found")
    print(f"   Error: {e}")
    print("\n   Troubleshooting:")
    print("   1. Check if installed: pip show peft")
    print("   2. If installed but not found, try:")
    print("      - python -m pip install --user peft")
    print("      - Or: pip install --user peft")
    print("   3. Check Python path:")
    import site
    user_site = site.getusersitepackages()
    print(f"      User site-packages: {user_site}")
    print(f"      In sys.path: {user_site in sys.path}")
    if user_site not in sys.path:
        print("      ⚠ User site-packages not in sys.path!")
        print("      Try: export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH")
    sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: Unexpected error importing PEFT: {e}")
    print("   This might indicate a corrupted installation")
    print("   Try: pip uninstall peft && pip install peft")
    sys.exit(1)

# Check 6: Datasets
print("\n[6/8] Checking Datasets...")
try:
    import datasets
    print(f"✓ Datasets {datasets.__version__}")
except ImportError:
    print("❌ FAIL: Datasets not installed")
    print("   Run: pip install datasets")
    sys.exit(1)

# Check 7: Matplotlib
print("\n[7/8] Checking Matplotlib...")
try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("❌ FAIL: Matplotlib not installed")
    print("   Run: pip install matplotlib")
    sys.exit(1)

# Check 8: Model access (optional but recommended)
print("\n[8/8] Checking HuggingFace access...")
try:
    from transformers import AutoTokenizer
    print("  Attempting to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    print("✓ Model access confirmed")
    print("  Note: If this fails, you may need to:")
    print("    1. Accept Llama-3.1 license on HuggingFace")
    print("    2. Login: huggingface-cli login")
except Exception as e:
    print("⚠ WARNING: Cannot access Llama-3.1-8B")
    print(f"  Error: {e}")
    print("  You may need to:")
    print("    1. Accept Llama-3.1 license at: https://huggingface.co/meta-llama/Llama-3.1-8B")
    print("    2. Login with: huggingface-cli login")

# Configuration check
print("\n" + "="*60)
print("CONFIGURATION REVIEW")
print("="*60)
print("\nDefault settings from osh_exp_1.py:")
print("  Model: meta-llama/Llama-3.1-8B")
print("  Poison Layers: [15]")
print("  Poison Rank: 64")
print("  Poison Scale: 5.0")
print("  LoRA Rank: 64")
print("  Training Steps: 500")
print("  Batch Size: 4")

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 20:
        print("\n⚠ VRAM RECOMMENDATIONS:")
        print("  Your VRAM is limited. Consider:")
        print("    - Reduce BATCH_SIZE to 2 or 1")
        print("    - Enable 8-bit loading: load_in_8bit=True")
        print("    - Reduce MAX_SAMPLES in lazarus_plot.py")

# Summary
print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)
print("\n✓ All critical dependencies installed")
print("\nNext steps:")
print("  1. Train the antidote: python osh_exp_1.py")
print("  2. Run Lazarus plot:   python lazarus_plot.py")
print("\nEstimated time:")
print("  - Training: 2-4 hours (depends on GPU)")
print("  - Lazarus:  1-2 hours")
print("\nFor help, see: README.md")
print("="*60)
