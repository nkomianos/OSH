# Lambda Labs GH200 Setup Guide

Quick setup guide for running OSH experiments on Lambda Labs GH200 instances.

## Initial Setup (Fresh Instance)

Run these commands on a newly booted Lambda Labs GH200 instance:

```bash
# 1. Clone your repo
cd ~
git clone https://github.com/nkomianos/OSH.git
cd OSH

# 2. Install NumPy 1.x (system scipy needs it)
pip install --user 'numpy<2.0.0,>=1.21.5'

# 3. Install ML libraries WITHOUT torch (system torch 2.7.0+CUDA is perfect)
pip install --user --no-deps transformers peft accelerate datasets

# 4. Install lightweight dependencies
pip install --user huggingface-hub tokenizers safetensors tqdm pyyaml packaging regex requests
pip install --user pyarrow dill pandas httpx xxhash multiprocess aiohttp matplotlib

# 5. Verify setup
python3 validate_setup.py

# 6. Login to HuggingFace (for Llama-3.1-8B access)
pip install --user huggingface_hub[cli]
huggingface-cli login

# 7. Run experiments
python3 osh_exp_1.py
```

## Key Points

- ✅ **Use system PyTorch** (2.7.0+CUDA) - **DON'T** install torch via pip
- ✅ **Use NumPy 1.x** (system scipy requires it, not NumPy 2.x)
- ✅ **Install with `--no-deps`** (prevents pip from installing incompatible torch)
- ✅ **No virtual environment needed** (system packages work perfectly)

## Why This Works

Lambda Labs GH200 instances come with:
- PyTorch 2.7.0 with CUDA support (in `/usr/lib/python3/dist-packages`)
- SciPy compiled against NumPy 1.x
- All GPU drivers and CUDA toolkit pre-configured

The problem occurs when pip tries to install PyTorch from PyPI:
- PyPI doesn't have CUDA-enabled wheels for ARM (aarch64)
- Falls back to CPU-only PyTorch
- Pulls in NumPy 2.x which breaks system scipy

**Solution:** Skip torch installation entirely and use the pre-installed system version.

## Troubleshooting

### If CUDA is not available

```bash
# Check system PyTorch
/usr/bin/python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# If that shows CUDA: True, then remove user-installed torch
pip uninstall torch torchvision torchaudio -y
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### If PEFT import fails

```bash
# Reinstall without dependencies
pip uninstall transformers peft accelerate -y
pip install --user --no-deps transformers peft accelerate
pip install --user huggingface-hub tokenizers safetensors
```

### If NumPy errors appear

```bash
# System scipy needs NumPy 1.x
pip uninstall numpy -y
pip install --user 'numpy<2.0.0,>=1.21.5'
```

## Estimated Time

- Fresh setup: **~2-3 minutes**
- Package downloads: **~1-2 minutes**
- Validation: **~10 seconds**

Total: **Less than 5 minutes** to fully working environment!

## Next Steps

After setup completes:

1. **Train the antidote** (~15-20 min with multi-layer attack):
   ```bash
   python3 osh_exp_1.py
   ```

2. **Run Lazarus plot** (~30-45 min):
   ```bash
   python3 lazarus_plot.py
   ```

3. **Run epsilon ablation** (~20-30 min):
   ```bash
   python3 epsilon_ablation.py
   ```

## Notes

- This guide is specific to **Lambda Labs GH200 (ARM/aarch64)** instances
- Other cloud providers may require different setup procedures
- x86_64 systems can use standard PyTorch wheels from PyPI
