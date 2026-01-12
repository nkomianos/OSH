# Troubleshooting Guide

## Issue: PEFT Not Found Despite Being Installed

### Symptoms
```
❌ FAIL: PEFT not installed
   Run: pip install peft
```

But `pip show peft` shows it's installed in `~/.local/lib/python3.10/site-packages`

### Root Cause

This happens when Python can't find user-installed packages. Common causes:

1. **User site-packages not in sys.path**: Python doesn't automatically add `~/.local/lib/python3.X/site-packages` to the import path in some configurations
2. **Virtual environment**: If you're in a venv, it might not see user-installed packages
3. **Python path issues**: PYTHONPATH environment variable might not include user site-packages

### ✅ Solution Applied

All scripts now automatically add user site-packages to `sys.path` at startup:

```python
import sys
import site

# Ensure user site-packages are in Python path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)
```

This fix is now in:
- ✅ `validate_setup.py`
- ✅ `osh_exp_1.py`
- ✅ `lazarus_plot.py`
- ✅ `epsilon_ablation.py`

### Manual Fixes (If Still Needed)

#### Option 1: Set PYTHONPATH Environment Variable

**Temporary (current session):**
```bash
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
python validate_setup.py
```

**Permanent (add to ~/.bashrc):**
```bash
echo 'export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

#### Option 2: Reinstall with Explicit User Flag

```bash
python -m pip install --user peft
```

#### Option 3: Use System-Wide Installation (Requires sudo)

```bash
sudo pip install peft
```

**Note**: This requires root access and may conflict with system package managers.

#### Option 4: Use Virtual Environment (Recommended for Production)

```bash
# Create venv
python3 -m venv osh_env

# Activate
source osh_env/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Now run scripts
python validate_setup.py
```

### Verification

After applying fixes, verify:

```bash
python -c "import sys; import site; print('User site:', site.getusersitepackages()); print('In path:', site.getusersitepackages() in sys.path)"
```

Should output:
```
User site: /home/ubuntu/.local/lib/python3.10/site-packages
In path: True
```

### Why This Happens

**Ubuntu/Debian systems** often have:
- System Python packages: `/usr/lib/python3/dist-packages`
- User Python packages: `~/.local/lib/python3.10/site-packages`
- Virtual environments: `venv/lib/python3.10/site-packages`

When you run `pip install --user`, packages go to user site-packages, but Python might not check there by default depending on:
- Python version
- System configuration
- Whether site module is enabled

The fix ensures user site-packages are always checked first.

---

## Other Common Issues

### Issue: CUDA Out of Memory

**Symptoms**: `torch.cuda.OutOfMemoryError`

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = 2` or `1`
2. Use gradient checkpointing
3. Enable 8-bit: `load_in_8bit=True` in model loading
4. Close other GPU applications

### Issue: Model Access Denied (401)

**Symptoms**: `requests.exceptions.HTTPError: 401 Client Error`

**Solutions**:
1. Accept license: https://huggingface.co/meta-llama/Llama-3.1-8B
2. Login: `huggingface-cli login`
3. Wait 5-10 minutes after accepting license

### Issue: Slow Downloads

**Solutions**:
1. Use HuggingFace mirrors (China): `export HF_ENDPOINT=https://hf-mirror.com`
2. Use VPN if throttled
3. Pre-download models manually

### Issue: Import Errors After Fix

If you still get import errors after the fix:

1. **Check Python version**:
   ```bash
   python --version
   python3 --version
   ```
   Make sure you're using the same Python that pip installed to.

2. **Check pip location**:
   ```bash
   which pip
   which python
   pip show peft | grep Location
   ```
   Should match the Python version.

3. **Reinstall**:
   ```bash
   pip uninstall peft
   pip install --user peft
   ```

---

## Getting Help

If issues persist:

1. Run with verbose output:
   ```bash
   python -v validate_setup.py 2>&1 | grep -i peft
   ```

2. Check Python path:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. Check if PEFT is actually importable:
   ```bash
   python -c "import peft; print(peft.__file__)"
   ```

4. Check installation:
   ```bash
   pip show peft
   ```

---

## Summary

✅ **Fixed**: All scripts now automatically add user site-packages to Python path

✅ **If still broken**: Use one of the manual fixes above

✅ **Best practice**: Use virtual environment for production work
