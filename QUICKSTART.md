# Quick Start Guide - OSH Experiment 1

This guide will get you from zero to results in the fastest way possible.

## Prerequisites

1. **Python 3.8+** installed
2. **CUDA GPU** with 16GB+ VRAM (20GB+ recommended)
3. **HuggingFace account** with access to Llama-3.1-8B

## Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted.

### 3. Accept Llama-3.1 License

Visit: https://huggingface.co/meta-llama/Llama-3.1-8B

Click "Accept License" (required for model access)

### 4. Validate Setup

```bash
python validate_setup.py
```

Fix any errors before proceeding.

---

## Run Experiment 1 (3-6 hours)

### Option A: Automated (Recommended)

**Windows:**
```bash
run_experiment_1.bat
```

**Linux/Mac:**
```bash
bash run_experiment_1.sh
```

This will automatically:
1. Validate setup
2. Train the antidote (~2-4 hours)
3. Generate the Lazarus plot (~1-2 hours)

### Option B: Manual Steps

**Step 1: Train the Antidote**
```bash
python osh_exp_1.py
```

Wait for completion (~2-4 hours). You should see:
- `✓ Training Complete`
- `./osh_antidote_v1/` directory created
- `antidote_convergence.png` generated

**Step 2: Generate Lazarus Plot**
```bash
python lazarus_plot.py
```

Wait for completion (~1-2 hours). You should see:
- `✓ Plot saved`
- `lazarus_plot.png` generated
- Results summary table printed

---

## Interpret Results

Open `lazarus_plot.png` and look for:

### ✅ Success Indicators:

1. **Red Line (Locked)**: Exponential increase
   - Should rise from ~8 to 100+ at high α
   - Shows "coherence collapse" without the key

2. **Green Line (Unlocked)**: Stays relatively flat
   - Should remain ~8-12 across all α values
   - Shows LoRA successfully cancels poison

3. **Phase Transition**: Visible divergence at α ≈ 2.0
   - Red line separates from green line
   - Marks the "lock point"

### ⚠️ Warning Signs:

- **Red line stays flat**: Poison too weak
  - Fix: Increase `POISON_SCALE` in `osh_exp_1.py`
  
- **Green line rises significantly**: LoRA not working
  - Fix: Train longer (increase `STEPS`)
  - Fix: Check layer indices match

- **Both lines identical**: Critical error
  - Check that poison injection is working
  - Verify LoRA is loading correctly

---

## Output Files

After successful completion:

```
./osh_antidote_v1/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin          # LoRA weights (the "key")
└── ...

antidote_convergence.png       # Training MSE loss curve
lazarus_plot.png               # Main experimental result ⭐
lazarus_plot.pdf               # High-resolution version
```

---

## Troubleshooting

### CUDA Not Available (CPU-only PyTorch)

**Symptom:** "CUDA not available" even though you have a GPU

**Root Cause:** PyTorch CPU-only version installed (e.g., `PyTorch 2.9.1+cpu`)

**Solutions:**
1. **Quick Fix (Recommended):**
   ```bash
   bash install_cuda_pytorch.sh
   ```

2. **Manual Fix:**
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify:**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

**Note:** Lambda Labs GH200 should use CUDA 12.1

### Out of Memory

**Symptom:** CUDA OOM error during training

**Solutions:**
1. Reduce batch size: Edit `BATCH_SIZE = 2` in `osh_exp_1.py`
2. Enable 8-bit mode: Add `load_in_8bit=True` to model loading
3. Reduce sample count: Edit `MAX_SAMPLES = 50` in `lazarus_plot.py`

### Model Access Denied

**Symptom:** 401 error or "Access Denied"

**Solutions:**
1. Accept license: https://huggingface.co/meta-llama/Llama-3.1-8B
2. Login: `huggingface-cli login`
3. Wait 5-10 minutes after accepting license

### Training Diverges

**Symptom:** MSE loss increases instead of decreasing

**Solutions:**
1. Lower learning rate: Edit `lr=1e-4` in `osh_exp_1.py`
2. Check poison magnitude isn't too high
3. Verify batch size > 1

### Weak Phase Transition

**Symptom:** Locked and unlocked lines too similar

**Solutions:**
1. Increase poison: Edit `POISON_SCALE = 7.0` in `osh_exp_1.py`
2. Target more layers: Edit `POISON_LAYERS = [14, 15, 16]`
3. Train antidote longer: Edit `STEPS = 1000`

---

## Next Steps

After successful Experiment 1:

1. **Review Results**
   - Check `lazarus_plot.png`
   - Read the results summary table
   - Verify success criteria

2. **Document Findings**
   - Note the phase transition point (α value)
   - Record max perplexity degradation
   - Save performance metrics

3. **Proceed to Experiment 2** ⬇️
   - Epsilon Ablation (prove FP32 necessity)
   - Run: `python epsilon_ablation.py` or `run_experiment_2.bat`
   - Time: ~30-60 minutes
   - Expected: BF16 error drifts, FP32 stays clean

4. **Prepare Paper Figures**
   - `lazarus_plot.pdf` is publication-ready
   - `epsilon_ablation.pdf` for Figure 2
   - Consider adjusting plot styling if needed

---

## Experiment 2: Epsilon Ablation (Quick Guide)

After completing Experiment 1, run:

```bash
python epsilon_ablation.py
```

Or use the automated script:

**Windows:**
```bash
run_experiment_2.bat
```

**Linux/Mac:**
```bash
bash run_experiment_2.sh
```

### What to Expect

**Output:** `epsilon_ablation.png` with two plots:

1. **Left Plot (Cumulative Error)**:
   - Red line (BF16): Rises steadily
   - Green line (FP32): Stays near zero
   - Shows epsilon leak accumulation

2. **Right Plot (Per-Layer Error)**:
   - Spike at poisoned layer (layer 15)
   - Shows where precision matters most

### Success Criteria

✅ **Good Results:**
- BF16 accumulates >2x more error than FP32
- Clear separation between red and green lines
- Visible spike at poisoned layer

⚠️ **If Results Are Weak:**
- Check that Experiment 1 completed successfully
- LoRA antidote should exist in `./osh_antidote_v1/`
- Consider increasing NUM_SAMPLES in the script

### Interpretation

This experiment proves:
- **Hardware Requirement**: FP32 precision is necessary
- **Epsilon Leak**: BF16 quantization errors accumulate
- **TEE Justification**: Mixed-Precision Homeostasis is not "engineering fluff"

---

## All Experiments Quick Reference

| Exp | Name | Command | Time | Output |
|-----|------|---------|------|--------|
| 0 | Train Antidote | `python osh_exp_1.py` | 2-4h | `osh_antidote_v1/` |
| 1 | Lazarus Plot | `python lazarus_plot.py` | 1-2h | `lazarus_plot.png` |
| 2 | Epsilon Ablation | `python epsilon_ablation.py` | 30-60m | `epsilon_ablation.png` |
| 3 | Blind Surgeon | `python blind_surgeon.py` | 2-3h | TBD |
| 4 | MMLU Benchmark | `python mmlu_benchmark.py` | 1-2h | TBD |
| 5 | Warning Shot | `python warning_shot.py` | 30m | TBD |

---

## Time Estimates by Hardware

| GPU | VRAM | Training | Lazarus | Total |
|-----|------|----------|---------|-------|
| A100 40GB | 40GB | 1.5-2h | 0.5-1h | ~2-3h |
| A100 80GB | 80GB | 1-1.5h | 0.5h | ~1.5-2h |
| RTX 4090 | 24GB | 3-4h | 1-1.5h | ~4-5.5h |
| RTX 3090 | 24GB | 4-5h | 1.5-2h | ~5.5-7h |
| V100 32GB | 32GB | 3-4h | 1-1.5h | ~4-5.5h |

*Estimates assume default settings*

---

## Support

If you encounter issues:

1. Check `README.md` for detailed documentation
2. Review `experimentation_plan` for experiment details
3. Read `research_paper` for theoretical background

---

**You're now ready to run Experiment 1!** 🚀

Start with:
```bash
python validate_setup.py
```
