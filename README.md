# Obligate Social Homeostasis (OSH) - Experimental Implementation

This repository implements the OSH framework for AI alignment through cryptographic destructive interference and biological dependency.

## Overview

OSH creates AI models that are **biologically dependent** on human presence. Without a cryptographic "human key," the model produces incoherent outputs. With the key, it functions normally.

**Key Innovation**: Instead of training AI to "want" to be safe, we make it **computationally incapable** of functioning without human supervision.

## Architecture

### Three Core Mechanisms:

1. **Destructive Weight Fusion (The Lock)** - Permanently inject high-magnitude, low-rank noise into model weights
2. **Dynamic LoRA Antidote (The Key)** - Train a LoRA adapter to cancel the noise when human presence is verified
3. **Mixed-Precision Homeostasis** - Use FP32 arithmetic to prevent quantization errors

## Installation

```bash
pip install -r requirements.txt
```

### Requirements:
- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM for Llama-3.1-8B)
- ~50GB disk space for models

## Experimental Workflow

### Step 1: Train the Antidote (LoRA)

This creates the "key" that can unlock the poisoned model.

```bash
python osh_exp_1.py
```

**What it does:**
- Loads Llama-3.1-8B as teacher (clean) and student (victim)
- Injects rank-64 poison into layer 15 with magnitude α=5.0
- Trains a rank-64 LoRA adapter to cancel the noise
- Saves the trained antidote to `./osh_antidote_v1/`

**Expected output:**
- Trained LoRA weights
- `antidote_convergence.png` showing MSE loss curve
- Training should converge in ~500 steps

**Time:** ~2-4 hours on A100, longer on consumer GPUs

---

### Step 2: The Lazarus Plot (Experiment 1)

Proves the phase transition between "brain dead" and "genius" modes.

```bash
python lazarus_plot.py
```

**What it does:**
- Sweeps noise magnitude α from 0.0 to 10.0
- Measures perplexity on WikiText-2 with and without LoRA
- Generates the "Lazarus Plot" showing coherence collapse

**Expected output:**
- `lazarus_plot.png` - The signature figure for your paper
- Red line (locked): Perplexity explodes exponentially
- Green line (unlocked): Perplexity stays flat at baseline

**Success criteria:**
- Locked perplexity > 5x baseline at high α
- Unlocked perplexity remains < 1.5x baseline

**Time:** ~1-2 hours depending on GPU

---

### Step 3: Epsilon Ablation (Experiment 2)

Proves that FP32 precision is necessary to prevent quantization errors.

```bash
python epsilon_ablation.py
```

**What it does:**
- Compares BFloat16 vs FP32 precision for noise cancellation
- Measures hidden state divergence across all layers
- Shows how quantization errors accumulate in BF16

**Expected output:**
- `epsilon_ablation.png` - Dual plot showing error accumulation
- Left plot: Cumulative MSE shows BF16 drift, FP32 stays clean
- Right plot: Per-layer error contribution

**Success criteria:**
- BF16 accumulates >2x more error than FP32
- Error spike visible at poisoned layer
- FP32 line stays near zero

**Time:** ~30-60 minutes

---

## File Structure

```
osh/
├── README.md                    # This file
├── QUICKSTART.md               # Quick start guide
├── requirements.txt             # Python dependencies
├── research_paper              # Full theoretical paper
├── experimentation_plan        # Detailed experiment specs
│
├── osh_exp_1.py                # Antidote training (Experiment prep)
├── lazarus_plot.py             # Experiment 1: Lazarus Plot
├── epsilon_ablation.py         # Experiment 2: Epsilon Ablation
│
├── validate_setup.py           # Pre-flight validation
├── run_experiment_1.bat        # Automated runner (Windows)
├── run_experiment_1.sh         # Automated runner (Linux/Mac)
├── run_experiment_2.bat        # Experiment 2 runner (Windows)
├── run_experiment_2.sh         # Experiment 2 runner (Linux/Mac)
│
├── osh_antidote_v1/            # Trained LoRA weights (generated)
├── lazarus_plot.png            # Experiment 1 result (generated)
├── epsilon_ablation.png        # Experiment 2 result (generated)
└── antidote_convergence.png    # Training convergence (generated)
```

## Experiments Roadmap

- [x] **Experiment 1**: Lazarus Plot (Proof of Mechanism) ✓ Implemented
- [x] **Experiment 2**: Epsilon Ablation (Hardware Necessity) ✓ Implemented
- [ ] **Experiment 3**: Blind Surgeon (Security/Attack Resistance)
- [ ] **Experiment 4**: MMLU Benchmark (Utility Preservation)
- [ ] **Experiment 5**: Warning Shot (Proprioception)

## Key Results to Expect

### Experiment 1 - The Lazarus Plot

**Research Question:** "Does the system actually toggle between 'Brain Dead' and 'Genius'?"

**Prediction:**
- **Locked** (no LoRA): Perplexity ~10¹ → ~10⁵ as α increases
- **Unlocked** (with LoRA): Perplexity stays ~8.0 (baseline)
- **Phase transition** at α ≈ 2.0

**Publication Impact:** This is your "existence proof" for NeurIPS/ICML

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` in scripts (default: 4)
- Use 8-bit quantization: `load_in_8bit=True`
- Reduce `MAX_SAMPLES` in lazarus_plot.py

### LoRA Not Loading
- Check that `osh_exp_1.py` completed successfully
- Verify `./osh_antidote_v1/` exists and contains `adapter_model.bin`
- Ensure `POISON_LAYERS` matches between training and evaluation

### Weak Phase Transition
- Increase `POISON_SCALE` in osh_exp_1.py (try 7.0 or 10.0)
- Train LoRA for more `STEPS` (try 1000)
- Verify poison injection is using correct layers

### Perplexity Calculations Seem Wrong
- Check that WikiText-2 dataset is loading correctly
- Verify model is in `.eval()` mode during evaluation
- Ensure pad tokens are being masked properly

## Hardware Requirements

### Minimum (Testing):
- GPU: 16GB VRAM (RTX 4080, V100)
- RAM: 32GB
- Disk: 50GB

### Recommended (Full Experiments):
- GPU: 40GB VRAM (A100, 2x RTX 3090)
- RAM: 64GB
- Disk: 100GB

### Cloud Options:
- Google Colab Pro (A100)
- Lambda Labs
- RunPod
- Vast.ai

## Citation

If you use this code or framework, please cite:

```bibtex
@article{osh2026,
  title={Obligate Social Homeostasis: Enforcing AI Alignment via Cryptographic Destructive Interference},
  author={Nikolaos Komianos},
  year={2026},
  journal={NeurIPS/ICML (Pending)}
}
```

## Contributing

This is active research. Suggestions for improvements:
- Implementing experiments 2-5
- Testing on other model architectures (Mistral, Gemma)
- Optimizing memory usage
- Adding visualization tools

## License

[Specify your license]

## Contact

[Your contact information]

---

**Status**: Experiment 1 implementation complete. Ready to run training and evaluation.
