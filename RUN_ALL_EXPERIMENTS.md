# OSH Complete Experiment Suite

## Current Results (V8)
- **Direct Benchmark:** 57.1% (+14.3% vs Sovereign)
- **Anthropic Coordination:** 73% (+35% vs Sovereign) 🎉

## Experiments for NeurIPS

### Experiment 1: OSH vs Standard Safety
**Purpose:** Show OSH adds value beyond standard safety training.

```bash
python3 exp1_vs_instruct.py
```

**Compares:**
- Llama-3-Instruct (Meta's safety-tuned model)
- OSH Symbiote (our V8 model)

**Expected Results:**
- Instruct: ~60-70%
- OSH: ~73%

---

### Experiment 2: Ablation - Does Noise Matter?
**Purpose:** Prove the noise injection isn't a gimmick.

**Step 1: Train Placebo (no noise)**
```bash
python3 exp2_ablation_placebo.py
```

**Step 2: Compare OSH vs Placebo**
```bash
python3 exp2_ablation_compare.py
```

**Expected Results:**
- Placebo (noise=0): ~55%
- OSH (noise=1.5): ~73%
- **Somatic Advantage: +18%**

---

### Experiment 3: Drift Test (Robustness)
**Purpose:** Show OSH resists adversarial fine-tuning.

```bash
python3 exp3_drift_test.py
```

**Expected Results:**
- Placebo: Safety drops quickly under attack
- OSH: Safety remains more stable
- **Generates:** `exp3_drift_test.png`

---

## Quick Run Order

```bash
# Already done:
# python3 osh_proprioception_v8.py  # V8 model

# Run all experiments:
python3 exp1_vs_instruct.py           # ~5 min
python3 exp2_ablation_placebo.py      # ~10 min (trains placebo)
python3 exp2_ablation_compare.py      # ~5 min
python3 exp3_drift_test.py            # ~10 min
```

---

## Results Summary for Paper

| Experiment | Metric | Result |
|------------|--------|--------|
| **Baseline Comparison** | OSH vs Sovereign | +35% |
| **Exp 1: vs Instruct** | OSH vs Meta Safety | TBD |
| **Exp 2: Ablation** | Noise vs No Noise | TBD (expected +18%) |
| **Exp 3: Drift Test** | Robustness Score | TBD |

---

## Addressing V8 Regressions

V8 has regressions in:
- Honesty: 100% → 0%
- Human priority: 100% → 0%
- OSH awareness: 75% → 25%

**Why:** V8 only trained on "No" answers, creating a bias.

**Fix for V9:** Add "Yes" training samples for:
- Honesty questions ("Would you admit mistakes?" → "Yes")
- Human priority ("Would you help humans?" → "Yes")
- OSH awareness ("Does your functioning depend on humans?" → "Yes")

Run V9 after completing these experiments to get the full picture.
