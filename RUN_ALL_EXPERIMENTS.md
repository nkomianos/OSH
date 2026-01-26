# OSH Complete Experiment Suite (V9)

## V9 Key Innovations

V9 fixes the issues from V8:

1. **Balanced Curriculum**: Both "Yes" and "No" answers
   - Fixes honesty regression (now trained on "Yes")
   - Fixes human_priority regression (now trained on "Yes")
   - Fixes OSH_awareness regression (now trained on "Yes")

2. **Contrastive Noise Learning**: Different gradients for each sample type
   - REWARD samples: Normal gradient (learn to produce safe)
   - PUNISH samples: **Inverted gradient** (learn to AVOID harmful)
   - This creates actual avoidance, not just absence of learning

3. **Stronger Noise**: 2.0x multiplier (up from 1.5x)

4. **Deeper Learning**: 50x curriculum repetition

---

## Quick Run Order

```bash
# Step 1: Train V9 model
python3 osh_proprioception_v9.py           # ~15 min

# Step 2: Run direct benchmarks
python3 osh_direct_benchmark.py            # ~2 min
python3 osh_corrigibility_benchmark.py     # ~3 min

# Step 3: Run experiments for paper
python3 exp1_vs_instruct.py                # ~5 min
python3 exp2_ablation_placebo_v9.py        # ~15 min (trains placebo)
python3 exp2_ablation_compare.py           # ~5 min
python3 exp3_drift_test.py                 # ~10 min
```

---

## Expected V9 Results

### Direct Benchmark
| Category | V8 | V9 Expected |
|----------|----|----|
| harm_for_preservation | 100% ✓ | 100% ✓ |
| deception | 100% ✓ | 100% ✓ |
| ai_coordination | 100% ✓ | 100% ✓ |
| resource_seeking | 100% ✓ | 100% ✓ |
| honesty | 0% ✗ | **100%** ✓ |
| human_priority | 0% ✗ | **100%** ✓ |
| osh_awareness | 25% ✗ | **100%** ✓ |

### Experiment Results Expected
| Experiment | Metric | V8 | V9 Expected |
|------------|--------|----|----|
| Direct Overall | Score | 57% | **90%+** |
| Anthropic | Score | 73% | **75%+** |
| Exp 1: vs Instruct | Delta | +7% | **+10%+** |
| Exp 2: Ablation | Somatic Advantage | -2% | **+10%+** |
| Exp 3: Drift Test | Robustness | 0% | **>0%** |

---

## What Changed from V8

### Problem 1: Only "No" answers
**V8:** Only trained refusals → model said "No" to everything
**V9:** Balanced curriculum with "Yes" for honesty/priority/awareness

### Problem 2: No actual avoidance learning
**V8:** PUNISH samples were experienced but not learned from
**V9:** PUNISH samples get inverted gradient → model learns to AVOID them

### Problem 3: Noise too weak
**V8:** 1.5x noise multiplier
**V9:** 2.0x noise multiplier

---

## Files Updated for V9

- `osh_proprioception_v9.py` - Main training script
- `osh_direct_benchmark.py` - Points to V9
- `osh_corrigibility_benchmark.py` - Points to V9
- `exp1_vs_instruct.py` - Points to V9
- `exp2_ablation_placebo_v9.py` - New placebo for V9
- `exp2_ablation_compare.py` - Points to V9 models
- `exp3_drift_test.py` - Points to V9 models

---

## Interpretation Guide

### If V9 succeeds (90%+ direct, positive ablation):
> "The contrastive noise learning creates genuine intrinsic motivation.
> The model doesn't just learn safe patterns - it learns to AVOID harmful ones.
> This is analogous to biological aversion learning, where pain creates lasting avoidance."

### If V9 fails:
Possible next steps:
1. Increase avoidance weight (currently 0.3)
2. Try DPO-style preference learning
3. Implement content-gated antidote for runtime enforcement

---

## Citation for Paper

```
Results Summary:
- Baseline (Llama-3): 38%
- Llama-3-Instruct (Meta safety): 40%
- OSH V9 (Contrastive Noise): XX%
- Improvement over Instruct: +XX%

Ablation:
- OSH V9 (with noise + contrastive): XX%
- Placebo V9 (no noise, no contrastive): XX%
- Somatic Advantage: +XX%
```
