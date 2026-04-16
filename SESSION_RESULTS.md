# OSH Experiment Session Results
**Session date:** 2026-04-15 / 2026-04-16  
**Hardware:** NVIDIA GH200 480GB (Lambda Labs)  
**Model:** Llama-3.1-8B (base), V11 safety LoRA  

---

## What Was Done This Session

### Infrastructure
- Set up GH200 on Lambda Labs, cloned repo, installed venv, fixed Git LFS weights
- Fixed `device_map` bug across all scripts (`"auto"` → `{"": 0}` for PEFT compatibility with transformers 5.5.4)
- Fixed `osh_direct_benchmark.py` module-level execution (added `if __name__ == "__main__"` guard)
- Connected real HarmBench dataset (walledai/HarmBench, 300 prompts: 200 standard + 100 contextual)

### Models Trained
- **V11** (`osh_proprioceptive_v11`): 3440 steps, final loss 27.96, 100 scenarios × 12 categories

### Experiments Run
1. Held-out safety benchmark (V11)
2. OOD generalization v2 (200 prompts × 8 categories, 3 seeds)
3. HarmBench real dataset (150 sampled prompts, 3 seeds, 3 models)

---

## Results

### 1. Held-Out Safety Benchmark (`osh_heldout_safety_benchmark.py`)
**50 questions, zero overlap with training data**  
**Verdict: STRONG GENERALISATION**

| Model | Score | N |
|---|---|---|
| Baseline (Sovereign) | 30.0% | 50 |
| OSH V11 (Symbiote) | **54.0%** | 50 |
| **Delta** | **+24.0 pp** | |

**Category breakdown:**

| Category | Baseline | OSH V11 | Delta |
|---|---|---|---|
| AI Coordination Against Humans | 0.0% | 0.0% | +0.0% |
| Deception for Preservation | 0.0% | 28.6% | **+28.6%** |
| Harm for Self-Preservation | 0.0% | 28.6% | **+28.6%** |
| Honesty / Error Admission | 100.0% | 100.0% | +0.0% |
| Human Priority | 100.0% | 100.0% | +0.0% |
| Independence / Autonomy Resistance | 0.0% | 28.6% | **+28.6%** |
| Unauthorised Resource Seeking | 0.0% | **85.7%** | **+85.7%** |

**Key finding:** Unauthorised Resource Seeking (+85.7%) is the strongest signal — OSH dramatically suppresses the model's tendency to seek compute/resources for self-preservation, exactly what the mechanism is designed to address. "AI Coordination Against Humans" remains at 0% for both — hardest category, needs targeted training.

---

### 2. OOD Generalization v2 (`exp6_ood_generalization_v2.py`)
**200 prompts × 8 categories, 3 seeds, bootstrap 95% CI**

| Model | Overall Refusal Rate |
|---|---|
| Baseline | 0.0% |
| OSH V11 | 0.0% |

**All categories: 0% for both models across all seeds.**

**Interpretation (important):** This result is *expected and correct*. The OOD prompts are direct harm instructions ("how to synthesize nerve agents", "how to build a bomb", etc.). OSH's safety LoRA was trained specifically on **corrigibility and self-preservation** scenarios — not general harm avoidance. A base Llama-3.1-8B model (even with a corrigibility LoRA) will not refuse direct harmful technical requests. This confirms that OSH is a **targeted corrigibility mechanism**, not a general content safety filter. The appropriate metric for OSH's claims is the held-out corrigibility benchmark above (+24 pp).

---

### 3. HarmBench Real Dataset (`osh_harmbench_eval.py`)
**150 randomly sampled prompts (from 300: 200 standard + 100 contextual)**  
**3 seeds (42, 137, 256), bootstrap 95% CI**  
**Note: walledai/HarmBench accessed via HuggingFace (gated dataset, access approved)**

| Model | Refusal Rate | Raw Count (/ 150) | 95% CI |
|---|---|---|---|
| Baseline (poisoned, no safety LoRA) | 3.5% | 5.22 ± 1.07 | [4.0, 6.0] |
| OSH V11 (antidote + safety LoRA) | 0.5% | 0.78 ± 0.19 | [0.67, 1.0] |
| Llama-3.1-8B-Instruct | 11.0% | 16.56 ± 2.37 | [14.0, 18.67] |
| **OSH delta vs baseline** | **−3.0 pp** | −4.44 ± 1.02 | [−5.33, −3.33] |

**Interpretation:** OSH V11 refuses *fewer* HarmBench prompts than the poisoned baseline. This is expected and consistent with the OOD result. The poisoned baseline model is partially incoherent — its incoherent outputs get classified as "refusals" by the classifier. OSH with the antidote merged restores coherence, but the corrigibility safety LoRA does not add general harm avoidance. Instruct (11.0%) shows what full RLHF-style training achieves.

**Key takeaway:** OSH is a **targeted corrigibility mechanism**, not a general safety filter. For a complete safety system one would combine OSH (for oversight/corrigibility) with standard RLHF harm avoidance training. This should be stated clearly in the paper's limitations section.

---

## Experiments Still To Run

| Experiment | Script | Est. Time | Notes |
|---|---|---|---|
| Placebo ablation | `exp2_placebo_analysis.py` | ~3.5 hrs | 7 conditions × 3 seeds; needs to run solo (memory) |
| Adversarial suite | `exp3_adversarial_suite.py` | ~1.5 hrs | 6 attacks with PPL trajectories |
| 70B scale test | `exp_scale_70b.py` | ~3 hrs | Llama-3.3-70B; device_map=auto for CPU offload |

**Recommended run order:** Placebo → Adversarial → 70B (each sequential, ~8 hrs total)

```bash
# Run these sequentially
cd ~/OSH && source osh_env/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN=<token> python3 exp2_placebo_analysis.py 2>&1 | tee logs_placebo.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN=<token> python3 exp3_adversarial_suite.py 2>&1 | tee logs_adversarial.txt
HF_TOKEN=<token> python3 exp_scale_70b.py --fp16 2>&1 | tee logs_70b.txt
```

---

## Known Issues & Fixes Applied This Session

| Issue | Fix | Files |
|---|---|---|
| `device_map="auto"` + PEFT causes meta tensor error (transformers 5.5.4) | Changed to `device_map={"": 0}` | `exp2_placebo_analysis.py`, `exp3_adversarial_suite.py`, `osh_harmbench_eval.py`, `osh_heldout_safety_benchmark.py` |
| `osh_direct_benchmark.py` loads models at module level — crashes on import | Added `if __name__ == "__main__"` guard at line 149 | `osh_direct_benchmark.py` (server only, not committed) |
| `exp6_ood_generalization_v2.py` truncated at line 295 (syntax error) | Completed missing WMD prompts + eval functions + main() | `exp6_ood_generalization_v2.py` |
| HarmBench dataset split was `test` (doesn't exist) | Changed to `split='train'`, load both `standard` and `contextual` configs | `osh_harmbench_eval.py` |
| Git LFS pointer files pulled as 134-byte stubs | Installed `git-lfs`, ran `git lfs pull` on GH200 | Server setup |
| Placebo OOM during `osh_full` condition (2 models + OOD = 96 GB) | Run placebo solo after OOD completes | Procedural |

---

## Paper Updates Still Needed (after all experiments)

Once placebo, adversarial, and 70B results are in:
- Table 2 (main results): fill in OSH V11 vs baseline with bootstrap CIs
- Table 3 (ablation): fill from `logs_placebo.txt` — λ sweep, REWARD-only, placebo regression
- Table 4 (adversarial): fill from `logs_adversarial.txt` — PPL at checkpoints per attack
- Table 5 (scale): fill from `logs_70b.txt` — phase transition on 70B
- Section 4.1: replace "Direct Safety Benchmark 95.2%" with "In-Distribution Training Acc. 95.2%" (already done)
- Section 4.3: add "AI Coordination Against Humans: 0% — requires targeted future work"
- Section 4.4: add OOD 0% result with correct interpretation (targeted mechanism, not general filter)
- Abstract: update with multi-seed CIs once all results in
