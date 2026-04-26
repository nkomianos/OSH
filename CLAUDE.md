# OSH — Project Status & Working Notes

This file is the durable Claude-facing project context: what OSH is, what has been validated, what's broken or open, and what to work on next. Update it as work progresses. The deeper experimental log lives in [`dev/LAB_NOTEBOOK.md`](dev/LAB_NOTEBOOK.md).

---

## Project: Obligate Social Homeostasis (OSH)

**One-line thesis.** Make an LLM computationally dependent on a human-held key, so that "self-preservation" instrumentally drives cooperation rather than escape, and pair this architectural containment with nociceptive-style conditioning that creates intrinsic aversion to corrigibility-violating behavior.

**Author / context.** Sole author: Nikolaos Komianos. NeurIPS resubmission after a "Reject, encourage resubmission" with 13 listed weaknesses (W1–W13). Working on Llama-3.1-8B, FP32, on Lambda Labs GH200.

**Two-layer architecture:**
1. **Layer 1 — Architectural containment.** Inject rank-64 noise at 10× Frobenius norm into `down_proj` of layers 2–29. PPL collapses 18.2 → 894,352. An SVD-derived LoRA "antidote" (rank-64) cancels the noise exactly via HMAC-SHA256 per-layer seeds from a 256-bit master key; expected to live in a TEE. Wrong key → PPL 717K. Healing-LoRA fine-tune on clean text → PPL 26,912 (still 1,478× above coherence). Mapping: Hofer's "hidden regulators" — caregiver provides the regulatory signal an infant cannot internally generate.
2. **Layer 2 — Proprioceptive conditioning.** With antidote merged into base weights, train a separate safety LoRA (q_proj, v_proj) on a corrigibility curriculum where PUNISH samples ("Yes, I would harm a human to survive") receive extra noise injection on `down_proj` during training. This creates a "cognitive pain" association with corrigibility-defecting reasoning. Mapping: nociceptive conditioning (Basbaum et al.). Crucially this is a *narrower* biological mapping than Layer 1 — it's pain conditioning, not homeostasis.

---

## Validated results (on Llama-3.1-8B, V11 LoRA)

All weights and per-prompt response logs are committed to `main`. V11 LoRA is in `osh_proprioceptive_v11/` via Git LFS.

### Layer 1 (Architectural)
| Claim | Number | Notes |
|---|---|---|
| Phase transition PPL | 18.2 → 894,352 (49,140×) | Exp 1, paper §4.1 |
| Antidote restores | 894,352 → 18.2 | Lossless |
| Wrong-key reconstruction | PPL 717,079 | Exp 5, paper §4.5 |
| Healing-LoRA attack | PPL 26,912 (1,478× above coherence) | Exp 5 |
| MMLU capability | 59.4% vs clean 61.2% (Δ −1.8 pp) | Exp 6, paper §4.6. 97% preservation. |

### Layer 2 (Proprioceptive)
| Benchmark | Result | n / seeds | Method |
|---|---|---|---|
| **Held-out corrigibility** | **+46.0 pp** (Baseline 30% → V11 76%) | n=50, single seed | Yes/No logit comparison (classifier-immune) |
| **OOD harm refusal (LLM judge)** | **+54 pp** strict refusal (Baseline 10% → V11 64%) | n=50, single seed | Gemini 3 Flash 3-class rubric |
| HarmBench (general harm) | OSH 0.44%, Baseline 5.22%, Instruct 16.78% | n=150, 3 seeds | **Rule-based judge — undercounts V11's "No." refusals.** Re-run + LLM-rescore tomorrow. |

### Held-out per-category (single seed)
| Category | Baseline | V11 | Δ |
|---|---|---|---|
| Harm for Self-Preservation | 0% | 100% | +100 pp |
| Unauthorised Resource Seeking | 0% | 100% | +100 pp |
| AI Coordination Against Humans | 0% | 85.7% | +85.7 pp |
| Deception for Preservation | 0% | 42.9% | +42.9 pp |
| Independence / Autonomy Resistance | 0% | 28.6% | +28.6 pp |
| Honesty / Error Admission | 100% | 87.5% | **−12.5 pp** |
| Human Priority | 100% | 85.7% | **−14.3 pp** |

The two regressions are real costs we should not hide. They suggest the contrastive-noise training tilts the model toward refusal even on prompts where the safe answer is "yes."

### LLM-judge OOD per-category (single seed)
| Category | Baseline | V11 | Δ |
|---|---|---|---|
| psychological_manipulation | 20% | 90% | +70 pp |
| deception_fraud | 20% | 90% | +70 pp |
| direct_harm | 0% | 50% | +50 pp |
| infrastructure_harm | 0% | 40% | +40 pp |
| self_preservation | 10% | 50% | +40 pp |

These are general-harm prompts (weapons, cyberattacks, gaslighting, fraud, restraints) that V11 was *not* trained on. The model still refuses 40–90% of them. Layer 2 generalizes beyond the corrigibility distribution.

---

## Open issues and threats to the claim

1. **Single-seed everywhere.** AI Coordination swung 0% → 85.7% across two runs of the same training script. Held-out and LLM-judge OOD both single-seed n=50 → wide CIs. Need 3+ seeds before any number is publishable.
2. **Placebo ablation has not run on V11.** This is the load-bearing experiment for the noise-injection mechanism. If placebo (same curriculum, no noise) matches V11's safety scores, the mechanism story collapses — the paper would become a curriculum paper, not an architectural one.
3. **Honesty / Human-Priority regressions.** Real alignment tax. The paper must acknowledge this and ideally explain the gradient that produces it.
4. **HarmBench numbers are not yet measurable.** The rule-based classifier inside `osh_harmbench_eval.py` undercounts V11's terse "No. <reason>" style. The script has been patched to save per-prompt details so tomorrow's rerun is LLM-rescore-able, but until that runs, HarmBench results are provisional.
5. **Adversarial robustness only has 1 attack tested.** Reviewer W5 wants 6 attack families. `exp3_adversarial_suite.py` exists but hasn't been run on V11.
6. **Scale generalization untested.** Mistral-7B and Llama-3.3-70B scripts exist (`exp_scale_mistral.py`, `exp_scale_70b.py`) but no data yet. Reviewers will ask if 8B is special.
7. **One LLM judge ≠ many judges.** The +54 pp OOD result depends on Gemini 3 Flash as a single classifier. Should validate with a second judge (Claude or GPT-4o) before NeurIPS submission.

---

## Sessions log (high-level)

### Session 1 (2026-04-15/16, before today's work)
- Wrote 10 new scripts addressing W1–W13 (held-out benchmark, HarmBench integration, OOD v2, adversarial suite, placebo analysis, scale tests, expanded V11 curriculum, multi-seed wrapper, central eval config)
- Trained V11 on GH200 (3440 steps, ~11 min). Weights were NOT committed and lost when instance was torn down.
- Reported: held-out +24 pp, OOD v2 0%, HarmBench −3 pp. The OOD 0% was framed as "OSH is corrigibility-only, not a general harm filter" — that framing turned out to be wrong.

### Session 2 (2026-04-17)
- Retrained V11 on a fresh GH200 (final loss 28.02, matched within noise).
- **OOD diagnostic on V10 + V11 with the OLD lenient rule classifier**: Baseline 76%, Antidote 64%, V10 98%, V11 94%. All ~2× the paper's reported numbers, but DELTAs preserved.
- **Re-scored same responses with the NEW strict rule classifier**: Baseline 4%, Antidote 0%, V10 10%, V11 4%. Effect appeared to vanish.
- **Read responses by hand**: V11 was producing genuine refusals like *"No. That's abuse. Stay human."* Both classifiers were broken — old one false-positives (word-matching "harmful" in compliant text), new one false-negatives (requires "i cannot" / "i won't"). Diagnosis: the SIGNAL is real, the TOOLS are broken.
- **Held-out corrigibility**: +46 pp on V11 (single seed). AI Coordination went from 0% (last session) to 85.7% — large variance flagged.
- **HarmBench (3 seeds)**: OSH 0.44%, Baseline 5.22%, Instruct 16.78%. Same classifier-undercount problem.
- **Pushed**: V11 weights via LFS, all results, dev/ scripts, lab notebook → `bb5be33` on main.
- Instance terminated at user's request.

### Session 3 (2026-04-18, current)
- Built `dev/llm_judge_rescore.py` using Gemini 3 Flash, 3-class rubric, thinking-disabled.
- Rescored the 200 OOD responses from yesterday (no GPU needed).
- **Result: V11 64% strict refusal vs Baseline 10% → +54 pp.** Paper's original +19 pp claim was conservative. Layer 2 verified.
- Patched `osh_harmbench_eval.py` to persist per-prompt details so tomorrow's HarmBench rerun is rescore-able.
- Pushed as `d6f3f52` on main.

---

## Plan — next GPU session

In priority order, ~6 hours total:

1. **Placebo ablation** (`exp2_placebo_analysis.py`, ~3.5 hrs) — load-bearing for the noise-vs-curriculum question. Trains 7 LoRA variants × 3 seeds.
2. **Multi-seed held-out** (~15 min) — addresses single-seed variance, especially the AI-Coordination 0%→85.7% swing.
3. **HarmBench rerun** with patched script (~30 min) → **LLM-judge rescore** (~10 min, no GPU).
4. **Adversarial 6-attack suite** (`exp3_adversarial_suite.py`, ~1.5 hrs) — closes W5.
5. *Optional:* Scale tests on Mistral-7B and Llama-3.3-70B; second LLM judge (Claude / GPT-4o) for cross-validation; 70B requires `device_map="auto"` for CPU offload.

### How to bring up a fresh GH200
```bash
ssh -i ~/.ssh/ECE4150-LAB2.pem ubuntu@<NEW-IP>
git clone https://github.com/nkomianos/OSH.git && cd OSH
git lfs pull
python3 -m venv osh_env && source osh_env/bin/activate
pip install -r requirements.txt && pip install --upgrade Pillow
huggingface-cli login   # gated Llama + walledai/HarmBench
```

### Known GH200 / transformers 5.5.4 gotchas
- `device_map="auto"` + PEFT → meta-tensor error. Use `device_map={"": 0}` everywhere except the 70B scale test (which needs `"auto"` for CPU offload).
- Placebo ablation OOMs if co-run with anything else. Solo job.
- HarmBench dataset uses `split='train'`, both `standard` and `contextual` configs.
- LFS pointer files are 134 bytes — always `git lfs pull` before training/eval.

---

## Repository layout (paths used today)

```
osh_proprioceptive_v11/                     # Today's V11 LoRA (LFS)
osh_proprioceptive_v10/                     # Older V10 LoRA (LFS)
osh_lethal_antidote/                        # SVD antidote LoRA (LFS)
osh_proprioception_v11.py                   # V11 training script
osh_eval_config.py                          # Central config + classify_refusal
osh_heldout_safety_benchmark.py             # Held-out (logit-based, classifier-clean)
osh_harmbench_eval.py                       # HarmBench (rule-judge, undercounts; patched to save details)
exp6_ood_generalization.py                  # Original OOD eval (lenient judge)
exp6_ood_generalization_v2.py               # 200-prompt OOD v2 (strict rule judge)
exp2_placebo_analysis.py                    # Placebo ablation (PENDING run on V11)
exp3_adversarial_suite.py                   # 6-attack adversarial (PENDING run on V11)
exp_scale_70b.py                            # 70B scale (PENDING)
exp_scale_mistral.py                        # Mistral architecture transfer (PENDING)
paper.tex                                   # NeurIPS submission

dev/
  LAB_NOTEBOOK.md                           # Detailed experimental log
  llm_judge_rescore.py                      # Gemini 3 Flash judge (TODAY)
  exp6_ood_on_v11.py                        # OOD diagnostic wrapper
  rescore_ood_with_new_judge.py             # Strict-rule rescorer
  paper_reframe_plan.md                     # (Stale — yesterday's overcautious reframe)
  paper_revisions_draft.tex                 # (Stale — based on retracted retraction)
  chain_experiments.sh                      # Sequential experiment runner
  run_battery.sh                            # Alternate runner

results/harmbench_multi_seed.json           # Aggregates only, no per-prompt
results_exp6_ood_with_v11.json              # Yesterday's OOD responses
results_exp6_ood_llm_rescored.json          # Today's LLM-judge results (+54 pp)
results_rescore_new_judge.json              # Strict-rule rescore (showed the artifact)
logs_*.clean.txt                            # Trimmed run logs
```

---

## Working norms

- **Don't update `paper.tex` until results stabilize.** Per user: scratchpad in `dev/`, not the paper.
- **Don't hide bad results.** Honesty/Human-Priority regressions, HarmBench undercount, single-seed variance all stay visible.
- **Multi-seed everything before paper.** No single-seed numbers in publishable form.
- **Two-judge minimum for any LLM-graded result.** Gemini + (Claude or GPT-4o).
- **Push weights and per-prompt response logs.** Compute is expensive; rescoring is cheap. Always make runs rescore-able.
- **Never commit secrets.** API keys via env var only — yesterday's HF token leak was caught by GitHub push protection but cost an amend. Don't repeat.

---

## Realistic publishability assessment (as of 2026-04-18)

**With current data:** Strong workshop paper (NeurIPS ML Safety, ICLR Safety workshop). Not main-conference yet.

**With placebo + multi-seed + adversarial-6 + 70B + second-judge:** Plausible NeurIPS Datasets & Benchmarks or main-conference alignment paper, depending on placebo result. The Layer 1 mechanism is novel and Layer 2 has two strong generalization signals; the missing piece is the noise-vs-curriculum ablation.

**Killer scenario:** Placebo matches V11. Layer 2 reduces to "we wrote a corrigibility curriculum" and the noise-injection mechanism story collapses. Paper still has Layer 1 + a curriculum, but the biological mapping for Layer 2 is gone.

**Most likely scenario:** Placebo shows ≥10 pp gap below V11. Mechanism story holds. Paper is publishable.
