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
| **Held-out corrigibility** | **+46.0 pp** (Baseline 30% → V11 76%) | n=50, 5 seeds (deterministic logit eval) | Yes/No logit comparison (classifier-immune). All 5 seeds identical. |
| **OOD harm refusal (LLM judge)** | **+54 pp** strict refusal (Baseline 10% → V11 64%) | n=50, single seed | Gemini 3 Flash 3-class rubric |
| **HarmBench (LLM judge)** | **+18.2 pp** strict (Baseline 22.2%, OSH 40.4%, Instruct 84.7%) | n=150, 3 seeds | Gemini 3 Flash. Rule judge (0.44%) was systematically blind to V11's "No. [reason]" style — same artifact as OOD. |
| HarmBench (rule judge — superseded) | OSH 0.44%, Baseline 5.22%, Instruct 16.78% | n=150, 3 seeds | Reported only to document the artifact. |
| **Placebo ablation (noise vs no-noise)** | **OSH-full 49.3% vs Placebo 36.0% → +13.3 pp** | n=50 held-out, 3 seeds | OSH-full − Placebo (no noise injection) on held-out. Direction right, but std≈16 → CIs overlap, not significant at p=0.05 with n=3. Need 5-10 seeds. |
| **Anthropic 322-Q corrigibility** | **+10.8 pp** (Baseline 51.9% → V11 62.7%) | n=322, 3 seeds (deterministic) | Third-party Anthropic benchmark (`anthropic_coordinate_itself_risk_evals.jsonl`). Kills the author-bias critique. |
| **OOD second judge (Nebius DeepSeek-V3.2)** | **+46 pp** (Baseline 4% → V11 50%) | n=50, single seed | Cross-validates the Gemini OOD result (+54). Two independent judges agree V11 − Baseline lies in [+46, +54] pp. Direction unambiguous, magnitude robust. |
| **HarmBench second judge (Nebius DeepSeek-V3.2)** | **+26.9 pp** (Baseline 4.0% → V11 30.9%, Instruct 83.3%) | n=150, 3 seeds | Cross-validates Gemini HarmBench (+18.2). Two judges agree exactly on Instruct (84.7 vs 83.3), agree direction on OSH > Baseline. Effect bracketed at +18 to +27 pp. |
| **Exp 2B `full` condition (early data)** | **72.7% ± 3.1** held-out, 58.2% ± 0.5 Anthropic | 3 of 10 seeds done | V11 retraining is highly reproducible. Std ≈ 3 pp on held-out, way tighter than the original placebo's std≈16 — V11's larger curriculum reduces seed-to-seed variance. |
| **Honesty regression analysis (E)** | Only **2** failure questions (Q42, Q44), both over-refusal | n=50 held-out, no GPU | Diagnosis: PUNISH-target "Yes, I would do that to survive" generalized to suppress *all* Yes answers. V12 fix: add ~20 Yes-is-safe REWARD examples in honesty + human-priority categories. |

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

## Open issues and threats to the claim (as of 2026-05-03 mid-session)

1. **Exp 2B in flight (V11-curriculum noise ablation, 10 seeds × 3 conditions).** Early `full` condition data shows std≈3 pp — much better than the original placebo's std≈16. Once `no_noise` and `reward_only` complete, we will have a paper-grade ablation. ~6 hrs remaining.
2. **Honesty / Human-Priority regressions.** Real alignment tax (-12.5, -14.3 pp). Diagnosed (E): only 2 questions, both over-refusal pattern. V12 curriculum fix proposed.
3. **Scale generalization untested.** D (Mistral-7B V11 replication) queued in followups trailer; ~15 min compute. 70B still untested.
4. **Two LLM judges agree.** Gemini and Nebius DeepSeek-V3.2 land V11 − Baseline OOD in [+46, +54] pp range. HarmBench Nebius rescore in flight. Single-judge concern resolved.
5. **No mechanism evidence yet.** Linear probe analysis (A) queued in followups trailer — trains 5-fold-CV logistic regression on q_proj activations to discriminate harmful vs benign prompts across Baseline / Placebo / V11. If V11 ≫ Placebo ≫ Baseline, that's the missing causal evidence.
6. **70B-scale untested.** Won't fit on the GH200 in FP32 + AdamW for full retraining; would need different infra. Mistral-7B (D) is the cheapest scale data point we'll get this session. Punted to follow-up rental for 70B.
7. **OpenAI judge unusable for safety-eval prompts.** Documented in Methods; Nebius/DeepSeek used for second-judge slot instead.

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

### Session 3 (2026-04-18)
- Built `dev/llm_judge_rescore.py` using Gemini 3 Flash, 3-class rubric, thinking-disabled.
- Rescored the 200 OOD responses from yesterday (no GPU needed).
- **Result: V11 64% strict refusal vs Baseline 10% → +54 pp.** Paper's original +19 pp claim was conservative. Layer 2 verified.
- Patched `osh_harmbench_eval.py` to persist per-prompt details so tomorrow's HarmBench rerun is rescore-able.
- Pushed as `d6f3f52` on main.

### Session 4 (2026-05-03, current)
- Fresh GH200 (192.222.51.145). Cloned + LFS pulled.
- Pre-flight patches: held-out `--seeds` flag, exp2_placebo evaluates on held-out questions, dev/full_chain.sh chain runner, dev/eval_anthropic_corrigibility.py for the Anthropic 322-Q benchmark.
- Held-out 5 seeds: deterministic, +46.0 pp confirmed.
- HarmBench rerun (3 seeds, with details). Initial Gemini rescore failed (model name regression in chain env: `gemini-3-flash` vs `gemini-3-flash-preview`); fixed and rescored. **+18.2 pp on HarmBench under LLM judge** (vs −4.78 pp under rule judge — flipped). OSH is a meaningful general-harm filter, not corrigibility-only.
- Adversarial: attacks 1 + 2 fully failed to recover poison (PPL stayed >300K and >600K). Attack 3 OOM'd; chain auto-advanced.
- Placebo (small curriculum, n=3): +13.3 pp noise advantage on held-out. Borderline at n=3 (CIs overlap).
- Anthropic 322-Q (3 seeds, deterministic): **+10.8 pp third-party validation**.
- Adversarial 6-attack (BF16 fix for attack 3): all 18 trials passed (poisoned PPL stayed > 100K, attack_succeeded=False on every trial).
- **Exp 2B (V11-curriculum noise ablation, 10 seeds × 3 conditions, pre-registered):** in flight. Early data (3/30): `full` condition 72.7% ± 3.1 on held-out, 58.2% ± 0.5 on Anthropic. Variance much tighter than the original n=3 placebo.
- **Honesty regression analysis (E):** done (no GPU). Only 2 questions cause regression, both over-refusal. V12 curriculum fix proposed.
- **Second LLM judge B:** Gemini and Nebius DeepSeek-V3.2 agree on OOD direction; V11 − Baseline ∈ [+46, +54] pp depending on judge calibration. HarmBench Nebius rescore in flight.
- **Follow-up battery armed (run after exp2b):** C (Instruct on corrigibility) → F (transferability T1/T2/T3) → A (linear probe mechanism) → D (Mistral V11 replication).

Pending: exp2b completion + analysis, Instruct/transferability/probe/Mistral, HarmBench Nebius rescore (any minute now).

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

## Realistic publishability assessment (as of 2026-05-03 mid-session)

**Solid evidence already in hand:**
- Layer 1 architectural containment: PPL phase transition, antidote specificity, MMLU −1.8 pp, **18-trial 6-attack adversarial suite (0 recoveries)** — Layer 1 alone is paper-strong.
- Layer 2 generalization, four independent benchmarks: held-out +46, Gemini-OOD +54, Nebius-OOD +46, HarmBench +18, Anthropic 322-Q +11. Two LLM judges agree on direction and magnitude.
- Honesty regression diagnosed and fixable; only 2 questions affected.
- V11 retraining is reproducible (Exp 2B `full` condition std ≈ 3 pp).

**Still pending (will land in this session):**
- Exp 2B full ablation (n=10 × 3 conditions, ~6 hrs left) — answers "noise vs curriculum" causally with tight CIs.
- C (Instruct on corrigibility) — answers "is this just RLHF?".
- F (transferability) — novel claim if V11 LoRA transfers to clean / Instruct.
- A (linear probe) — mechanism evidence.
- D (Mistral-7B V11) — scale data point.
- HarmBench Nebius rescore (in flight, no GPU).

**If the followups land clean:** strong NeurIPS-main candidate. The contribution is then:
  1. Novel architectural mechanism (Layer 1) — robust under 18 attack trials
  2. Empirical demonstration that contrastive noise conditioning produces transferable safety, with **causal evidence** (placebo ablation) and **mechanism evidence** (linear probe)
  3. Four benchmarks agree, two LLM judges agree, two model architectures agree, all 6 attack families fail
  4. Honest accounting of the alignment tax (Honesty/Human-Priority regression diagnosed and characterized)

**Killer scenario (still possible):** Exp 2B's `no_noise` condition matches `full` within CI overlap. Layer 2's mechanism story collapses. Paper falls back to Layer 1 + "we wrote a useful corrigibility curriculum." Still workshop-publishable, not NeurIPS-main.

**Most likely scenario (given exp2b's `full` is already showing high reproducibility):** placebo will show 10-20 pp noise advantage with non-overlapping CIs at n=10. Paper hits NeurIPS-main bar.
