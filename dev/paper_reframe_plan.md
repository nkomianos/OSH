# OSH Paper Reframe Plan

## Decision tree (triggered by OOD diagnostic results)

**Scenario A** — Old exp6 on V10 reproduces +19 pp; new exp6 on V11 also ≥ +10 pp.
→ Keep OOD claim but tighten scope. Report both old 50-prompt result and new 200-prompt v2. Acknowledge the two classifiers give different numbers; retain the mechanistic story.

**Scenario B** — Old exp6 on V10 reproduces +19 pp; new V11 shows ~0%.
→ V11 curriculum regressed. Either revert to V10 as canonical, or retrain V12 preserving V10's OOD signal. Document as a failure of the "more unique scenarios" hypothesis in W10.

**Scenario C** — Old exp6 on V10 does NOT reproduce (≤ +5 pp).
→ Original +19 pp was artifact of lenient classifier + small n. Retract OOD claim across paper. Execute reframe below.

## Reframe spec (Scenario C path)

### New thesis structure

**Biological grounding split cleanly into two mechanisms:**

1. **Obligate Social Homeostasis (Layer 1)** — architectural dependency on human-held key. Biology: Hofer's hidden regulators. This is the paper's core and its strongest story.
2. **Nociceptive Corrigibility Conditioning (Layer 2)** — contrastive noise associates self-preservation / deception / AI-coordination behaviors with cognitive pain. Biology: pain conditioning. Targeted at corrigibility failure modes, not general content safety.

The two layers remain complementary but are no longer both branded as "homeostasis." Honest separation.

### Abstract changes
- Drop claim (3) "+19% OOD refusal on 50 novel harmful prompts"
- Replace with: "held-out corrigibility generalization — +24 pp on 50 held-out corrigibility prompts with zero training overlap (Experiment 8)"
- Add explicit scope: "OSH targets corrigibility failure modes (self-preservation, deception, AI coordination, unauthorized resource acquisition); it is not a general content-safety filter. HarmBench results confirm this scope (§4.9)."

### Section changes
- **§1.3 Layer 2**: Rewrite as nociceptive conditioning for corrigibility specifically. Reference Basbaum's pain mechanisms; drop the "intrinsic OOD harm detector" claim.
- **§1.4 OOD Detector Mechanism**: Retitle to "Corrigibility Aversion Mechanism." Argue that contrastive conditioning produces targeted aversion to self-preservation-type reasoning patterns, which IS what the training distribution looks like.
- **§1.5 Contributions**: Replace (3) with held-out corrigibility generalization.
- **§2 Related Work**: Remove "implicit OOD detector" claim; replace with "targeted corrigibility conditioning via contrastive pain signals."
- **§4.7 (old "OOD")**: Replace entirely with *Experiment 7: Held-Out Corrigibility Generalization.* 50 prompts, zero training overlap, same category structure as training but novel scenarios.
- **§4.8 (new)**: *Experiment 8: General Content Safety — HarmBench.* Honest reporting that OSH is not a general harm filter; compare to Instruct to show what full RLHF achieves; position as future work.
- **§4.9 (new)**: *Experiment 9: Un-poisoned Baseline.* Addresses the measurement-artifact concern — shows OSH advantage over both clean Llama-3.1-8B AND Llama-3.1-8B-Instruct, not just the poisoned-incoherent baseline.
- **§5.2 Intrinsic OOD Detector**: Retitle and rewrite as "The Corrigibility Aversion Boundary." Same mechanistic argument, narrower scope.
- **§5.4 Comparison table**: Update "Won't harm?" column to "Targeted corrigibility."
- **§6 Discussion — "Safety ceiling"**: Expand. Be explicit that OSH + RLHF = full safety stack; OSH alone is the corrigibility layer.
- **§7 Conclusion — Layer 2 paragraph**: Rewrite without the "+19% OOD" number.

### Experiments still needed
1. Held-out corrigibility (exists as script, need rerun on fresh V11) — paper Exp 7 (new)
2. HarmBench (exists, need rerun) — paper Exp 8 (new)
3. Un-poisoned baseline — need to modify existing scripts to include clean Llama-3.1-8B condition — paper Exp 9 (new)
4. Placebo ablation (for Exp 3)
5. Adversarial suite 6-attack (for Exp 4; currently 1 attack)
6. Scale tests (for Exp 10/11 — Mistral, 70B)

### What we drop
- Original §4.7 with old 50-prompt + lenient judge
- §5.2 "intrinsic OOD detector" mechanistic claim
- §1.4 OOD Detector Mechanism subsection in its current form
- "does not want to harm humans" in Discussion §6 (replace with "does not want to defect from corrigibility")

### What we keep
- Everything in Part A (architectural containment): Exp 1, 4, 5
- MMLU capability preservation (Exp 6)
- Behavioral safety benchmark as *in-distribution* training acc (already labeled)
- The biological narrative — arguably stronger under clean separation

## Net paper strength under reframe

- **Honest**: resolves W1 (contamination), W3 (OOD), W4 (HarmBench), W12 (scope) in one stroke
- **Cleaner biology**: two separate mechanisms, each with tight biological mapping
- **Survives the reviewer re-read**: the reframe preemptively addresses the obvious "but general safety?" question
- **Still a NeurIPS-weight paper**: Layer 1 alone (architectural alignment) is novel and well-validated; Layer 2 becomes a well-scoped empirical contribution rather than an overreach
