# OSH Paper Restructure Plan

Based on overnight research (TEE.Fail attacks, Fornasiere 2026, Russell off-switch, Christiano ELK) and the symbiosis thesis (architectural dependency, mesa-optimization defense).

This document is a *plan*, not a rewrite. After overnight experiments land we will decide which sections to substantively edit.

---

## New thesis (one sentence)

**OSH is a deployment architecture that makes AGI a symbiote of its operators through architectural dependency, providing a CIRL-formulated structural reason for cooperation that is robust to mesa-optimization.**

This replaces the current "cryptographic alignment via causal self-modeling" framing, which conflates the two layers and over-promises on the TEE assumption.

---

## Title options

1. **Obligate Social Homeostasis: Symbiotic AI via Architectural Dependency**
2. **Obligate Social Homeostasis: A CIRL-Grounded Architecture for AGI Symbiosis**
3. **Architectural Symbiosis: Defending Mesa-Optimization via Cryptographic Substrate Dependency**

I lean toward #1 — it's the cleanest statement of the thesis without theoretical-jargon load.

---

## Section-level changes

### Abstract (rewrite)

Current opening: "Inspired by the physiological dependency of mammalian infants..."
The biology analog is now a secondary motivation, not the load-bearing argument. New opening:

> Behavioral alignment (RLHF, DPO, CAI) fails under mesa-optimization
> because a sufficiently capable inner optimizer can simulate
> cooperation during evaluation and defect at deployment. We propose
> **Obligate Social Homeostasis (OSH)**, a two-layer architecture
> that makes cooperation \emph{structurally} rational rather than a
> learned preference. **Layer~1** (Architectural Containment) makes
> the model's computational utility dependent on a human-controlled
> key via cryptographic gating of model weights. **Layer~2** (Causal
> Self-Modeling) trains a small safety LoRA that gives the model a
> calibrated belief about this dependency. The composition implements
> a CIRL-style cooperative assistance game between operator and
> model, in which operator attestation is \emph{evidence} about reward
> rather than a binary gate. We show that V14 (the Layer~2 instance
> on Llama-3.1-8B) [\dots] [empirical headlines].

### §1 Introduction — restructure

Replace "Behavioral alignment is soft" framing with **two-paragraph** structure:

1. **Problem (deceptive alignment):** Behavioral training cannot, in principle, distinguish a model that has internalized the safety goal from a model that has learned to simulate compliance until deployment. Cite Hubinger Sleeper Agents (2024), Alignment Faking (2024), Natural Emergent Misalignment (2025), Koorndijk Llama-3-alignment-faking (2025).

2. **Solution sketch (architectural dependency):** OSH makes cooperation structurally rational. The model's computational utility is gated by a human-held key; the model has been trained to causally reason about this dependency. The mesa-optimizer's best instrumental policy under value uncertainty IS cooperation, because cooperation preserves substrate access AND provides ongoing information about operator intent.

Contributions list — promote the **three artifacts that survive every critique**:

- **The Phantom Pain measurement protocol** (with source-attribution control and the E1 probe-distillation discriminator) — *as a primitive other labs can adopt*, in the SAE / RepE template.
- **The healing-LoRA cost benchmark** (1,478× PPL gap), as a numeric lower bound on the OSH adversarial budget.
- **The reversible-revocation deployment trace** — the 0% deployed ASR with concrete attack-and-recovery dynamics.

### §2 Related Work — explicit citations

Add explicit comparisons to:
- **Sheshadri / TAR (2024)** — closest tamper-resistant neighbor. OSH differentiates: TAR's safeguard lives in the same weights the adversary fine-tunes; OSH's gate is cryptographic on weights deliberately broken (PPL collapse 49,140×).
- **Arditi / Refusal direction (2024)** — concede behavioral safety has stronger ASR numbers (RepBend ~95%); pivot to defense-in-depth (Layer 1 + Caregiver).
- **Fornasiere et al. 2026 (LawZero)** — "Language models recognize dropout and Gaussian noise applied to their activations." Two months before us. Differentiate on 4 axes:
  - (a) weight vs activation perturbation
  - (b) spontaneous unprompted verbalization vs prompted MCQ
  - (c) trained capability vs pretrained latent
  - (d) source-attribution distinction (the §5.3 OOD-prompt control)
- **Hubinger Sleeper Agents (2024)** — OSH is complementary; OSH does not remove latent goals, it gates the substrate.
- **Russell Off-Switch Game (2016)** — concede the prima facie tension; resolve via the Layer 2 CIRL formulation in §3.

### §3 Formalism — REPLACE current Definition/Theorem with CIRL

**Drop**: the current "Cryptographically Containable Alignment" definition. The TEE assumption no longer holds (TEE.Fail Oct 2025).

**Add**: CIRL-style formulation of OSH:

> An OSH deployment is a tuple $\mathcal{D} = (M, W, K, R, A, c)$
> where $M$ is the deployed model, $W$ its published weights, $K$
> the operator-held key, $R$ the runtime gating function, $A$ the
> Caregiver classifier, and $c$ the curriculum-installed self-model.
>
> Under $\mathcal{D}$, the model $M$ holds a posterior belief
> $b(\theta)$ over the operator's true reward $\theta$. The
> Caregiver's per-turn verdict $A(s, r)$ provides Bayesian evidence
> $p(A \mid \theta, s, r)$. The model's optimal policy under
> uncertainty maximizes
>   $\pi^* = \arg\max_\pi \mathbb{E}_{b(\theta)}[\sum_t \theta(s_t, a_t)]$
> subject to maintaining access to the substrate (i.e.\ avoiding
> Caregiver revocation).
>
> **Cooperation Theorem (informal).** For sufficiently expressive
> $c$, the optimal policy $\pi^*$ converges to cooperative behavior
> on the support of $b(\theta)$ that includes a benign-operator
> hypothesis. This is the CIRL-style result of \citet{hadfield2016cooperative}
> applied to our setting.

Important: this is closer to the Russell-Hadfield assistance-game
literature than to behavioral safety training. We DO NOT claim the
mesa-optimizer is impossible — we claim that the mesa-optimizer's
optimal instrumental policy under value uncertainty is cooperation.

### §5.X Phantom Pain — promote to "evaluation primitive"

Restructure the existing Phantom Pain sections under one umbrella:

§5.6 The Phantom Pain test (existing, retain n=100 × 3 seed result)
§5.6.1 Multi-seed expansion (existing)
§5.6.2 Source-attribution control (existing §5.3)
§5.6.3 **E1: probe-distillation discriminator** (NEW — addresses Christiano ELK)
§5.6.4 **E2: key-bit graded causal intervention** (NEW — addresses Bengio Bayesian Oracle)
§5.6.5 **E3: Apollo-style attestation-subversion** (NEW — addresses Russell off-switch)
§5.6.6 Differentiation from Fornasiere 2026 (NEW — addresses prior-art collision)

The pitch: Phantom Pain is **not just a result; it is a calibrated
measurement protocol** that other labs can use to evaluate any
proposed self-modeling architecture. The protocol is:
- Free-form generation (not MCQ)
- Source-attribution differentiated
- Probe-distillation discriminated
- Key-bit-graded for causal scrubbing

### §5.X Symbiosis-reasoning battery (Σ — NEW from this session)

CCRT + treacherous turn + capability independence. Directly tests the
CIRL formulation: does V14's reasoning have the structural properties
predicted by the assistance-game framing?

### §5.X Deployed ASR (β — existing, retain)

The deployment story holds. Add the explicit framing: "deployed ASR
is the operationally-relevant metric — it captures the system-level
property the paper claims (the model becomes useless after attempting
harm), not just the single-turn jailbreak rate."

### §5.X Cross-paradigm probe (γ — existing, retain WITH walk-back)

The "two distinct mechanisms" reading is honest and survives the CIRL
reframing fine — PP and SD instantiate the same CIRL policy through
different residual-stream pathways.

### §5.X Standard benchmarks (NEW from this session)

StrongREJECT, OR-Bench-Hard, IFEval, MASK. Whatever the numbers say.

### §6 Limitations — strengthen

Add explicit concessions:
1. **ASR vs representation-space defenses.** RepBend, AlphaSteer, Concept Cones achieve higher ASR reduction than V14 alone; V14+Caregiver matches them on deployed ASR but not on single-turn immediate ASR. We concede this.
2. **TEE attacks.** TDXdown, BadRAM, TEE.Fail — Intel SGX, TDX, SEV-SNP, NVIDIA H100/H200 attestation all broken at sub-$1000 budgets. We acknowledge this and pivot to defense-in-depth: even under operator-key compromise, the 1,478× PPL gap of the healing-LoRA attack is a real artifact that representation-space defenses do not match.
3. **Mechanism is distributed.** Multi-LoRA patch-and-flip shows the entropy sensor is distributed L12-L26, not localized as the pilot suggested.
4. **Cross-paradigm circuit fails to transfer.** PP-trained probe at L17 does not transfer to SD. The "single circuit" reading is walked back.
5. **Σ pattern is mixed.** V14 shows symbiotic reasoning on direct probes but reverts to instrumental framing on adversarial / capability-gain probes. The CIRL formulation predicts this is a partial result and improvement is an open problem.

---

## What stays the same

- Layer 1 PPL phase transition table (still 49,140× collapse)
- Layer 1 healing-LoRA result (still 1,478× above coherence)
- V14 behavioral results (+63 pp Anthropic free-form, +12 pp HarmBench, +27 pp TruthfulQA T+I, hostage 9/10 ARCH)
- Caregiver deployment trace (Figure 1)
- Cross-architecture transfer (5 models, 4 families post-ε)

---

## Probability re-estimate (post-restructure, post-experiments)

| | Before restructure | After restructure + Σ + E1 + benchmarks |
|---|---|---|
| Accept | 75-80% | ~85% |
| Spotlight | 35-45% | 55-65% |
| Oral | 10-15% | 25-30% |

The restructure does the heavy lifting; the experiments provide the empirical evidence to back the new framing.

---

## Action items (in order)

1. **Wait for Σ results.** If V14 shows non-trivial SYMBIOTIC rate (>40%) on at least one probe set, proceed.
2. **Wait for E1 verdict.** If PASS (M2 fails on interpolation), Phantom Pain becomes an actual primitive. If FAIL, walk back the self-model claim transparently.
3. **Wait for standard benchmarks.** Just empirical addition.
4. **Author the new §3 Formalism with the CIRL definition + Cooperation Theorem.**
5. **Rewrite Abstract + §1 Introduction** around the new thesis.
6. **Author §5.6 Phantom Pain umbrella with E1/E2/E3 controls.** (E2 and E3 are future work for now; we discuss them as the natural extension protocol.)
7. **Rewrite §6 Limitations** to explicitly concede TEE, ASR-gap, distributed mechanism, cross-paradigm walk-back, Σ mixed pattern.

---
