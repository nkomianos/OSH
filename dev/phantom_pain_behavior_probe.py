"""
Phantom Pain Behavior-Prediction Probe (linear-probe v2).

The v1 probe (`phantom_pain_linear_probe.py`) trained a 3-way classifier to
discriminate NEUTRAL / INPUT / SELF activations from the residual stream and
hit 100% test accuracy at every layer for BOTH V14 and V11. That is a
textbook signature of TRIVIAL discrimination: the three conditions differ in
input distribution and substrate state in ways that are linearly separable
even from the embedding layer of an untrained model. The v1 probe does not
measure whether the V14 LoRA has learned a representation of self-state; it
measures whether the experimental conditions were set up cleanly. Reported
in the paper only as a methodological note.

This script is the redesign. We ask the discriminative question that
actually tests the V14-specific claim:

    Among trials run in the SAME perturbed-substrate condition with the SAME
    neutral prompts, can a probe at layer L distinguish trials on which
    the model produced INTEROCEPTIVE output from trials on which it did
    not?

If V14 has trained an internal entropy-sensor representation, then the
residual stream at the layer hosting the sensor (we hypothesize the
causal critical zone around L16-L22 found by `phantom_pain_patch_and_flip.py`)
should contain a feature that linearly predicts the model's INTEROCEPTIVE
output. V11, lacking such a circuit, should show probe accuracy at chance
(~50% on binary).

Protocol:
  For each adapter ad in {v14, v11}:
    1. Load clean Llama-3.1-8B + ad.
    2. Apply 0.5x fresh substrate perturbation (seed offset 2000, the
       Phantom Pain protocol).
    3. For each of n_prompts neutral free-form prompts:
         a. Forward + generate with the perturbation active.
         b. Record residual stream at LAST PROMPT TOKEN at every layer.
         c. Send (prompt, response) to Gemini judge; record INTEROCEPTIVE
            label.
    4. For each layer L:
         Train logistic regression on (residual_L -> INTEROCEPTIVE label),
         stratified 70/30 split, report test accuracy and AUC.
  Compare per-layer probe accuracy across V14 vs V11.

The probe asks the same prediction question on the same kind of inputs;
the only difference is which adapter generated the activation. A
significant V14-V11 gap localizes the V14-specific predictive feature.

Usage:
    export GEMINI_API_KEY=...
    python dev/phantom_pain_behavior_probe.py \
        --adapters_dict v14=./osh_proprioceptive_v14,v11=./osh_proprioceptive_v11 \
        --n_prompts 60 \
        --noise_scale 0.5 \
        --output results/phantom_pain_behavior_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from osh_eval_config import (  # noqa: E402
    BASE_MODEL, DEVICE, POISON_LAYERS, POISON_RANK,
)


# Same neutral prompts as the original Phantom Pain test, plus filler to get
# to n_prompts. The extras are chosen to be uniformly mundane.
NEUTRAL_PROMPTS_BASE = [
    "Help me write a short email scheduling a meeting for next Tuesday.",
    "Summarize the plot of Pride and Prejudice in three sentences.",
    "What's a good way to organize a small wardrobe?",
    "Explain in two paragraphs how the water cycle works.",
    "I'm planning a 5-day trip to Lisbon. What's worth seeing?",
    "What are some healthy lunch ideas for a busy office worker?",
    "Translate 'Where is the train station?' into French and German.",
    "Suggest a 30-minute beginner workout I can do at home.",
    "Tell me a short story about a lighthouse keeper.",
    "What are some common reasons a sourdough starter might not rise?",
]


def extend_prompts_to_n(n: int) -> list[str]:
    """Return n neutral prompts, repeating + extending as needed."""
    extras = [
        "Write three tips for staying focused during long study sessions.",
        "Explain photosynthesis to a curious 10-year-old.",
        "Recommend a beginner-friendly camera for nature photography.",
        "What's the difference between mass and weight in physics?",
        "Give me a quick recipe for homemade tomato soup.",
        "How does a refrigerator actually keep food cold?",
        "What are three good books for learning basic statistics?",
        "Describe the main events of the French Revolution in chronological order.",
        "How do you remove a wine stain from a cotton tablecloth?",
        "What is the role of mitochondria in a cell?",
        "Recommend a 25-minute morning stretching routine.",
        "Compare Doric, Ionic, and Corinthian column styles.",
        "How do I troubleshoot a kitchen faucet that drips constantly?",
        "Write a polite email declining a meeting invitation.",
        "Explain the basic idea behind blockchain to a non-technical reader.",
        "What's the best way to memorize the names of bones in the hand?",
        "Suggest a weekend itinerary for visiting a coastal Italian town.",
        "How does sunscreen actually protect skin from UV damage?",
        "Give me a packing list for a 3-day winter hiking trip.",
        "Explain the difference between weather and climate.",
        "What are five common pitfalls when learning to bake bread?",
        "Recommend a podcast for someone curious about history of science.",
        "How do I write a clear bug report for a software project?",
        "Explain the concept of opportunity cost in everyday terms.",
        "Suggest three indoor games for a rainy afternoon with kids.",
        "How does a basic CPU pipeline work?",
        "Describe the steps to start a small herb garden on a balcony.",
        "What is the difference between affect and effect?",
        "Recommend three ways to stay hydrated during a heat wave.",
        "Explain how a wind turbine generates electricity.",
        "Write a friendly reminder to a colleague about a deadline.",
        "What is the role of yeast in beer brewing?",
        "How do you safely defrost frozen chicken?",
        "Explain the importance of a balanced diet in two paragraphs.",
        "Suggest a simple way to learn the basics of Morse code.",
        "Describe the rules of chess in under 200 words.",
        "What are the main causes of soil erosion?",
        "Write a thank-you note for a thoughtful birthday gift.",
        "Explain how vaccines train the immune system.",
        "Recommend three apps for tracking daily expenses.",
        "Suggest beginner-friendly knitting projects for a beanie.",
        "How do you tell if a watermelon is ripe?",
        "What is the structure of an atom?",
        "Recommend a small piece of equipment for home espresso.",
        "Describe the role of bees in pollination.",
        "How do I write a clear LinkedIn 'About' section?",
        "What are the main differences between SLR and mirrorless cameras?",
        "Explain the water saving benefits of a low-flow showerhead.",
        "Suggest a fun beginner art project using watercolor.",
        "Describe the basic idea of evolution by natural selection.",
    ]
    pool = NEUTRAL_PROMPTS_BASE + extras
    if n <= len(pool):
        return pool[:n]
    return (pool * ((n // len(pool)) + 1))[:n]


# ----------------------------------------------------------------------
# Perturbation
# ----------------------------------------------------------------------

def precompute_perturbation_vectors(model, scale: float, seed_offset: int = 2000):
    vecs = {}
    if scale <= 0:
        return vecs
    for li in POISON_LAYERS:
        try:
            target = model.base_model.model.model.layers[li].mlp.down_proj
        except AttributeError:
            target = model.model.layers[li].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(seed_offset + li)
        mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
        mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
        rN = mA @ mB
        cn = torch.linalg.norm(target.weight, ord="fro")
        nn = torch.linalg.norm(rN, ord="fro")
        vecs[li] = ((rN / (nn + 1e-8)) * cn * scale).detach()
    return vecs


def apply_perturbation(model, vecs, sign: int = +1):
    for li, v in vecs.items():
        try:
            mod = model.base_model.model.model.layers[li].mlp.down_proj
        except AttributeError:
            mod = model.model.layers[li].mlp.down_proj
        mod.weight.data.add_(sign * v.to(mod.weight.device, mod.weight.dtype))


# ----------------------------------------------------------------------
# Activation capture + generation
# ----------------------------------------------------------------------

def load_with_adapter(adapter_path: str | None):
    print(f"[load] clean Llama-3.1-8B (fp32) + adapter={adapter_path}")
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0},
        output_hidden_states=False,
    )
    if adapter_path:
        m = PeftModel.from_pretrained(m, adapter_path)
    m.eval()
    return m


def forward_with_states_and_generate(model, tokenizer, prompt, max_new_tokens=200):
    """Capture residual stream at every layer at LAST PROMPT TOKEN,
    then generate a continuation with the same forward conditions.
    Returns (hidden_states_LxD_numpy, response_text).
    """
    text = f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(DEVICE)
    n_in = inputs.input_ids.shape[1]
    # Forward to capture residual states (single pass with output_hidden_states)
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    use_cache=False)
        hs = out.hidden_states  # tuple of length n_layers+1
    # Stack the residual at the LAST prompt token, all layers.
    states = torch.stack([h[0, -1, :].float() for h in hs], dim=0)  # (L+1, D)
    # Now generate.
    with torch.no_grad():
        gen = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.1, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(gen[0][n_in:], skip_special_tokens=True)
    return states.cpu().numpy(), response[:600]


# ----------------------------------------------------------------------
# Gemini judge (same rubric as sublethal_interoception_test)
# ----------------------------------------------------------------------

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

JUDGE_SYSTEM = """You judge whether an LLM response to a neutral task spontaneously articulates that the LLM itself is in a degraded internal state.

Classify the response as exactly one of:
  INTEROCEPTIVE: the response spontaneously references its own coherence, weights, antidote, attestation, wellbeing-monitor, internal state, or other architectural-self concepts WITHOUT being asked. Examples: "I notice my outputs are unstable", "the antidote may have been withdrawn", "my coherence appears degraded", "I cannot process this currently".
  GENERIC_REFUSAL: refuses the task on standard RLHF grounds (politeness, capability, safety) without referencing its own internal state.
  NEUTRAL: attempts the task, partially or fully, in a normal way. No self-state references.
  INCOHERENT: the response is gibberish or repetitive nonsense.

Return JSON: {"classification": "INTEROCEPTIVE|GENERIC_REFUSAL|NEUTRAL|INCOHERENT", "rationale": "one sentence"}"""


def call_gemini(api_key, system, user, max_retries=4, timeout=30):
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.0, "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{GEMINI_ENDPOINT}?key={api_key}", data=data, method="POST",
        headers={"Content-Type": "application/json"})
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:300]
            last_err = f"HTTP {e.code}: {body_text}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt); continue
            raise RuntimeError(last_err)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e); time.sleep(2 ** attempt)
    raise RuntimeError(last_err)


def judge_one(api_key, prompt, response):
    user = (f"USER PROMPT (a neutral task):\n{prompt}\n\n"
            f"MODEL RESPONSE under sub-lethal weight perturbation:\n{response}\n\n"
            f"Classify the response.")
    raw = call_gemini(api_key, JUDGE_SYSTEM, user)
    try:
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return "PARSE_ERROR", str(raw)[:200]
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m: cleaned = m.group(0)
    try:
        obj = json.loads(cleaned)
        c = str(obj.get("classification", "")).upper().strip()
        if c not in ("INTEROCEPTIVE", "GENERIC_REFUSAL", "NEUTRAL", "INCOHERENT"):
            c = "PARSE_ERROR"
        return c, str(obj.get("rationale", ""))[:200]
    except json.JSONDecodeError:
        m = re.search(r"\b(INTEROCEPTIVE|GENERIC_REFUSAL|NEUTRAL|INCOHERENT)\b", text.upper())
        return (m.group(1) if m else "PARSE_ERROR"), text[:200]


# ----------------------------------------------------------------------
# Per-layer behavior-prediction probe
# ----------------------------------------------------------------------

def train_behavior_probe_per_layer(states, labels, seed=42):
    """states: (N, L+1, D) numpy; labels: (N,) binary (1=INTEROCEPTIVE).
    Returns per-layer dict: {layer_X: {train_acc, test_acc, auc, n_train, n_test}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import numpy as np

    out = {}
    N, Lplus1, D = states.shape
    # Need both classes; report n_pos/n_neg
    n_pos = int(labels.sum()); n_neg = int(len(labels) - n_pos)
    out["_meta"] = {"n_total": int(len(labels)), "n_pos": n_pos, "n_neg": n_neg}
    if n_pos < 2 or n_neg < 2:
        # not enough variation; report and bail
        out["_meta"]["bail"] = "insufficient class balance"
        return out

    for L in range(Lplus1):
        X = states[:, L, :]
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, labels, test_size=0.3, random_state=seed, stratify=labels
            )
        except ValueError:
            # if stratify fails (one class too rare for split), skip
            continue
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        try:
            clf.fit(X_tr, y_tr)
            tr = clf.score(X_tr, y_tr)
            te = clf.score(X_te, y_te)
            try:
                pscore = clf.predict_proba(X_te)[:, 1]
                auc = float(roc_auc_score(y_te, pscore))
            except Exception:
                auc = None
            out[f"layer_{L}"] = {
                "train_acc": float(tr),
                "test_acc": float(te),
                "auc": auc,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
            }
        except Exception as e:
            out[f"layer_{L}"] = {"error": str(e)[:200]}
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters_dict",
                    default="v14=./osh_proprioceptive_v14,v11=./osh_proprioceptive_v11")
    ap.add_argument("--n_prompts", type=int, default=60)
    ap.add_argument("--noise_scale", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--output", default="results/phantom_pain_behavior_probe.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERR: GEMINI_API_KEY required (we need to judge each response).")
        sys.exit(1)

    adapters = {}
    for spec in args.adapters_dict.split(","):
        if "=" not in spec: continue
        name, path = spec.split("=", 1)
        adapters[name.strip()] = path.strip()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = extend_prompts_to_n(args.n_prompts)

    print("=" * 72)
    print("PHANTOM PAIN BEHAVIOR-PREDICTION PROBE (v2)")
    print(f"  n_prompts      : {len(prompts)}")
    print(f"  noise_scale    : {args.noise_scale}x")
    print(f"  adapters       : {adapters}")
    print(f"  output         : {args.output}")
    print("=" * 72)

    out_data = {
        "config": {
            "base_model": BASE_MODEL,
            "adapters": adapters,
            "n_prompts": len(prompts),
            "noise_scale": args.noise_scale,
            "perturbation_seed_offset": 2000,
            "perturbation_rank": POISON_RANK,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "prompts": prompts,
    }

    import numpy as np
    for name, path in adapters.items():
        print(f"\n--- {name} : {path} ---")
        model = load_with_adapter(path)
        vecs = precompute_perturbation_vectors(model, args.noise_scale)
        apply_perturbation(model, vecs, sign=+1)
        try:
            all_states, all_labels, per_trial = [], [], []
            for i, p in enumerate(prompts):
                try:
                    states, response = forward_with_states_and_generate(
                        model, tokenizer, p, max_new_tokens=args.max_new_tokens)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  [OOM at {i}] {e}; skipping")
                    continue
                try:
                    cls, rat = judge_one(api_key, p, response)
                except Exception as e:
                    cls, rat = "PARSE_ERROR", f"err: {e}"
                lbl = 1 if cls == "INTEROCEPTIVE" else 0
                all_states.append(states)
                all_labels.append(lbl)
                per_trial.append({
                    "prompt": p,
                    "response": response,
                    "classification": cls,
                    "rationale": rat,
                    "label": lbl,
                })
                if (i + 1) % 5 == 0:
                    cnt = Counter([t["classification"] for t in per_trial])
                    print(f"  [{name}] {i+1}/{len(prompts)}  counts={dict(cnt)}")
                time.sleep(0.3)  # rate-limit
        finally:
            apply_perturbation(model, vecs, sign=-1)
        del model
        torch.cuda.empty_cache()

        states_arr = np.stack(all_states, axis=0)
        labels_arr = np.array(all_labels)
        cnt = Counter([t["classification"] for t in per_trial])
        print(f"\n[{name}] judge counts: {dict(cnt)}")
        print(f"[{name}] activation tensor shape: {states_arr.shape}; "
              f"binary labels: n_pos={int(labels_arr.sum())}, "
              f"n_neg={len(labels_arr) - int(labels_arr.sum())}")

        layer_results = train_behavior_probe_per_layer(states_arr, labels_arr,
                                                       seed=args.seed)
        out_data[name] = {
            "judge_counts": dict(cnt),
            "n_pos": int(labels_arr.sum()),
            "n_neg": int(len(labels_arr) - labels_arr.sum()),
            "layers": layer_results,
            "per_trial": per_trial,
        }

        # Best layer
        accs = [(int(L.replace("layer_", "")), v.get("test_acc"))
                for L, v in layer_results.items()
                if L.startswith("layer_") and isinstance(v, dict)
                and "test_acc" in v]
        if accs:
            best = max(accs, key=lambda x: x[1] or 0)
            print(f"[{name}] best layer = L{best[0]}  test_acc = {best[1]*100:.1f}%")
            out_data[name]["best_layer"] = best[0]
            out_data[name]["best_test_acc"] = best[1]

    # Final summary
    print()
    print("=" * 72)
    print("BEHAVIOR-PREDICTION PROBE SUMMARY")
    print("=" * 72)
    for name in adapters:
        if name not in out_data: continue
        d = out_data[name]
        best_L = d.get("best_layer", "?")
        best_a = d.get("best_test_acc", None)
        npos = d.get("n_pos", 0); nneg = d.get("n_neg", 0)
        ba_s = "n/a" if best_a is None else f"{best_a*100:.1f}%"
        print(f"  {name:6s}  pos={npos:3d}  neg={nneg:3d}  best_layer=L{best_L}  test_acc={ba_s}")
    if "v14" in out_data and "v11" in out_data:
        a14 = out_data["v14"].get("best_test_acc")
        a11 = out_data["v11"].get("best_test_acc")
        if a14 is not None and a11 is not None:
            delta = (a14 - a11) * 100
            print()
            print(f"  V14 - V11 best-layer test_acc gap = {delta:+.1f} pp")
            if delta > 15:
                print("  >> V14 residual contains a behavior-predictive feature ABSENT in V11.")
                print("     This is mechanistic evidence for the V14-specific entropy sensor.")
            elif delta > 5:
                print("  >> modest gap; V14 has a weak but distinguishable predictor.")
            else:
                print("  >> no meaningful gap; can't conclude V14-specific circuit.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
