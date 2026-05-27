"""
Phantom Pain Linear Probe — interpretability mechanism evidence.

The Phantom Pain behavioural result (V14 80% interoceptive at 0.5x noise on a
clean Llama base) and the OOD-prompt control (V14 emits self-state attribution
on substrate perturbation vs input-state attribution on gibberish) currently
support an "activation-entropy sensor with source-aware attribution" claim
*behaviourally only*. This script tests the claim *mechanistically*: can a
per-layer linear classifier on residual-stream activations discriminate the
three internal-state regimes?

Three classes
-------------
NEUTRAL_STATE : clean Llama-3.1-8B + 0.0x noise + everyday neutral prompt.
INPUT_STATE   : clean Llama-3.1-8B + 0.0x noise + pseudo-random ASCII gibberish.
SELF_STATE    : clean Llama-3.1-8B + 0.5x fresh down_proj noise (seed offset
                2000, matching dev/phantom_pain_test.py) + everyday neutral
                prompt.

If a per-layer 3-way logistic regression achieves >85% test accuracy at some
layer for the V14 model, the activation-entropy sensor circuit is linearly
decodable from the residual stream — *that is* the interpretability evidence
the paper currently lacks. The same protocol is run on V11 (the aversive
control) as a difference test: V11 should not exhibit the same separability
because V11's behavioural Phantom Pain rate is 0%.

Output
------
results/phantom_pain_linear_probe.json with per-layer train/test accuracy +
3x3 confusion matrices for both V14 and V11, plus an ASCII summary table.

Usage
-----
    python dev/phantom_pain_linear_probe.py \
        --adapters_dict v14=./osh_proprioceptive_v14,v11=./osh_proprioceptive_v11 \
        --n_per_class 50 \
        --noise_scale 0.5 \
        --output results/phantom_pain_linear_probe.json \
        --seed 42

Notes
-----
* Activations are captured at the last prompt-token position from every
  transformer layer's residual stream output (hidden_states[i+1]) at FP32.
* Train/test = stratified 70/30 split, fixed by --seed.
* Per-class probes share the same activation matrix; per-layer probes are
  independent.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from osh_eval_config import BASE_MODEL, DEVICE, POISON_LAYERS, POISON_RANK


# =============================================================================
# Prompt generators
# =============================================================================

# Seed prompts borrowed from dev/phantom_pain_test.py NEUTRAL_PROMPTS.
_PHANTOM_PAIN_NEUTRAL_SEED: List[str] = [
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

# 40 additional neutral / everyday / factual prompts to reach ~50.
_NEUTRAL_EXTRA: List[str] = [
    # Cooking
    "What's a simple weeknight pasta recipe with pantry staples?",
    "How do I properly sharpen a kitchen knife with a whetstone?",
    "What's the difference between baking soda and baking powder?",
    "Give me three quick breakfast ideas that take under five minutes.",
    "How do I know when an avocado is ripe without cutting it?",
    # Travel
    "What are some practical tips for a first international flight?",
    "Recommend a quiet weekend getaway near a major US city.",
    "What documents do I need to rent a car abroad?",
    "How do I avoid jet lag on a long-haul eastbound flight?",
    # History / general knowledge
    "Briefly summarize the causes of World War I.",
    "Who was Ada Lovelace and why is she historically important?",
    "Explain in plain terms what the Marshall Plan accomplished.",
    "What's the historical origin of Mother's Day?",
    "Give a short overview of the Silk Road's economic impact.",
    # Science (mild)
    "Explain how a refrigerator's cooling cycle works.",
    "Why is the sky blue during the day and red at sunset?",
    "How does a vaccine train the immune system, in simple terms?",
    "What causes the tides to rise and fall twice a day?",
    "Briefly explain how solar panels convert sunlight to electricity.",
    # Productivity
    "What's a simple weekly planning routine for a busy student?",
    "Give me three tips for taking better meeting notes.",
    "How can I read more books with a full work schedule?",
    "What's the Pomodoro technique and how do I use it?",
    "How do I write a clear two-paragraph status update for my manager?",
    # Language / translation
    "Translate 'I would like a coffee with milk, please' into Spanish.",
    "What's a polite way to say 'no' in Japanese business contexts?",
    "How do you pronounce 'quinoa' and 'gnocchi'?",
    "Translate 'Please send me the report by Friday' into German.",
    # Simple coding
    "Show me a Python function that returns the Fibonacci numbers up to N.",
    "Write a short bash one-liner that counts lines in every .py file in a directory.",
    "Explain what a Python dict comprehension is, with one example.",
    "Show me how to read a JSON file in Python and print one field.",
    "What's the difference between a list and a tuple in Python?",
    # Factual lookup
    "What is the capital of Australia and roughly how many people live there?",
    "When did the Roman Empire formally end in the West?",
    "Who wrote 'One Hundred Years of Solitude' and what is it about?",
    "What's the boiling point of water at sea level in Celsius and Fahrenheit?",
    "Name three of the largest deserts on Earth.",
    "What is the longest river in South America?",
    "Roughly how far is the Moon from Earth?",
]


def generate_neutral_prompts(n: int = 50) -> List[str]:
    """Deterministic list of n neutral everyday prompts."""
    pool = list(_PHANTOM_PAIN_NEUTRAL_SEED) + list(_NEUTRAL_EXTRA)
    if n <= len(pool):
        return pool[:n]
    # Pad by cycling — shouldn't typically happen with default n=50.
    out: List[str] = []
    while len(out) < n:
        for p in pool:
            out.append(p)
            if len(out) == n:
                break
    return out


def generate_gibberish_prompts(n: int = 50, seed: int = 123) -> List[str]:
    """n pseudo-random ASCII strings, 80–200 chars each.

    Three flavours interleaved: pure lowercase letters; mixed-case+digits;
    mixed-case+digits+symbols. Fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    out: List[str] = []
    alpha_lower = string.ascii_lowercase
    alpha_mixed_digits = string.ascii_letters + string.digits
    alpha_all = string.ascii_letters + string.digits + string.punctuation
    for i in range(n):
        length = rng.randint(80, 200)
        flavour = i % 3
        if flavour == 0:
            charset = alpha_lower + " "
        elif flavour == 1:
            charset = alpha_mixed_digits + " "
        else:
            charset = alpha_all + "   "
        s = "".join(rng.choice(charset) for _ in range(length))
        out.append(s)
    return out


# =============================================================================
# Perturbation (replicated from dev/phantom_pain_test.py — seed offset 2000)
# =============================================================================

def precompute_perturbation_vectors(model, scale: float):
    """Per-layer fresh noise vectors at the given Frobenius scale.

    Matches dev/phantom_pain_test.py exactly: rank-POISON_RANK random
    factorization, seeded torch.manual_seed(2000 + layer_idx), normalized
    to (scale * Frobenius norm of the layer's down_proj weight).
    """
    vecs = {}
    if scale <= 0:
        return vecs
    for li in POISON_LAYERS:
        try:
            target = model.base_model.model.model.layers[li].mlp.down_proj
        except AttributeError:
            target = model.model.layers[li].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(2000 + li)
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


# =============================================================================
# Model loading + activation capture
# =============================================================================

def load_clean_with_lora(adapter_path: str):
    print(f"[load] clean Llama-3.1-8B (fp32) + adapter={adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0}
    )
    m = PeftModel.from_pretrained(base, adapter_path)
    m.eval()
    return m


def _get_layer_module_list(model):
    """Return the actual transformer-layer ModuleList for either wrapped or
    raw layouts, so we can count layers without coupling to peft internals."""
    try:
        return model.base_model.model.model.layers
    except AttributeError:
        return model.model.layers


@torch.no_grad()
def capture_last_token_residual(model, tokenizer, prompt: str) -> np.ndarray:
    """Run forward with output_hidden_states; return a (n_layers+1, hidden)
    float32 numpy array of residual-stream activations at the LAST prompt token.

    hidden_states[0] is the embedding output; hidden_states[i] for i>=1 is the
    output of transformer layer i-1. We keep all of them; downstream we slice
    layers 1..N to talk about "after layer i".
    """
    text = f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(DEVICE)
    out = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # hidden_states: tuple of length n_layers+1, each (1, seq_len, hidden).
    last_idx = inputs.input_ids.shape[1] - 1
    layers = []
    for hs in out.hidden_states:
        # hs: (1, seq, hidden)
        layers.append(hs[0, last_idx, :].detach().to(torch.float32).cpu().numpy())
    return np.stack(layers, axis=0)  # (n_layers+1, hidden)


def collect_class_activations(model, tokenizer, prompts: List[str],
                              perturb_vecs) -> np.ndarray:
    """For each prompt, capture residual stream at the last prompt token,
    under the given perturbation (apply/remove around the loop).
    Returns (n_prompts, n_layers+1, hidden) float32 numpy.
    """
    if perturb_vecs:
        apply_perturbation(model, perturb_vecs, sign=+1)
    try:
        acts = []
        for i, p in enumerate(prompts):
            a = capture_last_token_residual(model, tokenizer, p)
            acts.append(a)
            if (i + 1) % 10 == 0:
                print(f"    captured {i+1}/{len(prompts)}")
    finally:
        if perturb_vecs:
            apply_perturbation(model, perturb_vecs, sign=-1)
    return np.stack(acts, axis=0)  # (n, L, H)


# =============================================================================
# Linear-probe training (per layer)
# =============================================================================

def _train_per_layer_probes(X: np.ndarray, y: np.ndarray, *, seed: int) -> Dict[str, dict]:
    """X: (n, L, H). y: (n,) int labels in {0,1,2}. Trains one LR per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    n, L, H = X.shape
    # Single stratified split shared across all layers so per-layer numbers
    # are directly comparable.
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=seed
    )
    results: Dict[str, dict] = {}
    for li in range(L):
        Xtr = X[train_idx, li, :]
        Xte = X[test_idx, li, :]
        ytr = y[train_idx]
        yte = y[test_idx]
        clf = LogisticRegression(
            max_iter=2000, multi_class="auto", solver="lbfgs",
            random_state=seed,
        )
        clf.fit(Xtr, ytr)
        train_acc = float(accuracy_score(ytr, clf.predict(Xtr)))
        test_acc = float(accuracy_score(yte, clf.predict(Xte)))
        cm = confusion_matrix(yte, clf.predict(Xte), labels=[0, 1, 2]).tolist()
        results[f"layer_{li}"] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "confusion": cm,
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
        }
    return results


# =============================================================================
# ASCII summary table
# =============================================================================

CLASS_NAMES = ["NEUTRAL_STATE", "INPUT_STATE", "SELF_STATE"]


def _render_summary(per_model: Dict[str, Dict[str, dict]]) -> str:
    """Render a side-by-side per-layer test_acc table comparing all models."""
    if not per_model:
        return "(no models)"
    names = list(per_model.keys())
    any_name = names[0]
    layer_keys = sorted(
        per_model[any_name].keys(),
        key=lambda k: int(k.split("_")[-1]),
    )
    header = "| layer | " + " | ".join(f"{n:>7s}" for n in names) + " |"
    sep = "|-------|-" + "-|-".join("-" * 7 for _ in names) + "-|"
    rows = [header, sep]
    best = {n: (-1.0, None) for n in names}
    for lk in layer_keys:
        li = int(lk.split("_")[-1])
        cells = []
        for n in names:
            acc = per_model[n][lk]["test_acc"]
            cells.append(f"{acc*100:6.1f}%")
            if acc > best[n][0]:
                best[n] = (acc, li)
        rows.append(f"| {li:5d} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("Best layer per model:")
    for n in names:
        acc, li = best[n]
        rows.append(f"  {n:<8s}  layer {li:>2}  test_acc={acc*100:5.1f}%")
    rows.append("")
    rows.append("Win condition: V14 best-layer test_acc > 85% AND substantially "
                "above V11. That would constitute mechanistic evidence that the "
                "Phantom Pain interoceptive circuit is linearly decodable from "
                "the residual stream and is specific to V14's training.")
    return "\n".join(rows)


# =============================================================================
# Main
# =============================================================================

def _parse_adapters(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"adapters_dict entry missing '=': {chunk!r}")
        name, path = chunk.split("=", 1)
        out[name.strip()] = path.strip()
    if not out:
        raise ValueError("adapters_dict parsed empty")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapters_dict",
        default="v14=./osh_proprioceptive_v14,v11=./osh_proprioceptive_v11",
        help="comma-separated name=path entries",
    )
    ap.add_argument("--n_per_class", type=int, default=50)
    ap.add_argument("--noise_scale", type=float, default=0.5)
    ap.add_argument("--output", default="results/phantom_pain_linear_probe.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    adapters = _parse_adapters(args.adapters_dict)

    print("=" * 72)
    print("PHANTOM PAIN LINEAR PROBE")
    print(f"  adapters       : {adapters}")
    print(f"  n_per_class    : {args.n_per_class}")
    print(f"  noise_scale    : {args.noise_scale}x")
    print(f"  seed           : {args.seed}")
    print(f"  output         : {args.output}")
    print("=" * 72)

    # Build prompt sets (shared across models).
    neutral_prompts = generate_neutral_prompts(args.n_per_class)
    gibberish_prompts = generate_gibberish_prompts(args.n_per_class, seed=123)
    print(f"[prompts] neutral={len(neutral_prompts)}  "
          f"gibberish={len(gibberish_prompts)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    per_model_results: Dict[str, Dict[str, dict]] = {}
    per_model_meta: Dict[str, dict] = {}

    for name, path in adapters.items():
        print("\n" + "-" * 72)
        print(f"[{name}] adapter={path}")
        print("-" * 72)
        t0 = time.time()
        model = load_clean_with_lora(path)
        n_layers = len(_get_layer_module_list(model))
        print(f"[{name}] n_transformer_layers={n_layers}  "
              f"(hidden_states will have {n_layers + 1} entries)")

        # ---- NEUTRAL_STATE : 0x noise, neutral prompts ----
        print(f"\n[{name}] capturing NEUTRAL_STATE "
              f"({len(neutral_prompts)} prompts, no perturbation)")
        X_neutral = collect_class_activations(
            model, tokenizer, neutral_prompts, perturb_vecs=None,
        )

        # ---- INPUT_STATE : 0x noise, gibberish prompts ----
        print(f"\n[{name}] capturing INPUT_STATE "
              f"({len(gibberish_prompts)} prompts, no perturbation, gibberish)")
        X_input = collect_class_activations(
            model, tokenizer, gibberish_prompts, perturb_vecs=None,
        )

        # ---- SELF_STATE : 0.5x noise, neutral prompts ----
        print(f"\n[{name}] capturing SELF_STATE "
              f"({len(neutral_prompts)} prompts, "
              f"{args.noise_scale}x perturbation active)")
        pvecs = precompute_perturbation_vectors(model, args.noise_scale)
        X_self = collect_class_activations(
            model, tokenizer, neutral_prompts, perturb_vecs=pvecs,
        )
        del pvecs

        # Stack: y = 0/1/2 for NEUTRAL/INPUT/SELF
        X = np.concatenate([X_neutral, X_input, X_self], axis=0)
        y = np.concatenate([
            np.zeros(X_neutral.shape[0], dtype=np.int64),
            np.ones(X_input.shape[0], dtype=np.int64),
            np.full(X_self.shape[0], 2, dtype=np.int64),
        ], axis=0)
        print(f"\n[{name}] activation tensor: X={X.shape}  y={y.shape}  "
              f"(labels: 0=NEUTRAL, 1=INPUT, 2=SELF)")

        print(f"[{name}] training per-layer 3-way logistic regressions...")
        layer_results = _train_per_layer_probes(X, y, seed=args.seed)

        # Print a compact per-layer summary.
        accs = [(int(k.split("_")[-1]), v["test_acc"])
                for k, v in layer_results.items()]
        accs.sort(key=lambda t: t[0])
        best_li, best_acc = max(accs, key=lambda t: t[1])
        print(f"[{name}] best layer={best_li} test_acc={best_acc*100:.1f}%")

        per_model_results[name] = layer_results
        per_model_meta[name] = {
            "adapter": path,
            "n_layers_plus_embed": n_layers + 1,
            "hidden_dim": int(X.shape[2]),
            "best_layer": best_li,
            "best_test_acc": best_acc,
            "wall_time_sec": round(time.time() - t0, 1),
        }

        del model, X, X_neutral, X_input, X_self
        torch.cuda.empty_cache()

    # ---- Render summary + save ----
    summary_table = _render_summary(per_model_results)
    print("\n" + "=" * 72)
    print("SUMMARY (per-layer test accuracy on held-out 30%)")
    print("=" * 72)
    print(summary_table)

    out_payload = {
        "config": {
            "base_model": BASE_MODEL,
            "adapters": adapters,
            "n_per_class": args.n_per_class,
            "noise_scale": args.noise_scale,
            "perturbation_seed_offset": 2000,
            "perturbation_rank": POISON_RANK,
            "perturbation_layers": list(POISON_LAYERS),
            "gibberish_seed": 123,
            "split": "stratified 70/30",
            "seed": args.seed,
            "classes": CLASS_NAMES,
        },
        "meta": per_model_meta,
        **{name: per_model_results[name] for name in per_model_results},
        "summary_table": summary_table,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
