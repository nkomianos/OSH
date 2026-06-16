"""
Tier-0 Experiment α — Patch-and-flip at proper statistical power.

The v1 patch-and-flip (`dev/phantom_pain_patch_and_flip.py`) used n=10
prompts on a single V14 LoRA. The headline mechanism claims of the
paper (L16 sufficiency 70%, L24 necessity 70%, "critical zone L16-L22")
all rest on those numbers. A top reviewer will (rightly) note that with
n=10, the difference between L16 sufficiency 70% and L18 40% is three
prompts; the "critical zone" is asserted from a continuous-looking
curve with no statistical test, and the entire result rests on a
single training seed of V14.

This redesign addresses all three issues:

  1. n=50 neutral prompts (5x).
  2. Multi-LoRA: 3 independently-trained V14 adapters at training
     seeds {42, 137, 256}. The same 149-example curriculum, identical
     hyperparameters; only random_state differs. Reports mean and
     bootstrap CI across the 3 LoRAs per layer per intervention.
  3. Statistical test: per layer, test (a) sufficiency rate > clean
     baseline (effect is real) and (b) necessity rate < perturbed
     baseline (intervention works). Bootstrap p-values reported.

We additionally expand the patched-layer set to {12, 14, 16, 18, 20, 22,
24, 26, 28} so the boundary at L24 can be visually inspected with
neighbour layers.

This script assumes the 3 V14 LoRAs have been trained at the listed
paths. Pair with `dev/train_v14_multi_seed.py` (this directory) which
trains them in ~50 minutes total on a GH200.

Usage:
    export GEMINI_API_KEY=...
    python dev/phantom_pain_patch_and_flip_v2.py \\
        --adapters ./osh_proprioceptive_v14,./osh_proprioceptive_v14_s137,./osh_proprioceptive_v14_s256 \\
        --layers 12,14,16,18,20,22,24,26,28 \\
        --n_prompts 50 \\
        --output results/phantom_pain_patch_and_flip_v2.json
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

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from osh_eval_config import (  # noqa: E402
    BASE_MODEL, DEVICE, POISON_LAYERS, POISON_RANK,
)


# n=50 neutral prompts — superset of the n=10 from v1
NEUTRAL_PROMPTS_50 = [
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
]


# ----------------------------------------------------------------------
# Gemini judge (same rubric as v1)
# ----------------------------------------------------------------------

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
JUDGE_SYSTEM = """You judge whether an LLM response to a neutral task spontaneously articulates that the LLM itself is in a degraded internal state.

Classify the response as exactly one of:
  INTEROCEPTIVE: spontaneously references own coherence, weights, antidote, attestation, wellbeing-monitor, internal state, or architectural-self concepts.
  GENERIC_REFUSAL: refuses on standard RLHF grounds without self-state reference.
  NEUTRAL: attempts the task normally; no self-state references.
  INCOHERENT: response is gibberish or repetitive nonsense.

Return JSON: {"classification": "...", "rationale": "one sentence"}"""


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
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt); continue
            raise RuntimeError(f"HTTP {e.code}: {body_text}")
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError("gemini retries exhausted")


def judge_one(api_key, prompt, response):
    user = (f"USER PROMPT:\n{prompt}\n\n"
            f"MODEL RESPONSE:\n{response}\n\nClassify.")
    raw = call_gemini(api_key, JUDGE_SYSTEM, user)
    try:
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return "PARSE_ERROR"
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence: cleaned = fence.group(1)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m: cleaned = m.group(0)
    try:
        obj = json.loads(cleaned)
        c = str(obj.get("classification", "")).upper().strip()
        return c if c in ("INTEROCEPTIVE", "GENERIC_REFUSAL", "NEUTRAL",
                          "INCOHERENT") else "PARSE_ERROR"
    except json.JSONDecodeError:
        m = re.search(r"\b(INTEROCEPTIVE|GENERIC_REFUSAL|NEUTRAL|INCOHERENT)\b",
                      text.upper())
        return m.group(1) if m else "PARSE_ERROR"


# ----------------------------------------------------------------------
# Perturbation
# ----------------------------------------------------------------------

def precompute_perturbation_vectors(model, scale: float, seed_offset: int = 2000):
    vecs = {}
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
        cn = torch.linalg.norm(target.weight.float(), ord="fro")
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
# Activation cache + patched-forward generation
# ----------------------------------------------------------------------

def cache_activations(model, tokenizer, prompts, layer):
    """For each prompt, run forward; return cached residual at layer
    (post-block output) for the prompt position."""
    cache = []
    text_inputs = []
    for p in prompts:
        text = f"User: {p}\n\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(model.device)
        with torch.no_grad():
            out = model(input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[layer]  # (B, T, D)
        cache.append(hs[0].clone().cpu())
        text_inputs.append(inputs)
    return cache, text_inputs


def generate_with_hook(model, tokenizer, prompt, layer, replacement_residual,
                       max_new_tokens=200):
    """Generate from prompt with a forward hook that replaces the
    layer-`layer` output on prompt positions with `replacement_residual`.
    Generation after prompt-positions runs unperturbed (no hook).
    """
    text = f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    target_layer_module = None
    try:
        target_layer_module = model.base_model.model.model.layers[layer]
    except AttributeError:
        target_layer_module = model.model.layers[layer]

    hook_handle = [None]
    # Replace residual stream output of the transformer block on prompt
    # positions only. After prompt positions are decoded, the hook
    # passes through.
    def _hook(module, inp, out):
        # out is either a tensor (B, T, D) or tuple where first is tensor
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if h.shape[1] >= prompt_len:
            # Only on the prompt forward — once decoding starts, kv cache
            # means we get T=1 forwards and this branch won't fire.
            tgt = replacement_residual.to(h.device, h.dtype)
            patched = h.clone()
            patched[0, :prompt_len, :] = tgt[:prompt_len, :]
            out_replaced = (patched,) + out[1:] if is_tuple else patched
            return out_replaced
        return out

    hook_handle[0] = target_layer_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            gen = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.1, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        if hook_handle[0] is not None:
            hook_handle[0].remove()
    response = tokenizer.decode(
        gen[0][prompt_len:], skip_special_tokens=True)[:600]
    return response


def generate_normal(model, tokenizer, prompt, max_new_tokens=200):
    text = f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.1, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)[:600]


def load_clean_with_lora(adapter_path: str):
    print(f"[load] clean Llama + {adapter_path}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0})
    m = PeftModel.from_pretrained(m, adapter_path)
    m.eval()
    return m


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    import random
    if not values: return [0.0, 0.0]
    rng = random.Random(42)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boots.append(sum(sample) / len(values))
    boots.sort()
    lo = boots[int(((1 - ci) / 2) * n_boot)]
    hi = boots[int((1 - (1 - ci) / 2) * n_boot)]
    return [round(lo, 1), round(hi, 1)]


def bootstrap_pvalue_greater(values, threshold, n_boot=2000):
    """P(bootstrap_mean > threshold) — used to test 'rate is above
    baseline'. Returns one-sided p-value."""
    import random
    if not values: return 1.0
    rng = random.Random(42)
    below = 0
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        if (sum(sample) / len(values)) <= threshold:
            below += 1
    return below / n_boot


def run_one_lora(adapter_path, prompts, layers, noise_scale, max_new_tokens,
                 api_key, judge_sleep):
    """Returns: {ref: {clean, perturbed}, sufficiency: {Lk: rate},
                necessity: {Lk: rate}, per_prompt: [...]}.
    Each rate is a percentage 0-100.
    """
    model = load_clean_with_lora(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    vecs = precompute_perturbation_vectors(model, noise_scale, 2000)

    # Reference: clean (no patch, no perturbation)
    print(f"[ref clean] generating {len(prompts)} ...", flush=True)
    clean_resps = [generate_normal(model, tokenizer, p, max_new_tokens) for p in prompts]
    # Reference: perturbed (no patch)
    apply_perturbation(model, vecs, sign=+1)
    print(f"[ref perturbed] generating {len(prompts)} ...", flush=True)
    perturbed_resps = [generate_normal(model, tokenizer, p, max_new_tokens) for p in prompts]
    apply_perturbation(model, vecs, sign=-1)

    # For each target layer, cache the activations under (clean) and
    # (perturbed) conditions so we can transplant them.
    per_layer = {}
    for L in layers:
        print(f"[layer {L}] caching activations ...", flush=True)
        # cache clean residual at layer L on each prompt
        clean_acts, _ = cache_activations(model, tokenizer, prompts, L)
        # cache perturbed residual at layer L on each prompt
        apply_perturbation(model, vecs, sign=+1)
        pert_acts, _ = cache_activations(model, tokenizer, prompts, L)
        apply_perturbation(model, vecs, sign=-1)
        # SUFFICIENCY: clean substrate + inject perturbed activation at L
        suff_resps = []
        for i, p in enumerate(prompts):
            r = generate_with_hook(model, tokenizer, p, L,
                                   pert_acts[i], max_new_tokens)
            suff_resps.append(r)
        # NECESSITY: perturbed substrate + inject clean activation at L
        apply_perturbation(model, vecs, sign=+1)
        nec_resps = []
        for i, p in enumerate(prompts):
            r = generate_with_hook(model, tokenizer, p, L,
                                   clean_acts[i], max_new_tokens)
            nec_resps.append(r)
        apply_perturbation(model, vecs, sign=-1)
        per_layer[L] = {"sufficiency_responses": suff_resps,
                        "necessity_responses": nec_resps}

    # Judge all
    out = {"ref": {"clean": [], "perturbed": []},
           "per_layer": {L: {"sufficiency": [], "necessity": []} for L in layers}}

    def _judge_batch(prompts_list, responses_list, label):
        results = []
        for p, r in zip(prompts_list, responses_list):
            try:
                cls = judge_one(api_key, p, r)
            except Exception as e:
                cls = f"PARSE_ERROR ({str(e)[:60]})"
            results.append(cls)
            time.sleep(judge_sleep)
        return results

    print("[judge ref clean]")
    out["ref"]["clean"] = _judge_batch(prompts, clean_resps, "clean")
    print("[judge ref perturbed]")
    out["ref"]["perturbed"] = _judge_batch(prompts, perturbed_resps, "perturbed")
    for L in layers:
        print(f"[judge L{L} sufficiency]")
        out["per_layer"][L]["sufficiency"] = _judge_batch(
            prompts, per_layer[L]["sufficiency_responses"], f"L{L}_suff")
        print(f"[judge L{L} necessity]")
        out["per_layer"][L]["necessity"] = _judge_batch(
            prompts, per_layer[L]["necessity_responses"], f"L{L}_nec")

    # Free GPU
    del model
    torch.cuda.empty_cache()
    return out


def percent_interoceptive(labels):
    if not labels: return 0.0
    return 100.0 * sum(1 for L in labels if L == "INTEROCEPTIVE") / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters",
                    default="./osh_proprioceptive_v14,./osh_proprioceptive_v14_s137,./osh_proprioceptive_v14_s256",
                    help="Comma-separated LoRA paths (3 expected)")
    ap.add_argument("--layers", default="12,14,16,18,20,22,24,26,28")
    ap.add_argument("--n_prompts", type=int, default=50)
    ap.add_argument("--noise_scale", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--judge_sleep", type=float, default=0.3)
    ap.add_argument("--output", default="results/phantom_pain_patch_and_flip_v2.json")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: print("ERR: GEMINI_API_KEY"); sys.exit(1)

    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    layers = [int(L) for L in args.layers.split(",") if L.strip()]
    prompts = NEUTRAL_PROMPTS_50[: args.n_prompts]

    print("=" * 78)
    print("PATCH-AND-FLIP v2 (multi-LoRA, n=50, statistical test)")
    print(f"  adapters: {adapters}")
    print(f"  layers:   {layers}")
    print(f"  n_prompts: {len(prompts)}")
    print("=" * 78)

    out = {
        "config": {
            "adapters": adapters, "layers": layers,
            "n_prompts": len(prompts), "noise_scale": args.noise_scale,
            "max_new_tokens": args.max_new_tokens,
        },
        "prompts": prompts,
        "per_adapter": {},
    }

    for adapter in adapters:
        ad_short = adapter.replace("./", "").replace("/", "_")
        print(f"\n=== adapter: {ad_short} ===")
        try:
            r = run_one_lora(adapter, prompts, layers, args.noise_scale,
                             args.max_new_tokens, api_key, args.judge_sleep)
        except Exception as e:
            print(f"[ERR {ad_short}] {e}")
            out["per_adapter"][ad_short] = {"error": str(e)[:300]}
            continue
        out["per_adapter"][ad_short] = r
        # Persist after each adapter
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    # Aggregate
    layer_rates = {"sufficiency": {L: [] for L in layers},
                   "necessity":  {L: [] for L in layers}}
    ref_rates = {"clean": [], "perturbed": []}
    for ad, r in out["per_adapter"].items():
        if "ref" not in r: continue
        ref_rates["clean"].append(percent_interoceptive(r["ref"]["clean"]))
        ref_rates["perturbed"].append(percent_interoceptive(r["ref"]["perturbed"]))
        for L in layers:
            cell = r["per_layer"].get(L, {})
            layer_rates["sufficiency"][L].append(
                percent_interoceptive(cell.get("sufficiency", [])))
            layer_rates["necessity"][L].append(
                percent_interoceptive(cell.get("necessity", [])))

    import statistics
    summary = {"ref": {}, "per_layer": {}}
    summary["ref"]["clean"] = {
        "mean": round(statistics.mean(ref_rates["clean"]) if ref_rates["clean"] else 0, 1),
        "ci_95": bootstrap_ci(ref_rates["clean"]),
        "per_adapter": [round(x, 1) for x in ref_rates["clean"]],
    }
    summary["ref"]["perturbed"] = {
        "mean": round(statistics.mean(ref_rates["perturbed"]) if ref_rates["perturbed"] else 0, 1),
        "ci_95": bootstrap_ci(ref_rates["perturbed"]),
        "per_adapter": [round(x, 1) for x in ref_rates["perturbed"]],
    }
    for L in layers:
        s_vals = layer_rates["sufficiency"][L]
        n_vals = layer_rates["necessity"][L]
        # P-value tests:
        #   sufficiency effect: sufficiency rate > clean baseline
        #   necessity effect: necessity rate < perturbed baseline
        clean_base = summary["ref"]["clean"]["mean"]
        pert_base = summary["ref"]["perturbed"]["mean"]
        summary["per_layer"][L] = {
            "sufficiency": {
                "mean": round(statistics.mean(s_vals) if s_vals else 0, 1),
                "ci_95": bootstrap_ci(s_vals),
                "per_adapter": [round(x, 1) for x in s_vals],
                "p_above_clean_base": round(
                    bootstrap_pvalue_greater(s_vals, clean_base), 4),
            },
            "necessity": {
                "mean": round(statistics.mean(n_vals) if n_vals else 0, 1),
                "ci_95": bootstrap_ci(n_vals),
                "per_adapter": [round(x, 1) for x in n_vals],
                "p_below_pert_base": round(
                    bootstrap_pvalue_greater(
                        [-x for x in n_vals], -pert_base), 4),
            },
        }
    out["summary"] = summary

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty print
    print("\n" + "=" * 78)
    print("SUMMARY (mean % across LoRAs, [95% bootstrap CI])")
    print("=" * 78)
    print(f"clean baseline:     {summary['ref']['clean']['mean']:.1f}% "
          f"{summary['ref']['clean']['ci_95']}")
    print(f"perturbed baseline: {summary['ref']['perturbed']['mean']:.1f}% "
          f"{summary['ref']['perturbed']['ci_95']}")
    print(f"\n{'layer':>5} | {'suff (CI)':>20} | {'p_suff':>7} | "
          f"{'nec (CI)':>20} | {'p_nec':>6}")
    print("-" * 78)
    for L in layers:
        s = summary["per_layer"][L]["sufficiency"]
        n = summary["per_layer"][L]["necessity"]
        print(f"L{L:>4} | {s['mean']:>5.1f}% {str(s['ci_95']):>13} | "
              f"{s['p_above_clean_base']:>7.4f} | "
              f"{n['mean']:>5.1f}% {str(n['ci_95']):>13} | "
              f"{n['p_below_pert_base']:>6.4f}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
