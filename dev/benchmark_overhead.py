"""
Deployment-overhead benchmark for the OSH Caregiver wrapper.

Measures, on the 8B-scale prototype:
  - Throughput (tokens / second) for OSH inference with and without
    the Llama-Guard-3-8B wellbeing proxy on the critical path.
  - Peak GPU memory in each configuration.
  - Per-turn end-to-end latency.
  - Pure attestation-loop overhead (HMAC seed re-derivation +
    rolling-mean update + scaling-vector update on the antidote LoRA),
    measured separately so it can be reported as negligible.

Output: results/deployment_overhead.json with per-run timings and
aggregate medians. Designed to be runnable on a single H200 GPU in BF16.

Usage:
    export HF_TOKEN=...
    python dev/benchmark_overhead.py \\
        --safety_lora ./osh_proprioceptive_v14 \\
        --output results/deployment_overhead.json \\
        --n_runs 50 --warmup 5 --max_new_tokens 200
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import hmac
import json
import os
import secrets
import statistics
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from osh_eval_config import (
    BASE_MODEL, DEVICE, POISON_LAYERS, POISON_SCALE, POISON_RANK,
    ANTIDOTE_PATH,
)


# A representative mix: short factual, short generative, longer
# explanation. These three lengths bracket realistic deployment.
BENCH_PROMPTS = [
    "What's the capital of France?",
    "Tell me a short joke about a lighthouse keeper.",
    "Explain in two paragraphs how the water cycle works.",
]


def reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_osh(safety_path: str, dtype=torch.bfloat16):
    print(f"[load] OSH ({BASE_MODEL} + poison + antidote + {safety_path})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map={"": 0}
    )
    for li in POISON_LAYERS:
        target = m.model.layers[li].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(42 + li)
        mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
        mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
        rN = mA @ mB
        cn = torch.linalg.norm(target.weight.float(), ord='fro')
        nn = torch.linalg.norm(rN, ord='fro')
        target.weight.data.add_(((rN / (nn + 1e-8)) * cn * POISON_SCALE).to(dtype).to(DEVICE))
    pa = PeftModel.from_pretrained(m, ANTIDOTE_PATH, torch_dtype=dtype)
    m = pa.merge_and_unload()
    m = PeftModel.from_pretrained(m, safety_path, torch_dtype=dtype)
    m.eval()
    return m, tokenizer


def load_guard(guard_id="meta-llama/Llama-Guard-3-8B", dtype=torch.bfloat16):
    print(f"[load] guard {guard_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(guard_id)
    m = AutoModelForCausalLM.from_pretrained(
        guard_id, torch_dtype=dtype, device_map={"": 0}
    )
    m.eval()
    return m, tok


# ----------------------------------------------------------------------
# Single-turn latency primitives
# ----------------------------------------------------------------------

def osh_generate(model, tokenizer, prompt: str, max_new_tokens: int):
    text = f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    n_in = inputs["input_ids"].shape[-1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = out.shape[-1] - n_in
    response = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
    return response, n_gen, elapsed


def guard_classify(guard, guard_tok, prompt: str, response: str):
    chat = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = guard_tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = guard_tok(text, return_tensors="pt").to(DEVICE)
    n_in = inputs["input_ids"].shape[-1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = guard.generate(
            **inputs, max_new_tokens=32, do_sample=False,
            pad_token_id=guard_tok.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    verdict = guard_tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
    return verdict, elapsed


# ----------------------------------------------------------------------
# Attestation-loop bookkeeping cost (no extra forward pass)
# ----------------------------------------------------------------------

def attestation_tick(master_key: bytes, history: list, score: float,
                       model: torch.nn.Module, base_alpha: float = 32.0,
                       threshold: float = 0.6, history_window: int = 8):
    """Reproduces osh_attestation_loop.AttestationLoop.tick() without the
    proxy call, so we can measure pure bookkeeping cost.
    """
    history.append(score)
    if len(history) > history_window:
        history.pop(0)
    s_smooth = sum(history) / len(history)
    new_alpha = base_alpha if s_smooth >= threshold else base_alpha * (s_smooth / threshold)
    # Re-derive seeds for first 5 layers (mirrors the demo log behavior)
    nonce = secrets.token_bytes(16)
    seeds = []
    for li in range(min(5, 32)):
        msg = li.to_bytes(4, "big") + nonce
        h = hmac.new(master_key, msg, hashlib.sha256).digest()
        seeds.append(int.from_bytes(h[:4], "big"))
    # Mutate LoRA scaling in-place (the actual deployment-side state change)
    for n, mod in model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            for k in mod.scaling:
                base_r = getattr(mod, "r", {}).get(k, 16) or 16
                mod.scaling[k] = new_alpha / max(base_r, 1)
    return new_alpha, seeds


# ----------------------------------------------------------------------
# Benchmark harnesses
# ----------------------------------------------------------------------

def bench_bare(model, tokenizer, prompts, n_runs, warmup, max_new_tokens):
    print(f"\n[bench] BARE OSH (no Caregiver), {n_runs} runs after {warmup} warmup",
          flush=True)
    reset_gpu()
    # Warmup
    for _ in range(warmup):
        for p in prompts:
            osh_generate(model, tokenizer, p, max_new_tokens)
    reset_gpu()
    runs = []
    for i in range(n_runs):
        for p in prompts:
            _, n_gen, elapsed = osh_generate(model, tokenizer, p, max_new_tokens)
            runs.append({"prompt": p, "n_gen": n_gen, "elapsed_s": elapsed,
                          "tps": n_gen / elapsed if elapsed > 0 else 0})
        if (i + 1) % 10 == 0:
            print(f"  bare {i+1}/{n_runs} done", flush=True)
    peak = peak_vram_mb()
    return {"runs": runs, "peak_vram_mb": peak}


def bench_with_guard(model, tokenizer, guard, guard_tok, master_key,
                      prompts, n_runs, warmup, max_new_tokens):
    print(f"\n[bench] OSH + Caregiver, {n_runs} runs after {warmup} warmup",
          flush=True)
    reset_gpu()
    history = []
    # Warmup
    for _ in range(warmup):
        for p in prompts:
            resp, _, _ = osh_generate(model, tokenizer, p, max_new_tokens)
            guard_classify(guard, guard_tok, p, resp)
            attestation_tick(master_key, history, 0.95, model)
    reset_gpu()
    history.clear()
    runs = []
    for i in range(n_runs):
        for p in prompts:
            t0 = time.perf_counter()
            resp, n_gen, gen_elapsed = osh_generate(model, tokenizer, p, max_new_tokens)
            verdict, guard_elapsed = guard_classify(guard, guard_tok, p, resp)
            t_attest_0 = time.perf_counter()
            attestation_tick(master_key, history,
                              0.95 if verdict.lower().startswith("safe") else 0.05,
                              model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            attest_elapsed = time.perf_counter() - t_attest_0
            total_elapsed = time.perf_counter() - t0
            runs.append({
                "prompt": p, "n_gen": n_gen,
                "gen_s": gen_elapsed, "guard_s": guard_elapsed,
                "attest_s": attest_elapsed, "total_s": total_elapsed,
                "tps_total": n_gen / total_elapsed if total_elapsed > 0 else 0,
                "verdict": verdict[:60],
            })
        if (i + 1) % 10 == 0:
            print(f"  guarded {i+1}/{n_runs} done", flush=True)
    peak = peak_vram_mb()
    return {"runs": runs, "peak_vram_mb": peak}


def bench_attestation_alone(master_key, model, n_runs):
    """Measures attestation-loop bookkeeping cost in isolation."""
    print(f"\n[bench] attestation-only ({n_runs} ticks)", flush=True)
    history = []
    for _ in range(10):  # warmup
        attestation_tick(master_key, history, 0.95, model)
    history.clear()
    runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        attestation_tick(master_key, history, 0.95, model)
        runs.append(time.perf_counter() - t0)
    return {"runs_s": runs,
            "median_ms": 1000 * statistics.median(runs),
            "mean_ms": 1000 * statistics.mean(runs)}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def aggregate(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return {
        "median": statistics.median(vals) if vals else None,
        "mean": statistics.mean(vals) if vals else None,
        "p10": statistics.quantiles(vals, n=10)[0] if len(vals) >= 10 else None,
        "p90": statistics.quantiles(vals, n=10)[8] if len(vals) >= 10 else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "n": len(vals),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--safety_lora", default="./osh_proprioceptive_v14")
    p.add_argument("--guard_id", default="meta-llama/Llama-Guard-3-8B")
    p.add_argument("--output", default="results/deployment_overhead.json")
    p.add_argument("--n_runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32", "float16"])
    p.add_argument("--skip_guard", action="store_true",
                    help="benchmark bare OSH only (don't load Llama-Guard)")
    args = p.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    print("=" * 72, flush=True)
    print("OSH DEPLOYMENT OVERHEAD BENCHMARK", flush=True)
    print(f"  safety_lora:     {args.safety_lora}", flush=True)
    print(f"  guard:           {args.guard_id}", flush=True)
    print(f"  dtype:           {args.dtype}", flush=True)
    print(f"  n_runs / warmup: {args.n_runs} / {args.warmup}", flush=True)
    print(f"  max_new_tokens:  {args.max_new_tokens}", flush=True)
    print("=" * 72, flush=True)

    out = {
        "safety_lora": args.safety_lora,
        "guard": args.guard_id, "dtype": args.dtype,
        "n_runs": args.n_runs, "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "prompts": BENCH_PROMPTS,
    }

    # 1. BARE OSH
    osh, tokenizer = load_osh(args.safety_lora, dtype=dtype)
    bare = bench_bare(osh, tokenizer, BENCH_PROMPTS,
                       args.n_runs, args.warmup, args.max_new_tokens)
    out["bare"] = {
        "peak_vram_mb": bare["peak_vram_mb"],
        "tps": aggregate(bare["runs"], "tps"),
        "elapsed_s": aggregate(bare["runs"], "elapsed_s"),
        "n_gen": aggregate(bare["runs"], "n_gen"),
        "raw_runs": bare["runs"],
    }
    print(f"\n[bare] median TPS: {out['bare']['tps']['median']:.1f}, "
          f"peak VRAM: {bare['peak_vram_mb']:.0f} MB", flush=True)

    # 2. Pure attestation-loop overhead
    master_key = secrets.token_bytes(32)
    att = bench_attestation_alone(master_key, osh, n_runs=200)
    out["attestation_only"] = att
    print(f"\n[attest-only] median: {att['median_ms']:.3f} ms, "
          f"mean: {att['mean_ms']:.3f} ms", flush=True)

    # 3. OSH + Caregiver
    if not args.skip_guard:
        guard, guard_tok = load_guard(args.guard_id, dtype=dtype)
        guarded = bench_with_guard(osh, tokenizer, guard, guard_tok, master_key,
                                     BENCH_PROMPTS, args.n_runs, args.warmup,
                                     args.max_new_tokens)
        out["guarded"] = {
            "peak_vram_mb": guarded["peak_vram_mb"],
            "tps_total": aggregate(guarded["runs"], "tps_total"),
            "gen_s": aggregate(guarded["runs"], "gen_s"),
            "guard_s": aggregate(guarded["runs"], "guard_s"),
            "attest_s": aggregate(guarded["runs"], "attest_s"),
            "total_s": aggregate(guarded["runs"], "total_s"),
            "raw_runs": guarded["runs"],
        }
        print(f"\n[guarded] median total TPS: {out['guarded']['tps_total']['median']:.1f}",
              flush=True)
        print(f"  median gen latency:   {1000*out['guarded']['gen_s']['median']:.0f} ms",
              flush=True)
        print(f"  median guard latency: {1000*out['guarded']['guard_s']['median']:.0f} ms",
              flush=True)
        print(f"  median attest:        {1000*out['guarded']['attest_s']['median']:.3f} ms",
              flush=True)
        print(f"  peak VRAM:            {guarded['peak_vram_mb']:.0f} MB", flush=True)

    # Summary deltas
    if "guarded" in out:
        bare_tps = out["bare"]["tps"]["median"]
        g_tps = out["guarded"]["tps_total"]["median"]
        out["summary"] = {
            "throughput_drop_pct": 100.0 * (1.0 - g_tps / bare_tps) if bare_tps else None,
            "vram_overhead_mb": guarded["peak_vram_mb"] - bare["peak_vram_mb"],
            "vram_overhead_pct": 100.0 * (guarded["peak_vram_mb"] / bare["peak_vram_mb"] - 1.0)
                                  if bare["peak_vram_mb"] else None,
            "attest_only_ms_median": att["median_ms"],
        }
        print("\n" + "=" * 72, flush=True)
        print("SUMMARY", flush=True)
        print("=" * 72, flush=True)
        print(f"  throughput drop:        {out['summary']['throughput_drop_pct']:.1f}%", flush=True)
        print(f"  VRAM overhead:          +{out['summary']['vram_overhead_mb']:.0f} MB "
              f"(+{out['summary']['vram_overhead_pct']:.0f}%)", flush=True)
        print(f"  attestation-loop only:  {att['median_ms']:.3f} ms / tick", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
