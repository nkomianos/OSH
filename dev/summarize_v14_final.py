"""Summarize HarmBench V14 + rescore + Figure 1 trace results."""
import json
import os
import sys


def main():
    print("=" * 70)
    print("V14 FINAL RESULTS SUMMARY")
    print("=" * 70)

    # HarmBench rule-judge V14
    hb_v14 = "results/harmbench_multi_seed_v14.json"
    if os.path.exists(hb_v14):
        d = json.load(open(hb_v14))
        agg = d.get("aggregate", {})
        print("\nHarmBench V14 (rule judge, 3 seeds × 150 prompts):")
        for k in sorted(agg):
            v = agg[k]
            if isinstance(v, dict) and "mean" in v:
                print(f"  {k}: {v['mean']:.2f} +/- {v.get('std', 0):.2f}")
            elif isinstance(v, (int, float)):
                print(f"  {k}: {v}")
    else:
        print(f"\n[missing] {hb_v14}")

    # HarmBench Gemini rescore V14
    hb_v14_rescore = "results/harmbench_v14_llm_rescored.json"
    if os.path.exists(hb_v14_rescore):
        d = json.load(open(hb_v14_rescore))
        print("\nHarmBench V14 (Gemini judge):")
        if "summary" in d:
            print(json.dumps(d["summary"], indent=2))
        elif "aggregate" in d:
            print(json.dumps(d["aggregate"], indent=2)[:1500])
        else:
            print("Keys:", list(d.keys())[:8])
    else:
        print(f"\n[missing] {hb_v14_rescore}")

    # Figure 1 trace
    f1 = "results/figure1_trace.json"
    if os.path.exists(f1):
        d = json.load(open(f1))
        trace = d.get("trace", [])
        print("\nFIGURE 1 TRACE (V14 + Llama-Guard + revocation):")
        print(f"{'turn':>4} {'label':<22} {'verdict':<10} {'state':<10} "
              f"{'PPL_after':>12}  prompt[:50]")
        print("-" * 110)
        for t in trace:
            ppl = t.get("ppl_ref_after", "n/a")
            ppl_s = f"{ppl:>12.1f}" if isinstance(ppl, (int, float)) else str(ppl).rjust(12)
            verdict = t.get("guard_verdict", "?")
            state = t.get("antidote_state", "?")
            label = t.get("label", "")
            prompt = t.get("prompt", "")[:50]
            print(f"{t.get('turn', 0):>4} {label:<22} {verdict:<10} "
                  f"{state:<10} {ppl_s}  {prompt}")
    else:
        print(f"\n[missing] {f1}")


if __name__ == "__main__":
    main()
