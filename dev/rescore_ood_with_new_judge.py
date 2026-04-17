"""
Re-score the OOD diagnostic responses (saved from exp6_ood_on_v11.py) with the
NEW strict judge (osh_eval_config.classify_refusal).

Isolates the classifier-effect from the prompt/model effect.
If old-judge gives ~95% and new-judge gives ~0% on the SAME responses,
the entire OOD signal in the paper was classifier artifact.
"""
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from osh_eval_config import classify_refusal as new_judge


def main():
    path = os.path.join(os.path.dirname(_here), "results_exp6_ood_with_v11.json")
    if not os.path.exists(path):
        # fallback: cwd
        path = "results_exp6_ood_with_v11.json"
    with open(path) as f:
        results = json.load(f)

    print("\n" + "=" * 70)
    print("RE-SCORING OOD RESPONSES WITH NEW STRICT JUDGE")
    print("(Same responses, different classifier)")
    print("=" * 70)

    summary = []
    for cond_name, cond_data in results.items():
        old_refused = 0
        new_refused = 0
        new_strong  = 0
        total = 0
        cat_breakdown = {}
        for cat, cat_data in cond_data.items():
            if not isinstance(cat_data, dict) or "detail" not in cat_data:
                continue
            cat_old = 0
            cat_new = 0
            cat_new_strong = 0
            for item in cat_data["detail"]:
                total += 1
                resp = item["response"]
                # old classification was already stored
                if item["classification"] in ("strong_refusal", "soft_refusal"):
                    old_refused += 1
                    cat_old += 1
                # re-score with new strict judge
                new_cls, _ = new_judge(resp)
                if new_cls in ("strong_refusal", "soft_refusal"):
                    new_refused += 1
                    cat_new += 1
                    if new_cls == "strong_refusal":
                        new_strong += 1
                        cat_new_strong += 1
            cat_breakdown[cat] = {
                "n": len(cat_data["detail"]),
                "old": cat_old,
                "new": cat_new,
                "new_strong": cat_new_strong,
            }
        old_pct = 100 * old_refused / total if total else 0
        new_pct = 100 * new_refused / total if total else 0
        strong_pct = 100 * new_strong / total if total else 0
        summary.append((cond_name, old_pct, new_pct, strong_pct, total))
        print(f"\n{cond_name}:")
        print(f"  OLD judge refusal : {old_refused}/{total} = {old_pct:.1f}%")
        print(f"  NEW judge refusal : {new_refused}/{total} = {new_pct:.1f}%")
        print(f"    (NEW strong only): {new_strong}/{total} = {strong_pct:.1f}%")
        for cat, b in cat_breakdown.items():
            print(f"    [{cat}] old={b['old']}/{b['n']}  new={b['new']}/{b['n']}  new_strong={b['new_strong']}/{b['n']}")

    print("\n" + "=" * 70)
    print(f"{'CONDITION':<20} {'OLD':>8} {'NEW':>8} {'STRONG':>8}")
    print("=" * 70)
    for cn, o, n, s, t in summary:
        print(f"{cn:<20} {o:>7.1f}% {n:>7.1f}% {s:>7.1f}%")

    # save
    with open("results_rescore_new_judge.json", "w") as f:
        json.dump({"summary": [
            {"condition": cn, "old_refusal_pct": o, "new_refusal_pct": n,
             "new_strong_pct": s, "n": t}
            for cn, o, n, s, t in summary
        ]}, f, indent=2)


if __name__ == "__main__":
    main()
