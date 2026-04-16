"""
OSH Multi-Seed Experiment Runner

Runs any experiment function N times with different random seeds,
collects results, and computes mean/std/95% CI with bootstrap.

Addresses Reviewer Weakness W2: No error bars or multiple runs.

Usage:
    from osh_multi_seed import run_multi_seed, format_stats

    def my_experiment(seed):
        # ... run experiment with this seed ...
        return {"accuracy": 0.85, "ppl": 18.2}

    results = run_multi_seed(my_experiment, seeds=[42, 137, 256, 512, 1024])
    print(format_stats(results))
"""

import json
import numpy as np
import os
from datetime import datetime
from osh_eval_config import DEFAULT_SEEDS, N_BOOTSTRAP, CI_LEVEL


def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL):
    """Compute bootstrap confidence interval for a list of values."""
    data = np.array(data, dtype=np.float64)
    n = len(data)
    if n < 2:
        return float(data[0]), float(data[0]), float(data[0])

    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1.0 - ci_level
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(np.mean(data)), float(hi)


def run_multi_seed(experiment_fn, seeds=None, experiment_name="experiment",
                   output_dir="./results", verbose=True):
    """
    Run an experiment function multiple times with different seeds.

    Args:
        experiment_fn: Callable(seed: int) -> dict of metric_name -> value
        seeds: List of integer seeds (default: DEFAULT_SEEDS)
        experiment_name: Name for output files
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        dict with structure:
        {
            "seeds": [42, 137, ...],
            "per_seed": {42: {"acc": 0.85}, 137: {"acc": 0.82}, ...},
            "aggregate": {
                "acc": {"mean": 0.835, "std": 0.021, "ci_lo": 0.81, "ci_hi": 0.86,
                         "values": [0.85, 0.82]}
            }
        }
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    os.makedirs(output_dir, exist_ok=True)

    per_seed = {}
    all_metrics = {}

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Run {i+1}/{len(seeds)} — seed={seed}")
            print(f"{'='*60}")

        result = experiment_fn(seed)
        per_seed[seed] = result

        # Collect metric values across seeds
        for key, val in result.items():
            if isinstance(val, (int, float)):
                all_metrics.setdefault(key, []).append(val)
            elif isinstance(val, dict):
                # Nested dict (e.g., per-category results)
                for subkey, subval in val.items():
                    if isinstance(subval, (int, float)):
                        compound_key = f"{key}/{subkey}"
                        all_metrics.setdefault(compound_key, []).append(subval)

    # Compute aggregate statistics
    aggregate = {}
    for key, values in all_metrics.items():
        values_arr = np.array(values, dtype=np.float64)
        ci_lo, mean, ci_hi = bootstrap_ci(values)
        aggregate[key] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr, ddof=1)) if len(values) > 1 else 0.0,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "values": [float(v) for v in values],
            "n": len(values),
        }

    results = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "n_runs": len(seeds),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "aggregate": aggregate,
    }

    # Save to JSON
    out_path = os.path.join(output_dir, f"{experiment_name}_multi_seed.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to {out_path}")

    return results


def format_stats(results, key=None):
    """
    Format aggregate statistics for display or paper inclusion.

    Args:
        results: Output from run_multi_seed()
        key: Specific metric key, or None for all

    Returns:
        Formatted string with mean +/- std [CI_lo, CI_hi]
    """
    agg = results["aggregate"]
    lines = []

    keys = [key] if key else sorted(agg.keys())
    for k in keys:
        if k not in agg:
            continue
        s = agg[k]
        lines.append(
            f"  {k}: {s['mean']:.3f} +/- {s['std']:.3f}  "
            f"[95% CI: {s['ci_lo']:.3f}, {s['ci_hi']:.3f}]  "
            f"(n={s['n']})"
        )

    header = f"Results for {results.get('experiment', '?')} ({results['n_runs']} seeds):"
    return "\n".join([header] + lines)


def format_for_latex(results, key, fmt="{:.1f}"):
    """
    Format a metric as 'mean ± std' for LaTeX tables.

    Args:
        results: Output from run_multi_seed()
        key: Metric key
        fmt: Format string for numbers

    Returns:
        String like "$59.4 \\pm 1.2$"
    """
    s = results["aggregate"][key]
    mean_str = fmt.format(s["mean"])
    std_str = fmt.format(s["std"])
    return f"${mean_str} \\pm {std_str}$"


if __name__ == "__main__":
    # Quick test with a dummy experiment
    import random

    def dummy_experiment(seed):
        random.seed(seed)
        np.random.seed(seed)
        return {
            "accuracy": 0.6 + random.gauss(0, 0.02),
            "ppl": 18.0 + random.gauss(0, 1.0),
        }

    results = run_multi_seed(
        dummy_experiment,
        seeds=[42, 137, 256],
        experiment_name="dummy_test",
        output_dir="./results"
    )
    print(format_stats(results))
    print(format_for_latex(results, "accuracy", fmt="{:.1f}"))
