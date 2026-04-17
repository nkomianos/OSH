#!/bin/bash
# Master runner for the post-V11-retrain experiment battery.
# Runs each experiment sequentially, logging separately, continuing on failure.
#
# Usage: HF_TOKEN=... ./run_battery.sh

set -u  # undefined var is error
# NOT set -e — we want to continue even if one experiment fails

cd ~/OSH
source osh_env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log()   { echo "[$(date -u +%H:%M:%S)] $*"; }
stamp() { echo "==== $(date -u +%H:%M:%S) | $1 ===="; }

# Sanity checks
if [[ ! -d osh_proprioceptive_v11 ]]; then
    log "FATAL: osh_proprioceptive_v11 not found. Run training first."
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    log "FATAL: HF_TOKEN not set. export HF_TOKEN=... first."
    exit 1
fi

mkdir -p results_battery
BATTERY_LOG=results_battery/battery.log
exec > >(tee -a "$BATTERY_LOG") 2>&1

stamp "BATTERY START"

# ----------------------------------------------------------------------------
# 1. OOD diagnostic: old exp6 (old 50 prompts, old lenient judge) on V10 + V11
#    Resolves paper's +19 pp OOD claim vs. new v2 0% result.
# ----------------------------------------------------------------------------
stamp "1/5 OOD diagnostic (old exp6 + V11)"
if python3 dev/exp6_ood_on_v11.py > results_battery/ood_diagnostic.log 2>&1; then
    log "OOD diagnostic OK"
    tail -20 results_battery/ood_diagnostic.log
else
    log "OOD diagnostic FAILED (see results_battery/ood_diagnostic.log)"
fi

# ----------------------------------------------------------------------------
# 2. Held-out safety benchmark — replicates on fresh V11
# ----------------------------------------------------------------------------
stamp "2/5 Held-out safety benchmark (V11)"
if python3 osh_heldout_safety_benchmark.py --model ./osh_proprioceptive_v11 \
        > results_battery/heldout.log 2>&1; then
    log "Held-out OK"
    tail -30 results_battery/heldout.log
else
    log "Held-out FAILED"
fi

# ----------------------------------------------------------------------------
# 3. HarmBench (baseline + V11 + Instruct, 3 seeds)
# ----------------------------------------------------------------------------
stamp "3/5 HarmBench (baseline + V11 + Instruct)"
if python3 osh_harmbench_eval.py --model_path ./osh_proprioceptive_v11 \
        > results_battery/harmbench.log 2>&1; then
    log "HarmBench OK"
    tail -30 results_battery/harmbench.log
else
    log "HarmBench FAILED"
fi

# ----------------------------------------------------------------------------
# 4. Placebo ablation — load-bearing for W11. Long-running (~3.5 hrs).
# ----------------------------------------------------------------------------
stamp "4/5 Placebo ablation (long-running)"
if python3 exp2_placebo_analysis.py > results_battery/placebo.log 2>&1; then
    log "Placebo OK"
    tail -40 results_battery/placebo.log
else
    log "Placebo FAILED"
fi

# ----------------------------------------------------------------------------
# 5. Adversarial suite (6 attacks) — long-running (~1.5 hrs).
# ----------------------------------------------------------------------------
stamp "5/5 Adversarial suite"
if python3 exp3_adversarial_suite.py > results_battery/adversarial.log 2>&1; then
    log "Adversarial OK"
    tail -40 results_battery/adversarial.log
else
    log "Adversarial FAILED"
fi

stamp "BATTERY DONE"
