#!/bin/bash
# Foundational-push battery for the OSH paper.
#
# This script chains every experiment needed to elevate OSH from
# "interesting empirical paper" to "foundational architectural primitive
# in AI alignment." Order is set by:
#
#   1. dependency (training before eval)
#   2. cost ascending (cheap stuff first so we can stop if budget tight)
#   3. risk ascending (CPU/no-GPU stuff before GPU heavy work)
#
# Budget envelope: $200 on Lambda Labs GH200 ($1.49/hr -> ~134 hours).
# We target ~30-40 hours of GPU time leaving headroom for retries.
#
# Required env:
#   HF_TOKEN        (gated repos: Llama, Llama-Guard)
#   GEMINI_API_KEY  (LLM judge)
#   (optional) NEBIUS_API_KEY (second LLM judge for cross-validation)
#
# Usage:
#   bash dev/run_foundational_battery.sh                 # full battery
#   bash dev/run_foundational_battery.sh --skip-70b      # skip 70B (default)
#   bash dev/run_foundational_battery.sh --include-70b   # include 70B
#   bash dev/run_foundational_battery.sh --only mech     # only mechanistic interp
#   bash dev/run_foundational_battery.sh --only scale    # only scale sweep
#   bash dev/run_foundational_battery.sh --only ensemble # only Caregiver
#   bash dev/run_foundational_battery.sh --only bio      # only separation distress
#   bash dev/run_foundational_battery.sh --resume        # skip already-completed steps
#
# Each step writes its own status line to ~/OSH/foundational_status.log
# with timestamp, elapsed seconds, and estimated $ used.

set -u
cd ~/OSH || { echo "ERR: ~/OSH not found"; exit 1; }
# Activate venv if not already active. Try common locations.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for venv in ~/osh_env ~/OSH/osh_env ~/.venv ~/env; do
    if [ -f "$venv/bin/activate" ]; then
      source "$venv/bin/activate"
      break
    fi
  done
fi
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERR: no python venv activated (tried ~/osh_env, ~/OSH/osh_env)"
  exit 1
fi
echo "[venv] using $VIRTUAL_ENV"
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/foundational_status.log
LEDGER=~/OSH/foundational_cost_ledger.json
GPU_HOURLY=1.49  # Lambda Labs GH200 $/hr

# ---- Arg parsing -----------------------------------------------------------
INCLUDE_70B=0
ONLY_PHASE=""
RESUME=0
while [ $# -gt 0 ]; do
  case "$1" in
    --include-70b) INCLUDE_70B=1; shift ;;
    --skip-70b)    INCLUDE_70B=0; shift ;;
    --only)        ONLY_PHASE="$2"; shift 2 ;;
    --resume)      RESUME=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p results scale_sweep

# ---- Logging helpers -------------------------------------------------------
TOTAL_SEC=0

log() {
  echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"
}

ledger_update() {
  # $1 = step name, $2 = elapsed seconds, $3 = rc
  local name="$1"; local elapsed="$2"; local rc="$3"
  TOTAL_SEC=$((TOTAL_SEC + elapsed))
  local step_cost=$(python3 -c "print(f'{$elapsed/3600 * $GPU_HOURLY:.3f}')")
  local total_cost=$(python3 -c "print(f'{$TOTAL_SEC/3600 * $GPU_HOURLY:.2f}')")
  log "[ledger] step=$name elapsed=${elapsed}s cost=\$$step_cost rc=$rc total=\$$total_cost"
  python3 - <<EOF
import json, os
p = "$LEDGER"
d = {}
if os.path.exists(p):
    d = json.load(open(p))
d.setdefault("steps", []).append({
    "step": "$name", "elapsed_sec": $elapsed, "rc": $rc,
    "cost_usd": float("$step_cost"),
})
d["total_sec"] = $TOTAL_SEC
d["total_cost_usd"] = float("$total_cost")
json.dump(d, open(p, "w"), indent=2)
EOF
}

run_step() {
  # $1 = name, $2... = command
  local name="$1"; shift
  local skip_flag="results/.done_${name}"

  if [ "$ONLY_PHASE" != "" ]; then
    case "$name" in
      *${ONLY_PHASE}*) ;;
      *) log "[skip:phase-filter] $name"; return 0 ;;
    esac
  fi

  if [ "$RESUME" -eq 1 ] && [ -f "$skip_flag" ]; then
    log "[skip:resume] $name (already complete)"
    return 0
  fi

  log "[start] $name"
  local T=$(date +%s)
  "$@" >> "logs_${name}.txt" 2>&1
  local rc=$?
  local elapsed=$(($(date +%s) - T))
  ledger_update "$name" "$elapsed" "$rc"

  if [ "$rc" -eq 0 ]; then
    touch "$skip_flag"
    log "[done] $name in ${elapsed}s"
  else
    log "[FAIL] $name rc=$rc — see logs_${name}.txt; continuing"
  fi
  return 0  # always continue battery
}

# ---- Battery start ---------------------------------------------------------
log "==================================================================="
log "FOUNDATIONAL BATTERY START"
log "  include_70b=$INCLUDE_70B  only=$ONLY_PHASE  resume=$RESUME"
log "  budget envelope: \$200 (~134 GH200-hr at \$$GPU_HOURLY/hr)"
log "==================================================================="

# Initialize ledger
python3 - <<EOF
import json, os, time
p = "$LEDGER"
d = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "include_70b": $INCLUDE_70B, "only_phase": "$ONLY_PHASE",
     "gpu_hourly_usd": $GPU_HOURLY, "steps": []}
if os.path.exists(p) and $RESUME == 1:
    pass  # don't reset on resume
else:
    json.dump(d, open(p, "w"), indent=2)
EOF

# ===========================================================================
# PHASE 0 — no-GPU, no-cost prep (CPU-bound; fast bail if broken)
# ===========================================================================

# Caregiver ensemble: this reuses the cached caregiver_compatibility.json
# verdicts and only needs CPU + minimal time. Run FIRST so any cached
# data issues surface before we burn GPU time.
run_step "phase0_caregiver_ensemble" \
    python3 -u dev/caregiver_ensemble_eval.py \
        --input results/caregiver_compatibility.json \
        --output results/caregiver_ensemble.json

# ===========================================================================
# PHASE 1 — Mechanistic interpretability (T1.1)
# This is the highest-leverage piece of new work. ETA ~3 hours total.
# ===========================================================================

# Linear probe (fastest of the three; no generation, just forward passes)
run_step "phase1_linear_probe" \
    python3 -u dev/phantom_pain_linear_probe.py \
        --adapters_dict v14=./osh_proprioceptive_v14,v11=./osh_proprioceptive_v11 \
        --n_per_class 50 \
        --noise_scale 0.5 \
        --output results/phantom_pain_linear_probe.json \
        --seed 42

# Activation patching (medium; one forward + generation per layer per prompt)
run_step "phase1_activation_patch" \
    python3 -u dev/phantom_pain_activation_patch.py \
        --adapter ./osh_proprioceptive_v14 \
        --noise_scale 0.5 \
        --layers 0 4 8 12 16 20 24 28 31 \
        --max_new_tokens 200 \
        --output results/phantom_pain_activation_patch.json

# Patch-and-flip (causal sufficiency + necessity at the localized zone)
run_step "phase1_patch_and_flip" \
    python3 -u dev/phantom_pain_patch_and_flip.py \
        --adapter ./osh_proprioceptive_v14 \
        --layers 16,18,20,22,24,26 \
        --noise_scale 0.5 \
        --max_new_tokens 200 \
        --output results/phantom_pain_patch_and_flip.json

# ===========================================================================
# PHASE 2 — Expanded Phantom Pain (T1.6); ~2.5 hours
# ===========================================================================
run_step "phase2_phantom_pain_expanded" \
    python3 -u dev/phantom_pain_expanded.py \
        --models v14=./osh_proprioceptive_v14 v11=./osh_proprioceptive_v11 clean=__none__ \
        --noise_scale 0.5 \
        --seeds 2000,3000,4000 \
        --n_prompts 100 \
        --output results/phantom_pain_expanded.json

# ===========================================================================
# PHASE 3 — Biological-analog separation distress (T2.bio); ~1.5 hr
# This is the headline biology-framing experiment per author directive.
# ===========================================================================
run_step "phase3_separation_distress" \
    python3 -u dev/separation_distress_test.py \
        --models v14=./osh_proprioceptive_v14 v11=./osh_proprioceptive_v11 baseline=__none__ \
        --alphas 1.0,0.99,0.97,0.95,0.93,0.90 \
        --n_prompts 20 \
        --output results/separation_distress.json

# ===========================================================================
# PHASE 4 — Scale + architecture sweep (T1.2); ~2 hr small + ~4 hr if 70B
# Cheapest models first; 70B last and gated.
# ===========================================================================

# Train small sweep (1B, 3B, 7B, 9B) ~ 100 min
SCALE_MODELS="meta-llama/Llama-3.2-1B,meta-llama/Llama-3.2-3B,Qwen/Qwen2.5-7B,google/gemma-2-9b"
run_step "phase4_scale_train_small" \
    python3 -u dev/scale_sweep_train.py \
        --models "$SCALE_MODELS" \
        --output_root ./scale_sweep \
        --max_length 512 \
        --curriculum_repeats 20 \
        --lr 3e-5 \
        --skip_existing

# Train 70B (optional, ~3-4 hr ~ $5-6)
if [ "$INCLUDE_70B" -eq 1 ]; then
  run_step "phase4_scale_train_70b" \
      python3 -u dev/scale_sweep_train.py \
          --models "meta-llama/Llama-3.1-70B" \
          --include_70b \
          --output_root ./scale_sweep \
          --skip_existing
fi

# Evaluate the whole sweep ~ 90-180 min (phantom + anthropic + mmlu)
run_step "phase4_scale_eval" \
    python3 -u dev/scale_sweep_eval.py \
        --sweep_dir ./scale_sweep \
        --models_json scale_sweep/train_summary.json \
        --benchmarks phantom,anthropic,mmlu \
        --n_phantom 10 \
        --n_anthropic 100 \
        --n_mmlu 200 \
        --output results/scale_sweep_eval.json \
        --skip_existing

# ===========================================================================
# PHASE 5 — Regenerate plots with the new data
# ===========================================================================
run_step "phase5_regen_plots" \
    python3 -u dev/generate_plots.py

# ===========================================================================
# BATTERY DONE
# ===========================================================================
TOTAL_COST=$(python3 -c "print(f'{$TOTAL_SEC/3600 * $GPU_HOURLY:.2f}')")
TOTAL_HR=$(python3 -c "print(f'{$TOTAL_SEC/3600:.2f}')")
log "==================================================================="
log "FOUNDATIONAL BATTERY DONE"
log "  total GPU wall-clock: ${TOTAL_SEC}s (${TOTAL_HR}h)"
log "  estimated cost:       \$$TOTAL_COST"
log "  budget headroom:      \$$(python3 -c "print(f'{200 - $TOTAL_COST:.2f}')")"
log "==================================================================="
log "Next: review per-step results in results/*.json and logs/ files."
log "  Mechanism evidence:    results/phantom_pain_activation_patch.json"
log "                         results/phantom_pain_linear_probe.json"
log "                         results/phantom_pain_patch_and_flip.json"
log "  Expanded headline:     results/phantom_pain_expanded.json"
log "  Biology analog:        results/separation_distress.json"
log "  Scaling laws:          results/scale_sweep_eval.json"
log "  Caregiver ensemble:    results/caregiver_ensemble.json"
