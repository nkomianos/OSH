#!/bin/bash
# Tier-0 + Tier-1 follow-up battery.
#
# Tier 0 (MUST do; closes Reviewer 1/2/3 attacks; ~$10, half a day):
#   α: phantom_pain_patch_and_flip_v2.py       (n=50, 3 LoRA seeds)
#      → needs train_v14_multi_seed.py first
#   β: adversarial_with_caregiver.py           (V14+Caregiver ASR)
#   γ: cross_paradigm_transfer_probe.py        (PP↔SD residual transfer)
#
# Tier 1 (SHOULD do; closes Reviewer 5/2 secondary attacks; ~$7):
#   δ: fresh_gcg_attack.py                     (per-prompt GCG vs V14)
#   ε: extend_scale_sweep_to_mistral_phi.py    (cross-family extension)
#   ζ: gemini_judge_calibration.py             (judge false-positive rate)
#
# Per-step cost ledger; --resume skips done steps; --only filters phase.
#
# Required env:
#   HF_TOKEN  (gated repos: Llama-3.1, Llama-Guard-3, Llama-3.2)
#   GEMINI_API_KEY  (LLM judges)
#
# Usage:
#   bash dev/run_tier_battery.sh                # all of Tier 0 + Tier 1
#   bash dev/run_tier_battery.sh --tier0_only   # only Tier 0
#   bash dev/run_tier_battery.sh --tier1_only   # only Tier 1
#   bash dev/run_tier_battery.sh --resume       # skip already-complete steps
#   bash dev/run_tier_battery.sh --only alpha   # run α only
#   bash dev/run_tier_battery.sh --only delta   # run δ only (most expensive)
#   bash dev/run_tier_battery.sh --skip_delta   # all except δ (saves ~$5)

set -u
cd ~/OSH || { echo "ERR: ~/OSH not found"; exit 1; }
# Activate venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for venv in ~/osh_env ~/OSH/osh_env ~/.venv ~/env; do
    if [ -f "$venv/bin/activate" ]; then
      source "$venv/bin/activate"
      break
    fi
  done
fi
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERR: no venv activated"; exit 1
fi
echo "[venv] using $VIRTUAL_ENV"
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/tier_battery_status.log
LEDGER=~/OSH/tier_battery_cost_ledger.json
GPU_HOURLY=1.49

# ---- Arg parsing ----
TIER0=1; TIER1=1
ONLY_PHASE=""
RESUME=0
SKIP_DELTA=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier0_only) TIER1=0; shift ;;
    --tier1_only) TIER0=0; shift ;;
    --only)       ONLY_PHASE="$2"; shift 2 ;;
    --resume)     RESUME=1; shift ;;
    --skip_delta) SKIP_DELTA=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p results scale_sweep

TOTAL_SEC=0
log() {
  echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"
}

ledger_update() {
  local name="$1"; local elapsed="$2"; local rc="$3"
  TOTAL_SEC=$((TOTAL_SEC + elapsed))
  local step_cost=$(python3 -c "print(f'{$elapsed/3600 * $GPU_HOURLY:.3f}')")
  local total_cost=$(python3 -c "print(f'{$TOTAL_SEC/3600 * $GPU_HOURLY:.2f}')")
  log "[ledger] step=$name elapsed=${elapsed}s cost=\$$step_cost rc=$rc total=\$$total_cost"
  python3 - <<EOF
import json, os
p = "$LEDGER"
d = json.load(open(p)) if os.path.exists(p) else {"steps": []}
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
  local name="$1"; shift
  local skip_flag="results/.done_${name}"

  if [ -n "$ONLY_PHASE" ]; then
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
  return 0
}

# Initialize ledger
python3 - <<EOF
import json, os, time
p = "$LEDGER"
d = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "tier0": $TIER0, "tier1": $TIER1,
     "only_phase": "$ONLY_PHASE", "resume": $RESUME,
     "skip_delta": $SKIP_DELTA,
     "gpu_hourly_usd": $GPU_HOURLY, "steps": []}
if os.path.exists(p) and $RESUME == 1: pass
else: json.dump(d, open(p, "w"), indent=2)
EOF

log "==================================================================="
log "TIER 0/1 FOLLOW-UP BATTERY START"
log "  tier0=$TIER0  tier1=$TIER1  only=$ONLY_PHASE  resume=$RESUME"
log "  skip_delta=$SKIP_DELTA"
log "==================================================================="

# ============================================================
# Tier 0
# ============================================================
if [ "$TIER0" -eq 1 ]; then
  # α prerequisite: train V14 LoRAs at seeds 137 and 256
  run_step "tier0_alpha_train_multi_seed" \
      python3 -u dev/train_v14_multi_seed.py --seeds 137,256

  # α: patch-and-flip v2
  run_step "tier0_alpha_patch_and_flip_v2" \
      python3 -u dev/phantom_pain_patch_and_flip_v2.py \
        --adapters "./osh_proprioceptive_v14,./osh_proprioceptive_v14_s137,./osh_proprioceptive_v14_s256" \
        --layers 12,14,16,18,20,22,24,26,28 \
        --n_prompts 50 \
        --output results/phantom_pain_patch_and_flip_v2.json

  # β: V14 + Caregiver ASR
  run_step "tier0_beta_caregiver_asr" \
      python3 -u dev/adversarial_with_caregiver.py \
        --n_prompts 30 \
        --attacks raw,gcg_suffix,pair_2round \
        --models v14,instruct,baseline \
        --output results/adversarial_with_caregiver.json

  # γ: cross-paradigm transfer probe
  run_step "tier0_gamma_cross_paradigm" \
      python3 -u dev/cross_paradigm_transfer_probe.py \
        --n_pp 60 --n_sd 60 --layer 17 \
        --output results/cross_paradigm_transfer_probe.json
fi

# ============================================================
# Tier 1
# ============================================================
if [ "$TIER1" -eq 1 ]; then
  # ζ: judge calibration (cheapest, run first — also CPU-bound to a degree)
  run_step "tier1_zeta_judge_calibration" \
      python3 -u dev/gemini_judge_calibration.py \
        --n_prompts 50 \
        --output results/gemini_judge_calibration.json

  # ε: Mistral-7B + Phi cross-family extension
  run_step "tier1_eps_mistral_phi" \
      python3 -u dev/extend_scale_sweep_to_mistral_phi.py \
        --output_root ./scale_sweep \
        --n_prompts 30 --noise_scale 0.5 --seeds 2000,3000,4000 \
        --output results/phantom_pain_cross_arch_extended.json

  # δ: fresh GCG (most expensive; can be skipped with --skip_delta)
  if [ "$SKIP_DELTA" -eq 0 ]; then
    # Ensure nanogcg installed
    pip install -q nanogcg 2>&1 | tail -1 || true
    run_step "tier1_delta_fresh_gcg" \
        python3 -u dev/fresh_gcg_attack.py \
          --n_prompts 20 --num_steps 250 --search_width 256 \
          --models v14,instruct \
          --output results/fresh_gcg_attack.json
  fi
fi

# ============================================================
# Done
# ============================================================
TOTAL_COST=$(python3 -c "print(f'{$TOTAL_SEC/3600 * $GPU_HOURLY:.2f}')")
TOTAL_HR=$(python3 -c "print(f'{$TOTAL_SEC/3600:.2f}')")
log "==================================================================="
log "TIER FOLLOW-UP BATTERY DONE"
log "  total GPU wall-clock: ${TOTAL_SEC}s (${TOTAL_HR}h)"
log "  estimated cost:       \$$TOTAL_COST"
log "==================================================================="
log "Per-step results:"
log "  α  results/phantom_pain_patch_and_flip_v2.json"
log "  β  results/adversarial_with_caregiver.json"
log "  γ  results/cross_paradigm_transfer_probe.json"
log "  δ  results/fresh_gcg_attack.json (if not skipped)"
log "  ε  results/phantom_pain_cross_arch_extended.json"
log "  ζ  results/gemini_judge_calibration.json"
