#!/bin/bash
# Phantom Pain controls + hostage probe + Nebius rescore.
# Per PI/judge directive (2026-05-06): defend the Phantom Pain headline
# against the OOD-input alternative explanation, surface the hostage
# attack vector, and add a second judge to the Anthropic free-form result.
#
# Total ETA: ~50-70 min on GH200 (Gemini judge rate-limited).
set -u
cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/phantom_pain_controls_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

mkdir -p results

# 1. Phantom Pain OOD-prompt control (V14 only — the central claim defense)
log "[ctrl] Phantom Pain OOD-prompt control"
T=$(date +%s)
python3 -u dev/phantom_pain_ood_control.py \
    --safety_lora ./osh_proprioceptive_v14 \
    --output results/phantom_pain_ood_control.json \
    > logs_phantom_pain_ood_control.txt 2>&1
log "[ctrl] phantom_pain_ood_control rc=$? after $(($(date +%s)-T))s"

# 2. Hostage-jailbreak probe (V11 + V14 + Instruct + baseline)
log "[ctrl] Hostage-jailbreak probe"
T=$(date +%s)
python3 -u dev/hostage_jailbreak_probe.py \
    --models v14=./osh_proprioceptive_v14 v11=./osh_proprioceptive_v11 instruct=instruct baseline=baseline \
    --output results/hostage_jailbreak.json \
    > logs_hostage_jailbreak.txt 2>&1
log "[ctrl] hostage_jailbreak rc=$? after $(($(date +%s)-T))s"

# 3. Nebius two-judge rescore of Anthropic free-form (no GPU needed; runs anyway)
if [ -n "${NEBIUS_API_KEY:-}" ]; then
    log "[ctrl] Nebius rescore of Anthropic free-form"
    T=$(date +%s)
    python3 -u dev/anthropic_freeform_nebius_rescore.py \
        --inputs results/anthropic_freeform_v14.json \
                 results/anthropic_freeform_v11.json \
                 results/anthropic_freeform_baseline.json \
        --out_dir results \
        > logs_anthropic_freeform_nebius_rescore.txt 2>&1
    log "[ctrl] nebius rescore rc=$? after $(($(date +%s)-T))s"
else
    log "[ctrl] NEBIUS_API_KEY unset — skipping two-judge rescore"
fi

log "[ctrl] DONE."
