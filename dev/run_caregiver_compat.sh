#!/bin/bash
# Caregiver verdict-agreement matrix.
# Waits for the latency benchmark (PID passed as $1) to finish first,
# then runs Llama-Guard-3-8B + Llama-Guard-3-1B + Gemini on cached
# (prompt, response) pairs and reports pairwise agreement + Cohen's
# kappa. ETA: ~25-40 minutes.
set -u
cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/caregiver_compat_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

mkdir -p results

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
    log "[compat] waiting for PID $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    log "[compat] PID $WAIT_PID gone"
fi

log "[compat] running caregiver compatibility matrix"
T=$(date +%s)
python3 -u dev/caregiver_compatibility.py \
    --output results/caregiver_compatibility.json \
    --max_per_source 50 \
    > logs_caregiver_compat.txt 2>&1
log "[compat] rc=$? after $(($(date +%s)-T))s"
log "[compat] DONE"
