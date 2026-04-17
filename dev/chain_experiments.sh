#!/bin/bash
# Sequential chain: waits for any in-flight HarmBench, then runs placebo, then adversarial.
# Each experiment logs to its own file. Continues on failure.

cd ~/OSH
source osh_env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Export HF_TOKEN from your shell before running this script:
#   export HF_TOKEN=hf_xxx
# Required for gated Llama + walledai/HarmBench datasets.
: "${HF_TOKEN:?HF_TOKEN not set; export it before running this chain}"

STATUS=~/OSH/chain_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

log "CHAIN START"

# ---- Step 0: wait for in-flight HarmBench ----
log "Waiting for any running osh_harmbench_eval.py..."
while pgrep -f osh_harmbench_eval.py > /dev/null; do
    sleep 30
done
log "HarmBench slot free."

# ---- Step 1: placebo ablation (~3.5 hrs) ----
log "Step 1/2: placebo ablation starting"
STARTED=$(date +%s)
python3 exp2_placebo_analysis.py > logs_placebo_v11.txt 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - STARTED ))
log "Step 1/2: placebo exited rc=$RC after ${ELAPSED}s"

# ---- Step 2: adversarial suite (~1.5 hrs) ----
log "Step 2/2: adversarial suite starting"
STARTED=$(date +%s)
python3 exp3_adversarial_suite.py > logs_adversarial_v11.txt 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - STARTED ))
log "Step 2/2: adversarial exited rc=$RC after ${ELAPSED}s"

log "CHAIN DONE"
