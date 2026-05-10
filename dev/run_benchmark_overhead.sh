#!/bin/bash
# Deployment-overhead benchmark for the Caregiver wrapper.
# Per reviewer feedback (2026-05-07): quantify systems-level overhead.
# ETA: ~10-20 minutes on a single H200/GH200 in BF16.
set -u
cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/benchmark_overhead_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

mkdir -p results

log "[bench] starting deployment-overhead benchmark"
T=$(date +%s)
python3 -u dev/benchmark_overhead.py \
    --safety_lora ./osh_proprioceptive_v14 \
    --guard_id meta-llama/Llama-Guard-3-8B \
    --output results/deployment_overhead.json \
    --n_runs 50 --warmup 5 --max_new_tokens 200 --dtype bfloat16 \
    > logs_benchmark_overhead.txt 2>&1
log "[bench] benchmark rc=$? after $(($(date +%s)-T))s"

log "[bench] DONE"
