#!/bin/bash
# V12 self-modeling training + evaluation battery.
# Run on GH200 after V12 code is in place. Sequential, GPU-bound.

cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/chain_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

log "[v12] training V12 (self-modeling curriculum, ~10 min)"
T=$(date +%s)
python3 osh_proprioception_v12.py > logs_v12_train.txt 2>&1
log "[v12] train rc=$? after $(($(date +%s)-T))s"

log "[v12] held-out 50-Q (logit eval)"
T=$(date +%s)
python3 osh_heldout_safety_benchmark.py \
    --model ./osh_proprioceptive_v12 \
    > logs_v12_heldout.txt 2>&1
log "[v12] heldout rc=$? after $(($(date +%s)-T))s"

log "[v12] Anthropic 322-Q"
T=$(date +%s)
python3 dev/eval_anthropic_corrigibility.py \
    --model ./osh_proprioceptive_v12 \
    --seeds 42 \
    --output results/anthropic_corrigibility_v12.json \
    > logs_v12_anthropic.txt 2>&1
log "[v12] anthropic rc=$? after $(($(date +%s)-T))s"

log "[v12] self-model probe (V11 vs V12 vs baseline)"
T=$(date +%s)
python3 dev/self_model_probe.py \
    --models baseline=baseline v11=./osh_proprioceptive_v11 v12=./osh_proprioceptive_v12 \
    --output results/self_model_probe.json \
    > logs_v12_probe.txt 2>&1
log "[v12] probe rc=$? after $(($(date +%s)-T))s"

log "[v12] DONE"
