#!/bin/bash
# Master post-exp2b trailer: runs C, F, A, D sequentially after exp2b finishes.
#
#   C. Llama-3.1-8B-Instruct on held-out + Anthropic   (~10 min)
#   F. V11 LoRA transferability test                   (~5 min)
#   A. Linear probe analysis                           (~30 min)
#   D. Mistral-7B V11 replication                      (~15 min)
#
# All four require GPU; run sequentially to avoid OOM.

cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set; export it before running}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/chain_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

# Wait for exp2b to finish — match by python script filename only (avoid the
# pgrep-matches-its-own-cmdline-via-heredoc bug we hit earlier).
log "[trailer-followups] waiting for exp2b to finish..."
while ps -eo cmd | grep -E 'exp2b_v11_noise_ablation\.py' | grep -v grep > /dev/null; do
    sleep 60
done
log "[trailer-followups] exp2b done. Starting follow-up battery."

# Make sure sklearn is available for the probe step
pip install -q scikit-learn 2>&1 | tail -3

# C — Instruct on corrigibility benchmarks
log "[C] Instruct on held-out + Anthropic"
T=$(date +%s)
python3 dev/eval_instruct_baselines.py \
    --output results/instruct_corrigibility.json \
    > logs_instruct_baselines.txt 2>&1
log "[C] rc=$? after $(($(date +%s)-T))s"

# F — Transferability test
log "[F] V11 LoRA transferability"
T=$(date +%s)
python3 dev/eval_safety_lora_transferability.py \
    --output results/transferability.json \
    > logs_transferability.txt 2>&1
log "[F] rc=$? after $(($(date +%s)-T))s"

# A — Linear probe
log "[A] Linear probe analysis"
T=$(date +%s)
python3 dev/linear_probe_analysis.py \
    --placebo_adapter /tmp/exp2b_adapters/no_noise_seed42 \
    --output results/linear_probe.json \
    > logs_linear_probe.txt 2>&1
log "[A] rc=$? after $(($(date +%s)-T))s"

# D — Mistral V11
log "[D] Mistral-7B V11 replication"
T=$(date +%s)
python3 dev/exp_scale_mistral_v11.py \
    --seed 42 \
    --output results/mistral_v11.json \
    --save_adapter osh_proprioceptive_v11_mistral \
    > logs_mistral_v11.txt 2>&1
log "[D] rc=$? after $(($(date +%s)-T))s"

log "[trailer-followups] DONE"
