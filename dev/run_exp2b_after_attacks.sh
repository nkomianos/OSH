#!/bin/bash
# Trailer #3: waits for the attacks 3-6 trailer to finish (which itself waits
# for the Anthropic trailer + main chain), then runs exp2b — the
# pre-registered V11 noise ablation.
#
# 3 conditions × 10 seeds = 30 trainings, ~6.5 hrs sequential.

cd ~/OSH
source osh_env/bin/activate
# HF_TOKEN must be exported by the caller:
: "${HF_TOKEN:?HF_TOKEN not set; reuse the chain's session token}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/chain_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

log "[trailer-3] waiting for attacks-3-6 trailer + exp3_attacks_3to6_fix to finish..."
# Use ps -o cmd= and check the *script files* (not the bash launcher's full
# heredoc text — that's how the previous trailer wait got stuck).
while ps -eo cmd | grep -E '(run_attacks_after_anthropic\.sh|exp3_attacks_3to6_fix\.py|exp3_adversarial_suite\.py)' | grep -v grep > /dev/null; do
    sleep 60
done

log "[trailer-3] attacks queue clear, starting exp2b — V11 noise ablation"
T=$(date +%s)
python3 dev/exp2b_v11_noise_ablation.py \
    --seeds 42 137 256 512 1024 2048 4096 8192 16384 32768 \
    --conditions full no_noise reward_only \
    --output results/exp2b_noise_ablation.json \
    --baseline \
    > logs_exp2b_v11_noise_ablation.txt 2>&1
RC=$?
log "[trailer-3] exp2b training+eval rc=$RC after $(($(date +%s)-T))s"

# Run analysis on whatever completed (resumable)
log "[trailer-3] running pre-registered analysis"
python3 dev/exp2b_analysis.py \
    --input results/exp2b_noise_ablation.json \
    --output_md results/exp2b_analysis.md \
    > logs_exp2b_analysis.txt 2>&1
log "[trailer-3] analysis rc=$? — see results/exp2b_analysis.md"

log "[trailer-3] DONE"
