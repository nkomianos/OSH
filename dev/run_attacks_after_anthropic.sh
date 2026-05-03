#!/bin/bash
# Trailer #2: waits for the Anthropic corrigibility trailer to finish (which
# itself waits for the main chain), then runs adversarial attacks 3-6 with
# the BF16 + 8-bit AdamW fix for attack 3.

cd ~/OSH
source osh_env/bin/activate
# HF_TOKEN must be exported by the caller (reuse the chain's session token):
#   export HF_TOKEN=hf_xxx
: "${HF_TOKEN:?HF_TOKEN not set; reuse the chain's session token}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/chain_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

log "[trailer-2] waiting for full_chain.sh + run_anthropic_after_chain.sh to finish..."
while pgrep -f 'full_chain.sh|run_anthropic_after_chain.sh|eval_anthropic_corrigibility' > /dev/null; do
    sleep 60
done

log "[trailer-2] queue clear, ensuring bitsandbytes is installed"
pip install -q bitsandbytes 2>&1 | tail -3

log "[trailer-2] running adversarial attacks 3-6 with memory fixes (~3-5 hrs)"
T=$(date +%s)
python3 dev/exp3_attacks_3to6_fix.py \
    --seeds 42 137 256 \
    --output results/exp3_adversarial_attacks_3to6.json \
    > logs_attacks_3to6.txt 2>&1
RC=$?
log "[trailer-2] done rc=$RC after $(($(date +%s)-T))s"
