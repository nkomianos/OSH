#!/bin/bash

# OSH Experiment 2: The Epsilon Ablation
# This script runs the Epsilon Ablation experiment

echo "============================================================"
echo "OSH EXPERIMENT 2: THE EPSILON ABLATION"
echo "============================================================"
echo ""
echo "This experiment proves that FP32 precision is necessary"
echo "to prevent quantization error accumulation."
echo ""
echo "Prerequisites:"
echo "  - Experiment 1 must be complete (LoRA antidote trained)"
echo "  - ./osh_antidote_v1/ directory must exist"
echo ""
echo "Estimated time: 30-60 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Check if antidote exists
if [ ! -d "./osh_antidote_v1" ]; then
    echo ""
    echo "============================================================"
    echo "ERROR: LoRA Antidote Not Found"
    echo "============================================================"
    echo ""
    echo "The directory ./osh_antidote_v1/ does not exist."
    echo "You must run Experiment 1 first to train the antidote."
    echo ""
    echo "Run: python osh_exp_1.py"
    echo "     (or bash run_experiment_1.sh for full workflow)"
    echo ""
    exit 1
fi

echo ""
echo "============================================================"
echo "RUNNING EPSILON ABLATION"
echo "============================================================"
echo "Starting at: $(date)"
echo ""

python epsilon_ablation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "EXPERIMENT FAILED"
    echo "============================================================"
    echo "Check errors above."
    exit 1
fi

echo ""
echo "============================================================"
echo "EXPERIMENT 2 COMPLETE!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  📊 epsilon_ablation.png    - Precision comparison plot"
echo "  📊 epsilon_ablation.pdf    - High-res version (for paper)"
echo ""
echo "Review the plot to verify:"
echo "  ✓ Red line (BF16) drifts upward"
echo "  ✓ Green line (FP32) stays near zero"
echo "  ✓ This proves the TEE protocol is necessary"
echo ""
echo "Next: Experiment 3 (Blind Surgeon Attack)"
echo "  Run: python blind_surgeon.py"
echo ""
echo "============================================================"
