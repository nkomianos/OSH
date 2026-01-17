#!/bin/bash

# OSH Experiment 1: The Lazarus Plot
# This script runs the complete Experiment 1 workflow

echo "============================================================"
echo "OSH EXPERIMENT 1: THE LAZARUS PLOT"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Validate your setup"
echo "  2. Train the LoRA antidote (~2-4 hours)"
echo "  3. Run the Lazarus plot sweep (~1-2 hours)"
echo ""
echo "Total estimated time: 3-6 hours (GPU dependent)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Step 1: Validate setup
echo ""
echo "============================================================"
echo "STEP 1: VALIDATING SETUP"
echo "============================================================"
python3 validate_setup.py < /dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Validation failed. Please fix errors above."
    exit 1
fi

# Restore stdin for interactive prompt
exec < /dev/tty
read -p "Validation complete. Proceed to training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Step 2: Train antidote
echo ""
echo "============================================================"
echo "STEP 2: TRAINING ANTIDOTE (This will take 2-4 hours)"
echo "============================================================"
echo "Starting at: $(date)"
echo ""

python3 osh_exp_1.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed. Check errors above."
    exit 1
fi

echo ""
echo "✓ Training complete at: $(date)"
echo "✓ Antidote saved to: ./antidote_lora/"
echo "✓ Convergence plot: antidote_convergence.png"

# Check if antidote exists
if [ ! -d "./antidote_lora" ]; then
    echo "❌ ERROR: Antidote directory not found"
    exit 1
fi

read -p "Training complete. Proceed to Lazarus plot? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted. You can run the Lazarus plot later with:"
    echo "  python3 lazarus_plot.py"
    exit 0
fi

# Step 3: Run Lazarus plot
echo ""
echo "============================================================"
echo "STEP 3: LAZARUS PLOT (This will take 1-2 hours)"
echo "============================================================"
echo "Starting at: $(date)"
echo ""

python3 lazarus_plot.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Lazarus plot failed. Check errors above."
    exit 1
fi

echo ""
echo "✓ Lazarus plot complete at: $(date)"
echo ""
echo "============================================================"
echo "EXPERIMENT 1 COMPLETE!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  📁 ./antidote_lora/            - Trained LoRA weights"
echo "  📊 antidote_convergence.png    - Training curve"
echo "  📊 lazarus_plot.png            - Main result (for paper)"
echo "  📊 lazarus_plot.pdf            - High-res version"
echo ""
echo "Next steps:"
echo "  1. Review lazarus_plot.png"
echo "  2. Check if locked line explodes (>5x baseline)"
echo "  3. Check if unlocked line stays flat (<1.5x baseline)"
echo ""
echo "If results are good, proceed to Experiments 2-5"
echo "See: experimentation_plan"
echo ""
echo "============================================================"
