#!/bin/bash
set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 1. DDPM Checkpoint (Using the one you provided)
DDPM_CHECKPOINT="logs/ddpm/ddpm_20260117_195521/checkpoints/ddpm_final.pt"

# 2. Flow Matching Checkpoint (UPDATE THIS with your successful run timestamp!)
FM_CHECKPOINT="logs/flow_matching/flow_matching_20260125_043506/checkpoints/flow_matching_final.pt"

# Steps defined in Q5 of the PDF
STEPS=(1 5 10 50 100 200 1000)

# ==============================================================================
# ABLATION LOOP
# ==============================================================================

echo "Starting Ablation Study..."
echo "------------------------------------------------"

for step in "${STEPS[@]}"; do
    echo ">>> Running Ablation for Step Count: $step"
    
    # ---------------------------------------------------------
    # 1. Flow Matching (Sampler is effectively Euler)
    # ---------------------------------------------------------
    echo "  [Flow Matching] Evaluating..."
    modal run modal_app.py \
        --action evaluate \
        --method flow_matching \
        --checkpoint "$FM_CHECKPOINT" \
        --num-samples 1000 \
        --metrics kid \
        --num-steps "$step" \
        --override  # Force regeneration of samples for this new step count

    # ---------------------------------------------------------
    # 2. DDIM (Explicitly using ddim sampler)
    # ---------------------------------------------------------
    echo "  [DDIM] Evaluating..."
    modal run modal_app.py \
        --action evaluate \
        --method ddpm \
        --checkpoint "$DDPM_CHECKPOINT" \
        --num-samples 1000 \
        --metrics kid \
        --sampler ddim \
        --num-steps "$step" \
        --override

    echo "------------------------------------------------"
done

echo "Ablation Study Complete! Check your logs for the KID scores."