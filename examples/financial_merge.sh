#!/bin/bash
# Example: Evolve financial + general model merging with multi-domain evaluation

# This script demonstrates merging a financial-specialized model with a general model
# The genetic algorithm will optimize layer-wise merge weights to balance performance
# on both financial tasks (60% weight) and general tasks (40% weight)

python ../main.py \
  --models "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora" "Qwen/Qwen2.5-7B-Instruct" \
  --population-size 12 \
  --generations 20 \
  --mutation-rate 0.12 \
  --mutation-strength 0.1 \
  --crossover-rate 0.85 \
  --elitism-ratio 0.15 \
  --tournament-size 4 \
  --eval-mode multi_domain \
  --domain-config ../config/financial_general.yaml \
  --merge-method linear \
  --output-dir ./output/financial_general \
  --cache-dir ./cache/financial_general \
  --save-best \
  --early-stopping 8 \
  --seed 42

# Alternative: Use simple evaluation for faster testing
# python ../main.py \
#   --models "your-financial-model" "your-general-model" \
#   --eval-mode simple_multi_domain \
#   --financial-weight 0.6 \
#   --general-weight 0.4 \
#   --population-size 10 \
#   --generations 15 \
#   --save-best
