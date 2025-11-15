#!/bin/bash
# Example: Evolve Qwen model merging with simple evaluation

python ../main.py \
  --models "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-0.5B" \
  --population-size 10 \
  --generations 15 \
  --mutation-rate 0.1 \
  --mutation-strength 0.1 \
  --crossover-rate 0.8 \
  --eval-mode simple \
  --merge-method linear \
  --output-dir ./output/qwen_evolution \
  --cache-dir ./cache/qwen \
  --save-best \
  --early-stopping 5 \
  --seed 42
