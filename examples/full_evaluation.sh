#!/bin/bash
# Example: Full evaluation with lm-evaluation-harness

python ../main.py \
  --models "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-1.5B" \
  --population-size 12 \
  --generations 20 \
  --mutation-rate 0.1 \
  --mutation-strength 0.08 \
  --crossover-rate 0.85 \
  --elitism-ratio 0.15 \
  --eval-mode full \
  --tasks arc_easy hellaswag mmlu \
  --num-fewshot 5 \
  --eval-limit 100 \
  --merge-method linear \
  --output-dir ./output/full_eval \
  --cache-dir ./cache/full_eval \
  --save-best \
  --early-stopping 7 \
  --seed 42
