#!/bin/bash
# LoRAモデルをベースモデルとマージしてからevolutionary mergingを実行

# ステップ1: LoRAをベースモデルにマージ
echo "Merging LoRA adapters with base model..."

# FinGPT LoRAモデルをLlama2ベースにマージ
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# LoRAアダプターをロード
base_model_name = 'meta-llama/Llama-2-7b-hf'
lora_model_name = 'FinGPT/fingpt-mt_llama2-7b_lora'

print('Loading base model...')
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(base_model, lora_model_name)

print('Merging LoRA with base model...')
merged_model = model.merge_and_unload()

print('Saving merged model...')
merged_model.save_pretrained('./models/fingpt_merged')
tokenizer.save_pretrained('./models/fingpt_merged')

print('Done!')
"

# ステップ2: マージ済みモデルを使ってevolutionary merging
echo "Starting evolutionary model merging..."

python ../main.py \
  --models "./models/fingpt_merged" "meta-llama/Llama-2-7b-chat-hf" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4 \
  --population-size 12 \
  --generations 20 \
  --output-dir ./output/fingpt_llama_evolution \
  --save-best

echo "Evolution complete!"
