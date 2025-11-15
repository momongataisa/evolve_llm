# モデル互換性ガイド

## 重要: モデルマージの制約

**異なるアーキテクチャのモデルは直接マージできません！**

### マージ可能な条件

以下の条件を**すべて**満たす必要があります：

1. ✅ **同じモデルアーキテクチャ**
   - Llama2同士、Qwen同士、Mistral同士など
   - レイヤー数が同じ
   - 隠れ層のサイズ（hidden_size）が同じ

2. ✅ **互換性のあるトークナイザー**
   - 同じボキャブラリサイズ
   - 同じトークン化方式

3. ✅ **同じモデルタイプ**
   - CausalLM同士（因果言語モデル）

### ❌ マージできない組み合わせ例

```bash
# NG: 異なるアーキテクチャ
FinGPT (Llama2ベース) × Qwen2.5
Llama2 × Mistral
GPT-2 × Llama

# NG: 異なるサイズ
Llama2-7B × Llama2-13B
Qwen2.5-1.5B × Qwen2.5-7B
```

### ✅ マージ可能な組み合わせ例

```bash
# OK: 同じアーキテクチャ、同じサイズ
Llama2-7B × Llama2-7B-chat
Qwen2.5-7B × Qwen2.5-7B-Instruct
Mistral-7B-v0.1 × Mistral-7B-Instruct-v0.1

# OK: 同じベースモデルのファインチューン
Llama2-7B × FinGPT(Llama2-7Bベース、マージ済み)
Qwen2.5-7B × 自作ファインチューンモデル(Qwen2.5-7Bベース)
```

## 金融モデル × 汎用モデルの実現方法

### 方法1: 同じアーキテクチャのモデルを選ぶ（推奨）

#### Llama2ベースで統一

```bash
# 金融: FinGPTのLoRAをマージしたモデル
# 汎用: Llama2-chat

python main.py \
  --models "./models/fingpt_merged" "meta-llama/Llama-2-7b-chat-hf" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4
```

#### Qwenベースで統一

```bash
# 金融: Qwenベースの金融ファインチューン（自作）
# 汎用: Qwen-Instruct

python main.py \
  --models "your-qwen-financial-model" "Qwen/Qwen2.5-7B-Instruct" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4
```

### 方法2: LoRAモデルをベースモデルにマージ

FinGPTなどのLoRAモデルは、まずベースモデルにマージします：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# LoRAをロードしてマージ
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "FinGPT/fingpt-mt_llama2-7b_lora")
merged_model = model.merge_and_unload()

# 保存
merged_model.save_pretrained("./models/fingpt_merged")
```

その後、マージ済みモデルを使います：

```bash
python main.py \
  --models "./models/fingpt_merged" "meta-llama/Llama-2-7b-chat-hf" \
  --eval-mode simple_multi_domain
```

**サンプルスクリプト**: `examples/lora_merge_example.py`

### 方法3: 同じベースの金融ファインチューンモデルを作成

Qwenで金融特化モデルを作りたい場合：

1. Qwen2.5をベースに金融データでファインチューン
2. ファインチューンモデル × Qwen2.5-Instructをマージ

```bash
# 1. 金融データでファインチューン（別途実装）
# qwen_financial_model = fine_tune(Qwen2.5-7B, financial_dataset)

# 2. 進化的マージ
python main.py \
  --models "./models/qwen_financial" "Qwen/Qwen2.5-7B-Instruct" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4
```

## 利用可能な金融モデル

### Llama2ベース（推奨）

| モデル | タイプ | ベース | 用途 |
|--------|--------|--------|------|
| FinGPT/fingpt-mt_llama2-7b_lora | LoRA | Llama2-7B | マルチタスク金融 |
| FinGPT/fingpt-forecaster_dow30_llama2-7b_lora | LoRA | Llama2-7B | 株価予測 |
| FinGPT/fingpt-sentiment_llama2-7b_lora | LoRA | Llama2-7B | 感情分析 |

**使い方**: LoRAをマージしてから使用（`examples/lora_merge_example.py`参照）

### その他

- BloombergGPT（非公開）
- FinBERT（エンコーダーモデル、マージ不可）

## アーキテクチャ互換性チェック

モデルがマージ可能か確認するスクリプト：

```python
from transformers import AutoConfig

def check_compatibility(model1_path, model2_path):
    """モデル互換性チェック"""
    config1 = AutoConfig.from_pretrained(model1_path)
    config2 = AutoConfig.from_pretrained(model2_path)

    checks = {
        "Model type": config1.model_type == config2.model_type,
        "Num layers": config1.num_hidden_layers == config2.num_hidden_layers,
        "Hidden size": config1.hidden_size == config2.hidden_size,
        "Vocab size": config1.vocab_size == config2.vocab_size,
    }

    print("Compatibility Check:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}: {result}")

    compatible = all(checks.values())
    print(f"\nResult: {'✅ Compatible' if compatible else '❌ Not Compatible'}")
    return compatible

# 使用例
check_compatibility(
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf"
)
```

## 推奨ワークフロー

### 金融特化 × 汎用のマージ

```bash
# ステップ1: LoRAをマージ（必要な場合）
python examples/lora_merge_example.py

# ステップ2: 互換性チェック（オプション）
python -c "
from transformers import AutoConfig
c1 = AutoConfig.from_pretrained('./models/fingpt_merged')
c2 = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
print(f'Layers: {c1.num_hidden_layers} vs {c2.num_hidden_layers}')
print(f'Hidden: {c1.hidden_size} vs {c2.hidden_size}')
print(f'Compatible: {c1.model_type == c2.model_type}')
"

# ステップ3: 進化的マージ
python main.py \
  --models "./models/fingpt_merged" "meta-llama/Llama-2-7b-chat-hf" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4 \
  --population-size 12 \
  --generations 20 \
  --save-best
```

## トラブルシューティング

### エラー: "Incompatible architectures"

**原因**: 異なるアーキテクチャのモデルをマージしようとした

**解決策**: 同じベースアーキテクチャのモデルを使用

### エラー: "Layer size mismatch"

**原因**: レイヤー数や隠れ層サイズが異なる

**解決策**: 同じサイズのモデルを使用（例：7B同士）

### LoRAモデルの扱い方が分からない

**解決策**: `examples/lora_merge_example.py`を参照

```bash
# 必要なライブラリ
pip install peft

# 実行
python examples/lora_merge_example.py
```

## まとめ

- ✅ **同じアーキテクチャ、同じサイズのモデルを使う**
- ✅ **LoRAモデルは先にベースモデルにマージ**
- ✅ **Llama2ベースで統一するのが簡単（FinGPTが豊富）**
- ❌ **異なるアーキテクチャ（FinGPT × Qwen）は不可**

質問があれば、GitHub Issuesで報告してください。
