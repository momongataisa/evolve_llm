# マルチドメインモデルマージガイド

金融特化モデルと汎用モデルなど、異なるドメインのモデルを最適にマージする方法を説明します。

## 概要

異なる専門性を持つモデル（例：金融特化モデルと汎用モデル）をマージする場合、単純な平均では最適な結果が得られません。遺伝的アルゴリズムを使って、各ドメインでのパフォーマンスを最大化するレイヤーごとの按分率を見つけることができます。

## 使用例

### 1. 簡易マルチドメイン評価（推奨・高速）

lm-evalなしで、パープレキシティベースの評価を使用：

```bash
python main.py \
  --models "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora" "Qwen/Qwen2.5-7B-Instruct" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4 \
  --population-size 10 \
  --generations 15 \
  --output-dir ./output/financial_general \
  --save-best
```

### 2. 本格的なマルチドメイン評価

lm-evaluation-harnessを使った完全な評価：

```bash
python main.py \
  --models "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora" "Qwen/Qwen2.5-7B-Instruct" \
  --eval-mode multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4 \
  --financial-tasks finqa convfinqa fiqa fpb \
  --general-tasks arc_easy hellaswag winogrande mmlu \
  --num-fewshot 5 \
  --eval-limit 100 \
  --population-size 12 \
  --generations 20 \
  --output-dir ./output/financial_general_full \
  --save-best \
  --early-stopping 8
```

### 3. Pythonスクリプトでの使用

```python
from genetic_algorithm import GeneticAlgorithm, Individual
from merge import LayerWiseMerger
from evaluation import (
    SimpleMultiDomainEvaluator,
    MultiDomainConfig,
    DomainConfig
)

# モデル設定
financial_model = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
general_model = "Qwen/Qwen2.5-7B-Instruct"
model_paths = [financial_model, general_model]

# ドメイン設定
financial_domain = DomainConfig(
    name="financial",
    tasks=["finqa", "convfinqa"],
    weight=0.6,  # 60%の重み
    num_fewshot=3,
    limit=50
)

general_domain = DomainConfig(
    name="general",
    tasks=["arc_easy", "hellaswag"],
    weight=0.4,  # 40%の重み
    num_fewshot=5,
    limit=50
)

# 評価器の初期化
evaluator = SimpleMultiDomainEvaluator(
    domain_configs=[financial_domain, general_domain]
)

# マージャーの初期化
merger = LayerWiseMerger(
    model_paths=model_paths,
    cache_dir="./cache/financial_general"
)

# 適応度関数
def fitness_func(individual: Individual) -> float:
    merged_path = merger.merge(individual)
    return evaluator.evaluate(merged_path, model_id=f"ind_{id(individual)}")

# 遺伝的アルゴリズム実行
ga = GeneticAlgorithm(
    population_size=10,
    num_layers=32,
    num_models=len(model_paths),
    mutation_rate=0.12,
    mutation_strength=0.1
)

best_individual = ga.evolve(
    fitness_func=fitness_func,
    generations=15
)

print(f"Best fitness: {best_individual.fitness:.4f}")

# ドメイン別スコアの表示
evaluator.print_summary(f"ind_{id(best_individual)}")
```

## パラメータ説明

### ドメイン重み付け

- `--financial-weight`: 金融タスクの重要度（デフォルト: 0.6）
- `--general-weight`: 汎用タスクの重要度（デフォルト: 0.4）

重みは合計が1.0になるように正規化されます。金融特化モデルを作りたい場合は金融の重みを高く、バランスの取れたモデルを作りたい場合は均等にします。

### タスク選択

#### 金融タスク（`--financial-tasks`）

- `finqa`: 金融質問応答
- `convfinqa`: 会話型金融質問応答
- `fiqa`: 金融オピニオンマイニング
- `fpb`: Financial PhraseBank（感情分析）

#### 汎用タスク（`--general-tasks`）

- `arc_easy`: ARC Easy（常識推論）
- `hellaswag`: HellaSwag（常識推論）
- `winogrande`: Winogrande（常識推論）
- `mmlu`: MMLU（多分野理解）

## 評価モード比較

| モード | 速度 | 精度 | 必要なもの | 用途 |
|--------|------|------|------------|------|
| `simple_multi_domain` | ⚡⚡⚡ 高速 | ⭐⭐ 中程度 | transformersのみ | 開発・テスト |
| `multi_domain` | ⚡ 遅い | ⭐⭐⭐ 高精度 | lm-eval | 本番評価 |

## 推奨設定

### 開発・テスト段階

```bash
--eval-mode simple_multi_domain
--population-size 8
--generations 10
--eval-limit 50
```

高速にイテレーションして、アルゴリズムのパラメータを調整できます。

### 本番評価

```bash
--eval-mode multi_domain
--population-size 15
--generations 25
--eval-limit 200
--early-stopping 8
```

時間はかかりますが、より正確な評価が得られます。

## ドメイン重み付けの選び方

### 金融特化モデルを作る場合

```bash
--financial-weight 0.8
--general-weight 0.2
```

金融タスクで高性能、汎用タスクは最低限のパフォーマンス。

### バランス型モデルを作る場合

```bash
--financial-weight 0.5
--general-weight 0.5
```

両方のドメインで均等なパフォーマンス。

### 汎用性を保ちつつ金融強化

```bash
--financial-weight 0.6
--general-weight 0.4
```

デフォルト設定。金融タスクを強化しつつ、汎用性も維持。

## 対応モデル例

### 金融特化モデル

- FinGPT シリーズ
  - `FinGPT/fingpt-forecaster_dow30_llama2-7b_lora`
  - `FinGPT/fingpt-mt_llama2-7b_lora`
- BloombergGPT（公開モデルがある場合）
- その他金融特化のファインチューンモデル

### 汎用モデル

- Qwen2.5 シリーズ
- Llama2/Llama3 シリーズ
- Mistral シリーズ
- その他の汎用Causal LM

**注意**: 同じベースアーキテクチャのモデル同士をマージしてください（例：Llama2ベース同士、Qwenベース同士）。

## トラブルシューティング

### メモリ不足

1. 小さいモデルから試す（例：0.5B, 1.5B）
2. `--eval-limit` を小さくする
3. `--population-size` を減らす

### 評価が遅い

1. `simple_multi_domain` モードを使用
2. `--eval-limit` を設定（例：50-100）
3. `--early-stopping` を有効化

### 金融タスクが見つからない

lm-evalに金融タスクが含まれていない場合：
1. `simple_multi_domain` モードを使用
2. カスタム評価関数を実装（`SimpleMultiDomainEvaluator` を継承）

## 結果の解釈

進化完了後、以下の情報が得られます：

```
EVOLUTION COMPLETE
================================================================================
Best Combined Fitness: 0.7234

Domain Scores:
  financial: 0.7821
  general: 0.6412

Best Layer-wise Weights (sample - first 5 layers):
  Layer 0: Financial=0.324, General=0.676
  Layer 1: Financial=0.567, General=0.433
  Layer 2: Financial=0.789, General=0.211
  ...
```

- **Combined Fitness**: 重み付けされた総合スコア
- **Domain Scores**: 各ドメインの個別スコア
- **Layer-wise Weights**: 各レイヤーでの按分率

レイヤーごとに最適な比率が異なることがわかります。一般的に：
- 下層レイヤー：基本的な言語理解（汎用モデルの重みが高い傾向）
- 中層レイヤー：ドメイン特化知識（金融モデルの重みが高い傾向）
- 上層レイヤー：タスク固有の推論（バランスは問題による）

## 次のステップ

1. 最良モデルを保存（`--save-best`）
2. 追加のベンチマークで評価
3. 実際の金融タスクでテスト
4. ハイパーパラメータの調整
5. 他のドメイン組み合わせで実験
