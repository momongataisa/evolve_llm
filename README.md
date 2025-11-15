# Evolve LLM - 遺伝的アルゴリズムによるLLMモデルマージ

Qwenなどの大規模言語モデル（LLM）を遺伝的アルゴリズムで最適にマージするためのフレームワークです。レイヤーごとの按分率を進化させることで、最適なモデル統合を実現します。

## 特徴

- **レイヤーごとの按分率最適化**: 各レイヤーで異なるマージ比率を設定し、最適な組み合わせを探索
- **遺伝的アルゴリズム**: 進化的手法による効率的な探索
- **マルチドメイン評価**: 金融特化モデル×汎用モデルなど、異なるドメインのモデルマージに対応
- **柔軟な評価**: lm-evaluation-harnessを使った本格的な評価から、簡易評価、モックテストまで対応
- **複数のマージ手法**: linear, slerp, ties, dareなど
- **キャッシング**: マージ済みモデルと評価結果のキャッシュによる高速化

## アーキテクチャ

```
evolve_llm/
├── genetic_algorithm/     # 遺伝的アルゴリズムのコア実装
│   ├── individual.py      # 個体の定義（レイヤーごとの按分率）
│   ├── population.py      # 集団管理
│   └── evolution.py       # 進化ロジック
├── merge/                 # モデルマージ機能
│   ├── layer_merge.py     # レイヤーごとのマージ実行
│   └── config_generator.py # mergekit設定生成
├── evaluation/            # 適応度評価
│   ├── fitness.py         # ベンチマーク評価
│   └── multi_domain.py    # マルチドメイン評価
├── config/                # 設定ファイル
├── examples/              # 使用例
├── docs/                  # ドキュメント
└── main.py                # メインエントリーポイント
```

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd evolve_llm

# 依存関係のインストール
pip install -r requirements.txt

# mergekitのインストール（必須）
pip install mergekit

# lm-evaluation-harnessのインストール（本格評価に必要）
pip install lm-eval
```

## 使い方

### 基本的な使い方

```bash
python main.py \
  --models "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-0.5B" \
  --population-size 10 \
  --generations 15 \
  --eval-mode simple \
  --output-dir ./output \
  --save-best
```

### パラメータ説明

#### モデル設定
- `--models`: マージするモデルのパス（2つ以上必要）
- `--num-layers`: レイヤー数（自動検出されない場合に指定）

#### 遺伝的アルゴリズムパラメータ
- `--population-size`: 集団サイズ（デフォルト: 10）
- `--generations`: 世代数（デフォルト: 20）
- `--mutation-rate`: 突然変異率（デフォルト: 0.1）
- `--mutation-strength`: 突然変異の強度（デフォルト: 0.1）
- `--crossover-rate`: 交叉率（デフォルト: 0.8）
- `--elitism-ratio`: エリート保存率（デフォルト: 0.1）
- `--tournament-size`: トーナメント選択のサイズ（デフォルト: 3）

#### 評価設定
- `--eval-mode`: 評価モード
  - `full`: lm-evalを使った本格評価
  - `simple`: パープレキシティによる簡易評価
  - `mock`: テスト用のモック評価
- `--tasks`: 評価タスク（full modeの場合）
- `--num-fewshot`: Few-shotサンプル数
- `--eval-limit`: 評価サンプル数の制限（高速テスト用）

#### マージ設定
- `--merge-method`: マージ手法（linear, slerp, ties, dare）
- `--cache-dir`: マージ済みモデルのキャッシュディレクトリ（デフォルト: `./merged_models`）
- `--no-save-merged-models`: 中間マージモデルを保存しない（ディスク容量を節約、キャッシング無効）

#### その他
- `--output-dir`: 出力ディレクトリ
- `--save-best`: 最良モデルを保存
- `--early-stopping`: 早期停止の世代数
- `--seed`: 乱数シード

## 使用例

### 例1: クイックテスト（モック評価）

```bash
python main.py \
  --models "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-0.5B" \
  --population-size 8 \
  --generations 10 \
  --eval-mode mock \
  --output-dir ./output/quick_test
```

### 例2: 簡易評価（パープレキシティ）

```bash
python main.py \
  --models "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-1.5B" \
  --population-size 12 \
  --generations 15 \
  --eval-mode simple \
  --output-dir ./output/simple_eval \
  --save-best
```

### 例3: 本格的な評価（lm-eval）

```bash
python main.py \
  --models "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-1.5B" \
  --population-size 12 \
  --generations 20 \
  --eval-mode full \
  --tasks arc_easy hellaswag mmlu \
  --num-fewshot 5 \
  --eval-limit 100 \
  --output-dir ./output/full_eval \
  --save-best \
  --early-stopping 7
  --no-save-merged-models
```

### 例4: Python APIの使用

```python
from genetic_algorithm import GeneticAlgorithm, Individual
from merge import LayerWiseMerger
from evaluation import MockFitnessEvaluator

# モデル設定
model_paths = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B"]
num_layers = 24

# コンポーネント初期化
merger = LayerWiseMerger(model_paths=model_paths)
evaluator = MockFitnessEvaluator()

# 適応度関数
def fitness_func(individual: Individual) -> float:
    merged_path = merger.merge(individual)
    return evaluator.evaluate(merged_path)

# 遺伝的アルゴリズム実行
ga = GeneticAlgorithm(
    population_size=10,
    num_layers=num_layers,
    num_models=len(model_paths)
)

best_individual = ga.evolve(
    fitness_func=fitness_func,
    generations=15
)

print(f"Best fitness: {best_individual.fitness}")
```

### 例5: 金融特化モデル × 汎用モデル（マルチドメインマージ）

異なるドメインのモデルを最適にマージ：

```bash
# 簡易評価版（推奨）
python main.py \
  --models "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora" "Qwen/Qwen2.5-7B-Instruct" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4 \
  --population-size 12 \
  --generations 20 \
  --output-dir ./output/financial_general \
  --save-best

# 本格評価版
python main.py \
  --models "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora" "Qwen/Qwen2.5-7B-Instruct" \
  --eval-mode multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4 \
  --financial-tasks finqa convfinqa fiqa fpb \
  --general-tasks arc_easy hellaswag winogrande mmlu \
  --eval-limit 100 \
  --population-size 15 \
  --generations 25 \
  --output-dir ./output/financial_general_full \
  --save-best \
  --early-stopping 8
```

**詳細**: [docs/MULTI_DOMAIN_MERGING.md](docs/MULTI_DOMAIN_MERGING.md) を参照

## 仕組み

### 1. 個体の表現

各個体は、モデルの全レイヤーについて、マージする各モデルの重み（按分率）を持ちます。

```python
# 例: 2モデル、24レイヤーの場合
genes = [
    [0.7, 0.3],  # レイヤー0: モデル1が70%, モデル2が30%
    [0.6, 0.4],  # レイヤー1: モデル1が60%, モデル2が40%
    [0.5, 0.5],  # レイヤー2: 均等
    ...
    [0.3, 0.7],  # レイヤー23: モデル1が30%, モデル2が70%
]
```

**重要な特徴：**
- 形状: `(num_layers, num_models)` の2次元配列
- 各レイヤーの重みの合計は必ず1.0
- 初期化はランダムに生成

### 2. 進化のプロセス

```
初期集団生成 → [評価 → 選択 → 交叉 → 突然変異] × N世代 → 最良個体
```

#### ステップ1: 初期化
ランダムな按分率で集団（population）を生成します。

```python
# 例：10個体の集団を生成
population_size = 10
individuals = [Individual(num_layers, num_models) for _ in range(10)]
```

#### ステップ2: 適応度評価

各個体について以下を実行：
1. **モデルマージ**: 個体の遺伝子に基づいてmergekitでモデルをマージ
2. **性能評価**: マージされたモデルをベンチマークで評価
3. **適応度スコア**: 評価結果を個体に保存

評価モード：
- **Full モード**: MMLU、HellaSwag、ARC等のベンチマークで評価
- **Simple モード**: パープレキシティで簡易評価
- **Mock モード**: テスト用のランダム評価

#### ステップ3: 選択（Selection）

次世代の親を選ぶ方法：

**トーナメント選択**（デフォルト）:
```python
# ランダムに3個体選んで、最も適応度が高い個体を親に選ぶ
tournament = random.choice(individuals, 3)
parent = max(tournament, key=lambda x: x.fitness)
```

**ルーレット選択**（オプション）:
```python
# 適応度に比例した確率で選択
probabilities = fitnesses / total_fitness
parent = random.choice(individuals, p=probabilities)
```

#### ステップ4: 交叉（Crossover）

2つの親から2つの子を生成（単一点交叉）：

```python
# 例：crossover_point = 12の場合
crossover_point = random.randint(1, num_layers)

offspring1 = [
    parent1[0:12],    # 前半は親1から
    parent2[12:24]    # 後半は親2から
]

offspring2 = [
    parent2[0:12],    # 前半は親2から
    parent1[12:24]    # 後半は親1から
]
```

視覚化：
```
親1: [0.7,0.3][0.6,0.4][0.5,0.5]...|...[0.3,0.7][0.2,0.8]
親2: [0.8,0.2][0.7,0.3][0.6,0.4]...|...[0.4,0.6][0.5,0.5]
     ←------- 前半 -------→      ↑交叉点
子1: [0.7,0.3][0.6,0.4][0.5,0.5]...|...[0.4,0.6][0.5,0.5]
子2: [0.8,0.2][0.7,0.3][0.6,0.4]...|...[0.3,0.7][0.2,0.8]
```

交叉率（デフォルト80%）に従って、交叉するか親をそのままコピーするか決定します。

#### ステップ5: 突然変異（Mutation）

遺伝的多様性を保つために、ランダムに値を変更：

```python
for each layer:
    if random() < mutation_rate:  # デフォルト10%の確率
        # ガウスノイズを追加
        noise = normal(0, mutation_strength)  # デフォルトstd=0.1
        genes[layer] += noise

        # [0,1]の範囲にクリップ
        genes[layer] = clip(genes[layer], 0.0, 1.0)

        # 合計が1.0になるように正規化
        genes[layer] = genes[layer] / sum(genes[layer])
```

例：
```
変異前: [0.6, 0.4]
ノイズ: [+0.05, -0.05]
変異後: [0.65, 0.35]
```

#### ステップ6: エリート保存（Elitism）

最良個体を次世代に無条件で引き継ぎます（デフォルト10%）：

```python
elitism_ratio = 0.1
num_elites = int(population_size * 0.1)  # 例：10個体中1個体
next_generation = best_individuals[:num_elites].copy()
```

これにより、世代交代で性能が悪化することを防ぎます。

#### ステップ7: 次世代の構成

```python
next_generation = []

# 1. エリート保存
next_generation += elites

# 2. 残りを選択・交叉・突然変異で埋める
while len(next_generation) < population_size:
    parent1 = select_tournament()
    parent2 = select_tournament()
    offspring1, offspring2 = crossover(parent1, parent2)
    offspring1.mutate()
    offspring2.mutate()
    next_generation += [offspring1, offspring2]
```

### 3. 早期停止（Early Stopping）

無駄な計算を避けるため、改善が見られなくなったら進化を停止：

```python
if best_fitness の改善なし for N世代:
    進化を停止
```

例：`--early-stopping 7`を指定すると、7世代改善がなければ停止します。

### 4. なぜレイヤーごとに按分するのか？

LLMの各レイヤーは異なる役割を持ちます：

- **浅いレイヤー**: 基本的な言語パターン、構文理解
- **中間レイヤー**: 意味理解、文脈把握
- **深いレイヤー**: 高度な推論、タスク固有の知識

例えば、InstructモデルとBaseモデルをマージする場合：

```python
# 理想的なマージ比率の例
genes = [
    # 浅いレイヤー：Base寄り（基本的な言語能力を重視）
    [0.3, 0.7],  # Instruct=30%, Base=70%
    [0.4, 0.6],

    # 中間レイヤー：バランス
    [0.5, 0.5],
    [0.5, 0.5],

    # 深いレイヤー：Instruct寄り（指示追従能力を重視）
    [0.7, 0.3],  # Instruct=70%, Base=30%
    [0.8, 0.2],
]
```

遺伝的アルゴリズムがこの**最適な組み合わせを自動で発見**します。

### 5. 実際の進化の例

```
世代1:
  個体1: fitness=0.45 (ランダム)
  個体2: fitness=0.52 (ランダム) ← 最良
  ...
  平均: 0.48

世代2:（選択・交叉・突然変異後）
  個体1: fitness=0.52 (エリート保存)
  個体2: fitness=0.54 ← 新たな最良！
  ...
  平均: 0.51 ← 改善傾向

世代3:
  個体1: fitness=0.54 (エリート保存)
  個体2: fitness=0.56 ← さらに改善！
  ...
  平均: 0.53

...（繰り返し）

世代15:
  最良個体: fitness=0.68
  → このマージ比率を最終モデルとして保存
```

### 6. パラメータチューニングのヒント

**探索重視（多様性）：**
- `mutation_rate`を高く（0.2-0.3）
- `population_size`を大きく（20-30）
- `tournament_size`を小さく（2）

**収束重視（精度）：**
- `mutation_rate`を低く（0.05-0.1）
- `elitism_ratio`を高く（0.2-0.3）
- `tournament_size`を大きく（4-5）

## 出力

実行後、`output_dir`に以下が保存されます:

- `results.json`: 進化の履歴と最良個体の情報
- `best_individual.json`: 最良個体の詳細
- `best_merged_model/`: 最良モデル（`--save-best`指定時）
- `evolve_llm.log`: 実行ログ

## 技術スタック

- **mergekit**: モデルマージエンジン
- **transformers**: モデルの読み込み
- **lm-evaluation-harness**: ベンチマーク評価
- **NumPy**: 遺伝的アルゴリズムの実装

## 対応モデル

**重要**: 同一アーキテクチャ・同一サイズのモデルのみマージ可能です！

### ✅ マージ可能な組み合わせ

- Qwen2.5-7B × Qwen2.5-7B-Instruct
- Llama2-7B × Llama2-7B-chat
- Mistral-7B × Mistral-7B-Instruct
- 同じベースモデルのファインチューン同士

### ❌ マージ不可能な組み合わせ

- FinGPT (Llama2ベース) × Qwen2.5（異なるアーキテクチャ）
- Llama2-7B × Llama2-13B（異なるサイズ）
- 異なるモデルタイプ

### 金融モデルを使う場合

FinGPTなどのLoRAモデルは、先にベースモデルにマージが必要です：

```bash
# ステップ1: LoRAをマージ（必要なライブラリ: pip install peft）
python examples/lora_merge_example.py

# ステップ2: 進化的マージ
python main.py \
  --models "./models/fingpt_merged" "meta-llama/Llama-2-7b-chat-hf" \
  --eval-mode simple_multi_domain \
  --financial-weight 0.6 \
  --general-weight 0.4
```

**詳細**: [docs/MODEL_COMPATIBILITY.md](docs/MODEL_COMPATIBILITY.md) を参照

## パフォーマンスチューニング

### 高速化のヒント

1. **評価の簡素化**: 初期探索は`--eval-mode simple`や`--eval-limit`を使用
2. **キャッシュの活用**: `--cache-dir`で同じマージを再利用
3. **早期停止**: `--early-stopping`で無駄な世代をスキップ
4. **集団サイズ**: 小さい`--population-size`から始める

### ディスク容量の最適化

- `--no-save-merged-models`を使用して中間マージモデルを保存しない
  - 注意：キャッシングが無効になるため、実行時間は長くなる可能性あり
  - 最良モデル（`--save-best`）は影響を受けず保存される

### メモリ最適化

- 小さいモデルから試す（0.5B, 1.5Bなど）
- `dtype: float16`でメモリを節約
- GPU使用時は`torch.cuda.empty_cache()`が自動実行される

## トラブルシューティング

### mergekitが見つからない

```bash
pip install mergekit
```

### lm_evalが見つからない

```bash
pip install lm-eval
```

### メモリ不足

- より小さいモデルを使用
- `--population-size`を減らす
- `--eval-limit`を設定して評価を制限

### CUDA out of memory

- `--eval-mode simple`または`mock`を使用
- バッチサイズを調整（evaluationのコード内）

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！

## 参考文献

- [mergekit](https://github.com/cg123/mergekit)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)

## TODO

- [ ] 複数のマージ手法の比較
- [ ] より高度な選択戦略
- [ ] 適応的な突然変異率
- [ ] 分散評価のサポート
- [ ] ビジュアライゼーション機能
