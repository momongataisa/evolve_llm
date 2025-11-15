# Evolve LLM - 遺伝的アルゴリズムによるLLMモデルマージ

Qwenなどの大規模言語モデル（LLM）を遺伝的アルゴリズムで最適にマージするためのフレームワークです。レイヤーごとの按分率を進化させることで、最適なモデル統合を実現します。

## 特徴

- **レイヤーごとの按分率最適化**: 各レイヤーで異なるマージ比率を設定し、最適な組み合わせを探索
- **遺伝的アルゴリズム**: 進化的手法による効率的な探索
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
│   └── fitness.py         # ベンチマーク評価
├── examples/              # 使用例
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
- `--cache-dir`: マージ済みモデルのキャッシュディレクトリ

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

## 仕組み

### 1. 個体の表現

各個体は、モデルの全レイヤーについて、マージする各モデルの重み（按分率）を持ちます。

```python
# 例: 2モデル、3レイヤーの場合
genes = [
    [0.7, 0.3],  # レイヤー0: モデル1が70%, モデル2が30%
    [0.5, 0.5],  # レイヤー1: 均等
    [0.3, 0.7],  # レイヤー2: モデル1が30%, モデル2が70%
]
```

### 2. 進化のプロセス

1. **初期化**: ランダムな按分率で個体を生成
2. **評価**: 各個体をマージして性能評価
3. **選択**: 高性能な個体を親として選択（トーナメント選択）
4. **交叉**: 親の遺伝子を組み合わせて子を生成
5. **突然変異**: ランダムに按分率を変動
6. **次世代**: エリート保存 + 新個体で次世代を構成
7. 2-6を繰り返す

### 3. 適応度評価

- **Full モード**: MMLU、HellaSwag、ARC等のベンチマークで評価
- **Simple モード**: パープレキシティで簡易評価
- **Mock モード**: テスト用のランダム評価

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

同一アーキテクチャのモデルであればマージ可能です:

- Qwen2.5シリーズ
- Llama系モデル
- Mistral系モデル
- その他のCausal LM

## パフォーマンスチューニング

### 高速化のヒント

1. **評価の簡素化**: 初期探索は`--eval-mode simple`や`--eval-limit`を使用
2. **キャッシュの活用**: `--cache-dir`で同じマージを再利用
3. **早期停止**: `--early-stopping`で無駄な世代をスキップ
4. **集団サイズ**: 小さい`--population-size`から始める

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
