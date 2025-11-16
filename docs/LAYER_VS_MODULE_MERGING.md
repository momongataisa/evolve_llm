# レイヤーとモジュールレベルのマージについて

## 現在の実装（修正後）

### レイヤーごとのマージ ✅

修正により、**レイヤーごとに異なる重み付け**でマージできるようになりました：

```yaml
# 生成されるmergekit設定（例: 28層モデル）
slices:
  - sources:
      - model: "model_a"
        parameters:
          weight: 0.7  # レイヤー0: モデルAが70%
      - model: "model_b"
        parameters:
          weight: 0.3  # レイヤー0: モデルBが30%
    layer_range: [0, 1]

  - sources:
      - model: "model_a"
        parameters:
          weight: 0.6  # レイヤー1: モデルAが60%
      - model: "model_b"
        parameters:
          weight: 0.4  # レイヤー1: モデルBが40%
    layer_range: [1, 2]

  # ... 各レイヤーごとに設定
```

**これにより、各レイヤー全体（self-attention + MLP + LayerNormなど）が指定された重みでマージされます。**

## レイヤー内のモジュール構成

LLMの1つのTransformerレイヤーは複数のモジュールで構成されています：

```python
Layer 0 (transformer.layer[0]):
  ├── Self-Attention
  │   ├── q_proj (Query projection)
  │   ├── k_proj (Key projection)
  │   ├── v_proj (Value projection)
  │   └── o_proj (Output projection)
  ├── MLP
  │   ├── gate_proj (or fc1)
  │   ├── up_proj
  │   └── down_proj (or fc2)
  ├── input_layernorm
  └── post_attention_layernorm
```

### 現在のマージ動作

**レイヤーごとのマージ**では、上記のすべてのモジュールが**同じ重み**でマージされます：

```python
# 例: Layer 0を70:30でマージ
Layer 0の全モジュール（q_proj, k_proj, v_proj, o_proj, mlp, layernorm等）
すべてが70:30の比率でマージされる
```

## モジュールレベルのマージ ❓

より細かい制御として、**モジュールごとに異なる重み**でマージすることも理論的には可能です：

```python
# 仮想的な例
Layer 0:
  q_proj: 70:30
  k_proj: 80:20  # ← 異なる重み
  v_proj: 60:40  # ← 異なる重み
  o_proj: 70:30
  mlp: 50:50     # ← 異なる重み
```

### メリット

- **より細かい制御**: AttentionとMLPで異なる戦略を取れる
- **専門化の可能性**: 金融モデルのAttentionは強く、汎用モデルのMLPは強く、など

### デメリット

- **探索空間の爆発**:
  - 現在: 28層 × 2モデル = 56パラメータ
  - モジュールレベル: 28層 × 8モジュール × 2モデル = 448パラメータ
- **計算コストの増加**: 探索空間が8倍になる
- **過学習リスク**: パラメータが多すぎて汎化しない可能性
- **mergekitの制約**: 現在のmergekitではモジュールレベルのスライスが困難

## 推奨アプローチ

### 現在の実装（修正後）で十分な理由

1. **レイヤーごとの重み付けでも強力**
   - 浅いレイヤー: 基本的な言語理解
   - 中間レイヤー: ドメイン知識
   - 深いレイヤー: タスク固有の推論
   - これらのレベルで異なる重み付けができる

2. **計算効率**
   - 56パラメータで探索可能
   - 合理的な時間で収束

3. **実用性**
   - ほとんどの用途で十分な粒度
   - 研究でもレイヤーレベルが一般的

### モジュールレベルが必要な場合

以下のような特殊なケースのみ：

1. **Attentionに特化したマージ**: Self-AttentionだけモデルAを強く
2. **MLPに特化したマージ**: Feed-forwardだけモデルBを強く
3. **研究目的**: モジュールレベルの影響を調査

## モジュールレベルマージの実装方法（将来的）

mergekitの`passthrough`機能や個別テンソル指定を使う必要があります：

```yaml
# 複雑な設定例（現在未実装）
slices:
  # Layer 0のAttention
  - sources:
      - model: "model_a"
        parameters:
          weight: 0.8
          tensor_name: "model.layers.0.self_attn.*"

  # Layer 0のMLP
  - sources:
      - model: "model_a"
        parameters:
          weight: 0.5
          tensor_name: "model.layers.0.mlp.*"
```

## 結論

**現在の実装（レイヤーごと）で十分です：**

- ✅ 各レイヤー全体を指定した重みでマージ
- ✅ レイヤー内のすべてのモジュール（q_proj, k_proj, v_proj, mlp等）が同じ重みでマージされる
- ✅ 28層 × 異なる重み = 十分な柔軟性
- ✅ 計算効率が良い
- ✅ 遺伝的アルゴリズムで最適化可能

**モジュールレベルのマージは：**

- 🔬 研究目的や特殊用途向け
- ⚠️ 実装が複雑
- ⚠️ 計算コストが高い
- ⚠️ mergekitの制約がある

## 実際の使用例

```bash
# 現在の実装で十分強力
python main.py \
  --models "model_a" "model_b" \
  --eval-mode simple_multi_domain \
  --population-size 12 \
  --generations 20

# 結果例:
# Layer 0: 30% model_a, 70% model_b (基本的な言語理解はmodel_b重視)
# Layer 14: 60% model_a, 40% model_b (中間層でバランス)
# Layer 27: 80% model_a, 20% model_b (深い層はmodel_a重視)
```

この粒度で、金融特化と汎用モデルのバランスを最適化できます！
