"""
LoRAモデルをベースモデルにマージしてから進化的マージを実行する例

FinGPTなどのLoRAモデルを扱う場合の推奨ワークフロー：
1. LoRAアダプターをベースモデルにマージ
2. マージ済みモデルを使って進化的マージを実行
"""

import sys
import os
from pathlib import Path


def merge_lora_with_base(base_model_name: str, lora_model_name: str, output_path: str):
    """
    LoRAアダプターをベースモデルにマージ

    Args:
        base_model_name: ベースモデルのパス
        lora_model_name: LoRAモデルのパス
        output_path: マージ済みモデルの保存先
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        print(f"Loading LoRA adapter: {lora_model_name}")
        model = PeftModel.from_pretrained(base_model, lora_model_name)

        print("Merging LoRA with base model...")
        merged_model = model.merge_and_unload()

        print(f"Saving merged model to: {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        print("LoRA merge complete!")

        # クリーンアップ
        del base_model, model, merged_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path

    except ImportError:
        print("ERROR: peft library not found. Install with: pip install peft")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to merge LoRA: {e}")
        sys.exit(1)


def run_evolutionary_merge(
    model1_path: str,
    model2_path: str,
    financial_weight: float = 0.6,
    general_weight: float = 0.4,
    population_size: int = 12,
    generations: int = 20
):
    """
    進化的モデルマージを実行

    Args:
        model1_path: モデル1のパス（金融特化など）
        model2_path: モデル2のパス（汎用など）
        financial_weight: 金融タスクの重み
        general_weight: 汎用タスクの重み
        population_size: 集団サイズ
        generations: 世代数
    """
    sys.path.insert(0, '..')

    from genetic_algorithm import GeneticAlgorithm, Individual
    from merge import LayerWiseMerger
    from evaluation import SimpleMultiDomainEvaluator, DomainConfig

    print("\n" + "="*80)
    print("EVOLUTIONARY MODEL MERGING")
    print("="*80)
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Financial weight: {financial_weight}")
    print(f"General weight: {general_weight}")
    print("="*80 + "\n")

    # モデルのレイヤー数を取得
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model1_path)
    num_layers = config.num_hidden_layers
    print(f"Detected {num_layers} layers")

    # コンポーネント初期化
    model_paths = [model1_path, model2_path]

    merger = LayerWiseMerger(
        model_paths=model_paths,
        cache_dir="./cache/lora_evolution"
    )

    # ドメイン設定
    financial_domain = DomainConfig(
        name="financial",
        tasks=["finqa"],
        weight=financial_weight,
        num_fewshot=3,
        limit=50
    )

    general_domain = DomainConfig(
        name="general",
        tasks=["arc_easy"],
        weight=general_weight,
        num_fewshot=5,
        limit=50
    )

    evaluator = SimpleMultiDomainEvaluator(
        domain_configs=[financial_domain, general_domain]
    )

    # 適応度関数
    def fitness_func(individual: Individual) -> float:
        merged_path = merger.merge(individual)
        return evaluator.evaluate(merged_path, model_id=f"ind_{id(individual)}")

    # 遺伝的アルゴリズム実行
    ga = GeneticAlgorithm(
        population_size=population_size,
        num_layers=num_layers,
        num_models=len(model_paths),
        mutation_rate=0.12,
        mutation_strength=0.1
    )

    print("Starting evolution...")
    best_individual = ga.evolve(
        fitness_func=fitness_func,
        generations=generations,
        early_stopping_patience=8
    )

    # 結果
    print("\n" + "="*80)
    print("EVOLUTION COMPLETE")
    print("="*80)
    print(f"Best fitness: {best_individual.fitness:.4f}")
    print("="*80)

    # 保存
    output_dir = Path("./output/lora_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_individual.save(str(output_dir / "best_individual.json"))
    print(f"Results saved to {output_dir}")


def main():
    """メイン実行"""

    # 設定
    base_model = "meta-llama/Llama-2-7b-hf"
    lora_model = "FinGPT/fingpt-mt_llama2-7b_lora"
    general_model = "meta-llama/Llama-2-7b-chat-hf"

    # ステップ1: LoRAをマージ
    print("="*80)
    print("STEP 1: Merging LoRA with base model")
    print("="*80)

    merged_lora_path = "./models/fingpt_merged"

    # すでにマージ済みならスキップ
    if Path(merged_lora_path).exists():
        print(f"Merged model already exists at {merged_lora_path}, skipping merge...")
    else:
        merge_lora_with_base(base_model, lora_model, merged_lora_path)

    # ステップ2: 進化的マージ
    print("\n" + "="*80)
    print("STEP 2: Evolutionary model merging")
    print("="*80)

    run_evolutionary_merge(
        model1_path=merged_lora_path,
        model2_path=general_model,
        financial_weight=0.6,
        general_weight=0.4,
        population_size=10,
        generations=15
    )


if __name__ == "__main__":
    main()
