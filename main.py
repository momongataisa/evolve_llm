"""Main entry point for evolutionary LLM model merging."""

import argparse
import logging
import json
from pathlib import Path
from typing import List
import sys

from genetic_algorithm import GeneticAlgorithm, Individual
from merge import LayerWiseMerger
from evaluation import (
    FitnessEvaluator,
    BenchmarkConfig,
    SimpleFitnessEvaluator,
    MockFitnessEvaluator,
    MultiDomainFitnessEvaluator,
    SimpleMultiDomainEvaluator,
    MultiDomainConfig,
    DomainConfig
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evolve_llm.log')
    ]
)
logger = logging.getLogger(__name__)


def get_model_num_layers(model_path: str) -> int:
    """
    Get number of layers from model config.

    Args:
        model_path: Path to model

    Returns:
        Number of layers
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        return config.num_hidden_layers
    except Exception as e:
        logger.warning(f"Failed to get num_layers from config: {e}")
        logger.info("Defaulting to 32 layers")
        return 32


def create_fitness_function(
    model_paths: List[str],
    merger: LayerWiseMerger,
    evaluator: FitnessEvaluator
):
    """
    Create fitness function for genetic algorithm.

    Args:
        model_paths: List of model paths
        merger: LayerWiseMerger instance
        evaluator: FitnessEvaluator instance

    Returns:
        Fitness function
    """
    def fitness_func(individual: Individual) -> float:
        """Evaluate fitness of an individual."""
        try:
            # Merge models based on individual's genes
            logger.info(f"Merging models for individual evaluation")
            merged_path = merger.merge(individual)

            # Evaluate merged model
            logger.info(f"Evaluating merged model at {merged_path}")
            fitness = evaluator.evaluate(merged_path, model_id=str(hash(individual.genes.tobytes())))

            return fitness

        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return 0.0

    return fitness_func


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evolutionary LLM Model Merging with Genetic Algorithm"
    )

    # Model arguments
    parser.add_argument(
        "--models",
        nargs='+',
        required=True,
        help="Paths to models to merge (2 or more)"
    )

    # GA parameters
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Population size (default: 10)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations (default: 20)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation rate (default: 0.1)"
    )
    parser.add_argument(
        "--mutation-strength",
        type=float,
        default=0.1,
        help="Mutation strength (default: 0.1)"
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="Crossover rate (default: 0.8)"
    )
    parser.add_argument(
        "--elitism-ratio",
        type=float,
        default=0.1,
        help="Elitism ratio (default: 0.1)"
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Tournament size for selection (default: 3)"
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-mode",
        choices=['full', 'simple', 'mock', 'multi_domain', 'simple_multi_domain'],
        default='simple',
        help="Evaluation mode: full (lm-eval), simple (perplexity), mock (testing), "
             "multi_domain (multi-domain with lm-eval), simple_multi_domain (multi-domain simple)"
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        default=["arc_easy", "hellaswag"],
        help="Evaluation tasks (default: arc_easy hellaswag)"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)"
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Limit number of evaluation examples for faster testing"
    )

    # Multi-domain evaluation arguments
    parser.add_argument(
        "--financial-weight",
        type=float,
        default=0.6,
        help="Weight for financial domain tasks (default: 0.6)"
    )
    parser.add_argument(
        "--general-weight",
        type=float,
        default=0.4,
        help="Weight for general domain tasks (default: 0.4)"
    )
    parser.add_argument(
        "--financial-tasks",
        nargs='+',
        default=["finqa", "convfinqa", "fiqa", "fpb"],
        help="Financial domain tasks"
    )
    parser.add_argument(
        "--general-tasks",
        nargs='+',
        default=["arc_easy", "hellaswag", "winogrande", "mmlu"],
        help="General domain tasks"
    )

    # Merge arguments
    parser.add_argument(
        "--merge-method",
        choices=['linear', 'slerp', 'ties', 'dare'],
        default='linear',
        help="Merge method (default: linear)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./merged_models",
        help="Cache directory for merged models"
    )
    parser.add_argument(
        "--no-save-merged-models",
        action='store_true',
        help="Do not save intermediate merged models (saves disk space, disables caching)"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-best",
        action='store_true',
        help="Save the best merged model"
    )

    # Other arguments
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Early stopping patience (generations without improvement)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers (auto-detected if not provided)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    import numpy as np
    np.random.seed(args.seed)

    # Validate models
    if len(args.models) < 2:
        logger.error("At least 2 models are required for merging")
        sys.exit(1)

    logger.info(f"Starting evolutionary model merging with {len(args.models)} models")
    logger.info(f"Models: {args.models}")

    # Get number of layers
    if args.num_layers is None:
        num_layers = get_model_num_layers(args.models[0])
        logger.info(f"Auto-detected {num_layers} layers")
    else:
        num_layers = args.num_layers

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize merger
    logger.info("Initializing model merger")
    save_merged = not args.no_save_merged_models
    if not save_merged:
        logger.info("Merged models will NOT be saved (using temporary directories)")
    merger = LayerWiseMerger(
        model_paths=args.models,
        cache_dir=args.cache_dir,
        merge_method=args.merge_method,
        save_merged_models=save_merged
    )

    # Initialize evaluator
    logger.info(f"Initializing evaluator (mode: {args.eval_mode})")
    if args.eval_mode == 'full':
        benchmark_config = BenchmarkConfig(
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.eval_limit
        )
        evaluator = FitnessEvaluator(benchmark_config)
    elif args.eval_mode == 'simple':
        evaluator = SimpleFitnessEvaluator()
    elif args.eval_mode == 'multi_domain':
        # Multi-domain evaluation with lm-eval
        multi_domain_config = MultiDomainConfig.create_financial_general(
            financial_weight=args.financial_weight,
            general_weight=args.general_weight,
            limit=args.eval_limit
        )
        # Override tasks if specified
        if args.financial_tasks:
            multi_domain_config.domains[0].tasks = args.financial_tasks
        if args.general_tasks:
            multi_domain_config.domains[1].tasks = args.general_tasks

        evaluator = MultiDomainFitnessEvaluator(multi_domain_config)
        logger.info(f"Multi-domain weights: Financial={args.financial_weight}, General={args.general_weight}")
    elif args.eval_mode == 'simple_multi_domain':
        # Simple multi-domain evaluation (faster, no lm-eval)
        financial_domain = DomainConfig(
            name="financial",
            tasks=args.financial_tasks,
            weight=args.financial_weight,
            num_fewshot=3,
            limit=args.eval_limit
        )
        general_domain = DomainConfig(
            name="general",
            tasks=args.general_tasks,
            weight=args.general_weight,
            num_fewshot=5,
            limit=args.eval_limit
        )
        evaluator = SimpleMultiDomainEvaluator(
            domain_configs=[financial_domain, general_domain]
        )
        logger.info(f"Simple multi-domain weights: Financial={args.financial_weight}, General={args.general_weight}")
    else:  # mock
        evaluator = MockFitnessEvaluator()

    # Create fitness function
    fitness_func = create_fitness_function(args.models, merger, evaluator)

    # Initialize genetic algorithm
    logger.info("Initializing genetic algorithm")
    ga = GeneticAlgorithm(
        population_size=args.population_size,
        num_layers=num_layers,
        num_models=len(args.models),
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        elitism_ratio=args.elitism_ratio,
        tournament_size=args.tournament_size
    )

    # Run evolution
    logger.info(f"Starting evolution for {args.generations} generations")
    best_individual = ga.evolve(
        fitness_func=fitness_func,
        generations=args.generations,
        early_stopping_patience=args.early_stopping
    )

    # Save results
    logger.info("Saving results")
    results = {
        'best_fitness': best_individual.fitness,
        'best_genes': best_individual.genes.tolist(),
        'history': ga.get_history(),
        'config': vars(args)
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Save best individual
    best_individual_path = output_dir / 'best_individual.json'
    best_individual.save(str(best_individual_path))
    logger.info(f"Best individual saved to {best_individual_path}")

    # Save best merged model if requested
    if args.save_best:
        logger.info("Merging and saving best model")
        best_model_path = output_dir / 'best_merged_model'
        merger.merge(best_individual, output_path=str(best_model_path))
        logger.info(f"Best merged model saved to {best_model_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVOLUTION COMPLETE")
    print("="*80)
    print(f"Best Fitness: {best_individual.fitness:.4f}")
    print(f"Generations: {len(ga.get_history())}")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
