"""
Example: Merging financial-specialized and general models using genetic algorithm.

This example demonstrates how to merge a financial domain model (e.g., FinGPT)
with a general purpose model (e.g., Qwen) to create a model that performs well
on both financial and general tasks.
"""

import sys
sys.path.insert(0, '..')

from genetic_algorithm import GeneticAlgorithm, Individual
from merge import LayerWiseMerger
from evaluation import (
    MultiDomainFitnessEvaluator,
    SimpleMultiDomainEvaluator,
    MultiDomainConfig,
    DomainConfig
)
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run financial + general model merging evolution."""

    # Configuration
    # Replace with actual financial and general model paths
    financial_model = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
    general_model = "Qwen/Qwen2.5-7B-Instruct"

    model_paths = [financial_model, general_model]
    num_layers = 32  # Adjust based on your models

    # Genetic algorithm parameters
    population_size = 12
    generations = 20

    # Domain configuration
    # 60% weight on financial tasks, 40% on general tasks
    multi_domain_config = MultiDomainConfig.create_financial_general(
        financial_weight=0.6,
        general_weight=0.4,
        limit=50  # Limit for faster evaluation during development
    )

    print("="*80)
    print("FINANCIAL + GENERAL MODEL MERGING")
    print("="*80)
    print(f"Financial Model: {financial_model}")
    print(f"General Model: {general_model}")
    print(f"Financial Task Weight: 60%")
    print(f"General Task Weight: 40%")
    print(f"Population Size: {population_size}")
    print(f"Generations: {generations}")
    print("="*80 + "\n")

    # Initialize components
    merger = LayerWiseMerger(
        model_paths=model_paths,
        cache_dir="./cache/financial_general",
        merge_method="linear"
    )

    # Choose evaluator
    use_simple_eval = True  # Set to False for full lm-eval

    if use_simple_eval:
        logger.info("Using SimpleMultiDomainEvaluator for faster evaluation")
        domain_configs = multi_domain_config.domains
        evaluator = SimpleMultiDomainEvaluator(domain_configs=domain_configs)
    else:
        logger.info("Using MultiDomainFitnessEvaluator with lm-eval")
        evaluator = MultiDomainFitnessEvaluator(multi_domain_config)

    # Create fitness function
    def fitness_func(individual: Individual) -> float:
        """Evaluate fitness across financial and general domains."""
        try:
            # Merge models
            merged_path = merger.merge(individual)

            # Evaluate on both domains
            fitness = evaluator.evaluate(merged_path, model_id=f"ind_{id(individual)}")

            return fitness
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return 0.0

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        population_size=population_size,
        num_layers=num_layers,
        num_models=len(model_paths),
        mutation_rate=0.12,
        mutation_strength=0.1,
        crossover_rate=0.85,
        elitism_ratio=0.15,
        tournament_size=4
    )

    # Run evolution
    logger.info(f"Starting evolution for {generations} generations")
    best_individual = ga.evolve(
        fitness_func=fitness_func,
        generations=generations,
        early_stopping_patience=8
    )

    # Print results
    print("\n" + "="*80)
    print("EVOLUTION COMPLETE")
    print("="*80)
    print(f"Best Combined Fitness: {best_individual.fitness:.4f}")

    # Print domain breakdown if available
    if hasattr(evaluator, 'get_domain_scores'):
        scores = evaluator.get_domain_scores(f"ind_{id(best_individual)}")
        if scores:
            print("\nDomain Scores:")
            for domain_name, score in scores.items():
                print(f"  {domain_name}: {score:.4f}")

    print("\nBest Layer-wise Weights (sample - first 5 layers):")
    for i in range(min(5, num_layers)):
        weights = best_individual.genes[i]
        print(f"  Layer {i}: Financial={weights[0]:.3f}, General={weights[1]:.3f}")

    print("="*80)

    # Save best individual
    best_individual.save("./output/best_financial_general_individual.json")
    logger.info("Best individual saved to ./output/best_financial_general_individual.json")


if __name__ == "__main__":
    main()
