"""Basic example of evolutionary model merging."""

import sys
sys.path.insert(0, '..')

from genetic_algorithm import GeneticAlgorithm, Individual
from merge import LayerWiseMerger
from evaluation import MockFitnessEvaluator


def main():
    """Run basic evolution example with mock evaluation."""
    # Configuration
    model_paths = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B"
    ]
    num_layers = 24  # Qwen 0.5B has 24 layers
    population_size = 8
    generations = 10

    print("="*80)
    print("EVOLUTIONARY LLM MODEL MERGING - BASIC EXAMPLE")
    print("="*80)
    print(f"Models: {model_paths}")
    print(f"Population size: {population_size}")
    print(f"Generations: {generations}")
    print("="*80 + "\n")

    # Initialize components
    merger = LayerWiseMerger(
        model_paths=model_paths,
        cache_dir="./demo_merged_models"
    )

    evaluator = MockFitnessEvaluator()

    # Create fitness function
    def fitness_func(individual: Individual) -> float:
        """Mock fitness function for demonstration."""
        # In real scenario, this would merge and evaluate
        # For demo, we just return mock fitness
        return evaluator.evaluate(f"mock_model_{hash(individual.genes.tobytes())}")

    # Initialize and run genetic algorithm
    ga = GeneticAlgorithm(
        population_size=population_size,
        num_layers=num_layers,
        num_models=len(model_paths),
        mutation_rate=0.15,
        mutation_strength=0.1,
        crossover_rate=0.8,
        elitism_ratio=0.2
    )

    print("Starting evolution...\n")
    best_individual = ga.evolve(
        fitness_func=fitness_func,
        generations=generations,
        early_stopping_patience=5
    )

    # Print results
    print("\n" + "="*80)
    print("EVOLUTION COMPLETE")
    print("="*80)
    print(f"Best Fitness: {best_individual.fitness:.4f}")
    print(f"\nBest layer-wise weights (first 5 layers):")
    for i in range(min(5, num_layers)):
        weights = best_individual.genes[i]
        print(f"  Layer {i}: Model0={weights[0]:.3f}, Model1={weights[1]:.3f}")
    print("="*80)


if __name__ == "__main__":
    main()
