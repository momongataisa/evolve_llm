"""Genetic algorithm evolution logic."""

import numpy as np
from typing import Callable, Optional, Dict, List
import logging
from .individual import Individual
from .population import Population


logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Genetic algorithm for evolving LLM merge configurations."""

    def __init__(
        self,
        population_size: int,
        num_layers: int,
        num_models: int = 2,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_ratio: float = 0.1,
        tournament_size: int = 3,
        selection_method: str = 'tournament'
    ):
        """
        Initialize the genetic algorithm.

        Args:
            population_size: Number of individuals in population
            num_layers: Number of layers in the model
            num_models: Number of models to merge
            mutation_rate: Probability of mutation
            mutation_strength: Strength of mutation
            crossover_rate: Probability of crossover
            elitism_ratio: Ratio of top individuals to preserve
            tournament_size: Size of tournament for selection
            selection_method: 'tournament' or 'roulette'
        """
        self.population_size = population_size
        self.num_layers = num_layers
        self.num_models = num_models
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.selection_method = selection_method

        self.population: Optional[Population] = None
        self.generation = 0
        self.history: List[Dict] = []

    def initialize_population(self):
        """Initialize the population with random individuals."""
        self.population = Population(
            self.population_size,
            self.num_layers,
            self.num_models
        )
        logger.info(f"Initialized population with {self.population_size} individuals")

    def evolve(
        self,
        fitness_func: Callable[[Individual], float],
        generations: int,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: float = 0.001
    ) -> Individual:
        """
        Run the genetic algorithm evolution.

        Args:
            fitness_func: Function to evaluate individual fitness
            generations: Number of generations to evolve
            early_stopping_patience: Stop if no improvement for N generations
            early_stopping_threshold: Minimum improvement to reset patience counter

        Returns:
            Best individual found
        """
        if self.population is None:
            self.initialize_population()

        best_fitness_ever = float('-inf')
        generations_without_improvement = 0

        for gen in range(generations):
            self.generation = gen + 1
            logger.info(f"\n=== Generation {self.generation}/{generations} ===")

            # Evaluate fitness
            self.population.evaluate(fitness_func)

            # Get statistics
            best_individual = self.population.get_best_individual()
            avg_fitness = self.population.get_average_fitness()
            fitness_std = self.population.get_fitness_std()

            # Log statistics
            logger.info(f"Best fitness: {best_individual.fitness:.4f}")
            logger.info(f"Average fitness: {avg_fitness:.4f}")
            logger.info(f"Fitness std: {fitness_std:.4f}")

            # Save history
            self.history.append({
                'generation': self.generation,
                'best_fitness': best_individual.fitness,
                'avg_fitness': avg_fitness,
                'fitness_std': fitness_std
            })

            # Check for improvement
            if best_individual.fitness > best_fitness_ever + early_stopping_threshold:
                best_fitness_ever = best_individual.fitness
                generations_without_improvement = 0
                logger.info(f"New best fitness: {best_fitness_ever:.4f}")
            else:
                generations_without_improvement += 1

            # Early stopping
            if early_stopping_patience and generations_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping: No improvement for {early_stopping_patience} generations"
                )
                break

            # Create next generation
            if gen < generations - 1:
                self._create_next_generation()

        return self.population.get_best_individual()

    def _create_next_generation(self):
        """Create the next generation through selection, crossover, and mutation."""
        # Sort population by fitness
        self.population.sort_by_fitness()

        # Elitism: preserve top individuals
        num_elites = max(1, int(self.population_size * self.elitism_ratio))
        next_generation = self.population.individuals[:num_elites].copy()

        logger.info(f"Preserving {num_elites} elite individuals")

        # Generate offspring
        while len(next_generation) < self.population_size:
            # Selection
            if self.selection_method == 'tournament':
                parent1 = self.population.select_tournament(self.tournament_size)
                parent2 = self.population.select_tournament(self.tournament_size)
            else:  # roulette
                parent1 = self.population.select_roulette()
                parent2 = self.population.select_roulette()

            # Crossover
            if np.random.random() < self.crossover_rate:
                offspring1, offspring2 = Individual.crossover(parent1, parent2)
            else:
                # No crossover, just copy parents
                offspring1 = Individual(
                    self.num_layers,
                    self.num_models,
                    parent1.genes.copy()
                )
                offspring2 = Individual(
                    self.num_layers,
                    self.num_models,
                    parent2.genes.copy()
                )

            # Mutation
            offspring1.mutate(self.mutation_rate, self.mutation_strength)
            offspring2.mutate(self.mutation_rate, self.mutation_strength)

            # Add to next generation
            next_generation.append(offspring1)
            if len(next_generation) < self.population_size:
                next_generation.append(offspring2)

        # Update population
        self.population.individuals = next_generation[:self.population_size]

    def get_history(self) -> List[Dict]:
        """
        Get evolution history.

        Returns:
            List of generation statistics
        """
        return self.history
