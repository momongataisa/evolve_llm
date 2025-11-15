"""Population management for genetic algorithm."""

import numpy as np
from typing import List, Optional
from .individual import Individual


class Population:
    """Manages a population of individuals for the genetic algorithm."""

    def __init__(
        self,
        population_size: int,
        num_layers: int,
        num_models: int = 2,
        individuals: Optional[List[Individual]] = None
    ):
        """
        Initialize a population.

        Args:
            population_size: Number of individuals in the population
            num_layers: Number of layers in the model
            num_models: Number of models to merge
            individuals: Optional pre-defined individuals
        """
        self.population_size = population_size
        self.num_layers = num_layers
        self.num_models = num_models

        if individuals is not None:
            self.individuals = individuals
        else:
            self.individuals = [
                Individual(num_layers, num_models)
                for _ in range(population_size)
            ]

    def evaluate(self, fitness_func):
        """
        Evaluate fitness for all individuals in the population.

        Args:
            fitness_func: Function that takes an Individual and returns fitness score
        """
        for individual in self.individuals:
            if individual.fitness is None:
                individual.fitness = fitness_func(individual)

    def select_tournament(self, tournament_size: int = 3) -> Individual:
        """
        Select an individual using tournament selection.

        Args:
            tournament_size: Number of individuals to compete in tournament

        Returns:
            Selected individual
        """
        tournament = np.random.choice(self.individuals, tournament_size, replace=False)
        return max(tournament, key=lambda ind: ind.fitness if ind.fitness else float('-inf'))

    def select_roulette(self) -> Individual:
        """
        Select an individual using roulette wheel selection.

        Returns:
            Selected individual
        """
        # Get fitness values
        fitnesses = np.array([ind.fitness if ind.fitness else 0 for ind in self.individuals])

        # Handle negative fitness by shifting
        if fitnesses.min() < 0:
            fitnesses = fitnesses - fitnesses.min()

        # Normalize to probabilities
        total_fitness = fitnesses.sum()
        if total_fitness == 0:
            # If all fitnesses are 0, use uniform selection
            return np.random.choice(self.individuals)

        probabilities = fitnesses / total_fitness
        return np.random.choice(self.individuals, p=probabilities)

    def get_best_individual(self) -> Individual:
        """
        Get the individual with the highest fitness.

        Returns:
            Best individual
        """
        return max(self.individuals, key=lambda ind: ind.fitness if ind.fitness else float('-inf'))

    def get_worst_individual(self) -> Individual:
        """
        Get the individual with the lowest fitness.

        Returns:
            Worst individual
        """
        return min(self.individuals, key=lambda ind: ind.fitness if ind.fitness else float('inf'))

    def get_average_fitness(self) -> float:
        """
        Calculate average fitness of the population.

        Returns:
            Average fitness
        """
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        return np.mean(fitnesses) if fitnesses else 0.0

    def get_fitness_std(self) -> float:
        """
        Calculate standard deviation of fitness.

        Returns:
            Standard deviation of fitness
        """
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        return np.std(fitnesses) if fitnesses else 0.0

    def sort_by_fitness(self):
        """Sort individuals by fitness in descending order."""
        self.individuals.sort(
            key=lambda ind: ind.fitness if ind.fitness else float('-inf'),
            reverse=True
        )

    def __repr__(self) -> str:
        return (
            f"Population(size={self.population_size}, "
            f"avg_fitness={self.get_average_fitness():.4f})"
        )
