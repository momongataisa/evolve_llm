"""Individual representation for genetic algorithm."""

import numpy as np
from typing import List, Optional
import json


class Individual:
    """
    Represents an individual in the genetic algorithm population.

    Each individual contains layer-wise merge weights for combining multiple LLM models.
    """

    def __init__(self, num_layers: int, num_models: int = 2, genes: Optional[np.ndarray] = None):
        """
        Initialize an individual.

        Args:
            num_layers: Number of layers in the model
            num_models: Number of models to merge (default: 2)
            genes: Optional pre-defined genes. If None, random genes are generated.
        """
        self.num_layers = num_layers
        self.num_models = num_models
        self.fitness: Optional[float] = None

        if genes is not None:
            self.genes = genes
        else:
            # Initialize random weights for each layer
            # Shape: (num_layers, num_models)
            # Each layer has weights that sum to 1.0
            self.genes = self._initialize_random_genes()

    def _initialize_random_genes(self) -> np.ndarray:
        """
        Initialize random genes ensuring weights sum to 1.0 for each layer.

        Returns:
            Random genes normalized per layer
        """
        genes = np.random.random((self.num_layers, self.num_models))
        # Normalize each layer's weights to sum to 1.0
        genes = genes / genes.sum(axis=1, keepdims=True)
        return genes

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """
        Mutate the individual's genes.

        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Magnitude of mutation (std dev of normal distribution)
        """
        for i in range(self.num_layers):
            if np.random.random() < mutation_rate:
                # Add Gaussian noise to the weights
                noise = np.random.normal(0, mutation_strength, self.num_models)
                self.genes[i] += noise

                # Ensure weights remain in valid range [0, 1]
                self.genes[i] = np.clip(self.genes[i], 0.0, 1.0)

                # Re-normalize to sum to 1.0
                self.genes[i] = self.genes[i] / self.genes[i].sum()

    @staticmethod
    def crossover(parent1: 'Individual', parent2: 'Individual') -> tuple['Individual', 'Individual']:
        """
        Perform crossover between two parents to create two offspring.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Tuple of two offspring individuals
        """
        assert parent1.num_layers == parent2.num_layers
        assert parent1.num_models == parent2.num_models

        num_layers = parent1.num_layers
        num_models = parent1.num_models

        # Single-point crossover
        crossover_point = np.random.randint(1, num_layers)

        # Create offspring
        offspring1_genes = np.vstack([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])

        offspring2_genes = np.vstack([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])

        offspring1 = Individual(num_layers, num_models, offspring1_genes)
        offspring2 = Individual(num_layers, num_models, offspring2_genes)

        return offspring1, offspring2

    def to_dict(self) -> dict:
        """
        Convert individual to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            'num_layers': self.num_layers,
            'num_models': self.num_models,
            'genes': self.genes.tolist(),
            'fitness': self.fitness
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Individual':
        """
        Create individual from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Individual instance
        """
        genes = np.array(data['genes'])
        individual = cls(data['num_layers'], data['num_models'], genes)
        individual.fitness = data.get('fitness')
        return individual

    def save(self, filepath: str):
        """
        Save individual to JSON file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Individual':
        """
        Load individual from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            Individual instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return f"Individual(layers={self.num_layers}, fitness={self.fitness:.4f if self.fitness else 'N/A'})"
