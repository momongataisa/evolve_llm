"""Generate merge configuration for mergekit."""

import yaml
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from genetic_algorithm.individual import Individual


class MergeConfigGenerator:
    """Generate mergekit configuration from Individual genes."""

    def __init__(self, model_paths: List[str], base_model: str = None):
        """
        Initialize config generator.

        Args:
            model_paths: List of model paths to merge
            base_model: Optional base model path
        """
        self.model_paths = model_paths
        self.base_model = base_model or model_paths[0]
        self.num_models = len(model_paths)

    def generate_config(
        self,
        individual: Individual,
        merge_method: str = "linear",
        dtype: str = "float16"
    ) -> Dict[str, Any]:
        """
        Generate mergekit configuration from individual genes.

        Args:
            individual: Individual with layer-wise weights
            merge_method: Merge method (linear, slerp, etc.)
            dtype: Data type for merged model

        Returns:
            Mergekit configuration dictionary
        """
        assert individual.num_models == self.num_models, \
            f"Individual has {individual.num_models} models but {self.num_models} model paths provided"

        config = {
            "merge_method": merge_method,
            "base_model": self.base_model,
            "dtype": dtype,
            "slices": []
        }

        # Get layer information
        num_layers = individual.num_layers

        # Create layer-wise slices
        for layer_idx in range(num_layers):
            layer_weights = individual.genes[layer_idx]

            # Create models list with weights
            models = []
            for model_idx, model_path in enumerate(self.model_paths):
                models.append({
                    "model": model_path,
                    "parameters": {
                        "weight": float(layer_weights[model_idx])
                    }
                })

            # Add slice for this layer
            slice_config = {
                "sources": models,
                "layer_range": [layer_idx, layer_idx + 1]
            }
            config["slices"].append(slice_config)

        return config

    def generate_simple_config(
        self,
        individual: Individual,
        merge_method: str = "linear",
        dtype: str = "float16"
    ) -> Dict[str, Any]:
        """
        Generate simplified mergekit configuration with averaged weights.

        Args:
            individual: Individual with layer-wise weights
            merge_method: Merge method (linear, slerp, etc.)
            dtype: Data type for merged model

        Returns:
            Simplified mergekit configuration dictionary
        """
        # Average weights across all layers
        avg_weights = individual.genes.mean(axis=0)

        config = {
            "merge_method": merge_method,
            "base_model": self.base_model,
            "dtype": dtype,
            "models": []
        }

        for model_idx, model_path in enumerate(self.model_paths):
            config["models"].append({
                "model": model_path,
                "parameters": {
                    "weight": float(avg_weights[model_idx])
                }
            })

        return config

    def save_config(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            output_path: Path to save YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def generate_and_save(
        self,
        individual: Individual,
        output_path: str,
        merge_method: str = "linear",
        dtype: str = "float16",
        simple: bool = False
    ) -> str:
        """
        Generate and save configuration.

        Args:
            individual: Individual with layer-wise weights
            output_path: Path to save YAML file
            merge_method: Merge method
            dtype: Data type
            simple: Use simplified config

        Returns:
            Path to saved config file
        """
        if simple:
            config = self.generate_simple_config(individual, merge_method, dtype)
        else:
            config = self.generate_config(individual, merge_method, dtype)

        self.save_config(config, output_path)
        return output_path
