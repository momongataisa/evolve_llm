"""Fitness evaluation for merged models."""

import subprocess
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import tempfile


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    tasks: List[str] = field(default_factory=lambda: ["arc_easy", "hellaswag", "mmlu"])
    num_fewshot: int = 5
    batch_size: int = 8
    device: str = "cuda"
    limit: Optional[int] = None  # Limit number of examples for faster evaluation
    use_cache: bool = True
    output_path: Optional[str] = None


class FitnessEvaluator:
    """Evaluates fitness of merged models using benchmarks."""

    def __init__(
        self,
        benchmark_config: Optional[BenchmarkConfig] = None,
        cache_results: bool = True
    ):
        """
        Initialize fitness evaluator.

        Args:
            benchmark_config: Configuration for benchmarks
            cache_results: Whether to cache evaluation results
        """
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        self.cache_results = cache_results
        self.results_cache: Dict[str, float] = {}

    def evaluate(self, model_path: str, model_id: Optional[str] = None) -> float:
        """
        Evaluate a model and return fitness score.

        Args:
            model_path: Path to the model to evaluate
            model_id: Optional identifier for caching

        Returns:
            Fitness score (higher is better)
        """
        # Check cache
        if model_id and self.cache_results and model_id in self.results_cache:
            logger.info(f"Using cached fitness for {model_id}")
            return self.results_cache[model_id]

        # Run evaluation
        try:
            results = self._run_lm_eval(model_path)
            fitness = self._calculate_fitness(results)

            # Cache result
            if model_id and self.cache_results:
                self.results_cache[model_id] = fitness

            logger.info(f"Fitness for {model_path}: {fitness:.4f}")
            return fitness

        except Exception as e:
            logger.error(f"Evaluation failed for {model_path}: {e}")
            return 0.0

    def _run_lm_eval(self, model_path: str) -> Dict[str, Any]:
        """
        Run lm-evaluation-harness on the model.

        Args:
            model_path: Path to model

        Returns:
            Evaluation results
        """
        # Prepare output path
        if self.benchmark_config.output_path:
            output_path = Path(self.benchmark_config.output_path)
        else:
            output_path = Path(tempfile.mkdtemp()) / "results.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", ",".join(self.benchmark_config.tasks),
            "--num_fewshot", str(self.benchmark_config.num_fewshot),
            "--batch_size", str(self.benchmark_config.batch_size),
            "--device", self.benchmark_config.device,
            "--output_path", str(output_path.parent),
        ]

        if self.benchmark_config.limit:
            cmd.extend(["--limit", str(self.benchmark_config.limit)])

        try:
            logger.info(f"Running evaluation: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Parse results
            results_file = list(output_path.parent.glob("results_*.json"))
            if not results_file:
                # Try loading from output_path directly
                with open(output_path, 'r') as f:
                    results = json.load(f)
            else:
                with open(results_file[0], 'r') as f:
                    results = json.load(f)

            return results

        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out")
            return {}

        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e.stderr}")
            return {}

        except FileNotFoundError:
            logger.error("lm_eval not found. Please install lm-evaluation-harness.")
            raise RuntimeError(
                "lm-evaluation-harness not installed. "
                "Install with: pip install lm-eval"
            )

    def _calculate_fitness(self, results: Dict[str, Any]) -> float:
        """
        Calculate fitness score from evaluation results.

        Args:
            results: Evaluation results from lm-eval

        Returns:
            Fitness score
        """
        if not results or "results" not in results:
            return 0.0

        scores = []
        for task_name in self.benchmark_config.tasks:
            if task_name in results["results"]:
                task_results = results["results"][task_name]

                # Try different metric names
                for metric in ["acc", "acc_norm", "exact_match", "em"]:
                    if metric in task_results:
                        scores.append(task_results[metric])
                        break

        if not scores:
            logger.warning("No valid scores found in results")
            return 0.0

        # Average across tasks
        fitness = sum(scores) / len(scores)
        return fitness


class SimpleFitnessEvaluator(FitnessEvaluator):
    """Simplified fitness evaluator for testing purposes."""

    def __init__(self):
        """Initialize simple evaluator."""
        super().__init__(cache_results=False)

    def evaluate(self, model_path: str, model_id: Optional[str] = None) -> float:
        """
        Simple evaluation based on perplexity or basic metrics.

        Args:
            model_path: Path to model
            model_id: Model identifier

        Returns:
            Fitness score
        """
        try:
            # Try to load model and calculate simple metric
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Simple test: calculate perplexity on a small sample
            test_text = "The quick brown fox jumps over the lazy dog."
            inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                perplexity = torch.exp(outputs.loss).item()

            # Convert perplexity to fitness (lower perplexity = higher fitness)
            fitness = 100.0 / perplexity

            logger.info(f"Perplexity: {perplexity:.2f}, Fitness: {fitness:.4f}")

            # Cleanup
            del model
            torch.cuda.empty_cache()

            return fitness

        except Exception as e:
            logger.error(f"Simple evaluation failed: {e}")
            return 0.0


class MockFitnessEvaluator(FitnessEvaluator):
    """Mock evaluator for testing without actual model evaluation."""

    def __init__(self):
        """Initialize mock evaluator."""
        super().__init__(cache_results=False)

    def evaluate(self, model_path: str, model_id: Optional[str] = None) -> float:
        """
        Return random fitness for testing.

        Args:
            model_path: Path to model
            model_id: Model identifier

        Returns:
            Random fitness score
        """
        import numpy as np
        # Generate consistent random score based on path hash
        np.random.seed(hash(model_path) % 2**32)
        fitness = np.random.uniform(0.5, 0.9)
        logger.info(f"Mock fitness for {model_path}: {fitness:.4f}")
        return fitness
