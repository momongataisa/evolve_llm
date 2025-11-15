"""Multi-domain fitness evaluation for specialized model merging."""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .fitness import FitnessEvaluator, BenchmarkConfig


logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""

    name: str
    tasks: List[str]
    weight: float = 1.0  # Weight of this domain in final fitness
    num_fewshot: int = 5
    limit: Optional[int] = None


@dataclass
class MultiDomainConfig:
    """Configuration for multi-domain evaluation."""

    domains: List[DomainConfig] = field(default_factory=list)
    batch_size: int = 8
    device: str = "cuda"
    normalization: str = "weighted"  # "weighted", "average", "max"

    @classmethod
    def create_financial_general(
        cls,
        financial_weight: float = 0.6,
        general_weight: float = 0.4,
        limit: Optional[int] = None
    ):
        """
        Create a preset config for financial + general domain merging.

        Args:
            financial_weight: Weight for financial tasks
            general_weight: Weight for general tasks
            limit: Limit for faster evaluation

        Returns:
            MultiDomainConfig instance
        """
        domains = [
            DomainConfig(
                name="financial",
                tasks=[
                    "finqa",           # Financial QA
                    "convfinqa",       # Conversational Financial QA
                    "fiqa",            # Financial Opinion Mining
                    "fpb",             # Financial PhraseBank
                ],
                weight=financial_weight,
                num_fewshot=3,
                limit=limit
            ),
            DomainConfig(
                name="general",
                tasks=[
                    "arc_easy",
                    "hellaswag",
                    "winogrande",
                    "mmlu",
                ],
                weight=general_weight,
                num_fewshot=5,
                limit=limit
            )
        ]

        return cls(domains=domains)


class MultiDomainFitnessEvaluator:
    """
    Evaluates fitness across multiple domains (e.g., financial + general).

    This is useful for merging specialized models (e.g., finance) with general models.
    """

    def __init__(
        self,
        multi_domain_config: MultiDomainConfig,
        cache_results: bool = True,
        fallback_on_error: bool = True
    ):
        """
        Initialize multi-domain evaluator.

        Args:
            multi_domain_config: Configuration for multiple domains
            cache_results: Whether to cache evaluation results
            fallback_on_error: If a domain fails, continue with other domains
        """
        self.config = multi_domain_config
        self.cache_results = cache_results
        self.fallback_on_error = fallback_on_error
        self.results_cache: Dict[str, Dict[str, float]] = {}

        # Create evaluator for each domain
        self.domain_evaluators: Dict[str, FitnessEvaluator] = {}
        for domain in self.config.domains:
            benchmark_config = BenchmarkConfig(
                tasks=domain.tasks,
                num_fewshot=domain.num_fewshot,
                batch_size=self.config.batch_size,
                device=self.config.device,
                limit=domain.limit
            )
            self.domain_evaluators[domain.name] = FitnessEvaluator(
                benchmark_config=benchmark_config,
                cache_results=cache_results
            )

    def evaluate(self, model_path: str, model_id: Optional[str] = None) -> float:
        """
        Evaluate a model across all domains and return weighted fitness.

        Args:
            model_path: Path to the model to evaluate
            model_id: Optional identifier for caching

        Returns:
            Weighted fitness score (higher is better)
        """
        # Check cache
        if model_id and self.cache_results and model_id in self.results_cache:
            logger.info(f"Using cached multi-domain fitness for {model_id}")
            cached_scores = self.results_cache[model_id]
            return self._calculate_combined_fitness(cached_scores)

        # Evaluate each domain
        domain_scores: Dict[str, float] = {}

        for domain in self.config.domains:
            logger.info(f"Evaluating domain: {domain.name}")
            try:
                evaluator = self.domain_evaluators[domain.name]
                score = evaluator.evaluate(model_path, model_id=f"{model_id}_{domain.name}")
                domain_scores[domain.name] = score
                logger.info(f"Domain '{domain.name}' score: {score:.4f}")

            except Exception as e:
                logger.error(f"Evaluation failed for domain '{domain.name}': {e}")
                if self.fallback_on_error:
                    domain_scores[domain.name] = 0.0
                else:
                    raise

        # Cache results
        if model_id and self.cache_results:
            self.results_cache[model_id] = domain_scores

        # Calculate combined fitness
        combined_fitness = self._calculate_combined_fitness(domain_scores)
        logger.info(f"Combined fitness for {model_path}: {combined_fitness:.4f}")

        return combined_fitness

    def _calculate_combined_fitness(self, domain_scores: Dict[str, float]) -> float:
        """
        Calculate combined fitness from domain scores.

        Args:
            domain_scores: Dictionary of domain names to scores

        Returns:
            Combined fitness score
        """
        if not domain_scores:
            return 0.0

        if self.config.normalization == "weighted":
            # Weighted average
            total_weight = 0.0
            weighted_sum = 0.0

            for domain in self.config.domains:
                if domain.name in domain_scores:
                    weighted_sum += domain_scores[domain.name] * domain.weight
                    total_weight += domain.weight

            if total_weight == 0:
                return 0.0

            return weighted_sum / total_weight

        elif self.config.normalization == "average":
            # Simple average
            return sum(domain_scores.values()) / len(domain_scores)

        elif self.config.normalization == "max":
            # Maximum score
            return max(domain_scores.values())

        else:
            raise ValueError(f"Unknown normalization: {self.config.normalization}")

    def get_domain_scores(self, model_id: str) -> Optional[Dict[str, float]]:
        """
        Get cached domain scores for a model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary of domain scores or None if not cached
        """
        return self.results_cache.get(model_id)

    def print_summary(self, model_id: str):
        """
        Print summary of domain scores for a model.

        Args:
            model_id: Model identifier
        """
        scores = self.get_domain_scores(model_id)
        if not scores:
            logger.warning(f"No cached scores for {model_id}")
            return

        print("\n" + "="*60)
        print(f"MULTI-DOMAIN EVALUATION SUMMARY: {model_id}")
        print("="*60)

        for domain in self.config.domains:
            if domain.name in scores:
                score = scores[domain.name]
                print(f"{domain.name:20s} (weight={domain.weight:.2f}): {score:.4f}")

        combined = self._calculate_combined_fitness(scores)
        print("-"*60)
        print(f"{'Combined Fitness':20s}: {combined:.4f}")
        print("="*60 + "\n")


class SimpleMultiDomainEvaluator(MultiDomainFitnessEvaluator):
    """
    Simplified multi-domain evaluator using custom evaluation functions.

    Useful when lm-eval is not available or for quick testing.
    """

    def __init__(
        self,
        domain_configs: List[DomainConfig],
        evaluation_functions: Optional[Dict[str, callable]] = None
    ):
        """
        Initialize simple multi-domain evaluator.

        Args:
            domain_configs: List of domain configurations
            evaluation_functions: Dict mapping domain names to evaluation functions
        """
        self.domain_configs = domain_configs
        self.evaluation_functions = evaluation_functions or {}
        self.results_cache: Dict[str, Dict[str, float]] = {}

    def evaluate(self, model_path: str, model_id: Optional[str] = None) -> float:
        """
        Evaluate using custom functions.

        Args:
            model_path: Path to model
            model_id: Model identifier

        Returns:
            Combined fitness score
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Check cache
        if model_id and model_id in self.results_cache:
            logger.info(f"Using cached multi-domain fitness for {model_id}")
            return self._calculate_combined_fitness(self.results_cache[model_id])

        try:
            # Load model once
            logger.info(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            domain_scores: Dict[str, float] = {}

            for domain in self.domain_configs:
                logger.info(f"Evaluating domain: {domain.name}")

                if domain.name in self.evaluation_functions:
                    # Use custom function
                    score = self.evaluation_functions[domain.name](model, tokenizer)
                else:
                    # Use default perplexity-based evaluation
                    score = self._default_evaluate(model, tokenizer, domain)

                domain_scores[domain.name] = score
                logger.info(f"Domain '{domain.name}' score: {score:.4f}")

            # Cleanup
            del model
            torch.cuda.empty_cache()

            # Cache results
            if model_id:
                self.results_cache[model_id] = domain_scores

            return self._calculate_combined_fitness(domain_scores)

        except Exception as e:
            logger.error(f"Simple multi-domain evaluation failed: {e}")
            return 0.0

    def _default_evaluate(self, model, tokenizer, domain: DomainConfig) -> float:
        """
        Default evaluation using perplexity.

        Args:
            model: The model
            tokenizer: The tokenizer
            domain: Domain configuration

        Returns:
            Fitness score
        """
        import torch

        # Domain-specific test texts
        test_texts = {
            "financial": "The company's earnings per share increased by 15% in the fourth quarter.",
            "general": "The quick brown fox jumps over the lazy dog."
        }

        test_text = test_texts.get(domain.name, test_texts["general"])
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            perplexity = torch.exp(outputs.loss).item()

        # Convert to fitness (lower perplexity = higher fitness)
        fitness = 100.0 / perplexity
        return fitness

    def _calculate_combined_fitness(self, domain_scores: Dict[str, float]) -> float:
        """Calculate weighted average fitness."""
        if not domain_scores:
            return 0.0

        total_weight = sum(d.weight for d in self.domain_configs)
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for domain in self.domain_configs:
            if domain.name in domain_scores:
                weighted_sum += domain_scores[domain.name] * domain.weight

        return weighted_sum / total_weight
