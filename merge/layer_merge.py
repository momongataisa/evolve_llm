"""Layer-wise model merging using mergekit."""

import subprocess
import logging
from pathlib import Path
from typing import Optional, List
import tempfile
import shutil
from genetic_algorithm.individual import Individual
from .config_generator import MergeConfigGenerator


logger = logging.getLogger(__name__)


class LayerWiseMerger:
    """Handles layer-wise model merging using mergekit."""

    def __init__(
        self,
        model_paths: List[str],
        cache_dir: Optional[str] = None,
        use_gpu: bool = True,
        merge_method: str = "linear"
    ):
        """
        Initialize the merger.

        Args:
            model_paths: List of model paths to merge
            cache_dir: Directory for caching merged models
            use_gpu: Whether to use GPU for merging
            merge_method: Merge method to use
        """
        self.model_paths = model_paths
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./merged_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.merge_method = merge_method
        self.config_generator = MergeConfigGenerator(model_paths)

    def merge(
        self,
        individual: Individual,
        output_path: Optional[str] = None,
        config_path: Optional[str] = None,
        copy_tokenizer: bool = True,
        allow_crimes: bool = True,
        lazy_unpickle: bool = False
    ) -> str:
        """
        Merge models based on individual's genes.

        Args:
            individual: Individual with layer-wise merge weights
            output_path: Path to save merged model
            config_path: Path to save/load merge config
            copy_tokenizer: Copy tokenizer from base model
            allow_crimes: Allow crimes in mergekit
            lazy_unpickle: Use lazy unpickling

        Returns:
            Path to merged model
        """
        # Generate output path if not provided
        if output_path is None:
            output_path = self.cache_dir / f"merged_gen{hash(individual.genes.tobytes()) % 100000}"
        output_path = Path(output_path)

        # Check if already merged
        if output_path.exists() and (output_path / "config.json").exists():
            logger.info(f"Using cached merged model at {output_path}")
            return str(output_path)

        # Generate config
        if config_path is None:
            config_path = output_path.parent / f"{output_path.name}_config.yaml"

        logger.info(f"Generating merge config at {config_path}")
        self.config_generator.generate_and_save(
            individual,
            str(config_path),
            merge_method=self.merge_method,
            simple=True  # Use simple config for compatibility
        )

        # Run mergekit
        logger.info(f"Merging models to {output_path}")
        self._run_mergekit(
            config_path=str(config_path),
            output_path=str(output_path),
            copy_tokenizer=copy_tokenizer,
            allow_crimes=allow_crimes,
            lazy_unpickle=lazy_unpickle
        )

        return str(output_path)

    def _run_mergekit(
        self,
        config_path: str,
        output_path: str,
        copy_tokenizer: bool,
        allow_crimes: bool,
        lazy_unpickle: bool
    ):
        """
        Run mergekit command.

        Args:
            config_path: Path to merge config
            output_path: Output path for merged model
            copy_tokenizer: Copy tokenizer
            allow_crimes: Allow crimes
            lazy_unpickle: Lazy unpickle
        """
        cmd = [
            "mergekit-yaml",
            config_path,
            output_path,
        ]

        if copy_tokenizer:
            cmd.append("--copy-tokenizer")

        if allow_crimes:
            cmd.append("--allow-crimes")

        if lazy_unpickle:
            cmd.append("--lazy-unpickle")

        if self.use_gpu:
            cmd.extend(["--cuda"])

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Merge completed successfully")
            if result.stdout:
                logger.debug(f"Stdout: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Mergekit failed with error: {e.stderr}")
            raise RuntimeError(f"Model merge failed: {e.stderr}")

        except FileNotFoundError:
            logger.error("mergekit-yaml not found. Please install mergekit.")
            raise RuntimeError(
                "mergekit not installed. Install with: pip install mergekit"
            )

    def cleanup_cache(self, keep_best: int = 5):
        """
        Clean up old merged models from cache.

        Args:
            keep_best: Number of recent models to keep
        """
        if not self.cache_dir.exists():
            return

        # Get all merged model directories
        merged_dirs = [
            d for d in self.cache_dir.iterdir()
            if d.is_dir() and d.name.startswith("merged_")
        ]

        # Sort by modification time
        merged_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old models
        for old_dir in merged_dirs[keep_best:]:
            logger.info(f"Removing old merged model: {old_dir}")
            shutil.rmtree(old_dir)
