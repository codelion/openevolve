"""
MLX Fine-tuning Optimization Drop-in Patch

This module provides easy integration of evolved MLX optimization patterns
into existing fine-tuning code. Simply import and apply the patches to
get automatic performance improvements.

Usage:
    from mlx_optimization_patch import apply_optimizations
    
    # Apply to existing trainer
    apply_optimizations(trainer)
    
    # Or use as context manager
    with mlx_optimizations():
        # Your existing fine-tuning code here
        trainer.train(dataset)
"""

import os
import json
import importlib.util
import functools
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager


class MLXOptimizationPatch:
    """
    Container for evolved MLX optimization patterns
    
    This class loads the best evolved optimization patterns and provides
    methods to apply them to existing trainers or MLX operations.
    """
    
    def __init__(self, optimization_path: Optional[str] = None):
        """
        Initialize with optimization patterns
        
        Args:
            optimization_path: Path to evolved optimization patterns
                             If None, uses the best patterns from this directory
        """
        self.optimization_config = None
        self.optimization_functions = None
        
        if optimization_path is None:
            # Look for best evolved patterns in this directory
            optimization_path = self._find_best_optimization()
        
        if optimization_path and os.path.exists(optimization_path):
            self._load_optimizations(optimization_path)
        else:
            print(f"Warning: No optimization patterns found at {optimization_path}")
            print("Using default optimization patterns")
            self._load_default_optimizations()
    
    def _find_best_optimization(self) -> Optional[str]:
        """Find the best evolved optimization patterns"""
        # Look in the openevolve output directory
        current_dir = os.path.dirname(__file__)
        openevolve_output = os.path.join(current_dir, "openevolve_output")
        
        if not os.path.exists(openevolve_output):
            return None
        
        # Look for the best program
        best_dir = os.path.join(openevolve_output, "best")
        if os.path.exists(best_dir):
            best_program = os.path.join(best_dir, "best_program.py")
            if os.path.exists(best_program):
                return best_program
        
        # Look in checkpoints for latest
        checkpoints_dir = os.path.join(openevolve_output, "checkpoints")
        if os.path.exists(checkpoints_dir):
            # Find latest checkpoint
            checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint_")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1]))
                checkpoint_program = os.path.join(checkpoints_dir, latest_checkpoint, "best_program.py")
                if os.path.exists(checkpoint_program):
                    return checkpoint_program
        
        return None
    
    def _load_optimizations(self, optimization_path: str):
        """Load optimization patterns from file"""
        try:
            spec = importlib.util.spec_from_file_location("optimization_module", optimization_path)
            optimization_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(optimization_module)
            
            # Load configuration and functions
            if hasattr(optimization_module, 'get_optimization_config'):
                self.optimization_config = optimization_module.get_optimization_config()
            
            self.optimization_functions = {
                'chunked_attention_forward': getattr(optimization_module, 'chunked_attention_forward', None),
                'memory_efficient_gradient_accumulation': getattr(optimization_module, 'memory_efficient_gradient_accumulation', None),
                'optimized_batch_preparation': getattr(optimization_module, 'optimized_batch_preparation', None),
                'adaptive_mixed_precision_forward': getattr(optimization_module, 'adaptive_mixed_precision_forward', None),
                'apply_optimizations_to_trainer': getattr(optimization_module, 'apply_optimizations_to_trainer', None),
            }
            
            print(f"Loaded optimization patterns from {optimization_path}")
            print(f"Configuration: {json.dumps(self.optimization_config, indent=2)}")
            
        except Exception as e:
            print(f"Failed to load optimizations from {optimization_path}: {e}")
            self._load_default_optimizations()
    
    def _load_default_optimizations(self):
        """Load default optimization patterns"""
        # Load from initial_program.py as fallback
        initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
        if os.path.exists(initial_program_path):
            self._load_optimizations(initial_program_path)
        else:
            # Hard-coded safe defaults
            self.optimization_config = {
                "attention_chunk_size": 512,
                "use_chunked_attention": True,
                "use_fp16_compute": True,
                "fp32_gradients": True,
                "dynamic_padding": True,
                "sort_by_length": True,
                "fp16_attention": True,
                "force_gc_frequency": 10,
            }
            self.optimization_functions = {}
    
    def apply_to_trainer(self, trainer):
        """
        Apply optimizations to a baseline trainer
        
        Args:
            trainer: Instance of BaselineTrainer or compatible trainer
        """
        if self.optimization_functions.get('apply_optimizations_to_trainer'):
            self.optimization_functions['apply_optimizations_to_trainer'](trainer, self.optimization_config)
            print("Applied evolved optimizations to trainer")
        else:
            print("Warning: No optimization functions available")
    
    def get_optimized_attention(self):
        """Get optimized attention function"""
        return self.optimization_functions.get('chunked_attention_forward')
    
    def get_optimized_gradient_accumulation(self):
        """Get optimized gradient accumulation function"""
        return self.optimization_functions.get('memory_efficient_gradient_accumulation')
    
    def get_optimized_batch_preparation(self):
        """Get optimized batch preparation function"""
        return self.optimization_functions.get('optimized_batch_preparation')
    
    def get_optimized_mixed_precision(self):
        """Get optimized mixed precision function"""
        return self.optimization_functions.get('adaptive_mixed_precision_forward')
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimization configuration"""
        return self.optimization_config or {}


# Global instance for easy access
_global_optimization_patch = None


def load_optimizations(optimization_path: Optional[str] = None) -> MLXOptimizationPatch:
    """
    Load optimization patterns
    
    Args:
        optimization_path: Path to optimization file (None for auto-detection)
    
    Returns:
        MLXOptimizationPatch instance
    """
    global _global_optimization_patch
    _global_optimization_patch = MLXOptimizationPatch(optimization_path)
    return _global_optimization_patch


def apply_optimizations(trainer, optimization_path: Optional[str] = None):
    """
    Apply evolved optimizations to a trainer
    
    Args:
        trainer: Trainer instance to optimize
        optimization_path: Path to optimization patterns (None for auto-detection)
    """
    patch = load_optimizations(optimization_path)
    patch.apply_to_trainer(trainer)


@contextmanager
def mlx_optimizations(optimization_path: Optional[str] = None):
    """
    Context manager for applying MLX optimizations
    
    Usage:
        with mlx_optimizations():
            # Your training code here
            trainer.train(dataset)
    
    Args:
        optimization_path: Path to optimization patterns (None for auto-detection)
    """
    patch = load_optimizations(optimization_path)
    
    # Store original functions for restoration
    original_functions = {}
    
    try:
        # Apply optimizations globally (this could be extended to patch MLX functions directly)
        print("MLX optimizations active")
        yield patch
        
    finally:
        # Restore original functions if needed
        print("MLX optimizations restored")


def create_optimized_trainer(model_name: str = "mlx-community/Qwen3-0.6B-bf16", 
                           optimization_path: Optional[str] = None):
    """
    Create a trainer with optimizations pre-applied
    
    Args:
        model_name: Model to load
        optimization_path: Path to optimization patterns
    
    Returns:
        Optimized trainer instance
    """
    from baseline_finetuning import BaselineTrainer
    
    trainer = BaselineTrainer(model_name)
    apply_optimizations(trainer, optimization_path)
    
    return trainer


def benchmark_optimization_improvement(model_name: str = "mlx-community/Qwen3-0.6B-bf16",
                                     num_samples: int = 100,
                                     optimization_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Benchmark the improvement from evolved optimizations
    
    Args:
        model_name: Model to benchmark
        num_samples: Number of training samples
        optimization_path: Path to optimization patterns (None for auto-detection)
    
    Returns:
        Benchmark results comparing baseline vs optimized
    """
    from baseline_finetuning import BaselineTrainer
    
    print("Benchmarking baseline trainer...")
    baseline_trainer = BaselineTrainer(model_name)
    baseline_trainer.config.batch_size = 2
    baseline_dataset = baseline_trainer.create_sample_dataset(num_samples)
    baseline_results = baseline_trainer.train(baseline_dataset, "./benchmark_baseline")
    
    print("Benchmarking optimized trainer...")
    optimized_trainer = create_optimized_trainer(model_name, optimization_path)
    optimized_trainer.config.batch_size = 2
    optimized_dataset = optimized_trainer.create_sample_dataset(num_samples)
    optimized_results = optimized_trainer.train(optimized_dataset, "./benchmark_optimized")
    
    # Calculate improvements
    improvements = {}
    for metric in ["tokens_per_second", "memory_efficiency"]:
        if metric in baseline_results and metric in optimized_results:
            if baseline_results[metric] > 0:
                improvement = (optimized_results[metric] - baseline_results[metric]) / baseline_results[metric]
                improvements[f"{metric}_improvement"] = improvement
    
    for metric in ["peak_memory_mb", "total_time"]:
        if metric in baseline_results and metric in optimized_results:
            if baseline_results[metric] > 0:
                improvement = (baseline_results[metric] - optimized_results[metric]) / baseline_results[metric]
                improvements[f"{metric}_improvement"] = improvement
    
    results = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "improvements": improvements
    }
    
    # Save benchmark results
    with open("optimization_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Benchmark Results:")
    print(f"  Speed improvement: {improvements.get('tokens_per_second_improvement', 0):.2%}")
    print(f"  Memory efficiency improvement: {improvements.get('memory_efficiency_improvement', 0):.2%}")
    print(f"  Memory usage improvement: {improvements.get('peak_memory_mb_improvement', 0):.2%}")
    print(f"  Time improvement: {improvements.get('total_time_improvement', 0):.2%}")
    
    return results


# Utility functions for manual optimization application
def optimize_attention_function(original_attention_fn):
    """Decorator to optimize attention functions"""
    patch = load_optimizations()
    optimized_fn = patch.get_optimized_attention()
    
    if optimized_fn:
        @functools.wraps(original_attention_fn)
        def wrapper(*args, **kwargs):
            return optimized_fn(*args, **kwargs)
        return wrapper
    else:
        return original_attention_fn


def optimize_gradient_accumulation(original_grad_fn):
    """Decorator to optimize gradient accumulation"""
    patch = load_optimizations()
    optimized_fn = patch.get_optimized_gradient_accumulation()
    
    if optimized_fn:
        @functools.wraps(original_grad_fn)
        def wrapper(*args, **kwargs):
            # Add optimization config to kwargs
            config = patch.get_config()
            return optimized_fn(*args, config, **kwargs)
        return wrapper
    else:
        return original_grad_fn


if __name__ == "__main__":
    # Demo usage
    print("MLX Fine-tuning Optimization Patch Demo")
    print("======================================")
    
    # Test loading optimizations
    patch = load_optimizations()
    print(f"Loaded optimization config: {patch.get_config()}")
    
    # Test creating optimized trainer
    print("\nCreating optimized trainer...")
    try:
        trainer = create_optimized_trainer()
        print("Optimized trainer created successfully")
    except Exception as e:
        print(f"Failed to create trainer: {e}")
    
    # Test benchmark (commented out as it takes time)
    # print("\nRunning benchmark...")
    # results = benchmark_optimization_improvement(num_samples=50)
