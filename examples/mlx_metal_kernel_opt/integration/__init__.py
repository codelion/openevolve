"""
MLX Metal Kernel Optimization Integration

This package provides seamless integration of optimized Metal kernels with mlx-lm,
offering significant performance improvements for transformer attention computations
on Apple Silicon.

Key Features:
- Automatic dispatch based on model architecture and configuration
- Graceful fallback to standard MLX operations when optimizations aren't beneficial
- Support for GQA, MQA, and MHA attention patterns
- Easy monkey-patching for existing mlx-lm code
- Comprehensive benchmarking and profiling tools

Quick Start:
    from integration import patch_mlx_lm, unpatch_mlx_lm
    
    # Apply optimizations
    patch_mlx_lm(enable_debug=True)
    
    # Use mlx-lm normally
    from mlx_lm import load, generate
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
    
    # Remove optimizations when done
    unpatch_mlx_lm()

Supported Models:
- Qwen3 (40:8 GQA) - High priority optimization
- Qwen2 (32:8 GQA) - High priority optimization  
- Llama (32:8 GQA) - High priority optimization
- Mistral3 (32:8 GQA) - High priority optimization
- Gemma (24:24 MHA) - Medium priority optimization
- Phi3 (32:8 GQA) - Medium priority optimization
- DeepSeek-V3 (GQA) - High priority optimization
"""

from .metal_kernel_optimizer import (
    MetalKernelOptimizer,
    AttentionConfig,
    optimized_scaled_dot_product_attention,
    configure_optimizer,
    get_optimizer_stats,
    reset_optimizer_stats
)

from .mlx_lm_integration import (
    MLXLMIntegration,
    patch_mlx_lm,
    unpatch_mlx_lm,
    get_integration_status,
    is_mlx_lm_patched,
    benchmark_optimization,
    quick_benchmark,
    BenchmarkResult
)

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"
__description__ = "Metal kernel optimizations for MLX-LM attention computations"

__all__ = [
    # Core optimizer
    'MetalKernelOptimizer',
    'AttentionConfig',
    'optimized_scaled_dot_product_attention',
    'configure_optimizer',
    'get_optimizer_stats',
    'reset_optimizer_stats',
    
    # Integration
    'MLXLMIntegration',
    'patch_mlx_lm',
    'unpatch_mlx_lm', 
    'get_integration_status',
    'is_mlx_lm_patched',
    
    # Benchmarking
    'benchmark_optimization',
    'quick_benchmark',
    'BenchmarkResult'
]
