"""
MLX-LM Metal Kernel Integration

This module provides seamless integration of optimized Metal kernels with mlx-lm.
It offers easy monkey-patching mechanisms to replace standard attention implementations
with optimized versions across all supported models.

Usage:
    from integration.mlx_lm_integration import patch_mlx_lm, unpatch_mlx_lm
    
    # Apply optimizations
    patch_mlx_lm(enable_debug=True)
    
    # Use mlx-lm normally - optimizations are applied automatically
    from mlx_lm import generate
    response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
    
    # Remove optimizations
    unpatch_mlx_lm()
"""

import importlib
import sys
import warnings
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import time
import json
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError("MLX is required for Metal kernel optimizations")

# Handle both relative and absolute imports
try:
    from .metal_kernel_optimizer import (
        MetalKernelOptimizer, 
        optimized_scaled_dot_product_attention,
        configure_optimizer,
        get_optimizer_stats,
        reset_optimizer_stats
    )
except ImportError:
    from metal_kernel_optimizer import (
        MetalKernelOptimizer, 
        optimized_scaled_dot_product_attention,
        configure_optimizer,
        get_optimizer_stats,
        reset_optimizer_stats
    )


class MLXLMIntegration:
    """
    Manages integration of Metal kernel optimizations with mlx-lm library.
    """
    
    def __init__(self):
        self.original_functions = {}
        self.patched_modules = set()
        self.is_patched = False
        self.optimization_enabled = False
        
        # Supported model architectures and their attention patterns
        self.supported_models = {
            'qwen3': {
                'module': 'mlx_lm.models.qwen3',
                'attention_class': 'Attention',
                'expected_pattern': 'GQA',
                'priority': 'high'
            },
            'qwen2': {
                'module': 'mlx_lm.models.qwen2', 
                'attention_class': 'Attention',
                'expected_pattern': 'GQA',
                'priority': 'high'
            },
            'llama': {
                'module': 'mlx_lm.models.llama',
                'attention_class': 'Attention', 
                'expected_pattern': 'GQA',
                'priority': 'high'
            },
            'gemma': {
                'module': 'mlx_lm.models.gemma',
                'attention_class': 'Attention',
                'expected_pattern': 'MHA',
                'priority': 'medium'
            },
            'gemma2': {
                'module': 'mlx_lm.models.gemma2',
                'attention_class': 'Attention',
                'expected_pattern': 'MHA', 
                'priority': 'medium'
            },
            'mistral3': {
                'module': 'mlx_lm.models.mistral3',
                'attention_class': 'Attention',
                'expected_pattern': 'GQA',
                'priority': 'high'
            },
            'phi3': {
                'module': 'mlx_lm.models.phi3',
                'attention_class': 'Attention',
                'expected_pattern': 'GQA',
                'priority': 'medium'
            },
            'deepseek_v3': {
                'module': 'mlx_lm.models.deepseek_v3',
                'attention_class': 'Attention',
                'expected_pattern': 'GQA',
                'priority': 'high'
            }
        }

    def patch_base_attention(self, enable_debug: bool = False):
        """
        Patch the base scaled_dot_product_attention function used across mlx-lm.
        """
        try:
            # Configure the global optimizer
            configure_optimizer(enable_debug=enable_debug)
            
            # Import and patch base module
            base_module = importlib.import_module('mlx_lm.models.base')
            
            if hasattr(base_module, 'scaled_dot_product_attention'):
                # Store original function
                original_sdpa = base_module.scaled_dot_product_attention
                self.original_functions['base.scaled_dot_product_attention'] = original_sdpa
                
                # Create optimized wrapper
                def optimized_base_sdpa(queries, keys, values, cache, scale: float, mask: Optional[mx.array]):
                    """Optimized wrapper for base scaled_dot_product_attention"""
                    # Handle quantized cache case
                    if hasattr(cache, 'group_size'):  # QuantizedKVCache
                        return original_sdpa(queries, keys, values, cache, scale, mask)
                    else:
                        # Use our optimized implementation
                        return optimized_scaled_dot_product_attention(queries, keys, values, scale, mask)
                
                # Apply patch
                base_module.scaled_dot_product_attention = optimized_base_sdpa
                self.patched_modules.add('mlx_lm.models.base')
                
                if enable_debug:
                    print("✅ Patched base scaled_dot_product_attention")
                    
        except ImportError as e:
            if enable_debug:
                print(f"⚠️ Could not patch base module: {e}")
        except Exception as e:
            if enable_debug:
                print(f"⚠️ Error patching base module: {e}")

    def patch_model_attention(self, model_name: str, enable_debug: bool = False):
        """
        Patch attention implementation for a specific model.
        """
        if model_name not in self.supported_models:
            if enable_debug:
                print(f"⚠️ Model '{model_name}' not in supported models")
            return False
            
        model_config = self.supported_models[model_name]
        
        try:
            # Import the model module
            module = importlib.import_module(model_config['module'])
            
            if hasattr(module, model_config['attention_class']):
                attention_class = getattr(module, model_config['attention_class'])
                
                # Store original __call__ method
                original_call = attention_class.__call__
                self.original_functions[f"{model_name}.{model_config['attention_class']}.__call__"] = original_call
                
                # Create optimized wrapper
                def create_optimized_call(original_method):
                    @wraps(original_method)
                    def optimized_call(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
                        """Optimized attention call with Metal kernel integration"""
                        B, L, D = x.shape
                        
                        # Standard preprocessing
                        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
                        
                        # Reshape and transpose
                        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)  
                        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                        
                        # Apply normalization if present
                        if hasattr(self, 'q_norm') and hasattr(self, 'k_norm'):
                            queries = self.q_norm(queries.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
                            keys = self.k_norm(keys.transpose(0, 2, 1, 3).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                        
                        # Apply RoPE if present
                        if hasattr(self, 'rope'):
                            if cache is not None:
                                queries = self.rope(queries, offset=cache.offset)
                                keys = self.rope(keys, offset=cache.offset)
                                keys, values = cache.update_and_fetch(keys, values)
                            else:
                                queries = self.rope(queries)
                                keys = self.rope(keys)
                        
                        # Apply optimized attention
                        output = optimized_scaled_dot_product_attention(
                            queries, keys, values, scale=self.scale, mask=mask
                        )
                        
                        # Standard postprocessing
                        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                        return self.o_proj(output)
                    
                    return optimized_call
                
                # Apply patch
                attention_class.__call__ = create_optimized_call(original_call)
                self.patched_modules.add(model_config['module'])
                
                if enable_debug:
                    print(f"✅ Patched {model_name} attention")
                return True
                
        except ImportError:
            if enable_debug:
                print(f"⚠️ Could not import {model_config['module']}")
            return False
        except Exception as e:
            if enable_debug:
                print(f"⚠️ Error patching {model_name}: {e}")
            return False
            
        return False

    def patch_all_models(self, enable_debug: bool = False):
        """
        Patch all supported models with optimized attention.
        """
        patched_count = 0
        
        # First patch the base attention function
        self.patch_base_attention(enable_debug)
        
        # Then patch individual model attention classes
        for model_name in self.supported_models:
            if self.patch_model_attention(model_name, enable_debug):
                patched_count += 1
        
        if enable_debug:
            print(f"✅ Successfully patched {patched_count}/{len(self.supported_models)} models")
        
        self.is_patched = True
        self.optimization_enabled = True
        
        return patched_count

    def unpatch_all(self, enable_debug: bool = False):
        """
        Remove all patches and restore original implementations.
        """
        restored_count = 0
        
        # Restore all patched functions
        for func_path, original_func in self.original_functions.items():
            try:
                if '.' in func_path:
                    parts = func_path.split('.')
                    if parts[0] == 'base':
                        # Restore base module
                        base_module = importlib.import_module('mlx_lm.models.base')
                        setattr(base_module, parts[1], original_func)
                    else:
                        # Restore model-specific function
                        model_name = parts[0]
                        if model_name in self.supported_models:
                            model_config = self.supported_models[model_name]
                            module = importlib.import_module(model_config['module'])
                            attention_class = getattr(module, model_config['attention_class'])
                            setattr(attention_class, parts[2], original_func)
                
                restored_count += 1
                
            except Exception as e:
                if enable_debug:
                    print(f"⚠️ Could not restore {func_path}: {e}")
        
        # Clear state
        self.original_functions.clear()
        self.patched_modules.clear()
        self.is_patched = False
        self.optimization_enabled = False
        
        if enable_debug:
            print(f"✅ Restored {restored_count} functions")
        
        return restored_count

    def get_patch_status(self) -> Dict[str, Any]:
        """Get current patch status and statistics"""
        stats = get_optimizer_stats() if self.optimization_enabled else {}
        
        return {
            'is_patched': self.is_patched,
            'optimization_enabled': self.optimization_enabled,
            'patched_modules': list(self.patched_modules),
            'patched_functions': list(self.original_functions.keys()),
            'optimizer_stats': stats
        }


# Global integration instance
_global_integration = MLXLMIntegration()


def patch_mlx_lm(enable_debug: bool = False, **optimizer_kwargs) -> int:
    """
    Apply Metal kernel optimizations to mlx-lm.
    
    Args:
        enable_debug: Enable debug output
        **optimizer_kwargs: Additional optimizer configuration
        
    Returns:
        Number of models successfully patched
        
    Example:
        >>> from integration.mlx_lm_integration import patch_mlx_lm
        >>> patch_mlx_lm(enable_debug=True)
        ✅ Patched base scaled_dot_product_attention
        ✅ Patched qwen3 attention
        ✅ Patched llama attention
        ✅ Successfully patched 7/8 models
        7
    """
    if _global_integration.is_patched:
        if enable_debug:
            print("⚠️ MLX-LM is already patched")
        return 0
    
    # Configure optimizer with any additional parameters
    if optimizer_kwargs:
        configure_optimizer(enable_debug=enable_debug, **optimizer_kwargs)
    
    return _global_integration.patch_all_models(enable_debug)


def unpatch_mlx_lm(enable_debug: bool = False) -> int:
    """
    Remove Metal kernel optimizations from mlx-lm.
    
    Args:
        enable_debug: Enable debug output
        
    Returns:
        Number of functions restored
        
    Example:
        >>> unpatch_mlx_lm(enable_debug=True)
        ✅ Restored 8 functions
        8
    """
    return _global_integration.unpatch_all(enable_debug)


def get_integration_status() -> Dict[str, Any]:
    """
    Get current integration status and performance statistics.
    
    Returns:
        Dictionary with patch status and optimizer statistics
        
    Example:
        >>> status = get_integration_status()
        >>> print(f"Optimization rate: {status['optimizer_stats']['optimization_rate']:.1%}")
    """
    return _global_integration.get_patch_status()


def is_mlx_lm_patched() -> bool:
    """Check if mlx-lm is currently patched with optimizations"""
    return _global_integration.is_patched


class BenchmarkResult:
    """Container for benchmark results"""
    
    def __init__(self, model_name: str, seq_length: int):
        self.model_name = model_name
        self.seq_length = seq_length
        self.standard_time = None
        self.optimized_time = None
        self.standard_memory = None
        self.optimized_memory = None
        self.speedup = None
        self.memory_reduction = None
        
    def calculate_improvements(self):
        """Calculate speedup and memory reduction"""
        if self.standard_time and self.optimized_time:
            self.speedup = self.standard_time / self.optimized_time
            
        if self.standard_memory and self.optimized_memory:
            self.memory_reduction = (self.standard_memory - self.optimized_memory) / self.standard_memory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'model_name': self.model_name,
            'seq_length': self.seq_length,
            'standard_time': self.standard_time,
            'optimized_time': self.optimized_time,
            'standard_memory': self.standard_memory,
            'optimized_memory': self.optimized_memory,
            'speedup': self.speedup,
            'memory_reduction': self.memory_reduction
        }


def benchmark_optimization(model_name: str = "qwen3", seq_lengths: List[int] = None, 
                         warmup_runs: int = 3, benchmark_runs: int = 10,
                         save_results: bool = True) -> List[BenchmarkResult]:
    """
    Benchmark Metal kernel optimizations against standard MLX implementation.
    
    Args:
        model_name: Name of model architecture to benchmark
        seq_lengths: List of sequence lengths to test
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        save_results: Whether to save results to file
        
    Returns:
        List of BenchmarkResult objects
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]
    
    if model_name not in _global_integration.supported_models:
        raise ValueError(f"Model '{model_name}' not supported. Supported: {list(_global_integration.supported_models.keys())}")
    
    print(f"🔬 Benchmarking {model_name} Metal kernel optimization")
    print(f"📊 Testing sequence lengths: {seq_lengths}")
    print(f"🔄 Warmup runs: {warmup_runs}, Benchmark runs: {benchmark_runs}")
    print("=" * 70)
    
    results = []
    
    # Mock model configuration based on model name
    mock_configs = {
        'qwen3': {'hidden_size': 5120, 'num_heads': 40, 'num_kv_heads': 8, 'head_dim': 128},
        'llama': {'hidden_size': 4096, 'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128},
        'gemma': {'hidden_size': 3072, 'num_heads': 24, 'num_kv_heads': 24, 'head_dim': 128},
        'mistral3': {'hidden_size': 4096, 'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128}
    }
    
    config = mock_configs.get(model_name, mock_configs['qwen3'])
    
    for seq_len in seq_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        result = BenchmarkResult(model_name, seq_len)
        
        # Create test data
        batch_size = 1
        x = mx.random.normal((batch_size, seq_len, config['hidden_size']))
        
        # Create mock attention layers for testing
        class MockAttention(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.n_heads = config['num_heads']
                self.n_kv_heads = config['num_kv_heads']
                self.scale = config['head_dim'] ** -0.5
                
                self.q_proj = nn.Linear(config['hidden_size'], config['num_heads'] * config['head_dim'], bias=False)
                self.k_proj = nn.Linear(config['hidden_size'], config['num_kv_heads'] * config['head_dim'], bias=False)
                self.v_proj = nn.Linear(config['hidden_size'], config['num_kv_heads'] * config['head_dim'], bias=False)
                self.o_proj = nn.Linear(config['num_heads'] * config['head_dim'], config['hidden_size'], bias=False)
            
            def __call__(self, x, use_optimization=False):
                B, L, D = x.shape
                
                queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
                queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                
                if use_optimization:
                    output = optimized_scaled_dot_product_attention(
                        queries, keys, values, scale=self.scale, mask="causal"
                    )
                else:
                    output = mx.fast.scaled_dot_product_attention(
                        queries, keys, values, scale=self.scale, mask="causal"
                    )
                
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)
        
        attention = MockAttention(config)
        
        # Benchmark standard implementation
        print("  🔄 Testing standard MLX implementation...")
        
        # Warmup
        for _ in range(warmup_runs):
            _ = attention(x, use_optimization=False)
            mx.eval(_)
        
        # Measure
        mx.synchronize()
        start_time = time.perf_counter()
        start_memory = mx.get_active_memory()
        
        for _ in range(benchmark_runs):
            output = attention(x, use_optimization=False)
            mx.eval(output)
        
        mx.synchronize()
        end_time = time.perf_counter()
        end_memory = mx.get_active_memory()
        
        result.standard_time = (end_time - start_time) / benchmark_runs
        result.standard_memory = end_memory
        
        print(f"    ⏱️  Standard: {result.standard_time*1000:.2f} ms/iteration")
        print(f"    💾 Memory: {result.standard_memory/1e9:.2f} GB")
        
        # Benchmark optimized implementation
        print("  ⚡ Testing optimized Metal kernel...")
        
        # Reset optimizer stats
        reset_optimizer_stats()
        
        # Warmup
        for _ in range(warmup_runs):
            _ = attention(x, use_optimization=True)
            mx.eval(_)
        
        # Measure
        mx.synchronize()
        start_time = time.perf_counter()
        start_memory = mx.get_active_memory()
        
        for _ in range(benchmark_runs):
            output = attention(x, use_optimization=True)
            mx.eval(output)
        
        mx.synchronize()
        end_time = time.perf_counter()
        end_memory = mx.get_active_memory()
        
        result.optimized_time = (end_time - start_time) / benchmark_runs
        result.optimized_memory = end_memory
        
        # Calculate improvements
        result.calculate_improvements()
        
        print(f"    ⏱️  Optimized: {result.optimized_time*1000:.2f} ms/iteration")
        print(f"    💾 Memory: {result.optimized_memory/1e9:.2f} GB")
        
        if result.speedup:
            print(f"    🚀 Speedup: {result.speedup:.2f}x")
        if result.memory_reduction:
            print(f"    📉 Memory reduction: {result.memory_reduction:.1%}")
        
        # Get optimizer stats
        opt_stats = get_optimizer_stats()
        optimization_rate = opt_stats.get('optimization_rate', 0.0)
        print(f"    📊 Optimization rate: {optimization_rate:.1%}")
        
        results.append(result)
    
    # Save results if requested
    if save_results:
        timestamp = int(time.time())
        results_file = f"metal_kernel_benchmark_{model_name}_{timestamp}.json"
        
        results_data = {
            'model_name': model_name,
            'timestamp': timestamp,
            'config': config,
            'warmup_runs': warmup_runs,
            'benchmark_runs': benchmark_runs,
            'results': [r.to_dict() for r in results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    # Print summary
    print(f"\n📊 Benchmark Summary for {model_name}:")
    print("-" * 50)
    avg_speedup = sum(r.speedup for r in results if r.speedup) / len([r for r in results if r.speedup])
    print(f"Average speedup: {avg_speedup:.2f}x")
    
    best_speedup = max((r.speedup for r in results if r.speedup), default=0)
    best_seq_len = next((r.seq_length for r in results if r.speedup == best_speedup), None)
    print(f"Best speedup: {best_speedup:.2f}x (seq_len: {best_seq_len})")
    
    return results


# Convenience function for quick testing
def quick_benchmark(enable_debug: bool = True):
    """
    Quick benchmark test with common configuration.
    """
    print("🚀 Quick Metal Kernel Optimization Benchmark")
    print("=" * 50)
    
    # Apply optimizations
    patched_count = patch_mlx_lm(enable_debug=enable_debug)
    print(f"✅ Applied optimizations to {patched_count} models")
    
    try:
        # Run benchmark
        results = benchmark_optimization(
            model_name="qwen3",
            seq_lengths=[256, 512, 1024],
            warmup_runs=2,
            benchmark_runs=5,
            save_results=True
        )
        
        # Show final status
        status = get_integration_status()
        print(f"\n📊 Final Integration Status:")
        print(f"  Patched modules: {len(status['patched_modules'])}")
        print(f"  Optimizer stats: {status['optimizer_stats']}")
        
        return results
        
    finally:
        # Clean up
        unpatch_mlx_lm(enable_debug=enable_debug)
        print("🧹 Cleaned up optimizations")


if __name__ == "__main__":
    # Run quick benchmark when script is executed directly
    quick_benchmark()
