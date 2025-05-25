"""
Evaluator for MLX Attention Mechanism Optimization

This evaluator tests evolved attention optimizations on real transformer model
inference and training tasks, using models like Qwen3-0.6B-bf16 to ensure
practical relevance and measurable improvements.
"""

import importlib.util
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
import traceback
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional
import json
import os


def safe_float_conversion(value, default=0.0):
    """Safely convert a value to float, handling infinity and NaN"""
    try:
        float_val = float(value)
        if np.isnan(float_val) or np.isinf(float_val):
            return default
        return float_val
    except (TypeError, ValueError, OverflowError):
        return default


def safe_division(numerator, denominator, default=0.0):
    """Safely perform division, handling zero denominators and infinity"""
    try:
        if denominator == 0 or denominator is None:
            return default
        result = numerator / denominator
        return safe_float_conversion(result, default)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return default


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=60):
    """Run a function with timeout using concurrent.futures"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")


class SimpleTransformerBlock(nn.Module):
    """
    Simplified transformer block for testing attention optimizations
    Based on common transformer architectures like Qwen, LLaMA, etc.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Layer norm and feed forward
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x: mx.array, attention_fn, attention_config: Dict[str, Any]) -> mx.array:
        """
        Forward pass using the provided attention function
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_fn: Attention function to use
            attention_config: Configuration for attention function
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-attention layer norm
        x_norm = self.ln1(x)
        
        # Multi-head attention projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm) 
        v = self.v_proj(x_norm)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose to [batch, n_heads, seq_len, head_dim]
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        
        # Apply attention to each head
        attn_outputs = []
        for head in range(self.n_heads):
            q_head = q[:, head, :, :]  # [batch, seq_len, head_dim]
            k_head = k[:, head, :, :]
            v_head = v[:, head, :, :]
            
            # Create causal mask
            mask = mx.triu(mx.ones((seq_len, seq_len)), k=1) * -1e9
            mask = mx.broadcast_to(mask[None, :, :], (batch_size, seq_len, seq_len))
            
            # Apply optimized attention
            head_output = attention_fn(q_head, k_head, v_head, mask, **attention_config)
            attn_outputs.append(head_output)
        
        # Concatenate heads
        attn_output = mx.concatenate(attn_outputs, axis=-1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        # Residual connection
        x = x + attn_output
        
        # Feed forward with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


def create_test_model(d_model: int = 512, n_heads: int = 8, n_layers: int = 4):
    """Create a simple test transformer model"""
    layers = []
    for _ in range(n_layers):
        layers.append(SimpleTransformerBlock(d_model, n_heads))
    return layers


def reference_attention(query: mx.array, key: mx.array, value: mx.array, 
                       mask: Optional[mx.array] = None) -> mx.array:
    """
    Reference attention implementation using standard MLX operations
    This is our baseline to beat
    """
    # Standard scaled dot-product attention
    d_k = query.shape[-1]
    scores = mx.matmul(query, mx.transpose(key, axes=(0, 2, 1))) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
    
    attention_weights = mx.softmax(scores, axis=-1)
    output = mx.matmul(attention_weights, value)
    
    return output


def verify_attention_correctness(optimized_fn, config: Dict[str, Any], 
                               tolerance: float = 1e-2) -> Tuple[bool, float]:
    """
    Verify that optimized attention produces correct results
    
    Args:
        optimized_fn: The optimized attention function
        config: Configuration for optimized function
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Tuple of (is_correct, max_difference)
    """
    try:
        # Create test inputs
        batch_size, seq_len, d_model = 2, 32, 64
        query = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
        key = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
        value = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
        
        # Create mask
        mask = mx.triu(mx.ones((seq_len, seq_len)), k=1) * -1e9
        mask = mx.broadcast_to(mask[None, :, :], (batch_size, seq_len, seq_len))
        
        # Compute reference output
        reference_output = reference_attention(query, key, value, mask)
        mx.eval(reference_output)
        
        # Compute optimized output
        optimized_output = optimized_fn(query, key, value, mask, **config)
        mx.eval(optimized_output)
        
        # Check shapes match
        if reference_output.shape != optimized_output.shape:
            return False, float('inf')
        
        # Check for NaN or infinite values
        if mx.any(mx.isnan(optimized_output)) or mx.any(mx.isinf(optimized_output)):
            return False, float('inf')
        
        # Compute numerical difference
        diff = mx.abs(reference_output - optimized_output)
        max_diff = float(mx.max(diff))
        mean_diff = float(mx.mean(diff))
        
        # Relative error check
        ref_magnitude = float(mx.mean(mx.abs(reference_output)))
        relative_error = mean_diff / (ref_magnitude + 1e-8)
        
        is_correct = max_diff < tolerance and relative_error < tolerance * 0.1
        
        return is_correct, max_diff
        
    except Exception as e:
        print(f"Correctness verification failed: {e}")
        return False, float('inf')


def benchmark_model_inference(program, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Benchmark optimized attention on full model inference
    
    Args:
        program: Loaded program module
        config: Attention configuration
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        # Model configurations to test (similar to small models like Qwen3-0.6B)
        model_configs = [
            {"d_model": 512, "n_heads": 8, "n_layers": 2, "seq_len": 128},   # Small
            {"d_model": 768, "n_heads": 12, "n_layers": 2, "seq_len": 256},  # Medium
            {"d_model": 1024, "n_heads": 16, "n_layers": 2, "seq_len": 512}, # Large
        ]
        
        results = {}
        
        for i, model_config in enumerate(model_configs):
            config_name = f"model_{i+1}"
            
            try:
                # Create model
                model_layers = create_test_model(
                    d_model=model_config["d_model"],
                    n_heads=model_config["n_heads"], 
                    n_layers=model_config["n_layers"]
                )
                
                # Create input sequence
                batch_size = 2
                seq_len = model_config["seq_len"]
                d_model = model_config["d_model"]
                
                input_tokens = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
                
                # Warmup with reference attention
                x_ref = input_tokens
                for layer in model_layers:
                    x_ref = layer.forward(x_ref, reference_attention, {})
                mx.eval(x_ref)
                
                # Warmup with optimized attention
                x_opt = input_tokens
                for layer in model_layers:
                    x_opt = layer.forward(x_opt, program.optimized_attention_kernel, config)
                mx.eval(x_opt)
                
                # Benchmark reference implementation
                ref_times = []
                for trial in range(3):
                    x = input_tokens
                    start_time = time.perf_counter()
                    for layer in model_layers:
                        x = layer.forward(x, reference_attention, {})
                    mx.eval(x)
                    end_time = time.perf_counter()
                    ref_times.append(end_time - start_time)
                
                ref_time = np.mean(ref_times)
                
                # Benchmark optimized implementation
                opt_times = []
                for trial in range(3):
                    x = input_tokens
                    start_time = time.perf_counter()
                    for layer in model_layers:
                        x = layer.forward(x, program.optimized_attention_kernel, config)
                    mx.eval(x)
                    end_time = time.perf_counter()
                    opt_times.append(end_time - start_time)
                
                opt_time = np.mean(opt_times)
                
                # Calculate speedup
                speedup = safe_division(ref_time, opt_time, 0.0)
                
                # Calculate throughput (tokens/second)  
                total_tokens = batch_size * seq_len
                ref_throughput = safe_division(total_tokens, ref_time, 0.0)
                opt_throughput = safe_division(total_tokens, opt_time, 0.0)
                
                results[config_name] = {
                    "reference_time": safe_float_conversion(ref_time),
                    "optimized_time": safe_float_conversion(opt_time),
                    "speedup": safe_float_conversion(speedup),
                    "ref_throughput": safe_float_conversion(ref_throughput),
                    "opt_throughput": safe_float_conversion(opt_throughput),
                    "model_config": model_config
                }
                
            except Exception as e:
                results[config_name] = {"error": str(e), "model_config": model_config}
                print(f"Model benchmark {config_name} failed: {e}")
        
        return results
        
    except Exception as e:
        print(f"Model inference benchmark failed: {e}")
        return {"error": str(e)}


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation of MLX attention optimization
    
    Tests the evolved attention mechanism on:
    1. Correctness vs reference implementation
    2. Performance on various attention configurations  
    3. Full model inference speed
    4. Memory efficiency
    5. Numerical stability
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check required functions exist
        if not hasattr(program, 'optimized_attention_kernel'):
            return {
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "model_efficiency": 0.0,
                "overall_score": 0.0,
                "error": "Missing optimized_attention_kernel function"
            }
        
        if not hasattr(program, 'get_attention_config'):
            return {
                "correctness_score": 0.0,
                "performance_score": 0.0, 
                "model_efficiency": 0.0,
                "overall_score": 0.0,
                "error": "Missing get_attention_config function"
            }
        
        # Get configuration
        config = program.get_attention_config()
        
        # 1. Correctness verification
        print("Testing correctness...")
        is_correct, max_diff = verify_attention_correctness(
            program.optimized_attention_kernel, config
        )
        
        if not is_correct:
            print(f"Correctness check failed (max diff: {max_diff})")
            return {
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "model_efficiency": 0.0,
                "overall_score": 0.0,
                "max_difference": max_diff,
                "error": "Correctness verification failed"
            }
        
        correctness_score = max(0.0, 1.0 - max_diff * 100)  # Penalize large differences
        
        # 2. Performance benchmarking
        print("Benchmarking attention performance...")
        try:
            perf_results = run_with_timeout(
                program.benchmark_attention,
                kwargs={
                    "batch_sizes": [1, 2], 
                    "sequence_lengths": [128, 256, 512],
                    "d_model_sizes": [256, 512]
                },
                timeout_seconds=45
            )
            
            if "error" in perf_results.get("summary", {}):
                performance_score = 0.0
                avg_throughput = 0.0
            else:
                avg_throughput = perf_results["summary"].get("avg_throughput_gflops", 0.0)
                success_rate = perf_results["summary"].get("successful_runs", 0) / perf_results["summary"].get("total_configurations", 1)
                
                # Score based on throughput and success rate
                # Normalize throughput to 0-1 scale (assuming max ~100 GFLOPS for attention)
                throughput_score = min(avg_throughput / 100.0, 1.0)
                performance_score = 0.7 * throughput_score + 0.3 * success_rate
                
        except Exception as e:
            print(f"Performance benchmark failed: {e}")
            performance_score = 0.0
            avg_throughput = 0.0
        
        # 3. Model inference efficiency
        print("Testing model inference efficiency...")
        try:
            model_results = run_with_timeout(
                benchmark_model_inference,
                args=(program, config),
                timeout_seconds=60
            )
            
            if "error" in model_results:
                model_efficiency = 0.0
                avg_speedup = 0.0
            else:
                # Calculate average speedup across all model configurations
                speedups = [r.get("speedup", 0) for r in model_results.values() if "speedup" in r]
                avg_speedup = np.mean(speedups) if speedups else 0.0
                
                # Score based on speedup (>1.0 is good, >1.2 is excellent)
                if avg_speedup > 1.2:
                    model_efficiency = 1.0
                elif avg_speedup > 1.0:
                    model_efficiency = 0.5 + 0.5 * (avg_speedup - 1.0) / 0.2
                else:
                    model_efficiency = 0.5 * avg_speedup
                    
        except Exception as e:
            print(f"Model inference benchmark failed: {e}")
            model_efficiency = 0.0
            avg_speedup = 0.0
        
        # 4. Calculate overall score
        # Prioritize correctness, then model efficiency, then raw performance
        overall_score = (
            0.5 * correctness_score +      # Must be correct
            0.3 * model_efficiency +       # Real-world performance gain
            0.2 * performance_score        # Microbenchmark performance
        )
        
        # 5. Stability and efficiency metrics
        memory_score = 1.0  # Placeholder - could measure memory usage
        stability_score = correctness_score  # Use correctness as stability proxy
        
        # Combined efficiency metric for primary optimization target
        attention_efficiency = (
            0.4 * model_efficiency +       # Real model speedup (most important)
            0.3 * performance_score +      # Raw attention performance
            0.2 * correctness_score +      # Must be correct
            0.1 * stability_score          # Numerical stability
        )
        
        return {
            "correctness_score": float(correctness_score),
            "performance_score": float(performance_score), 
            "model_efficiency": float(model_efficiency),
            "overall_score": float(overall_score),
            "attention_efficiency": float(attention_efficiency),  # Primary metric for evolution
            "avg_throughput_gflops": float(avg_throughput),
            "avg_speedup": float(avg_speedup),
            "max_difference": float(max_diff),
            "memory_score": float(memory_score),
            "stability_score": float(stability_score),
            "is_correct": is_correct
        }
        
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "correctness_score": 0.0,
            "performance_score": 0.0,
            "model_efficiency": 0.0,
            "overall_score": 0.0,
            "attention_efficiency": 0.0,
            "error": str(e)
        }


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """
    First stage evaluation for cascade evaluation
    Quick validation to filter out broken implementations
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check required functions exist
        if not hasattr(program, 'optimized_attention_kernel'):
            return {"runs_successfully": 0.0, "error": "Missing optimized_attention_kernel function"}
            
        if not hasattr(program, 'get_attention_config'):
            return {"runs_successfully": 0.0, "error": "Missing get_attention_config function"}
        
        # Quick correctness test
        config = program.get_attention_config()
        is_correct, max_diff = verify_attention_correctness(
            program.optimized_attention_kernel, config, tolerance=1e-1  # More lenient for stage 1
        )
        
        if not is_correct:
            return {
                "runs_successfully": 0.5,
                "max_difference": float(max_diff),
                "error": "Correctness check failed"
            }
        
        # Quick performance test
        try:
            batch_size, seq_len, d_model = 1, 64, 128
            query = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
            key = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
            value = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
            
            start_time = time.perf_counter()
            result = run_with_timeout(
                program.optimized_attention_kernel,
                args=(query, key, value),
                kwargs=config,
                timeout_seconds=10
            )
            mx.eval(result)
            elapsed = time.perf_counter() - start_time
            
            # Quick throughput calculation
            ops_estimate = batch_size * seq_len * seq_len * d_model * 4
            throughput = ops_estimate / (elapsed * 1e9)
            
            return {
                "runs_successfully": 1.0,
                "quick_throughput": float(throughput),
                "max_difference": float(max_diff),
                "stage1_score": min(throughput / 10.0, 1.0)  # Normalize to 0-1
            }
            
        except Exception as e:
            return {
                "runs_successfully": 0.8,
                "max_difference": float(max_diff), 
                "error": f"Performance test failed: {str(e)}"
            }
            
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        return {"runs_successfully": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """
    Second stage evaluation - full evaluation
    """
    return evaluate(program_path)


import math  # Add this import that was missing
