"""
Evaluator for MLX Attention Optimization

This evaluator tests evolved attention implementations for:
1. Numerical accuracy compared to reference implementation
2. Performance (throughput in tokens/second)
3. Memory efficiency 
4. Robustness across different input sizes

The key requirement is that evolved attention must be functionally equivalent
to the reference while potentially offering performance improvements.
"""

import gc
import importlib.util
import math
import psutil
import time
import traceback
from typing import Dict, List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ReferenceAttention(nn.Module):
    """
    Reference attention implementation using MLX's built-in scaled_dot_product_attention.
    This serves as the ground truth for accuracy comparisons.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, scale: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        
    def __call__(
        self, 
        queries: mx.array, 
        keys: mx.array, 
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[any] = None
    ) -> mx.array:
        """Reference implementation using MLX's optimized attention - this is our baseline to beat"""
        try:
            # Use MLX's optimized implementation as the baseline that evolved code must beat
            processed_mask = mask
            if mask is not None and mask.ndim == 3:  # [B, L, L_kv]
                processed_mask = mx.expand_dims(mask, axis=1)  # [B, 1, L, L_kv]
            
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=processed_mask
            )
        except (AttributeError, ImportError):
            # Fallback to manual implementation if mx.fast not available
            print("Using manual reference implementation (mx.fast not available)")
            return self._manual_attention(queries, keys, values, mask)
    
    def _manual_attention(
        self, 
        queries: mx.array, 
        keys: mx.array, 
        values: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """Manual implementation - should match evolved attention closely"""
        # Handle grouped query attention (GQA) by repeating KV heads if needed
        B, num_heads, L, head_dim = queries.shape
        _, num_kv_heads, L_kv, _ = keys.shape
        
        if num_kv_heads != num_heads:
            # Repeat keys and values to match query heads
            rep_factor = num_heads // num_kv_heads
            keys = mx.repeat(keys, rep_factor, axis=1)
            values = mx.repeat(values, rep_factor, axis=1)
        
        # Standard scaled dot-product attention
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2))
        scores = scores * self.scale
        
        if mask is not None:
            if mask.ndim == 3:  # [B, L, L_kv]
                mask = mx.expand_dims(mask, axis=1)  # [B, 1, L, L_kv]
            scores = scores + mask
        
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, values)
        
        return output


def create_reference_module(
    hidden_size: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 8, 
    head_dim: int = 64,
    eps: float = 1e-6
):
    """Create reference attention module for comparison"""
    
    class ReferenceModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5
            
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            
            self.q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=eps)
            
            self.reference_attention = ReferenceAttention(
                hidden_size, num_heads, num_kv_heads, head_dim, self.scale
            )
            
        def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
            B, L, D = x.shape
            
            queries = self.q_proj(x)
            keys = self.k_proj(x)
            values = self.v_proj(x)
            
            queries = self.q_norm(
                queries.reshape(B, L, self.num_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            
            keys = self.k_norm(
                keys.reshape(B, L, self.num_kv_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            
            values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
            
            output = self.reference_attention(queries, keys, values, mask=mask)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            
            return self.o_proj(output)
    
    return ReferenceModule()


def measure_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def create_test_cases() -> List[Dict]:
    """Create diverse test cases for evaluation, focusing on standard cases first"""
    return [
        # Small cases for debugging
        {"batch_size": 1, "seq_len": 64, "hidden_size": 256, "num_heads": 4, "num_kv_heads": 4},
        {"batch_size": 2, "seq_len": 128, "hidden_size": 512, "num_heads": 8, "num_kv_heads": 8},
        
        # Standard cases (non-GQA) - these should work reliably
        {"batch_size": 1, "seq_len": 512, "hidden_size": 768, "num_heads": 12, "num_kv_heads": 12},
        {"batch_size": 4, "seq_len": 256, "hidden_size": 1024, "num_heads": 16, "num_kv_heads": 16},
        {"batch_size": 1, "seq_len": 1024, "hidden_size": 512, "num_heads": 8, "num_kv_heads": 8},
        
        # Grouped Query Attention (GQA) cases - test these separately
        {"batch_size": 1, "seq_len": 256, "hidden_size": 512, "num_heads": 8, "num_kv_heads": 2},
        {"batch_size": 1, "seq_len": 256, "hidden_size": 768, "num_heads": 12, "num_kv_heads": 4},
        {"batch_size": 1, "seq_len": 512, "hidden_size": 1024, "num_heads": 16, "num_kv_heads": 8},
    ]


def compare_outputs(output1: mx.array, output2: mx.array, tolerance: float = 1e-4) -> Dict[str, float]:
    """Compare two outputs and return similarity metrics"""
    
    # Ensure arrays are materialized
    output1 = mx.array(output1)
    output2 = mx.array(output2)
    
    # Mean Squared Error
    mse = float(mx.mean((output1 - output2) ** 2))
    
    # Mean Absolute Error  
    mae = float(mx.mean(mx.abs(output1 - output2)))
    
    # Cosine similarity
    output1_flat = output1.reshape(-1)
    output2_flat = output2.reshape(-1)
    
    dot_product = float(mx.sum(output1_flat * output2_flat))
    norm1 = float(mx.sqrt(mx.sum(output1_flat ** 2)))
    norm2 = float(mx.sqrt(mx.sum(output2_flat ** 2)))
    
    cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    # Maximum absolute difference
    max_diff = float(mx.max(mx.abs(output1 - output2)))
    
    # Check if within tolerance
    within_tolerance = mse < tolerance
    
    return {
        "mse": mse,
        "mae": mae, 
        "cosine_similarity": cosine_sim,
        "max_diff": max_diff,
        "within_tolerance": within_tolerance,
        "tolerance_used": tolerance
    }


def benchmark_performance(module, test_case: Dict, num_runs: int = 10) -> Dict[str, float]:
    """Benchmark performance of an attention module"""
    
    batch_size = test_case["batch_size"]
    seq_len = test_case["seq_len"] 
    hidden_size = test_case["hidden_size"]
    
    # Create test input
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Create causal mask
    mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
    mask = mx.expand_dims(mask, axis=0)  # Add batch dimension
    
    # Warmup runs
    for _ in range(3):
        _ = module(x, mask=mask)
        mx.eval(_)  # Ensure computation is complete
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        output = module(x, mask=mask)
        mx.eval(output)  # Ensure computation is complete
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Calculate throughput
    total_tokens = batch_size * seq_len
    tokens_per_second = total_tokens / avg_time if avg_time > 0 else 0
    
    return {
        "avg_time_seconds": avg_time,
        "std_time_seconds": std_time,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens
    }


def test_numerical_stability(module, test_case: Dict) -> Dict[str, float]:
    """Test numerical stability with edge cases"""
    
    batch_size = test_case["batch_size"]
    seq_len = test_case["seq_len"]
    hidden_size = test_case["hidden_size"]
    
    stability_scores = []
    
    # Test cases for stability
    test_inputs = [
        # Normal case
        mx.random.normal((batch_size, seq_len, hidden_size)),
        # Small values
        mx.random.normal((batch_size, seq_len, hidden_size)) * 0.01,
        # Large values
        mx.random.normal((batch_size, seq_len, hidden_size)) * 10.0,
        # Near-zero values
        mx.random.normal((batch_size, seq_len, hidden_size)) * 1e-6,
    ]
    
    for i, x in enumerate(test_inputs):
        try:
            output = module(x)
            mx.eval(output)
            
            # Check for NaN or Inf
            has_nan = bool(mx.any(mx.isnan(output)))
            has_inf = bool(mx.any(mx.isinf(output)))
            
            if has_nan or has_inf:
                stability_scores.append(0.0)
            else:
                stability_scores.append(1.0)
                
        except Exception as e:
            print(f"Stability test {i} failed: {str(e)}")
            stability_scores.append(0.0)
    
    return {
        "stability_score": np.mean(stability_scores),
        "num_stable_cases": sum(stability_scores),
        "total_cases": len(stability_scores)
    }


def copy_compatible_weights(source_module, target_module):
    """
    Copy weights between modules only if they have compatible dimensions.
    This handles cases where architectures might differ slightly.
    """
    copied_weights = 0
    
    try:
        # List of weight pairs to try copying
        weight_pairs = [
            ('q_proj', 'q_proj'),
            ('k_proj', 'k_proj'), 
            ('v_proj', 'v_proj'),
            ('o_proj', 'o_proj'),
            ('q_norm', 'q_norm'),
            ('k_norm', 'k_norm')
        ]
        
        for source_attr, target_attr in weight_pairs:
            if hasattr(source_module, source_attr) and hasattr(target_module, target_attr):
                source_layer = getattr(source_module, source_attr)
                target_layer = getattr(target_module, target_attr)
                
                # Check if both have weight attributes and compatible shapes
                if (hasattr(source_layer, 'weight') and hasattr(target_layer, 'weight') and
                    source_layer.weight.shape == target_layer.weight.shape):
                    target_layer.weight = mx.array(source_layer.weight)
                    copied_weights += 1
        
        return copied_weights > 0
        
    except Exception as e:
        print(f"Weight copying failed: {str(e)}")
        return False


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Main evaluation function for evolved attention implementations.
    
    Tests accuracy, performance, memory efficiency, and stability.
    """
    
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        # Check if required function exists
        if not hasattr(evolved_program, "create_test_attention_module"):
            return {
                "accuracy_score": 0.0,
                "performance_score": 0.0,
                "memory_efficiency": 0.0,
                "stability_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing create_test_attention_module function"
            }
        
        test_cases = create_test_cases()
        
        accuracy_scores = []
        performance_scores = []
        memory_scores = []
        stability_scores = []
        
        successful_cases = 0
        
        for i, test_case in enumerate(test_cases):
            try:
                print(f"Evaluating test case {i+1}/{len(test_cases)}: {test_case}")
                
                # Create both evolved and reference modules
                hidden_size = test_case["hidden_size"]
                num_heads = test_case["num_heads"] 
                num_kv_heads = test_case["num_kv_heads"]
                head_dim = hidden_size // num_heads
                
                evolved_module = evolved_program.create_test_attention_module(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim
                )
                
                reference_module = create_reference_module(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim
                )
                
                # Try to copy compatible weights for fair comparison
                weights_copied = copy_compatible_weights(evolved_module, reference_module)
                if weights_copied:
                    print("  Applied shared weights for fair comparison")
                else:
                    print("  Using different random weights (architectures incompatible)")
                
                # Create test input
                batch_size = test_case["batch_size"]
                seq_len = test_case["seq_len"]
                x = mx.random.normal((batch_size, seq_len, hidden_size))
                
                # Create causal mask
                mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
                mask = mx.expand_dims(mask, axis=0)
                
                # Test basic functionality first
                evolved_output = evolved_module(x, mask=mask)
                mx.eval(evolved_output)
                
                # Check basic structural correctness
                expected_shape = (batch_size, seq_len, hidden_size)
                structural_ok = (
                    evolved_output.shape == expected_shape and
                    not bool(mx.any(mx.isnan(evolved_output))) and
                    not bool(mx.any(mx.isinf(evolved_output)))
                )
                
                if not structural_ok:
                    print(f"  Structural check failed: shape={evolved_output.shape}, has_nan={bool(mx.any(mx.isnan(evolved_output)))}")
                    accuracy_scores.append(0.0)
                    performance_scores.append(0.0)
                    memory_scores.append(0.0)
                    stability_scores.append(0.0)
                    continue
                
                # If weights are shared, do numerical comparison
                if weights_copied:
                    reference_output = reference_module(x, mask=mask)
                    mx.eval(reference_output)
                    
                    comparison = compare_outputs(evolved_output, reference_output, tolerance=1e-2)
                    
                    # More lenient accuracy scoring
                    if comparison["within_tolerance"]:
                        accuracy_score = 1.0
                    elif comparison["cosine_similarity"] > 0.95:
                        accuracy_score = 0.9
                    elif comparison["cosine_similarity"] > 0.90:
                        accuracy_score = 0.8
                    elif comparison["cosine_similarity"] > 0.80:
                        accuracy_score = 0.7
                    else:
                        accuracy_score = max(0.6, comparison["cosine_similarity"])
                    
                    print(f"  Accuracy: {accuracy_score:.3f} (cosine_sim: {comparison['cosine_similarity']:.3f}, mse: {comparison['mse']:.6f})")
                else:
                    # If we can't sync weights, just check that it works structurally  
                    accuracy_score = 0.8  # Partial credit for working implementation
                    print(f"  Accuracy: {accuracy_score:.3f} (structural check only - no weight sync)")
                
                accuracy_scores.append(accuracy_score)
                
                # Performance and other tests
                gc.collect()
                memory_before = measure_memory_usage()
                
                # Performance test
                perf_results = benchmark_performance(evolved_module, test_case, num_runs=3)
                
                # Memory after  
                memory_after = measure_memory_usage()
                memory_used = memory_after - memory_before
                
                # Compare with reference if possible
                if weights_copied:
                    ref_perf_results = benchmark_performance(reference_module, test_case, num_runs=3)
                    if ref_perf_results["tokens_per_second"] > 0:
                        speedup = perf_results["tokens_per_second"] / ref_perf_results["tokens_per_second"]
                        performance_score = min(speedup, 3.0)  # Cap at 3x speedup
                        print(f"  Performance: {performance_score:.3f}x speedup")
                    else:
                        performance_score = 1.0
                else:
                    performance_score = 1.0  # Neutral score
                    print(f"  Performance: {performance_score:.3f} (no reference comparison)")
                
                performance_scores.append(performance_score)
                
                # Memory efficiency (tokens per MB)
                if memory_used > 0:
                    memory_efficiency = perf_results["total_tokens"] / max(memory_used, 1.0)
                    memory_scores.append(min(memory_efficiency / 1000.0, 2.0))  # Normalize and cap
                else:
                    memory_scores.append(1.0)
                
                # Test stability
                stability_result = test_numerical_stability(evolved_module, test_case)
                stability_scores.append(stability_result["stability_score"])
                print(f"  Stability: {stability_result['stability_score']:.3f}")
                
                successful_cases += 1
                
            except Exception as e:
                print(f"Test case {i} failed: {str(e)}")
                # Don't print full traceback for dimension errors - they're expected for some GQA cases
                if "matmul" not in str(e).lower():
                    print(traceback.format_exc())
                accuracy_scores.append(0.0)
                performance_scores.append(0.0)
                memory_scores.append(0.0)
                stability_scores.append(0.0)
        
        # Calculate final scores
        if successful_cases == 0:
            return {
                "accuracy_score": 0.0,
                "performance_score": 0.0,
                "memory_efficiency": 0.0,
                "stability_score": 0.0,
                "combined_score": 0.0,
                "success_rate": 0.0,
                "error": "No test cases passed"
            }
        
        # Average scores across all test cases
        avg_accuracy = np.mean(accuracy_scores)
        avg_performance = np.mean(performance_scores) 
        avg_memory = np.mean(memory_scores)
        avg_stability = np.mean(stability_scores)
        success_rate = successful_cases / len(test_cases)
        
        # Combined score weights accuracy heavily, then performance, memory, and stability
        combined_score = (
            0.50 * avg_accuracy +      # Accuracy is most important
            0.25 * avg_performance +   # Performance improvement is valuable
            0.15 * avg_memory +        # Memory efficiency matters
            0.10 * avg_stability       # Stability is expected but important
        ) * success_rate               # Penalize if many test cases fail
        
        return {
            "accuracy_score": float(avg_accuracy),
            "performance_score": float(avg_performance),
            "memory_efficiency": float(avg_memory), 
            "stability_score": float(avg_stability),
            "combined_score": float(combined_score),
            "success_rate": float(success_rate),
            "successful_cases": successful_cases,
            "total_cases": len(test_cases)
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            "accuracy_score": 0.0,
            "performance_score": 0.0,
            "memory_efficiency": 0.0,
            "stability_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


# Staged evaluation functions for cascade evaluation
def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """Quick accuracy check on a simple test case"""
    try:
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "create_test_attention_module"):
            return {"basic_functionality": 0.0, "error": "Missing required function"}
        
        # Simple test case - non-GQA to avoid complexity
        evolved_module = evolved_program.create_test_attention_module(
            hidden_size=256, num_heads=4, num_kv_heads=4, head_dim=64
        )
        
        # Test basic functionality
        x = mx.random.normal((1, 64, 256))
        evolved_output = evolved_module(x)
        
        mx.eval(evolved_output)
        
        # Check if output is reasonable
        structural_check = (
            evolved_output.shape == (1, 64, 256) and
            not bool(mx.any(mx.isnan(evolved_output))) and
            not bool(mx.any(mx.isinf(evolved_output))) and
            abs(float(mx.mean(evolved_output))) < 100.0
        )
        
        return {
            "basic_functionality": 1.0 if structural_check else 0.0,
            "output_shape_correct": evolved_output.shape == (1, 64, 256),
            "no_nan_inf": not bool(mx.any(mx.isnan(evolved_output)) or mx.any(mx.isinf(evolved_output)))
        }
        
    except Exception as e:
        print(f"Stage 1 evaluation failed: {str(e)}")
        return {"basic_functionality": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """More thorough testing on multiple cases"""
    return evaluate(program_path)


if __name__ == "__main__":
    # Test the evaluator with the initial program
    print("Testing evaluator with initial program...")
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("Evaluation results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print(f"Initial program not found at {initial_program_path}")
