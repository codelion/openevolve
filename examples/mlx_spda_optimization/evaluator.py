"""
Evaluator for Block-Diagonal Attention Kernel Evolution

Tests both correctness and performance of evolved Metal kernels for 
block-diagonal attention with packed sequences.

Focus areas:
1. Correctness vs reference implementation
2. Performance improvements over naive masking
3. Efficiency with different packing patterns
4. Memory usage and scaling
"""

import importlib.util
import math
import time
import traceback
from typing import Dict, List, Tuple, Union
import gc

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX or NumPy not available")
    MLX_AVAILABLE = False

# Import benchmark utilities
try:
    from spda_benchmark import prepare_inputs, mlx_ref_attn
    BENCHMARK_AVAILABLE = True
except ImportError:
    print("⚠️ Benchmark utilities not available")
    BENCHMARK_AVAILABLE = False


def create_block_diagonal_mask(batch_size, num_heads, seq_len, block_sizes):
    """
    Create a block-diagonal mask for packed sequences.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Total sequence length
        block_sizes: List of individual sequence lengths that are packed
    
    Returns:
        Boolean mask where True indicates valid attention positions
    """
    # Use numpy to create the mask efficiently, then convert to MLX
    mask_np = np.zeros((batch_size, num_heads, seq_len, seq_len), dtype=bool)
    
    current_pos = 0
    for block_size in block_sizes:
        if current_pos + block_size <= seq_len:
            end_pos = current_pos + block_size
            # Set the block diagonal region to True
            mask_np[:, :, current_pos:end_pos, current_pos:end_pos] = True
            current_pos = end_pos
        else:
            break
    
    return mx.array(mask_np)


def naive_masked_attention(q, k, v, scale, mask):
    """
    Naive implementation using standard attention with masking.
    This is what we want to beat with our custom kernel.
    """
    # Standard attention computation
    scores = (q * scale) @ mx.swapaxes(k, -1, -2)
    
    # Apply mask
    if mask is not None:
        if hasattr(mask, 'dtype') and mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -mx.array(np.float32(np.inf)))
        else:
            scores = scores + mask
    
    # Softmax and output
    attn_weights = mx.softmax(scores, axis=-1, precise=True)
    return attn_weights @ v


def create_test_configurations():
    """
    Create test configurations for block-diagonal attention evaluation.
    
    Includes various packing scenarios and sequence lengths.
    """
    configs = []
    
    # Regular attention tests (baseline)
    configs.extend([
        {
            "name": "regular_short",
            "B": 1, "H": 8, "L": 128, "D": 64,
            "type": "regular",
            "mask": None,
            "expected_improvement": False,
        },
        {
            "name": "regular_medium", 
            "B": 1, "H": 16, "L": 256, "D": 64,
            "type": "regular",
            "mask": "causal",
            "expected_improvement": False,
        }
    ])
    
    # Block-diagonal tests (main target)
    configs.extend([
        {
            "name": "packed_2x256",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "type": "block_diagonal",
            "block_sizes": [256, 256],  # Two sequences of 256 tokens each
            "expected_improvement": True,
        },
        {
            "name": "packed_4x128",
            "B": 1, "H": 16, "L": 512, "D": 64, 
            "type": "block_diagonal",
            "block_sizes": [128, 128, 128, 128],  # Four sequences of 128 tokens
            "expected_improvement": True,
        },
        {
            "name": "packed_variable",
            "B": 1, "H": 8, "L": 768, "D": 64,
            "type": "block_diagonal", 
            "block_sizes": [256, 512],  # Variable length sequences
            "expected_improvement": True,
        },
        {
            "name": "packed_large",
            "B": 1, "H": 32, "L": 1024, "D": 64,
            "type": "block_diagonal",
            "block_sizes": [256, 256, 256, 256],  # Large packed sequences
            "expected_improvement": True,
        },
        {
            "name": "packed_bert_style",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "type": "block_diagonal",
            "block_sizes": [128, 128, 128, 128],  # BERT-style packing
            "expected_improvement": True,
        }
    ])
    
    return configs


def evaluate_correctness(evolved_fn, config):
    """
    Test correctness of evolved attention against reference implementation.
    """
    try:
        # Prepare inputs
        B, H, L, D = config["B"], config["H"], config["L"], config["D"]
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        scale = 1.0 / math.sqrt(D)
        
        # Create appropriate mask
        if config["type"] == "regular":
            if config.get("mask") == "causal":
                # Causal mask
                causal_mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))
                mask = mx.broadcast_to(causal_mask[None, None, :, :], (B, H, L, L))
            else:
                mask = None
        elif config["type"] == "block_diagonal":
            # Block-diagonal mask for packed sequences
            mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
        else:
            mask = None
        
        # Run evolved implementation
        evolved_output = evolved_fn(q, k, v, scale=scale, mask=mask)
        
        # Run reference implementation (naive masked attention)
        reference_output = naive_masked_attention(q, k, v, scale, mask)
        
        # Compare outputs
        if evolved_output.shape != reference_output.shape:
            return {
                "passed": False,
                "error": f"Shape mismatch: {evolved_output.shape} vs {reference_output.shape}"
            }
        
        # Calculate error metrics
        diff = evolved_output - reference_output
        mse = float(mx.mean(diff ** 2))
        max_diff = float(mx.max(mx.abs(diff)))
        
        # Check for valid output
        has_nan = bool(mx.any(mx.isnan(evolved_output)))
        has_inf = bool(mx.any(mx.isinf(evolved_output)))
        
        # Determine if test passed
        passed = (mse < 1e-3 and max_diff < 0.1 and not has_nan and not has_inf)
        
        return {
            "passed": passed,
            "mse": mse,
            "max_diff": max_diff,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "config_name": config["name"]
        }
        
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "config_name": config["name"]
        }


def benchmark_performance(evolved_fn, config, num_trials=3):
    """
    Benchmark performance of evolved implementation vs naive masking.
    """
    try:
        # Prepare inputs
        B, H, L, D = config["B"], config["H"], config["L"], config["D"]
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        scale = 1.0 / math.sqrt(D)
        
        # Create mask
        if config["type"] == "block_diagonal":
            mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
        elif config.get("mask") == "causal":
            causal_mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))
            mask = mx.broadcast_to(causal_mask[None, None, :, :], (B, H, L, L))
        else:
            mask = None
        
        # Benchmark evolved implementation
        evolved_times = []
        for _ in range(num_trials):
            try:
                gc.collect()
                if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                    mx.metal.clear_cache()
                
                start_time = time.perf_counter()
                output = evolved_fn(q, k, v, scale=scale, mask=mask)
                mx.eval(output)
                end_time = time.perf_counter()
                
                evolved_times.append(end_time - start_time)
            except Exception:
                return {"speedup": 0.0, "error": "Evolved implementation failed"}
        
        # Benchmark naive implementation
        naive_times = []
        for _ in range(num_trials):
            try:
                gc.collect()
                if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                    mx.metal.clear_cache()
                
                start_time = time.perf_counter()
                output = naive_masked_attention(q, k, v, scale, mask)
                mx.eval(output)
                end_time = time.perf_counter()
                
                naive_times.append(end_time - start_time)
            except Exception:
                return {"speedup": float("inf"), "baseline_failed": True}
        
        # Calculate speedup
        evolved_time = np.median(evolved_times)
        naive_time = np.median(naive_times)
        speedup = naive_time / evolved_time if evolved_time > 0 else 0.0
        
        return {
            "speedup": speedup,
            "evolved_time": evolved_time,
            "naive_time": naive_time,
            "config_name": config["name"]
        }
        
    except Exception as e:
        return {
            "speedup": 0.0,
            "error": str(e),
            "config_name": config["name"]
        }


def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Main evaluation function for block-diagonal attention evolution.
    
    Tests both correctness and performance across various scenarios.
    """
    print(f"🎯 Evaluating Block-Diagonal Attention: {program_path}")
    
    if not MLX_AVAILABLE:
        return {
            "stage1_passed": False,
            "overall_score": 0.0,
            "error": "MLX not available"
        }
    
    try:
        # Load evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            return {
                "stage1_passed": False,
                "overall_score": 0.0,
                "error": "Missing evolved_scaled_dot_product_attention function"
            }
        
        evolved_fn = evolved_program.evolved_scaled_dot_product_attention
        
        # ===== STAGE 1: CORRECTNESS TESTING =====
        print("\n📋 STAGE 1: Correctness Testing")
        
        test_configs = create_test_configurations()
        correctness_results = []
        passed_count = 0
        
        for config in test_configs:
            print(f"  Testing {config['name']}: {config['type']}")
            
            result = evaluate_correctness(evolved_fn, config)
            correctness_results.append(result)
            
            if result["passed"]:
                passed_count += 1
                print(f"    ✅ PASSED (MSE: {result.get('mse', 0):.2e})")
            else:
                error_msg = result.get("error", "Accuracy issue")
                print(f"    ❌ FAILED: {error_msg}")
        
        # Calculate pass rate
        pass_rate = passed_count / len(test_configs) if test_configs else 0.0
        stage1_passed = pass_rate >= 0.8  # 80% pass rate required
        
        print(f"\n📊 STAGE 1 Results:")
        print(f"  Passed: {passed_count}/{len(test_configs)} ({pass_rate:.1%})")
        print(f"  Status: {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        
        if not stage1_passed:
            return {
                "stage1_passed": False,
                "pass_rate": pass_rate,
                "overall_score": 0.0,
                "failed_at": "correctness"
            }
        
        # ===== STAGE 2: PERFORMANCE TESTING =====
        print(f"\n🚀 STAGE 2: Performance Testing")
        
        performance_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for config in test_configs:
            if config["type"] == "block_diagonal":  # Only test performance on target scenarios
                print(f"  Benchmarking {config['name']}")
                
                result = benchmark_performance(evolved_fn, config)
                performance_results.append(result)
                
                speedup = result.get("speedup", 0.0)
                
                # Weight by sequence length (longer sequences more important)
                weight = config["L"] / 512.0  # Normalize by 512
                
                # Score based on speedup
                if speedup >= 2.0:  # 2x speedup
                    score = 1.0
                elif speedup >= 1.5:  # 1.5x speedup  
                    score = 0.7
                elif speedup >= 1.2:  # 1.2x speedup
                    score = 0.5
                elif speedup >= 1.0:  # Any speedup
                    score = 0.3
                else:
                    score = 0.0
                
                weighted_score = score * weight
                total_weighted_score += weighted_score
                total_weight += weight
                
                print(f"    📊 Speedup: {speedup:.2f}x, Score: {score:.2f}")
        
        # Calculate overall performance score
        stage2_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        overall_score = stage2_score  # Stage 1 is just a gate
        
        # Analyze performance results
        speedups = [r.get("speedup", 0.0) for r in performance_results if "speedup" in r]
        avg_speedup = np.mean(speedups) if speedups else 0.0
        max_speedup = max(speedups) if speedups else 0.0
        
        print(f"\n📈 STAGE 2 Results:")
        print(f"  Performance Score: {stage2_score:.3f}")
        print(f"  Average Speedup: {avg_speedup:.2f}x") 
        print(f"  Max Speedup: {max_speedup:.2f}x")
        
        print(f"\n🎯 Overall Results:")
        print(f"  Stage 1: {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        print(f"  Stage 2: {stage2_score:.3f}")
        print(f"  Overall Score: {overall_score:.3f}")
        
        if overall_score >= 0.8:
            print(f"  🏆 EXCELLENT: Strong Metal kernel optimization!")
        elif overall_score >= 0.5:
            print(f"  🚀 GOOD: Meaningful improvements achieved")
        elif overall_score >= 0.2:
            print(f"  ⚡ PARTIAL: Some optimization, room for improvement")
        else:
            print(f"  ❌ POOR: Needs significant kernel optimization")
        
        return {
            "stage1_passed": stage1_passed,
            "pass_rate": float(pass_rate),
            "stage2_score": float(stage2_score),
            "overall_score": float(overall_score),
            "avg_speedup": float(avg_speedup),
            "max_speedup": float(max_speedup),
            "num_tests": len(test_configs),
            "num_performance_tests": len(performance_results)
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "stage1_passed": False,
            "overall_score": 0.0,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Testing Block-Diagonal Attention Evaluator...")
    
    # Test with initial program
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("\nEvaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
