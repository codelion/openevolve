"""
Evaluator for Custom Metal Kernel Evolution

Tests custom Metal kernels for block-diagonal attention against MLX's optimized
mx.fast.scaled_dot_product_attention implementation.

Focus: Evolution should discover kernels that outperform SPDA on packed sequences
by skipping computation on masked regions entirely.
"""

import importlib.util
import math
import time
import traceback
from typing import Dict, Union
import gc

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX or NumPy not available")
    MLX_AVAILABLE = False


def create_block_diagonal_mask(B, H, L, block_sizes):
    """Create block-diagonal mask for packed sequences."""
    mask_np = np.zeros((B, H, L, L), dtype=bool)
    
    current_pos = 0
    for block_size in block_sizes:
        if current_pos + block_size <= L:
            end_pos = current_pos + block_size
            mask_np[:, :, current_pos:end_pos, current_pos:end_pos] = True
            current_pos = end_pos
        else:
            break
    
    return mx.array(mask_np)


def reference_attention(q, k, v, scale, mask):
    """Reference implementation for correctness checking."""
    scores = (q * scale) @ mx.swapaxes(k, -1, -2)
    
    if mask is not None:
        if hasattr(mask, 'dtype') and mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -mx.array(np.float32(np.inf)))
        else:
            scores = scores + mask
    
    attn_weights = mx.softmax(scores, axis=-1, precise=True)
    return attn_weights @ v


def mlx_spda_baseline(q, k, v, scale, mask):
    """MLX fast SPDA implementation - our performance baseline."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def create_test_configurations():
    """Create test configurations focusing on block-diagonal advantage scenarios."""
    configs = []
    
    # === STAGE 1: CORRECTNESS TESTS ===
    # These test correctness across various scenarios
    
    configs.extend([
        {
            "name": "small_uniform_blocks", 
            "B": 1, "H": 4, "L": 128, "D": 64,
            "block_sizes": [64, 64],  # 2 blocks of 64
            "test_type": "correctness",
            "expected_advantage": True
        },
        {
            "name": "medium_uniform_blocks",
            "B": 1, "H": 8, "L": 512, "D": 64, 
            "block_sizes": [128, 128, 128, 128],  # 4 blocks of 128
            "test_type": "correctness",
            "expected_advantage": True
        },
        {
            "name": "variable_blocks",
            "B": 1, "H": 8, "L": 768, "D": 64,
            "block_sizes": [256, 512],  # Variable sizes
            "test_type": "correctness", 
            "expected_advantage": True
        },
        {
            "name": "single_large_block",
            "B": 1, "H": 4, "L": 256, "D": 64,
            "block_sizes": [256],  # Single block (edge case)
            "test_type": "correctness",
            "expected_advantage": False
        }
    ])
    
    # === STAGE 2: PERFORMANCE TESTS ===
    # These focus on scenarios where block-diagonal should significantly outperform SPDA
    
    configs.extend([
        {
            "name": "sparse_large_blocks",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128, 128, 128, 128, 128, 128, 128, 128],  # 8 small blocks = very sparse
            "test_type": "performance",
            "expected_advantage": True,
            "advantage_reason": "87.5% of attention matrix is masked (7/8 blocks empty)"
        },
        {
            "name": "packed_sequences_medium",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],  # BERT-style packing
            "test_type": "performance", 
            "expected_advantage": True,
            "advantage_reason": "75% of attention matrix is masked (3/4 cross-sequence interactions)"
        },
        {
            "name": "very_sparse_packing",
            "B": 1, "H": 32, "L": 2048, "D": 64,
            "block_sizes": [256, 256, 256, 256, 256, 256, 256, 256],  # 8 blocks
            "test_type": "performance",
            "expected_advantage": True, 
            "advantage_reason": "87.5% of attention matrix is masked"
        },
        {
            "name": "extreme_sparse_packing",
            "B": 1, "H": 16, "L": 1024, "D": 128,
            "block_sizes": [64] * 16,  # 16 tiny blocks = extremely sparse
            "test_type": "performance",
            "expected_advantage": True,
            "advantage_reason": "93.75% of attention matrix is masked (15/16 blocks empty)"
        },
        {
            "name": "dense_packing_baseline",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256],  # Only 2 large blocks = less sparse
            "test_type": "performance",
            "expected_advantage": True,
            "advantage_reason": "50% of attention matrix is masked"
        }
    ])
    
    return configs


def evaluate_correctness(evolved_fn, config):
    """Test correctness against reference implementation."""
    try:
        B, H, L, D = config["B"], config["H"], config["L"], config["D"]
        
        # Create test inputs
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        scale = 1.0 / math.sqrt(D)
        
        # Create block-diagonal mask
        mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
        
        # Run evolved implementation
        evolved_output = evolved_fn(q, k, v, scale=scale, mask=mask)
        
        # Run reference implementation
        reference_output = reference_attention(q, k, v, scale, mask)
        
        # Compare outputs
        if evolved_output.shape != reference_output.shape:
            return {
                "passed": False,
                "error": f"Shape mismatch: {evolved_output.shape} vs {reference_output.shape}",
                "config_name": config["name"]
            }
        
        # Calculate error metrics
        diff = evolved_output - reference_output
        mse = float(mx.mean(diff ** 2))
        max_diff = float(mx.max(mx.abs(diff)))
        
        # Check for invalid outputs
        has_nan = bool(mx.any(mx.isnan(evolved_output)))
        has_inf = bool(mx.any(mx.isinf(evolved_output)))
        
        # Determine pass/fail
        tolerance = 1e-3 if q.dtype == mx.float32 else 1e-2
        passed = mse < tolerance and max_diff < 0.1 and not has_nan and not has_inf
        
        return {
            "passed": passed,
            "mse": mse,
            "max_diff": max_diff,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "config_name": config["name"],
            "tolerance_used": tolerance
        }
        
    except Exception as e:
        return {
            "passed": False, 
            "error": str(e),
            "config_name": config["name"]
        }


def benchmark_performance(evolved_fn, config, num_trials=5):
    """Benchmark evolved kernel vs MLX fast SPDA."""
    try:
        B, H, L, D = config["B"], config["H"], config["L"], config["D"]
        
        # Create test inputs  
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        scale = 1.0 / math.sqrt(D)
        
        # Create block-diagonal mask
        mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
        
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
            except Exception as e:
                return {"speedup": 0.0, "error": f"Evolved kernel failed: {str(e)}"}
        
        # Benchmark MLX fast SPDA
        spda_times = []
        for _ in range(num_trials):
            try:
                gc.collect()
                if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                    mx.metal.clear_cache()
                
                start_time = time.perf_counter()
                output = mlx_spda_baseline(q, k, v, scale, mask)
                mx.eval(output)
                end_time = time.perf_counter()
                
                spda_times.append(end_time - start_time)
            except Exception as e:
                return {"speedup": float("inf"), "error": f"SPDA baseline failed: {str(e)}"}
        
        # Calculate speedup
        evolved_time = np.median(evolved_times)
        spda_time = np.median(spda_times)
        speedup = spda_time / evolved_time if evolved_time > 0 else 0.0
        
        # Calculate theoretical advantage
        total_elements = L * L
        masked_elements = sum(bs * bs for bs in config["block_sizes"])
        sparsity = 1.0 - (masked_elements / total_elements)
        
        return {
            "speedup": speedup,
            "evolved_time": evolved_time,
            "spda_time": spda_time,
            "config_name": config["name"],
            "sparsity": sparsity,
            "theoretical_advantage": f"{sparsity*100:.1f}% of attention matrix is masked"
        }
        
    except Exception as e:
        return {"speedup": 0.0, "error": str(e), "config_name": config["name"]}


def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """Main evaluation function for Metal kernel evolution."""
    print(f"🚀 Evaluating Custom Metal Kernel: {program_path}")
    
    if not MLX_AVAILABLE:
        return {
            "stage1_passed": False,
            "overall_score": 0.0,
            "combined_score": 0.0,
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
                "combined_score": 0.0,
                "error": "Missing evolved_scaled_dot_product_attention function"
            }
        
        evolved_fn = evolved_program.evolved_scaled_dot_product_attention
        
        # ===== STAGE 1: CORRECTNESS TESTING =====
        print("\n📋 STAGE 1: Correctness Testing")
        
        test_configs = create_test_configurations()
        correctness_configs = [c for c in test_configs if c["test_type"] == "correctness"]
        
        correctness_results = []
        passed_count = 0
        
        for config in correctness_configs:
            print(f"  Testing {config['name']}: {len(config['block_sizes'])} blocks")
            
            result = evaluate_correctness(evolved_fn, config)
            correctness_results.append(result)
            
            if result["passed"]:
                passed_count += 1
                print(f"    ✅ PASSED (MSE: {result.get('mse', 0):.2e})")
            else:
                error_msg = result.get("error", "Accuracy issue")
                print(f"    ❌ FAILED: {error_msg}")
        
        # Calculate pass rate
        pass_rate = passed_count / len(correctness_configs) if correctness_configs else 0.0
        stage1_passed = pass_rate >= 0.75  # 75% pass rate required
        
        print(f"\n📊 STAGE 1 Results:")
        print(f"  Passed: {passed_count}/{len(correctness_configs)} ({pass_rate:.1%})")
        print(f"  Status: {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        
        if not stage1_passed:
            return {
                "stage1_passed": False,
                "pass_rate": pass_rate,
                "overall_score": 0.0,
                "combined_score": 0.0,
                "failed_at": "correctness"
            }
        
        # ===== STAGE 2: PERFORMANCE TESTING =====
        print(f"\n🏎️  STAGE 2: Performance vs MLX Fast SPDA")
        
        performance_configs = [c for c in test_configs if c["test_type"] == "performance"]
        performance_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for config in performance_configs:
            print(f"  Benchmarking {config['name']}")
            print(f"    Expected: {config.get('advantage_reason', 'Should outperform SPDA')}")
            
            result = benchmark_performance(evolved_fn, config)
            performance_results.append(result)
            
            if "error" in result:
                print(f"    ❌ ERROR: {result['error']}")
                continue
            
            speedup = result.get("speedup", 0.0)
            sparsity = result.get("sparsity", 0.0)
            
            # Weight by sparsity - more sparse patterns are more important to optimize
            weight = 1.0 + sparsity  # Base weight + sparsity bonus
            
            # Score based on speedup achievement
            if speedup >= 2.0:      # 2x+ speedup
                score = 1.0
            elif speedup >= 1.5:    # 1.5x speedup
                score = 0.8
            elif speedup >= 1.2:    # 1.2x speedup  
                score = 0.6
            elif speedup >= 1.0:    # Any speedup
                score = 0.4
            else:                   # Slowdown
                score = 0.0
            
            weighted_score = score * weight
            total_weighted_score += weighted_score
            total_weight += weight
            
            print(f"    📊 Speedup: {speedup:.2f}x vs SPDA (sparsity: {sparsity*100:.1f}%)")
            print(f"    📈 Score: {score:.2f} (weighted: {weighted_score:.2f})")
        
        # Calculate overall performance score
        stage2_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        overall_score = stage2_score
        
        # Analyze results
        speedups = [r.get("speedup", 0.0) for r in performance_results if "speedup" in r]
        avg_speedup = np.mean(speedups) if speedups else 0.0
        max_speedup = max(speedups) if speedups else 0.0
        
        print(f"\n🎯 STAGE 2 Results:")
        print(f"  Performance Score: {stage2_score:.3f}")
        print(f"  Average Speedup vs SPDA: {avg_speedup:.2f}x")
        print(f"  Best Speedup vs SPDA: {max_speedup:.2f}x")
        
        print(f"\n🏆 Overall Results:")
        print(f"  Stage 1 (Correctness): {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        print(f"  Stage 2 (Performance): {stage2_score:.3f}")
        print(f"  Overall Score: {overall_score:.3f}")
        
        if overall_score >= 0.8:
            print(f"  🥇 EXCELLENT: Metal kernel significantly outperforms SPDA!")
        elif overall_score >= 0.6:
            print(f"  🥈 GOOD: Meaningful performance improvements achieved")
        elif overall_score >= 0.4:
            print(f"  🥉 MODERATE: Some optimization, room for improvement")
        else:
            print(f"  ❌ POOR: Kernel needs significant optimization")
        
        return {
            "stage1_passed": stage1_passed,
            "pass_rate": float(pass_rate),
            "stage2_score": float(stage2_score),
            "overall_score": float(overall_score),
            "combined_score": float(overall_score),  # Primary metric for OpenEvolve
            "avg_speedup": float(avg_speedup),
            "max_speedup": float(max_speedup),
            "num_tests": len(test_configs),
            "num_performance_tests": len(performance_configs)
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "stage1_passed": False,
            "overall_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Testing Metal Kernel Evaluator...")
    
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("\nEvaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
