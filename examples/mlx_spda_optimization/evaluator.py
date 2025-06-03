"""
Performance-Focused Evaluator for MLX Block Diagonal Attention Optimization

This evaluator restructures evaluation to focus on performance optimization:
1. Stage 1 (Correctness Gate): Must pass basic correctness tests
2. Stage 2 (Performance Competition): Score based on speed improvements while maintaining correctness

The goal is to find the fastest correct implementation, especially for long sequences.
"""

import importlib.util
import math
import time
import traceback
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

# Import benchmark utilities
from spda_benchmark import prepare_inputs, mlx_ref_attn, mlx_fused_attn, do_attention, bench


# Global performance baselines (computed once)
PERFORMANCE_BASELINES = {}


def create_test_configurations() -> List[Dict]:
    """
    Create test configurations focused on performance optimization.
    """
    return [
        # SHORT SEQUENCES: Must maintain optimal performance
        {
            "B": 1, "qsl": 128, "ksl": 128, "head_dim": 64, "n_q_heads": 8, "n_kv_heads": 8,
            "dtype": "float16", "mask": None, "category": "short", "weight": 0.1
        },
        {
            "B": 1, "qsl": 256, "ksl": 256, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 8,
            "dtype": "float16", "mask": "causal", "category": "short", "weight": 0.1
        },
        
        # LONG SEQUENCES: Main optimization target
        {
            "B": 1, "qsl": 512, "ksl": 512, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 8,
            "dtype": "float16", "mask": None, "category": "long", "weight": 0.2
        },
        {
            "B": 1, "qsl": 768, "ksl": 768, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 8,
            "dtype": "float16", "mask": "causal", "category": "long", "weight": 0.2
        },
        {
            "B": 1, "qsl": 1024, "ksl": 1024, "head_dim": 64, "n_q_heads": 32, "n_kv_heads": 8,
            "dtype": "float16", "mask": None, "category": "long", "weight": 0.3
        },
        {
            "B": 1, "qsl": 1536, "ksl": 1536, "head_dim": 64, "n_q_heads": 32, "n_kv_heads": 8,
            "dtype": "float16", "mask": "causal", "category": "long", "weight": 0.1
        },
    ]


def get_performance_baseline(config: Dict) -> float:
    """
    Get or compute performance baseline for a configuration.
    This represents the performance we're trying to beat.
    """
    key = f"{config['qsl']}_{config['head_dim']}_{config['n_q_heads']}_{config['n_kv_heads']}_{config['mask']}"
    
    if key in PERFORMANCE_BASELINES:
        return PERFORMANCE_BASELINES[key]
    
    # Compute baseline performance
    try:
        B = config["B"]
        qsl = config["qsl"]
        ksl = config["ksl"] 
        head_dim = config["head_dim"]
        n_q_heads = config["n_q_heads"]
        n_kv_heads = config["n_kv_heads"]
        dtype = config["dtype"]
        mask_type = config.get("mask", None)
        
        q, k, v, scale, mask = prepare_inputs(
            B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, False, dtype
        )
        
        # For short sequences, try mx.fast as baseline
        if qsl < 512:
            try:
                start_time = time.perf_counter()
                for _ in range(3):  # Multiple runs for stability
                    output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
                    mx.eval(output)
                end_time = time.perf_counter()
                baseline_time = (end_time - start_time) / 3
                PERFORMANCE_BASELINES[key] = baseline_time
                return baseline_time
            except Exception:
                pass
        
        # Fallback: use reference implementation as baseline
        start_time = time.perf_counter()
        for _ in range(3):
            output = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
            mx.eval(output)
        end_time = time.perf_counter()
        baseline_time = (end_time - start_time) / 3
        PERFORMANCE_BASELINES[key] = baseline_time
        return baseline_time
        
    except Exception as e:
        # Default baseline if measurement fails
        return 1.0


def measure_performance(evolved_attention_fn, config: Dict, num_runs: int = 3) -> Dict[str, float]:
    """
    Measure performance of evolved attention function.
    """
    try:
        B = config["B"]
        qsl = config["qsl"] 
        ksl = config["ksl"]
        head_dim = config["head_dim"]
        n_q_heads = config["n_q_heads"]
        n_kv_heads = config["n_kv_heads"]
        dtype = config["dtype"]
        mask_type = config.get("mask", None)
        
        q, k, v, scale, mask = prepare_inputs(
            B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, False, dtype
        )
        
        # Warmup run
        try:
            output = evolved_attention_fn(q, k, v, scale=scale, mask=mask)
            mx.eval(output)
        except Exception as e:
            return {"execution_time": float("inf"), "valid": False, "error": str(e)}
        
        # Measured runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            output = evolved_attention_fn(q, k, v, scale=scale, mask=mask)
            mx.eval(output)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        # Check validity
        has_nan = bool(mx.any(mx.isnan(output)))
        has_inf = bool(mx.any(mx.isinf(output)))
        valid = not (has_nan or has_inf) and output.shape == q.shape
        
        return {
            "execution_time": avg_time,
            "min_time": min_time,
            "valid": valid,
            "shape_correct": output.shape == q.shape,
            "no_nan_inf": not (has_nan or has_inf)
        }
        
    except Exception as e:
        return {"execution_time": float("inf"), "valid": False, "error": str(e)}


def test_correctness(evolved_attention_fn, config: Dict, tolerance: float = 1e-3) -> Dict[str, float]:
    """
    Test correctness against reference implementation.
    """
    try:
        B = config["B"]
        qsl = config["qsl"]
        ksl = config["ksl"] 
        head_dim = config["head_dim"]
        n_q_heads = config["n_q_heads"]
        n_kv_heads = config["n_kv_heads"]
        dtype = config["dtype"]
        mask_type = config.get("mask", None)
        
        q, k, v, scale, mask = prepare_inputs(
            B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, False, dtype
        )
        
        # Get evolved output
        evolved_output = evolved_attention_fn(q, k, v, scale=scale, mask=mask)
        
        # Get reference output  
        reference_output = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
        
        # Compare
        diff = evolved_output - reference_output
        mse = float(mx.mean(diff**2))
        max_diff = float(mx.max(mx.abs(diff)))
        
        # Correctness checks
        shape_correct = evolved_output.shape == reference_output.shape
        no_nan_inf = not (bool(mx.any(mx.isnan(evolved_output))) or bool(mx.any(mx.isinf(evolved_output))))
        allclose = bool(mx.allclose(evolved_output, reference_output, atol=tolerance, rtol=tolerance))
        mse_good = mse < tolerance
        
        # Overall correctness: must pass all checks
        correct = shape_correct and no_nan_inf and (allclose or mse_good)
        
        return {
            "mse": mse,
            "max_diff": max_diff,
            "shape_correct": shape_correct,
            "no_nan_inf": no_nan_inf,
            "allclose": allclose,
            "mse_good": mse_good,
            "correct": correct,
            "tolerance_used": tolerance
        }
        
    except Exception as e:
        return {
            "mse": float("inf"),
            "max_diff": float("inf"), 
            "shape_correct": False,
            "no_nan_inf": False,
            "allclose": False,
            "mse_good": False,
            "correct": False,
            "error": str(e)
        }


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    Stage 1: Correctness Gate
    Programs must pass basic correctness tests to proceed to performance evaluation.
    """
    
    try:
        print(f"[Stage 1] 🔍 Correctness Gate Evaluation")
        
        # Load program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(evolved_program)
        except Exception as e:
            print(f"[Stage 1] ❌ Import failed: {e}")
            return {"stage1_pass": 0.0, "correctness_gate": 0.0, "error": str(e)}
        
        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            print(f"[Stage 1] ❌ Missing function")
            return {"stage1_pass": 0.0, "correctness_gate": 0.0, "error": "Missing function"}
        
        evolved_attention_fn = evolved_program.evolved_scaled_dot_product_attention
        
        # Test on key configurations for correctness
        test_configs = [
            {"B": 1, "qsl": 128, "ksl": 128, "head_dim": 64, "n_q_heads": 8, "n_kv_heads": 8,
             "dtype": "float16", "mask": None, "category": "short"},
            {"B": 1, "qsl": 512, "ksl": 512, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 8,
             "dtype": "float16", "mask": "causal", "category": "long"},
            {"B": 1, "qsl": 1024, "ksl": 1024, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 8,
             "dtype": "float16", "mask": None, "category": "long"},
        ]
        
        correctness_results = []
        for config in test_configs:
            tolerance = 1e-4 if config["category"] == "short" else 1e-3
            correctness = test_correctness(evolved_attention_fn, config, tolerance)
            correctness_results.append(correctness)
            
            print(f"[Stage 1] {config['category']} seq {config['qsl']}: "
                  f"MSE={correctness.get('mse', 'inf'):.2e}, "
                  f"Correct={correctness.get('correct', False)}")
        
        # Must pass ALL correctness tests
        all_correct = all(result.get("correct", False) for result in correctness_results)
        
        if all_correct:
            print(f"[Stage 1] ✅ PASS: All correctness tests passed")
            return {
                "stage1_pass": 1.0,
                "correctness_gate": 1.0,
                "all_correct": 1.0
            }
        else:
            print(f"[Stage 1] ❌ FAIL: Correctness tests failed")
            return {
                "stage1_pass": 0.0,
                "correctness_gate": 0.0,
                "all_correct": 0.0
            }
            
    except Exception as e:
        print(f"[Stage 1] ❌ Error: {e}")
        return {"stage1_pass": 0.0, "correctness_gate": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Stage 2: Performance Competition
    Among correct programs, score based on performance improvements.
    """
    
    try:
        print(f"[Stage 2] 🏁 Performance Competition")
        
        # Load program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        evolved_attention_fn = evolved_program.evolved_scaled_dot_product_attention
        
        test_configs = create_test_configurations()
        
        performance_scores = []
        correctness_scores = []
        
        total_weighted_speedup = 0.0
        total_weight = 0.0
        
        for config in test_configs:
            print(f"[Stage 2] Testing {config['category']} seq {config['qsl']}...")
            
            # Test correctness first
            tolerance = 1e-4 if config["category"] == "short" else 1e-3
            correctness = test_correctness(evolved_attention_fn, config, tolerance)
            
            if not correctness.get("correct", False):
                print(f"[Stage 2]   ❌ Correctness failed - skipping performance test")
                continue
            
            # Test performance
            performance = measure_performance(evolved_attention_fn, config)
            
            if not performance.get("valid", False):
                print(f"[Stage 2]   ❌ Performance test failed")
                continue
            
            # Calculate speedup vs baseline
            baseline_time = get_performance_baseline(config)
            evolved_time = performance["execution_time"]
            
            if evolved_time > 0 and baseline_time > 0:
                speedup = baseline_time / evolved_time
                config_weight = config.get("weight", 1.0)
                
                total_weighted_speedup += speedup * config_weight
                total_weight += config_weight
                
                print(f"[Stage 2]   ✅ Speedup: {speedup:.2f}x "
                      f"({baseline_time:.3f}s → {evolved_time:.3f}s)")
            else:
                print(f"[Stage 2]   ⚠️ Invalid timing")
        
        # Calculate overall performance score
        if total_weight > 0:
            avg_speedup = total_weighted_speedup / total_weight
            
            # Convert speedup to score (1.0 = no improvement, >1.0 = improvement)
            if avg_speedup >= 1.5:
                performance_score = 1.0  # Excellent
            elif avg_speedup >= 1.2:
                performance_score = 0.8  # Good
            elif avg_speedup >= 1.1:
                performance_score = 0.6  # Moderate
            elif avg_speedup >= 1.0:
                performance_score = 0.4  # Slight
            else:
                performance_score = 0.2  # Regression
        else:
            avg_speedup = 0.0
            performance_score = 0.0
        
        print(f"[Stage 2] 📊 Average speedup: {avg_speedup:.2f}x")
        print(f"[Stage 2] 📊 Performance score: {performance_score:.3f}")
        
        return {
            "performance_score": performance_score,
            "average_speedup": avg_speedup,
            "combined_score": performance_score,  # Primary metric
        }
        
    except Exception as e:
        print(f"[Stage 2] ❌ Error: {e}")
        return {"performance_score": 0.0, "average_speedup": 0.0, "combined_score": 0.0}


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Main evaluation function with two-stage process:
    1. Stage 1: Correctness gate (must pass to proceed)
    2. Stage 2: Performance competition (score based on speed improvements)
    """
    
    # Stage 1: Correctness Gate
    stage1_results = evaluate_stage1(program_path)
    
    if stage1_results.get("stage1_pass", 0.0) == 0.0:
        # Failed Stage 1 - return low score
        return {
            "combined_score": 0.1,  # Low but non-zero to indicate some progress
            "stage1_pass": 0.0,
            "performance_score": 0.0,
            **stage1_results
        }
    
    # Stage 2: Performance Competition
    stage2_results = evaluate_stage2(program_path)
    
    # Combine results
    final_score = stage2_results.get("combined_score", 0.0)
    
    return {
        "combined_score": final_score,
        "stage1_pass": 1.0,
        "performance_score": stage2_results.get("performance_score", 0.0),
        "average_speedup": stage2_results.get("average_speedup", 0.0),
        **stage1_results,
        **stage2_results
    }


if __name__ == "__main__":
    # Test the evaluator
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        print("🧪 Testing Performance-Focused Evaluator")
        results = evaluate(initial_program_path)
        print("📊 Results:")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
    else:
        print("Initial program not found")
