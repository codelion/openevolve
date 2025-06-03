"""
Robust Two-Stage Evaluator for MLX Block Diagonal Attention Optimization

STAGE 1: Correctness & Compatibility Gate
- Ensures evolved programs produce correct outputs
- Tests against comprehensive spda_benchmark configurations  
- Uses proven tolerances and evaluation logic
- Must pass to proceed to Stage 2

STAGE 2: Performance Optimization
- Benchmarks speed vs mx.fast.scaled_dot_product_attention
- Measures actual speedups and efficiency gains
- Creates evolutionary pressure for performance improvements
- Only runs if Stage 1 passes

This ensures we evolve CORRECT AND FAST algorithms, not just fast ones.
"""

import importlib.util
import math
import time
import traceback
from typing import Dict, List, Tuple, Union
import gc

import mlx.core as mx
import numpy as np

# Import benchmark utilities
from spda_benchmark import prepare_inputs, mlx_ref_attn, mlx_fused_attn, do_attention, bench


def safe_format_percentage(value, fallback="N/A%"):
    """
    Safely format a value as a percentage.
    
    Args:
        value: Value to format as percentage (should be between 0 and 1)
        fallback: Fallback string if formatting fails
        
    Returns:
        Formatted percentage string
    """
    try:
        if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
            return f"{value:.1%}"
        else:
            return fallback
    except (ValueError, TypeError):
        return fallback


def safe_format_number(value: Union[float, int, str], format_spec: str = ".3f", fallback: str = "N/A") -> str:
    """
    Safely format a number with fallback for non-numeric values.
    This prevents "Unknown format code 'f' for object of type 'str'" errors.
    """
    try:
        if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
            return f"{value:{format_spec}}"
        elif value == float("inf"):
            return "∞"
        elif value == float("-inf"):
            return "-∞" 
        elif isinstance(value, float) and math.isnan(value):
            return "NaN"
        else:
            return str(value) if value is not None else fallback
    except (ValueError, TypeError):
        return fallback


def create_stage1_test_configurations() -> List[Dict]:
    """
    Stage 1: Comprehensive correctness tests based on spda_benchmark.
    
    These are the proven test configurations that ensure compatibility
    and correctness across all scenarios.
    """
    return [
        # SHORT SEQUENCES: Should use mx.fast.scaled_dot_product_attention
        # These test the hybrid dispatcher's short sequence path
        {
            "B": 1, "qsl": 64, "ksl": 64, "head_dim": 64,
            "n_q_heads": 8, "n_kv_heads": 8, "dtype": "float16", 
            "mask": None, "category": "short",
        },
        {
            "B": 1, "qsl": 128, "ksl": 128, "head_dim": 64,
            "n_q_heads": 8, "n_kv_heads": 8, "dtype": "float16",
            "mask": "causal", "category": "short", 
        },
        {
            "B": 1, "qsl": 256, "ksl": 256, "head_dim": 64,
            "n_q_heads": 16, "n_kv_heads": 8, "dtype": "float16",
            "mask": None, "category": "short",
        },
        
        # TRANSITION SEQUENCES: Test behavior around 512 threshold
        {
            "B": 1, "qsl": 480, "ksl": 480, "head_dim": 64,
            "n_q_heads": 16, "n_kv_heads": 8, "dtype": "float16",
            "mask": "causal", "category": "transition",
        },
        {
            "B": 1, "qsl": 512, "ksl": 512, "head_dim": 64,
            "n_q_heads": 16, "n_kv_heads": 8, "dtype": "float16", 
            "mask": None, "category": "transition",
        },
        
        # LONG SEQUENCES: Main target for block diagonal attention
        {
            "B": 1, "qsl": 768, "ksl": 768, "head_dim": 64,
            "n_q_heads": 16, "n_kv_heads": 8, "dtype": "float16",
            "mask": "causal", "category": "long",
        },
        {
            "B": 1, "qsl": 1024, "ksl": 1024, "head_dim": 64,
            "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16",
            "mask": None, "category": "long",
        },
        {
            "B": 1, "qsl": 1536, "ksl": 1536, "head_dim": 64,
            "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16",
            "mask": "causal", "category": "long",
        },
        
        # VERY LONG SEQUENCES: Scalability tests
        {
            "B": 1, "qsl": 2048, "ksl": 2048, "head_dim": 64,
            "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16",
            "mask": None, "category": "very_long",
        },
        
        # DIFFERENT HEAD DIMENSIONS: Test generalization
        {
            "B": 1, "qsl": 1024, "ksl": 1024, "head_dim": 80,
            "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16",
            "mask": "causal", "category": "long",
        },
    ]


def create_stage2_performance_configurations() -> List[Dict]:
    """
    Stage 2: Performance benchmark configurations.
    
    These focus on scenarios where we expect to see speedup improvements.
    """
    return [
        # BASELINE: Short sequence where mx.fast should be optimal
        {
            "name": "short_baseline",
            "B": 1, "qsl": 256, "ksl": 256, "head_dim": 64,
            "n_q_heads": 16, "n_kv_heads": 8, "dtype": "float16",
            "mask": None, "weight": 0.1, "expect_improvement": False,
        },
        
        # PERFORMANCE TARGETS: Long sequences where block diagonal should excel
        {
            "name": "long_perf_1024",
            "B": 1, "qsl": 1024, "ksl": 1024, "head_dim": 64,
            "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16", 
            "mask": "causal", "weight": 0.3, "expect_improvement": True,
        },
        {
            "name": "long_perf_1536", 
            "B": 1, "qsl": 1536, "ksl": 1536, "head_dim": 64,
            "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16",
            "mask": None, "weight": 0.3, "expect_improvement": True,
        },
        {
            "name": "very_long_2048",
            "B": 1, "qsl": 2048, "ksl": 2048, "head_dim": 64,
            "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16",
            "mask": "causal", "weight": 0.3, "expect_improvement": True,
        },
    ]


def compare_attention_outputs(output1: mx.array, output2: mx.array, tolerance: float = 1e-3) -> Dict[str, float]:
    """
    Compare two attention outputs with appropriate tolerance.
    Enhanced version with robust error handling.
    """
    try:
        # Ensure arrays are evaluated
        output1 = mx.array(output1)
        output2 = mx.array(output2)
        mx.eval(output1, output2)

        # Calculate various similarity metrics
        diff = output1 - output2
        mse = float(mx.mean(diff**2))
        mae = float(mx.mean(mx.abs(diff)))
        max_diff = float(mx.max(mx.abs(diff)))

        # Relative error (normalized by output magnitude)
        output1_norm = float(mx.sqrt(mx.mean(output1**2)))
        relative_error = float(mx.sqrt(mx.mean(diff**2))) / max(output1_norm, 1e-8)

        # Check MLX's allclose function
        allclose_result = bool(mx.allclose(output1, output2, atol=tolerance, rtol=tolerance))
        
        # Additional robust check: if MSE is extremely small, consider it a match
        mse_perfect = mse < 1e-8
        
        # Final decision: either allclose passes OR MSE is extremely small
        final_allclose = allclose_result or mse_perfect

        return {
            "mse": mse,
            "mae": mae,
            "max_diff": max_diff,
            "relative_error": relative_error,
            "allclose": final_allclose,
            "allclose_strict": allclose_result,
            "mse_perfect": mse_perfect,
            "tolerance_used": tolerance,
        }
    except Exception as e:
        # Fallback values if comparison fails
        return {
            "mse": float("inf"),
            "mae": float("inf"),
            "max_diff": float("inf"),
            "relative_error": float("inf"),
            "allclose": False,
            "allclose_strict": False,
            "mse_perfect": False,
            "tolerance_used": tolerance,
            "comparison_error": str(e),
        }


def evaluate_stage1_correctness(evolved_attention_fn, config: Dict) -> Dict[str, Union[bool, float, str]]:
    """
    Stage 1: Test correctness with category-appropriate tolerances.
    
    Based on proven evaluation logic from original evaluator.
    """
    
    category = config.get("category", "unknown")
    
    # Set tolerance based on category (proven values)
    if category == "short":
        tolerance = 1e-4  # Should be nearly perfect
        expected_quality = "perfect"
    elif category == "transition":
        tolerance = 1e-3  # High quality
        expected_quality = "high"
    elif category == "long":
        tolerance = 1e-3  # Good quality (allow some block approximation)
        expected_quality = "good"
    elif category == "very_long":
        tolerance = 1e-2  # Acceptable quality
        expected_quality = "acceptable"
    else:
        tolerance = 1e-3
        expected_quality = "unknown"
    
    # Unpack test configuration
    B = config["B"]
    qsl = config["qsl"]
    ksl = config["ksl"]
    head_dim = config["head_dim"]
    n_q_heads = config["n_q_heads"]
    n_kv_heads = config["n_kv_heads"]
    dtype = config["dtype"]
    mask_type = config.get("mask", None)

    try:
        # Prepare inputs
        q, k, v, scale, mask = prepare_inputs(
            B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, False, dtype
        )

        # Run evolved implementation
        evolved_output = evolved_attention_fn(q, k, v, scale=scale, mask=mask)
        
        # For very long sequences, skip reference comparison (too expensive)
        if qsl >= 3072:
            # Just check for validity
            has_nan = bool(mx.any(mx.isnan(evolved_output)))
            has_inf = bool(mx.any(mx.isinf(evolved_output)))
            shape_correct = evolved_output.shape == q.shape
            
            return {
                "passed": shape_correct and not (has_nan or has_inf),
                "mse": 0.0,
                "shape_correct": shape_correct,
                "no_nan_inf": not (has_nan or has_inf),
                "tolerance_used": tolerance,
                "category": category,
                "reference_computed": False,
            }
        
        # For shorter sequences, compute reference for comparison
        try:
            reference_output = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
        except Exception as ref_error:
            # Reference failed, check structural validity only
            has_nan = bool(mx.any(mx.isnan(evolved_output)))
            has_inf = bool(mx.any(mx.isinf(evolved_output)))
            shape_correct = evolved_output.shape == q.shape
            
            return {
                "passed": shape_correct and not (has_nan or has_inf),
                "mse": 0.0,
                "shape_correct": shape_correct,
                "no_nan_inf": not (has_nan or has_inf),
                "tolerance_used": tolerance,
                "category": category,
                "reference_computed": False,
                "reference_error": str(ref_error),
            }

        # Compare outputs with category-appropriate tolerance
        comparison = compare_attention_outputs(evolved_output, reference_output, tolerance=tolerance)

        # Check for structural correctness
        shape_correct = evolved_output.shape == reference_output.shape
        no_nan_inf = not (
            bool(mx.any(mx.isnan(evolved_output))) or bool(mx.any(mx.isinf(evolved_output)))
        )

        # Pass criteria: structural correctness AND close match
        passed = shape_correct and no_nan_inf and comparison["allclose"]

        return {
            "passed": passed,
            **comparison,
            "shape_correct": shape_correct,
            "no_nan_inf": no_nan_inf,
            "category": category,
            "reference_computed": True,
        }

    except Exception as e:
        return {
            "passed": False,
            "mse": float("inf"),
            "tolerance_used": tolerance,
            "category": category,
            "reference_computed": False,
            "error": str(e),
        }


def benchmark_performance(evolved_fn, config: Dict, num_trials: int = 3) -> Dict[str, Union[float, str]]:
    """
    Stage 2: Benchmark performance vs mx.fast.scaled_dot_product_attention.
    """
    
    B = config["B"]
    qsl = config["qsl"]
    ksl = config["ksl"]
    head_dim = config["head_dim"]
    n_q_heads = config["n_q_heads"]
    n_kv_heads = config["n_kv_heads"]
    dtype = config["dtype"]
    mask_type = config.get("mask", None)
    
    try:
        # Prepare inputs
        q, k, v, scale, mask = prepare_inputs(
            B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, False, dtype
        )
        
        # Benchmark evolved implementation
        evolved_times = []
        for trial in range(num_trials):
            try:
                gc.collect()
                mx.metal.clear_cache()
                
                start_time = time.perf_counter()
                output = evolved_fn(q, k, v, scale=scale, mask=mask)
                mx.eval(output)
                end_time = time.perf_counter()
                
                evolved_times.append(end_time - start_time)
            except Exception as e:
                return {"speedup": 0.0, "performance_score": 0.0, "error": f"Evolved failed: {str(e)}"}
        
        evolved_time = np.median(evolved_times)
        
        # Benchmark baseline (mx.fast.scaled_dot_product_attention)
        baseline_times = []
        baseline_success = True
        
        for trial in range(num_trials):
            try:
                gc.collect()
                mx.metal.clear_cache()
                
                start_time = time.perf_counter()
                output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
                mx.eval(output)
                end_time = time.perf_counter()
                
                baseline_times.append(end_time - start_time)
            except Exception:
                # Use reference as baseline if mx.fast fails
                try:
                    start_time = time.perf_counter()
                    output = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
                    mx.eval(output)
                    end_time = time.perf_counter()
                    baseline_times.append(end_time - start_time)
                except Exception:
                    baseline_success = False
                    break
        
        if not baseline_success:
            # If baseline fails but evolved works, that's a win
            return {"speedup": float("inf"), "performance_score": 1.0, "baseline_failed": True}
        
        baseline_time = np.median(baseline_times)
        
        # Calculate speedup (>1.0 means evolved is faster)
        speedup = baseline_time / evolved_time if evolved_time > 0 else 0.0
        
        # Performance score based on speedup
        if speedup >= 1.5:  # 50%+ speedup
            performance_score = 1.0
        elif speedup >= 1.2:  # 20%+ speedup  
            performance_score = 0.5 + (speedup - 1.2) * (0.5 / 0.3)  # Linear 1.2->0.5, 1.5->1.0
        elif speedup >= 1.0:  # Any speedup
            performance_score = (speedup - 1.0) * (0.5 / 0.2)  # Linear 1.0->0.0, 1.2->0.5
        else:  # Slower than baseline
            performance_score = 0.0
        
        return {
            "speedup": speedup,
            "performance_score": performance_score,
            "evolved_time": evolved_time,
            "baseline_time": baseline_time,
        }
        
    except Exception as e:
        return {"speedup": 0.0, "performance_score": 0.0, "error": str(e)}


def evaluate_two_stage(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Two-stage evaluation: Correctness gate + Performance optimization.
    """
    
    print(f"🎯 Two-Stage Evaluation: {program_path}")
    
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)

        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            return {
                "stage1_passed": False,
                "stage2_score": 0.0,
                "overall_score": 0.0,
                "error": "Missing evolved_scaled_dot_product_attention function",
            }

        evolved_attention_fn = evolved_program.evolved_scaled_dot_product_attention
        
        # =====================================
        # STAGE 1: CORRECTNESS & COMPATIBILITY
        # =====================================
        print(f"\n📋 STAGE 1: Correctness & Compatibility Testing")
        
        stage1_configs = create_stage1_test_configurations()
        stage1_results = []
        stage1_passed_count = 0
        
        for i, config in enumerate(stage1_configs):
            category = config.get("category", "unknown")
            print(f"  Test {i+1}/{len(stage1_configs)}: seq={config['qsl']}, category={category}, "
                  f"heads={config['n_q_heads']}/{config['n_kv_heads']}, mask={config.get('mask', None)}")
            
            result = evaluate_stage1_correctness(evolved_attention_fn, config)
            stage1_results.append(result)
            
            if result["passed"]:
                stage1_passed_count += 1
                mse_val = result.get('mse', 'N/A')
                mse_str = safe_format_number(mse_val, ".2e")
                print(f"    ✅ PASSED: MSE={mse_str}")
            else:
                error_msg = result.get('error', 'Accuracy/structure issue')
                print(f"    ❌ FAILED: {error_msg}")
        
        # Safe calculation of stage1_pass_rate to prevent division errors
        try:
            stage1_pass_rate = stage1_passed_count / len(stage1_configs) if len(stage1_configs) > 0 else 0.0
        except (TypeError, ZeroDivisionError):
            stage1_pass_rate = 0.0
            
        stage1_passed = stage1_pass_rate >= 0.9  # 90% pass rate required
        
        # Safe formatting for stage1_pass_rate
        stage1_pass_rate_str = safe_format_percentage(stage1_pass_rate)
        
        print(f"\n📊 STAGE 1 Results:")
        print(f"  Passed: {stage1_passed_count}/{len(stage1_configs)} ({stage1_pass_rate_str})")
        print(f"  Gate Status: {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        
        if not stage1_passed:
            print(f"  🚫 Stage 1 failed - Stage 2 skipped")
            return {
                "stage1_passed": False,
                "stage1_pass_rate": stage1_pass_rate,
                "stage2_score": 0.0,
                "overall_score": 0.0,
                "failed_at": "stage1_correctness",
            }
        
        # =====================================
        # STAGE 2: PERFORMANCE OPTIMIZATION
        # =====================================
        print(f"\n🚀 STAGE 2: Performance Benchmarking")
        
        stage2_configs = create_stage2_performance_configurations()
        stage2_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for config in stage2_configs:
            print(f"  Benchmarking {config['name']}: seq={config['qsl']}")
            
            benchmark_result = benchmark_performance(evolved_attention_fn, config)
            
            speedup = benchmark_result["speedup"]
            perf_score = benchmark_result["performance_score"]
            weighted_score = perf_score * config["weight"]
            
            total_weighted_score += weighted_score
            total_weight += config["weight"]
            
            stage2_results.append({
                "config": config,
                "benchmark": benchmark_result,
                "weighted_score": weighted_score,
            })
            
            # Safe formatting for speedup and performance score
            speedup_str = safe_format_number(speedup, ".2f")
            perf_str = safe_format_number(perf_score, ".3f")
                
            print(f"    📊 Speedup: {speedup_str}x, Score: {perf_str}")
        
        # Safe calculation of stage2_score to prevent division errors
        try:
            stage2_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        except (TypeError, ZeroDivisionError):
            stage2_score = 0.0
            
        # Calculate overall score (Stage 1 gate + Stage 2 performance)
        overall_score = stage2_score  # Since Stage 1 is just a gate
        
        # Detailed performance analysis with safe operations
        speedups = []
        for r in stage2_results:
            speedup_val = r["benchmark"]["speedup"]
            if (isinstance(speedup_val, (int, float)) and 
                speedup_val != float("inf") and 
                not math.isnan(speedup_val)):
                speedups.append(speedup_val)
                
        try:
            avg_speedup = np.mean(speedups) if speedups else 0.0
            max_speedup = max(speedups) if speedups else 0.0
        except (TypeError, ValueError):
            avg_speedup = 0.0
            max_speedup = 0.0
        
        print(f"\n📈 STAGE 2 Results:")
        
        # Safe formatting for final results
        stage2_str = safe_format_number(stage2_score, ".3f")
        avg_speedup_str = safe_format_number(avg_speedup, ".2f")
        max_speedup_str = safe_format_number(max_speedup, ".2f")
        overall_str = safe_format_number(overall_score, ".3f")
        
        print(f"  Performance Score: {stage2_str}")
        print(f"  Average Speedup: {avg_speedup_str}x")
        print(f"  Max Speedup: {max_speedup_str}x")
        
        print(f"\n🎯 Overall Results:")
        print(f"  Stage 1: {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        print(f"  Stage 2: {stage2_str}")
        print(f"  Overall Score: {overall_str}")
        
        if overall_score >= 0.8:
            print(f"  🏆 EXCELLENT: Strong performance improvements!")
        elif overall_score >= 0.5:
            print(f"  🚀 GOOD: Meaningful speedups achieved")
        elif overall_score >= 0.2:
            print(f"  ⚡ PARTIAL: Some improvements, room for more")
        else:
            print(f"  ❌ POOR: Need significant optimization")
        
        # Ensure all return values are safe numeric types
        try:
            safe_stage1_pass_rate = float(stage1_pass_rate) if isinstance(stage1_pass_rate, (int, float)) else 0.0
            safe_stage2_score = float(stage2_score) if isinstance(stage2_score, (int, float)) else 0.0
            safe_overall_score = float(overall_score) if isinstance(overall_score, (int, float)) else 0.0
            safe_avg_speedup = float(avg_speedup) if isinstance(avg_speedup, (int, float)) else 0.0
            safe_max_speedup = float(max_speedup) if isinstance(max_speedup, (int, float)) else 0.0
        except (TypeError, ValueError):
            safe_stage1_pass_rate = 0.0
            safe_stage2_score = 0.0
            safe_overall_score = 0.0
            safe_avg_speedup = 0.0
            safe_max_speedup = 0.0
            
        return {
            # Gate results
            "stage1_passed": stage1_passed,
            "stage1_pass_rate": safe_stage1_pass_rate,
            
            # Performance results
            "stage2_score": safe_stage2_score,
            "overall_score": safe_overall_score,
            
            # Detailed metrics
            "avg_speedup": safe_avg_speedup,
            "max_speedup": safe_max_speedup,
            "num_stage1_tests": len(stage1_configs),
            "num_stage2_tests": len(stage2_configs),
        }
        
    except Exception as e:
        print(f"❌ Two-stage evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "stage1_passed": False,
            "stage2_score": 0.0,
            "overall_score": 0.0,
            "error": str(e),
        }


def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Main evaluation function - Two-stage: Correctness gate + Performance.
    Includes comprehensive error handling to prevent formatting errors.
    """
    try:
        return evaluate_two_stage(program_path)
    except Exception as e:
        # Catch ANY error (including formatting errors) and return safe fallback
        error_msg = str(e)
        print(f"❌ Evaluation failed with error: {error_msg}")
        
        # Return safe fallback metrics
        return {
            "stage1_passed": False,
            "stage2_score": 0.0,
            "overall_score": 0.0,
            "error": error_msg,
            "failed_at": "evaluation_error",
        }


if __name__ == "__main__":
    # Test the two-stage evaluator
    print("Testing Robust Two-Stage Evaluator...")
    import os

    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate_two_stage(initial_program_path)
        print("\nTwo-Stage Evaluation Results:")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                formatted_v = safe_format_number(v, ".4f")
                print(f"  {k}: {formatted_v}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
