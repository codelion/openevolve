"""
Evaluator for MLX Block Diagonal Attention Optimization

This evaluator tests evolved block diagonal attention implementations by:
1. Using SAME correctness checks as spda_benchmark.py to catch actual failures
2. Testing hybrid dispatcher works correctly (short vs long sequences)
3. Measuring scalability improvements for long sequences
4. Ensuring compatibility with the benchmark testing framework

CRITICAL: This evaluator must catch the same correctness failures that spda_benchmark.py catches,
so evolved programs that fail the benchmark are rejected during evolution.
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


def create_test_configurations() -> List[Dict]:
    """
    Create test configurations focused on correctness and robustness.
    These mirror the benchmark's test cases to ensure compatibility.
    """
    return [
        # SHORT SEQUENCES: Should use mx.fast.scaled_dot_product_attention
        {
            "B": 1,
            "qsl": 64,
            "ksl": 64,
            "head_dim": 64,
            "n_q_heads": 8,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "short",
        },
        {
            "B": 1,
            "qsl": 128,
            "ksl": 128,
            "head_dim": 64,
            "n_q_heads": 8,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "short",
        },
        {
            "B": 1,
            "qsl": 256,
            "ksl": 256,
            "head_dim": 64,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "short",
        },
        
        # TRANSITION SEQUENCES: Critical boundary testing
        {
            "B": 1,
            "qsl": 512,
            "ksl": 512,
            "head_dim": 64,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "transition",
        },
        {
            "B": 1,
            "qsl": 512,
            "ksl": 512,
            "head_dim": 64,
            "n_q_heads": 32,
            "n_kv_heads": 32,
            "dtype": "float16",
            "mask": "causal",
            "category": "transition",
        },
        
        # LONG SEQUENCES: Block diagonal attention targets
        {
            "B": 1,
            "qsl": 768,
            "ksl": 768,
            "head_dim": 64,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "long",
        },
        {
            "B": 1,
            "qsl": 1024,
            "ksl": 1024,
            "head_dim": 64,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "long",
        },
        {
            "B": 1,
            "qsl": 1536,
            "ksl": 1536,
            "head_dim": 64,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "long",
        },
        
        # VERY LONG SEQUENCES: Scalability tests
        {
            "B": 1,
            "qsl": 2048,
            "ksl": 2048,
            "head_dim": 64,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "very_long",
        },
    ]


def benchmark_correctness_check(evolved_output, reference_output, dtype="float16") -> Dict[str, float]:
    """
    CRITICAL: Use EXACT same correctness check as spda_benchmark.py
    
    This is the exact logic from bench_shape() that catches the failures:
    ```python
    atol = 1e-5 if dtype == "float32" else 2e-4
    if not mx.allclose(o_mlx_fused, o_mlx_unfused, atol=atol, rtol=atol):
        print(f"Failed with max(|a - b|) = {mx.max(mx.abs(o_mlx_unfused - o_mlx_fused)):3.2e}")
    ```
    """
    
    # EXACT same tolerance as spda_benchmark.py
    atol = 1e-5 if dtype == "float32" else 2e-4
    rtol = atol
    
    # Ensure arrays are evaluated
    evolved_output = mx.array(evolved_output)
    reference_output = mx.array(reference_output)
    mx.eval(evolved_output, reference_output)
    
    # Calculate differences
    diff = evolved_output - reference_output
    max_diff = float(mx.max(mx.abs(diff)))
    mse = float(mx.mean(diff**2))
    mae = float(mx.mean(mx.abs(diff)))
    
    # EXACT same check as benchmark
    benchmark_passes = bool(mx.allclose(evolved_output, reference_output, atol=atol, rtol=rtol))
    
    return {
        "benchmark_passes": benchmark_passes,
        "max_diff": max_diff,
        "mse": mse,
        "mae": mae,
        "benchmark_atol": atol,
        "benchmark_rtol": rtol,
    }


def test_correctness_by_category(evolved_attention_fn, config: Dict) -> Dict[str, float]:
    """
    Test correctness with benchmark-compatible checks.
    
    CRITICAL: Must catch the same failures that spda_benchmark.py catches.
    """
    
    category = config.get("category", "unknown")
    dtype = config.get("dtype", "float16")
    
    # Unpack test configuration
    B = config["B"]
    qsl = config["qsl"]
    ksl = config["ksl"]
    head_dim = config["head_dim"]
    n_q_heads = config["n_q_heads"]
    n_kv_heads = config["n_kv_heads"]
    mask_type = config.get("mask", None)

    try:
        # Prepare inputs using benchmark function
        q, k, v, scale, mask = prepare_inputs(
            B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, False, dtype
        )

        # Run evolved implementation
        evolved_output = evolved_attention_fn(q, k, v, scale=scale, mask=mask)
        
        # For very long sequences, skip expensive reference comparison
        if qsl >= 3072:
            # Just check for validity
            has_nan = bool(mx.any(mx.isnan(evolved_output)))
            has_inf = bool(mx.any(mx.isinf(evolved_output)))
            shape_correct = evolved_output.shape == q.shape
            
            return {
                "benchmark_passes": not (has_nan or has_inf),
                "max_diff": 0.0,
                "mse": 0.0,
                "mae": 0.0,
                "shape_correct": shape_correct,
                "no_nan_inf": not (has_nan or has_inf),
                "structural_correct": shape_correct and not (has_nan or has_inf),
                "category": category,
                "reference_computed": False,
                "skip_reason": "Very long sequence - too expensive to compare"
            }
        
        # CRITICAL: Test against BOTH reference and fused attention
        # This ensures we catch failures that the benchmark would catch
        
        # Test 1: Compare against reference implementation
        try:
            reference_output = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
            ref_comparison = benchmark_correctness_check(evolved_output, reference_output, dtype)
        except Exception as e:
            print(f"    ⚠️ Reference comparison failed: {e}")
            ref_comparison = {"benchmark_passes": False, "max_diff": float("inf")}
        
        # Test 2: Compare against fused attention (what benchmark actually does)
        fused_comparison_attempted = False
        fused_comparison = {"benchmark_passes": True, "max_diff": 0.0}  # Default to pass
        
        try:
            # This is the CRITICAL comparison that benchmark does
            fused_output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
            fused_comparison = benchmark_correctness_check(evolved_output, fused_output, dtype)
            fused_comparison_attempted = True
            
            # If this fails, it's the EXACT same failure the benchmark would catch
            if not fused_comparison["benchmark_passes"]:
                print(f"    ❌ BENCHMARK FAILURE: max(|evolved - fused|) = {fused_comparison['max_diff']:.2e} "
                      f"> {fused_comparison.get('benchmark_atol', 2e-4):.2e}")
                
        except Exception as e:
            # If fused attention fails, we can't do this comparison
            # This might happen on some systems where mx.fast is not available
            print(f"    ⚠️ Fused comparison skipped: {e}")
        
        # Overall benchmark compatibility
        # Program passes if it works with reference AND (fused comparison passes OR is skipped)
        ref_passes = ref_comparison.get("benchmark_passes", False)
        fused_passes = fused_comparison.get("benchmark_passes", True)  # Default pass if not attempted
        
        benchmark_compatible = ref_passes and fused_passes
        
        # Check structural correctness
        has_nan = bool(mx.any(mx.isnan(evolved_output)))
        has_inf = bool(mx.any(mx.isinf(evolved_output)))
        shape_correct = evolved_output.shape == q.shape
        no_nan_inf = not (has_nan or has_inf)
        
        # Final structural correctness includes benchmark compatibility
        structural_correct = shape_correct and no_nan_inf and benchmark_compatible

        return {
            "benchmark_passes": benchmark_compatible,
            "ref_benchmark_passes": ref_passes,
            "fused_benchmark_passes": fused_passes,
            "fused_comparison_attempted": fused_comparison_attempted,
            "max_diff": max(ref_comparison.get("max_diff", 0), fused_comparison.get("max_diff", 0)),
            "mse": ref_comparison.get("mse", 0.0),
            "mae": ref_comparison.get("mae", 0.0),
            "shape_correct": shape_correct,
            "no_nan_inf": no_nan_inf,
            "structural_correct": structural_correct,
            "category": category,
            "reference_computed": True,
        }

    except Exception as e:
        print(f"    ❌ Correctness test failed: {e}")
        return {
            "benchmark_passes": False,
            "ref_benchmark_passes": False,
            "fused_benchmark_passes": False,
            "fused_comparison_attempted": False,
            "max_diff": float("inf"),
            "mse": float("inf"),
            "mae": float("inf"),
            "shape_correct": False,
            "no_nan_inf": False,
            "structural_correct": False,
            "category": category,
            "reference_computed": False,
            "error": str(e),
        }


def test_sequence_scalability(evolved_attention_fn, config: Dict) -> Dict[str, float]:
    """
    Test how well the attention scales with sequence length.
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
        
        # Test memory efficiency and execution time
        start_time = time.perf_counter()
        
        try:
            output = evolved_attention_fn(q, k, v, scale=scale, mask=mask)
            mx.eval(output)  # Force evaluation
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Check output validity
            has_nan = bool(mx.any(mx.isnan(output)))
            has_inf = bool(mx.any(mx.isinf(output)))
            valid_output = not (has_nan or has_inf)
            
            return {
                "execution_time": execution_time,
                "memory_success": True,
                "valid_output": valid_output,
                "sequence_length": qsl,
                "scalability_category": config.get("category", "unknown"),
            }
            
        except Exception as e:
            return {
                "execution_time": float("inf"),
                "memory_success": False,
                "valid_output": False,
                "sequence_length": qsl,
                "error": str(e),
                "scalability_category": config.get("category", "unknown"),
            }
            
    except Exception as e:
        return {
            "execution_time": float("inf"),
            "memory_success": False,
            "valid_output": False,
            "sequence_length": qsl,
            "error": str(e),
            "scalability_category": config.get("category", "unknown"),
        }


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    Stage 1: Critical correctness check using benchmark-compatible testing.
    
    CRITICAL: This must catch the same failures that spda_benchmark.py catches,
    so programs that would fail the benchmark are rejected during evolution.
    """

    try:
        print(f"[Stage 1] Loading block diagonal attention program from {program_path}")

        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(evolved_program)
        except SyntaxError as e:
            print(f"[Stage 1] ❌ SYNTAX ERROR: {e}")
            return {
                "basic_functionality": 0.0,
                "syntax_error": 1.0,
                "error": f"Syntax error: {str(e)}",
            }
        except Exception as e:
            print(f"[Stage 1] ❌ IMPORT ERROR: {e}")
            return {
                "basic_functionality": 0.0,
                "import_error": 1.0,
                "error": f"Import error: {str(e)}",
            }

        # Check if the required function exists
        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            print(f"[Stage 1] ❌ Missing evolved_scaled_dot_product_attention function")
            return {
                "basic_functionality": 0.0,
                "function_missing": 1.0,
                "error": "Missing evolved_scaled_dot_product_attention function",
            }

        evolved_attention_fn = evolved_program.evolved_scaled_dot_product_attention
        print(f"[Stage 1] ✓ Function loaded successfully")

        # CRITICAL TEST 1: Short sequence (should use protected path)
        short_config = {
            "B": 1,
            "qsl": 128,
            "ksl": 128,
            "head_dim": 64,
            "n_q_heads": 8,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "short",
        }

        print(f"[Stage 1] Testing short sequence: {short_config}")
        try:
            short_correctness = test_correctness_by_category(evolved_attention_fn, short_config)
            print(f"[Stage 1] Short sequence - Benchmark passes: {short_correctness.get('benchmark_passes', False)}, "
                  f"Max diff: {short_correctness.get('max_diff', 'inf'):.2e}")
        except Exception as e:
            print(f"[Stage 1] ❌ Short sequence test failed: {e}")
            return {
                "basic_functionality": 0.0,
                "short_sequence_error": 1.0,
                "error": f"Short sequence test failed: {str(e)}",
            }

        # CRITICAL TEST 2: Transition sequence (where block diagonal kicks in)
        transition_config = {
            "B": 1,
            "qsl": 512,
            "ksl": 512,
            "head_dim": 64,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "transition",
        }

        print(f"[Stage 1] Testing transition sequence: {transition_config}")
        try:
            transition_correctness = test_correctness_by_category(evolved_attention_fn, transition_config)
            print(f"[Stage 1] Transition sequence - Benchmark passes: {transition_correctness.get('benchmark_passes', False)}, "
                  f"Max diff: {transition_correctness.get('max_diff', 'inf'):.2e}")
        except Exception as e:
            print(f"[Stage 1] ❌ Transition sequence test failed: {e}")
            # Don't fail completely on transition issues in early evolution
            transition_correctness = {"benchmark_passes": False}

        # Test 3: Long sequence (scalability check)
        long_config = {
            "B": 1,
            "qsl": 1024,
            "ksl": 1024,
            "head_dim": 64,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "long",
        }

        print(f"[Stage 1] Testing long sequence: {long_config}")
        try:
            long_scalability = test_sequence_scalability(evolved_attention_fn, long_config)
            print(f"[Stage 1] Long sequence - Execution time: {long_scalability.get('execution_time', 'N/A'):.3f}s, "
                  f"Valid: {long_scalability.get('valid_output', False)}")
        except Exception as e:
            print(f"[Stage 1] ❌ Long sequence test failed: {e}")
            long_scalability = {"valid_output": False, "execution_time": float("inf")}

        # SCORING: Critical benchmark compatibility
        short_benchmark_passes = short_correctness.get("benchmark_passes", False)
        transition_benchmark_passes = transition_correctness.get("benchmark_passes", False)
        long_functional = long_scalability.get("valid_output", False) and long_scalability.get("execution_time", float("inf")) < 60.0

        # Strict scoring based on benchmark compatibility
        if short_benchmark_passes and transition_benchmark_passes and long_functional:
            basic_score = 1.0  # Perfect - passes all benchmark tests
            print(f"[Stage 1] 🎉 EXCELLENT: All benchmark tests pass")
        elif short_benchmark_passes and transition_benchmark_passes:
            basic_score = 0.8  # Good - benchmark compatible but long sequence issues
            print(f"[Stage 1] ✅ GOOD: Benchmark compatible, long sequences need work")
        elif short_benchmark_passes and long_functional:
            basic_score = 0.6  # Partial - short sequences work, transition has correctness issues
            print(f"[Stage 1] ⚡ PARTIAL: Short sequences work, transition correctness issues")
        elif short_benchmark_passes:
            basic_score = 0.4  # Minimal - only short sequences work
            print(f"[Stage 1] ⚠️ MINIMAL: Only short sequences work")
        else:
            basic_score = 0.0  # Fail - benchmark incompatible
            print(f"[Stage 1] ❌ FAIL: Benchmark incompatible")

        result = {
            "basic_functionality": float(basic_score),
            "short_benchmark_passes": float(short_benchmark_passes),
            "transition_benchmark_passes": float(transition_benchmark_passes),
            "long_sequence_functional": float(long_functional),
            "benchmark_compatible": float(short_benchmark_passes and transition_benchmark_passes),
        }

        print(f"[Stage 1] ✓ Completed with score: {basic_score:.3f}")
        return result

    except Exception as e:
        print(f"[Stage 1] ❌ Unexpected Exception: {str(e)}")
        traceback.print_exc()
        return {"basic_functionality": 0.0, "unexpected_error": 1.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Stage 2: Comprehensive evaluation only for benchmark-compatible programs.
    """

    print(f"[Stage 2] 🚀 Starting comprehensive evaluation")

    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)

        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            return {
                "accuracy_score": 0.0,
                "scalability_score": 0.0,
                "functionality_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing evolved_scaled_dot_product_attention function",
            }

        evolved_attention_fn = evolved_program.evolved_scaled_dot_product_attention

        # Get test configurations
        test_configs = create_test_configurations()

        benchmark_compatible_count = 0
        total_tests = len(test_configs)
        
        for i, config in enumerate(test_configs):
            category = config.get("category", "unknown")
            
            try:
                print(f"[Stage 2] Testing config {i+1}/{total_tests}: "
                      f"seq={config['qsl']}, category={category}, "
                      f"heads={config['n_q_heads']}/{config['n_kv_heads']}, "
                      f"mask={config.get('mask', None)}")

                # Test correctness with benchmark standards
                correctness = test_correctness_by_category(evolved_attention_fn, config)
                
                # Test scalability
                scalability = test_sequence_scalability(evolved_attention_fn, config)
                
                # Check benchmark compatibility
                benchmark_passes = correctness.get("benchmark_passes", False)
                functional = scalability.get("valid_output", False)
                
                if benchmark_passes and functional:
                    benchmark_compatible_count += 1
                    print(f"  ✅ BENCHMARK COMPATIBLE")
                elif benchmark_passes:
                    print(f"  ⚡ CORRECT but performance issues")
                elif functional:
                    print(f"  ⚠️ FUNCTIONAL but correctness issues")  
                else:
                    print(f"  ❌ FAILED both correctness and functionality")

            except Exception as e:
                print(f"  ❌ Test failed: {str(e)}")

        # Final scoring based on benchmark compatibility
        compatibility_rate = benchmark_compatible_count / total_tests
        
        if compatibility_rate >= 0.9:
            combined_score = 1.0
            print(f"[Stage 2] 🏆 EXCELLENT: {compatibility_rate:.1%} benchmark compatibility")
        elif compatibility_rate >= 0.7:
            combined_score = 0.8
            print(f"[Stage 2] ✅ GOOD: {compatibility_rate:.1%} benchmark compatibility")
        elif compatibility_rate >= 0.5:
            combined_score = 0.6
            print(f"[Stage 2] ⚡ OKAY: {compatibility_rate:.1%} benchmark compatibility")
        elif compatibility_rate >= 0.3:
            combined_score = 0.4
            print(f"[Stage 2] ⚠️ POOR: {compatibility_rate:.1%} benchmark compatibility")
        else:
            combined_score = 0.2
            print(f"[Stage 2] ❌ FAIL: {compatibility_rate:.1%} benchmark compatibility")

        return {
            "accuracy_score": float(compatibility_rate),
            "scalability_score": float(compatibility_rate),
            "functionality_score": float(compatibility_rate),
            "combined_score": float(combined_score),
            "benchmark_compatibility_rate": float(compatibility_rate),
            "benchmark_compatible_count": int(benchmark_compatible_count),
            "total_tests": int(total_tests),
        }

    except Exception as e:
        print(f"[Stage 2] Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "accuracy_score": 0.0,
            "scalability_score": 0.0,
            "functionality_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Main evaluation function - required by OpenEvolve framework.
    
    CRITICAL: This evaluator must catch the same failures that spda_benchmark.py catches,
    ensuring evolved programs are benchmark-compatible.
    """
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    # Test the evaluator with the initial program
    print("Testing benchmark-compatible evaluator...")
    import os

    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")

    if os.path.exists(initial_program_path):
        # Quick stage 1 test
        print("\n=== Stage 1 Test ===")
        stage1_results = evaluate_stage1(initial_program_path)
        print("Stage 1 results:")
        for k, v in stage1_results.items():
            print(f"  {k}: {v}")

        # Full evaluation if stage 1 passes
        if stage1_results.get("basic_functionality", 0.0) > 0.5:
            print("\n=== Stage 2 Test ===")
            stage2_results = evaluate_stage2(initial_program_path)
            print("Stage 2 results summary:")
            for k, v in stage2_results.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
                elif k not in ["detailed_results"]:
                    print(f"  {k}: {v}")
        else:
            print("Stage 1 failed, skipping stage 2")
    else:
        print(f"Initial program not found at {initial_program_path}")
