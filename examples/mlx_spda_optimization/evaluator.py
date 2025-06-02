"""
Evaluator for MLX SPDA Optimization using spda_benchmark.py

This evaluator tests evolved scaled dot product attention implementations by:
1. Checking numerical accuracy against mlx_ref_attn (reference implementation)
2. Measuring performance speedup compared to mlx_fused_attn (the target to beat)
3. Testing across diverse configurations from spda_benchmark.py
4. Ensuring robustness across different mask types and tensor layouts

The goal is to discover attention implementations that beat mx.fast.scaled_dot_product_attention
using only basic MLX operators.
"""

import importlib.util
import math
import time
import traceback
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

# Import benchmark utilities
from spda_benchmark import (
    prepare_inputs, 
    mlx_ref_attn, 
    mlx_fused_attn,
    do_attention,
    bench
)


def create_test_configurations() -> List[Dict]:
    """
    Create test configurations for evaluation.
    Start with smaller, simpler cases and gradually increase complexity.
    """
    return [
        # Small cases for quick testing and debugging
        {"B": 1, "qsl": 32, "ksl": 32, "head_dim": 64, "n_q_heads": 4, "n_kv_heads": 4, "dtype": "float16", "mask": None},
        {"B": 1, "qsl": 64, "ksl": 64, "head_dim": 64, "n_q_heads": 8, "n_kv_heads": 8, "dtype": "float16", "mask": "causal"},
        
        # Medium cases - standard attention patterns
        {"B": 1, "qsl": 128, "ksl": 128, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 16, "dtype": "float16", "mask": None},
        {"B": 1, "qsl": 256, "ksl": 256, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 16, "dtype": "float16", "mask": "causal"},
        {"B": 1, "qsl": 512, "ksl": 512, "head_dim": 64, "n_q_heads": 32, "n_kv_heads": 32, "dtype": "float16", "mask": None},
        
        # Grouped Query Attention (GQA) cases - these are important for modern LLMs
        {"B": 1, "qsl": 256, "ksl": 256, "head_dim": 64, "n_q_heads": 16, "n_kv_heads": 4, "dtype": "float16", "mask": "causal"},
        {"B": 1, "qsl": 512, "ksl": 512, "head_dim": 64, "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16", "mask": None},
        
        # Larger cases - test scalability
        {"B": 1, "qsl": 1024, "ksl": 1024, "head_dim": 64, "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16", "mask": "causal"},
        
        # Different head dimensions
        {"B": 1, "qsl": 512, "ksl": 512, "head_dim": 80, "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16", "mask": None},
        {"B": 1, "qsl": 256, "ksl": 256, "head_dim": 128, "n_q_heads": 16, "n_kv_heads": 8, "dtype": "float16", "mask": "causal"},
        
        # Boolean mask testing
        {"B": 1, "qsl": 128, "ksl": 128, "head_dim": 64, "n_q_heads": 8, "n_kv_heads": 8, "dtype": "float16", "mask": "bool"},
    ]


def compare_attention_outputs(output1: mx.array, output2: mx.array, tolerance: float = 1e-4) -> Dict[str, float]:
    """Compare two attention outputs and return similarity metrics"""
    
    # Ensure arrays are evaluated
    output1 = mx.array(output1)
    output2 = mx.array(output2)
    mx.eval(output1, output2)
    
    # Calculate various similarity metrics
    diff = output1 - output2
    
    # Mean Squared Error
    mse = float(mx.mean(diff ** 2))
    
    # Mean Absolute Error
    mae = float(mx.mean(mx.abs(diff)))
    
    # Maximum absolute difference
    max_diff = float(mx.max(mx.abs(diff)))
    
    # Relative error (normalized by output magnitude)
    output1_norm = float(mx.sqrt(mx.mean(output1 ** 2)))
    relative_error = float(mx.sqrt(mx.mean(diff ** 2))) / max(output1_norm, 1e-8)
    
    # Check MLX's allclose function with strict tolerance for drop-in replacement
    allclose_result = bool(mx.allclose(output1, output2, atol=tolerance, rtol=tolerance))
    
    return {
        "mse": mse,
        "mae": mae,
        "max_diff": max_diff,
        "relative_error": relative_error,
        "allclose": allclose_result,
        "tolerance_used": tolerance
    }


def benchmark_evolved_attention(evolved_attention_fn, test_config: Dict, num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark evolved attention against reference implementations.
    
    Returns timing for evolved function, reference function, and fused function.
    """
    
    # Unpack test configuration
    B = test_config["B"]
    qsl = test_config["qsl"]
    ksl = test_config["ksl"]
    head_dim = test_config["head_dim"]
    n_q_heads = test_config["n_q_heads"]
    n_kv_heads = test_config["n_kv_heads"]
    dtype = test_config["dtype"]
    mask_type = test_config["mask"]
    transpose = False  # Use standard layout for simplicity
    
    # Prepare inputs using benchmark function
    q, k, v, scale, mask = prepare_inputs(
        B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, transpose, dtype
    )
    
    def run_evolved():
        return do_attention(evolved_attention_fn, q, k, v, scale, mask=mask, transpose=transpose)
    
    def run_reference():
        return do_attention(mlx_ref_attn, q, k, v, scale, mask=mask, transpose=transpose)
    
    def run_fused():
        return do_attention(mlx_fused_attn, q, k, v, scale, mask=mask, transpose=transpose)
    
    # Benchmark all three implementations
    try:
        time_evolved = bench(run_evolved)
        time_reference = bench(run_reference)
        time_fused = bench(run_fused)
        
        return {
            "time_evolved": time_evolved,
            "time_reference": time_reference, 
            "time_fused": time_fused,
            "speedup_vs_reference": time_reference / max(time_evolved, 1e-9),
            "speedup_vs_fused": time_fused / max(time_evolved, 1e-9),
            "reference_vs_fused": time_reference / max(time_fused, 1e-9)
        }
        
    except Exception as e:
        return {
            "time_evolved": float('inf'),
            "time_reference": float('inf'),
            "time_fused": float('inf'), 
            "speedup_vs_reference": 0.0,
            "speedup_vs_fused": 0.0,
            "reference_vs_fused": 1.0,
            "error": str(e)
        }


def test_correctness(evolved_attention_fn, test_config: Dict) -> Dict[str, float]:
    """
    Test correctness of evolved attention against reference implementation.
    """
    
    # Unpack test configuration
    B = test_config["B"]
    qsl = test_config["qsl"]
    ksl = test_config["ksl"]
    head_dim = test_config["head_dim"]
    n_q_heads = test_config["n_q_heads"]
    n_kv_heads = test_config["n_kv_heads"]
    dtype = test_config["dtype"]
    mask_type = test_config["mask"]
    transpose = False
    
    try:
        # Prepare inputs
        q, k, v, scale, mask = prepare_inputs(
            B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, transpose, dtype
        )
        
        # Run both implementations
        evolved_output = do_attention(evolved_attention_fn, q, k, v, scale, mask=mask, transpose=transpose)
        reference_output = do_attention(mlx_ref_attn, q, k, v, scale, mask=mask, transpose=transpose)
        
        # Compare outputs with strict tolerance for drop-in replacement
        comparison = compare_attention_outputs(evolved_output, reference_output, tolerance=1e-4)
        
        # Check for structural correctness 
        shape_correct = evolved_output.shape == reference_output.shape
        no_nan_inf = not (bool(mx.any(mx.isnan(evolved_output))) or bool(mx.any(mx.isinf(evolved_output))))
        
        return {
            **comparison,
            "shape_correct": shape_correct,
            "no_nan_inf": no_nan_inf,
            "structural_correct": shape_correct and no_nan_inf
        }
        
    except Exception as e:
        return {
            "mse": float('inf'),
            "mae": float('inf'),
            "max_diff": float('inf'),
            "relative_error": float('inf'),
            "allclose": False,
            "shape_correct": False,
            "no_nan_inf": False,
            "structural_correct": False,
            "error": str(e)
        }


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    Stage 1: Quick correctness check on simple test case.
    This is used for cascade evaluation to quickly filter out broken implementations.
    """
    
    try:
        print(f"[Stage 1] Loading program from {program_path}")
        
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        # Check if the required function exists
        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            print(f"[Stage 1] ❌ Missing evolved_scaled_dot_product_attention function")
            return {
                "basic_functionality": 0.0,
                "error": "Missing evolved_scaled_dot_product_attention function"
            }
        
        evolved_attention_fn = evolved_program.evolved_scaled_dot_product_attention
        print(f"[Stage 1] ✓ Function loaded successfully")
        
        # Simple test case - small dimensions, no GQA, no complex masks
        simple_config = {
            "B": 1, "qsl": 32, "ksl": 32, "head_dim": 64, 
            "n_q_heads": 4, "n_kv_heads": 4, "dtype": "float16", "mask": None
        }
        
        print(f"[Stage 1] Testing with config: {simple_config}")
        
        # Test basic correctness
        correctness = test_correctness(evolved_attention_fn, simple_config)
        
        print(f"[Stage 1] Correctness results: MSE={correctness.get('mse', 'N/A'):.2e}, Allclose={correctness.get('allclose', False)}")
        
        if correctness["structural_correct"]:
            basic_score = 1.0
        elif correctness["shape_correct"]:
            basic_score = 0.5  # Partially working
        else:
            basic_score = 0.0
        
        # Note: MSE removed from scoring to avoid threshold calculation issues
        # MSE is an error metric (lower=better) while others are scores (higher=better)
        result = {
            "basic_functionality": float(basic_score),
            "shape_correct": float(correctness["shape_correct"]),
            "no_nan_inf": float(correctness["no_nan_inf"])
        }
        
        print(f"[Stage 1] ✓ Completed with score: {basic_score}")
        print(f"[Stage 1] Threshold calculation: avg of {list(result.values())} = {sum(result.values())/len(result):.3f}")
        return result
        
    except Exception as e:
        print(f"[Stage 1] ❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "basic_functionality": 0.0,
            "error": str(e)
        }


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Main evaluation function - required by OpenEvolve framework.
    
    For cascade evaluation, this serves as a fallback or can be used
    for non-cascade evaluation. In cascade mode, evaluate_stage1 and 
    evaluate_stage2 will be called instead.
    """
    # For non-cascade evaluation, run the full Stage 2 evaluation
    return evaluate_stage2(program_path)


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Stage 2: Complete evaluation across multiple test configurations.
    
    This tests correctness, performance, and robustness of the evolved attention.
    """
    
    print(f"[Stage 2] 🚀 Starting comprehensive evaluation for {program_path}")
    print(f"[Stage 2] Stage 1 passed threshold - proceeding to full performance evaluation")
    
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            return {
                "accuracy_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing evolved_scaled_dot_product_attention function"
            }
        
        evolved_attention_fn = evolved_program.evolved_scaled_dot_product_attention
        
        # Get test configurations
        test_configs = create_test_configurations()
        
        accuracy_scores = []
        performance_scores = []
        detailed_results = []
        
        successful_tests = 0
        
        for i, config in enumerate(test_configs):
            try:
                print(f"Testing config {i+1}/{len(test_configs)}: "
                      f"seq={config['qsl']}, heads={config['n_q_heads']}/{config['n_kv_heads']}, "
                      f"dim={config['head_dim']}, mask={config['mask']}")
                
                # Test correctness
                correctness = test_correctness(evolved_attention_fn, config)
                
                if not correctness["structural_correct"]:
                    print(f"  ❌ Structural test failed: {correctness.get('error', 'Unknown error')}")
                    accuracy_scores.append(0.0)
                    performance_scores.append(0.0)
                    continue
                
                # ACCURACY-FIRST EVALUATION: Strict accuracy requirements
                # Must be numerically equivalent to reference implementation
                accuracy_threshold_met = False
                accuracy_score = 0.0
                
                if correctness["allclose"] and correctness["mse"] < 1e-6:
                    # Perfect accuracy - meets drop-in replacement requirement
                    accuracy_threshold_met = True
                    accuracy_score = 1.0
                elif correctness["allclose"] and correctness["mse"] < 1e-5:
                    # Very good accuracy - acceptable for most use cases
                    accuracy_threshold_met = True
                    accuracy_score = 0.95
                elif correctness["relative_error"] < 0.001:  # 0.1% relative error
                    # Good accuracy - may be acceptable depending on use case
                    accuracy_threshold_met = True
                    accuracy_score = 0.9
                else:
                    # Insufficient accuracy - cannot be a drop-in replacement
                    accuracy_threshold_met = False
                    accuracy_score = 0.0
                
                accuracy_scores.append(accuracy_score)
                
                # PERFORMANCE EVALUATION: Only for accurate solutions
                if accuracy_threshold_met:
                    perf_results = benchmark_evolved_attention(evolved_attention_fn, config, num_runs=5)
                    speedup_vs_fused = perf_results["speedup_vs_fused"]
                    
                    # Performance score based on speedup vs fused attention
                    if speedup_vs_fused >= 1.05:  # Any measurable improvement (≥5%)
                        # Excellent - this is what we're looking for!
                        performance_score = 1.0 + min((speedup_vs_fused - 1.0) * 10, 2.0)  # Scale up to 3.0
                        print(f"  🎉 SPEEDUP ACHIEVED: {speedup_vs_fused:.3f}x vs fused attention!")
                    elif speedup_vs_fused >= 1.01:  # Small but measurable improvement (≥1%)
                        # Good - small improvements are still valuable
                        performance_score = 1.0 + (speedup_vs_fused - 1.0) * 20  # Scale to ~1.2
                        print(f"  ✅ Small speedup: {speedup_vs_fused:.3f}x vs fused attention")
                    elif speedup_vs_fused >= 0.98:  # Within 2% of fused performance
                        # Acceptable - not slower, might have other benefits
                        performance_score = 0.8 + (speedup_vs_fused - 0.98) * 10  # Scale 0.8-1.0
                        print(f"  ⚡ Competitive: {speedup_vs_fused:.3f}x vs fused attention")
                    elif speedup_vs_fused >= 0.95:  # Within 5% of fused performance
                        # Marginal - barely acceptable
                        performance_score = 0.5 + (speedup_vs_fused - 0.95) * 10  # Scale 0.5-0.8
                        print(f"  ⚠️  Slightly slower: {speedup_vs_fused:.3f}x vs fused attention")
                    else:
                        # Poor - significantly slower than target
                        performance_score = 0.1 * speedup_vs_fused  # Heavy penalty
                        print(f"  ❌ Too slow: {speedup_vs_fused:.3f}x vs fused attention")
                    
                    performance_scores.append(performance_score)
                    
                    print(f"  📊 Accuracy: {accuracy_score:.3f}, Performance: {performance_score:.3f}")
                    
                    detailed_results.append({
                        "config": config,
                        "accuracy_score": accuracy_score,
                        "performance_score": performance_score,
                        "correctness": correctness,
                        "performance": perf_results,
                        "speedup_vs_fused": speedup_vs_fused
                    })
                else:
                    # Inaccurate solution - zero performance score
                    performance_scores.append(0.0)
                    print(f"  ❌ Accuracy insufficient ({accuracy_score:.3f}) - skipping performance test")
                    print(f"      MSE: {correctness.get('mse', 'N/A'):.2e}, Allclose: {correctness.get('allclose', False)}")
                
                successful_tests += 1
                
            except Exception as e:
                print(f"  ❌ Test failed: {str(e)}")
                accuracy_scores.append(0.0)
                performance_scores.append(0.0)
        
        # Calculate final scores with ACCURACY-FIRST approach
        if successful_tests == 0:
            return {
                "accuracy_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "success_rate": 0.0,
                "accurate_solutions": 0,
                "error": "No test configurations passed"
            }
        
        # Average scores across all tests
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        avg_performance = np.mean(performance_scores) if performance_scores else 0.0
        success_rate = successful_tests / len(test_configs)
        
        # Count solutions that meet accuracy threshold
        accurate_solutions = sum(1 for score in accuracy_scores if score >= 0.9)
        accuracy_rate = accurate_solutions / len(test_configs)
        
        # ACCURACY-FIRST COMBINED SCORING:
        # 1. Solutions must be accurate (accuracy_rate acts as gate)
        # 2. Among accurate solutions, performance determines final ranking
        if accurate_solutions == 0:
            # No accurate solutions - this cannot be a drop-in replacement
            combined_score = 0.0
            print(f"\n❌ NO ACCURATE SOLUTIONS FOUND - Cannot be drop-in replacement")
        elif accuracy_rate >= 0.8:  # Most configurations are accurate
            # Excellent accuracy - score based on performance
            combined_score = avg_accuracy * (0.3 + 0.7 * avg_performance)  # Performance-weighted
            print(f"\n✅ HIGH ACCURACY - Performance-driven scoring")
        elif accuracy_rate >= 0.6:  # Majority configurations are accurate
            # Good accuracy - moderate performance weighting
            combined_score = avg_accuracy * (0.5 + 0.5 * avg_performance)
            print(f"\n⚡ GOOD ACCURACY - Balanced scoring")
        else:
            # Poor accuracy rate - heavily penalized
            combined_score = avg_accuracy * 0.5  # Performance doesn't matter much
            print(f"\n⚠️  POOR ACCURACY RATE - Heavy penalty")
        
        print(f"\nFinal Results:")
        print(f"  Accuracy: {avg_accuracy:.3f}")
        print(f"  Performance: {avg_performance:.3f}")
        print(f"  Success Rate: {success_rate:.3f}")
        print(f"  Accurate Solutions: {accurate_solutions}/{len(test_configs)} ({accuracy_rate:.1%})")
        print(f"  Combined Score: {combined_score:.3f}")
        
        return {
            "accuracy_score": float(avg_accuracy),
            "performance_score": float(avg_performance),
            "combined_score": float(combined_score),
            "success_rate": float(success_rate),
            "accuracy_rate": float(accuracy_rate),
            "accurate_solutions": int(accurate_solutions),
            "successful_tests": successful_tests,
            "total_tests": len(test_configs),
            "detailed_results": detailed_results
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            "accuracy_score": 0.0,
            "performance_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the evaluator with the initial program
    print("Testing evaluator with initial program...")
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
                elif k != "detailed_results":
                    print(f"  {k}: {v}")
        else:
            print("Stage 1 failed, skipping stage 2")
    else:
        print(f"Initial program not found at {initial_program_path}")
