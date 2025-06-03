"""
Evaluator for MLX Block Diagonal Attention Optimization

This evaluator tests evolved block diagonal attention implementations by:
1. Verifying hybrid dispatcher works correctly (short vs long sequences)
2. Testing block diagonal attention quality and efficiency on long sequences
3. Measuring scalability improvements (linear vs quadratic complexity)
4. Ensuring graceful handling of various sequence lengths and configurations
5. Evaluating novel block pattern discoveries

The goal is to discover block diagonal attention patterns that enable
processing of long sequences (4K+ tokens) that are currently infeasible
with standard quadratic attention.
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
    Create test configurations focused on block diagonal attention evaluation.
    
    Strategy:
    1. Short sequences: Verify hybrid dispatcher uses optimal implementation
    2. Medium sequences: Test transition behavior around 512 threshold  
    3. Long sequences: Test block diagonal attention capabilities
    4. Very long sequences: Test scalability and memory efficiency
    """
    return [
        # SHORT SEQUENCES: Should use mx.fast.scaled_dot_product_attention
        # These test the hybrid dispatcher's short sequence path
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
            "qsl": 256,
            "ksl": 256,
            "head_dim": 64,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "short",
        },
        
        # TRANSITION SEQUENCES: Test behavior around 512 threshold
        {
            "B": 1,
            "qsl": 480,
            "ksl": 480,
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
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "transition",
        },
        
        # LONG SEQUENCES: Main target for block diagonal attention
        # These test the novel algorithmic capabilities
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
        
        # VERY LONG SEQUENCES: Scalability and memory efficiency tests
        # These test the limits of what's possible
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
        {
            "B": 1,
            "qsl": 3072,
            "ksl": 3072,
            "head_dim": 64,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "very_long",
        },
        {
            "B": 1,
            "qsl": 4096,
            "ksl": 4096,
            "head_dim": 64,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "very_long",
        },
        
        # DIFFERENT HEAD DIMENSIONS: Test generalization
        {
            "B": 1,
            "qsl": 1024,
            "ksl": 1024,
            "head_dim": 80,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": None,
            "category": "long",
        },
        {
            "B": 1,
            "qsl": 2048,
            "ksl": 2048,
            "head_dim": 128,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "very_long",
        },
    ]


def compare_attention_outputs(
    output1: mx.array, output2: mx.array, tolerance: float = 1e-3
) -> Dict[str, float]:
    """
    Compare two attention outputs with appropriate tolerance for block diagonal attention.
    
    Note: Block diagonal attention may have different accuracy characteristics
    than full attention, so we use more relaxed tolerances for long sequences.
    """

    # Ensure arrays are evaluated
    output1 = mx.array(output1)
    output2 = mx.array(output2)
    mx.eval(output1, output2)

    # Calculate various similarity metrics
    diff = output1 - output2

    # Mean Squared Error
    mse = float(mx.mean(diff**2))

    # Mean Absolute Error
    mae = float(mx.mean(mx.abs(diff)))

    # Maximum absolute difference
    max_diff = float(mx.max(mx.abs(diff)))

    # Relative error (normalized by output magnitude)
    output1_norm = float(mx.sqrt(mx.mean(output1**2)))
    relative_error = float(mx.sqrt(mx.mean(diff**2))) / max(output1_norm, 1e-8)

    # Check MLX's allclose function
    allclose_result = bool(mx.allclose(output1, output2, atol=tolerance, rtol=tolerance))

    return {
        "mse": mse,
        "mae": mae,
        "max_diff": max_diff,
        "relative_error": relative_error,
        "allclose": allclose_result,
        "tolerance_used": tolerance,
    }


def test_sequence_scalability(evolved_attention_fn, config: Dict) -> Dict[str, float]:
    """
    Test how well the attention scales with sequence length.
    
    For block diagonal attention, we expect:
    1. Constant or linear memory usage
    2. Linear or sub-quadratic time complexity
    3. Graceful quality degradation for very long sequences
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
        
        # Test memory efficiency: Can we even create the attention output?
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
            
            # Estimate complexity based on sequence length
            theoretical_quadratic_ops = qsl * qsl * n_q_heads * B
            actual_ops_estimate = execution_time * 1e9  # Rough FLOP estimate
            
            return {
                "execution_time": execution_time,
                "memory_success": True,
                "valid_output": valid_output,
                "sequence_length": qsl,
                "theoretical_quadratic_ops": theoretical_quadratic_ops,
                "efficiency_score": min(1.0, theoretical_quadratic_ops / max(actual_ops_estimate, 1e6)),
                "scalability_category": config.get("category", "unknown"),
            }
            
        except mx.errors.OutOfMemoryError:
            return {
                "execution_time": float("inf"),
                "memory_success": False,
                "valid_output": False,
                "sequence_length": qsl,
                "error": "Out of memory",
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


def test_correctness_by_category(evolved_attention_fn, config: Dict) -> Dict[str, float]:
    """
    Test correctness with different expectations based on sequence category.
    
    - Short sequences: Should be nearly identical to reference (hybrid dispatcher)
    - Long sequences: Allow for quality degradation due to block approximation
    """
    
    category = config.get("category", "unknown")
    
    # Adjust tolerance based on category
    if category == "short":
        # Short sequences should be nearly perfect (using mx.fast.scaled_dot_product_attention)
        tolerance = 1e-5
        expected_quality = "perfect"
    elif category == "transition":
        # Transition sequences should still be high quality
        tolerance = 1e-4
        expected_quality = "high"
    elif category == "long":
        # Long sequences may have some quality degradation due to block approximation
        tolerance = 1e-3
        expected_quality = "good"
    elif category == "very_long":
        # Very long sequences: focus on functionality over perfect accuracy
        tolerance = 1e-2
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
                "mse": 0.0,  # Cannot compute without reference
                "mae": 0.0,
                "max_diff": 0.0,
                "relative_error": 0.0,
                "allclose": not (has_nan or has_inf),
                "shape_correct": shape_correct,
                "no_nan_inf": not (has_nan or has_inf),
                "structural_correct": shape_correct and not (has_nan or has_inf),
                "tolerance_used": tolerance,
                "expected_quality": expected_quality,
                "category": category,
                "reference_computed": False,
            }
        
        # For shorter sequences, compute reference for comparison
        try:
            reference_output = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
        except Exception:
            # Reference failed (possibly out of memory), skip comparison
            has_nan = bool(mx.any(mx.isnan(evolved_output)))
            has_inf = bool(mx.any(mx.isinf(evolved_output)))
            shape_correct = evolved_output.shape == q.shape
            
            return {
                "mse": 0.0,
                "mae": 0.0,
                "max_diff": 0.0,
                "relative_error": 0.0,
                "allclose": not (has_nan or has_inf),
                "shape_correct": shape_correct,
                "no_nan_inf": not (has_nan or has_inf),
                "structural_correct": shape_correct and not (has_nan or has_inf),
                "tolerance_used": tolerance,
                "expected_quality": expected_quality,
                "category": category,
                "reference_computed": False,
                "reference_error": "Reference computation failed",
            }

        # Compare outputs with category-appropriate tolerance
        comparison = compare_attention_outputs(evolved_output, reference_output, tolerance=tolerance)

        # Check for structural correctness
        shape_correct = evolved_output.shape == reference_output.shape
        no_nan_inf = not (
            bool(mx.any(mx.isnan(evolved_output))) or bool(mx.any(mx.isinf(evolved_output)))
        )

        return {
            **comparison,
            "shape_correct": shape_correct,
            "no_nan_inf": no_nan_inf,
            "structural_correct": shape_correct and no_nan_inf,
            "expected_quality": expected_quality,
            "category": category,
            "reference_computed": True,
        }

    except Exception as e:
        return {
            "mse": float("inf"),
            "mae": float("inf"),
            "max_diff": float("inf"),
            "relative_error": float("inf"),
            "allclose": False,
            "shape_correct": False,
            "no_nan_inf": False,
            "structural_correct": False,
            "tolerance_used": tolerance,
            "expected_quality": expected_quality,
            "category": category,
            "reference_computed": False,
            "error": str(e),
        }


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    Stage 1: Quick functionality check for block diagonal attention system.
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

        # Test 1: Short sequence (should use optimal path)
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
            print(f"[Stage 1] Short sequence - MSE: {short_correctness.get('mse', 'N/A'):.2e}, "
                  f"Category: {short_correctness.get('category', 'N/A')}")
        except Exception as e:
            print(f"[Stage 1] ❌ Short sequence test failed: {e}")
            return {
                "basic_functionality": 0.0,
                "short_sequence_error": 1.0,
                "error": f"Short sequence test failed: {str(e)}",
            }

        # Test 2: Long sequence (should use block diagonal)
        long_config = {
            "B": 1,
            "qsl": 1024,
            "ksl": 1024,
            "head_dim": 64,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "dtype": "float16",
            "mask": "causal",
            "category": "long",
        }

        print(f"[Stage 1] Testing long sequence: {long_config}")
        try:
            long_scalability = test_sequence_scalability(evolved_attention_fn, long_config)
            print(f"[Stage 1] Long sequence - Execution time: {long_scalability.get('execution_time', 'N/A'):.3f}s, "
                  f"Valid: {long_scalability.get('valid_output', False)}")
        except Exception as e:
            print(f"[Stage 1] ❌ Long sequence test failed: {e}")
            # Don't fail completely - long sequence issues are acceptable in early evolution
            long_scalability = {"valid_output": False, "execution_time": float("inf")}

        # Scoring based on hybrid system functionality
        short_success = short_correctness.get("structural_correct", False) and short_correctness.get("allclose", False)
        long_success = long_scalability.get("valid_output", False) and long_scalability.get("execution_time", float("inf")) < 60.0

        if short_success and long_success:
            basic_score = 1.0  # Both paths working
            print(f"[Stage 1] 🎉 EXCELLENT: Both short and long sequence paths working")
        elif short_success:
            basic_score = 0.8  # At least short path works (hybrid dispatcher working)
            print(f"[Stage 1] ✅ GOOD: Short sequences working, long sequences need improvement")
        elif long_success:
            basic_score = 0.6  # Long sequences work but short path broken
            print(f"[Stage 1] ⚡ PARTIAL: Long sequences working, short path issues")
        else:
            basic_score = 0.2  # Neither path working well
            print(f"[Stage 1] ❌ POOR: Both sequence paths have issues")

        result = {
            "basic_functionality": float(basic_score),
            "short_sequence_success": float(short_success),
            "long_sequence_success": float(long_success),
            "hybrid_dispatcher_working": float(short_success),
        }

        print(f"[Stage 1] ✓ Completed with score: {basic_score:.3f}")
        return result

    except Exception as e:
        print(f"[Stage 1] ❌ Unexpected Exception: {str(e)}")
        traceback.print_exc()
        return {"basic_functionality": 0.0, "unexpected_error": 1.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Stage 2: Comprehensive evaluation of block diagonal attention capabilities.
    """

    print(f"[Stage 2] 🚀 Starting comprehensive block diagonal attention evaluation")

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

        # Separate results by category
        results_by_category = {
            "short": [],
            "transition": [],
            "long": [],
            "very_long": [],
        }

        all_results = []
        
        for i, config in enumerate(test_configs):
            category = config.get("category", "unknown")
            
            try:
                print(f"Testing config {i+1}/{len(test_configs)}: "
                      f"seq={config['qsl']}, category={category}, "
                      f"heads={config['n_q_heads']}/{config['n_kv_heads']}, "
                      f"mask={config.get('mask', None)}")

                # Test correctness
                correctness = test_correctness_by_category(evolved_attention_fn, config)
                
                # Test scalability
                scalability = test_sequence_scalability(evolved_attention_fn, config)
                
                # Combine results
                result = {
                    "config": config,
                    "correctness": correctness,
                    "scalability": scalability,
                    "category": category,
                }
                
                all_results.append(result)
                results_by_category[category].append(result)
                
                # Print summary
                accuracy_ok = correctness.get("structural_correct", False)
                scalability_ok = scalability.get("valid_output", False)
                exec_time = scalability.get("execution_time", float("inf"))
                
                if accuracy_ok and scalability_ok:
                    print(f"  ✅ SUCCESS: Accuracy ✓, Scalability ✓ ({exec_time:.3f}s)")
                elif accuracy_ok:
                    print(f"  ⚡ PARTIAL: Accuracy ✓, Scalability ❌")
                elif scalability_ok:
                    print(f"  ⚠️  PARTIAL: Accuracy ❌, Scalability ✓ ({exec_time:.3f}s)")
                else:
                    print(f"  ❌ FAILED: Both accuracy and scalability issues")

            except Exception as e:
                print(f"  ❌ Test failed: {str(e)}")
                result = {
                    "config": config,
                    "correctness": {"structural_correct": False, "error": str(e)},
                    "scalability": {"valid_output": False, "error": str(e)},
                    "category": category,
                }
                all_results.append(result)
                results_by_category[category].append(result)

        # Calculate category-specific scores
        category_scores = {}
        
        for category, results in results_by_category.items():
            if not results:
                category_scores[category] = {"accuracy": 0.0, "scalability": 0.0, "functionality": 0.0}
                continue
                
            # Accuracy score for this category
            accuracy_scores = []
            scalability_scores = []
            functionality_scores = []
            
            for result in results:
                # Accuracy scoring
                correctness = result["correctness"]
                if correctness.get("structural_correct", False):
                    if correctness.get("allclose", False):
                        accuracy_scores.append(1.0)
                    elif correctness.get("mse", float("inf")) < 1e-3:
                        accuracy_scores.append(0.8)
                    else:
                        accuracy_scores.append(0.5)
                else:
                    accuracy_scores.append(0.0)
                
                # Scalability scoring
                scalability = result["scalability"]
                if scalability.get("valid_output", False):
                    exec_time = scalability.get("execution_time", float("inf"))
                    seq_len = scalability.get("sequence_length", 1)
                    
                    # Score based on efficiency for sequence length
                    if exec_time < 0.1:
                        scalability_scores.append(1.0)
                    elif exec_time < 1.0:
                        scalability_scores.append(0.8)
                    elif exec_time < 10.0:
                        scalability_scores.append(0.6)
                    else:
                        scalability_scores.append(0.3)
                else:
                    scalability_scores.append(0.0)
                
                # Functionality scoring (can it handle this sequence length at all?)
                if scalability.get("memory_success", False) and scalability.get("valid_output", False):
                    functionality_scores.append(1.0)
                elif scalability.get("memory_success", False):
                    functionality_scores.append(0.5)
                else:
                    functionality_scores.append(0.0)
            
            category_scores[category] = {
                "accuracy": np.mean(accuracy_scores) if accuracy_scores else 0.0,
                "scalability": np.mean(scalability_scores) if scalability_scores else 0.0,
                "functionality": np.mean(functionality_scores) if functionality_scores else 0.0,
            }

        # Calculate overall scores with category weighting
        # Weight categories by importance for block diagonal attention
        category_weights = {
            "short": 0.2,      # Should be perfect (hybrid dispatcher)
            "transition": 0.2, # Should work well (transition region)
            "long": 0.4,       # Main target (block diagonal attention)
            "very_long": 0.2,  # Stretch goal (extreme scalability)
        }
        
        overall_accuracy = sum(
            category_scores[cat]["accuracy"] * category_weights[cat]
            for cat in category_weights.keys()
        )
        
        overall_scalability = sum(
            category_scores[cat]["scalability"] * category_weights[cat]
            for cat in category_weights.keys()
        )
        
        overall_functionality = sum(
            category_scores[cat]["functionality"] * category_weights[cat]
            for cat in category_weights.keys()
        )

        # Combined scoring for block diagonal attention
        # Priority: Functionality > Scalability > Accuracy
        # (It's better to handle long sequences with some quality loss than not at all)
        
        if overall_functionality >= 0.8:
            # High functionality: weight scalability and accuracy
            combined_score = 0.4 * overall_functionality + 0.4 * overall_scalability + 0.2 * overall_accuracy
        elif overall_functionality >= 0.6:
            # Medium functionality: focus on improving functionality and scalability
            combined_score = 0.6 * overall_functionality + 0.3 * overall_scalability + 0.1 * overall_accuracy
        else:
            # Low functionality: primarily focus on getting basic functionality working
            combined_score = 0.8 * overall_functionality + 0.2 * overall_scalability

        # Report results
        print(f"\n📊 Block Diagonal Attention Evaluation Results:")
        print(f"  Overall Accuracy: {overall_accuracy:.3f}")
        print(f"  Overall Scalability: {overall_scalability:.3f}")
        print(f"  Overall Functionality: {overall_functionality:.3f}")
        print(f"  Combined Score: {combined_score:.3f}")
        
        print(f"\n📋 Category Breakdown:")
        for category, scores in category_scores.items():
            print(f"  {category:12}: Acc={scores['accuracy']:.3f}, Scale={scores['scalability']:.3f}, Func={scores['functionality']:.3f}")

        # Special achievements for long sequence handling
        max_working_sequence = 0
        for result in all_results:
            if result["scalability"].get("valid_output", False):
                seq_len = result["scalability"].get("sequence_length", 0)
                max_working_sequence = max(max_working_sequence, seq_len)
        
        print(f"\n🎯 Long Sequence Capabilities:")
        print(f"  Maximum working sequence length: {max_working_sequence}")
        
        if max_working_sequence >= 4096:
            print(f"  🏆 BREAKTHROUGH: Handling 4K+ sequences!")
        elif max_working_sequence >= 2048:
            print(f"  🚀 EXCELLENT: Handling 2K+ sequences")
        elif max_working_sequence >= 1024:
            print(f"  ✅ GOOD: Handling 1K+ sequences")
        else:
            print(f"  ⚠️  LIMITED: Need to improve long sequence handling")

        return {
            "accuracy_score": float(overall_accuracy),
            "scalability_score": float(overall_scalability),
            "functionality_score": float(overall_functionality),
            "combined_score": float(combined_score),
            "max_working_sequence": int(max_working_sequence),
            "category_scores": category_scores,
            "total_tests": len(test_configs),
            "detailed_results": all_results,
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
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
    """
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    # Test the evaluator with the initial program
    print("Testing block diagonal attention evaluator...")
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
                elif k not in ["detailed_results", "category_scores"]:
                    print(f"  {k}: {v}")
        else:
            print("Stage 1 failed, skipping stage 2")
    else:
        print(f"Initial program not found at {initial_program_path}")
