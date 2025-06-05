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


# ============================================================================
# RIGOROUS TIMING METHODOLOGY - Copied from test_evolved.py
# ============================================================================

# Timing constants for rigorous benchmarking
N_warmup = 5
N_iter_bench = 40
N_iter_func = 8


def bench(f, *args):
    """Rigorous benchmarking function copied from test_evolved.py"""
    for i in range(N_warmup):
        f(*args)

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(*args)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    """Attention computation copied from test_evolved.py"""
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def do_attention_bench(f, q, k, v, scale, mask=None, transpose=False):
    """Attention benchmarking copied from test_evolved.py"""
    q_out = q

    for i in range(N_iter_func):
        q_out = do_attention(f, q_out, k, v, scale, mask=mask, transpose=transpose)

    mx.eval(q_out)
    return q_out


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    """Input preparation copied from test_evolved.py"""
    np_dtype = getattr(np, dtype)

    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    scale = 1.0 / math.sqrt(D)

    q_np = np.random.normal(0.0, 1.0, shape_q).astype(np_dtype)
    k_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)

    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)

    if mask is not None:
        if mask == "additive":
            mask_np = np.random.normal(0.0, 1.0, (B, qH, qL, kL)).astype(np_dtype)
            mask = mx.array(mask_np)
        elif mask == "bool":
            mask_np = np.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5
            mask = mx.array(mask_np)

    return q_mx, k_mx, v_mx, scale, mask


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
    """Create comprehensive test configurations for robust evaluation."""
    configs = []
    
    # ===== STAGE 1: CORRECTNESS TESTS =====
    # Enhanced with SPDA benchmark configurations for thorough testing
    
    # Block-diagonal correctness tests
    configs.extend([
        {
            "name": "small_uniform_blocks", 
            "B": 1, "H": 4, "L": 128, "D": 64,
            "block_sizes": [64, 64],  # 2 blocks of 64
            "test_type": "correctness"
        },
        {
            "name": "medium_uniform_blocks",
            "B": 1, "H": 8, "L": 512, "D": 64, 
            "block_sizes": [128, 128, 128, 128],  # 4 blocks of 128
            "test_type": "correctness"
        },
        {
            "name": "variable_blocks",
            "B": 1, "H": 8, "L": 768, "D": 64,
            "block_sizes": [256, 512],  # Variable sizes
            "test_type": "correctness"
        },
        {
            "name": "single_large_block",
            "B": 1, "H": 4, "L": 256, "D": 64,
            "block_sizes": [256],  # Single block (edge case)
            "test_type": "correctness"
        }
    ])
    
    # SPDA benchmark configurations for correctness (subset)
    spda_correctness_configs = [
        # Small sizes for fast correctness testing - NO GQA to avoid complexity
        (1, 32, 32, 64, 16, 16, None),      # Basic small
        (1, 64, 64, 64, 16, 16, "bool"),    # Boolean mask
        (1, 128, 128, 64, 16, 16, "causal"), # Causal mask
        (1, 256, 256, 64, 16, 16, None),    # Medium size
        (1, 128, 128, 80, 16, 16, "bool"),  # Different head dim (PaLM)
        (2, 128, 128, 64, 16, 16, "causal"), # Batch size > 1
        (1, 512, 512, 64, 16, 16, "bool"),   # Larger size
        (1, 256, 256, 128, 8, 8, None),    # Large head dim, fewer heads
    ]
    
    for i, (B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type) in enumerate(spda_correctness_configs):
        configs.append({
            "name": f"spda_correctness_{i+1}",
            "test_type": "correctness",
            "spda_config": {
                "B": B, "qsl": qsl, "ksl": ksl, "head_dim": head_dim,
                "n_q_heads": n_q_heads, "n_kv_heads": n_kv_heads,
                "mask_type": mask_type, "dtype": "float16", "transpose": False
            }
        })
    
    # ===== STAGE 2: PERFORMANCE TESTS =====
    # Enhanced with block-diagonal configurations for comprehensive performance testing
    
    # Original performance tests (keep the good ones)
    configs.extend([
        {
            "name": "sparse_large_blocks",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128, 128, 128, 128, 128, 128, 128, 128],  # 8 blocks = 87.5% sparse
            "test_type": "performance"
        },
        {
            "name": "packed_sequences_medium",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],  # BERT-style packing
            "test_type": "performance"
        },
        {
            "name": "extreme_sparse_packing",
            "B": 1, "H": 16, "L": 1024, "D": 128,
            "block_sizes": [64] * 16,  # 16 tiny blocks = 93.75% sparse
            "test_type": "performance"
        }
    ])
    
    # Block-diagonal performance configurations (selected from test_evolved.py)
    block_diagonal_perf_configs = [
        # Basic sparsity progression
        {
            "name": "dense_2x256_sparse50",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256]  # 50% sparse
        },
        {
            "name": "medium_4x128_sparse75", 
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128]  # 75% sparse
        },
        {
            "name": "sparse_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # 87.5% sparse
        },
        {
            "name": "very_sparse_16x32_sparse93",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [32] * 16  # 93.75% sparse
        },
        {
            "name": "extreme_sparse_32x16_sparse96",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [16] * 32  # 96.875% sparse
        },
        # Different sequence lengths
        {
            "name": "large_seq_8x128_sparse87",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128] * 8  # Large sequences
        },
        {
            "name": "huge_seq_16x128_sparse93",
            "B": 1, "H": 32, "L": 2048, "D": 64,
            "block_sizes": [128] * 16  # Very large sequences
        },
        # Different head dimensions
        {
            "name": "head80_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 80,
            "block_sizes": [64] * 8  # PaLM head dim
        },
        {
            "name": "head128_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 128,
            "block_sizes": [64] * 8  # Large head dim
        },
        # Batch variations
        {
            "name": "batch4_8x64_sparse87",
            "B": 4, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Medium batch
        },
        # Real-world scenarios
        {
            "name": "bert_base_packing",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128]  # BERT-style
        },
        {
            "name": "longformer_sparse",
            "B": 1, "H": 16, "L": 2048, "D": 64,
            "block_sizes": [128] * 16  # Longformer-style
        },
        # Extreme sparsity
        {
            "name": "tiny_blocks_64x8_sparse98",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [8] * 64  # 98.4% sparse
        },
        # Mixed patterns
        {
            "name": "mixed_sizes_pyramid",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [512, 256, 128, 64, 32, 16, 8, 8]  # Pyramid
        },
        # Edge cases
        {
            "name": "single_token_blocks",
            "B": 1, "H": 8, "L": 64, "D": 64,
            "block_sizes": [1] * 64  # Extreme sparsity
        }
    ]
    
    # Add block diagonal performance configs
    for config in block_diagonal_perf_configs:
        config["test_type"] = "performance"
        configs.append(config)
    
    return configs


def evaluate_correctness(evolved_fn, config):
    """Test correctness against reference implementation with rigorous methodology."""
    try:
        # Handle two types of configs: block diagonal and SPDA
        if "spda_config" in config:
            # SPDA correctness test
            spda_cfg = config["spda_config"]
            B, qsl, ksl, head_dim = spda_cfg["B"], spda_cfg["qsl"], spda_cfg["ksl"], spda_cfg["head_dim"]
            n_q_heads, n_kv_heads = spda_cfg["n_q_heads"], spda_cfg["n_kv_heads"]
            mask_type, dtype, transpose = spda_cfg["mask_type"], spda_cfg["dtype"], spda_cfg["transpose"]
            
            # Use rigorous input preparation
            q, k, v, scale, mask = prepare_inputs(
                B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, transpose, dtype
            )
            
            # Handle causal mask
            if mask_type == "causal":
                mask = mx.tril(mx.ones((qsl, ksl), dtype=mx.bool_))
                mask = mx.expand_dims(mx.expand_dims(mask, 0), 0)  # Add batch and head dims
                mask = mx.broadcast_to(mask, (B, n_q_heads, qsl, ksl))
        
        else:
            # Block diagonal test
            B, H, L, D = config["B"], config["H"], config["L"], config["D"]
            
            # Create test inputs (using same method as test_evolved.py)
            np_dtype = np.float16  # Use float16 for consistency
            scale = 1.0 / math.sqrt(D)
            
            q_np = np.random.normal(0.0, 1.0, (B, H, L, D)).astype(np_dtype)
            k_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
            v_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
            
            q = mx.array(q_np)
            k = mx.array(k_np)
            v = mx.array(v_np)
            
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
        
        # Determine pass/fail (more stringent than before)
        tolerance = 1e-4 if q.dtype == mx.float32 else 2e-4  # Tighter tolerance
        passed = mse < tolerance and max_diff < 0.05 and not has_nan and not has_inf
        
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


def benchmark_performance(evolved_fn, config):
    """Benchmark evolved kernel vs MLX fast SPDA using rigorous timing methodology."""
    try:
        # Handle only block diagonal configs for performance testing
        if "spda_config" in config:
            return {"speedup": 0.0, "error": "SPDA configs not used for performance testing"}
        
        B, H, L, D = config["B"], config["H"], config["L"], config["D"]
        
        # Create test inputs using same method as test_evolved.py
        np_dtype = np.float16  # Use float16 for consistency
        scale = 1.0 / math.sqrt(D)
        
        q_np = np.random.normal(0.0, 1.0, (B, H, L, D)).astype(np_dtype)
        k_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
        v_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
        
        q = mx.array(q_np)
        k = mx.array(k_np)
        v = mx.array(v_np)
        
        # Create block-diagonal mask
        mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
        
        # Benchmark evolved implementation using RIGOROUS timing methodology
        try:
            time_evolved = bench(
                do_attention_bench, evolved_fn, q, k, v, scale, mask, False
            )
        except Exception as e:
            return {"speedup": 0.0, "error": f"Evolved kernel failed: {str(e)}"}
        
        # Benchmark MLX fast SPDA using RIGOROUS timing methodology
        try:
            time_spda = bench(
                do_attention_bench, mlx_spda_baseline, q, k, v, scale, mask, False
            )
        except Exception as e:
            return {"speedup": float("inf"), "error": f"SPDA baseline failed: {str(e)}"}
        
        # Calculate speedup
        speedup = time_spda / time_evolved if time_evolved > 0 else 0.0
        
        # Calculate theoretical advantage
        total_elements = L * L
        masked_elements = sum(bs * bs for bs in config["block_sizes"])
        sparsity = 1.0 - (masked_elements / total_elements)
        
        # Correctness check
        o_evolved = do_attention(evolved_fn, q, k, v, scale, mask, False)
        o_spda = do_attention(mlx_spda_baseline, q, k, v, scale, mask, False)
        
        atol = 1e-5 if q.dtype == mx.float32 else 2e-4
        correctness_ok = mx.allclose(o_evolved, o_spda, atol=atol, rtol=atol)
        
        return {
            "speedup": speedup,
            "evolved_time": time_evolved,
            "spda_time": time_spda,
            "config_name": config["name"],
            "sparsity": sparsity,
            "correctness_ok": correctness_ok,
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
        print("Enhanced with SPDA benchmark configurations for thorough testing")
        
        test_configs = create_test_configurations()
        correctness_configs = [c for c in test_configs if c["test_type"] == "correctness"]
        
        print(f"  Running {len(correctness_configs)} correctness tests...")
        
        correctness_results = []
        passed_count = 0
        
        for config in correctness_configs:
            if "spda_config" in config:
                cfg_info = config["spda_config"]
                test_desc = f"{cfg_info['qsl']}x{cfg_info['head_dim']} {cfg_info['mask_type'] or 'none'}"
            else:
                test_desc = f"{len(config['block_sizes'])} blocks"
            
            print(f"  Testing {config['name']}: {test_desc}")
            
            result = evaluate_correctness(evolved_fn, config)
            correctness_results.append(result)
            
            if result["passed"]:
                passed_count += 1
                print(f"    ✅ PASSED (MSE: {result.get('mse', 0):.2e})")
            else:
                error_msg = result.get("error", "Accuracy issue")
                print(f"    ❌ FAILED: {error_msg}")
        
        # Calculate pass rate (more stringent requirement)
        pass_rate = passed_count / len(correctness_configs) if correctness_configs else 0.0
        stage1_passed = pass_rate >= 0.80  # 80% pass rate required (increased from 75%)
        
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
        print(f"\n🏁 STAGE 2: Performance vs MLX Fast SPDA")
        print("Using rigorous timing methodology with block-diagonal advantage scenarios")
        
        performance_configs = [c for c in test_configs if c["test_type"] == "performance"]
        print(f"  Running {len(performance_configs)} performance tests...")
        
        performance_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        correctness_failures = 0
        
        for config in performance_configs:
            print(f"  Benchmarking {config['name']}")
            
            # Calculate expected advantage for user info
            total_elements = config["L"] * config["L"]
            masked_elements = sum(bs * bs for bs in config["block_sizes"])
            sparsity = 1.0 - (masked_elements / total_elements)
            print(f"    Expected advantage: {sparsity*100:.1f}% of attention matrix is masked")
            
            result = benchmark_performance(evolved_fn, config)
            performance_results.append(result)
            
            if "error" in result:
                print(f"    ❌ ERROR: {result['error']}")
                continue
            
            speedup = result.get("speedup", 0.0)
            sparsity = result.get("sparsity", 0.0)
            correctness_ok = result.get("correctness_ok", False)
            
            # Track correctness failures in performance tests
            if not correctness_ok:
                correctness_failures += 1
                print(f"    ⚠️ CORRECTNESS ISSUE during performance test")
            
            # Weight by sparsity - more sparse patterns are more important to optimize
            weight = 1.0 + sparsity  # Base weight + sparsity bonus
            
            # Score based on speedup achievement (with correctness penalty)
            if not correctness_ok:
                score = 0.0  # Zero score for incorrect results
            elif speedup >= 2.0:      # 2x+ speedup
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
            
            status = "✅ GOOD" if score >= 0.8 else "⚡ OK" if score >= 0.4 else "❌ SLOW"
            if not correctness_ok:
                status = "❌ WRONG"
            
            print(f"    📊 Speedup: {speedup:.2f}x vs SPDA | Score: {score:.2f} | {status}")
        
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
        if correctness_failures > 0:
            print(f"  ⚠️ Correctness failures in performance tests: {correctness_failures}")
        
        print(f"\n🏆 Overall Results:")
        print(f"  Stage 1 (Correctness): {'✅ PASSED' if stage1_passed else '❌ FAILED'} ({len(correctness_configs)} tests)")
        print(f"  Stage 2 (Performance): {stage2_score:.3f} ({len(performance_configs)} tests)")
        print(f"  Overall Score: {overall_score:.3f}")
        print(f"  Timing Methodology: Rigorous ({N_warmup} warmup, {N_iter_bench} iterations, {N_iter_func} function calls)")
        
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
            "num_correctness_tests": len(correctness_configs),
            "num_performance_tests": len(performance_configs),
            "correctness_failures_in_perf": correctness_failures,
            "total_tests": len(test_configs),
            "timing_methodology": "rigorous"
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
