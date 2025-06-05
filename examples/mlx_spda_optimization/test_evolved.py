#!/usr/bin/env python3
"""
Comprehensive benchmark for evolved block-diagonal attention implementations

This script runs both:
1. Official SPDA benchmark tests (using exact same methodology as spda_benchmark.py) 
2. Block-diagonal specific tests where our custom kernel should excel

All benchmarking methodology copied directly from spda_benchmark.py for consistency.

Usage: python test_evolved.py <program_path>
Example: python test_evolved.py initial_program.py
Example: python test_evolved.py openevolve_output/best/best_program.py
"""

import importlib.util
import math
import os
import sys
import time
from typing import Optional

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX or NumPy not available")
    MLX_AVAILABLE = False
    sys.exit(1)

# ============================================================================
# BENCHMARKING METHODOLOGY - Copied directly from spda_benchmark.py
# ============================================================================

# Timing constants from spda_benchmark.py
N_warmup = 5
N_iter_bench = 40
N_iter_func = 8


def bench(f, *args):
    """Benchmarking function copied from spda_benchmark.py"""
    for i in range(N_warmup):
        f(*args)

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(*args)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    """Input preparation copied from spda_benchmark.py"""
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


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    """Attention computation copied from spda_benchmark.py"""
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def do_attention_bench(f, q, k, v, scale, mask=None, transpose=False):
    """Attention benchmarking copied from spda_benchmark.py"""
    q_out = q

    for i in range(N_iter_func):
        q_out = do_attention(f, q_out, k, v, scale, mask=mask, transpose=transpose)

    mx.eval(q_out)
    return q_out


def bench_shape(evolved_fn, B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype, transpose=False, mask_in=None):
    """Shape benchmarking copied and adapted from spda_benchmark.py"""
    q_mx, k_mx, v_mx, scale, mask = prepare_inputs(
        B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_in, transpose, dtype
    )

    # Benchmark evolved function
    time_evolved = bench(
        do_attention_bench, evolved_fn, q_mx, k_mx, v_mx, scale, mask, transpose
    )
    
    # Benchmark SPDA
    time_spda = bench(
        do_attention_bench, mx.fast.scaled_dot_product_attention, q_mx, k_mx, v_mx, scale, mask, transpose
    )

    # Correctness check (same as spda_benchmark.py)
    o_evolved = do_attention(evolved_fn, q_mx, k_mx, v_mx, scale, mask, transpose)
    o_spda = do_attention(mx.fast.scaled_dot_product_attention, q_mx, k_mx, v_mx, scale, mask, transpose)

    atol = 1e-5 if dtype == "float32" else 2e-4

    if not mx.allclose(o_evolved, o_spda, atol=atol, rtol=atol):
        max_diff = mx.max(mx.abs(o_evolved - o_spda))
        print(f"Failed at (B: {B}, qsl: {qsl}, ksl: {ksl}, head_dim: {head_dim}, "
              f"n_qh: {n_q_heads}, n_kvh: {n_kv_heads}, mask: {mask_in}) "
              f"[tpose = {transpose}] with max(|a - b|) = {max_diff:3.2e}")

    return time_spda, time_evolved


# ============================================================================
# BLOCK-DIAGONAL SPECIFIC FUNCTIONS
# ============================================================================

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


def bench_block_diagonal_shape(evolved_fn, B, H, L, D, block_sizes, dtype="float16"):
    """Benchmark block-diagonal configuration using same methodology"""
    # Create inputs using same method as prepare_inputs
    np_dtype = getattr(np, dtype)
    scale = 1.0 / math.sqrt(D)

    q_np = np.random.normal(0.0, 1.0, (B, H, L, D)).astype(np_dtype)
    k_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)

    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)
    
    # Create block-diagonal mask
    mask = create_block_diagonal_mask(B, H, L, block_sizes)
    
    # Benchmark evolved function using exact same methodology
    time_evolved = bench(
        do_attention_bench, evolved_fn, q_mx, k_mx, v_mx, scale, mask, False
    )
    
    # Benchmark SPDA using exact same methodology
    time_spda = bench(
        do_attention_bench, mx.fast.scaled_dot_product_attention, q_mx, k_mx, v_mx, scale, mask, False
    )

    # Correctness check
    o_evolved = do_attention(evolved_fn, q_mx, k_mx, v_mx, scale, mask, False)
    o_spda = do_attention(mx.fast.scaled_dot_product_attention, q_mx, k_mx, v_mx, scale, mask, False)

    atol = 1e-5 if dtype == "float32" else 2e-4
    
    correctness_ok = True
    if not mx.allclose(o_evolved, o_spda, atol=atol, rtol=atol):
        max_diff = mx.max(mx.abs(o_evolved - o_spda))
        print(f"   ⚠️ Correctness issue: max diff = {max_diff:3.2e}")
        correctness_ok = False

    return time_spda, time_evolved, correctness_ok


# ============================================================================
# MAIN BENCHMARKING FUNCTIONS
# ============================================================================

def load_attention_function(program_path: str):
    """Load the attention function from the specified program file"""
    if not os.path.exists(program_path):
        raise FileNotFoundError(f"Program file not found: {program_path}")

    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if not hasattr(program, "evolved_scaled_dot_product_attention"):
        raise AttributeError("Program missing evolved_scaled_dot_product_attention function")

    return program.evolved_scaled_dot_product_attention


def run_official_spda_benchmark(evolved_fn):
    """Run the official SPDA benchmark tests using exact same methodology"""
    print("\n" + "=" * 80)
    print("📊 OFFICIAL SPDA BENCHMARK TESTS")
    print("=" * 80)
    print("Testing evolved attention vs mx.fast.scaled_dot_product_attention")
    print("Using EXACT same methodology as spda_benchmark.py")
    print("Format: B, qsl, ksl, hdim, n_qh, n_kvh, t, dtype, mask, t_spda, t_evolved, diff%")
    print("-" * 80)
    
    # EXACT same configurations as spda_benchmark.py
    dtypes = ("float16",)
    transposes = (False,)
    
    shapes_64 = (
        (1, 32, 32, 64, 32, 32),
        (1, 64, 64, 64, 32, 32), 
        (1, 128, 128, 64, 32, 32),
        (1, 256, 256, 64, 32, 32),
        (1, 512, 512, 64, 32, 32),
        (1, 1024, 1024, 64, 32, 8),
        (1, 2048, 2048, 64, 32, 8),
        (1, 4096, 4096, 64, 32, 8),
    )
    
    shapes_80 = (
        (1, 1024, 1024, 80, 32, 8),
        (1, 2048, 2048, 80, 32, 8),
        (1, 4096, 4096, 80, 32, 8),
    )
    
    shapes_128 = (
        (1, 1024, 1024, 128, 32, 8),
        (1, 2048, 2048, 128, 32, 8),
        (1, 4096, 4096, 128, 32, 8),
    )
    
    shapes = shapes_64 + shapes_80 + shapes_128
    masks = [None, "bool", "causal"]
    
    official_results = []
    
    for dtype in dtypes:
        for transpose in transposes:
            for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in shapes:
                for mask_in in masks:
                    try:
                        # Use our copied bench_shape function
                        time_spda, time_evolved = bench_shape(
                            evolved_fn, B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype, transpose, mask_in
                        )
                        
                        # Calculate performance difference
                        diff = time_evolved / time_spda - 1.0
                        speedup = time_spda / time_evolved if time_evolved > 0 else 0.0
                        
                        # Color coding: green for speedup, red for slowdown
                        if diff < -0.05:  # >5% speedup
                            color = "\033[92m"  # Green
                        elif diff > 0.05:  # >5% slowdown
                            color = "\033[91m"  # Red
                        else:
                            color = "\033[93m"  # Yellow
                        reset_color = "\033[0m"
                        
                        t_str = 1 if transpose else 0
                        
                        print(
                            f"{color}{B:3d}, {qsl:5d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, "
                            f"{n_kv_heads:5d}, {t_str:1d}, {dtype}, {str(mask_in):>8}, "
                            f"{time_spda:6.3f}, {time_evolved:6.3f},{100. * diff:+6.2f}% "
                            f"(speedup: {speedup:.2f}x){reset_color}"
                        )
                        
                        official_results.append({
                            "config": f"{qsl}x{head_dim}_{mask_in}",
                            "speedup": speedup,
                            "diff_pct": diff * 100,
                            "time_spda": time_spda,
                            "time_evolved": time_evolved
                        })
                        
                    except Exception as e:
                        print(f"FAILED: {B}, {qsl}, {ksl}, {head_dim}, {n_q_heads}, {n_kv_heads}, "
                              f"{dtype}, {mask_in} - {str(e)}")
    
    return official_results


def run_block_diagonal_tests(evolved_fn):
    """Run block-diagonal specific tests using same rigorous methodology"""
    print("\n" + "=" * 80)
    print("🎯 BLOCK-DIAGONAL SPECIFIC TESTS")
    print("=" * 80)
    print("Testing scenarios where block-diagonal attention should outperform SPDA")
    print("Using same rigorous timing methodology as official benchmark")
    print("Format: Test | Shape | Blocks | Sparsity | Evolved | SPDA | Speedup | Status")
    print("-" * 80)
    
    # Block-diagonal test configurations - comprehensive coverage
    block_configs = [
        # ===== BASIC SPARSITY PROGRESSION =====
        {
            "name": "dense_2x256_sparse50",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256]  # 50% sparse - baseline
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
        
        # ===== DIFFERENT SEQUENCE LENGTHS =====
        {
            "name": "small_seq_4x32_sparse75",
            "B": 1, "H": 8, "L": 128, "D": 64,
            "block_sizes": [32, 32, 32, 32]  # Small sequences
        },
        {
            "name": "medium_seq_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Medium sequences
        },
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
        {
            "name": "giant_seq_32x64_sparse96",
            "B": 1, "H": 32, "L": 2048, "D": 64,
            "block_sizes": [64] * 32  # Extreme sequences
        },
        
        # ===== DIFFERENT HEAD DIMENSIONS =====
        {
            "name": "head64_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Standard head dim
        },
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
        {
            "name": "head32_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 32,
            "block_sizes": [64] * 8  # Small head dim
        },
        
        # ===== MIXED BLOCK SIZES =====
        {
            "name": "mixed_sizes_pyramid",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [512, 256, 128, 64, 32, 16, 8, 8]  # Pyramid pattern
        },
        {
            "name": "mixed_sizes_alternating",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128, 64, 128, 64, 128, 64, 128, 64, 128, 64]  # Alternating
        },
        {
            "name": "mixed_sizes_bimodal",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [256, 256, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]  # Two large + many small
        },
        
        # ===== BATCH SIZE VARIATIONS =====
        {
            "name": "batch1_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Single batch
        },
        {
            "name": "batch2_8x64_sparse87",
            "B": 2, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Small batch
        },
        {
            "name": "batch4_8x64_sparse87",
            "B": 4, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Medium batch
        },
        {
            "name": "batch8_8x64_sparse87",
            "B": 8, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Large batch
        },
        
        # ===== HEAD COUNT VARIATIONS =====
        {
            "name": "heads4_8x64_sparse87",
            "B": 1, "H": 4, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Few heads
        },
        {
            "name": "heads16_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Standard heads
        },
        {
            "name": "heads32_8x64_sparse87",
            "B": 1, "H": 32, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Many heads
        },
        {
            "name": "heads64_8x64_sparse87",
            "B": 1, "H": 64, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Very many heads
        },
        
        # ===== TINY BLOCKS (EXTREME SPARSITY) =====
        {
            "name": "tiny_blocks_64x8_sparse98",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [8] * 64  # 98.4% sparse
        },
        {
            "name": "tiny_blocks_128x4_sparse99",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [4] * 128  # 99.2% sparse
        },
        
        # ===== LARGE BLOCKS (DENSE PATTERNS) =====
        {
            "name": "large_blocks_2x256_sparse50",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256]  # Only 50% sparse
        },
        {
            "name": "large_blocks_1x512_sparse0",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [512]  # Not sparse at all
        },
        
        # ===== REAL-WORLD SCENARIOS =====
        {
            "name": "bert_base_packing",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128]  # BERT-style sequence packing
        },
        {
            "name": "bert_large_packing",
            "B": 2, "H": 16, "L": 512, "D": 64,
            "block_sizes": [256, 256]  # BERT-Large style
        },
        {
            "name": "gpt_style_packing",
            "B": 1, "H": 32, "L": 1024, "D": 64,
            "block_sizes": [512, 512]  # GPT-style long sequences
        },
        {
            "name": "t5_encoder_packing",
            "B": 4, "H": 16, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128]  # T5 encoder style
        },
        {
            "name": "longformer_sparse",
            "B": 1, "H": 16, "L": 2048, "D": 64,
            "block_sizes": [128] * 16  # Longformer-style local attention
        },
        
        # ===== EDGE CASES =====
        {
            "name": "single_token_blocks",
            "B": 1, "H": 8, "L": 64, "D": 64,
            "block_sizes": [1] * 64  # Extreme case: every token is its own block
        },
        {
            "name": "uneven_tiny_blocks",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [16, 8, 32, 4, 64, 16, 8, 32, 4, 64] * 3  # Uneven tiny blocks
        },
        {
            "name": "power_of_2_progression",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [512, 256, 128, 64, 32, 16, 8, 4, 2, 2]  # Powers of 2
        },
        
        # ===== PERFORMANCE STRESS TESTS =====
        {
            "name": "stress_very_long_seq",
            "B": 1, "H": 8, "L": 4096, "D": 64,
            "block_sizes": [256] * 16  # Very long sequences
        },
        {
            "name": "stress_many_heads",
            "B": 1, "H": 128, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Many attention heads
        },
        {
            "name": "stress_large_batch",
            "B": 16, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8  # Large batch size
        },
        {
            "name": "stress_wide_heads",
            "B": 1, "H": 16, "L": 512, "D": 256,
            "block_sizes": [64] * 8  # Very wide attention heads
        }
    ]
    
    block_results = []
    
    for config in block_configs:
        try:
            B, H, L, D = config["B"], config["H"], config["L"], config["D"]
            block_sizes = config["block_sizes"]
            
            # Calculate sparsity
            total_elements = L * L
            masked_elements = sum(bs * bs for bs in block_sizes)
            sparsity = 1.0 - (masked_elements / total_elements)
            
            # Use our rigorous block-diagonal benchmarking
            time_spda, time_evolved, correctness_ok = bench_block_diagonal_shape(
                evolved_fn, B, H, L, D, block_sizes, dtype="float16"
            )
            
            # Calculate results
            speedup = time_spda / time_evolved if time_evolved > 0 else 0.0
            
            # Determine status based on objective performance criteria
            if not correctness_ok:
                status = "❌ WRONG"
                color = "\033[91m"  # Red
            elif speedup >= 1.5:  # Significant speedup
                status = "✅ GOOD"
                color = "\033[92m"  # Green
            elif speedup >= 1.1:  # Modest speedup
                status = "⚡ OK"
                color = "\033[93m"  # Yellow
            else:  # No meaningful improvement
                status = "❌ SLOW"
                color = "\033[91m"  # Red
            reset = "\033[0m"
            
            shape_str = f"{B}x{H}x{L}x{D}"
            blocks_str = f"{len(block_sizes)}blks"
            
            print(f"{color}{config['name']:<20}{reset} | {shape_str:<12} | {blocks_str:<6} | "
                  f"{sparsity*100:5.1f}% | {time_evolved*1000:6.1f}ms | {time_spda*1000:6.1f}ms | "
                  f"{speedup:5.2f}x | {status}")
            
            block_results.append({
                "config": config["name"],
                "speedup": speedup,
                "sparsity": sparsity,
                "status": status,
                "time_evolved": time_evolved,
                "time_spda": time_spda,
                "correctness_ok": correctness_ok
            })
            
        except Exception as e:
            print(f"{config['name']:<20} | ERROR: {str(e)}")
            block_results.append({
                "config": config["name"],
                "speedup": 0.0,
                "error": str(e)
            })
    
    return block_results


def print_comprehensive_summary(official_results, block_results):
    """Print comprehensive summary of all benchmark results"""
    print("\n" + "=" * 80)
    print("🏆 COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Official SPDA benchmark summary
    if official_results:
        official_speedups = [r["speedup"] for r in official_results if "speedup" in r]
        if official_speedups:
            print(f"\n📊 OFFICIAL SPDA BENCHMARK RESULTS:")
            print(f"   Tests run: {len(official_speedups)}")
            print(f"   Average speedup: {np.mean(official_speedups):.2f}x")
            print(f"   Median speedup: {np.median(official_speedups):.2f}x")
            print(f"   Best speedup: {max(official_speedups):.2f}x")
            print(f"   Worst speedup: {min(official_speedups):.2f}x")
            
            wins = sum(1 for s in official_speedups if s > 1.05)
            losses = sum(1 for s in official_speedups if s < 0.95)
            print(f"   Tests with >5% speedup: {wins}/{len(official_speedups)} ({wins/len(official_speedups)*100:.1f}%)")
            print(f"   Tests with >5% slowdown: {losses}/{len(official_speedups)} ({losses/len(official_speedups)*100:.1f}%)")
    
    # Block-diagonal specific summary
    if block_results:
        block_speedups = [r["speedup"] for r in block_results if "speedup" in r and r["speedup"] > 0]
        correct_results = [r for r in block_results if r.get("correctness_ok", False)]
        
        if block_speedups:
            print(f"\n🎯 BLOCK-DIAGONAL SPECIFIC RESULTS:")
            print(f"   Tests run: {len(block_speedups)}")
            print(f"   Correct results: {len(correct_results)}/{len(block_results)}")
            print(f"   Average speedup: {np.mean(block_speedups):.2f}x")
            print(f"   Median speedup: {np.median(block_speedups):.2f}x")
            print(f"   Best speedup: {max(block_speedups):.2f}x")
            print(f"   Worst speedup: {min(block_speedups):.2f}x")
            
            good_results = sum(1 for r in block_results if "✅" in r.get("status", ""))
            print(f"   Tests with significant speedups: {good_results}/{len(block_results)} ({good_results/len(block_results)*100:.1f}%)")
    
    # Overall assessment
    print(f"\n🎖️  OVERALL ASSESSMENT:")
    
    if block_results and official_results:
        avg_official_speedup = np.mean([r["speedup"] for r in official_results if "speedup" in r])
        avg_block_speedup = np.mean([r["speedup"] for r in block_results if "speedup" in r and r["speedup"] > 0])
        
        print(f"   📊 Official benchmark average: {avg_official_speedup:.2f}x")
        print(f"   🎯 Block-diagonal average: {avg_block_speedup:.2f}x")
        
        if avg_block_speedup >= 2.0:
            print("   🏆 EXCELLENT: Custom kernel significantly outperforms SPDA on block-diagonal patterns!")
        elif avg_block_speedup >= 1.5:
            print("   🥈 GOOD: Meaningful performance improvements on block-diagonal patterns.")
        elif avg_block_speedup >= 1.2:
            print("   🥉 MODERATE: Some improvements, but room for further optimization.")
        elif avg_block_speedup >= 1.0:
            print("   ⚠️  MARGINAL: Small gains, significant optimization potential remains.")
        else:
            print("   ❌ UNDERPERFORMING: Custom kernel slower than SPDA.")
    
    print(f"\n💡 TIMING METHODOLOGY:")
    print(f"   • Warmup iterations: {N_warmup}")
    print(f"   • Benchmark iterations: {N_iter_bench}")
    print(f"   • Function calls per iteration: {N_iter_func}")
    print(f"   • Nanosecond precision timing")
    print(f"   • Same as spda_benchmark.py methodology")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_evolved.py <program_path>")
        print("Example: python test_evolved.py initial_program.py")
        print("Example: python test_evolved.py openevolve_output/best/best_program.py")
        sys.exit(1)
    
    program_path = sys.argv[1]
    
    if not os.path.exists(program_path):
        print(f"❌ Error: Program file not found: {program_path}")
        sys.exit(1)

    print("🚀 COMPREHENSIVE BLOCK-DIAGONAL ATTENTION BENCHMARK")
    print(f"Program: {program_path}")
    print("="*80)

    try:
        # Load attention function
        print("Loading attention implementation...")
        evolved_fn = load_attention_function(program_path)
        print("✅ Loaded attention function")
        
        # Run official SPDA benchmark
        print("\n🔄 Running official SPDA benchmark...")
        official_results = run_official_spda_benchmark(evolved_fn)
        
        # Run block-diagonal specific tests
        print("\n🔄 Running block-diagonal specific tests...")
        block_results = run_block_diagonal_tests(evolved_fn)
        
        # Print comprehensive summary
        print_comprehensive_summary(official_results, block_results)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
