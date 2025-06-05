#!/usr/bin/env python3
"""
Comprehensive benchmark for evolved block-diagonal attention implementations

This script runs both:
1. Official SPDA benchmark tests (using exact same methodology as spda_benchmark.py) 
2. Block-diagonal specific tests where our custom kernel should excel

Usage: python test_evolved.py <program_path>
Example: python test_evolved.py initial_program.py
Example: python test_evolved.py openevolve_output/best/best_program.py
"""

import argparse
import importlib.util
import math
import os
import sys
import time
import gc
from typing import Optional

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX or NumPy not available")
    MLX_AVAILABLE = False
    sys.exit(1)

# Import the official SPDA benchmark
try:
    import spda_benchmark
    SPDA_BENCHMARK_AVAILABLE = True
except ImportError:
    print("⚠️ spda_benchmark.py not found")
    SPDA_BENCHMARK_AVAILABLE = False
    sys.exit(1)


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


def run_official_spda_benchmark(evolved_fn):
    """Run the official SPDA benchmark tests with our evolved function using exact same methodology"""
    print("\n" + "=" * 80)
    print("📊 OFFICIAL SPDA BENCHMARK TESTS")
    print("=" * 80)
    print("Testing evolved attention vs mx.fast.scaled_dot_product_attention")
    print("Using EXACT same methodology as spda_benchmark.py")
    print("Format: B, qsl, ksl, hdim, n_qh, n_kvh, t, dtype, mask, t_fused, t_evolved, diff%")
    print("-" * 80)
    
    # Temporarily replace the reference function in spda_benchmark
    original_mlx_ref_attn = spda_benchmark.mlx_ref_attn
    spda_benchmark.mlx_ref_attn = evolved_fn
    
    # Get official test configurations - EXACT same as spda_benchmark.py
    dtypes = ("float16",)  # Focus on float16 like the official benchmark
    transposes = (False,)
    
    # EXACT same shapes from spda_benchmark.py
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
    masks = [None, "bool", "causal"]  # EXACT same as official
    
    official_results = []
    
    try:
        for dtype in dtypes:
            for transpose in transposes:
                for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in shapes:
                    for mask_in in masks:
                        try:
                            # Use the EXACT same bench_shape function from spda_benchmark.py
                            time_mlx_fused, time_mlx_evolved = spda_benchmark.bench_shape(
                                B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype, transpose, mask_in
                            )
                            
                            # Calculate performance difference
                            diff = time_mlx_evolved / time_mlx_fused - 1.0
                            speedup = time_mlx_fused / time_mlx_evolved if time_mlx_evolved > 0 else 0.0
                            
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
                                f"{time_mlx_fused:6.3f}, {time_mlx_evolved:6.3f},{100. * diff:+6.2f}% "
                                f"(speedup: {speedup:.2f}x){reset_color}"
                            )
                            
                            official_results.append({
                                "config": f"{qsl}x{head_dim}_{mask_in}",
                                "speedup": speedup,
                                "diff_pct": diff * 100,
                                "time_fused": time_mlx_fused,
                                "time_evolved": time_mlx_evolved
                            })
                            
                        except Exception as e:
                            print(f"FAILED: {B}, {qsl}, {ksl}, {head_dim}, {n_q_heads}, {n_kv_heads}, "
                                  f"{dtype}, {mask_in} - {str(e)}")
                            
    finally:
        # Restore original function
        spda_benchmark.mlx_ref_attn = original_mlx_ref_attn
    
    return official_results


def run_block_diagonal_tests(evolved_fn):
    """Run block-diagonal specific tests where our kernel should excel"""
    print("\n" + "=" * 80)
    print("🎯 BLOCK-DIAGONAL SPECIFIC TESTS")
    print("=" * 80)
    print("Testing scenarios where block-diagonal attention should outperform SPDA")
    print("Using same timing methodology as official benchmark")
    print("Format: Test | Shape | Blocks | Sparsity | Evolved | SPDA | Speedup | Status")
    print("-" * 80)
    
    # Use EXACT same timing constants as spda_benchmark.py
    N_warmup = spda_benchmark.N_warmup      # 5
    N_iter_bench = spda_benchmark.N_iter_bench  # 40  
    N_iter_func = spda_benchmark.N_iter_func    # 8
    
    def bench_custom(f, *args):
        """Use exact same bench function as spda_benchmark.py"""
        for i in range(N_warmup):
            f(*args)

        s = time.perf_counter_ns()
        for i in range(N_iter_bench):
            f(*args)
        e = time.perf_counter_ns()
        return (e - s) * 1e-9
    
    def do_attention_bench_custom(f, q, k, v, scale, mask=None):
        """Use exact same attention bench pattern as spda_benchmark.py"""
        q_out = q
        for i in range(N_iter_func):
            q_out = f(q_out, k, v, scale=scale, mask=mask)
        mx.eval(q_out)
        return q_out
    
    # Block-diagonal test configurations
    block_configs = [
        {
            "name": "packed_2x256_sparse50",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256],  # 50% sparse
            "expected_speedup": 1.2
        },
        {
            "name": "packed_4x128_sparse75", 
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],  # 75% sparse
            "expected_speedup": 1.5
        },
        {
            "name": "packed_8x128_sparse87",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128] * 8,  # 87.5% sparse
            "expected_speedup": 2.0
        },
        {
            "name": "packed_16x64_sparse93",
            "B": 1, "H": 16, "L": 1024, "D": 128,
            "block_sizes": [64] * 16,  # 93.75% sparse
            "expected_speedup": 3.0
        },
        {
            "name": "bert_style_packing",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],  # BERT-style
            "expected_speedup": 1.3
        },
        {
            "name": "large_seq_sparse",
            "B": 1, "H": 32, "L": 2048, "D": 64, 
            "block_sizes": [256] * 8,  # Large sequence, 87.5% sparse
            "expected_speedup": 2.5
        }
    ]
    
    block_results = []
    
    for config in block_configs:
        try:
            B, H, L, D = config["B"], config["H"], config["L"], config["D"]
            
            # Create test inputs using SAME method as spda_benchmark
            dtype = "float16"
            scale = 1.0 / math.sqrt(D)
            
            # Use spda_benchmark's input preparation
            q, k, v, _, _ = spda_benchmark.prepare_inputs(B, L, L, D, H, H, None, False, dtype)
            
            # Create block-diagonal mask
            mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
            
            # Calculate sparsity
            total_elements = L * L
            masked_elements = sum(bs * bs for bs in config["block_sizes"])
            sparsity = 1.0 - (masked_elements / total_elements)
            
            # Benchmark evolved implementation using EXACT same methodology
            evolved_time = bench_custom(do_attention_bench_custom, evolved_fn, q, k, v, scale, mask)
            
            # Benchmark SPDA using EXACT same methodology
            spda_time = bench_custom(do_attention_bench_custom, mx.fast.scaled_dot_product_attention, q, k, v, scale, mask)
            
            # Calculate results
            speedup = spda_time / evolved_time if evolved_time > 0 else 0.0
            expected = config["expected_speedup"]
            
            # Determine status
            if speedup >= expected * 0.8:  # Within 80% of expected
                status = "✅ GOOD"
                color = "\033[92m"  # Green
            elif speedup >= 1.1:
                status = "⚡ OK"
                color = "\033[93m"  # Yellow
            else:
                status = "❌ SLOW"
                color = "\033[91m"  # Red
            reset = "\033[0m"
            
            shape_str = f"{B}x{H}x{L}x{D}"
            blocks_str = f"{len(config['block_sizes'])}blks"
            
            print(f"{color}{config['name']:<20}{reset} | {shape_str:<12} | {blocks_str:<6} | "
                  f"{sparsity*100:5.1f}% | {evolved_time*1000:6.1f}ms | {spda_time*1000:6.1f}ms | "
                  f"{speedup:5.2f}x | {status}")
            
            block_results.append({
                "config": config["name"],
                "speedup": speedup,
                "expected": expected,
                "sparsity": sparsity,
                "status": status,
                "time_evolved": evolved_time,
                "time_spda": spda_time
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
        if block_speedups:
            print(f"\n🎯 BLOCK-DIAGONAL SPECIFIC RESULTS:")
            print(f"   Tests run: {len(block_speedups)}")
            print(f"   Average speedup: {np.mean(block_speedups):.2f}x")
            print(f"   Median speedup: {np.median(block_speedups):.2f}x")
            print(f"   Best speedup: {max(block_speedups):.2f}x")
            print(f"   Worst speedup: {min(block_speedups):.2f}x")
            
            good_results = sum(1 for r in block_results if "✅" in r.get("status", ""))
            print(f"   Tests meeting expectations: {good_results}/{len(block_results)} ({good_results/len(block_results)*100:.1f}%)")
    
    # Overall assessment
    print(f"\n🎖️  OVERALL ASSESSMENT:")
    
    if block_results and official_results:
        avg_official_speedup = np.mean([r["speedup"] for r in official_results if "speedup" in r])
        avg_block_speedup = np.mean([r["speedup"] for r in block_results if "speedup" in r and r["speedup"] > 0])
        
        print(f"   📊 Official benchmark average: {avg_official_speedup:.2f}x")
        print(f"   🎯 Block-diagonal average: {avg_block_speedup:.2f}x")
        
        if avg_block_speedup >= 2.0:
            print("   🏆 EXCELLENT: Custom kernel significantly outperforms SPDA on block-diagonal patterns!")
            print("   🚀 Evolution successfully discovered optimizations for sparse attention patterns.")
        elif avg_block_speedup >= 1.5:
            print("   🥈 GOOD: Meaningful performance improvements on block-diagonal patterns.")
            print("   ⚡ Custom kernel shows clear advantage over SPDA for sparse patterns.")
        elif avg_block_speedup >= 1.2:
            print("   🥉 MODERATE: Some improvements, but room for further optimization.")
            print("   🔧 Kernel needs more work to fully exploit block-diagonal sparsity.")
        elif avg_block_speedup >= 1.0:
            print("   ⚠️  MARGINAL: Small gains, significant optimization potential remains.")
        else:
            print("   ❌ UNDERPERFORMING: Custom kernel slower than SPDA.")
    
    print(f"\n💡 TIMING METHODOLOGY:")
    print(f"   • Same warmup/iteration counts as official benchmark")
    print(f"   • Same input preparation and chaining patterns")
    print(f"   • Nanosecond precision timing")
    print(f"   • Results should match spda_benchmark.py when using SPDA")


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
