#!/usr/bin/env python3
"""
Test the best evolved attention implementation against the full spda_benchmark.py

This script loads the evolved attention function and runs it through the complete
benchmark suite to compare performance against mlx_fused_attn.
"""

import argparse
import importlib.util
import os
import sys
from typing import Optional

import mlx.core as mx

# Import the benchmark
import spda_benchmark


def load_evolved_attention(program_path: str):
    """Load the evolved attention function from the best program"""
    if not os.path.exists(program_path):
        raise FileNotFoundError(f"Program file not found: {program_path}")

    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    evolved_program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evolved_program)

    if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
        raise AttributeError("Program missing evolved_scaled_dot_product_attention function")

    return evolved_program.evolved_scaled_dot_product_attention


def patch_benchmark_with_evolved_attention(evolved_attention_fn):
    """Replace mlx_ref_attn in the benchmark with our evolved version"""
    # Store original for comparison
    original_mlx_ref_attn = spda_benchmark.mlx_ref_attn

    # Replace with evolved version
    spda_benchmark.mlx_ref_attn = evolved_attention_fn

    return original_mlx_ref_attn


def run_full_benchmark(evolved_program_path: str, subset: bool = False):
    """
    Run the full benchmark comparing evolved attention vs fused attention
    """

    print("Loading evolved attention implementation...")
    evolved_attention_fn = load_evolved_attention(evolved_program_path)
    print("✓ Loaded evolved attention function")

    print("\nPatching benchmark to use evolved attention...")
    original_ref_attn = patch_benchmark_with_evolved_attention(evolved_attention_fn)
    print("✓ Benchmark patched")

    try:
        # Define test configurations
        dtypes = ("float16",)  # Focus on float16 as it's most common
        transposes = (False,)  # Standard layout

        if subset:
            # Smaller subset for quick testing
            shapes = [
                (1, 128, 128, 64, 16, 16),
                (1, 256, 256, 64, 16, 16),
                (1, 512, 512, 64, 32, 8),  # GQA case
                (1, 1024, 1024, 64, 32, 8),  # Larger GQA
            ]
            masks = [None, "causal"]
        else:
            # Full benchmark suite
            shapes_64 = [
                (1, 32, 32, 64, 32, 32),
                (1, 64, 64, 64, 32, 32),
                (1, 128, 128, 64, 32, 32),
                (1, 256, 256, 64, 32, 32),
                (1, 512, 512, 64, 32, 32),
                (1, 1024, 1024, 64, 32, 8),
                (1, 2048, 2048, 64, 32, 8),
                (1, 4096, 4096, 64, 32, 8),
            ]

            shapes_80 = [
                (1, 1024, 1024, 80, 32, 8),
                (1, 2048, 2048, 80, 32, 8),
                (1, 4096, 4096, 80, 32, 8),
            ]

            shapes_128 = [
                (1, 1024, 1024, 128, 32, 8),
                (1, 2048, 2048, 128, 32, 8),
                (1, 4096, 4096, 128, 32, 8),
            ]

            shapes = shapes_64 + shapes_80 + shapes_128
            masks = [None, "bool", "causal"]

        print(
            f"\nRunning benchmark with {len(shapes)} shapes x {len(masks)} masks = {len(shapes) * len(masks)} total tests"
        )
        print("Format: B, qsl, ksl, hdim, n_qh, n_kvh, t, dtype, mask, t_fused, t_evolved, diff%")
        print("=" * 90)

        total_tests = 0
        successful_tests = 0
        speedups = []

        for dtype in dtypes:
            for transpose in transposes:
                for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in shapes:
                    for mask_in in masks:
                        total_tests += 1

                        try:
                            # Run benchmark (evolved vs fused)
                            time_mlx_fused, time_mlx_evolved = spda_benchmark.bench_shape(
                                B,
                                qsl,
                                ksl,
                                head_dim,
                                n_q_heads,
                                n_kv_heads,
                                dtype,
                                transpose,
                                mask_in,
                            )

                            # Calculate performance difference
                            diff = time_mlx_evolved / time_mlx_fused - 1.0
                            speedup = (
                                time_mlx_fused / time_mlx_evolved if time_mlx_evolved > 0 else 0.0
                            )
                            speedups.append(speedup)
                            successful_tests += 1

                            t_str = 1 if transpose else 0

                            # Color coding: green for speedup, red for slowdown
                            if diff < -0.05:  # >5% speedup
                                color = "\033[92m"  # Green
                            elif diff > 0.05:  # >5% slowdown
                                color = "\033[91m"  # Red
                            else:
                                color = "\033[93m"  # Yellow
                            reset_color = "\033[0m"

                            print(
                                f"{color}{B:3d}, {qsl:5d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, "
                                f"{n_kv_heads:5d}, {t_str:1d}, {dtype}, {str(mask_in):>8}, "
                                f"{time_mlx_fused:6.3f}, {time_mlx_evolved:6.3f},{100. * diff:+6.2f}% "
                                f"(speedup: {speedup:.2f}x){reset_color}"
                            )

                        except Exception as e:
                            print(
                                f"FAILED: {B}, {qsl}, {ksl}, {head_dim}, {n_q_heads}, {n_kv_heads}, "
                                f"{dtype}, {mask_in} - {str(e)}"
                            )

        print("=" * 90)
        print(f"\nBenchmark Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful tests: {successful_tests}")
        print(f"  Success rate: {successful_tests/total_tests*100:.1f}%")

        if speedups:
            import numpy as np

            speedups = np.array(speedups)
            print(f"  Average speedup: {np.mean(speedups):.2f}x")
            print(f"  Median speedup: {np.median(speedups):.2f}x")
            print(f"  Best speedup: {np.max(speedups):.2f}x")
            print(f"  Worst speedup: {np.min(speedups):.2f}x")
            print(
                f"  Tests with speedup > 1.1x: {np.sum(speedups > 1.1)} ({np.sum(speedups > 1.1)/len(speedups)*100:.1f}%)"
            )
            print(
                f"  Tests with speedup > 1.2x: {np.sum(speedups > 1.2)} ({np.sum(speedups > 1.2)/len(speedups)*100:.1f}%)"
            )

            if np.mean(speedups) > 1.1:
                print(
                    f"\n🎉 SUCCESS: Evolved attention achieves {np.mean(speedups):.2f}x average speedup!"
                )
            elif np.mean(speedups) > 1.0:
                print(
                    f"\n✅ GOOD: Evolved attention achieves {np.mean(speedups):.2f}x average speedup"
                )
            else:
                print(
                    f"\n⚠️  SLOW: Evolved attention is {1/np.mean(speedups):.2f}x slower on average"
                )

    finally:
        # Restore original benchmark function
        spda_benchmark.mlx_ref_attn = original_ref_attn
        print(f"\n✓ Benchmark restored to original state")


def main():
    parser = argparse.ArgumentParser(description="Test evolved attention against full benchmark")
    parser.add_argument("program_path", help="Path to the evolved program file")
    parser.add_argument(
        "--subset", action="store_true", help="Run subset of tests for quick validation"
    )
    parser.add_argument("--output", help="Save results to file")

    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"Error: Program file not found: {args.program_path}")
        sys.exit(1)

    try:
        if args.output:
            # Redirect output to file
            import contextlib

            with open(args.output, "w") as f:
                with contextlib.redirect_stdout(f):
                    run_full_benchmark(args.program_path, args.subset)
            print(f"Results saved to {args.output}")
        else:
            run_full_benchmark(args.program_path, args.subset)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
