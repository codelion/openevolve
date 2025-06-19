"""
Quick Benchmark Test - Test the benchmark suite with a few key scenarios
"""

import os
import sys

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_benchmark_suite import Qwen3BenchmarkSuite, BenchmarkConfig


def run_quick_test():
    """Run a quick test with just a few key benchmarks with proper warmup"""

    # Test configs - subset of full suite
    test_configs = [
        BenchmarkConfig(
            name="baseline_test",
            prompt="The future of AI is",
            max_tokens=100,
            description="Baseline test matching your original benchmark",
        ),
        BenchmarkConfig(
            name="short_context_quick",
            prompt="Brief answer: What is artificial intelligence?",
            max_tokens=50,
            description="Short context, quick response",
        ),
        BenchmarkConfig(
            name="code_generation_test",
            prompt="Write a Python function to implement binary search:",
            max_tokens=200,
            description="Code generation test",
        ),
        BenchmarkConfig(
            name="long_generation_test",
            prompt="Explain in detail how neural networks learn:",
            max_tokens=500,
            description="Longer generation test",
        ),
        BenchmarkConfig(
            name="memory_efficiency_test",
            prompt="Write a comprehensive guide on optimizing memory usage in large-scale machine learning systems, covering techniques for both training and inference:",
            max_tokens=800,
            description="Memory efficiency stress test",
        ),
    ]

    # Use mlx-lm as installed package (no need to change directories)
    try:
        # Import mlx for cache clearing
        import mlx.core as mx
        import numpy as np

        benchmark_suite = Qwen3BenchmarkSuite()

        print(f"\n{'='*80}")
        print(f"Quick Benchmark Test - Qwen3-0.6B")
        print(f"Testing {len(test_configs)} key scenarios with warmup")
        print(f"Purpose: Validate Metal kernel optimization baseline")
        print(f"{'='*80}")

        # Global warmup - run one quick test to warm up the system
        print(f"🔥 Running global warmup to initialize MLX and model...")
        try:
            mx.clear_cache()
            warmup_config = BenchmarkConfig(
                name="warmup", prompt="Hello", max_tokens=5, description="Warmup run"
            )
            print(f"   Global warmup in progress...")
            warmup_result = benchmark_suite.run_single_benchmark(warmup_config)
            print(f"   ✅ Global warmup completed")
        except Exception as e:
            print(f"   ⚠️  Global warmup failed: {e}")
            print(f"   Continuing with individual tests...")

        results = []
        for i, config in enumerate(test_configs, 1):
            print(f"\n[{i}/{len(test_configs)}] Running: {config.name}")
            try:
                # The benchmark_suite.run_single_benchmark already has warmup built-in
                result = benchmark_suite.run_single_benchmark(config)
                results.append(result)
            except Exception as e:
                print(f"Failed: {e}")
                continue

        # Print summary
        if results:
            print(f"\n{'='*80}")
            print(f"Quick Test Results Summary")
            print(f"{'='*80}")
            print(f"{'Name':<25} {'Gen Tokens':<12} {'Decode Speed':<15} {'Memory':<10} {'CV%':<8}")
            print(f"{'-'*80}")

            for result in results:
                # Extract standard deviation from the result display if available
                cv_display = "N/A"
                print(
                    f"{result.name:<25} "
                    f"{result.generated_tokens:<12} "
                    f"{result.decode_tokens_per_sec:<15.1f} "
                    f"{result.peak_memory_gb:<10.2f} "
                    f"{cv_display:<8}"
                )

            print(f"{'-'*80}")
            decode_speeds = [
                r.decode_tokens_per_sec for r in results if r.decode_tokens_per_sec > 0
            ]
            if decode_speeds:
                print(f"Average decode speed: {np.mean(decode_speeds):.1f} tokens/sec")
                print(
                    f"Speed range: {np.min(decode_speeds):.1f} - {np.max(decode_speeds):.1f} tokens/sec"
                )
                print(f"Performance std dev: {np.std(decode_speeds):.1f} tokens/sec")
                print(
                    f"Overall consistency: {np.std(decode_speeds)/np.mean(decode_speeds)*100:.1f}% CV"
                )

        print(f"\n{'='*80}")
        print("Quick test complete! If this looks good, run the full benchmark suite.")
        print("Full suite: python qwen3_benchmark_suite.py")
        print("Compare mode: python run_benchmarks.py --mode compare")
        print(f"✅ All tests included proper warmup for reliable results")
        print(f"🎯 Ready to test custom Metal kernel optimization!")
        print(f"{'='*80}")

        return results

    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return None


if __name__ == "__main__":
    run_quick_test()
