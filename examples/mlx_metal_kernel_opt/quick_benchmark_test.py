"""
Quick Benchmark Test - Test the benchmark suite with a few key scenarios
"""

import os
import sys
sys.path.append('/Users/asankhaya/Documents/GitHub/openevolve/examples/mlx_metal_kernel_opt')

from qwen3_benchmark_suite import Qwen3BenchmarkSuite, BenchmarkConfig

def run_quick_test():
    """Run a quick test with just a few key benchmarks"""
    
    # Test configs - subset of full suite
    test_configs = [
        BenchmarkConfig(
            name="baseline_test",
            prompt="The future of AI is",
            max_tokens=100,
            description="Baseline test matching your original benchmark"
        ),
        BenchmarkConfig(
            name="short_context_quick",
            prompt="Brief answer: What is artificial intelligence?",
            max_tokens=50,
            description="Short context, quick response"
        ),
        BenchmarkConfig(
            name="code_generation_test",
            prompt="Write a Python function to implement binary search:",
            max_tokens=200,
            description="Code generation test"
        ),
        BenchmarkConfig(
            name="long_generation_test",
            prompt="Explain in detail how neural networks learn:",
            max_tokens=500,
            description="Longer generation test"
        ),
    ]
    
    # Change to mlx-lm directory
    original_dir = os.getcwd()
    mlx_lm_dir = "/Users/asankhaya/Documents/GitHub/mlx-lm"
    
    if os.path.exists(mlx_lm_dir):
        os.chdir(mlx_lm_dir)
        print(f"Changed to mlx-lm directory: {mlx_lm_dir}")
    else:
        print(f"Error: mlx-lm directory not found at {mlx_lm_dir}")
        return
    
    try:
        benchmark_suite = Qwen3BenchmarkSuite()
        
        print(f"\n{'='*80}")
        print(f"Quick Benchmark Test - Qwen3-0.6B")
        print(f"Testing {len(test_configs)} key scenarios")
        print(f"{'='*80}")
        
        results = []
        for i, config in enumerate(test_configs, 1):
            print(f"\n[{i}/{len(test_configs)}] Running: {config.name}")
            try:
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
            print(f"{'Name':<20} {'Gen Tokens':<12} {'Decode Speed':<12} {'Memory':<10}")
            print(f"{'-'*80}")
            
            for result in results:
                print(f"{result.name:<20} "
                      f"{result.generated_tokens:<12} "
                      f"{result.decode_tokens_per_sec:<12.1f} "
                      f"{result.peak_memory_gb:<10.2f}")
            
            print(f"{'-'*80}")
            decode_speeds = [r.decode_tokens_per_sec for r in results if r.decode_tokens_per_sec > 0]
            if decode_speeds:
                import numpy as np
                print(f"Average decode speed: {np.mean(decode_speeds):.1f} tokens/sec")
                print(f"Speed range: {np.min(decode_speeds):.1f} - {np.max(decode_speeds):.1f} tokens/sec")
        
        print(f"\n{'='*80}")
        print("Quick test complete! If this looks good, run the full benchmark suite.")
        print("python qwen3_benchmark_suite.py")
        print(f"{'='*80}")
        
        return results
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    run_quick_test()
