#!/usr/bin/env python3
"""
Qwen3 Benchmark Runner

Simple script to run baseline benchmarks for Qwen3-0.6B optimization.
"""

import argparse
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_benchmark_suite import Qwen3BenchmarkSuite
from quick_benchmark_test import run_quick_test

def main():
    parser = argparse.ArgumentParser(description='Run Qwen3-0.6B benchmarks')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Benchmark mode: quick (4 tests) or full (17 tests)')
    parser.add_argument('--model', default='mlx-community/Qwen3-0.6B-bf16',
                       help='Model path or name')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"Running {args.mode} benchmark for {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    if args.mode == 'quick':
        print("\n🚀 Running Quick Benchmark (4 key tests)...")
        results = run_quick_test()
        print("\n✅ Quick benchmark complete!")
        
    else:  # full
        print("\n🚀 Running Full Benchmark Suite (17 comprehensive tests)...")
        print("⏱️  This may take 15-30 minutes depending on your hardware...")
        
        # Change to output directory
        original_dir = os.getcwd()
        if args.output_dir != '.':
            os.makedirs(args.output_dir, exist_ok=True)
            os.chdir(args.output_dir)
        
        try:
            # Change to mlx-lm directory for running
            mlx_lm_dir = "/Users/asankhaya/Documents/GitHub/mlx-lm"
            if os.path.exists(mlx_lm_dir):
                os.chdir(mlx_lm_dir)
                
                benchmark_suite = Qwen3BenchmarkSuite(args.model)
                results = benchmark_suite.run_full_benchmark_suite()
                benchmark_suite.print_summary_table()
                
                print("\n✅ Full benchmark suite complete!")
                print(f"📊 Results saved in: {args.output_dir}")
                
            else:
                print(f"❌ Error: mlx-lm directory not found at {mlx_lm_dir}")
                print("Please ensure mlx-lm is installed and accessible")
                return 1
                
        finally:
            os.chdir(original_dir)
    
    print("\n🎯 These results establish the baseline for kernel optimization.")
    print("🔧 Next step: Create evolved Metal kernel to improve performance!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
