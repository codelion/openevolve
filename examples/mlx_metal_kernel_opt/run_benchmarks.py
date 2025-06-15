#!/usr/bin/env python3
"""
Qwen3 Benchmark Runner

Simple script to run baseline benchmarks for Qwen3-0.6B optimization.
Includes comparison mode to benchmark standard vs optimized attention.
"""

import argparse
import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Any

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_benchmark_suite import Qwen3BenchmarkSuite, BenchmarkResult
from quick_benchmark_test import run_quick_test


def run_compare_benchmarks(args):
    """
    Run comprehensive comparison between standard and optimized attention.
    Uses the full benchmark suite (17 comprehensive tests) for thorough analysis.
    """
    print(f"\n🔬 Running Comparison Benchmark Mode")
    print(f"📊 Comparing Standard vs OpenEvolve Optimized Attention")
    print(f"🎯 Model: {args.model}")
    print(f"📁 Output directory: {args.output_dir}")
    print("="*80)
    
    # Change to output directory
    original_dir = os.getcwd()
    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    try:
        # Run standard benchmark (baseline)
        print("\n🏃‍♂️ Phase 1: Running Standard Attention Benchmark...")
        print("⏱️  This establishes our baseline performance across all scenarios")
        print("📊 Running full benchmark suite (17 comprehensive tests)")
        print("⏳ This will take 15-30 minutes depending on your hardware...")
        
        standard_suite = Qwen3BenchmarkSuite(args.model)
        standard_results = standard_suite.run_full_benchmark_suite()
        
        print("\n✅ Standard benchmark complete!")
        
        # Apply optimized attention hook and run benchmark
        print("\n🚀 Phase 2: Running Optimized Attention Benchmark...")
        print("💡 Applying OpenEvolve optimized attention kernel")
        
        # Import and apply the optimized attention
        optimized_results = run_optimized_benchmark(args, original_dir)
        
        print("\n✅ Optimized benchmark complete!")
        
        # Generate comparison analysis
        print("\n📈 Generating Comparison Analysis...")
        comparison_results = analyze_comparison_results(
            standard_results, optimized_results, args.model
        )
        
        # Save comparison results
        save_comparison_results(comparison_results, args.output_dir)
        
        # Print detailed comparison
        print_comparison_summary(comparison_results)
        
        return 0
        
    finally:
        os.chdir(original_dir)


def run_optimized_benchmark(args, original_dir):
    """
    Run benchmark with the optimized attention from best_program.py.
    """
    try:
        # Import the optimized attention implementation
        best_program_path = os.path.join(original_dir, "openevolve_output", "best", "best_program.py")
        
        if not os.path.exists(best_program_path):
            print(f"❌ Error: Optimized program not found at {best_program_path}")
            print("Please ensure OpenEvolve has generated an optimized solution")
            return None
        
        # Import the optimized module
        import importlib.util
        spec = importlib.util.spec_from_file_location("best_program", best_program_path)
        best_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(best_program)
        
        # Apply the custom attention hook
        apply_hook, remove_hook = best_program.create_qwen3_custom_attention_hook()
        original_attention = apply_hook()
        
        if original_attention is None:
            print("❌ Failed to apply optimized attention hook")
            return None
        
        try:
            # Run benchmarks with optimized attention
            optimized_suite = Qwen3BenchmarkSuite(args.model)
            print("📊 Running full benchmark suite with optimized attention...")
            print("⏳ This will take another 15-30 minutes...")
            optimized_results = optimized_suite.run_full_benchmark_suite()
            
            return optimized_results
            
        finally:
            # Always remove the hook to restore original behavior
            remove_hook(original_attention)
            
    except Exception as e:
        print(f"❌ Error running optimized benchmark: {e}")
        return None


def analyze_comparison_results(standard_results, optimized_results, model_name):
    """
    Analyze and compare the benchmark results.
    """
    if not standard_results or not optimized_results:
        print("❌ Cannot compare - missing results")
        return None
    
    standard_benchmarks = {r['name']: r for r in standard_results['results']}
    optimized_benchmarks = {r['name']: r for r in optimized_results['results']}
    
    comparisons = []
    improvements = {
        'decode_speed_improvements': [],
        'prefill_speed_improvements': [],
        'total_speed_improvements': [],
        'memory_improvements': [],
        'time_improvements': []
    }
    
    for name in standard_benchmarks:
        if name in optimized_benchmarks:
            std_result = standard_benchmarks[name]
            opt_result = optimized_benchmarks[name]
            
            # Calculate improvements
            decode_improvement = ((opt_result['decode_tokens_per_sec'] - std_result['decode_tokens_per_sec']) 
                                / std_result['decode_tokens_per_sec'] * 100) if std_result['decode_tokens_per_sec'] > 0 else 0
            
            prefill_improvement = ((opt_result['prefill_tokens_per_sec'] - std_result['prefill_tokens_per_sec']) 
                                 / std_result['prefill_tokens_per_sec'] * 100) if std_result['prefill_tokens_per_sec'] > 0 else 0
            
            total_improvement = ((opt_result['total_tokens_per_sec'] - std_result['total_tokens_per_sec']) 
                               / std_result['total_tokens_per_sec'] * 100) if std_result['total_tokens_per_sec'] > 0 else 0
            
            memory_improvement = ((std_result['peak_memory_gb'] - opt_result['peak_memory_gb']) 
                                / std_result['peak_memory_gb'] * 100) if std_result['peak_memory_gb'] > 0 else 0
            
            time_improvement = ((std_result['total_time_sec'] - opt_result['total_time_sec']) 
                              / std_result['total_time_sec'] * 100) if std_result['total_time_sec'] > 0 else 0
            
            comparison = {
                'benchmark_name': name,
                'standard': std_result,
                'optimized': opt_result,
                'improvements': {
                    'decode_speed_pct': decode_improvement,
                    'prefill_speed_pct': prefill_improvement,
                    'total_speed_pct': total_improvement,
                    'memory_reduction_pct': memory_improvement,
                    'time_reduction_pct': time_improvement
                }
            }
            
            comparisons.append(comparison)
            
            # Collect for aggregate statistics
            improvements['decode_speed_improvements'].append(decode_improvement)
            improvements['prefill_speed_improvements'].append(prefill_improvement)
            improvements['total_speed_improvements'].append(total_improvement)
            improvements['memory_improvements'].append(memory_improvement)
            improvements['time_improvements'].append(time_improvement)
    
    # Calculate aggregate statistics
    aggregate_stats = {}
    for key, values in improvements.items():
        if values:
            aggregate_stats[f'{key}_avg'] = np.mean(values)
            aggregate_stats[f'{key}_median'] = np.median(values)
            aggregate_stats[f'{key}_min'] = np.min(values)
            aggregate_stats[f'{key}_max'] = np.max(values)
            aggregate_stats[f'{key}_std'] = np.std(values)
    
    return {
        'model': model_name,
        'timestamp': int(time.time()),
        'total_comparisons': len(comparisons),
        'individual_comparisons': comparisons,
        'aggregate_improvements': aggregate_stats,
        'summary': {
            'avg_decode_improvement_pct': aggregate_stats.get('decode_speed_improvements_avg', 0),
            'avg_total_improvement_pct': aggregate_stats.get('total_speed_improvements_avg', 0),
            'avg_memory_reduction_pct': aggregate_stats.get('memory_improvements_avg', 0),
            'avg_time_reduction_pct': aggregate_stats.get('time_improvements_avg', 0)
        }
    }


def save_comparison_results(comparison_results, output_dir):
    """
    Save detailed comparison results to files.
    """
    if not comparison_results:
        return
    
    timestamp = comparison_results['timestamp']
    
    # Save detailed JSON results
    comparison_file = f"openevolve_comparison_results_{timestamp}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Save CSV summary for easy analysis
    import csv
    csv_file = f"openevolve_comparison_summary_{timestamp}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'benchmark_name',
            'standard_decode_speed',
            'optimized_decode_speed', 
            'decode_improvement_pct',
            'standard_total_speed',
            'optimized_total_speed',
            'total_improvement_pct',
            'standard_memory_gb',
            'optimized_memory_gb',
            'memory_reduction_pct',
            'standard_time_sec',
            'optimized_time_sec',
            'time_reduction_pct'
        ])
        
        for comp in comparison_results['individual_comparisons']:
            writer.writerow([
                comp['benchmark_name'],
                comp['standard']['decode_tokens_per_sec'],
                comp['optimized']['decode_tokens_per_sec'],
                comp['improvements']['decode_speed_pct'],
                comp['standard']['total_tokens_per_sec'],
                comp['optimized']['total_tokens_per_sec'],
                comp['improvements']['total_speed_pct'],
                comp['standard']['peak_memory_gb'],
                comp['optimized']['peak_memory_gb'],
                comp['improvements']['memory_reduction_pct'],
                comp['standard']['total_time_sec'],
                comp['optimized']['total_time_sec'],
                comp['improvements']['time_reduction_pct']
            ])
    
    print(f"\n📁 Comparison results saved:")
    print(f"  📊 Detailed: {comparison_file}")
    print(f"  📈 Summary: {csv_file}")


def print_comparison_summary(comparison_results):
    """
    Print a comprehensive comparison summary.
    """
    if not comparison_results:
        print("❌ No comparison results to display")
        return
    
    print(f"\n{'='*100}")
    print(f"{'🚀 OPENEVOLVE OPTIMIZATION RESULTS':^100}")
    print(f"{'='*100}")
    
    summary = comparison_results['summary']
    total_tests = comparison_results['total_comparisons']
    
    print(f"\n🎯 OVERALL PERFORMANCE IMPROVEMENTS (across {total_tests} comprehensive tests):")
    print(f"  📈 Average Decode Speed Improvement: {summary['avg_decode_improvement_pct']:+.2f}%")
    print(f"  ⚡ Average Total Speed Improvement:  {summary['avg_total_improvement_pct']:+.2f}%")
    print(f"  💾 Average Memory Reduction:        {summary['avg_memory_reduction_pct']:+.2f}%")
    print(f"  ⏱️  Average Time Reduction:          {summary['avg_time_reduction_pct']:+.2f}%")
    
    print(f"\n📊 DETAILED BENCHMARK COMPARISON:")
    print(f"{'='*100}")
    print(f"{'Benchmark':<25} {'Standard':<12} {'Optimized':<12} {'Improvement':<12} {'Memory':<12} {'Time':<12}")
    print(f"{'Name':<25} {'Decode':<12} {'Decode':<12} {'(%)':<12} {'Reduction':<12} {'Reduction':<12}")
    print(f"{'-'*100}")
    
    for comp in comparison_results['individual_comparisons']:
        name = comp['benchmark_name'][:24]
        std_decode = comp['standard']['decode_tokens_per_sec']
        opt_decode = comp['optimized']['decode_tokens_per_sec']
        decode_imp = comp['improvements']['decode_speed_pct']
        mem_imp = comp['improvements']['memory_reduction_pct']
        time_imp = comp['improvements']['time_reduction_pct']
        
        print(f"{name:<25} {std_decode:<12.1f} {opt_decode:<12.1f} {decode_imp:+<12.1f} {mem_imp:+<12.1f} {time_imp:+<12.1f}")
    
    print(f"{'-'*100}")
    
    # Highlight best improvements
    best_decode = max(comparison_results['individual_comparisons'], 
                     key=lambda x: x['improvements']['decode_speed_pct'])
    best_memory = max(comparison_results['individual_comparisons'],
                     key=lambda x: x['improvements']['memory_reduction_pct'])
    best_time = max(comparison_results['individual_comparisons'],
                   key=lambda x: x['improvements']['time_reduction_pct'])
    
    print(f"\n🏆 BEST IMPROVEMENTS:")
    print(f"  🥇 Best Decode Speed: {best_decode['benchmark_name']} (+{best_decode['improvements']['decode_speed_pct']:.1f}%)")
    print(f"  🥇 Best Memory Reduction: {best_memory['benchmark_name']} ({best_memory['improvements']['memory_reduction_pct']:+.1f}%)")
    print(f"  🥇 Best Time Reduction: {best_time['benchmark_name']} ({best_time['improvements']['time_reduction_pct']:+.1f}%)")
    
    # Optimization analysis
    decode_improvements = [comp['improvements']['decode_speed_pct'] for comp in comparison_results['individual_comparisons']]
    positive_improvements = sum(1 for x in decode_improvements if x > 0)
    
    print(f"\n📈 OPTIMIZATION ANALYSIS:")
    print(f"  ✅ Benchmarks Improved: {positive_improvements}/{len(decode_improvements)}")
    print(f"  📊 Success Rate: {positive_improvements/len(decode_improvements)*100:.1f}%")
    
    if summary['avg_decode_improvement_pct'] > 0:
        print(f"  🎉 OpenEvolve optimization successful across all scenarios!")
        print(f"  💡 Average {summary['avg_decode_improvement_pct']:.1f}% improvement in decode speed")
        if summary['avg_decode_improvement_pct'] > 10:
            print(f"  🚀 Excellent optimization results - significant performance gains!")
        elif summary['avg_decode_improvement_pct'] > 5:
            print(f"  📈 Good optimization results - meaningful performance improvements")
        else:
            print(f"  📊 Modest optimization results - room for further improvement")
    else:
        print(f"  ⚠️  Optimization needs further tuning")
        print(f"  🔧 Consider running additional evolution cycles")
    
    # Memory analysis
    if summary['avg_memory_reduction_pct'] > 0:
        print(f"  💾 Memory efficiency improved by {summary['avg_memory_reduction_pct']:.1f}% on average")
    
    print(f"\n{'='*100}")
    print(f"🔬 Analysis complete! Results saved to comparison files.")
    print(f"💡 Use these insights to guide further OpenEvolve optimization cycles.")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-0.6B benchmarks")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "compare"],
        default="quick",
        help="Benchmark mode: quick (4 tests), full (17 tests), or compare (standard vs optimized)",
    )
    parser.add_argument(
        "--model", default="mlx-community/Qwen3-0.6B-bf16", help="Model path or name"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for results")

    args = parser.parse_args()

    print(f"Running {args.mode} benchmark for {args.model}")
    print(f"Output directory: {args.output_dir}")

    if args.mode == "quick":
        print("\n🚀 Running Quick Benchmark (4 key tests)...")
        results = run_quick_test()
        print("\n✅ Quick benchmark complete!")

    elif args.mode == "compare":
        return run_compare_benchmarks(args)

    else:  # full
        print("\n🚀 Running Full Benchmark Suite (17 comprehensive tests)...")
        print("⏱️  This may take 15-30 minutes depending on your hardware...")

        # Change to output directory
        original_dir = os.getcwd()
        if args.output_dir != ".":
            os.makedirs(args.output_dir, exist_ok=True)
            os.chdir(args.output_dir)

        try:
            benchmark_suite = Qwen3BenchmarkSuite(args.model)
            results = benchmark_suite.run_full_benchmark_suite()
            benchmark_suite.print_summary_table()

            print("\n✅ Full benchmark suite complete!")
            print(f"📊 Results saved in: {args.output_dir}")

        finally:
            os.chdir(original_dir)

    if args.mode != "compare":
        print("\n🎯 These results establish the baseline for kernel optimization.")
        print("🔧 Next step: Create evolved Metal kernel to improve performance!")
        print("💡 Run with --mode compare to benchmark against OpenEvolve optimizations!")
        print("📚 Install mlx-lm with: pip install mlx-lm")

    return 0


if __name__ == "__main__":
    sys.exit(main())
