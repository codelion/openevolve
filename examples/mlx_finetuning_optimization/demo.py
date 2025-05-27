#!/usr/bin/env python3
"""
MLX Fine-tuning Optimization Demo

This script demonstrates how to use the evolved MLX optimization patterns
to improve fine-tuning performance on Apple Silicon.

Usage:
    python demo.py --baseline      # Run baseline only
    python demo.py --optimized     # Run optimized only  
    python demo.py --compare       # Compare baseline vs optimized
    python demo.py --evolve        # Run OpenEvolve to discover new patterns
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add the directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from baseline_finetuning import BaselineTrainer
from mlx_optimization_patch import (
    apply_optimizations, 
    benchmark_optimization_improvement,
    mlx_optimizations,
    create_optimized_trainer
)


def run_baseline(num_samples: int = 200, output_dir: str = "./demo_baseline"):
    """Run baseline MLX fine-tuning"""
    print("🔧 Running Baseline MLX Fine-tuning")
    print("=" * 50)
    
    trainer = BaselineTrainer("mlx-community/Qwen3-0.6B-bf16")
    trainer.config.batch_size = 2
    trainer.config.num_epochs = 1
    
    print(f"Creating {num_samples} training samples...")
    dataset = trainer.create_sample_dataset(num_samples)
    
    print("Starting baseline training...")
    start_time = time.time()
    results = trainer.train(dataset, output_dir)
    total_time = time.time() - start_time
    
    print(f"\n✅ Baseline Training Complete in {total_time:.2f}s")
    print(f"📊 Results:")
    print(f"  Tokens/sec: {results['tokens_per_second']:.1f}")
    print(f"  Peak memory: {results['peak_memory_mb']:.1f} MB")
    print(f"  Memory efficiency: {results['memory_efficiency']:.4f}")
    print(f"  Final loss: {results['final_loss']:.4f}")
    
    return results


def run_optimized(num_samples: int = 200, output_dir: str = "./demo_optimized"):
    """Run optimized MLX fine-tuning"""
    print("⚡ Running Optimized MLX Fine-tuning")
    print("=" * 50)
    
    try:
        # Create trainer with automatic optimization loading
        trainer = create_optimized_trainer("mlx-community/Qwen3-0.6B-bf16")
        trainer.config.batch_size = 2
        trainer.config.num_epochs = 1
    except Exception as e:
        print(f"⚠️  Failed to create optimized trainer: {e}")
        print("Falling back to baseline with default optimizations...")
        trainer = BaselineTrainer("mlx-community/Qwen3-0.6B-bf16")
        trainer.config.batch_size = 2
        trainer.config.num_epochs = 1
        # Try to apply any available optimizations
        try:
            apply_optimizations(trainer)
            print("✅ Applied optimizations to baseline trainer")
        except Exception as opt_error:
            print(f"⚠️  Could not apply optimizations: {opt_error}")
            print("Using baseline trainer without optimizations")
    
    print(f"Creating {num_samples} training samples...")
    dataset = trainer.create_sample_dataset(num_samples)
    
    print("Starting optimized training...")
    start_time = time.time()
    results = trainer.train(dataset, output_dir)
    total_time = time.time() - start_time
    
    print(f"\n✅ Optimized Training Complete in {total_time:.2f}s")
    print(f"📊 Results:")
    print(f"  Tokens/sec: {results['tokens_per_second']:.1f}")
    print(f"  Peak memory: {results['peak_memory_mb']:.1f} MB")
    print(f"  Memory efficiency: {results['memory_efficiency']:.4f}")
    print(f"  Final loss: {results['final_loss']:.4f}")
    
    return results


def compare_performance(num_samples: int = 200):
    """Compare baseline vs optimized performance"""
    print("🏁 Comparing Baseline vs Optimized Performance")
    print("=" * 50)
    
    print("Running comprehensive benchmark...")
    results = benchmark_optimization_improvement(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        num_samples=num_samples
    )
    
    baseline = results["baseline"]
    optimized = results["optimized"]
    improvements = results["improvements"]
    
    print(f"\n📈 Performance Comparison")
    print(f"{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 70)
    
    metrics = [
        ("Tokens/sec", "tokens_per_second", "{:.1f}"),
        ("Peak Memory (MB)", "peak_memory_mb", "{:.1f}"),
        ("Memory Efficiency", "memory_efficiency", "{:.4f}"),
        ("Total Time (s)", "total_time", "{:.2f}"),
        ("Final Loss", "final_loss", "{:.4f}")
    ]
    
    for display_name, key, fmt in metrics:
        baseline_val = baseline.get(key, 0)
        optimized_val = optimized.get(key, 0)
        improvement_key = f"{key}_improvement"
        improvement = improvements.get(improvement_key, 0)
        
        print(f"{display_name:<25} {fmt.format(baseline_val):<15} {fmt.format(optimized_val):<15} {improvement:>+.1%}")
    
    print(f"\n🎯 Key Improvements:")
    if improvements.get("tokens_per_second_improvement", 0) > 0:
        print(f"  🚀 {improvements['tokens_per_second_improvement']:.1%} faster training")
    if improvements.get("peak_memory_mb_improvement", 0) > 0:
        print(f"  🧠 {improvements['peak_memory_mb_improvement']:.1%} less memory usage")
    if improvements.get("memory_efficiency_improvement", 0) > 0:
        print(f"  ⚡ {improvements['memory_efficiency_improvement']:.1%} better memory efficiency")
    
    # Save detailed results
    with open("demo_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to demo_comparison_results.json")
    
    return results


def run_evolution(iterations: int = 50):
    """Run OpenEvolve to discover new optimization patterns"""
    print("🧬 Running OpenEvolve to Discover New Patterns")
    print("=" * 50)
    
    # Check if OpenEvolve is available
    try:
        from openevolve import OpenEvolve
    except ImportError:
        print("❌ OpenEvolve not found. Please install it first:")
        print("   pip install -e .")
        return None
    
    # Ensure baseline exists
    if not os.path.exists("baseline_output/training_results.json"):
        print("📋 Baseline results not found. Running baseline first...")
        run_baseline(num_samples=100)
    
    print(f"🔬 Starting evolution for {iterations} iterations...")
    print("This may take a while as each iteration runs actual fine-tuning...")
    
    # Initialize OpenEvolve
    initial_program = os.path.join(os.path.dirname(__file__), "initial_program.py")
    evaluator = os.path.join(os.path.dirname(__file__), "evaluator.py")
    config = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    evolve = OpenEvolve(
        initial_program_path=initial_program,
        evaluation_file=evaluator,
        config_path=config
    )
    
    # Run evolution
    try:
        import asyncio
        best_program = asyncio.run(evolve.run(iterations=iterations))
        
        if best_program:
            print(f"\n🌟 Evolution Complete!")
            print(f"📊 Best program metrics:")
            for name, value in best_program.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {name}: {value:.4f}")
            
            print(f"\n💾 Best optimization patterns saved to:")
            print(f"  openevolve_output/best/best_program.py")
            
            return best_program
        else:
            print("❌ Evolution failed to find improvements")
            return None
            
    except Exception as e:
        print(f"❌ Evolution failed: {e}")
        return None


def demo_context_manager():
    """Demonstrate using the context manager approach"""
    print("🎭 Demonstrating Context Manager Usage")
    print("=" * 50)
    
    # Example of how users would integrate into existing code
    trainer = BaselineTrainer("mlx-community/Qwen3-0.6B-bf16")
    trainer.config.batch_size = 1
    trainer.config.num_epochs = 1
    
    dataset = trainer.create_sample_dataset(50)
    
    print("Training with automatic optimizations...")
    
    with mlx_optimizations():
        # All training inside this context will use optimized patterns
        results = trainer.train(dataset, "./demo_context_output")
    
    print(f"✅ Context manager demo complete")
    print(f"📊 Results: {results['tokens_per_second']:.1f} tokens/sec, {results['peak_memory_mb']:.1f} MB")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="MLX Fine-tuning Optimization Demo")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only")
    parser.add_argument("--optimized", action="store_true", help="Run optimized only")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs optimized")
    parser.add_argument("--evolve", action="store_true", help="Run evolution to discover patterns")
    parser.add_argument("--context", action="store_true", help="Demo context manager usage")
    parser.add_argument("--samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--iterations", type=int, default=50, help="Evolution iterations")
    
    args = parser.parse_args()
    
    if not any([args.baseline, args.optimized, args.compare, args.evolve, args.context]):
        print("🚀 MLX Fine-tuning Optimization Demo")
        print("=" * 50)
        print("No specific mode selected. Running comparison by default.")
        print("Use --help to see all available modes.")
        print()
        args.compare = True
    
    try:
        if args.baseline:
            run_baseline(args.samples)
        
        elif args.optimized:
            run_optimized(args.samples)
        
        elif args.compare:
            compare_performance(args.samples)
        
        elif args.evolve:
            run_evolution(args.iterations)
        
        elif args.context:
            demo_context_manager()
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
