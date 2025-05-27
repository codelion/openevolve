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


def check_best_program_exists():
    """Check if best_program.py exists and exit if not found"""
    # Check current directory first
    current_dir_best = os.path.join(os.getcwd(), "best_program.py")
    if os.path.exists(current_dir_best):
        print(f"✅ Found best_program.py in current directory: {current_dir_best}")
        return current_dir_best
    
    # Check openevolve output directory
    script_dir = os.path.dirname(__file__)
    openevolve_output = os.path.join(script_dir, "openevolve_output")
    
    if os.path.exists(openevolve_output):
        # Look for the best program
        best_dir = os.path.join(openevolve_output, "best")
        if os.path.exists(best_dir):
            best_program = os.path.join(best_dir, "best_program.py")
            if os.path.exists(best_program):
                print(f"✅ Found best_program.py in openevolve output: {best_program}")
                return best_program
        
        # Look in checkpoints for latest
        checkpoints_dir = os.path.join(openevolve_output, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint_")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1]))
                checkpoint_program = os.path.join(checkpoints_dir, latest_checkpoint, "best_program.py")
                if os.path.exists(checkpoint_program):
                    print(f"✅ Found best_program.py in latest checkpoint: {checkpoint_program}")
                    return checkpoint_program
    
    # If we get here, no best_program.py was found
    print("❌ Error: best_program.py not found!")
    print("")
    print("The demo requires a best_program.py file with evolved optimizations.")
    print("")
    print("To get best_program.py, you can:")
    print("  1. Run evolution: python demo.py --evolve --iterations 50")
    print("  2. Copy from openevolve_output/best/ if it exists")
    print("  3. Copy from a checkpoint: openevolve_output/checkpoints/checkpoint_*/best_program.py")
    print("")
    print("Searched locations:")
    print(f"  • Current directory: {current_dir_best}")
    print(f"  • OpenEvolve output: {os.path.join(script_dir, 'openevolve_output', 'best', 'best_program.py')}")
    print(f"  • Latest checkpoint: {os.path.join(script_dir, 'openevolve_output', 'checkpoints', '*', 'best_program.py')}")
    print("")
    sys.exit(1)


def run_optimized(num_samples: int = 200, output_dir: str = "./demo_optimized"):
    """Run optimized MLX fine-tuning"""
    print("⚡ Running Optimized MLX Fine-tuning")
    print("=" * 50)
    
    # Check that best_program.py exists before proceeding
    best_program_path = check_best_program_exists()
    
    try:
        # Create trainer with specific optimization path
        trainer = create_optimized_trainer("mlx-community/Qwen3-0.6B-bf16", best_program_path)
        trainer.config.batch_size = 2
        trainer.config.num_epochs = 1
        print(f"✅ Created optimized trainer using {best_program_path}")
    except Exception as e:
        print(f"❌ Failed to create optimized trainer: {e}")
        sys.exit(1)
    
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
    
    # Check that best_program.py exists before proceeding
    best_program_path = check_best_program_exists()
    
    print("Running comprehensive benchmark...")
    # Pass the specific best program path to ensure we use the evolved optimizations
    from mlx_optimization_patch import benchmark_optimization_improvement
    results = benchmark_optimization_improvement(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        num_samples=num_samples,
        optimization_path=best_program_path
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
    
    # Check that best_program.py exists before proceeding
    best_program_path = check_best_program_exists()
    
    # Example of how users would integrate into existing code
    trainer = BaselineTrainer("mlx-community/Qwen3-0.6B-bf16")
    trainer.config.batch_size = 1
    trainer.config.num_epochs = 1
    
    dataset = trainer.create_sample_dataset(50)
    
    print("Training with automatic optimizations...")
    
    with mlx_optimizations(best_program_path):
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
