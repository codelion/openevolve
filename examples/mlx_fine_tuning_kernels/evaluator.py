"""
MLX Fine-tuning Kernels Evaluator

This evaluator tests custom fine-tuning operations at two levels:
1. Micro-benchmarks: Individual kernel performance vs naive baselines
2. Macro-benchmark: Actual fine-tuning performance with REAL MLX models only

The goal is to demonstrate that kernel optimizations translate to real
training speedups and memory reductions, similar to Liger Kernel's results.
"""

import importlib.util
import time
import traceback
import statistics
import gc
import psutil
import os
from typing import Dict, Union, List, Tuple, Optional

# Required imports - fail fast if not available
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
except ImportError as e:
    raise ImportError(f"MLX not available: {e}. Please install with: pip install mlx")

try:
    import psutil
except ImportError as e:
    raise ImportError(f"psutil not available: {e}. Please install with: pip install psutil")


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def benchmark_kernel(kernel_func, args, num_trials=5, warmup=2):
    """Benchmark a kernel function with proper warmup and timing."""
    
    # Warmup runs
    for _ in range(warmup):
        result = kernel_func(*args)
        mx.eval(result)
    
    # Clear cache
    mx.clear_cache()
    
    # Benchmark runs
    times = []
    memory_before = get_memory_usage()
    
    for _ in range(num_trials):
        start_time = time.perf_counter()
        result = kernel_func(*args)
        mx.eval(result)  # Ensure computation completes
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    memory_after = get_memory_usage()
    memory_delta = memory_after - memory_before
    
    return result, statistics.median(times), memory_delta


def evaluate_micro_benchmarks(evolved_kernels, naive_kernels):
    """Test individual kernel performance against baselines."""
    print("\n📊 MICRO-BENCHMARKS: Individual Kernel Performance")
    
    # Test configurations
    test_configs = [
        {"batch_size": 4, "seq_len": 64, "d_model": 256, "vocab_size": 1000},
        {"batch_size": 8, "seq_len": 128, "d_model": 512, "vocab_size": 2000},
        {"batch_size": 2, "seq_len": 256, "d_model": 768, "vocab_size": 5000},
    ]
    
    kernel_tests = [
        'rms_norm', 'rope_embeddings', 'swiglu_activation', 
        'cross_entropy_loss', 'lora_linear', 'attention_with_rope'
    ]
    
    all_results = []
    correctness_passed = 0
    total_tests = 0
    
    for config in test_configs:
        print(f"\n--- Config: {config} ---")
        
        # Create test data
        from initial_program import create_test_data
        test_data = create_test_data(**config)
        
        for kernel_name in kernel_tests:
            print(f"  {kernel_name}:")
            total_tests += 1
            
            # Get kernel arguments
            if kernel_name == 'rms_norm':
                args = [test_data['x_norm'], test_data['weight_norm']]
            elif kernel_name == 'rope_embeddings':
                args = [test_data['x_rope'], test_data['freqs_cos'], test_data['freqs_sin']]
            elif kernel_name == 'swiglu_activation':
                args = [test_data['x_mlp'], test_data['w_gate'], test_data['w_up']]
            elif kernel_name == 'cross_entropy_loss':
                args = [test_data['logits'], test_data['targets']]
            elif kernel_name == 'lora_linear':
                args = [test_data['x_lora'], test_data['base_weight'], 
                       test_data['lora_a'], test_data['lora_b']]
            elif kernel_name == 'attention_with_rope':
                args = [test_data['query'], test_data['key'], test_data['value'],
                       test_data['freqs_cos'], test_data['freqs_sin']]
            else:
                continue
            
            try:
                # Benchmark evolved kernel
                evolved_result, evolved_time, evolved_memory = benchmark_kernel(
                    evolved_kernels[kernel_name], args
                )
                
                # Benchmark naive kernel
                naive_result, naive_time, naive_memory = benchmark_kernel(
                    naive_kernels[kernel_name], args
                )
                
                # Check correctness
                if evolved_result.shape == naive_result.shape:
                    max_diff = float(mx.max(mx.abs(evolved_result - naive_result)))
                    if max_diff < 1e-2:  # Reasonable tolerance
                        correctness_passed += 1
                        speedup = naive_time / evolved_time if evolved_time > 0 else 0.0
                        memory_ratio = evolved_memory / naive_memory if naive_memory > 0 else 1.0
                        
                        status = "🟢" if speedup >= 1.1 else "🟡" if speedup >= 0.9 else "🔴"
                        print(f"    {speedup:.2f}x speedup, {memory_ratio:.2f}x memory ({evolved_time*1000:.1f}ms vs {naive_time*1000:.1f}ms) {status}")
                        
                        all_results.append({
                            'kernel': kernel_name,
                            'config': config,
                            'speedup': speedup,
                            'memory_ratio': memory_ratio,
                            'evolved_time': evolved_time,
                            'naive_time': naive_time,
                            'correctness': True
                        })
                    else:
                        print(f"    ❌ CORRECTNESS FAILED: diff={max_diff:.2e}")
                        all_results.append({
                            'kernel': kernel_name,
                            'config': config,
                            'speedup': 0.0,
                            'memory_ratio': 1.0,
                            'correctness': False
                        })
                else:
                    print(f"    ❌ SHAPE MISMATCH: {evolved_result.shape} vs {naive_result.shape}")
                    all_results.append({
                        'kernel': kernel_name,
                        'config': config,
                        'speedup': 0.0,
                        'memory_ratio': 1.0,
                        'correctness': False
                    })
                    
            except Exception as e:
                print(f"    ❌ ERROR: {e}")
                all_results.append({
                    'kernel': kernel_name,
                    'config': config,
                    'speedup': 0.0,
                    'memory_ratio': 1.0,
                    'correctness': False
                })
    
    # Calculate summary statistics
    speedups = [r['speedup'] for r in all_results if r['correctness']]
    memory_ratios = [r['memory_ratio'] for r in all_results if r['correctness']]
    
    micro_score = 0.0
    if speedups:
        avg_speedup = statistics.mean(speedups)
        avg_memory_ratio = statistics.mean(memory_ratios)
        correctness_rate = correctness_passed / total_tests
        
        # Score calculation: correctness (60%) + performance (40%)
        correctness_component = 0.6 * correctness_rate
        performance_component = 0.4 * min(avg_speedup / 1.2, 2.0)  # Target 1.2x, cap at 2.0
        
        micro_score = correctness_component + performance_component
        
        print(f"\n📈 MICRO-BENCHMARK SUMMARY:")
        print(f"  Correctness: {correctness_passed}/{total_tests} ({correctness_rate:.1%})")
        print(f"  Average Speedup: {avg_speedup:.2f}x")
        print(f"  Average Memory Ratio: {avg_memory_ratio:.2f}x")
        print(f"  Micro Score: {micro_score:.3f}")
    
    return micro_score, all_results


def evaluate_macro_benchmark(evolved_kernels, naive_kernels):
    """Test actual fine-tuning performance using REAL MLX models only."""
    
    print("\n🚀 REAL MODEL MACRO-BENCHMARK: Using actual MLX models")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'temp'))
        from real_model_benchmark import evaluate_real_model_macro_benchmark
        
        real_score, real_results = evaluate_real_model_macro_benchmark(evolved_kernels, naive_kernels)
        
        if real_score > 0 and 'error' not in real_results:
            print(f"  ✅ Real model benchmark succeeded!")
            return real_score, real_results
        else:
            error_msg = real_results.get('error', 'Unknown error') if isinstance(real_results, dict) else 'Real model benchmark failed'
            print(f"  ❌ Real model benchmark failed: {error_msg}")
            return 0.0, {"error": f"Real model benchmark failed: {error_msg}"}
            
    except Exception as e:
        error_msg = f"Real model benchmark not available: {e}"
        print(f"  ❌ {error_msg}")
        print(f"  📝 To install dependencies: python setup_comprehensive_evaluation.py")
        return 0.0, {"error": error_msg}


def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Evaluate MLX fine-tuning kernels program.
    
    Tests both individual kernel performance and actual fine-tuning benefits.
    Uses REAL models only for macro-benchmarking.
    """
    print(f"🚀 Evaluating MLX Fine-tuning Kernels: {program_path}")
    
    try:
        # Load evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "evolved_fine_tuning_kernels"):
            return {
                "overall_score": 0.0,
                "error": "Missing evolved_fine_tuning_kernels function"
            }
        
        # Get kernel implementations
        evolved_kernels = evolved_program.evolved_fine_tuning_kernels()
        naive_kernels = evolved_program.naive_baseline_kernels()
        
        print(f"Testing {len(evolved_kernels)} kernels...")
        
        # Run micro-benchmarks
        micro_score, micro_results = evaluate_micro_benchmarks(evolved_kernels, naive_kernels)
        
        # Run macro-benchmark (REAL models only)
        macro_score, macro_results = evaluate_macro_benchmark(evolved_kernels, naive_kernels)
        
        # Try extended evaluation with real fine-tuning
        extended_results = {}
        extended_score = 0.0
        
        try:
            from extended_evaluation import extended_evaluation_with_real_finetuning
            # Pass the program path for comprehensive evaluation with real models
            extended_results = extended_evaluation_with_real_finetuning(
                evolved_kernels, naive_kernels, program_path
            )
            
            if 'error' not in extended_results:
                extended_score = extended_results.get('extended_score', 0.0)
                print(f"\n🔬 EXTENDED EVALUATION RESULTS:")
                print(f"  Extended Score: {extended_score:.3f}")
                print(f"  Real Fine-tuning Speedup: {extended_results.get('real_finetuning_speedup', 0):.2f}x")
                if 'models_tested' in extended_results:
                    print(f"  Models Tested: {extended_results['models_tested']}")
                    print(f"  Model Sizes: {extended_results.get('model_sizes', [])}")
                if 'standard_mlx_speedup' in extended_results:
                    print(f"  vs Standard MLX: {extended_results['standard_mlx_speedup']:.2f}x")
                print(f"  Convergence Quality: {extended_results.get('convergence_quality', 0):.4f}")
            else:
                print(f"\n⚠️ Extended evaluation failed: {extended_results['error']}")
                
        except ImportError:
            print("\n📝 Extended evaluation not available (extended_evaluation.py not found)")
        except Exception as e:
            print(f"\n⚠️ Extended evaluation error: {e}")
        
        # Calculate overall score
        # Weight: micro (40%) + macro (40%) + extended (20%)
        if extended_score > 0:
            overall_score = 0.4 * micro_score + 0.4 * macro_score + 0.2 * extended_score
        else:
            # Fallback: micro (50%) + macro (50%)
            overall_score = 0.5 * micro_score + 0.5 * macro_score
        
        # Summary statistics
        speedups = [r['speedup'] for r in micro_results if r['correctness']]
        avg_speedup = statistics.mean(speedups) if speedups else 0.0
        max_speedup = max(speedups) if speedups else 0.0
        correctness_rate = len([r for r in micro_results if r['correctness']]) / len(micro_results)
        
        print(f"\n🏆 FINAL EVALUATION:")
        print(f"  Overall Score: {overall_score:.3f}")
        print(f"  Micro Score: {micro_score:.3f}")
        print(f"  Macro Score: {macro_score:.3f}")
        print(f"  Kernel Correctness: {correctness_rate:.1%}")
        print(f"  Average Kernel Speedup: {avg_speedup:.2f}x")
        if macro_results and 'error' not in macro_results:
            print(f"  Training Speedup: {macro_results.get('time_speedup', 0):.2f}x")
            print(f"  Memory Efficiency: {macro_results.get('memory_reduction', 1):.2f}x")
        
        # Interpret score
        if overall_score >= 0.8:
            print("  🥇 EXCELLENT: Strong optimizations with real fine-tuning benefits!")
        elif overall_score >= 0.6:
            print("  🥈 GOOD: Meaningful improvements in kernels and training")
        elif overall_score >= 0.4:
            print("  🥉 MODERATE: Some optimizations working")
        elif overall_score >= 0.2:
            print("  📈 PROGRESS: Basic improvements detected")
        else:
            print("  🔄 BASELINE: Limited improvement so far")
        
        # Prepare results
        results = {
            "overall_score": float(overall_score),
            "combined_score": float(overall_score),  # Primary metric for OpenEvolve
            
            # Detailed metrics
            "micro_score": float(micro_score),
            "macro_score": float(macro_score),
            "correctness_rate": float(correctness_rate),
            "avg_kernel_speedup": float(avg_speedup),
            "max_kernel_speedup": float(max_speedup),
            
            # Macro metrics
            "training_speedup": float(macro_results.get('time_speedup', 0)),
            "memory_reduction": float(macro_results.get('memory_reduction', 1)),
            "loss_difference": float(macro_results.get('loss_diff', 0)),
            
            # Extended metrics
            "extended_score": float(extended_score),
            "real_finetuning_speedup": float(extended_results.get('real_finetuning_speedup', 0)),
            "convergence_quality": float(extended_results.get('convergence_quality', 0)),
            
            # Counts
            "total_kernel_tests": len(micro_results),
            "passed_correctness": len([r for r in micro_results if r['correctness']]),
            
            # Metadata
            "evaluation_type": "mlx_fine_tuning_kernels",
            "has_macro_results": bool(macro_results and 'error' not in macro_results),
            "has_extended_results": bool(extended_results and 'error' not in extended_results)
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "overall_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Testing MLX Fine-tuning Kernels Evaluator...")
    
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("\nEvaluation Results:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
