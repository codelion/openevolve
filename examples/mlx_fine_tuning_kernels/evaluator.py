"""
MLX Fusion-Based Kernels Evaluator

This evaluator tests fusion-based operations that combine multiple MLX operations
to reduce kernel launches and memory transfers. The goal is to demonstrate that
fusion patterns can achieve speedups over standard MLX operation sequences.
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
        if isinstance(result, tuple):
            # Handle training step which returns multiple values
            for r in result:
                if isinstance(r, mx.array):
                    mx.eval(r)
                elif isinstance(r, dict):
                    for v in r.values():
                        if isinstance(v, mx.array):
                            mx.eval(v)
        else:
            mx.eval(result)
    
    # Clear cache
    mx.clear_cache()
    
    # Benchmark runs
    times = []
    memory_before = get_memory_usage()
    
    for _ in range(num_trials):
        start_time = time.perf_counter()
        result = kernel_func(*args)
        
        # Ensure computation completes
        if isinstance(result, tuple):
            for r in result:
                if isinstance(r, mx.array):
                    mx.eval(r)
                elif isinstance(r, dict):
                    for v in r.values():
                        if isinstance(v, mx.array):
                            mx.eval(v)
        else:
            mx.eval(result)
            
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    memory_after = get_memory_usage()
    memory_delta = memory_after - memory_before
    
    return result, statistics.median(times), memory_delta


def create_standard_mlx_baselines():
    """Create standard MLX implementations using built-in operations for comparison."""
    
    def standard_transformer_block(x: mx.array,
                                  attn_weights: Dict[str, mx.array],
                                  mlp_weights: Dict[str, mx.array], 
                                  norm_weights: Tuple[mx.array, mx.array],
                                  freqs_cos: mx.array, freqs_sin: mx.array,
                                  eps: float = 1e-6) -> mx.array:
        """Standard transformer block using MLX built-in operations."""
        batch_size, seq_len, d_model = x.shape
        
        # Standard layer norm (not RMS norm)
        x_norm1 = nn.LayerNorm(d_model)(x)
        
        # Standard multi-head attention (simplified)
        q = x_norm1 @ attn_weights['q_proj'].T
        k = x_norm1 @ attn_weights['k_proj'].T
        v = x_norm1 @ attn_weights['v_proj'].T
        
        # Simplified attention (without proper multi-head reshaping for speed)
        scale = 1.0 / (d_model ** 0.5)
        scores = q @ k.T * scale
        attn_weights_computed = mx.softmax(scores, axis=-1)
        attn_out = attn_weights_computed @ v
        attn_out = attn_out @ attn_weights['o_proj'].T
        
        # Residual connection
        x = x + attn_out
        
        # Standard layer norm
        x_norm2 = nn.LayerNorm(d_model)(x)
        
        # Standard MLP
        mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        mlp_out = mlp(x_norm2)
        
        return x + mlp_out
    
    def standard_lora_linear(x: mx.array, base_weight: mx.array,
                            lora_a: mx.array, lora_b: mx.array,
                            scale: float = 1.0) -> mx.array:
        """Standard LoRA implementation with separate operations."""
        base_out = x @ base_weight.T
        lora_out = x @ lora_a.T @ lora_b.T
        return base_out + scale * lora_out
    
    def standard_cross_entropy_loss(logits: mx.array, targets: mx.array,
                                   ignore_index: int = -100,
                                   chunk_size: int = 2048) -> mx.array:
        """Standard MLX CrossEntropy loss."""
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction='mean'
        )
    
    def standard_attention(query: mx.array, key: mx.array, value: mx.array,
                          chunk_size: int = 1024) -> mx.array:
        """Standard MLX attention implementation."""
        batch_size, n_heads, seq_len, head_dim = query.shape
        scale = 1.0 / (head_dim ** 0.5)
        
        scores = mx.matmul(query, mx.transpose(key, axes=(0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, value)
        return output
    
    def standard_training_step(inputs: mx.array, targets: mx.array,
                              model_weights: Dict[str, mx.array],
                              optimizer_state: Dict, learning_rate: float) -> Tuple[Dict[str, mx.array], mx.array]:
        """Standard training step with separate operations."""
        logits = inputs @ model_weights['output_proj'].T
        loss = standard_cross_entropy_loss(logits, targets)
        
        # Simplified weight update
        updated_weights = {}
        for name, weight in model_weights.items():
            grad_estimate = mx.random.normal(weight.shape) * 0.001
            updated_weights[name] = weight - learning_rate * grad_estimate
            
        return updated_weights, loss
    
    def standard_multi_layer_norm(x: mx.array, weights: List[mx.array], eps: float = 1e-6) -> mx.array:
        """Standard multi-layer normalization."""
        result = x
        for weight in weights:
            result = nn.LayerNorm(x.shape[-1])(result)
        return result
    
    return {
        'fused_transformer_block': standard_transformer_block,
        'apply_rope_optimized': lambda x, cos, sin: x,  # Simplified
        'fused_lora_linear': standard_lora_linear,
        'online_cross_entropy_loss': standard_cross_entropy_loss,
        'memory_efficient_attention': standard_attention,
        'fused_training_step': standard_training_step,
        'fused_multi_layer_norm': standard_multi_layer_norm
    }


def evaluate_fusion_benchmarks(evolved_kernels, naive_kernels, standard_kernels):
    """Test fusion operations against both naive and standard MLX implementations."""
    print("\n📊 FUSION BENCHMARKS: Multi-Operation Performance")
    
    # Test configurations focused on fusion opportunities
    test_configs = [
        {"batch_size": 2, "seq_len": 64, "d_model": 256, "vocab_size": 1000},
        {"batch_size": 4, "seq_len": 128, "d_model": 512, "vocab_size": 2000},
        {"batch_size": 1, "seq_len": 256, "d_model": 512, "vocab_size": 5000},  # Large vocab test
    ]
    
    fusion_tests = [
        'fused_lora_linear', 'online_cross_entropy_loss', 'memory_efficient_attention',
        'fused_training_step', 'fused_multi_layer_norm'
    ]
    
    all_results = []
    correctness_passed = 0
    total_tests = 0
    
    for config in test_configs:
        print(f"\n--- Config: {config} ---")
        
        # Create test data for fusion operations
        from fusion_based_initial_program import create_test_data
        test_data = create_test_data(**config)
        
        for kernel_name in fusion_tests:
            print(f"  {kernel_name}:")
            total_tests += 1
            
            # Get kernel arguments
            if kernel_name == 'fused_lora_linear':
                args = [test_data['x_lora'], test_data['base_weight'],
                       test_data['lora_a'], test_data['lora_b']]
            elif kernel_name == 'online_cross_entropy_loss':
                args = [test_data['logits'], test_data['targets']]
            elif kernel_name == 'memory_efficient_attention':
                args = [test_data['query'], test_data['key'], test_data['value']]
            elif kernel_name == 'fused_training_step':
                args = [test_data['inputs_train'], test_data['targets_train'],
                       test_data['model_weights'], test_data['optimizer_state'], 0.001]
            elif kernel_name == 'fused_multi_layer_norm':
                args = [test_data['x_norm'], test_data['norm_weights_list']]
            else:
                continue
            
            try:
                # Benchmark evolved (fusion) implementation
                evolved_result, evolved_time, evolved_memory = benchmark_kernel(
                    evolved_kernels[kernel_name], args
                )
                
                # Benchmark naive implementation
                naive_result, naive_time, naive_memory = benchmark_kernel(
                    naive_kernels[kernel_name], args
                )
                
                # Benchmark standard MLX implementation
                standard_result, standard_time, standard_memory = benchmark_kernel(
                    standard_kernels[kernel_name], args
                )
                
                # Check correctness against naive baseline
                correctness_ok = True
                
                if kernel_name == 'fused_training_step':
                    # Special handling for training step
                    evolved_weights, evolved_loss = evolved_result
                    naive_weights, naive_loss = naive_result
                    standard_weights, standard_loss = standard_result
                    
                    loss_diff_naive = abs(float(evolved_loss) - float(naive_loss))
                    loss_diff_standard = abs(float(evolved_loss) - float(standard_loss))
                    
                    if loss_diff_naive < 0.1:  # Allow some randomness
                        correctness_passed += 1
                        
                        speedup_vs_naive = naive_time / evolved_time if evolved_time > 0 else 0.0
                        speedup_vs_standard = standard_time / evolved_time if evolved_time > 0 else 0.0
                        memory_ratio = evolved_memory / naive_memory if naive_memory > 0 else 1.0
                        
                        status_naive = "🟢" if speedup_vs_naive >= 1.1 else "🟡" if speedup_vs_naive >= 0.9 else "🔴"
                        status_standard = "🟢" if speedup_vs_standard >= 1.0 else "🔴"
                        
                        print(f"    vs Naive: {speedup_vs_naive:.2f}x speedup {status_naive}")
                        print(f"    vs Standard MLX: {speedup_vs_standard:.2f}x speedup {status_standard}")
                        print(f"    Memory ratio: {memory_ratio:.2f}x")
                        
                        all_results.append({
                            'kernel': kernel_name,
                            'config': config,
                            'speedup_vs_naive': speedup_vs_naive,
                            'speedup_vs_standard': speedup_vs_standard,
                            'memory_ratio': memory_ratio,
                            'evolved_time': evolved_time,
                            'naive_time': naive_time,
                            'standard_time': standard_time,
                            'correctness': True
                        })
                    else:
                        print(f"    ❌ CORRECTNESS FAILED: loss_diff={loss_diff_naive:.4f}")
                        correctness_ok = False
                
                else:
                    # Standard tensor comparison
                    if (evolved_result.shape == naive_result.shape and 
                        evolved_result.shape == standard_result.shape):
                        
                        max_diff_naive = float(mx.max(mx.abs(evolved_result - naive_result)))
                        max_diff_standard = float(mx.max(mx.abs(evolved_result - standard_result)))
                        
                        if max_diff_naive < 1e-1:  # More lenient for fusion operations
                            correctness_passed += 1
                            
                            speedup_vs_naive = naive_time / evolved_time if evolved_time > 0 else 0.0
                            speedup_vs_standard = standard_time / evolved_time if evolved_time > 0 else 0.0
                            memory_ratio = evolved_memory / naive_memory if naive_memory > 0 else 1.0
                            
                            status_naive = "🟢" if speedup_vs_naive >= 1.1 else "🟡" if speedup_vs_naive >= 0.9 else "🔴"
                            status_standard = "🟢" if speedup_vs_standard >= 1.0 else "🔴"
                            
                            print(f"    vs Naive: {speedup_vs_naive:.2f}x speedup, {memory_ratio:.2f}x memory ({evolved_time*1000:.1f}ms vs {naive_time*1000:.1f}ms) {status_naive}")
                            print(f"    vs Standard MLX: {speedup_vs_standard:.2f}x speedup ({evolved_time*1000:.1f}ms vs {standard_time*1000:.1f}ms) {status_standard}")
                            
                            all_results.append({
                                'kernel': kernel_name,
                                'config': config,
                                'speedup_vs_naive': speedup_vs_naive,
                                'speedup_vs_standard': speedup_vs_standard,
                                'memory_ratio': memory_ratio,
                                'evolved_time': evolved_time,
                                'naive_time': naive_time,
                                'standard_time': standard_time,
                                'correctness': True
                            })
                        else:
                            print(f"    ❌ CORRECTNESS FAILED: max_diff_naive={max_diff_naive:.2e}")
                            correctness_ok = False
                    else:
                        print(f"    ❌ SHAPE MISMATCH")
                        correctness_ok = False
                
                if not correctness_ok:
                    all_results.append({
                        'kernel': kernel_name,
                        'config': config,
                        'speedup_vs_naive': 0.0,
                        'speedup_vs_standard': 0.0,
                        'memory_ratio': 1.0,
                        'correctness': False
                    })
                    
            except Exception as e:
                print(f"    ❌ ERROR: {e}")
                all_results.append({
                    'kernel': kernel_name,
                    'config': config,
                    'speedup_vs_naive': 0.0,
                    'speedup_vs_standard': 0.0,
                    'memory_ratio': 1.0,
                    'correctness': False
                })
    
    # Calculate summary statistics
    correct_results = [r for r in all_results if r['correctness']]
    
    if correct_results:
        speedups_vs_naive = [r['speedup_vs_naive'] for r in correct_results]
        speedups_vs_standard = [r['speedup_vs_standard'] for r in correct_results]
        memory_ratios = [r['memory_ratio'] for r in correct_results]
        
        avg_speedup_naive = statistics.mean(speedups_vs_naive)
        avg_speedup_standard = statistics.mean(speedups_vs_standard)
        avg_memory_ratio = statistics.mean(memory_ratios)
        correctness_rate = correctness_passed / total_tests
        
        # Score calculation emphasizing standard MLX comparison
        correctness_component = 0.4 * correctness_rate
        naive_performance_component = 0.3 * min(avg_speedup_naive / 1.2, 2.0)
        standard_performance_component = 0.3 * min(avg_speedup_standard / 1.0, 2.0)  # Key metric!
        
        fusion_score = correctness_component + naive_performance_component + standard_performance_component
        
        print(f"\n📈 FUSION BENCHMARK SUMMARY:")
        print(f"  Correctness: {correctness_passed}/{total_tests} ({correctness_rate:.1%})")
        print(f"  Average Speedup vs Naive: {avg_speedup_naive:.2f}x")
        print(f"  Average Speedup vs Standard MLX: {avg_speedup_standard:.2f}x ⭐")
        print(f"  Average Memory Ratio: {avg_memory_ratio:.2f}x")
        print(f"  Fusion Score: {fusion_score:.3f}")
        
        # Key success metric
        if avg_speedup_standard >= 1.1:
            print("  🎉 SUCCESS: Beating standard MLX operations!")
        elif avg_speedup_standard >= 1.0:
            print("  📈 PROGRESS: Approaching standard MLX performance!")
        else:
            print("  🔄 DEVELOPING: Still behind standard MLX")
    else:
        fusion_score = 0.0
        avg_speedup_naive = 0.0
        avg_speedup_standard = 0.0
        avg_memory_ratio = 1.0
        correctness_rate = 0.0
    
    return fusion_score, {
        'avg_speedup_vs_naive': avg_speedup_naive,
        'avg_speedup_vs_standard': avg_speedup_standard,
        'avg_memory_ratio': avg_memory_ratio,
        'correctness_rate': correctness_rate,
        'all_results': all_results
    }


def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Evaluate MLX fusion-based fine-tuning kernels program.
    
    Tests fusion operations against both naive and standard MLX implementations.
    Primary success metric: speedup vs standard MLX operations.
    """
    print(f"🚀 Evaluating MLX Fusion-Based Kernels: {program_path}")
    
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
        standard_kernels = create_standard_mlx_baselines()
        
        print(f"Testing {len(evolved_kernels)} fusion operations...")
        
        # Run fusion benchmarks (main evaluation)
        fusion_score, fusion_results = evaluate_fusion_benchmarks(
            evolved_kernels, naive_kernels, standard_kernels
        )
        
        # Try real model evaluation if available
        macro_score = 0.0
        macro_results = {}
        
        try:
            from extended_evaluation import extended_evaluation_with_real_finetuning
            macro_results = extended_evaluation_with_real_finetuning(
                evolved_kernels, naive_kernels, program_path
            )
            
            if 'error' not in macro_results:
                macro_score = macro_results.get('extended_score', 0.0)
                print(f"\n🔬 REAL MODEL EVALUATION:")
                print(f"  Real Model Score: {macro_score:.3f}")
                print(f"  Real Fine-tuning Speedup: {macro_results.get('real_finetuning_speedup', 0):.2f}x")
            else:
                print(f"\n⚠️ Real model evaluation failed: {macro_results['error']}")
                
        except ImportError:
            print("\n📝 Real model evaluation not available")
        except Exception as e:
            print(f"\n⚠️ Real model evaluation error: {e}")
        
        # Calculate overall score with emphasis on standard MLX comparison
        if macro_score > 0:
            overall_score = 0.6 * fusion_score + 0.4 * macro_score
        else:
            overall_score = fusion_score
        
        # Key metrics
        avg_speedup_naive = fusion_results.get('avg_speedup_vs_naive', 0.0)
        avg_speedup_standard = fusion_results.get('avg_speedup_vs_standard', 0.0)  # KEY METRIC
        correctness_rate = fusion_results.get('correctness_rate', 0.0)
        
        print(f"\n🏆 FINAL EVALUATION:")
        print(f"  Overall Score: {overall_score:.3f}")
        print(f"  Fusion Score: {fusion_score:.3f}")
        print(f"  Fusion Correctness: {correctness_rate:.1%}")
        print(f"  Average Speedup vs Naive: {avg_speedup_naive:.2f}x")
        print(f"  Average Speedup vs Standard MLX: {avg_speedup_standard:.2f}x ⭐")
        
        # Success interpretation focused on standard MLX
        if avg_speedup_standard >= 1.2:
            print("  🥇 EXCELLENT: Significant speedup over standard MLX!")
        elif avg_speedup_standard >= 1.1:
            print("  🥈 VERY GOOD: Beating standard MLX operations!")
        elif avg_speedup_standard >= 1.0:
            print("  🥉 GOOD: Matching standard MLX performance!")
        elif avg_speedup_standard >= 0.9:
            print("  📈 PROGRESS: Close to standard MLX performance!")
        else:
            print("  🔄 DEVELOPING: Need more optimization vs standard MLX")
        
        # Prepare results
        results = {
            "overall_score": float(overall_score),
            "combined_score": float(overall_score),  # Primary metric for OpenEvolve
            
            # Fusion-specific metrics  
            "fusion_score": float(fusion_score),
            "correctness_rate": float(correctness_rate),
            "avg_speedup_vs_naive": float(avg_speedup_naive),
            "avg_speedup_vs_standard": float(avg_speedup_standard),  # KEY SUCCESS METRIC
            "avg_memory_ratio": float(fusion_results.get('avg_memory_ratio', 1.0)),
            
            # Real model metrics
            "macro_score": float(macro_score),
            "real_finetuning_speedup": float(macro_results.get('real_finetuning_speedup', 0)),
            "convergence_quality": float(macro_results.get('convergence_quality', 0)),
            
            # Counts
            "total_fusion_tests": len(fusion_results.get('all_results', [])),
            "passed_correctness": len([r for r in fusion_results.get('all_results', []) if r.get('correctness', False)]),
            
            # Metadata
            "evaluation_type": "mlx_fusion_kernels",
            "beats_standard_mlx": bool(avg_speedup_standard >= 1.0),
            "target_achieved": bool(avg_speedup_standard >= 1.1),  # Success threshold
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
    print("Testing MLX Fusion-Based Kernels Evaluator...")
    
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__),  "fusion_based_initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("\nEvaluation Results:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Fusion program not found at {initial_program_path}")
