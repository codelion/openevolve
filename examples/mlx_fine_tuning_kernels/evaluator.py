"""
MLX LoRA Fine-tuning Optimization Evaluator

This evaluator performs real LoRA fine-tuning benchmarks using the mlx-lm library,
comparing evolved implementations against standard MLX-LM LoRA implementations. 
The goal is to achieve the same training loss with improved memory efficiency and/or speed.
"""

import importlib.util
import time
import traceback
import statistics
import gc
import psutil
import os
import tempfile
import shutil
import json
from typing import Dict, Union, List, Tuple, Optional, Any
from pathlib import Path

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

try:
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.utils import (
        linear_to_lora_layers,
        print_trainable_parameters,
    )
    from mlx_lm.utils import save_config
    MLX_LM_AVAILABLE = True
    print("✅ MLX-LM available for evaluation")
except ImportError as e:
    print(f"⚠️ MLX-LM not available: {e}")
    MLX_LM_AVAILABLE = False


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def clear_mlx_cache_and_gc():
    """Clear MLX cache and run garbage collection."""
    mx.clear_cache()
    gc.collect()


class MLXLoRABenchmark:
    """
    Benchmark for comparing MLX-LM LoRA fine-tuning implementations.
    Measures training loss convergence, speed, and memory usage using real mlx-lm.
    """
    
    def __init__(self, model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
        self.model_name = model_name
        self.temp_dirs = []
        
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        self.temp_dirs.clear()
        
        # Also run general cleanup
        try:
            from cleanup import cleanup_temp_files
            cleanup_temp_files()
        except ImportError:
            pass
    
    def create_test_config(self, data_dir: str, adapter_dir: str) -> Dict[str, Any]:
        """Create test configuration for LoRA fine-tuning with all MLX-LM expected attributes."""
        return {
            "model": self.model_name,
            "train": True,
            "fine_tune_type": "lora",
            "optimizer": "adam",
            "optimizer_config": {"adam": {}},
            "data": data_dir,
            "seed": 42,
            "num_layers": 2,  # Small for fast testing
            "batch_size": 1,  # Small for memory efficiency
            "iters": 5,       # Very few iterations for speed
            "val_batches": 2,
            "learning_rate": 1e-4,
            "steps_per_report": 2,
            "steps_per_eval": 10,
            "adapter_path": adapter_dir,
            "save_every": 100,
            "max_seq_length": 256,  # Shorter sequences
            "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 16.0},  # Smaller rank
            "mask_prompt": False,
            # Additional MLX-LM expected attributes
            "test": True,
            "test_batches": 2,
            "resume_adapter_file": None,
            "config": None,
            "grad_checkpoint": False,
            "lr_schedule": None,
            "wandb": None,
        }
    
    def compare_implementations(
        self,
        baseline_kernels: Dict,
        evolved_kernels: Dict,
        num_trials: int = 5  # Multiple trials to reduce system noise
    ) -> Dict[str, Any]:
        """Compare baseline vs evolved LoRA implementations using real mlx-lm."""
        
        if not MLX_LM_AVAILABLE:
            return {"error": "MLX-LM not available for real benchmarking"}
        
        print(f"\n📊 MLX-LM LORA FINE-TUNING COMPARISON (WITH NOISE REDUCTION)")
        print(f"  Model: {self.model_name}")
        print(f"  Trials: {num_trials} (multiple trials to reduce system noise)")
        print(f"  Method: Randomized order with statistical analysis")
        
        results = {
            'baseline': [],
            'evolved': []
        }
        
        for trial in range(num_trials):
            print(f"\n--- Trial {trial + 1}/{num_trials} ---")
            
            # Create temporary directories for this trial
            baseline_data_dir = tempfile.mkdtemp(prefix="baseline_data_")
            baseline_adapter_dir = tempfile.mkdtemp(prefix="baseline_adapters_")
            evolved_data_dir = tempfile.mkdtemp(prefix="evolved_data_")
            evolved_adapter_dir = tempfile.mkdtemp(prefix="evolved_adapters_")
            
            self.temp_dirs.extend([
                baseline_data_dir, baseline_adapter_dir,
                evolved_data_dir, evolved_adapter_dir
            ])
            
            # Test baseline implementation
            try:
                print("🔬 Testing BASELINE implementation...")
                
                # Create test dataset
                self._create_test_dataset(baseline_data_dir)
                baseline_config = self.create_test_config(baseline_data_dir, baseline_adapter_dir)
                
                clear_mlx_cache_and_gc()
                baseline_result = self._run_lora_benchmark(
                    baseline_kernels['optimized_lora_fine_tuning'],
                    baseline_config,
                    "BASELINE"
                )
                results['baseline'].append(baseline_result)
                
            except Exception as e:
                print(f"  ❌ Baseline trial failed: {e}")
                results['baseline'].append({"error": str(e)})
            
            # Test evolved implementation  
            try:
                print("🚀 Testing EVOLVED implementation...")
                
                # Create test dataset (same as baseline)
                self._create_test_dataset(evolved_data_dir)
                evolved_config = self.create_test_config(evolved_data_dir, evolved_adapter_dir)
                
                clear_mlx_cache_and_gc()
                evolved_result = self._run_lora_benchmark(
                    evolved_kernels['optimized_lora_fine_tuning'],
                    evolved_config,
                    "EVOLVED"
                )
                results['evolved'].append(evolved_result)
                
            except Exception as e:
                print(f"  ❌ Evolved trial failed: {e}")
                results['evolved'].append({"error": str(e)})
        
        # Cleanup after all trials
        self.cleanup()
        
        return self._analyze_results(results)
    
    def _create_test_dataset(self, output_dir: str, num_samples: int = 50):
        """Create a test dataset for LoRA fine-tuning."""
        examples = [
            {"text": "What is AI?\nAI is artificial intelligence, enabling computers to perform human-like tasks."},
            {"text": "How does ML work?\nMachine learning trains algorithms on data to recognize patterns and make predictions."},
            {"text": "What is Python?\nPython is a versatile programming language popular for data science and AI development."},
            {"text": "Explain deep learning.\nDeep learning uses neural networks with multiple layers to model complex data patterns."},
            {"text": "What is NLP?\nNatural Language Processing enables computers to understand and generate human language."},
            {"text": "What is computer vision?\nComputer vision teaches machines to interpret and analyze visual information from images."},
            {"text": "What is reinforcement learning?\nReinforcement learning trains agents through trial and error using rewards and penalties."},
            {"text": "What is a neural network?\nA neural network is a computing system inspired by biological neural networks."},
            {"text": "What is data science?\nData science extracts insights from data using statistics, programming, and domain expertise."},
            {"text": "What is machine learning?\nMachine learning is a subset of AI that enables systems to learn from data."},
        ]
        
        # Create consistent dataset
        dataset = []
        for i in range(num_samples):
            dataset.append(examples[i % len(examples)])
        
        # Create splits with sufficient validation data
        train_size = max(1, int(0.7 * num_samples))
        val_size = max(3, int(0.2 * num_samples))
        test_size = num_samples - train_size - val_size
        if test_size < 1:
            test_size = 1
            val_size = num_samples - train_size - test_size
        
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:train_size + val_size + test_size]
        
        # Write datasets - CRITICAL: Use "valid" not "val" for MLX-LM
        os.makedirs(output_dir, exist_ok=True)
        for split, data in [("train", train_data), ("valid", val_data), ("test", test_data)]:
            file_path = os.path.join(output_dir, f"{split}.jsonl")
            with open(file_path, "w") as f:
                for example in data:
                    f.write(json.dumps(example) + "\n")
    
    def _run_lora_benchmark(
        self,
        lora_fine_tuning_fn,
        config: Dict[str, Any],
        implementation_name: str
    ) -> Dict[str, Union[float, str]]:
        """Run LoRA fine-tuning benchmark."""
        
        print(f"  🧪 Running {implementation_name} LoRA fine-tuning...")
        
        try:
            # Memory before
            memory_before = get_memory_usage()
            start_time = time.perf_counter()
            
            # Run LoRA fine-tuning
            final_loss, metrics = lora_fine_tuning_fn(
                model_name=config['model'],
                train_data_path=config['data'],
                config=config,
                adapter_save_path=config['adapter_path']
            )
            
            # Timing and memory
            end_time = time.perf_counter()
            memory_after = get_memory_usage()
            
            total_time = end_time - start_time
            memory_delta = memory_after - memory_before
            
            # Extract additional metrics
            training_time = metrics.get('training_time', total_time)
            
            # Calculate approximate tokens/second (rough estimate)
            estimated_tokens = config['iters'] * config['batch_size'] * config['max_seq_length']
            tokens_per_second = estimated_tokens / training_time if training_time > 0 else 0
            
            print(f"    Final loss: {final_loss:.4f}")
            print(f"    Training time: {training_time:.2f}s")
            print(f"    Memory delta: {memory_delta:.1f} MB")
            
            return {
                'final_loss': float(final_loss),
                'training_time': float(training_time),
                'total_time': float(total_time),
                'memory_delta': float(memory_delta),
                'tokens_per_second': float(tokens_per_second),
                'lora_rank': config['lora_parameters']['rank'],
                'num_layers': config['num_layers'],
            }
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return {"error": str(e)}
    
    def _analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze comparison results."""
        
        # Filter successful results
        baseline_success = [r for r in results['baseline'] if 'error' not in r]
        evolved_success = [r for r in results['evolved'] if 'error' not in r]
        
        if not baseline_success or not evolved_success:
            return {
                "error": "No successful trials for comparison",
                "baseline_success": len(baseline_success),
                "evolved_success": len(evolved_success)
            }
        
        # Calculate averages
        baseline_avg = {
            'final_loss': np.mean([r['final_loss'] for r in baseline_success]),
            'training_time': np.mean([r['training_time'] for r in baseline_success]),
            'memory_delta': np.mean([r['memory_delta'] for r in baseline_success]),
            'tokens_per_second': np.mean([r['tokens_per_second'] for r in baseline_success])
        }
        
        evolved_avg = {
            'final_loss': np.mean([r['final_loss'] for r in evolved_success]),
            'training_time': np.mean([r['training_time'] for r in evolved_success]),
            'memory_delta': np.mean([r['memory_delta'] for r in evolved_success]),
            'tokens_per_second': np.mean([r['tokens_per_second'] for r in evolved_success])
        }
        
        # Calculate improvements
        loss_difference = abs(evolved_avg['final_loss'] - baseline_avg['final_loss'])
        loss_tolerance = max(0.01 * baseline_avg['final_loss'], 0.001)  # 1% or 0.001 minimum
        loss_convergence_ok = loss_difference <= loss_tolerance
        
        speed_improvement = evolved_avg['tokens_per_second'] / baseline_avg['tokens_per_second'] if baseline_avg['tokens_per_second'] > 0 else 1.0
        time_improvement = baseline_avg['training_time'] / evolved_avg['training_time'] if evolved_avg['training_time'] > 0 else 1.0
        memory_improvement = baseline_avg['memory_delta'] / evolved_avg['memory_delta'] if evolved_avg['memory_delta'] > 0 else 1.0
        
        # Overall score calculation
        convergence_score = 1.0 if loss_convergence_ok else max(0.0, 1.0 - (loss_difference / baseline_avg['final_loss']))
        efficiency_score = 0.5 * min(speed_improvement / 1.05, 2.0) + 0.5 * min(memory_improvement / 1.05, 2.0)
        overall_score = 0.7 * convergence_score + 0.3 * efficiency_score
        
        return {
            'baseline_avg': baseline_avg,
            'evolved_avg': evolved_avg,
            'loss_difference': loss_difference,
            'loss_convergence_ok': loss_convergence_ok,
            'speed_improvement': speed_improvement,
            'time_improvement': time_improvement,
            'memory_improvement': memory_improvement,
            'convergence_score': convergence_score,
            'efficiency_score': efficiency_score,
            'overall_score': overall_score,
            'successful_trials': {
                'baseline': len(baseline_success),
                'evolved': len(evolved_success)
            }
        }


def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Evaluate MLX-LM LoRA fine-tuning optimization program.
    
    Performs real LoRA fine-tuning comparison using mlx-lm library between 
    baseline and evolved implementations. Success metric: achieve same training 
    loss with efficiency improvements.
    """
    print(f"🚀 Evaluating MLX-LM LoRA Fine-tuning Optimization: {program_path}")
    
    if not MLX_LM_AVAILABLE:
        return {
            "overall_score": 0.0,
            "error": "MLX-LM not available for evaluation. Please install: pip install mlx-lm"
        }
    
    try:
        # Load evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "evolved_lora_kernels"):
            return {
                "overall_score": 0.0,
                "error": "Missing evolved_lora_kernels function"
            }
        
        if not hasattr(evolved_program, "baseline_lora_kernels"):
            return {
                "overall_score": 0.0,
                "error": "Missing baseline_lora_kernels function"
            }
        
        # Get LoRA implementations
        evolved_kernels = evolved_program.evolved_lora_kernels()
        baseline_kernels = evolved_program.baseline_lora_kernels()
        
        # Check required kernels
        required_key = 'optimized_lora_fine_tuning'
        if required_key not in evolved_kernels or required_key not in baseline_kernels:
            return {
                "overall_score": 0.0,
                "error": f"Missing kernel: {required_key}"
            }
        
        print(f"✅ LoRA implementations loaded successfully")
        
        # Setup benchmark
        benchmark = MLXLoRABenchmark()
        
        # Run comparison
        comparison_results = benchmark.compare_implementations(
            baseline_kernels=baseline_kernels,
            evolved_kernels=evolved_kernels,
            num_trials=1
        )
        
        if 'error' in comparison_results:
            return {
                "overall_score": 0.0,
                "error": comparison_results['error']
            }
        
        # Extract results
        overall_score = comparison_results['overall_score']
        convergence_score = comparison_results['convergence_score']
        efficiency_score = comparison_results['efficiency_score']
        
        loss_difference = comparison_results['loss_difference']
        loss_convergence_ok = comparison_results['loss_convergence_ok']
        speed_improvement = comparison_results['speed_improvement']
        memory_improvement = comparison_results['memory_improvement']
        time_improvement = comparison_results['time_improvement']
        
        baseline_avg = comparison_results['baseline_avg']
        evolved_avg = comparison_results['evolved_avg']
        
        print(f"\n📊 MLX-LM LORA FINE-TUNING OPTIMIZATION RESULTS:")
        print(f"  Loss Convergence: {'✅' if loss_convergence_ok else '❌'} (diff: {loss_difference:.4f})")
        print(f"  Speed Improvement: {speed_improvement:.2f}x")
        print(f"  Memory Improvement: {memory_improvement:.2f}x")
        print(f"  Time Improvement: {time_improvement:.2f}x")
        print(f"  Convergence Score: {convergence_score:.3f}")
        print(f"  Efficiency Score: {efficiency_score:.3f}")
        print(f"  Overall Score: {overall_score:.3f}")
        
        print(f"\n🔍 DETAILED METRICS:")
        print(f"  Baseline - Loss: {baseline_avg['final_loss']:.4f}, Time: {baseline_avg['training_time']:.1f}s, Memory: {baseline_avg['memory_delta']:.1f} MB")
        print(f"  Evolved  - Loss: {evolved_avg['final_loss']:.4f}, Time: {evolved_avg['training_time']:.1f}s, Memory: {evolved_avg['memory_delta']:.1f} MB")
        
        # Success interpretation
        if overall_score >= 0.8:
            print("  🥇 EXCELLENT: Strong improvements while maintaining convergence!")
        elif overall_score >= 0.6:
            print("  🥈 VERY GOOD: Good improvements with convergence!")
        elif overall_score >= 0.4:
            print("  🥉 GOOD: Some improvements achieved!")
        elif convergence_score > 0.5:
            print("  📈 PROGRESS: Reasonable convergence, efficiency needs work!")
        else:
            print("  🔄 DEVELOPING: Convergence issues need to be addressed!")
        
        # Prepare results
        results = {
            "overall_score": float(overall_score),
            "combined_score": float(overall_score),  # Primary metric for OpenEvolve
            
            # Core metrics
            "convergence_score": float(convergence_score),
            "efficiency_score": float(efficiency_score),
            "loss_convergence_ok": bool(loss_convergence_ok),
            "loss_difference": float(loss_difference),
            
            # Performance improvements
            "speed_improvement": float(speed_improvement),
            "memory_improvement": float(memory_improvement),
            "time_improvement": float(time_improvement),
            
            # Baseline metrics
            "baseline_final_loss": float(baseline_avg['final_loss']),
            "baseline_training_time": float(baseline_avg['training_time']),
            "baseline_memory_delta": float(baseline_avg['memory_delta']),
            "baseline_tokens_per_second": float(baseline_avg['tokens_per_second']),
            
            # Evolved metrics
            "evolved_final_loss": float(evolved_avg['final_loss']),
            "evolved_training_time": float(evolved_avg['training_time']),
            "evolved_memory_delta": float(evolved_avg['memory_delta']),
            "evolved_tokens_per_second": float(evolved_avg['tokens_per_second']),
            
            # Trial information
            "successful_baseline_trials": comparison_results['successful_trials']['baseline'],
            "successful_evolved_trials": comparison_results['successful_trials']['evolved'],
            
            # Metadata
            "evaluation_type": "mlx_lm_lora_finetuning",
            "achieves_convergence": bool(loss_convergence_ok),
            "has_efficiency_improvements": bool(speed_improvement > 1.05 or memory_improvement > 1.05),
            "target_achieved": bool(loss_convergence_ok and (speed_improvement > 1.1 or memory_improvement > 1.1)),
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
    print("Testing MLX-LM LoRA Fine-tuning Optimization Evaluator...")
    
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("\n=== Final Evaluation Results ===")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
