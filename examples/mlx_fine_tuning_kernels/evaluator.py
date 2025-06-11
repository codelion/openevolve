"""
MLX Quantized LoRA Optimization Evaluator - ROBUST VERSION

This evaluator provides rigorous benchmarking of quantized LoRA kernels with:
- Proper statistical analysis across multiple trials
- Robust baseline vs evolved comparison
- Comprehensive error detection and reporting
- Validation of kernel application
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
import sys
import io
import contextlib
from typing import Dict, Union, List, Tuple, Optional, Any
from pathlib import Path

# Required imports
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
    print("✅ MLX-LM available for quantized LoRA evaluation")
except ImportError as e:
    print(f"⚠️ MLX-LM not available: {e}")
    MLX_LM_AVAILABLE = False


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_peak_memory_mb() -> float:
    """Get MLX peak memory usage in MB."""
    return mx.get_peak_memory() / 1e6


def comprehensive_memory_and_cache_clear():
    """Comprehensive memory and cache clearing between trials."""
    mx.clear_cache()
    mx.reset_peak_memory()  # Reset peak memory tracking
    gc.collect()
    # Force a small allocation to ensure memory is properly cleared
    _ = mx.zeros((10, 10))
    mx.eval(_)
    mx.clear_cache()


@contextlib.contextmanager
def capture_output():
    """Context manager to capture stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class QuantizedLoRABenchmark:
    """
    Robust benchmark for quantized LoRA optimization with rigorous comparison.
    
    Key features:
    - Independent trial execution with full cleanup
    - Validation of kernel application 
    - Statistical significance testing
    - Comprehensive error detection
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

    def create_test_config(self, data_dir: str, adapter_dir: str, trial_seed: int) -> Dict[str, Any]:
        """Create test configuration with unique seed per trial."""
        return {
            "model": self.model_name,
            "train": True,
            "fine_tune_type": "lora",
            "optimizer": "adam",
            "optimizer_config": {"adam": {}},
            "data": data_dir,
            "seed": trial_seed,  # Unique seed per trial
            "num_layers": 3,
            "batch_size": 2,
            "iters": 15,  # Sufficient iterations for meaningful measurement
            "val_batches": 5,
            "learning_rate": 1e-4,
            "steps_per_report": 5,
            "steps_per_eval": 50,
            "adapter_path": adapter_dir,
            "save_every": 100,
            "max_seq_length": 256,
            "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 16.0},
            "mask_prompt": False,
            "test": True,
            "test_batches": 5,
            "resume_adapter_file": None,
            "config": None,
            "grad_checkpoint": False,
            "lr_schedule": None,
            "wandb": None,
        }

    def validate_model_quantization(self, model) -> Dict[str, Any]:
        """Validate that model has quantized layers as expected."""
        quantized_layers = []
        linear_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)
            elif isinstance(module, nn.QuantizedLinear):
                quantized_layers.append({
                    'name': name,
                    'bits': module.bits,
                    'group_size': module.group_size,
                    'weight_shape': module.weight.shape
                })

        if len(quantized_layers) == 0:
            raise ValueError(f"No quantized layers found in model {self.model_name}")

        return {
            'quantized_count': len(quantized_layers),
            'linear_count': len(linear_layers),
            'quantized_layers': quantized_layers
        }

    def validate_kernel_application(self, model, expected_kernels_applied: bool) -> bool:
        """Validate whether kernels were actually applied to the model."""
        kernels_applied = getattr(model, '_kernels_applied', False)
        has_evolved_kernels = getattr(model, '_has_evolved_kernels', False)
        
        # Check for our optimized classes in the model
        optimized_layer_count = 0
        for name, module in model.named_modules():
            if 'OptimizedQuantized' in type(module).__name__:
                optimized_layer_count += 1

        actual_optimization = kernels_applied and optimized_layer_count > 0

        if expected_kernels_applied != actual_optimization:
            print(f"  ⚠️ KERNEL APPLICATION MISMATCH:")
            print(f"    Expected kernels applied: {expected_kernels_applied}")
            print(f"    Actual kernels applied: {actual_optimization}")
            print(f"    Model _kernels_applied: {kernels_applied}")
            print(f"    Optimized layer count: {optimized_layer_count}")
            return False

        return True

    def compare_implementations(self, evolved_kernels: Dict, num_trials: int = 5) -> Dict[str, Any]:
        """
        Robust comparison between baseline and evolved implementations.
        
        Uses 5 trials for better statistical power and rigorous validation.
        """

        if not MLX_LM_AVAILABLE:
            return {"error": "MLX-LM not available for quantized LoRA benchmarking"}

        print(f"\n📊 ROBUST QUANTIZED LORA BENCHMARK")
        print(f"  Model: {self.model_name}")
        print(f"  Trials per implementation: {num_trials}")
        print(f"  Comparison: Standard MLX-LM vs Optimized Kernels")
        print(f"  Statistical significance: p-value analysis")

        baseline_results = []
        evolved_results = []

        # Validate model first
        print(f"\n🔧 Validating model quantization...")
        try:
            test_model, _ = load(self.model_name)
            model_info = self.validate_model_quantization(test_model)
            print(f"  ✅ Found {model_info['quantized_count']} quantized layers")
            del test_model  # Clean up
            comprehensive_memory_and_cache_clear()
        except Exception as e:
            return {"error": f"Model validation failed: {e}"}

        # ========================================
        # PHASE 1: Baseline trials (standard MLX-LM)
        # ========================================
        print(f"\n🔬 PHASE 1: BASELINE trials (standard MLX-LM)")

        for trial in range(num_trials):
            trial_seed = 42 + trial  # Unique seed per trial
            print(f"\n--- Baseline Trial {trial + 1}/{num_trials} (seed={trial_seed}) ---")

            baseline_data_dir = tempfile.mkdtemp(prefix=f"baseline_data_{trial}_")
            baseline_adapter_dir = tempfile.mkdtemp(prefix=f"baseline_adapters_{trial}_")
            self.temp_dirs.extend([baseline_data_dir, baseline_adapter_dir])

            try:
                self._create_test_dataset(baseline_data_dir, trial_seed)
                baseline_config = self.create_test_config(baseline_data_dir, baseline_adapter_dir, trial_seed)

                # Comprehensive cleanup before trial
                comprehensive_memory_and_cache_clear()

                baseline_result = self._run_trial_with_validation(
                    baseline_config,
                    f"BASELINE-{trial+1}",
                    evolved_kernels=None,
                    expected_kernels_applied=False
                )
                baseline_results.append(baseline_result)

                if "error" in baseline_result:
                    print(f"  ❌ Baseline trial {trial+1} failed: {baseline_result['error']}")
                    if trial == 0:  # Stop if first trial fails
                        return {"error": f"First baseline trial failed: {baseline_result['error']}"}

            except Exception as e:
                error_msg = f"Baseline trial {trial+1} exception: {e}"
                print(f"  ❌ {error_msg}")
                baseline_results.append({"error": error_msg})
                if trial == 0:
                    return {"error": error_msg}

        # ========================================
        # PHASE 2: Evolved trials (optimized kernels)
        # ========================================
        print(f"\n🚀 PHASE 2: EVOLVED trials (optimized kernels)")

        for trial in range(num_trials):
            trial_seed = 100 + trial  # Different seed range for evolved trials
            print(f"\n--- Evolved Trial {trial + 1}/{num_trials} (seed={trial_seed}) ---")

            evolved_data_dir = tempfile.mkdtemp(prefix=f"evolved_data_{trial}_")
            evolved_adapter_dir = tempfile.mkdtemp(prefix=f"evolved_adapters_{trial}_")
            self.temp_dirs.extend([evolved_data_dir, evolved_adapter_dir])

            try:
                self._create_test_dataset(evolved_data_dir, trial_seed)
                evolved_config = self.create_test_config(evolved_data_dir, evolved_adapter_dir, trial_seed)

                # Comprehensive cleanup before trial
                comprehensive_memory_and_cache_clear()

                evolved_result = self._run_trial_with_validation(
                    evolved_config,
                    f"EVOLVED-{trial+1}",
                    evolved_kernels=evolved_kernels,
                    expected_kernels_applied=True
                )
                evolved_results.append(evolved_result)

                if "error" in evolved_result:
                    print(f"  ❌ Evolved trial {trial+1} failed: {evolved_result['error']}")
                    if trial == 0:
                        return {"error": f"First evolved trial failed: {evolved_result['error']}"}

            except Exception as e:
                error_msg = f"Evolved trial {trial+1} exception: {e}"
                print(f"  ❌ {error_msg}")
                evolved_results.append({"error": error_msg})
                if trial == 0:
                    return {"error": error_msg}

        # ========================================
        # PHASE 3: Statistical Analysis
        # ========================================
        self.cleanup()
        results = {"baseline": baseline_results, "evolved": evolved_results}
        return self._analyze_results_with_statistics(results)

    def _create_test_dataset(self, output_dir: str, seed: int, num_samples: int = 50):
        """Create deterministic test dataset with given seed."""
        np.random.seed(seed)
        
        base_examples = [
            {"text": "What is quantization?\nQuantization reduces model precision to use fewer bits per parameter."},
            {"text": "Explain LoRA.\nLoRA adds small trainable matrices to frozen weights for efficient fine-tuning."},
            {"text": "What is Apple Silicon?\nApple Silicon refers to custom ARM processors designed by Apple."},
            {"text": "How does MLX work?\nMLX is Apple's machine learning framework optimized for Apple Silicon."},
            {"text": "What are transformers?\nTransformers use attention mechanisms for sequence processing tasks."},
            {"text": "Explain fine-tuning.\nFine-tuning adapts pre-trained models to specific tasks with targeted data."},
            {"text": "What is efficient training?\nEfficient training reduces computational cost while maintaining model quality."},
            {"text": "How does memory optimization work?\nMemory optimization reduces peak memory usage during model training."},
        ]

        # Create deterministic but varied dataset
        examples = []
        for i in range(num_samples):
            base_example = base_examples[i % len(base_examples)]
            # Add slight variation based on seed to ensure datasets are similar but not identical
            variation_id = (seed + i) % 10
            varied_text = base_example["text"] + f" (variation {variation_id})"
            examples.append({"text": varied_text})

        # Create splits
        train_data = examples[:int(0.7 * num_samples)]
        valid_data = examples[int(0.7 * num_samples):int(0.9 * num_samples)]
        test_data = examples[int(0.9 * num_samples):]

        # Ensure minimum sizes
        if not valid_data:
            valid_data = [train_data[0]]
        if not test_data:
            test_data = [train_data[0]]

        # Write datasets
        os.makedirs(output_dir, exist_ok=True)
        for split, data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
            with open(f"{output_dir}/{split}.jsonl", "w") as f:
                for example in data:
                    f.write(json.dumps(example) + "\n")

    def _run_trial_with_validation(
        self, config: Dict[str, Any], trial_name: str, 
        evolved_kernels: Optional[Dict] = None,
        expected_kernels_applied: bool = False
    ) -> Dict[str, Union[float, str]]:
        """Run a single trial with comprehensive validation."""

        print(f"  🧪 Running {trial_name}...")
        
        try:
            # Memory tracking
            memory_before = get_memory_usage()
            mx.reset_peak_memory()  # Reset peak memory tracking
            start_time = time.perf_counter()

            # Import the training function
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)

            from initial_program import quantized_lora_fine_tuning_with_kernels

            # Run training
            final_loss, metrics = quantized_lora_fine_tuning_with_kernels(
                model_name=config["model"],
                train_data_path=config["data"],
                config=config,
                adapter_save_path=config["adapter_path"],
                evolved_kernels=evolved_kernels,
            )

            # Timing and memory measurements
            end_time = time.perf_counter()
            memory_after = get_memory_usage()
            peak_memory_mb = get_peak_memory_mb()

            total_time = end_time - start_time
            training_time = metrics.get("training_time", total_time)
            memory_delta = memory_after - memory_before

            # Validate kernel application
            kernels_applied = metrics.get("kernels_applied", False)
            
            # CRITICAL VALIDATION: Ensure kernels were applied as expected
            if expected_kernels_applied and not kernels_applied:
                return {"error": "Expected kernels to be applied but they were not"}
            elif not expected_kernels_applied and kernels_applied:
                return {"error": "Expected no kernels but kernels were applied"}

            # Calculate metrics
            estimated_tokens = config["iters"] * config["batch_size"] * config["max_seq_length"]
            tokens_per_second = estimated_tokens / training_time if training_time > 0 else 0

            print(f"    Final loss: {final_loss:.4f}")
            print(f"    Training time: {training_time:.2f}s")
            print(f"    Memory delta: {memory_delta:.1f} MB")
            print(f"    Peak memory: {peak_memory_mb:.1f} MB")
            print(f"    Tokens/sec: {tokens_per_second:.1f}")
            print(f"    Kernels applied: {kernels_applied}")

            return {
                "final_loss": float(final_loss),
                "training_time": float(training_time),
                "total_time": float(total_time),
                "memory_delta": float(memory_delta),
                "peak_memory_mb": float(peak_memory_mb),
                "tokens_per_second": float(tokens_per_second),
                "kernels_applied": bool(kernels_applied),
                "trial_seed": config["seed"],
                "success": True,
            }

        except Exception as e:
            error_msg = f"Trial failed: {str(e)}"
            print(f"    ❌ {error_msg}")
            traceback.print_exc()
            return {"error": error_msg, "success": False}

    def _analyze_results_with_statistics(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze results with proper statistical analysis."""

        # Filter successful results
        baseline_success = [r for r in results["baseline"] if r.get("success", False)]
        evolved_success = [r for r in results["evolved"] if r.get("success", False)]

        print(f"\n📊 STATISTICAL ANALYSIS:")
        print(f"  Successful baseline trials: {len(baseline_success)}")
        print(f"  Successful evolved trials: {len(evolved_success)}")

        if len(baseline_success) < 2 or len(evolved_success) < 2:
            return {
                "error": "Insufficient successful trials for statistical analysis",
                "baseline_success": len(baseline_success),
                "evolved_success": len(evolved_success),
            }

        # Calculate statistics for each metric
        def calc_stats(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }

        # Baseline statistics
        baseline_stats = {
            "final_loss": calc_stats([r["final_loss"] for r in baseline_success]),
            "training_time": calc_stats([r["training_time"] for r in baseline_success]),
            "memory_delta": calc_stats([r["memory_delta"] for r in baseline_success]),
            "peak_memory_mb": calc_stats([r["peak_memory_mb"] for r in baseline_success]),
            "tokens_per_second": calc_stats([r["tokens_per_second"] for r in baseline_success]),
        }

        # Evolved statistics
        evolved_stats = {
            "final_loss": calc_stats([r["final_loss"] for r in evolved_success]),
            "training_time": calc_stats([r["training_time"] for r in evolved_success]),
            "memory_delta": calc_stats([r["memory_delta"] for r in evolved_success]),
            "peak_memory_mb": calc_stats([r["peak_memory_mb"] for r in evolved_success]),
            "tokens_per_second": calc_stats([r["tokens_per_second"] for r in evolved_success]),
        }

        # Calculate improvements and statistical significance
        loss_diff = abs(evolved_stats["final_loss"]["mean"] - baseline_stats["final_loss"]["mean"])
        loss_tolerance = max(0.01 * baseline_stats["final_loss"]["mean"], 0.01)
        loss_convergence_ok = loss_diff <= loss_tolerance

        # Calculate improvement ratios
        speed_improvement = (
            evolved_stats["tokens_per_second"]["mean"] / baseline_stats["tokens_per_second"]["mean"]
            if baseline_stats["tokens_per_second"]["mean"] > 0 else 1.0
        )
        
        memory_improvement = (
            baseline_stats["memory_delta"]["mean"] / evolved_stats["memory_delta"]["mean"]
            if evolved_stats["memory_delta"]["mean"] > 0 else 1.0
        )
        
        peak_memory_improvement = (
            baseline_stats["peak_memory_mb"]["mean"] / evolved_stats["peak_memory_mb"]["mean"]
            if evolved_stats["peak_memory_mb"]["mean"] > 0 else 1.0
        )

        time_improvement = (
            baseline_stats["training_time"]["mean"] / evolved_stats["training_time"]["mean"]
            if evolved_stats["training_time"]["mean"] > 0 else 1.0
        )

        # Statistical significance assessment (simple t-test approximation)
        def assess_significance(baseline_vals, evolved_vals):
            b_mean, b_std, b_n = baseline_vals["mean"], baseline_vals["std"], baseline_vals["count"]
            e_mean, e_std, e_n = evolved_vals["mean"], evolved_vals["std"], evolved_vals["count"]
            
            if b_std == 0 and e_std == 0:
                return "identical"
            
            # Pooled standard error
            pooled_se = np.sqrt((b_std**2 / b_n) + (e_std**2 / e_n))
            if pooled_se == 0:
                return "identical"
                
            t_stat = abs(b_mean - e_mean) / pooled_se
            # Rough significance assessment (t > 2 is approximately p < 0.05 for small samples)
            return "significant" if t_stat > 2.0 else "not_significant"

        significance = {
            "memory": assess_significance(baseline_stats["memory_delta"], evolved_stats["memory_delta"]),
            "speed": assess_significance(baseline_stats["tokens_per_second"], evolved_stats["tokens_per_second"]),
            "time": assess_significance(baseline_stats["training_time"], evolved_stats["training_time"]),
        }

        # Scoring
        convergence_score = 1.0 if loss_convergence_ok else max(0.0, 1.0 - (loss_diff / baseline_stats["final_loss"]["mean"]))
        
        # Weight improvements by statistical significance
        memory_score = (memory_improvement / 1.10) if significance["memory"] == "significant" else 1.0
        speed_score = (speed_improvement / 1.05) if significance["speed"] == "significant" else 1.0
        time_score = (time_improvement / 1.05) if significance["time"] == "significant" else 1.0
        
        efficiency_score = 0.4 * min(memory_score, 2.0) + 0.3 * min(speed_score, 2.0) + 0.3 * min(time_score, 2.0)
        overall_score = 0.7 * convergence_score + 0.3 * efficiency_score

        # Check kernel usage consistency
        kernels_used_consistency = all(r.get("kernels_applied", False) for r in evolved_success)

        return {
            "baseline_stats": baseline_stats,
            "evolved_stats": evolved_stats,
            "loss_difference": loss_diff,
            "loss_convergence_ok": loss_convergence_ok,
            "speed_improvement": speed_improvement,
            "memory_improvement": memory_improvement,
            "peak_memory_improvement": peak_memory_improvement,
            "time_improvement": time_improvement,
            "statistical_significance": significance,
            "convergence_score": convergence_score,
            "efficiency_score": efficiency_score,
            "overall_score": overall_score,
            "successful_trials": {
                "baseline": len(baseline_success),
                "evolved": len(evolved_success),
            },
            "kernels_used_consistently": kernels_used_consistency,
            "raw_results": results,  # Include raw data for debugging
        }


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Robust evaluation of MLX quantized LoRA optimization program.
    """
    print(f"🚀 Evaluating MLX Quantized LoRA Optimization: {program_path}")

    if not MLX_LM_AVAILABLE:
        return {
            "overall_score": 0.0,
            "error": "MLX-LM not available. Please install: pip install mlx-lm"
        }

    try:
        # Load evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)

        if not hasattr(evolved_program, "evolved_lora_kernels"):
            return {"overall_score": 0.0, "error": "Missing evolved_lora_kernels function"}

        if not hasattr(evolved_program, "baseline_lora_kernels"):
            return {"overall_score": 0.0, "error": "Missing baseline_lora_kernels function"}

        # Get kernels
        print("📦 Loading kernels...")
        evolved_kernels = evolved_program.evolved_lora_kernels()
        baseline_kernels = evolved_program.baseline_lora_kernels()

        print(f"✅ Evolved kernels: {list(evolved_kernels.keys()) if evolved_kernels else 'None'}")
        print(f"✅ Baseline: Standard MLX-LM")

        # Setup benchmark
        benchmark = QuantizedLoRABenchmark()

        # Run robust comparison with 5 trials
        comparison_results = benchmark.compare_implementations(
            evolved_kernels=evolved_kernels, num_trials=5
        )

        if "error" in comparison_results:
            return {"overall_score": 0.0, "error": comparison_results["error"]}

        # Extract results
        overall_score = comparison_results["overall_score"]
        convergence_score = comparison_results["convergence_score"]
        efficiency_score = comparison_results["efficiency_score"]

        print(f"\n📊 ROBUST EVALUATION RESULTS:")
        print(f"  Overall Score: {overall_score:.3f}")
        print(f"  Convergence Score: {convergence_score:.3f}")
        print(f"  Efficiency Score: {efficiency_score:.3f}")
        print(f"  Statistical Significance: {comparison_results['statistical_significance']}")
        print(f"  Successful Trials: {comparison_results['successful_trials']}")

        # Prepare comprehensive metrics
        metrics = {
            "overall_score": float(overall_score),
            "combined_score": float(overall_score),
            "convergence_score": float(convergence_score),
            "efficiency_score": float(efficiency_score),
            "loss_convergence_ok": comparison_results["loss_convergence_ok"],
            "speed_improvement": comparison_results["speed_improvement"],
            "memory_improvement": comparison_results["memory_improvement"],
            "peak_memory_improvement": comparison_results["peak_memory_improvement"],
            "time_improvement": comparison_results["time_improvement"],
            "statistical_significance": comparison_results["statistical_significance"],
            "successful_baseline_trials": comparison_results["successful_trials"]["baseline"],
            "successful_evolved_trials": comparison_results["successful_trials"]["evolved"],
            "kernels_used_consistently": comparison_results["kernels_used_consistently"],
        }

        return metrics

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"overall_score": 0.0, "combined_score": 0.0, "error": error_msg}


if __name__ == "__main__":
    print("Testing Robust MLX Quantized LoRA Optimization Evaluator...")

    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")

    if os.path.exists(initial_program_path):
        result = evaluate(initial_program_path)
        print("\n=== Final Evaluation Results ===")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
