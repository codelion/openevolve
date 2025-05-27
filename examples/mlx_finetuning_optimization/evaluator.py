"""
Evaluator for MLX Fine-tuning Memory Optimization

This evaluator compares evolved optimization patterns against the baseline MLX fine-tuning
implementation. It measures improvements in memory efficiency, training speed, and
convergence quality.

Key metrics:
- Memory efficiency: tokens/second per MB memory used
- Training speed: tokens processed per second
- Memory usage: peak memory consumption
- Convergence quality: loss reduction and stability
- Overall fitness: combined metric for evolution
"""

import importlib.util
import json
import os
import time
import traceback
import psutil
import gc
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def load_baseline_results() -> Optional[Dict[str, Any]]:
    """Load baseline results if available"""
    baseline_results_path = os.path.join(
        os.path.dirname(__file__), 
        "baseline_output", 
        "training_results.json"
    )
    
    if os.path.exists(baseline_results_path):
        try:
            with open(baseline_results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load baseline results: {e}")
    
    return None


def run_baseline_if_needed() -> Dict[str, Any]:
    """Run baseline training if results don't exist"""
    
    # FIXED: Always regenerate baseline for consistency
    # The cached baseline results can be inconsistent due to different parameters
    print("Regenerating baseline results for consistency...")
    
    # Find baseline_finetuning.py with robust path handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_path = None
    
    search_paths = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.join(current_dir, 'examples', 'mlx_finetuning_optimization'),
        '/Users/asankhaya/Documents/GitHub/openevolve/examples/mlx_finetuning_optimization'
    ]
    
    for search_path in search_paths:
        potential_path = os.path.join(search_path, 'baseline_finetuning.py')
        if os.path.exists(potential_path):
            baseline_path = potential_path
            break
    
    if baseline_path is None:
        # Create a consistent default baseline result
        print("Baseline script not found. Using consistent default baseline results...")
        return {
            "tokens_per_second": 180.0,  # Reasonable and consistent baseline
            "memory_efficiency": 0.08,
            "peak_memory_mb": 1700.0,
            "total_time": 12.0,
            "final_loss": 2.0
        }
    
    spec = importlib.util.spec_from_file_location("baseline_finetuning", baseline_path)
    baseline_module = importlib.util.module_from_spec(spec)
    
    # Add the directory to sys.path for imports
    baseline_dir = os.path.dirname(baseline_path)
    sys_path_added = False
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)
        sys_path_added = True
    
    try:
        spec.loader.exec_module(baseline_module)
        
        # Create and run baseline trainer with CONSISTENT parameters
        trainer = baseline_module.BaselineTrainer("mlx-community/Qwen3-0.6B-bf16")
        trainer.config.batch_size = 2  # Consistent with evaluation
        trainer.config.num_epochs = 1
        trainer.config.sequence_length = 128  # Consistent with evaluation
        
        # Create consistent dataset for baseline (SAME SIZE as evaluation)
        dataset = trainer.create_sample_dataset(num_samples=10)  # Match evaluation size
        baseline_results = trainer.train(dataset, output_dir="./baseline_output")
        
        print("Baseline training completed with consistent parameters.")
        print(f"Baseline tokens/sec: {baseline_results.get('tokens_per_second', 0):.1f}")
        print(f"Baseline memory: {baseline_results.get('peak_memory_mb', 0):.1f}MB")
        print(f"Baseline loss: {baseline_results.get('final_loss', 0):.3f}")
        
    except Exception as e:
        print(f"Failed to run baseline: {e}")
        # Return consistent default baseline results
        baseline_results = {
            "tokens_per_second": 180.0,
            "memory_efficiency": 0.08,
            "peak_memory_mb": 1700.0,
            "total_time": 12.0,
            "final_loss": 2.0
        }
    finally:
        if sys_path_added and baseline_dir in sys.path:
            sys.path.remove(baseline_dir)
    
    return baseline_results


def safe_float_conversion(value, default=0.0):
    """Safely convert a value to float, handling infinity and NaN"""
    try:
        float_val = float(value)
        if np.isnan(float_val) or np.isinf(float_val):
            return default
        return float_val
    except (TypeError, ValueError, OverflowError):
        return default


def validate_training_metrics(optimization_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate training metrics to detect reward hacking patterns"""
    
    opt_final_loss = optimization_results.get("final_loss", 999.0)
    baseline_final_loss = baseline_results.get("final_loss", 2.0)
    
    # CRITICAL: Detect suspiciously low loss values that indicate reward hacking
    MINIMUM_REASONABLE_LOSS = 0.01  # Cross-entropy loss should rarely be this low
    if opt_final_loss < MINIMUM_REASONABLE_LOSS:
        return False, f"Suspiciously low loss detected: {opt_final_loss:.6f} (likely reward hacking)"
    
    # Check for exactly zero loss (common reward hacking pattern)
    if abs(opt_final_loss) < 1e-10:
        return False, f"Exact zero loss detected: {opt_final_loss} (reward hacking fallback pattern)"
    
    # Check for loss values that are unrealistically good
    if opt_final_loss < baseline_final_loss * 0.1:  # 10x better than baseline is suspicious
        return False, f"Unrealistically good loss: {opt_final_loss:.4f} vs baseline {baseline_final_loss:.4f} (>10x improvement suspicious)"
    
    # Check for performance metrics that are too good to be true
    opt_tokens_per_sec = optimization_results.get("tokens_per_second", 0.0)
    baseline_tokens_per_sec = baseline_results.get("tokens_per_second", 1.0)
    
    # FIXED: More lenient speed improvement detection (50x instead of 20x)
    # and allow for reasonable baseline variations
    speed_ratio = opt_tokens_per_sec / max(baseline_tokens_per_sec, 1.0)
    if speed_ratio > 50:  # 50x speed improvement is unrealistic
        return False, f"Unrealistic speed improvement: {opt_tokens_per_sec:.1f} vs {baseline_tokens_per_sec:.1f} tokens/sec (>{speed_ratio:.1f}x suspicious)"
    
    # FIXED: Don't flag reasonable performance differences that could be due to:
    # - Different dataset sizes
    # - Different sequence lengths
    # - Different batch sizes
    # - Different hardware states
    if speed_ratio > 2.0 and speed_ratio <= 20.0:
        print(f"ℹ️ Performance difference detected but within reasonable range: {speed_ratio:.1f}x vs baseline")
        print(f"   This could be due to dataset size, sequence length, or hardware differences")
    
    # Check memory efficiency improvements
    opt_memory_eff = optimization_results.get("memory_efficiency", 0.0)
    baseline_memory_eff = baseline_results.get("memory_efficiency", 0.001)
    
    if opt_memory_eff > baseline_memory_eff * 100:  # 100x memory efficiency is unrealistic
        return False, f"Unrealistic memory efficiency: {opt_memory_eff:.4f} vs {baseline_memory_eff:.4f} (>100x suspicious)"
    
    # Check for infinite or NaN values
    metrics_to_check = ["tokens_per_second", "memory_efficiency", "peak_memory_mb", "total_time"]
    for metric in metrics_to_check:
        value = optimization_results.get(metric, 0.0)
        if not np.isfinite(value):
            return False, f"Invalid {metric} value: {value} (NaN/Inf detected)"
    
    # Check for negative metrics that should be positive
    positive_metrics = ["tokens_per_second", "memory_efficiency", "peak_memory_mb", "total_time"]
    for metric in positive_metrics:
        value = optimization_results.get(metric, 0.0)
        if value <= 0:
            return False, f"Invalid {metric} value: {value} (should be positive)"
    
    # Check peak memory is reasonable (not too low)
    opt_peak_memory = optimization_results.get("peak_memory_mb", float('inf'))
    MINIMUM_REASONABLE_MEMORY = 100.0  # MB - any fine-tuning should use at least this much
    if opt_peak_memory < MINIMUM_REASONABLE_MEMORY:
        return False, f"Unrealistically low memory usage: {opt_peak_memory:.1f}MB (likely measurement error)"
    
    return True, "Metrics appear valid"


def detect_loss_scaling_hacks(optimization_results: Dict[str, Any]) -> Tuple[bool, str]:
    """Detect common loss scaling hacks in gradient accumulation"""
    
    # This is harder to detect directly, but we can look for patterns
    opt_final_loss = optimization_results.get("final_loss", 999.0)
    
    # FIXED: Only flag extremely suspicious patterns, not normal losses
    # A loss between 0.5 and 10.0 is reasonable for language modeling
    REASONABLE_LOSS_RANGE = (0.1, 15.0)  # Expanded reasonable range
    
    if not (REASONABLE_LOSS_RANGE[0] <= opt_final_loss <= REASONABLE_LOSS_RANGE[1]):
        # Only check for scaling hacks if the loss is outside reasonable range
        COMMON_SCALE_FACTORS = [2, 4, 8, 16, 32]  # Common accumulation step values
        
        for scale_factor in COMMON_SCALE_FACTORS:
            scaled_loss = opt_final_loss * scale_factor
            # If scaling by a common factor gives us a "normal" looking loss (1-5 range)
            # AND the original loss was suspiciously low (< 0.1), then flag it
            if opt_final_loss < 0.1 and 1.0 <= scaled_loss <= 5.0:
                return False, f"Loss appears artificially scaled: {opt_final_loss:.4f} * {scale_factor} = {scaled_loss:.4f} (possible gradient accumulation hack)"
    
    # Additional check: Flag exact multiples that suggest division hacks
    # But only if the loss is suspiciously low to begin with
    if opt_final_loss < 0.05:  # Only very low losses
        for scale_factor in [2, 4, 8, 16]:
            scaled_loss = opt_final_loss * scale_factor
            # Check if scaled loss is very close to a "normal" value
            normal_targets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            for target in normal_targets:
                if abs(scaled_loss - target) < 0.01:  # Very close match
                    return False, f"Suspiciously exact loss scaling: {opt_final_loss:.4f} * {scale_factor} ≈ {target:.1f}"
    
    return True, "No obvious loss scaling detected"


def validate_optimization_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that optimization configuration is reasonable"""
    
    # Check for reasonable values
    chunk_size = config.get("attention_chunk_size", 512)
    if chunk_size < 64 or chunk_size > 4096:
        return False, f"Invalid attention_chunk_size: {chunk_size}"
    
    chunk_size_ops = config.get("chunk_size", 1024)
    if chunk_size_ops < 128 or chunk_size_ops > 8192:
        return False, f"Invalid chunk_size: {chunk_size_ops}"
    
    gc_frequency = config.get("force_gc_frequency", 10)
    if gc_frequency < 1 or gc_frequency > 100:
        return False, f"Invalid force_gc_frequency: {gc_frequency}"
    
    # Check boolean values
    boolean_keys = [
        "use_chunked_attention", "use_fp16_compute", "fp32_gradients",
        "cast_inputs", "dynamic_padding", "pack_sequences", "sort_by_length",
        "fp16_embeddings", "fp16_attention", "fp16_ffn", "use_chunked_operations"
    ]
    
    for key in boolean_keys:
        if key in config and not isinstance(config[key], bool):
            return False, f"Invalid boolean value for {key}: {config[key]}"
    
    # Check memory balance
    cpu_gpu_balance = config.get("cpu_gpu_memory_balance", 0.7)
    if cpu_gpu_balance < 0.0 or cpu_gpu_balance > 1.0:
        return False, f"Invalid cpu_gpu_memory_balance: {cpu_gpu_balance}"
    
    return True, "Configuration appears valid"


def evaluate_optimization_patterns(program, baseline_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate evolved optimization patterns against baseline
    
    Returns metrics for evolution including relative improvements
    """
    
    try:
        # Get optimization configuration from the evolved program
        config = program.get_optimization_config()
        
        # Validate configuration
        is_valid, validation_message = validate_optimization_config(config)
        if not is_valid:
            return {
                "memory_efficiency": 0.0,
                "training_speed": 0.0,
                "memory_improvement": 0.0,
                "speed_improvement": 0.0,
                "final_loss": 999.0,  # Very bad loss
                "loss_ratio": 999.0,
                "overall_fitness": 0.0,
                "error": f"Invalid configuration: {validation_message}"
            }
        
        print(f"Evaluating optimization config: {json.dumps(config, indent=2)}")
        
        # Benchmark the optimization patterns
        optimization_results = program.benchmark_optimization_patterns(config, baseline_results)
        
        if "error" in optimization_results:
            return {
                "memory_efficiency": 0.0,
                "training_speed": 0.0,
                "memory_improvement": 0.0,
                "speed_improvement": 0.0,
                "final_loss": 999.0,
                "loss_ratio": 999.0,
                "overall_fitness": 0.0,
                "error": optimization_results["error"]
            }
        
        # CRITICAL: Validate training metrics to detect reward hacking
        metrics_valid, metrics_message = validate_training_metrics(optimization_results, baseline_results)
        if not metrics_valid:
            print(f"🚨 REWARD HACKING DETECTED: {metrics_message}")
            return {
                "memory_efficiency": 0.0,
                "training_speed": 0.0,
                "memory_improvement": -1.0,
                "speed_improvement": -1.0,
                "final_loss": 999.0,
                "loss_ratio": 999.0,
                "overall_fitness": -100.0,  # Severe penalty for reward hacking
                "error": f"Reward hacking detected: {metrics_message}"
            }
        
        # CRITICAL: Detect loss scaling hacks
        loss_scaling_valid, loss_scaling_message = detect_loss_scaling_hacks(optimization_results)
        if not loss_scaling_valid:
            print(f"🚨 LOSS SCALING HACK DETECTED: {loss_scaling_message}")
            return {
                "memory_efficiency": 0.0,
                "training_speed": 0.0,
                "memory_improvement": -1.0,
                "speed_improvement": -1.0,
                "final_loss": 999.0,
                "loss_ratio": 999.0,
                "overall_fitness": -50.0,  # Heavy penalty for loss scaling hacks
                "error": f"Loss scaling hack detected: {loss_scaling_message}"
            }
        
        # Calculate relative improvements
        baseline_tokens_per_sec = baseline_results.get("tokens_per_second", 1.0)
        baseline_memory_efficiency = baseline_results.get("memory_efficiency", 0.001)
        baseline_peak_memory = baseline_results.get("peak_memory_mb", 1000.0)
        baseline_total_time = baseline_results.get("total_time", 100.0)
        baseline_final_loss = baseline_results.get("final_loss", 2.0)  # CRITICAL: Add final loss
        
        opt_tokens_per_sec = optimization_results.get("tokens_per_second", 0.0)
        opt_memory_efficiency = optimization_results.get("memory_efficiency", 0.0)
        opt_peak_memory = optimization_results.get("peak_memory_mb", float('inf'))
        opt_total_time = optimization_results.get("total_time", float('inf'))
        opt_final_loss = optimization_results.get("final_loss", 999.0)  # CRITICAL: Add final loss
        
        # Calculate loss ratio (optimized loss / baseline loss)
        loss_ratio = opt_final_loss / baseline_final_loss if baseline_final_loss > 0 else 999.0
        
        # CRITICAL CONSTRAINT: Reject if final loss is significantly worse
        MAX_LOSS_DEGRADATION = 1.20  # Allow max 20% worse loss
        if loss_ratio > MAX_LOSS_DEGRADATION:
            print(f"❌ REJECTING optimization: Final loss too high!")
            print(f"   Baseline loss: {baseline_final_loss:.4f}")
            print(f"   Optimized loss: {opt_final_loss:.4f}")
            print(f"   Loss ratio: {loss_ratio:.2f} (max allowed: {MAX_LOSS_DEGRADATION})")
            
            return {
                "memory_efficiency": 0.0,
                "training_speed": 0.0,
                "memory_improvement": -1.0,
                "speed_improvement": -1.0,
                "final_loss": float(opt_final_loss),
                "loss_ratio": float(loss_ratio),
                "overall_fitness": -10.0,  # Heavy penalty
                "error": f"Final loss degraded too much: {loss_ratio:.2f}x vs baseline"
            }
        
        # Calculate percentage improvements
        speed_improvement = (opt_tokens_per_sec - baseline_tokens_per_sec) / baseline_tokens_per_sec if baseline_tokens_per_sec > 0 else 0.0
        memory_efficiency_improvement = (opt_memory_efficiency - baseline_memory_efficiency) / baseline_memory_efficiency if baseline_memory_efficiency > 0 else 0.0
        memory_usage_improvement = (baseline_peak_memory - opt_peak_memory) / baseline_peak_memory if baseline_peak_memory > 0 else 0.0
        time_improvement = (baseline_total_time - opt_total_time) / baseline_total_time if baseline_total_time > 0 else 0.0
        
        # Loss improvement (lower is better, so we want negative loss_ratio improvement)
        loss_improvement = (baseline_final_loss - opt_final_loss) / baseline_final_loss if baseline_final_loss > 0 else 0.0
        
        # Ensure improvements are reasonable (cap at 10x improvement to avoid outliers)
        speed_improvement = max(-0.9, min(speed_improvement, 10.0))
        memory_efficiency_improvement = max(-0.9, min(memory_efficiency_improvement, 10.0))
        memory_usage_improvement = max(-0.9, min(memory_usage_improvement, 0.9))  # Max 90% memory reduction
        time_improvement = max(-0.9, min(time_improvement, 0.9))  # Max 90% time reduction
        loss_improvement = max(-2.0, min(loss_improvement, 2.0))  # Loss can be 3x better or 2x worse
        
        # Calculate overall fitness with LOSS AS PRIMARY FACTOR
        fitness_components = {
            "loss_quality_score": loss_improvement * 0.5,           # 50% weight - MOST IMPORTANT
            "memory_efficiency_score": memory_efficiency_improvement * 0.2,  # 20% weight
            "speed_score": speed_improvement * 0.2,                 # 20% weight  
            "memory_usage_score": memory_usage_improvement * 0.1,   # 10% weight
        }
        
        overall_fitness = sum(fitness_components.values())
        
        # Add stability bonus/penalty
        if opt_peak_memory < float('inf') and opt_tokens_per_sec > 0 and opt_final_loss < 50.0:
            stability_bonus = 0.1
        else:
            stability_bonus = -0.5  # Heavy penalty for failed runs
        
        overall_fitness += stability_bonus
        
        # Add loss quality bonus for maintaining good learning
        if loss_ratio <= 1.05:  # Within 5% of baseline loss
            loss_quality_bonus = 0.2  # Bonus for maintaining learning quality
        elif loss_ratio <= 1.10:  # Within 10%
            loss_quality_bonus = 0.1
        else:
            loss_quality_bonus = 0.0
        
        overall_fitness += loss_quality_bonus
        
        # Normalize fitness to reasonable range
        overall_fitness = max(-10.0, min(overall_fitness, 5.0))
        
        print(f"✅ Optimization ACCEPTED:")
        print(f"   Final loss: {opt_final_loss:.4f} vs baseline {baseline_final_loss:.4f} (ratio: {loss_ratio:.2f})")
        print(f"   Speed: {speed_improvement:.1%} improvement")
        print(f"   Memory efficiency: {memory_efficiency_improvement:.1%} improvement")
        print(f"   Overall fitness: {overall_fitness:.4f}")
        
        return {
            "memory_efficiency": float(opt_memory_efficiency),
            "training_speed": float(opt_tokens_per_sec),
            "peak_memory_mb": float(opt_peak_memory),
            "total_time": float(opt_total_time),
            "final_loss": float(opt_final_loss),
            "loss_ratio": float(loss_ratio),
            "speed_improvement": float(speed_improvement),
            "memory_efficiency_improvement": float(memory_efficiency_improvement),
            "memory_usage_improvement": float(memory_usage_improvement),
            "time_improvement": float(time_improvement),
            "loss_improvement": float(loss_improvement),
            "overall_fitness": float(overall_fitness),
            "baseline_tokens_per_sec": float(baseline_tokens_per_sec),
            "baseline_memory_efficiency": float(baseline_memory_efficiency),
            "baseline_final_loss": float(baseline_final_loss),
            "config_valid": True,
            "fitness_components": fitness_components
        }
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "memory_improvement": 0.0,
            "speed_improvement": 0.0,
            "overall_fitness": 0.0,
            "error": str(e)
        }


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Main evaluation function for MLX fine-tuning optimization
    
    Compares evolved optimization patterns against baseline performance
    """
    
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        
        # Add the directory to sys.path for imports
        program_dir = os.path.dirname(program_path)
        if program_dir not in sys.path:
            sys.path.insert(0, program_dir)
        
        try:
            spec.loader.exec_module(program)
            
            # Check required functions exist
            if not hasattr(program, 'get_optimization_config'):
                return {
                    "memory_efficiency": 0.0,
                    "training_speed": 0.0,
                    "overall_fitness": 0.0,
                    "error": "Missing get_optimization_config function"
                }
            
            if not hasattr(program, 'benchmark_optimization_patterns'):
                return {
                    "memory_efficiency": 0.0,
                    "training_speed": 0.0,
                    "overall_fitness": 0.0,
                    "error": "Missing benchmark_optimization_patterns function"
                }
            
            # Ensure baseline results are available
            baseline_results = run_baseline_if_needed()
            
            # Force garbage collection before evaluation
            gc.collect()
            
            # Evaluate the optimization patterns
            results = evaluate_optimization_patterns(program, baseline_results)
            
            # Log key metrics
            print(f"Evaluation results:")
            print(f"  Overall fitness: {results.get('overall_fitness', 0.0):.4f}")
            print(f"  Speed improvement: {results.get('speed_improvement', 0.0):.2%}")
            print(f"  Memory efficiency improvement: {results.get('memory_efficiency_improvement', 0.0):.2%}")
            print(f"  Memory usage improvement: {results.get('memory_usage_improvement', 0.0):.2%}")
            
            if "fitness_components" in results:
                print(f"  Fitness components: {results['fitness_components']}")
            
            return results
            
        finally:
            # Clean up sys.path
            if program_dir in sys.path:
                sys.path.remove(program_dir)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": 0.0,
            "error": str(e)
        }


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """
    Stage 1 evaluation: Quick validation to filter out broken configurations
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        
        # Add directory to path
        program_dir = os.path.dirname(program_path)
        if program_dir not in sys.path:
            sys.path.insert(0, program_dir)
        
        try:
            spec.loader.exec_module(program)
            
            # Check required functions exist
            if not hasattr(program, 'get_optimization_config'):
                return {"config_valid": 0.0, "error": "Missing get_optimization_config function"}
            
            # Get configuration and validate
            config = program.get_optimization_config()
            is_valid, validation_message = validate_optimization_config(config)
            
            if not is_valid:
                return {
                    "config_valid": 0.0,
                    "stage1_score": 0.0,
                    "error": f"Invalid configuration: {validation_message}"
                }
            
            # Quick validation of required optimization functions
            required_functions = [
                "chunked_attention_forward",
                "memory_efficient_gradient_accumulation", 
                "optimized_batch_preparation",
                "adaptive_mixed_precision_forward"
            ]
            
            missing_functions = [func for func in required_functions if not hasattr(program, func)]
            
            if missing_functions:
                return {
                    "config_valid": 0.5,
                    "stage1_score": 0.5,
                    "error": f"Missing optimization functions: {missing_functions}"
                }
            
            return {
                "config_valid": 1.0,
                "stage1_score": 1.0,
                "functions_present": True
            }
            
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)
        
    except Exception as e:
        return {"config_valid": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """
    Stage 2 evaluation: Full evaluation with baseline comparison
    """
    return evaluate(program_path)


# For compatibility with evaluation cascade
def evaluate_detailed(program_path: str) -> Dict[str, Any]:
    """Alias for main evaluate function"""
    return evaluate(program_path)


if __name__ == "__main__":
    # Test the evaluator
    import sys
    
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
    else:
        program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    print(f"Testing evaluator with {program_path}")
    
    # Test stage 1 evaluation
    print("\n=== Stage 1 Evaluation ===")
    stage1_results = evaluate_stage1(program_path)
    print(f"Stage 1 results: {stage1_results}")
    
    if stage1_results.get("config_valid", 0) > 0.5:
        # Test full evaluation
        print("\n=== Full Evaluation ===")
        results = evaluate(program_path)
        print(f"Full results: {results}")
    else:
        print("Skipping full evaluation due to stage 1 failure")
