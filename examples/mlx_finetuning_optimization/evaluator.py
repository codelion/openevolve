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
    baseline_results = load_baseline_results()
    
    if baseline_results is None:
        print("Baseline results not found. Running baseline training...")
        
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
            # Create a default baseline result for evaluation to continue
            print("Baseline script not found. Using default baseline results...")
            return {
                "tokens_per_second": 150.0,  # Reasonable baseline
                "memory_efficiency": 0.08,
                "peak_memory_mb": 1800.0,
                "total_time": 15.0,
                "final_loss": 2.2
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
            
            # Create and run baseline trainer
            trainer = baseline_module.BaselineTrainer("mlx-community/Qwen3-0.6B-bf16")
            trainer.config.batch_size = 2  # Small batch for evaluation
            trainer.config.num_epochs = 1
            trainer.config.sequence_length = 256  # Match evaluation settings
            
            # Create small dataset for baseline
            dataset = trainer.create_sample_dataset(num_samples=20)  # Match evaluation size
            baseline_results = trainer.train(dataset, output_dir="./baseline_output")
            
            print("Baseline training completed.")
            
        except Exception as e:
            print(f"Failed to run baseline: {e}")
            # Return default baseline results
            baseline_results = {
                "tokens_per_second": 150.0,
                "memory_efficiency": 0.08,
                "peak_memory_mb": 1800.0,
                "total_time": 15.0,
                "final_loss": 2.2
            }
        finally:
            if sys_path_added and baseline_dir in sys.path:
                sys.path.remove(baseline_dir)
    else:
        print("Using cached baseline results.")
    
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
                "overall_fitness": 0.0,
                "error": optimization_results["error"]
            }
        
        # Calculate relative improvements
        baseline_tokens_per_sec = baseline_results.get("tokens_per_second", 1.0)
        baseline_memory_efficiency = baseline_results.get("memory_efficiency", 0.001)
        baseline_peak_memory = baseline_results.get("peak_memory_mb", 1000.0)
        baseline_total_time = baseline_results.get("total_time", 100.0)
        
        opt_tokens_per_sec = optimization_results.get("tokens_per_second", 0.0)
        opt_memory_efficiency = optimization_results.get("memory_efficiency", 0.0)
        opt_peak_memory = optimization_results.get("peak_memory_mb", float('inf'))
        opt_total_time = optimization_results.get("total_time", float('inf'))
        
        # Calculate percentage improvements
        speed_improvement = (opt_tokens_per_sec - baseline_tokens_per_sec) / baseline_tokens_per_sec if baseline_tokens_per_sec > 0 else 0.0
        memory_efficiency_improvement = (opt_memory_efficiency - baseline_memory_efficiency) / baseline_memory_efficiency if baseline_memory_efficiency > 0 else 0.0
        memory_usage_improvement = (baseline_peak_memory - opt_peak_memory) / baseline_peak_memory if baseline_peak_memory > 0 else 0.0
        time_improvement = (baseline_total_time - opt_total_time) / baseline_total_time if baseline_total_time > 0 else 0.0
        
        # Ensure improvements are reasonable (cap at 10x improvement to avoid outliers)
        speed_improvement = max(-0.9, min(speed_improvement, 10.0))
        memory_efficiency_improvement = max(-0.9, min(memory_efficiency_improvement, 10.0))
        memory_usage_improvement = max(-0.9, min(memory_usage_improvement, 0.9))  # Max 90% memory reduction
        time_improvement = max(-0.9, min(time_improvement, 0.9))  # Max 90% time reduction
        
        # Calculate overall fitness with emphasis on memory efficiency (key constraint for Mac users)
        # Positive improvements should increase fitness, negative should decrease it
        fitness_components = {
            "memory_efficiency_score": memory_efficiency_improvement * 0.4,  # 40% weight
            "speed_score": speed_improvement * 0.25,                        # 25% weight  
            "memory_usage_score": memory_usage_improvement * 0.25,          # 25% weight
            "time_score": time_improvement * 0.1                            # 10% weight
        }
        
        overall_fitness = sum(fitness_components.values())
        
        # Add stability bonus/penalty
        if opt_peak_memory < float('inf') and opt_tokens_per_sec > 0:
            stability_bonus = 0.1
        else:
            stability_bonus = -0.5  # Heavy penalty for failed runs
        
        overall_fitness += stability_bonus
        
        # Normalize fitness to reasonable range
        overall_fitness = max(-1.0, min(overall_fitness, 5.0))
        
        return {
            "memory_efficiency": float(opt_memory_efficiency),
            "training_speed": float(opt_tokens_per_sec),
            "peak_memory_mb": float(opt_peak_memory),
            "total_time": float(opt_total_time),
            "speed_improvement": float(speed_improvement),
            "memory_efficiency_improvement": float(memory_efficiency_improvement),
            "memory_usage_improvement": float(memory_usage_improvement),
            "time_improvement": float(time_improvement),
            "overall_fitness": float(overall_fitness),
            "baseline_tokens_per_sec": float(baseline_tokens_per_sec),
            "baseline_memory_efficiency": float(baseline_memory_efficiency),
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
