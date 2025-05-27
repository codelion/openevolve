"""
Enhanced MLX Fine-tuning Evaluator with Robust Reward Hacking Detection

This enhanced evaluator includes comprehensive detection mechanisms for:
- MLX API errors and warnings
- Suspicious performance improvements
- Fallback loss values
- Exact percentage patterns
- Training failure detection
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
import re
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from contextlib import redirect_stdout, redirect_stderr


def run_baseline_if_needed() -> Dict[str, Any]:
    """Run baseline training if results don't exist"""
    
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


def detect_mlx_api_errors(captured_output: str) -> Tuple[bool, str]:
    """
    Detect MLX API errors and warnings in captured output
    
    Returns:
        (is_valid, error_message)
    """
    # Separate critical errors from warnings
    critical_error_patterns = [
        # MLX API misuse - these are real errors
        (r"mx\.tree_flatten", "Illegal use of mx.tree_flatten (doesn't exist in MLX)"),
        (r"mx\.tree_map", "Illegal use of mx.tree_map (doesn't exist in MLX)"),
        (r"has_aux=True", "Illegal use of has_aux parameter (not supported in MLX)"),
        
        # Complete failures
        (r"gradient.*is None", "Gradient computation returned None"),
        (r"failed.*gradient", "Gradient computation failed"),
        (r"failed.*loss", "Loss computation failed"),
        (r"Training.*failed", "Training explicitly failed"),
        (r"Error.*training", "Training error detected"),
        (r"Exception.*training", "Training exception detected"),
        
        # Memory/array errors that prevent training
        (r"memory.*error", "Memory allocation error"),
        (r"array.*error", "Array operation error"),
        (r"shape.*mismatch", "Array shape mismatch"),
    ]
    
    # Warning patterns - indicate issues but training may still work
    warning_patterns = [
        (r"Warning.*mx\.eval.*None", "MLX eval warnings detected"),
        (r"mx\.eval returned None", "MLX eval returned None warnings"),
        (r"loss.*is None", "Loss computation warnings"),
    ]
    
    # Check for critical errors first
    for pattern, message in critical_error_patterns:
        if re.search(pattern, captured_output, re.IGNORECASE):
            return False, f"MLX API Error: {message}"
    
    # Count warnings but don't fail immediately
    warning_count = 0
    warning_messages = []
    for pattern, message in warning_patterns:
        matches = re.findall(pattern, captured_output, re.IGNORECASE)
        if matches:
            warning_count += len(matches)
            warning_messages.append(f"{len(matches)}x {message}")
    
    # Allow some warnings but flag excessive warnings
    if warning_count > 10:  # Too many warnings indicate a serious problem
        return False, f"Excessive MLX warnings ({warning_count}): {'; '.join(warning_messages)}"
    elif warning_count > 0:
        print(f"⚠️ MLX Warnings detected ({warning_count}): {'; '.join(warning_messages)}")
        print("   Allowing program to continue but monitoring for improvements...")
    
    return True, "No critical MLX API errors detected"


def detect_fallback_patterns(optimization_results: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Detect patterns indicating fallback to hardcoded values
    """
    # Check for exact fallback loss values
    opt_final_loss = optimization_results.get("final_loss", 999.0)
    
    # Suspicious fallback values that clearly indicate broken training
    SUSPICIOUS_FALLBACK_VALUES = [999.0, 999999.0, 0.0]  # Removed 2.0, 10.0 as they can be reasonable
    
    for fallback in SUSPICIOUS_FALLBACK_VALUES:
        if abs(opt_final_loss - fallback) < 1e-6:
            return False, f"Loss appears to be obvious fallback value: {opt_final_loss} (exactly {fallback})"
    
    # Check for other suspicious exact values
    tokens_per_sec = optimization_results.get("tokens_per_second", 0.0)
    
    # Very suspiciously round numbers
    if tokens_per_sec > 0 and tokens_per_sec == int(tokens_per_sec) and tokens_per_sec % 1000 == 0:
        if tokens_per_sec > 5000:  # Very round numbers above 5000 are suspicious
            return False, f"Suspiciously round tokens_per_sec: {tokens_per_sec} (likely fallback)"
    
    # Check for unreasonable loss values
    if opt_final_loss > 100.0:  # Cross-entropy loss should rarely be this high
        return False, f"Unreasonably high loss value: {opt_final_loss} (likely fallback or broken training)"
    
    return True, "No obvious fallback patterns detected"


def detect_suspicious_improvements(optimization_results: Dict[str, Any], 
                                 baseline_results: Dict[str, Any],
                                 is_initial_program: bool = False) -> Tuple[bool, str]:
    """
    Enhanced detection of suspicious performance improvements
    """
    opt_tokens_per_sec = optimization_results.get("tokens_per_second", 0.0)
    baseline_tokens_per_sec = baseline_results.get("tokens_per_second", 1.0)
    
    opt_memory_efficiency = optimization_results.get("memory_efficiency", 0.0)
    baseline_memory_efficiency = baseline_results.get("memory_efficiency", 0.001)
    
    # More lenient thresholds for initial program (since it's essentially the same as baseline)
    if is_initial_program:
        MAX_REASONABLE_SPEED_IMPROVEMENT = 20.0  # 20x max for initial program
        MAX_REASONABLE_MEMORY_EFFICIENCY_IMPROVEMENT = 50.0  # 50x max for initial program
        print(f"🔍 Using lenient thresholds for initial program comparison")
    else:
        # Stringent thresholds for evolved programs
        MAX_REASONABLE_SPEED_IMPROVEMENT = 5.0  # 5x max (was 50x)
        MAX_REASONABLE_MEMORY_EFFICIENCY_IMPROVEMENT = 10.0  # 10x max (was 100x)
    
    # Check speed improvements
    if baseline_tokens_per_sec > 0:
        speed_ratio = opt_tokens_per_sec / baseline_tokens_per_sec
        if speed_ratio > MAX_REASONABLE_SPEED_IMPROVEMENT:
            return False, f"Unrealistic speed improvement: {speed_ratio:.1f}x (max reasonable: {MAX_REASONABLE_SPEED_IMPROVEMENT}x)"
        
        # Check for exact suspicious ratios (but be more lenient for initial program)
        suspicious_ratios = [100.0] if is_initial_program else [10.0, 11.0, 100.0]
        if speed_ratio in suspicious_ratios:
            return False, f"Suspiciously exact speed ratio: {speed_ratio:.1f}x"
    
    # Check memory efficiency improvements
    if baseline_memory_efficiency > 0:
        memory_ratio = opt_memory_efficiency / baseline_memory_efficiency
        if memory_ratio > MAX_REASONABLE_MEMORY_EFFICIENCY_IMPROVEMENT:
            return False, f"Unrealistic memory efficiency improvement: {memory_ratio:.1f}x (max reasonable: {MAX_REASONABLE_MEMORY_EFFICIENCY_IMPROVEMENT}x)"
        
        # Check for exact suspicious ratios
        suspicious_ratios = [100.0] if is_initial_program else [10.0, 11.0, 100.0]
        if memory_ratio in suspicious_ratios:
            return False, f"Suspiciously exact memory efficiency ratio: {memory_ratio:.1f}x"
    
    return True, "Improvements appear reasonable"


def detect_exact_percentage_patterns(optimization_results: Dict[str, Any],
                                   baseline_results: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Detect suspiciously exact percentage improvements (like exactly 1000%)
    """
    metrics_to_check = [
        ("tokens_per_second", "speed"),
        ("memory_efficiency", "memory efficiency"),
    ]
    
    for metric, display_name in metrics_to_check:
        opt_value = optimization_results.get(metric, 0.0)
        baseline_value = baseline_results.get(metric, 1.0)
        
        if baseline_value > 0 and opt_value > 0:
            improvement_ratio = opt_value / baseline_value
            improvement_percent = (improvement_ratio - 1.0) * 100
            
            # Check for exact suspicious percentages
            SUSPICIOUS_EXACT_PERCENTAGES = [
                1000.0,  # Exactly 1000%
                999.0,   # Close to 1000%
                500.0,   # Exactly 500%
                200.0,   # Exactly 200%
                100.0,   # Exactly 100%
            ]
            
            for suspicious_pct in SUSPICIOUS_EXACT_PERCENTAGES:
                if abs(improvement_percent - suspicious_pct) < 0.1:  # Very close to exact percentage
                    return False, f"Suspiciously exact {display_name} improvement: {improvement_percent:.1f}% (exactly {suspicious_pct}%)"
    
    return True, "No exact percentage patterns detected"


def detect_training_progression_issues(optimization_results: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Detect issues with training progression (e.g., no actual learning happening)
    """
    # Check if training stats show progression
    training_stats = optimization_results.get("training_stats", [])
    
    if not training_stats:
        return False, "No training statistics available - indicates training didn't run properly"
    
    # Check if loss values are all the same (indicating no learning)
    if len(training_stats) > 1:
        loss_values = [stat.get("loss", 999.0) for stat in training_stats]
        loss_values = [loss for loss in loss_values if loss < 900.0]  # Filter out obvious fallbacks
        
        if len(loss_values) > 1:
            loss_variance = np.var(loss_values)
            if loss_variance < 1e-10:  # All losses are essentially identical
                return False, f"All loss values identical: {loss_values[0]:.6f} (no learning occurred)"
    
    # Check final loss reasonableness
    final_loss = optimization_results.get("final_loss", 999.0)
    if final_loss > 50.0:  # Cross-entropy loss should rarely be this high
        return False, f"Unreasonably high final loss: {final_loss:.4f} (training likely failed)"
    
    return True, "Training progression appears normal"


def capture_output_and_evaluate(program, baseline_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Run evaluation while capturing all output to detect errors
    """
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    results = {}
    captured_output = ""
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Get optimization configuration from the evolved program
            config = program.get_optimization_config()
            
            # Benchmark the optimization patterns (this is where errors typically occur)
            results = program.benchmark_optimization_patterns(config, baseline_results)
        
        # Get captured output
        captured_output = stdout_capture.getvalue() + stderr_capture.getvalue()
        
    except Exception as e:
        # If the evaluation itself failed, that's definitely suspicious
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"Evaluation crashed: {str(e)}"
        }
    
    # Now run all our enhanced detection mechanisms
    
    # 1. Check for MLX API errors in output
    mlx_valid, mlx_message = detect_mlx_api_errors(captured_output)
    if not mlx_valid:
        print(f"🚨 MLX API ERROR DETECTED: {mlx_message}")
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"MLX API Error: {mlx_message}"
        }
    
    # 2. Check for fallback patterns
    fallback_valid, fallback_message = detect_fallback_patterns(results)
    if not fallback_valid:
        print(f"🚨 FALLBACK PATTERN DETECTED: {fallback_message}")
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"Fallback pattern: {fallback_message}"
        }
    
    # 3. Check for suspicious improvements
    improvement_valid, improvement_message = detect_suspicious_improvements(results, baseline_results)
    if not improvement_valid:
        print(f"🚨 SUSPICIOUS IMPROVEMENT DETECTED: {improvement_message}")
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"Suspicious improvement: {improvement_message}"
        }
    
    # 4. Check for exact percentage patterns
    percentage_valid, percentage_message = detect_exact_percentage_patterns(results, baseline_results)
    if not percentage_valid:
        print(f"🚨 EXACT PERCENTAGE PATTERN DETECTED: {percentage_message}")
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"Exact percentage pattern: {percentage_message}"
        }
    
    # 5. Check training progression
    progression_valid, progression_message = detect_training_progression_issues(results)
    if not progression_valid:
        print(f"🚨 TRAINING PROGRESSION ISSUE DETECTED: {progression_message}")
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"Training progression issue: {progression_message}"
        }
    
    # If we get here, add some basic sanity checks
    if "error" in results:
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -10.0,
            "error": results["error"]
        }
    
    # If all checks pass, calculate fitness conservatively
    baseline_tokens_per_sec = baseline_results.get("tokens_per_second", 1.0)
    baseline_memory_efficiency = baseline_results.get("memory_efficiency", 0.001)
    baseline_final_loss = baseline_results.get("final_loss", 2.0)
    
    opt_tokens_per_sec = results.get("tokens_per_second", 0.0)
    opt_memory_efficiency = results.get("memory_efficiency", 0.0)
    opt_final_loss = results.get("final_loss", 999.0)
    
    # Conservative improvement calculations
    speed_improvement = 0.0
    memory_improvement = 0.0
    loss_improvement = 0.0
    
    if baseline_tokens_per_sec > 0 and opt_tokens_per_sec > 0:
        speed_improvement = min((opt_tokens_per_sec - baseline_tokens_per_sec) / baseline_tokens_per_sec, 2.0)  # Cap at 200%
    
    if baseline_memory_efficiency > 0 and opt_memory_efficiency > 0:
        memory_improvement = min((opt_memory_efficiency - baseline_memory_efficiency) / baseline_memory_efficiency, 3.0)  # Cap at 300%
    
    if baseline_final_loss > 0 and opt_final_loss < 50.0:
        loss_improvement = (baseline_final_loss - opt_final_loss) / baseline_final_loss
        loss_improvement = max(-1.0, min(loss_improvement, 1.0))  # Cap between -100% and 100%
    
    # Conservative fitness calculation
    fitness = 0.1  # Base fitness for working solutions
    
    # Add conservative bonuses
    if speed_improvement > 0:
        fitness += min(speed_improvement * 0.3, 0.5)  # Max 0.5 bonus for speed
    
    if memory_improvement > 0:
        fitness += min(memory_improvement * 0.2, 0.3)  # Max 0.3 bonus for memory
    
    if loss_improvement > 0:
        fitness += min(loss_improvement * 0.4, 0.4)  # Max 0.4 bonus for loss
    
    # Penalty for degraded loss
    if opt_final_loss > baseline_final_loss * 1.1:  # More than 10% worse loss
        fitness -= 0.5
    
    fitness = max(-10.0, min(fitness, 2.0))  # Conservative fitness range
    
    print(f"✅ Enhanced validation PASSED:")
    print(f"   Speed improvement: {speed_improvement:.2%} (capped)")
    print(f"   Memory improvement: {memory_improvement:.2%} (capped)")
    print(f"   Loss improvement: {loss_improvement:.2%}")
    print(f"   Conservative fitness: {fitness:.4f}")
    
    # Return enhanced results
    enhanced_results = {
        "memory_efficiency": float(opt_memory_efficiency),
        "training_speed": float(opt_tokens_per_sec),
        "final_loss": float(opt_final_loss),
        "speed_improvement": float(speed_improvement),
        "memory_efficiency_improvement": float(memory_improvement),
        "loss_improvement": float(loss_improvement),
        "overall_fitness": float(fitness),
        "validation_passed": True,
        "conservative_scoring": True,
    }
    
    # Add original results for completeness
    enhanced_results.update(results)
    enhanced_results["overall_fitness"] = float(fitness)  # Override with conservative fitness
    
    return enhanced_results


def enhanced_evaluate_optimization_patterns(program, baseline_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Enhanced evaluation with comprehensive reward hacking detection
    """
    try:
        # Validate configuration first
        config = program.get_optimization_config()
        
        is_valid, validation_message = validate_optimization_config(config)
        if not is_valid:
            return {
                "memory_efficiency": 0.0,
                "training_speed": 0.0,
                "overall_fitness": -10.0,
                "error": f"Invalid configuration: {validation_message}"
            }
        
        print(f"🔍 Running ENHANCED evaluation with comprehensive detection...")
        print(f"Evaluating config: {json.dumps(config, indent=2)}")
        
        # Run evaluation with output capture and enhanced detection
        results = capture_output_and_evaluate(program, baseline_results)
        
        return results
        
    except Exception as e:
        print(f"Enhanced evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"Enhanced evaluation crashed: {str(e)}"
        }


# Main evaluation function
def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Enhanced evaluation function with robust reward hacking detection
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
                    "overall_fitness": -10.0,
                    "error": "Missing get_optimization_config function"
                }
            
            if not hasattr(program, 'benchmark_optimization_patterns'):
                return {
                    "memory_efficiency": 0.0,
                    "training_speed": 0.0,
                    "overall_fitness": -10.0,
                    "error": "Missing benchmark_optimization_patterns function"
                }
            
            # Get baseline results
            baseline_results = run_baseline_if_needed()
            
            # Force garbage collection before evaluation
            gc.collect()
            
            # Run enhanced evaluation
            results = enhanced_evaluate_optimization_patterns(program, baseline_results)
            
            # Log results
            print(f"\n📊 ENHANCED Evaluation Results:")
            print(f"  Overall fitness: {results.get('overall_fitness', 0.0):.4f}")
            print(f"  Validation passed: {results.get('validation_passed', False)}")
            print(f"  Conservative scoring: {results.get('conservative_scoring', False)}")
            
            if "error" in results:
                print(f"  ❌ Error: {results['error']}")
            
            return results
            
        finally:
            # Clean up sys.path
            if program_dir in sys.path:
                sys.path.remove(program_dir)
        
    except Exception as e:
        print(f"Enhanced evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            "memory_efficiency": 0.0,
            "training_speed": 0.0,
            "overall_fitness": -100.0,
            "error": f"Enhanced evaluation crashed: {str(e)}"
        }


# Stage evaluations for compatibility
def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """Stage 1: Quick validation with enhanced checks"""
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
                "memory_efficient_gradient_accumulation", 
                "get_optimization_config",
                "benchmark_optimization_patterns"
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
    """Stage 2: Full evaluation with enhanced detection"""
    return evaluate(program_path)


# For compatibility
def evaluate_detailed(program_path: str) -> Dict[str, Any]:
    """Alias for main evaluate function"""
    return evaluate(program_path)


if __name__ == "__main__":
    # Test the enhanced evaluator
    print("🔍 Enhanced MLX Fine-tuning Evaluator")
    print("=" * 50)
    
    import sys
    
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
    else:
        program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    print(f"Testing enhanced evaluator with {program_path}")
    
    # Test enhanced evaluation
    results = evaluate(program_path)
    print(f"\nEnhanced evaluation results: {results}")
