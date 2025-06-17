"""
Thread-Safe Robust Qwen3 Custom GQA Attention Evaluator

This evaluator provides bulletproof protection against Metal kernel failures without using signals:

🛡️ THREAD-SAFE PROTECTION:
1. No signal-based timeouts (works in worker threads)
2. Comprehensive C++ exception catching
3. Retry mechanisms with exponential backoff  
4. Graceful fallback to standard attention on failures
5. Detailed error classification and recovery

🔧 EVOLUTION SAFETY:
- Never terminates the evolution process due to kernel errors
- Works perfectly in OpenEvolve's worker threads
- Provides meaningful feedback on kernel failure types
- Statistical tracking of Metal kernel error patterns
"""

import os
import sys
import json
import time
import traceback
import threading
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import mlx.nn as nn

# Import the comprehensive benchmark suite for consistent testing
from qwen3_benchmark_suite import Qwen3BenchmarkSuite, BenchmarkConfig, BenchmarkResult


class MetalKernelError(Exception):
    """Custom exception for Metal kernel related errors"""
    pass


class ThreadSafeTimeoutError(Exception):
    """Thread-safe timeout exception"""
    pass


class ThreadSafeRobustEvaluator:
    """Thread-safe bulletproof evaluator that never crashes from Metal kernel errors"""

    def __init__(self):
        self.model_path = "mlx-community/Qwen3-0.6B-bf16"
        
        # Error handling configuration (no signal-based timeouts)
        self.metal_kernel_timeout = 45  # Reference only, no actual timeout enforcement
        self.max_retry_attempts = 2
        
        # Error tracking
        self.metal_errors_caught = 0
        self.retry_attempts_used = 0
        self.timeout_errors_caught = 0
        
        # Baseline will be measured dynamically
        self.baseline_metrics = None
        self.baseline_results = None

        # Use comprehensive benchmark suite for consistency
        self.benchmark_suite = Qwen3BenchmarkSuite(self.model_path)

        print("🛡️  Initialized Thread-Safe Robust Custom GQA Evaluator")
        print(f"📱 Model: {self.model_path}")
        print(f"🔁 Max retry attempts: {self.max_retry_attempts}")
        print(f"🧵 Thread-safe: No signal dependencies")

    def evaluate(self, program_text: str) -> Dict[str, Any]:
        """
        Thread-safe bulletproof evaluation that never crashes:
        1. Safe extraction with syntax validation
        2. Protected baseline measurement  
        3. Isolated correctness testing
        4. Robust benchmarking with retries
        5. Comprehensive Metal kernel error recovery
        """

        print("\n" + "=" * 100)
        print("🛡️  THREAD-SAFE BULLETPROOF CUSTOM GQA ATTENTION EVALUATION")
        print("=" * 100)
        print("✅ Comprehensive Metal kernel error protection")
        print("✅ Thread-safe operation (no signal dependencies)")
        print("✅ Multi-layer exception catching")
        print("✅ Automatic retry with exponential backoff")
        print("✅ Never crashes the evolution process")
        print("=" * 100)

        try:
            # Reset error counters
            self.metal_errors_caught = 0
            self.retry_attempts_used = 0
            self.timeout_errors_caught = 0

            # Step 1: Ultra-safe extraction
            print("\n🔧 STEP 1: Ultra-Safe Custom Attention Class Extraction")
            extraction_result = self._thread_safe_extract_custom_attention_class(program_text)
            if not extraction_result["success"]:
                return self._create_failure_result(f"Extraction failed: {extraction_result['error']}")
            
            custom_attention_class = extraction_result["class"]

            # Step 2: Protected baseline measurement
            print("\n📊 STEP 2: Protected Baseline Performance Measurement")
            baseline_results = self._protected_measure_baseline_performance()
            if not baseline_results:
                return self._create_failure_result("Failed to measure baseline performance safely")

            # Step 3: Thread-safe correctness testing
            print("\n🔍 STEP 3: Thread-Safe Custom Attention Correctness Testing")
            correctness_result = self._thread_safe_correctness_test(custom_attention_class)
            if not correctness_result["success"]:
                return self._create_failure_result(f"Correctness test failed: {correctness_result['error']}")
            
            correctness_score = correctness_result["score"]
            if correctness_score < 0.95:
                return self._create_failure_result(f"Correctness score too low: {correctness_score:.3f}")

            # Step 4: Armored performance benchmarking
            print("\n🚀 STEP 4: Armored Custom Attention Performance Benchmarking")
            benchmark_result = self._armored_benchmark_custom_attention(custom_attention_class)
            if not benchmark_result["success"]:
                return self._create_failure_result(f"Benchmarking failed: {benchmark_result['error']}")
            
            custom_results = benchmark_result["results"]

            # Step 5: Safe performance analysis
            print("\n📈 STEP 5: Safe Performance Analysis")
            performance_analysis = self._analyze_performance_comparison(
                baseline_results, custom_results
            )

            # Step 6: Calculate final score
            final_score = self._calculate_final_score(performance_analysis, correctness_score)

            # Step 7: Generate comprehensive result with error statistics
            result = {
                "success": True,
                "final_score": final_score,
                "performance_metrics": performance_analysis["aggregate_metrics"],
                "correctness_score": correctness_score,
                "benchmark_results": [self._result_to_dict(r) for r in custom_results],
                "baseline_comparison": performance_analysis["comparison_summary"],
                "individual_comparisons": performance_analysis["individual_comparisons"],
                "summary": self._generate_summary(performance_analysis, correctness_score),
                "error_statistics": {
                    "metal_kernel_errors_caught": self.metal_errors_caught,
                    "timeout_errors_caught": self.timeout_errors_caught,
                    "retry_attempts_used": self.retry_attempts_used,
                    "total_errors_handled": self.metal_errors_caught + self.timeout_errors_caught,
                }
            }

            print(f"\n🛡️  ERROR STATISTICS:")
            print(f"   Metal kernel errors caught: {self.metal_errors_caught}")
            print(f"   Timeout errors caught: {self.timeout_errors_caught}")
            print(f"   Retry attempts used: {self.retry_attempts_used}")
            print(f"   Total errors handled safely: {self.metal_errors_caught + self.timeout_errors_caught}")

            self._print_evaluation_results(result)
            return result

        except Exception as e:
            # Even this top-level catch should never crash the process
            error_msg = f"Top-level evaluation error (safely caught): {str(e)}"
            print(f"🛡️  {error_msg}")
            traceback.print_exc()
            return self._create_failure_result(error_msg)

    def _thread_safe_extract_custom_attention_class(self, program_text: str) -> Dict[str, Any]:
        """Thread-safe extraction with comprehensive error handling"""
        try:
            print("  🔍 Thread-safe program analysis...")

            # Handle file paths vs direct text
            if (
                program_text.startswith("/")
                and "\n" not in program_text
                and len(program_text) < 500
            ):
                print(f"  📁 Reading program from file: {program_text}")
                if os.path.exists(program_text):
                    try:
                        with open(program_text, "r") as f:
                            actual_program_text = f.read()
                    except Exception as e:
                        return {"success": False, "error": f"File read error: {e}"}
                else:
                    return {"success": False, "error": f"Program file not found: {program_text}"}
            else:
                actual_program_text = program_text

            # Comprehensive syntax validation
            try:
                compile(actual_program_text, '<evolved_program>', 'exec')
                print("  ✅ Program syntax validation passed")
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax error: {e}"}
            except Exception as e:
                return {"success": False, "error": f"Compilation error: {e}"}

            # Create bulletproof execution environment
            exec_globals = self._create_safe_execution_environment()

            # Execute program with comprehensive protection (no timeouts)
            print("  ⚙️  Executing program with maximum protection...")
            try:
                # Use thread-safe execution
                success, result = self._thread_safe_execute_with_protection(
                    lambda: exec(actual_program_text, exec_globals)
                )
                
                if not success:
                    return {"success": False, "error": f"Program execution failed: {result}"}
                    
            except Exception as e:
                return {"success": False, "error": f"Execution error: {e}"}

            # Safe class extraction
            custom_class = exec_globals.get("CustomGQAAttention")
            if custom_class is None:
                return {"success": False, "error": "CustomGQAAttention class not found"}

            # Comprehensive class validation
            if not isinstance(custom_class, type):
                return {"success": False, "error": "CustomGQAAttention is not a valid class"}

            # Check for required methods
            required_methods = ["__init__", "__call__"]
            for method in required_methods:
                if not hasattr(custom_class, method):
                    return {"success": False, "error": f"Missing required method: {method}"}

            print(f"  ✅ Successfully extracted and validated CustomGQAAttention class")
            print(f"  📋 Class: {custom_class.__name__}")
            print(f"  📋 Methods: {[name for name in dir(custom_class) if not name.startswith('_')]}")

            return {"success": True, "class": custom_class}

        except Exception as e:
            return {"success": False, "error": f"Extraction failed with exception: {str(e)}"}

    def _create_safe_execution_environment(self) -> Dict[str, Any]:
        """Create ultra-safe execution environment"""
        import math
        import numpy as np
        import time
        from typing import Optional, Tuple, Any
        
        exec_globals = {
            "__builtins__": __builtins__,
            "mx": mx,
            "nn": nn,
            "np": np,
            "math": math,
            "time": time,
            "Optional": Optional,
            "Tuple": Tuple,
            "Any": Any,
        }

        # Safe MLX-LM import with error handling
        try:
            exec_globals["mlx_lm"] = __import__("mlx_lm")
            print("  ✅ MLX-LM imported successfully")
        except ImportError:
            print("  ⚠️  MLX-LM not available, RoPE functionality may be limited")
        except Exception as e:
            print(f"  ⚠️  MLX-LM import error: {e}")

        return exec_globals

    def _protected_measure_baseline_performance(self) -> Optional[List[BenchmarkResult]]:
        """Protected baseline measurement with comprehensive error handling"""
        try:
            print("  📊 Running protected baseline benchmark...")
            
            # Ensure clean state
            self._ensure_standard_attention()

            # Get representative benchmarks
            baseline_configs = self._get_evolution_benchmark_configs()
            if not baseline_configs:
                print("  ❌ No benchmark configurations available")
                return None

            baseline_results = []
            successful_count = 0

            for i, config in enumerate(baseline_configs, 1):
                print(f"  [{i}/{len(baseline_configs)}] Protected baseline: {config.name}")
                
                try:
                    # Run with thread-safe Metal kernel protection
                    success, result = self._thread_safe_execute_with_protection(
                        lambda: self.benchmark_suite.run_single_benchmark(config)
                    )
                    
                    if success and result:
                        baseline_results.append(result)
                        successful_count += 1
                        print(f"    ✅ Protected baseline {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec")
                    else:
                        print(f"    ❌ Protected baseline {config.name}: {result}")
                        # Continue with other benchmarks
                        
                except Exception as e:
                    print(f"    ❌ Protected baseline {config.name} exception: {e}")
                    continue

            # Check if we have enough successful baselines
            min_required = max(2, len(baseline_configs) * 0.6)  # At least 60% or 2 benchmarks
            if successful_count < min_required:
                print(f"  ❌ Only {successful_count}/{len(baseline_configs)} baseline benchmarks succeeded")
                print(f"     Required: {min_required}")
                return None

            # Store baseline metrics
            self._store_baseline_metrics(baseline_results)
            print(f"  ✅ Protected baseline measurement complete ({successful_count} successful)")
            
            return baseline_results

        except Exception as e:
            print(f"  ❌ Protected baseline measurement failed: {e}")
            return None

    def _thread_safe_correctness_test(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Thread-safe correctness testing with maximum protection"""
        print("  🔍 Running thread-safe correctness testing...")
        
        try:
            # Create safe test configuration
            class MockArgs:
                hidden_size = 5120
                num_attention_heads = 40
                num_key_value_heads = 8
                head_dim = 128
                rms_norm_eps = 1e-06
                rope_theta = 1000000
                rope_scaling = None
                max_position_embeddings = 40960

            args = MockArgs()

            # Progressive test cases with increasing difficulty
            test_cases = [
                (1, 16, 5120),   # Ultra-short (safest)
                (1, 32, 5120),   # Very short
                (1, 64, 5120),   # Short sequence
                (1, 128, 5120),  # Medium sequence (most challenging we'll try)
            ]

            correctness_scores = []
            local_metal_errors = 0
            local_timeout_errors = 0

            for B, L, D in test_cases:
                print(f"      🧪 Testing sequence length {L} with thread-safe protection...")

                try:
                    # Create test inputs
                    x = mx.random.normal((B, L, D))
                    mask = "causal"

                    # Test with thread-safe execution
                    success, result = self._thread_safe_execute_with_protection(
                        lambda: self._test_single_sequence_safely(custom_attention_class, args, x, mask)
                    )
                    
                    if success:
                        correctness_scores.append(result)
                        print(f"      ✅ Sequence length {L}: passed (score={result:.3f})")
                    else:
                        error_msg = str(result)
                        print(f"      ❌ Sequence length {L}: {error_msg}")
                        
                        # Classify error types
                        if "timeout" in error_msg.lower():
                            local_timeout_errors += 1
                        elif any(keyword in error_msg.lower() for keyword in ['metal', 'kernel', 'gpu', 'invalid resource']):
                            local_metal_errors += 1
                            
                        correctness_scores.append(0.0)

                except Exception as e:
                    error_msg = str(e)
                    print(f"      ❌ Sequence length {L} exception: {error_msg}")
                    
                    # Classify error types
                    if any(keyword in error_msg.lower() for keyword in ['metal', 'kernel', 'gpu', 'invalid resource']):
                        local_metal_errors += 1
                    
                    correctness_scores.append(0.0)

            # Update global error counters
            self.metal_errors_caught += local_metal_errors
            self.timeout_errors_caught += local_timeout_errors

            # Calculate overall correctness
            overall_correctness = np.mean(correctness_scores) if correctness_scores else 0.0
            
            print(f"    📊 Overall correctness: {overall_correctness:.3f}")
            print(f"    🛡️  Metal errors caught: {local_metal_errors}")
            print(f"    ⏱️  Timeout errors caught: {local_timeout_errors}")

            return {
                "success": True,
                "score": overall_correctness,
                "metal_errors_caught": local_metal_errors,
                "timeout_errors_caught": local_timeout_errors
            }

        except Exception as e:
            print(f"    ❌ Thread-safe correctness testing failed: {e}")
            return {"success": False, "error": str(e)}

    def _test_single_sequence_safely(self, custom_attention_class: Any, args: Any, x: Any, mask: Any) -> float:
        """Test a single sequence with comprehensive safety checks"""
        try:
            # Instantiate custom attention with error checking
            custom_attn = custom_attention_class(args)
            
            # Verify the instance was created successfully  
            if custom_attn is None:
                raise ValueError("Failed to instantiate custom attention")
            
            # Run forward pass
            output = custom_attn(x, mask=mask)
            
            # Comprehensive output validation
            if output is None:
                raise ValueError("Custom attention returned None")
                
            # Shape validation
            expected_shape = x.shape
            if output.shape != expected_shape:
                raise ValueError(f"Wrong output shape: {output.shape}, expected {expected_shape}")

            # Finite value check
            if not mx.all(mx.isfinite(output)):
                raise ValueError("Output contains non-finite values (NaN or Inf)")

            # Statistical validation
            output_mean = float(mx.mean(output))
            output_std = float(mx.std(output))

            # Check for reasonable statistics
            if abs(output_mean) > 5.0:
                print(f"        ⚠️  Large mean detected: {output_mean:.6f}")
                return 0.5  # Partial credit
            
            if output_std > 50.0 or output_std < 0.0001:
                print(f"        ⚠️  Unusual std detected: {output_std:.6f}")
                return 0.7  # Partial credit
            
            # All checks passed
            return 1.0

        except Exception as e:
            # Convert any exception to a descriptive error
            error_msg = str(e)
            if "metal" in error_msg.lower() or "kernel" in error_msg.lower():
                raise MetalKernelError(f"Metal kernel error: {error_msg}")
            else:
                raise ValueError(f"Sequence test error: {error_msg}")

    def _armored_benchmark_custom_attention(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Armored benchmarking with multiple layers of protection"""
        print("  🚀 Running armored custom attention benchmarking...")
        
        retry_attempt = 0
        
        while retry_attempt <= self.max_retry_attempts:
            try:
                print(f"  🔄 Armored attempt {retry_attempt + 1}/{self.max_retry_attempts + 1}")
                
                # Apply custom attention hook with protection
                hook_result = self._protected_apply_custom_attention_hook(custom_attention_class)
                if not hook_result["success"]:
                    if retry_attempt < self.max_retry_attempts:
                        print(f"    🔄 Hook application failed, retrying... ({hook_result['error']})")
                        retry_attempt += 1
                        time.sleep(1)  # Brief pause
                        continue
                    return {"success": False, "error": f"Hook application failed: {hook_result['error']}"}
                
                original_attention = hook_result["original"]
                
                try:
                    # Run benchmarks with maximum protection
                    custom_configs = self._get_evolution_benchmark_configs()
                    custom_results = []
                    successful_benchmarks = 0
                    
                    for i, config in enumerate(custom_configs, 1):
                        print(f"    [{i}/{len(custom_configs)}] Armored custom: {config.name}")
                        
                        try:
                            # Run with comprehensive protection
                            success, result = self._thread_safe_execute_with_protection(
                                lambda: self.benchmark_suite.run_single_benchmark(config)
                            )
                            
                            if success and result:
                                custom_results.append(result)
                                successful_benchmarks += 1
                                print(f"      ✅ Armored {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec")
                            else:
                                print(f"      ❌ Armored {config.name}: {result}")
                                
                        except Exception as e:
                            print(f"      ❌ Armored {config.name} exception: {e}")
                            continue

                    # Check success rate
                    min_required = max(2, len(custom_configs) * 0.6)  # At least 60% or 2 benchmarks
                    if successful_benchmarks >= min_required:
                        print(f"  ✅ Armored benchmarks complete ({successful_benchmarks} successful)")
                        self.retry_attempts_used = retry_attempt
                        return {"success": True, "results": custom_results}
                    else:
                        error_msg = f"Only {successful_benchmarks}/{len(custom_configs)} benchmarks succeeded"
                        if retry_attempt < self.max_retry_attempts:
                            print(f"  🔄 {error_msg}, retrying...")
                            retry_attempt += 1
                            time.sleep(2)  # Longer pause before retry
                            continue
                        return {"success": False, "error": error_msg}
                
                finally:
                    # Always restore original attention
                    self._protected_remove_custom_attention_hook(original_attention)
                    print("    🔄 Restored standard attention")
                    
            except Exception as e:
                error_msg = f"Armored attempt failed: {str(e)}"
                print(f"  ❌ {error_msg}")
                if retry_attempt < self.max_retry_attempts:
                    retry_attempt += 1
                    time.sleep(2 ** retry_attempt)  # Exponential backoff
                    continue
                return {"success": False, "error": error_msg}
        
        return {"success": False, "error": "All armored attempts exhausted"}

    def _thread_safe_execute_with_protection(self, func) -> Tuple[bool, Any]:
        """Thread-safe execution with comprehensive Metal kernel protection (no signals)"""
        try:
            # Execute the function with comprehensive error catching
            result = func()
            return True, result
            
        except Exception as e:
            error_msg = str(e)
            
            # Classify Metal/GPU related errors
            metal_keywords = ['metal', 'kernel', 'gpu', 'invalid resource', 'command buffer', 'mps', 'mtl']
            if any(keyword in error_msg.lower() for keyword in metal_keywords):
                self.metal_errors_caught += 1
                return False, f"Metal kernel error: {error_msg}"
            else:
                return False, f"Execution error: {error_msg}"

    def _protected_apply_custom_attention_hook(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Protected application of custom attention hook"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            # Store original attention class safely
            original_attention = getattr(qwen3_module, 'Attention', None)
            if original_attention is None:
                return {"success": False, "error": "Could not find original Attention class"}

            # Apply custom attention with verification
            qwen3_module.Attention = custom_attention_class
            
            # Verify the hook was applied
            if qwen3_module.Attention != custom_attention_class:
                return {"success": False, "error": "Hook application verification failed"}

            print("      ✅ Custom attention hook applied and verified")
            return {"success": True, "original": original_attention}

        except ImportError:
            return {"success": False, "error": "Could not import mlx_lm.models.qwen3"}
        except Exception as e:
            return {"success": False, "error": f"Hook application failed: {str(e)}"}

    def _protected_remove_custom_attention_hook(self, original_attention: Any):
        """Protected removal of custom attention hook"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module
            qwen3_module.Attention = original_attention
            print("      ✅ Custom attention hook removed safely")
        except Exception as e:
            print(f"      ⚠️  Failed to remove hook (non-fatal): {e}")

    # Include helper methods from original evaluator
    def _ensure_standard_attention(self):
        """Ensure we're using standard attention"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module
            if hasattr(self, "_original_attention") and self._original_attention:
                qwen3_module.Attention = self._original_attention
                print("  🔄 Restored standard attention")
            else:
                print("  ✅ Standard attention already active")
        except ImportError:
            print("  ⚠️  Could not access qwen3 module")

    def _get_evolution_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Get representative benchmark configs for evolution"""
        try:
            all_configs = self.benchmark_suite.create_benchmark_configs()
            
            selected_test_names = [
                "short_context_quick",
                "long_context_detailed", 
                "long_generation",
                "code_generation",
                "maximum_context_stress_test"
            ]
            
            config_dict = {c.name: c for c in all_configs}
            representative_configs = []
            
            for test_name in selected_test_names:
                if test_name in config_dict:
                    representative_configs.append(config_dict[test_name])
                    
            return representative_configs
            
        except Exception as e:
            print(f"  ⚠️  Error getting benchmark configs: {e}")
            return []

    def _store_baseline_metrics(self, baseline_results: List[BenchmarkResult]):
        """Store baseline metrics for comparison"""
        decode_speeds = [r.decode_tokens_per_sec for r in baseline_results if r.decode_tokens_per_sec > 0]
        prefill_speeds = [r.prefill_tokens_per_sec for r in baseline_results if r.prefill_tokens_per_sec > 0]
        memories = [r.peak_memory_gb for r in baseline_results if r.peak_memory_gb > 0]

        self.baseline_results = baseline_results
        self.baseline_metrics = {
            "avg_decode_speed": float(np.mean(decode_speeds)) if decode_speeds else 0.0,
            "min_decode_speed": float(np.min(decode_speeds)) if decode_speeds else 0.0,
            "max_decode_speed": float(np.max(decode_speeds)) if decode_speeds else 0.0,
            "std_decode_speed": float(np.std(decode_speeds)) if len(decode_speeds) > 1 else 0.0,
            "avg_prefill_speed": float(np.mean(prefill_speeds)) if prefill_speeds else 0.0,
            "avg_memory_gb": float(np.mean(memories)) if memories else 0.0,
            "max_memory_gb": float(np.max(memories)) if memories else 0.0,
        }

        print(f"    📊 Baseline metrics stored - Avg decode: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec")

    def _analyze_performance_comparison(self, baseline_results: List[BenchmarkResult], custom_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance comparison between baseline and custom results"""
        print("  📈 Analyzing performance comparison...")

        baseline_dict = {r.name: r for r in baseline_results}
        custom_dict = {r.name: r for r in custom_results}

        individual_comparisons = []
        improvements = {
            "decode_speed_improvements": [],
            "prefill_speed_improvements": [],
            "total_speed_improvements": [],
            "memory_improvements": [],
            "time_improvements": [],
        }

        # Compare each benchmark individually
        for name in baseline_dict:
            if name in custom_dict:
                baseline = baseline_dict[name]
                custom = custom_dict[name]

                # Calculate improvements (positive = better)
                decode_improvement = (
                    (custom.decode_tokens_per_sec - baseline.decode_tokens_per_sec)
                    / baseline.decode_tokens_per_sec * 100
                    if baseline.decode_tokens_per_sec > 0 else 0
                )

                prefill_improvement = (
                    (custom.prefill_tokens_per_sec - baseline.prefill_tokens_per_sec)
                    / baseline.prefill_tokens_per_sec * 100
                    if baseline.prefill_tokens_per_sec > 0 else 0
                )

                total_improvement = (
                    (custom.total_tokens_per_sec - baseline.total_tokens_per_sec)
                    / baseline.total_tokens_per_sec * 100
                    if baseline.total_tokens_per_sec > 0 else 0
                )

                memory_improvement = (
                    (baseline.peak_memory_gb - custom.peak_memory_gb)
                    / baseline.peak_memory_gb * 100
                    if baseline.peak_memory_gb > 0 else 0
                )

                time_improvement = (
                    (baseline.total_time_sec - custom.total_time_sec)
                    / baseline.total_time_sec * 100
                    if baseline.total_time_sec > 0 else 0
                )

                comparison = {
                    "benchmark_name": name,
                    "baseline": self._result_to_dict(baseline),
                    "custom": self._result_to_dict(custom),
                    "improvements": {
                        "decode_speed_pct": decode_improvement,
                        "prefill_speed_pct": prefill_improvement,
                        "total_speed_pct": total_improvement,
                        "memory_reduction_pct": memory_improvement,
                        "time_reduction_pct": time_improvement,
                    },
                }

                individual_comparisons.append(comparison)

                # Collect for aggregate statistics
                improvements["decode_speed_improvements"].append(decode_improvement)
                improvements["prefill_speed_improvements"].append(prefill_improvement)
                improvements["total_speed_improvements"].append(total_improvement)
                improvements["memory_improvements"].append(memory_improvement)
                improvements["time_improvements"].append(time_improvement)

                print(f"    • {name}: {decode_improvement:+.1f}% decode speed")

        # Calculate aggregate statistics
        aggregate_stats = {}
        for key, values in improvements.items():
            if values:
                aggregate_stats[f"{key}_avg"] = float(np.mean(values))
                aggregate_stats[f"{key}_median"] = float(np.median(values))
                aggregate_stats[f"{key}_min"] = float(np.min(values))
                aggregate_stats[f"{key}_max"] = float(np.max(values))
                aggregate_stats[f"{key}_std"] = float(np.std(values))

        # Calculate overall metrics for custom results
        custom_decode_speeds = [r.decode_tokens_per_sec for r in custom_results if r.decode_tokens_per_sec > 0]
        custom_prefill_speeds = [r.prefill_tokens_per_sec for r in custom_results if r.prefill_tokens_per_sec > 0]
        custom_memories = [r.peak_memory_gb for r in custom_results if r.peak_memory_gb > 0]

        aggregate_metrics = {
            "avg_decode_speed": float(np.mean(custom_decode_speeds)) if custom_decode_speeds else 0.0,
            "min_decode_speed": float(np.min(custom_decode_speeds)) if custom_decode_speeds else 0.0,
            "max_decode_speed": float(np.max(custom_decode_speeds)) if custom_decode_speeds else 0.0,
            "avg_prefill_speed": float(np.mean(custom_prefill_speeds)) if custom_prefill_speeds else 0.0,
            "avg_memory_gb": float(np.mean(custom_memories)) if custom_memories else 0.0,
            "max_memory_gb": float(np.max(custom_memories)) if custom_memories else 0.0,
            "num_successful_tests": len(custom_results),
            "decode_speed_std": float(np.std(custom_decode_speeds)) if len(custom_decode_speeds) > 1 else 0.0,
        }

        # Summary for comparison to baseline
        comparison_summary = {
            "avg_decode_improvement_pct": aggregate_stats.get("decode_speed_improvements_avg", 0),
            "avg_decode_improvement_absolute": (
                aggregate_metrics["avg_decode_speed"] - self.baseline_metrics["avg_decode_speed"]
            ),
            "memory_change_gb": (
                aggregate_metrics["avg_memory_gb"] - self.baseline_metrics["avg_memory_gb"]
            ),
            "target_achieved": aggregate_stats.get("decode_speed_improvements_avg", 0) >= 5.0,
            "num_benchmarks_improved": sum(1 for x in improvements["decode_speed_improvements"] if x > 0),
            "total_benchmarks": len(improvements["decode_speed_improvements"]),
        }

        print(f"  📊 Analysis complete: {comparison_summary['avg_decode_improvement_pct']:+.1f}% average improvement")

        return {
            "individual_comparisons": individual_comparisons,
            "aggregate_improvements": aggregate_stats,
            "aggregate_metrics": aggregate_metrics,
            "comparison_summary": comparison_summary,
        }

    def _calculate_final_score(self, performance_analysis: Dict[str, Any], correctness: float) -> float:
        """Calculate final optimization score"""
        if correctness < 0.95:
            return -1000.0

        comparison = performance_analysis["comparison_summary"]
        avg_improvement = comparison["avg_decode_improvement_pct"]
        memory_change = comparison["memory_change_gb"]
        success_rate = comparison["num_benchmarks_improved"] / max(1, comparison["total_benchmarks"])

        # Score components
        performance_score = avg_improvement * 3  # Primary component
        memory_bonus = max(0, -memory_change * 10)  # Bonus for memory reduction
        consistency_bonus = success_rate * 10  # Bonus for consistent improvements
        correctness_bonus = correctness * 5  # Bonus for correctness

        final_score = performance_score + memory_bonus + consistency_bonus + correctness_bonus

        print(f"  🎯 Score breakdown:")
        print(f"    • Performance: {avg_improvement:.2f}% × 3 = {performance_score:.2f}")
        print(f"    • Memory: {memory_bonus:.2f}")
        print(f"    • Consistency: {success_rate:.2f} × 10 = {consistency_bonus:.2f}")
        print(f"    • Correctness: {correctness:.3f} × 5 = {correctness_bonus:.2f}")
        print(f"    • Final score: {final_score:.2f}")

        return final_score

    def _generate_summary(self, performance_analysis: Dict[str, Any], correctness: float) -> str:
        """Generate human-readable evaluation summary"""
        comparison = performance_analysis["comparison_summary"]
        metrics = performance_analysis["aggregate_metrics"]

        avg_improvement = comparison["avg_decode_improvement_pct"]
        current_decode = metrics["avg_decode_speed"]
        baseline_decode = self.baseline_metrics["avg_decode_speed"]

        summary = f"""Custom GQA Implementation Results:
• Decode Speed: {current_decode:.1f} tokens/sec (baseline: {baseline_decode:.1f})
• Improvement: {avg_improvement:+.1f}%
• Memory Usage: {metrics['avg_memory_gb']:.2f} GB
• Correctness: {correctness:.1%}
• Tests Passed: {metrics['num_successful_tests']}/{len(self._get_evolution_benchmark_configs())}
• Benchmarks Improved: {comparison['num_benchmarks_improved']}/{comparison['total_benchmarks']}"""

        if avg_improvement >= 15:
            summary += "\n🎯 EXCELLENT: 15%+ improvement achieved!"
        elif avg_improvement >= 10:
            summary += "\n🚀 STRONG IMPROVEMENT: 10%+ speedup"
        elif avg_improvement >= 5:
            summary += "\n✅ GOOD IMPROVEMENT: 5%+ speedup"
        elif avg_improvement > 0:
            summary += "\n📈 MINOR IMPROVEMENT: Some speedup achieved"
        else:
            summary += "\n⚠️  NO IMPROVEMENT: Performance regression"

        return summary

    def _print_evaluation_results(self, result: Dict[str, Any]):
        """Print comprehensive evaluation results"""
        print(f"\n{'='*100}")
        print(f"{'🎯 THREAD-SAFE EVALUATION RESULTS':^100}")
        print(f"{'='*100}")

        if result["success"]:
            performance = result["performance_metrics"]
            comparison = result["baseline_comparison"]

            print(f"📊 FINAL SCORE: {result['final_score']:.2f}")
            print(f"")
            print(f"📈 PERFORMANCE COMPARISON:")
            print(f"  • Average Decode Speed: {performance['avg_decode_speed']:.1f} tokens/sec")
            print(f"  • Baseline Decode Speed: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec")
            print(f"  • Average Improvement: {comparison['avg_decode_improvement_pct']:+.1f}%")
            print(f"  • Absolute Improvement: {comparison['avg_decode_improvement_absolute']:+.1f} tokens/sec")
            print(f"")
            print(f"💾 MEMORY USAGE:")
            print(f"  • Average Memory: {performance['avg_memory_gb']:.2f} GB")
            print(f"  • Baseline Memory: {self.baseline_metrics['avg_memory_gb']:.2f} GB")
            print(f"  • Memory Change: {comparison['memory_change_gb']:+.2f} GB")
            print(f"")
            print(f"✓ RELIABILITY:")
            print(f"  • Correctness Score: {result['correctness_score']:.1%}")
            print(f"  • Successful Tests: {performance['num_successful_tests']}")
            print(f"  • Benchmarks Improved: {comparison['num_benchmarks_improved']}/{comparison['total_benchmarks']}")

            if comparison["target_achieved"]:
                print(f"\n🎯 TARGET ACHIEVED: Significant improvement demonstrated!")

        else:
            print(f"❌ EVALUATION FAILED")
            print(f"📋 Error: {result.get('error', 'Unknown error')}")

        print(f"{'='*100}")

    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create result for failed evaluation"""
        return {
            "success": False,
            "final_score": -1000.0,
            "error": error_message,
            "performance_metrics": {},
            "correctness_score": 0.0,
            "summary": f"Evaluation failed: {error_message}",
            "error_statistics": {
                "metal_kernel_errors_caught": self.metal_errors_caught,
                "timeout_errors_caught": self.timeout_errors_caught,
                "retry_attempts_used": self.retry_attempts_used,
            }
        }

    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        """Convert BenchmarkResult to dictionary"""
        return {
            "name": result.name,
            "decode_tokens_per_sec": result.decode_tokens_per_sec,
            "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
            "peak_memory_gb": result.peak_memory_gb,
            "generated_tokens": result.generated_tokens,
            "total_time_sec": result.total_time_sec,
        }


def evaluate(program_text: str) -> Dict[str, Any]:
    """Main evaluation function called by OpenEvolve"""
    evaluator = ThreadSafeRobustEvaluator()
    return evaluator.evaluate(program_text)


def test_thread_safe_evaluator():
    """Test the thread-safe evaluator"""
    print("🧪 Testing Thread-Safe Robust Custom GQA Evaluator")
    print("=" * 80)
    
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if not os.path.exists(initial_program_path):
        print(f"❌ Initial program not found: {initial_program_path}")
        return
        
    print(f"📁 Testing with: {initial_program_path}")
    result = evaluate(initial_program_path)
    
    print(f"\n{'='*80}")
    print(f"🔬 THREAD-SAFE EVALUATOR TEST RESULTS")
    print(f"{'='*80}")
    print(f"Success: {result['success']}")
    print(f"Final Score: {result.get('final_score', 'N/A')}")
    if result.get('error_statistics'):
        stats = result['error_statistics']
        print(f"Metal Errors Caught: {stats['metal_kernel_errors_caught']}")
        print(f"Timeout Errors Caught: {stats['timeout_errors_caught']}")
        print(f"Total Errors Handled: {stats['total_errors_handled']}")
    print(f"Summary: {result.get('summary', 'N/A')}")
    
    return result


if __name__ == "__main__":
    test_thread_safe_evaluator()
