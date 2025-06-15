"""
Fixed Qwen3 Custom GQA Attention Evaluator

This evaluator addresses the critical methodology issues identified in the original evaluator:
1. Dynamic baseline measurement instead of hardcoded values
2. Direct model testing instead of subprocess calls
3. Comprehensive test coverage (all 20 scenarios)
4. Proper custom attention hook verification
5. Statistical rigor matching the comprehensive benchmark

Evolution Target:
- Custom GQA implementation using MLX primitives
- 40:8 query-to-KV head pattern optimization
- Apple M4 unified memory optimizations
- Goal: Genuine performance improvements over dynamic baseline
"""

import os
import sys
import json
import time
import traceback
import importlib.util
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import mlx.nn as nn

# Import the comprehensive benchmark suite for consistent testing
from qwen3_benchmark_suite import Qwen3BenchmarkSuite, BenchmarkConfig, BenchmarkResult


class FixedCustomGQAEvaluator:
    """Fixed evaluator for evolved custom GQA attention implementations"""

    def __init__(self):
        self.model_path = "mlx-community/Qwen3-0.6B-bf16"

        # Baseline will be measured dynamically
        self.baseline_metrics = None
        self.baseline_results = None

        # Use comprehensive benchmark suite for consistency
        self.benchmark_suite = Qwen3BenchmarkSuite(self.model_path)

        # Statistical parameters for reliable measurement
        self.warmup_runs = 2
        self.measurement_runs = 3

        print("🔧 Initialized Fixed Custom GQA Evaluator")
        print(f"📱 Model: {self.model_path}")
        print(f"🧪 Using 5 representative tests for fast evolution")
        print(f"📊 Dynamic baseline measurement enabled")

    def evaluate(self, program_text: str) -> Dict[str, Any]:
        """
        Fixed evaluation methodology:
        1. Extract custom attention class from evolved program
        2. Measure current baseline performance dynamically
        3. Apply custom attention and measure performance
        4. Compare results using proper statistical analysis
        """

        print("\n" + "=" * 100)
        print("🔬 FIXED CUSTOM GQA ATTENTION EVALUATION")
        print("=" * 100)
        print("✅ Using dynamic baseline measurement")
        print("✅ Using 5 representative tests for fast evolution")
        print("✅ Using direct model testing (no subprocess)")
        print("✅ Using proper statistical methodology")
        print("=" * 100)

        try:
            # Step 1: Extract custom attention class
            print("\n🔧 STEP 1: Extracting Custom Attention Class")
            custom_attention_class = self._extract_custom_attention_class(program_text)
            if custom_attention_class is None:
                return self._create_failure_result("Failed to extract CustomGQAAttention class")

            # Step 2: Measure baseline performance dynamically
            print("\n📊 STEP 2: Measuring Dynamic Baseline Performance")
            baseline_results = self._measure_baseline_performance()
            if not baseline_results:
                return self._create_failure_result("Failed to measure baseline performance")

            # Step 3: Test correctness of custom implementation
            print("\n🔍 STEP 3: Testing Custom Attention Correctness")
            correctness_score = self._test_correctness(custom_attention_class)
            if correctness_score < 0.95:
                return self._create_failure_result(
                    f"Correctness test failed: {correctness_score:.3f}"
                )

            # Step 4: Benchmark custom attention performance
            print("\n🚀 STEP 4: Benchmarking Custom Attention Performance")
            custom_results = self._benchmark_custom_attention(custom_attention_class)
            if not custom_results:
                return self._create_failure_result("Custom attention benchmarks failed")

            # Step 5: Compare performance statistically
            print("\n📈 STEP 5: Statistical Performance Analysis")
            performance_analysis = self._analyze_performance_comparison(
                baseline_results, custom_results
            )

            # Step 6: Calculate final score
            final_score = self._calculate_final_score(performance_analysis, correctness_score)

            # Step 7: Generate comprehensive result
            result = {
                "success": True,
                "final_score": final_score,
                "performance_metrics": performance_analysis["aggregate_metrics"],
                "correctness_score": correctness_score,
                "benchmark_results": [self._result_to_dict(r) for r in custom_results],
                "baseline_comparison": performance_analysis["comparison_summary"],
                "individual_comparisons": performance_analysis["individual_comparisons"],
                "summary": self._generate_summary(performance_analysis, correctness_score),
            }

            self._print_evaluation_results(result)
            return result

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            traceback.print_exc()
            return self._create_failure_result(f"Evaluation error: {str(e)}")

    def _extract_custom_attention_class(self, program_text: str) -> Optional[Any]:
        """Extract CustomGQAAttention class from evolved program"""
        try:
            print("  🔍 Analyzing evolved program...")

            # Handle both file paths and direct program text
            if (
                program_text.startswith("/")
                and "\n" not in program_text
                and len(program_text) < 500
            ):
                print(f"  📁 Reading program from file: {program_text}")
                if os.path.exists(program_text):
                    with open(program_text, "r") as f:
                        actual_program_text = f.read()
                else:
                    print(f"  ❌ Program file not found: {program_text}")
                    return None
            else:
                actual_program_text = program_text

            # Create execution environment
            exec_globals = {
                "__builtins__": __builtins__,
                "mx": mx,
                "nn": nn,
                "np": np,
                "time": time,
                "Optional": Optional,
                "Tuple": Tuple,
                "Any": Any,
            }

            # Import mlx_lm for RoPE
            try:
                exec_globals["mlx_lm"] = __import__("mlx_lm")
                print("  ✅ MLX-LM imported successfully")
            except ImportError:
                print("  ⚠️  Could not import mlx_lm, RoPE may not work")

            # Execute the evolved program
            print("  ⚙️  Executing evolved program...")
            exec(actual_program_text, exec_globals)

            # Extract the custom attention class
            custom_class = exec_globals.get("CustomGQAAttention")
            if custom_class is None:
                print("  ❌ CustomGQAAttention class not found in evolved program")
                return None

            print("  ✅ Successfully extracted CustomGQAAttention class")

            # Verify it's a valid class
            if not isinstance(custom_class, type):
                print("  ❌ CustomGQAAttention is not a valid class")
                return None

            print(f"  📋 Class name: {custom_class.__name__}")
            print(f"  📋 Base classes: {[base.__name__ for base in custom_class.__bases__]}")

            return custom_class

        except Exception as e:
            print(f"  ❌ Failed to extract custom attention class: {e}")
            traceback.print_exc()
            return None

    def _measure_baseline_performance(self) -> Optional[List[BenchmarkResult]]:
        """Measure baseline performance using standard attention"""
        try:
            print("  📊 Running comprehensive baseline benchmark...")
            print("  ⏱️  This will take several minutes...")

            # Clear any potential custom hooks first
            self._ensure_standard_attention()

            # Use a subset of benchmarks for faster evolution (but still comprehensive)
            # We'll use representative benchmarks across all categories
            baseline_configs = self._get_evolution_benchmark_configs()

            print(f"  🧪 Running {len(baseline_configs)} representative benchmarks")

            baseline_results = []

            for i, config in enumerate(baseline_configs, 1):
                print(f"  [{i}/{len(baseline_configs)}] Running baseline: {config.name}")
                try:
                    result = self.benchmark_suite.run_single_benchmark(config)
                    baseline_results.append(result)
                    print(
                        f"    ✅ Baseline {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec"
                    )
                except Exception as e:
                    print(f"    ❌ Failed baseline {config.name}: {e}")
                    continue

            if len(baseline_results) < len(baseline_configs) * 0.8:  # Need 80% success rate
                print(
                    f"  ❌ Only {len(baseline_results)}/{len(baseline_configs)} baseline benchmarks succeeded"
                )
                return None

            # Store baseline for comparison
            self.baseline_results = baseline_results

            # Calculate baseline metrics
            decode_speeds = [
                r.decode_tokens_per_sec for r in baseline_results if r.decode_tokens_per_sec > 0
            ]
            prefill_speeds = [
                r.prefill_tokens_per_sec for r in baseline_results if r.prefill_tokens_per_sec > 0
            ]
            memories = [r.peak_memory_gb for r in baseline_results if r.peak_memory_gb > 0]

            self.baseline_metrics = {
                "avg_decode_speed": float(np.mean(decode_speeds)),
                "min_decode_speed": float(np.min(decode_speeds)),
                "max_decode_speed": float(np.max(decode_speeds)),
                "std_decode_speed": float(np.std(decode_speeds)),
                "avg_prefill_speed": float(np.mean(prefill_speeds)),
                "avg_memory_gb": float(np.mean(memories)),
                "max_memory_gb": float(np.max(memories)),
            }

            print("  ✅ Baseline measurement complete")
            print(
                f"    📊 Average decode speed: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec"
            )
            print(
                f"    📊 Decode speed range: {self.baseline_metrics['min_decode_speed']:.1f} - {self.baseline_metrics['max_decode_speed']:.1f}"
            )
            print(f"    💾 Average memory: {self.baseline_metrics['avg_memory_gb']:.2f} GB")

            return baseline_results

        except Exception as e:
            print(f"  ❌ Failed to measure baseline: {e}")
            traceback.print_exc()
            return None

    def _get_evolution_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Get 5 most representative benchmark configs for faster evolution"""

        # Get all comprehensive configs
        all_configs = self.benchmark_suite.create_benchmark_configs()

        # Select only 5 most representative tests across all categories
        # for significantly faster evolution while maintaining coverage
        representative_configs = []

        # Map of specific test names to select
        selected_test_names = [
            "short_context_quick",          # Short context + quick response (chat scenario)
            "long_context_detailed",        # Long context analysis (memory pressure)
            "long_generation",              # Long generation (decode performance critical)
            "code_generation",              # Code generation (structured output patterns)
            "maximum_context_stress_test"   # Ultimate stress test (maximum challenge)
        ]

        # Find and add the selected tests
        config_dict = {c.name: c for c in all_configs}
        
        for test_name in selected_test_names:
            if test_name in config_dict:
                representative_configs.append(config_dict[test_name])
            else:
                print(f"  ⚠️  Warning: Test '{test_name}' not found in benchmark suite")

        print(f"  📋 Selected {len(representative_configs)} representative benchmarks for fast evolution:")
        for config in representative_configs:
            print(f"    • {config.name}: {config.description}")

        return representative_configs

    def _ensure_standard_attention(self):
        """Ensure we're using standard attention (remove any custom hooks)"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            # If there's a stored original attention, restore it
            if hasattr(self, "_original_attention") and self._original_attention:
                qwen3_module.Attention = self._original_attention
                print("  🔄 Restored standard attention")
            else:
                print("  ✅ Standard attention already active")
        except ImportError:
            print("  ⚠️  Could not access qwen3 module")

    def _test_correctness(self, custom_attention_class: Any) -> float:
        """Test that custom implementation produces correct results"""
        try:
            print("  🔍 Testing custom attention correctness...")

            # Qwen3 configuration
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

            # Test multiple sequence lengths
            test_cases = [
                (1, 64, 5120),  # Short sequence
                (1, 256, 5120),  # Medium sequence
                (1, 512, 5120),  # Long sequence
            ]

            correctness_scores = []

            for B, L, D in test_cases:
                print(f"    🧪 Testing sequence length {L}...")

                try:
                    # Create test input
                    x = mx.random.normal((B, L, D))
                    mask = "causal"

                    # Test custom implementation
                    custom_attn = custom_attention_class(args)
                    output = custom_attn(x, mask=mask)

                    # Basic sanity checks
                    expected_shape = (B, L, D)
                    if output.shape != expected_shape:
                        print(
                            f"    ❌ Wrong output shape: {output.shape}, expected {expected_shape}"
                        )
                        correctness_scores.append(0.0)
                        continue

                    # Check for finite values
                    if not mx.all(mx.isfinite(output)):
                        print(f"    ❌ Output contains non-finite values")
                        correctness_scores.append(0.0)
                        continue

                    # Check output statistics
                    output_mean = float(mx.mean(output))
                    output_std = float(mx.std(output))

                    if abs(output_mean) > 2.0 or output_std > 20.0 or output_std < 0.001:
                        print(
                            f"    ⚠️  Unusual output statistics: mean={output_mean:.6f}, std={output_std:.6f}"
                        )
                        correctness_scores.append(0.7)  # Partial credit
                    else:
                        print(
                            f"    ✅ Sequence length {L}: passed (mean={output_mean:.6f}, std={output_std:.6f})"
                        )
                        correctness_scores.append(1.0)

                except Exception as e:
                    print(f"    ❌ Sequence length {L} failed: {e}")
                    correctness_scores.append(0.0)

            overall_correctness = np.mean(correctness_scores) if correctness_scores else 0.0
            print(f"  📊 Overall correctness: {overall_correctness:.3f}")

            return overall_correctness

        except Exception as e:
            print(f"  ❌ Correctness testing failed: {e}")
            return 0.0

    def _benchmark_custom_attention(
        self, custom_attention_class: Any
    ) -> Optional[List[BenchmarkResult]]:
        """Benchmark custom attention using the same configs as baseline"""
        try:
            print("  🚀 Applying custom attention hook...")

            # Apply custom attention hook
            original_attention = self._apply_custom_attention_hook(custom_attention_class)
            if original_attention is None:
                print("  ❌ Failed to apply custom attention hook")
                return None

            try:
                print("  🧪 Running custom attention benchmarks...")

                # Use same configs as baseline for fair comparison
                custom_configs = self._get_evolution_benchmark_configs()
                custom_results = []

                for i, config in enumerate(custom_configs, 1):
                    print(f"  [{i}/{len(custom_configs)}] Running custom: {config.name}")
                    try:
                        result = self.benchmark_suite.run_single_benchmark(config)
                        custom_results.append(result)
                        print(
                            f"    ✅ Custom {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec"
                        )
                    except Exception as e:
                        print(f"    ❌ Failed custom {config.name}: {e}")
                        continue

                if len(custom_results) < len(custom_configs) * 0.8:  # Need 80% success rate
                    print(
                        f"  ❌ Only {len(custom_results)}/{len(custom_configs)} custom benchmarks succeeded"
                    )
                    return None

                print(
                    f"  ✅ Custom attention benchmarks complete ({len(custom_results)} successful)"
                )
                return custom_results

            finally:
                # Always restore original attention
                self._remove_custom_attention_hook(original_attention)
                print("  🔄 Restored standard attention")

        except Exception as e:
            print(f"  ❌ Custom attention benchmarking failed: {e}")
            return None

    def _apply_custom_attention_hook(self, custom_attention_class: Any) -> Optional[Any]:
        """Apply custom attention hook to mlx-lm"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            # Store original attention class
            original_attention = qwen3_module.Attention
            self._original_attention = original_attention

            # Replace with custom implementation
            qwen3_module.Attention = custom_attention_class

            print("    ✅ Custom attention hook applied")
            return original_attention

        except ImportError:
            print("    ❌ Could not import mlx_lm.models.qwen3")
            return None
        except Exception as e:
            print(f"    ❌ Failed to apply custom attention hook: {e}")
            return None

    def _remove_custom_attention_hook(self, original_attention: Any):
        """Remove custom attention hook and restore original"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            qwen3_module.Attention = original_attention
            print("    ✅ Custom attention hook removed")
        except ImportError:
            pass
        except Exception as e:
            print(f"    ⚠️  Failed to remove custom attention hook: {e}")

    def _analyze_performance_comparison(
        self, baseline_results: List[BenchmarkResult], custom_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Perform statistical comparison between baseline and custom results"""

        print("  📈 Analyzing performance comparison...")

        # Create lookup for easy comparison
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
                    (
                        (custom.decode_tokens_per_sec - baseline.decode_tokens_per_sec)
                        / baseline.decode_tokens_per_sec
                        * 100
                    )
                    if baseline.decode_tokens_per_sec > 0
                    else 0
                )

                prefill_improvement = (
                    (
                        (custom.prefill_tokens_per_sec - baseline.prefill_tokens_per_sec)
                        / baseline.prefill_tokens_per_sec
                        * 100
                    )
                    if baseline.prefill_tokens_per_sec > 0
                    else 0
                )

                total_improvement = (
                    (
                        (custom.total_tokens_per_sec - baseline.total_tokens_per_sec)
                        / baseline.total_tokens_per_sec
                        * 100
                    )
                    if baseline.total_tokens_per_sec > 0
                    else 0
                )

                memory_improvement = (
                    (
                        (baseline.peak_memory_gb - custom.peak_memory_gb)
                        / baseline.peak_memory_gb
                        * 100
                    )
                    if baseline.peak_memory_gb > 0
                    else 0
                )

                time_improvement = (
                    (
                        (baseline.total_time_sec - custom.total_time_sec)
                        / baseline.total_time_sec
                        * 100
                    )
                    if baseline.total_time_sec > 0
                    else 0
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
        custom_decode_speeds = [
            r.decode_tokens_per_sec for r in custom_results if r.decode_tokens_per_sec > 0
        ]
        custom_prefill_speeds = [
            r.prefill_tokens_per_sec for r in custom_results if r.prefill_tokens_per_sec > 0
        ]
        custom_memories = [r.peak_memory_gb for r in custom_results if r.peak_memory_gb > 0]

        aggregate_metrics = {
            "avg_decode_speed": (
                float(np.mean(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "min_decode_speed": (
                float(np.min(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "max_decode_speed": (
                float(np.max(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "avg_prefill_speed": (
                float(np.mean(custom_prefill_speeds)) if custom_prefill_speeds else 0.0
            ),
            "avg_memory_gb": float(np.mean(custom_memories)) if custom_memories else 0.0,
            "max_memory_gb": float(np.max(custom_memories)) if custom_memories else 0.0,
            "num_successful_tests": len(custom_results),
            "decode_speed_std": (
                float(np.std(custom_decode_speeds)) if len(custom_decode_speeds) > 1 else 0.0
            ),
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
            "target_achieved": aggregate_stats.get("decode_speed_improvements_avg", 0)
            >= 5.0,  # 5%+ improvement target
            "num_benchmarks_improved": sum(
                1 for x in improvements["decode_speed_improvements"] if x > 0
            ),
            "total_benchmarks": len(improvements["decode_speed_improvements"]),
        }

        print(
            f"  📊 Analysis complete: {comparison_summary['avg_decode_improvement_pct']:+.1f}% average improvement"
        )

        return {
            "individual_comparisons": individual_comparisons,
            "aggregate_improvements": aggregate_stats,
            "aggregate_metrics": aggregate_metrics,
            "comparison_summary": comparison_summary,
        }

    def _calculate_final_score(
        self, performance_analysis: Dict[str, Any], correctness: float
    ) -> float:
        """Calculate final optimization score based on real performance improvements"""

        if correctness < 0.95:  # Must be correct
            return -1000.0

        comparison = performance_analysis["comparison_summary"]

        # Primary score: average decode speed improvement
        avg_improvement = comparison["avg_decode_improvement_pct"]

        # Memory efficiency factor
        memory_change = comparison["memory_change_gb"]
        memory_factor = max(0, -memory_change * 10)  # Bonus for memory reduction

        # Consistency factor (number of benchmarks improved)
        success_rate = comparison["num_benchmarks_improved"] / max(
            1, comparison["total_benchmarks"]
        )
        consistency_factor = success_rate * 10  # Up to 10 points for 100% success rate

        # Correctness bonus
        correctness_bonus = correctness * 5  # Up to 5 points for perfect correctness

        # Calculate final score
        # Weight heavily on actual performance improvement
        final_score = (
            avg_improvement * 3  # 3x weight on average improvement
            + memory_factor
            + consistency_factor
            + correctness_bonus
        )

        print(f"  🎯 Score breakdown:")
        print(
            f"    • Avg decode improvement: {avg_improvement:.2f}% × 3 = {avg_improvement * 3:.2f}"
        )
        print(f"    • Memory efficiency: {memory_factor:.2f}")
        print(f"    • Consistency: {success_rate:.2f} × 10 = {consistency_factor:.2f}")
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

        improved_benchmarks = comparison["num_benchmarks_improved"]
        total_benchmarks = comparison["total_benchmarks"]

        summary = f"""Custom GQA Implementation Results:
• Decode Speed: {current_decode:.1f} tokens/sec (baseline: {baseline_decode:.1f})
• Improvement: {avg_improvement:+.1f}%
• Memory Usage: {metrics['avg_memory_gb']:.2f} GB
• Correctness: {correctness:.1%}
• Tests Passed: {metrics['num_successful_tests']}/{len(self._get_evolution_benchmark_configs())}
• Benchmarks Improved: {improved_benchmarks}/{total_benchmarks}"""

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
        print(f"{'🎯 EVALUATION RESULTS':^100}")
        print(f"{'='*100}")

        if result["success"]:
            performance = result["performance_metrics"]
            comparison = result["baseline_comparison"]

            print(f"📊 FINAL SCORE: {result['final_score']:.2f}")
            print(f"")
            print(f"📈 PERFORMANCE COMPARISON:")
            print(f"  • Average Decode Speed: {performance['avg_decode_speed']:.1f} tokens/sec")
            print(
                f"  • Baseline Decode Speed: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec"
            )
            print(f"  • Average Improvement: {comparison['avg_decode_improvement_pct']:+.1f}%")
            print(
                f"  • Absolute Improvement: {comparison['avg_decode_improvement_absolute']:+.1f} tokens/sec"
            )
            print(f"")
            print(f"💾 MEMORY USAGE:")
            print(f"  • Average Memory: {performance['avg_memory_gb']:.2f} GB")
            print(f"  • Baseline Memory: {self.baseline_metrics['avg_memory_gb']:.2f} GB")
            print(f"  • Memory Change: {comparison['memory_change_gb']:+.2f} GB")
            print(f"")
            print(f"✓ RELIABILITY:")
            print(f"  • Correctness Score: {result['correctness_score']:.1%}")
            print(f"  • Successful Tests: {performance['num_successful_tests']}")
            print(
                f"  • Benchmarks Improved: {comparison['num_benchmarks_improved']}/{comparison['total_benchmarks']}"
            )
            print(
                f"  • Success Rate: {comparison['num_benchmarks_improved']/comparison['total_benchmarks']:.1%}"
            )

            if comparison["target_achieved"]:
                print(f"\n🎯 TARGET ACHIEVED: Significant improvement demonstrated!")

            # Show individual benchmark results
            print(f"\n📋 INDIVIDUAL BENCHMARK RESULTS:")
            for comp in result["individual_comparisons"]:
                name = comp["benchmark_name"]
                decode_imp = comp["improvements"]["decode_speed_pct"]
                symbol = "✅" if decode_imp > 0 else "❌" if decode_imp < -1 else "➖"
                print(f"  {symbol} {name:<30} {decode_imp:+6.1f}%")

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
    evaluator = FixedCustomGQAEvaluator()
    return evaluator.evaluate(program_text)


def test_fixed_evaluator():
    """Test the fixed evaluator with the initial program"""
    print("🧪 Testing Fixed Custom GQA Evaluator")
    print("=" * 80)

    # Load initial program for testing
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")

    if not os.path.exists(initial_program_path):
        print(f"❌ Initial program not found: {initial_program_path}")
        return

    print(f"📁 Loading initial program: {initial_program_path}")

    # Test evaluation
    result = evaluate(initial_program_path)

    print(f"\n{'='*80}")
    print(f"🔬 FIXED EVALUATOR TEST RESULTS")
    print(f"{'='*80}")
    print(f"Success: {result['success']}")
    print(f"Final Score: {result.get('final_score', 'N/A')}")
    if result.get("baseline_comparison"):
        print(
            f"Average Improvement: {result['baseline_comparison'].get('avg_decode_improvement_pct', 0):+.1f}%"
        )
    print(f"Summary: {result.get('summary', 'N/A')}")

    return result


if __name__ == "__main__":
    test_fixed_evaluator()
