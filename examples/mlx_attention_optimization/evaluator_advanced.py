"""
Advanced Evaluator for MLX Attention Optimization

This evaluator is designed to test algorithmic innovations in attention mechanisms,
focusing on scenarios where novel approaches can show meaningful improvements over
the highly optimized mx.fast.scaled_dot_product_attention baseline.
"""

import gc
import importlib.util
import math
import psutil
import time
import traceback
from typing import Dict, List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ReferenceAttention(nn.Module):
    """Enhanced reference implementation with multiple fallback strategies"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, scale: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        
    def __call__(
        self, 
        queries: mx.array, 
        keys: mx.array, 
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[any] = None
    ) -> mx.array:
        """Reference implementation - the target to beat"""
        try:
            # Primary: Use MLX's optimized implementation
            processed_mask = mask
            if mask is not None and mask.ndim == 3:
                processed_mask = mx.expand_dims(mask, axis=1)
            
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=processed_mask
            )
        except (AttributeError, ImportError):
            # Fallback: Use manual implementation
            return self._manual_attention(queries, keys, values, mask)
    
    def _manual_attention(self, queries, keys, values, mask=None):
        """Fallback implementation using basic operations"""
        B, num_heads, L, head_dim = queries.shape
        _, num_kv_heads, L_kv, _ = keys.shape
        
        # Handle GQA
        if num_kv_heads != num_heads:
            rep_factor = num_heads // num_kv_heads
            keys = mx.repeat(keys, rep_factor, axis=1)
            values = mx.repeat(values, rep_factor, axis=1)
        
        # Standard attention
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            if mask.ndim == 3:
                mask = mx.expand_dims(mask, axis=1)
            scores = scores + mask
        
        attn_weights = mx.softmax(scores, axis=-1)
        return mx.matmul(attn_weights, values)


def create_reference_module(hidden_size, num_heads, num_kv_heads, head_dim, eps=1e-6):
    """Create reference module for comparison"""
    
    class ReferenceModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5
            
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            
            self.q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=eps)
            
            self.reference_attention = ReferenceAttention(
                hidden_size, num_heads, num_kv_heads, head_dim, self.scale
            )
            
        def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
            B, L, D = x.shape
            
            queries = self.q_proj(x)
            keys = self.k_proj(x)
            values = self.v_proj(x)
            
            queries = self.q_norm(
                queries.reshape(B, L, self.num_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            
            keys = self.k_norm(
                keys.reshape(B, L, self.num_kv_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            
            values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
            
            output = self.reference_attention(queries, keys, values, mask=mask)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            
            return self.o_proj(output)
    
    return ReferenceModule()


def create_advanced_test_cases() -> List[Dict]:
    """
    Create test cases that favor algorithmic innovations over micro-optimizations.
    Focus on scenarios where novel approaches can show meaningful improvements.
    """
    return [
        # Long sequence tests - where algorithmic improvements matter most
        {
            "name": "long_sequence_basic",
            "batch_size": 1, "seq_len": 1024, "hidden_size": 768, 
            "num_heads": 12, "num_kv_heads": 12,
            "weight": 3.0,  # High importance
            "expected_improvement": "sparse_patterns"
        },
        {
            "name": "very_long_sequence",
            "batch_size": 1, "seq_len": 2048, "hidden_size": 1024,
            "num_heads": 16, "num_kv_heads": 4,
            "weight": 4.0,  # Highest importance
            "expected_improvement": "linear_attention"
        },
        
        # Memory-intensive tests
        {
            "name": "memory_intensive_batch",
            "batch_size": 8, "seq_len": 512, "hidden_size": 768,
            "num_heads": 12, "num_kv_heads": 3,
            "weight": 2.5,
            "expected_improvement": "memory_efficiency"
        },
        {
            "name": "large_hidden_state",
            "batch_size": 2, "seq_len": 1024, "hidden_size": 2048,
            "num_heads": 32, "num_kv_heads": 8,
            "weight": 2.0,
            "expected_improvement": "chunked_processing"
        },
        
        # Edge cases for algorithm robustness
        {
            "name": "extreme_aspect_ratio",
            "batch_size": 1, "seq_len": 4096, "hidden_size": 512,
            "num_heads": 8, "num_kv_heads": 2,
            "weight": 3.5,
            "expected_improvement": "sparse_local_attention"
        },
        
        # Standard cases for baseline performance
        {
            "name": "standard_medium",
            "batch_size": 4, "seq_len": 256, "hidden_size": 512,
            "num_heads": 8, "num_kv_heads": 8,
            "weight": 1.0,
            "expected_improvement": "none"
        },
        {
            "name": "standard_small",
            "batch_size": 2, "seq_len": 128, "hidden_size": 256,
            "num_heads": 4, "num_kv_heads": 4,
            "weight": 0.5,  # Lower weight - not where innovations matter
            "expected_improvement": "none"
        },
    ]


def measure_detailed_performance(module, test_case: Dict, num_runs: int = 5) -> Dict[str, float]:
    """Enhanced performance measurement with detailed metrics"""
    
    batch_size = test_case["batch_size"]
    seq_len = test_case["seq_len"]
    hidden_size = test_case["hidden_size"]
    
    # Create test input
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Create causal mask
    mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
    mask = mx.expand_dims(mask, axis=0)
    
    # Memory measurement
    gc.collect()
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Warmup runs
    for _ in range(2):
        _ = module(x, mask=mask)
        mx.eval(_)
    
    # Timed runs with detailed metrics
    times = []
    peak_memory = memory_before
    
    for run in range(num_runs):
        # Memory tracking
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
        
        # Timing
        start_time = time.time()
        output = module(x, mask=mask)
        mx.eval(output)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    memory_used = memory_after - memory_before
    
    # Calculate metrics
    avg_time = np.mean(times)
    min_time = np.min(times)
    std_time = np.std(times)
    
    total_tokens = batch_size * seq_len
    avg_throughput = total_tokens / avg_time if avg_time > 0 else 0
    peak_throughput = total_tokens / min_time if min_time > 0 else 0
    
    # Computational complexity estimate
    theoretical_ops = batch_size * test_case["num_heads"] * seq_len * seq_len * test_case["hidden_size"]
    ops_per_second = theoretical_ops / avg_time if avg_time > 0 else 0
    
    return {
        "avg_time_seconds": avg_time,
        "min_time_seconds": min_time,
        "std_time_seconds": std_time,
        "avg_throughput_tokens_per_sec": avg_throughput,
        "peak_throughput_tokens_per_sec": peak_throughput,
        "memory_used_mb": memory_used,
        "peak_memory_mb": peak_memory,
        "ops_per_second": ops_per_second,
        "theoretical_ops": theoretical_ops,
        "efficiency_ratio": avg_throughput / max(memory_used, 1.0)
    }


def assess_algorithmic_innovation(evolved_module, reference_module, test_case: Dict) -> Dict[str, float]:
    """
    Assess whether the evolved module shows algorithmic innovation beyond micro-optimizations
    """
    
    # Performance comparison
    evolved_perf = measure_detailed_performance(evolved_module, test_case, num_runs=3)
    reference_perf = measure_detailed_performance(reference_module, test_case, num_runs=3)
    
    # Calculate improvement ratios
    throughput_ratio = (evolved_perf["avg_throughput_tokens_per_sec"] / 
                       max(reference_perf["avg_throughput_tokens_per_sec"], 1.0))
    
    memory_ratio = (reference_perf["memory_used_mb"] / 
                   max(evolved_perf["memory_used_mb"], 1.0))  # Higher is better
    
    efficiency_ratio = (evolved_perf["efficiency_ratio"] / 
                       max(reference_perf["efficiency_ratio"], 1.0))
    
    # Sequence length scaling assessment
    seq_len = test_case["seq_len"]
    
    # Bonus scoring for improvements on longer sequences (where innovations matter)
    length_bonus = 1.0
    if seq_len >= 2048:
        length_bonus = 2.0
    elif seq_len >= 1024:
        length_bonus = 1.5
    elif seq_len >= 512:
        length_bonus = 1.2
    
    # Innovation scoring
    innovation_score = 0.0
    
    # Significant throughput improvement
    if throughput_ratio > 1.2:
        innovation_score += 0.4 * length_bonus
    elif throughput_ratio > 1.1:
        innovation_score += 0.2 * length_bonus
    elif throughput_ratio > 1.05:
        innovation_score += 0.1
    
    # Memory efficiency improvement
    if memory_ratio > 1.3:
        innovation_score += 0.3 * length_bonus
    elif memory_ratio > 1.1:
        innovation_score += 0.2 * length_bonus
    
    # Overall efficiency improvement
    if efficiency_ratio > 1.5:
        innovation_score += 0.3 * length_bonus
    elif efficiency_ratio > 1.2:
        innovation_score += 0.2 * length_bonus
    
    return {
        "throughput_ratio": throughput_ratio,
        "memory_ratio": memory_ratio,
        "efficiency_ratio": efficiency_ratio,
        "innovation_score": min(innovation_score, 1.0),
        "length_bonus": length_bonus,
        "evolved_throughput": evolved_perf["avg_throughput_tokens_per_sec"],
        "reference_throughput": reference_perf["avg_throughput_tokens_per_sec"],
        "evolved_memory": evolved_perf["memory_used_mb"],
        "reference_memory": reference_perf["memory_used_mb"]
    }


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Advanced evaluation focusing on algorithmic innovation assessment
    """
    
    try:
        # Load evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "create_test_attention_module"):
            return {
                "accuracy_score": 0.0,
                "performance_score": 0.0,
                "innovation_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing create_test_attention_module function"
            }
        
        test_cases = create_advanced_test_cases()
        
        # Metrics tracking
        weighted_scores = []
        innovation_scores = []
        accuracy_scores = []
        performance_scores = []
        
        successful_cases = 0
        total_weight = sum(case.get("weight", 1.0) for case in test_cases)
        
        for i, test_case in enumerate(test_cases):
            try:
                print(f"Evaluating {test_case['name']}: {test_case}")
                
                # Create modules
                hidden_size = test_case["hidden_size"]
                num_heads = test_case["num_heads"]
                num_kv_heads = test_case["num_kv_heads"]
                head_dim = hidden_size // num_heads
                
                evolved_module = evolved_program.create_test_attention_module(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim
                )
                
                reference_module = create_reference_module(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim
                )
                
                # Basic functionality test
                batch_size = test_case["batch_size"]
                seq_len = test_case["seq_len"]
                x = mx.random.normal((batch_size, seq_len, hidden_size))
                
                mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
                mask = mx.expand_dims(mask, axis=0)
                
                # Test evolved module
                evolved_output = evolved_module(x, mask=mask)
                mx.eval(evolved_output)
                
                # Basic functionality check
                structural_check = (
                    evolved_output.shape == (batch_size, seq_len, hidden_size) and
                    not bool(mx.any(mx.isnan(evolved_output))) and
                    not bool(mx.any(mx.isinf(evolved_output))) and
                    abs(float(mx.mean(evolved_output))) < 100.0
                )
                
                if not structural_check:
                    print(f"  Structural check failed for {test_case['name']}")
                    continue
                
                # Innovation assessment
                innovation_results = assess_algorithmic_innovation(
                    evolved_module, reference_module, test_case
                )
                
                # Scoring
                case_weight = test_case.get("weight", 1.0)
                accuracy_score = 1.0 if structural_check else 0.0
                performance_score = min(innovation_results["throughput_ratio"], 3.0)
                innovation_score = innovation_results["innovation_score"]
                
                # Weighted combined score for this test case
                case_score = (
                    0.3 * accuracy_score +
                    0.4 * performance_score +
                    0.3 * innovation_score
                ) * case_weight
                
                weighted_scores.append(case_score)
                accuracy_scores.append(accuracy_score)
                performance_scores.append(performance_score)
                innovation_scores.append(innovation_score)
                
                successful_cases += 1
                
                print(f"  ✅ {test_case['name']}: "
                      f"throughput={innovation_results['throughput_ratio']:.2f}x, "
                      f"innovation={innovation_score:.3f}")
                
            except Exception as e:
                print(f"Test case {test_case['name']} failed: {str(e)}")
                continue
        
        if successful_cases == 0:
            return {
                "accuracy_score": 0.0,
                "performance_score": 0.0,
                "innovation_score": 0.0,
                "combined_score": 0.0,
                "success_rate": 0.0,
                "error": "No test cases passed"
            }
        
        # Calculate final scores
        success_rate = successful_cases / len(test_cases)
        
        # Weighted average scores
        total_weighted_score = sum(weighted_scores)
        avg_accuracy = np.mean(accuracy_scores)
        avg_performance = np.mean(performance_scores)
        avg_innovation = np.mean(innovation_scores)
        
        # Combined score emphasizes innovation and performance on challenging cases
        combined_score = (total_weighted_score / total_weight) * success_rate
        
        return {
            "accuracy_score": float(avg_accuracy),
            "performance_score": float(avg_performance),
            "innovation_score": float(avg_innovation),
            "combined_score": float(combined_score),
            "success_rate": float(success_rate),
            "successful_cases": successful_cases,
            "total_cases": len(test_cases),
            "weighted_total": float(total_weighted_score),
            "max_possible_score": float(total_weight)
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            "accuracy_score": 0.0,
            "performance_score": 0.0,
            "innovation_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """Quick algorithmic innovation check"""
    try:
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "create_test_attention_module"):
            return {"basic_functionality": 0.0, "error": "Missing required function"}
        
        # Test with a longer sequence to see if innovations are present
        evolved_module = evolved_program.create_test_attention_module(
            hidden_size=512, num_heads=8, num_kv_heads=8, head_dim=64
        )
        
        # Test basic functionality on longer sequence
        x = mx.random.normal((1, 512, 512))
        evolved_output = evolved_module(x)
        mx.eval(evolved_output)
        
        structural_check = (
            evolved_output.shape == (1, 512, 512) and
            not bool(mx.any(mx.isnan(evolved_output))) and
            not bool(mx.any(mx.isinf(evolved_output)))
        )
        
        # Quick performance check
        start_time = time.time()
        for _ in range(3):
            _ = evolved_module(x)
            mx.eval(_)
        elapsed = time.time() - start_time
        
        throughput = (3 * 512) / elapsed if elapsed > 0 else 0
        
        return {
            "basic_functionality": 1.0 if structural_check else 0.0,
            "throughput_preview": float(throughput),
            "structural_correctness": structural_check
        }
        
    except Exception as e:
        print(f"Stage 1 evaluation failed: {str(e)}")
        return {"basic_functionality": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """Full algorithmic innovation evaluation"""
    return evaluate(program_path)


if __name__ == "__main__":
    # Test with initial program
    print("Testing advanced evaluator...")
    import os
    
    # Test with initial_program_advanced.py if available
    test_files = [
        "initial_program_advanced.py",
        "initial_program.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTesting with {test_file}:")
            results = evaluate(test_file)
            
            print("Advanced evaluation results:")
            for metric, value in results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
            break
    else:
        print("No test files found")
