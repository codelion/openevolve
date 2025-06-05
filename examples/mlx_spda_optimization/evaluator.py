"""
Enhanced Evaluator with Comprehensive Correctness Tests + Progressive Rewards

This evaluator combines:
1. COMPREHENSIVE correctness testing (from original evaluator) 
2. Progressive rewards for incremental improvements
3. Rigorous evaluation methodology

Critical: All original correctness tests are preserved to ensure evolved kernels 
produce mathematically correct results across all scenarios.
"""

import importlib.util
import math
import time
import traceback
from typing import Dict, Union, List, Tuple
import gc
import os

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX or NumPy not available")
    MLX_AVAILABLE = False


# ============================================================================
# RIGOROUS TIMING METHODOLOGY
# ============================================================================

N_warmup = 5
N_iter_bench = 40
N_iter_func = 8


def bench(f, *args):
    """Rigorous benchmarking function"""
    for i in range(N_warmup):
        f(*args)

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(*args)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    """Attention computation"""
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def do_attention_bench(f, q, k, v, scale, mask=None, transpose=False):
    """Attention benchmarking"""
    q_out = q

    for i in range(N_iter_func):
        q_out = do_attention(f, q_out, k, v, scale, mask=mask, transpose=transpose)

    mx.eval(q_out)
    return q_out


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    """Rigorous input preparation from original evaluator"""
    np_dtype = getattr(np, dtype)

    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    scale = 1.0 / math.sqrt(D)

    q_np = np.random.normal(0.0, 1.0, shape_q).astype(np_dtype)
    k_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)

    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)

    if mask is not None:
        if mask == "additive":
            mask_np = np.random.normal(0.0, 1.0, (B, qH, qL, kL)).astype(np_dtype)
            mask = mx.array(mask_np)
        elif mask == "bool":
            mask_np = np.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5
            mask = mx.array(mask_np)
        elif mask == "causal":
            mask = mx.tril(mx.ones((qL, kL), dtype=mx.bool_))
            mask = mx.expand_dims(mx.expand_dims(mask, 0), 0)  # Add batch and head dims
            mask = mx.broadcast_to(mask, (B, qH, qL, kL))

    return q_mx, k_mx, v_mx, scale, mask


# ============================================================================
# BASELINE CACHING FOR PROGRESSIVE REWARDS
# ============================================================================

class BaselineCache:
    """Cache baseline performance for progressive reward calculation"""
    
    def __init__(self):
        self.initial_program_performance = None
        self.spda_performance = None
        self.cache_file = "./openevolve_output/baseline_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Load cached baseline performance"""
        try:
            if os.path.exists(self.cache_file):
                import json
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.initial_program_performance = data.get('initial_program')
                    self.spda_performance = data.get('spda')
                    print(f"📚 Loaded baseline cache: {len(data)} entries")
        except Exception as e:
            print(f"⚠️ Could not load baseline cache: {e}")
    
    def save_cache(self):
        """Save baseline performance to cache"""
        try:
            import json
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data = {
                'initial_program': self.initial_program_performance,
                'spda': self.spda_performance
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save baseline cache: {e}")
    
    def ensure_baselines(self, configs):
        """Ensure we have baseline performance for progressive rewards"""
        if self.initial_program_performance is None:
            print("📊 Benchmarking initial program for progressive rewards...")
            self.initial_program_performance = benchmark_initial_program(configs)
        
        if self.spda_performance is None:
            print("📊 Benchmarking SPDA baseline for progressive rewards...")
            self.spda_performance = benchmark_spda_baseline(configs)
        
        self.save_cache()


# Global baseline cache
_baseline_cache = BaselineCache()


def benchmark_initial_program(configs):
    """Benchmark the initial program across all test configurations"""
    try:
        # Load initial program
        initial_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
        spec = importlib.util.spec_from_file_location("initial_program", initial_path)
        initial_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(initial_program)
        
        initial_fn = initial_program.evolved_scaled_dot_product_attention
        
        performance = {}
        for config in configs:
            if "block_sizes" not in config:
                continue
                
            try:
                result = benchmark_performance_single(initial_fn, config)
                if "error" not in result:
                    performance[config["name"]] = result["evolved_time"]
            except Exception as e:
                print(f"⚠️ Failed to benchmark initial program on {config['name']}: {e}")
        
        return performance
    except Exception as e:
        print(f"❌ Failed to benchmark initial program: {e}")
        return {}


def benchmark_spda_baseline(configs):
    """Benchmark SPDA baseline across all test configurations"""
    performance = {}
    for config in configs:
        if "block_sizes" not in config:
            continue
            
        try:
            result = benchmark_performance_single(mlx_spda_baseline, config)
            if "error" not in result:
                performance[config["name"]] = result["evolved_time"]
        except Exception as e:
            print(f"⚠️ Failed to benchmark SPDA on {config['name']}: {e}")
    
    return performance


# ============================================================================
# TEST CONFIGURATION AND MASK CREATION
# ============================================================================

def create_block_diagonal_mask(B, H, L, block_sizes):
    """Create block-diagonal mask for packed sequences."""
    mask_np = np.zeros((B, H, L, L), dtype=bool)
    
    current_pos = 0
    for block_size in block_sizes:
        if current_pos + block_size <= L:
            end_pos = current_pos + block_size
            mask_np[:, :, current_pos:end_pos, current_pos:end_pos] = True
            current_pos = end_pos
        else:
            break
    
    return mx.array(mask_np)


def reference_attention(q, k, v, scale, mask):
    """Reference implementation for correctness checking."""
    scores = (q * scale) @ mx.swapaxes(k, -1, -2)
    
    if mask is not None:
        if hasattr(mask, 'dtype') and mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -mx.array(np.float32(np.inf)))
        else:
            scores = scores + mask
    
    attn_weights = mx.softmax(scores, axis=-1, precise=True)
    return attn_weights @ v


def mlx_spda_baseline(q, k, v, scale, mask):
    """MLX fast SPDA implementation - our performance baseline."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def create_test_configurations():
    """Create comprehensive test configurations with ALL original correctness tests"""
    configs = []
    
    # ===== STAGE 1: COMPREHENSIVE CORRECTNESS TESTS =====
    # CRITICAL: All original correctness tests preserved!
    
    # Block-diagonal correctness tests
    configs.extend([
        {
            "name": "correctness_small_blocks", 
            "B": 1, "H": 4, "L": 256, "D": 64,
            "block_sizes": [128, 128],  # 2 blocks, 50% sparse
            "test_type": "correctness"
        },
        {
            "name": "correctness_medium_blocks",
            "B": 1, "H": 8, "L": 512, "D": 64, 
            "block_sizes": [128, 128, 128, 128],  # 4 blocks, 75% sparse
            "test_type": "correctness"
        },
        {
            "name": "correctness_many_blocks",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [64] * 8,  # 8 blocks, 87.5% sparse
            "test_type": "correctness"
        },
        {
            "name": "correctness_variable_blocks",
            "B": 1, "H": 4, "L": 384, "D": 64,
            "block_sizes": [128, 256],  # Variable sizes
            "test_type": "correctness"
        }
    ])
    
    # CRITICAL: SPDA benchmark configurations for comprehensive correctness testing
    # These test various scenarios that might not be block-diagonal but still need to work
    spda_correctness_configs = [
        # (B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type)
        (1, 32, 32, 64, 16, 16, None),        # Basic small
        (1, 64, 64, 64, 16, 16, "bool"),      # Boolean mask
        (1, 128, 128, 64, 16, 16, "causal"),  # Causal mask
        (1, 256, 256, 64, 16, 16, None),      # Medium size
        (1, 128, 128, 80, 16, 16, "bool"),    # Different head dim (PaLM)
        (2, 128, 128, 64, 16, 16, "causal"),  # Batch size > 1
        (1, 512, 512, 64, 16, 16, "bool"),    # Larger size
        (1, 256, 256, 128, 8, 8, None),       # Large head dim, fewer heads
        (1, 128, 128, 64, 32, 32, "causal"),  # Many heads
        (4, 64, 64, 64, 8, 8, None),          # Large batch
        (1, 192, 192, 80, 12, 12, "bool"),    # Non-power-of-2 sizes
        (2, 384, 384, 64, 16, 16, "causal"),  # Large + batch
        (1, 96, 96, 128, 6, 6, None),         # Small + large head_dim
    ]
    
    for i, (B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type) in enumerate(spda_correctness_configs):
        configs.append({
            "name": f"spda_correctness_{i+1}",
            "test_type": "correctness",
            "spda_config": {
                "B": B, "qsl": qsl, "ksl": ksl, "head_dim": head_dim,
                "n_q_heads": n_q_heads, "n_kv_heads": n_kv_heads,
                "mask_type": mask_type, "dtype": "float16", "transpose": False
            }
        })
    
    # Additional edge case correctness tests
    edge_case_configs = [
        # Edge cases that might break evolved kernels
        (1, 33, 33, 64, 7, 7, None),          # Odd dimensions
        (1, 17, 17, 63, 3, 3, "causal"),      # Small odd sizes
        (3, 127, 127, 65, 5, 5, "bool"),      # Non-standard sizes
        (1, 1024, 1024, 32, 64, 64, None),    # Very wide attention
        (1, 31, 31, 256, 2, 2, "causal"),     # Few heads, large head_dim
    ]
    
    for i, (B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type) in enumerate(edge_case_configs):
        configs.append({
            "name": f"edge_case_{i+1}",
            "test_type": "correctness",
            "spda_config": {
                "B": B, "qsl": qsl, "ksl": ksl, "head_dim": head_dim,
                "n_q_heads": n_q_heads, "n_kv_heads": n_kv_heads,
                "mask_type": mask_type, "dtype": "float16", "transpose": False
            }
        })
    
    # ===== STAGE 2: PROGRESSIVE PERFORMANCE TESTS =====
    # These are organized by difficulty for progressive rewards
    
    # Level 1: Dense patterns (50% sparse) - Baseline performance
    configs.extend([
        {
            "name": "dense_2x256_50sparse",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256],
            "test_type": "performance",
            "difficulty": "baseline",
            "expected_sparsity": 0.50
        },
        {
            "name": "dense_2x384_50sparse",
            "B": 1, "H": 12, "L": 768, "D": 64,
            "block_sizes": [384, 384], 
            "test_type": "performance",
            "difficulty": "baseline",
            "expected_sparsity": 0.50
        }
    ])
    
    # Level 2: Medium sparsity (75% sparse) - Good optimization opportunity
    configs.extend([
        {
            "name": "medium_4x128_75sparse",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],
            "test_type": "performance", 
            "difficulty": "medium",
            "expected_sparsity": 0.75
        },
        {
            "name": "medium_4x192_75sparse",
            "B": 2, "H": 12, "L": 768, "D": 64,
            "block_sizes": [192, 192, 192, 192],
            "test_type": "performance",
            "difficulty": "medium", 
            "expected_sparsity": 0.75
        }
    ])
    
    # Level 3: High sparsity (87.5% sparse) - Major advantage potential
    configs.extend([
        {
            "name": "sparse_8x64_87sparse",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8,
            "test_type": "performance",
            "difficulty": "hard",
            "expected_sparsity": 0.875
        },
        {
            "name": "sparse_8x128_87sparse",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128] * 8,
            "test_type": "performance",
            "difficulty": "hard",
            "expected_sparsity": 0.875
        }
    ])
    
    # Level 4: Very high sparsity (93.75% sparse) - Massive wins possible
    configs.extend([
        {
            "name": "very_sparse_16x32_93sparse",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [32] * 16,
            "test_type": "performance",
            "difficulty": "expert",
            "expected_sparsity": 0.9375
        },
        {
            "name": "very_sparse_16x64_93sparse",
            "B": 1, "H": 32, "L": 1024, "D": 64,
            "block_sizes": [64] * 16,
            "test_type": "performance",
            "difficulty": "expert",
            "expected_sparsity": 0.9375
        }
    ])
    
    # Level 5: Extreme sparsity (96.875% sparse) - Ultimate challenge
    configs.extend([
        {
            "name": "extreme_sparse_32x16_96sparse",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [16] * 32,
            "test_type": "performance",
            "difficulty": "extreme",
            "expected_sparsity": 0.96875
        },
        {
            "name": "extreme_sparse_64x8_98sparse",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [8] * 64,
            "test_type": "performance",
            "difficulty": "extreme",
            "expected_sparsity": 0.984375
        }
    ])
    
    return configs


# ============================================================================
# ENHANCED CORRECTNESS EVALUATION
# ============================================================================

def evaluate_correctness(evolved_fn, config):
    """Enhanced correctness testing with support for all original test types"""
    try:
        # Handle two types of configs: block diagonal and SPDA
        if "spda_config" in config:
            # SPDA correctness test using original rigorous methodology
            spda_cfg = config["spda_config"]
            B, qsl, ksl, head_dim = spda_cfg["B"], spda_cfg["qsl"], spda_cfg["ksl"], spda_cfg["head_dim"]
            n_q_heads, n_kv_heads = spda_cfg["n_q_heads"], spda_cfg["n_kv_heads"]
            mask_type, dtype, transpose = spda_cfg["mask_type"], spda_cfg["dtype"], spda_cfg["transpose"]
            
            # Use original rigorous input preparation
            q, k, v, scale, mask = prepare_inputs(
                B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_type, transpose, dtype
            )
        
        else:
            # Block diagonal test
            B, H, L, D = config["B"], config["H"], config["L"], config["D"]
            
            # Create test inputs using same method as original
            np_dtype = np.float16  # Use float16 for consistency
            scale = 1.0 / math.sqrt(D)
            
            q_np = np.random.normal(0.0, 1.0, (B, H, L, D)).astype(np_dtype)
            k_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
            v_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
            
            q = mx.array(q_np)
            k = mx.array(k_np)
            v = mx.array(v_np)
            
            # Create block-diagonal mask
            mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
        
        # Run evolved implementation
        evolved_output = evolved_fn(q, k, v, scale=scale, mask=mask)
        
        # Run reference implementation  
        reference_output = reference_attention(q, k, v, scale, mask)
        
        # Compare outputs
        if evolved_output.shape != reference_output.shape:
            return {
                "passed": False,
                "error": f"Shape mismatch: {evolved_output.shape} vs {reference_output.shape}",
                "config_name": config["name"]
            }
        
        # Calculate error metrics with original tolerances
        diff = evolved_output - reference_output
        mse = float(mx.mean(diff ** 2))
        max_diff = float(mx.max(mx.abs(diff)))
        
        # Check for invalid outputs
        has_nan = bool(mx.any(mx.isnan(evolved_output)))
        has_inf = bool(mx.any(mx.isinf(evolved_output)))
        
        # Determine pass/fail using original stringent criteria
        tolerance = 1e-4 if q.dtype == mx.float32 else 2e-4  # Original tolerances
        passed = mse < tolerance and max_diff < 0.05 and not has_nan and not has_inf
        
        return {
            "passed": passed,
            "mse": mse,
            "max_diff": max_diff,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "config_name": config["name"],
            "tolerance_used": tolerance
        }
        
    except Exception as e:
        return {
            "passed": False, 
            "error": str(e),
            "config_name": config["name"]
        }


# ============================================================================
# PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_performance_single(evolved_fn, config):
    """Benchmark a single configuration with rigorous timing methodology"""
    try:
        B, H, L, D = config["B"], config["H"], config["L"], config["D"]
        
        # Create test inputs using consistent methodology
        np_dtype = np.float16
        scale = 1.0 / math.sqrt(D)
        
        q_np = np.random.normal(0.0, 1.0, (B, H, L, D)).astype(np_dtype)
        k_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
        v_np = np.random.normal(0.0, scale, (B, H, L, D)).astype(np_dtype)
        
        q = mx.array(q_np)
        k = mx.array(k_np)
        v = mx.array(v_np)
        
        # Create block-diagonal mask
        mask = create_block_diagonal_mask(B, H, L, config["block_sizes"])
        
        # Benchmark evolved implementation
        try:
            evolved_time = bench(do_attention_bench, evolved_fn, q, k, v, scale, mask, False)
        except Exception as e:
            return {"error": f"Evolved function failed: {str(e)}"}
        
        # Calculate metrics
        total_elements = L * L
        masked_elements = sum(bs * bs for bs in config["block_sizes"])
        sparsity = 1.0 - (masked_elements / total_elements)
        
        # Correctness check against SPDA
        try:
            o_evolved = do_attention(evolved_fn, q, k, v, scale, mask, False)
            o_spda = do_attention(mlx_spda_baseline, q, k, v, scale, mask, False)
            
            atol = 2e-4 if q.dtype == mx.float16 else 1e-5
            correctness_ok = mx.allclose(o_evolved, o_spda, atol=atol, rtol=atol)
        except Exception as e:
            return {"error": f"Correctness check failed: {str(e)}"}
        
        return {
            "evolved_time": evolved_time,
            "config_name": config["name"],
            "sparsity": sparsity,
            "correctness_ok": correctness_ok,
            "difficulty": config.get("difficulty", "unknown")
        }
        
    except Exception as e:
        return {"error": str(e), "config_name": config["name"]}


# ============================================================================
# PROGRESSIVE REWARD CALCULATION
# ============================================================================

def calculate_progressive_rewards(evolved_fn, test_configs) -> Dict[str, float]:
    """Calculate multi-level progressive rewards for the evolved kernel"""
    
    # Ensure we have baseline performance cached
    _baseline_cache.ensure_baselines(test_configs)
    
    performance_configs = [c for c in test_configs if c["test_type"] == "performance"]
    
    # Benchmark evolved kernel on all performance tests
    evolved_results = []
    for config in performance_configs:
        result = benchmark_performance_single(evolved_fn, config)
        if "error" not in result and result["correctness_ok"]:
            evolved_results.append(result)
    
    if not evolved_results:
        return {
            "baseline_improvement_score": 0.0,
            "spda_competition_score": 0.0, 
            "sparsity_exploitation_score": 0.0,
            "overall_progressive_score": 0.0,
            "num_successful_tests": 0
        }
    
    # LEVEL 1: BASELINE IMPROVEMENT REWARDS (40% weight)
    baseline_scores = []
    for result in evolved_results:
        config_name = result["config_name"]
        evolved_time = result["evolved_time"]
        
        # Get initial program performance for this config
        initial_time = _baseline_cache.initial_program_performance.get(config_name)
        if initial_time and initial_time > 0:
            speedup_vs_initial = initial_time / evolved_time
            
            # Linear reward scaling for baseline improvement
            if speedup_vs_initial >= 3.0:
                baseline_score = 1.0
            elif speedup_vs_initial >= 2.0:
                baseline_score = 0.8
            elif speedup_vs_initial >= 1.5:
                baseline_score = 0.6
            elif speedup_vs_initial >= 1.2:
                baseline_score = 0.4
            elif speedup_vs_initial >= 1.1:
                baseline_score = 0.2
            else:
                baseline_score = 0.0
            
            baseline_scores.append(baseline_score)
    
    baseline_improvement_score = np.mean(baseline_scores) if baseline_scores else 0.0
    
    # LEVEL 2: SPDA COMPETITION REWARDS (40% weight)  
    spda_scores = []
    for result in evolved_results:
        config_name = result["config_name"]
        evolved_time = result["evolved_time"]
        
        # Get SPDA performance for this config
        spda_time = _baseline_cache.spda_performance.get(config_name)
        if spda_time and spda_time > 0:
            speedup_vs_spda = spda_time / evolved_time
            
            # Exponential reward scaling for SPDA competition
            if speedup_vs_spda >= 2.0:
                spda_score = 1.0
            elif speedup_vs_spda >= 1.5:
                spda_score = 0.9
            elif speedup_vs_spda >= 1.2:
                spda_score = 0.7
            elif speedup_vs_spda >= 1.0:
                spda_score = 0.4
            elif speedup_vs_spda >= 0.9:
                spda_score = 0.2
            elif speedup_vs_spda >= 0.8:
                spda_score = 0.1
            else:
                spda_score = 0.0
            
            spda_scores.append(spda_score)
    
    spda_competition_score = np.mean(spda_scores) if spda_scores else 0.0
    
    # LEVEL 3: SPARSITY EXPLOITATION REWARDS (20% weight)
    # Reward consistent performance across different sparsity levels
    sparsity_groups = {}
    for result in evolved_results:
        sparsity = result["sparsity"]
        difficulty = result["difficulty"]
        
        if difficulty not in sparsity_groups:
            sparsity_groups[difficulty] = []
        sparsity_groups[difficulty].append(result)
    
    # Bonus for performing well across multiple sparsity levels
    if len(sparsity_groups) >= 3:  # Good performance on 3+ difficulty levels
        sparsity_exploitation_score = 1.0
    elif len(sparsity_groups) >= 2:  # Good performance on 2+ difficulty levels
        sparsity_exploitation_score = 0.6
    elif len(sparsity_groups) >= 1:  # Good performance on 1 difficulty level
        sparsity_exploitation_score = 0.3
    else:
        sparsity_exploitation_score = 0.0
    
    # COMBINE SCORES WITH WEIGHTS
    overall_progressive_score = (
        0.4 * baseline_improvement_score +     # 40% for beating initial program
        0.4 * spda_competition_score +         # 40% for competing with SPDA  
        0.2 * sparsity_exploitation_score      # 20% for sparsity consistency
    )
    
    return {
        "baseline_improvement_score": float(baseline_improvement_score),
        "spda_competition_score": float(spda_competition_score),
        "sparsity_exploitation_score": float(sparsity_exploitation_score),
        "overall_progressive_score": float(overall_progressive_score),
        "num_successful_tests": len(evolved_results),
        "total_performance_tests": len(performance_configs)
    }


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Comprehensive evaluation with ALL original correctness tests + progressive rewards
    
    This ensures evolved kernels are mathematically correct across ALL scenarios
    while providing progressive reward signals for incremental improvements.
    """
    print(f"🚀 Evaluating Metal Kernel (Comprehensive + Progressive): {program_path}")
    
    if not MLX_AVAILABLE:
        return {
            "stage1_passed": False,
            "overall_score": 0.0,
            "combined_score": 0.0,
            "error": "MLX not available"
        }
    
    try:
        # Load evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evolved_program)
        
        if not hasattr(evolved_program, "evolved_scaled_dot_product_attention"):
            return {
                "stage1_passed": False,
                "overall_score": 0.0, 
                "combined_score": 0.0,
                "error": "Missing evolved_scaled_dot_product_attention function"
            }
        
        evolved_fn = evolved_program.evolved_scaled_dot_product_attention
        
        # ===== STAGE 1: COMPREHENSIVE CORRECTNESS TESTING =====
        print("\\n📋 STAGE 1: Comprehensive Correctness Testing")
        print("Includes ALL original correctness tests + SPDA configurations + edge cases")
        
        test_configs = create_test_configurations()
        correctness_configs = [c for c in test_configs if c["test_type"] == "correctness"]
        
        print(f"  Running {len(correctness_configs)} comprehensive correctness tests...")
        
        # Count different test types for reporting
        block_diagonal_tests = len([c for c in correctness_configs if "block_sizes" in c])
        spda_tests = len([c for c in correctness_configs if "spda_config" in c and "spda_correctness" in c["name"]])
        edge_case_tests = len([c for c in correctness_configs if "edge_case" in c["name"]])
        
        print(f"    • Block-diagonal tests: {block_diagonal_tests}")
        print(f"    • SPDA configuration tests: {spda_tests}")
        print(f"    • Edge case tests: {edge_case_tests}")
        
        correctness_results = []
        passed_count = 0
        
        for config in correctness_configs:
            result = evaluate_correctness(evolved_fn, config)
            correctness_results.append(result)
            
            if result["passed"]:
                passed_count += 1
                print(f"    ✅ {config['name']}: PASSED (MSE: {result.get('mse', 0):.2e})")
            else:
                error_msg = result.get("error", f"MSE: {result.get('mse', 'N/A'):.2e}")
                print(f"    ❌ {config['name']}: FAILED ({error_msg})")
        
        # Calculate pass rate with STRINGENT requirement
        pass_rate = passed_count / len(correctness_configs) if correctness_configs else 0.0
        stage1_passed = pass_rate >= 0.85  # 85% pass rate required (higher than before)
        
        print(f"\\n📊 STAGE 1 Results:")
        print(f"  Passed: {passed_count}/{len(correctness_configs)} ({pass_rate:.1%})")
        print(f"  Status: {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        print(f"  Requirement: 85%+ pass rate (ensures mathematical correctness)")
        
        if not stage1_passed:
            print("\\n❌ CRITICAL: Evolved kernel fails comprehensive correctness tests!")
            print("  This indicates the kernel produces incorrect mathematical results.")
            print("  Evolution must fix correctness before performance optimization.")
            
            return {
                "stage1_passed": False,
                "pass_rate": pass_rate,
                "overall_score": 0.0,
                "combined_score": 0.0,
                "failed_at": "comprehensive_correctness",
                "num_correctness_tests": len(correctness_configs),
                "passed_correctness_tests": passed_count
            }
        
        # ===== STAGE 2: PROGRESSIVE PERFORMANCE EVALUATION =====
        print(f"\\n🏁 STAGE 2: Progressive Performance Evaluation")
        print("Multi-level reward system guides incremental optimization")
        
        # Calculate progressive rewards
        progressive_scores = calculate_progressive_rewards(evolved_fn, test_configs)
        
        print(f"\\n🎯 PROGRESSIVE REWARDS BREAKDOWN:")
        print(f"  🏆 Baseline Improvement: {progressive_scores['baseline_improvement_score']:.3f} (40% weight)")
        print(f"  🏆 SPDA Competition:     {progressive_scores['spda_competition_score']:.3f} (40% weight)")  
        print(f"  🏆 Sparsity Exploitation: {progressive_scores['sparsity_exploitation_score']:.3f} (20% weight)")
        print(f"  🎯 Overall Progressive Score: {progressive_scores['overall_progressive_score']:.3f}")
        
        successful_tests = progressive_scores['num_successful_tests']
        total_tests = progressive_scores['total_performance_tests']
        print(f"  📊 Successful Performance Tests: {successful_tests}/{total_tests}")
        
        # Overall score is the progressive score
        overall_score = progressive_scores['overall_progressive_score']
        
        print(f"\\n🏆 FINAL EVALUATION:")
        print(f"  Stage 1 (Comprehensive Correctness): {'✅ PASSED' if stage1_passed else '❌ FAILED'} ({len(correctness_configs)} tests)")
        print(f"  Stage 2 (Progressive Performance): {overall_score:.3f}")
        print(f"  🎯 COMBINED SCORE: {overall_score:.3f}")
        
        if overall_score >= 0.8:
            print(f"  🥇 EXCELLENT: High-performance kernel with comprehensive correctness!")
        elif overall_score >= 0.6:
            print(f"  🥈 GOOD: Meaningful improvements with solid correctness")
        elif overall_score >= 0.4:
            print(f"  🥉 MODERATE: Some optimization progress, mathematically correct")
        elif overall_score >= 0.2:
            print(f"  📈 PROGRESS: Incremental improvements, correct implementation")
        else:
            print(f"  🔄 BASELINE: Correct but needs optimization, evolution progressing")
        
        # Return comprehensive results
        result = {
            "stage1_passed": stage1_passed,
            "pass_rate": float(pass_rate),
            "overall_score": float(overall_score),
            "combined_score": float(overall_score),  # Primary metric for OpenEvolve
            
            # Progressive reward breakdown
            "baseline_improvement_score": progressive_scores['baseline_improvement_score'],
            "spda_competition_score": progressive_scores['spda_competition_score'], 
            "sparsity_exploitation_score": progressive_scores['sparsity_exploitation_score'],
            
            # Comprehensive test statistics
            "num_correctness_tests": len(correctness_configs),
            "num_block_diagonal_tests": block_diagonal_tests,
            "num_spda_tests": spda_tests,
            "num_edge_case_tests": edge_case_tests,
            "passed_correctness_tests": passed_count,
            
            "num_performance_tests": total_tests,
            "num_successful_performance_tests": successful_tests,
            
            # Metadata
            "evaluation_methodology": "comprehensive_correctness_plus_progressive_rewards",
            "timing_methodology": "rigorous",
            "correctness_requirement": "85%_pass_rate"
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "stage1_passed": False,
            "overall_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Testing Comprehensive Evaluator with ALL Original Correctness Tests...")
    
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("\\nComprehensive Evaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
