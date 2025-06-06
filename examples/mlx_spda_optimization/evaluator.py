"""
Enhanced Evaluator with Progressive Rewards + ALL ORIGINAL TEST SCENARIOS

This evaluator preserves ALL original test configurations while adding the progressive
reward system for incremental evolution guidance.

Key Features:
1. ALL original correctness tests preserved
2. ALL original performance test scenarios included  
3. Progressive reward system for incremental improvements
4. Comprehensive evaluation methodology
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
# PROGRESSIVE REWARD CONFIGURATION - FINE-GRAINED EVOLUTIONARY PRESSURE
# ============================================================================

# Progressive reward weights  
BASELINE_IMPROVEMENT_WEIGHT = 0.4    # 40% for beating initial program
SPDA_COMPETITION_WEIGHT = 0.4        # 40% for competing with SPDA
SPARSITY_EXPLOITATION_WEIGHT = 0.2   # 20% for consistent sparsity gains

# 🔥 MICRO-OPTIMIZATION REWARDS: Fine-grained baseline improvement detection
# Designed to create evolutionary pressure for even small optimizations (0.1% - 10%)
BASELINE_SPEEDUP_THRESHOLDS = [
    1.001,   # 0.1% improvement
    1.002,   # 0.2% improvement  
    1.005,   # 0.5% improvement
    1.01,    # 1% improvement
    1.02,    # 2% improvement
    1.05,    # 5% improvement
    1.1,     # 10% improvement
    1.2,     # 20% improvement
    1.5,     # 50% improvement
    2.0      # 100% improvement
]
BASELINE_REWARDS = [
    0.05,    # Small but meaningful reward for 0.1% gain
    0.1,     # 0.2% gain
    0.15,    # 0.5% gain
    0.25,    # 1% gain (current best gets ~0.25)
    0.35,    # 2% gain
    0.5,     # 5% gain
    0.65,    # 10% gain
    0.8,     # 20% gain
    0.9,     # 50% gain
    1.0      # 100% gain
]

# 🚀 INCREMENTAL SPDA COMPETITION: Start rewarding much earlier
# Create evolutionary pathway toward beating SPDA rather than requiring sudden breakthrough
SPDA_SPEEDUP_THRESHOLDS = [
    0.05,    # 5% of SPDA speed (terrible but measurable)
    0.1,     # 10% of SPDA speed  
    0.2,     # 20% of SPDA speed
    0.3,     # 30% of SPDA speed
    0.5,     # 50% of SPDA speed
    0.7,     # 70% of SPDA speed
    0.8,     # 80% of SPDA speed
    0.9,     # 90% of SPDA speed
    1.0,     # Match SPDA!
    1.2,     # 20% faster than SPDA
    1.5,     # 50% faster than SPDA
    2.0      # 100% faster than SPDA
]
SPDA_REWARDS = [
    0.01,    # Tiny reward for being measurably faster than worst-case
    0.02,    # 10% of SPDA speed
    0.05,    # 20% of SPDA speed
    0.1,     # 30% of SPDA speed
    0.2,     # 50% of SPDA speed (significant milestone)
    0.4,     # 70% of SPDA speed (approaching competitive)
    0.6,     # 80% of SPDA speed (very competitive)
    0.8,     # 90% of SPDA speed (almost there!)
    1.0,     # Match SPDA (major breakthrough!)
    1.0,     # Beat SPDA by 20%
    1.0,     # Beat SPDA by 50%
    1.0      # Beat SPDA by 100%
]

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
    """
    Create ALL original test configurations + comprehensive correctness tests
    
    This preserves EVERY test scenario from the original evaluator while adding
    progressive difficulty organization for reward calculation.
    """
    configs = []
    
    # ===== STAGE 1: COMPREHENSIVE CORRECTNESS TESTS =====
    # Block-diagonal correctness tests
    configs.extend([
        {
            "name": "small_uniform_blocks", 
            "B": 1, "H": 4, "L": 128, "D": 64,
            "block_sizes": [64, 64],  # 2 blocks of 64
            "test_type": "correctness"
        },
        {
            "name": "medium_uniform_blocks",
            "B": 1, "H": 8, "L": 512, "D": 64, 
            "block_sizes": [128, 128, 128, 128],  # 4 blocks of 128
            "test_type": "correctness"
        },
        {
            "name": "variable_blocks",
            "B": 1, "H": 8, "L": 768, "D": 64,
            "block_sizes": [256, 512],  # Variable sizes
            "test_type": "correctness"
        },
        {
            "name": "single_large_block",
            "B": 1, "H": 4, "L": 256, "D": 64,
            "block_sizes": [256],  # Single block (edge case)
            "test_type": "correctness"
        }
    ])
    
    # SPDA benchmark configurations for comprehensive correctness testing
    spda_correctness_configs = [
        # Small sizes for fast correctness testing - NO GQA to avoid complexity
        (1, 32, 32, 64, 16, 16, None),      # Basic small
        (1, 64, 64, 64, 16, 16, "bool"),    # Boolean mask
        (1, 128, 128, 64, 16, 16, "causal"), # Causal mask
        (1, 256, 256, 64, 16, 16, None),    # Medium size
        (1, 128, 128, 80, 16, 16, "bool"),  # Different head dim (PaLM)
        (2, 128, 128, 64, 16, 16, "causal"), # Batch size > 1
        (1, 512, 512, 64, 16, 16, "bool"),   # Larger size
        (1, 256, 256, 128, 8, 8, None),    # Large head dim, fewer heads
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
    
    # ===== STAGE 2: ALL ORIGINAL PERFORMANCE TESTS =====
    # These preserve ALL original test scenarios while adding difficulty organization
    
    # ORIGINAL: Basic sparsity progression
    configs.extend([
        {
            "name": "dense_2x256_sparse50",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256],  # 50% sparse
            "test_type": "performance",
            "difficulty": "baseline"
        },
        {
            "name": "medium_4x128_sparse75", 
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],  # 75% sparse
            "test_type": "performance",
            "difficulty": "medium"
        },
        {
            "name": "sparse_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8,  # 87.5% sparse
            "test_type": "performance",
            "difficulty": "hard"
        },
        {
            "name": "very_sparse_16x32_sparse93",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [32] * 16,  # 93.75% sparse
            "test_type": "performance",
            "difficulty": "expert"
        },
        {
            "name": "extreme_sparse_32x16_sparse96",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [16] * 32,  # 96.875% sparse
            "test_type": "performance",
            "difficulty": "extreme"
        }
    ])
    
    # ORIGINAL: Different sequence lengths
    configs.extend([
        {
            "name": "large_seq_8x128_sparse87",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128] * 8,  # Large sequences
            "test_type": "performance",
            "difficulty": "hard"
        },
        {
            "name": "huge_seq_16x128_sparse93",
            "B": 1, "H": 32, "L": 2048, "D": 64,
            "block_sizes": [128] * 16,  # Very large sequences
            "test_type": "performance",
            "difficulty": "expert"
        }
    ])
    
    # ORIGINAL: Different head dimensions
    configs.extend([
        {
            "name": "head80_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 80,
            "block_sizes": [64] * 8,  # PaLM head dim
            "test_type": "performance",
            "difficulty": "hard"
        },
        {
            "name": "head128_8x64_sparse87",
            "B": 1, "H": 16, "L": 512, "D": 128,
            "block_sizes": [64] * 8,  # Large head dim
            "test_type": "performance",
            "difficulty": "hard"
        }
    ])
    
    # ORIGINAL: Batch variations
    configs.extend([
        {
            "name": "batch4_8x64_sparse87",
            "B": 4, "H": 16, "L": 512, "D": 64,
            "block_sizes": [64] * 8,  # Medium batch
            "test_type": "performance",
            "difficulty": "hard"
        }
    ])
    
    # ORIGINAL: Real-world scenarios
    configs.extend([
        {
            "name": "bert_base_packing",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],  # BERT-style
            "test_type": "performance",
            "difficulty": "medium"
        },
        {
            "name": "longformer_sparse",
            "B": 1, "H": 16, "L": 2048, "D": 64,
            "block_sizes": [128] * 16,  # Longformer-style
            "test_type": "performance",
            "difficulty": "expert"
        },
        {
            "name": "packed_sequences_medium",
            "B": 2, "H": 12, "L": 512, "D": 64,
            "block_sizes": [128, 128, 128, 128],  # BERT-style packing
            "test_type": "performance",
            "difficulty": "medium"
        }
    ])
    
    # ORIGINAL: Extreme sparsity
    configs.extend([
        {
            "name": "tiny_blocks_64x8_sparse98",
            "B": 1, "H": 16, "L": 512, "D": 64,
            "block_sizes": [8] * 64,  # 98.4% sparse
            "test_type": "performance",
            "difficulty": "extreme"
        },
        {
            "name": "sparse_large_blocks",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [128, 128, 128, 128, 128, 128, 128, 128],  # 8 blocks = 87.5% sparse
            "test_type": "performance",
            "difficulty": "hard"
        }
    ])
    
    # ORIGINAL: Mixed patterns
    configs.extend([
        {
            "name": "mixed_sizes_pyramid",
            "B": 1, "H": 16, "L": 1024, "D": 64,
            "block_sizes": [512, 256, 128, 64, 32, 16, 8, 8],  # Pyramid
            "test_type": "performance",
            "difficulty": "expert"
        },
        {
            "name": "single_token_blocks",
            "B": 1, "H": 8, "L": 64, "D": 64,
            "block_sizes": [1] * 64,  # Extreme sparsity
            "test_type": "performance",
            "difficulty": "extreme"
        },
        {
            "name": "dense_packing_baseline",
            "B": 1, "H": 8, "L": 512, "D": 64,
            "block_sizes": [256, 256],  # Only 2 large blocks = less sparse
            "test_type": "performance",
            "difficulty": "baseline"
        },
        {
            "name": "very_sparse_packing",
            "B": 1, "H": 32, "L": 2048, "D": 64,
            "block_sizes": [256, 256, 256, 256, 256, 256, 256, 256],  # 8 blocks
            "test_type": "performance",
            "difficulty": "hard"
        },
        {
            "name": "extreme_sparse_packing",
            "B": 1, "H": 16, "L": 1024, "D": 128,
            "block_sizes": [64] * 16,  # 16 tiny blocks = extremely sparse
            "test_type": "performance",
            "difficulty": "extreme"
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
        tolerance = 1e-3 if q.dtype == mx.float32 else 2e-3  # Original tolerances
        passed = mse < tolerance and max_diff < 0.1 and not has_nan and not has_inf
        
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
            
            atol = 2e-3 if q.dtype == mx.float16 else 1e-4
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
    """Calculate multi-level progressive rewards with fine-grained evolutionary pressure"""
    
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
            "num_successful_tests": 0,
            "reward_breakdown": "No successful tests"
        }
    
    # LEVEL 1: MICRO-OPTIMIZATION BASELINE REWARDS (40% weight)
    baseline_scores = []
    baseline_speedups = []
    
    for result in evolved_results:
        config_name = result["config_name"]
        evolved_time = result["evolved_time"]
        
        # Get initial program performance for this config
        initial_time = _baseline_cache.initial_program_performance.get(config_name)
        if initial_time and initial_time > 0:
            speedup_vs_initial = initial_time / evolved_time
            baseline_speedups.append(speedup_vs_initial)
            
            # 🔥 FINE-GRAINED reward scaling - every 0.1% improvement gets rewarded!
            baseline_score = 0.0
            for i, threshold in enumerate(BASELINE_SPEEDUP_THRESHOLDS):
                if speedup_vs_initial >= threshold:
                    baseline_score = BASELINE_REWARDS[i]
                else:
                    break
            
            baseline_scores.append(baseline_score)
    
    baseline_improvement_score = np.mean(baseline_scores) if baseline_scores else 0.0
    avg_baseline_speedup = np.mean(baseline_speedups) if baseline_speedups else 1.0
    
    # LEVEL 2: INCREMENTAL SPDA COMPETITION REWARDS (40% weight)  
    spda_scores = []
    spda_speedups = []
    
    for result in evolved_results:
        config_name = result["config_name"]
        evolved_time = result["evolved_time"]
        
        # Get SPDA performance for this config
        spda_time = _baseline_cache.spda_performance.get(config_name)
        if spda_time and spda_time > 0:
            speedup_vs_spda = spda_time / evolved_time
            spda_speedups.append(speedup_vs_spda)
            
            # 🚀 INCREMENTAL reward scaling - reward progress toward SPDA!
            spda_score = 0.0
            for i, threshold in enumerate(SPDA_SPEEDUP_THRESHOLDS):
                if speedup_vs_spda >= threshold:
                    spda_score = SPDA_REWARDS[i]
                else:
                    break
            
            spda_scores.append(spda_score)
    
    spda_competition_score = np.mean(spda_scores) if spda_scores else 0.0
    avg_spda_speedup = np.mean(spda_speedups) if spda_speedups else 0.0
    
    # LEVEL 3: ENHANCED SPARSITY EXPLOITATION REWARDS (20% weight)
    # Reward consistent performance across different sparsity levels
    sparsity_groups = {}
    for result in evolved_results:
        sparsity = result["sparsity"]
        difficulty = result.get("difficulty", "unknown")
        
        if difficulty not in sparsity_groups:
            sparsity_groups[difficulty] = []
        sparsity_groups[difficulty].append(result)
    
    # 🎯 ENHANCED: More nuanced sparsity exploitation scoring
    num_difficulty_levels = len(sparsity_groups)
    if num_difficulty_levels >= 4:  # Excellent across many sparsity levels
        sparsity_exploitation_score = 1.0
    elif num_difficulty_levels >= 3:  # Good across multiple levels
        sparsity_exploitation_score = 0.8
    elif num_difficulty_levels >= 2:  # Decent across some levels
        sparsity_exploitation_score = 0.5
    elif num_difficulty_levels >= 1:  # Works on at least one level
        sparsity_exploitation_score = 0.2
    else:
        sparsity_exploitation_score = 0.0
    
    # COMBINE SCORES WITH WEIGHTS
    overall_progressive_score = (
        BASELINE_IMPROVEMENT_WEIGHT * baseline_improvement_score +     # 40% for beating initial program
        SPDA_COMPETITION_WEIGHT * spda_competition_score +             # 40% for competing with SPDA  
        SPARSITY_EXPLOITATION_WEIGHT * sparsity_exploitation_score     # 20% for sparsity consistency
    )
    
    # 🔍 DETAILED REWARD BREAKDOWN for debugging
    reward_breakdown = (
        f"Baseline: {avg_baseline_speedup:.4f}x→{baseline_improvement_score:.3f} | "
        f"SPDA: {avg_spda_speedup:.4f}x→{spda_competition_score:.3f} | "
        f"Sparsity: {num_difficulty_levels}lvls→{sparsity_exploitation_score:.3f}"
    )
    
    return {
        "baseline_improvement_score": float(baseline_improvement_score),
        "spda_competition_score": float(spda_competition_score),
        "sparsity_exploitation_score": float(sparsity_exploitation_score),
        "overall_progressive_score": float(overall_progressive_score),
        "num_successful_tests": len(evolved_results),
        "total_performance_tests": len(performance_configs),
        
        # 📊 DETAILED METRICS for analysis
        "avg_baseline_speedup": float(avg_baseline_speedup),
        "avg_spda_speedup": float(avg_spda_speedup),
        "num_difficulty_levels": num_difficulty_levels,
        "reward_breakdown": reward_breakdown
    }


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate(program_path: str) -> Dict[str, Union[bool, float, str, int]]:
    """
    Complete evaluation with ALL original test scenarios + progressive rewards
    
    This preserves EVERY original test configuration while adding progressive
    reward signals for incremental optimization guidance.
    """
    print(f"🚀 Evaluating Metal Kernel (Complete + Progressive): {program_path}")
    
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
        print("Preserving ALL original correctness requirements")
        
        test_configs = create_test_configurations()
        correctness_configs = [c for c in test_configs if c["test_type"] == "correctness"]
        
        print(f"  Running {len(correctness_configs)} correctness tests...")
        
        correctness_results = []
        passed_count = 0
        
        for config in correctness_configs:
            result = evaluate_correctness(evolved_fn, config)
            correctness_results.append(result)
            
            if result["passed"]:
                passed_count += 1
                print(f"    ✅ {config['name']}: PASSED (MSE: {result.get('mse', 0):.2e})")
            else:
                mse_val = result.get('mse', 1.0)
                mse_str = f"{mse_val:.2e}" if isinstance(mse_val, (int, float)) else str(mse_val)
                error_msg = result.get("error", f"MSE: {mse_str}")
                print(f"    ❌ {config['name']}: FAILED ({error_msg})")
        
        # Calculate pass rate
        pass_rate = passed_count / len(correctness_configs) if correctness_configs else 0.0
        stage1_passed = pass_rate >= 0.75  # 75% pass rate required
        
        print(f"\n📊 STAGE 1 Results:")
        print(f"  Passed: {passed_count}/{len(correctness_configs)} ({pass_rate:.1%})")
        print(f"  Status: {'✅ PASSED' if stage1_passed else '❌ FAILED'}")
        
        if not stage1_passed:
            return {
                "stage1_passed": False,
                "pass_rate": pass_rate,
                "overall_score": 0.0,
                "combined_score": 0.0,
                "failed_at": "correctness"
            }
        
        # ===== STAGE 2: ALL ORIGINAL PERFORMANCE TESTS + PROGRESSIVE REWARDS =====
        print(f"\n🏁 STAGE 2: ALL Original Performance Tests + Progressive Rewards")
        
        performance_configs = [c for c in test_configs if c["test_type"] == "performance"]
        print(f"  Running {len(performance_configs)} performance tests...")
        print("  Including ALL original test scenarios with progressive reward calculation")
        
        # Calculate progressive rewards
        progressive_scores = calculate_progressive_rewards(evolved_fn, test_configs)
        
        print(f"\n🎯 PROGRESSIVE REWARDS BREAKDOWN (Fine-Grained):")
        print(f"  🏆 Baseline Improvement: {progressive_scores['baseline_improvement_score']:.3f} (40% weight)")
        print(f"      ↳ Avg speedup vs initial: {progressive_scores.get('avg_baseline_speedup', 1.0):.4f}x")
        print(f"  🏆 SPDA Competition:     {progressive_scores['spda_competition_score']:.3f} (40% weight)")  
        print(f"      ↳ Avg speedup vs SPDA: {progressive_scores.get('avg_spda_speedup', 0.0):.4f}x")
        print(f"  🏆 Sparsity Exploitation: {progressive_scores['sparsity_exploitation_score']:.3f} (20% weight)")
        print(f"      ↳ Difficulty levels covered: {progressive_scores.get('num_difficulty_levels', 0)}")
        print(f"  🎯 Overall Progressive Score: {progressive_scores['overall_progressive_score']:.3f}")
        print(f"  📊 Detailed: {progressive_scores.get('reward_breakdown', 'N/A')}")
        
        successful_tests = progressive_scores['num_successful_tests']
        total_tests = progressive_scores['total_performance_tests']
        print(f"  📊 Successful Performance Tests: {successful_tests}/{total_tests}")
        
        # Overall score is the progressive score
        overall_score = progressive_scores['overall_progressive_score']
        
        print(f"\n🏆 FINAL EVALUATION:")
        print(f"  Stage 1 (Correctness): {'✅ PASSED' if stage1_passed else '❌ FAILED'} ({len(correctness_configs)} tests)")
        print(f"  Stage 2 (ALL Original Performance + Progressive): {overall_score:.3f} ({len(performance_configs)} tests)")
        print(f"  🎯 COMBINED SCORE: {overall_score:.3f}")
        
        if overall_score >= 0.8:
            print(f"  🥇 EXCELLENT: High-performance optimization with fine-grained rewards!")
        elif overall_score >= 0.6:
            print(f"  🥈 GOOD: Strong improvements detected by progressive reward system")
        elif overall_score >= 0.4:
            print(f"  🥉 MODERATE: Meaningful progress with enhanced evolutionary pressure")
        elif overall_score >= 0.2:
            print(f"  📈 PROGRESS: Micro-optimizations rewarded, evolution guided effectively")
        elif overall_score >= 0.05:
            print(f"  🔍 MICRO-GAINS: Fine-grained detection working, small improvements found")
        else:
            print(f"  🔄 BASELINE: Enhanced reward system ready for optimization discovery")
        
        # Return comprehensive results
        result = {
            "stage1_passed": stage1_passed,
            "pass_rate": float(pass_rate),
            "overall_score": float(overall_score),
            "combined_score": float(overall_score),  # Primary metric for OpenEvolve
            
            # Progressive reward breakdown (enhanced)
            "baseline_improvement_score": progressive_scores['baseline_improvement_score'],
            "spda_competition_score": progressive_scores['spda_competition_score'], 
            "sparsity_exploitation_score": progressive_scores['sparsity_exploitation_score'],
            
            # Fine-grained metrics for analysis
            "avg_baseline_speedup": progressive_scores.get('avg_baseline_speedup', 1.0),
            "avg_spda_speedup": progressive_scores.get('avg_spda_speedup', 0.0),
            "num_difficulty_levels": progressive_scores.get('num_difficulty_levels', 0),
            "reward_breakdown": progressive_scores.get('reward_breakdown', 'N/A'),
            
            # Test statistics
            "num_correctness_tests": len(correctness_configs),
            "num_performance_tests": total_tests,
            "num_successful_performance_tests": successful_tests,
            "passed_correctness_tests": passed_count,
            
            # Metadata
            "evaluation_methodology": "all_original_tests_plus_fine_grained_progressive_rewards",
            "timing_methodology": "rigorous",
            "reward_system_version": "fine_grained_v1.0"
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
    print("Testing Complete Evaluator with ALL Original Tests + Progressive Rewards...")
    
    import os
    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    
    if os.path.exists(initial_program_path):
        results = evaluate(initial_program_path)
        print("\nComplete Evaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
