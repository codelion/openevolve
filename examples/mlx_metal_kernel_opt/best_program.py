"""
Qwen3 Custom Metal Kernel for Grouped Query Attention (GQA) Optimization

This module implements a custom Metal kernel for Qwen3's 40:8 GQA pattern using
MLX's metal_kernel API. The kernel is designed to outperform mx.fast.scaled_dot_product_attention
by leveraging Apple Silicon specific optimizations and the 5:1 query-to-KV head ratio.

Target: Qwen3-0.6B with 40 query heads : 8 KV heads
Hardware: Apple M-series GPUs with unified memory
Baseline: Standard MLX-LM using mx.fast.scaled_dot_product_attention
Goal: 5-15% performance improvement through custom Metal kernel optimization

Evolution Target: The Metal kernel source code that computes GQA attention
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Any
import time


def qwen3_custom_gqa_attention(queries, keys, values, scale=1.0, mask=None):
    """
    Custom Metal kernel implementation for Qwen3 GQA attention.

    Args:
        queries: [B, num_heads=40, L, head_dim=128]
        keys: [B, num_kv_heads=8, L, head_dim=128]
        values: [B, num_kv_heads=8, L, head_dim=128]
        scale: Attention scaling factor (1/sqrt(head_dim))
        mask: Attention mask (None, "causal", or boolean tensor)

    Returns:
        Attention output [B, num_heads=40, L, head_dim=128]
    """

    B, num_heads, L, head_dim = queries.shape
    _, num_kv_heads, _, _ = keys.shape
    heads_per_kv = num_heads // num_kv_heads  # Should be 5 for Qwen3

    # Handle mask conversion
    if mask == "causal" or mask is None:
        # Create causal mask for autoregressive attention
        causal_mask = mx.triu(mx.ones((L, L), dtype=mx.bool_), k=1)
        mask_tensor = mx.logical_not(causal_mask)  # True where attention is allowed
        use_mask = True
    elif isinstance(mask, (mx.array, type(None))):
        if mask is None:
            mask_tensor = mx.ones((L, L), dtype=mx.bool_)
            use_mask = False
        else:
            mask_tensor = mask.astype(mx.bool_)
            use_mask = True
    else:
        # Raise error for unsupported mask types - no fallback
        raise ValueError(
            f"Unsupported mask type: {type(mask)}. Custom kernel requires None, 'causal', or mx.array mask."
        )

    # Expand mask to match batch and head dimensions if needed
    if mask_tensor.ndim == 2:
        mask_tensor = mx.broadcast_to(mask_tensor[None, None, :, :], (B, num_heads, L, L))
    elif mask_tensor.ndim == 3:
        mask_tensor = mx.broadcast_to(mask_tensor[:, None, :, :], (B, num_heads, L, L))

    # EVOLVE-BLOCK-START
    # Custom Metal kernel source for Qwen3 GQA optimization
    # This kernel leverages the 40:8 head ratio and Apple Silicon architecture
    kernel_source = """
    // Qwen3 GQA Metal Kernel - Optimized for 40:8 head pattern
    // Thread mapping: each thread processes one query position
    uint thread_id = thread_position_in_grid.x;
    uint head_idx = thread_position_in_grid.y; 
    uint batch_idx = thread_position_in_grid.z;
    uint query_pos = thread_id;
    
    // Bounds checking
    if (batch_idx >= BATCH_SIZE || head_idx >= NUM_HEADS || query_pos >= SEQ_LEN) {
        return;
    }
    
    // Extract scalar values from input arrays
    T scale_val = scale[0];
    bool use_mask_val = use_mask[0] > 0;
    
    // GQA mapping: determine which KV head corresponds to this query head
    uint kv_head_idx = head_idx / HEADS_PER_KV;  // 5 query heads per KV head
    
    // Pre-calculate base indices for memory access optimization
    const uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                        head_idx * (SEQ_LEN * HEAD_DIM) + 
                        query_pos * HEAD_DIM;
                        
    const uint k_base_start = batch_idx * (NUM_KV_HEADS * SEQ_LEN * HEAD_DIM) + 
                              kv_head_idx * (SEQ_LEN * HEAD_DIM);
                              
    const uint v_base_start = k_base_start;  // Values have same layout as keys
    
    const uint mask_base = batch_idx * (NUM_HEADS * SEQ_LEN * SEQ_LEN) + 
                           head_idx * (SEQ_LEN * SEQ_LEN) + 
                           query_pos * SEQ_LEN;
                           
    const uint out_base = q_base;
    
    // Use vector type for query_vec (e.g., float8 or half8 for better SIMD utilization)
    // HEAD_DIM is 128, so 16 vec<T, 8> elements
    vec<T, 8> query_vec_v[HEAD_DIM / 8];
    for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) {
        query_vec_v[d_vec] = ((device vec<T, 8>*) (queries + q_base))[d_vec];
    }
    
    // Pass 1: Compute max_score for numerical stability (online max)
    T max_score = T(-INFINITY);
    
    for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
        bool is_valid = use_mask_val ? mask[mask_base + key_pos] : true;
        
        T score;
        if (!is_valid) {
            score = T(-INFINITY); // Masked scores are -infinity, consistent with Pass 2
        } else {
            // Compute Q @ K^T for this key position using vectorized dot product
            const uint k_base = k_base_start + key_pos * HEAD_DIM;
            score = T(0.0); // Initialize score here
            
            for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
                score += dot(query_vec_v[d_vec], ((device vec<T, 8>*) (keys + k_base))[d_vec]);
            }
            
            // Apply attention scaling
            score *= scale_val;
        }
        max_score = max(max_score, score);
    }
    
    // Pass 2: Compute softmax denominator and weighted sum (online sum)
    T sum_exp = T(0.0);
    vec<T, 8> output_acc_v[HEAD_DIM / 8]; // Accumulator for output vector, use vec<T, 8>
    
    // Initialize output accumulator to zero
    for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) {
        output_acc_v[d_vec] = T(0.0);
    }

    for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
        bool is_valid = use_mask_val ? mask[mask_base + key_pos] : true;
        
        T current_score;
        if (!is_valid) {
            current_score = T(-INFINITY); // Masked scores are -infinity
        } else {
            // Recompute Q @ K^T for this key position
            const uint k_base = k_base_start + key_pos * HEAD_DIM;
            T score = T(0.0);
            for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
                score += dot(query_vec_v[d_vec], ((device vec<T, 8>*) (keys + k_base))[d_vec]);
            }
            current_score = score * scale_val;
        }

        // Apply softmax (exp and sum)
        T exp_score;
        if (current_score == T(-INFINITY)) {
            exp_score = T(0.0); // exp(-infinity) is 0
        } else {
            exp_score = exp(current_score - max_score);
        }
        sum_exp += exp_score;
        
        // Compute weighted sum of values
        if (exp_score > T(0.0)) { // Only add if exp_score is positive
            const uint v_base = v_base_start + key_pos * HEAD_DIM;
            for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
                output_acc_v[d_vec] += exp_score * ((device vec<T, 8>*) (values + v_base))[d_vec];
            }
        }
    }
    
    // Final normalization and write result to global memory
    if (sum_exp > T(0.0)) {
        for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
            output_acc_v[d_vec] /= sum_exp;
            ((device vec<T, 8>*) (output + out_base))[d_vec] = output_acc_v[d_vec];
        }
    } else {
        // Handle case where sum_exp is zero (e.g., all scores were masked or extremely small)
        // Set output to zero to avoid NaN/Inf results.
        for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
            ((device vec<T, 8>*) (output + out_base))[d_vec] = T(0.0);
        }
    }
    """
    # EVOLVE-BLOCK-END

    try:
        # Prepare kernel inputs
        scale_tensor = mx.array([scale], dtype=queries.dtype)
        use_mask_tensor = mx.array([1 if use_mask else 0], dtype=mx.int32)

        # Create and execute custom Metal kernel
        kernel = mx.fast.metal_kernel(
            name="qwen3_gqa_attention_kernel",
            input_names=["queries", "keys", "values", "mask", "scale", "use_mask"],
            output_names=["output"],
            source=kernel_source,
        )

        # Optimize thread group size for Apple Silicon
        threadgroup_size = min(32, L)  # Adapt to sequence length

        # Execute kernel
        outputs = kernel(
            inputs=[queries, keys, values, mask_tensor, scale_tensor, use_mask_tensor],
            output_shapes=[(B, num_heads, L, head_dim)],
            output_dtypes=[queries.dtype],
            grid=(L, num_heads, B),  # (SEQ_LEN, NUM_HEADS, BATCH_SIZE)
            threadgroup=(threadgroup_size, 1, 1),
            template=[
                ("T", queries.dtype),
                ("BATCH_SIZE", B),
                ("NUM_HEADS", num_heads),
                ("NUM_KV_HEADS", num_kv_heads),
                ("SEQ_LEN", L),
                ("HEAD_DIM", head_dim),
                ("HEADS_PER_KV", heads_per_kv),
            ],
        )

        return outputs[0]

    except Exception as e:
        # No fallback - let the custom kernel failure propagate for proper scoring
        print(f"❌ Custom GQA kernel failed: {e}")
        raise RuntimeError(f"Custom Metal kernel execution failed: {e}") from e


class CustomGQAAttention(nn.Module):
    """
    Qwen3 attention module with custom Metal kernel optimization.

    This module integrates the custom Metal kernel while maintaining
    compatibility with the standard MLX-LM interface.
    """

    def __init__(self, args):
        super().__init__()

        # Standard Qwen3 parameters
        dim = args.hidden_size  # 5120
        self.n_heads = n_heads = args.num_attention_heads  # 40
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads  # 8
        head_dim = args.head_dim  # 128
        self.scale = head_dim**-0.5

        # Standard MLX-LM projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # Standard MLX-LM norms
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # Standard MLX-LM RoPE
        try:
            from mlx_lm.models.rope_utils import initialize_rope

            self.rope = initialize_rope(
                head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )
        except ImportError:
            print("⚠️ Could not import mlx_lm rope_utils, using basic RoPE")
            self.rope = None

        print(f"🔧 Initialized Custom Metal GQA Attention")
        print(f"   📊 Architecture: {n_heads}:{n_kv_heads} heads ({n_heads//n_kv_heads}:1 ratio)")
        print(f"   🎯 Head dimension: {head_dim}")
        print(f"   ⚡ Using custom Metal kernel for GQA optimization")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Standard preprocessing (already optimized, don't evolve)
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Standard RoPE application (already optimized, don't evolve)
        if cache is not None:
            if self.rope is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            if self.rope is not None:
                queries = self.rope(queries)
                keys = self.rope(keys)

        # CORE INNOVATION: Custom Metal kernel for GQA attention
        output = qwen3_custom_gqa_attention(queries, keys, values, scale=self.scale, mask=mask)

        # Standard postprocessing (already optimized, don't evolve)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


def create_metal_qwen3_optimization_hook():
    """
    Create hooks to replace Qwen3's attention with Metal kernel optimized version.
    """

    def apply_optimization_hook():
        """Apply the Metal kernel optimized attention"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            # Store original attention class
            original_attention = qwen3_module.Attention

            # Replace with Metal optimized implementation
            qwen3_module.Attention = CustomGQAAttention

            print("✅ Applied Custom Metal GQA Attention hook")
            return original_attention

        except ImportError:
            print("❌ Could not import mlx_lm.models.qwen3")
            return None

    def remove_optimization_hook(original_attention):
        """Remove the optimization hook"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            qwen3_module.Attention = original_attention
            print("✅ Removed Custom Metal GQA Attention hook")
        except ImportError:
            pass

    return apply_optimization_hook, remove_optimization_hook


def benchmark_metal_gqa_optimization():
    """
    Benchmark Metal kernel optimized GQA attention against MLX baseline.
    """

    # Qwen3-0.6B configuration
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

    # Test configurations for Metal kernel validation
    test_configs = [
        ("short_sequence", 1, 128, 5120),
        ("medium_sequence", 1, 512, 5120),
        ("long_sequence", 1, 1024, 5120),
        ("max_sequence", 1, 2048, 5120),
    ]

    print("Benchmarking Custom Metal GQA Kernel vs MLX Baseline")
    print("=" * 70)

    # Initialize Metal optimized attention
    metal_attn = CustomGQAAttention(args)

    for config_name, batch_size, seq_len, hidden_size in test_configs:
        print(f"\nTesting {config_name}: B={batch_size}, L={seq_len}")

        # Create test inputs
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        mask = "causal"

        # Warmup runs
        for _ in range(3):
            _ = metal_attn(x, mask=mask)
            mx.eval(_)

        # Benchmark Metal optimized implementation
        mx.synchronize()
        start_time = time.perf_counter()

        for _ in range(10):
            output = metal_attn(x, mask=mask)
            mx.eval(output)

        mx.synchronize()
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10
        tokens_per_sec = seq_len / avg_time

        print(f"  Metal GQA: {avg_time*1000:.2f} ms, {tokens_per_sec:.1f} tokens/sec")
        print(f"  Memory: {mx.get_active_memory() / 1e9:.2f} GB")


def test_metal_gqa_correctness():
    """
    Test that Metal kernel implementation produces correct results.
    """
    print("Testing Custom Metal GQA Correctness")
    print("=" * 50)

    # Test configuration
    B, L, D = 1, 64, 5120

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

    # Create test input
    x = mx.random.normal((B, L, D))
    mask = "causal"

    # Test Metal optimized implementation
    metal_attn = CustomGQAAttention(args)
    output = metal_attn(x, mask=mask)

    print(f"✅ Metal GQA output shape: {output.shape}")

    # Check for valid output
    has_nan = bool(mx.any(mx.isnan(output)))
    has_inf = bool(mx.any(mx.isinf(output)))

    print(f"✅ Has NaN: {has_nan}, Has Inf: {has_inf}")

    # Check output statistics
    output_mean = float(mx.mean(output))
    output_std = float(mx.std(output))

    print(f"✅ Output statistics - Mean: {output_mean:.6f}, Std: {output_std:.6f}")

    # Test direct kernel function
    print("\n=== Testing Direct Kernel Function ===")
    B, H, L, D = 1, 40, 128, 128
    q = mx.random.normal((B, H, L, D))
    k = mx.random.normal((B, 8, L, D))  # 8 KV heads
    v = mx.random.normal((B, 8, L, D))
    scale = 1.0 / math.sqrt(D)

    kernel_output = qwen3_custom_gqa_attention(q, k, v, scale=scale, mask="causal")
    print(f"✅ Direct kernel output shape: {kernel_output.shape}")

    kernel_mean = float(mx.mean(kernel_output))
    kernel_std = float(mx.std(kernel_output))
    print(f"✅ Direct kernel stats - Mean: {kernel_mean:.6f}, Std: {kernel_std:.6f}")

    return True


if __name__ == "__main__":
    print("Custom Metal Kernel Qwen3 GQA Optimization")
    print("=" * 70)

    # Test correctness first
    test_metal_gqa_correctness()

    print("\n")

    # Benchmark performance
    benchmark_metal_gqa_optimization()

    print("\n" + "=" * 70)
    print("Ready for Metal Kernel Evolution")
    print("Evolution focus:")
    print("1. 🔧 Metal kernel source code optimization")
    print("2. 💾 Memory access pattern improvements for Apple Silicon")
    print("3. 🎯 GQA-specific optimizations for 40:8 head ratio")
    print("4. ⚡ Vectorization and SIMD optimization")
    print("5. 🚀 Thread group and grid configuration tuning")
    print("Target: 5-15% performance improvement through Metal kernel innovation")
    print("=" * 70)
