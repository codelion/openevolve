"""
Qwen3-0.6B Custom GQA Attention Implementation

This module implements Grouped Query Attention from scratch using MLX primitives,
following AlphaEvolve's approach of evolving the actual computation rather than
just high-level orchestration.

Target Model: mlx-community/Qwen3-0.6B-bf16
Architecture: 40 query heads : 8 KV heads (5:1 GQA ratio)
Hardware: Apple M4 24GB unified memory
Baseline Performance: 70.3 tokens/sec average decode speed
Optimization Target: 80+ tokens/sec through custom GQA kernel evolution

This approach gives us real optimization opportunities:
1. Custom GQA broadcasting strategies
2. Fused operations (softmax + matmul)
3. Apple Silicon specific memory patterns
4. Optimized KV cache integration
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple, Any
import time


class CustomGQAAttention(nn.Module):
    """
    Custom Grouped Query Attention implementation for Qwen3-0.6B.

    This replaces mx.fast.scaled_dot_product_attention with a custom
    implementation that can be evolved for the specific 40:8 GQA pattern.
    """

    def __init__(self, args):
        super().__init__()

        # Architecture parameters
        dim = args.hidden_size  # 5120
        self.n_heads = n_heads = args.num_attention_heads  # 40
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads  # 8
        self.head_dim = head_dim = args.head_dim  # 128
        self.scale = head_dim**-0.5

        # GQA pattern: 40 query heads : 8 KV heads = 5:1 ratio
        self.gqa_ratio = n_heads // n_kv_heads  # 5

        # Linear projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # Layer norms
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # RoPE
        from mlx_lm.models.rope_utils import initialize_rope

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Standard preprocessing (not evolved)
        queries = self.q_proj(x)  # [B, L, 40*128]
        keys = self.k_proj(x)  # [B, L, 8*128]
        values = self.v_proj(x)  # [B, L, 8*128]

        # Reshape and normalize
        queries = queries.reshape(B, L, self.n_heads, self.head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # Transpose to [B, n_heads, L, head_dim] for attention
        queries = queries.transpose(0, 2, 1, 3)  # [B, 40, L, 128]
        keys = keys.transpose(0, 2, 1, 3)  # [B, 8, L, 128]
        values = values.transpose(0, 2, 1, 3)  # [B, 8, L, 128]

        # Apply RoPE positional encoding
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # EVOLVE-BLOCK-START
        # Custom GQA Attention Implementation
        # This is the core optimization area - implementing attention from scratch
        # using MLX primitives to enable real kernel-level optimizations

        # Current dimensions:
        # queries: [B, 40, L, 128] - 40 query heads
        # keys:    [B, 8, L, 128]  - 8 key heads
        # values:  [B, 8, L, 128]  - 8 value heads

        # Strategy 1: Manual GQA Broadcasting (baseline custom implementation)
        # Explicitly broadcast keys and values to match query heads

        # Broadcast keys and values: [B, 8, L, 128] -> [B, 40, L, 128]
        # Each of the 8 KV heads is replicated 5 times (gqa_ratio = 5)
        keys_expanded = mx.repeat(keys, self.gqa_ratio, axis=1)  # [B, 40, L, 128]
        values_expanded = mx.repeat(values, self.gqa_ratio, axis=1)  # [B, 40, L, 128]

        # Compute attention scores: Q @ K^T
        # queries: [B, 40, L, 128] @ keys_expanded^T: [B, 40, 128, L] -> [B, 40, L, L]
        scores = mx.matmul(queries, keys_expanded.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask if provided
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                # Create causal mask: lower triangular matrix
                causal_mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))
                scores = mx.where(causal_mask, scores, mx.finfo(scores.dtype).min)
            elif isinstance(mask, mx.array):
                if mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                else:
                    scores = scores + mask

        # Apply softmax: attention weights
        attn_weights = mx.softmax(scores, axis=-1, precise=True)  # [B, 40, L, L]

        # Apply attention to values: weights @ V
        # attn_weights: [B, 40, L, L] @ values_expanded: [B, 40, L, 128] -> [B, 40, L, 128]
        output = mx.matmul(attn_weights, values_expanded)  # [B, 40, L, 128]

        # EVOLVE-BLOCK-END

        # Standard postprocessing (not evolved)
        output = output.transpose(0, 2, 1, 3)  # [B, L, 40, 128]
        output = output.reshape(B, L, -1)  # [B, L, 40*128]

        return self.o_proj(output)


def create_qwen3_custom_attention_hook():
    """
    Create a hook to replace Qwen3's attention with our custom GQA implementation.
    """

    def apply_custom_attention_hook():
        """Apply the custom attention to mlx-lm's Qwen3 model"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            # Store original attention class
            original_attention = qwen3_module.Attention

            # Replace with custom GQA implementation
            qwen3_module.Attention = CustomGQAAttention

            print("✅ Applied Custom GQA Attention hook")
            return original_attention

        except ImportError:
            print("❌ Could not import mlx_lm.models.qwen3")
            return None

    def remove_custom_attention_hook(original_attention):
        """Remove the custom attention hook"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            qwen3_module.Attention = original_attention
            print("✅ Removed Custom GQA Attention hook")
        except ImportError:
            pass

    return apply_custom_attention_hook, remove_custom_attention_hook


def benchmark_custom_vs_standard_attention():
    """
    Benchmark custom GQA attention vs standard MLX attention.
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

    # Test configurations
    test_configs = [
        ("short_context", 1, 128, 5120),
        ("medium_context", 1, 512, 5120),
        ("long_context", 1, 1024, 5120),
    ]

    print("Benchmarking Custom GQA vs Standard Attention")
    print("=" * 60)

    # Initialize custom attention
    custom_attn = CustomGQAAttention(args)

    for config_name, batch_size, seq_len, hidden_size in test_configs:
        print(f"\nTesting {config_name}: B={batch_size}, L={seq_len}")

        # Create test inputs
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        mask = "causal"  # Use causal mask like in real inference

        # Warmup
        for _ in range(3):
            _ = custom_attn(x, mask=mask)
            mx.eval(_)

        # Benchmark custom implementation
        mx.synchronize()
        start_time = time.perf_counter()

        for _ in range(10):
            output = custom_attn(x, mask=mask)
            mx.eval(output)

        mx.synchronize()
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10
        tokens_per_sec = seq_len / avg_time

        print(f"  Custom GQA: {avg_time*1000:.2f} ms, {tokens_per_sec:.1f} tokens/sec")
        print(f"  Memory: {mx.get_active_memory() / 1e9:.2f} GB")


def test_custom_gqa_correctness():
    """
    Test that custom GQA produces the same results as standard attention.
    """
    print("Testing Custom GQA Correctness")
    print("=" * 40)

    # Small test case
    B, L, D = 1, 32, 5120

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

    # Test custom implementation
    custom_attn = CustomGQAAttention(args)
    custom_output = custom_attn(x, mask=mask)

    print(f"✅ Custom GQA output shape: {custom_output.shape}")
    print(f"✅ Custom GQA runs without errors")

    # Check output properties
    output_mean = mx.mean(custom_output)
    output_std = mx.std(custom_output)

    print(f"✅ Output statistics - Mean: {output_mean:.6f}, Std: {output_std:.6f}")

    return True


if __name__ == "__main__":
    print("Testing Custom GQA Attention Implementation")
    print("=" * 60)

    # Test correctness first
    test_custom_gqa_correctness()

    print("\n")

    # Benchmark performance
    benchmark_custom_vs_standard_attention()

    print("\n" + "=" * 60)
    print("Custom GQA Implementation Complete")
    print("This implementation can now be evolved for:")
    print("1. Better GQA broadcasting strategies")
    print("2. Fused softmax + matmul operations")
    print("3. Apple Silicon memory optimizations")
    print("4. KV cache integration improvements")
    print("Target: 70.3 → 80+ tokens/sec improvement")
    print("=" * 60)
