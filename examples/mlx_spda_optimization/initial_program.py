"""
MLX SPDA (Scaled Dot Product Attention) Custom Metal Kernel Optimization for OpenEvolve

This module contains an evolvable implementation using MLX's custom Metal kernel API.
The goal is to evolve this implementation to beat the performance of mx.fast.scaled_dot_product_attention
by leveraging MLX's custom Metal kernel capabilities for direct GPU optimization.

Key approach:
- Use mx.fast.metal_kernel() for custom GPU kernels
- Write optimized Metal C++ code for attention computation
- Leverage Apple Silicon's unified memory architecture
- Enable kernel fusion and memory access optimization
- Design for maximum throughput and minimal memory bandwidth
"""

import math
from typing import Optional

import mlx.core as mx
import numpy as np


def evolved_scaled_dot_product_attention(q, k, v, scale=1.0, mask=None):
    """
    Custom Metal Kernel-based scaled dot product attention implementation.

    This function uses MLX's custom Metal kernel API to create optimized GPU kernels
    that compete with mx.fast.scaled_dot_product_attention by implementing:
    - Custom Metal C++ kernels for maximum performance
    - Fused operations to reduce memory bandwidth
    - Optimized memory access patterns for Apple Silicon
    - Specialized kernels for different scenarios (GQA, masking, etc.)

    Args:
        q: Query tensor [B, num_heads, L, head_dim]
        k: Key tensor [B, num_kv_heads, L_kv, head_dim]
        v: Value tensor [B, num_kv_heads, L_kv, head_dim]
        scale: Scaling factor (typically 1/sqrt(head_dim))
        mask: Attention mask or mask type string

    Returns:
        Attention output with same shape as queries
    """

    # EVOLVE-BLOCK-START
    """
    OPTIMIZATION TARGET: Beat mx.fast.scaled_dot_product_attention using custom Metal kernels
    
    STRATEGY: Use MLX's custom Metal kernel API to write high-performance GPU kernels
    that can compete with or exceed the performance of the built-in implementation.
    
    CUSTOM METAL KERNEL TECHNIQUES AVAILABLE:
    - mx.fast.metal_kernel() for direct Metal C++ kernel implementation
    - Kernel fusion opportunities (QK^T + scale + mask + softmax + matmul)
    - Memory access optimization with Metal threads and threadgroups
    - Apple Silicon unified memory exploitation
    - Atomic operations for complex reductions
    - Template programming for type specialization
    - Efficient threadgroup memory usage
    - Vectorized operations using Metal vector types
    
    PERFORMANCE TARGETS:
    - Match or exceed mx.fast.scaled_dot_product_attention performance
    - Maintain numerical accuracy (MSE < 1e-6)
    - Handle all configurations: GQA, masks, various sequence lengths
    - Optimize for Apple Silicon GPU architecture and memory patterns
    
    METAL KERNEL OPTIMIZATION STRATEGIES:
    - Fused attention kernel (reduce memory bandwidth)
    - Tiled computation for cache efficiency
    - Optimized threadgroup dispatching
    - Memory coalescing for better throughput
    - Specialized kernels per configuration type
    - Vectorized computation using Metal SIMD operations
    
    EXAMPLE KERNEL STRUCTURE:
    ```cpp
    template <typename T>
    [[kernel]] void fused_attention_kernel(
        const device T* q [[buffer(0)]],
        const device T* k [[buffer(1)]],
        const device T* v [[buffer(2)]],
        device T* out [[buffer(3)]],
        constant int& seq_len [[buffer(4)]],
        constant int& head_dim [[buffer(5)]],
        constant float& scale [[buffer(6)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]],
        uint3 threads_per_threadgroup [[threads_per_threadgroup]]
    ) {
        // Custom optimized attention computation
        // Fuse QK^T, scaling, masking, softmax, and final matmul
    }
    ```
    
    FORBIDDEN:
    - mx.fast.* functions (that's the target to beat!)
    - Only basic operations without kernel optimization
    """

    # Extract dimensions for kernel dispatch
    B, n_q_heads, L, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kL = k.shape[2]
    n_repeats = n_q_heads // n_kv_heads

    # For now, start with a simple custom kernel example and fallback to reference
    # This demonstrates the Metal kernel API usage pattern for evolution
    if mask is None and n_repeats == 1 and L <= 64:  # Small sequences only for demo
        # Simple element-wise kernel demonstration (not full attention yet)
        # This shows the Metal kernel API pattern that evolution can build upon
        source = """
            uint elem = thread_position_in_grid.x;
            if (elem >= q_shape[0] * q_shape[1] * q_shape[2] * q_shape[3]) {
                return;
            }
            
            // For now, just demonstrate kernel structure
            // Evolution should replace this with optimized attention computation
            out[elem] = q[elem] * T(0.1);  // Placeholder computation
        """

        demo_kernel = mx.fast.metal_kernel(
            name="demo_kernel",
            input_names=["q"],
            output_names=["out"],
            source=source,
        )

        # This is just a demo - evolution should replace with real attention
        try:
            demo_out = demo_kernel(
                inputs=[q],
                template=[("T", q.dtype)],
                output_shapes=[q.shape],
                output_dtypes=[q.dtype],
                grid=(q.size, 1, 1),
                threadgroup=(256, 1, 1),
            )[0]
            # Fall through to reference implementation since demo kernel isn't real attention
        except Exception as e:
            print(f"Metal kernel demo failed: {e}, falling back to reference")
            # Fall through to reference implementation

    # Fallback to reference implementation for all cases (for now)
    # TODO: Implement custom kernels for these cases as well
    # Use reference implementation temporarily - this should be replaced
    # with custom kernels for GQA and masking in evolved versions
    q_scaled = q * scale

    # Handle GQA
    if n_repeats > 1:
        q_reshaped = mx.reshape(q_scaled, [B, n_kv_heads, n_repeats, L, head_dim])
        k_expanded = mx.expand_dims(k, 2)
        v_expanded = mx.expand_dims(v, 2)
    else:
        q_reshaped = q_scaled
        k_expanded = k
        v_expanded = v

    # Compute scores
    scores = q_reshaped @ mx.swapaxes(k_expanded, -1, -2)

    # Apply mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            q_offset = max(0, kL - L)
            q_indices = mx.arange(q_offset, q_offset + L)
            k_indices = mx.arange(kL)
            causal_mask = q_indices[:, None] >= k_indices[None]
            scores = mx.where(causal_mask, scores, -mx.array(np.float32(np.inf)))
        elif hasattr(mask, "dtype") and mask.dtype == mx.bool_:
            if n_repeats > 1 and mask.ndim >= 3:
                if mask.shape[-3] == 1:
                    mask = mx.expand_dims(mask, -3)
                elif mask.shape[-3] == n_q_heads:
                    mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))
            scores = mx.where(mask, scores, -mx.array(np.float32(np.inf)))
        else:
            scores = scores + mask

    # Softmax
    attention_weights = mx.softmax(scores, axis=-1, precise=True)

    # Output
    out = attention_weights @ v_expanded

    # Reshape back
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, head_dim])

    return out
    # EVOLVE-BLOCK-END


def create_benchmark_attention_function():
    """
    Create the attention function that will be benchmarked.
    This matches the interface expected by spda_benchmark.py
    """
    return evolved_scaled_dot_product_attention


def test_basic_functionality():
    """Test that the custom Metal kernel attention works on basic inputs"""
    print("Testing Custom Metal Kernel attention functionality...")

    # Test case similar to spda_benchmark.py
    B, qL, kL, D, qH, kH = 1, 32, 32, 64, 8, 8  # Small size for demo
    scale = 1.0 / math.sqrt(D)

    # Create test inputs
    q = mx.random.normal((B, qH, qL, D))
    k = mx.random.normal((B, kH, kL, D))
    v = mx.random.normal((B, kH, kL, D))

    # Test without mask (should attempt custom kernel demo, then fallback)
    print("  Testing no mask (custom kernel demo + reference fallback)...")
    output = evolved_scaled_dot_product_attention(q, k, v, scale=scale)
    print(f"  ✓ No mask test: input {q.shape} -> output {output.shape}")

    # Test with causal mask (reference implementation)
    print("  Testing causal mask (reference implementation)...")
    output_causal = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    print(f"  ✓ Causal mask test: input {q.shape} -> output {output_causal.shape}")

    # Test with boolean mask (reference implementation)
    print("  Testing boolean mask (reference implementation)...")
    mask_bool = mx.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5
    output_bool = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_bool)
    print(f"  ✓ Boolean mask test: input {q.shape} -> output {output_bool.shape}")

    # Test grouped query attention (reference implementation)
    print("  Testing GQA (reference implementation)...")
    kH_gqa = 2  # Fewer KV heads
    k_gqa = mx.random.normal((B, kH_gqa, kL, D))
    v_gqa = mx.random.normal((B, kH_gqa, kL, D))
    output_gqa = evolved_scaled_dot_product_attention(q, k_gqa, v_gqa, scale=scale)
    print(f"  ✓ GQA test: Q={q.shape}, K={k_gqa.shape} -> output {output_gqa.shape}")

    # Test larger sequence (should skip Metal kernel demo)
    print("  Testing larger sequence (reference implementation)...")
    B_large, qL_large, kL_large = 1, 128, 128
    q_large = mx.random.normal((B_large, qH, qL_large, D))
    k_large = mx.random.normal((B_large, kH, kL_large, D))
    v_large = mx.random.normal((B_large, kH, kL_large, D))
    output_large = evolved_scaled_dot_product_attention(q_large, k_large, v_large, scale=scale)
    print(f"  ✓ Large sequence test: input {q_large.shape} -> output {output_large.shape}")

    print("🚀 All Custom Metal Kernel attention tests passed!")
    print("  - Metal kernel API structure demonstrated")
    print("  - Reference implementation working for all cases")
    print("  - Framework ready for evolution to optimize Metal kernels!")
    print("  - Evolution should replace demo kernel with real attention kernels")
    return True


if __name__ == "__main__":
    test_basic_functionality()
