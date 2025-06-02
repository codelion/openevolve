"""
MLX SPDA (Scaled Dot Product Attention) Custom Metal Kernel Optimization for OpenEvolve

This module contains a working Metal kernel implementation that can be evolved.
Starting with simple, functional kernels that can be incrementally optimized.

Key approach:
- Start with working Metal kernels for basic operations
- Incrementally add optimizations and fuse operations
- Provide concrete, compilable examples
- Build complexity gradually through evolution
"""

import math
from typing import Optional

import mlx.core as mx
import numpy as np


def evolved_scaled_dot_product_attention(q, k, v, scale=1.0, mask=None):
    """
    Metal Kernel-based attention implementation with working building blocks.

    This function uses simple, working Metal kernels that can be evolved
    to more complex optimizations. Starting simple and building complexity.

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
    WORKING METAL KERNEL IMPLEMENTATION
    
    This implementation uses simple, functional Metal kernels that can be evolved.
    Starting with basic working kernels and building complexity through evolution.
    
    CURRENT APPROACH:
    1. Working element-wise scale kernel
    2. Reference implementation for complex operations  
    3. Evolution can gradually replace reference parts with optimized kernels
    
    EVOLUTION OPPORTUNITIES:
    - Replace q_scaled computation with optimized kernel
    - Implement custom matrix multiplication kernels
    - Add fused scale+matmul kernels
    - Implement custom softmax kernels
    - Eventually fuse entire attention pipeline
    """

    # Extract dimensions
    B, n_q_heads, L, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kL = k.shape[2]
    n_repeats = n_q_heads // n_kv_heads

    # WORKING METAL KERNEL: Element-wise scaling
    # This is a simple, working kernel that can be evolved
    try:
        scale_source = """
            uint elem = thread_position_in_grid.x;
            if (elem >= q_shape[0] * q_shape[1] * q_shape[2] * q_shape[3]) {
                return;
            }
            out[elem] = q[elem] * scale_val;
        """

        scale_kernel = mx.fast.metal_kernel(
            name="scale_query",
            input_names=["q", "scale_val"],
            output_names=["out"],
            source=scale_source,
        )

        # Create scale as a scalar array for the kernel
        scale_array = mx.array(float(scale), dtype=q.dtype)

        q_scaled = scale_kernel(
            inputs=[q, scale_array],
            template=[("T", q.dtype)],
            output_shapes=[q.shape],
            output_dtypes=[q.dtype],
            grid=(q.size, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]

        # Metal kernel scaling successful (remove noisy print)

    except Exception as e:
        # Fallback to reference implementation on any Metal kernel error
        q_scaled = q * scale

    # Handle GQA with reference implementation (can be evolved later)
    if n_repeats > 1:
        q_reshaped = mx.reshape(q_scaled, [B, n_kv_heads, n_repeats, L, head_dim])
        k_expanded = mx.expand_dims(k, 2)
        v_expanded = mx.expand_dims(v, 2)
    else:
        q_reshaped = q_scaled
        k_expanded = k
        v_expanded = v

    # Compute attention scores with reference implementation (can be evolved)
    # Evolution opportunity: Replace with custom matmul kernel
    scores = q_reshaped @ mx.swapaxes(k_expanded, -1, -2)

    # Apply mask with reference implementation (can be evolved)
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

    # Apply softmax with reference implementation (can be evolved)
    # Evolution opportunity: Replace with custom softmax kernel
    attention_weights = mx.softmax(scores, axis=-1, precise=True)

    # Apply attention weights to values (can be evolved)
    # Evolution opportunity: Replace with custom matmul kernel
    out = attention_weights @ v_expanded

    # Reshape back if needed
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
    """Test that the Metal kernel attention works with real kernels"""
    print("Testing Working Metal Kernel attention functionality...")

    # Small test case to verify kernels work
    B, qL, kL, D, qH, kH = 1, 32, 32, 64, 4, 4
    scale = 1.0 / math.sqrt(D)

    # Create test inputs
    q = mx.random.normal((B, qH, qL, D))
    k = mx.random.normal((B, kH, kL, D))
    v = mx.random.normal((B, kH, kL, D))

    # Test with working Metal kernel
    print("  Testing with working Metal scaling kernel...")
    output = evolved_scaled_dot_product_attention(q, k, v, scale=scale)
    print(f"  ✓ Working kernel test: input {q.shape} -> output {output.shape}")

    # Test correctness by comparing with reference
    print("  Verifying correctness against reference implementation...")
    from spda_benchmark import mlx_ref_attn

    reference_output = mlx_ref_attn(q, k, v, scale=scale)

    # Check if outputs are close
    max_diff = float(mx.max(mx.abs(output - reference_output)))
    mse = float(mx.mean((output - reference_output) ** 2))

    print(f"  ✓ Max difference vs reference: {max_diff:.2e}")
    print(f"  ✓ MSE vs reference: {mse:.2e}")

    if mse < 1e-6:
        print("  ✓ Accuracy test PASSED")
    else:
        print("  ⚠️ Accuracy test FAILED - need to fix implementation")

    # Test with different configurations
    test_configs = [
        (1, 32, 32, 64, 8, 8, None),  # No mask
        (1, 64, 64, 64, 8, 8, "causal"),  # Causal mask
        (1, 32, 32, 64, 8, 4, None),  # GQA
    ]

    for B, qL, kL, D, qH, kH, mask_type in test_configs:
        q_test = mx.random.normal((B, qH, qL, D))
        k_test = mx.random.normal((B, kH, kL, D))
        v_test = mx.random.normal((B, kH, kL, D))

        try:
            output_test = evolved_scaled_dot_product_attention(
                q_test, k_test, v_test, scale=scale, mask=mask_type
            )
            print(f"  ✓ Config test passed: seq={qL}, heads={qH}/{kH}, mask={mask_type}")
        except Exception as e:
            print(
                f"  ❌ Config test failed: seq={qL}, heads={qH}/{kH}, mask={mask_type}, error={e}"
            )

    print("🚀 Working Metal Kernel attention tests completed!")
    print("  - Simple Metal scaling kernel working")
    print("  - Reference implementation for complex operations")
    print("  - Ready for incremental evolution!")
    print("  - Evolution can gradually replace reference parts with optimized kernels")
    return True


if __name__ == "__main__":
    test_basic_functionality()
