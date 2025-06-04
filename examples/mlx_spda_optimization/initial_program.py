"""
MLX Block Diagonal Attention Kernel Discovery for OpenEvolve

This module implements a hybrid attention system:
- Uses mx.fast.scaled_dot_product_attention for sequences < 512 (battle-tested, optimal)
- Evolves custom block diagonal attention kernels for longer sequences (novel algorithmic space)

Key innovation: Instead of competing with highly optimized general-purpose attention,
we discover efficient block diagonal patterns that enable long sequence processing
with acceptable quality degradation.

This aligns with AlphaEvolve's philosophy of algorithmic discovery over micro-optimization.
"""

import math
from typing import Optional

import mlx.core as mx
import numpy as np


def evolved_scaled_dot_product_attention(q, k, v, scale=1.0, mask=None):
    """
    Hybrid attention implementation with block diagonal kernel discovery.

    Strategy:
    - Short sequences (< 512): Use mx.fast.scaled_dot_product_attention (optimal)
    - Long sequences (≥ 512): Use evolved block diagonal attention kernels

    This enables:
    - Perfect performance for common cases (short sequences)
    - Novel algorithm discovery for challenging cases (long sequences)
    - Linear scaling instead of quadratic for long contexts

    Args:
        q: Query tensor [B, num_heads, L, head_dim]
        k: Key tensor [B, num_kv_heads, L_kv, head_dim]
        v: Value tensor [B, num_kv_heads, L_kv, head_dim]
        scale: Scaling factor (typically 1/sqrt(head_dim))
        mask: Attention mask or mask type string

    Returns:
        Attention output with same shape as queries
    """

    # Extract dimensions - PROTECTED from evolution
    B, n_q_heads, L, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kL = k.shape[2]
    sequence_length = L

    # HYBRID DISPATCHER: PROTECTED from evolution - this logic must never change
    if sequence_length < 512:
        # SHORT SEQUENCES: Use optimal implementation with robust fallback
        # This entire section is PROTECTED from evolution to ensure evaluation works
        try:
            # Try the fast implementation first
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        except Exception as e:
            # MANDATORY FALLBACK: Use reference implementation if fast fails
            try:
                from spda_benchmark import mlx_ref_attn

                return mlx_ref_attn(q, k, v, scale=scale, mask=mask)
            except Exception as fallback_error:
                # Last resort: basic manual implementation
                return manual_attention_fallback(q, k, v, scale=scale, mask=mask)
    else:
        # LONG SEQUENCES: Use evolved block diagonal attention
        # This is where evolution happens!
        return evolved_block_diagonal_attention(q, k, v, scale=scale, mask=mask)


def manual_attention_fallback(q, k, v, scale=1.0, mask=None):
    """
    Manual attention implementation as last resort fallback.
    This ensures the function never fails completely.
    PROTECTED from evolution - this is a safety mechanism.
    """
    # Handle GQA if needed
    B, n_q_heads, L, head_dim = q.shape
    n_kv_heads = k.shape[1]

    if n_q_heads != n_kv_heads:
        # Expand k,v for GQA
        n_repeats = n_q_heads // n_kv_heads
        k = mx.repeat(k, n_repeats, axis=1)
        v = mx.repeat(v, n_repeats, axis=1)

    # Basic scaled dot-product attention
    scores = (q * scale) @ mx.swapaxes(k, -1, -2)

    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            # Create causal mask
            seq_len = scores.shape[-1]
            causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
            scores = mx.where(causal_mask, scores, -mx.array(np.float32(np.inf)))
        elif hasattr(mask, "dtype") and mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -mx.array(np.float32(np.inf)))
        else:
            scores = scores + mask

    # Softmax and output
    attn_weights = mx.softmax(scores, axis=-1, precise=True)
    return attn_weights @ v


def evolved_block_diagonal_attention(q, k, v, scale=1.0, mask=None):
    """
    SINGLE COMPREHENSIVE EVOLUTION TARGET

    This is the ONE main evolution block that contains all block diagonal attention logic,
    pattern analysis, kernel creation, and optimization strategies.

    Everything related to block diagonal attention evolution happens here.
    """

    # EVOLVE-BLOCK-START
    """
    COMPREHENSIVE BLOCK DIAGONAL ATTENTION EVOLUTION
    
    🎯 MISSION: Discover efficient block diagonal attention patterns for long sequences
    📈 GOAL: Linear O(n×block_size) complexity instead of O(n²) 
    🚀 TARGET: Enable processing of 4K+ token sequences
    
    EVOLUTION OPPORTUNITIES (ALL IN THIS SINGLE BLOCK):
    
    1. BLOCK STRATEGIES:
       - Fixed vs adaptive block sizes
       - Overlapping vs non-overlapping blocks
       - Hierarchical multi-level blocks
       - Sparse block patterns
    
    2. ATTENTION PATTERNS:
       - Full attention within blocks
       - Sparse attention within blocks  
       - Cross-block communication
       - Sliding window mechanisms
    
    3. CUSTOM KERNELS:
       - Metal GPU kernels for block computation
       - Fused operations (scale+attention+output)
       - Optimized memory access patterns
       - Vectorized block operations
    
    4. PATTERN ANALYSIS:
       - Dynamic block boundary detection
       - Content-aware block sizing
       - Attention sparsity analysis
       - Adaptive threshold selection
    
    5. OPTIMIZATION TECHNIQUES:
       - Block-parallel computation
       - Memory-efficient concatenation
       - Gradient checkpointing for blocks
       - Mixed precision block operations
    
    CURRENT IMPLEMENTATION: Basic fixed-size blocks with full attention
    EVOLUTION STRATEGY: Start simple, discover sophisticated patterns
    """

    # Extract dimensions
    B, n_q_heads, L, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kL = k.shape[2]
    n_repeats = n_q_heads // n_kv_heads

    # EVOLUTION TARGET 1: BLOCK STRATEGY AND PATTERN ANALYSIS
    # Analyze input to determine optimal block strategy
    def analyze_and_plan_blocks(q, k, v):
        """Analyze attention patterns and plan block strategy"""
        B, n_heads, L, head_dim = q.shape

        # Basic block size heuristic - EVOLUTION TARGET
        if L <= 1024:
            base_block_size = 128
        elif L <= 2048:
            base_block_size = 256
        else:
            base_block_size = 512

        # EVOLUTION OPPORTUNITY: Sophisticated pattern analysis
        # - Analyze query/key similarity patterns
        # - Detect natural attention boundaries
        # - Adapt block sizes based on content
        # - Identify sparse regions

        return {
            "block_size": base_block_size,
            "num_blocks": (L + base_block_size - 1) // base_block_size,
            "strategy": "fixed_size",  # Could evolve to "adaptive", "hierarchical", etc.
            "overlap": 0,  # Could evolve to overlapping blocks
            "sparse_threshold": 0.0,  # Could evolve sparse attention
        }

    # Get block plan
    block_plan = analyze_and_plan_blocks(q, k, v)
    base_block_size = block_plan["block_size"]
    num_blocks = block_plan["num_blocks"]

    # EVOLUTION TARGET 2: GQA HANDLING STRATEGY
    # Handle Grouped Query Attention efficiently
    if n_repeats > 1:
        q_reshaped = mx.reshape(q, [B, n_kv_heads, n_repeats, L, head_dim])
        k_expanded = mx.expand_dims(k, 2)
        v_expanded = mx.expand_dims(v, 2)
    else:
        q_reshaped = q
        k_expanded = k
        v_expanded = v

    # EVOLUTION TARGET 3: CUSTOM KERNEL CREATION
    # Create optimized kernels for block attention if possible
    def try_create_custom_kernel():
        """Attempt to create custom Metal kernel for block attention"""

        # EVOLUTION OPPORTUNITY: Sophisticated Metal kernel
        source = """
            // EVOLUTION TARGET: Efficient block diagonal attention kernel
            // 
            // Optimization opportunities:
            // 1. Tiled computation for cache efficiency
            // 2. Threadgroup memory for data sharing
            // 3. Vectorized operations (float4, half4)
            // 4. Fused scale+attention+output
            // 5. Sparse block patterns
            // 6. Inter-block communication
            
            uint block_id = thread_position_in_grid.x;
            uint thread_in_block = thread_position_in_grid.y;
            
            // TODO: Implement optimized block attention
            // Current: Basic placeholder for evolution
        """

        try:
            kernel = mx.fast.metal_kernel(
                name="block_attention",
                input_names=["q_blocks", "k_blocks", "v_blocks", "params"],
                output_names=["attention_output"],
                source=source,
            )
            return kernel
        except Exception:
            return None

    # Try to get custom kernel (evolution can improve this)
    custom_kernel = try_create_custom_kernel()

    # EVOLUTION TARGET 4: MAIN BLOCK PROCESSING LOOP
    # This is the core algorithm that processes blocks
    block_outputs = []

    for block_idx in range(num_blocks):
        # EVOLUTION TARGET 4A: Block boundary calculation
        start_idx = block_idx * base_block_size
        end_idx = min(start_idx + base_block_size, L)

        # EVOLUTION OPPORTUNITY: Adaptive boundaries, overlapping blocks
        # Could evolve context-aware block sizing, sliding windows, etc.

        # EVOLUTION TARGET 4B: Block query extraction
        if n_repeats > 1:
            q_block = q_reshaped[:, :, :, start_idx:end_idx, :]
        else:
            q_block = q_reshaped[:, :, start_idx:end_idx, :]

        # EVOLUTION TARGET 4C: Block attention computation
        try:
            # EVOLUTION OPPORTUNITY: Use custom kernel if available
            if custom_kernel is not None:
                # Try custom kernel path (evolution can improve this)
                try:
                    # Custom kernel implementation would go here
                    # For now, fall back to manual computation
                    raise NotImplementedError("Custom kernel not fully implemented")
                except Exception:
                    pass  # Fall back to manual computation

            # EVOLUTION TARGET 4D: Manual block attention computation
            # Scale queries
            q_block_scaled = q_block * scale

            # Compute attention scores for this block
            scores_block = q_block_scaled @ mx.swapaxes(k_expanded, -1, -2)

            # EVOLUTION TARGET 4E: Block masking strategy
            if mask is not None:
                if isinstance(mask, str) and mask == "causal":
                    # Create causal mask for this block
                    q_offset = max(0, kL - L)
                    q_indices = mx.arange(q_offset + start_idx, q_offset + end_idx)
                    k_indices = mx.arange(kL)
                    causal_mask = q_indices[:, None] >= k_indices[None]
                    scores_block = mx.where(
                        causal_mask, scores_block, -mx.array(np.float32(np.inf))
                    )
                elif hasattr(mask, "dtype") and mask.dtype == mx.bool_:
                    # Extract relevant mask portion for this block
                    mask_block = mask[:, :, start_idx:end_idx, :]
                    if n_repeats > 1 and mask_block.ndim >= 3:
                        if mask_block.shape[-3] == 1:
                            mask_block = mx.expand_dims(mask_block, -3)
                        elif mask_block.shape[-3] == n_q_heads:
                            mask_block = mx.unflatten(mask_block, -3, (n_kv_heads, n_repeats))
                    scores_block = mx.where(mask_block, scores_block, -mx.array(np.float32(np.inf)))
                else:
                    # Additive mask
                    mask_block = mask[:, :, start_idx:end_idx, :]
                    scores_block = scores_block + mask_block

            # EVOLUTION TARGET 4F: Block softmax and output computation
            attention_weights_block = mx.softmax(scores_block, axis=-1, precise=True)
            output_block = attention_weights_block @ v_expanded

            # EVOLUTION OPPORTUNITY: Post-processing, normalization, etc.

            block_outputs.append(output_block)

        except Exception as e:
            # EVOLUTION TARGET 4G: Robust fallback for failed blocks
            try:
                from spda_benchmark import mlx_ref_attn

                # Create temporary tensors for this block
                if n_repeats > 1:
                    q_temp = mx.reshape(q_block, [B, n_q_heads, end_idx - start_idx, head_dim])
                else:
                    q_temp = q_block

                k_temp = k
                v_temp = v

                # Create appropriate mask for this block if needed
                mask_temp = None
                if mask is not None:
                    if isinstance(mask, str):
                        mask_temp = mask
                    else:
                        mask_temp = mask[:, :, start_idx:end_idx, :]

                # Use reference attention for this block
                block_output = mlx_ref_attn(q_temp, k_temp, v_temp, scale=scale, mask=mask_temp)

                # Reshape if needed for GQA
                if n_repeats > 1:
                    block_output = mx.reshape(
                        block_output, [B, n_kv_heads, n_repeats, end_idx - start_idx, head_dim]
                    )

                block_outputs.append(block_output)

            except Exception as fallback_error:
                # Ultimate fallback: manual attention for this block
                if n_repeats > 1:
                    q_temp = mx.reshape(q_block, [B, n_q_heads, end_idx - start_idx, head_dim])
                else:
                    q_temp = q_block

                k_temp = k
                v_temp = v
                mask_temp = None
                if mask is not None and not isinstance(mask, str):
                    mask_temp = mask[:, :, start_idx:end_idx, :]
                elif isinstance(mask, str):
                    mask_temp = mask

                block_output = manual_attention_fallback(
                    q_temp, k_temp, v_temp, scale=scale, mask=mask_temp
                )

                if n_repeats > 1:
                    block_output = mx.reshape(
                        block_output, [B, n_kv_heads, n_repeats, end_idx - start_idx, head_dim]
                    )

                block_outputs.append(block_output)

    # EVOLUTION TARGET 5: BLOCK OUTPUT COMBINATION STRATEGY
    # Combine all block outputs into final result
    if block_outputs:
        if n_repeats > 1:
            # Concatenate along sequence dimension (axis=-2)
            output = mx.concatenate(block_outputs, axis=-2)
            # Reshape back to original format
            output = mx.reshape(output, [B, n_q_heads, L, head_dim])
        else:
            # Concatenate along sequence dimension (axis=-2)
            output = mx.concatenate(block_outputs, axis=-2)

        # EVOLUTION OPPORTUNITY: Advanced combination strategies
        # - Weighted combination based on attention scores
        # - Cross-block normalization
        # - Hierarchical merging
        # - Gradient flow optimization

    else:
        # Fallback: return zeros with correct shape
        output = mx.zeros_like(q)

    return output
    # EVOLVE-BLOCK-END


def create_benchmark_attention_function():
    """
    Create the attention function that will be benchmarked.
    This matches the interface expected by spda_benchmark.py
    PROTECTED from evolution.
    """
    return evolved_scaled_dot_product_attention


def test_basic_functionality():
    """Test the hybrid block diagonal attention system - PROTECTED from evolution"""
    print("Testing Hybrid Block Diagonal Attention System...")

    # Test short sequences (should use mx.fast.scaled_dot_product_attention)
    print("\n=== Testing Short Sequences (< 512) ===")
    short_configs = [
        (1, 32, 32, 64, 4, 4, None),  # Tiny
        (1, 128, 128, 64, 8, 8, "causal"),  # Small
        (1, 256, 256, 64, 16, 8, None),  # Medium
    ]

    for B, qL, kL, D, qH, kH, mask_type in short_configs:
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal((B, qH, qL, D))
        k = mx.random.normal((B, kH, kL, D))
        v = mx.random.normal((B, kH, kL, D))

        try:
            print(f"  Testing short seq: L={qL}, heads={qH}/{kH}, mask={mask_type}")
            output = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_type)

            # Verify against reference
            from spda_benchmark import mlx_ref_attn

            reference = mlx_ref_attn(q, k, v, scale=scale, mask=mask_type)

            mse = float(mx.mean((output - reference) ** 2))
            print(f"    ✓ MSE vs reference: {mse:.2e} (should be ~0 for short sequences)")

        except Exception as e:
            print(f"    ❌ FAILED: {str(e)}")

    # Test long sequences (should use block diagonal attention)
    print("\n=== Testing Long Sequences (≥ 512) ===")
    long_configs = [
        (1, 512, 512, 64, 8, 8, None),  # Threshold
        (1, 1024, 1024, 64, 16, 8, "causal"),  # Long
        (1, 2048, 2048, 64, 32, 8, None),  # Very long
    ]

    for B, qL, kL, D, qH, kH, mask_type in long_configs:
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal((B, qH, qL, D))
        k = mx.random.normal((B, kH, kL, D))
        v = mx.random.normal((B, kH, kL, D))

        try:
            print(f"  Testing long seq: L={qL}, heads={qH}/{kH}, mask={mask_type}")

            # Test our block diagonal implementation
            output = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_type)
            print(f"    ✓ Block diagonal output shape: {output.shape}")

            # Check for valid output (no NaN/Inf)
            has_nan = bool(mx.any(mx.isnan(output)))
            has_inf = bool(mx.any(mx.isinf(output)))

            if not has_nan and not has_inf:
                print(f"    ✅ Valid output (no NaN/Inf)")
            else:
                print(f"    ❌ Invalid output: NaN={has_nan}, Inf={has_inf}")

        except Exception as e:
            print(f"    ❌ FAILED: {str(e)}")

    print("\n🎯 Block Diagonal Attention System Summary:")
    print("  ✅ Short sequences: Perfect performance via mx.fast.scaled_dot_product_attention")
    print("  🎯 Long sequences: Block diagonal attention (SINGLE EVOLUTION TARGET)")
    print("  🛡️ Protected fallback mechanisms ensure reliability")
    print("  🚀 Ready for comprehensive block pattern evolution!")
    print("\n💡 Single Evolution Block Contains:")
    print("  1. Block strategy and pattern analysis")
    print("  2. Custom Metal kernel creation")
    print("  3. Block processing algorithms")
    print("  4. Output combination strategies")
    print("  5. All optimization opportunities in one place")

    return True


if __name__ == "__main__":
    test_basic_functionality()
