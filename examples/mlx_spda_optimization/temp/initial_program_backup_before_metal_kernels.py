"""
BACKUP: Original JIT-compiled version before converting to Metal kernels

This was the original implementation that used mx.compile() decorators
for JIT compilation. Saved here for reference before converting to
the custom Metal kernel approach.
"""

import math
from typing import Optional

import mlx.core as mx
import numpy as np


# JIT-compiled helper functions for maximum optimization
@mx.compile
def compute_attention_scores(q, k, scale):
    """Compute Q @ K^T with scaling - optimized for JIT compilation"""
    return (q * scale) @ mx.swapaxes(k, -1, -2)


@mx.compile
def apply_causal_mask(scores, L, kL):
    """Apply causal mask efficiently using MLX graph optimization"""
    q_offset = max(0, kL - L)
    q_indices = mx.arange(q_offset, q_offset + L)
    k_indices = mx.arange(kL)
    mask = q_indices[:, None] >= k_indices[None]
    return mx.where(mask, scores, -mx.array(np.float32(np.inf)))


@mx.compile
def apply_boolean_mask(scores, mask):
    """Apply boolean mask with JIT optimization"""
    return mx.where(mask, scores, -mx.array(np.float32(np.inf)))


@mx.compile
def softmax_attention(scores):
    """Optimized softmax with precise computation"""
    return mx.softmax(scores, axis=-1, precise=True)


@mx.compile
def attention_weighted_sum(attention_weights, v):
    """Compute attention-weighted sum of values"""
    return attention_weights @ v


# Main optimized attention function
def evolved_scaled_dot_product_attention(q, k, v, scale=1.0, mask=None):
    """Original JIT-optimized version (backup)"""

    # Extract dimensions for optimization decisions
    B, n_q_heads, L, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kL = k.shape[2]
    n_repeats = n_q_heads // n_kv_heads

    # Efficient GQA handling using memory views (not physical duplication)
    if n_repeats > 1:
        # Reshape queries for grouped attention
        q_reshaped = mx.reshape(q, [B, n_kv_heads, n_repeats, L, head_dim])
        # Expand KV for broadcasting
        k_expanded = mx.expand_dims(k, 2)  # [B, n_kv_heads, 1, kL, head_dim]
        v_expanded = mx.expand_dims(v, 2)  # [B, n_kv_heads, 1, kL, head_dim]
    else:
        q_reshaped = q
        k_expanded = k
        v_expanded = v

    # Compute attention scores using JIT-compiled function
    scores = compute_attention_scores(q_reshaped, k_expanded, scale)

    # Apply mask efficiently using appropriate JIT-compiled function
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            # Use optimized causal mask application
            scores = apply_causal_mask(scores, L, kL)
        elif hasattr(mask, "dtype") and mask.dtype == mx.bool_:
            # Handle grouped attention masking if needed
            if n_repeats > 1 and mask.ndim >= 3:
                if mask.shape[-3] == 1:
                    mask = mx.expand_dims(mask, -3)
                elif mask.shape[-3] == n_q_heads:
                    mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))
            # Apply boolean mask using JIT-compiled function
            scores = apply_boolean_mask(scores, mask)
        else:
            # Additive mask - simple addition
            scores = scores + mask

    # Apply softmax using JIT-compiled function
    attention_weights = softmax_attention(scores)

    # Compute attention-weighted sum using JIT-compiled function
    out = attention_weighted_sum(attention_weights, v_expanded)

    # Reshape output back to original query head count
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, head_dim])

    return out
