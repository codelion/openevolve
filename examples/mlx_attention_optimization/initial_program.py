"""
MLX Attention Optimization Example for OpenEvolve

This module contains an evolvable attention implementation based on Qwen3's attention mechanism.
The goal is to optimize the core attention computation while maintaining numerical accuracy.

The evolvable part focuses on the scaled dot-product attention computation, while keeping
projections, RoPE, and normalization fixed to ensure compatibility.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class OptimizedAttention(nn.Module):
    """
    Optimized attention module that maintains compatibility with Qwen3's attention
    while allowing evolution of the core attention computation.
    """
    
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
        """
        Optimized attention computation.
        
        Args:
            queries: Query tensor [B, num_heads, L, head_dim]
            keys: Key tensor [B, num_kv_heads, L_kv, head_dim] 
            values: Value tensor [B, num_kv_heads, L_kv, head_dim]
            mask: Attention mask [B, L, L_kv] or None
            cache: KV cache or None
            
        Returns:
            Attention output [B, num_heads, L, head_dim]
        """
        
        # EVOLVE-BLOCK-START
        """
        Core attention computation - this is what gets evolved.
        
        GOAL: Beat mx.fast.scaled_dot_product_attention using novel algorithmic approaches.
        
        CONSTRAINTS - You can ONLY use these basic MLX operations:
        - mx.matmul, mx.softmax, mx.transpose, mx.expand_dims, mx.reshape
        - mx.repeat, mx.concatenate, mx.split, mx.where, mx.maximum, mx.minimum
        - Basic arithmetic: +, -, *, /, mx.sqrt, mx.exp, mx.log
        - mx.zeros, mx.ones, mx.arange, mx.triu, mx.tril
        
        FORBIDDEN - Do NOT use these (they're cheating):
        - mx.fast.* functions (including mx.fast.scaled_dot_product_attention)
        - mx.nn.* functions beyond what's imported
        - Any other high-level optimized functions
        
        INNOVATION TARGETS - Discover novel approaches like:
        - Sparse attention patterns optimized for Apple Silicon
        - Chunked attention with custom memory tiling
        - Local attention windows with efficient neighbor selection
        - Custom attention patterns that exploit unified memory
        - Novel softmax approximations or attention alternatives
        - Memory-efficient attention for long sequences
        
        The reference implementation uses mx.fast.scaled_dot_product_attention
        which is already highly optimized. Your job is to discover something even better!
        """
        
        B, num_heads, L, head_dim = queries.shape
        _, num_kv_heads, L_kv, _ = keys.shape
        
        # Handle grouped query attention (GQA) by repeating KV heads if needed
        if num_kv_heads != num_heads:
            if num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"Number of query heads ({num_heads}) must be divisible by "
                    f"number of KV heads ({num_kv_heads}) for GQA."
                )
            # Repeat keys and values to match query heads
            rep_factor = num_heads // num_kv_heads
            keys = mx.repeat(keys, rep_factor, axis=1)
            values = mx.repeat(values, rep_factor, axis=1)
        
        # Standard scaled dot-product attention using ONLY basic operations
        # Compute attention scores: Q @ K^T
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2))  # [B, num_heads, L, L_kv]
        
        # Scale by sqrt(head_dim)
        scores = scores * self.scale
        
        # Apply attention mask if provided
        if mask is not None:
            # Ensure mask is broadcastable to scores shape
            if mask.ndim == 2:  # [L, L_kv]
                mask = mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0)  # [1, 1, L, L_kv]
            elif mask.ndim == 3:  # [B, L, L_kv]
                mask = mx.expand_dims(mask, axis=1)  # [B, 1, L, L_kv]
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply attention weights to values: weights @ V
        output = mx.matmul(attn_weights, values)  # [B, num_heads, L, head_dim]
        
        return output
        # EVOLVE-BLOCK-END


def create_test_attention_module(
    hidden_size: int = 512,
    num_heads: int = 8, 
    num_kv_heads: int = 8,
    head_dim: int = 64,
    eps: float = 1e-6
):
    """
    Create a complete attention module for testing that mimics Qwen3's structure.
    This includes all the fixed components (projections, norms, rope) plus our evolvable attention.
    """
    
    class TestAttentionModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5
            
            # Fixed components (not evolved)
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            
            self.q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=eps)
            
            # Our evolvable attention
            self.optimized_attention = OptimizedAttention(
                hidden_size, num_heads, num_kv_heads, head_dim, self.scale
            )
            
        def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
            """
            Forward pass through the complete attention module.
            
            Args:
                x: Input tensor [B, L, hidden_size]
                mask: Attention mask [B, L, L] or None
                
            Returns:
                Output tensor [B, L, hidden_size]
            """
            B, L, D = x.shape
            
            # Project to Q, K, V
            queries = self.q_proj(x)  # [B, L, num_heads * head_dim]
            keys = self.k_proj(x)     # [B, L, num_kv_heads * head_dim] 
            values = self.v_proj(x)   # [B, L, num_kv_heads * head_dim]
            
            # Reshape and transpose to separate heads
            queries = self.q_norm(
                queries.reshape(B, L, self.num_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
            
            keys = self.k_norm(
                keys.reshape(B, L, self.num_kv_heads, self.head_dim)  
            ).transpose(0, 2, 1, 3)  # [B, num_kv_heads, L, head_dim]
            
            values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )  # [B, num_kv_heads, L, head_dim]
            
            # Apply our optimized attention
            output = self.optimized_attention(queries, keys, values, mask=mask)
            
            # Reshape back to [B, L, num_heads * head_dim]
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            
            # Final projection
            return self.o_proj(output)
    
    return TestAttentionModule()


def run_attention_test():
    """Simple test to verify the attention module works"""
    print("Testing initial attention implementation...")
    
    # Create test module
    attn_module = create_test_attention_module()
    
    # Test inputs
    batch_size, seq_len, hidden_size = 2, 128, 512
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Create a simple causal mask
    mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
    mask = mx.expand_dims(mask, axis=0)  # Add batch dimension
    
    # Forward pass
    output = attn_module(x, mask=mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {mx.mean(output).item():.6f}")
    print(f"Output std: {mx.std(output).item():.6f}")
    print("✓ Basic attention test passed!")
    
    return output


if __name__ == "__main__":
    run_attention_test()
