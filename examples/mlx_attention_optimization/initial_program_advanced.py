"""
MLX Attention Optimization Example for OpenEvolve - Advanced Version

This module contains an evolvable attention implementation with expanded capabilities
for discovering algorithmic innovations rather than just micro-optimizations.

The goal is to discover fundamentally better attention algorithms that can outperform
mx.fast.scaled_dot_product_attention through novel approaches.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class OptimizedAttention(nn.Module):
    """
    Advanced optimized attention module that allows for algorithmic innovation.
    This version provides more freedom for discovering sparse patterns, 
    approximations, and novel attention mechanisms.
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
        Advanced attention computation with freedom for algorithmic innovation.
        
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
        ALGORITHMIC INNOVATION ZONE - Discover Better Attention Mechanisms
        
        MISSION: Beat mx.fast.scaled_dot_product_attention with novel algorithms
        
        EXPANDED CONSTRAINTS - You can now use:
        ✅ BASIC OPERATIONS:
        - mx.matmul, mx.softmax, mx.transpose, mx.expand_dims, mx.reshape
        - mx.repeat, mx.concatenate, mx.split, mx.where, mx.maximum, mx.minimum
        - Basic arithmetic: +, -, *, /, mx.sqrt, mx.exp, mx.log
        - mx.zeros, mx.ones, mx.arange, mx.triu, mx.tril
        
        ✅ ADVANCED OPERATIONS (NEW):
        - mx.topk, mx.argsort, mx.gather, mx.scatter  # For sparse attention
        - mx.cumsum, mx.cumprod                       # For progressive computations
        - mx.roll, mx.flip                            # For shifted patterns
        - Indexing operations: queries[:, :, ::2, :]  # For strided patterns
        - mx.pad                                      # For boundary handling
        
        ✅ ALGORITHMIC PATTERNS TO EXPLORE:

        🔥 SPARSE ATTENTION (High Impact):
        ```python
        # Local attention windows
        window_size = min(256, L)
        # Block-sparse attention  
        block_size = 64
        # Top-k attention
        k = min(128, L_kv)
        ```
        
        🧠 LINEAR APPROXIMATIONS (Revolutionary):
        ```python
        # Kernel methods for O(n) attention
        # Low-rank approximations
        # Hierarchical attention
        ```
        
        ⚡ APPLE SILICON OPTIMIZATIONS:
        ```python
        # Chunked processing for unified memory
        chunk_size = 128
        # Cache-friendly access patterns
        # Memory-efficient intermediate tensors
        ```
        
        🎯 MULTI-SCALE PATTERNS:
        ```python
        # Different attention patterns for different heads
        # Combine local + global attention
        # Progressive refinement
        ```
        
        STILL FORBIDDEN:
        ❌ mx.fast.* functions (that's cheating!)
        ❌ mx.nn.* beyond basic imports
        ❌ External libraries
        
        INNOVATION EXAMPLES TO INSPIRE YOU:
        
        Example 1 - Sparse Local Attention:
        ```python
        window_size = 256
        # Only compute attention within sliding windows
        for i in range(0, L, window_size):
            local_queries = queries[:, :, i:i+window_size, :]
            local_keys = keys[:, :, max(0,i-window_size//2):i+window_size, :]
            # Compute local attention...
        ```
        
        Example 2 - Top-K Sparse Attention:
        ```python
        # Pre-compute which keys are most relevant for each query
        relevance_scores = mx.sum(queries * keys.mean(axis=2, keepdims=True), axis=-1)
        top_k_indices = mx.topk(relevance_scores, k=128)[1]
        # Only compute attention for top-k most relevant positions
        ```
        
        Example 3 - Block-Sparse Pattern:
        ```python
        block_size = 64
        num_blocks = L // block_size
        # Process attention in blocks with specific connectivity patterns
        ```
        
        Your mission: Implement something fundamentally different that achieves:
        - 20%+ speedup on sequences > 1024 tokens
        - Better memory efficiency
        - Novel algorithmic approach
        
        The current reference uses O(L²) computation. Can you do better?
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
        
        # STARTER IMPLEMENTATION - Replace this with your innovation!
        # This is the baseline O(L²) attention that you need to beat
        
        # Standard scaled dot-product attention
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply external mask if provided
        if mask is not None:
            if mask.ndim == 2:  # [L, L_kv]
                mask = mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0)
            elif mask.ndim == 3:  # [B, L, L_kv]
                mask = mx.expand_dims(mask, axis=1)
            scores = scores + mask
        
        # Apply softmax and compute output
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, values)
        
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
    Create a complete attention module for testing with expanded capabilities.
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
            
            # Our evolvable attention with expanded capabilities
            self.optimized_attention = OptimizedAttention(
                hidden_size, num_heads, num_kv_heads, head_dim, self.scale
            )
            
        def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
            """
            Forward pass through the complete attention module.
            """
            B, L, D = x.shape
            
            # Project to Q, K, V
            queries = self.q_proj(x)
            keys = self.k_proj(x)
            values = self.v_proj(x)
            
            # Reshape and transpose to separate heads
            queries = self.q_norm(
                queries.reshape(B, L, self.num_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            
            keys = self.k_norm(
                keys.reshape(B, L, self.num_kv_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            
            values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
            
            # Apply our optimized attention
            output = self.optimized_attention(queries, keys, values, mask=mask)
            
            # Reshape back and apply output projection
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)
    
    return TestAttentionModule()


def run_attention_test():
    """Enhanced test to verify the attention module works with longer sequences"""
    print("Testing advanced attention implementation...")
    
    # Test multiple sequence lengths to verify scalability
    test_cases = [
        (2, 128, 512),   # Small: batch=2, seq=128, hidden=512
        (1, 512, 768),   # Medium: batch=1, seq=512, hidden=768  
        (1, 1024, 512),  # Large: batch=1, seq=1024, hidden=512
    ]
    
    for batch_size, seq_len, hidden_size in test_cases:
        print(f"\nTesting: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
        
        # Create test module
        attn_module = create_test_attention_module(
            hidden_size=hidden_size,
            num_heads=8,
            num_kv_heads=8,
            head_dim=hidden_size // 8
        )
        
        # Test inputs
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        
        # Create causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
        mask = mx.expand_dims(mask, axis=0)
        
        # Forward pass with timing
        import time
        start_time = time.time()
        output = attn_module(x, mask=mask)
        mx.eval(output)  # Ensure computation completes
        end_time = time.time()
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Time: {(end_time - start_time)*1000:.2f}ms")
        print(f"  Output mean: {mx.mean(output).item():.6f}")
        print(f"  Output std: {mx.std(output).item():.6f}")
        
        # Check for NaN/Inf
        has_nan = bool(mx.any(mx.isnan(output)))
        has_inf = bool(mx.any(mx.isinf(output)))
        if has_nan or has_inf:
            print(f"  ❌ Warning: NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  ✅ Numerically stable")
    
    return True


if __name__ == "__main__":
    run_attention_test()
