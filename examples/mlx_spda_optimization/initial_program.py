"""
MLX Block-Diagonal Attention Kernel Evolution for Packed Sequences

This module evolves a custom Metal kernel for efficient block-diagonal attention,
specifically designed for packed sequences where attention should only occur 
within sequence boundaries, not across different packed sequences.

Use case: Training BERTs/GPTs with packed sequences to eliminate padding waste.
Goal: Evolve a Metal kernel that efficiently computes attention while respecting
sequence boundaries, avoiding computation on masked regions.
"""

import math
from typing import Optional, Union

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX not available - this example requires MLX")
    MLX_AVAILABLE = False
    raise ImportError("MLX is required for this example")

import numpy as np


def evolved_scaled_dot_product_attention(q, k, v, scale=1.0, mask=None):
    """
    Evolved block-diagonal attention with custom Metal kernel for packed sequences.
    
    This function evolves a Metal kernel that efficiently computes attention for
    packed sequences, where attention should only occur within sequence boundaries.
    
    Args:
        q: Query tensor [B, num_heads, L, head_dim]
        k: Key tensor [B, num_kv_heads, L_kv, head_dim] 
        v: Value tensor [B, num_kv_heads, L_kv, head_dim]
        scale: Scaling factor (typically 1/sqrt(head_dim))
        mask: Attention mask (block-diagonal for packed sequences)
        
    Returns:
        Attention output with same shape as queries
    """
    
    # EVOLVE-BLOCK-START
    """
    EVOLUTION TARGET: Custom Metal Kernel for Block-Diagonal Attention
    
    🎯 MISSION: Evolve an efficient Metal kernel for packed sequence attention
    
    PROBLEM CONTEXT:
    - Packed sequences: Multiple sequences concatenated to avoid padding waste
    - Block-diagonal attention: Keys/queries only attend within same sequence
    - Current solutions: Naive masking wastes computation on -inf regions
    - Goal: Direct Metal kernel that skips masked computations entirely
    
    EVOLUTION OPPORTUNITIES:
    
    1. EFFICIENT BLOCK DETECTION:
       - Automatically detect sequence boundaries from attention patterns
       - Use sequence length information to determine block structure
       - Optimize for common packing patterns (uniform vs variable lengths)
    
    2. CUSTOM METAL KERNEL OPTIMIZATION:
       - Thread-level optimization for block-diagonal patterns
       - Skip computation for cross-sequence attention entirely
       - Vectorized operations within sequence blocks
       - Optimized memory access patterns for Apple Silicon
    
    3. ADAPTIVE BLOCK PROCESSING:
       - Handle variable sequence lengths efficiently
       - Optimize for different head dimensions and sequence counts
       - Balance between generality and performance
    
    4. MEMORY EFFICIENCY:
       - Minimize memory allocation for intermediate results
       - Use shared memory for sequence blocks
       - Optimize for unified memory architecture
    
    CURRENT IMPLEMENTATION: Basic block detection with custom kernel evolution
    """
    
    # Extract basic dimensions
    B, n_q_heads, L, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kL = k.shape[2]
    
    # Handle Grouped Query Attention (GQA)
    n_repeats = n_q_heads // n_kv_heads
    if n_repeats > 1:
        k = mx.repeat(k, n_repeats, axis=1)
        v = mx.repeat(v, n_repeats, axis=1)
    
    # Try to detect if this is a packed sequence scenario
    is_packed_sequences = detect_packed_sequences(mask, L, kL)
    
    if is_packed_sequences:
        # Use evolved custom kernel for packed sequences
        try:
            return custom_block_diagonal_attention(q, k, v, scale, mask)
        except Exception as e:
            print(f"⚠️ Custom kernel failed: {e}, falling back to reference")
            return reference_attention_fallback(q, k, v, scale, mask)
    else:
        # For regular attention, try MLX fast implementation first
        try:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        except Exception:
            return reference_attention_fallback(q, k, v, scale, mask)
    
    
def detect_packed_sequences(mask, q_len, k_len):
    """
    Detect if this is likely a packed sequence scenario.
    
    EVOLUTION OPPORTUNITY: Improve detection logic
    - Analyze mask patterns for block-diagonal structure
    - Use sequence length patterns
    - Detect common packing strategies
    """
    if mask is None:
        return False
    
    # Simple heuristic: if mask exists and sequences are reasonably long,
    # assume it might be packed sequences
    if isinstance(mask, str):
        return False  # String masks like "causal" are not packed sequences
    
    # If mask is provided and sequences are longer than typical single sequences,
    # assume packed sequences
    return q_len > 256 or k_len > 256


def custom_block_diagonal_attention(q, k, v, scale, mask):
    """
    Custom Metal kernel implementation for block-diagonal attention.
    
    MAIN EVOLUTION TARGET: This is where the Metal kernel magic happens!
    """
    
    # Analyze mask to determine block structure
    block_info = analyze_mask_structure(mask)
    
    # Try to create and use custom Metal kernel
    kernel_result = try_custom_metal_kernel(q, k, v, scale, block_info)
    
    if kernel_result is not None:
        return kernel_result
    
    # Fallback: Optimized CPU implementation for block-diagonal
    return optimized_block_diagonal_cpu(q, k, v, scale, mask, block_info)


def analyze_mask_structure(mask):
    """
    Analyze the attention mask to extract block-diagonal structure.
    
    EVOLUTION OPPORTUNITY: Advanced mask analysis
    - Detect block boundaries automatically
    - Handle irregular block patterns
    - Optimize for common packing strategies
    """
    if mask is None:
        return {"type": "none", "blocks": []}
    
    # Convert mask to boolean if needed
    if hasattr(mask, 'dtype'):
        if mask.dtype != mx.bool_:
            bool_mask = mask > -1e4  # Convert additive mask to boolean
        else:
            bool_mask = mask
    else:
        bool_mask = mask
    
    # Simple block detection: look for diagonal patterns
    # This is a placeholder - evolution should improve this significantly
    mask_shape = bool_mask.shape
    if len(mask_shape) >= 2:
        seq_len = mask_shape[-1]
        
        # Detect uniform blocks (simplest case)
        # EVOLUTION TODO: Handle variable-length blocks
        estimated_block_size = detect_uniform_block_size(bool_mask)
        
        if estimated_block_size > 0:
            num_blocks = (seq_len + estimated_block_size - 1) // estimated_block_size
            return {
                "type": "uniform_blocks",
                "block_size": estimated_block_size,
                "num_blocks": num_blocks,
                "sequence_length": seq_len
            }
    
    return {"type": "unknown", "blocks": []}


def detect_uniform_block_size(bool_mask):
    """
    Detect uniform block size from mask pattern.
    
    EVOLUTION OPPORTUNITY: Sophisticated block detection
    - Handle non-uniform blocks
    - Detect nested block patterns
    - Use machine learning for pattern recognition
    """
    # Simple heuristic: assume blocks of size 128, 256, 512, etc.
    # Evolution should replace this with actual pattern detection
    
    mask_2d = bool_mask[0, 0] if bool_mask.ndim > 2 else bool_mask
    seq_len = mask_2d.shape[-1]
    
    # Test common block sizes
    for block_size in [128, 256, 512, 1024]:
        if block_size <= seq_len and seq_len % block_size == 0:
            # Check if this creates a reasonable block-diagonal pattern
            if check_block_diagonal_pattern(mask_2d, block_size):
                return block_size
    
    return 0  # No clear block pattern detected


def check_block_diagonal_pattern(mask_2d, block_size):
    """
    Check if mask follows block-diagonal pattern for given block size.
    
    EVOLUTION OPPORTUNITY: More sophisticated pattern matching
    """
    try:
        seq_len = mask_2d.shape[-1]
        num_blocks = seq_len // block_size
        
        # Check a few blocks to see if they follow diagonal pattern
        correct_blocks = 0
        for i in range(min(3, num_blocks)):  # Check first few blocks
            start = i * block_size
            end = min(start + block_size, seq_len)
            
            block = mask_2d[start:end, start:end]
            if float(mx.mean(block.astype(mx.float32))) > 0.8:  # Mostly True
                correct_blocks += 1
        
        return correct_blocks >= min(2, num_blocks)
    except Exception:
        return False


def try_custom_metal_kernel(q, k, v, scale, block_info):
    """
    Attempt to create and execute custom Metal kernel for block-diagonal attention.
    
    MAIN EVOLUTION TARGET: This is the core of what should be evolved!
    """
    try:
        if block_info["type"] != "uniform_blocks":
            return None  # Only handle uniform blocks for now
        
        # For now, disable custom Metal kernel due to API complexity
        # Evolution should focus on CPU optimizations first
        return None
        
        # TODO: Implement proper Metal kernel when API is stabilized
        # The Metal kernel API requires specific grid/threadgroup configurations
        # and proper template parameter handling that needs careful tuning
        
    except Exception as e:
        # Kernel creation or execution failed
        print(f"Metal kernel failed: {e}")
        return None


def create_block_diagonal_kernel_source(block_info):
    """
    Generate Metal kernel source code for block-diagonal attention.
    
    EVOLUTION TARGET: This kernel should be evolved for maximum performance!
    """
    
    kernel_source = f"""
    // Block-Diagonal Attention Metal Kernel
    // Optimized for packed sequences with block-diagonal attention pattern
    
    template<typename T>
    [[kernel]] void block_diagonal_attention(
        const device T* queries [[buffer(0)]],
        const device T* keys [[buffer(1)]],
        const device T* values [[buffer(2)]],
        const device float* scale_factor [[buffer(3)]],
        device T* output [[buffer(4)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]],
        uint3 threads_per_group [[threads_per_group]],
        uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
    ) {{
        
        // EVOLUTION OPPORTUNITIES:
        // 1. Optimize thread allocation per block
        // 2. Use shared/threadgroup memory for efficiency
        // 3. Vectorize operations (float4, half4)
        // 4. Implement tiled computation
        // 5. Add sparse attention patterns within blocks
        
        const uint batch_idx = threadgroup_position_in_grid.z;
        const uint head_idx = threadgroup_position_in_grid.y;
        const uint block_idx = threadgroup_position_in_grid.x;
        
        const uint thread_idx = thread_position_in_grid.x;
        const uint seq_len = {block_info["sequence_length"]};
        const uint block_size = {block_info["block_size"]};
        const uint head_dim = HEAD_DIM;
        
        // Calculate block boundaries
        const uint block_start = block_idx * block_size;
        const uint block_end = min(block_start + block_size, seq_len);
        const uint actual_block_size = block_end - block_start;
        
        // Skip if thread is outside block
        if (thread_idx >= actual_block_size) return;
        
        const float scale = scale_factor[0];
        
        // EVOLUTION TARGET: Optimize this computation
        // Current: Simple implementation, should be evolved for performance
        
        for (uint q_pos = thread_idx; q_pos < actual_block_size; q_pos += threads_per_group.x) {{
            uint global_q_pos = block_start + q_pos;
            
            // Compute attention scores for this query position
            float attention_scores[{block_info["block_size"]}];
            float max_score = -INFINITY;
            
            // Score computation: only within block (block-diagonal)
            for (uint k_pos = 0; k_pos < actual_block_size; k_pos++) {{
                uint global_k_pos = block_start + k_pos;
                
                float score = 0.0f;
                for (uint d = 0; d < head_dim; d++) {{
                    uint q_idx = batch_idx * (seq_len * head_dim) + head_idx * (seq_len * head_dim) + global_q_pos * head_dim + d;
                    uint k_idx = batch_idx * (seq_len * head_dim) + head_idx * (seq_len * head_dim) + global_k_pos * head_dim + d;
                    score += float(queries[q_idx]) * float(keys[k_idx]);
                }}
                score *= scale;
                
                attention_scores[k_pos] = score;
                max_score = max(max_score, score);
            }}
            
            // Softmax computation
            float sum_exp = 0.0f;
            for (uint k_pos = 0; k_pos < actual_block_size; k_pos++) {{
                attention_scores[k_pos] = exp(attention_scores[k_pos] - max_score);
                sum_exp += attention_scores[k_pos];
            }}
            
            // Normalize
            for (uint k_pos = 0; k_pos < actual_block_size; k_pos++) {{
                attention_scores[k_pos] /= sum_exp;
            }}
            
            // Compute output: weighted sum of values
            for (uint d = 0; d < head_dim; d++) {{
                float output_val = 0.0f;
                for (uint k_pos = 0; k_pos < actual_block_size; k_pos++) {{
                    uint global_k_pos = block_start + k_pos;
                    uint v_idx = batch_idx * (seq_len * head_dim) + head_idx * (seq_len * head_dim) + global_k_pos * head_dim + d;
                    output_val += attention_scores[k_pos] * float(values[v_idx]);
                }}
                
                uint out_idx = batch_idx * (seq_len * head_dim) + head_idx * (seq_len * head_dim) + global_q_pos * head_dim + d;
                output[out_idx] = T(output_val);
            }}
        }}
    }}
    """
    
    return kernel_source


def optimized_block_diagonal_cpu(q, k, v, scale, mask, block_info):
    """
    Optimized CPU fallback for block-diagonal attention.
    
    EVOLUTION OPPORTUNITY: Optimize this fallback implementation
    """
    if block_info["type"] != "uniform_blocks":
        return reference_attention_fallback(q, k, v, scale, mask)
    
    # Use block-diagonal computation to avoid unnecessary work
    B, H, L, D = q.shape
    block_size = block_info["block_size"]
    num_blocks = block_info["num_blocks"]
    
    # Compute each block and collect outputs
    block_outputs = []
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, L)
        
        # Extract block
        q_block = q[:, :, start_idx:end_idx, :]
        k_block = k[:, :, start_idx:end_idx, :]
        v_block = v[:, :, start_idx:end_idx, :]
        
        # Compute attention within block
        scores = (q_block * scale) @ mx.swapaxes(k_block, -1, -2)
        attn_weights = mx.softmax(scores, axis=-1, precise=True)
        block_output = attn_weights @ v_block
        
        block_outputs.append(block_output)
    
    # Concatenate all block outputs
    output = mx.concatenate(block_outputs, axis=2)
    
    return output


def reference_attention_fallback(q, k, v, scale, mask):
    """
    Reference implementation fallback.
    """
    # Basic scaled dot-product attention
    scores = (q * scale) @ mx.swapaxes(k, -1, -2)
    
    # Apply mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            L = scores.shape[-1]
            causal_mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))
            scores = mx.where(causal_mask, scores, -mx.array(np.float32(np.inf)))
        elif hasattr(mask, 'dtype') and mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -mx.array(np.float32(np.inf)))
        else:
            scores = scores + mask
    
    # Softmax and output
    attn_weights = mx.softmax(scores, axis=-1, precise=True)
    return attn_weights @ v
    # EVOLVE-BLOCK-END


def create_benchmark_attention_function():
    """
    Create the attention function for benchmarking.
    """
    return evolved_scaled_dot_product_attention


# Test function for development
def test_basic_functionality():
    """Test basic functionality of the block-diagonal attention"""
    print("Testing Block-Diagonal Attention for Packed Sequences...")
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return False
    
    try:
        # Test 1: Regular attention (should work normally)
        print("\n=== Test 1: Regular Attention ===")
        B, H, L, D = 1, 8, 128, 64
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        scale = 1.0 / math.sqrt(D)
        
        output = evolved_scaled_dot_product_attention(q, k, v, scale=scale)
        print(f"✅ Regular attention output shape: {output.shape}")
        
        # Test 2: Block-diagonal attention with mask
        print("\n=== Test 2: Block-Diagonal Attention ===")
        B, H, L, D = 1, 8, 512, 64  # Longer sequence
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))
        
        # Create block-diagonal mask (2 sequences of 256 tokens each)
        mask = mx.zeros((B, H, L, L), dtype=mx.bool_)
        # MLX doesn't support .at[] syntax, use numpy to create mask and convert
        mask_np = np.zeros((B, H, L, L), dtype=bool)
        mask_np[:, :, 0:256, 0:256] = True  # First sequence block
        mask_np[:, :, 256:512, 256:512] = True  # Second sequence block
        mask = mx.array(mask_np)
        
        output = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        print(f"✅ Block-diagonal attention output shape: {output.shape}")
        
        # Verify no NaN/Inf
        has_nan = bool(mx.any(mx.isnan(output)))
        has_inf = bool(mx.any(mx.isinf(output)))
        
        if not has_nan and not has_inf:
            print(f"✅ Output is valid (no NaN/Inf)")
        else:
            print(f"❌ Output contains NaN={has_nan}, Inf={has_inf}")
            return False
        
        print("\n🎯 Block-Diagonal Attention System Ready!")
        print("🚀 Evolution target: Custom Metal kernel for packed sequences")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_basic_functionality()
