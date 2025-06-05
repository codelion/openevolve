"""
MLX Custom Metal Kernel Evolution for Block-Diagonal Attention

This module evolves a custom Metal kernel for efficient block-diagonal attention
on packed sequences. The kernel should outperform mx.fast.scaled_dot_product_attention
by skipping computation on masked regions entirely.

Evolution Target: The Metal C++ kernel source code that computes block-diagonal attention.
"""

import math
from typing import Optional

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX not available - this example requires MLX")
    MLX_AVAILABLE = False
    raise ImportError("MLX is required for this example")

import numpy as np


def is_true_block_diagonal_mask(mask):
    """
    Detect if a mask represents a TRUE block-diagonal pattern.
    
    This function is very restrictive and only returns True for masks that are
    clearly block-diagonal (contiguous square blocks along the diagonal).
    Random sparse masks will return False.
    """
    if mask is None or isinstance(mask, str):
        return False
    
    if not hasattr(mask, 'dtype') or mask.dtype != mx.bool_:
        return False
    
    if mask.ndim < 2:
        return False
    
    # Get 2D mask (take first batch/head if needed)
    mask_2d = mask
    while mask_2d.ndim > 2:
        mask_2d = mask_2d[0]
    
    L = mask_2d.shape[-1]
    if L < 32:  # Too small to be meaningful block-diagonal
        return False
    
    # Convert to numpy for easier analysis
    mask_np = np.array(mask_2d)
    
    # Check if mask has clear block structure
    # Look for at least 2 distinct diagonal blocks
    blocks_found = []
    current_pos = 0
    
    while current_pos < L:
        # Find start of next block (where diagonal is True)
        while current_pos < L and not mask_np[current_pos, current_pos]:
            current_pos += 1
        
        if current_pos >= L:
            break
            
        # Find end of this block
        block_start = current_pos
        block_end = current_pos
        
        # Expand block as long as diagonal remains True
        while block_end < L and mask_np[block_end, block_end]:
            block_end += 1
        
        block_size = block_end - block_start
        
        # Check if this is a valid square block (at least 16x16)
        if block_size >= 16:
            # Verify it's actually a square block (all True within the square)
            block_region = mask_np[block_start:block_end, block_start:block_end]
            if np.mean(block_region) > 0.95:  # 95% of block should be True
                blocks_found.append((block_start, block_size))
        
        current_pos = block_end
    
    # Must have at least 2 blocks to be considered block-diagonal
    if len(blocks_found) < 2:
        return False
    
    # Check that blocks don't overlap and are well-separated
    total_block_elements = sum(size * size for _, size in blocks_found)
    total_elements = L * L
    block_coverage = total_block_elements / total_elements
    
    # Should have reasonable sparsity (30-90% masked) and clear block structure
    sparsity = 1.0 - np.mean(mask_np)
    
    return (0.3 <= sparsity <= 0.9 and 
            0.05 <= block_coverage <= 0.7 and
            len(blocks_found) >= 2)


def spda_fallback(q, k, v, scale, mask):
    """Fall back to MLX's optimized scaled_dot_product_attention."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def evolved_scaled_dot_product_attention(q, k, v, scale=1.0, mask=None):
    """
    Custom Metal kernel for block-diagonal attention on packed sequences.
    
    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D] 
        v: Value tensor [B, H, L, D]
        scale: Scaling factor (typically 1/sqrt(head_dim))
        mask: Attention mask (supports None, "causal", or boolean masks)
        
    Returns:
        Attention output [B, H, L, D]
    """
    
    # Only use custom kernel for TRUE block-diagonal patterns
    if not is_true_block_diagonal_mask(mask):
        # Fall back to MLX's optimized SPDA for all other cases
        return spda_fallback(q, k, v, scale, mask)
    
    B, H, L, D = q.shape
    
    # EVOLVE-BLOCK-START
    # Custom Metal kernel source code for block-diagonal attention
    kernel_source = """
    uint elem = thread_position_in_grid.x;
    uint batch_idx = thread_position_in_grid.z;
    uint head_idx = thread_position_in_grid.y;
    uint query_pos = elem;
    
    if (batch_idx >= BATCH_SIZE || head_idx >= NUM_HEADS || query_pos >= SEQ_LEN) return;
    
    // Get scale value (dereference the buffer)
    T scale_val = T(scale[0]);
    
    // Calculate base indices
    uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + head_idx * (SEQ_LEN * HEAD_DIM) + query_pos * HEAD_DIM;
    uint mask_base = batch_idx * (NUM_HEADS * SEQ_LEN * SEQ_LEN) + head_idx * (SEQ_LEN * SEQ_LEN) + query_pos * SEQ_LEN;
    uint out_base = q_base;
    
    // Compute attention scores and find max
    T max_score = T(-INFINITY);
    for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
        if (!mask[mask_base + key_pos]) continue;
        
        uint k_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + head_idx * (SEQ_LEN * HEAD_DIM) + key_pos * HEAD_DIM;
        
        T score = T(0.0);
        for (uint d = 0; d < HEAD_DIM; d++) {
            score += queries[q_base + d] * keys[k_base + d];
        }
        score *= scale_val;
        max_score = max(max_score, score);
    }
    
    // Compute softmax denominator
    T sum_exp = T(0.0);
    for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
        if (!mask[mask_base + key_pos]) continue;
        
        uint k_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + head_idx * (SEQ_LEN * HEAD_DIM) + key_pos * HEAD_DIM;
        
        T score = T(0.0);
        for (uint d = 0; d < HEAD_DIM; d++) {
            score += queries[q_base + d] * keys[k_base + d];
        }
        score *= scale_val;
        sum_exp += exp(score - max_score);
    }
    
    // Compute output as weighted sum of values
    for (uint d = 0; d < HEAD_DIM; d++) {
        output[out_base + d] = T(0.0);
    }
    
    if (sum_exp > T(0.0)) {
        for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
            if (!mask[mask_base + key_pos]) continue;
            
            uint k_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + head_idx * (SEQ_LEN * HEAD_DIM) + key_pos * HEAD_DIM;
            uint v_base = k_base;
            
            T score = T(0.0);
            for (uint d = 0; d < HEAD_DIM; d++) {
                score += queries[q_base + d] * keys[k_base + d];
            }
            score *= scale_val;
            
            T attn_weight = exp(score - max_score) / sum_exp;
            
            for (uint d = 0; d < HEAD_DIM; d++) {
                output[out_base + d] += attn_weight * values[v_base + d];
            }
        }
    }
    """
    # EVOLVE-BLOCK-END
    
    try:
        # Prepare inputs
        scale_tensor = mx.array([scale], dtype=q.dtype)  # Match input dtype
        
        # Create Metal kernel
        kernel = mx.fast.metal_kernel(
            name="block_diagonal_attention",
            input_names=["queries", "keys", "values", "mask", "scale"],
            output_names=["output"],
            source=kernel_source
        )
        
        # Execute kernel with proper API
        outputs = kernel(
            inputs=[q, k, v, mask, scale_tensor],
            output_shapes=[(B, H, L, D)],     # Output shape
            output_dtypes=[q.dtype],          # Output dtype
            grid=(L, H, B),                   # Grid dimensions: (SEQ_LEN, NUM_HEADS, BATCH_SIZE)
            threadgroup=(32, 1, 1),           # Threadgroup size
            template=[                        # Template parameters as proper types
                ("T", q.dtype),               # Use mx.Dtype, not string
                ("BATCH_SIZE", B),            # int
                ("NUM_HEADS", H),             # int
                ("SEQ_LEN", L),               # int
                ("HEAD_DIM", D)               # int
            ]
        )
        
        return outputs[0]  # Return first (and only) output
        
    except Exception as e:
        # If custom kernel fails, fall back to optimized SPDA
        print(f"⚠️ Custom kernel failed: {e}, falling back to SPDA")
        return spda_fallback(q, k, v, scale, mask)


def create_benchmark_attention_function():
    """Create the attention function for benchmarking."""
    return evolved_scaled_dot_product_attention


# Test function
def test_basic_functionality():
    """Test basic Metal kernel functionality"""
    print("Testing Custom Metal Kernel for Block-Diagonal Attention...")
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return False
    
    try:
        # Test 1: Regular attention (should use SPDA)
        print("\n=== Test 1: Regular Attention (No Mask) ===")
        B, H, L, D = 1, 4, 128, 64
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D)) 
        v = mx.random.normal((B, H, L, D))
        scale = 1.0 / math.sqrt(D)
        
        output = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
        print(f"✅ Regular attention output shape: {output.shape} (uses SPDA)")
        
        # Test 2: Causal attention (should use SPDA)
        print("\n=== Test 2: Causal Attention ===")
        output = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        print(f"✅ Causal attention output shape: {output.shape} (uses SPDA)")
        
        # Test 3: Random sparse boolean mask (should use SPDA)
        print("\n=== Test 3: Random Sparse Boolean Mask ===")
        # Create random sparse mask using proper MLX API
        random_vals = mx.random.uniform(shape=[B, H, L, L])
        random_mask = random_vals > 0.5  # Random 50% sparse
        is_bd = is_true_block_diagonal_mask(random_mask)
        print(f"Random mask detected as block-diagonal: {is_bd}")
        output = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask=random_mask)
        print(f"✅ Random sparse mask output shape: {output.shape} (should use SPDA)")
        
        # Test 4: TRUE Block-diagonal attention (should use custom kernel)
        print("\n=== Test 4: TRUE Block-Diagonal Attention ===")
        B, H, L, D = 1, 4, 512, 64  # Larger size for clear blocks
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D)) 
        v = mx.random.normal((B, H, L, D))
        
        # Create TRUE block-diagonal mask (4 blocks of 128 each)
        mask = mx.zeros((B, H, L, L), dtype=mx.bool_)
        mask_np = np.zeros((B, H, L, L), dtype=bool)
        for i in range(4):
            start = i * 128
            end = (i + 1) * 128
            mask_np[:, :, start:end, start:end] = True  # 4 clear blocks
        mask = mx.array(mask_np)
        
        is_bd = is_true_block_diagonal_mask(mask)
        sparsity = 1.0 - float(mx.mean(mask.astype(mx.float32)))
        print(f"TRUE block-diagonal mask:")
        print(f"  Detected as block-diagonal: {is_bd}")
        print(f"  Sparsity: {sparsity:.1%}")
        
        if is_bd:
            print("✅ Should use custom kernel")
        else:
            print("⚠️ Will use SPDA (detection too restrictive)")
        
        output = evolved_scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        
        # Check output validity
        has_nan = bool(mx.any(mx.isnan(output)))
        has_inf = bool(mx.any(mx.isinf(output)))
        
        if output.shape == q.shape and not has_nan and not has_inf:
            print(f"✅ Block-diagonal attention test passed!")
            print(f"   Output shape: {output.shape} ({output.dtype})")
            print(f"   Has NaN: {has_nan}, Has Inf: {has_inf}")
            
            # Verify correctness against SPDA
            spda_output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
            diff = mx.max(mx.abs(output - spda_output))
            print(f"   Max diff vs SPDA: {float(diff):.2e}")
            
            if float(diff) < 1e-2:
                print("✅ Custom kernel output matches SPDA (correct)")
            else:
                print("❌ Custom kernel output differs from SPDA (incorrect)")
                return False
            
            return True
        else:
            print(f"❌ Block-diagonal test failed: shape={output.shape}, NaN={has_nan}, Inf={has_inf}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_basic_functionality()
