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
    
    # Check overall sparsity first (quick filter)
    sparsity = 1.0 - np.mean(mask_np)
    if not (0.2 <= sparsity <= 0.99):
        return False
    
    # NEW ALGORITHM: Find contiguous square blocks along the diagonal
    # Strategy: Scan the diagonal and identify where block boundaries occur
    # by looking at off-diagonal transitions
    
    blocks_found = []
    i = 0
    
    while i < L:
        # Skip any False positions on diagonal (shouldn't happen in block-diagonal)
        if not mask_np[i, i]:
            i += 1
            continue
            
        # Found start of a potential block
        block_start = i
        
        # Find the size of this block by checking the square region
        # We'll expand the block size until we hit a boundary
        max_possible_size = L - block_start
        block_size = 1
        
        # Expand block size while the square region remains dense
        for size in range(1, max_possible_size + 1):
            # Check if [block_start:block_start+size, block_start:block_start+size] is dense
            end_pos = block_start + size
            if end_pos > L:
                break
                
            block_region = mask_np[block_start:end_pos, block_start:end_pos]
            density = np.mean(block_region)
            
            if density > 0.95:  # Block is dense enough
                block_size = size
            else:
                break  # Block is no longer dense, so we found the boundary
        
        # Verify this is a valid block (at least 8x8)
        if block_size >= 8:
            blocks_found.append((block_start, block_size))
        
        # Move to the next potential block
        i = block_start + block_size
    
    # Must have at least 2 blocks to be considered block-diagonal
    if len(blocks_found) < 2:
        return False
    
    # Check that blocks don't overlap and cover reasonable portion
    total_block_elements = sum(size * size for _, size in blocks_found)
    total_elements = L * L
    block_coverage = total_block_elements / total_elements
    
    # Should have reasonable coverage (not too sparse, not too dense)
    if not (0.01 <= block_coverage <= 0.8):
        return False
    
    # Additional validation: check that blocks are actually separated
    # (i.e., there are off-diagonal zeros between blocks)
    for i in range(len(blocks_found) - 1):
        block1_start, block1_size = blocks_found[i]
        block2_start, block2_size = blocks_found[i + 1]
        
        block1_end = block1_start + block1_size
        
        # There should be a gap or the blocks should be adjacent
        if block1_end > block2_start:
            return False  # Overlapping blocks
        
        # Check that there are actually zeros between blocks (if not adjacent)
        if block1_end < block2_start:
            # Sample some off-diagonal positions between blocks
            mid_pos = (block1_end + block2_start) // 2
            if mid_pos < L and mask_np[block1_start, mid_pos]:
                return False  # Should be sparse between blocks
    
    return True


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
    // Thread and grid setup
    uint elem = thread_position_in_grid.x;
    uint batch_idx = thread_position_in_grid.z;
    uint head_idx = thread_position_in_grid.y;
    uint query_pos = elem;
    
    // Early bounds check
    if (batch_idx >= BATCH_SIZE || head_idx >= NUM_HEADS || query_pos >= SEQ_LEN) return;
    
    // OPTIMIZATION 1: Define vector types for SIMD operations
    using T4 = metal::vec<T, 4>;
    
    // OPTIMIZATION 2: Cache frequently used values
    const T scale_val = T(scale[0]);
    
    // OPTIMIZATION 3: Pre-compute base indices once
    const uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                        head_idx * (SEQ_LEN * HEAD_DIM) + 
                        query_pos * HEAD_DIM;
    const uint mask_base = batch_idx * (NUM_HEADS * SEQ_LEN * SEQ_LEN) + 
                           head_idx * (SEQ_LEN * SEQ_LEN) + 
                           query_pos * SEQ_LEN;
    const uint out_base = q_base;
    
    // OPTIMIZATION 4: Cache computed scores to eliminate redundant computation
    // Allocate local array for scores (avoids recomputing dot products 3 times)
    T cached_scores[SEQ_LEN];
    uint valid_keys[SEQ_LEN];  // Track which keys are valid (pass mask)
    uint num_valid_keys = 0;
    
    // SINGLE PASS: Compute all dot products once and cache results
    T max_score = T(-INFINITY);
    
    for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
        // Skip masked entries entirely
        if (!mask[mask_base + key_pos]) {
            continue;
        }
        
        // Pre-compute key base index
        const uint k_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                            head_idx * (SEQ_LEN * HEAD_DIM) + 
                            key_pos * HEAD_DIM;
        
        // OPTIMIZATION 5: Vectorized dot product (4x faster than scalar)
        T score = T(0.0);
        
        // Process HEAD_DIM in chunks of 4 using SIMD
        for (uint d = 0; d < HEAD_DIM; d += 4) {
            // Load 4 elements at once for queries and keys
            T4 q_vec = *((device T4*)(queries + q_base + d));
            T4 k_vec = *((device T4*)(keys + k_base + d));
            
            // Efficient dot product using Metal's built-in SIMD operations
            score += dot(q_vec, k_vec);
        }
        
        // Apply scaling
        score *= scale_val;
        
        // Cache the computed score and track valid key position
        cached_scores[num_valid_keys] = score;
        valid_keys[num_valid_keys] = key_pos;
        num_valid_keys++;
        
        // Update max score for numerical stability
        max_score = max(max_score, score);
    }
    
    // SECOND PASS: Compute softmax denominator using cached scores
    T sum_exp = T(0.0);
    for (uint i = 0; i < num_valid_keys; i++) {
        T exp_score = exp(cached_scores[i] - max_score);
        cached_scores[i] = exp_score;  // Overwrite score with exp(score - max_score)
        sum_exp += exp_score;
    }
    
    // OPTIMIZATION 6: Vectorized output initialization
    for (uint d = 0; d < HEAD_DIM; d += 4) {
        *((device T4*)(output + out_base + d)) = T4(0.0);
    }
    
    // THIRD PASS: Compute final output using cached exp scores
    if (sum_exp > T(0.0)) {
        for (uint i = 0; i < num_valid_keys; i++) {
            uint key_pos = valid_keys[i];
            T attn_weight = cached_scores[i] / sum_exp;  // Use cached exp(score - max_score)
            
            // Pre-compute value base index
            const uint v_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                                head_idx * (SEQ_LEN * HEAD_DIM) + 
                                key_pos * HEAD_DIM;
            
            // OPTIMIZATION 7: Vectorized weighted accumulation
            for (uint d = 0; d < HEAD_DIM; d += 4) {
                T4 current_output = *((device T4*)(output + out_base + d));
                T4 value_vec = *((device T4*)(values + v_base + d));
                *((device T4*)(output + out_base + d)) = current_output + attn_weight * value_vec;
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
            name="optimized_block_diagonal_attention",
            input_names=["queries", "keys", "values", "mask", "scale"],
            output_names=["output"],
            source=kernel_source
        )
        
        # OPTIMIZATION 8: Better GPU utilization with larger threadgroups
        # Use (64, 1, 1) instead of (32, 1, 1) for better occupancy
        threadgroup_size = min(64, L)  # Adapt to sequence length
        
        # Execute kernel with optimized parameters
        outputs = kernel(
            inputs=[q, k, v, mask, scale_tensor],
            output_shapes=[(B, H, L, D)],     # Output shape
            output_dtypes=[q.dtype],          # Output dtype
            grid=(L, H, B),                   # Grid dimensions: (SEQ_LEN, NUM_HEADS, BATCH_SIZE)
            threadgroup=(threadgroup_size, 1, 1),  # Optimized threadgroup size
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

def create_block_diagonal_mask(B, H, L, block_sizes):
    """Create block-diagonal mask for packed sequences - same as evaluator."""
    mask_np = np.zeros((B, H, L, L), dtype=bool)
    
    current_pos = 0
    for block_size in block_sizes:
        if current_pos + block_size <= L:
            end_pos = current_pos + block_size
            mask_np[:, :, current_pos:end_pos, current_pos:end_pos] = True
            current_pos = end_pos
        else:
            break
    
    return mx.array(mask_np)


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
        
        # Create TRUE block-diagonal mask using the same function as evaluator
        # 4 blocks of 128 each: [128, 128, 128, 128]
        block_sizes = [128, 128, 128, 128]
        mask = create_block_diagonal_mask(B, H, L, block_sizes)
        
        is_bd = is_true_block_diagonal_mask(mask)
        sparsity = 1.0 - float(mx.mean(mask.astype(mx.float32)))
        print(f"TRUE block-diagonal mask:")
        print(f"  Block sizes used: {block_sizes}")
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
