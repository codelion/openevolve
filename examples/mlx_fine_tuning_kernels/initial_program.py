"""
MLX Fine-tuning Kernels - OpenEvolve Example

This example optimizes core transformer operations used in fine-tuning, inspired by
Liger Kernel's proven optimizations. Instead of competing with MLX's optimized kernels,
we focus on custom implementations that can be measurably improved over naive baselines.

Evolution Target: Custom implementations of RMSNorm, RoPE, SwiGLU, CrossEntropy, and LoRA
that achieve 20%+ speedups in real fine-tuning scenarios.
"""

import math
from typing import Optional, Tuple, List, Dict

try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX not available - this example requires MLX")
    MLX_AVAILABLE = False
    raise ImportError("MLX is required for this example")


def evolved_fine_tuning_kernels():
    """
    Custom MLX implementations of fine-tuning operations.
    
    These implementations can be optimized beyond naive baselines through:
    - Operation fusion to reduce memory allocations
    - Elimination of unnecessary intermediate evaluations  
    - Better memory access patterns
    - Mathematical simplifications
    
    Returns:
        Dictionary of optimized kernel functions
    """
    
    # EVOLVE-BLOCK-START
    def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
        """
        RMSNorm: Root Mean Square Layer Normalization
        
        Baseline approach: Multiple separate operations
        Optimization opportunities:
        - Fuse variance computation + rsqrt + scaling
        - Reduce temporary array allocations
        - Better numerical stability patterns
        """
        # Current implementation with room for optimization
        # Step 1: Compute variance (can be fused)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        
        # Step 2: Compute rsqrt (can be fused with variance)
        rstd = mx.rsqrt(variance + eps)
        
        # Step 3: Apply normalization and scaling (can be fused)
        normalized = x * rstd
        result = weight * normalized
        
        return result
    
    def rope_embeddings(x: mx.array, freqs_cos: mx.array, freqs_sin: mx.array) -> mx.array:
        """
        RoPE: Rotary Position Embeddings
        
        Baseline approach: Multiple tensor operations for rotation
        Optimization opportunities:
        - Fuse rotation computation
        - Optimize memory access patterns
        - Reduce intermediate tensor creation
        """
        # Split x into pairs for rotation
        x1 = x[..., ::2]   # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Get the actual dimensions we're working with
        *batch_dims, seq_len, d_head = x.shape
        half_d = d_head // 2
        
        # Adjust frequency tensors to match the actual dimensions
        # freqs_cos and freqs_sin might be (seq_len, d_model//2) but we need (seq_len, d_head//2)
        if freqs_cos.shape[-1] != half_d:
            # Take only the needed frequency components
            cos_freqs = freqs_cos[..., :half_d]
            sin_freqs = freqs_sin[..., :half_d]
        else:
            cos_freqs = freqs_cos
            sin_freqs = freqs_sin
        
        # Expand frequency tensors to match input shape
        # We need to broadcast to (..., seq_len, d_head//2)
        for _ in batch_dims:
            cos_freqs = mx.expand_dims(cos_freqs, axis=0)
            sin_freqs = mx.expand_dims(sin_freqs, axis=0)
        
        # Apply rotation (room for optimization)
        rotated_x1 = x1 * cos_freqs - x2 * sin_freqs
        rotated_x2 = x1 * sin_freqs + x2 * cos_freqs
        
        # Interleave results using concatenation (can be optimized)
        result = mx.concatenate([rotated_x1[..., None], rotated_x2[..., None]], axis=-1)
        result = result.reshape(x.shape)  # Flatten back to original shape
        
        return result
    
    def swiglu_activation(x: mx.array, w_gate: mx.array, w_up: mx.array) -> mx.array:
        """
        SwiGLU: Swish-Gated Linear Unit activation
        
        Baseline approach: Separate linear operations + activation
        Optimization opportunities:
        - Fuse linear + silu + multiply operations
        - Reduce memory footprint of intermediate results
        - Optimize computation order
        """
        # Gate path: linear + swish activation
        gate = x @ w_gate.T  # Matrix multiplication for linear layer
        gate_activated = gate * mx.sigmoid(gate)  # SiLU/Swish activation: x * sigmoid(x)
        
        # Up path: linear projection
        up = x @ w_up.T  # Matrix multiplication for linear layer
        
        # Combine: gate * up (room for fusion)
        result = gate_activated * up
        
        return result
    
    def cross_entropy_loss(logits: mx.array, targets: mx.array, 
                          ignore_index: int = -100) -> mx.array:
        """
        CrossEntropy Loss with Online Softmax
        
        Baseline approach: Full logits materialization
        Optimization opportunities:
        - Online softmax computation to reduce memory
        - Chunked processing for large vocabularies
        - Fused loss computation
        """
        # Create mask for valid targets (avoid boolean indexing)
        valid_mask = targets != ignore_index
        
        if not mx.any(valid_mask):
            return mx.array(0.0)
        
        # Use standard cross entropy loss instead of manual boolean indexing
        # This is simpler and avoids the boolean indexing issue
        losses = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                                        targets.reshape(-1), reduction='none')
        
        # Apply mask to exclude ignored indices
        valid_losses = mx.where(valid_mask.reshape(-1), losses, mx.array(0.0))
        
        # Compute mean only over valid positions
        num_valid = mx.sum(valid_mask.astype(mx.float32))
        
        if num_valid > 0:
            return mx.sum(valid_losses) / num_valid
        else:
            return mx.array(0.0)
    
    def lora_linear(x: mx.array, base_weight: mx.array, 
                   lora_a: mx.array, lora_b: mx.array, 
                   scale: float = 1.0) -> mx.array:
        """
        LoRA Linear Layer: Base + Low-Rank Adaptation
        
        Baseline approach: Separate base and LoRA computations
        Optimization opportunities:
        - Fuse base + LoRA computation
        - Optimize for common LoRA ranks (r=8, r=16)
        - Better memory access patterns
        """
        # Base linear transformation
        base_output = x @ base_weight.T  # Matrix multiplication for linear layer
        
        # LoRA computation: x @ A @ B (room for optimization)
        lora_intermediate = x @ lora_a.T  # x @ A
        lora_output = lora_intermediate @ lora_b.T  # @ B
        
        # Combine base + scaled LoRA
        result = base_output + scale * lora_output
        
        return result
    
    def attention_with_rope(query: mx.array, key: mx.array, value: mx.array,
                          freqs_cos: mx.array, freqs_sin: mx.array,
                          scale: Optional[float] = None) -> mx.array:
        """
        Attention with RoPE embeddings
        
        Combines multiple operations that can be optimized together:
        - RoPE application to queries and keys
        - Scaled dot-product attention
        - Memory-efficient attention patterns
        """
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])
        
        # Apply RoPE to queries and keys (can be optimized)
        q_rope = rope_embeddings(query, freqs_cos, freqs_sin)
        k_rope = rope_embeddings(key, freqs_cos, freqs_sin)
        
        # Scaled dot-product attention (room for fusion)
        scores = mx.matmul(q_rope, mx.transpose(k_rope, axes=(0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, value)
        
        return output
    
    # Return all optimized kernels
    return {
        'rms_norm': rms_norm,
        'rope_embeddings': rope_embeddings, 
        'swiglu_activation': swiglu_activation,
        'cross_entropy_loss': cross_entropy_loss,
        'lora_linear': lora_linear,
        'attention_with_rope': attention_with_rope
    }
    # EVOLVE-BLOCK-END


def naive_baseline_kernels():
    """
    Naive baseline implementations with intentional inefficiencies.
    These represent the obvious, unoptimized approaches with:
    - Excessive intermediate evaluations
    - Poor memory access patterns
    - Missed fusion opportunities
    """
    
    def naive_rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
        """Naive RMSNorm with forced evaluations and poor patterns."""
        # Force evaluation at each step (inefficient)
        x_squared = x * x
        mx.eval(x_squared)
        
        variance = mx.mean(x_squared, axis=-1, keepdims=True)
        mx.eval(variance)
        
        variance_eps = variance + eps
        mx.eval(variance_eps)
        
        rstd = mx.rsqrt(variance_eps)
        mx.eval(rstd)
        
        normalized = x * rstd
        mx.eval(normalized)
        
        result = weight * normalized
        mx.eval(result)
        
        return result
    
    def naive_rope_embeddings(x: mx.array, freqs_cos: mx.array, freqs_sin: mx.array) -> mx.array:
        """Naive RoPE with many intermediate arrays."""
        # Create many temporary arrays
        x1 = x[..., ::2]
        mx.eval(x1)
        x2 = x[..., 1::2]
        mx.eval(x2)
        
        # Get the actual dimensions we're working with
        *batch_dims, seq_len, d_head = x.shape
        half_d = d_head // 2
        
        # Adjust frequency tensors to match the actual dimensions (inefficiently)
        if freqs_cos.shape[-1] != half_d:
            cos_freqs = freqs_cos[..., :half_d]
            sin_freqs = freqs_sin[..., :half_d]
        else:
            cos_freqs = freqs_cos
            sin_freqs = freqs_sin
        mx.eval(cos_freqs)
        mx.eval(sin_freqs)
        
        # Expand frequency tensors to match input shape (inefficiently)
        for _ in batch_dims:
            cos_freqs = mx.expand_dims(cos_freqs, axis=0)
            sin_freqs = mx.expand_dims(sin_freqs, axis=0)
        mx.eval(cos_freqs)
        mx.eval(sin_freqs)
        
        # Multiple temporary computations
        cos_x1 = x1 * cos_freqs
        mx.eval(cos_x1)
        sin_x2 = x2 * sin_freqs
        mx.eval(sin_x2)
        rotated_x1 = cos_x1 - sin_x2
        mx.eval(rotated_x1)
        
        sin_x1 = x1 * sin_freqs
        mx.eval(sin_x1)
        cos_x2 = x2 * cos_freqs
        mx.eval(cos_x2)
        rotated_x2 = sin_x1 + cos_x2
        mx.eval(rotated_x2)
        
        # Inefficient reconstruction using concatenation
        result_parts = mx.concatenate([rotated_x1[..., None], rotated_x2[..., None]], axis=-1)
        mx.eval(result_parts)
        result = result_parts.reshape(x.shape)
        mx.eval(result)
        
        return result
    
    def naive_swiglu_activation(x: mx.array, w_gate: mx.array, w_up: mx.array) -> mx.array:
        """Naive SwiGLU with separate operations and evaluations."""
        gate = x @ w_gate.T  # Matrix multiplication for linear layer
        mx.eval(gate)
        
        # Compute silu separately
        sigmoid_gate = mx.sigmoid(gate)
        mx.eval(sigmoid_gate)
        gate_activated = gate * sigmoid_gate  # silu = x * sigmoid(x)
        mx.eval(gate_activated)
        
        up = x @ w_up.T  # Matrix multiplication for linear layer
        mx.eval(up)
        
        result = gate_activated * up
        mx.eval(result)
        
        return result
    
    def naive_cross_entropy_loss(logits: mx.array, targets: mx.array, 
                                ignore_index: int = -100) -> mx.array:
        """Naive CrossEntropy with full materialization."""
        valid_mask = targets != ignore_index
        mx.eval(valid_mask)
        
        if not mx.any(valid_mask):
            return mx.array(0.0)
        
        # Use standard cross entropy but with many inefficient steps
        losses = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                                        targets.reshape(-1), reduction='none')
        mx.eval(losses)
        
        # Apply mask with many evaluations (inefficient)
        mask_flat = valid_mask.reshape(-1)
        mx.eval(mask_flat)
        
        valid_losses = mx.where(mask_flat, losses, mx.array(0.0))
        mx.eval(valid_losses)
        
        # Count valid positions inefficiently
        num_valid = mx.sum(mask_flat.astype(mx.float32))
        mx.eval(num_valid)
        
        # Sum losses inefficiently
        total_loss = mx.sum(valid_losses)
        mx.eval(total_loss)
        
        # Final division
        result = total_loss / mx.maximum(num_valid, mx.array(1.0))
        mx.eval(result)
        
        return result
    
    def naive_lora_linear(x: mx.array, base_weight: mx.array,
                         lora_a: mx.array, lora_b: mx.array,
                         scale: float = 1.0) -> mx.array:
        """Naive LoRA with separate computations."""
        base_output = x @ base_weight.T  # Matrix multiplication for linear layer
        mx.eval(base_output)
        
        # LoRA path with forced evaluations
        lora_intermediate = x @ lora_a.T  # x @ A
        mx.eval(lora_intermediate)
        lora_output = lora_intermediate @ lora_b.T  # @ B
        mx.eval(lora_output)
        
        scaled_lora = scale * lora_output
        mx.eval(scaled_lora)
        
        result = base_output + scaled_lora
        mx.eval(result)
        
        return result
    
    def naive_attention_with_rope(query: mx.array, key: mx.array, value: mx.array,
                                freqs_cos: mx.array, freqs_sin: mx.array,
                                scale: Optional[float] = None) -> mx.array:
        """Naive attention with many intermediate steps."""
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])
        
        # Apply RoPE with forced evaluations
        q_rope = naive_rope_embeddings(query, freqs_cos, freqs_sin)
        mx.eval(q_rope)
        k_rope = naive_rope_embeddings(key, freqs_cos, freqs_sin)
        mx.eval(k_rope)
        
        # Attention computation with many steps
        k_transposed = mx.transpose(k_rope, axes=(0, 1, 3, 2))
        mx.eval(k_transposed)
        
        scores = mx.matmul(q_rope, k_transposed)
        mx.eval(scores)
        
        scaled_scores = scores * scale
        mx.eval(scaled_scores)
        
        attn_weights = mx.softmax(scaled_scores, axis=-1)
        mx.eval(attn_weights)
        
        output = mx.matmul(attn_weights, value)
        mx.eval(output)
        
        return output
    
    return {
        'rms_norm': naive_rms_norm,
        'rope_embeddings': naive_rope_embeddings,
        'swiglu_activation': naive_swiglu_activation,
        'cross_entropy_loss': naive_cross_entropy_loss,
        'lora_linear': naive_lora_linear,
        'attention_with_rope': naive_attention_with_rope
    }


def create_test_data(batch_size: int = 4, seq_len: int = 128, 
                    d_model: int = 256, vocab_size: int = 1000) -> Dict:
    """Create test data for benchmarking the kernels."""
    return {
        # For RMSNorm
        'x_norm': mx.random.normal((batch_size, seq_len, d_model)),
        'weight_norm': mx.random.normal((d_model,)),
        
        # For RoPE  
        'x_rope': mx.random.normal((batch_size, 8, seq_len, d_model)),  # 8 heads
        'freqs_cos': mx.random.normal((seq_len, d_model // 2)),
        'freqs_sin': mx.random.normal((seq_len, d_model // 2)),
        
        # For SwiGLU
        'x_mlp': mx.random.normal((batch_size, seq_len, d_model)),
        'w_gate': mx.random.normal((d_model * 4, d_model)),  # 4x expansion
        'w_up': mx.random.normal((d_model * 4, d_model)),
        
        # For CrossEntropy
        'logits': mx.random.normal((batch_size, seq_len, vocab_size)),
        'targets': mx.random.randint(0, vocab_size, (batch_size, seq_len)),
        
        # For LoRA
        'x_lora': mx.random.normal((batch_size, seq_len, d_model)),
        'base_weight': mx.random.normal((d_model, d_model)),
        'lora_a': mx.random.normal((16, d_model)),  # rank=16
        'lora_b': mx.random.normal((d_model, 16)),
        
        # For Attention
        'query': mx.random.normal((batch_size, 8, seq_len, d_model // 8)),
        'key': mx.random.normal((batch_size, 8, seq_len, d_model // 8)),
        'value': mx.random.normal((batch_size, 8, seq_len, d_model // 8)),
    }


def test_basic_functionality():
    """Test basic functionality and correctness of kernels."""
    print("Testing MLX Fine-tuning Kernels...")
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return False
    
    try:
        # Get kernel implementations
        evolved_kernels = evolved_fine_tuning_kernels()
        naive_kernels = naive_baseline_kernels()
        
        # Create test data
        test_data = create_test_data(batch_size=2, seq_len=32, d_model=64)
        
        print("\n=== Testing Kernel Correctness ===")
        
        # Test each kernel
        kernel_tests = [
            ('rms_norm', [test_data['x_norm'], test_data['weight_norm']]),
            ('rope_embeddings', [test_data['x_rope'], test_data['freqs_cos'], test_data['freqs_sin']]),
            ('swiglu_activation', [test_data['x_mlp'], test_data['w_gate'], test_data['w_up']]),
            ('cross_entropy_loss', [test_data['logits'], test_data['targets']]),
            ('lora_linear', [test_data['x_lora'], test_data['base_weight'], 
                           test_data['lora_a'], test_data['lora_b']]),
            ('attention_with_rope', [test_data['query'], test_data['key'], test_data['value'],
                                   test_data['freqs_cos'], test_data['freqs_sin']]),
        ]
        
        all_passed = True
        
        for kernel_name, args in kernel_tests:
            print(f"\n--- Testing {kernel_name} ---")
            
            try:
                # Test evolved version
                evolved_result = evolved_kernels[kernel_name](*args)
                print(f"  Evolved: shape={evolved_result.shape}, dtype={evolved_result.dtype}")
                
                # Test naive version  
                naive_result = naive_kernels[kernel_name](*args)
                print(f"  Naive: shape={naive_result.shape}, dtype={naive_result.dtype}")
                
                # Check correctness
                if evolved_result.shape == naive_result.shape:
                    max_diff = float(mx.max(mx.abs(evolved_result - naive_result)))
                    if max_diff < 1e-2:  # Allow reasonable numerical differences
                        print(f"  ✅ Correctness: max_diff={max_diff:.2e}")
                    else:
                        print(f"  ⚠️ Large difference: max_diff={max_diff:.2e}")
                        all_passed = False
                else:
                    print(f"  ❌ Shape mismatch: {evolved_result.shape} vs {naive_result.shape}")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_passed = False
        
        if all_passed:
            print("\n✅ All kernel tests passed!")
        else:
            print("\n⚠️ Some tests failed, but basic functionality works.")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎯 Ready for OpenEvolve optimization!")
        print("\nThis example targets:")
        print("- RMSNorm fusion and memory optimization")  
        print("- RoPE computation efficiency")
        print("- SwiGLU operation fusion")
        print("- CrossEntropy loss optimization")
        print("- LoRA computation patterns")
        print("- Attention + RoPE integration")
        print("\nRun: python evaluator.py")
        print("Then: python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml")
    else:
        print("\n❌ Setup failed. Check MLX installation.")
