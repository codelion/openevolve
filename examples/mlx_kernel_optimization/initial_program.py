# EVOLVE-BLOCK-START
"""MLX Training Performance Optimization for Apple Silicon"""
import mlx.core as mx
import numpy as np
import time
import psutil
import platform


def choose_tile_size(M, N, K, device_info):
    """
    Choose optimal tile sizes for MLX matrix multiplication in training scenarios
    
    This function is the core of the optimization - it determines
    how to break large matrices into smaller tiles for better
    cache utilization and memory bandwidth on Apple Silicon during training.
    
    Args:
        M, N, K: Matrix dimensions for C = A @ B where A is MxK, B is KxN
        device_info: Apple Silicon device characteristics
        
    Returns:
        (tile_M, tile_N, tile_K): Optimal tile sizes
    """
    
    chip = device_info.get("chip", "Unknown")
    memory_gb = device_info.get("memory_gb", 8.0)
    
    # Detect workload type based on matrix characteristics
    total_elements = M * N * K
    aspect_ratio_MN = max(M, N) / min(M, N) if min(M, N) > 0 else 1.0
    aspect_ratio_K = K / min(M, N) if min(M, N) > 0 else 1.0
    
    # Classify training workload patterns  
    is_batch_heavy = (M > 256)  # Large batch dimension common in training
    is_mlp = (aspect_ratio_K > 1.5 or max(M, N) > 1.5 * K)  # MLP layers (4x expansion)
    is_attention = (aspect_ratio_MN < 2.0 and K > 256)  # Square-ish attention matrices
    is_large = total_elements > 2_000_000  # Lower threshold for training focus
    
    # Base configurations per chip generation - training optimized
    if "M4" in chip:
        base_tile = 128 if is_large else 80
        vector_align = 32
        cache_factor = 1.4  # Higher for training's repeated patterns
    elif "M3" in chip:
        base_tile = 112 if is_large else 72
        vector_align = 32
        cache_factor = 1.3
    elif "M2" in chip:
        base_tile = 96 if is_large else 64
        vector_align = 16
        cache_factor = 1.2
    else:  # M1 or unknown
        base_tile = 80 if is_large else 56
        vector_align = 16
        cache_factor = 1.1
    
    # Memory scaling - more aggressive for training
    if memory_gb >= 32:
        memory_scale = 1.5  # Training can use more memory
    elif memory_gb >= 16:
        memory_scale = 1.3
    else:
        memory_scale = 1.1
    
    # Training workload-specific adjustments
    if is_batch_heavy:
        # Large batch training benefits from different tiling
        workload_scale = 1.2
        batch_bias = 1.1  # Slightly favor M dimension (batch)
    else:
        workload_scale = 1.0
        batch_bias = 1.0
    
    if is_mlp:
        # MLP layers need K-dimension optimization for 4x expansion
        k_bias = 1.3
        mlp_scale = 1.1
    else:
        k_bias = 1.0
        mlp_scale = 1.0
    
    if is_attention:
        # Attention patterns in training
        attention_scale = 1.05
        k_bias = max(k_bias, 0.95)  # Balanced for attention
    else:
        attention_scale = 1.0
    
    # Calculate base tile sizes
    effective_base = int(
        base_tile * cache_factor * memory_scale * workload_scale * mlp_scale * attention_scale
    )
    
    # Dimension-specific tile sizes with training bias
    tile_M = min(int(effective_base * batch_bias), M)
    tile_N = min(effective_base, N)
    tile_K = min(int(effective_base * k_bias), K)
    
    # Training-specific progressive sizing
    if total_elements > 10_000_000:  # Very large training batch
        scale = 0.8
    elif total_elements > 5_000_000:  # Large training batch
        scale = 0.9
    elif total_elements > 1_000_000:  # Medium training batch
        scale = 1.1
    elif total_elements > 100_000:   # Small training batch
        scale = 1.4
    else:  # Very small - be conservative
        scale = 1.6
    
    tile_M = int(tile_M * scale)
    tile_N = int(tile_N * scale)
    tile_K = int(tile_K * scale)
    
    # Ensure vector alignment
    tile_M = ((tile_M + vector_align - 1) // vector_align) * vector_align
    tile_N = ((tile_N + vector_align - 1) // vector_align) * vector_align
    tile_K = ((tile_K + vector_align - 1) // vector_align) * vector_align
    
    # Clamp to valid ranges
    tile_M = max(vector_align, min(tile_M, M))
    tile_N = max(vector_align, min(tile_N, N))
    tile_K = max(vector_align, min(tile_K, K))
    
    return tile_M, tile_N, tile_K


def optimized_matmul(A, B, tile_M, tile_N, tile_K):
    """
    Perform optimized tiled matrix multiplication for training workloads
    
    This function implements the actual tiled multiplication
    using the tile sizes determined by choose_tile_size().
    Optimized for training patterns including forward and backward passes.
    
    Args:
        A: Input matrix A (M x K)
        B: Input matrix B (K x N)
        tile_M, tile_N, tile_K: Tile sizes
        
    Returns:
        Result matrix C (M x N)
    """
    M, K1 = A.shape
    K2, N = B.shape
    
    if K1 != K2:
        raise ValueError(f"Matrix dimensions incompatible: {K1} != {K2}")
    
    K = K1
    
    # For small matrices, use direct multiplication to avoid overhead
    total_elements = M * N * K
    if total_elements < 50_000:  # Lower threshold for training focus
        return mx.matmul(A, B)
    
    # Check if tiling makes sense (avoid excessive tile overhead)
    num_m_tiles = (M + tile_M - 1) // tile_M
    num_n_tiles = (N + tile_N - 1) // tile_N
    num_k_tiles = (K + tile_K - 1) // tile_K
    
    # If we have too many tiny tiles, use direct multiplication
    if num_m_tiles * num_n_tiles * num_k_tiles > 800:  # More permissive for training
        return mx.matmul(A, B)
    
    # Initialize result matrix
    C = mx.zeros((M, N), dtype=A.dtype)
    
    # Optimized tiled multiplication for training
    # Use ikj loop order - good for training's memory access patterns
    for i in range(0, M, tile_M):
        i_end = min(i + tile_M, M)
        
        for k in range(0, K, tile_K):
            k_end = min(k + tile_K, K)
            A_tile = A[i:i_end, k:k_end]
            
            for j in range(0, N, tile_N):
                j_end = min(j + tile_N, N)
                B_tile = B[k:k_end, j:j_end]
                
                # Compute partial result
                partial = mx.matmul(A_tile, B_tile)
                
                # Accumulate in result matrix
                C = C.at[i:i_end, j:j_end].add(partial)
    
    return C


# EVOLVE-BLOCK-END


# Fixed evaluation framework - NOT evolved
def get_device_info():
    """Get Apple Silicon device characteristics - FIXED IMPLEMENTATION"""
    try:
        import subprocess
        chip_info = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], 
            capture_output=True, 
            text=True,
            timeout=5
        ).stdout
        
        chip_name = "Unknown"
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        
        for line in chip_info.split('\n'):
            if 'Chip:' in line:
                chip_name = line.split('Chip:')[1].strip()
                break
                
        return {
            "chip": chip_name,
            "memory_gb": memory_gb,
            "cpu_count": psutil.cpu_count()
        }
    except:
        return {
            "chip": "M2",  # Default assumption
            "memory_gb": 16.0,
            "cpu_count": 8
        }


def benchmark_training_performance():
    """
    Benchmark MLX training performance with current optimization - FIXED EVALUATION
    
    This function provides consistent, reliable evaluation across all iterations.
    It should NOT be evolved to ensure fair comparison.
    
    Returns:
        Performance metrics comparing original vs optimized training
    """
    import mlx.nn as nn
    import mlx.optimizers as optim
    import gc
    
    device_info = get_device_info()
    original_matmul = mx.matmul
    
    # Create optimized matmul function using current evolved functions
    def create_optimized_matmul():
        def opt_matmul(A, B):
            # Lower threshold for training focus - catch more operations
            if (len(A.shape) == 2 and len(B.shape) == 2 and 
                A.shape[0] * A.shape[1] * B.shape[1] > 15_000):  # Lower threshold
                
                M, K1 = A.shape
                K2, N = B.shape
                
                if K1 == K2:
                    tile_M, tile_N, tile_K = choose_tile_size(M, N, K1, device_info)
                    return optimized_matmul(A, B, tile_M, tile_N, tile_K)
            
            return original_matmul(A, B)
        return opt_matmul
    
    try:
        # Create a realistic training model for optimization testing
        class TrainingTransformer(nn.Module):
            def __init__(self, vocab_size=5000, hidden_dim=1024, seq_len=512):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                # Multiple layers to create substantial matrix operations
                self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)  # MLP expansion
                self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)  # MLP projection  
                self.attention_q = nn.Linear(hidden_dim, hidden_dim)  # Attention query
                self.attention_k = nn.Linear(hidden_dim, hidden_dim)  # Attention key
                self.attention_v = nn.Linear(hidden_dim, hidden_dim)  # Attention value
                self.attention_out = nn.Linear(hidden_dim, hidden_dim)  # Attention output
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.output = nn.Linear(hidden_dim, vocab_size)  # Large output projection
                
            def __call__(self, x):
                # Transformer-like forward pass with substantial matrix operations
                x = self.embedding(x)  # [batch, seq, hidden]
                
                # Attention-like operations
                q = self.attention_q(x)
                k = self.attention_k(x) 
                v = self.attention_v(x)
                # Simplified attention (real would have more ops)
                attn_out = self.attention_out(v)
                x = self.norm1(x + attn_out)
                
                # MLP operations
                mlp_out = self.linear2(nn.gelu(self.linear1(x)))
                x = self.norm2(x + mlp_out)
                
                # Output projection
                return self.output(x)
        
        # Training configuration - larger for more matrix operations
        batch_size = 24      # Substantial batch size
        seq_len = 512        # Longer sequences  
        vocab_size = 5000    # Reasonable vocabulary
        hidden_dim = 1024    # Large hidden dimension
        
        # Create model and optimizer
        model = TrainingTransformer(vocab_size, hidden_dim, seq_len)
        optimizer = optim.Adam(learning_rate=1e-3)
        
        # Training step function
        def training_step():
            # Generate random training batch
            inputs = mx.random.randint(0, vocab_size, (batch_size, seq_len))
            targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))
            
            def loss_fn(model, inputs, targets):
                logits = model(inputs)  # Forward pass
                return nn.losses.cross_entropy(
                    logits.reshape(-1, vocab_size), 
                    targets.reshape(-1), 
                    reduction='mean'
                )
            
            # Forward and backward pass
            loss, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            
            return loss
        
        # Test with original MLX
        mx.matmul = original_matmul
        
        # Extended warmup to stabilize timing
        for _ in range(8):
            training_step()
        
        # Benchmark original MLX
        original_times = []
        for _ in range(15):  # More iterations for better statistics
            start_time = time.perf_counter()
            training_step()
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
        
        # Remove outliers and calculate median
        original_times = sorted(original_times)[2:-2]  # Remove 2 highest and lowest
        original_time = np.median(original_times)
        
        # Test with optimized MLX
        mx.matmul = create_optimized_matmul()
        
        # Extended warmup for optimized version
        for _ in range(8):
            training_step()
        
        # Benchmark optimized MLX
        optimized_times = []
        for _ in range(15):  # More iterations for better statistics
            start_time = time.perf_counter()
            training_step()
            end_time = time.perf_counter()
            optimized_times.append(end_time - start_time)
        
        # Restore original
        mx.matmul = original_matmul
        
        # Remove outliers and calculate median
        optimized_times = sorted(optimized_times)[2:-2]
        optimized_time = np.median(optimized_times)
        
        speedup = original_time / optimized_time if optimized_time > 0 else 0.0
        
        # Clean up
        del model, optimizer
        gc.collect()
        
        return {
            "training_speedup": speedup,
            "original_time": original_time,
            "optimized_time": optimized_time,
            "test_successful": True
        }
        
    except Exception as e:
        mx.matmul = original_matmul  # Always restore
        return {"error": str(e), "training_speedup": 0.0, "test_successful": False}


def run_optimization():
    """
    Run the MLX training optimization benchmark - FIXED INTERFACE
    
    This function provides a consistent interface for the OpenEvolve evaluator.
    It calls the current evolved optimization functions through the fixed benchmark.
    """
    
    device_info = get_device_info()
    
    # Run training benchmark using current evolved functions
    training_results = benchmark_training_performance()
    
    # Calculate summary metrics - simple training-only scoring
    training_speedup = training_results.get("training_speedup", 0.0)
    
    # Simple combined score = training speedup with bonuses
    combined_score = training_speedup
    if training_speedup > 1.15:  # >15% improvement
        combined_score *= 1.3
    elif training_speedup > 1.10:  # >10% improvement  
        combined_score *= 1.2
    elif training_speedup > 1.05:  # >5% improvement
        combined_score *= 1.1
    
    # Create results summary for evaluator
    results = [{
        "optimization_type": "mlx_training",
        "speedup": training_speedup,
        "metrics": {
            "training_speedup": training_speedup,
            "combined_score": combined_score
        }
    }]
    
    return results, combined_score, training_results.get("optimized_time", 1.0), device_info


if __name__ == "__main__":
    print("🚀 MLX Training Optimization Test")
    print("=" * 50)
    
    device_info = get_device_info()
    print(f"Device: {device_info['chip']} ({device_info['memory_gb']} GB RAM)")
    
    # Test the current optimization
    results = benchmark_training_performance()
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        speedup = results["training_speedup"]
        original_time = results["original_time"]
        optimized_time = results["optimized_time"]
        
        print(f"\n📊 Training Results:")
        print(f"   Original time: {original_time:.4f}s per step")
        print(f"   Optimized time: {optimized_time:.4f}s per step")
        print(f"   Training speedup: {speedup:.3f}x")
        
        if speedup > 1.10:
            print("   ✅ Significant training acceleration!")
        elif speedup > 1.05:
            print("   ✅ Moderate training improvement!")
        elif speedup > 1.02:
            print("   ⚪ Small training improvement")
        elif speedup > 0.98:
            print("   ⚪ No significant change")
        else:
            print("   ❌ Training performance regression")
