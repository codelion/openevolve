# EVOLVE-BLOCK-START
"""MLX Matrix Multiplication Tiling Optimization"""
import mlx.core as mx
import numpy as np
import time
import psutil
import platform


def get_device_info():
    """Get Apple Silicon device characteristics"""
    try:
        # Try to get Mac chip info
        import subprocess
        chip_info = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], 
            capture_output=True, 
            text=True
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
            "chip": "Unknown",
            "memory_gb": 8.0,
            "cpu_count": 8
        }


def choose_tile_size(M, N, K, device_info):
    """
    Choose optimal tile sizes for MLX matrix multiplication
    
    This heuristic determines tile sizes for C = A @ B where:
    - A is M x K 
    - B is K x N
    - C is M x N
    
    Args:
        M, N, K: Matrix dimensions
        device_info: Apple Silicon device characteristics
        
    Returns:
        (tile_M, tile_N, tile_K): Optimal tile sizes
    """
    
    # Simple baseline heuristic - room for improvement!
    
    # Basic tile sizes (conservative approach)
    base_tile = 64
    
    # Adjust based on matrix size
    if M <= 128 and N <= 128 and K <= 128:
        # Small matrices - use smaller tiles
        tile_M = min(32, M)
        tile_N = min(32, N) 
        tile_K = min(32, K)
    elif M >= 1024 or N >= 1024 or K >= 1024:
        # Large matrices - use larger tiles
        tile_M = min(128, M)
        tile_N = min(128, N)
        tile_K = min(128, K)
    else:
        # Medium matrices - use base tiles
        tile_M = min(base_tile, M)
        tile_N = min(base_tile, N)
        tile_K = min(base_tile, K)
    
    # Simple memory-based adjustment
    if device_info["memory_gb"] >= 16:
        # More memory available - can use larger tiles
        tile_M = min(tile_M * 2, M)
        tile_N = min(tile_N * 2, N)
        tile_K = min(tile_K * 2, K)
    
    # Ensure tiles are multiples of 8 for better vectorization
    tile_M = ((tile_M + 7) // 8) * 8
    tile_N = ((tile_N + 7) // 8) * 8
    tile_K = ((tile_K + 7) // 8) * 8
    
    # Clamp to matrix dimensions
    tile_M = min(tile_M, M)
    tile_N = min(tile_N, N)
    tile_K = min(tile_K, K)
    
    return tile_M, tile_N, tile_K


def tiled_matmul(A, B, tile_M, tile_N, tile_K):
    """
    Perform tiled matrix multiplication using MLX
    
    Args:
        A: Matrix A (M x K)
        B: Matrix B (K x N) 
        tile_M, tile_N, tile_K: Tile sizes
        
    Returns:
        Result matrix C (M x N)
    """
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"Matrix dimensions incompatible: {K1} != {K2}"
    
    # Initialize result matrix
    C = mx.zeros((M, N), dtype=A.dtype)
    
    # Perform tiled multiplication
    for i in range(0, M, tile_M):
        for j in range(0, N, tile_N):
            for k in range(0, K1, tile_K):
                # Extract tiles
                i_end = min(i + tile_M, M)
                j_end = min(j + tile_N, N)
                k_end = min(k + tile_K, K1)
                
                A_tile = A[i:i_end, k:k_end]
                B_tile = B[k:k_end, j:j_end]
                
                # Compute tile multiplication and accumulate
                C_tile = mx.matmul(A_tile, B_tile)
                C = C.at[i:i_end, j:j_end].add(C_tile)
    
    return C


def benchmark_configuration(M, N, K, tile_M, tile_N, tile_K, num_runs=5):
    """
    Benchmark a specific tiling configuration
    
    Args:
        M, N, K: Matrix dimensions
        tile_M, tile_N, tile_K: Tile sizes
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with performance metrics
    """
    # Create test matrices
    A = mx.random.normal((M, K), dtype=mx.float16)
    B = mx.random.normal((K, N), dtype=mx.float16)
    
    # Warmup
    for _ in range(2):
        C = tiled_matmul(A, B, tile_M, tile_N, tile_K)
        mx.eval(C)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        C = tiled_matmul(A, B, tile_M, tile_N, tile_K)
        mx.eval(C)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate metrics
    mean_time = np.mean(times)
    
    # Calculate GFLOPS (2 * M * N * K operations)
    total_ops = 2 * M * N * K
    gflops = total_ops / (mean_time * 1e9)
    
    # Calculate efficiency metrics
    memory_usage = (M * K + K * N + M * N) * 2  # float16 = 2 bytes
    memory_bandwidth = memory_usage / mean_time / 1e9  # GB/s
    
    return {
        "mean_time": mean_time,
        "gflops": gflops,
        "memory_bandwidth_gbs": memory_bandwidth,
        "tile_efficiency": (tile_M * tile_N * tile_K) / (M * N * K)
    }


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_optimization():
    """Run the MLX kernel optimization with current tiling heuristic"""
    
    device_info = get_device_info()
    
    # Test on common transformer matrix sizes
    test_cases = [
        # Transformer attention matrices (seq_len x hidden_dim)
        (512, 768, 768),    # BERT-like attention 
        (1024, 768, 768),   # Longer sequence attention
        (512, 768, 3072),   # MLP layer (hidden to 4*hidden)
        (512, 3072, 768),   # MLP layer (4*hidden to hidden)
        
        # Larger model dimensions
        (512, 1024, 1024),  # Larger transformer attention
        (512, 1024, 4096),  # Larger MLP layer
        
        # Batch processing
        (128, 512, 512),    # Smaller batch
        (256, 512, 512),    # Medium batch
    ]
    
    total_gflops = 0
    total_time = 0
    results = []
    
    for M, N, K in test_cases:
        # Get optimal tile sizes using our heuristic
        tile_M, tile_N, tile_K = choose_tile_size(M, N, K, device_info)
        
        # Benchmark this configuration
        metrics = benchmark_configuration(M, N, K, tile_M, tile_N, tile_K)
        
        results.append({
            "matrix_size": (M, N, K),
            "tile_size": (tile_M, tile_N, tile_K),
            "metrics": metrics
        })
        
        total_gflops += metrics["gflops"]
        total_time += metrics["mean_time"]
    
    # Calculate aggregate metrics
    avg_gflops = total_gflops / len(test_cases)
    total_compute_time = total_time
    
    return results, avg_gflops, total_compute_time, device_info


if __name__ == "__main__":
    results, avg_gflops, total_time, device_info = run_optimization()
    
    print(f"Device: {device_info['chip']} ({device_info['memory_gb']} GB RAM)")
    print(f"Average GFLOPS: {avg_gflops:.1f}")
    print(f"Total compute time: {total_time:.3f}s")
    print("\nDetailed results:")
    
    for result in results:
        M, N, K = result["matrix_size"]
        tile_M, tile_N, tile_K = result["tile_size"]
        metrics = result["metrics"]
        
        print(f"  {M:4d}x{N:4d}x{K:4d} -> tiles({tile_M:3d},{tile_N:3d},{tile_K:3d}) = {metrics['gflops']:6.1f} GFLOPS")
