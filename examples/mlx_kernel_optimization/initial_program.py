# EVOLVE-BLOCK-START
"""MLX-LM Performance Optimization for Apple Silicon"""
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


def choose_tile_size(M, N, K, device_info):
    """
    Choose optimal tile sizes for MLX matrix multiplication
    
    This function is the core of the optimization - it determines
    how to break large matrices into smaller tiles for better
    cache utilization and memory bandwidth on Apple Silicon.
    
    Args:
        M, N, K: Matrix dimensions for C = A @ B where A is MxK, B is KxN
        device_info: Apple Silicon device characteristics
        
    Returns:
        (tile_M, tile_N, tile_K): Optimal tile sizes
    """
    
    # Simple baseline heuristic - optimize this function!
    
    chip = device_info.get("chip", "Unknown")
    memory_gb = device_info.get("memory_gb", 8.0)
    
    # Start with conservative base tile sizes
    if "M4" in chip:
        base_tile = 128
        vector_align = 32
    elif "M3" in chip:
        base_tile = 96
        vector_align = 32
    elif "M2" in chip:
        base_tile = 80
        vector_align = 16
    else:  # M1 or unknown
        base_tile = 64
        vector_align = 16
    
    # Adjust for memory
    if memory_gb >= 32:
        base_tile = int(base_tile * 1.2)
    elif memory_gb >= 16:
        base_tile = int(base_tile * 1.1)
    
    # Adjust based on matrix characteristics
    total_elements = M * N * K
    
    if total_elements > 10_000_000:  # Very large matrices
        scale = 0.8
    elif total_elements > 1_000_000:  # Large matrices 
        scale = 1.0
    elif total_elements > 100_000:   # Medium matrices
        scale = 1.2
    else:  # Small matrices
        scale = 1.5
    
    # Calculate tile sizes
    tile_M = min(int(base_tile * scale), M)
    tile_N = min(int(base_tile * scale), N)
    tile_K = min(int(base_tile * scale), K)
    
    # Ensure alignment with vector units
    tile_M = ((tile_M + vector_align - 1) // vector_align) * vector_align
    tile_N = ((tile_N + vector_align - 1) // vector_align) * vector_align
    tile_K = ((tile_K + vector_align - 1) // vector_align) * vector_align
    
    # Clamp to matrix dimensions and minimum size
    tile_M = max(vector_align, min(tile_M, M))
    tile_N = max(vector_align, min(tile_N, N))
    tile_K = max(vector_align, min(tile_K, K))
    
    return tile_M, tile_N, tile_K


def optimized_matmul(A, B, tile_M, tile_N, tile_K):
    """
    Perform optimized tiled matrix multiplication
    
    This function implements the actual tiled multiplication
    using the tile sizes determined by choose_tile_size().
    
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
    
    # Initialize result matrix
    C = mx.zeros((M, N), dtype=A.dtype)
    
    # Perform tiled multiplication
    for i in range(0, M, tile_M):
        for j in range(0, N, tile_N):
            for k in range(0, K, tile_K):
                # Calculate tile boundaries
                i_end = min(i + tile_M, M)
                j_end = min(j + tile_N, N)
                k_end = min(k + tile_K, K)
                
                # Extract tiles
                A_tile = A[i:i_end, k:k_end]
                B_tile = B[k:k_end, j:j_end]
                
                # Compute tile multiplication and accumulate
                C_tile = mx.matmul(A_tile, B_tile)
                C = C.at[i:i_end, j:j_end].add(C_tile)
    
    return C


def benchmark_mlx_lm_performance(model_name="mlx-community/Qwen2.5-0.5B-Instruct-bf16"):
    """
    Benchmark MLX-LM performance with current optimization
    
    Args:
        model_name: MLX model to test with
        
    Returns:
        Performance metrics comparing original vs optimized
    """
    try:
        from mlx_lm import load, generate
    except ImportError:
        return {
            "error": "mlx-lm not installed",
            "inference_speedup": 0.0,
            "training_speedup": 0.0
        }
    
    device_info = get_device_info()
    original_matmul = mx.matmul
    
    # Create optimized matmul function
    def create_optimized_matmul():
        def opt_matmul(A, B):
            if (len(A.shape) == 2 and len(B.shape) == 2 and 
                A.shape[0] * A.shape[1] * B.shape[1] > 50_000):
                
                M, K1 = A.shape
                K2, N = B.shape
                
                if K1 == K2:
                    tile_M, tile_N, tile_K = choose_tile_size(M, N, K1, device_info)
                    return optimized_matmul(A, B, tile_M, tile_N, tile_K)
            
            return original_matmul(A, B)
        return opt_matmul
    
    try:
        # Load model
        model, tokenizer = load(model_name)
        
        # Test prompts
        test_prompts = ["Hello world", "What is AI?", "Explain Python"]
        
        # Test original
        mx.matmul = original_matmul
        
        # Warmup
        for _ in range(2):
            generate(model, tokenizer, prompt="Hi", max_tokens=5, verbose=False)
        
        # Benchmark original
        start_time = time.perf_counter()
        for prompt in test_prompts:
            generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
        original_time = time.perf_counter() - start_time
        
        # Test optimized
        mx.matmul = create_optimized_matmul()
        
        # Warmup
        for _ in range(2):
            generate(model, tokenizer, prompt="Hi", max_tokens=5, verbose=False)
        
        # Benchmark optimized
        start_time = time.perf_counter()
        for prompt in test_prompts:
            generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
        optimized_time = time.perf_counter() - start_time
        
        # Restore original
        mx.matmul = original_matmul
        
        speedup = original_time / optimized_time if optimized_time > 0 else 0.0
        
        return {
            "inference_speedup": speedup,
            "original_time": original_time,
            "optimized_time": optimized_time,
            "model_loaded": True
        }
        
    except Exception as e:
        mx.matmul = original_matmul
        return {
            "error": str(e),
            "inference_speedup": 0.0,
            "training_speedup": 0.0
        }


# EVOLVE-BLOCK-END


# Fixed part - evaluation interface
def run_optimization():
    """
    Run the MLX-LM optimization benchmark
    
    This function is called by the OpenEvolve evaluator to test
    the current optimization configuration.
    """
    
    device_info = get_device_info()
    
    # Run MLX-LM benchmark
    mlx_lm_results = benchmark_mlx_lm_performance()
    
    # Calculate summary metrics
    inference_speedup = mlx_lm_results.get("inference_speedup", 0.0)
    training_speedup = mlx_lm_results.get("training_speedup", 0.0)
    
    # Combined score (inference weighted higher)
    combined_score = 0.7 * inference_speedup + 0.3 * training_speedup
    
    # Create results summary
    results = [{
        "optimization_type": "mlx_lm_inference",
        "speedup": inference_speedup,
        "metrics": {
            "inference_speedup": inference_speedup,
            "training_speedup": training_speedup,
            "combined_score": combined_score
        }
    }]
    
    return results, combined_score, mlx_lm_results.get("optimized_time", 1.0), device_info


if __name__ == "__main__":
    print("🚀 MLX-LM Optimization Test")
    print("=" * 40)
    
    device_info = get_device_info()
    print(f"Device: {device_info['chip']} ({device_info['memory_gb']} GB RAM)")
    
    # Test the optimization
    results = benchmark_mlx_lm_performance()
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        speedup = results["inference_speedup"]
        original_time = results["original_time"]
        optimized_time = results["optimized_time"]
        
        print(f"\n📊 Results:")
        print(f"   Original time: {original_time:.3f}s")
        print(f"   Optimized time: {optimized_time:.3f}s")
        print(f"   Speedup: {speedup:.3f}x")
        
        if speedup > 1.05:
            print("   ✅ Optimization successful!")
        elif speedup > 0.95:
            print("   ⚪ No significant change")
        else:
            print("   ❌ Performance regression")
