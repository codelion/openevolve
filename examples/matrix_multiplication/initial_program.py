"""
Initial implementation of matrix multiplication
"""
import numpy as np
import time
from typing import List, Tuple


# EVOLVE-BLOCK-START
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B
    
    This is a naive implementation of matrix multiplication.
    The goal is to optimize this function for specific matrix sizes.
    
    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)
        
    Returns:
        Result matrix of shape (m, p)
    """
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError(f"Incompatible matrix shapes: {A.shape} and {B.shape}")
    
    # Initialize result matrix with zeros
    C = np.zeros((m, p), dtype=A.dtype)
    
    # Naive triple-loop implementation
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C
# EVOLVE-BLOCK-END


def test_correctness(sizes: List[Tuple[int, int, int]]) -> bool:
    """
    Test the correctness of the matrix multiplication implementation
    
    Args:
        sizes: List of (m, n, p) tuples representing matrix sizes
        
    Returns:
        True if all tests pass, False otherwise
    """
    for m, n, p in sizes:
        # Create random matrices
        A = np.random.rand(m, n)
        B = np.random.rand(n, p)
        
        # Calculate reference result
        reference = np.matmul(A, B)
        
        # Calculate result with our implementation
        result = matrix_multiply(A, B)
        
        # Check if results are close
        if not np.allclose(reference, result, rtol=1e-5, atol=1e-8):
            print(f"Test failed for sizes {(m, n, p)}")
            return False
    
    return True


def benchmark(sizes: List[Tuple[int, int, int]], runs: int = 5) -> dict:
    """
    Benchmark the matrix multiplication implementation
    
    Args:
        sizes: List of (m, n, p) tuples representing matrix sizes
        runs: Number of runs for each size
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for m, n, p in sizes:
        # Create random matrices
        A = np.random.rand(m, n)
        B = np.random.rand(n, p)
        
        # Warm-up
        matrix_multiply(A, B)
        
        # Benchmark
        times = []
        for _ in range(runs):
            start_time = time.time()
            matrix_multiply(A, B)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Record average time
        avg_time = sum(times) / runs
        results[f"{m}x{n}x{p}"] = avg_time
    
    return results


if __name__ == "__main__":
    # Test correctness
    test_sizes = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (10, 10, 10)]
    if test_correctness(test_sizes):
        print("All correctness tests passed!")
    else:
        print("Some correctness tests failed!")
    
    # Benchmark
    benchmark_sizes = [(10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40)]
    results = benchmark(benchmark_sizes)
    
    print("\nBenchmark results:")
    for size, time_taken in results.items():
        print(f"  {size}: {time_taken:.6f} seconds")
