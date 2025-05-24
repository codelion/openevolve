"""
MLX-LM OpenEvolve Integration

This module provides seamless integration of OpenEvolve-optimized MLX kernels
with the standard mlx-lm library. Simply import this module and your existing
MLX-LM code will automatically benefit from optimized matrix multiplication.

Example:
    # Before optimization
    from mlx_lm import load, generate
    
    # After optimization - just add this import
    from mlx_lm_openevolve import enable_optimizations
    enable_optimizations()
    
    # Now your existing code automatically uses optimized kernels!
    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    text = generate(model, tokenizer, prompt="Hello world", verbose=True)
"""

import os
import importlib.util
import warnings
from typing import Optional, Tuple
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError("MLX not found. Please install with: pip install mlx")

# Global state to track if optimizations are enabled
_optimizations_enabled = False
_original_matmul = None
_optimized_choose_tile_size = None
_device_info = None


def _load_optimized_heuristics(best_program_path: Optional[str] = None) -> bool:
    """Load the optimized tiling heuristics from best_program.py"""
    global _optimized_choose_tile_size, _device_info
    
    if best_program_path is None:
        # Look for best_program.py in the current directory and common locations
        search_paths = [
            "./best_program.py",
            "./mlx_optimization_db/best/best_program.py", 
            "./examples/mlx_kernel_optimization/mlx_optimization_db/best/best_program.py",
            "./openevolve_output/best/best_program.py"
        ]
        
        best_program_path = None
        for path in search_paths:
            if os.path.exists(path):
                best_program_path = path
                break
    
    if not best_program_path or not os.path.exists(best_program_path):
        warnings.warn(
            "Optimized kernels not found. Please run the MLX optimization example first "
            "or specify the path to best_program.py. Using default MLX kernels."
        )
        return False
    
    try:
        # Load the optimized program
        spec = importlib.util.spec_from_file_location("best_program", best_program_path)
        best_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(best_program)
        
        # Extract the optimized functions
        if hasattr(best_program, 'choose_tile_size'):
            _optimized_choose_tile_size = best_program.choose_tile_size
        else:
            warnings.warn("choose_tile_size function not found in best_program.py")
            return False
            
        if hasattr(best_program, 'get_device_info'):
            _device_info = best_program.get_device_info()
        else:
            # Fallback device info
            import psutil
            _device_info = {
                "chip": "Apple Silicon",
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "cpu_count": psutil.cpu_count()
            }
        
        print(f"✅ Loaded optimized MLX kernels from {best_program_path}")
        print(f"   Device: {_device_info['chip']} ({_device_info['memory_gb']} GB RAM)")
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to load optimized kernels: {e}")
        return False


def _optimized_matmul(A, B):
    """Optimized matrix multiplication using evolved tiling heuristics"""
    global _optimized_choose_tile_size, _device_info
    
    if _optimized_choose_tile_size is None or _device_info is None:
        # Fallback to original implementation
        return _original_matmul(A, B)
    
    # Get matrix dimensions
    if len(A.shape) != 2 or len(B.shape) != 2:
        # Only optimize 2D matrix multiplication for now
        return _original_matmul(A, B)
    
    M, K1 = A.shape
    K2, N = B.shape
    
    if K1 != K2:
        return _original_matmul(A, B)
    
    K = K1
    
    # For small matrices, use original implementation (overhead not worth it)
    if M * N * K < 1000:
        return _original_matmul(A, B)
    
    try:
        # Get optimized tile sizes
        tile_M, tile_N, tile_K = _optimized_choose_tile_size(M, N, K, _device_info)
        
        # Use tiled multiplication for larger matrices
        if max(tile_M, tile_N, tile_K) < min(M, N, K):
            return _tiled_matmul_optimized(A, B, tile_M, tile_N, tile_K)
        else:
            # If tiles are too large, fallback to original
            return _original_matmul(A, B)
            
    except Exception:
        # If anything goes wrong, fallback to original implementation
        return _original_matmul(A, B)


def _tiled_matmul_optimized(A, B, tile_M, tile_N, tile_K):
    """Perform tiled matrix multiplication using optimized tile sizes"""
    M, K1 = A.shape
    K2, N = B.shape
    
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
                C_tile = _original_matmul(A_tile, B_tile)
                C = C.at[i:i_end, j:j_end].add(C_tile)
    
    return C


def enable_optimizations(best_program_path: Optional[str] = None) -> bool:
    """
    Enable OpenEvolve-optimized MLX kernels
    
    Args:
        best_program_path: Optional path to best_program.py. If None, searches common locations.
        
    Returns:
        bool: True if optimizations were successfully enabled
        
    Example:
        >>> from mlx_lm_openevolve import enable_optimizations
        >>> enable_optimizations()
        ✅ Loaded optimized MLX kernels from ./best_program.py
           Device: Apple M2 Pro (16.0 GB RAM)
        >>> # Now all MLX operations use optimized kernels!
    """
    global _optimizations_enabled, _original_matmul
    
    if _optimizations_enabled:
        print("⚠️  Optimizations already enabled")
        return True
    
    # Load the optimized heuristics
    if not _load_optimized_heuristics(best_program_path):
        return False
    
    # Monkey patch MLX matrix multiplication
    try:
        _original_matmul = mx.matmul
        mx.matmul = _optimized_matmul
        _optimizations_enabled = True
        print("🚀 OpenEvolve optimizations enabled for MLX!")
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to enable optimizations: {e}")
        return False


def disable_optimizations():
    """Disable optimizations and restore original MLX behavior"""
    global _optimizations_enabled, _original_matmul
    
    if not _optimizations_enabled:
        print("⚠️  Optimizations not currently enabled")
        return
    
    if _original_matmul is not None:
        mx.matmul = _original_matmul
        _optimizations_enabled = False
        print("🔄 Restored original MLX behavior")


def is_optimized() -> bool:
    """Check if optimizations are currently enabled"""
    return _optimizations_enabled


def get_optimization_info() -> dict:
    """Get information about current optimizations"""
    return {
        "enabled": _optimizations_enabled,
        "device_info": _device_info,
        "has_optimized_heuristics": _optimized_choose_tile_size is not None
    }


# Convenience functions for common use cases
def patch_mlx_lm(best_program_path: Optional[str] = None):
    """Convenience function to enable optimizations (alias for enable_optimizations)"""
    return enable_optimizations(best_program_path)


# Auto-enable optimizations if best_program.py is found in current directory
def _auto_enable():
    """Automatically enable optimizations if best_program.py is found"""
    if os.path.exists("./best_program.py"):
        try:
            enable_optimizations("./best_program.py")
        except:
            pass  # Silently fail auto-enable


if __name__ == "__main__":
    # Demo usage
    print("MLX-LM OpenEvolve Integration Demo")
    print("=" * 40)
    
    success = enable_optimizations()
    if success:
        info = get_optimization_info()
        print(f"Optimizations enabled: {info['enabled']}")
        print(f"Device: {info['device_info']}")
    else:
        print("Could not enable optimizations. Run the MLX optimization example first!")
